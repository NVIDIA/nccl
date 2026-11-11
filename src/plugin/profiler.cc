/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param.h"
#include "checks.h"
#include "comm.h"
#include "utils.h"
#include "proxy.h"
#include "profiler.h"
#include "transport.h"
#include "plugin.h"
#include "compiler.h"
#include "device.h"
#include "sym_kernels.h"
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include <time.h>
#include "os.h"

extern ncclProfiler_t* getNcclProfiler_v1(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v2(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v3(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v4(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v5(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v6(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v7(void* lib, void** target);
extern ncclProfiler_t* getNcclProfiler_v8(void* lib, void** target);
extern void setNcclProfiler_v1(void* target);
extern void setNcclProfiler_v2(void* target);
extern void setNcclProfiler_v3(void* target);
extern void setNcclProfiler_v4(void* target);
extern void setNcclProfiler_v5(void* target);
extern void setNcclProfiler_v6(void* target);
extern void setNcclProfiler_v7(void* target);

// Observability threshold for pending+active ops. We keep enqueueing beyond
// this so KernelCh events stay paired with their parent task events.
#define NCCL_PROFILER_DEFAULT_MAX_INFLIGHT (MAXCHANNELS * MAX_PROFILER_EVENTS_PER_CHANNEL * 4)

// GPU->CPU time calibration, owned and used solely by one profiler thread.
//
// The device profiler stamps GPU globaltimer ns into KernelCh timestamps.
// To also report those on the plugin's chosen CPU clock (v8 gpuClockSync), the
// thread periodically launches a 1-thread probe kernel
// (ncclProfilerProbeGpuClock) sandwiched between CPU clock reads, takes the
// lowest-latency (gpu, cpu) anchor, and fits a near-unity linear map
// cpu = cpuRef + slope*(gpu - gpuRef). The slope is refined from consecutive
// anchors (clamped near 1.0 to reject bad samples).
struct ncclProfilerClockCalib {
  bool ready;                 // probe stream + pinned buffer allocated
  bool valid;                 // at least one good anchor taken
  cudaStream_t stream;
  unsigned long long* pinned; // probe output (host-pinned, kernel-visible)
  uint64_t gpuRef, cpuRef;
  double slope;
  bool havePrev;
  uint64_t prevGpu, prevCpu;
  uint64_t nextCalibCpu;      // recalibrate once the CPU clock reaches this
  // Half-width of the CPU interval bracketing the current anchor's GPU clock
  // read, i.e. the +/- bound we report to the plugin as tsCpuUncertainty.
  uint64_t uncertainty;
};

struct ncclProfilerThread {
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
  // GPU->CPU time calibration (only touched by this thread).
  struct ncclProfilerClockCalib clockCalib;
  // Signalled when iterationActive transitions to false; lets a concurrent
  // ncclProfilerThreadDestroy wait for an in-flight progress pass to finish.
  std::condition_variable condIterationInactive;
  int stop;
  int refCount;
  // Captured at thread-create so a defensive cudaSetDevice at startup
  // gives the plugin a valid primary context even if it was loaded by a
  // user thread bound to a different device.
  int cudaDev;
  volatile uint32_t* abortFlag;
  // True while the thread is iterating `active` outside the mutex and may
  // be calling into the plugin. ncclProfilerThreadDestroy waits this out
  // before tearing down per-comm state.
  bool iterationActive;
  // Posters append to pending under mutex; thread splices it into active.
  struct ncclProfilerWorkOp* pending;
  struct ncclProfilerWorkOp* pendingTail;
  struct ncclProfilerWorkOp* active;
  struct ncclProfilerWorkOp* activeTail;
  struct ncclMemoryStack opStack;
  struct ncclMemoryPool opPool;

  // Backpressure observability, all under mutex.
  size_t inflight;
  size_t maxInflightSeen;
  size_t maxInflight;
  uint64_t droppedOps;  // Allocation failures only; never used for soft cap.
};

enum ncclProfilerThreadAction {
  NCCL_PROFILER_THREAD_PROGRESS,
  NCCL_PROFILER_THREAD_STOP,
  // Stop requested but ops are still queued; drain them so join() returns.
  NCCL_PROFILER_THREAD_CLEANUP_AND_STOP,
};

static std::mutex profilerMutex;
static int profilerPluginRefCount;

struct ncclProfilerPlugin {
  void* lib;
  ncclProfiler_t api;
  void* target;
  void (*activate)(void*);
  bool legacyNetPlugin;
  // The word init() is handed, and the one delivery reads back. A lone plugin
  // owns ncclProfilerEventMask itself, exactly as before multi-plugin support,
  // so there is no union to rebuild; several plugins each own `mask` below.
  int* maskPtr;
  int mask;
  // v8 timestamp mode requested by this plugin. Multi-plugin dispatch uses
  // this instead of the process-wide calibration mode so opt-out is isolated.
  uint32_t tsMode;
};

struct ncclProfilerMultiContext {
  int count;
  struct {
    void* context;
    bool initialized;
  } slots[NCCL_PROFILER_MAX_PLUGINS];
};

static struct ncclProfilerPlugin profilerPlugins[NCCL_PROFILER_MAX_PLUGINS];
static int profilerPluginCount;
static ncclProfiler_t* ncclProfiler;
// Pre-v6 plugins retain implicit proxy ancestors for NetPlugin events, including RAS masks.
static bool profilerLegacyNetPlugin;

// The v8 timestamp opt-in, as negotiated by whichever plugin is loaded: the
// CPU clock to stamp into tsCpu (an ncclProfilerCpuClock_t) and whether
// GPU-timed events should also be mapped onto it. Zero means no plugin has
// asked, so NCCL reads no clocks, runs no GPU clock calibration and leaves
// every tsCpu at NCCL_PROFILER_TS_NONE.
//
// Process-wide rather than per-comm: the toggles are negotiated per
// communicator but the stamping sites (group API, proxy control, net
// callbacks) have no comm in hand, and one plugin library answers for every
// comm in the process. The per-comm answer is still recorded in
// comm->profiler.config. Once any comm has asked, stamping stays on until the
// plugin is unloaded.
//
// Packed into one atomic word rather than two because comms initialize
// concurrently: one thread may publish while another reads on an event path,
// and a single word keeps the clock selection and the GPU toggle from being
// observed half-applied. The first selected CPU clock remains fixed; a later
// plugin on that same clock may enable GPU calibration. The reset to 0 happens
// under profilerMutex when the last comm has gone.
#define PROF_TS_MODE(clock, gpuSync) (((uint32_t)(clock) & 0xffu) | ((gpuSync) ? 0x100u : 0u))
#define PROF_TS_MODE_CLOCK(mode) ((mode) & 0xffu)
#define PROF_TS_MODE_GPUSYNC(mode) (((mode) & 0x100u) != 0u)

static uint32_t profilerTsMode = 0;

static inline uint32_t profilerTsModeGet(void) {
  return COMPILER_ATOMIC_LOAD(&profilerTsMode, std::memory_order_relaxed);
}

static uint32_t profilerApplyConfig(const ncclProfilerConfig_t* config) {
  uint32_t clock = config->cpuClock;
  if (clock != ncclProfilerCpuClockMonotonic && clock != ncclProfilerCpuClockRealtime) clock = ncclProfilerCpuClockNone;
  bool gpuSync = clock != ncclProfilerCpuClockNone && config->gpuClockSync;
  uint32_t mode = PROF_TS_MODE(clock, gpuSync);
  uint32_t expected = COMPILER_ATOMIC_LOAD(&profilerTsMode, std::memory_order_relaxed);
  bool published = false;
  while (mode) {
    uint32_t desired = expected;
    if (expected == 0) {
      desired = mode;
    } else if (PROF_TS_MODE_CLOCK(expected) == PROF_TS_MODE_CLOCK(mode)) {
      desired |= mode & 0x100u;
    }
    if (desired == expected) break;
    if (COMPILER_ATOMIC_COMPARE_EXCHANGE(&profilerTsMode, &expected, desired, std::memory_order_release,
                                         std::memory_order_relaxed)) {
      published = true;
      break;
    }
  }
  if (published) {
    INFO(NCCL_INIT | NCCL_PROFILE, "Profiler: plugin requested NCCL-sourced timings on %s%s",
         clock == ncclProfilerCpuClockRealtime ? "CLOCK_REALTIME" : "CLOCK_MONOTONIC",
         gpuSync ? "; enabling GPU->CPU clock calibration" : "");
  }
  return mode;
}

static void profilerApplyPluginConfig(struct ncclProfilerPlugin* plugin, const ncclProfilerConfig_t* config) {
  uint32_t mode = profilerApplyConfig(config);
  uint32_t expected = COMPILER_ATOMIC_LOAD(&plugin->tsMode, std::memory_order_relaxed);
  while (mode) {
    uint32_t desired = expected;
    if (expected == 0) {
      desired = mode;
    } else if (PROF_TS_MODE_CLOCK(expected) == PROF_TS_MODE_CLOCK(mode)) {
      desired |= mode & 0x100u;
    }
    if (desired == expected) break;
    if (COMPILER_ATOMIC_COMPARE_EXCHANGE(&plugin->tsMode, &expected, desired, std::memory_order_release,
                                         std::memory_order_relaxed))
      break;
  }
}

// Read the CPU clock the plugin selected. Returns NCCL_PROFILER_TS_NONE when
// no plugin opted in, which is also what every unrecognized clock maps to.
static inline uint64_t profilerCpuNow(uint32_t mode) {
  clockid_t id;
  switch (PROF_TS_MODE_CLOCK(mode)) {
  case ncclProfilerCpuClockMonotonic:
    id = CLOCK_MONOTONIC;
    break;
  case ncclProfilerCpuClockRealtime:
    id = CLOCK_REALTIME;
    break;
  default:
    return NCCL_PROFILER_TS_NONE;
  }
  struct timespec t;
  clock_gettime(id, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// CLOCK_MONOTONIC regardless of what the plugin selected. Used only to
// schedule recalibration, which must not be thrown off by a CLOCK_REALTIME
// step.
static inline uint64_t profilerMonoNow(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + (uint64_t)t.tv_nsec;
}

// ---------------------------------------------------------------------------
// GPU->CPU time calibration (profiler-thread local).
// ---------------------------------------------------------------------------
#define NCCL_PROF_CLOCK_PROBES 8                    // best-of-N per anchor
#define NCCL_PROF_CLOCK_RECALIB_NS (1000000000ull)      // re-anchor every ~1s

// The active thread's calibration, set at thread start. Only ever
// read/written by the owning profiler thread.
static thread_local struct ncclProfilerClockCalib* profilerTlCalib = nullptr;

static void profilerClockCalibInit(struct ncclProfilerClockCalib* c) {
  memset(c, 0, sizeof(*c));
  c->slope = 1.0;
  if (cudaStreamCreateWithFlags(&c->stream, cudaStreamNonBlocking) != cudaSuccess) {
    c->stream = nullptr;
    return;
  }
  if (cudaMallocHost((void**)&c->pinned, sizeof(*c->pinned)) != cudaSuccess) {
    cudaStreamDestroy(c->stream);
    c->stream = nullptr;
    c->pinned = nullptr;
    return;
  }
  c->ready = true;
}

static void profilerClockCalibFree(struct ncclProfilerClockCalib* c) {
  if (c->pinned) {
    cudaFreeHost(c->pinned);
    c->pinned = nullptr;
  }
  if (c->stream) {
    cudaStreamDestroy(c->stream);
    c->stream = nullptr;
  }
  c->ready = false;
  c->valid = false;
}

// The clock probe can fail when the CUDA context has been poisoned by
// unrelated GPU work (e.g. a driver-contained NVLink/hardware
// fault). That is not a profiler bug, so report it as INFO log.
static inline void profilerClockProbeReport() {
  INFO(NCCL_INIT | NCCL_PROFILE, "[Profiler Thread] GPU clock probe failed "
                                 "(likely a CUDA context error from unrelated GPU work); "
                                 "GPU->CPU timestamp calibration paused. "
                                 "Informational -- not itself a profiler fault.");
}

// Fold one completed probe round-trip into the best-so-far
// anchor. The GPU read its clock (now in *c->pinned) somewhere inside
// the CPU interval [t0, t1], so the round-trip midpoint is the
// least-biased CPU estimate of that instant; we keep the sample with
// the shortest round-trip, whose midpoint is tightest.
static inline void profilerClockKeepBest(struct ncclProfilerClockCalib* c, uint64_t t0, uint64_t t1, uint64_t* bestLat,
                                         uint64_t* gpuOut, uint64_t* cpuOut, bool* got) {
  uint64_t lat = (t1 > t0) ? (t1 - t0) : 0;
  if (lat >= *bestLat) return;
  *bestLat = lat;
  *gpuOut = (uint64_t)*c->pinned;
  *cpuOut = t0 + lat / 2;
  *got = true;
}

// One lowest-latency (gpu, cpu) anchor: launch the probe N times and keep the
// sample whose CPU round-trip was shortest (its midpoint best approximates the
// instant the GPU read its clock).
static bool profilerClockProbe(struct ncclProfilerClockCalib* c, uint32_t mode, uint64_t* gpuOut, uint64_t* cpuOut,
                               uint64_t* uncOut) {
  if (!c->ready) return false;
  uint64_t bestLat = ~0ull;
  bool got = false;
  for (int i = 0; i < NCCL_PROF_CLOCK_PROBES; i++) {
    uint64_t t0 = profilerCpuNow(mode);
    if (ncclProfilerProbeGpuClock(c->pinned, c->stream) != ncclSuccess) {
      profilerClockProbeReport();
      break;
    }
    if (cudaStreamSynchronize(c->stream) != cudaSuccess) {
      (void)cudaGetLastError();
      profilerClockProbeReport();
      break;
    }
    profilerClockKeepBest(c, t0, profilerCpuNow(mode), &bestLat, gpuOut, cpuOut, &got);
  }
  // The kept sample's GPU read happened inside a CPU interval of width
  // bestLat and we report its midpoint, so the error bound is half that.
  if (got) *uncOut = bestLat / 2;
  return got;
}

// Take a fresh anchor and (when a prior anchor is far enough away)
// refine the slope. globaltimer and the CPU clock both advance
// ~1ns/tick, so a sane slope is within a few percent of 1.0; values
// outside that come from a perturbed sample and are rejected.
static void profilerClockCalibUpdate(struct ncclProfilerClockCalib* c, uint32_t mode) {
  uint64_t gpu, cpu, unc;
  if (!profilerClockProbe(c, mode, &gpu, &cpu, &unc)) return;
  if (c->havePrev && gpu > c->prevGpu) {
    double s = (double)(cpu - c->prevCpu) / (double)(gpu - c->prevGpu);
    if (s > 0.95 && s < 1.05) c->slope = s;
  }
  c->prevGpu = gpu;
  c->prevCpu = cpu;
  c->havePrev = true;
  c->gpuRef = gpu;
  c->cpuRef = cpu;
  c->uncertainty = unc;
  c->valid = true;
}

// Recalibrate on first use and roughly once per
// NCCL_PROF_CLOCK_RECALIB_NS.
static void profilerClockCalibMaybe(struct ncclProfilerClockCalib* c, uint32_t mode) {
  uint64_t now = profilerMonoNow();
  if (!c->valid || now >= c->nextCalibCpu) {
    profilerClockCalibUpdate(c, mode);
    c->nextCalibCpu = now + NCCL_PROF_CLOCK_RECALIB_NS;
  }
}

// Map a GPU globaltimer ns onto the configured CPU clock using this thread's
// calibration, along with the anchor's +/- error bound. Yields 0
// (NCCL_PROFILER_TS_NONE) when uncalibrated or tsGpu==0.
static inline void profilerMapGpuToCpu(uint64_t tsGpu, uint64_t* tsCpu, uint64_t* tsCpuUncertainty) {
  *tsCpu = NCCL_PROFILER_TS_NONE;
  *tsCpuUncertainty = 0;
  struct ncclProfilerClockCalib* c = profilerTlCalib;
  if (c == nullptr || !c->valid || tsGpu == 0) return;
  int64_t d = (int64_t)tsGpu - (int64_t)c->gpuRef;
  *tsCpu = (uint64_t)((int64_t)c->cpuRef + (int64_t)(c->slope * (double)d));
  *tsCpuUncertainty = c->uncertainty;
}

extern thread_local int ncclGroupDepth;
thread_local ncclProfilerApiState_t ncclProfilerApiState;

#define MAX_STR_LEN 256

enum {
  profilerPluginLoadFailed = -1,
  profilerPluginLoadReady = 0,
  profilerPluginLoadSuccess = 1,
};
static int profilerPluginStatus = profilerPluginLoadReady;
static pid_t pid;

// Every call into a plugin goes through here. For v1-v6 this selects the
// shim's thread-local plugin, which is how its stopEvent/recordEventState
// adapters recover the owner of an event handle; native v8 needs no TLS shim
// and has activate == nullptr. See profiler_shim.h.
static ncclProfiler_t* profilerActivate(struct ncclProfilerPlugin* plugin) {
  if (plugin->activate) plugin->activate(plugin->target);
  return &plugin->api;
}

static void profilerTrimRange(const char** start, const char** end) {
  while (*start < *end && isspace((unsigned char)**start)) (*start)++;
  while (*end > *start && isspace((unsigned char)*(*end - 1))) (*end)--;
}

static bool profilerGetPlugin(void* lib, struct ncclProfilerPlugin* plugin) {
  static ncclProfiler_t* (*const get[])(void*, void**) = {getNcclProfiler_v8, getNcclProfiler_v7, getNcclProfiler_v6,
                                                          getNcclProfiler_v5, getNcclProfiler_v4, getNcclProfiler_v3,
                                                          getNcclProfiler_v2, getNcclProfiler_v1};
  static void (*const activate[])(void*) = {nullptr,
                                            setNcclProfiler_v7,
                                            setNcclProfiler_v6,
                                            setNcclProfiler_v5,
                                            setNcclProfiler_v4,
                                            setNcclProfiler_v3,
                                            setNcclProfiler_v2,
                                            setNcclProfiler_v1};
  for (int i = 0; i < 8; i++) {
    ncclProfiler_t* api = get[i](lib, &plugin->target);
    if (api) {
      plugin->api = *api;
      plugin->activate = activate[i];
      plugin->legacyNetPlugin = i >= 3;
      return true;
    }
  }
  return false;
}

static bool profilerPluginAdd(void* lib, int listPosition, const char* requestedName) {
  for (int i = 0; i < profilerPluginCount; i++) {
    if (profilerPlugins[i].lib == lib) {
      INFO(NCCL_INIT, "External profiler plugin is already loaded, skipping it");
      ncclClosePluginLibHandle(lib, ncclPluginTypeProfiler);
      return false;
    }
  }

  if (profilerPluginCount == NCCL_PROFILER_MAX_PLUGINS) {
    WARN("NCCL supports at most %d profiler plugins; ignoring additional entry %d", NCCL_PROFILER_MAX_PLUGINS,
         listPosition + 1);
    ncclClosePluginLibHandle(lib, ncclPluginTypeProfiler);
    return false;
  }

  struct ncclProfilerPlugin plugin = {};
  plugin.lib = lib;
  if (!profilerGetPlugin(lib, &plugin)) {
    if (requestedName) ATTN("External profiler plugin %s is unsupported", requestedName);
    else INFO(NCCL_INIT, "External profiler plugin is unsupported");
    ncclClosePluginLibHandle(lib, ncclPluginTypeProfiler);
    return false;
  }

  profilerPlugins[profilerPluginCount++] = plugin;
  if (plugin.legacyNetPlugin) COMPILER_ATOMIC_STORE(&profilerLegacyNetPlugin, true, std::memory_order_relaxed);
  INFO(NCCL_INIT, "Successfully loaded external profiler plugin %s", plugin.api.name);
  return true;
}

static ncclResult_t ncclProfilerPluginLoad(void) {
  if (profilerPluginLoadFailed == profilerPluginStatus) return ncclSuccess;

  std::lock_guard<std::mutex> lock(profilerMutex);
  if (profilerPluginLoadSuccess == profilerPluginStatus) {
    profilerPluginRefCount++;
    return ncclSuccess;
  }

  const char* names = ncclGetEnv("NCCL_PROFILER_PLUGIN");
  bool disabled = false;
  // Unset or empty (whitespace-only) loads libnccl-profiler.so, matching
  // the pre-list behavior of NCCL_PROFILER_PLUGIN.
  bool loadDefault = (names == nullptr);
  int listEntries = 0;
  if (names) {
    INFO(NCCL_ENV, "NCCL_PROFILER_PLUGIN set by environment to %s", names);
    const char* start = names;
    const char* end = names + strlen(names);
    profilerTrimRange(&start, &end);
    if (end - start == 4 && strncasecmp(start, "none", 4) == 0) {
      disabled = true;
    } else {
      loadDefault = (start == end);
    }
  }

  if (!disabled) {
    if (loadDefault) {
      void* lib = ncclOpenProfilerPluginLib(nullptr);
      if (lib) {
        ncclDetachPluginLibHandle(lib, ncclPluginTypeProfiler);
        profilerPluginAdd(lib, 0, nullptr);
      }
    } else {
      const char* name = names;
      for (int position = 0; name; position++) {
        listEntries = position + 1;
        const char* comma = strchr(name, ',');
        const char* start = name;
        const char* end = comma ? comma : name + strlen(name);
        profilerTrimRange(&start, &end);
        if (start != end) {
          char* entry = strndup(start, (size_t)(end - start));
          if (!entry) {
            WARN("Failed to copy NCCL_PROFILER_PLUGIN entry");
            break;
          }
          void* lib = ncclOpenProfilerPluginLib(entry);
          if (lib) {
            ncclDetachPluginLibHandle(lib, ncclPluginTypeProfiler);
            profilerPluginAdd(lib, position, entry);
          } else {
            ATTN("External profiler plugin %s was not found", entry);
          }
          free(entry);
        }
        name = comma ? comma + 1 : nullptr;
      }
    }
  }

  if (profilerPluginCount < NCCL_PROFILER_MAX_PLUGINS && !disabled) {
    void* lib = ncclGetNetPluginLib(ncclPluginTypeProfiler);
    if (lib) {
      ncclDetachPluginLibHandle(lib, ncclPluginTypeProfiler);
      profilerPluginAdd(lib, names ? listEntries : 0, nullptr);
    }
  }

  if (profilerPluginCount == 0) {
    profilerPluginStatus = profilerPluginLoadFailed;
    return ncclSuccess;
  }

  ncclProfiler = &profilerPlugins[0].api;
  for (int i = 0; i < profilerPluginCount; i++) {
    profilerPlugins[i].maskPtr = profilerPluginCount == 1 ? &ncclProfilerEventMask : &profilerPlugins[i].mask;
  }
  profilerPluginRefCount++;
  profilerPluginStatus = profilerPluginLoadSuccess;

  // Store the pid of the process loading the profiler.
  // This is attached to the proxyOp event descriptor
  // so the plugin can figure out if the parent event
  // is in the same address space or not
  pid = ncclOsGetPid();
  return ncclSuccess;
}

static ncclResult_t ncclProfilerPluginUnload(void) {
  std::lock_guard<std::mutex> lock(profilerMutex);
  if (profilerPluginRefCount == 0) return ncclSuccess;
  if (--profilerPluginRefCount == 0) {
    for (int i = 0; i < profilerPluginCount; i++) {
      INFO(NCCL_DESTROY, "PROFILER/Plugin: Closing profiler plugin %s", profilerPlugins[i].api.name);
      NCCLCHECK(ncclClosePluginLibHandle(profilerPlugins[i].lib, ncclPluginTypeProfiler));
    }
    memset(profilerPlugins, 0, sizeof(profilerPlugins));
    profilerPluginCount = 0;
    ncclProfiler = nullptr;
    COMPILER_ATOMIC_STORE(&profilerLegacyNetPlugin, false, std::memory_order_relaxed);
    COMPILER_ATOMIC_STORE(&profilerTsMode, 0u, std::memory_order_relaxed);
    COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, 0, std::memory_order_relaxed);
    profilerPluginStatus = profilerPluginLoadReady;
  }
  return ncclSuccess;
}

#define ENABLE_TIMER 0
#include "timer.h"

#if ENABLE_TIMER
// These counters are used to measure profiler overheads for different part of the code
// These counters are only useful/meaningful in controlled test environments where there
// is only one thread updating each set of counters, i.e., every communicator has its
// own proxy thread and the network uses only one thread to make progress (this is true
// for net_ib plugin but might not be true for net_socket plugin).
static int64_t elapsedCount;
static int64_t initCount, finalizeCount;
static int64_t groupStartCount, groupStopCount;
static int64_t taskStartCount, taskStopCount;
static int64_t proxyOpStartCount, proxyOpStopCount;
static int64_t proxyStepStartCount, proxyStepStopCount;
static int64_t proxyCtrlStartCount, proxyCtrlStopCount;
static int64_t proxyOpRecordCount, proxyStepRecordCount, proxyCtrlRecordCount;

static double elapsedTs[2];
static double initTs[2], finalizeTs[2];
static double groupStartTs[2], groupStopTs[2];
static double taskStartTs[2], taskStopTs[2];
static double proxyOpStartTs[2], proxyOpStopTs[2];
static double proxyStepStartTs[2], proxyStepStopTs[2];
static double proxyCtrlStartTs[2], proxyCtrlStopTs[2];
static double proxyOpRecordTs[2], proxyStepRecordTs[2], proxyCtrlRecordTs[2];

#define TIME_START_EVENT(event) \
  do { \
    (event##Count)++; \
    (event##Ts)[0] = gettime(); \
  } while (0)

#define TIME_STOP_EVENT(event) \
  do { \
    double val = gettime() - (event##Ts)[0]; \
    (event##Ts)[1] += val; \
  } while (0)

#define TIME_PRINT_EVENTS(name) \
  do { \
    printf("%s ", name); \
    if (elapsedCount) printf("[elapsed] %g/%ld = %g ", elapsedTs[1], elapsedCount, elapsedTs[1] / elapsedCount); \
    if (initCount) printf("[init] %g/%ld = %g ", initTs[1], initCount, initTs[1] / initCount); \
    if (finalizeCount) printf("[finalize] %g/%ld = %g ", finalizeTs[1], finalizeCount, finalizeTs[1] / finalizeCount); \
    if (groupStartCount) \
      printf("[groupStart] %g/%ld = %g ", groupStartTs[1], groupStartCount, groupStartTs[1] / groupStartCount); \
    if (groupStopCount) \
      printf("[groupStop] %g/%ld = %g ", groupStopTs[1], groupStopCount, groupStopTs[1] / groupStopCount); \
    if (taskStartCount) \
      printf("[taskStart] %g/%ld = %g ", taskStartTs[1], taskStartCount, taskStartTs[1] / taskStartCount); \
    if (taskStopCount) printf("[taskStop] %g/%ld = %g ", taskStopTs[1], taskStopCount, taskStopTs[1] / taskStopCount); \
    if (proxyOpStartCount) \
      printf("[proxyOpStart] %g/%ld = %g ", proxyOpStartTs[1], proxyOpStartCount, \
             proxyOpStartTs[1] / proxyOpStartCount); \
    if (proxyOpStopCount) \
      printf("[proxyOpStop] %g/%ld = %g ", proxyOpStopTs[1], proxyOpStopCount, proxyOpStopTs[1] / proxyOpStopCount); \
    if (proxyStepStartCount) \
      printf("[proxyStepStart] %g/%ld = %g ", proxyStepStartTs[1], proxyStepStartCount, \
             proxyStepStartTs[1] / proxyStepStartCount); \
    if (proxyStepStopCount) \
      printf("[proxyStepStop] %g/%ld = %g ", proxyStepStopTs[1], proxyStepStopCount, \
             proxyStepStopTs[1] / proxyStepStopCount); \
    if (proxyCtrlStartCount) \
      printf("[proxyCtrlStart] %g/%ld = %g ", proxyCtrlStartTs[1], proxyCtrlStartCount, \
             proxyCtrlStartTs[1] / proxyCtrlStartCount); \
    if (proxyCtrlStopCount) \
      printf("[proxyCtrlStop] %g/%ld = %g ", proxyCtrlStopTs[1], proxyCtrlStopCount, \
             proxyCtrlStopTs[1] / proxyCtrlStopCount); \
    if (proxyOpRecordCount) \
      printf("[proxyOpRecord] %g/%ld = %g ", proxyOpRecordTs[1], proxyOpRecordCount, \
             proxyOpRecordTs[1] / proxyOpRecordCount); \
    if (proxyStepRecordCount) \
      printf("[proxyStepRecord] %g/%ld = %g ", proxyStepRecordTs[1], proxyStepRecordCount, \
             proxyStepRecordTs[1] / proxyStepRecordCount); \
    if (proxyCtrlRecordCount) \
      printf("[proxyCtrlRecord] %g/%ld = %g", proxyCtrlRecordTs[1], proxyCtrlRecordCount, \
             proxyCtrlRecordTs[1] / proxyCtrlRecordCount); \
    printf("\n"); \
  } while (0)
#else
#define TIME_START_EVENT(event) \
  do { \
  } while (0)
#define TIME_STOP_EVENT(event) \
  do { \
  } while (0)
#define TIME_PRINT_EVENTS(name) \
  do { \
  } while (0)
#endif

int ncclProfilerEventMask;       // Set by profiler

// RAS out-of-band override of the event mask; re-applied at the end of ncclProfilerPluginInit so a new
// comm's plugin init() cannot clobber a mask set job-wide via the RAS client.
static bool ncclProfilerRasOverrideActive = false;
static int ncclProfilerRasOverrideMask = 0;

// Print enabled profiler event types
static void printProfilerEventMask(int mask, const char* name) {
  if (!mask && !name) return;

  char enabled[512] = {0};
  int pos = 0;
  if (mask & ncclProfileGroup) pos += sprintf(enabled + pos, "Group ");
  if (mask & ncclProfileColl) pos += sprintf(enabled + pos, "Coll ");
  if (mask & ncclProfileP2p) pos += sprintf(enabled + pos, "P2p ");
  if (mask & ncclProfileProxyOp) pos += sprintf(enabled + pos, "ProxyOp ");
  if (mask & ncclProfileProxyStep) pos += sprintf(enabled + pos, "ProxyStep ");
  if (mask & ncclProfileProxyCtrl) pos += sprintf(enabled + pos, "ProxyCtrl ");
  if (mask & ncclProfileKernelCh) pos += sprintf(enabled + pos, "KernelCh ");
  if (mask & ncclProfileKernelPhase) pos += sprintf(enabled + pos, "KernelPhase ");
  if (mask & ncclProfileNetPlugin) pos += sprintf(enabled + pos, "NetPlugin ");
  if (mask & ncclProfileGroupApi) pos += sprintf(enabled + pos, "GroupApi ");
  if (mask & ncclProfileCollApi) pos += sprintf(enabled + pos, "CollApi ");
  if (mask & ncclProfileP2pApi) pos += sprintf(enabled + pos, "P2pApi ");
  if (mask & ncclProfileKernelLaunch) pos += sprintf(enabled + pos, "KernelLaunch ");
  if (mask & ncclProfileCeColl) pos += sprintf(enabled + pos, "CeColl ");
  if (mask & ncclProfileCeSync) pos += sprintf(enabled + pos, "CeSync ");
  if (mask & ncclProfileCeBatch) pos += sprintf(enabled + pos, "CeBatch ");
  if (name) {
    INFO(NCCL_INIT, "Profiler %s event mask: 0x%x (%d) - Enabled: %s", name, mask, mask, enabled);
  } else {
    INFO(NCCL_INIT, "Profiler event mask: 0x%x (%d) - Enabled: %s", mask, mask, enabled);
  }
}

// KernelPhase sub-events are gated on ncclProfileKernelCh, so KernelCh must be set
// whenever KernelPhase is. Enforce that on any incoming mask and log if it changes.
// Also preserve implicit proxy ancestors for legacy NetPlugin events.
static int ncclProfilerEnforceMaskDeps(int mask, const char* source, bool legacyNetPlugin) {
  if (legacyNetPlugin && (mask & ncclProfileNetPlugin)) {
    int required = ncclProfileProxyOp | ncclProfileProxyStep;
    if ((mask & required) != required) {
      mask |= required;
      INFO(NCCL_INIT, "Profiler: enabling ProxyOp and ProxyStep ancestors for legacy NetPlugin events (%s)", source);
    }
  }
  if ((mask & ncclProfileKernelPhase) && !(mask & ncclProfileKernelCh)) {
    mask |= ncclProfileKernelCh;
    INFO(NCCL_INIT,
         "Profiler: enabling ncclProfileKernelCh implicitly (%s set ncclProfileKernelPhase, which requires it)",
         source);
  }
  return mask;
}

static int profilerEventGate(int type) {
  switch (type) {
  case ncclProfileGroupApi:
    return ncclProfileGroupApi | ncclProfileP2pApi | ncclProfileCollApi | ncclProfileKernelLaunch | ncclProfileGroup |
           ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh |
           ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  case ncclProfileCollApi:
    return ncclProfileCollApi | ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh |
           ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  case ncclProfileP2pApi:
    return ncclProfileP2pApi | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh |
           ncclProfileNetPlugin;
  case ncclProfileGroup:
    return ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep |
           ncclProfileKernelCh | ncclProfileNetPlugin;
  case ncclProfileColl:
    return ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;
  case ncclProfileP2p:
    return ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin;
  case ncclProfileProxyOp:
    return ncclProfileProxyOp | ncclProfileProxyStep;
  case ncclProfileProxyStep:
    return ncclProfileProxyStep;
  case ncclProfileCeColl:
    return ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
  default:
    return type;
  }
}

static int profilerMaskWithDeps(int mask, bool legacyNetPlugin) {
  if ((mask & ncclProfileKernelPhase) && !(mask & ncclProfileKernelCh)) mask |= ncclProfileKernelCh;
  if (legacyNetPlugin && (mask & ncclProfileNetPlugin)) mask |= ncclProfileProxyOp | ncclProfileProxyStep;
  return mask;
}

// Delivery always reads *plugin->maskPtr, the same word init() received. A later
// write through that pointer (exampleProfilerStart) is immediately visible here.
static int profilerPluginEffectiveMask(const struct ncclProfilerPlugin* plugin) {
  return profilerMaskWithDeps(COMPILER_ATOMIC_LOAD(plugin->maskPtr, std::memory_order_relaxed),
                              plugin->legacyNetPlugin);
}

// Called from the enqueue and proxy-progress hot paths, so it must not write
// ncclProfilerEventMask in steady state: that word is read by every enqueueing
// thread and dirtying it from the proxy thread false-shares the line.
static int profilerRefreshEventMask() {
  if (COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideActive, std::memory_order_acquire)) {
    int mask = COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideMask, std::memory_order_relaxed);
    COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, mask, std::memory_order_relaxed);
    return mask;
  }
  // No plugin, or one that was handed &ncclProfilerEventMask by init(): the word
  // is already current, including any stashed-pointer write, so just read it.
  if (COMPILER_EXPECT(profilerPluginCount <= 1, 1)) {
    bool legacyNetPlugin = COMPILER_ATOMIC_LOAD(&profilerLegacyNetPlugin, std::memory_order_relaxed);
    return profilerMaskWithDeps(COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed),
                                legacyNetPlugin);
  }
  // Several plugins: each owns its mask, so the process-wide word is the union.
  int mask = 0;
  for (int i = 0; i < profilerPluginCount; i++) {
    mask |= profilerPluginEffectiveMask(&profilerPlugins[i]);
  }
  if (mask != COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed)) {
    COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, mask, std::memory_order_relaxed);
  }
  return mask;
}

int ncclProfilerCurrentEventMask(void) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1)) {
    return COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  }
  return profilerRefreshEventMask();
}

// After a successful init(), enforce KernelPhase dependencies through maskPtr
// so the plugin and NCCL observe the same value.
static int profilerApplyMaskDeps(struct ncclProfilerPlugin* plugin) {
  int mask = COMPILER_ATOMIC_LOAD(plugin->maskPtr, std::memory_order_relaxed);
  mask = ncclProfilerEnforceMaskDeps(mask, plugin->api.name, plugin->legacyNetPlugin);
  COMPILER_ATOMIC_STORE(plugin->maskPtr, mask, std::memory_order_relaxed);
  return mask;
}

static bool profilerLive(const struct ncclProfilerEventRef* event) {
  for (int slot = 0; slot < profilerPluginCount; slot++)
    if (event->h[slot]) return true;
  return false;
}

static void profilerSetParents(ncclProfilerEventDescr_t* descr, const struct ncclProfilerEventRef* parent,
                               const struct ncclProfilerEventRef* parentGroup, int slot) {
  if (parent) descr->parentObj = parent->h[slot];
  if (!parentGroup) return;
  if (descr->type == ncclProfileColl) descr->coll.parentGroup = parentGroup->h[slot];
  else if (descr->type == ncclProfileP2p) descr->p2p.parentGroup = parentGroup->h[slot];
}

enum profilerTimestampKind {
  profilerTimestampNone,
  profilerTimestampHost,
  profilerTimestampGpu,
};

static void profilerStampDescr(ncclProfilerEventDescr_t* descr, uint32_t mode, profilerTimestampKind kind,
                               uint64_t tsGpu) {
  descr->tsCpu = NCCL_PROFILER_TS_NONE;
  descr->tsCpuUncertainty = 0;
  if (kind == profilerTimestampHost) {
    descr->tsCpu = profilerCpuNow(mode);
  } else if (kind == profilerTimestampGpu && mode == profilerTsModeGet() && PROF_TS_MODE_GPUSYNC(mode)) {
    profilerMapGpuToCpu(tsGpu, &descr->tsCpu, &descr->tsCpuUncertainty);
  }
}

static void profilerStampArgs(ncclProfilerEventStateArgs_t* args, uint32_t mode, profilerTimestampKind kind,
                              uint64_t tsGpu) {
  args->tsCpu = NCCL_PROFILER_TS_NONE;
  args->tsCpuUncertainty = 0;
  if (kind == profilerTimestampHost) {
    args->tsCpu = profilerCpuNow(mode);
  } else if (kind == profilerTimestampGpu && mode == profilerTsModeGet() && PROF_TS_MODE_GPUSYNC(mode)) {
    profilerMapGpuToCpu(tsGpu, &args->tsCpu, &args->tsCpuUncertainty);
  }
}

// count==1 preserves the legacy fast path: ctx is the plugin's own context,
// including a legal NULL context, and callers already gate delivery with the
// process-wide mask (including a RAS override). count>1 uses NCCL's own
// ncclProfilerMultiContext. The process-wide mask is then only the union, so
// each slot must be gated against its own mask, or the RAS override replacing
// it, before remapping parent handles and delivering the event.
static void profilerStartDispatch(void* ctx, struct ncclProfilerEventRef* out, ncclProfilerEventDescr_t* descr,
                                  const struct ncclProfilerEventRef* parent,
                                  const struct ncclProfilerEventRef* parentGroup, profilerTimestampKind timestampKind,
                                  uint64_t tsGpu) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1)) return;
  memset(out, 0, sizeof(*out));
  if (COMPILER_EXPECT(profilerPluginCount <= 1, 1)) {
    profilerStampDescr(descr, profilerTsModeGet(), timestampKind, tsGpu);
    profilerSetParents(descr, parent, parentGroup, 0);
    profilerActivate(&profilerPlugins[0])->startEvent(ctx, &out->h[0], descr);
    return;
  }
  if (ctx == NULL) return;
  struct ncclProfilerMultiContext* multi = (struct ncclProfilerMultiContext*)ctx;
  int gate = profilerEventGate(descr->type);
  bool ras = COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideActive, std::memory_order_acquire);
  int rasMask = ras ? COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideMask, std::memory_order_relaxed) : 0;
  for (int slot = 0; slot < multi->count; slot++) {
    if (!multi->slots[slot].initialized) continue;
    int mask = ras ? rasMask : profilerPluginEffectiveMask(&profilerPlugins[slot]);
    if (!(mask & gate)) continue;
    ncclProfilerEventDescr_t slotDescr = *descr;
    profilerStampDescr(&slotDescr, COMPILER_ATOMIC_LOAD(&profilerPlugins[slot].tsMode, std::memory_order_relaxed),
                       timestampKind, tsGpu);
    profilerSetParents(&slotDescr, parent, parentGroup, slot);
    profilerActivate(&profilerPlugins[slot])->startEvent(multi->slots[slot].context, &out->h[slot], &slotDescr);
  }
}

static void profilerStart(void* ctx, struct ncclProfilerEventRef* out, ncclProfilerEventDescr_t* descr,
                          const struct ncclProfilerEventRef* parent, const struct ncclProfilerEventRef* parentGroup) {
  profilerStartDispatch(ctx, out, descr, parent, parentGroup, profilerTimestampHost, 0);
}

static void profilerStartNone(void* ctx, struct ncclProfilerEventRef* out, ncclProfilerEventDescr_t* descr,
                              const struct ncclProfilerEventRef* parent,
                              const struct ncclProfilerEventRef* parentGroup) {
  profilerStartDispatch(ctx, out, descr, parent, parentGroup, profilerTimestampNone, 0);
}

static void profilerStartGpu(void* ctx, struct ncclProfilerEventRef* out, ncclProfilerEventDescr_t* descr,
                             const struct ncclProfilerEventRef* parent, const struct ncclProfilerEventRef* parentGroup,
                             uint64_t tsGpu) {
  profilerStartDispatch(ctx, out, descr, parent, parentGroup, profilerTimestampGpu, tsGpu);
}

static void profilerStopDispatch(struct ncclProfilerEventRef* event, profilerTimestampKind timestampKind,
                                 uint64_t tsGpu) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1)) return;
  if (COMPILER_EXPECT(profilerPluginCount <= 1, 1)) {
    if (event->h[0]) {
      ncclProfilerEventDescr_t descr = {};
      profilerStampDescr(&descr, profilerTsModeGet(), timestampKind, tsGpu);
      profilerActivate(&profilerPlugins[0])->stopEvent(event->h[0], &descr);
    }
  } else {
    for (int slot = 0; slot < profilerPluginCount; slot++) {
      if (event->h[slot]) {
        ncclProfilerEventDescr_t descr = {};
        profilerStampDescr(&descr, COMPILER_ATOMIC_LOAD(&profilerPlugins[slot].tsMode, std::memory_order_relaxed),
                           timestampKind, tsGpu);
        profilerActivate(&profilerPlugins[slot])->stopEvent(event->h[slot], &descr);
      }
    }
  }
  memset(event, 0, sizeof(*event));
}

static void profilerStop(struct ncclProfilerEventRef* event) {
  profilerStopDispatch(event, profilerTimestampHost, 0);
}

static void profilerStopNone(struct ncclProfilerEventRef* event) {
  profilerStopDispatch(event, profilerTimestampNone, 0);
}

static void profilerStopGpu(struct ncclProfilerEventRef* event, uint64_t tsGpu) {
  profilerStopDispatch(event, profilerTimestampGpu, tsGpu);
}

static void profilerRecordDispatch(struct ncclProfilerEventRef* event, ncclProfilerEventState_t eState,
                                   ncclProfilerEventStateArgs_t* args, profilerTimestampKind timestampKind,
                                   uint64_t tsGpu) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1) || !profilerLive(event)) return;
  if (COMPILER_EXPECT(profilerPluginCount <= 1, 1)) {
    profilerStampArgs(args, profilerTsModeGet(), timestampKind, tsGpu);
    profilerActivate(&profilerPlugins[0])->recordEventState(event->h[0], eState, args);
    return;
  }
  for (int slot = 0; slot < profilerPluginCount; slot++) {
    if (event->h[slot]) {
      ncclProfilerEventStateArgs_t slotArgs = *args;
      profilerStampArgs(&slotArgs, COMPILER_ATOMIC_LOAD(&profilerPlugins[slot].tsMode, std::memory_order_relaxed),
                        timestampKind, tsGpu);
      profilerActivate(&profilerPlugins[slot])->recordEventState(event->h[slot], eState, &slotArgs);
    }
  }
}

static void profilerRecord(struct ncclProfilerEventRef* event, ncclProfilerEventState_t eState,
                           ncclProfilerEventStateArgs_t* args) {
  ncclProfilerEventStateArgs_t local = {};
  if (args == nullptr) args = &local;
  profilerRecordDispatch(event, eState, args, profilerTimestampHost, 0);
}

static void profilerRecordGpu(struct ncclProfilerEventRef* event, ncclProfilerEventState_t eState,
                              ncclProfilerEventStateArgs_t* args, uint64_t tsGpu) {
  ncclProfilerEventStateArgs_t local = {};
  if (args == nullptr) args = &local;
  profilerRecordDispatch(event, eState, args, profilerTimestampGpu, tsGpu);
}

// Net plugins start several events on one pHandle (IB: one per QP; socket: one
// per helper-thread socket). Each start must return a unique token.
// One plugin: pass through the plugin's own handle (same as before this work).
// Several plugins: malloc a boxed EventRef for the lifetime of that net event.
static void profilerStartNet(void* ctx, void** eHandle, ncclProfilerEventDescr_t* descr,
                             const struct ncclProfilerEventRef* parent) {
  struct ncclProfilerEventRef event = {};
  profilerStart(ctx, &event, descr, parent, nullptr);
  if (profilerPluginCount <= 1) {
    *eHandle = event.h[0];
    return;
  }
  if (!profilerLive(&event)) {
    *eHandle = nullptr;
    return;
  }
  struct ncclProfilerEventRef* box = (struct ncclProfilerEventRef*)malloc(sizeof(*box));
  if (box) *box = event;
  else profilerStop(&event);
  *eHandle = box;
}

static void profilerStopNet(void** eHandle) {
  if (!*eHandle) return;
  if (profilerPluginCount <= 1) {
    struct ncclProfilerEventRef event = {};
    event.h[0] = *eHandle;
    profilerStop(&event);
  } else {
    profilerStop((struct ncclProfilerEventRef*)*eHandle);
    free(*eHandle);
  }
  *eHandle = nullptr;
}

// A NULL eHandle is a net plugin one-off event (no prior start). It is forwarded to
// every loaded plugin's recordEventState as a NULL handle, as before multi-plugin support;
// plugins are responsible for ignoring or handling NULL handles
static void profilerRecordNet(void* eHandle, ncclProfilerEventState_t eState, ncclProfilerEventStateArgs_t* args) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1)) return;
  if (!eHandle) {
    for (int slot = 0; slot < profilerPluginCount; slot++) {
      ncclProfilerEventStateArgs_t slotArgs = *args;
      uint32_t mode = profilerPluginCount <= 1 ?
                        profilerTsModeGet() :
                        COMPILER_ATOMIC_LOAD(&profilerPlugins[slot].tsMode, std::memory_order_relaxed);
      profilerStampArgs(&slotArgs, mode, profilerTimestampHost, 0);
      // profilerRecord expects a non-NULL eHandle (it drops events whose handle is not live),
      // so call each plugin's recordEventState directly here.
      profilerActivate(&profilerPlugins[slot])->recordEventState(nullptr, eState, &slotArgs);
    }
    return;
  }
  if (profilerPluginCount <= 1) {
    struct ncclProfilerEventRef event = {};
    event.h[0] = eHandle;
    profilerRecord(&event, eState, args);
  } else {
    profilerRecord((struct ncclProfilerEventRef*)eHandle, eState, args);
  }
}

static ncclResult_t profilerInitMulti(void** output, uint64_t commId, const char* commName, int nNodes, int nranks,
                                      int rank, ncclProfilerConfig_t* commConfig) {
  struct ncclProfilerMultiContext* context = (struct ncclProfilerMultiContext*)calloc(1, sizeof(*context));
  if (!context) return ncclSystemError;
  context->count = profilerPluginCount;
  int initialized = 0;
  bool configStored = false;
  for (int slot = 0; slot < context->count; slot++) {
    struct ncclProfilerPlugin* plugin = &profilerPlugins[slot];
    int oldMask = COMPILER_ATOMIC_LOAD(plugin->maskPtr, std::memory_order_relaxed);
    ncclProfilerConfig_t config = {};
    int err = profilerActivate(plugin)->init(&context->slots[slot].context, commId, plugin->maskPtr, commName, nNodes,
                                             nranks, rank, ncclDebugLog, &config);
    if (err) {
      COMPILER_ATOMIC_STORE(plugin->maskPtr, oldMask, std::memory_order_relaxed);
      WARN("Profiler %s init failed with error '%d'; continuing without it", plugin->api.name, err);
      continue;
    }
    context->slots[slot].initialized = true;
    initialized++;
    if (!configStored) {
      *commConfig = config;
      configStored = true;
    }
    profilerApplyPluginConfig(plugin, &config);
    profilerApplyMaskDeps(plugin);
  }
  if (initialized == 0) {
    free(context);
    return ncclInternalError;
  }
  *output = context;
  return ncclSuccess;
}

static void profilerFinalizeMulti(void* opaque) {
  struct ncclProfilerMultiContext* context = (struct ncclProfilerMultiContext*)opaque;
  if (!context) return;
  for (int slot = 0; slot < context->count; slot++) {
    // Preserve the pre-multi-plugin contract: successful init with a NULL
    // context does not receive a finalize callback.
    if (context->slots[slot].initialized && context->slots[slot].context)
      profilerActivate(&profilerPlugins[slot])->finalize(context->slots[slot].context);
  }
  free(context);
}

void ncclProfilerSetRasOverride(int mask) {
  // A RAS-set mask must satisfy the same KernelPhase->KernelCh dependency as any other.
  // It must also preserve proxy ancestors when a legacy plugin requests NetPlugin.
  bool legacyNetPlugin = COMPILER_ATOMIC_LOAD(&profilerLegacyNetPlugin, std::memory_order_relaxed);
  mask = ncclProfilerEnforceMaskDeps(mask, "RAS override", legacyNetPlugin);
  // Store the 'active' flag (release) LAST, pairing with the acquire-load in ncclProfilerPluginInit so both
  // mask writes are visible there.
  COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, mask, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&ncclProfilerRasOverrideMask, mask, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&ncclProfilerRasOverrideActive, true, std::memory_order_release);
  // Override is recorded above regardless; only log the decoded mask when a plugin is loaded.
  if (ncclProfiler != NULL) {
    INFO(NCCL_INIT, "Profiler event mask set out-of-band via RAS:");
    printProfilerEventMask(mask, nullptr);
  }
}

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  TIME_START_EVENT(elapsed);
  TIME_START_EVENT(init);
  NCCLCHECK(ncclProfilerPluginLoad());
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // Feature handshake: NCCL supplies the defaults (all zero), the plugin
    // writes back the toggles it wants. A pre-v8 plugin never sees this (its
    // adapter drops the argument), so it keeps every default.
    ncclProfilerConfig_t config = {};
    int err;
    if (profilerPluginCount == 1) {
      err = profilerActivate(&profilerPlugins[0])
              ->init(&comm->profilerContext, comm->commHash, profilerPlugins[0].maskPtr, comm->config.commName,
                     comm->nNodes, comm->nRanks, comm->rank, ncclDebugLog, &config);
      if (!err) {
        profilerApplyPluginConfig(&profilerPlugins[0], &config);
        profilerApplyMaskDeps(&profilerPlugins[0]);
      }
    } else {
      err = profilerInitMulti(&comm->profilerContext, comm->commHash, comm->config.commName, comm->nNodes, comm->nRanks,
                              comm->rank, &config);
    }
    if (err) {
      ncclProfilerPluginUnload();
      ATTN("Profiler init failed with error '%d': %s. Continue without profiler.", err, strerror(errno));
    } else {
      comm->profiler.config = config;
      comm->profiler.contextValid = true;
      profilerRefreshEventMask();
      for (int i = 0; i < profilerPluginCount; i++) {
        printProfilerEventMask(profilerPluginEffectiveMask(&profilerPlugins[i]), profilerPlugins[i].api.name);
      }
    }
    printProfilerEventMask(ncclProfilerEventMask, nullptr);
  }
  // Re-apply any RAS override: the plugin's init() above rewrites the mask unconditionally and would
  // otherwise clobber a value set job-wide via the RAS client.
  if (COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideActive, std::memory_order_acquire)) {
    int mask = COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideMask, std::memory_order_relaxed);
    // RAS may set this mask before the plugin is loaded. Recheck dependencies using the loaded API version.
    bool legacyNetPlugin = COMPILER_ATOMIC_LOAD(&profilerLegacyNetPlugin, std::memory_order_relaxed);
    mask = ncclProfilerEnforceMaskDeps(mask, "RAS override", legacyNetPlugin);
    COMPILER_ATOMIC_STORE(&ncclProfilerRasOverrideMask, mask, std::memory_order_relaxed);
    COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, mask, std::memory_order_relaxed);
    INFO(NCCL_INIT, "Profiler event mask: re-applied RAS override 0x%x", mask);
  }
  TIME_STOP_EVENT(init);
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  TIME_START_EVENT(finalize);
  // Only finalize a context this comm actually created: init() may have failed here
  // while another comm keeps the plugin loaded. Clearing the flag also keeps a second
  // finalize from double-freeing the multi-plugin context or releasing its plugin
  // reference twice.
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && comm->profiler.contextValid) {
    comm->profiler.contextValid = false;
    if (profilerPluginCount == 1) {
      // Legacy plugins expect NCCL to skip finalize after a successful init
      // that returned a NULL context.
      if (comm->profilerContext) profilerActivate(&profilerPlugins[0])->finalize(comm->profilerContext);
    } else {
      profilerFinalizeMulti(comm->profilerContext);
    }
    comm->profilerContext = nullptr;
    profilerRefreshEventMask();
    ncclProfilerPluginUnload();
  }
  TIME_STOP_EVENT(finalize);
  TIME_STOP_EVENT(elapsed);
  TIME_PRINT_EVENTS("Profiler");
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = {0};
  eDescr.type = ncclProfileGroupApi;
  eDescr.rank = info->comm->rank;
  eDescr.groupApi.graphCaptured = isGraphCaptured;

  ncclProfilerApiState.eActivationMask = ncclProfilerCurrentEventMask();

  // Only count outermost groups when emitting group API events
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) &&
      (ncclProfilerApiState.eActivationMask & profilerEventGate(ncclProfileGroupApi))) {
    if (ncclProfilerApiState.profilerGroupDepth == 0) {
      eDescr.groupApi.groupDepth = ncclGroupDepth;
      profilerStart(info->comm->profilerContext, &ncclProfilerApiState.groupApiEventHandle, &eDescr, nullptr, nullptr);
      ncclProfilerApiState.profilerGroupDepth = ncclGroupDepth;
      ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupApiEvent() {
  if (ncclProfilerApiState.profilerGroupDepth == 0) profilerStop(&ncclProfilerApiState.groupApiEventHandle);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  bool shouldRecord = false;
  if (eState == ncclProfilerGroupStartApiStop && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStarted) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStopped;
    shouldRecord = true;
  } else if (eState == ncclProfilerGroupEndApiStart &&
             ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStopped) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateReset;
    shouldRecord = true;
  }
  if (shouldRecord) profilerRecord(&ncclProfilerApiState.groupApiEventHandle, eState, NULL);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = {0};
  eDescr.type = ncclProfileP2pApi;
  eDescr.rank = info->comm->rank;
  eDescr.p2pApi.func = ncclFuncToString(info->coll);
  eDescr.p2pApi.count = info->count;
  eDescr.p2pApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.p2pApi.stream = (void*)info->stream;
  eDescr.p2pApi.graphCaptured = isGraphCaptured;
  eDescr.p2pApi.userTag = info->collConfig.userProfilerTag;
  ncclProfilerApiState.eActivationMask = ncclProfilerCurrentEventMask();
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) &&
      (ncclProfilerApiState.eActivationMask & profilerEventGate(ncclProfileP2pApi))) {
    profilerStart(info->comm->profilerContext, &ncclProfilerApiState.p2pApiEventHandle, &eDescr,
                  &ncclProfilerApiState.groupApiEventHandle, nullptr);
  }
  if (isGraphCaptured && info->comm->config.graphUsageMode == 0) {
    INFO(NCCL_P2P | NCCL_ENV,
         "Comm config graphUsageMode is set to %d but the user is capturing graphs on the stream. Violating "
         "graphUsageMode semantics can lead to hangs!",
         info->comm->config.graphUsageMode);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopP2pApiEvent() {
  profilerStop(&ncclProfilerApiState.p2pApiEventHandle);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = {0};
  eDescr.type = ncclProfileCollApi;
  eDescr.rank = info->comm->rank;
  eDescr.collApi.func = ncclFuncToString(info->coll);
  eDescr.collApi.count = info->count;
  eDescr.collApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.collApi.stream = (void*)info->stream;
  eDescr.collApi.root = info->root;
  eDescr.collApi.graphCaptured = isGraphCaptured;
  eDescr.collApi.userTag = info->collConfig.userProfilerTag;
  ncclProfilerApiState.eActivationMask = ncclProfilerCurrentEventMask();
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) &&
      (ncclProfilerApiState.eActivationMask & profilerEventGate(ncclProfileCollApi))) {
    profilerStart(info->comm->profilerContext, &ncclProfilerApiState.collApiEventHandle, &eDescr,
                  &ncclProfilerApiState.groupApiEventHandle, nullptr);
  }
  if (isGraphCaptured && info->comm->config.graphUsageMode == 0) {
    INFO(NCCL_COLL | NCCL_ENV,
         "Comm config graphUsageMode is set to %d but the user is capturing graphs on the stream. Violating "
         "graphUsageMode semantics can lead to hangs!",
         info->comm->config.graphUsageMode);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCollApiEvent() {
  profilerStop(&ncclProfilerApiState.collApiEventHandle);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclProfilerEventDescr_t eDescr = {0};
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    const struct ncclProfilerEventRef* groupParent = nullptr;
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        groupParent = &ct->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      ct = ct->next;
    }
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        groupParent = &pt->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      pt = pt->next;
    }

  startKernelLaunchEvent:
    if (eActivationMask_ & ncclProfileKernelLaunch) {
      eDescr.type = ncclProfileKernelLaunch;
      eDescr.rank = plan->comm->rank;
      eDescr.kernelLaunch.stream = (void*)stream;
      profilerStart(plan->comm->profilerContext, &plan->kernelLaunchEventHandle, &eDescr, groupParent, nullptr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  profilerStop(&plan->kernelLaunchEventHandle);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStart);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        goto startGroup;
      }
      ct = ct->next;
    }
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        goto startGroup;
      }
      pt = pt->next;
    }

  startGroup:
    if (eActivationMask_ & profilerEventGate(ncclProfileGroup)) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileGroup;
      profilerStart(plan->comm->profilerContext, &plan->groupEventHandle, &eDescr, nullptr, nullptr);
    }
  }
  TIME_STOP_EVENT(groupStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStop);
  profilerStop(&plan->groupEventHandle);
  TIME_STOP_EVENT(groupStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStart);
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
      int enable = ct->eActivationMask & profilerEventGate(ncclProfileColl);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = {0};
        eDescr.type = ncclProfileColl;
        eDescr.rank = plan->comm->rank;
        eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func];
        eDescr.coll.func = ncclFuncToString(ct->func);
        eDescr.coll.sendBuff = ct->sendbuff;
        eDescr.coll.recvBuff = ct->recvbuff;
        eDescr.coll.count = ct->count;
        eDescr.coll.root = ct->root;
        eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);
        eDescr.coll.nChannels = ct->nChannels;
        eDescr.coll.nWarps = ct->nWarps;
        eDescr.coll.isSymColl = plan->isSymColl;
        if (plan->isSymColl) {
          const char* variant = ncclSymkKernelIdToString(static_cast<int>(ct->devFuncId));
          eDescr.coll.kernelVariant = variant;
          const char* underscore = strchr(variant, '_');
          eDescr.coll.algo = underscore ? underscore + 1 : variant;
          eDescr.coll.proto = "";
        } else {
          eDescr.coll.algo = ncclAlgoToString(ct->algorithm);
          eDescr.coll.proto = ncclProtoToString(ct->protocol);
          eDescr.coll.kernelVariant = nullptr;
        }
        eDescr.coll.userTag = ct->profilerTag;
        profilerStart(plan->comm->profilerContext, &ct->eventHandle, &eDescr, &ct->collApiEventHandle,
                      &plan->groupEventHandle);
      }
    }
    // seqNumber is updated even if the plugin is not active; RAS uses it too.
    // Skip persistent (graph-captured) collectives: this function may not run
    // on every rank, which would mismatch counters and false-positive in RAS.
    // KernelCh is the exception because it forces proxy events on all ranks.
    if (!plan->persistent || (COMPILER_EXPECT(ncclProfiler != NULL, 0) &&
                              (profilerLive(&plan->groupEventHandle) || profilerLive(&ct->collApiEventHandle)) &&
                              (ct->eActivationMask & ncclProfileKernelCh)))
      COMPILER_ATOMIC_FETCH_ADD(&plan->comm->seqNumber[ct->func], 1ULL, std::memory_order_relaxed);
    ct = ct->next;
  }
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      int enable = pt->eActivationMask & profilerEventGate(ncclProfileP2p);
      uint8_t profNChannels = pt->nChannels;
      if (pt->eActivationMask & ncclProfileKernelCh) profNChannels = (uint8_t)countOneBits(pt->channelMask);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = {0};
        eDescr.type = ncclProfileP2p;
        eDescr.rank = plan->comm->rank;
        eDescr.p2p.func = ncclFuncToString(pt->func);
        eDescr.p2p.buff = pt->buff;
        eDescr.p2p.count = pt->count;
        eDescr.p2p.datatype = ncclDatatypeToString(pt->datatype);
        eDescr.p2p.peer = pt->root;
        eDescr.p2p.nChannels = profNChannels;
        eDescr.p2p.userTag = pt->profilerTag;
        profilerStart(plan->comm->profilerContext, &pt->eventHandle, &eDescr, &pt->p2pApiEventHandle,
                      &plan->groupEventHandle);
      }
      pt = pt->next;
    }
  }
  TIME_STOP_EVENT(taskStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStop);
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    profilerStop(&ct->eventHandle);
    ct = ct->next;
  }
  struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  while (pt) {
    profilerStop(&pt->eventHandle);
    pt = pt->next;
  }
  TIME_STOP_EVENT(taskStop);
  return ncclSuccess;
}

// Bellow we set the proxy descriptor step number to DIVUP(step, args->sliceSteps).
// The reason is that for some ncclOp (e.g. AllReduce) one network transfer is
// made of sliceSteps steps rather than one step. In the profiler we are still
// interested in whole network transfers though, so we account for this when
// computing the actual network step number.
ncclResult_t ncclProfilerStartProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (sub->eActivationMask & profilerEventGate(ncclProfileProxyOp))) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileProxyOp;
    eDescr.rank = sub->rank;
    eDescr.proxyOp.pid = sub->pid;
    eDescr.proxyOp.channelId = sub->channelId;
    eDescr.proxyOp.peer = sub->peer;
    eDescr.proxyOp.nSteps = DIVUP(sub->nsteps, args->sliceSteps);
    eDescr.proxyOp.chunkSize = args->chunkSize * args->sliceSteps;
    eDescr.proxyOp.isSend = args->progress == ncclTransports[TRANSPORT_NET]->send.proxyProgress ? 1 : 0;
    profilerStart(sub->profilerContext, &sub->opEventHandle, &eDescr, &sub->taskEventHandle, nullptr);
  }
  TIME_STOP_EVENT(proxyOpStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStop);
  profilerStop(&args->subs[s].opEventHandle);
  TIME_STOP_EVENT(proxyOpStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  sub->pHandles[step_ % NCCL_STEPS].stepEventHandle = {};
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (sub->eActivationMask & profilerEventGate(ncclProfileProxyStep))) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileProxyStep;
    eDescr.rank = sub->rank;
    eDescr.proxyStep.step = step_;
    profilerStart(sub->profilerContext, &sub->pHandles[step_ % NCCL_STEPS].stepEventHandle, &eDescr,
                  &sub->opEventHandle, nullptr);
  }
  sub->pHandles[step_ % NCCL_STEPS].subArgPtr = sub;
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  sub->pHandles[step_ % NCCL_STEPS].stepEventHandle = {};
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (sub->eActivationMask & profilerEventGate(ncclProfileProxyStep))) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileProxyStep;
    eDescr.rank = sub->rank;
    eDescr.proxyStep.step = step_;
    profilerStart(sub->profilerContext, &sub->pHandles[step_ % NCCL_STEPS].stepEventHandle, &eDescr,
                  &sub->opEventHandle, nullptr);
  }
  sub->pHandles[step_ % NCCL_STEPS].subArgPtr = sub;
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStop);
  int step_ = DIVUP(stepId, args->sliceSteps);
  profilerStop(&args->subs[s].pHandles[step_ % NCCL_STEPS].stepEventHandle);
  TIME_STOP_EVENT(proxyStepStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, struct ncclProfilerEventRef* eHandle) {
  TIME_START_EVENT(proxyCtrlStart);
  memset(eHandle, 0, sizeof(*eHandle));
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (profilerRefreshEventMask() & ncclProfileProxyCtrl)) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileProxyCtrl;
    profilerStart(profilerContext, eHandle, &eDescr, nullptr, nullptr);
  }
  TIME_STOP_EVENT(proxyCtrlStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyCtrlEvent(struct ncclProfilerEventRef* eHandle) {
  TIME_START_EVENT(proxyCtrlStop);
  profilerStop(eHandle);
  TIME_STOP_EVENT(proxyCtrlStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProfilerWorkOp* op, uint64_t start) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (op->eActivationMask & ncclProfileKernelCh)) {
    ncclProfilerEventDescr_t eDescr = {};
    eDescr.type = ncclProfileKernelCh;
    eDescr.kernelCh.channelId = op->channelId;
    eDescr.kernelCh.pTimer = start;
    profilerStartGpu(op->profilerContext, &op->kernelEventHandle, &eDescr, &op->taskEventHandle, nullptr, start);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProfilerWorkOp* op, uint64_t stop) {
  if (profilerLive(&op->kernelEventHandle)) {
    ncclProfilerEventStateArgs_t a = {};
    a.kernelCh.pTimer = stop;
    profilerRecordGpu(&op->kernelEventHandle, ncclProfilerKernelChStop, &a, stop);
    profilerStopGpu(&op->kernelEventHandle, stop);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int s, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyOpRecord);
  ncclProfilerEventStateArgs_t a = {};
  profilerRecord(&args->subs[s].opEventHandle, eState, &a);
  TIME_STOP_EVENT(proxyOpRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int s, struct ncclProxyArgs* args, int stepId,
                                                   ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyStepRecord);
  int step_ = DIVUP(stepId, args->sliceSteps);
  ncclProfilerEventStateArgs_t a = {};
  a.proxyStep.transSize = args->subs[s].transSize;
  profilerRecord(&args->subs[s].pHandles[step_ % NCCL_STEPS].stepEventHandle, eState, &a);
  TIME_STOP_EVENT(proxyStepRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(struct ncclProfilerEventRef* eHandle, int appended,
                                                   ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyCtrlRecord);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (profilerRefreshEventMask() & ncclProfileProxyCtrl)) {
    ncclProfilerEventStateArgs_t args = {};
    args.proxyCtrl.appendedProxyOps = appended;
    profilerRecord(eHandle, eState, &args);
  }
  TIME_STOP_EVENT(proxyCtrlRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerAddPidToProxyOp(struct ncclProxyOp* op) {
  op->pid = pid;
  return ncclSuccess;
}

// Caller must hold pt->mutex.
static struct ncclProfilerWorkOp* profilerAllocOp(struct ncclProfilerThread* pt) {
  return ncclMemoryPoolAlloc<struct ncclProfilerWorkOp>(&pt->opPool, &pt->opStack);
}

// Caller must hold pt->mutex.
static void profilerRecycleList(struct ncclProfilerThread* pt, struct ncclProfilerWorkOp* head) {
  while (head) {
    struct ncclProfilerWorkOp* next = head->next;
    profilerStop(&head->kernelEventHandle);
    ncclMemoryPoolFree(&pt->opPool, head);
    if (pt->inflight > 0) pt->inflight--;
    head = next;
  }
}

static const char* profilerPhaseNames[] = {"initial_sync", "compute", "final_sync"};

static void profilerFireKernelPhaseEvents(struct ncclProfilerWorkOp* op) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1)) return;
  if (!(op->eActivationMask & ncclProfileKernelPhase)) return;
  if (op->workPhases == nullptr) return;

  int ch = op->channelId;
  uint64_t wc = op->workCounter;
  int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
  // Several tasks in a sym plan can share this channel's device record. Fold
  // boundaries in a local copy so every task can report the same phase interval.
  uint64_t ts[MAX_PROFILER_PHASES];
  memcpy(ts, op->workPhases[ch].data[slot].timestamps, sizeof(ts));

  // Fold a missing intermediate boundary onto the adjacent outer one; if both are
  // missing the kernel didn't instrument phases, so skip rather than mislabel.
  bool haveAfterOpen = ts[NCCL_KERNEL_PHASE_AFTER_OPEN] != 0;
  bool haveBeforeClose = ts[NCCL_KERNEL_PHASE_BEFORE_CLOSE] != 0;
  if (!haveAfterOpen && !haveBeforeClose) return;
  if (!haveAfterOpen && ts[NCCL_KERNEL_PHASE_BEGIN] != 0)
    ts[NCCL_KERNEL_PHASE_AFTER_OPEN] = ts[NCCL_KERNEL_PHASE_BEGIN];
  if (!haveBeforeClose && ts[NCCL_KERNEL_PHASE_END] != 0)
    ts[NCCL_KERNEL_PHASE_BEFORE_CLOSE] = ts[NCCL_KERNEL_PHASE_END];

  for (int i = 0; i < MAX_PROFILER_PHASES - 1; i++) {
    if (ts[i] == 0 || ts[i + 1] == 0) continue;
    if (ts[i] == ts[i + 1]) continue; // skip zero-duration (synthesized) boundary
    struct ncclProfilerEventRef phaseHandle = {};
    ncclProfilerEventDescr_t eDescr = {};
    eDescr.type = ncclProfileKernelPhase;
    eDescr.kernelPhase.channelId = op->channelId;
    eDescr.kernelPhase.phaseId = i;
    eDescr.kernelPhase.phaseName = profilerPhaseNames[i];
    eDescr.kernelPhase.pTimer = ts[i];
    profilerStartGpu(op->profilerContext, &phaseHandle, &eDescr, &op->kernelEventHandle, nullptr, ts[i]);
    if (profilerLive(&phaseHandle)) {
      ncclProfilerEventStateArgs_t a = {};
      a.kernelCh.pTimer = ts[i + 1];
      profilerRecordGpu(&phaseHandle, ncclProfilerKernelPhaseStop, &a, ts[i + 1]);
      profilerStopGpu(&phaseHandle, ts[i + 1]);
    }
  }
}

// Runs without pt->mutex held: plugin callbacks may block, so we mustn't
// stall posters. Completed ops are returned as a local list and recycled by
// the caller under lock; *outNewTail receives the new tail of pt->active.
//
// Concurrency: caller sets pt->iterationActive=true under mutex before
// invoking this; ncclProfilerThreadDestroy waits on condIterationInactive
// until cleanupAndStop clears it. That ordering keeps a comm's
// profilerContext alive across the plugin callbacks below.
static struct ncclProfilerWorkOp* profilerProgressOps(struct ncclProfilerThread* pt,
                                                      struct ncclProfilerWorkOp** outNewTail) {
  struct ncclProfilerWorkOp* recycled = nullptr;
  struct ncclProfilerWorkOp* prev = nullptr;
  struct ncclProfilerWorkOp* op = pt->active;
  while (op) {
    struct ncclProfilerWorkOp* next = op->next;
    int ch = op->channelId;
    uint64_t wc = op->workCounter;
    int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;

    // Use `<=` not `==`: the device wraps around MAX_PROFILER_EVENTS_PER_CHANNEL
    // slots, so if the host falls behind the kernel may already have overwritten
    // this slot with a counter > wc (matches the old proxy progress check).
    if (!op->started && wc <= op->workStarted[ch].data[slot].counter) {
      ncclProfilerStartKernelChEvent(op, op->workStarted[ch].data[slot].timestamp);
      op->started = true;
    }
    // Also wait for the phases counter so phase child events fire under a live
    // kernelCh parent. `<=` for the same slot-wraparound reason as above.
    if (op->started && !op->completed && wc <= op->workCompleted[ch].data[slot].counter) {
      bool phasesReady = !(op->eActivationMask & ncclProfileKernelPhase) || op->workPhases == nullptr ||
                         wc <= op->workPhases[ch].data[slot].counter;
      if (phasesReady) {
        profilerFireKernelPhaseEvents(op);
        ncclProfilerStopKernelChEvent(op, op->workCompleted[ch].data[slot].timestamp);
        op->completed = true;
      }
    }
    if (op->completed) {
      if (prev) prev->next = next;
      else pt->active = next;
      op->next = recycled;
      recycled = op;
    } else {
      prev = op;
    }
    op = next;
  }
  *outNewTail = prev;
  return recycled;
}

// Caller must hold pt->mutex. O(1) splice via pt->activeTail.
static inline void appendWorkToActiveQueue(struct ncclProfilerThread* pt) {
  if (pt->pending == nullptr) return;
  if (pt->activeTail) {
    pt->activeTail->next = pt->pending;
  } else {
    pt->active = pt->pending;
  }
  pt->activeTail = pt->pendingTail;
  pt->pending = nullptr;
  pt->pendingTail = nullptr;
}

// Block until there is work or a stop/abort request. On exit, pending has
// been folded into active and pt->iterationActive is set to true (paired
// with cleanupAndStop) when the returned action is PROGRESS/CLEANUP_AND_STOP.
static inline ncclProfilerThreadAction waitForAction(struct ncclProfilerThread* pt, bool* outStop) {
  std::unique_lock<std::mutex> lock(pt->mutex);
  bool aborted = pt->abortFlag && COMPILER_ATOMIC_LOAD(pt->abortFlag, std::memory_order_relaxed);
  while (pt->pending == nullptr && pt->active == nullptr && !pt->stop && !aborted) {
    pt->cond.wait(lock);
    aborted = pt->abortFlag && COMPILER_ATOMIC_LOAD(pt->abortFlag, std::memory_order_relaxed);
  }
  bool stop = pt->stop;
  *outStop = stop;
  appendWorkToActiveQueue(pt);
  if ((stop || aborted) && pt->active == nullptr) return NCCL_PROFILER_THREAD_STOP;
  pt->iterationActive = true;
  if ((stop || aborted) && pt->active != nullptr) return NCCL_PROFILER_THREAD_CLEANUP_AND_STOP;
  return NCCL_PROFILER_THREAD_PROGRESS;
}

// Recycle completed ops, publish new active tail, and (when stopping)
// drain ops whose kernels will never run. Clears iterationActive so
// ncclProfilerThreadDestroy can proceed. Returns true to exit the thread.
static inline bool cleanupAndStop(struct ncclProfilerThread* pt, struct ncclProfilerWorkOp* recycled,
                                  struct ncclProfilerWorkOp* newActiveTail, bool drainStuck) {
  std::lock_guard<std::mutex> lock(pt->mutex);
  profilerRecycleList(pt, recycled);
  pt->activeTail = newActiveTail;
  if (drainStuck) {
    struct ncclProfilerWorkOp* stuck = pt->active;
    pt->active = nullptr;
    pt->activeTail = nullptr;
    profilerRecycleList(pt, stuck);
  }
  pt->iterationActive = false;
  pt->condIterationInactive.notify_all();
  return pt->active == nullptr;
}

// Keep KernelCh lag bounded: reset after any completed op and only back off
// briefly when the active head has not completed yet.
static inline unsigned updateProgressInterval(struct ncclProfilerThread* pt, unsigned spinUs, bool madeProgress) {
  if (pt->active == nullptr || madeProgress) return 1;
  return spinUs < 10 ? spinUs * 2 : 10;
}

static void* ncclProfilerThreadFunc(void* arg) {
  struct ncclProfilerThread* pt = (struct ncclProfilerThread*)arg;

  // The thread only reads host-pinned memory, but plugins it dispatches
  // into may make context-dependent driver calls; bind defensively.
  cudaError_t setDevErr = cudaSetDevice(pt->cudaDev);
  if (setDevErr != cudaSuccess) {
    INFO(NCCL_INIT | NCCL_PROFILE,
         "[Profiler Thread] cudaSetDevice(%d) failed: %s; continuing without explicit device binding", pt->cudaDev,
         cudaGetErrorString(setDevErr));
    (void)cudaGetLastError();
  }
  INFO(NCCL_INIT, "[Profiler Thread] started (cudaDev=%d, maxInflight=%zu, tsMode=0x%x)", pt->cudaDev, pt->maxInflight,
       profilerTsModeGet());

  // Clock calibration is set up on first use below rather than here: a comm
  // created after this thread started can be the one that opts into
  // gpuClockSync, so deciding once at startup would leave that comm mapping
  // GPU timestamps against a clockCalib that was never initialized.
  // calibSetup latches the one-time attempt so a failed init is not retried.
  bool calibSetup = false;

  unsigned spinUs = 1;
  while (true) {
    bool stop;
    ncclProfilerThreadAction action = waitForAction(pt, &stop);
    if (action == NCCL_PROFILER_THREAD_STOP) break;
    uint32_t tsMode = profilerTsModeGet();
    if (!calibSetup && PROF_TS_MODE_GPUSYNC(tsMode)) {
      calibSetup = true;
      // This thread launches its own GPU work (the clock probe kernel). Run it
      // in relaxed capture mode so that work stays invisible to app-thread
      // graph captures, which it would otherwise invalidate. Safe: the mode is
      // per-thread and this thread never captures.
      cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;
      if (cudaThreadExchangeStreamCaptureMode(&captureMode) != cudaSuccess) {
        (void)cudaGetLastError();
      }
      profilerClockCalibInit(&pt->clockCalib);
      profilerTlCalib = &pt->clockCalib;
    }
    if (pt->clockCalib.ready) profilerClockCalibMaybe(&pt->clockCalib, tsMode);

    struct ncclProfilerWorkOp* newActiveTail = nullptr;
    struct ncclProfilerWorkOp* recycled = profilerProgressOps(pt, &newActiveTail);
    bool madeProgress = (recycled != nullptr);

    bool drainStuck = (action == NCCL_PROFILER_THREAD_CLEANUP_AND_STOP);
    bool exitNow = cleanupAndStop(pt, recycled, newActiveTail, drainStuck);
    if (stop && exitNow) break;

    if (pt->active && !madeProgress) {
      std::this_thread::sleep_for(std::chrono::microseconds(spinUs));
    }
    spinUs = updateProgressInterval(pt, spinUs, madeProgress);
  }

  size_t hwm;
  uint64_t drops;
  {
    std::lock_guard<std::mutex> lock(pt->mutex);
    hwm = pt->maxInflightSeen;
    drops = pt->droppedOps;
  }
  if (drops != 0 || hwm != 0) {
    INFO(NCCL_INIT | NCCL_PROFILE, "[Profiler Thread] exiting: maxInflightSeen=%zu, droppedOps=%lu, cap=%zu", hwm,
         (unsigned long)drops, pt->maxInflight);
  }
  profilerClockCalibFree(&pt->clockCalib);
  profilerTlCalib = nullptr;
  return nullptr;
}

ncclResult_t ncclProfilerThreadCreate(struct ncclComm* comm, struct ncclComm* parent) {
  if (!ncclProfilerPluginLoaded()) return ncclSuccess;
  if (parent && parent->shareResources && parent->profiler.profilerThread) {
    comm->profiler.profilerThread = parent->profiler.profilerThread;
    std::lock_guard<std::mutex> lock(comm->profiler.profilerThread->mutex);
    comm->profiler.profilerThread->refCount++;
    return ncclSuccess;
  }

  struct ncclProfilerThread* pt = new ncclProfilerThread{};
  pt->stop = 0;
  pt->refCount = 1;
  pt->cudaDev = comm->cudaDev;
  pt->abortFlag = comm->abortFlag;
  pt->iterationActive = false;
  pt->pending = nullptr;
  pt->pendingTail = nullptr;
  pt->active = nullptr;
  pt->activeTail = nullptr;
  pt->inflight = 0;
  pt->maxInflightSeen = 0;
  pt->droppedOps = 0;
  pt->maxInflight = NCCL_PROFILER_DEFAULT_MAX_INFLIGHT;
  ncclMemoryStackConstruct(&pt->opStack);
  ncclMemoryPoolConstruct(&pt->opPool);
  pt->thread = std::thread(ncclProfilerThreadFunc, pt);
  comm->profiler.profilerThread = pt;
  return ncclSuccess;
}

// Splice out and recycle every op referencing `ctx`; returns new list tail.
// Caller must hold mutex.
static struct ncclProfilerWorkOp* profilerPurgeByContext(struct ncclProfilerThread* pt,
                                                         struct ncclProfilerWorkOp** list, void* ctx) {
  struct ncclProfilerWorkOp* lastKept = nullptr;
  struct ncclProfilerWorkOp** link = list;
  while (*link) {
    if ((*link)->profilerContext == ctx) {
      struct ncclProfilerWorkOp* dead = *link;
      *link = dead->next;
      profilerStop(&dead->kernelEventHandle);
      ncclMemoryPoolFree(&pt->opPool, dead);
      if (pt->inflight > 0) pt->inflight--;
    } else {
      lastKept = *link;
      link = &(*link)->next;
    }
  }
  return lastKept;
}

ncclResult_t ncclProfilerThreadDestroy(struct ncclComm* comm) {
  struct ncclProfilerThread* pt = comm->profiler.profilerThread;
  if (pt == nullptr) return ncclSuccess;
  bool shouldJoin = false;
  {
    std::unique_lock<std::mutex> lock(pt->mutex);
    // Wait out any in-flight progressOps before touching the queues:
    // ncclProfilerPluginFinalize destroys this comm's profilerContext right
    // after we return, and the thread may be mid-callback against it.
    while (pt->iterationActive) pt->condIterationInactive.wait(lock);

    pt->pendingTail = profilerPurgeByContext(pt, &pt->pending, comm->profilerContext);
    pt->activeTail = profilerPurgeByContext(pt, &pt->active, comm->profilerContext);

    pt->refCount--;
    if (pt->refCount == 0) {
      pt->stop = 1;
      shouldJoin = true;
    }
    pt->cond.notify_one();
  }
  if (shouldJoin) {
    if (pt->thread.joinable()) pt->thread.join();
    // opStack owns opPool's backing storage; destructing it frees both.
    ncclMemoryStackDestruct(&pt->opStack);
    delete pt;
  }
  comm->profiler.profilerThread = nullptr;
  return ncclSuccess;
}

// Enqueue one KernelCh op for (channelId, wc) against taskEventHandle. Does NOT
// advance workCounter: the caller owns the per-channel bump so both halves of a
// send/recv pair (which the device counts as one fused work per channel) can
// post against the same device slot.
static void profilerEnqueueOp(struct ncclProfilerThread* pt, struct ncclComm* comm, int channelId, uint64_t wc,
                              int eActivationMask, struct ncclProfilerEventRef taskEventHandle, bool sym) {
  bool dropped = false;
  uint64_t dropTotal = 0;
  size_t inflightAtDrop = 0;
  size_t maxInflightCap = 0;
  {
    std::lock_guard<std::mutex> lock(pt->mutex);
    struct ncclProfilerWorkOp* op = nullptr;
    op = profilerAllocOp(pt);
    if (op == nullptr) {
      pt->droppedOps++;
      dropped = true;
      dropTotal = pt->droppedOps;
      inflightAtDrop = pt->inflight;
      maxInflightCap = pt->maxInflight;
    } else {
      op->channelId = channelId;
      op->workCounter = wc;
      op->eActivationMask = eActivationMask;
      op->taskEventHandle = taskEventHandle;
      op->profilerContext = comm->profilerContext;
      // Sym collectives poll a dedicated buffer set (see ncclProfilerCommState).
      op->workStarted = sym ? comm->profiler.symWorkStarted : comm->profiler.workStarted;
      op->workCompleted = sym ? comm->profiler.symWorkCompleted : comm->profiler.workCompleted;
      // Null for regular/p2p: they never publish a phases counter, so the completion
      // check would otherwise wait on it forever.
      op->workPhases = sym ? comm->profiler.symWorkPhases : nullptr;
      op->kernelEventHandle = {};
      op->started = false;
      op->completed = false;
      op->next = nullptr;

      if (pt->pendingTail) {
        pt->pendingTail->next = op;
      } else {
        pt->pending = op;
      }
      pt->pendingTail = op;
      pt->inflight++;
      if (pt->inflight > pt->maxInflightSeen) pt->maxInflightSeen = pt->inflight;
      if (pt->inflight > pt->maxInflight && (pt->inflight & (pt->inflight - 1)) == 0) {
        INFO(NCCL_PROFILE,
             "Profiler: KernelCh inflight ops exceeded soft cap "
             "(inflight=%zu, cap=%zu). Check that the profiler plugin is draining events.",
             pt->inflight, pt->maxInflight);
      }
      pt->cond.notify_one();
    }
  }

  if (!dropped) return;

  // INFO on a power-of-two ramp so logs surface allocation failure without spam.
  if ((dropTotal & (dropTotal - 1)) == 0) {
    INFO(NCCL_PROFILE,
         "Profiler: dropping KernelCh op after allocation failure "
         "(dropped=%lu, inflight=%zu, softCap=%zu).",
         (unsigned long)dropTotal, inflightAtDrop, maxInflightCap);
  }
}

// Invoked from the captured hostStreamPlanCallback. Must:
//   * Always return ncclSuccess (errors on this path poison the captured
//     stream and desync the host workCounter from the kernel).
//   * Bump comm->profiler.workCounter[channelId] exactly once per call,
//     even on allocation failure, to stay in lock-step with the device kernel.
static ncclResult_t profilerPostWorkInternal(struct ncclComm* comm, int channelId, int eActivationMask,
                                             struct ncclProfilerEventRef taskEventHandle) {
  struct ncclProfilerThread* pt = comm->profiler.profilerThread;
  if (pt == nullptr || !(eActivationMask & ncclProfileKernelCh)) return ncclSuccess;
  uint64_t wc = ++comm->profiler.workCounter[channelId];
  profilerEnqueueOp(pt, comm, channelId, wc, eActivationMask, taskEventHandle, /*sym=*/false);
  return ncclSuccess;
}

// Reserve per-channel sym workCounters into the args buffer pre-launch, before the
// driver snapshots it at cuLaunchKernel (the device kernel reads them from there). The
// matching KernelCh ops are enqueued later by ncclProfilerPostPlanWork() from the host
// callback. No-op for the clean kernel (graph capture / profiling off). Uses the
// dedicated sym counters, which never share the regular/p2p sequence (ncclProfilerCommState).
static uint64_t* profilerGetSymCounters(struct ncclKernelPlan* plan, struct ncclTaskColl* task) {
  if ((1ull << task->devFuncId) & ncclGenkKernelMask()) {
    struct ncclGenkDevWorkArgs* args = (struct ncclGenkDevWorkArgs*)plan->kernelSymArgs;
    return args->profilerMode ? ncclSymkGetProfilerCounters(args) : nullptr;
  }
  struct ncclSymkDevWorkArgs* args = (struct ncclSymkDevWorkArgs*)plan->kernelSymArgs;
  return args->profilerMode ? ncclSymkGetProfilerCounters(args) : nullptr;
}

void ncclProfilerReserveSymCounters(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!ncclProfilerPluginLoaded() || comm->profiler.profilerThread == nullptr) return;
  if (!plan->isSymColl) return;
  struct ncclTaskColl* sct = ncclIntruQueueHead(&plan->collTaskQueue);
  if (sct == nullptr || !(sct->eActivationMask & ncclProfileKernelCh)) return;
  uint64_t* counters = profilerGetSymCounters(plan, sct);
  if (counters == nullptr) return;
  int nChannels = countOneBits(plan->channelMask);
  for (int c = 0; c < nChannels; c++) counters[c] = ++comm->profiler.symWorkCounter[c];
}

// Enqueue the KernelCh ops for a sym plan using the counters reserved pre-launch by
// ncclProfilerReserveSymCounters() (read back from the args buffer, not re-bumped).
// No-op for the clean kernel (graph capture / profiling off).
static void profilerPostPlanWorkSym(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclTaskColl* sct = ncclIntruQueueHead(&plan->collTaskQueue);
  if (sct == nullptr || !(sct->eActivationMask & ncclProfileKernelCh)) return;
  struct ncclProfilerThread* pt = comm->profiler.profilerThread;
  if (pt == nullptr) return;
  uint64_t* counters = profilerGetSymCounters(plan, sct);
  if (counters == nullptr) return;
  for (struct ncclTaskColl* ct = sct; ct != nullptr; ct = ct->next) {
    if (ct->eActivationMask & ncclProfileKernelCh) {
      for (int c = ct->channelLo; c <= ct->channelHi; c++) {
        // A shared channel still publishes only one record for the entire plan.
        // Reuse its counter for each task; these are shared, not per-task timings.
        profilerEnqueueOp(pt, comm, c, counters[c], ct->eActivationMask, ct->eventHandle, /*sym=*/true);
      }
    }
  }
}

// Posts must mirror the device kernel's per-channel workCounter advance
// exactly (`ncclShmem.channel.workCounter += ncclShmem.nWorks` in
// src/device/common.h::profiler). Per-task mapping:
//   * Coll: one bump per channel in [channelLo, channelHi].
//   * P2p:  one bump per channel in channelMask per addP2pToPlan() pair;
//           sibling tasks of a pair must not double-post. Multiple pairs
//           that overlap a channel each contribute one bump.
//   * Sym-coll: one bump per channel in [0, countOneBits(channelMask)), with
//     the counter mirrored into the kernel args buffer (see below).
//   * Bcast/RMA/CE: piggyback or no KernelCh.
// Drift breaks the slot-counter invariant in profilerProgressOps, leaks
// ops into pt->active, and (eventually) corrupts unrelated CUDA driver
// state by exhausting the op pool. Return value is discarded by callers.
ncclResult_t ncclProfilerPostPlanWork(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!ncclProfilerPluginLoaded() || !comm->profiler.profilerThread) return ncclSuccess;

  if (plan->isSymColl) {
    profilerPostPlanWorkSym(comm, plan);
    return ncclSuccess;
  }

  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    if (ct->eActivationMask & ncclProfileKernelCh) {
      for (int c = ct->channelLo; c <= ct->channelHi; c++) {
        (void)profilerPostWorkInternal(comm, c, ct->eActivationMask, ct->eventHandle);
      }
    }
    ct = ct->next;
  }

  // Both halves of a send/recv pair carry KernelCh over their own channels, as
  // before the thread decouple. The device counts the fused p2p work once per
  // channel, so the pair shares a single workCounter bump per channel and both
  // halves' ops read the same device slot. Siblings are enqueued back-to-back
  // (see scheduleP2pTasksToPlan) and share p2pPairId; PID 0 is the unassigned
  // sentinel, treated as a single-task pair.
  struct ncclProfilerThread* thr = comm->profiler.profilerThread;
  struct ncclTaskP2p* p2pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
  uint16_t lastPid = 0;
  while (p2pt) {
    uint16_t pid = p2pt->p2pPairId;
    bool isFirstOfPair = (pid == 0) || (pid != lastPid);
    lastPid = pid;
    if (isFirstOfPair) {
      struct ncclTaskP2p* t0 = p2pt;
      struct ncclTaskP2p* t1 = (pid != 0 && p2pt->next && p2pt->next->p2pPairId == pid) ? p2pt->next : nullptr;
      uint64_t m0 = (t0->eActivationMask & ncclProfileKernelCh) ? t0->channelMask : 0;
      uint64_t m1 = (t1 && (t1->eActivationMask & ncclProfileKernelCh)) ? t1->channelMask : 0;
      for (int c = 0; c < MAXCHANNELS; c++) {
        uint64_t bit = 1ull << c;
        if (!((m0 | m1) & bit)) continue;
        uint64_t wc = ++comm->profiler.workCounter[c];
        if (m0 & bit) profilerEnqueueOp(thr, comm, c, wc, t0->eActivationMask, t0->eventHandle, /*sym=*/false);
        if (m1 & bit) profilerEnqueueOp(thr, comm, c, wc, t1->eActivationMask, t1->eventHandle, /*sym=*/false);
      }
    }
    p2pt = p2pt->next;
  }

  return ncclSuccess;
}

bool ncclProfilerPluginLoaded(void) {
  return (COMPILER_EXPECT(ncclProfiler != NULL, 0));
}

uint8_t ncclProfilerDeviceMode(int eActivationMask) {
  if (!ncclProfilerPluginLoaded()) return ncclDevProfilerModeNone;

  uint8_t mode = ncclDevProfilerModeNone;
  if (eActivationMask & ncclProfileKernelCh) mode |= ncclDevProfilerModeKernelCh;
  if (eActivationMask & ncclProfileKernelPhase) {
    mode |= ncclDevProfilerModeKernelCh | ncclDevProfilerModeKernelPhase;
  }
  return mode;
}

ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (COMPILER_EXPECT(ncclProfiler == NULL, 1)) return ncclSuccess;
  if (type == ncclProfilerNetEventStart) {
    struct ncclProxyEventHandle* proxyEvent = (struct ncclProxyEventHandle*)pHandle;
    struct ncclProxySubArgs* sub = proxyEvent->subArgPtr;
    if (sub->eActivationMask & ncclProfileNetPlugin) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileNetPlugin;
      eDescr.rank = sub->rank;
      eDescr.netPlugin.id = pluginId;
      eDescr.netPlugin.data = extData;
      // NetPlugin parents remain ProxyStep handles; an unavailable parent is NULL.
      profilerStartNet(sub->profilerContext, eHandle, &eDescr, &proxyEvent->stepEventHandle);
    }
  } else if (type == ncclProfilerNetEventStop) {
    profilerStopNet(eHandle);
  } else if (type == ncclProfilerNetEventUpdate) {
    ncclProfilerEventStateArgs_t args = {};
    args.netPlugin.data = extData;
    profilerRecordNet(*eHandle, ncclProfilerNetPluginUpdate, &args);
  } else {
    ncclProfilerEventStateArgs_t args = {};
    args.netPlugin.data = extData;
    profilerRecordNet(*eHandle, ncclProfilerNetPluginUpdate, &args);
    profilerStopNet(eHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (profilerRefreshEventMask() & profilerEventGate(ncclProfileCeColl))) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileCeColl;
    eDescr.rank = comm->rank;
    eDescr.ceColl.seqNumber = comm->ceColl.ceSeqNum;
    eDescr.ceColl.func = ncclFuncToString(args->func);
    eDescr.ceColl.sendBuff = args->sendBuff;
    eDescr.ceColl.recvBuff = args->recvBuff;
    eDescr.ceColl.count = args->nElts;
    eDescr.ceColl.root = args->rootRank;
    eDescr.ceColl.datatype = ncclDatatypeToString(args->datatype);
    eDescr.ceColl.syncStrategy = comm->symkState.hasLsaMultimem ? "MC" : "UC";
    eDescr.ceColl.intraBatchSync = false;
    eDescr.ceColl.batchSize = 0;
    eDescr.ceColl.numBatches = 0;
    eDescr.ceColl.ceSeqNum = comm->ceColl.ceSeqNum;
    eDescr.ceColl.stream = (void*)stream;
    eDescr.ceColl.userTag = args->userTag;
    profilerStartNone(comm->profilerContext, &args->ceCollProfHandle, &eDescr, &args->collApiEventHandle, nullptr);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  (void)comm;
  (void)stream;
  if (args) profilerStopNone(&args->ceCollProfHandle);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream,
                                          struct ncclProfilerEventRef* ceSyncHandle) {
  (void)stream;
  memset(ceSyncHandle, 0, sizeof(*ceSyncHandle));
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && args && profilerLive(&args->ceCollProfHandle) &&
      (profilerRefreshEventMask() & ncclProfileCeSync)) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileCeSync;
    eDescr.rank = comm->rank;
    eDescr.ceCollSync.isComplete = comm->ceColl.useCompletePtr;
    eDescr.ceCollSync.nRanks = comm->nRanks;
    profilerStartNone(comm->profilerContext, ceSyncHandle, &eDescr, &args->ceCollProfHandle, nullptr);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, struct ncclProfilerEventRef* ceSyncHandle,
                                         cudaStream_t stream) {
  (void)comm;
  (void)stream;
  profilerStopNone(ceSyncHandle);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                           struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                           struct ncclProfilerEventRef* ceBatchHandle) {
  (void)stream;
  memset(ceBatchHandle, 0, sizeof(*ceBatchHandle));
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && args && profilerLive(&args->ceCollProfHandle) &&
      (profilerRefreshEventMask() & ncclProfileCeBatch)) {
    ncclProfilerEventDescr_t eDescr = {0};
    eDescr.type = ncclProfileCeBatch;
    eDescr.rank = comm->rank;
    eDescr.ceCollBatch.numOps = params->numOps;
    size_t totalBytes = 0;
    for (int i = 0; i < params->numOps; i++) totalBytes += params->sizes[i];
    eDescr.ceCollBatch.totalBytes = totalBytes;
    eDescr.ceCollBatch.useIntraSync = params->intraBatchSync;
    profilerStartNone(comm->profilerContext, ceBatchHandle, &eDescr, &args->ceCollProfHandle, nullptr);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, struct ncclProfilerEventRef* ceBatchHandle,
                                          cudaStream_t stream) {
  (void)comm;
  (void)stream;
  profilerStopNone(ceBatchHandle);
  return ncclSuccess;
}
