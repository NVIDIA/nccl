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
#include <cstring>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <chrono>
#include "os.h"

extern ncclProfiler_t* getNcclProfiler_v1(void* lib);
extern ncclProfiler_t* getNcclProfiler_v2(void* lib);
extern ncclProfiler_t* getNcclProfiler_v3(void* lib);
extern ncclProfiler_t* getNcclProfiler_v4(void* lib);
extern ncclProfiler_t* getNcclProfiler_v5(void* lib);
extern ncclProfiler_t* getNcclProfiler_v6(void* lib);
extern ncclProfiler_t* getNcclProfiler_v7(void* lib);

// Observability threshold for pending+active ops. We keep enqueueing beyond
// this so KernelCh events stay paired with their parent task events.
#define NCCL_PROFILER_DEFAULT_MAX_INFLIGHT (MAXCHANNELS * MAX_PROFILER_EVENTS_PER_CHANNEL * 4)

struct ncclProfilerThread {
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cond;
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
  uint64_t droppedOps; // Allocation failures only; never used for soft cap.
};

enum ncclProfilerThreadAction {
  NCCL_PROFILER_THREAD_PROGRESS,
  NCCL_PROFILER_THREAD_STOP,
  // Stop requested but ops are still queued; drain them so join() returns.
  NCCL_PROFILER_THREAD_CLEANUP_AND_STOP,
};

static std::mutex profilerMutex;
static int profilerPluginRefCount;
static void* profilerPluginLib;
static ncclProfiler_t* ncclProfiler;

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

static ncclResult_t ncclProfilerPluginLoad(void) {
  const char* profilerName;
  if (profilerPluginLoadFailed == profilerPluginStatus) {
    return ncclSuccess;
  }

  std::lock_guard<std::mutex> lock(profilerMutex);
  if (profilerPluginLoadSuccess == profilerPluginStatus) {
    ++profilerPluginRefCount;
    goto exit;
  }

  if ((profilerName = ncclGetEnv("NCCL_PROFILER_PLUGIN")) != nullptr) {
    INFO(NCCL_ENV, "NCCL_PROFILER_PLUGIN set by environment to %s", profilerName);
    if (strcasecmp(profilerName, "none") == 0) goto fail;
  }
  profilerPluginLib = ncclOpenProfilerPluginLib(profilerName);
  if (profilerPluginLib == nullptr) {
    profilerPluginLib = ncclGetNetPluginLib(ncclPluginTypeProfiler);
    if (nullptr == profilerPluginLib) {
      goto fail;
    }
    profilerName = nullptr;
  } else if (ncclPluginLibPaths[ncclPluginTypeProfiler]) {
    profilerName = ncclPluginLibPaths[ncclPluginTypeProfiler];
  }

  ncclProfiler = getNcclProfiler_v7(profilerPluginLib);
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v6(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v5(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v4(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v3(profilerPluginLib);
  }
  if (ncclProfiler == nullptr) {
    ncclProfiler = getNcclProfiler_v2(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    ncclProfiler = getNcclProfiler_v1(profilerPluginLib);
  }
  if (ncclProfiler == NULL) {
    if (profilerName) INFO(NCCL_INIT, "External profiler plugin %s is unsupported", profilerName);
    goto fail;
  }
  if (profilerName) INFO(NCCL_INIT, "Successfully loaded external profiler plugin %s", profilerName);

  ++profilerPluginRefCount;
  profilerPluginStatus = profilerPluginLoadSuccess;

  // Store the pid of the process loading the profiler.
  // This is attached to the proxyOp event descriptor
  // so the plugin can figure out if the parent event
  // is in the same address space or not
  pid = ncclOsGetPid();

exit:
  return ncclSuccess;
fail:
  if (profilerPluginLib) NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));
  profilerPluginLib = nullptr;
  profilerPluginStatus = profilerPluginLoadFailed;
  goto exit;
}

static ncclResult_t ncclProfilerPluginUnload(void) {
  std::lock_guard<std::mutex> lock(profilerMutex);
  if (0 == (--profilerPluginRefCount)) {
    if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
      INFO(NCCL_DESTROY, "PROFILER/Plugin: Closing profiler plugin %s", ncclProfiler->name);
    }
    NCCLCHECK(ncclClosePluginLib(profilerPluginLib, ncclPluginTypeProfiler));
    profilerPluginLib = nullptr;
    ncclProfiler = nullptr;
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
static void printProfilerEventMask(int mask) {
  if (!mask) return;

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
  INFO(NCCL_INIT, "Profiler event mask: 0x%x (%d) - Enabled: %s", mask, mask, enabled);
}

// KernelPhase sub-events are gated on ncclProfileKernelCh, so KernelCh must be set
// whenever KernelPhase is. Enforce that on any incoming mask and log if it changes.
static int ncclProfilerEnforceMaskDeps(int mask, const char* source) {
  if ((mask & ncclProfileKernelPhase) && !(mask & ncclProfileKernelCh)) {
    mask |= ncclProfileKernelCh;
    INFO(NCCL_INIT,
         "Profiler: enabling ncclProfileKernelCh implicitly (%s set ncclProfileKernelPhase, which requires it)",
         source);
  }
  return mask;
}

void ncclProfilerSetRasOverride(int mask) {
  // A RAS-set mask must satisfy the same KernelPhase->KernelCh dependency as any other.
  mask = ncclProfilerEnforceMaskDeps(mask, "RAS override");
  // Store the 'active' flag (release) LAST, pairing with the acquire-load in ncclProfilerPluginInit so both
  // mask writes are visible there.
  COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, mask, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&ncclProfilerRasOverrideMask, mask, std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&ncclProfilerRasOverrideActive, true, std::memory_order_release);
  // Override is recorded above regardless; only log the decoded mask when a plugin is loaded.
  if (ncclProfiler != NULL) {
    INFO(NCCL_INIT, "Profiler event mask set out-of-band via RAS:");
    printProfilerEventMask(mask);
  }
}

ncclResult_t ncclProfilerPluginInit(struct ncclComm* comm) {
  TIME_START_EVENT(elapsed);
  TIME_START_EVENT(init);
  ncclProfilerPluginLoad();
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    int err = ncclProfiler->init(&comm->profilerContext, comm->commHash, &ncclProfilerEventMask, comm->config.commName,
                                 comm->nNodes, comm->nRanks, comm->rank, ncclDebugLog);
    if (err) {
      ncclProfilerPluginUnload();
      INFO(NCCL_INIT, "Profiler init failed with error '%d': %s. Continue without profiler.", err, strerror(errno));
    }

    // KernelPhase requires KernelCh; enable it implicitly (logs if it changes the mask).
    ncclProfilerEventMask = ncclProfilerEnforceMaskDeps(ncclProfilerEventMask, "plugin");

    printProfilerEventMask(ncclProfilerEventMask);
  }
  // Re-apply any RAS override: the plugin's init() above rewrites the mask unconditionally and would
  // otherwise clobber a value set job-wide via the RAS client.
  if (COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideActive, std::memory_order_acquire)) {
    int mask = COMPILER_ATOMIC_LOAD(&ncclProfilerRasOverrideMask, std::memory_order_relaxed);
    COMPILER_ATOMIC_STORE(&ncclProfilerEventMask, mask, std::memory_order_relaxed);
    INFO(NCCL_INIT, "Profiler event mask: re-applied RAS override 0x%x", mask);
  }
  TIME_STOP_EVENT(init);
  return ncclSuccess;
}

ncclResult_t ncclProfilerPluginFinalize(struct ncclComm* comm) {
  TIME_START_EVENT(finalize);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && comm->profilerContext) {
    ncclProfiler->finalize(comm->profilerContext);
  }
  ncclProfilerPluginUnload();
  TIME_STOP_EVENT(finalize);
  TIME_STOP_EVENT(elapsed);
  TIME_PRINT_EVENTS("Profiler");
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = {0};
  eDescr.type = ncclProfileGroupApi;
  eDescr.groupApi.graphCaptured = isGraphCaptured;

  ncclProfilerApiState.eActivationMask = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
  int groupApiMask = ncclProfileGroupApi | ncclProfileP2pApi | ncclProfileCollApi | ncclProfileKernelLaunch |
                     ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep |
                     ncclProfileKernelCh | ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync |
                     ncclProfileCeBatch;

  // Only count outermost groups when emitting group API events
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & groupApiMask)) {
    if (ncclProfilerApiState.profilerGroupDepth == 0) {
      eDescr.groupApi.groupDepth = ncclGroupDepth;
      ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.groupApiEventHandle, &eDescr);
      ncclProfilerApiState.profilerGroupDepth = ncclGroupDepth;
      ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStarted;
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupApiEvent() {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && groupApiEventHandle && ncclProfilerApiState.profilerGroupDepth == 0) {
    ncclProfiler->stopEvent(groupApiEventHandle);
    ncclProfilerApiState.groupApiEventHandle = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordGroupApiEventState(ncclProfilerEventState_t eState) {
  void* groupApiEventHandle = ncclProfilerApiState.groupApiEventHandle;
  bool shouldRecord = false;
  if (eState == ncclProfilerGroupStartApiStop && ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStarted) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateStopped;
    shouldRecord = true;
  } else if (eState == ncclProfilerGroupEndApiStart &&
             ncclProfilerApiState.state == ncclProfilerGroupApiStartStateStopped) {
    ncclProfilerApiState.state = ncclProfilerGroupApiStartStateReset;
    shouldRecord = true;
  }

  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && groupApiEventHandle && shouldRecord) {
    ncclProfiler->recordEventState(groupApiEventHandle, eState, NULL);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartP2pApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = {0};
  eDescr.type = ncclProfileP2pApi;
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;
  eDescr.p2pApi.func = ncclFuncToString(info->coll);
  eDescr.p2pApi.count = info->count;
  eDescr.p2pApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.p2pApi.stream = (void*)info->stream;
  eDescr.p2pApi.graphCaptured = isGraphCaptured;
  eDescr.p2pApi.userTag = info->collConfig.userProfilerTag;
  int p2pApiMask = ncclProfileP2pApi | ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep |
                   ncclProfileKernelCh | ncclProfileNetPlugin;
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & p2pApiMask)) {
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.p2pApiEventHandle, &eDescr);
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
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ncclProfilerApiState.p2pApiEventHandle) {
    ncclProfiler->stopEvent(ncclProfilerApiState.p2pApiEventHandle);
    ncclProfilerApiState.p2pApiEventHandle = nullptr;
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartCollApiEvent(struct ncclInfo* info, bool isGraphCaptured) {
  ncclProfilerEventDescr_t eDescr = {0};
  eDescr.type = ncclProfileCollApi;
  eDescr.parentObj = ncclProfilerApiState.groupApiEventHandle;
  eDescr.collApi.func = ncclFuncToString(info->coll);
  eDescr.collApi.count = info->count;
  eDescr.collApi.datatype = ncclDatatypeToString(info->datatype);
  eDescr.collApi.stream = (void*)info->stream;
  eDescr.collApi.root = info->root;
  eDescr.collApi.graphCaptured = isGraphCaptured;
  eDescr.collApi.userTag = info->collConfig.userProfilerTag;
  int collApiMask = ncclProfileCollApi | ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep |
                    ncclProfileKernelCh | ncclProfileNetPlugin | ncclProfileCeColl | ncclProfileCeSync |
                    ncclProfileCeBatch;
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (ncclProfilerApiState.eActivationMask & collApiMask)) {
    ncclProfiler->startEvent(info->comm->profilerContext, &ncclProfilerApiState.collApiEventHandle, &eDescr);
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
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ncclProfilerApiState.collApiEventHandle) {
    ncclProfiler->stopEvent(ncclProfilerApiState.collApiEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelLaunchEvent(struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclProfilerEventDescr_t eDescr = {0};
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    void* groupApiEventHandle = NULL;
    // Check if any collective in the plan has a set event activation mask
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    int eActivationMask_ = 0;
    while (ct) {
      if (ct->eActivationMask) {
        eActivationMask_ = ct->eActivationMask;
        groupApiEventHandle = ct->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      ct = ct->next;
    }
    // Check if any pt2pt in the plan has a set event activation mask
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        groupApiEventHandle = pt->groupApiEventHandle;
        goto startKernelLaunchEvent;
      }
      pt = pt->next;
    }

  startKernelLaunchEvent:
    if (eActivationMask_ & ncclProfileKernelLaunch) {
      eDescr.type = ncclProfileKernelLaunch;
      eDescr.parentObj = groupApiEventHandle;
      eDescr.kernelLaunch.stream = (void*)stream;
      ncclProfiler->startEvent(plan->comm->profilerContext, &plan->kernelLaunchEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelLaunchEvent(struct ncclKernelPlan* plan) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && plan->kernelLaunchEventHandle) {
    ncclProfiler->stopEvent(plan->kernelLaunchEventHandle);
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStart);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // Check if any collective in the plan has a set event activation mask
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
    // Check if any pt2pt in the plan has a set event activation mask
    while (pt) {
      if (pt->eActivationMask) {
        eActivationMask_ = pt->eActivationMask;
        goto startGroup;
      }
      pt = pt->next;
    }

  startGroup:
    if (eActivationMask_ & (ncclProfileGroup | ncclProfileColl | ncclProfileP2p | ncclProfileProxyOp |
                            ncclProfileProxyStep | ncclProfileKernelCh | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileGroup;
      ncclProfiler->startEvent(plan->comm->profilerContext, &plan->groupEventHandle, &eDescr);
    }
  }
  TIME_STOP_EVENT(groupStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopGroupEvent(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(groupStop);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && plan->groupEventHandle) {
    ncclProfiler->stopEvent(plan->groupEventHandle);
    // Null after stopping: the next callback (graph replay) re-creates it, keeping
    // start/stop balanced and making any stray double-stop a safe no-op.
    plan->groupEventHandle = nullptr;
  }
  TIME_STOP_EVENT(groupStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStart);
  struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
  while (ct) {
    if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
      int enable = ct->eActivationMask & (ncclProfileColl | ncclProfileProxyOp | ncclProfileProxyStep |
                                          ncclProfileKernelCh | ncclProfileNetPlugin);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = {0};
        eDescr.type = ncclProfileColl;
        eDescr.coll.parentGroup = plan->groupEventHandle;
        eDescr.parentObj = ct->collApiEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.coll.seqNumber = plan->comm->seqNumber[ct->func];
        eDescr.coll.func = ncclFuncToString(ct->func);
        eDescr.coll.sendBuff = ct->sendbuff;
        eDescr.coll.recvBuff = ct->recvbuff;
        eDescr.coll.count = ct->count;
        eDescr.coll.root = ct->root;
        eDescr.coll.datatype = ncclDatatypeToString(ct->datatype);
        eDescr.coll.nChannels = ct->nChannels;
        // Sym plans post all KernelCh ops against the head task's event
        // (profilerPostPlanWorkSym), covering countOneBits(plan->channelMask)
        // channels, while ct->nChannels stays 0. Advertise the real count on the
        // head so completion-gating plugins balance their per-channel refs.
        if (plan->isSymColl && (ct->eActivationMask & ncclProfileKernelCh) &&
            ct == ncclIntruQueueHead(&plan->collTaskQueue))
          eDescr.coll.nChannels = (uint8_t)countOneBits(plan->channelMask);
        eDescr.coll.nWarps = ct->nWarps;
        eDescr.coll.isSymColl = plan->isSymColl;
        if (plan->isSymColl) {
          // Override algo with the kernel variant (e.g. "AGxLL_R") so perftest
          // tuning mode (-U 1) shows a meaningful algorithm column.
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
        ncclProfiler->startEvent(plan->comm->profilerContext, &ct->eventHandle, &eDescr);
      }
    }
    // comm->seqNumber values are updated even if the plugin is not active, since they are used by RAS as well.
    // The test for "persistent" is a workaround for graph-captured collectives.  In their case this function may not be
    // consistently invoked on all the ranks, which would lead to mismatched counter values and thus false-positive
    // reports from RAS.  Instead, we choose not to include graph-captured collectives in our counts.  An exception is
    // made if ncclProfileKernelCh profiler events are active, as they result in proxy events always being added, which
    // gives the consistency.
    if (!plan->persistent ||
        (COMPILER_EXPECT(ncclProfiler != NULL, 0) && (plan->groupEventHandle || ct->collApiEventHandle) &&
         (ct->eActivationMask & ncclProfileKernelCh)))
      COMPILER_ATOMIC_FETCH_ADD(&plan->comm->seqNumber[ct->func], 1ULL, std::memory_order_relaxed);
    ct = ct->next;
  }
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      int enable = pt->eActivationMask & (ncclProfileP2p | ncclProfileProxyOp | ncclProfileProxyStep |
                                          ncclProfileKernelCh | ncclProfileNetPlugin);
      // Both halves of a send/recv pair carry KernelCh over their own channels
      // (see ncclProfilerPostPlanWork). Advertise the matching per-direction
      // count so completion-gating plugins balance their per-event refs.
      uint8_t profNChannels = pt->nChannels;
      if (pt->eActivationMask & ncclProfileKernelCh) profNChannels = (uint8_t)countOneBits(pt->channelMask);
      if (enable) {
        ncclProfilerEventDescr_t eDescr = {0};
        eDescr.type = ncclProfileP2p;
        eDescr.p2p.parentGroup = plan->groupEventHandle;
        eDescr.parentObj = pt->p2pApiEventHandle;
        eDescr.rank = plan->comm->rank;
        eDescr.p2p.func = ncclFuncToString(pt->func);
        eDescr.p2p.buff = pt->buff;
        eDescr.p2p.count = pt->count;
        eDescr.p2p.datatype = ncclDatatypeToString(pt->datatype);
        eDescr.p2p.peer = pt->root;
        eDescr.p2p.nChannels = profNChannels;
        eDescr.p2p.userTag = pt->profilerTag;
        ncclProfiler->startEvent(plan->comm->profilerContext, &pt->eventHandle, &eDescr);
      }
      pt = pt->next;
    }
  }
  TIME_STOP_EVENT(taskStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopTaskEvents(struct ncclKernelPlan* plan) {
  TIME_START_EVENT(taskStop);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // Null each handle after stopping: the next callback (graph replay) re-creates
    // them, keeping start/stop balanced and making a stray double-stop a safe no-op.
    struct ncclTaskColl* ct = ncclIntruQueueHead(&plan->collTaskQueue);
    while (ct) {
      if (ct->eventHandle) {
        ncclProfiler->stopEvent(ct->eventHandle);
        ct->eventHandle = nullptr;
      }
      ct = ct->next;
    }
    struct ncclTaskP2p* pt = ncclIntruQueueHead(&plan->p2pTaskQueue);
    while (pt) {
      if (pt->eventHandle) {
        ncclProfiler->stopEvent(pt->eventHandle);
        pt->eventHandle = nullptr;
      }
      pt = pt->next;
    }
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
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyOp | ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileProxyOp;
      eDescr.parentObj = sub->taskEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyOp.pid = sub->pid;
      eDescr.proxyOp.channelId = sub->channelId;
      eDescr.proxyOp.peer = sub->peer;
      eDescr.proxyOp.nSteps = DIVUP(sub->nsteps, args->sliceSteps);
      eDescr.proxyOp.chunkSize = args->chunkSize * args->sliceSteps;
      eDescr.proxyOp.isSend = args->progress == ncclTransports[TRANSPORT_NET]->send.proxyProgress ? 1 : 0;
      ncclProfiler->startEvent(sub->profilerContext, &sub->opEventHandle, &eDescr);
    }
  }
  TIME_STOP_EVENT(proxyOpStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyOpEvent(int s, struct ncclProxyArgs* args) {
  TIME_START_EVENT(proxyOpStop);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfiler->stopEvent(sub->opEventHandle);
    sub->opEventHandle = NULL;
  }
  TIME_STOP_EVENT(proxyOpStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartSendProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_ % NCCL_STEPS].stepEventHandle, &eDescr);
    }
  }
  sub->pHandles[step_ % NCCL_STEPS].subArgPtr = sub;
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartRecvProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStart);
  struct ncclProxySubArgs* sub = &args->subs[s];
  int step_ = DIVUP(stepId, args->sliceSteps);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (sub->eActivationMask & (ncclProfileProxyStep | ncclProfileNetPlugin)) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileProxyStep;
      eDescr.parentObj = sub->opEventHandle;
      eDescr.rank = sub->rank;
      eDescr.proxyStep.step = step_;
      ncclProfiler->startEvent(sub->profilerContext, &sub->pHandles[step_ % NCCL_STEPS].stepEventHandle, &eDescr);
    }
  }
  sub->pHandles[step_ % NCCL_STEPS].subArgPtr = sub;
  TIME_STOP_EVENT(proxyStepStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyStepEvent(int s, struct ncclProxyArgs* args, int stepId) {
  TIME_START_EVENT(proxyStepStop);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->pHandles[step_ % NCCL_STEPS].stepEventHandle) {
      ncclProfiler->stopEvent(sub->pHandles[step_ % NCCL_STEPS].stepEventHandle);
      sub->pHandles[step_ % NCCL_STEPS].stepEventHandle = NULL;
    }
  }
  TIME_STOP_EVENT(proxyStepStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartProxyCtrlEvent(void* profilerContext, void** eHandle) {
  TIME_START_EVENT(proxyCtrlStart);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // for proxy control events we allow profiling mode to change on a per event basis
    int eActivationMaskProxy = COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed);
    if (eActivationMaskProxy & ncclProfileProxyCtrl) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileProxyCtrl;
      ncclProfiler->startEvent(profilerContext, eHandle, &eDescr);
      TIME_STOP_EVENT(proxyCtrlStart);
      return ncclSuccess;
    }
  }
  *eHandle = NULL;
  TIME_STOP_EVENT(proxyCtrlStart);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopProxyCtrlEvent(void* eHandle) {
  TIME_START_EVENT(proxyCtrlStop);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && eHandle) {
    ncclProfiler->stopEvent(eHandle);
  }
  TIME_STOP_EVENT(proxyCtrlStop);
  return ncclSuccess;
}

ncclResult_t ncclProfilerStartKernelChEvent(struct ncclProfilerWorkOp* op, uint64_t start) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (op->eActivationMask & ncclProfileKernelCh) {
      ncclProfilerEventDescr_t eDescr = {};
      eDescr.type = ncclProfileKernelCh;
      eDescr.parentObj = op->taskEventHandle;
      eDescr.kernelCh.channelId = op->channelId;
      eDescr.kernelCh.pTimer = start;
      ncclProfiler->startEvent(op->profilerContext, &op->kernelEventHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerStopKernelChEvent(struct ncclProfilerWorkOp* op, uint64_t stop) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (op->kernelEventHandle) {
      ncclProfilerEventStateArgs_t a = {};
      a.kernelCh.pTimer = stop;
      ncclProfiler->recordEventState(op->kernelEventHandle, ncclProfilerKernelChStop, &a);
      ncclProfiler->stopEvent(op->kernelEventHandle);
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyOpEventState(int s, struct ncclProxyArgs* args, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyOpRecord);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    ncclProfilerEventStateArgs_t a = {};
    ncclProfiler->recordEventState(sub->opEventHandle, eState, &a);
  }
  TIME_STOP_EVENT(proxyOpRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyStepEventState(int s, struct ncclProxyArgs* args, int stepId,
                                                   ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyStepRecord);
  struct ncclProxySubArgs* sub = &args->subs[s];
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && sub->opEventHandle) {
    int step_ = DIVUP(stepId, args->sliceSteps);
    if (sub->pHandles[step_ % NCCL_STEPS].stepEventHandle) {
      ncclProfilerEventStateArgs_t a = {};
      a.proxyStep.transSize = sub->transSize;
      ncclProfiler->recordEventState(sub->pHandles[step_ % NCCL_STEPS].stepEventHandle, eState, &a);
    }
  }
  TIME_STOP_EVENT(proxyStepRecord);
  return ncclSuccess;
}

ncclResult_t ncclProfilerRecordProxyCtrlEventState(void* eHandle, int appended, ncclProfilerEventState_t eState) {
  TIME_START_EVENT(proxyCtrlRecord);
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && eHandle &&
      COMPILER_ATOMIC_LOAD(&ncclProfilerEventMask, std::memory_order_relaxed) & ncclProfileProxyCtrl) {
    ncclProfilerEventStateArgs_t args = {};
    args.proxyCtrl.appendedProxyOps = appended;
    ncclProfiler->recordEventState(eHandle, eState, &args);
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
  uint64_t* ts = op->workPhases[ch].data[slot].timestamps;

  // Fold a missing intermediate boundary onto the adjacent outer one; if both are
  // missing the kernel didn't instrument phases, so skip rather than mislabel.
  bool haveAfterOpen = ts[NCCL_KERNEL_PHASE_AFTER_OPEN] != 0;
  bool haveBeforeClose = ts[NCCL_KERNEL_PHASE_BEFORE_CLOSE] != 0;
  if (!haveAfterOpen && !haveBeforeClose) {
    memset(ts, 0, sizeof(uint64_t) * MAX_PROFILER_PHASES);
    return;
  }
  if (!haveAfterOpen && ts[NCCL_KERNEL_PHASE_BEGIN] != 0)
    ts[NCCL_KERNEL_PHASE_AFTER_OPEN] = ts[NCCL_KERNEL_PHASE_BEGIN];
  if (!haveBeforeClose && ts[NCCL_KERNEL_PHASE_END] != 0)
    ts[NCCL_KERNEL_PHASE_BEFORE_CLOSE] = ts[NCCL_KERNEL_PHASE_END];

  for (int i = 0; i < MAX_PROFILER_PHASES - 1; i++) {
    if (ts[i] == 0 || ts[i + 1] == 0) continue;
    if (ts[i] == ts[i + 1]) continue; // skip zero-duration (synthesized) boundary
    void* phaseHandle = nullptr;
    ncclProfilerEventDescr_t eDescr = {};
    eDescr.type = ncclProfileKernelPhase;
    eDescr.parentObj = op->kernelEventHandle;
    eDescr.kernelPhase.channelId = op->channelId;
    eDescr.kernelPhase.phaseId = i;
    eDescr.kernelPhase.phaseName = profilerPhaseNames[i];
    eDescr.kernelPhase.pTimer = ts[i];
    ncclProfiler->startEvent(op->profilerContext, &phaseHandle, &eDescr);
    if (phaseHandle) {
      ncclProfilerEventStateArgs_t a = {};
      // Phase stop reuses kernelCh.pTimer (same single-timestamp shape); see profiler_v7.h.
      a.kernelCh.pTimer = ts[i + 1];
      ncclProfiler->recordEventState(phaseHandle, ncclProfilerKernelPhaseStop, &a);
      ncclProfiler->stopEvent(phaseHandle);
    }
  }
  memset(ts, 0, sizeof(uint64_t) * MAX_PROFILER_PHASES);
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
  INFO(NCCL_INIT, "[Profiler Thread] started (cudaDev=%d, maxInflight=%zu)", pt->cudaDev, pt->maxInflight);

  unsigned spinUs = 1;
  while (true) {
    bool stop;
    ncclProfilerThreadAction action = waitForAction(pt, &stop);
    if (action == NCCL_PROFILER_THREAD_STOP) break;

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
                              int eActivationMask, void* taskEventHandle, bool sym) {
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
      op->workPhases = sym ? comm->profiler.symWorkPhases : comm->profiler.workPhases;
      op->kernelEventHandle = nullptr;
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
                                             void* taskEventHandle) {
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
void ncclProfilerReserveSymCounters(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  if (!ncclProfilerPluginLoaded() || comm->profiler.profilerThread == nullptr) return;
  if (!plan->isSymColl) return;
  struct ncclTaskColl* sct = ncclIntruQueueHead(&plan->collTaskQueue);
  if (sct == nullptr || !(sct->eActivationMask & ncclProfileKernelCh)) return;
  struct ncclSymkDevWorkArgs* argsBuf = (struct ncclSymkDevWorkArgs*)plan->kernelSymArgs;
  if (!argsBuf->profilerEnabled) return;
  uint64_t* counters = argsBuf->getProfilerCounters();
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
  struct ncclSymkDevWorkArgs* argsBuf = (struct ncclSymkDevWorkArgs*)plan->kernelSymArgs;
  if (!argsBuf->profilerEnabled) return;
  uint64_t* counters = argsBuf->getProfilerCounters();
  int nChannels = countOneBits(plan->channelMask);
  for (int c = 0; c < nChannels; c++) {
    profilerEnqueueOp(pt, comm, c, counters[c], sct->eActivationMask, sct->eventHandle, /*sym=*/true);
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

ncclResult_t ncclProfilerCallback(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (type == ncclProfilerNetEventStart) { // start
      struct ncclProxyEventHandle* p = (struct ncclProxyEventHandle*)pHandle;
      struct ncclProxySubArgs* sub = p->subArgPtr;
      if (sub->eActivationMask & ncclProfileNetPlugin) {
        ncclProfilerEventDescr_t eDescr = {0};
        eDescr.type = ncclProfileNetPlugin;
        eDescr.parentObj = p->stepEventHandle;
        eDescr.rank = sub->rank;
        eDescr.netPlugin.id = pluginId;
        eDescr.netPlugin.data = extData;
        ncclProfiler->startEvent(sub->profilerContext, eHandle, &eDescr);
      }
    } else if (type == ncclProfilerNetEventStop) { // stop
      ncclProfiler->stopEvent(*eHandle);
    } else if (type == ncclProfilerNetEventUpdate) { // update
      ncclProfilerEventStateArgs_t args = {};
      args.netPlugin.data = extData;
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);
    } else { // update and stop
      ncclProfilerEventStateArgs_t args = {};
      args.netPlugin.data = extData;
      ncclProfiler->recordEventState(*eHandle, ncclProfilerNetPluginUpdate, &args);
      ncclProfiler->stopEvent(*eHandle);
    }
  }
  return ncclSuccess;
}

// ============================================================================
// CE Profiler Functions (Simple Wrappers)
// ============================================================================

/*
 * CE Collective start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    // Check if CE Coll events are enabled (or child events CeSync/CeBatch which need CeColl)
    int ceCollMask = ncclProfileCeColl | ncclProfileCeSync | ncclProfileCeBatch;
    if (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ceCollMask) {
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileCeColl;
      eDescr.parentObj = args->collApiEventHandle;
      eDescr.rank = comm->rank;

      eDescr.ceColl.seqNumber = comm->ceColl.ceSeqNum;
      eDescr.ceColl.func = ncclFuncToString(args->func);
      eDescr.ceColl.sendBuff = args->sendBuff;
      eDescr.ceColl.recvBuff = args->recvBuff;
      eDescr.ceColl.count = args->nElts;
      eDescr.ceColl.root = args->rootRank;
      eDescr.ceColl.datatype = ncclDatatypeToString(args->datatype);
      // NVLS multicast isn't available across cliques - report UC for cross-clique
      eDescr.ceColl.syncStrategy = (comm->nvlsSupport && !comm->p2pCrossClique) ? "MC" : "UC";
      eDescr.ceColl.intraBatchSync = false;
      eDescr.ceColl.batchSize = 0;
      eDescr.ceColl.numBatches = 0;
      eDescr.ceColl.ceSeqNum = comm->ceColl.ceSeqNum;
      eDescr.ceColl.stream = (void*)stream;
      eDescr.ceColl.userTag = args->userTag;

      ncclProfiler->startEvent(comm->profilerContext, &args->ceCollProfHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

/*
 * CE Collective stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeCollEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle) {
      ncclProfiler->stopEvent(args->ceCollProfHandle);
    }
  }
  return ncclSuccess;
}

/*
 * CE Sync start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeSyncEvent(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream,
                                          void** ceSyncHandle) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle &&
        (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileCeSync)) {
      // CeSync only needs to check if it's enabled; parent CeColl is implicitly started via ceCollMask
      ncclProfilerEventDescr_t eDescr = {0};
      eDescr.type = ncclProfileCeSync;
      eDescr.parentObj = args->ceCollProfHandle;
      eDescr.rank = comm->rank;

      eDescr.ceCollSync.isComplete = comm->ceColl.useCompletePtr;
      eDescr.ceCollSync.nRanks = comm->nRanks;

      ncclProfiler->startEvent(comm->profilerContext, ceSyncHandle, &eDescr);
    }
  }
  return ncclSuccess;
}

/*
 * CE Sync stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeSyncEvent(struct ncclComm* comm, void* ceSyncHandle, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ceSyncHandle) {
    ncclProfiler->stopEvent(ceSyncHandle);
  }
  return ncclSuccess;
}

/*
 * CE Batch start event - calls plugin startEvent callback
 */
ncclResult_t ncclProfilerStartCeBatchEvent(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                           struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                           void** ceBatchHandle) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0)) {
    if (args && args->ceCollProfHandle) {
      // CeBatch only needs to check if it's enabled; parent CeColl is implicitly started via ceCollMask
      if (__atomic_load_n(&ncclProfilerEventMask, __ATOMIC_RELAXED) & ncclProfileCeBatch) {
        ncclProfilerEventDescr_t eDescr = {0};
        eDescr.type = ncclProfileCeBatch;
        eDescr.parentObj = args->ceCollProfHandle;
        eDescr.rank = comm->rank;

        eDescr.ceCollBatch.numOps = params->numOps;

        size_t totalBytes = 0;
        for (int i = 0; i < params->numOps; i++) {
          totalBytes += params->sizes[i];
        }
        eDescr.ceCollBatch.totalBytes = totalBytes;
        eDescr.ceCollBatch.useIntraSync = params->intraBatchSync;

        ncclProfiler->startEvent(comm->profilerContext, ceBatchHandle, &eDescr);
      }
    }
  }
  return ncclSuccess;
}

/*
 * CE Batch stop event - calls plugin stopEvent callback
 */
ncclResult_t ncclProfilerStopCeBatchEvent(struct ncclComm* comm, void* ceBatchHandle, cudaStream_t stream) {
  if (COMPILER_EXPECT(ncclProfiler != NULL, 0) && ceBatchHandle) {
    ncclProfiler->stopEvent(ceBatchHandle);
  }
  return ncclSuccess;
}
