/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "comm.h"
#include "progress_monitor.h"
#include "debug.h"
#include "param.h"
#include "checks.h"
#include "compiler.h"
#include "utils.h"
#include "alloc.h"
#include "gdrwrap.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

NCCL_PARAM(ProgressCounterMonitorPollMs, "PROGRESS_COUNTER_MONITOR_POLL_MS", 1000);
NCCL_PARAM(ProgressCounterMonitorStaleMs, "PROGRESS_COUNTER_MONITOR_STALE_MS", 5000);
NCCL_PARAM(ProgressCounterMonitorStaleWarnSec, "PROGRESS_COUNTER_MONITOR_STALE_WARN_SEC", 600);

static constexpr int kProgressCounterMonitorMinPollMs = 50;
static constexpr int kProgressCounterMonitorMinStaleMs = 1000;

// One CPU sampler per used device, sharing the process-wide GDR handle. Empty
// workers sleep until another communicator registers. Control objects live until exit.
struct ncclGpuProgressCounterMonitor {
  int cudaDev;
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  bool shouldStop;
  uint64_t lastErrorWarnNs;
  struct ncclIntruQueue<struct ncclComm, &ncclComm::nextProgressRegistration> registrations;
};

// The registry mutex protects installation and termination state.
// The per-device mutex protects registrations and GDR mapping lifetimes.
static std::mutex gpuProgressCounterMonitorsMu;
static struct ncclGpuProgressCounterMonitor* gpuProgressCounterMonitors[kRasMaxCudaDevices] = {};
static bool progressCounterMonitorTerminationRegistered;
static bool progressCounterMonitorTerminating;

// Query gathering and host publication share this lock; never hold it across device reads.
// If both are needed, acquire the per-device mutex before ncclCommsMutex.
extern std::mutex ncclCommsMutex;

static int progressCounterMonitorPollIntervalMs() {
  return std::max((int)ncclParamProgressCounterMonitorPollMs(), kProgressCounterMonitorMinPollMs);
}

// Consecutive samples are continuous only if they fall within the limit, so allow two poll intervals.
int ncclProgressCounterMonitorStaleThresholdMs() {
  return std::max({(int)ncclParamProgressCounterMonitorStaleMs(), kProgressCounterMonitorMinStaleMs,
                   2 * progressCounterMonitorPollIntervalMs()});
}

// Rate-limit sampler errors; caller holds the per-device mutex.
static bool progressCounterMonitorErrorLogAllowed(struct ncclGpuProgressCounterMonitor* g, uint64_t nowNs) {
  int64_t warnSec = ncclParamProgressCounterMonitorStaleWarnSec();
  if (warnSec <= 0) return false;
  uint64_t intervalNs = (uint64_t)warnSec * 1000000000ULL;
  if (g->lastErrorWarnNs != 0 && nowNs - g->lastErrorWarnNs < intervalNs) return false;
  g->lastErrorWarnNs = nowNs;
  return true;
}

// Publish a sample read between beginNs and nowNs, updating the history against the previous sample;
// across a sampling gap no advance is observed. The caller holds ncclCommsMutex and the device mutex.
void ncclProgressCounterMonitorPublish(struct ncclComm* comm, const struct ncclProgressCountersBlock* sample,
                                       uint64_t beginNs, uint64_t nowNs, uint64_t staleNs) {
  const struct ncclProgressCountersBlock* previous = comm->hostCountersBlock;
  bool continuous = comm->rasCounterLastRefreshNs != 0 && nowNs - comm->rasCounterLastRefreshNs < staleNs;

  uint64_t activeMask = 0, previousActiveMask = 0;
  for (int channel = 0; channel < MAXCHANNELS; channel++) {
    activeMask |= sample->collOpActive[channel];
    previousActiveMask |= previous->collOpActive[channel];
  }
  uint32_t uninstrumentedMask =
    COMPILER_ATOMIC_LOAD(&comm->rasCounterUninstrumentedSlotMask, std::memory_order_relaxed);
  uint32_t graphUnsafeMask = COMPILER_ATOMIC_LOAD(&comm->rasHostCountGraphUnsafeMask, std::memory_order_relaxed);

  for (int slot = 0; slot < NCCL_NUM_PROGRESS_COUNTERS; slot++) {
    uint64_t completed = sample->completedWorkCount[slot];
    // Even with incomplete coverage, an increase proves instrumented-path progress.
    bool completedAdvanced = continuous && completed > previous->completedWorkCount[slot];
    // A count was reached no later than its first observation, even after a sampling gap.
    if (slot < NCCL_NUM_FUNCTIONS && completed != previous->completedWorkCount[slot])
      comm->rasCounterCompletedChangedNs[slot] = nowNs;

    bool pending = (activeMask & (1ULL << slot)) != 0;
    // Host counts exist only for legacy slots, and need full coverage without graph replays.
    if (slot < NCCL_NUM_FUNCTIONS && !((uninstrumentedMask | graphUnsafeMask) & (1ULL << slot))) {
      uint64_t launched = COMPILER_ATOMIC_LOAD(&comm->seqNumber[slot], std::memory_order_relaxed);
      pending |= launched > completed;
    }

    uint64_t sinceNs = comm->rasCounterNoProgressSinceNs[slot];
    if (!pending) {
      sinceNs = 0;
    } else if (!continuous || completedAdvanced || sinceNs == 0 ||
               ((activeMask & ~previousActiveMask) & (1ULL << slot))) {
      // A newly observed active bit must not inherit an earlier host-only wait.
      sinceNs = nowNs;
    }
    comm->rasCounterNoProgressSinceNs[slot] = sinceNs;
  }
  memcpy(comm->hostCountersBlock, sample, sizeof(*sample));
  // Freshness includes the read window; observed progress uses its end.
  comm->rasCounterLastRefreshNs = beginNs;
  COMPILER_ATOMIC_STORE(&comm->rasCounterMonitorState, ncclCounterRunning, std::memory_order_release);
}

// Stop host readers at exit.
static void progressCounterMonitorTerminate() {
  struct ncclGpuProgressCounterMonitor* monitors[kRasMaxCudaDevices];
  {
    std::lock_guard<std::mutex> globalLock(gpuProgressCounterMonitorsMu);
    if (progressCounterMonitorTerminating) return;
    progressCounterMonitorTerminating = true;
    std::copy_n(gpuProgressCounterMonitors, kRasMaxCudaDevices, monitors);
  }
  // Stop and join without holding the registry mutex.
  for (auto* g : monitors) {
    if (g == nullptr) continue;
    std::lock_guard<std::mutex> lock(g->mutex);
    g->shouldStop = true;
    g->cv.notify_one();
  }
  for (auto* g : monitors) {
    if (g == nullptr) continue;
    if (g->thread.joinable()) g->thread.join();
  }
}

// Hold the registration mutex while sampling. Every completed read is published; a slow one is dated by its start,
// so the query reports it as stale.
static void progressCounterMonitorLoop(struct ncclGpuProgressCounterMonitor* g) {
  std::unique_lock<std::mutex> lock(g->mutex);
  while (!g->shouldStop) {
    while (!g->shouldStop && ncclIntruQueueEmpty(&g->registrations)) {
      g->cv.wait(lock);
    }
    if (g->shouldStop) break;
    int pollMs = progressCounterMonitorPollIntervalMs();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(pollMs);
    while (!g->shouldStop) {
      if (g->cv.wait_until(lock, deadline) == std::cv_status::timeout) break;
    }
    if (g->shouldStop) break;

    uint64_t staleNs = (uint64_t)ncclProgressCounterMonitorStaleThresholdMs() * 1000000ULL;
    for (struct ncclComm* comm = ncclIntruQueueHead(&g->registrations); comm != nullptr;
         comm = comm->nextProgressRegistration) {
      struct ncclProgressCountersBlock sample;
      uint64_t beginNs = clockNano();
      // Read aligned fields separately, not as an atomic block snapshot. This
      // relies on the platform's CPU/GPU single-field visibility guarantees.
      static_assert(sizeof(sample) % sizeof(uint64_t) == 0, "Counter block must contain whole 64-bit fields");
      const uint64_t* mapped = (const uint64_t*)comm->mappedCountersBlock;
      for (size_t i = 0; i < sizeof(sample) / sizeof(uint64_t); i++) {
        uint64_t value = COMPILER_ATOMIC_LOAD(mapped + i, std::memory_order_relaxed);
        memcpy((char*)&sample + i * sizeof(value), &value, sizeof(value));
      }
      uint64_t nowNs = clockNano();
      if (nowNs - beginNs >= staleNs && progressCounterMonitorErrorLogAllowed(g, nowNs))
        INFO(NCCL_RAS,
             "RAS progress counter read on cudaDev %d rank %d took %.3fs, longer than the %.3fs freshness limit",
             g->cudaDev, comm->rank, (nowNs - beginNs) / 1e9, staleNs / 1e9);
      std::lock_guard<std::mutex> commsLock(ncclCommsMutex);
      ncclProgressCounterMonitorPublish(comm, &sample, beginNs, nowNs, staleNs);
    }
  }
}

// Register only mapped counter allocations. The sampler does not enable the
// normal work FIFO/RMA GDRCopy paths.
ncclResult_t ncclProgressCounterMonitorInit(struct ncclComm* comm) {
  if (comm->mappedCountersBlock == nullptr) return comm->hostCountersBlock == nullptr ? ncclSuccess : ncclSystemError;
  if (comm->cudaDev < 0 || comm->cudaDev >= kRasMaxCudaDevices) return ncclInvalidArgument;
  if (comm->progressCounterMonitor != nullptr) return ncclInvalidUsage;
  struct ncclGpuProgressCounterMonitor* g;
  {
    std::lock_guard<std::mutex> globalLock(gpuProgressCounterMonitorsMu);
    if (progressCounterMonitorTerminating) return ncclSystemError;
    if (!progressCounterMonitorTerminationRegistered) {
      if (atexit(progressCounterMonitorTerminate) != 0) return ncclSystemError;
      progressCounterMonitorTerminationRegistered = true;
    }
    g = gpuProgressCounterMonitors[comm->cudaDev];
    if (g == nullptr) {
      g = new (std::nothrow) ncclGpuProgressCounterMonitor{};
      if (g == nullptr) return ncclSystemError;
      g->cudaDev = comm->cudaDev;
      ncclIntruQueueConstruct(&g->registrations);
      gpuProgressCounterMonitors[comm->cudaDev] = g;
    }
  }

  std::unique_lock<std::mutex> lock(g->mutex);
  if (g->shouldStop) return ncclSystemError;
  if (!g->thread.joinable()) {
    try {
      g->thread = std::thread(progressCounterMonitorLoop, g);
    } catch (const std::exception& e) {
      INFO(NCCL_RAS, "RAS progress counter sampler thread creation failed (%s)", e.what());
      return ncclSystemError;
    }
    ncclSetThreadName(g->thread, "NCCL CtrMon %d", g->cudaDev);
  }
  comm->progressCounterMonitor = g;
  ncclIntruQueueEnqueue(&g->registrations, comm);
  COMPILER_ATOMIC_STORE(&comm->rasCounterMonitorStartNs, clockNano(), std::memory_order_relaxed);
  COMPILER_ATOMIC_STORE(&comm->rasCounterMonitorState, ncclCounterRunning, std::memory_order_release);
  g->cv.notify_one();
  INFO(NCCL_RAS,
       "RAS progress counter sampler registered rank %d cudaDev %d "
       "(NCCL_RAS_DEVICE_ENABLE=1)",
       comm->rank, comm->cudaDev);
  return ncclSuccess;
}

static void progressCounterSetStickyBit(uint32_t* mask, int bit) {
  uint32_t expected = COMPILER_ATOMIC_LOAD(mask, std::memory_order_relaxed);
  while (!(expected & (1u << bit)) &&
         !COMPILER_ATOMIC_COMPARE_EXCHANGE(mask, &expected, expected | (1u << bit), std::memory_order_relaxed,
                                           std::memory_order_relaxed)) {
  }
}

void ncclProgressCounterMarkUninstrumented(struct ncclComm* comm, int func) {
  if (comm->mappedCountersBlock == nullptr || func < 0 || func >= ncclNumFuncs) return;
  progressCounterSetStickyBit(&comm->rasCounterUninstrumentedSlotMask, func);
}

void ncclProgressCounterMarkGraphUnsafe(struct ncclComm* comm, int func) {
  if (comm->mappedCountersBlock == nullptr || func < 0 || func >= NCCL_NUM_FUNCTIONS) return;
  progressCounterSetStickyBit(&comm->rasHostCountGraphUnsafeMask, func);
}

static bool progressCounterRegistrationMatches(struct ncclComm* registered, struct ncclComm* target) {
  return registered == target;
}

// Unregister before the shared GDR destructor releases counter storage.
ncclResult_t ncclProgressCounterMonitorDestroy(struct ncclComm* comm) {
  auto* g = comm->progressCounterMonitor;
  if (g == nullptr) return ncclSuccess;
  std::lock_guard<std::mutex> lock(g->mutex);
  ncclIntruQueueDelete(&g->registrations, comm, progressCounterRegistrationMatches);
  comm->nextProgressRegistration = nullptr;
  comm->progressCounterMonitor = nullptr;
  return ncclSuccess;
}
