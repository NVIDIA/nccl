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

static int progressCounterMonitorPollIntervalMs() {
  return std::max((int)ncclParamProgressCounterMonitorPollMs(), kProgressCounterMonitorMinPollMs);
}

static int progressCounterMonitorStaleThresholdMs() {
  return std::max((int)ncclParamProgressCounterMonitorStaleMs(), kProgressCounterMonitorMinStaleMs);
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

// Hold the registration mutex while sampling.
// Update the host mirror only after a successful, within-budget read.
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

    uint64_t staleNs = (uint64_t)progressCounterMonitorStaleThresholdMs() * 1000000ULL;
    uint64_t readBudgetNs = std::min<uint64_t>((uint64_t)pollMs * 1000000ULL, staleNs);
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
      if (nowNs - beginNs >= readBudgetNs) {
        if (progressCounterMonitorErrorLogAllowed(g, nowNs))
          INFO(NCCL_RAS,
               "RAS progress counter sample unavailable on cudaDev %d rank %d "
               "(read %.3fs, limit %.3fs)",
               g->cudaDev, comm->rank, (nowNs - beginNs) / 1e9, readBudgetNs / 1e9);
        continue;
      }
      memcpy(comm->hostCountersBlock, &sample, sizeof(sample));
    }
  }
}

// Register only mapped counter allocations. The sampler does not enable the
// normal work FIFO/RMA GDRCopy paths.
ncclResult_t ncclProgressCounterMonitorInit(struct ncclComm* comm) {
  if (comm->mappedCountersBlock == nullptr) return ncclSuccess;
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
  g->cv.notify_one();
  INFO(NCCL_RAS,
       "RAS progress counter sampler registered rank %d cudaDev %d "
       "(NCCL_RAS_DEVICE_ENABLE=1)",
       comm->rank, comm->cudaDev);
  return ncclSuccess;
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
