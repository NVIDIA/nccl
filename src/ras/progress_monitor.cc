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

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <condition_variable>
#include <mutex>
#include <thread>

NCCL_PARAM(ProgressCounterMonitorPollMs, "PROGRESS_COUNTER_MONITOR_POLL_MS", 1000);
NCCL_PARAM(ProgressCounterMonitorStaleMs, "PROGRESS_COUNTER_MONITOR_STALE_MS", 5000);
NCCL_PARAM(ProgressCounterMonitorStaleWarnSec, "PROGRESS_COUNTER_MONITOR_STALE_WARN_SEC", 600);

static constexpr int kProgressCounterMonitorMinPollMs = 50;
static constexpr int kProgressCounterMonitorMinStaleMs = 1000;

// One worker per CUDA device, shared by all registered communicators.
// mutex protects mutable worker state and the registration list;
// warningMutex protects the per-GPU warning timestamps.
struct ncclGpuProgressCounterMonitor {
  int cudaDev;
  std::thread thread;
  std::mutex mutex;
  std::condition_variable cv;
  bool running;
  bool shouldStop;
  bool copyInFlight;
  bool copyStallWarned;
  uint64_t copyStartNs;
  cudaStream_t sideStream;
  cudaEvent_t copyDone;
  std::mutex warningMutex;
  uint64_t lastStaleWarnNs;
  uint64_t lastErrorWarnNs;
  int destroyRefs; // Pins the worker during unregister cleanup.
  struct ncclIntruQueue<struct ncclComm, &ncclComm::nextProgressRegistration> registrations;
};

// Lock order: gpuProgressCounterMonitorsMu before ncclGpuProgressCounterMonitor::mutex.
static std::mutex gpuProgressCounterMonitorsMu;
static struct ncclGpuProgressCounterMonitor* gpuProgressCounterMonitors[kRasMaxCudaDevices] = {};

// Return the configured polling interval, clamped to the supported minimum.
static int progressCounterMonitorPollIntervalMs() {
  int pollMs = (int)ncclParamProgressCounterMonitorPollMs();
  return std::max(pollMs, kProgressCounterMonitorMinPollMs);
}

// Return the stale-copy threshold, clamped to the supported minimum.
static int progressCounterMonitorStaleThresholdMs() {
  int staleMs = (int)ncclParamProgressCounterMonitorStaleMs();
  return std::max(staleMs, kProgressCounterMonitorMinStaleMs);
}

// Return the warning rate-limit interval, or zero when warnings are disabled.
static uint64_t progressCounterMonitorWarnIntervalNs() {
  int64_t warnSec = ncclParamProgressCounterMonitorStaleWarnSec();
  if (warnSec <= 0) return 0;
  return warnSec * 1000000000ULL;
}

// Reserve a per-GPU warning slot when the category's rate-limit interval has elapsed.
static bool progressCounterMonitorShouldWarn(struct ncclGpuProgressCounterMonitor* g, uint64_t* lastWarnNs,
                                             uint64_t nowNs, uint64_t warnIntervalNs) {
  if (warnIntervalNs == 0) return false;
  std::lock_guard<std::mutex> lock(g->warningMutex);
  if (*lastWarnNs == 0 || nowNs - *lastWarnNs >= warnIntervalNs) {
    *lastWarnNs = nowNs;
    return true;
  }
  return false;
}

// Return whether the per-GPU rate limiter permits another monitor error log.
static bool progressCounterMonitorErrorLogAllowed(struct ncclGpuProgressCounterMonitor* g) {
  return progressCounterMonitorShouldWarn(g, &g->lastErrorWarnNs, clockNano(), progressCounterMonitorWarnIntervalNs());
}

// Periodically copy device progress counters for all communicators registered on one CUDA device.
static void progressCounterMonitorLoop(struct ncclGpuProgressCounterMonitor* g) {
  // Bind the shared worker to its CUDA device for its lifetime.
  if (cudaSetDevice(g->cudaDev) != cudaSuccess) {
    WARN("NCCL progress counter monitor: cudaSetDevice(%d) failed; progress-counter mirrors will "
         "remain stale; Init will surface the failure.",
         g->cudaDev);
    {
      std::lock_guard<std::mutex> lock(g->mutex);
      g->shouldStop = true;
    }
    g->cv.notify_one();
    return;
  }

  // Keep counter-monitor CUDA calls independent of application graph capture.
  cudaStreamCaptureMode captureMode = cudaStreamCaptureModeRelaxed;
  cudaError_t captureErr = cudaThreadExchangeStreamCaptureMode(&captureMode);
  if (captureErr != cudaSuccess) {
    WARN("NCCL progress counter monitor: failed to set relaxed CUDA stream capture mode (%s)",
         cudaGetErrorString(captureErr));
    {
      std::lock_guard<std::mutex> lock(g->mutex);
      g->shouldStop = true;
    }
    g->cv.notify_one();
    return;
  }

  {
    std::lock_guard<std::mutex> lock(g->mutex);
    g->running = true;
  }
  g->cv.notify_one();

  while (true) {
    // Protect registration lifetimes while copies are queued.
    std::unique_lock<std::mutex> lock(g->mutex);
    auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(progressCounterMonitorPollIntervalMs());
    while (!g->shouldStop) {
      if (g->cv.wait_until(lock, deadline) == std::cv_status::timeout) break;
    }
    if (g->shouldStop) break;

    if (g->copyInFlight) {
      cudaError_t e = cudaEventQuery(g->copyDone);
      if (e == cudaErrorNotReady) {
        uint64_t nowNs = clockNano();
        uint64_t staleNs = (g->copyStartNs != 0 && nowNs >= g->copyStartNs) ? nowNs - g->copyStartNs : 0;
        if (!g->copyStallWarned && staleNs >= (uint64_t)progressCounterMonitorStaleThresholdMs() * 1000000ULL) {
          g->copyStallWarned = true;
          uint64_t warnIntervalNs = progressCounterMonitorWarnIntervalNs();
          if (progressCounterMonitorShouldWarn(g, &g->lastStaleWarnNs, nowNs, warnIntervalNs)) {
            INFO(NCCL_RAS,
                 "NCCL progress counter monitor: counter DMA still pending after %.1fs (cudaDev %d, %zu comms); "
                 "progress-counter mirrors may be stale",
                 staleNs / 1e9, g->cudaDev, (size_t)g->registrations.nElems);
          }
        }
        continue;
      }
      if (e != cudaSuccess && progressCounterMonitorErrorLogAllowed(g)) {
        INFO(NCCL_RAS, "NCCL progress counter monitor: counter DMA completion check failed on cudaDev %d (%s)",
             g->cudaDev, cudaGetErrorString(e));
      }
      g->copyInFlight = false;
      g->copyStartNs = 0;
      g->copyStallWarned = false;
    }

    bool copiedAny = false;
    int copyLaunchFailures = 0;
    int firstFailRank = -1;
    uint64_t firstFailCommHash = 0;
    cudaError_t firstFail = cudaSuccess;
    for (struct ncclComm* comm = ncclIntruQueueHead(&g->registrations); comm != nullptr;
         comm = comm->nextProgressRegistration) {
      if (comm->deviceCountersBlock == nullptr || comm->hostCountersBlock == nullptr) continue;
      cudaError_t e = cudaMemcpyAsync(comm->hostCountersBlock, comm->deviceCountersBlock,
                                      sizeof(struct ncclProgressCountersBlock), cudaMemcpyDeviceToHost, g->sideStream);
      if (e != cudaSuccess) {
        if (copyLaunchFailures == 0) {
          firstFailRank = comm->rank;
          firstFailCommHash = comm->commHash;
          firstFail = e;
        }
        copyLaunchFailures++;
      } else {
        copiedAny = true;
      }
    }
    if (copyLaunchFailures != 0 && progressCounterMonitorErrorLogAllowed(g)) {
      INFO(NCCL_RAS,
           "NCCL progress counter monitor: counter DMA launch failed for %d/%zu comms on cudaDev %d; "
           "first failure rank %d comm %016" PRIx64 " (%s)",
           copyLaunchFailures, (size_t)g->registrations.nElems, g->cudaDev, firstFailRank, firstFailCommHash,
           cudaGetErrorString(firstFail));
    }

    if (copiedAny) {
      cudaError_t e = cudaEventRecord(g->copyDone, g->sideStream);
      if (e == cudaSuccess) {
        g->copyInFlight = true;
        g->copyStartNs = clockNano();
      } else if (progressCounterMonitorErrorLogAllowed(g)) {
        INFO(NCCL_RAS, "NCCL progress counter monitor: counter DMA event record failed on cudaDev %d (%s)", g->cudaDev,
             cudaGetErrorString(e));
      }
    }
  }
}

// Restore the caller's CUDA device after monitor setup or teardown.
static void restoreCudaDevice(struct ncclGpuProgressCounterMonitor* g, int savedDev, bool restore) {
  if (restore) {
    cudaError_t e = cudaSetDevice(savedDev);
    if (e != cudaSuccess && progressCounterMonitorErrorLogAllowed(g)) {
      INFO(NCCL_RAS, "NCCL progress counter monitor: failed to restore cudaDev %d (%s)", savedDev,
           cudaGetErrorString(e));
    }
  }
}

// Release a teardown reference and destroy the unused per-device worker when safe.
static void releaseGpuProgressCounterMonitorDestroyRef(struct ncclGpuProgressCounterMonitor* g) {
  bool doDelete = false;
  {
    std::lock_guard<std::mutex> globalLock(gpuProgressCounterMonitorsMu);
    std::lock_guard<std::mutex> lock(g->mutex);
    g->destroyRefs--;
    if (ncclIntruQueueEmpty(&g->registrations) && g->destroyRefs == 0) doDelete = true;
  }

  if (!doDelete) return;

  // An empty registration list removes g from gpuProgressCounterMonitors before the final destroy reference is
  // released, so new registrations cannot find it while it is joined and deleted below.
  g->cv.notify_one();
  if (g->thread.joinable()) g->thread.join();

  int savedDev = -1;
  bool restore = (cudaGetDevice(&savedDev) == cudaSuccess);
  cudaError_t setErr = cudaSetDevice(g->cudaDev);
  if (setErr != cudaSuccess && progressCounterMonitorErrorLogAllowed(g)) {
    INFO(NCCL_RAS, "NCCL progress counter monitor: cudaSetDevice(%d) during destroy failed (%s)", g->cudaDev,
         cudaGetErrorString(setErr));
  }
  if (g->copyDone != nullptr) cudaEventDestroy(g->copyDone);
  if (g->sideStream != nullptr) cudaStreamDestroy(g->sideStream);
  restoreCudaDevice(g, savedDev, restore);
  delete g;
}

// Create and start the shared progress-counter worker for one CUDA device.
static ncclResult_t createGpuProgressCounterMonitor(int cudaDev, struct ncclGpuProgressCounterMonitor** out) {
  ncclResult_t ret = ncclSuccess;
  int savedDev = -1;
  bool restore = (cudaGetDevice(&savedDev) == cudaSuccess);
  struct ncclGpuProgressCounterMonitor* g = new (std::nothrow) ncclGpuProgressCounterMonitor{};
  if (g == nullptr) return ncclSystemError;

  g->cudaDev = cudaDev;
  g->shouldStop = false;
  g->running = false;
  g->copyInFlight = false;
  g->copyStallWarned = false;
  g->copyStartNs = 0;
  g->sideStream = nullptr;
  g->copyDone = nullptr;
  g->lastStaleWarnNs = 0;
  g->lastErrorWarnNs = 0;
  g->destroyRefs = 0;
  ncclIntruQueueConstruct(&g->registrations);

  int param = (int)ncclParamProgressCounterMonitorPollMs();
  if (param < kProgressCounterMonitorMinPollMs) {
    TRACE(NCCL_RAS, "NCCL progress counter monitor: NCCL_PROGRESS_COUNTER_MONITOR_POLL_MS=%d clamped to %d", param,
          kProgressCounterMonitorMinPollMs);
  }
  int staleParam = (int)ncclParamProgressCounterMonitorStaleMs();
  if (staleParam < kProgressCounterMonitorMinStaleMs) {
    TRACE(NCCL_RAS, "NCCL progress counter monitor: NCCL_PROGRESS_COUNTER_MONITOR_STALE_MS=%d clamped to %d",
          staleParam, kProgressCounterMonitorMinStaleMs);
  }

  CUDACHECKGOTO(cudaSetDevice(cudaDev), ret, fail);
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&g->sideStream, cudaStreamNonBlocking), ret, fail);
  CUDACHECKGOTO(cudaEventCreateWithFlags(&g->copyDone, cudaEventDisableTiming), ret, fail);

  g->thread = std::thread(progressCounterMonitorLoop, g);
  ncclSetThreadName(g->thread, "NCCL CtrMon %d", cudaDev);

  {
    std::unique_lock<std::mutex> lock(g->mutex);
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(2000);
    while (!g->running && !g->shouldStop) {
      if (g->cv.wait_until(lock, deadline) == std::cv_status::timeout) break;
    }
    if (!g->running) {
      g->shouldStop = true;
      lock.unlock();
      g->cv.notify_one();
      WARN("NCCL progress counter monitor: cudaDev %d thread failed to come up within 2000 ms (worker initialization "
           "may have failed)",
           cudaDev);
      if (g->thread.joinable()) g->thread.join();
      ret = ncclSystemError;
      goto fail;
    }
  }

  *out = g;
  restoreCudaDevice(g, savedDev, restore);
  return ncclSuccess;

fail:
  if (g->copyDone != nullptr) cudaEventDestroy(g->copyDone);
  if (g->sideStream != nullptr) cudaStreamDestroy(g->sideStream);
  restoreCudaDevice(g, savedDev, restore);
  delete g;
  return ret;
}

// Register a communicator with its CUDA device's shared progress-counter worker.
ncclResult_t ncclProgressCounterMonitorInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  comm->nextProgressRegistration = nullptr;
  // No device counter storage means counter-monitor registration is disabled.
  if (comm->deviceCountersBlock == nullptr) return ncclSuccess;
  if (comm->cudaDev < 0 || comm->cudaDev >= kRasMaxCudaDevices) return ncclInvalidArgument;

  struct ncclGpuProgressCounterMonitor* g = nullptr;
  {
    std::lock_guard<std::mutex> globalLock(gpuProgressCounterMonitorsMu);
    g = gpuProgressCounterMonitors[comm->cudaDev];
    if (g == nullptr) {
      NCCLCHECKGOTO(createGpuProgressCounterMonitor(comm->cudaDev, &g), ret, fail);
      gpuProgressCounterMonitors[comm->cudaDev] = g;
      TRACE(NCCL_RAS, "NCCL progress counter monitor started for cudaDev %d: pollIntervalMs=%d", g->cudaDev,
            progressCounterMonitorPollIntervalMs());
    }

    {
      std::lock_guard<std::mutex> lock(g->mutex);
      ncclIntruQueueEnqueue(&g->registrations, comm);
    }
  }
  g->cv.notify_one();

  TRACE(NCCL_RAS, "NCCL progress counter monitor registered comm %016lx rank %d cudaDev %d",
        (unsigned long)comm->commHash, comm->rank, comm->cudaDev);
  return ncclSuccess;

fail:
  return ret;
}

static bool progressCounterRegistrationMatches(struct ncclComm* registered, struct ncclComm* target) {
  return registered == target;
}

// Unregister a communicator and drain copies before releasing its counter storage.
ncclResult_t ncclProgressCounterMonitorDestroy(struct ncclComm* comm) {
  if (comm->cudaDev < 0 || comm->cudaDev >= kRasMaxCudaDevices) return ncclSuccess;

  struct ncclGpuProgressCounterMonitor* g = nullptr;
  bool haveDestroyRef = false;

  {
    std::lock_guard<std::mutex> globalLock(gpuProgressCounterMonitorsMu);
    g = gpuProgressCounterMonitors[comm->cudaDev];
    if (g != nullptr) {
      std::lock_guard<std::mutex> lock(g->mutex);
      if (ncclIntruQueueDelete(&g->registrations, comm, progressCounterRegistrationMatches) != nullptr) {
        comm->nextProgressRegistration = nullptr;
        // Keep g alive while unregister synchronizes its stream.
        g->destroyRefs++;
        haveDestroyRef = true;
        if (ncclIntruQueueEmpty(&g->registrations)) {
          gpuProgressCounterMonitors[g->cudaDev] = nullptr;
          g->shouldStop = true;
        }
      }
    }
  }

  if (haveDestroyRef) {
    g->cv.notify_one();
    // Drain copies that may still reference this communicator's buffers.
    int savedDev = -1;
    bool restore = (cudaGetDevice(&savedDev) == cudaSuccess);
    cudaError_t setErr = cudaSetDevice(g->cudaDev);
    if (setErr != cudaSuccess) {
      if (progressCounterMonitorErrorLogAllowed(g)) {
        INFO(NCCL_RAS, "NCCL progress counter monitor: cudaSetDevice(%d) during unregister failed (%s)", g->cudaDev,
             cudaGetErrorString(setErr));
      }
    } else if (g->sideStream != nullptr) {
      cudaError_t e = cudaStreamSynchronize(g->sideStream);
      if (e != cudaSuccess && progressCounterMonitorErrorLogAllowed(g)) {
        INFO(NCCL_RAS, "NCCL progress counter monitor: side-stream sync during unregister failed on cudaDev %d (%s)",
             g->cudaDev, cudaGetErrorString(e));
      }
    }
    restoreCudaDevice(g, savedDev, restore);
  }

  if (haveDestroyRef) releaseGpuProgressCounterMonitorDestroyRef(g);
  return ncclSuccess;
}
