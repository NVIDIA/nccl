/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include <limits.h>

// NetworkDirect shares the existing InfiniBand controls with equivalent
// semantics instead of exposing a second transport-specific environment API.
NCCL_PARAM(NdDisable, "IB_DISABLE", 0);
NCCL_PARAM(NdUseInline, "IB_USE_INLINE", 0);
NCCL_PARAM(NdQpsPerConnection, "IB_QPS_PER_CONNECTION", 1);
NCCL_PARAM(NdResiliencyPortFailover, "IB_RESILIENCY_PORT_FAILOVER", 0);

// Global device state
int ncclNNdDevs = -1;
int ncclNMergedNdDevs = -1;
struct ncclNdDev ncclNdDevs[MAX_ND_DEVS];
struct ncclNdMergedDev ncclNdMergedDevs[MAX_ND_VDEVS];
union ncclSocketAddress ncclNdSocketAddr;
static std::atomic<uint64_t> ncclNdNextCommId{1};
static constexpr LONG ncclNdConnectionPending = 0;
static constexpr LONG ncclNdConnectionSucceeded = 1;
static constexpr LONG ncclNdConnectionFailed = 2;

// Return whether shutdown statistics contain an operational anomaly worth reporting.
static bool ncclNdStatsRequireReport(const struct ncclNdStats* stats) {
  return stats->fatalErrorCount.load(std::memory_order_relaxed) != 0 ||
         stats->commBackpressure.load(std::memory_order_relaxed) != 0 ||
         stats->requestBackpressure.load(std::memory_order_relaxed) != 0 ||
         stats->mrAdmissionFailures.load(std::memory_order_relaxed) != 0 ||
         stats->connectionFailures.load(std::memory_order_relaxed) != 0 ||
         stats->connectionTimeouts.load(std::memory_order_relaxed) != 0 ||
         stats->dataTimeouts.load(std::memory_order_relaxed) != 0 ||
         stats->peerDisconnects.load(std::memory_order_relaxed) != 0 ||
         stats->cqErrors.load(std::memory_order_relaxed) != 0 ||
         stats->registrationFailures.load(std::memory_order_relaxed) != 0 ||
         stats->providerNotifications.load(std::memory_order_relaxed) != 0 ||
         stats->adapterHealthFailures.load(std::memory_order_relaxed) != 0 ||
         stats->linkEvents.load(std::memory_order_relaxed) != 0 ||
         stats->railFailures.load(std::memory_order_relaxed) != 0 ||
         stats->railFailovers.load(std::memory_order_relaxed) != 0 ||
         stats->replayedRequests.load(std::memory_order_relaxed) != 0 ||
         stats->failoverFailures.load(std::memory_order_relaxed) != 0;
}

// Increment the active-resource count and update the peak if this sets a new maximum.
void ncclNdStatsAddResource(std::atomic<int>* active, std::atomic<int>* peak) {
  int value = active->fetch_add(1, std::memory_order_relaxed) + 1;
  int observed = peak->load(std::memory_order_relaxed);
  while (value > observed && !peak->compare_exchange_weak(observed, value, std::memory_order_relaxed)) {
  }
}

// Decrement a live-resource counter and diagnose accounting underflow.
void ncclNdStatsRemoveResource(std::atomic<int>* active) {
  int previous = active->fetch_sub(1, std::memory_order_relaxed);
  if (previous <= 0) {
    active->store(0, std::memory_order_relaxed);
    WARN("NET/ND : Resource accounting underflow");
  }
}

// Resolve a physical or virtual device to its primary physical adapter.
static int ncclNdPrimaryDevice(int dev) {
  if (dev >= 0 && dev < ncclNNdDevs) return dev;
  int vDev = dev - ncclNNdDevs;
  if (vDev >= 0 && vDev < ncclNMergedNdDevs && ncclNdMergedDevs[vDev].vProps.ndevs > 0) {
    int physical = ncclNdMergedDevs[vDev].vProps.devs[0];
    if (physical >= 0 && physical < ncclNNdDevs) return physical;
  }
  return -1;
}

// Assign a stable communicator ID and begin connection accounting.
void ncclNdStatsConnectionStart(struct ncclNdNetCommBase* base, int dev) {
  if (base->commId != 0) return;
  base->commId = ncclNdNextCommId.fetch_add(1, std::memory_order_relaxed);
  base->primaryDev = ncclNdPrimaryDevice(dev);
  base->stats = base->primaryDev >= 0 ? &ncclNdDevs[base->primaryDev].stats : NULL;
  InterlockedExchange(&base->connectionOutcome, ncclNdConnectionPending);
  if (base->primaryDev >= 0) {
    ncclNdDevs[base->primaryDev].stats.connectionAttempts.fetch_add(1, std::memory_order_relaxed);
  }
}

// Record the first successful outcome for a connection attempt.
void ncclNdStatsConnectionComplete(struct ncclNdNetCommBase* base) {
  if (InterlockedCompareExchange(&base->connectionOutcome, ncclNdConnectionSucceeded, ncclNdConnectionPending) ==
        ncclNdConnectionPending &&
      base->primaryDev >= 0 && base->primaryDev < ncclNNdDevs) {
    ncclNdDevs[base->primaryDev].stats.connectionSuccesses.fetch_add(1, std::memory_order_relaxed);
  }
}

// Record the first failed outcome and whether it timed out.
void ncclNdStatsConnectionFailed(struct ncclNdNetCommBase* base, bool timedOut) {
  if (InterlockedCompareExchange(&base->connectionOutcome, ncclNdConnectionFailed, ncclNdConnectionPending) ==
        ncclNdConnectionPending &&
      base->primaryDev >= 0 && base->primaryDev < ncclNNdDevs) {
    struct ncclNdStats* stats = &ncclNdDevs[base->primaryDev].stats;
    stats->connectionFailures.fetch_add(1, std::memory_order_relaxed);
    if (timedOut) stats->connectionTimeouts.fetch_add(1, std::memory_order_relaxed);
  }
}

// Report live-resource leaks and lifetime transport counters for one adapter.
void ncclNdLogStats(struct ncclNdDev* dev) {
  struct ncclNdStats* stats = &dev->stats;
  int activeComms = stats->activeComms.load(std::memory_order_relaxed);
  int activeQps = stats->activeQps.load(std::memory_order_relaxed);
  int activeCqs = stats->activeCqs.load(std::memory_order_relaxed);
  int activeMrs = stats->activeMrs.load(std::memory_order_relaxed);
  int activeRequests = stats->activeRequests.load(std::memory_order_relaxed);
  bool hasLiveResources = activeComms || activeQps || activeCqs || activeMrs || activeRequests;
  if (hasLiveResources) {
    WARN("NET/ND : Device %d finalizing with live resources "
         "(comm=%d qp=%d cq=%d mr=%d request=%d)",
         dev->device, activeComms, activeQps, activeCqs, activeMrs, activeRequests);
  }
  if (!hasLiveResources && !ncclNdStatsRequireReport(stats)) return;
  INFO(NCCL_NET,
       "NET/ND : Device %d stats active=%d/%d/%d/%d/%d "
       "peak=%d/%d/%d/%d/%d connection=%llu/%llu/%llu timeout=%llu "
       "fatal=%d dataTimeout=%llu disconnect=%llu cqError=%llu regFailure=%llu "
       "backpressure=%llu/%llu mrAdmission=%llu inlineWrites=%llu health=%llu "
       "provider=%llu adapterHealth=%llu linkEvents=%llu rail=%llu/%llu replay=%llu/%llu",
       dev->device, activeComms, activeQps, activeCqs, activeMrs, activeRequests,
       stats->peakComms.load(std::memory_order_relaxed), stats->peakQps.load(std::memory_order_relaxed),
       stats->peakCqs.load(std::memory_order_relaxed), stats->peakMrs.load(std::memory_order_relaxed),
       stats->peakRequests.load(std::memory_order_relaxed),
       (unsigned long long)stats->connectionAttempts.load(std::memory_order_relaxed),
       (unsigned long long)stats->connectionSuccesses.load(std::memory_order_relaxed),
       (unsigned long long)stats->connectionFailures.load(std::memory_order_relaxed),
       (unsigned long long)stats->connectionTimeouts.load(std::memory_order_relaxed),
       stats->fatalErrorCount.load(std::memory_order_relaxed),
       (unsigned long long)stats->dataTimeouts.load(std::memory_order_relaxed),
       (unsigned long long)stats->peerDisconnects.load(std::memory_order_relaxed),
       (unsigned long long)stats->cqErrors.load(std::memory_order_relaxed),
       (unsigned long long)stats->registrationFailures.load(std::memory_order_relaxed),
       (unsigned long long)stats->commBackpressure.load(std::memory_order_relaxed),
       (unsigned long long)stats->requestBackpressure.load(std::memory_order_relaxed),
       (unsigned long long)stats->mrAdmissionFailures.load(std::memory_order_relaxed),
       (unsigned long long)stats->inlineWrites.load(std::memory_order_relaxed),
       (unsigned long long)stats->healthChecks.load(std::memory_order_relaxed),
       (unsigned long long)stats->providerNotifications.load(std::memory_order_relaxed),
       (unsigned long long)stats->adapterHealthFailures.load(std::memory_order_relaxed),
       (unsigned long long)stats->linkEvents.load(std::memory_order_relaxed),
       (unsigned long long)stats->railFailures.load(std::memory_order_relaxed),
       (unsigned long long)stats->railFailovers.load(std::memory_order_relaxed),
       (unsigned long long)stats->replayedRequests.load(std::memory_order_relaxed),
       (unsigned long long)stats->failoverFailures.load(std::memory_order_relaxed));
}

// Reserve one communicator slot on a physical adapter.
static bool ncclNdTryAcquireDeviceComm(struct ncclNdDev* dev) {
  int current = dev->stats.activeComms.load(std::memory_order_relaxed);
  while (current < dev->maxComms) {
    if (dev->stats.activeComms.compare_exchange_weak(current, current + 1, std::memory_order_acq_rel)) {
      int peak = dev->stats.peakComms.load(std::memory_order_relaxed);
      while (current + 1 > peak &&
             !dev->stats.peakComms.compare_exchange_weak(peak, current + 1, std::memory_order_relaxed)) {
      }
      return true;
    }
  }
  dev->stats.commBackpressure.fetch_add(1, std::memory_order_relaxed);
  return false;
}

// Reserve communicator capacity on every rail, rolling back partial admission.
bool ncclNdTryAcquireComm(struct ncclNdNetCommBase* base) {
  if (base->admissionMask != 0) return true;
  uint32_t acquired = 0;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    int devIndex = base->vProps.devs[i];
    if (devIndex < 0 || devIndex >= ncclNNdDevs || i >= 32 || !ncclNdTryAcquireDeviceComm(&ncclNdDevs[devIndex])) {
      for (int rollback = 0; rollback < i; rollback++) {
        if (acquired & (1u << rollback)) {
          ncclNdStatsRemoveResource(&ncclNdDevs[base->vProps.devs[rollback]].stats.activeComms);
        }
      }
      return false;
    }
    acquired |= 1u << i;
  }
  base->admissionMask = acquired;
  base->primaryDev = base->vProps.devs[0];
  base->stats = &ncclNdDevs[base->primaryDev].stats;

  // Snapshot each rail's failure generation so later link events are detectable.
  for (int i = 0; i < base->vProps.ndevs; i++) {
    base->linkFailureGenerations[i] =
      InterlockedCompareExchange64(&ncclNdDevs[base->vProps.devs[i]].linkFailureGeneration, 0, 0);
  }
  return true;
}

// Release every physical-rail slot held by a communicator.
void ncclNdReleaseComm(struct ncclNdNetCommBase* base) {
  uint32_t acquired = base->admissionMask;
  base->admissionMask = 0;
  for (int i = 0; i < base->vProps.ndevs && i < 32; i++) {
    int devIndex = base->vProps.devs[i];
    if ((acquired & (1u << i)) && devIndex >= 0 && devIndex < ncclNNdDevs) {
      ncclNdStatsRemoveResource(&ncclNdDevs[devIndex].stats.activeComms);
    }
  }
}

// Reserve and initialize a request-pool slot with its completion deadline.
ncclResult_t ncclNdGetRequest(struct ncclNdNetCommBase* base, int type, struct ncclNdRequest** req) {
  ncclNdScopedSrwLock lock(&base->reqLock);
  for (int i = 0; i < NET_ND_MAX_REQUESTS; i++) {
    struct ncclNdRequest* r = &base->reqs[i];
    if (r->type == NCCL_NET_ND_REQ_UNUSED) {
      memset(r, 0, sizeof(*r));
      r->type = type;
      r->base = base;
      int64_t timeoutSeconds = NCCL_ND_DATA_TIMEOUT_SECONDS;
      if (timeoutSeconds > 0) {
        ULONGLONG now = GetTickCount64();
        ULONGLONG maxSeconds = (ULLONG_MAX - now) / 1000;
        r->deadlineMs = now + (ULONGLONG)(timeoutSeconds > (int64_t)maxSeconds ? maxSeconds : timeoutSeconds) * 1000;
      }
      *req = r;
      if (base->primaryDev >= 0 && base->primaryDev < ncclNNdDevs) {
        struct ncclNdStats* stats = &ncclNdDevs[base->primaryDev].stats;
        ncclNdStatsAddResource(&stats->activeRequests, &stats->peakRequests);
      }
      return ncclSuccess;
    }
  }
  *req = NULL;
  if (base->primaryDev >= 0 && base->primaryDev < ncclNNdDevs) {
    ncclNdDevs[base->primaryDev].stats.requestBackpressure.fetch_add(1, std::memory_order_relaxed);
  }
  return ncclSuccess;
}

// Return a request slot to its communicator pool.
ncclResult_t ncclNdFreeRequest(struct ncclNdRequest* r) {
  if (r == NULL) return ncclSuccess;
  ncclNdScopedSrwLock lock(&r->base->reqLock);
  if (r->type != NCCL_NET_ND_REQ_UNUSED && r->base->primaryDev >= 0 && r->base->primaryDev < ncclNNdDevs) {
    ncclNdStatsRemoveResource(&ncclNdDevs[r->base->primaryDev].stats.activeRequests);
  }
  r->type = NCCL_NET_ND_REQ_UNUSED;
  return ncclSuccess;
}

// NetworkDirect plugin function table
ncclNet_t ncclNetNd = {
  "NetworkDirect",
  ncclNdInit,
  ncclNdDevices,
  ncclNdGetProperties,
  ncclNdListen,
  ncclNdConnect,
  ncclNdAccept,
  ncclNdRegMr,
  ncclNdRegMrDmaBuf,
  ncclNdDeregMr,
  ncclNdIsend,
  ncclNdIrecv,
  ncclNdIflush,
  ncclNdTest,
  ncclNdCloseSend,
  ncclNdCloseRecv,
  ncclNdCloseListen,
  NULL, // getDeviceMr
  NULL, // irecvConsumed
  ncclNdMakeVDevice,
  ncclNdFinalize,
  ncclNdSetNetAttr,
};
