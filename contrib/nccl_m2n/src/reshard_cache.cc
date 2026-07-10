/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstdio>
#include <cstring>
#include <vector>
#include "nccl.h"

// Host-only TU: pull host-visible types/decls from nccl_device.h. Scope
// the macro to this include so it doesn't leak to subsequent headers.
#define NCCL_HOSTLIB_ONLY
#include "nccl_device.h"
#undef NCCL_HOSTLIB_ONLY

#include "reshard_types.h"
#include "m2n_log.h"
#include "m2n_checks.h"
#include "reshard_internal.h"

struct DevCommCacheEntry {
  ncclComm_t comm;
  int numCtas;
  int ginSignalCount;
  cudaStream_t stream;
  bool valid;
  ncclDevComm devComm;
};

/* One pool entry per (comm, dev) — a non-blocking stream paired with
 * the back-edge event we record on it.  The event is reused across
 * calls so we avoid cudaEvent{Create,Destroy} on every reshard. */
struct StreamPoolEntry {
  ncclComm_t comm;
  int dev;
  cudaStream_t stream = nullptr;
  cudaEvent_t event = nullptr;
};

static WindowCache gInternalWindowCache = {};
static DevCommCacheEntry gDevcommCache[MAX_DEVCOMM_CACHE_ENTRIES];
static int gDevcommCacheCount = 0;
static int gDevcommCacheNextIdx = 0;

/* CUDA handle lifetimes are tied to ncclM2nFinalize() (so we can
 * destroy them before the CUDA context tears down) — see cacheFinalize.
 * The vector itself just owns memory; handles inside it are released
 * explicitly there before the vector is cleared. */
static std::vector<StreamPoolEntry> gStreamPool;

static ncclWindow_t* findCachedWindowByPtr(WindowCache* cache, ncclComm_t comm, void* buffer, size_t size) {
  for (int i = 0; i < cache->count; i++) {
    WindowCacheEntry& e = cache->entries[i];
    if (e.valid && e.comm == comm && e.windowBuffer == buffer && e.windowSize == size) return &e.window;
  }
  return nullptr;
}

static ncclResult_t cacheWindow(WindowCache* cache, ncclComm_t comm, void* windowBuffer, size_t windowSize,
                                ncclWindow_t window) {
  int idx;
  if (cache->count >= MAX_WINDOW_CACHE_ENTRIES) {
    idx = cache->nextIdx;
    WindowCacheEntry& old = cache->entries[idx];
    RESHARD_WARN(-1, "Window cache full (%d entries), replacing entry at index %d", MAX_WINDOW_CACHE_ENTRIES, idx);
    if (old.valid) NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(old.comm, old.window));
    cache->nextIdx = (cache->nextIdx + 1) % MAX_WINDOW_CACHE_ENTRIES;
  } else {
    idx = cache->count++;
  }
  WindowCacheEntry& e = cache->entries[idx];
  e.comm = comm;
  e.windowBuffer = windowBuffer;
  e.windowSize = windowSize;
  e.window = window;
  e.valid = true;
  return ncclSuccess;
}

ncclWindow_t* findCachedInternalWindowByPtr(ncclComm_t comm, void* buffer, size_t size) {
  return findCachedWindowByPtr(&gInternalWindowCache, comm, buffer, size);
}

ncclResult_t cacheInternalWindow(ncclComm_t comm, void* buffer, size_t size, ncclWindow_t window) {
  return cacheWindow(&gInternalWindowCache, comm, buffer, size, window);
}

ncclDevComm* findCachedDevComm(ncclComm_t comm, int numCtas, int signalCount, cudaStream_t stream) {
  for (int i = 0; i < gDevcommCacheCount; i++) {
    DevCommCacheEntry& e = gDevcommCache[i];
    if (e.valid && e.comm == comm && e.numCtas == numCtas && e.ginSignalCount == signalCount && e.stream == stream)
      return &e.devComm;
  }
  return nullptr;
}

ncclResult_t cacheDevComm(ncclComm_t comm, int numCtas, int signalCount, const ncclDevComm* devComm,
                          cudaStream_t stream) {
  int idx;
  if (gDevcommCacheCount >= MAX_DEVCOMM_CACHE_ENTRIES) {
    idx = gDevcommCacheNextIdx;
    DevCommCacheEntry& old = gDevcommCache[idx];
    RESHARD_WARN(-1, "DevComm cache full (%d entries), replacing entry at index %d", MAX_DEVCOMM_CACHE_ENTRIES, idx);
    if (old.valid) NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(old.comm, &old.devComm));
    gDevcommCacheNextIdx = (gDevcommCacheNextIdx + 1) % MAX_DEVCOMM_CACHE_ENTRIES;
  } else {
    idx = gDevcommCacheCount++;
  }
  DevCommCacheEntry& e = gDevcommCache[idx];
  e.comm = comm;
  e.numCtas = numCtas;
  e.ginSignalCount = signalCount;
  e.stream = stream;
  e.devComm = *devComm;
  e.valid = true;
  return ncclSuccess;
}

void cacheFinalize() {
  for (int i = 0; i < gInternalWindowCache.count; i++) {
    WindowCacheEntry& e = gInternalWindowCache.entries[i];
    if (e.valid) {
      NCCL_M2N_CHECK_WARN(ncclCommWindowDeregister(e.comm, e.window));
      e.valid = false;
    }
  }
  gInternalWindowCache.count = 0;
  gInternalWindowCache.nextIdx = 0;

  for (int i = 0; i < gDevcommCacheCount; i++) {
    DevCommCacheEntry& e = gDevcommCache[i];
    if (e.valid) {
      NCCL_M2N_CHECK_WARN(ncclDevCommDestroy(e.comm, &e.devComm));
      e.valid = false;
    }
  }
  gDevcommCacheCount = 0;

  for (StreamPoolEntry& e : gStreamPool) {
    if (e.event != nullptr) NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(e.event));
    if (e.stream != nullptr) NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(e.stream));
  }
  gStreamPool.clear();
}

ncclResult_t streamPoolAcquire(ncclComm_t comm, int dev, cudaStream_t* outStream, cudaEvent_t* outEvent) {
  if (outStream == nullptr || outEvent == nullptr) return ncclInvalidArgument;
  /* Pool disabled (NCCL_RESHARD_STREAM_POOL_SIZE <= 0) — caller
   * should have gated on reshardGetStreamPoolSize() > 0; defend
   * anyway so a forgotten gate doesn't UB. */
  const int maxEntries = reshardGetStreamPoolSize();
  if (maxEntries <= 0) return ncclInvalidArgument;

  /* Find existing entry for (comm, dev). */
  for (StreamPoolEntry& e : gStreamPool) {
    if (e.comm == comm && e.dev == dev) {
      *outStream = e.stream;
      *outEvent = e.event;
      return ncclSuccess;
    }
  }
  /* Pool full — soft fall-through.  Caller checks *outEvent ==
   * nullptr and runs on the user's default stream for this call. */
  if ((int)gStreamPool.size() >= maxEntries) {
    RESHARD_WARN(-1,
                 "Stream pool full (%d entries, "
                 "NCCL_RESHARD_STREAM_POOL_SIZE=%d); "
                 "falling through to the caller's default stream for this (comm, "
                 "dev) pair.  Bump NCCL_RESHARD_STREAM_POOL_SIZE if your "
                 "workload "
                 "uses more distinct (comm, dev) pairs.",
                 (int)gStreamPool.size(), maxEntries);
    *outStream = nullptr;
    *outEvent = nullptr;
    return ncclSuccess;
  }
  /* Lazy-create stream + event for the new (comm, dev). */
  StreamPoolEntry fresh;
  fresh.comm = comm;
  fresh.dev = dev;
  if (cudaStreamCreateWithFlags(&fresh.stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&fresh.event, cudaEventDisableTiming) != cudaSuccess) {
    if (fresh.event != nullptr) NCCL_M2N_CUDACHECK_WARN(cudaEventDestroy(fresh.event));
    if (fresh.stream != nullptr) NCCL_M2N_CUDACHECK_WARN(cudaStreamDestroy(fresh.stream));
    return ncclSystemError;
  }
  gStreamPool.push_back(fresh);
  *outStream = fresh.stream;
  *outEvent = fresh.event;
  return ncclSuccess;
}
