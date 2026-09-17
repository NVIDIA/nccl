/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"
#include <windows.h>
#include "alloc.h"
#include <unordered_map>

// Page size for alignment
static size_t ncclNdPageSize = 0;
static std::once_flag ncclNdPageSizeFlag;
static constexpr size_t ncclNdGpuPageSize = 64 * 1024;
// WinOF peer-direct MR extension; not part of the Microsoft NDv2 SPI.
static constexpr ULONG ncclNdMrFlagNvPeerDirect = 0x10000000;

struct ncclNdSyncMemopsState {
  CUdeviceptr pointer;
  unsigned int originalValue;
  int refs;
};

static std::mutex ncclNdSyncMemopsMutex;
static std::unordered_map<uint64_t, ncclNdSyncMemopsState> ncclNdSyncMemopsStates;

static ncclResult_t ncclNdCudaCallResult(CUresult cudaResult, const char* operation) {
  if (cudaResult == CUDA_SUCCESS) return ncclSuccess;
  const char* error = "unknown";
  if (CUPFN(cuGetErrorString) != NULL) (void)CUPFN(cuGetErrorString(cudaResult, &error));
  WARN("NET/ND : %s failed: CUDA error %d '%s'", operation, cudaResult, error);
  return ncclUnhandledCudaError;
}

// Hold allocation-wide synchronous CUDA semantics while any cached ND MR can
// expose a legacy CUDA allocation to GPUDirect RDMA. Windows CUDA VMM
// allocations return CUDA_ERROR_NOT_SUPPORTED and need no attribute tracking.
static ncclResult_t ncclNdAcquireSyncMemops(void* data, uint64_t* bufferId, bool* tracked) {
  *bufferId = 0;
  *tracked = false;
  if (CUPFN(cuPointerGetAttribute) == NULL || CUPFN(cuPointerSetAttribute) == NULL) {
    WARN("NET/ND : CUDA pointer attribute functions are unavailable");
    return ncclUnhandledCudaError;
  }

  unsigned int originalValue = 0;
  CUresult cudaResult =
    CUPFN(cuPointerGetAttribute(&originalValue, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)data));
  if (cudaResult == CUDA_ERROR_NOT_SUPPORTED || cudaResult == CUDA_ERROR_INVALID_VALUE) return ncclSuccess;
  NCCLCHECK(ncclNdCudaCallResult(cudaResult, "querying CU_POINTER_ATTRIBUTE_SYNC_MEMOPS"));

  uint64_t id = 0;
  NCCLCHECK(ncclNdCudaCallResult(CUPFN(cuPointerGetAttribute(&id, CU_POINTER_ATTRIBUTE_BUFFER_ID, (CUdeviceptr)data)),
                                 "querying CU_POINTER_ATTRIBUTE_BUFFER_ID"));

  std::lock_guard<std::mutex> lock(ncclNdSyncMemopsMutex);
  auto existing = ncclNdSyncMemopsStates.find(id);
  if (existing != ncclNdSyncMemopsStates.end()) {
    existing->second.refs++;
    *bufferId = id;
    *tracked = true;
    return ncclSuccess;
  }

  if (originalValue == 0) {
    unsigned int enabled = 1;
    cudaResult = CUPFN(cuPointerSetAttribute(&enabled, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)data));
    if (cudaResult == CUDA_ERROR_NOT_SUPPORTED || cudaResult == CUDA_ERROR_INVALID_VALUE) return ncclSuccess;
    NCCLCHECK(ncclNdCudaCallResult(cudaResult, "setting CU_POINTER_ATTRIBUTE_SYNC_MEMOPS"));
  }

  ncclNdSyncMemopsStates.emplace(id, ncclNdSyncMemopsState{(CUdeviceptr)data, originalValue, 1});
  *bufferId = id;
  *tracked = true;
  return ncclSuccess;
}

static ncclResult_t ncclNdReleaseSyncMemops(uint64_t bufferId) {
  std::lock_guard<std::mutex> lock(ncclNdSyncMemopsMutex);
  auto existing = ncclNdSyncMemopsStates.find(bufferId);
  if (existing == ncclNdSyncMemopsStates.end() || existing->second.refs <= 0) {
    WARN("NET/ND : Invalid CUDA allocation synchronization reference for buffer %llu", (unsigned long long)bufferId);
    return ncclInternalError;
  }
  if (--existing->second.refs != 0) return ncclSuccess;

  ncclResult_t result = ncclSuccess;
  if (existing->second.originalValue == 0) {
    uint64_t currentId = 0;
    CUresult cudaResult =
      CUPFN(cuPointerGetAttribute(&currentId, CU_POINTER_ATTRIBUTE_BUFFER_ID, existing->second.pointer));
    if (cudaResult != CUDA_SUCCESS) {
      result = ncclNdCudaCallResult(cudaResult, "revalidating CU_POINTER_ATTRIBUTE_BUFFER_ID");
    } else if (currentId != bufferId) {
      WARN("NET/ND : CUDA allocation changed before synchronization state could be restored");
      result = ncclInvalidArgument;
    } else {
      unsigned int disabled = 0;
      result = ncclNdCudaCallResult(
        CUPFN(cuPointerSetAttribute(&disabled, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, existing->second.pointer)),
        "restoring CU_POINTER_ATTRIBUTE_SYNC_MEMOPS");
    }
  }
  ncclNdSyncMemopsStates.erase(existing);
  return result;
}

// Cache and return the Windows system page size.
static size_t ncclNdGetPageSize(void) {
  std::call_once(ncclNdPageSizeFlag, []() {
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    ncclNdPageSize = sysInfo.dwPageSize;
  });
  return ncclNdPageSize;
}

// CUDA capability bits alone do not prove that a particular ND provider can
// pin a particular GPU. Probe the complete path once per GPU/adapter pair and
// cache only a successful CUDA allocation + ND registration + deregistration.
ncclResult_t ncclNdGetGpuDirectSupport(struct ncclNdDev* dev, int* supported, int* forceFlush) {
  *supported = 0;
  *forceFlush = 0;

  int cudaDev = -1;
  cudaError_t cudaResult = cudaGetDevice(&cudaDev);
  if (cudaResult != cudaSuccess || cudaDev < 0 || cudaDev >= NCCL_ND_MAX_CUDA_DEVS) {
    (void)cudaGetLastError();
    return ncclSuccess;
  }

  if (ncclCudaLibraryInit() != ncclSuccess) return ncclSuccess;

  std::lock_guard<std::mutex> lock(dev->mutex);
  if (dev->gpuCaps.gpuDirectSupported[cudaDev] != -1) {
    *supported = dev->gpuCaps.gpuDirectSupported[cudaDev];
    *forceFlush = dev->gpuCaps.forceFlush[cudaDev];
    return ncclSuccess;
  }

  // Check CUDA's coarse capability gates before allocating probe resources.
  int cudaGdr = 0;
  int flushOptions = 0;
  int writeOrdering = cudaGPUDirectRDMAWritesOrderingNone;
  bool cudaAttrsOk =
    cudaDeviceGetAttribute(&cudaGdr, cudaDevAttrGPUDirectRDMASupported, cudaDev) == cudaSuccess &&
    cudaDeviceGetAttribute(&flushOptions, cudaDevAttrGPUDirectRDMAFlushWritesOptions, cudaDev) == cudaSuccess &&
    cudaDeviceGetAttribute(&writeOrdering, cudaDevAttrGPUDirectRDMAWritesOrdering, cudaDev) == cudaSuccess;
  if (!cudaAttrsOk) {
    (void)cudaGetLastError();
    INFO(NCCL_NET, "NET/ND : Device [%d] GPU %d GPUDirect RDMA capability query failed; will retry", dev->device,
         cudaDev);
    return ncclSuccess;
  }

  int probeSupported = 0;
  int probeForceFlush = writeOrdering < cudaGPUDirectRDMAWritesOrderingOwner;
  if (!cudaGdr || (probeForceFlush && !(flushOptions & cudaFlushGPUDirectRDMAWritesOptionHost)) ||
      CUPFN(cuPointerGetAttribute) == NULL || CUPFN(cuPointerSetAttribute) == NULL) {
    dev->gpuCaps.gpuDirectSupported[cudaDev] = 0;
    dev->gpuCaps.forceFlush[cudaDev] = 0;
    INFO(NCCL_NET,
         "NET/ND : Device [%d] GPU %d does not support GPUDirect RDMA "
         "(cudaAttrs=%s cudaGdr=%d flushOptions=0x%x writeOrdering=%d "
         "hostFlush=%s cuPointerGetAttribute=%s cuPointerSetAttribute=%s)",
         dev->device, cudaDev, cudaAttrsOk ? "ok" : "error", cudaGdr, flushOptions, writeOrdering,
         (flushOptions & cudaFlushGPUDirectRDMAWritesOptionHost) ? "yes" : "no",
         CUPFN(cuPointerGetAttribute) != NULL ? "yes" : "no", CUPFN(cuPointerSetAttribute) != NULL ? "yes" : "no");
    return ncclSuccess;
  }

  void* buffer = NULL;
  HANDLE ovFile = NULL;
  struct IND2MemoryRegion* mr = NULL;
  bool registered = false;
  uint64_t bufferId = 0;
  bool syncMemopsTracked = false;
  ncclResult_t result = ncclSuccess;

  // Allocate one GPU page, enable synchronous memory semantics, and register it with ND.
  CUDACHECKGOTO(cudaMalloc(&buffer, ncclNdGpuPageSize), result, cleanup);
  NCCLCHECKGOTO(ncclNdAcquireSyncMemops(buffer, &bufferId, &syncMemopsTracked), result, cleanup);

  NCCLCHECKGOTO(wrap_nd_create_overlapped_file(dev->adapter, &ovFile), result, cleanup);
  NCCLCHECKGOTO(wrap_nd_create_memory_region(dev->adapter, ovFile, &mr), result, cleanup);
  ncclNdStatsAddResource(&dev->stats.activeMrs, &dev->stats.peakMrs);

  {
    OVERLAPPED ov = {};
    ULONG flags = ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ | ND_MR_FLAG_ALLOW_REMOTE_WRITE |
                  ncclNdMrFlagNvPeerDirect;
    NCCLCHECKGOTO(wrap_nd_register_memory(mr, buffer, ncclNdGpuPageSize, flags, &ov), result, cleanup);
    registered = true;
    probeSupported = 1;
  }

cleanup:
  // Unwind every probe resource before caching the observed capability.
  if (registered) {
    OVERLAPPED ov = {};
    NCCLCHECKIGNORE(wrap_nd_deregister_memory(mr, &ov), result);
    if (result != ncclSuccess) probeSupported = 0;
  }
  if (mr != NULL) {
    (void)wrap_nd_release(mr);
    ncclNdStatsRemoveResource(&dev->stats.activeMrs);
  }
  if (ovFile != NULL) CloseHandle(ovFile);
  if (syncMemopsTracked) {
    NCCLCHECKIGNORE(ncclNdReleaseSyncMemops(bufferId), result);
    if (result != ncclSuccess) probeSupported = 0;
  }
  if (buffer != NULL && cudaFree(buffer) != cudaSuccess) {
    (void)cudaGetLastError();
    probeSupported = 0;
  }

  // Operational probe failures can be caused by temporary CUDA, BAR, or
  // provider resource pressure. Keep the pair unknown so a later properties
  // query can retry; cache only a completed successful probe.
  if (probeSupported) {
    dev->gpuCaps.gpuDirectSupported[cudaDev] = 1;
    dev->gpuCaps.forceFlush[cudaDev] = probeForceFlush;
  }
  *supported = probeSupported;
  *forceFlush = probeSupported ? probeForceFlush : 0;
  INFO(NCCL_NET, "NET/ND : Device [%d] GPU %d GPUDirect RDMA probe %s%s%s", dev->device, cudaDev,
       probeSupported ? "passed" : "failed", *forceFlush ? " (flush required)" : "",
       probeSupported ? "" : " (will retry)");
  return ncclSuccess;
}

// Find a compatible cached registration that contains the requested range.
static struct ncclNdMr* ncclNdMrCacheLookup(struct ncclNdMrCache* cache, uintptr_t start, uintptr_t end,
                                            enum ncclNdMemType memType) {
  for (int i = 0; i < cache->population; i++) {
    struct ncclNdMr* mr = &cache->slots[i];
    if (mr->memType == memType && mr->start <= start && end <= mr->end) {
      return mr;
    }
  }
  return NULL;
}

// Grow the cache if needed and add a newly registered range.
static ncclResult_t ncclNdMrCacheInsert(struct ncclNdMrCache* cache, uintptr_t start, uintptr_t end,
                                        enum ncclNdMemType memType, uint64_t bufferId, bool syncMemopsTracked,
                                        struct IND2MemoryRegion* mr, HANDLE ovFile, struct ncclNdMr** retMr) {
  if (cache->population >= cache->capacity) {
    // Grow geometrically to keep inserts amortized constant time.
    int newCapacity = cache->capacity ? cache->capacity * 2 : 16;
    NCCLCHECK(ncclRealloc(&cache->slots, cache->capacity, newCapacity));
    cache->capacity = newCapacity;
  }

  struct ncclNdMr* cacheMr = &cache->slots[cache->population++];
  cacheMr->start = start;
  cacheMr->end = end;
  cacheMr->memType = memType;
  cacheMr->bufferId = bufferId;
  cacheMr->syncMemopsTracked = syncMemopsTracked;
  cacheMr->refs = 1;
  cacheMr->mr = mr;
  cacheMr->ovFile = ovFile;
  *retMr = cacheMr;

  return ncclSuccess;
}

// Remove a cache entry while keeping the active slots compact.
static ncclResult_t ncclNdMrCacheRemove(struct ncclNdMrCache* cache, struct ncclNdMr* mr) {
  int idx = mr - cache->slots;
  if (idx < 0 || idx >= cache->population) {
    WARN("NET/ND : Invalid MR cache entry");
    return ncclInternalError;
  }

  // Fill the removed slot with the last live entry to keep lookups dense.
  cache->population--;
  if (idx < cache->population) {
    cache->slots[idx] = cache->slots[cache->population];
  }

  return ncclSuccess;
}

// Register memory on a single device. MRs are cached globally per adapter and
// can therefore outlive the communicator that first registered the range. Give
// every cached MR its own overlapped file instead of borrowing a communicator
// handle that another cache user could close underneath it.
static ncclResult_t ncclNdRegMrDevice(struct ncclNdDev* dev, void* data, size_t size, enum ncclNdMemType memType,
                                      struct IND2MemoryRegion** retMr) {
  *retMr = NULL;
  std::lock_guard<std::mutex> lock(dev->mutex);

  // Normalize the range to provider page granularity and enforce adapter limits.
  size_t pageSize = memType == ncclNdMemTypeDevice ? ncclNdGpuPageSize : ncclNdGetPageSize();
  uintptr_t addr = (uintptr_t)data;
  uintptr_t alignedAddr = addr & ~(pageSize - 1);
  size_t offset = addr - alignedAddr;
  if (data == NULL || size == 0 || size > SIZE_MAX - offset) return ncclInvalidArgument;
  size_t alignedSize = size + offset;
  if (alignedSize > SIZE_MAX - (pageSize - 1)) return ncclInvalidArgument;
  alignedSize = ((alignedSize + pageSize - 1) / pageSize) * pageSize;
  if (alignedSize > UINTPTR_MAX - alignedAddr) return ncclInvalidArgument;
  uintptr_t alignedEnd = alignedAddr + alignedSize;
  if (dev->adapterInfo.MaxRegistrationSize != 0 && alignedSize > dev->adapterInfo.MaxRegistrationSize) {
    WARN("NET/ND : Registration size %zu exceeds adapter limit %zu", alignedSize, dev->adapterInfo.MaxRegistrationSize);
    return ncclInvalidArgument;
  }

  // Reuse a compatible adapter-global registration before consuming provider resources.
  struct ncclNdMr* cacheMr = ncclNdMrCacheLookup(&dev->mrCache, alignedAddr, alignedEnd, memType);
  if (cacheMr) {
    cacheMr->refs++;
    *retMr = cacheMr->mr;
    return ncclSuccess;
  }

  if (dev->mrCache.population >= dev->maxMrs) {
    dev->stats.mrAdmissionFailures.fetch_add(1, std::memory_order_relaxed);
    WARN("NET/ND : Device %d cached MR limit %d reached", dev->device, dev->maxMrs);
    return ncclSystemError;
  }

  // Give a new cached MR an independent overlapped file and provider lifetime.
  struct IND2MemoryRegion* mr = NULL;
  HANDLE ovFile = NULL;
  bool registered = false;
  uint64_t bufferId = 0;
  bool syncMemopsTracked = false;
  ncclResult_t result = ncclSuccess;
  OVERLAPPED ov = {};
  ULONG flags = ND_MR_FLAG_ALLOW_LOCAL_WRITE | ND_MR_FLAG_ALLOW_REMOTE_READ | ND_MR_FLAG_ALLOW_REMOTE_WRITE;
  NCCLCHECKGOTO(wrap_nd_create_overlapped_file(dev->adapter, &ovFile), result, fail);
  NCCLCHECKGOTO(wrap_nd_create_memory_region(dev->adapter, ovFile, &mr), result, fail);
  ncclNdStatsAddResource(&dev->stats.activeMrs, &dev->stats.peakMrs);

  // Enable synchronous GPU memory semantics, then register the aligned range.
  if (memType == ncclNdMemTypeDevice) {
    NCCLCHECKGOTO(ncclNdAcquireSyncMemops(data, &bufferId, &syncMemopsTracked), result, fail);
    flags |= ncclNdMrFlagNvPeerDirect;
  }
  NCCLCHECKGOTO(wrap_nd_register_memory(mr, (void*)alignedAddr, alignedSize, flags, &ov), result, fail);
  registered = true;

  // Publish the entry only after provider registration completes.
  NCCLCHECKGOTO(ncclNdMrCacheInsert(&dev->mrCache, alignedAddr, alignedEnd, memType, bufferId, syncMemopsTracked, mr,
                                    ovFile, &cacheMr),
                result, fail);
  *retMr = mr;
  return ncclSuccess;

fail:
  // Unwind in reverse order; a failed cache insert must also deregister.
  if (registered) {
    OVERLAPPED deregOv = {};
    NCCLCHECKIGNORE(wrap_nd_deregister_memory(mr, &deregOv), result);
  }
  if (mr != NULL) {
    (void)wrap_nd_release(mr);
    ncclNdStatsRemoveResource(&dev->stats.activeMrs);
  }
  if (ovFile != NULL) CloseHandle(ovFile);
  if (syncMemopsTracked) NCCLCHECKIGNORE(ncclNdReleaseSyncMemops(bufferId), result);
  return result;
}

// Release one reference and destroy the adapter MR when it reaches zero.
static ncclResult_t ncclNdDeregMrDevice(struct ncclNdDev* dev, struct IND2MemoryRegion* mr) {
  std::lock_guard<std::mutex> lock(dev->mutex);

  // Locate the provider object and its shared reference count.
  struct ncclNdMr* cacheMr = NULL;
  for (int i = 0; i < dev->mrCache.population; i++) {
    if (dev->mrCache.slots[i].mr == mr) {
      cacheMr = &dev->mrCache.slots[i];
      break;
    }
  }

  if (!cacheMr) {
    WARN("NET/ND : MR not found in cache");
    return ncclInternalError;
  }

  // Keep the provider registration alive until the final user releases it.
  if (cacheMr->refs <= 0) {
    WARN("NET/ND : Invalid MR cache reference count %d", cacheMr->refs);
    return ncclInternalError;
  }
  cacheMr->refs--;
  if (cacheMr->refs == 0) {
    HANDLE ovFile = cacheMr->ovFile;
    uint64_t bufferId = cacheMr->bufferId;
    bool syncMemopsTracked = cacheMr->syncMemopsTracked;
    OVERLAPPED ov = {};
    ncclResult_t result = wrap_nd_deregister_memory(mr, &ov);

    // Never retain a dead provider object in the cache after the caller has
    // released its last reference, even when deregistration reports failure.
    (void)wrap_nd_release(mr);
    ncclNdStatsRemoveResource(&dev->stats.activeMrs);

    // Remove the dead provider object before closing its overlapped file.
    NCCLCHECKIGNORE(ncclNdMrCacheRemove(&dev->mrCache, cacheMr), result);
    if (ovFile != NULL) CloseHandle(ovFile);
    if (syncMemopsTracked) NCCLCHECKIGNORE(ncclNdReleaseSyncMemops(bufferId), result);
    return result;
  }

  return ncclSuccess;
}

// Register a range on every physical rail and return one aggregate handle.
ncclResult_t ncclNdRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  struct ncclNdNetCommBase* base = (struct ncclNdNetCommBase*)comm;
  if (mhandle == NULL) return ncclInvalidArgument;
  *mhandle = NULL;
  if (base == NULL || data == NULL || size == 0) return ncclInvalidArgument;

  if (type != NCCL_PTR_HOST && type != NCCL_PTR_CUDA) return ncclInvalidArgument;
  enum ncclNdMemType memType = type == NCCL_PTR_CUDA ? ncclNdMemTypeDevice : ncclNdMemTypeHost;

  if (memType == ncclNdMemTypeDevice) {
    INFO(NCCL_NET, "NET/ND : Registering GPU memory at %p size %zu (GPUDirect RDMA support depends on adapter)", data,
         size);
  }

  // Allocate one aggregate handle for all communicator rails.
  struct ncclNdMrHandle* handle;
  NCCLCHECK(ncclCalloc(&handle, 1));

  // Validate the authoritative rail count before starting any registration.
  int ndevs = base->vProps.ndevs;
  if (ndevs <= 0 || ndevs > NCCL_ND_MAX_DEVS_PER_NIC || ndevs != base->ndevs) {
    WARN("NET/ND : Inconsistent device count: vProps.ndevs=%d base->ndevs=%d", ndevs, base->ndevs);
    free(handle);
    return ncclInternalError;
  }
  handle->ndevs = ndevs;
  ncclResult_t result = ncclSuccess;
  int registeredDevices = 0;
  for (int i = 0; i < ndevs; i++) {
    int devIdx = base->vProps.devs[i];
    // Validate each rail and its current-GPU capability before registering.
    if (devIdx < 0 || devIdx >= ncclNNdDevs) {
      WARN("NET/ND : Invalid device index %d in vProps (ncclNNdDevs=%d)", devIdx, ncclNNdDevs);
      result = ncclInternalError;
      goto fail;
    }
    handle->devs[i] = devIdx;
    struct ncclNdDev* dev = &ncclNdDevs[devIdx];
    if (memType == ncclNdMemTypeDevice) {
      int supported = 0, forceFlush = 0;
      NCCLCHECKGOTO(ncclNdGetGpuDirectSupport(dev, &supported, &forceFlush), result, fail);
      if (!supported) {
        WARN("NET/ND : Device [%d] cannot register memory from the current CUDA device", devIdx);
        result = ncclInternalError;
        goto fail;
      }
    }
    NCCLCHECKGOTO(ncclNdRegMrDevice(dev, data, size, memType, &handle->mrs[i]), result, fail);
    registeredDevices++;
  }

  *mhandle = handle;
  return ncclSuccess;

fail:
  // Roll back every earlier rail before releasing the aggregate handle.
  for (int i = 0; i < registeredDevices; i++) {
    int devIdx = handle->devs[i];
    ncclResult_t cleanupResult = ncclNdDeregMrDevice(&ncclNdDevs[devIdx], handle->mrs[i]);
    if (cleanupResult != ncclSuccess) {
      WARN("NET/ND : Failed to roll back memory registration on device %d", devIdx);
    }
  }
  free(handle);
  if (base->primaryDev >= 0 && base->primaryDev < ncclNNdDevs) {
    ncclNdDevs[base->primaryDev].stats.registrationFailures.fetch_add(1, std::memory_order_relaxed);
  }
  return result;
}

// Reject DMA-BUF registration, which NetworkDirect does not support.
ncclResult_t ncclNdRegMrDmaBuf(void*, void*, size_t, int, uint64_t, int, void**) {
  WARN("NET/ND : DMA-BUF registration not yet supported");
  return ncclInternalError;
}

// Deregister an aggregate handle from every physical rail.
ncclResult_t ncclNdDeregMr(void*, void* mhandle) {
  struct ncclNdMrHandle* handle = (struct ncclNdMrHandle*)mhandle;

  if (!handle) return ncclSuccess;
  if (handle->ndevs <= 0 || handle->ndevs > NCCL_ND_MAX_DEVS_PER_NIC) {
    WARN("NET/ND : Invalid MR handle device count %d", handle->ndevs);
    return ncclInvalidArgument;
  }

  // Validate every live entry before releasing any of them so malformed
  // handles remain intact and can be diagnosed or retried safely.
  for (int i = 0; i < handle->ndevs; i++) {
    if (handle->mrs[i] != NULL && (handle->devs[i] < 0 || handle->devs[i] >= ncclNNdDevs)) {
      WARN("NET/ND : Invalid device index %d in MR handle", handle->devs[i]);
      return ncclInvalidArgument;
    }
  }

  // Release every rail even if an earlier deregistration reports failure.
  ncclResult_t result = ncclSuccess;
  for (int i = 0; i < handle->ndevs; i++) {
    if (handle->mrs[i]) {
      int devIdx = handle->devs[i];
      struct ncclNdDev* dev = &ncclNdDevs[devIdx];
      NCCLCHECKIGNORE(ncclNdDeregMrDevice(dev, handle->mrs[i]), result);
      handle->mrs[i] = NULL;
    }
  }

  free(handle);
  return result;
}

// Finalization is the last ownership boundary for adapter-global registrations.
// Drain every provider object even if a caller leaked a registration handle so
// the provider DLL is never unloaded while MRs still reference it.
ncclResult_t ncclNdFinalizeMrCache(struct ncclNdDev* dev) {
  std::lock_guard<std::mutex> lock(dev->mutex);
  ncclResult_t result = ncclSuccess;
  for (int i = 0; i < dev->mrCache.population; i++) {
    struct ncclNdMr* entry = &dev->mrCache.slots[i];
    if (entry->refs != 0) {
      WARN("NET/ND : Finalizing adapter %d with MR %p still referenced %d time(s)", dev->device, entry->mr,
           entry->refs);
    }
    if (entry->mr != NULL) {
      OVERLAPPED ov = {};
      NCCLCHECKIGNORE(wrap_nd_deregister_memory(entry->mr, &ov), result);
      (void)wrap_nd_release(entry->mr);
      entry->mr = NULL;
      ncclNdStatsRemoveResource(&dev->stats.activeMrs);
    }
    if (entry->ovFile != NULL) {
      if (!CloseHandle(entry->ovFile) && result == ncclSuccess) result = ncclSystemError;
      entry->ovFile = NULL;
    }
    if (entry->syncMemopsTracked) {
      NCCLCHECKIGNORE(ncclNdReleaseSyncMemops(entry->bufferId), result);
      entry->syncMemopsTracked = false;
    }
  }
  free(dev->mrCache.slots);
  dev->mrCache.slots = NULL;
  dev->mrCache.capacity = 0;
  dev->mrCache.population = 0;
  return result;
}
