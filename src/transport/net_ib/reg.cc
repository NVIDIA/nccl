/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "common.h"

ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize-1)/pageSize;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int slot=0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) { // didn't find in cache
      if (cache->population == cache->capacity) { // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2*cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags = IBV_ACCESS_LOCAL_WRITE|IBV_ACCESS_REMOTE_WRITE|IBV_ACCESS_REMOTE_READ|IBV_ACCESS_REMOTE_ATOMIC;
      bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        if (!ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
          NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags));
        } else {
          NCCLCHECK(wrap_mlx5dv_reg_dmabuf_mr(&mr, base->pd, offset, pages*pageSize, addr, fd, flags, MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT));
        }
      } else {
        if (relaxedOrdering) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages*pageSize, addr, flags));
        }
        else {
          NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages*pageSize, flags));
        }
      }
      TRACE(NCCL_INIT|NCCL_NET,"regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr, (long long)pages*pageSize, mr->rkey, mr->lkey, fd);
      if (slot != cache->population) memmove(cache->slots+slot+1, cache->slots+slot, (cache->population-slot)*sizeof(struct ncclIbMr));
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot].addr) &&
        ((addr-cache->slots[slot].addr)/pageSize+pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  assert(size > 0);
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) malloc(sizeof(struct ncclIbMrHandle));
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i), ret, fail);
  }
  *mhandle = (void*) mhandleWrapper;
exit:
  return ret;
fail:
  free(mhandleWrapper);
  goto exit;
}

ncclResult_t ncclIbRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, offset, fd, 0ULL, mhandle);
}

ncclResult_t ncclIbRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclIbRegMrDmaBufInternal(comm, data, size, type, 0ULL, -1, 0, mhandle);
}

ncclResult_t ncclIbDeregMrInternal(ncclIbNetCommDevBase* base, ibv_mr* mhandle) {
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int i=0; i < cache->population; i++) {
    if (mhandle == cache->slots[i].mr) {
      if (0 == --cache->slots[i].refs) {
        memmove(&cache->slots[i], &cache->slots[--cache->population], sizeof(struct ncclIbMr));
        if (cache->population == 0) {
          free(cache->slots);
          cache->slots = NULL;
          cache->capacity = 0;
        }
        NCCLCHECK(wrap_ibv_dereg_mr(mhandle));
      }
      return ncclSuccess;
    }
  }
  WARN("NET/IB: could not find mr %p inside cache of %d entries", mhandle, cache->population);
  return ncclInternalError;
}

ncclResult_t ncclIbDeregMr(void* comm, void* mhandle) {
  if (mhandle == NULL) return ncclSuccess;

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*) mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*) comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
