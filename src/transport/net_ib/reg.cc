/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "common.h"

ncclResult_t ncclIbRegMrDmaBufInternal2(ncclIbNetCommDevBase* base, void* data, size_t size, int type, uint64_t offset,
                                        int fd, uint64_t mrFlags, ibv_mr** mhandle) {
  static thread_local uintptr_t pageSize = 0;
  if (pageSize == 0) pageSize = sysconf(_SC_PAGESIZE);
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  uintptr_t addr = (uintptr_t)data & -pageSize;
  size_t pages = ((uintptr_t)data + size - addr + pageSize - 1) / pageSize;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  for (int slot = 0; /*true*/; slot++) {
    if (slot == cache->population || addr < cache->slots[slot].addr) {
      // didn't find in cache
      if (cache->population == cache->capacity) {
        // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      // Deregister / register
      struct ibv_mr* mr;
      unsigned int flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
      bool relaxedOrdering = ncclIbRelaxedOrderingEnabled && (mrFlags & NCCL_NET_MR_FLAG_FORCE_SO) == 0;
      if (relaxedOrdering) flags |= IBV_ACCESS_RELAXED_ORDERING;
      if (fd != -1) {
        /* DMA-BUF support */
        if (!ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect) {
          NCCLCHECK(wrap_ibv_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags));
        } else {
          NCCLCHECK(wrap_mlx5dv_reg_dmabuf_mr(&mr, base->pd, offset, pages * pageSize, addr, fd, flags,
                                              MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT));
        }
      } else {
        if (relaxedOrdering) {
          // Use IBVERBS_1.8 API - needed for IBV_ACCESS_RELAXED_ORDERING support
          NCCLCHECK(wrap_ibv_reg_mr_iova2(&mr, base->pd, (void*)addr, pages * pageSize, addr, flags));
        } else {
          NCCLCHECK(wrap_ibv_reg_mr(&mr, base->pd, (void*)addr, pages * pageSize, flags));
        }
      }
      TRACE(NCCL_INIT | NCCL_NET, "regAddr=0x%lx size=%lld rkey=0x%x lkey=0x%x fd=%d", (unsigned long)addr,
            (long long)pages * pageSize, mr->rkey, mr->lkey, fd);
      if (slot != cache->population) {
        memmove(cache->slots + slot + 1, cache->slots + slot, (cache->population - slot) * sizeof(struct ncclIbMr));
      }
      cache->slots[slot].addr = addr;
      cache->slots[slot].pages = pages;
      cache->slots[slot].refs = 1;
      cache->slots[slot].mr = mr;
      cache->slots[slot].cuda = type == NCCL_PTR_CUDA;
      cache->slots[slot].dmabuf = fd != -1;
      cache->slots[slot].dataDirect = ncclIbDevs[base->ibDevN].capsProvider.mlx5.dataDirect;
      cache->slots[slot].parked = false;
      cache->slots[slot].access = flags;
      cache->population += 1;
      *mhandle = mr;
      return ncclSuccess;
    } else if ((addr >= cache->slots[slot].addr) &&
               ((addr - cache->slots[slot].addr) / pageSize + pages) <= cache->slots[slot].pages) {
      cache->slots[slot].refs += 1;
      *mhandle = cache->slots[slot].mr;
      return ncclSuccess;
    }
  }
  return ncclSuccess;
}

// Park (resume=false) the registration of a GPU MR on a placeholder host page, so it no longer pins the GPU pages, or
// register the range again on the pages that back it now (resume=true). The new registration is moved into the
// existing ibv_mr, so every handle referencing it stays valid, including handles captured in CUDA graphs; NCCL reads
// the keys from the ibv_mr on every post. An MR shared by several handles is moved once.
static ncclResult_t ncclIbMrMove(ncclIbNetCommDevBase* base, ibv_mr* mr, bool resume, int dmabufFlags) {
  static uintptr_t pageSize = sysconf(_SC_PAGESIZE);
  static void* placeholder = [] {
    void* page = nullptr;
    return posix_memalign(&page, pageSize, pageSize) == 0 ? page : nullptr;
  }();
  struct ncclIbMrCache* cache = &ncclIbDevs[base->ibDevN].mrCache;
  std::lock_guard<std::mutex> lock(ncclIbDevs[base->ibDevN].mutex);
  struct ncclIbMr* slot = nullptr;
  for (int i = 0; i < cache->population && slot == nullptr; i++) {
    if (cache->slots[i].mr == mr) slot = cache->slots + i;
  }
  if (slot == nullptr) {
    WARN("NET/IB: could not find mr %p inside cache of %d entries", mr, cache->population);
    return ncclInternalError;
  }
  // Host memory keeps its pages; parking and resuming an MR twice is a no-op.
  if (!slot->cuda || slot->parked != resume) return ncclSuccess;

  struct ibv_mr* tmp;
  size_t len = slot->pages * pageSize;
  if (!resume) {
    if (placeholder == nullptr) {
      WARN("NET/IB: failed to allocate the placeholder page for parked MRs");
      return ncclSystemError;
    }
    NCCLCHECK(wrap_ibv_reg_mr(&tmp, mr->pd, placeholder, pageSize, IBV_ACCESS_LOCAL_WRITE));
  } else if (slot->dmabuf) {
    int fd;
    CUCHECK(cuMemGetHandleForAddressRange((void*)&fd, (CUdeviceptr)slot->addr, len, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
                                          dmabufFlags));
    ncclResult_t ret = slot->dataDirect
                         ? wrap_mlx5dv_reg_dmabuf_mr(&tmp, mr->pd, 0, len, slot->addr, fd, slot->access,
                                                     MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT)
                         : wrap_ibv_reg_dmabuf_mr(&tmp, mr->pd, 0, len, slot->addr, fd, slot->access);
    (void)close(fd);
    NCCLCHECK(ret);
  } else if (slot->access & IBV_ACCESS_RELAXED_ORDERING) {
    NCCLCHECK(wrap_ibv_reg_mr_iova2(&tmp, mr->pd, (void*)slot->addr, len, slot->addr, slot->access));
  } else {
    NCCLCHECK(wrap_ibv_reg_mr(&tmp, mr->pd, (void*)slot->addr, len, slot->access));
  }
  std::swap(mr->handle, tmp->handle);
  std::swap(mr->lkey, tmp->lkey);
  std::swap(mr->rkey, tmp->rkey);
  // tmp now holds the previous registration.
  NCCLCHECK(wrap_ibv_dereg_mr(tmp));
  slot->parked = !resume;
  return ncclSuccess;
}

ncclResult_t ncclIbMrHandleMove(void* comm, void* mhandle, bool resume, int dmabufFlags) {
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    NCCLCHECK(ncclIbMrMove(ncclIbGetNetCommDevBase(base, i), mhandleWrapper->mrs[i], resume, dmabufFlags));
  }
  return ncclSuccess;
}

/* DMA-BUF support */
ncclResult_t ncclIbRegMrDmaBufInternal(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       uint64_t mrFlags, void** mhandle) {
  ncclResult_t ret = ncclSuccess;
  if (size == 0) {
    WARN("NET/IB: Invalid memory registration size %zu, expected > 0", size);
    return ncclInternalError;
  }
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)malloc(sizeof(struct ncclIbMrHandle));
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECKGOTO(ncclIbRegMrDmaBufInternal2(devComm, data, size, type, offset, fd, mrFlags, mhandleWrapper->mrs + i),
                  ret, fail);
  }
  *mhandle = (void*)mhandleWrapper;
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
  for (int i = 0; i < cache->population; i++) {
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

  struct ncclIbMrHandle* mhandleWrapper = (struct ncclIbMrHandle*)mhandle;
  struct ncclIbNetCommBase* base = (struct ncclIbNetCommBase*)comm;
  for (int i = 0; i < base->vProps.ndevs; i++) {
    // Each ncclIbNetCommDevBase is at different offset in send and recv netComms
    struct ncclIbNetCommDevBase* devComm = ncclIbGetNetCommDevBase(base, i);
    NCCLCHECK(ncclIbDeregMrInternal(devComm, mhandleWrapper->mrs[i]));
  }
  free(mhandleWrapper);
  return ncclSuccess;
}
