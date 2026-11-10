/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "argcheck.h" // Need some checks here since we access comm
#include "nccl.h"
#include "comm.h"
#include "net.h"
#include "register.h"
#include "transport.h"
#include "group.h"

NCCL_PARAM(LocalRegister, "LOCAL_REGISTER", 1);

// Blocks are the cuMem allocation granularity, which one array would cost on its own; without cuMem, cudaMalloc has
// no such granularity and a small block keeps a few registrations cheap.
#define NCCL_REG_IPC_ADDR_BLOCK_CUMEM CUDA_IPC_MIN
#define NCCL_REG_IPC_ADDR_BLOCK_NOCUMEM (16 * 1024)

static ncclResult_t regIpcAddrPoolGrow(struct ncclComm* comm, cudaStream_t stream) {
  struct ncclRegIpcAddrPool* pool = &comm->regIpcAddrPool;
  struct ncclRegIpcAddrBlock* block = NULL;
  ncclResult_t ret = ncclSuccess;
  NCCLCHECK(ncclCalloc(&block, 1));
  NCCLCHECKGOTO(ncclCudaCallocAsync(&block->base, (size_t)pool->arraysPerBlock * pool->arrayEntries, stream,
                                    comm->memManager),
                ret, fail);
  block->next = pool->blocks;
  pool->blocks = block;
  pool->nextArray = 0;
  return ncclSuccess;
fail:
  free(block);
  return ret;
}

ncclResult_t ncclRegIpcAddrsAlloc(struct ncclComm* comm, int arrayEntries, cudaStream_t stream, uintptr_t** addrs) {
  struct ncclRegIpcAddrPool* pool = &comm->regIpcAddrPool;
  // ipcIndexSize (localRanks, or nRanks across cliques) does not change during the life of a communicator
  if (arrayEntries <= 0 || (pool->arrayEntries != 0 && pool->arrayEntries != arrayEntries)) {
    WARN("rank %d IPC address array of %d entries requested from a pool of %d-entry arrays", comm->rank, arrayEntries,
         pool->arrayEntries);
    return ncclInternalError;
  }
  if (pool->arrayEntries == 0) {
    size_t blockBytes = ncclCuMemEnable() ? NCCL_REG_IPC_ADDR_BLOCK_CUMEM : NCCL_REG_IPC_ADDR_BLOCK_NOCUMEM;
    size_t arraysPerBlock = blockBytes / ((size_t)arrayEntries * sizeof(uintptr_t));
    pool->arrayEntries = arrayEntries;
    pool->arraysPerBlock = arraysPerBlock > 0 ? (int)arraysPerBlock : 1;
  }
  if (pool->nFree > 0) {
    *addrs = pool->freeArrays[--pool->nFree];
    return ncclSuccess;
  }
  if (pool->blocks == NULL || pool->nextArray == pool->arraysPerBlock) NCCLCHECK(regIpcAddrPoolGrow(comm, stream));
  *addrs = pool->blocks->base + (size_t)pool->nextArray * pool->arrayEntries;
  pool->nextArray++;
  return ncclSuccess;
}

ncclResult_t ncclRegIpcAddrsFree(struct ncclComm* comm, uintptr_t* addrs, bool clear) {
  struct ncclRegIpcAddrPool* pool = &comm->regIpcAddrPool;
  if (clear) {
    // Zero the array on the stream that carries every copy into it, so that a kernel which still reads it faults
    // (offset from address 0) instead of reading the addresses of the next record that takes the array.
    cudaStream_t stream;
    cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
    cudaError_t res;
    NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                      /*concurrent=*/false, &stream));
    CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    res = cudaMemsetAsync(addrs, 0, (size_t)pool->arrayEntries * sizeof(uintptr_t), stream);
    if (res == cudaSuccess && pool->clearEvent == NULL) {
      res = cudaEventCreateWithFlags(&pool->clearEvent, cudaEventDisableTiming);
    }
    if (res == cudaSuccess) res = cudaEventRecord(pool->clearEvent, stream);
    CUDACHECK(cudaThreadExchangeStreamCaptureMode(&mode));
    CUDACHECK(res);
    NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->hostStream,
                                      /*concurrent=*/false));
  }
  if (pool->nFree == pool->freeCap) {
    int freeCap = pool->freeCap ? 2 * pool->freeCap : 64;
    NCCLCHECK(ncclRealloc(&pool->freeArrays, pool->freeCap, freeCap));
    pool->freeCap = freeCap;
  }
  pool->freeArrays[pool->nFree++] = addrs;
  return ncclSuccess;
}

static ncclResult_t regIpcAddrPoolDestroy(struct ncclComm* comm) {
  struct ncclRegIpcAddrPool* pool = &comm->regIpcAddrPool;
  if (pool->clearEvent) {
    // the zeroing of released arrays is asynchronous; the blocks must not be unmapped under it
    CUDACHECK(cudaEventSynchronize(pool->clearEvent));
    CUDACHECK(cudaEventDestroy(pool->clearEvent));
  }
  while (pool->blocks) {
    struct ncclRegIpcAddrBlock* block = pool->blocks;
    pool->blocks = block->next;
    NCCLCHECK(ncclCudaFree(block->base, comm->memManager));
    free(block);
  }
  free(pool->freeArrays);
  memset(pool, 0, sizeof(*pool));
  return ncclSuccess;
}

ncclResult_t ncclRegLocalIsValid(struct ncclReg* reg, bool* isValid) {
  if (reg && isValid) {
    if (reg->localRefs) *isValid = true;
    else *isValid = false;
  }
  return ncclSuccess;
}

ncclResult_t ncclRegister(struct ncclComm* comm, void* data, size_t size, bool isGraph, void** handle) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));

  *handle = NULL;

  struct ncclRegCache* cache = &comm->regCache;
  uintptr_t pageSize = cache->pageSize;
  uintptr_t begAddr = (uintptr_t)data & -pageSize;
  uintptr_t endAddr = ((uintptr_t)data + size + pageSize - 1) & -pageSize;

  if (comm->checkMode != ncclCheckModeDefault) NCCLCHECK(CudaPtrCheck(data, comm, "buff", "ncclCommRegister"));

  bool hasSysmemSegment = false;
  if (ncclCuMemEnable()) {
    CUdeviceptr base;
    size_t baseSize;
    int numSegments;
    int legacyIpcCap;
    CUCHECK(cuMemGetAddressRange(&base, &baseSize, (CUdeviceptr)data));
    CUmemorytype memType;
    CUCHECK(cuPointerGetAttribute(&memType, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, (CUdeviceptr)data));
    if (memType == CU_MEMORYTYPE_HOST) {
      hasSysmemSegment = true;
    } else {
      // Check for a Sysmem segment is only valid with cuMem based allocators, so a IS_LEGACY_CUDA_IPC check is
      // required to ensure that we're calling ncclCuMemGetAddressRange only when necessary.
      CUCHECK(cuPointerGetAttribute((void*)&legacyIpcCap, CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE,
                                    (CUdeviceptr)base));
      if (!legacyIpcCap) {
        NCCLCHECK(ncclCuMemGetAddressRange((CUdeviceptr)data, size, (CUdeviceptr*)&base, &baseSize, &numSegments,
                                           &hasSysmemSegment));
      }
    }
  }
  if (hasSysmemSegment) {
    INFO(NCCL_REG, "Skipping registration for buffer %p size %zi since it contains segments backed by CPU memory", data,
         size);
    return ncclSuccess;
  } else {
    INFO(NCCL_REG, "register comm %p buffer %p size %zi", comm, data, size);
  }

  for (int slot = 0; /*true*/; slot++) {
    if ((slot == cache->population) || (begAddr < cache->slots[slot]->begAddr)) {
      if (cache->population == cache->capacity) {
        // must grow cache
        cache->capacity = cache->capacity < 32 ? 32 : 2 * cache->capacity;
        NCCLCHECK(ncclRealloc(&cache->slots, cache->population, cache->capacity));
      }
      memmove(cache->slots + slot + 1, cache->slots + slot, (cache->population - slot) * sizeof(struct ncclReg*));
      NCCLCHECK(ncclCalloc(cache->slots + slot, 1));
      struct ncclReg* regSlot = cache->slots[slot];
      regSlot->begAddr = begAddr;
      regSlot->endAddr = endAddr;
      if (isGraph) regSlot->graphRefs = 1;
      else regSlot->localRefs = 1;
      cache->population += 1;
      *handle = regSlot;
      goto exit;
    } else if ((cache->slots[slot]->begAddr <= begAddr) && (cache->slots[slot]->endAddr >= endAddr)) {
      if (isGraph) cache->slots[slot]->graphRefs++;
      else cache->slots[slot]->localRefs++;
      *handle = cache->slots[slot];
      goto exit;
    }
  }

exit:
  return ncclSuccess;
}

static ncclResult_t regCleanup(struct ncclComm* comm, struct ncclReg* reg, bool clearIpcAddrs) {
  if (reg->state & NET_REG_COMPLETE) {
    struct ncclRegNetHandles* netHandle = reg->netHandleHead;
    struct ncclRegNetHandles* netHandlePrev;
    while (netHandle) {
      if (ncclNetDeregBuffer(comm, netHandle->proxyConn, netHandle->handle) != ncclSuccess) {
        ATTN("rank %d deregister NET buffer handle %p proxy rank %d failed", comm->rank, netHandle->handle,
             netHandle->proxyConn->rank);
      }
      netHandlePrev = netHandle;
      netHandle = netHandle->next;
      free(netHandlePrev);
    }
  }
  if (reg->state & NVLS_REG_COMPLETE) {
    NCCLCHECK(ncclNvlsUbDeregister(comm, reg));
  }
  if (reg->state & COLLNET_REG_COMPLETE) {
    if (ncclCollnetDeregBuffer(comm, reg->collnetProxyconn, reg->collnetHandle) != ncclSuccess) {
      ATTN("rank %d deregister COLLNET buffer handle %p proxy rank %d failed", comm->rank, reg->collnetHandle,
           reg->collnetProxyconn->rank);
    }
  }
  if (reg->state & IPC_REG_COMPLETE) {
    if (reg->ipcInfos) {
      for (int i = 0; i < reg->ipcInfosSize; ++i)
        if (reg->ipcInfos[i]) {
          if (ncclIpcDeregBuffer(comm, reg->ipcInfos[i]) != ncclSuccess) {
            ATTN("rank %d deregister IPC buffer %p peerRank %d failed", comm->rank, reg->ipcInfos[i]->baseAddr,
                 reg->ipcInfos[i]->peerRank);
          }
          free(reg->ipcInfos[i]);
        }
      free(reg->ipcInfos);
    }
    if (reg->regIpcAddrs.hostPeerRmtAddrs) free(reg->regIpcAddrs.hostPeerRmtAddrs);
    if (reg->regIpcAddrs.devPeerRmtAddrs) {
      NCCLCHECK(ncclRegIpcAddrsFree(comm, reg->regIpcAddrs.devPeerRmtAddrs, clearIpcAddrs));
    }
  }
  return ncclSuccess;
}

ncclResult_t ncclRegCleanup(struct ncclComm* comm) {
  struct ncclRegCache* cache = &comm->regCache;
  ncclResult_t ret = ncclSuccess, res;
  for (int i = 0; i < cache->population; i++) {
    struct ncclReg* reg = cache->slots[i];
    INFO(NCCL_DESTROY, "Cleanup buffer %p pages %lx", (void*)reg->begAddr,
         (reg->endAddr - reg->begAddr) / cache->pageSize);
    // keep going after a failure so that every record and the address blocks are released
    res = regCleanup(comm, reg, /*clearIpcAddrs=*/false);
    if (res != ncclSuccess && ret == ncclSuccess) ret = res;
    free(reg);
  }
  free(cache->slots);
  res = regIpcAddrPoolDestroy(comm);
  if (res != ncclSuccess && ret == ncclSuccess) ret = res;
  return ret;
}

NCCL_API(ncclResult_t, ncclCommRegister, const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NCCLCHECK(PtrCheck(handle, "ncclCommRegister", "handle"));

  if (!ncclParamLocalRegister() || ncclP2pUsesMemcpy()) {
    *handle = NULL;
    INFO(NCCL_REG, "Skipping registration for buffer %p size %zi (LocalRegister=%ld, P2pUsesMemcpy=%d)", buff, size,
         ncclParamLocalRegister(), ncclP2pUsesMemcpy());
  } else {
    NCCLCHECK(ncclRegister(comm, buff, size, false, handle));
  }
  return ncclSuccess;
}

ncclResult_t ncclCommGraphRegister(const ncclComm_t comm, void* buff, size_t size, void** handle) {
  NCCLCHECK(PtrCheck(handle, "ncclCommGraphRegister", "handle"));

  if (ncclP2pUsesMemcpy()) {
    *handle = NULL;
    INFO(NCCL_REG, "Skipping graph registration for buffer %p size %zi (P2pUsesMemcpy=%d)", buff, size,
         ncclP2pUsesMemcpy());
  } else {
    NCCLCHECK(ncclRegister(comm, buff, size, true, handle));
  }
  return ncclSuccess;
}

static ncclResult_t commDeregister(struct ncclComm* comm, bool isGraph, struct ncclReg* reg) {
  NCCLCHECK(CommCheck(comm, "ncclCommRegister", "comm"));
  struct ncclRegCache* cache = &comm->regCache;
  int slot;
  int saveDev;
  if (reg == NULL) goto exit;
  CUDACHECK(cudaGetDevice(&saveDev));
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  for (slot = 0; slot < cache->population && cache->slots[slot] != reg; slot++);
  if (slot == cache->population) {
    WARN("Deregister: Could not find handle");
    return ncclInvalidUsage;
  }
  if (isGraph) --reg->graphRefs;
  else --reg->localRefs;
  if (reg->localRefs || reg->graphRefs) return ncclSuccess;
  NCCLCHECK(regCleanup(comm, reg, /*clearIpcAddrs=*/true));
  free(reg);
  memmove(cache->slots + slot, cache->slots + slot + 1, (cache->population - slot - 1) * sizeof(struct ncclReg*));
  cache->population -= 1;
  CUDACHECK(cudaSetDevice(saveDev));
exit:
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDeregister, const ncclComm_t comm, void* handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle) {
  NCCLCHECK(commDeregister(comm, false, (struct ncclReg*)handle));
  return ncclSuccess;
}

ncclResult_t ncclCommGraphDeregister(const ncclComm_t comm, struct ncclReg* handle) {
  NCCLCHECK(commDeregister(comm, true, handle));
  return ncclSuccess;
}
