/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "gdrwrap.h"
#include "comm.h"
#include "bootstrap.h"
#include "compiler.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "dev_runtime.h"
#include "os.h"


extern int64_t ncclParamDmaBufEnable();
extern int64_t ncclParamIbDataDirect();

NCCL_PARAM(RmaProxyDumpSignal, "RMA_PROXY_DUMP_SIGNAL", -1);
NCCL_PARAM(RmaProxyQueueSize, "RMA_PROXY_QUEUE_SIZE", -1);

#include <signal.h>
static ncclRmaProxyState* ncclLastRmaProxyState;

ncclResult_t dumpRmaProxyState(struct ncclRmaProxyState* rmaProxyState);
void ncclDumpRmaProxyState(int signal);

// ---- Internal helpers ----

static ncclResult_t getDmaBufFd(void *addr, size_t length, int *fd,
                                bool forceNonDataDirect = false) {
  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

#if CUDA_VERSION >= 11070
  static size_t hostPageSize = ncclOsGetPageSize();
  size_t alignedSize = length;
  ALIGN_SIZE(alignedSize, hostPageSize);

#if CUDA_VERSION >= 12080
  if (ncclParamIbDataDirect() && !forceNonDataDirect) {
    CUresult status = pfn_cuMemGetHandleForAddressRange(
      (void *)fd, (CUdeviceptr)addr, alignedSize, CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD,
      CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE);
    if (status == CUDA_SUCCESS) return ncclSuccess;
  }
#endif
  CUresult status = pfn_cuMemGetHandleForAddressRange((void *)fd, (CUdeviceptr)addr, alignedSize,
                                                      CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 0);
  if (status == CUDA_SUCCESS) return ncclSuccess;
#endif

  return ncclInvalidUsage;
}

// Check if the RMA plugin supports DMA-BUF, if so we can try to get the DMA-BUF handle from CUDA,
// if that fails we fallback to non-DMA-BUF
static ncclResult_t ncclRmaProxyRegMrSym(ncclRma_t *rmaComm, void *rmaCollComm, ncclNetProperties_t props, void *addr,
                                         size_t size, int type, int mr_flags, void **mhandle) {
  if (type == NCCL_PTR_HOST) {
    NCCLCHECK(rmaComm->regMrSym(rmaCollComm, addr, size, type, mr_flags, mhandle));
  } else if (type == NCCL_PTR_CUDA) {
    ncclResult_t dmabufResult = ncclInvalidUsage;
    if (ncclParamDmaBufEnable() && (props.ptrSupport & NCCL_PTR_DMABUF)) {
      ncclResult_t registrationResult = ncclSuccess;
      int dmabufFd = -1;
      dmabufResult = getDmaBufFd(addr, size, &dmabufFd);
      if (dmabufResult == ncclSuccess) {
        registrationResult = rmaComm->regMrSymDmaBuf(rmaCollComm, addr, size, type, 0, dmabufFd,
                                                     mr_flags, mhandle);
        close(dmabufFd);
      }
      if (registrationResult != ncclSuccess) {
        dmabufFd = -1;
        dmabufResult = getDmaBufFd(addr, size, &dmabufFd, true);
        if (dmabufResult == ncclSuccess) {
          NCCLCHECK(rmaComm->regMrSymDmaBuf(rmaCollComm, addr, size, type, 0, dmabufFd,
                                            mr_flags, mhandle));
          close(dmabufFd);
        }
      }
    }
    // Fallback to non-DMA-BUF if the DMA-BUF handle is not supported
    if (dmabufResult != ncclSuccess) {
      NCCLCHECK(rmaComm->regMrSym(rmaCollComm, addr, size, type, mr_flags, mhandle));
    }
  } else {
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}
static uint64_t isPowerOfTwo(uint64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

// ---- Context lifecycle ----

static ncclResult_t ncclRmaProxyCtxAlloc(struct ncclComm* comm, ncclRma_t* rmaComm, struct ncclRmaProxyCtx* rmaProxyCtx) {
  // The clean up in case of failure will be done by the ncclRmaProxyDestroyContext function invoked by the caller.
  // Allocate the signals on the GPU and then register the memory region with the RMA plugin.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and signals.
  size_t signalsBufSize = (comm->nRanks + 1) * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&rmaProxyCtx->signalsDev, &rmaProxyCtx->signalsCumemhandle,
                           CU_MEM_HANDLE_TYPE_NONE, signalsBufSize, comm->memManager));
  CUDACHECK(cudaMemset(rmaProxyCtx->signalsDev, 0, signalsBufSize));
  NCCLCHECK(ncclRmaProxyRegMrSym(rmaComm, rmaProxyCtx->rmaCollComm, rmaProxyCtx->props, rmaProxyCtx->signalsDev, signalsBufSize,
                                 NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO | NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET,
                                 &rmaProxyCtx->signalsMhandle));
  // Allocate the host buffer to track the expected values of the signals
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->signalsHost, signalsBufSize));
  // Allocate the sequence numbers for the per-rank network function descriptors
  // These are allocated as CPU-accessible memory (either GDR or host memory)
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->opSeqs, &rmaProxyCtx->opSeqsDev,
                                  comm->nRanks, 0, &rmaProxyCtx->opSeqsGdrHandle, comm->memManager));
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->readySeqs, &rmaProxyCtx->readySeqsDev,
                                  comm->nRanks, 0, &rmaProxyCtx->readySeqsGdrHandle, comm->memManager));
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->doneSeqs, &rmaProxyCtx->doneSeqsDev,
                                  comm->nRanks, 0, &rmaProxyCtx->doneSeqsGdrHandle, comm->memManager));
  // Sanitize and set up the lock-free circular buffer queue size
  uint64_t queueSize = ncclParamRmaProxyQueueSize();
  uint32_t maxRequests = NCCL_NET_MAX_REQUESTS * rmaProxyCtx->props.maxRecvs;
  rmaProxyCtx->maxInflightRequests = maxRequests;
  if (queueSize == -1) {
    queueSize = maxRequests;
  }
  if (queueSize > maxRequests) {
    INFO(NCCL_NET,
         "NCCL_RMA_PROXY_QUEUE_SIZE is greater than the maximum outstanding requests (%d), using the default/maximum value instead",
         maxRequests);
    queueSize = maxRequests;
  }
  if (queueSize < 1) {
    INFO(NCCL_NET,
         "NCCL_RMA_PROXY_QUEUE_SIZE is less than 1, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  if (!isPowerOfTwo(queueSize)) {
    INFO(NCCL_NET,
         "NCCL_RMA_PROXY_QUEUE_SIZE is not a power of two, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  rmaProxyCtx->queueSize = queueSize;

  // Allocate lock-free circular buffer for pending Descs
  size_t circularBufLength = comm->nRanks * queueSize;
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->circularBuffers, circularBufLength));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->pis, comm->nRanks));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->cis, comm->nRanks));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->inflightRequests, comm->nRanks));

  // Allocate per-peer InProgress queues (kept as linked list, single consumer)
  rmaProxyCtx->inProgressQueues = ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(&comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->inProgressQueues[i]);
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyCtxAllocGraph(struct ncclComm* comm, ncclRma_t* rmaComm, struct ncclRmaProxyCtx* rmaProxyCtx) {
  // The clean up in case of failure will be done by the ncclRmaProxyDestroyContext function invoked by the caller.
  size_t signalsBufSize = (comm->nRanks + 1) * sizeof(uint64_t);
  // Allocate the CPU-accessible signal for graph capture and then register the memory region with the RMA plugin.
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->cpuAccessSignals, &rmaProxyCtx->cpuAccessSignalsDev,
                                  comm->nRanks + 1, 0, &rmaProxyCtx->cpuAccessSignalsGdrHandle, comm->memManager));
  int cpuAccessSignalsType = (rmaProxyCtx->cpuAccessSignalsGdrHandle != NULL) ? NCCL_PTR_CUDA : NCCL_PTR_HOST;
  NCCLCHECK(ncclRmaProxyRegMrSym(rmaComm, rmaProxyCtx->rmaCollComm, rmaProxyCtx->props, rmaProxyCtx->cpuAccessSignalsDev, signalsBufSize,
                                 cpuAccessSignalsType, NCCL_NET_MR_FLAG_FORCE_SO,
                                 &rmaProxyCtx->cpuAccessSignalsMhandle));
  // Allocate the host buffer to track the expected values of the signals
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->cpuAccessSignalsHost, signalsBufSize));

  // Allocate the flush buffer on the GPU and then register the memory region with the RMA plugin.
  size_t flushBufSize = comm->nRanks * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&rmaProxyCtx->flushBufDev, &rmaProxyCtx->flushBufCumemhandle,
                            CU_MEM_HANDLE_TYPE_NONE, flushBufSize, comm->memManager));
  CUDACHECK(cudaMemset(rmaProxyCtx->flushBufDev, 0, flushBufSize));
  NCCLCHECK(ncclRmaProxyRegMrSym(rmaComm, rmaProxyCtx->rmaCollComm, rmaProxyCtx->props, rmaProxyCtx->flushBufDev, flushBufSize,
                                  NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                  &rmaProxyCtx->flushBufMhandle));
  // Allocate and initialize persistent descriptor queue
  rmaProxyCtx->persistentQueues = ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(&comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->persistentQueues[i]);
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyCreateContext(struct ncclComm *comm, void *collComm, ncclNetProperties_t props,
                                       void **outRmaProxyCtx) {
  ncclResult_t ret = ncclSuccess;
  // Get the RMA plugin interface
  ncclRma_t *rmaComm = (ncclRma_t *)comm->rmaState.rmaProxyState.ncclRma;

  ncclRmaConfig_t config = { 1, comm->config.trafficClass, 1 };

  // Allocate the RMA proxy context
  struct ncclRmaProxyCtx *rmaProxyCtx = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&rmaProxyCtx, 1), ret, fail);

  rmaProxyCtx->comm = comm;
  rmaProxyCtx->rmaCollComm = collComm;
  rmaProxyCtx->props = props;
  NCCLCHECK(rmaComm->createContext(collComm, &config, &rmaProxyCtx->rmaCtx));

  NCCLCHECKGOTO(ncclRmaProxyCtxAlloc(comm, rmaComm, rmaProxyCtx), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyCtxAllocGraph(comm, rmaComm, rmaProxyCtx), ret, fail);

  *outRmaProxyCtx = rmaProxyCtx;

  return ncclSuccess;
fail:
  ncclRmaProxyDestroyContext(rmaComm, rmaProxyCtx);
  return ret;
}

ncclResult_t ncclRmaProxyDestroyContext(ncclRma_t* rmaComm, void* rmaProxyCtx){
  if (!rmaProxyCtx) return ncclSuccess;
  struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

  NCCLCHECK(rmaComm->destroyContext(ctx->rmaCtx));

  // Free descriptors remaining in circular buffers
  if (ctx->circularBuffers) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[i], std::memory_order_relaxed);
      uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[i], std::memory_order_relaxed);
      // Free any remaining pending descriptors
      for (uint32_t j = ci; j < pi; j++) {
        uint32_t idx = j & (ctx->queueSize - 1);
        struct ncclRmaProxyDesc *desc = ctx->circularBuffers[i * ctx->queueSize + idx];
        if (desc != NULL) {
          NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &desc));
        }
      }
    }
    free(ctx->circularBuffers);
  }

  // Free PI/CI arrays
  free(ctx->pis);
  free(ctx->cis);
  free(ctx->inflightRequests);

  // Free InProgress queues and their Descs
  if (ctx->inProgressQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      struct ncclRmaProxyDesc *desc = ncclIntruQueueHead(&ctx->inProgressQueues[i]);
      while (desc != NULL) {
        struct ncclRmaProxyDesc *nextDesc = desc->next;
        ncclIntruQueueDequeue(&ctx->inProgressQueues[i]);
        NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &desc));
        desc = nextDesc;
      }
    }
  }

  // Free persistent descriptor queue and their Descs
  if (ctx->persistentQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      struct ncclRmaProxyDesc *desc = ncclIntruQueueHead(&ctx->persistentQueues[i]);
      while (desc != NULL) {
        struct ncclRmaProxyDesc *nextDesc = desc->next;
        ncclIntruQueueDequeue(&ctx->persistentQueues[i]);
        NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &desc));
        desc = nextDesc;
      }
    }
  }

  // Free counters (using GDR-aware deallocation)
  if (ctx->opSeqs) NCCLCHECK(freeMemCPUAccessible(ctx->opSeqs, ctx->opSeqsGdrHandle, ctx->comm->memManager));
  if (ctx->readySeqs) NCCLCHECK(freeMemCPUAccessible(ctx->readySeqs, ctx->readySeqsGdrHandle, ctx->comm->memManager));
  if (ctx->doneSeqs) NCCLCHECK(freeMemCPUAccessible(ctx->doneSeqs, ctx->doneSeqsGdrHandle, ctx->comm->memManager));

  // Free signals
  if (rmaComm && ctx->rmaCollComm && ctx->signalsMhandle) {
    NCCLCHECK(rmaComm->deregMrSym(ctx->rmaCollComm, ctx->signalsMhandle));
  }
  if (ctx->signalsDev) NCCLCHECK(ncclCudaFree(ctx->signalsDev, ctx->comm->memManager));

  // Free flush buffer
  if (rmaComm && ctx->rmaCollComm && ctx->flushBufMhandle)
    rmaComm->deregMrSym(ctx->rmaCollComm, ctx->flushBufMhandle);
  if (ctx->flushBufDev) ncclCudaFree(ctx->flushBufDev, ctx->comm->memManager);

  // Free CPU-accessible signals
  if (rmaComm && ctx->rmaCollComm && ctx->cpuAccessSignalsMhandle) {
    rmaComm->deregMrSym(ctx->rmaCollComm, ctx->cpuAccessSignalsMhandle);
  }
  if (ctx->cpuAccessSignals) {
    freeMemCPUAccessible(ctx->cpuAccessSignals, ctx->cpuAccessSignalsGdrHandle, ctx->comm->memManager);
  }

  // Free host signals buffers
  free(ctx->signalsHost);
  free(ctx->cpuAccessSignalsHost);

  free(ctx);

  return ncclSuccess;
}

// ---- Memory registration ----

ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
    void* rmaHostWins[NCCL_RMA_MAX_CONNECTIONS]){
      struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
      for (int n = 0; n < rmaProxyState->rmaCommCount; n++) {
          NCCLCHECK(ncclRmaProxyRegMrSym(rmaProxyState->ncclRma, rmaProxyState->rmaComms[n], rmaProxyState->props[n], address, size,
                                         NCCL_PTR_CUDA, 0, &rmaHostWins[n]));
        if (rmaHostWins[n] == NULL) {
          WARN("rank %d - RMA Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
          return ncclSystemError;
        }
      }
      return ncclSuccess;
}

ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_RMA_MAX_CONNECTIONS]){
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  for (int n = 0; n < rmaProxyState->rmaCommCount; n++) {
    NCCLCHECK(rmaProxyState->ncclRma->deregMrSym(rmaProxyState->rmaComms[n], rmaHostWins[n]));
  }
  return ncclSuccess;
}

// ---- Progress thread and global lifecycle ----

void* ncclRmaProxyProgressThread(struct ncclRmaProxyState* rmaProxyState_) {
  struct ncclRmaProxyState* rmaProxyState = (struct ncclRmaProxyState*)rmaProxyState_;
  const int sig = ncclParamRmaProxyDumpSignal();
  if (sig != -1) signal(sig, ncclDumpRmaProxyState);
  ncclLastRmaProxyState = rmaProxyState;
  while (1) {
    std::unique_lock<std::mutex> lock(rmaProxyState->mutex);
    if (rmaProxyState->rmaProgress == 1) {
      lock.unlock();
      for (int n=0; n<rmaProxyState->rmaProxyCtxCount; n++) {
        ncclResult_t ret = ncclRmaProxyProgress(rmaProxyState->ncclRma, rmaProxyState->rmaProxyCtxs[n]);
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE_32(&rmaProxyState->asyncResult, ret, std::memory_order_release);
          INFO_LOC(NCCL_ALL, "-> %d [RMA Proxy Progress Thread]", ret);
          rmaProxyState->rmaProgress = -2;
          return NULL;
        }
      }
      std::this_thread::yield();
    } else if (rmaProxyState->rmaProgress == 2) {
      // Pause requested for reclaim: acknowledge and sleep.
      // Main thread will do the actual freeing while we're paused.
      rmaProxyState->rmaProgress = 0;
      rmaProxyState->cond.notify_one();
      rmaProxyState->cond.wait(lock);
    } else if (rmaProxyState->rmaProgress == -1) {
      return NULL;
    } else if (rmaProxyState->rmaProgress == 0) {
      rmaProxyState->cond.wait(lock);
    } else {
      INFO_LOC(NCCL_ALL, "[RMA Proxy Progress Thread] state unknown %d", rmaProxyState->rmaProgress);
      rmaProxyState->rmaProgress = -2;
      return NULL;
    }
  }
}

ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaProxyState *rmaProxyState = &comm->rmaState.rmaProxyState;
  rmaProxyState->comm = comm;
  if (rmaProxyState->ncclRma == NULL) {
    WARN("RMA not supported.");
    return ncclInvalidUsage;
  }
  if (rmaProxyState->connected) return ncclSuccess;

  rmaProxyState->rmaInstance = comm->rmaContext;

  int ndev = 0;
  NCCLCHECK(rmaProxyState->ncclRma->devices(&ndev));
  if (ndev <= 0) {
    WARN("No RMA-capable devices found.");
    return ncclInternalError;
  }

  int rmaCommCount;
  int localRmaDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalRmaDevs(comm, localRmaDevs, &rmaProxyState->rmaCommCount));
  rmaCommCount = std::min<int>(rmaProxyState->rmaCommCount, NCCL_RMA_MAX_CONNECTIONS);
  rmaCommCount = std::min<int>(rmaCommCount, ndev);

  int* allCommCounts = NULL;
  void** handles = NULL;
  char* allHandles = NULL;

  // Get the min local net count from all ranks
  NCCLCHECK(ncclCalloc(&allCommCounts, comm->nRanks));
  allCommCounts[comm->rank] = rmaCommCount;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allCommCounts, sizeof(int)), ret, fail);
  for (int i = 0; i < comm->nRanks; i++) {
    rmaCommCount = std::min<int>(rmaCommCount, allCommCounts[i]);
  }
  free(allCommCounts);
  allCommCounts = NULL;

  if (rmaCommCount == 0) {
    WARN("Rma connect : min local net count is zero");
    ret = ncclSystemError;
    goto fail;
  }
  rmaProxyState->rmaCommCount = rmaCommCount;

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) handles[r] = allHandles + r * NCCL_NET_HANDLE_MAXSIZE;

  for (int n = 0; n < rmaCommCount; n++) {
    void* listenComm;
    NCCLCHECKGOTO(
      rmaProxyState->ncclRma->listen(rmaProxyState->rmaInstance, localRmaDevs[n],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);
    NCCLCHECKGOTO(
      rmaProxyState->ncclRma->connect(comm->netContext, handles, comm->nRanks, comm->rank,
                                      listenComm, rmaProxyState->rmaComms + n),
      ret, fail);
    NCCLCHECKGOTO(rmaProxyState->ncclRma->getProperties(localRmaDevs[n], &rmaProxyState->props[n]), ret, fail);
    NCCLCHECKGOTO(rmaProxyState->ncclRma->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;

  // Create virtual RMA proxy contexts
  rmaProxyState->rmaProxyCtxCount = comm->config.numRmaCtx;
  NCCLCHECK(ncclCalloc(&rmaProxyState->rmaProxyCtxs, rmaProxyState->rmaProxyCtxCount));
  for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
    // Round-robin mapping to physical RMA communicator contexts
    int rmaCommIdx = n % rmaProxyState->rmaCommCount;
    NCCLCHECKGOTO(ncclRmaProxyCreateContext(comm, rmaProxyState->rmaComms[rmaCommIdx], rmaProxyState->props[rmaCommIdx],
                                              &rmaProxyState->rmaProxyCtxs[n]),
                  ret, fail);
  }

  // Start / wake up the progress thread.
  rmaProxyState->rmaProgress = 1;
  rmaProxyState->thread = std::thread(ncclRmaProxyProgressThread, rmaProxyState);
  ncclSetThreadName(rmaProxyState->thread, "NCCL RMA Proxy Progress%2d", comm->cudaDev);

  INFO(NCCL_INIT, "Rank %d ncclRmaProxyConnectOnce: rmaCommCount %d rmaProxyCtxCount:%d", comm->rank, rmaCommCount, rmaProxyState->rmaProxyCtxCount);

exit:
  if (ret == ncclSuccess) rmaProxyState->connected = true;
  return ret;
fail:
  free(allCommCounts);
  free(allHandles);
  free(handles);
  goto exit;
}

ncclResult_t ncclRmaProxyFinalize(struct ncclComm* comm) {
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  if (!rmaProxyState->connected) return ncclSuccess;

  {
    std::lock_guard<std::mutex> lock(rmaProxyState->mutex);
    rmaProxyState->rmaProgress = -1;
    rmaProxyState->cond.notify_one();
  }
  rmaProxyState->thread.join();

  // Destroy all virtual RMA proxy contexts
  if (rmaProxyState->rmaProxyCtxs) {
    for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
      if (rmaProxyState->rmaProxyCtxs[n] != NULL) {
        NCCLCHECK(ncclRmaProxyDestroyContext(rmaProxyState->ncclRma, rmaProxyState->rmaProxyCtxs[n]));
        rmaProxyState->rmaProxyCtxs[n] = NULL;
      }
    }
    // Free the dynamically allocated context array
    free(rmaProxyState->rmaProxyCtxs);
    rmaProxyState->rmaProxyCtxs = NULL;
  }

  // Close all physical RMA communicators
  for (int n = 0; n < rmaProxyState->rmaCommCount; n++) {
    if (rmaProxyState->rmaComms[n] != NULL) {
      NCCLCHECK(rmaProxyState->ncclRma->closeColl(rmaProxyState->rmaComms[n]));
      rmaProxyState->rmaComms[n] = NULL;
    }
  }

  memset((void*)rmaProxyState, 0, sizeof(*rmaProxyState));
  return ncclSuccess;
}

// ---- Debug ----

ncclResult_t dumpRmaProxyState(struct ncclRmaProxyState* rmaProxyState) {
  ncclLastRmaProxyState = rmaProxyState;
  if (rmaProxyState->comm) {
    printf("Rank %d RMA Proxy State:\n", rmaProxyState->comm->rank);
    printf("  rmaProgress: %d\n", rmaProxyState->rmaProgress);
    printf("  rmaCommCount: %d\n", rmaProxyState->rmaCommCount);
    printf("  rmaProxyCtxCount:%d\n", rmaProxyState->rmaProxyCtxCount);
    printf("  connected: %d\n", rmaProxyState->connected);

    // dump per-context information
    for (int i = 0; i < rmaProxyState->rmaProxyCtxCount; i++) {
      struct ncclRmaProxyCtx* ctx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[i];
      printf("  rmaCtx[%d]: %p\n", i, ctx);
      printf("    rmaCollComms: %p\n", ctx->rmaCollComm);
      if (ctx && ctx->comm) {
        printf("    nRanks: %d, myRank: %d\n", ctx->comm->nRanks, ctx->comm->rank);
        printf("    queueSize: %zu\n", ctx->queueSize);
        // dump per-peer information
        for (int peer = 0; peer < ctx->comm->nRanks; peer++) {
          uint64_t readySeq = COMPILER_ATOMIC_LOAD(&ctx->readySeqs[peer], std::memory_order_acquire);
          uint64_t doneSeq = COMPILER_ATOMIC_LOAD(&ctx->doneSeqs[peer], std::memory_order_acquire);
          uint64_t opSeq = COMPILER_ATOMIC_LOAD(&ctx->opSeqs[peer], std::memory_order_acquire);
          uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_acquire);
          uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_acquire);
          printf("      Peer %d: readySeq: %lu, doneSeq: %lu, opSeq: %lu, PI: %u, CI: %u\n",
                 peer, readySeq, doneSeq, opSeq, pi, ci);

          // Count and print pending Descs from circular buffer
          int pendingCount = pi - ci;
          printf("        Pending Descs: %d\n", pendingCount);
          for (uint32_t j = ci; j < pi; j++) {
            uint32_t idx = j & (ctx->queueSize - 1);
            struct ncclRmaProxyDesc* desc = ctx->circularBuffers[peer * ctx->queueSize + idx];
            if (desc != NULL) {
              printf("          Desc: seq=%lu targetRank=%d size=%zu\n",
                    desc->opSeq, desc->putSignal.targetRank, desc->putSignal.size);
            }
          }

          // Count in-progress Descs
          int inProgressCount = 0;
          struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
          while (desc != NULL) {
            inProgressCount++;
            desc = desc->next;
          }
          printf("        In-progress Descs: %d\n", inProgressCount);
          // print all in-progress Descs
          desc = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
          while (desc != NULL) {
            printf("          Desc: seq=%lu targetRank=%d size=%zu\n",
                  desc->opSeq, desc->putSignal.targetRank, desc->putSignal.size);
            desc = desc->next;
          }
        }
      } else {
        printf("    rmaCtx[%d]: NULL\n", i);
      }
    }
  }
  return ncclSuccess;
}

void ncclDumpRmaProxyState(int signal) {
  dumpRmaProxyState(ncclLastRmaProxyState);
}
