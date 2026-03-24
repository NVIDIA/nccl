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
#if !defined(NCCL_OS_WINDOWS)
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
#else
#define NCCL_GIN_PROXY_VERSION 100  /* stub value; GIN proxy not used at runtime on Windows */
#endif
#include "os.h"


extern int64_t ncclParamDmaBufEnable();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamGinEnable();
extern int64_t ncclParamGinType();

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

// Check if the GIN plugin supports DMA-BUF, if so we can try to get the DMA-BUF handle from CUDA,
// if that fails we fallback to non-DMA-BUF
static ncclResult_t ncclRmaProxyRegMrSym(ncclGin_t *ginComm, void *ginCollComm, ncclNetProperties_t props, void *addr,
                                         size_t size, int type, int mr_flags, void **mhandle,
                                         void **ginHandle) {
  if (type == NCCL_PTR_HOST) {
    NCCLCHECK(ginComm->regMrSym(ginCollComm, addr, size, type, mr_flags, mhandle, ginHandle));
  } else if (type == NCCL_PTR_CUDA) {
    ncclResult_t dmabufResult = ncclInvalidUsage;
    if (ncclParamDmaBufEnable() && (props.ptrSupport & NCCL_PTR_DMABUF)) {
      ncclResult_t registrationResult = ncclSuccess;
      int dmabufFd = -1;
      dmabufResult = getDmaBufFd(addr, size, &dmabufFd);
      if (dmabufResult == ncclSuccess) {
        registrationResult = ginComm->regMrSymDmaBuf(ginCollComm, addr, size, type, 0, dmabufFd,
                                                     mr_flags, mhandle, ginHandle);
        close(dmabufFd);
      }
      if (registrationResult != ncclSuccess) {
        dmabufFd = -1;
        dmabufResult = getDmaBufFd(addr, size, &dmabufFd, true);
        if (dmabufResult == ncclSuccess) {
          NCCLCHECK(ginComm->regMrSymDmaBuf(ginCollComm, addr, size, type, 0, dmabufFd,
                                            mr_flags, mhandle, ginHandle));
          close(dmabufFd);
        }
      }
    }
    // Fallback to non-DMA-BUF if the DMA-BUF handle is not supported
    if (dmabufResult != ncclSuccess) {
      NCCLCHECK(ginComm->regMrSym(ginCollComm, addr, size, type, mr_flags, mhandle, ginHandle));
    }
  } else {
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}
static uint64_t isPowerOfTwo(uint64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

// ---- Context lifecycle ----

static ncclResult_t ncclRmaProxyCtxAlloc(struct ncclComm* comm, ncclGin_t* ginComm, struct ncclRmaProxyCtx* rmaProxyCtx) {
  // The clean up in case of failure will be done by the ncclRmaProxyDestroyContext function invoked by the caller.
  // Allocate the signals on the GPU and then register the memory region with the GIN plugin.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and signals.
  size_t signalsBufSize = (comm->nRanks + 1) * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&rmaProxyCtx->signalsDev, &rmaProxyCtx->signalsCumemhandle,
                           CU_MEM_HANDLE_TYPE_NONE, signalsBufSize, comm->memManager));
  CUDACHECK(cudaMemset(rmaProxyCtx->signalsDev, 0, signalsBufSize));
  NCCLCHECK(ncclRmaProxyRegMrSym(ginComm, rmaProxyCtx->ginCollComm, rmaProxyCtx->props, rmaProxyCtx->signalsDev, signalsBufSize,
                                 NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                 &rmaProxyCtx->signalsMhandle, &rmaProxyCtx->signalsGinHandle));
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

  // Allocate per-peer InProgress queues (kept as linked list, single consumer)
  rmaProxyCtx->inProgressQueues = ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(&comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->inProgressQueues[i]);
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyCtxAllocGraph(struct ncclComm* comm, ncclGin_t* ginComm, struct ncclRmaProxyCtx* rmaProxyCtx) {
  // The clean up in case of failure will be done by the ncclRmaProxyDestroyContext function invoked by the caller.
  size_t signalsBufSize = (comm->nRanks + 1) * sizeof(uint64_t);
  // Allocate the CPU-accessible signal for graph capture and then register the memory region with the GIN plugin.
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->cpuAccessSignals, &rmaProxyCtx->cpuAccessSignalsDev,
                                  comm->nRanks + 1, 0, &rmaProxyCtx->cpuAccessSignalsGdrHandle, comm->memManager));
  NCCLCHECK(ncclRmaProxyRegMrSym(ginComm, rmaProxyCtx->ginCollComm, rmaProxyCtx->props, rmaProxyCtx->cpuAccessSignalsDev, signalsBufSize,
                                 NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                 &rmaProxyCtx->cpuAccessSignalsMhandle, &rmaProxyCtx->cpuAccessSignalsGinHandle));
  // Allocate the host buffer to track the expected values of the signals
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->cpuAccessSignalsHost, signalsBufSize));

  // Allocate the flush buffer on the GPU and then register the memory region with the GIN plugin.
  size_t flushBufSize = comm->nRanks * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&rmaProxyCtx->flushBufDev, &rmaProxyCtx->flushBufCumemhandle,
                            CU_MEM_HANDLE_TYPE_NONE, flushBufSize, comm->memManager));
  CUDACHECK(cudaMemset(rmaProxyCtx->flushBufDev, 0, flushBufSize));
  NCCLCHECK(ncclRmaProxyRegMrSym(ginComm, rmaProxyCtx->ginCollComm, rmaProxyCtx->props, rmaProxyCtx->flushBufDev, flushBufSize,
                                  NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                  &rmaProxyCtx->flushBufMhandle, &rmaProxyCtx->flushBufGinHandle));
  // Allocate and initialize persistent descriptor queue
  rmaProxyCtx->persistentQueues = ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(&comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->persistentQueues[i]);
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyCreateContext(struct ncclComm *comm, void *collComm, ncclNetProperties_t props,
                                       void **outRmaProxyCtx, ncclNetDeviceHandle_t **outDevHandle) {
  ncclResult_t ret = ncclSuccess;
  // Get the GIN plugin interface
  ncclGin_t *ginComm = (ncclGin_t *)comm->rmaState.rmaProxyState.ncclGin;
  ncclNetDeviceHandle_t *devHandle = nullptr;

  ncclGinConfig_t config = { 0, 0, 1, 0, comm->config.trafficClass };

  // Allocate the RMA proxy context
  struct ncclRmaProxyCtx *rmaProxyCtx = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&rmaProxyCtx, 1), ret, fail);

  rmaProxyCtx->comm = comm;
  rmaProxyCtx->ginCollComm = collComm;
  rmaProxyCtx->props = props;
  NCCLCHECK(ginComm->createContext(collComm, &config, &rmaProxyCtx->ginCtx, NULL));

  NCCLCHECKGOTO(ncclRmaProxyCtxAlloc(comm, ginComm, rmaProxyCtx), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyCtxAllocGraph(comm, ginComm, rmaProxyCtx), ret, fail);

  // Allocate and initialize device handle
  NCCLCHECKGOTO(ncclCalloc(&devHandle, 1), ret, fail);
  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  devHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  devHandle->handle = (void *)rmaProxyCtx;
  devHandle->size = 0;
  devHandle->needsProxyProgress = 1;

  rmaProxyCtx->devHandle = devHandle;

  *outDevHandle = devHandle;
  *outRmaProxyCtx = rmaProxyCtx;

  return ncclSuccess;
fail:
  ncclRmaProxyDestroyContext(ginComm, rmaProxyCtx);
  return ret;
}

ncclResult_t ncclRmaProxyDestroyContext(ncclGin_t* ginComm, void* rmaProxyCtx){
  if (!rmaProxyCtx) return ncclSuccess;
  struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

  NCCLCHECK(ginComm->destroyContext(ctx->ginCtx));

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
          NCCLCHECK(ncclRmaProxyDestroyDescNonPersistent(desc));
        }
      }
    }
    free(ctx->circularBuffers);
  }

  // Free PI/CI arrays
  free(ctx->pis);
  free(ctx->cis);

  // Free InProgress queues and their Descs
  if (ctx->inProgressQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      struct ncclRmaProxyDesc *desc = ncclIntruQueueHead(&ctx->inProgressQueues[i]);
      while (desc != NULL) {
        struct ncclRmaProxyDesc *nextDesc = desc->next;
        ncclIntruQueueDequeue(&ctx->inProgressQueues[i]);
        NCCLCHECK(ncclRmaProxyDestroyDescNonPersistent(desc));
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
        NCCLCHECK(ncclRmaProxyDestroyDescPersistent(ctx->comm, desc));
        desc = nextDesc;
      }
    }
  }

  // Free counters (using GDR-aware deallocation)
  if (ctx->opSeqs) NCCLCHECK(freeMemCPUAccessible(ctx->opSeqs, ctx->opSeqsGdrHandle, ctx->comm->memManager));
  if (ctx->readySeqs) NCCLCHECK(freeMemCPUAccessible(ctx->readySeqs, ctx->readySeqsGdrHandle, ctx->comm->memManager));
  if (ctx->doneSeqs) NCCLCHECK(freeMemCPUAccessible(ctx->doneSeqs, ctx->doneSeqsGdrHandle, ctx->comm->memManager));

  // Free signals
  if (ginComm && ctx->ginCollComm && ctx->signalsMhandle)
    NCCLCHECK(ginComm->deregMrSym(ctx->ginCollComm, ctx->signalsMhandle));
  if (ctx->signalsDev) NCCLCHECK(ncclCudaFree(ctx->signalsDev, ctx->comm->memManager));

  // Free flush buffer
  if (ginComm && ctx->ginCollComm && ctx->flushBufMhandle)
    ginComm->deregMrSym(ctx->ginCollComm, ctx->flushBufMhandle);
  if (ctx->flushBufDev) ncclCudaFree(ctx->flushBufDev, ctx->comm->memManager);

  // Free CPU-accessible signals
  if (ginComm && ctx->ginCollComm && ctx->cpuAccessSignalsMhandle)
    ginComm->deregMrSym(ctx->ginCollComm, ctx->cpuAccessSignalsMhandle);
  if (ctx->cpuAccessSignals) freeMemCPUAccessible(ctx->cpuAccessSignals, ctx->cpuAccessSignalsGdrHandle, ctx->comm->memManager);

  // Free host signals buffers
  free(ctx->signalsHost);
  free(ctx->cpuAccessSignalsHost);

  // Note: devHandle->handle points to ctx itself, so we don't free it separately
  free(ctx->devHandle);

  free(ctx);

  return ncclSuccess;
}

// ---- Memory registration ----

ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
    void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS],
    ncclGinWindow_t rmaDevWins[NCCL_GIN_MAX_CONNECTIONS]){
      struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
      for (int n = 0; n < rmaProxyState->ginCommCount; n++) {
          NCCLCHECK(ncclRmaProxyRegMrSym(rmaProxyState->ncclGin, rmaProxyState->ginComms[n], rmaProxyState->props[n], address, size,
                                         NCCL_PTR_CUDA, 0, &rmaHostWins[n], &rmaDevWins[n]));
        if (rmaHostWins[n] == NULL) {
          WARN("rank %d - GIN Symmetric register failed: buff %p, size %ld", comm->rank, address, size);
          return ncclSystemError;
        }
      }
      return ncclSuccess;
}

ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONNECTIONS]){
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  for (int n = 0; n < rmaProxyState->ginCommCount; n++) {
    NCCLCHECK(rmaProxyState->ncclGin->deregMrSym(rmaProxyState->ginComms[n], rmaHostWins[n]));
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
    if (rmaProxyState->ginProgress == 1) {
      lock.unlock();
      for (int n=0; n<rmaProxyState->rmaProxyCtxCount; n++) {
        ncclResult_t ret = ncclRmaProxyProgress(rmaProxyState->ncclGin, rmaProxyState->rmaProxyCtxs[n]);
        if (ret != ncclSuccess) {
          COMPILER_ATOMIC_STORE_32(&rmaProxyState->asyncResult, ret, std::memory_order_release);
          INFO(NCCL_ALL,"%s:%d -> %d [RMA Proxy Progress Thread]", __FILE__, __LINE__, ret);
          rmaProxyState->ginProgress = -2;
          return NULL;
        }
      }
      std::this_thread::yield();
    } else if (rmaProxyState->ginProgress == 2) {
      // Pause requested for reclaim: acknowledge and sleep.
      // Main thread will do the actual freeing while we're paused.
      rmaProxyState->ginProgress = 0;
      rmaProxyState->cond.notify_one();
      rmaProxyState->cond.wait(lock);
    } else if (rmaProxyState->ginProgress == -1) {
      return NULL;
    } else if (rmaProxyState->ginProgress == 0) {
      rmaProxyState->cond.wait(lock);
    } else {
      INFO(NCCL_ALL,"%s:%d -> [RMA Proxy Progress Thread] state unknown %d", __FILE__, __LINE__, rmaProxyState->ginProgress);
      rmaProxyState->ginProgress = -2;
      return NULL;
    }
  }
}

ncclResult_t ncclRmaProxyConnectOnce(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaProxyState *rmaProxyState = &comm->rmaState.rmaProxyState;
  rmaProxyState->comm = comm;
  if (rmaProxyState->ncclGin == NULL) {
    WARN("GIN not supported.");
    return ncclInvalidUsage;
  }
  if (ncclParamGinEnable() == 0) {
    WARN("GIN is disabled.");
    return ncclInternalError;
  }
  if (rmaProxyState->connected) return ncclSuccess;

  rmaProxyState->ginInstance = comm->rmaGinContext;

  int ndev = 0;
  NCCLCHECK(rmaProxyState->ncclGin->devices(&ndev));
  if (ndev <= 0) {
    WARN("No GIN-capable devices found.");
    return ncclInternalError;
  }

  ncclNetProperties_t props;
  NCCLCHECK(rmaProxyState->ncclGin->getProperties(0, &props));
  rmaProxyState->ginType = props.netDeviceType;
  if (rmaProxyState->ginType != NCCL_NET_DEVICE_GIN_PROXY) {
    WARN("RMA proxy backend type mismatch.");
    return ncclInternalError;
  }

  int ginCommCount;
  int localGinDevs[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalGinDevs(comm, localGinDevs, &rmaProxyState->ginCommCount));
  ginCommCount = std::min<int>(rmaProxyState->ginCommCount, NCCL_GIN_MAX_CONNECTIONS);
  ginCommCount = std::min<int>(ginCommCount, ndev);

  int* allCommCounts = NULL;
  void** handles = NULL;
  char* allHandles = NULL;

  // Get the min local net count from all ranks
  NCCLCHECK(ncclCalloc(&allCommCounts, comm->nRanks));
  allCommCounts[comm->rank] = ginCommCount;
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allCommCounts, sizeof(int)), ret, fail);
  for (int i = 0; i < comm->nRanks; i++) {
    ginCommCount = std::min<int>(ginCommCount, allCommCounts[i]);
  }
  free(allCommCounts);
  allCommCounts = NULL;

  if (ginCommCount == 0) {
    WARN("Gin connect : min local net count is zero");
    ret = ncclSystemError;
    goto fail;
  }
  rmaProxyState->ginCommCount = ginCommCount;

  NCCLCHECKGOTO(ncclCalloc(&allHandles, (size_t)comm->nRanks * NCCL_NET_HANDLE_MAXSIZE), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&handles, comm->nRanks), ret, fail);
  for (int r = 0; r < comm->nRanks; r++) handles[r] = allHandles + r * NCCL_NET_HANDLE_MAXSIZE;

  for (int n = 0; n < ginCommCount; n++) {
    void* listenComm;
    NCCLCHECKGOTO(
      rmaProxyState->ncclGin->listen(rmaProxyState->ginInstance, localGinDevs[n],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);
    NCCLCHECKGOTO(
      rmaProxyState->ncclGin->connect(comm->netContext, handles, comm->nRanks, comm->rank,
                                      listenComm, rmaProxyState->ginComms + n),
      ret, fail);
    NCCLCHECKGOTO(rmaProxyState->ncclGin->getProperties(localGinDevs[n], &rmaProxyState->props[n]), ret, fail);
    NCCLCHECKGOTO(rmaProxyState->ncclGin->closeListen(listenComm), ret, fail);
  }
  free(handles);
  handles = NULL;
  free(allHandles);
  allHandles = NULL;

  // Create virtual RMA proxy contexts
  rmaProxyState->rmaProxyCtxCount = comm->config.numRmaCtx;
  NCCLCHECK(ncclCalloc(&rmaProxyState->rmaProxyCtxs, rmaProxyState->rmaProxyCtxCount));
  NCCLCHECK(ncclCalloc(&rmaProxyState->rmaProxyDevHandles, rmaProxyState->rmaProxyCtxCount));
  for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
    // Round-robin mapping to physical GIN communicator contexts
    int ginCommIdx = n % rmaProxyState->ginCommCount;
    NCCLCHECKGOTO(ncclRmaProxyCreateContext(comm, rmaProxyState->ginComms[ginCommIdx], rmaProxyState->props[ginCommIdx],
                                              &rmaProxyState->rmaProxyCtxs[n], &rmaProxyState->rmaProxyDevHandles[n]),
                  ret, fail);
  }

  // Check whether we need proxy progress and if so, start / wake up the progress thread.
  rmaProxyState->needsProxyProgress = 0;
  for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
    if (rmaProxyState->rmaProxyDevHandles[n]->needsProxyProgress) rmaProxyState->needsProxyProgress = 1;
  }
  if (rmaProxyState->needsProxyProgress) {
    rmaProxyState->ginProgress = 1;
    rmaProxyState->thread = std::thread(ncclRmaProxyProgressThread, rmaProxyState);
    ncclSetThreadName(rmaProxyState->thread, "NCCL RMA Proxy Progress%2d", comm->cudaDev);
  }

  INFO(NCCL_INIT, "Rank %d ncclRmaProxyConnectOnce: ginCommCount %d rmaProxyCtxCount:%d needsProxyProgress %d", comm->rank, ginCommCount, rmaProxyState->rmaProxyCtxCount, rmaProxyState->needsProxyProgress);

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

  if (rmaProxyState->needsProxyProgress) {
    {
      std::lock_guard<std::mutex> lock(rmaProxyState->mutex);
      rmaProxyState->ginProgress = -1;
      rmaProxyState->cond.notify_one();
    }
    rmaProxyState->thread.join();
  }

  // Destroy all virtual RMA proxy contexts
  if (rmaProxyState->rmaProxyCtxs) {
    for (int n = 0; n < rmaProxyState->rmaProxyCtxCount; n++) {
      if (rmaProxyState->rmaProxyCtxs[n] != NULL) {
        NCCLCHECK(ncclRmaProxyDestroyContext(rmaProxyState->ncclGin, rmaProxyState->rmaProxyCtxs[n]));
        rmaProxyState->rmaProxyCtxs[n] = NULL;
      }
    }
    // Free the dynamically allocated context array
    free(rmaProxyState->rmaProxyCtxs);
    rmaProxyState->rmaProxyCtxs = NULL;
  }

  // Free the device handles array
  if (rmaProxyState->rmaProxyDevHandles) {
    free(rmaProxyState->rmaProxyDevHandles);
    rmaProxyState->rmaProxyDevHandles = NULL;
  }

  // Close all physical GIN communicators
  for (int n = 0; n < rmaProxyState->ginCommCount; n++) {
    if (rmaProxyState->ginComms[n] != NULL) {
      NCCLCHECK(rmaProxyState->ncclGin->closeColl(rmaProxyState->ginComms[n]));
      rmaProxyState->ginComms[n] = NULL;
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
    printf("  ginProgress: %d\n", rmaProxyState->ginProgress);
    printf("  ginCommCount: %d\n", rmaProxyState->ginCommCount);
    printf("  rmaProxyCtxCount:%d\n", rmaProxyState->rmaProxyCtxCount);
    printf("  connected: %d\n", rmaProxyState->connected);
    printf("  needsProxyProgress: %d\n", rmaProxyState->needsProxyProgress);

    // dump per-context information
    for (int i = 0; i < rmaProxyState->rmaProxyCtxCount; i++) {
      struct ncclRmaProxyCtx* ctx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[i];
      printf("  rmaCtx[%d]: %p\n", i, ctx);
      printf("    rmaDevHandles: %p\n", ctx->devHandle);
      printf("    rmaCollComms: %p\n", ctx->ginCollComm);
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
