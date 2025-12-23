/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <assert.h>
#include "nccl.h"
#include "comm.h"
#include "gin/gin_host.h"
#include "alloc.h"
#include "checks.h"
#include "gdrwrap.h"
#include "plugin/nccl_net.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"
#include "compiler.h"

NCCL_PARAM(GinProxyQueueSize, "GIN_PROXY_QUEUE_SIZE", -1);
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamDmaBufEnable();

struct ginProxyGfdState {
  ncclGinProxyOp_t op;
  uint16_t counterId;
  int done;
  void *request;
};

// a member might be on the GPU, if it has a *GdrHandle counterpart
struct ginProxyHostGpuCtx {
  size_t queueSize;

  // size = nRanks * queueSize
  ncclGinProxyGfd_t *queues;
  void *cisGdrHandle;
  // Consumed Indices, one per rank
  uint32_t *cis;
  // to decrease the number of reads/writes to cis which might be on the GPU
  uint32_t *cisShadow;
  // Seen Indices one per rank
  uint32_t *sis;

  // same size as queues
  struct ginProxyGfdState *states;
  // same size as queues
  uint64_t *inlines;
  // inlines is registered as a memory region with the GIN plugin
  void *inlinesMhandle;
  void *inlinesGinHandle;
};

struct ginProxyCtx {
  struct ncclComm *comm;
  void *collComm;
  ncclNetDeviceHandle_v11_t *devHandle;
  ncclNetProperties_t props;

  // GPU queues, if GDR on the GPU, else on the CPU
  // Queue size, must be a power of 2
  struct ginProxyHostGpuCtx *hostGpuCtx;

  void *countersGdrHandle;
  uint64_t *counters;
  uint64_t *countersDev;
  CUmemGenericAllocationHandle signalsCumemhandle;
  void *signalsMhandle;
  void *signalsGinHandle;
  uint64_t *signalsDev;
  int hasError;
};

// Depending on GDR, allocate memory on the CPU or GPU.
// host_flags is not used for now, but it is here for future use.
template <typename T>
static ncclResult_t allocMemCPUAccessible(T **ptr, T **devPtr, size_t nelem, int host_flags,
                                          void **gdrHandle, bool forceHost = false) {
  if (ncclGdrCopy && !forceHost) {
    NCCLCHECK(ncclGdrCudaCalloc(ptr, devPtr, nelem, gdrHandle));
  } else {
    NCCLCHECK(ncclCuMemHostAlloc((void **)ptr, NULL, nelem * sizeof(T)));
    memset((void *)*ptr, 0, nelem * sizeof(T));
    *devPtr = *ptr;
    if (gdrHandle) *gdrHandle = NULL;  // Mark as host allocated by nulling GDR handle
  }
  return ncclSuccess;
}

// Depending on GDR, free memory on the CPU or GPU.
template <typename T>
static ncclResult_t freeMemCPUAccessible(T *ptr, void *gdrHandle) {
  if (gdrHandle != NULL) {  // If a GDR handle exists, it was GDR memory
    NCCLCHECK(ncclGdrCudaFree(gdrHandle));
  } else {  // Otherwise, it was host memory (or GDR was off)
    NCCLCHECK(ncclCuMemHostFree(ptr));
  }
  return ncclSuccess;
}

static ncclResult_t getDmaBufFd(void *addr, size_t length, int *fd,
                                bool forceNonDataDirect = false) {
  if (ncclParamDmaBufEnable() == 0) return ncclInvalidUsage;

#if CUDA_VERSION >= 11070
  static size_t hostPageSize = sysconf(_SC_PAGESIZE);
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

static ncclResult_t proxyGinPollCompletions(ncclGin_t *ginComm, void *collComm,
                                            struct ginProxyCtx *ctx,
                                            struct ginProxyHostGpuCtx *hostGpuCtx) {
  for (int targetRank = 0; targetRank < ctx->comm->nRanks; targetRank++) {
    // loop on all seen but unconsumed GFDs
    for (uint32_t i = hostGpuCtx->cisShadow[targetRank]; i < hostGpuCtx->sis[targetRank]; i++) {
      uint32_t idx = i & (hostGpuCtx->queueSize - 1);
      struct ginProxyGfdState *state =
        &hostGpuCtx->states[targetRank * hostGpuCtx->queueSize + idx];
      // no need to poll if already done
      if (!state->done) {
        ginComm->test(collComm, state->request, &state->done);
        if (state->done) {
          TRACE(NCCL_NET, "GFD completed - stateIdx: %lu, request: %p", state - hostGpuCtx->states,
                state->request);
          // update the counter specified in the GFD
          if (state->op & ncclGinProxyOpWithCounter) {
            COMPILER_ATOMIC_STORE(&ctx->counters[state->counterId], ctx->counters[state->counterId] + 1,
                              std::memory_order_relaxed);
            TRACE(NCCL_NET, "Updated counter %d to %ld", state->counterId,
                  ctx->counters[state->counterId]);
          }
        }
      }
      // allow holes in the CI space to get resolved
      if (state->done && i == hostGpuCtx->cisShadow[targetRank]) {
        // tell the GPU that we have consumed the GFD
        COMPILER_ATOMIC_STORE(&hostGpuCtx->cis[targetRank], ++hostGpuCtx->cisShadow[targetRank],
                          std::memory_order_relaxed);
        TRACE(NCCL_NET, "Updated cis[%u] to %u", targetRank, hostGpuCtx->cisShadow[targetRank]);
      }
    }
  }

  return ncclSuccess;
}

static int proxyGinPollGfd(struct ginProxyCtx *ctx, ginProxyHostGpuCtx *hostGpuCtx, int targetRank,
                           ncclGinProxyGfd_t *gfd, struct ginProxyGfdState **state) {
  ncclGinProxyGfd_t *q = hostGpuCtx->queues + targetRank * hostGpuCtx->queueSize;
  uint32_t idx = hostGpuCtx->sis[targetRank] & (hostGpuCtx->queueSize - 1);
  ncclGinProxyQword_t qword;
  COMPILER_ATOMIC_LOAD_DEST(&q[idx].qword[ncclGinProxyGfdHeader].raw, &qword.raw, std::memory_order_relaxed);
  if (qword.flag.v == 0) {
    return 0;
  }

  // We know for sure that the first qword is there, copy it.
  gfd->qword[ncclGinProxyGfdHeader] = q[idx].qword[ncclGinProxyGfdHeader];
  // Wait for and copy the other qwords.
  for (int k = 1; k < ncclGinProxyGfdQwords; k++) {
    do {
      COMPILER_ATOMIC_LOAD_DEST(&q[idx].qword[k].raw, &qword.raw, std::memory_order_relaxed);
    } while (qword.flag.v == 0);
    gfd->qword[k] = qword;
  }
  // Now we have the full GFD in the local struct.

  // Reset the GFD in the queue. This lets the producer know that the GFD is consumed.
  for (int k = 0; k < ncclGinProxyGfdQwords; k++) {
    COMPILER_ATOMIC_STORE(&q[idx].qword[k].raw, 0, std::memory_order_relaxed);
  }

  // set the counter_id into the state
  uint32_t stateIdx = targetRank * hostGpuCtx->queueSize + idx;
  *state = &hostGpuCtx->states[stateIdx];
  (*state)->op = (ncclGinProxyOp_t)(gfd->qword[ncclGinProxyGfdHeader].header.op);
  (*state)->counterId = gfd->qword[ncclGinProxyGfdCompletion].completion.counterId;
  (*state)->done = 0;
  (*state)->request = NULL;

  TRACE(NCCL_NET,
        "GFD to target PE %d raw idx: %u, idx: %u - op: %#lx, size: %lu, srcOff: %lu, dstOff: %lu, "
        "srcHandle: %lu, dstHandle: %lu, counterId: %u, signalId: %u, stateIdx: %u",
        targetRank, hostGpuCtx->sis[targetRank], idx, gfd->qword[ncclGinProxyGfdHeader].header.op,
        gfd->qword[ncclGinProxyGfdHeader].header.size,
        gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff,
        gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff,
        gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle,
        gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle,
        gfd->qword[ncclGinProxyGfdCompletion].completion.counterId,
        gfd->qword[ncclGinProxyGfdCompletion].completion.signalId, stateIdx);

  hostGpuCtx->sis[targetRank]++;

  return 1;
}

static int mapGfdOpToCollNetOp(ncclGinProxyGfd_t *gfd) {
  switch (gfd->qword[ncclGinProxyGfdHeader].header.op &
          (ncclGinProxyOpComplMask & ~ncclGinProxyOpWithCounter)) {
    case ncclGinProxyOpWithSignalInc:
      return NCCL_NET_SIGNAL_OP_INC;
    case ncclGinProxyOpWithSignalAdd:
      return NCCL_NET_SIGNAL_OP_ADD;
    default:
      return -1;
  }
}

static ncclResult_t proxyGinProcessGfd(ncclGin_t *ginComm, void *collComm, struct ginProxyCtx *ctx,
                                       struct ginProxyHostGpuCtx *hostGpuCtx, int targetRank,
                                       ncclGinProxyGfd_t *gfd, struct ginProxyGfdState *state) {
  int signalOp;
  uint64_t signalVal;

  uint64_t size = gfd->qword[ncclGinProxyGfdHeader].header.size;
  uint64_t srcOff;
  void *srcHandle;
  if (gfd->qword[ncclGinProxyGfdHeader].header.op & ncclGinProxyOpWithInline) {
    uint64_t *inlineVal = &hostGpuCtx->inlines[gfd - hostGpuCtx->queues];
    srcOff = (uint64_t)&inlineVal[0] - (uint64_t)hostGpuCtx->inlines;
    // reconstruct the inline value from the two qwords
    *inlineVal = gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow;
    if (size == 8) {
      *inlineVal |= (uint64_t)gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 << 32;
      *inlineVal |= (uint64_t)gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh << 48;
    }
    srcHandle = hostGpuCtx->inlinesMhandle;
  } else {
    srcOff = gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff;
    srcHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle;
  }
  uint64_t dstOff = gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff;
  void *dstHandle = (void *)(uint64_t)gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle;

  switch (gfd->qword[ncclGinProxyGfdHeader].header.op & ncclGinProxyOpBaseMask) {
    case ncclGinProxyOpPut:
      signalOp = mapGfdOpToCollNetOp(gfd);
      if (signalOp == -1) {
        // First cast from 63 bits to 64 bits and then to void * to avoid warnings
        NCCLCHECK(ginComm->iput(collComm, srcOff, srcHandle, size, dstOff, dstHandle,
                                targetRank, &state->request));
      } else {
        // reconstruct the signal value from the two qwords
        signalVal = gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow;
        signalVal |= (uint64_t)gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 << 16;
        signalVal |= (uint64_t)gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh << 32;
        uint64_t signalOff =
          gfd->qword[ncclGinProxyGfdCompletion].completion.signalId * sizeof(uint64_t);
        NCCLCHECK(ginComm->iputSignal(collComm, srcOff, srcHandle, size, dstOff, dstHandle,
                                      targetRank, signalOff, ctx->signalsGinHandle, signalVal,
                                      signalOp, &state->request));
      }
      break;
    default:
      // this error should already have been checked in pollGfd
      assert(0);
  }
  TRACE(NCCL_NET, "GFD submitted into GIN plugin - stateIdx: %lu, request: %p",
        state - hostGpuCtx->states, state->request);
  return ncclSuccess;
}

static uint64_t isPowerOfTwo(uint64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

// Check if the GIN plugin supports DMA-BUF, if so we can try to get the DMA-BUF handle from CUDA,
// if that fails we fallback to non-DMA-BUF
static ncclResult_t ncclGinProxyRegMrSym(ncclGin_t *ginComm, struct ginProxyCtx *ctx, void *addr,
                                         size_t size, int type, int mr_flags, void **mhandle,
                                         void **ginHandle) {
  if (type == NCCL_PTR_HOST) {
    NCCLCHECK(ginComm->regMrSym(ctx->collComm, addr, size, type, mr_flags, mhandle, ginHandle));
  } else if (type == NCCL_PTR_CUDA) {
    ncclResult_t dmabufResult = ncclInvalidUsage;
    if (ncclParamDmaBufEnable() && (ctx->props.ptrSupport & NCCL_PTR_DMABUF)) {
      ncclResult_t registrationResult = ncclSuccess;
      int dmabufFd = -1;
      dmabufResult = getDmaBufFd(addr, size, &dmabufFd);
      if (dmabufResult == ncclSuccess) {
        registrationResult = ginComm->regMrSymDmaBuf(ctx->collComm, addr, size, type, 0, dmabufFd,
                                                     mr_flags, mhandle, ginHandle);
        close(dmabufFd);
      }
      if (registrationResult != ncclSuccess) {
        dmabufFd = -1;
        dmabufResult = getDmaBufFd(addr, size, &dmabufFd, true);
        if (dmabufResult == ncclSuccess) {
          NCCLCHECK(ginComm->regMrSymDmaBuf(ctx->collComm, addr, size, type, 0, dmabufFd,
                                            mr_flags, mhandle, ginHandle));
          close(dmabufFd);
        }
      }
    }
    // Fallback to non-DMA-BUF if the DMA-BUF handle is not supported
    if (dmabufResult != ncclSuccess) {
      NCCLCHECK(ginComm->regMrSym(ctx->collComm, addr, size, type, mr_flags, mhandle, ginHandle));
    }
  } else {
    return ncclInvalidUsage;
  }

  return ncclSuccess;
}

ncclResult_t ncclGinProxyCreateContext(struct ncclComm *comm, void *collComm, int devId,
                                       int nSignals, int nCounters, void **outGinCtx,
                                       ncclNetDeviceHandle_v11_t **outDevHandle) {
  ncclGin_t *ginComm = (ncclGin_t *)comm->sharedRes->ginState.ncclGin;

  if (!ncclGdrCopy)
    INFO(NCCL_NET, "GIN Proxy will not be using GDRCopy");

  struct ginProxyCtx *proxyCtx = NULL;
  NCCLCHECK(ncclCalloc(&proxyCtx, 1));

  proxyCtx->comm = comm;
  proxyCtx->collComm = collComm;

  // Sanitize the queue size
  NCCLCHECK(ginComm->getProperties(devId, &proxyCtx->props));
  uint64_t queueSize = ncclParamGinProxyQueueSize();
  uint32_t maxRequests = NCCL_NET_MAX_REQUESTS * proxyCtx->props.maxRecvs;
  if (queueSize == -1) {
    queueSize = maxRequests;
  }
  if (queueSize > maxRequests) {
    INFO(NCCL_NET,
         "NCCL_GIN_PROXY_QUEUE_SIZE is greater than the maximum outstanding requests in the GIN "
         "plugin (%d), using the default/maximum value instead",
         maxRequests);
    queueSize = maxRequests;
  }
  if (queueSize < 1) {
    INFO(NCCL_NET,
         "NCCL_GIN_PROXY_QUEUE_SIZE is less than 1, using the default/maximum value instead");
    queueSize = maxRequests;
  }
  if (!isPowerOfTwo(queueSize)) {
    INFO(
      NCCL_NET,
      "NCCL_GIN_PROXY_QUEUE_SIZE is not a power of two, using the default/maximum value instead");
    queueSize = maxRequests;
  }

  // Allocate the counters on the GPU or CPU depending on GDR
  NCCLCHECK(allocMemCPUAccessible(&proxyCtx->counters, &proxyCtx->countersDev, nCounters,
                                  CU_MEMHOSTALLOC_WRITECOMBINED,
                                  &proxyCtx->countersGdrHandle));

  // Allocate the signals on the GPU and then register the memory region with the GIN plugin.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and
  // signals.
  size_t signalsBufSize = nSignals * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&proxyCtx->signalsDev, &proxyCtx->signalsCumemhandle,
                           CU_MEM_HANDLE_TYPE_NONE, signalsBufSize));
  CUDACHECK(cudaMemset(proxyCtx->signalsDev, 0, signalsBufSize));
  NCCLCHECK(ncclGinProxyRegMrSym(ginComm, proxyCtx, proxyCtx->signalsDev, signalsBufSize,
                                 NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                 &proxyCtx->signalsMhandle, &proxyCtx->signalsGinHandle));

  NCCLCHECK(ncclCalloc(&proxyCtx->hostGpuCtx, 1));
  struct ginProxyHostGpuCtx *hostGpuCtx = proxyCtx->hostGpuCtx;
  hostGpuCtx->queueSize = queueSize;
  size_t queuesLength = hostGpuCtx->queueSize * comm->nRanks;
  NCCLCHECK(ncclCalloc(&hostGpuCtx->states, queuesLength));
  NCCLCHECK(ncclCalloc(&hostGpuCtx->cisShadow, comm->nRanks));
  NCCLCHECK(ncclCalloc(&hostGpuCtx->sis, comm->nRanks));
  NCCLCHECK(ncclCalloc(&hostGpuCtx->inlines, queuesLength));
  NCCLCHECK(ncclGinProxyRegMrSym(ginComm, proxyCtx, hostGpuCtx->inlines,
                                       queuesLength * sizeof(uint64_t), NCCL_PTR_HOST, 0,
                                       &hostGpuCtx->inlinesMhandle, &hostGpuCtx->inlinesGinHandle));

  ncclGinProxyGpuCtx_t devGpuCtx_h;
  devGpuCtx_h.nranks = comm->nRanks;
  devGpuCtx_h.queueSize = hostGpuCtx->queueSize;
  devGpuCtx_h.counters = proxyCtx->countersDev;
  devGpuCtx_h.signals = proxyCtx->signalsDev;
  NCCLCHECK(ncclCudaCalloc(&devGpuCtx_h.pis, comm->nRanks));

  // Allocate the GFD queues, CIs, counters, signals and test/wait variables on the either the CPU
  // or GPU.
  NCCLCHECK(allocMemCPUAccessible(&hostGpuCtx->queues, &devGpuCtx_h.queues, queuesLength, 0,
                                        NULL, true /*forceHost*/));
  NCCLCHECK(allocMemCPUAccessible(&hostGpuCtx->cis, &devGpuCtx_h.cis, comm->nRanks,
                                        CU_MEMHOSTALLOC_WRITECOMBINED, &hostGpuCtx->cisGdrHandle));

  ncclGinProxyGpuCtx_t *devGpuCtx_d = NULL;
  NCCLCHECK(ncclCudaCalloc(&devGpuCtx_d, 1));
  // Copy the proxy's devGpuCtx to the GPU
  NCCLCHECK(ncclCudaMemcpy(devGpuCtx_d, &devGpuCtx_h, 1));

  ncclNetDeviceHandle_v11_t *devHandle = NULL;
  NCCLCHECK(ncclCalloc(&devHandle, 1));
  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  devHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  devHandle->handle = (void *)devGpuCtx_d;
  devHandle->size = 0;
  devHandle->needsProxyProgress = 1;

  proxyCtx->devHandle = devHandle;

  *outDevHandle = devHandle;
  *outGinCtx = proxyCtx;

  return ncclSuccess;
}

ncclResult_t ncclGinProxyRegister(ncclGin_t *ginComm, void *ginCtx, void *addr, size_t size,
                                  int type, int mr_flags, void **mhandle, void **ginHandle) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  // Register the memory region with the GIN plugin
  NCCLCHECK(ncclGinProxyRegMrSym(ginComm, ctx, addr, size, type, mr_flags, mhandle, ginHandle));
  return ncclSuccess;
}

ncclResult_t ncclGinProxyDeregister(ncclGin_t *ginComm, void *ginCtx, void *mhandle) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  // Deregister the memory region with the GIN plugin
  NCCLCHECK(ginComm->deregMrSym(ctx->collComm, mhandle));
  return ncclSuccess;
}

ncclResult_t ncclGinProxyDestroyContext(ncclGin_t *ginComm, void *ginCtx) {
  if (!ginCtx) return ncclSuccess;
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;

  // Free counters
  if (ctx) {
    if (ctx->counters || ctx->countersGdrHandle)
      freeMemCPUAccessible(ctx->counters, ctx->countersGdrHandle);

    // Free signals
    if (ginComm && ctx->collComm && ctx->signalsMhandle)
      ginComm->deregMrSym(ctx->collComm, ctx->signalsMhandle);
    if (ctx->signalsDev) ncclCudaFree(ctx->signalsDev);

    // Free hostGpuCtx and its allocations
    struct ginProxyHostGpuCtx *hostGpuCtx = ctx->hostGpuCtx;
    if (hostGpuCtx) {
      if (hostGpuCtx->cisShadow) free(hostGpuCtx->cisShadow);
      if (hostGpuCtx->sis) free(hostGpuCtx->sis);
      if (hostGpuCtx->states) free(hostGpuCtx->states);
      if (hostGpuCtx->inlines) free(hostGpuCtx->inlines);
      if (ginComm && ctx->collComm && hostGpuCtx->inlinesMhandle)
        ginComm->deregMrSym(ctx->collComm, hostGpuCtx->inlinesMhandle);
      if (hostGpuCtx->queues) freeMemCPUAccessible(hostGpuCtx->queues, NULL);
      if (hostGpuCtx->cis || hostGpuCtx->cisGdrHandle)
        freeMemCPUAccessible(hostGpuCtx->cis, hostGpuCtx->cisGdrHandle);
      free(hostGpuCtx);
    }

    ncclNetDeviceHandle_v11_t *devHandle = (ncclNetDeviceHandle_v11_t *)ctx->devHandle;
    if (devHandle) {
      if (devHandle->handle) ncclCudaFree((void *)devHandle->handle);
      free(devHandle);
    }

    free(ctx);
  }

  return ncclSuccess;
}

ncclResult_t ncclGinProxyProgress(ncclGin_t *ginComm, void *ginCtx) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;

  NCCLCHECK(proxyGinPollCompletions(ginComm, ctx->collComm, ctx, ctx->hostGpuCtx));
  for (int targetRank = 0; targetRank < ctx->comm->nRanks; targetRank++) {
    // Poll on the GFD queue
    ncclGinProxyGfd_t gfd;
    struct ginProxyGfdState *state = NULL;
    if (proxyGinPollGfd(ctx, ctx->hostGpuCtx, targetRank, &gfd, &state)) {
      ncclResult_t ret =
        proxyGinProcessGfd(ginComm, ctx->collComm, ctx, ctx->hostGpuCtx, targetRank, &gfd, state);
      if (ret) ctx->hasError = ret;
      NCCLCHECK(ret);
    }
    if (ginComm->ginProgress) ginComm->ginProgress(ctx->collComm);
  }

  return ncclSuccess;
}

ncclResult_t ncclGinProxyQueryLastError(ncclGin_t *ginComm, void *ginCtx, bool *hasError) {
  struct ginProxyCtx *ctx = (struct ginProxyCtx *)ginCtx;
  *hasError = ctx->hasError;
  return ncclSuccess;
}
