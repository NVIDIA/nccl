#include <assert.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "gdrwrap.h"
#include "comm.h"
#include "bootstrap.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "dev_runtime.h"
#include "nccl_device/gin/proxy/gin_proxy_device_host_common.h"


extern int64_t ncclParamDmaBufEnable();
extern int64_t ncclParamIbDataDirect();
extern int64_t ncclParamGinEnable();
extern int64_t ncclParamGinType();

NCCL_PARAM(RmaProxyDumpSignal, "RMA_PROXY_DUMP_SIGNAL", -1);
NCCL_PARAM(RmaProxyQueueSize, "RMA_PROXY_QUEUE_SIZE", -1);

#include <signal.h>
static ncclRmaProxyState* ncclLastRmaProxyState;

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
          uint64_t readySeq = __atomic_load_n(&ctx->readySeqs[peer], __ATOMIC_ACQUIRE);
          uint64_t doneSeq = __atomic_load_n(&ctx->doneSeqs[peer], __ATOMIC_ACQUIRE);
          uint64_t opSeq = __atomic_load_n(&ctx->opSeqs[peer], __ATOMIC_ACQUIRE);
          uint32_t pi = __atomic_load_n(&ctx->pis[peer], __ATOMIC_ACQUIRE);
          uint32_t ci = __atomic_load_n(&ctx->cis[peer], __ATOMIC_ACQUIRE);
          printf("      Peer %d: readySeq: %lu, doneSeq: %lu, opSeq: %lu, PI: %u, CI: %u\n",
                 peer, readySeq, doneSeq, opSeq, pi, ci);

          // Count and print pending Descs from circular buffer
          int pendingCount = pi - ci;
          printf("        Pending Descs: %d\n", pendingCount);
          for (uint32_t j = ci; j < pi; j++) {
            uint32_t idx = j & (ctx->queueSize - 1);
            struct ncclRmaProxyDesc* desc = ctx->pendingQueues[peer * ctx->queueSize + idx];
            if (desc != NULL) {
              printf("          Desc: seq=%lu targetRank=%d size=%zu\n",
                    desc->seq, desc->targetRank, desc->size);
            }
          }

          // Count in-progress Descs
          int inProgressCount = 0;
          struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&ctx->rmaProxyInProgressQueues[peer]);
          while (desc != NULL) {
            inProgressCount++;
            desc = desc->next;
          }
          printf("        In-progress Descs: %d\n", inProgressCount);
          // print all in-progress Descs
          desc = ncclIntruQueueHead(&ctx->rmaProxyInProgressQueues[peer]);
          while (desc != NULL) {
            printf("          Desc: seq=%lu targetRank=%d size=%zu\n",
                  desc->seq, desc->targetRank, desc->size);
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

static uint64_t isPowerOfTwo(uint64_t n) { return (n > 0) && ((n & (n - 1)) == 0); }

ncclResult_t ncclRmaProxyCreateContext(struct ncclComm *comm, void *collComm, ncclNetProperties_t props,
                                       void **outRmaProxyCtx, ncclNetDeviceHandle_t **outDevHandle) {
  // Get the GIN plugin interface
  ncclGin_t *ginComm = (ncclGin_t *)comm->rmaState.rmaProxyState.ncclGin;

  // Allocate the RMA proxy context
  struct ncclRmaProxyCtx *rmaProxyCtx = NULL;
  NCCLCHECK(ncclCalloc(&rmaProxyCtx, 1));

  rmaProxyCtx->comm = comm;
  rmaProxyCtx->ginCollComm = collComm;
  rmaProxyCtx->props = props;

  // Allocate the signals on the GPU and then register the memory region with the GIN plugin.
  // Enforcing strong ordering on the signals mr is vital to ensure ordering between puts and signals.
  size_t signalsBufSize = (comm->nRanks + 1) * sizeof(uint64_t);
  NCCLCHECK(ncclCuMemAlloc((void **)&rmaProxyCtx->signalsDev, &rmaProxyCtx->signalsCumemhandle,
                           CU_MEM_HANDLE_TYPE_NONE, signalsBufSize));
  CUDACHECK(cudaMemset(rmaProxyCtx->signalsDev, 0, signalsBufSize));
  NCCLCHECK(ncclRmaProxyRegMrSym(ginComm, rmaProxyCtx->ginCollComm, rmaProxyCtx->props, rmaProxyCtx->signalsDev, signalsBufSize,
                                 NCCL_PTR_CUDA, NCCL_NET_MR_FLAG_FORCE_SO,
                                 &rmaProxyCtx->signalsMhandle, &rmaProxyCtx->signalsGinHandle));

  // Allocate the host buffer to track the expected values of the signals
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->signalsHost, signalsBufSize));

  // Allocate the sequence numbers for the per-rank network function descriptors
  // These are allocated as CPU-accessible memory (either GDR or host memory)
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->opSeqs, &rmaProxyCtx->opSeqsDev,
                                  comm->nRanks, 0, &rmaProxyCtx->opSeqsGdrHandle));
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->readySeqs, &rmaProxyCtx->readySeqsDev,
                                  comm->nRanks, 0, &rmaProxyCtx->readySeqsGdrHandle));
  NCCLCHECK(allocMemCPUAccessible(&rmaProxyCtx->doneSeqs, &rmaProxyCtx->doneSeqsDev,
                                  comm->nRanks, 0, &rmaProxyCtx->doneSeqsGdrHandle));

  // Sanitize and set up the lock-free circular buffer queue size
  uint64_t queueSize = ncclParamRmaProxyQueueSize();
  uint32_t maxRequests = NCCL_NET_MAX_REQUESTS * props.maxRecvs;
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
  size_t pendingQueuesLength = comm->nRanks * queueSize;
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->pendingQueues, pendingQueuesLength));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->pis, comm->nRanks));
  NCCLCHECK(ncclCalloc(&rmaProxyCtx->cis, comm->nRanks));

  // Allocate per-peer InProgress queues (kept as linked list, single consumer)
  rmaProxyCtx->rmaProxyInProgressQueues = ncclMemoryStackAlloc<struct ncclIntruQueue<struct ncclRmaProxyDesc, &ncclRmaProxyDesc::next>>(&comm->memPermanent, comm->nRanks);
  for (int i = 0; i < comm->nRanks; i++) {
    ncclIntruQueueConstruct(&rmaProxyCtx->rmaProxyInProgressQueues[i]);
  }

  // Allocate and initialize device handle
  ncclNetDeviceHandle_t *devHandle = NULL;
  NCCLCHECK(ncclCalloc(&devHandle, 1));
  devHandle->netDeviceType = NCCL_NET_DEVICE_GIN_PROXY;
  devHandle->netDeviceVersion = NCCL_GIN_PROXY_VERSION;
  devHandle->handle = (void *)rmaProxyCtx;
  devHandle->size = 0;
  devHandle->needsProxyProgress = 1;

  rmaProxyCtx->devHandle = devHandle;

  *outDevHandle = devHandle;
  *outRmaProxyCtx = rmaProxyCtx;

  return ncclSuccess;
}

// Poll and test completion of InProgress Descs for a given peer
// Returns after testing head Desc (stops on first incomplete to enforce FIFO)
static ncclResult_t ncclRmaProxyPollCompletion(ncclGin_t *ncclGin, struct ncclRmaProxyCtx *ctx, int peer) {
  while (true) {
    struct ncclRmaProxyDesc *inProgressDesc = ncclIntruQueueHead(&ctx->rmaProxyInProgressQueues[peer]);
    if (inProgressDesc == NULL) break;  // No InProgress Descs

    int done = 0;
    NCCLCHECK(ncclGin->test(ctx->ginCollComm, inProgressDesc->request, &done));
    if (done) {
      INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollCompletion: targetRank=%d descSeq=%lu COMPLETED, updating doneSeq",
        ctx->comm->rank, inProgressDesc->targetRank, inProgressDesc->seq);

      // Update the doneSeq for the target rank with RELEASE to ensure GPU sees it
      __atomic_store_n(&ctx->doneSeqs[inProgressDesc->targetRank], inProgressDesc->seq, __ATOMIC_RELEASE); // sync with the custreamWait aquire semantic
      // Dequeue and free the completed Desc
      ncclIntruQueueDequeue(&ctx->rmaProxyInProgressQueues[peer]);
      ncclMemoryPoolFree(&ctx->comm->memPool_ncclRmaProxyDesc, inProgressDesc);

      free(inProgressDesc);
    } else {
      // Head is not done - stop testing to enforce FIFO completion order
      break;
    }
  }
  return ncclSuccess;
}

// Poll and issue ready Pending Descs for a given peer
// Moves ready Descs from pending queue to InProgress queue
static ncclResult_t ncclRmaProxyPollDesc(ncclGin_t *ncclGin, struct ncclRmaProxyCtx *ctx, int peer) {
  while (true) {
    // Lock-free dequeue: Check if queue has entries
    uint32_t ci = __atomic_load_n(&ctx->cis[peer], __ATOMIC_RELAXED);
    uint32_t pi = __atomic_load_n(&ctx->pis[peer], __ATOMIC_ACQUIRE);

    if (ci >= pi) {
      break;  // Empty queue
    }

    // Read descriptor from queue
    uint32_t idx = ci & (ctx->queueSize - 1);
    struct ncclRmaProxyDesc *pendingDesc = ctx->pendingQueues[peer * ctx->queueSize + idx];

    // Check if this Desc is ready to be issued
    uint64_t readySeq = __atomic_load_n(&ctx->readySeqs[peer], __ATOMIC_ACQUIRE);
    if (readySeq >= pendingDesc->seq) {
      // Advance CI with RELEASE to ensure descriptor is consumed
      __atomic_store_n(&ctx->cis[peer], ci + 1, __ATOMIC_RELEASE);

      // Issue the network operation
      if (pendingDesc->signal.op == 0) {
        // No signal operation
        NCCLCHECK(ncclGin->iput(ctx->ginCollComm,
          pendingDesc->srcOff, pendingDesc->srcHandle, pendingDesc->size,
          pendingDesc->dstOff, pendingDesc->dstHandle,
          pendingDesc->targetRank, &pendingDesc->request));
      } else {
        // Signal operation needed
        NCCLCHECK(ncclGin->iputSignal(ctx->ginCollComm,
          pendingDesc->srcOff, pendingDesc->srcHandle, pendingDesc->size,
          pendingDesc->dstOff, pendingDesc->dstHandle,
          pendingDesc->targetRank, pendingDesc->signal.offset, pendingDesc->signal.signalMhandle,
          pendingDesc->signal.val, pendingDesc->signal.op, &pendingDesc->request));
      }

      // Enqueue to InProgress queue (no lock needed - progress thread only)
      ncclIntruQueueEnqueue(&ctx->rmaProxyInProgressQueues[peer], pendingDesc);

      INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollDesc: targetRank=%d descSeq=%lu readySeq=%lu srcOff=%lu srcHandle=%p dstOff=%lu dstHandle=%p size=%lu - issuing network operation",
        ctx->comm->rank, pendingDesc->targetRank, pendingDesc->seq, readySeq, pendingDesc->srcOff, pendingDesc->srcHandle, pendingDesc->dstOff, pendingDesc->dstHandle, pendingDesc->size);
    } else {
      // ReadySeq not ready yet - stop processing this peer's pending queue to maintain FIFO order
      break;
    }
  }
  return ncclSuccess;
}

// Checks the RMA proxy progress.
ncclResult_t ncclRmaProxyProgress(ncclGin_t *ncclGin, void *rmaProxyCtx) {
  struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

  // Loop through each peer's queues
  for (int i = 0; i < ctx->comm->nRanks; i++) {
    // Step 1: Poll completion of InProgress Descs
    NCCLCHECK(ncclRmaProxyPollCompletion(ncclGin, ctx, i));

    // Step 2: Poll and issue ready Pending Descs
    NCCLCHECK(ncclRmaProxyPollDesc(ncclGin, ctx, i));
  }
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyDestroyContext(ncclGin_t* ginComm, void* rmaProxyCtx){
  if (!rmaProxyCtx) return ncclSuccess;
  struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

  // Free descriptors remaining in circular buffers
  if (ctx->pendingQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      uint32_t ci = __atomic_load_n(&ctx->cis[i], __ATOMIC_RELAXED);
      uint32_t pi = __atomic_load_n(&ctx->pis[i], __ATOMIC_RELAXED);
      // Free any remaining pending descriptors
      for (uint32_t j = ci; j < pi; j++) {
        uint32_t idx = j & (ctx->queueSize - 1);
        struct ncclRmaProxyDesc *desc = ctx->pendingQueues[i * ctx->queueSize + idx];
        if (desc != NULL) {
          free(desc);
        }
      }
    }
    free(ctx->pendingQueues);
  }

  // Free PI/CI arrays
  if (ctx->pis) free(ctx->pis);
  if (ctx->cis) free(ctx->cis);

  // Free InProgress queues and their Descs
  if (ctx->rmaProxyInProgressQueues) {
    for (int i = 0; i < ctx->comm->nRanks; i++) {
      struct ncclRmaProxyDesc *desc = ncclIntruQueueHead(&ctx->rmaProxyInProgressQueues[i]);
      while (desc != NULL) {
        struct ncclRmaProxyDesc *nextDesc = desc->next;
        ncclIntruQueueDequeue(&ctx->rmaProxyInProgressQueues[i]);
        free(desc);
        desc = nextDesc;
      }
    }
  }

  // Free counters (using GDR-aware deallocation)
  if (ctx->opSeqs) freeMemCPUAccessible(ctx->opSeqs, ctx->opSeqsGdrHandle);
  if (ctx->readySeqs) freeMemCPUAccessible(ctx->readySeqs, ctx->readySeqsGdrHandle);
  if (ctx->doneSeqs) freeMemCPUAccessible(ctx->doneSeqs, ctx->doneSeqsGdrHandle);

  // Free signals
  if (ginComm && ctx->ginCollComm && ctx->signalsMhandle)
    ginComm->deregMrSym(ctx->ginCollComm, ctx->signalsMhandle);
  if (ctx->signalsDev) ncclCudaFree(ctx->signalsDev);

  // Free host signals buffer
  if (ctx->signalsHost) free(ctx->signalsHost);

  ncclNetDeviceHandle_t *devHandle = (ncclNetDeviceHandle_t *)ctx->devHandle;
  if (devHandle) {
    // Note: devHandle->handle points to ctx itself, so we don't free it separately
    free(devHandle);
  }

  free(ctx);

  return ncclSuccess;
}


ncclResult_t ncclRmaProxyRegister(struct ncclComm* comm, void* address, size_t size,
    void* rmaHostWins[NCCL_GIN_MAX_CONTEXTS],
    ncclGinWindow_t rmaDevWins[NCCL_GIN_MAX_CONTEXTS]){
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

ncclResult_t ncclRmaProxyDeregister(struct ncclComm* comm, void* rmaHostWins[NCCL_GIN_MAX_CONTEXTS]){
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  for (int n = 0; n < rmaProxyState->ginCommCount; n++) {
    NCCLCHECK(rmaProxyState->ncclGin->deregMrSym(rmaProxyState->ginComms[n], rmaHostWins[n]));
  }
  return ncclSuccess;
}

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
          __atomic_store_n(&rmaProxyState->asyncResult, ret, __ATOMIC_RELEASE);
          INFO(NCCL_ALL,"%s:%d -> %d [RMA Proxy Progress Thread]", __FILE__, __LINE__, ret);
          rmaProxyState->ginProgress = -2;
          return NULL;
        }
      }
      std::this_thread::yield();
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

  NCCLCHECK(rmaProxyState->ncclGin->init(&rmaProxyState->ginInstance, comm->commHash, ncclDebugLog));

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
  int64_t localNets[NCCL_TOPO_MAX_NODES];
  NCCLCHECK(ncclTopoGetLocalNets(comm->topo, comm->rank, localNets, &rmaProxyState->ginCommCount));
  ginCommCount = std::min<int>(rmaProxyState->ginCommCount, NCCL_GIN_MAX_CONTEXTS);
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
      rmaProxyState->ncclGin->listen(rmaProxyState->ginInstance, localNets[n],
                                allHandles + NCCL_NET_HANDLE_MAXSIZE * comm->rank, &listenComm),
      ret, fail);
    NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allHandles, NCCL_NET_HANDLE_MAXSIZE), ret,
                  fail);
    NCCLCHECKGOTO(rmaProxyState->ncclGin->connect(comm->netContext, handles, comm->nRanks, comm->rank,
                                             listenComm, rmaProxyState->ginComms + n),
                  ret, fail);
    NCCLCHECKGOTO(rmaProxyState->ncclGin->getProperties(localNets[n], &rmaProxyState->props[n]), ret, fail);
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

  // Finalize the GIN instance
  NCCLCHECK(rmaProxyState->ncclGin->finalize(rmaProxyState->ginInstance));
  memset((void*)rmaProxyState, 0, sizeof(*rmaProxyState));
  return ncclSuccess;
}

ncclResult_t ncclRmaPutProxy(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA proxy is connected
  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  int ctx = plan->rmaArgs->ctx;
  int nRmaTasksProxy = plan->rmaArgs->nRmaTasksProxy;
  struct ncclRmaProxyCtx * rmaProxyCtx = (struct ncclRmaProxyCtx *)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  // Allocate 2*nRmaTasksProxy CUstreamBatchMemOpParams
  CUstreamBatchMemOpParams* batchParams = NULL;
  NCCLCHECK(ncclCalloc(&batchParams, 2*nRmaTasksProxy));

  int batchIdx = 0;

  for (int i = 0; i < nRmaTasksProxy; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
    int peer = task->peer;

    // Check for available slot in the circular buffer
    uint32_t pi = __atomic_load_n(&rmaProxyCtx->pis[peer], __ATOMIC_RELAXED);
    uint32_t ci = __atomic_load_n(&rmaProxyCtx->cis[peer], __ATOMIC_ACQUIRE);

    // If queue is full, flush pending batch ops to allow progress thread to free slots
    while ((pi - ci) >= rmaProxyCtx->queueSize) {
      if (batchIdx > 0) {
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams), ret, fail);
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams+nRmaTasksProxy), ret, fail);
        batchIdx = 0;
      }
      // Yield to allow progress thread to run and process pending entries
      std::this_thread::yield();
      // Re-read both PI and CI to get fresh values
      pi = __atomic_load_n(&rmaProxyCtx->pis[peer], __ATOMIC_RELAXED);
      ci = __atomic_load_n(&rmaProxyCtx->cis[peer], __ATOMIC_ACQUIRE);
    }

    ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);

    assert(task->ctx == ctx);

    struct ncclRmaProxyDesc *desc = NULL;
    NCCLCHECK(ncclCalloc(&desc, 1));
    desc->srcOff = task->srcWinOffset;
    desc->srcHandle = ncclDevrGetRmaDevWin(task->srcWinHost, ctx);
    desc->dstOff = task->peerWinOffset;
    desc->dstHandle = ncclDevrGetRmaDevWin(task->peerWinHost, ctx);
    desc->size = task->count * ncclTypeSize(task->datatype);
    desc->targetRank = task->peer;
    desc->seq = rmaProxyCtx->opSeqs[task->peer]++;
    desc->rmaDescState = ncclRmaDescStatePending;
    desc->request = NULL;

    // If the signal mode is none, we do not need to set the signal operation
    if (task->signalMode == NCCL_SIGNAL_NONE) {
      desc->signal.op = 0;
    }
    // If the signal mode is NCCL_SIGNAL, we use the per-rank signal for the target rank
    else if (task->signalMode == NCCL_SIGNAL) {
      desc->signal.op = NCCL_NET_SIGNAL_OP_ADD;
      desc->signal.offset = comm->rank * sizeof(uint64_t); // Write to our rank slot in peer's buffer
      desc->signal.signalMhandle = rmaProxyCtx->signalsMhandle;
      desc->signal.val = 1;
    }

    // Prepare the readySeq write operation
    batchParams[batchIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    batchParams[batchIdx].writeValue.address = (CUdeviceptr)&rmaProxyCtx->readySeqsDev[task->peer];
    batchParams[batchIdx].writeValue.value = desc->seq;
    batchParams[batchIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;

    // Prepare the doneSeq wait operation
    batchParams[batchIdx+nRmaTasksProxy].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    batchParams[batchIdx+nRmaTasksProxy].waitValue.address = (CUdeviceptr)&rmaProxyCtx->doneSeqsDev[task->peer];
    batchParams[batchIdx+nRmaTasksProxy].waitValue.value = desc->seq;
    batchParams[batchIdx+nRmaTasksProxy].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;

    INFO(NCCL_COLL, "ncclRmaPutProxy enqueued Desc: rank=%d peer=%d ctx=%d size=%ld signalMode=%d readySeq=%lu doneSeq=%lu",
      comm->rank, task->peer, ctx, task->count * ncclTypeSize(task->datatype), task->signalMode, (uint64_t)desc->seq, (uint64_t)desc->seq);

    // Write descriptor to queue
    uint32_t idx = pi & (rmaProxyCtx->queueSize - 1);
    rmaProxyCtx->pendingQueues[peer * rmaProxyCtx->queueSize + idx] = desc;

    // Advance PI with RELEASE to ensure descriptor write is visible
    __atomic_store_n(&rmaProxyCtx->pis[peer], pi + 1, __ATOMIC_RELEASE);
    batchIdx++;

    // Free the task
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

  // Execute ready operations (readySeq writes) first, then done operations (doneSeq waits)
  if (batchIdx == nRmaTasksProxy) {
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, 2*batchIdx, batchParams), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams+nRmaTasksProxy), ret, fail);
  }

exit:
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}



ncclResult_t ncclRmaWaitSignalProxy(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA proxy is connected
  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaProxyCtx* proxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);

  // Assert task func is ncclFuncWaitSignal
  assert(task->func == ncclFuncWaitSignal);
  // Assert task context is the same as the plan context
  assert(task->ctx == ctx);
  // Assert the plan has exactly one RMA proxy task
  assert(plan->rmaArgs->nRmaTasksProxy == 1);

  size_t opIdx = 0;
  CUstreamBatchMemOpParams* batchParams = nullptr;

  NCCLCHECK(ncclCalloc(&batchParams, task->npeers));

  // Use per-rank signal for the target rank
  if (task->signalMode == NCCL_SIGNAL) {
    for (int i = 0; i < task->npeers; i++) {
      int peerRank = task->peers[i];
      // Calculate the expected signal value from this peer
      uint64_t waitValue = proxyCtx->signalsHost[peerRank] + task->nsignals[i];

      // Update our expectation for future waits
      proxyCtx->signalsHost[peerRank] = waitValue;

      // Add wait operation to batch
      batchParams[opIdx] = {};
      batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      batchParams[opIdx].waitValue.address = (CUdeviceptr)&proxyCtx->signalsDev[peerRank];
      batchParams[opIdx].waitValue.value64 = waitValue;
      batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
      opIdx++;
    }

    // Execute all wait operations in a single batch
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);
  }

  // Free the task
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);

exit:
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}
