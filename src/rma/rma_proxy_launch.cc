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
#include "compiler.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"
#include "dev_runtime.h"

// ---- Descriptor build ----

static ncclResult_t ncclRmaProxyPutDescFromTask(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                          struct ncclKernelPlan* plan, struct ncclTaskRma* task, struct ncclRmaProxyDesc* desc)
{
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;

  desc->rmaDescType = ncclRmaDescTypePutSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->putSignal.srcOff = task->srcWinOffset;
  desc->putSignal.srcHandle = ncclDevrGetRmaDevWin(task->srcWinHost, task->ctx);
  desc->putSignal.dstOff = task->peerWinOffset;
  desc->putSignal.dstHandle = ncclDevrGetRmaDevWin(task->peerWinHost, task->ctx);
  desc->putSignal.size = task->count * ncclTypeSize(task->datatype);
  desc->putSignal.targetRank = task->peer;
  desc->putSignal.request = NULL;
  if (!persistent) {
    desc->opSeq = rmaProxyCtx->opSeqs[task->peer]++;
    desc->readySeq = &rmaProxyCtx->readySeqs[task->peer];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[task->peer];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[task->peer];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[task->peer];
    desc->doneSeqGdrHandle = rmaProxyCtx->doneSeqsGdrHandle;
  }
  else {
    desc->opSeq = 1;
    // Allocation during graph capture, off the execution critical path
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle, comm->memManager), ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle, comm->memManager), ret, fail);
    desc->persistPlan = plan;
  }

  if (task->signalMode == NCCL_SIGNAL_NONE) {
    desc->putSignal.signal.op = 0;
  }
  else if (task->signalMode == NCCL_SIGNAL) {
    desc->putSignal.signal.op = NCCL_NET_SIGNAL_OP_ADD;
    desc->putSignal.signal.offset = comm->rank * sizeof(uint64_t);
    desc->putSignal.signal.signalMhandle = persistent ? rmaProxyCtx->cpuAccessSignalsMhandle : rmaProxyCtx->signalsMhandle;
    desc->putSignal.signal.val = 1;
  }
exit:
  return ret;
fail:
  if (desc->readySeq)
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
  goto exit;
}

static ncclResult_t ncclRmaProxyWaitDescFromTask(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                          struct ncclKernelPlan* plan, struct ncclTaskRma* task, struct ncclRmaProxyDesc* desc)
{
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;

  desc->rmaDescType = ncclRmaDescTypeWaitSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->waitSignal.npeers = task->npeers;
  if (!persistent) {
    desc->waitSignal.waitPeers = task->peers;
    desc->waitSignal.waitSignals = task->nsignals;
  }
  else {
    desc->opSeq = 1;
    desc->waitSignal.waitPeers = task->peers;
    desc->waitSignal.waitSignals = task->nsignals;
    // If GDR is not used, we need to flush the NIC-GPU path to ensure the data is visible in vidmem
    desc->waitSignal.needFlush = (rmaProxyCtx->cpuAccessSignalsGdrHandle == NULL);
    task->peers = nullptr;
    task->nsignals = nullptr;
    // Allocation during graph capture, off the execution critical path
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle, comm->memManager), ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle, comm->memManager), ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq)
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
  goto exit;
}

// ---- Descriptor destroy ----

ncclResult_t ncclRmaProxyDestroyDescNonPersistent(struct ncclRmaProxyDesc* desc) {
  free(desc);
  return ncclSuccess;
}

ncclResult_t ncclRmaProxyDestroyDescPersistent(struct ncclComm* comm, struct ncclRmaProxyDesc* desc) {
  if (desc->readySeqGdrHandle || desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
  }
  if (desc->doneSeqGdrHandle || desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
  }
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    free(desc->waitSignal.waitPeers);
    free(desc->waitSignal.waitSignals);
  }
  free(desc);
  return ncclSuccess;
}

// ---- Queue management ----

bool ncclRmaProxyCircularBufFull(struct ncclRmaProxyCtx* ctx, int peer) {
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_relaxed);
  uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_acquire);
  return (pi - ci) >= ctx->queueSize;
}

bool ncclRmaProxyCircularBufEmpty(struct ncclRmaProxyCtx* ctx, int peer) {
  uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_acquire);
  return ci >= pi;
}

static inline ncclResult_t ncclRmaProxyEnqueueNonPersistentDesc(
    struct ncclRmaProxyCtx* ctx, int peer, struct ncclRmaProxyDesc* desc) {
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_relaxed);
  uint32_t idx = pi & (ctx->queueSize - 1);
  ctx->circularBuffers[peer * ctx->queueSize + idx] = desc;
  // Advance PI with RELEASE to ensure descriptor write is visible
  COMPILER_ATOMIC_STORE_32(&ctx->pis[peer], pi + 1, std::memory_order_release);
  return ncclSuccess;
}

static inline ncclResult_t ncclRmaProxyEnqueuePersistentDesc(
    struct ncclRmaProxyCtx* ctx, int peer, struct ncclRmaProxyDesc* desc) {
  ncclIntruQueueEnqueue(&ctx->persistentQueues[peer], desc);
  // Transition to persistent with RELEASE to guarantee all preceding writes are visible to proxy
  COMPILER_ATOMIC_STORE(&desc->persistDescValid, true, std::memory_order_release);
  return ncclSuccess;
}

// ---- Launch-time API ----

ncclResult_t ncclRmaProxyPutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int ctx = plan->rmaArgs->ctx;
  int nRmaTasksProxy = plan->rmaArgs->nRmaTasksProxy;
  struct ncclRmaProxyCtx * rmaProxyCtx = (struct ncclRmaProxyCtx *)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  int opsPerTask = persistent ? 3 : 2;
  struct ncclRmaProxyDesc *desc = nullptr;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECK(ncclCalloc(&batchParams, opsPerTask * nRmaTasksProxy));

  int batchIdx = 0;

  for (int i = 0; i < nRmaTasksProxy; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
    int peer = task->peer;

    // Non-graph: wait for circular buffer slot, flushing batch if needed
    if (!persistent) {
      while (ncclRmaProxyCircularBufFull(rmaProxyCtx, peer)) {
        if (batchIdx > 0) {
          NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams), ret, fail);
          NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams + nRmaTasksProxy), ret, fail);
          batchIdx = 0;
        }
        std::this_thread::yield();
      }
    }
    ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);
    assert(task->ctx == ctx);

    NCCLCHECKGOTO(ncclCalloc(&desc, 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutDescFromTask(comm, rmaProxyCtx, plan, task, desc), ret, fail);

    // Prepare the readySeq write operation
    batchParams[batchIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    batchParams[batchIdx].writeValue.address = (CUdeviceptr)desc->readySeqDev;
    batchParams[batchIdx].writeValue.value = desc->opSeq;
    batchParams[batchIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;

    // Prepare the doneSeq wait operation
    batchParams[batchIdx+nRmaTasksProxy].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    batchParams[batchIdx+nRmaTasksProxy].waitValue.address = (CUdeviceptr)desc->doneSeqDev;
    batchParams[batchIdx+nRmaTasksProxy].waitValue.value = desc->opSeq;
    batchParams[batchIdx+nRmaTasksProxy].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;

    // Graph: extra reset op for doneSeq
    if (persistent) {
      batchParams[batchIdx + 2 * nRmaTasksProxy].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
      batchParams[batchIdx + 2 * nRmaTasksProxy].writeValue.address = (CUdeviceptr)desc->doneSeqDev;
      batchParams[batchIdx + 2 * nRmaTasksProxy].writeValue.value = 0;
      batchParams[batchIdx + 2 * nRmaTasksProxy].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
    }

    INFO(NCCL_COLL, "ncclRmaProxyPutLaunch enqueued Desc: rank=%d peer=%d ctx=%d size=%ld signalMode=%d readySeq=%lu doneSeq=%lu persistent=%d",
      comm->rank, task->peer, ctx, task->count * ncclTypeSize(task->datatype), task->signalMode, (uint64_t)desc->opSeq, (uint64_t)desc->opSeq, persistent);

    // Enqueue descriptor to appropriate queue
    if (!persistent) {
      NCCLCHECKGOTO(ncclRmaProxyEnqueueNonPersistentDesc(rmaProxyCtx, peer, desc), ret, fail);
    } else {
      NCCLCHECKGOTO(ncclRmaProxyEnqueuePersistentDesc(rmaProxyCtx, peer, desc), ret, fail);
    }
    desc = nullptr;

    batchIdx++;

    // Free the task
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

  // Execute batch
  if (batchIdx == nRmaTasksProxy) {
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opsPerTask*nRmaTasksProxy, batchParams), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, batchIdx, batchParams+nRmaTasksProxy), ret, fail);
  }

exit:
  free(batchParams);
  return ret;
fail:
  free(desc);
  goto exit;
}



ncclResult_t ncclRmaProxyWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);

  assert(task->func == ncclFuncWaitSignal);
  assert(task->ctx == ctx);
  assert(plan->rmaArgs->nRmaTasksProxy == 1);

  size_t opIdx = 0;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  struct ncclRmaProxyDesc* desc = nullptr;

  if (task->signalMode == NCCL_SIGNAL) {
    if (!persistent) {
      // Non-graph: direct cuStreamWaitValue on vidmem signals
      NCCLCHECKGOTO(ncclCalloc(&batchParams, task->npeers), ret, fail);
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        uint64_t waitValue = rmaProxyCtx->signalsHost[peerRank] + task->nsignals[i];
        rmaProxyCtx->signalsHost[peerRank] = waitValue;

        batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        batchParams[opIdx].waitValue.address = (CUdeviceptr)&rmaProxyCtx->signalsDev[peerRank];
        batchParams[opIdx].waitValue.value64 = waitValue;
        batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
        opIdx++;
      }
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);
    }
    else {
      // Graph: create persistent desc, proxy polls CPU-accessible signals
      NCCLCHECKGOTO(ncclCalloc(&batchParams, 3), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&desc, 1), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyWaitDescFromTask(comm, rmaProxyCtx, plan, task, desc), ret, fail);

      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
      batchParams[opIdx].writeValue.address = (CUdeviceptr)desc->readySeqDev;
      batchParams[opIdx].writeValue.value = 1;
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;

      batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      batchParams[opIdx].waitValue.address = (CUdeviceptr)desc->doneSeqDev;
      batchParams[opIdx].waitValue.value = 1;
      batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
      opIdx++;

      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
      batchParams[opIdx].writeValue.address = (CUdeviceptr)desc->doneSeqDev;
      batchParams[opIdx].writeValue.value = 0;
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;

      // Enqueue to own rank's persistent queue — ownership of desc transfers to the queue
      NCCLCHECKGOTO(ncclRmaProxyEnqueuePersistentDesc(rmaProxyCtx, comm->rank, desc), ret, fail);
      desc = nullptr;

      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);
    }
  }

exit:
  free(task->peers);
  free(task->nsignals);
  free(batchParams);
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  return ret;
fail:
  free(desc);
  goto exit;
}

// ---- Graph reclaim ----

// Free persistent descs belonging to the given plan.
// ncclRmaProxyProgress is paused when this function is called, so no concurrent access to persistentQueues.
static ncclResult_t ncclRmaProxyReclaimPersistDescs(struct ncclRmaProxyState* proxyState, struct ncclKernelPlan* plan) {
  struct ncclComm* comm = proxyState->comm;

  for (int c = 0; c < proxyState->rmaProxyCtxCount; c++) {
    struct ncclRmaProxyCtx* proxyCtx = (struct ncclRmaProxyCtx*)proxyState->rmaProxyCtxs[c];
    if (proxyCtx == nullptr) continue;

    for (int peer = 0; peer < proxyCtx->comm->nRanks; peer++) {
      struct ncclRmaProxyDesc* prev = nullptr;
      struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&proxyCtx->persistentQueues[peer]);
      while (desc != nullptr) {
        struct ncclRmaProxyDesc* next = desc->next;
        if (desc->persistPlan == plan) {
          if (prev == nullptr) {
            proxyCtx->persistentQueues[peer].head = next;
            if (next == nullptr) proxyCtx->persistentQueues[peer].tail = nullptr;
          } else {
            prev->next = next;
            if (next == nullptr) proxyCtx->persistentQueues[peer].tail = prev;
          }

          NCCLCHECK(ncclRmaProxyDestroyDescPersistent(comm, desc));
        } else {
          prev = desc;
        }
        desc = next;
      }
    }
  }

  return ncclSuccess;
}

// Destroy persistent descs belonging to the plan.
// Called from reclaimPlan when the CUDA graph is destroyed.
// Pauses the proxy thread, frees descs on the main thread, then resumes the proxy.
ncclResult_t ncclRmaProxyReclaimPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclRmaProxyState* proxyState = &comm->rmaState.rmaProxyState;
  if (!proxyState->connected || !proxyState->needsProxyProgress) return ncclSuccess;

  // Step 1: Request proxy to pause and wait for acknowledgment
  {
    std::unique_lock<std::mutex> lock(proxyState->mutex);
    proxyState->ginProgress = 2;
    proxyState->cond.notify_one();
    proxyState->cond.wait(lock, [&]{ return proxyState->ginProgress == 0; });
  }

  // Step 2: Free persistent descs on main thread
  NCCLCHECK(ncclRmaProxyReclaimPersistDescs(proxyState, plan));

  // Step 3: Resume proxy
  {
    std::lock_guard<std::mutex> lock(proxyState->mutex);
    proxyState->ginProgress = 1;
    proxyState->cond.notify_one();
  }

  return ncclSuccess;
}
