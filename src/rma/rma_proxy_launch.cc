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

// ============================================================================
// Descriptor build
// ============================================================================

// Helper function to build a single put-signal op.
// Used by ncclRmaProxyPutBuildDesc and ncclRmaProxyPutGroupBuildDesc.
ncclResult_t ncclRmaProxyPutBuildOp(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx, int ctx,
                                    bool persistent, struct ncclDevrWindow* srcWin, size_t srcOff,
                                    struct ncclDevrWindow* peerWin, size_t peerOff, size_t size, int peer,
                                    ncclSignalMode_t signalMode, struct ncclRmaPutSignalOp* op) {
  op->srcOff = ncclDevrGetWinOffset(srcWin) + srcOff;
  op->srcHandle = ncclDevrGetRmaWin(srcWin, ctx);
  op->dstOff = ncclDevrGetWinOffset(peerWin) + peerOff;
  op->dstHandle = ncclDevrGetRmaWin(peerWin, ctx);
  op->size = size;
  op->targetRank = peer;
  op->request = nullptr;

  if (signalMode == NCCL_SIGNAL_NONE) {
    op->signal.op = 0;
  } else if (signalMode == NCCL_SIGNAL) {
    op->signal.op = NCCL_NET_SIGNAL_OP_ADD;
    op->signal.offset = comm->rank * sizeof(uint64_t);
    op->signal.signalMhandle = persistent ? rmaProxyCtx->cpuAccessSignalsMhandle : rmaProxyCtx->signalsMhandle;
    op->signal.val = 1;
  }
  return ncclSuccess;
}

// Build the put descriptor
ncclResult_t ncclRmaProxyPutBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                      struct ncclKernelPlan* plan, struct ncclDevrWindow* srcWinHost,
                                      size_t srcWinOffset, struct ncclDevrWindow* peerWinHost, size_t peerWinOffset,
                                      size_t size, int peer, int ctx, ncclSignalMode_t signalMode,
                                      struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;

  desc->rmaDescType = ncclRmaDescTypePutSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;

  // Inner-struct fields shared with the group builder.
  NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent, srcWinHost, srcWinOffset, peerWinHost,
                                       peerWinOffset, size, peer, signalMode, &desc->putSignal),
                ret, fail);

  // Desc-level seq + persist state: per-peer slot for single puts.
  if (!persistent) {
    desc->opSeq = ++rmaProxyCtx->opSeqs[peer];
    desc->readySeq = &rmaProxyCtx->readySeqs[peer];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[peer];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[peer];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[peer];
    desc->doneSeqGdrHandle = rmaProxyCtx->doneSeqsGdrHandle;
  } else {
    desc->opSeq = 1;
    // Allocation during graph capture, off the execution critical path
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

// Build a put-signal-group descriptor over a caller-supplied array of
// pre-filled ops (can be populated via ncclRmaProxyPutBuildOp).
ncclResult_t ncclRmaProxyPutGroupBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                           struct ncclKernelPlan* plan, int nOps, struct ncclRmaPutSignalOp** ops,
                                           int ctx, struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;
  int slot = comm->rank;

  desc->rmaDescType = ncclRmaDescTypePutSignalGroup;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;

  // Take ownership of the caller's pre-filled ops array.
  desc->putSignalGroup.nOps = nOps;
  desc->putSignalGroup.ops = *ops;
  desc->putSignalGroup.nIssued = 0;
  desc->putSignalGroup.nCompleted = 0;
  *ops = nullptr;

  // Desc-level seq + persist state: same shape as single-put, slot=rank.
  if (!persistent) {
    desc->opSeq = ++rmaProxyCtx->opSeqs[slot];
    desc->readySeq = &rmaProxyCtx->readySeqs[slot];
    desc->readySeqDev = &rmaProxyCtx->readySeqsDev[slot];
    desc->readySeqGdrHandle = rmaProxyCtx->readySeqsGdrHandle;
    desc->doneSeq = &rmaProxyCtx->doneSeqs[slot];
    desc->doneSeqDev = &rmaProxyCtx->doneSeqsDev[slot];
    desc->doneSeqGdrHandle = rmaProxyCtx->doneSeqsGdrHandle;
  } else {
    desc->opSeq = 1;
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

static ncclResult_t ncclRmaProxyPutDescFromTask(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                struct ncclKernelPlan* plan, struct ncclTaskRma* task,
                                                struct ncclRmaProxyDesc* desc) {
  return ncclRmaProxyPutBuildDesc(comm, rmaProxyCtx, plan, task->srcWinHost, task->srcWinOffset, task->peerWinHost,
                                  task->peerWinOffset, task->count * ncclTypeSize(task->datatype), task->peer,
                                  task->ctx, task->signalMode, desc);
}

// Build a wait-signal descriptor.
ncclResult_t ncclRmaProxyWaitBuildDesc(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                       struct ncclKernelPlan* plan, int npeers, int** peers, int** nsignals,
                                       struct ncclRmaProxyDesc* desc) {
  ncclResult_t ret = ncclSuccess;
  bool persistent = plan->persistent;

  desc->rmaDescType = ncclRmaDescTypeWaitSignal;
  desc->rmaDescState = ncclRmaDescStateReady;
  desc->waitSignal.npeers = npeers;
  // Transfer ownership: desc takes the arrays, caller's locals are nulled.
  desc->waitSignal.waitPeers = *peers;
  desc->waitSignal.waitSignals = *nsignals;
  *peers = nullptr;
  *nsignals = nullptr;
  desc->persistPlan = nullptr;
  desc->persistDescValid = false;
  if (persistent) {
    desc->opSeq = 1;
    // If GDR is not used, we need to flush the NIC-GPU path to ensure the data is visible in vidmem
    desc->waitSignal.needFlush = (rmaProxyCtx->cpuAccessSignalsGdrHandle == NULL);
    // Allocation during graph capture, off the execution critical path
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->readySeq, &desc->readySeqDev, 1, 0, &desc->readySeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    NCCLCHECKGOTO(allocMemCPUAccessible(&desc->doneSeq, &desc->doneSeqDev, 1, 0, &desc->doneSeqGdrHandle,
                                        comm->memManager),
                  ret, fail);
    desc->persistPlan = plan;
  }
exit:
  return ret;
fail:
  if (desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
    desc->readySeq = nullptr;
    desc->readySeqDev = nullptr;
    desc->readySeqGdrHandle = nullptr;
  }
  if (desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
    desc->doneSeq = nullptr;
    desc->doneSeqDev = nullptr;
    desc->doneSeqGdrHandle = nullptr;
  }
  goto exit;
}

// Build the wait descriptor from an ncclTaskRma
static ncclResult_t ncclRmaProxyWaitDescFromTask(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                 struct ncclKernelPlan* plan, struct ncclTaskRma* task,
                                                 struct ncclRmaProxyDesc* desc) {
  return ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, task->npeers, &task->peers, &task->nsignals, desc);
}

// ============================================================================
// Descriptor destroy
// ============================================================================

static ncclResult_t ncclRmaProxyDestroyDescNonPersistent(struct ncclRmaProxyDesc* desc) {
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    free(desc->waitSignal.waitPeers);
    free(desc->waitSignal.waitSignals);
  } else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
    free(desc->putSignalGroup.ops);
  }
  free(desc);
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyDestroyDescPersistent(struct ncclComm* comm, struct ncclRmaProxyDesc* desc) {
  if (desc->readySeqGdrHandle || desc->readySeq) {
    freeMemCPUAccessible(desc->readySeq, desc->readySeqGdrHandle, comm->memManager);
  }
  if (desc->doneSeqGdrHandle || desc->doneSeq) {
    freeMemCPUAccessible(desc->doneSeq, desc->doneSeqGdrHandle, comm->memManager);
  }
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    free(desc->waitSignal.waitPeers);
    free(desc->waitSignal.waitSignals);
  } else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
    free(desc->putSignalGroup.ops);
  }
  free(desc);
  return ncclSuccess;
}

// Destroy a descriptor.
ncclResult_t ncclRmaProxyDestroyDesc(struct ncclComm* comm, struct ncclRmaProxyDesc** desc) {
  ncclResult_t ret;
  if ((*desc)->persistPlan == nullptr) {
    ret = ncclRmaProxyDestroyDescNonPersistent(*desc);
  } else {
    ret = ncclRmaProxyDestroyDescPersistent(comm, *desc);
  }
  *desc = nullptr;
  return ret;
}

// ============================================================================
// Queue management
// ============================================================================

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

// Returns true if the queue this descriptor would enqueue into is full.
// Peer / persistence / type are all derived from the descriptor.
//
//   PutSignal,      non-persistent  : pis[targetRank] circular buffer full
//   PutSignalGroup, non-persistent  : pis[comm->rank] circular buffer full
//   *,              persistent      : false  (linked list, unbounded)
//   WaitSignal,     non-persistent  : false  (no queue exists)
bool ncclRmaProxyEnqueueFull(struct ncclRmaProxyCtx* ctx, const struct ncclRmaProxyDesc* desc) {
  // Persistent queues are unbounded.
  if (desc->persistPlan != nullptr) return false;
  // Non-persistent wait has no queue.
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) return false;
  // Non-persistent put or put-group: derive peer from the desc.
  int peer;
  switch (desc->rmaDescType) {
  case ncclRmaDescTypePutSignal:
    peer = desc->putSignal.targetRank;
    break;
  case ncclRmaDescTypePutSignalGroup:
    peer = ctx->comm->rank;
    break;
  default:
    WARN("ncclRmaProxyEnqueueFull: unknown desc type %d", desc->rmaDescType);
    return false;
  }
  return ncclRmaProxyCircularBufFull(ctx, peer);
}

static inline ncclResult_t ncclRmaProxyEnqueueNonPersistentDesc(struct ncclRmaProxyCtx* ctx, int peer,
                                                                struct ncclRmaProxyDesc* desc) {
  if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
    // Destroy the scratch desc for wait non-persistent
    return ncclRmaProxyDestroyDesc(ctx->comm, &desc);
  }
  // Non-persistent puts use a bounded circular buffer; caller is required
  // to have checked ncclRmaProxyEnqueueFull. Catch silent overflow in debug.
  assert(!ncclRmaProxyCircularBufFull(ctx, peer));
  uint32_t pi = COMPILER_ATOMIC_LOAD_32(&ctx->pis[peer], std::memory_order_relaxed);
  uint32_t idx = pi & (ctx->queueSize - 1);
  ctx->circularBuffers[peer * ctx->queueSize + idx] = desc;
  // Advance PI with RELEASE to ensure descriptor write is visible
  COMPILER_ATOMIC_STORE_32(&ctx->pis[peer], pi + 1, std::memory_order_release);
  return ncclSuccess;
}

static inline ncclResult_t ncclRmaProxyEnqueuePersistentDesc(struct ncclRmaProxyCtx* ctx, int peer,
                                                             struct ncclRmaProxyDesc* desc) {
  ncclIntruQueueEnqueue(&ctx->persistentQueues[peer], desc);
  // Transition to persistent with RELEASE to guarantee all preceding writes are visible to proxy
  COMPILER_ATOMIC_STORE(&desc->persistDescValid, true, std::memory_order_release);
  return ncclSuccess;
}

// Enqueue a built descriptor onto its target queue.
//
//   (type, persistent)             | action
//   -------------------------------+----------------------------------------
//   PutSignal,      non-persist    | push to pis[targetRank] circular buffer
//                                  | (proxy picks up and issues iput; proxy
//                                  |  frees the desc on completion)
//   PutSignal,      persistent     | append to persistentQueues[targetRank]
//                                  | (proxy traverses on each graph replay;
//                                  |  freed at graph reclaim)
//   PutSignalGroup, non-persist    | push to pis[comm->rank] circular buffer
//                                  | (rides the to-self slot; proxy issues
//                                  |  all nOps iputs in parallel and frees
//                                  |  the desc once all issued ops complete)
//   PutSignalGroup, persistent     | append to persistentQueues[comm->rank]
//                                  | (proxy issues all nOps iputs on each
//                                  |  graph replay; freed at graph reclaim)
//   WaitSignal,     persistent     | append to persistentQueues[comm->rank]
//                                  | (proxy polls signals on each replay;
//                                  |  freed at graph reclaim)
//   WaitSignal,     non-persist    | DESTROY -- the desc is a scratch object
//                                  | whose only purpose was to carry fields
//                                  | into ncclRmaProxyWaitParams. Destroyed
//                                  | in place.
ncclResult_t ncclRmaProxyEnqueueDesc(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc** desc) {
  int peer;
  switch ((*desc)->rmaDescType) {
  case ncclRmaDescTypePutSignal:
    peer = (*desc)->putSignal.targetRank;
    break;
  case ncclRmaDescTypeWaitSignal:
    peer = rmaProxyCtx->comm->rank;
    break;
  case ncclRmaDescTypePutSignalGroup:
    peer = rmaProxyCtx->comm->rank;
    break;
  default:
    WARN("ncclRmaProxyEnqueueDesc: unknown desc type %d", (*desc)->rmaDescType);
    return ncclInternalError;
  }

  // Check if the target queue is full and yield if it is.
  while (ncclRmaProxyEnqueueFull(rmaProxyCtx, *desc)) {
    std::this_thread::yield();
  }

  bool persistent = ((*desc)->persistPlan != nullptr);
  ncclResult_t ret;
  if (!persistent) {
    ret = ncclRmaProxyEnqueueNonPersistentDesc(rmaProxyCtx, peer, *desc);
  } else {
    ret = ncclRmaProxyEnqueuePersistentDesc(rmaProxyCtx, peer, *desc);
  }
  *desc = nullptr;
  return ret;
}

// ============================================================================
// Prep params
// ============================================================================

// Returns the number of stream-batch memops ncclRmaProxyPutStartParams emits.
int ncclRmaProxyPutStartNumOps(bool persistent) {
  return 1;
}

// Fill the stream-batch memop param that triggers the proxy to launch the descriptor
ncclResult_t ncclRmaProxyPutStartParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignal) {
    WARN("ncclRmaProxyPutStartParams: descriptor is not a put-signal type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }

  params[0].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
  params[0].writeValue.address = (CUdeviceptr)desc->readySeqDev;
  params[0].writeValue.value = desc->opSeq;
  params[0].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  return ncclSuccess;
}

// Returns the number of stream-batch memops ncclRmaProxyPutDoneParams emits
int ncclRmaProxyPutDoneNumOps(bool persistent) {
  return persistent ? 2 : 1;
}

// Fill stream-batch memop params for the proxy completion fence
ncclResult_t ncclRmaProxyPutDoneParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignal) {
    WARN("ncclRmaProxyPutDoneParams: descriptor is not a put-signal type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }

  params[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
  params[0].waitValue.address = (CUdeviceptr)desc->doneSeqDev;
  params[0].waitValue.value = desc->opSeq;
  params[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;

  bool persistent = (desc->persistPlan != nullptr);
  if (persistent) {
    params[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    params[1].writeValue.address = (CUdeviceptr)desc->doneSeqDev;
    params[1].writeValue.value = 0;
    params[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  }
  return ncclSuccess;
}

// Returns the number of stream-batch memops ncclRmaProxyPutGroupStartParams emits.
int ncclRmaProxyPutGroupStartNumOps(bool persistent) {
  return 1;
}

// Fill the stream-batch memop param that triggers the proxy to launch the group.
// Bytes-equivalent to ncclRmaProxyPutStartParams, with the type guard targeting
// PutSignalGroup. The group has one shared (readySeq, opSeq) regardless of nOps.
ncclResult_t ncclRmaProxyPutGroupStartParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignalGroup) {
    WARN("ncclRmaProxyPutGroupStartParams: descriptor is not a put-signal-group type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }

  params[0].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
  params[0].writeValue.address = (CUdeviceptr)desc->readySeqDev;
  params[0].writeValue.value = desc->opSeq;
  params[0].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  return ncclSuccess;
}

// Returns the number of stream-batch memops ncclRmaProxyPutGroupDoneParams emits.
int ncclRmaProxyPutGroupDoneNumOps(bool persistent) {
  return persistent ? 2 : 1;
}

// Fill stream-batch memop params for the group's completion fence.
// Bytes-equivalent to ncclRmaProxyPutDoneParams, with the type guard targeting
// PutSignalGroup. The wait + (optional) reset target the group's shared doneSeq.
ncclResult_t ncclRmaProxyPutGroupDoneParams(struct ncclRmaProxyDesc* desc, CUstreamBatchMemOpParams* params) {
  if (desc->rmaDescType != ncclRmaDescTypePutSignalGroup) {
    WARN("ncclRmaProxyPutGroupDoneParams: descriptor is not a put-signal-group type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }

  params[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
  params[0].waitValue.address = (CUdeviceptr)desc->doneSeqDev;
  params[0].waitValue.value = desc->opSeq;
  params[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;

  bool persistent = (desc->persistPlan != nullptr);
  if (persistent) {
    params[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    params[1].writeValue.address = (CUdeviceptr)desc->doneSeqDev;
    params[1].writeValue.value = 0;
    params[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  }
  return ncclSuccess;
}

// Returns the number of stream-batch memops waitSignal will emit for the descriptor
int ncclRmaProxyWaitNumStreamOps(const struct ncclRmaProxyDesc* desc) {
  bool persistent = (desc->persistPlan != nullptr);
  return persistent ? 3 : desc->waitSignal.npeers;
}

// Fill stream-batch memop params for a wait-signal descriptor.
ncclResult_t ncclRmaProxyWaitParams(struct ncclRmaProxyCtx* rmaProxyCtx, struct ncclRmaProxyDesc* desc,
                                    CUstreamBatchMemOpParams* params) {
  if (desc->rmaDescType != ncclRmaDescTypeWaitSignal) {
    WARN("ncclRmaProxyWaitParams: descriptor is not a wait-signal type (%d)", desc->rmaDescType);
    return ncclInternalError;
  }

  bool persistent = (desc->persistPlan != nullptr);
  if (!persistent) {
    int npeers = desc->waitSignal.npeers;
    int* peers = desc->waitSignal.waitPeers;
    int* nsignals = desc->waitSignal.waitSignals;
    for (int i = 0; i < npeers; i++) {
      int peer = peers[i];
      uint64_t waitValue = rmaProxyCtx->signalsHost[peer] + nsignals[i];
      rmaProxyCtx->signalsHost[peer] = waitValue;

      params[i].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      params[i].waitValue.address = (CUdeviceptr)&rmaProxyCtx->signalsDev[peer];
      params[i].waitValue.value64 = waitValue;
      params[i].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
    }
  } else {
    params[0].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    params[0].writeValue.address = (CUdeviceptr)desc->readySeqDev;
    params[0].writeValue.value = desc->opSeq;
    params[0].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;

    params[1].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
    params[1].waitValue.address = (CUdeviceptr)desc->doneSeqDev;
    params[1].waitValue.value = desc->opSeq;
    params[1].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;

    params[2].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
    params[2].writeValue.address = (CUdeviceptr)desc->doneSeqDev;
    params[2].writeValue.value = 0;
    params[2].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
  }

  return ncclSuccess;
}

// ============================================================================
// Launch-time API
// ============================================================================

ncclResult_t ncclRmaProxyPutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int ctx = plan->rmaArgs->ctx;
  int nRmaTasksProxy = plan->rmaArgs->nRmaTasksProxy;
  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  int startOps = ncclRmaProxyPutStartNumOps(persistent);
  int doneOps = ncclRmaProxyPutDoneNumOps(persistent);
  int opsPerTask = startOps + doneOps;

  struct ncclRmaProxyDesc** descs = nullptr;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECK(ncclCalloc(&descs, nRmaTasksProxy));
  NCCLCHECK(ncclCalloc(&batchParams, opsPerTask * nRmaTasksProxy));

  // Phase 1: build all descriptors and fill all stream-batch memop params
  // up front. After this loop, every batchParams slot is populated and every
  // descs[i] points to a fresh heap-allocated descriptor we still own.
  for (int i = 0; i < nRmaTasksProxy; i++) {
    struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);
    assert(task->ctx == ctx);

    NCCLCHECKGOTO(ncclCalloc(&descs[i], 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutDescFromTask(comm, rmaProxyCtx, plan, task, descs[i]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutStartParams(descs[i], &batchParams[i * startOps]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutDoneParams(descs[i], &batchParams[nRmaTasksProxy * startOps + i * doneOps]), ret,
                  fail);

    INFO(NCCL_COLL,
         "ncclRmaProxyPutLaunch enqueued Desc: rank=%d peer=%d ctx=%d size=%ld signalMode=%d readySeq=%lu doneSeq=%lu "
         "persistent=%d",
         comm->rank, task->peer, ctx, task->count * ncclTypeSize(task->datatype), task->signalMode,
         (uint64_t)descs[i]->opSeq, (uint64_t)descs[i]->opSeq, persistent);

    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

  // Phase 2: enqueue descriptors and flush their stream memops in chunks
  // bounded by the per-peer circular buffer's capacity.
  {
    int count = 0;
    while (count < nRmaTasksProxy) {
      int i = count;
      for (; i < nRmaTasksProxy; i++) {
        if (ncclRmaProxyEnqueueFull(rmaProxyCtx, descs[i])) {
          break;
        }
        // EnqueueDesc transfers ownership to the proxy queue and nulls
        // descs[i] via its desc** parameter.
        NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &descs[i]), ret, fail);
      }

      int pending = i - count;
      if (pending == 0) {
        // No descs fit this round; let the proxy drain a slot.
        std::this_thread::yield();
        continue;
      }

      if (count == 0 && i == nRmaTasksProxy) {
        // Fast path: every desc fit in a single pass. Issue start + done
        // as one contiguous batch over the whole batchParams.
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opsPerTask * nRmaTasksProxy, batchParams), ret, fail);
      } else {
        // Partial flush: issue start ops for [count..i) and done blocks for
        // the same range. They live in separate regions of batchParams so
        // require two cuStreamBatchMemOp calls.
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, pending * startOps, &batchParams[count * startOps]), ret, fail);
        NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, pending * doneOps,
                                             &batchParams[nRmaTasksProxy * startOps + count * doneOps]),
                      ret, fail);
      }
      count = i;
    }
  }

exit:
  free(batchParams);
  free(descs);
  return ret;
fail:
  if (descs != nullptr) {
    for (int i = 0; i < nRmaTasksProxy; i++) {
      if (descs[i] == nullptr) continue;
      (void)ncclRmaProxyDestroyDesc(comm, &descs[i]);
    }
  }
  goto exit;
}

ncclResult_t ncclRmaProxyWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  if (!comm->rmaState.rmaProxyState.connected) {
    WARN("RMA proxy is not connected");
    return ncclInternalError;
  }

  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaProxyCtx* rmaProxyCtx = (struct ncclRmaProxyCtx*)comm->rmaState.rmaProxyState.rmaProxyCtxs[ctx];

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueProxy);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueProxy);

  assert(task->func == ncclFuncWaitSignal);
  assert(task->ctx == ctx);
  assert(plan->rmaArgs->nRmaTasksProxy == 1);

  CUstreamBatchMemOpParams* batchParams = nullptr;
  struct ncclRmaProxyDesc* desc = nullptr;

  if (task->signalMode == NCCL_SIGNAL) {
    NCCLCHECKGOTO(ncclCalloc(&desc, 1), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitDescFromTask(comm, rmaProxyCtx, plan, task, desc), ret, fail);

    int nOps = ncclRmaProxyWaitNumStreamOps(desc);
    NCCLCHECKGOTO(ncclCalloc(&batchParams, nOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, desc, batchParams), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &desc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nOps, batchParams), ret, fail);
  }

exit:
  if (desc != nullptr) {
    (void)ncclRmaProxyDestroyDesc(comm, &desc);
  }
  free(task->peers);
  free(task->nsignals);
  free(batchParams);
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  return ret;
fail:
  goto exit;
}

// ============================================================================
// Graph reclaim
// ============================================================================

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

          NCCLCHECK(ncclRmaProxyDestroyDesc(comm, &desc));
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
  if (!proxyState->connected) return ncclSuccess;

  // Step 1: Request proxy to pause and wait for acknowledgment
  {
    std::unique_lock<std::mutex> lock(proxyState->mutex);
    proxyState->rmaProgress = 2;
    proxyState->cond.notify_one();
    proxyState->cond.wait(lock, [&] { return proxyState->rmaProgress == 0; });
  }

  // Step 2: Free persistent descs on main thread
  NCCLCHECK(ncclRmaProxyReclaimPersistDescs(proxyState, plan));

  // Step 3: Resume proxy
  {
    std::lock_guard<std::mutex> lock(proxyState->mutex);
    proxyState->rmaProgress = 1;
    proxyState->cond.notify_one();
  }

  return ncclSuccess;
}
