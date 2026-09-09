/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "checks.h"
#include "compiler.h"
#include "comm.h"
#include "rma/rma.h"
#include "rma/rma_proxy.h"

// Helper functions to manage request credits
static inline bool ncclRmaProxyCanIssueRequest(struct ncclRmaProxyCtx* ctx, int targetRank) {
  return ctx->inflightRequests[targetRank] < ctx->maxInflightRequests;
}

// Issue one putSignal op via the network.
static ncclResult_t ncclRmaProxyIssuePutSignal(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx,
                                               struct ncclRmaPutSignalOp* ps) {
  if (ps->signal.op == 0) {
    NCCLCHECK(ncclRma->iput(ctx->rmaCtx, 0, ps->srcOff, ps->srcHandle, ps->size, ps->dstOff, ps->dstHandle,
                            ps->targetRank, /*optFlags*/ 0, &ps->request));
  } else {
    NCCLCHECK(ncclRma->iputSignal(ctx->rmaCtx, 0, ps->srcOff, ps->srcHandle, ps->size, ps->dstOff, ps->dstHandle,
                                  ps->targetRank, ps->signal.offset, ps->signal.signalMhandle, ps->signal.val,
                                  ps->signal.op, /*isStrongSignal*/ true, /*optFlags*/ 0, &ps->request));
  }
  // Defensive: RMA proxy iput/iputSignal should return a non-NULL request with inflightReqeusts checked.
  if (ps->request == nullptr) {
    WARN("RMA proxy iput/iputSignal returned success with NULL request");
    return ncclInternalError;
  }
  ctx->inflightRequests[ps->targetRank]++;
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyTestPutSignal(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx,
                                              struct ncclRmaPutSignalOp* ps, int* done) {
  NCCLCHECK(ncclRma->test(ctx->rmaCollComm, ps->request, done));
  if (*done) {
    if (ctx->inflightRequests[ps->targetRank] == 0) {
      WARN("RMA proxy completed request with no inflight credit");
      return ncclInternalError;
    }
    ctx->inflightRequests[ps->targetRank]--;
    ps->request = nullptr;
  }
  return ncclSuccess;
}

static ncclResult_t ncclRmaProxyProgressPutSignalGroup(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx,
                                                       struct ncclRmaPutSignalGroupOp* group) {
  // Test the completion of the issued ops.
  for (int i = 0; i < group->nIssued; i++) {
    if (group->ops[i].request == nullptr) continue;
    int done = 0;
    NCCLCHECK(ncclRmaProxyTestPutSignal(ncclRma, ctx, &group->ops[i], &done));
    if (done) group->nCompleted++;
  }

  while (group->nIssued < group->nOps) {
    struct ncclRmaPutSignalOp* op = &group->ops[group->nIssued];
    int targetRank = op->targetRank;
    if (!ncclRmaProxyCanIssueRequest(ctx, targetRank)) break;

    NCCLCHECK(ncclRmaProxyIssuePutSignal(ncclRma, ctx, op));
    group->nIssued++;
  }
  return ncclSuccess;
}

// Poll and test completion of InProgress Descs for a given peer.
// Returns after testing head Desc (stops on first incomplete to enforce FIFO).
static ncclResult_t ncclRmaProxyPollNonPersistCompletion(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx, int peer) {
  while (true) {
    struct ncclRmaProxyDesc* head = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
    if (head == nullptr) break;  // No InProgress Descs

    bool fullyDone = false;
    if (head->rmaDescType == ncclRmaDescTypePutSignal) {
      int done = 0;
      NCCLCHECK(ncclRmaProxyTestPutSignal(ncclRma, ctx, &head->putSignal, &done));
      fullyDone = (done != 0);
    } else if (head->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ncclRma, ctx, &head->putSignalGroup));
      fullyDone = (head->putSignalGroup.nCompleted == head->putSignalGroup.nOps);
    }

    if (!fullyDone) break;  // FIFO at this slot - don't peek behind head

    INFO(NCCL_COLL,
         "Rank %d ncclRmaProxyPollNonPersistCompletion: peer=%d type=%d descSeq=%lu COMPLETED, updating doneSeq",
         ctx->comm->rank, peer, head->rmaDescType, head->opSeq);

    // Update the doneSeq with RELEASE to ensure GPU sees it
    // sync with the custreamWait acquire semantic
    COMPILER_ATOMIC_STORE(head->doneSeq, head->opSeq, std::memory_order_release);
    // Dequeue and free the completed Desc
    ncclIntruQueueDequeue(&ctx->inProgressQueues[peer]);
    NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &head));
  }
  return ncclSuccess;
}

// Poll and issue ready Pending Descs for a given peer
// Moves ready Descs from pending queue to InProgress queue
static ncclResult_t ncclRmaProxyPollNonPersistDesc(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx, int peer) {
  while (true) {
    if (ncclRmaProxyCircularBufEmpty(ctx, peer)) break;

    uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
    uint32_t idx = ci & (ctx->queueSize - 1);
    struct ncclRmaProxyDesc* pendingDesc = ctx->circularBuffers[peer * ctx->queueSize + idx];

    // Check if this Desc is ready to be issued
    uint64_t readySeq = COMPILER_ATOMIC_LOAD(pendingDesc->readySeq, std::memory_order_acquire);
    if (readySeq < pendingDesc->opSeq) {
      // ReadySeq not ready yet - stop processing this peer's pending queue to maintain FIFO order
      break;
    }

    if (pendingDesc->rmaDescType == ncclRmaDescTypePutSignal) {
      if (!ncclRmaProxyCanIssueRequest(ctx, pendingDesc->putSignal.targetRank)) break;

      NCCLCHECK(ncclRmaProxyIssuePutSignal(ncclRma, ctx, &pendingDesc->putSignal));
      INFO(NCCL_COLL,
           "Rank %d ncclRmaProxyPollNonPersistDesc: targetRank=%d descSeq=%lu readySeq=%lu srcOff=%lu srcHandle=%p "
           "dstOff=%lu dstHandle=%p size=%lu - issuing network operation",
           ctx->comm->rank, pendingDesc->putSignal.targetRank, pendingDesc->opSeq, readySeq,
           pendingDesc->putSignal.srcOff, pendingDesc->putSignal.srcHandle, pendingDesc->putSignal.dstOff,
           pendingDesc->putSignal.dstHandle, pendingDesc->putSignal.size);
    } else if (pendingDesc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ncclRma, ctx, &pendingDesc->putSignalGroup));
      if (pendingDesc->putSignalGroup.nIssued < pendingDesc->putSignalGroup.nOps) break;
      INFO(NCCL_COLL,
           "Rank %d ncclRmaProxyPollNonPersistDesc: Group(nOps=%d) descSeq=%lu readySeq=%lu - issuing all ops",
           ctx->comm->rank, pendingDesc->putSignalGroup.nOps, pendingDesc->opSeq, readySeq);
    }

    // Enqueue to InProgress queue (no lock needed - progress thread only)
    ncclIntruQueueEnqueue(&ctx->inProgressQueues[peer], pendingDesc);
    // Advance CI with RELEASE after ownership transfers to the in-progress queue.
    COMPILER_ATOMIC_STORE_32(&ctx->cis[peer], ci + 1, std::memory_order_release);
  }
  return ncclSuccess;
}

// Blocking strict-order loopback iput flush.
// Issues a local iput to the flush buffer registered with strict ordering and
// spins until the NIC reports completion, guaranteeing that all prior data
// written through the NIC-GPU path is visible in vidmem.
static ncclResult_t ncclRmaProxyFlushNicGpuPath(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx) {
  void* request = nullptr;
  size_t flushOff = (size_t)ctx->comm->rank * sizeof(uint64_t);
  NCCLCHECK(ncclRma->iput(ctx->rmaCtx, 0, flushOff, ctx->flushBufMhandle, sizeof(uint64_t), flushOff,
                          ctx->flushBufMhandle, ctx->comm->rank, /*optFlags*/ 0, &request));
  int done = 0;
  while (!done) {
    NCCLCHECK(ncclRma->test(ctx->rmaCollComm, request, &done));
  }
  return ncclSuccess;
}

// Poll persistent descriptors for a given peer (graph mode).
static ncclResult_t ncclRmaProxyPollPersistDesc(ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx, int peer) {
  struct ncclRmaProxyDesc* desc = ncclIntruQueueHead(&ctx->persistentQueues[peer]);
  while (desc != nullptr) {
    if (!COMPILER_ATOMIC_LOAD(&desc->persistDescValid, std::memory_order_acquire)) {
      desc = desc->next;
      continue;
    }

    if (desc->rmaDescType == ncclRmaDescTypePutSignal) {
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          if (!ncclRmaProxyCanIssueRequest(ctx, desc->putSignal.targetRank)) {
            desc = desc->next;
            continue;
          }

          NCCLCHECK(ncclRmaProxyIssuePutSignal(ncclRma, ctx, &desc->putSignal));
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal issued", ctx->comm->rank, peer);
        }
      } else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        int done = 0;
        NCCLCHECK(ncclRmaProxyTestPutSignal(ncclRma, ctx, &desc->putSignal, &done));
        if (done) {
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal completed, doneSeq set",
               ctx->comm->rank, peer);
        }
      }
    } else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      // Group descs only land at peer == comm->rank (see ncclRmaProxyEnqueueDesc).
      // State machine mirrors PutSignal but issues / tests nOps requests.
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          desc->putSignalGroup.nIssued = 0;
          desc->putSignalGroup.nCompleted = 0;

          NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ncclRma, ctx, &desc->putSignalGroup));
          if (desc->putSignalGroup.nIssued > 0) {
            COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
            desc->rmaDescState = ncclRmaDescStateInProgress;
            INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d Group(nOps=%d) issued", ctx->comm->rank, peer,
                 desc->putSignalGroup.nOps);
          }
        }
      } else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        NCCLCHECK(ncclRmaProxyProgressPutSignalGroup(ncclRma, ctx, &desc->putSignalGroup));
        if (desc->putSignalGroup.nCompleted == desc->putSignalGroup.nOps) {
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d Group complete, doneSeq set", ctx->comm->rank,
               peer);
        }
      }
    } else if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d WaitSignal ready, start polling signals",
               ctx->comm->rank, peer);
        }
      } else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        bool signalsAllArrived = true;
        for (int i = 0; i < desc->waitSignal.npeers; i++) {
          int peerRank = desc->waitSignal.waitPeers[i];
          size_t signalSlot = ncclRmaSignalSlot(ctx->comm->nRanks, desc->waitSignal.waitSignalIdxs[i], peerRank);
          uint64_t signalVal = COMPILER_ATOMIC_LOAD(&ctx->cpuAccessSignals[signalSlot], std::memory_order_acquire);
          if (signalVal < ctx->cpuAccessSignalsHost[signalSlot] + desc->waitSignal.waitSignals[i]) {
            signalsAllArrived = false;
            break;
          }
        }

        if (signalsAllArrived) {
          if (desc->waitSignal.needFlush) {
            NCCLCHECK(ncclRmaProxyFlushNicGpuPath(ncclRma, ctx));
          }

          for (int i = 0; i < desc->waitSignal.npeers; i++) {
            int peerRank = desc->waitSignal.waitPeers[i];
            size_t signalSlot = ncclRmaSignalSlot(ctx->comm->nRanks, desc->waitSignal.waitSignalIdxs[i], peerRank);
            ctx->cpuAccessSignalsHost[signalSlot] += desc->waitSignal.waitSignals[i];
          }

          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d WaitSignal all arrived, doneSeq set",
               ctx->comm->rank, peer);
        }
      }
    }

    desc = desc->next;
  }
  return ncclSuccess;
}

// Checks the RMA proxy progress.
ncclResult_t ncclRmaProxyProgress(ncclRma_t* ncclRma, void* rmaProxyCtx) {
  struct ncclRmaProxyCtx* ctx = (struct ncclRmaProxyCtx*)rmaProxyCtx;

  if (ncclRma->rmaProgress) NCCLCHECK(ncclRma->rmaProgress(ctx->rmaCtx));

  // Loop through each peer's queues
  for (int i = 0; i < ctx->comm->nRanks; i++) {
    // Step 1: Poll completion of InProgress Descs (non-graph)
    NCCLCHECK(ncclRmaProxyPollNonPersistCompletion(ncclRma, ctx, i));

    // Step 2: Poll and issue ready Pending Descs (non-graph)
    NCCLCHECK(ncclRmaProxyPollNonPersistDesc(ncclRma, ctx, i));

    // Step 3: Poll persistent descriptors (graph mode)
    NCCLCHECK(ncclRmaProxyPollPersistDesc(ncclRma, ctx, i));
  }
  return ncclSuccess;
}
