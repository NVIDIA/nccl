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
#include "rma/rma_proxy.h"

// Issue one putSignal op via the network.
static ncclResult_t ncclRmaProxyIssuePutSignal(
    ncclRma_t* ncclRma, struct ncclRmaProxyCtx* ctx,
    struct ncclRmaPutSignalOp* ps) {
  if (ps->signal.op == 0) {
    NCCLCHECK(ncclRma->iput(ctx->rmaCtx, 0,
        ps->srcOff, ps->srcHandle, ps->size,
        ps->dstOff, ps->dstHandle,
        ps->targetRank, &ps->request));
  } else {
    NCCLCHECK(ncclRma->iputSignal(ctx->rmaCtx, 0,
        ps->srcOff, ps->srcHandle, ps->size,
        ps->dstOff, ps->dstHandle,
        ps->targetRank, ps->signal.offset, ps->signal.signalMhandle,
        ps->signal.val, ps->signal.op, /*isStrongSignal*/true, &ps->request));
  }
  return ncclSuccess;
}

// Poll and test completion of InProgress Descs for a given peer.
// Returns after testing head Desc (stops on first incomplete to enforce FIFO).
static ncclResult_t ncclRmaProxyPollNonPersistCompletion(ncclRma_t *ncclRma, struct ncclRmaProxyCtx *ctx, int peer) {
  while (true) {
    struct ncclRmaProxyDesc *head = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
    if (head == NULL) break;  // No InProgress Descs

    bool fullyDone = false;
    if (head->rmaDescType == ncclRmaDescTypePutSignal) {
      int done = 0;
      NCCLCHECK(ncclRma->test(ctx->rmaCollComm, head->putSignal.request, &done));
      fullyDone = (done != 0);
    } else if (head->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      for (int i = 0; i < head->putSignalGroup.nOps; i++) {
        if (head->putSignalGroup.ops[i].request == NULL) continue;
        int done = 0;
        NCCLCHECK(ncclRma->test(ctx->rmaCollComm,
                                head->putSignalGroup.ops[i].request, &done));
        if (done) {
          // Null the request so we skip this op on subsequent polls.
          head->putSignalGroup.ops[i].request = NULL;
          head->putSignalGroup.nRemaining--;
        }
      }
      fullyDone = (head->putSignalGroup.nRemaining == 0);
    }

    if (!fullyDone) break;  // FIFO at this slot - don't peek behind head

    INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollNonPersistCompletion: peer=%d type=%d descSeq=%lu COMPLETED, updating doneSeq",
      ctx->comm->rank, peer, head->rmaDescType, head->opSeq);

    // Update the doneSeq with RELEASE to ensure GPU sees it
    COMPILER_ATOMIC_STORE(head->doneSeq, head->opSeq, std::memory_order_release); // sync with the custreamWait acquire semantic
    // Dequeue and free the completed Desc
    ncclIntruQueueDequeue(&ctx->inProgressQueues[peer]);
    NCCLCHECK(ncclRmaProxyDestroyDesc(ctx->comm, &head));
  }
  return ncclSuccess;
}

// Poll and issue ready Pending Descs for a given peer
// Moves ready Descs from pending queue to InProgress queue
static ncclResult_t ncclRmaProxyPollNonPersistDesc(ncclRma_t *ncclRma, struct ncclRmaProxyCtx *ctx, int peer) {
  while (true) {
    if (ncclRmaProxyCircularBufEmpty(ctx, peer)) break;

    uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
    uint32_t idx = ci & (ctx->queueSize - 1);
    struct ncclRmaProxyDesc *pendingDesc = ctx->circularBuffers[peer * ctx->queueSize + idx];

    // Check if this Desc is ready to be issued
    uint64_t readySeq = COMPILER_ATOMIC_LOAD(pendingDesc->readySeq, std::memory_order_acquire);
    if (readySeq < pendingDesc->opSeq) {
      // ReadySeq not ready yet - stop processing this peer's pending queue to maintain FIFO order
      break;
    }

    // Advance CI with RELEASE to ensure descriptor is consumed
    COMPILER_ATOMIC_STORE_32(&ctx->cis[peer], ci + 1, std::memory_order_release);

    if (pendingDesc->rmaDescType == ncclRmaDescTypePutSignal) {
      NCCLCHECK(ncclRmaProxyIssuePutSignal(ncclRma, ctx, &pendingDesc->putSignal));
      INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollNonPersistDesc: targetRank=%d descSeq=%lu readySeq=%lu srcOff=%lu srcHandle=%p dstOff=%lu dstHandle=%p size=%lu - issuing network operation",
        ctx->comm->rank, pendingDesc->putSignal.targetRank, pendingDesc->opSeq, readySeq, pendingDesc->putSignal.srcOff, pendingDesc->putSignal.srcHandle, pendingDesc->putSignal.dstOff, pendingDesc->putSignal.dstHandle, pendingDesc->putSignal.size);
    } else if (pendingDesc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      pendingDesc->putSignalGroup.nRemaining = pendingDesc->putSignalGroup.nOps;
      for (int i = 0; i < pendingDesc->putSignalGroup.nOps; i++) {
        NCCLCHECK(ncclRmaProxyIssuePutSignal(
            ncclRma, ctx, &pendingDesc->putSignalGroup.ops[i]));
      }
      INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollNonPersistDesc: Group(nOps=%d) descSeq=%lu readySeq=%lu - issuing all ops",
        ctx->comm->rank, pendingDesc->putSignalGroup.nOps, pendingDesc->opSeq, readySeq);
    }

    // Enqueue to InProgress queue (no lock needed - progress thread only)
    ncclIntruQueueEnqueue(&ctx->inProgressQueues[peer], pendingDesc);
  }
  return ncclSuccess;
}

// Blocking strict-order loopback iput flush.
// Issues a local iput to the flush buffer registered with strict ordering and
// spins until the NIC reports completion, guaranteeing that all prior data
// written through the NIC-GPU path is visible in vidmem.
static ncclResult_t ncclRmaProxyFlushNicGpuPath(ncclRma_t *ncclRma, struct ncclRmaProxyCtx *ctx) {
  void* request = NULL;
  size_t flushOff = (size_t)ctx->comm->rank * sizeof(uint64_t);
  NCCLCHECK(ncclRma->iput(ctx->rmaCtx, 0, flushOff, ctx->flushBufMhandle, sizeof(uint64_t),
            flushOff, ctx->flushBufMhandle, ctx->comm->rank, &request));
  int done = 0;
  while (!done) {
    NCCLCHECK(ncclRma->test(ctx->rmaCollComm, request, &done));
  }
  return ncclSuccess;
}

// Poll persistent descriptors for a given peer (graph mode).
static ncclResult_t ncclRmaProxyPollPersistDesc(ncclRma_t *ncclRma, struct ncclRmaProxyCtx *ctx, int peer) {
  struct ncclRmaProxyDesc *desc = ncclIntruQueueHead(&ctx->persistentQueues[peer]);
  while (desc != NULL) {
    if (!COMPILER_ATOMIC_LOAD(&desc->persistDescValid, std::memory_order_acquire)) {
      desc = desc->next;
      continue;
    }

    if (desc->rmaDescType == ncclRmaDescTypePutSignal) {
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);

          NCCLCHECK(ncclRmaProxyIssuePutSignal(ncclRma, ctx, &desc->putSignal));

          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal issued", ctx->comm->rank, peer);
        }
      }
      else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        int done = 0;
        NCCLCHECK(ncclRma->test(ctx->rmaCollComm, desc->putSignal.request, &done));
        if (done) {
          desc->putSignal.request = NULL;
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal completed, doneSeq set",
            ctx->comm->rank, peer);
        }
      }
    }
    else if (desc->rmaDescType == ncclRmaDescTypePutSignalGroup) {
      // Group descs only land at peer == comm->rank (see ncclRmaProxyEnqueueDesc).
      // State machine mirrors PutSignal but issues / tests nOps requests.
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);

          desc->putSignalGroup.nRemaining = desc->putSignalGroup.nOps;
          for (int i = 0; i < desc->putSignalGroup.nOps; i++) {
            NCCLCHECK(ncclRmaProxyIssuePutSignal(
                ncclRma, ctx, &desc->putSignalGroup.ops[i]));
          }

          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d Group(nOps=%d) issued",
            ctx->comm->rank, peer, desc->putSignalGroup.nOps);
        }
      }
      else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        for (int i = 0; i < desc->putSignalGroup.nOps; i++) {
          if (desc->putSignalGroup.ops[i].request == NULL) continue;
          int done = 0;
          NCCLCHECK(ncclRma->test(ctx->rmaCollComm,
                                  desc->putSignalGroup.ops[i].request, &done));
          if (done) {
            // Null the request so we skip this op on subsequent polls.
            desc->putSignalGroup.ops[i].request = NULL;
            desc->putSignalGroup.nRemaining--;
          }
        }
        if (desc->putSignalGroup.nRemaining == 0) {
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d Group complete, doneSeq set",
            ctx->comm->rank, peer);
        }
      }
    }
    else if (desc->rmaDescType == ncclRmaDescTypeWaitSignal) {
      if (desc->rmaDescState == ncclRmaDescStateReady) {
        uint64_t readyVal = COMPILER_ATOMIC_LOAD(desc->readySeq, std::memory_order_acquire);
        if (readyVal == 1) {
          COMPILER_ATOMIC_STORE(desc->readySeq, (uint64_t)0, std::memory_order_relaxed);
          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d WaitSignal ready, start polling signals",
            ctx->comm->rank, peer);
        }
      }
      else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        bool signalsAllArrived = true;
        for (int i = 0; i < desc->waitSignal.npeers; i++) {
          int peerRank = desc->waitSignal.waitPeers[i];
          uint64_t signalVal = COMPILER_ATOMIC_LOAD(&ctx->cpuAccessSignals[peerRank], std::memory_order_acquire);
          if (signalVal < ctx->cpuAccessSignalsHost[peerRank] + desc->waitSignal.waitSignals[i]) {
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
            ctx->cpuAccessSignalsHost[peerRank] += desc->waitSignal.waitSignals[i];
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
ncclResult_t ncclRmaProxyProgress(ncclRma_t *ncclRma, void *rmaProxyCtx) {
  struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

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
