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

// Poll and test completion of InProgress Descs for a given peer
// Returns after testing head Desc (stops on first incomplete to enforce FIFO)
static ncclResult_t ncclRmaProxyPollNonPersistCompletion(ncclGin_t *ncclGin, struct ncclRmaProxyCtx *ctx, int peer) {
  while (true) {
    struct ncclRmaProxyDesc *inProgressDesc = ncclIntruQueueHead(&ctx->inProgressQueues[peer]);
    if (inProgressDesc == NULL) break;  // No InProgress Descs

    int done = 0;
    NCCLCHECK(ncclGin->test(ctx->ginCollComm, inProgressDesc->putSignal.request, &done));
    if (done) {
      INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollNonPersistCompletion: targetRank=%d descSeq=%lu COMPLETED, updating doneSeq",
        ctx->comm->rank, inProgressDesc->putSignal.targetRank, inProgressDesc->opSeq);

      // Update the doneSeq for the target rank with RELEASE to ensure GPU sees it
      COMPILER_ATOMIC_STORE(inProgressDesc->doneSeq, inProgressDesc->opSeq, std::memory_order_release); // sync with the custreamWait aquire semantic
      // Dequeue and free the completed Desc
      ncclIntruQueueDequeue(&ctx->inProgressQueues[peer]);
      NCCLCHECK(ncclRmaProxyDestroyDescNonPersistent(inProgressDesc));
    } else {
      // Head is not done - stop testing to enforce FIFO completion order
      break;
    }
  }
  return ncclSuccess;
}

// Poll and issue ready Pending Descs for a given peer
// Moves ready Descs from pending queue to InProgress queue
static ncclResult_t ncclRmaProxyPollNonPersistDesc(ncclGin_t *ncclGin, struct ncclRmaProxyCtx *ctx, int peer) {
  while (true) {
    if (ncclRmaProxyCircularBufEmpty(ctx, peer)) break;

    uint32_t ci = COMPILER_ATOMIC_LOAD_32(&ctx->cis[peer], std::memory_order_relaxed);
    uint32_t idx = ci & (ctx->queueSize - 1);
    struct ncclRmaProxyDesc *pendingDesc = ctx->circularBuffers[peer * ctx->queueSize + idx];

    // Check if this Desc is ready to be issued
    uint64_t readySeq = COMPILER_ATOMIC_LOAD(pendingDesc->readySeq, std::memory_order_acquire);
    if (readySeq >= pendingDesc->opSeq) {
      // Advance CI with RELEASE to ensure descriptor is consumed
      COMPILER_ATOMIC_STORE_32(&ctx->cis[peer], ci + 1, std::memory_order_release);

      // Issue the network operation
      if (pendingDesc->putSignal.signal.op == 0) {
        // No signal operation
        NCCLCHECK(ncclGin->iput(ctx->ginCtx, 0,
          pendingDesc->putSignal.srcOff, pendingDesc->putSignal.srcHandle, pendingDesc->putSignal.size,
          pendingDesc->putSignal.dstOff, pendingDesc->putSignal.dstHandle,
          pendingDesc->putSignal.targetRank, &pendingDesc->putSignal.request));
      } else {
        // Signal operation needed
        NCCLCHECK(ncclGin->iputSignal(ctx->ginCtx, 0,
          pendingDesc->putSignal.srcOff, pendingDesc->putSignal.srcHandle, pendingDesc->putSignal.size,
          pendingDesc->putSignal.dstOff, pendingDesc->putSignal.dstHandle,
          pendingDesc->putSignal.targetRank, pendingDesc->putSignal.signal.offset, pendingDesc->putSignal.signal.signalMhandle,
          pendingDesc->putSignal.signal.val, pendingDesc->putSignal.signal.op, &pendingDesc->putSignal.request));
      }

      // Enqueue to InProgress queue (no lock needed - progress thread only)
      ncclIntruQueueEnqueue(&ctx->inProgressQueues[peer], pendingDesc);

      INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollNonPersistDesc: targetRank=%d descSeq=%lu readySeq=%lu srcOff=%lu srcHandle=%p dstOff=%lu dstHandle=%p size=%lu - issuing network operation",
        ctx->comm->rank, pendingDesc->putSignal.targetRank, pendingDesc->opSeq, readySeq, pendingDesc->putSignal.srcOff, pendingDesc->putSignal.srcHandle, pendingDesc->putSignal.dstOff, pendingDesc->putSignal.dstHandle, pendingDesc->putSignal.size);
    } else {
      // ReadySeq not ready yet - stop processing this peer's pending queue to maintain FIFO order
      break;
    }
  }
  return ncclSuccess;
}

// Blocking strict-order loopback iput flush.
// Issues a local iput to the flush buffer registered with strict ordering and
// spins until the NIC reports completion, guaranteeing that all prior data
// written through the NIC-GPU path is visible in vidmem.
static ncclResult_t ncclRmaProxyFlushNicGpuPath(ncclGin_t *ncclGin, struct ncclRmaProxyCtx *ctx) {
  void* request = NULL;
  size_t flushOff = (size_t)ctx->comm->rank * sizeof(uint64_t);
  NCCLCHECK(ncclGin->iput(ctx->ginCtx, 0, flushOff, ctx->flushBufMhandle, sizeof(uint64_t),
            flushOff, ctx->flushBufMhandle, ctx->comm->rank, &request));
  int done = 0;
  while (!done) {
    NCCLCHECK(ncclGin->test(ctx->ginCollComm, request, &done));
  }
  return ncclSuccess;
}

// Poll persistent descriptors for a given peer (graph mode).
static ncclResult_t ncclRmaProxyPollPersistDesc(ncclGin_t *ncclGin, struct ncclRmaProxyCtx *ctx, int peer) {
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

          if (desc->putSignal.signal.op == 0) {
            NCCLCHECK(ncclGin->iput(ctx->ginCtx, 0,
              desc->putSignal.srcOff, desc->putSignal.srcHandle, desc->putSignal.size,
              desc->putSignal.dstOff, desc->putSignal.dstHandle,
              desc->putSignal.targetRank, &desc->putSignal.request));
          } else {
            NCCLCHECK(ncclGin->iputSignal(ctx->ginCtx, 0,
              desc->putSignal.srcOff, desc->putSignal.srcHandle, desc->putSignal.size,
              desc->putSignal.dstOff, desc->putSignal.dstHandle,
              desc->putSignal.targetRank, desc->putSignal.signal.offset, desc->putSignal.signal.signalMhandle,
              desc->putSignal.signal.val, desc->putSignal.signal.op, &desc->putSignal.request));
          }

          desc->rmaDescState = ncclRmaDescStateInProgress;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal issued", ctx->comm->rank, peer);
        }
      }
      else if (desc->rmaDescState == ncclRmaDescStateInProgress) {
        int done = 0;
        NCCLCHECK(ncclGin->test(ctx->ginCollComm, desc->putSignal.request, &done));
        if (done) {
          desc->putSignal.request = NULL;
          COMPILER_ATOMIC_STORE(desc->doneSeq, (uint64_t)1, std::memory_order_release);
          desc->rmaDescState = ncclRmaDescStateReady;
          INFO(NCCL_COLL, "Rank %d ncclRmaProxyPollPersistDesc: peer=%d PutSignal completed, doneSeq set",
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
            NCCLCHECK(ncclRmaProxyFlushNicGpuPath(ncclGin, ctx));
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
ncclResult_t ncclRmaProxyProgress(ncclGin_t *ncclGin, void *rmaProxyCtx) {
  struct ncclRmaProxyCtx *ctx = (struct ncclRmaProxyCtx *)rmaProxyCtx;

  // Loop through each peer's queues
  for (int i = 0; i < ctx->comm->nRanks; i++) {
    // Step 1: Poll completion of InProgress Descs (non-graph)
    NCCLCHECK(ncclRmaProxyPollNonPersistCompletion(ncclGin, ctx, i));

    // Step 2: Poll and issue ready Pending Descs (non-graph)
    NCCLCHECK(ncclRmaProxyPollNonPersistDesc(ncclGin, ctx, i));

    // Step 3: Poll persistent descriptors (graph mode)
    NCCLCHECK(ncclRmaProxyPollPersistDesc(ncclGin, ctx, i));
  }
  return ncclSuccess;
}
