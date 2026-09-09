/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "collectives.h"
#include "cudawrap.h"
#include "rma/rma.h"
#include "rma/rma_ce.h"
#include "ce_coll.h"

ncclResult_t ncclRmaCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  uint64_t* signalsDevBase = nullptr;
  uint64_t* ackInitHost = nullptr;
  size_t signalSlots = 0;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  comm->rmaState.rmaCeState.rmaCeCtxCount = comm->config.numRmaCtx;

  // CE only targets intra-node (LSA) peers, so signals are indexed by LSA-local rank.
  signalSlots = (size_t)comm->devrState.lsaSize * comm->config.numRmaSig;

  NCCLCHECKGOTO(ncclCalloc(&ackInitHost, signalSlots), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rmaState.rmaCeState.rmaCeCtxs, comm->rmaState.rmaCeState.rmaCeCtxCount), ret, fail);
  for (int i = 0; i < comm->rmaState.rmaCeState.rmaCeCtxCount; i++) {
    // Allocate the RMA CE context
    struct ncclRmaCeCtx* ceCtx;
    NCCLCHECKGOTO(ncclCalloc(&ceCtx, 1), ret, fail);
    comm->rmaState.rmaCeState.rmaCeCtxs[i] = ceCtx;

    // Initialize context
    ceCtx->comm = comm;

    // Allocate and register symmetric memory for signals
    size_t signalsBufSize = 3 * signalSlots * sizeof(uint64_t);
    ncclWindow_vidmem* signalsWinDev;
    ncclWindow_vidmem* signalsWinDevHost;

    NCCLCHECKGOTO(ncclCudaCalloc(&signalsDevBase, 3 * signalSlots, comm->memManager), ret, fail);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalsDevBase, signalsBufSize, NCCL_WIN_COLL_SYMMETRIC,
                                                &signalsWinDev),
                  ret, fail);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, signalsWinDev, &signalsWinDevHost), ret, fail);

    // Get the ncclDevrWindow from the winHost field
    ceCtx->signalsWin = (struct ncclDevrWindow*)signalsWinDevHost->winHost;
    ceCtx->signalsDev = signalsDevBase;
    ceCtx->graphSignalsDev = signalsDevBase + signalSlots;
    ceCtx->graphAckDev = signalsDevBase + 2 * signalSlots;
    signalsDevBase = nullptr;
    ceCtx->signalOffset = 0;
    ceCtx->graphSignalOffset = signalSlots * sizeof(uint64_t);
    ceCtx->graphAckOffset = 2 * signalSlots * sizeof(uint64_t);

    // Initialize ack flags to 1
    for (size_t r = 0; r < signalSlots; r++) ackInitHost[r] = 1;
    NCCLCHECKGOTO(ncclCudaMemcpy(ceCtx->graphAckDev, ackInitHost, signalSlots), ret, fail);

    // Allocate device-resident constants for graph-safe D2D signal/ack writes
    NCCLCHECKGOTO(ncclCudaCalloc(&ceCtx->signalConstDev, 2, comm->memManager), ret, fail);
    ceCtx->signalConstZeroDev = &ceCtx->signalConstDev[0];
    ceCtx->signalConstOneDev = &ceCtx->signalConstDev[1];
    {
      uint64_t zeroone[] = {0, 1};
      NCCLCHECKGOTO(ncclCudaMemcpy(ceCtx->signalConstDev, zeroone, 2), ret, fail);
    }

    // Allocate host buffer to track expected non-graph signal values
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalsHost, signalSlots), ret, fail);

    // Allocate host per-(sigIdx, rank) sequence counters.
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalOpSeqs, signalSlots), ret, fail);
    // Allocate device staging slots for non-graph signal values, with capacity comm->nRanks.
    NCCLCHECKGOTO(ncclCudaCalloc(&ceCtx->signalOpSeqsDev, comm->nRanks, comm->memManager), ret, fail);
  }

  INFO(NCCL_INIT, "Rank %d: finished init RMA CE contexts, numRmaCeCtxs %d", comm->rank, comm->config.numRmaCtx);

  // Create CE stream for parallel execution
  CUDACHECKGOTO(cudaStreamCreateWithFlags(&comm->rmaState.rmaCeState.ceStream, cudaStreamNonBlocking), ret, fail);

  // Create event for synchronization
  CUDACHECKGOTO(cudaEventCreateWithFlags(&comm->rmaState.rmaCeState.ceEvent, cudaEventDisableTiming), ret, fail);

  comm->rmaState.rmaCeState.initialized = true;

exit:
  free(ackInitHost);
  return ret;
fail:
  if (signalsDevBase) ncclCudaFree(signalsDevBase, comm->memManager);
  goto exit;
}

ncclResult_t ncclRmaCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // Clean up rmaCeInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->rmaCeInitTaskQueue)) {
    struct ncclRmaCeInitTask* task = ncclIntruQueueDequeue(&comm->rmaCeInitTaskQueue);
    free(task);
  }

  // Destroy CE stream and event
  if (comm->rmaState.rmaCeState.ceStream != NULL) {
    CUDACHECKGOTO(cudaStreamDestroy(comm->rmaState.rmaCeState.ceStream), ret, fail);
    comm->rmaState.rmaCeState.ceStream = NULL;
  }

  if (comm->rmaState.rmaCeState.ceEvent != NULL) {
    CUDACHECKGOTO(cudaEventDestroy(comm->rmaState.rmaCeState.ceEvent), ret, fail);
    comm->rmaState.rmaCeState.ceEvent = NULL;
  }

  for (int i = 0; i < comm->rmaState.rmaCeState.rmaCeCtxCount; i++) {
    struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[i];

    // Free per-rank operation sequence counters
    if (ceCtx->signalOpSeqs) free(ceCtx->signalOpSeqs);
    if (ceCtx->signalOpSeqsDev) NCCLCHECKGOTO(ncclCudaFree(ceCtx->signalOpSeqsDev, comm->memManager), ret, fail);

    // Free host signals buffer
    if (ceCtx->signalsHost) free(ceCtx->signalsHost);

    // Free device-resident constants
    if (ceCtx->signalConstDev) NCCLCHECKGOTO(ncclCudaFree(ceCtx->signalConstDev, comm->memManager), ret, fail);

    // Deregister and free signal window
    if (ceCtx->signalsWin) NCCLCHECKGOTO(ncclCommWindowDeregister(comm, ceCtx->signalsWin->vidmem), ret, fail);

    // Free signal device memory
    if (ceCtx->signalsDev) NCCLCHECKGOTO(ncclCudaFree(ceCtx->signalsDev, comm->memManager), ret, fail);

    // Free the context itself
    free(ceCtx);
    comm->rmaState.rmaCeState.rmaCeCtxs[i] = NULL;
  }

  // Reset the number of contexts and initialized flag
  comm->rmaState.rmaCeState.rmaCeCtxCount = 0;
  comm->rmaState.rmaCeState.initialized = false;

  free(comm->rmaState.rmaCeState.rmaCeCtxs);
  comm->rmaState.rmaCeState.rmaCeCtxs = NULL;

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t ncclRmaCePutLaunchPersist(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;

  int lsaSize = comm->devrState.lsaSize;
  int lsaSelf = comm->devrState.lsaSelf;

  // Reusable per-task batch params
  // signal and data operations can not be in the same batch as batched mem copy does not guarantee order of execution
  struct ncclCeBatchOpsParams dataParams = {};
  struct ncclCeBatchOpsParams signalParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&dataParams, 1), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&signalParams, 1), ret, fail);

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
    ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
    // Per-task context: a put-with-signal on ctx c raises the peer's ctx-c signal.
    struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[task->ctx];

    int peerLsaRank;
    NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, task->peer, &peerLsaRank), ret, fail);

    size_t bytes = task->count * ncclTypeSize(task->datatype);

    // Graph: wait for receiver's ack, then reset ack flag.
    // Ack handshake is only required for put with signals.
    if (task->signalMode != NCCL_SIGNAL_NONE) {
      size_t ackSlot = ncclRmaSignalSlot(lsaSize, task->signalIdx, peerLsaRank);
      CUdeviceptr ackAddr = (CUdeviceptr)&ceCtx->graphAckDev[ackSlot];
      CUstreamBatchMemOpParams ackOps[2] = {};
      ackOps[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      ackOps[0].waitValue.address = ackAddr;
      ackOps[0].waitValue.value64 = 1;
      ackOps[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
      ackOps[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
      ackOps[1].writeValue.address = ackAddr;
      ackOps[1].writeValue.value64 = 0;
      ackOps[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, 2, ackOps), ret, fail);
    }

    dataParams.numOps = 0;
    signalParams.numOps = 0;

    // Data movement
    if (bytes > 0) {
      void* peerBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, task->peerWinHost, task->peerWinOffset, peerLsaRank, &peerBuff), ret,
                    fail);
      if (peerBuff == NULL) {
        WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
        ret = ncclInvalidArgument;
        goto fail;
      }
      dataParams.srcs[dataParams.numOps] = const_cast<void*>(task->srcBuff);
      dataParams.dsts[dataParams.numOps] = peerBuff;
      dataParams.sizes[dataParams.numOps] = bytes;
      dataParams.numOps++;
    }

    // Graph: write signal=1 to peer's graphSignalsDev (separate from non-graph signals)
    if (task->signalMode != NCCL_SIGNAL_NONE) {
      size_t signalOffset = ncclRmaSignalOffset(lsaSize, task->signalIdx, lsaSelf);
      void* peerGraphSignal;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphSignalOffset + signalOffset, peerLsaRank,
                                          &peerGraphSignal),
                    ret, fail);
      signalParams.srcs[signalParams.numOps] = ceCtx->signalConstOneDev;
      signalParams.dsts[signalParams.numOps] = peerGraphSignal;
      signalParams.sizes[signalParams.numOps] = sizeof(uint64_t);
      signalParams.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &dataParams, stream), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &signalParams, stream), ret, fail);

    // Free the task after processing
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

exit:
  ncclCeFreeBatchOpsParams(&dataParams);
  ncclCeFreeBatchOpsParams(&signalParams);
  return ret;
fail:
  goto exit;
}

static ncclResult_t ncclRmaCePutLaunchNonPersist(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                                 cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;

  int lsaSize = comm->devrState.lsaSize;
  int lsaSelf = comm->devrState.lsaSelf;

  // signal and data operations can not be in the same batch as batched mem copy does not guarantee order of execution
  // we can not have signal operations to the same physical address in the same batch as batched mem copy does not
  // guarantee order of execution
  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* peerTaskQueues = nullptr;
  int* activePeers = nullptr;
  struct ncclCeBatchOpsParams dataParams = {};
  struct ncclCeBatchOpsParams signalParams = {};
  CUstreamBatchMemOpParams* seqStageOps = nullptr;
  struct ncclTaskRma* currentTask = nullptr;
  int nActivePeers = 0;

  if (nRmaTasksCe == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&peerTaskQueues, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&activePeers, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&dataParams, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&signalParams, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&seqStageOps, comm->nRanks), ret, fail);

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);
    int peer = task->peer;
    if (ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
      activePeers[nActivePeers++] = peer;
    }
    ncclIntruQueueEnqueue(&peerTaskQueues[peer], task);
  }

  while (nActivePeers > 0) {
    int nNextActivePeers = 0;
    int nSeqStageOps = 0;
    dataParams.numOps = 0;
    signalParams.numOps = 0;

    for (int i = 0; i < nActivePeers; i++) {
      int peer = activePeers[i];
      currentTask = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
      // Per-task context: each task's signal goes to the peer's own-context buffer.
      struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[currentTask->ctx];

      int peerLsaRank;
      NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, currentTask->peer, &peerLsaRank), ret, fail);

      size_t bytes = currentTask->count * ncclTypeSize(currentTask->datatype);

      // Data movement
      if (bytes > 0) {
        void* peerBuff;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, currentTask->peerWinHost, currentTask->peerWinOffset, peerLsaRank,
                                            &peerBuff),
                      ret, fail);
        if (peerBuff == NULL) {
          WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
          ret = ncclInvalidArgument;
          goto fail;
        }
        dataParams.srcs[dataParams.numOps] = const_cast<void*>(currentTask->srcBuff);
        dataParams.dsts[dataParams.numOps] = peerBuff;
        dataParams.sizes[dataParams.numOps] = bytes;
        dataParams.numOps++;
      }

      // Non-graph: write incrementing sequence to peer's signalsDev.
      // Each batch has at most one task per peer, so each signal batch has at
      // most one write to a given peer signal slot.
      if (currentTask->signalMode != NCCL_SIGNAL_NONE) {
        // Device staging slots hold one signal op per rank in this batch. Each batch processes at
        // most one task per active peer, so this should never trip; guard against silent overflow.
        if (nSeqStageOps >= comm->nRanks) {
          WARN("RMA CE: staged signal ops (%d) exceed staging capacity (%d)", nSeqStageOps, comm->nRanks);
          ret = ncclInternalError;
          goto fail;
        }
        size_t signalOffset = ncclRmaSignalOffset(lsaSize, currentTask->signalIdx, lsaSelf);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, currentTask->signalIdx, peerLsaRank);
        void* peerSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->signalOffset + signalOffset, peerLsaRank,
                                            &peerSignal),
                      ret, fail);
        ceCtx->signalOpSeqs[signalSlot]++;

        seqStageOps[nSeqStageOps].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
        seqStageOps[nSeqStageOps].writeValue.address = (CUdeviceptr)&ceCtx->signalOpSeqsDev[nSeqStageOps];
        seqStageOps[nSeqStageOps].writeValue.value64 = ceCtx->signalOpSeqs[signalSlot];
        seqStageOps[nSeqStageOps].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;

        signalParams.srcs[signalParams.numOps] = &ceCtx->signalOpSeqsDev[nSeqStageOps];
        signalParams.dsts[signalParams.numOps] = peerSignal;
        signalParams.sizes[signalParams.numOps] = sizeof(uint64_t);
        signalParams.numOps++;

        nSeqStageOps++;
      }

      ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, currentTask);
      currentTask = nullptr;

      if (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
        activePeers[nNextActivePeers++] = peer;
      }
    }

    // Issue batches in stream order. Staging writes must precede the memcpy
    // batch because signal mem copies read the staged sequence slots.
    if (nSeqStageOps > 0) {
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nSeqStageOps, seqStageOps), ret, fail);
    }
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &dataParams, stream), ret, fail);
    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &signalParams, stream), ret, fail);
    nActivePeers = nNextActivePeers;
  }

exit:
  if (currentTask != nullptr) ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, currentTask);
  if (peerTaskQueues != nullptr) {
    for (int peer = 0; peer < comm->nRanks; peer++) {
      while (!ncclIntruQueueEmpty(&peerTaskQueues[peer])) {
        struct ncclTaskRma* task = ncclIntruQueueDequeue(&peerTaskQueues[peer]);
        ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
      }
    }
  }
  ncclCeFreeBatchOpsParams(&dataParams);
  ncclCeFreeBatchOpsParams(&signalParams);
  free(peerTaskQueues);
  free(activePeers);
  free(seqStageOps);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclRmaCePutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  if (plan->persistent) {
    return ncclRmaCePutLaunchPersist(comm, plan, stream);
  }
  return ncclRmaCePutLaunchNonPersist(comm, plan, stream);
}

ncclResult_t ncclRmaCeWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  CUstreamBatchMemOpParams* batchParams = nullptr;
  struct ncclCeBatchOpsParams ceParams = {};

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;

  int lsaSize = comm->devrState.lsaSize;
  int lsaSelf = comm->devrState.lsaSelf;

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

  // A WaitSignal plan is single-context; use the wait task's own context.
  // (Declared before the goto-based checks so they don't cross its initialization.)
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[task->ctx];

  if (task->func != ncclFuncWaitSignal) {
    WARN("RMA CE task function is %d, expected %d", task->func, ncclFuncWaitSignal);
    goto invalid_task;
  }
  if (plan->rmaArgs->nRmaTasksCe != 1) {
    WARN("RMA CE task count is %d, expected 1", plan->rmaArgs->nRmaTasksCe);
    goto invalid_task;
  }

  if (task->signalMode == NCCL_SIGNAL) {
    if (!persistent) {
      // Non-graph: batch one cuStreamWaitValue per peer with incrementing expected value
      NCCLCHECKGOTO(ncclCalloc(&batchParams, task->npeers), ret, fail);
      size_t opIdx = 0;
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, peerRank, &peerLsaRank), ret, fail);
        size_t signalSlot = ncclRmaSignalSlot(lsaSize, task->signalIdxs[i], peerLsaRank);
        uint64_t waitValue = ceCtx->signalsHost[signalSlot] + task->nsignals[i];
        ceCtx->signalsHost[signalSlot] = waitValue;

        CUdeviceptr signalAddr = (CUdeviceptr)&ceCtx->signalsDev[signalSlot];
        batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        batchParams[opIdx].waitValue.address = signalAddr;
        batchParams[opIdx].waitValue.value64 = waitValue;
        batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
        opIdx++;
      }
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);
    } else {
      // Graph: wait-reset-ack cycle using separate graphSignalsDev (isolated from non-graph)
      NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceParams, 1), ret, fail);

      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, peerRank, &peerLsaRank), ret, fail);

        for (int s = 0; s < task->nsignals[i]; s++) {
          size_t signalSlot = ncclRmaSignalSlot(lsaSize, task->signalIdxs[i], peerLsaRank);
          CUdeviceptr graphSignalAddr = (CUdeviceptr)&ceCtx->graphSignalsDev[signalSlot];
          CUstreamBatchMemOpParams signalOps[2] = {};
          signalOps[0].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
          signalOps[0].waitValue.address = graphSignalAddr;
          signalOps[0].waitValue.value64 = 1;
          signalOps[0].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
          signalOps[1].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_64;
          signalOps[1].writeValue.address = graphSignalAddr;
          signalOps[1].writeValue.value64 = 0;
          signalOps[1].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
          NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, 2, signalOps), ret, fail);

          void* peerAck;
          size_t ackOffset = ncclRmaSignalOffset(lsaSize, task->signalIdxs[i], lsaSelf);
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphAckOffset + ackOffset, peerLsaRank,
                                              &peerAck),
                        ret, fail);
          ceParams.numOps = 0;
          ceParams.srcs[ceParams.numOps] = ceCtx->signalConstOneDev;
          ceParams.dsts[ceParams.numOps] = peerAck;
          ceParams.sizes[ceParams.numOps] = sizeof(uint64_t);
          ceParams.numOps++;
          NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceParams, stream), ret, fail);
        }
      }
    }
  }

  // Free the task
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);

exit:
  if (batchParams) free(batchParams);
  ncclCeFreeBatchOpsParams(&ceParams);
  return ret;
invalid_task:
  ret = ncclInternalError;
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
fail:
  goto exit;
}
