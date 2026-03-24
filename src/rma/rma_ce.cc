/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <assert.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "collectives.h"
#include "rma/rma.h"
#include "rma/rma_ce.h"

ncclResult_t ncclRmaCeInit(struct ncclComm* comm){
  ncclResult_t ret = ncclSuccess;
  uint64_t* signalsDevBase = nullptr;
  uint64_t* ackInitHost = nullptr;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  comm->rmaState.rmaCeState.rmaCeCtxCount = comm->config.numRmaCtx;

  NCCLCHECKGOTO(ncclCalloc(&ackInitHost, comm->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rmaState.rmaCeState.rmaCeCtxs, comm->rmaState.rmaCeState.rmaCeCtxCount), ret, fail);
  for (int i = 0; i < comm->rmaState.rmaCeState.rmaCeCtxCount; i++) {
    // Allocate the RMA CE context
    struct ncclRmaCeCtx* ceCtx;
    NCCLCHECKGOTO(ncclCalloc(&ceCtx, 1), ret, fail);
    comm->rmaState.rmaCeState.rmaCeCtxs[i] = ceCtx;

    // Initialize context
    ceCtx->comm = comm;

    // Allocate and register symmetric memory for signals
    int nRanks = comm->nRanks;
    size_t signalsBufSize = (3 * nRanks + 2) * sizeof(uint64_t);
    ncclWindow_vidmem* signalsWinDev;
    ncclWindow_vidmem* signalsWinDevHost;

    NCCLCHECKGOTO(ncclMemAlloc((void**)&signalsDevBase, signalsBufSize), ret, fail);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalsDevBase, signalsBufSize, NCCL_WIN_COLL_SYMMETRIC, &signalsWinDev), ret, fail);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, signalsWinDev, &signalsWinDevHost), ret, fail);

    // Get the ncclDevrWindow from the winHost field
    ceCtx->signalsWin = (struct ncclDevrWindow*)signalsWinDevHost->winHost;
    ceCtx->signalsDev = signalsDevBase;
    ceCtx->graphSignalsDev = signalsDevBase + nRanks + 1;
    ceCtx->graphAckDev = signalsDevBase + 2 * nRanks + 2;
    signalsDevBase = nullptr;
    ceCtx->signalOffset = 0;
    ceCtx->graphSignalOffset = (nRanks + 1) * sizeof(uint64_t);
    ceCtx->graphAckOffset = (2 * nRanks + 2) * sizeof(uint64_t);

    // Initialize ack flags to 1
    for (int r = 0; r < nRanks; r++) ackInitHost[r] = 1;
    CUDACHECKGOTO(cudaMemcpy(ceCtx->graphAckDev, ackInitHost, nRanks * sizeof(uint64_t), cudaMemcpyHostToDevice), ret, fail);

    // Allocate device-resident constants for graph-safe D2D signal/ack writes
    NCCLCHECKGOTO(ncclMemAlloc((void**)&ceCtx->signalConstDev, 2 * sizeof(uint64_t)), ret, fail);
    ceCtx->signalConstZeroDev = &ceCtx->signalConstDev[0];
    ceCtx->signalConstOneDev = &ceCtx->signalConstDev[1];
    {
      uint64_t zeroone[] = {0, 1};
      CUDACHECKGOTO(cudaMemcpy(ceCtx->signalConstDev, zeroone, sizeof(zeroone), cudaMemcpyHostToDevice), ret, fail);
    }

    // Allocate host buffer to track expected non-graph signal values
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalsHost, nRanks + 1), ret, fail);

    // Allocate per-rank operation sequence counters
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalOpSeqs, comm->nRanks), ret, fail);

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
  if (signalsDevBase) ncclMemFree(signalsDevBase);
  goto exit;
}

ncclResult_t ncclRmaCeFinalize(struct ncclComm* comm){
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

    // Free host signals buffer
    if (ceCtx->signalsHost) free(ceCtx->signalsHost);

    // Free device-resident constants
    if (ceCtx->signalConstDev) NCCLCHECKGOTO(ncclMemFree(ceCtx->signalConstDev), ret, fail);

    // Deregister and free signal window
    if (ceCtx->signalsWin) NCCLCHECKGOTO(ncclCommWindowDeregister(comm, ceCtx->signalsWin->vidmem), ret, fail);

    // Free signal device memory
    if (ceCtx->signalsDev) NCCLCHECKGOTO(ncclMemFree(ceCtx->signalsDev), ret, fail);

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

ncclResult_t ncclRmaCePutLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;
  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
    ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

    int peerLsaRank;
    NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, task->peer, &peerLsaRank), ret, fail);

    size_t bytes = task->count * ncclTypeSize(task->datatype);

    // Graph: wait for receiver's ack, then reset ack flag
    if (persistent) {
      CUdeviceptr ackAddr = (CUdeviceptr)&ceCtx->graphAckDev[task->peer];
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

    if (bytes > 0) {
      // Get the peer buffer from the peer window
      void* peerBuff;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, task->peerWinHost, task->peerWinOffset, peerLsaRank, &peerBuff), ret, fail);

      // Validate peer buffer
      if (peerBuff == NULL) {
        WARN("RMA CE: peerBuff is NULL after ncclDevrGetLsaRankPtr");
        ret = ncclInvalidArgument;
        goto fail;
      }

      // Copy the data to the peer buffer
      CUDACHECKGOTO(cudaMemcpyAsync(peerBuff, task->srcBuff, bytes, cudaMemcpyDeviceToDevice, stream), ret, fail);
    }

    // Write signal if needed for the target rank
    // CE over NVL only supports distinct signal
    if (task->signalMode != NCCL_SIGNAL_NONE) {
      size_t rankSlot = comm->rank * sizeof(uint64_t);

      if (!persistent) {
        // Non-graph: write incrementing sequence to peer's signalsDev
        void* peerSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->signalOffset + rankSlot, peerLsaRank, &peerSignal), ret, fail);
        ceCtx->signalOpSeqs[task->peer]++;
        CUDACHECKGOTO(cudaMemcpyAsync(peerSignal, &ceCtx->signalOpSeqs[task->peer], sizeof(uint64_t), cudaMemcpyHostToDevice, stream), ret, fail);
      } else {
        // Graph: write signal=1 to peer's graphSignalsDev (separate from non-graph signals)
        void* peerGraphSignal;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphSignalOffset + rankSlot, peerLsaRank, &peerGraphSignal), ret, fail);
        CUDACHECKGOTO(cudaMemcpyAsync(peerGraphSignal, ceCtx->signalConstOneDev, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream), ret, fail);
      }
    }

    // Free the task after processing
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclRmaCeWaitLaunch(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;
  CUstreamBatchMemOpParams* batchParams = nullptr;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  bool persistent = plan->persistent;
  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];

  struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
  ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

  // Assert task func is ncclFuncWaitSignal
  assert(task->func == ncclFuncWaitSignal);
  // Assert task context is the same as the plan context
  assert(task->ctx == ctx);
  // Assert the plan has exactly one RMA CE task
  assert(plan->rmaArgs->nRmaTasksCe == 1);

  if (task->signalMode == NCCL_SIGNAL) {
    if (!persistent) {
      // Non-graph: batch one cuStreamWaitValue per peer with incrementing expected value
      NCCLCHECKGOTO(ncclCalloc(&batchParams, task->npeers), ret, fail);
      size_t opIdx = 0;
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        uint64_t waitValue = ceCtx->signalsHost[peerRank] + task->nsignals[i];
        ceCtx->signalsHost[peerRank] = waitValue;

        CUdeviceptr signalAddr = (CUdeviceptr)&ceCtx->signalsDev[peerRank];
        batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
        batchParams[opIdx].waitValue.address = signalAddr;
        batchParams[opIdx].waitValue.value64 = waitValue;
        batchParams[opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_GEQ;
        opIdx++;
      }
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);
    }
    else {
      // Graph: wait-reset-ack cycle using separate graphSignalsDev (isolated from non-graph)
      for (int i = 0; i < task->npeers; i++) {
        int peerRank = task->peers[i];
        int peerLsaRank;
        NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, peerRank, &peerLsaRank), ret, fail);

        for (int s = 0; s < task->nsignals[i]; s++) {
          CUdeviceptr graphSignalAddr = (CUdeviceptr)&ceCtx->graphSignalsDev[peerRank];
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
          NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, ceCtx->graphAckOffset + comm->rank * sizeof(uint64_t), peerLsaRank, &peerAck), ret, fail);
          CUDACHECKGOTO(cudaMemcpyAsync(peerAck, ceCtx->signalConstOneDev, sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream), ret, fail);
        }
      }
    }
  }

  // Free the task
  ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);

exit:
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}
