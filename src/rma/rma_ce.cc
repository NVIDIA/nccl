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

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);

  comm->rmaState.rmaCeState.rmaCeCtxCount = comm->config.numRmaCtx;

  NCCLCHECKGOTO(ncclCalloc(&comm->rmaState.rmaCeState.rmaCeCtxs, comm->rmaState.rmaCeState.rmaCeCtxCount), ret, fail);
  for (int i = 0; i < comm->rmaState.rmaCeState.rmaCeCtxCount; i++) {
    // Allocate the RMA CE context
    struct ncclRmaCeCtx* ceCtx;
    NCCLCHECKGOTO(ncclCalloc(&ceCtx, 1), ret, fail);
    comm->rmaState.rmaCeState.rmaCeCtxs[i] = ceCtx;

    // Initialize context
    ceCtx->comm = comm;

    // Allocate and register symmetric memory for signals
    // Signal buffer layout: [0..nRanks-1] per-rank signals, [nRanks] aggregate signal
    size_t signalsBufSize = (comm->nRanks + 1) * sizeof(uint64_t);
    uint64_t* signalsDevBase;
    ncclWindow_vidmem* signalsWinDev;
    ncclWindow_vidmem* signalsWinDevHost;

    NCCLCHECKGOTO(ncclMemAlloc((void**)&signalsDevBase, signalsBufSize), ret, fail);
    NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, signalsDevBase, signalsBufSize, NCCL_WIN_COLL_SYMMETRIC, &signalsWinDev), ret, fail);
    NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, signalsWinDev, &signalsWinDevHost), ret, fail);

    // Get the ncclDevrWindow from the winHost field
    ceCtx->signalsWin = (struct ncclDevrWindow*)signalsWinDevHost->winHost;
    ceCtx->signalsDev = signalsDevBase;

    // Allocate host buffer to track expected signal values
    NCCLCHECKGOTO(ncclCalloc(&ceCtx->signalsHost, signalsBufSize), ret, fail);

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
  return ret;
fail:
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

ncclResult_t ncclRmaPutCe(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

  int nRmaTasksCe = plan->rmaArgs->nRmaTasksCe;
  int ctx = plan->rmaArgs->ctx;
  struct ncclRmaCeCtx* ceCtx = (struct ncclRmaCeCtx*)comm->rmaState.rmaCeState.rmaCeCtxs[ctx];

  for (int i = 0; i < nRmaTasksCe; i++) {
    struct ncclTaskRma* task = ncclIntruQueueHead(&plan->rmaTaskQueueCe);
    ncclIntruQueueDequeue(&plan->rmaTaskQueueCe);

    // Convert global peer rank to LSA rank index
    // LSA rank is computed as: peer % lsaSize (see dev_runtime.cc)
    int peerLsaRank = task->peer % comm->devrState.lsaSize;

    size_t bytes = task->count * ncclTypeSize(task->datatype);

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
      // Get the signal location in peer's signal buffer where we write to notify them
      // We write to offset [comm->rank] in peer's signal buffer, same as proxy version
      // So peer waits on their signalsDev[comm->rank] to see our signals
      void* peerSignal;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, ceCtx->signalsWin, comm->rank * sizeof(uint64_t), peerLsaRank, &peerSignal), ret, fail);

      // Increment our sequence number for operations to this peer
      ceCtx->signalOpSeqs[task->peer]++;

      // Write the absolute sequence number - peer will wait for this value
      CUDACHECKGOTO(cudaMemcpyAsync(peerSignal, &ceCtx->signalOpSeqs[task->peer], sizeof(uint64_t), cudaMemcpyHostToDevice, stream), ret, fail);
    }

    // Free the task after processing
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, task);
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclRmaWaitSignalCe(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // Make sure the RMA CE is initialized
  if (!comm->rmaState.rmaCeState.initialized) {
    WARN("RMA CE is not initialized");
    return ncclInternalError;
  }

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

  size_t opIdx = 0;
  CUstreamBatchMemOpParams* batchParams = nullptr;

  NCCLCHECK(ncclCalloc(&batchParams, task->npeers));

  // NVL only supports per-rank signal
  if (task->signalMode == NCCL_SIGNAL) {
    for (int i = 0; i < task->npeers; i++) {
      int peerRank = task->peers[i];

      // Calculate the expected signal value from this peer
      // We wait on signalsDev[peerRank] where peerRank writes their sequence numbers
      uint64_t waitValue = ceCtx->signalsHost[peerRank] + task->nsignals[i];

      // Update our expectation for future waits
      ceCtx->signalsHost[peerRank] = waitValue;

      // Add wait operation to batch
      // Wait on our local signal buffer at offset [peerRank] where peer writes to us
      batchParams[opIdx] = {};
      batchParams[opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_64;
      batchParams[opIdx].waitValue.address = (CUdeviceptr)&ceCtx->signalsDev[peerRank];
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
