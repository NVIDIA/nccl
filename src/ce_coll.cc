/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "register_inline.h"
#include <cuda.h>
#include "cudawrap.h"
#include "ce_coll.h"
#include "alloc.h"

// Static constant for graph synchronization
static const uint32_t GRAPH_SYNC_VALUE = 1;

// Static constants for intra-batch synchronization to improve CE collective performance with large scale
// Frequency of intra-batch synchronization
static const uint32_t CE_COLL_INTRA_BATCH_SYNC_FREQ = 8;
// Message threshold for intra-batch synchronization
static const uint64_t CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD = 512*1024*1024;

ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  uint8_t* ceDevBase;
  size_t ceDevBaseSize = alignUp(comm->nRanks*sizeof(uint32_t), 16) * 2;
  ncclWindow_vidmem* ceWinDev;
  ncclWindow_vidmem* ceWinDevHost;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  // Allocate and register memory for the symmetric memory
  NCCLCHECKGOTO(ncclMemAlloc((void**)&ceDevBase, ceDevBaseSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret, fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  // Get the ncclDevrWindow from the winHost field
  comm->ceColl.ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;

  comm->ceColl.baseUCSymReadyOffset = 0;
  comm->ceColl.baseUCSymComplOffset = alignUp(comm->nRanks*sizeof(uint32_t), 16);
  comm->ceColl.baseUCSymReadyPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymReadyOffset;
  comm->ceColl.baseUCSymComplPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymComplOffset;
  comm->ceColl.ceSeqNum = 0;
  comm->ceColl.useCompletePtr = false;
  comm->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d", comm->rank, comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum);

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // Clean up ceInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    free(task);
  }

  // Clean up CE resources
  if (comm->ceColl.baseUCSymReadyPtr != NULL) {
    if (comm->ceColl.ceSyncWin && comm->ceColl.ceSyncWin->vidmem) {
      NCCLCHECKGOTO(ncclCommWindowDeregister(comm, comm->ceColl.ceSyncWin->vidmem), ret, fail);
      NCCLCHECKGOTO(ncclMemFree(comm->ceColl.baseUCSymReadyPtr), ret, fail);
    }
    comm->ceColl.baseUCSymReadyPtr = NULL;
    comm->ceColl.baseUCSymComplPtr = NULL;
    comm->ceColl.ceSyncWin = NULL;
  }

exit:
  return ret;
fail:
  goto exit;
}

bool ncclCeImplemented(ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  int driverVersion;
  if (ncclCudaDriverVersion(&driverVersion) != ncclSuccess) return false;

  // CE is supported in CUDA 12.5 and later
  if (driverVersion >= 12050) {
    switch (coll) {
    case ncclFuncAllGather:
    case ncclFuncAlltoAll:
    case ncclFuncScatter:
    case ncclFuncGather:
      return true;
    default:
      return false;
    }
  }
  return false;
}

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int/*ncclDevRedOp_t*/ red, ncclDataType_t ty, ncclSymRegType_t winRegType) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not implemented");
    return false;
  }
  if (comm->nNodes > 1) {
    TRACE(NCCL_TUNING, "Skipping CE collective: comm is not a single node");
    return false;
  }
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping CE collective: symmetric support is not enabled");
    return false;
  }
  if (winRegType != ncclSymSendRegRecvReg && winRegType != ncclSymSendNonregRecvReg) {
    TRACE(NCCL_TUNING, "Skipping CE collective: window registration type %d is not supported", winRegType);
    return false;
  }
  return true;
}

ncclResult_t ncclPrepMCSync(struct ncclComm* comm, bool isComplete, CUstreamBatchMemOpParams* batchParams, size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  uint32_t* readyPtrs    = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Source pointer is either the constant graph sync value or the sequence number
  void* srcPtr = capturing ? (void*)&GRAPH_SYNC_VALUE : (void*)&currentSeq;
  // Wait value is either the constant graph sync value or the sequence number
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  // Use multi-cast address as destination pointer
  void* mcDstPtr;
  void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
  NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, comm->ceColl.ceSyncWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);

  // Write our own ready/complete flag to the multi-cast address
  CUDACHECKGOTO(cudaMemcpyAsync(
    mcDstPtr,
    srcPtr,
    sizeof(uint32_t),
    cudaMemcpyHostToDevice,
    stream), ret, fail);

  // Add local wait operations for every other rank
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = waitValue;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete,
                               CUstreamBatchMemOpParams* batchParams,
                               size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  uint32_t* readyPtrs    = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Write our own ready/complete flag to remote ranks using cudaMemcpyAsync
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    void * peerDstPtr;
    void* dstPtr = isComplete ? (void*)&completePtrs[comm->rank] : (void*)&readyPtrs[comm->rank];
    size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    if (capturing) {
      CUDACHECKGOTO(cudaMemcpyAsync(peerDstPtr, &GRAPH_SYNC_VALUE, sizeof(uint32_t), cudaMemcpyHostToDevice, stream), ret, fail);
    } else {
      CUDACHECKGOTO(cudaMemcpyAsync(peerDstPtr, &currentSeq, sizeof(uint32_t), cudaMemcpyHostToDevice, stream), ret, fail);
    }
  }

  // Add local wait operations for every other rank
  for (int r = 0; r < comm->nRanks; ++r) {
    if (r == comm->rank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address  = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = capturing ? GRAPH_SYNC_VALUE : currentSeq;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclMemOpSync(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  void* ceSyncHandle = NULL;

  // Get pointers to the ready and complete synchronization arrays
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // Allocate enough slots for all possible ops
  size_t batchSize = (comm->nvlsSupport ? NCCL_CE_SYNC_OPS_PER_RANK_MC : NCCL_CE_SYNC_OPS_PER_RANK_UC) * comm->nRanks;
  size_t opIdx = 0;

  // Prepare batch memory operations for synchronization
  CUstreamBatchMemOpParams* batchParams = nullptr;
  NCCLCHECKGOTO(ncclCalloc(&batchParams, batchSize), ret, fail);

  // Start CE sync profiling
  NCCLCHECKGOTO(ncclProfilerStartCeSyncEvent(comm, args, stream, &ceSyncHandle),
                ret, fail);

  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclPrepMCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclPrepUCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  }

  // For CUDA graph capture, add reset operation
  if (ncclCudaGraphValid(comm->planner.capturingGraph)) {
    for (int i = 0; i < comm->nRanks; i++) {
      batchParams[opIdx] = {};
      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[opIdx].writeValue.address = (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      batchParams[opIdx].writeValue.value = 0;
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;
    }
  }

  // Execute all memory operations in a single batch
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);

  // Toggle the flag for next call
  comm->ceColl.useCompletePtr = !comm->ceColl.useCompletePtr;

  // Stop CE sync profiling
  NCCLCHECKGOTO(ncclProfilerStopCeSyncEvent(comm, ceSyncHandle, stream),
                ret, fail);

exit:
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int nRanks) {
  ncclResult_t ret = ncclSuccess;

  params->srcs = nullptr;
  params->dsts = nullptr;
  params->sizes = nullptr;
  params->numOps = 0;
  params->intraBatchSync = false;
#if CUDART_VERSION >= 12080
  params->attrs = nullptr;
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif

  NCCLCHECKGOTO(ncclCalloc(&params->srcs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->dsts, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->sizes, nRanks), ret, fail);
#if CUDART_VERSION >= 12080
  NCCLCHECKGOTO(ncclCalloc(&params->attrs, nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&params->attrIdxs, nRanks), ret, fail);
#endif
exit:
  return ret;
fail:
  goto exit;
}

void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  if (params->srcs) free(params->srcs);
  if (params->dsts) free(params->dsts);
  if (params->sizes) free(params->sizes);
#if CUDART_VERSION >= 12080
  if (params->attrs) free(params->attrs);
  if (params->attrIdxs) free(params->attrIdxs);
#endif
}

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeCollArgs* args,
                                  struct ncclCeBatchOpsParams* params, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  bool capturing;
  int driverVersion;
  void* ceBatchHandle = NULL;

  // Check if there are any operations to perform
  if (params->numOps == 0) {
    return ncclSuccess;
  }

  // Check if we are in a CUDA graph capture
  capturing = ncclCudaGraphValid(comm->planner.capturingGraph);

  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, fail);

  // Start CE batch profiling
  NCCLCHECKGOTO(ncclProfilerStartCeBatchEvent(comm, args, params, stream, &ceBatchHandle),
                ret, fail);

  //--------------Graph capture--------------
  // cudaMemcpyBatchAsync is not supported during CUDA graph capture
  if (capturing) {
    for (int i =0; i < params->numOps; i++) {
      CUDACHECKGOTO(cudaMemcpyAsync(
        (void*)params->dsts[i],
        (void*)params->srcs[i],
        params->sizes[i],
        cudaMemcpyDeviceToDevice,
        stream), ret, fail);

      if (params->intraBatchSync && ((i+1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i+1) < params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
      }
    }
  }
  //--------------No graph capture--------------
  else {
    if (CUDART_VERSION >= 12080 && driverVersion >= 12080) {
#if CUDART_VERSION >= 12080
    // For CUDA 12.8+, use batch memory copy for better performance
    params->attrs[0] = {};
    params->attrs[0].srcAccessOrder = cudaMemcpySrcAccessOrderStream;
    params->attrs[0].flags = cudaMemcpyFlagPreferOverlapWithCompute;
    params->attrIdxs[0] = 0;
    params->numAttrs = 1;

    if (params->intraBatchSync) {
      // Find the maximum transfer size to determine number of rounds
      size_t maxSize = 0;
      size_t totalSize = 0;
      for (int i = 0; i < params->numOps; i++) {
        if (params->sizes[i] > maxSize) {
          maxSize = params->sizes[i];
        }
        totalSize += params->sizes[i];
      }

      size_t chunkSize = comm->ceColl.intraBatchSyncMsgThreshold / params->numOps;
      int numRounds = (maxSize + chunkSize - 1) / chunkSize;

      // Allocate temporary arrays for all chunked operations
      void** tmpDsts = nullptr;
      void** tmpSrcs = nullptr;
      size_t* tmpSizes = nullptr;
      NCCLCHECKGOTO(ncclCalloc(&tmpDsts, params->numOps * numRounds), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&tmpSrcs, params->numOps * numRounds), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&tmpSizes, params->numOps * numRounds), ret, fail);

      int opIdx = 0;
      for (int round = 0; round < numRounds; round++) {
        size_t offset = round * chunkSize;
        // Prepare chunk transfers for this round
        for (int i = 0; i < params->numOps; i++) {
          int index = (i+round) % params->numOps;
          if (offset < params->sizes[index]) {
            size_t remainingSize = params->sizes[index] - offset;
            size_t currentChunkSize = (remainingSize > chunkSize) ? chunkSize : remainingSize;

            tmpDsts[opIdx] = (void*)((uint8_t*)params->dsts[index] + offset);
            tmpSrcs[opIdx] = (void*)((uint8_t*)params->srcs[index] + offset);
            tmpSizes[opIdx] = currentChunkSize;
            opIdx++;
          }
        }
      }

      // Launch a single batch for all chunks
      if (opIdx > 0) {
        #if CUDART_VERSION >= 13000
        CUDACHECKGOTO(cudaMemcpyBatchAsync(
          tmpDsts, tmpSrcs, tmpSizes, opIdx,
          params->attrs, params->attrIdxs, params->numAttrs, stream), ret, fail);
        #else
        CUDACHECKGOTO(cudaMemcpyBatchAsync(
          tmpDsts, tmpSrcs, tmpSizes, opIdx,
          params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream), ret, fail);
        #endif
      }

      // Free temporary arrays
      if (tmpDsts) free(tmpDsts);
      if (tmpSrcs) free(tmpSrcs);
      if (tmpSizes) free(tmpSizes);
    } else {
      // Use single batch for all operations
      #if CUDART_VERSION >= 13000
      CUDACHECKGOTO(cudaMemcpyBatchAsync(
        params->dsts, params->srcs, params->sizes, params->numOps,
        params->attrs, params->attrIdxs, params->numAttrs, stream), ret, fail);
      #else
      CUDACHECKGOTO(cudaMemcpyBatchAsync(
        params->dsts, params->srcs, params->sizes, params->numOps,
        params->attrs, params->attrIdxs, params->numAttrs, nullptr, stream), ret, fail);
      #endif
    }
#endif
    } else {
      // For older CUDA versions, fall back to individual transfers
      for (int i = 0; i < params->numOps; i++) {
        CUDACHECKGOTO(cudaMemcpyAsync(
          (void*)params->dsts[i],
          (void*)params->srcs[i],
          params->sizes[i],
          cudaMemcpyDeviceToDevice,
          stream), ret, fail);

        if (params->intraBatchSync && ((i+1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i+1) < params->numOps)) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);
        }
      }
    }
  }

  // Stop CE batch profiling
  NCCLCHECKGOTO(ncclProfilerStopCeBatchEvent(comm, ceBatchHandle, stream),
                ret, fail);

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args,
                             cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  // Copy own data to receive buffer if operation is out-of-place
  if (myRecvBuff != mySendBuff) {
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Copy data to other ranks
  for (int r = 1; r < comm->nRanks; r++) {
    int targetRank = (comm->rank + r) % comm->nRanks;
    offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Check if we need to perform intra-batch synchronization
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && chunkBytes*batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream),
                ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  // Copy data to other ranks: send data chunk for each destination rank
  for (int r = 0; r < comm->nRanks; r++) {
    int dstRank = (comm->rank + r) % comm->nRanks;
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;

    if (dstRank == comm->rank) {
      // Local copy for own data
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    } else {
      // Remote copy to other ranks: send to rank dstRank's receive buffer at position comm->rank
      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }

  // Check if we need to perform intra-batch synchronization
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq && chunkBytes*batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerDstPtr;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, comm->nRanks), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Check if this is an in-place scatter operation
    bool isInPlace = (myRecvBuff == mySendBuff + comm->rank * chunkBytes);

    // Copy root's own data first if not in-place
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + comm->rank * chunkBytes;
      uint8_t* dstPtr = myRecvBuff;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // Root rank distributes data to other ranks
    for (int r = 1; r < comm->nRanks; r++) {
      int dstRank = (comm->rank + r) % comm->nRanks;
      uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
      uint8_t* dstPtr = isInPlace ? myRecvBuff + dstRank * chunkBytes : myRecvBuff;

      offset = dstPtr - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, dstRank, &peerDstPtr), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerDstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  }
  // Non-root ranks don't need to perform any copy operations

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootRank = args->rootRank;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

  if (comm->rank == rootRank) {
    // Root rank copies its own data to the correct position in receive buffer
    uint8_t* dstPtr = myRecvBuff + comm->rank * chunkBytes;
    if (mySendBuff != dstPtr) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // Non-root ranks send their data to root's receive buffer
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + comm->rank * chunkBytes;
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, args, &batchOpsParams, stream), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, args, stream), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;
  struct ncclCeCollArgs* args = plan->ceCollArgs;

  // Start CE collective profiling
  NCCLCHECKGOTO(ncclProfilerStartCeCollEvent(comm, args, stream),
                ret, fail);

  switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream),
                    ret, fail);
      break;
    case ncclFuncAlltoAll:
      NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream),
                    ret, fail);
      break;
    case ncclFuncScatter:
      NCCLCHECKGOTO(ncclCeScatter(comm, args, stream),
                    ret, fail);
      break;
    case ncclFuncGather:
      NCCLCHECKGOTO(ncclCeGather(comm, args, stream),
                    ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
  }

  // Stop CE collective profiling
  NCCLCHECKGOTO(ncclProfilerStopCeCollEvent(comm, args, stream),
                ret, fail);

exit:
  return ret;
fail:
  goto exit;
}
