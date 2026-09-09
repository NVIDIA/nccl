/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "register_inline.h"
#include <cuda.h>
#include "cudawrap.h"
#include "ce_coll.h"
#include "alloc.h"
#include "tuning.h"

// User override: when set (>= 0) the cost model is bypassed and this byte
// threshold decides multicast (sendSize <= threshold). -1 (unset) -> cost model.
NCCL_PARAM(CeCollAgMulticastThreshold, "CE_COLL_AG_MULTICAST_THRESHOLD", -1);

// Static constant for graph synchronization
static const uint32_t GRAPH_SYNC_VALUE = 1;

// Static constants for intra-batch synchronization to improve CE collective performance with large scale
// Frequency of intra-batch synchronization
static const uint32_t CE_COLL_INTRA_BATCH_SYNC_FREQ = 8;
// Message threshold for intra-batch synchronization
static const uint64_t CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD = 512 * 1024 * 1024;

// Maximum size of a single sub-chunk for hierarchical collective
static constexpr size_t HIER_COLL_MAX_CHUNK_SIZE = 64 * 1024 * 1024;
// Alignment of hierarchical-collective sub-chunks. Shared by the chunk-plan
// builder and the chunk-width computation, which must agree on it.
static constexpr size_t HIER_COLL_CHUNK_ALIGN = 8 * 1024;

// Minimum per-peer transfer size (bytes) for the hierarchical CE collectives to
// distribute their inter-node rail traffic across multiple internal RMA
// contexts. Below the threshold the per-context launch/progress overhead
// dominates, so the whole transfer stays on a single context. This only gates
// how many of the provisioned contexts a given collective USES -- the internal
// contexts themselves are always created at connect (ncclRmaProxyConnectOnce),
// so later, larger transfers distribute regardless of what ran before.
// 0 distributes regardless of size; -1 selects the built-in default. Like other
// NCCL tuning variables, it must be set identically on all ranks.
NCCL_PARAM(RmaMultiCtxThreshold, "RMA_MULTI_CTX_THRESHOLD", -1);
static constexpr int64_t HIER_COLL_MULTI_CTX_THRESHOLD_DEFAULT = 4 * 1024 * 1024;

// Number of internal RMA contexts used by hierarchical CE collectives.
// Values above NCCL_NUM_RMA_INT_CTX are clamped.
NCCL_PARAM(HierCeCollNumCtx, "HIER_CE_COLL_NUM_CTX", -1);

// Decide multicast vs unicast for CE AllGather: a CE-only tuning mask lets
// ncclTuningCompute pick the fastest CE method (UC vs MC).
int ncclCeAllGatherUseMulticast(struct ncclComm* comm, size_t perRankBytes, int captured, int inPlace) {
  if (!comm->symkState.hasLsaMultimem) return 0;

  int64_t thresholdOverride = comm->ceColl.agMulticastThreshold;
  if (thresholdOverride >= 0) {
    return ((int64_t)perRankBytes <= thresholdOverride) ? 1 : 0;
  }

  struct ncclTuningInput_t input = {};
  input.comm = comm;
  input.tuningMask = NCCL_TUNING_MASK_CE;
  input.func = ncclFuncAllGather;
  input.datatype = ncclInt8;
  input.nBytes = perRankBytes;
  input.count = perRankBytes;
  input.captured = captured;
  input.inPlace = inPlace;

  struct ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
  if (ncclTuningCompute(&input, &result) != ncclSuccess) return 0;
  return (result.ceMethodId == ncclCeMethodId_AllGather_MC) ? 1 : 0;
}

ncclResult_t ncclCeInit(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  uint8_t* ceDevBase = nullptr;
  // Sync window has lsaSize slots (one per LSA-local rank): one ready array + one complete array
  size_t ceDevBaseSize = alignUp(comm->devrState.lsaSize * sizeof(uint32_t), 16) * 2;
  ncclWindow_vidmem* ceWinDev = nullptr;
  ncclWindow_vidmem* ceWinDevHost = nullptr;

  // Ensure symmetric memory runtime is initialized
  NCCLCHECKGOTO(ncclDevrInitOnce(comm), ret, fail);
  // Allocate and register memory for the symmetric memory
  NCCLCHECKGOTO(ncclCudaCalloc((void**)&ceDevBase, ceDevBaseSize, comm->memManager), ret, fail);
  NCCLCHECKGOTO(ncclDevrWindowRegisterInGroup(comm, ceDevBase, ceDevBaseSize, NCCL_WIN_COLL_SYMMETRIC, &ceWinDev), ret,
                fail);
  NCCLCHECKGOTO(ncclShadowPoolToHost(&comm->devrState.shadows, ceWinDev, &ceWinDevHost), ret, fail);
  NCCLCHECKGOTO(ncclCudaCalloc(&comm->ceColl.ceSeqNumDev, 2, comm->memManager), ret, fail);
  // Get the ncclDevrWindow from the winHost field
  comm->ceColl.ceSyncWin = (struct ncclDevrWindow*)ceWinDevHost->winHost;

  comm->ceColl.baseUCSymReadyOffset = 0;
  comm->ceColl.baseUCSymComplOffset = alignUp(comm->devrState.lsaSize * sizeof(uint32_t), 16);
  comm->ceColl.baseUCSymReadyPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymReadyOffset;
  comm->ceColl.baseUCSymComplPtr = (uint8_t*)comm->ceColl.ceSyncWin->userPtr + comm->ceColl.baseUCSymComplOffset;
  comm->ceColl.ceSeqNum = 0;
  comm->ceColl.useCompletePtr = false;
  comm->ceColl.intraBatchSyncFreq = CE_COLL_INTRA_BATCH_SYNC_FREQ;
  comm->ceColl.intraBatchSyncMsgThreshold = CE_COLL_INTRA_BATCH_SYNC_MSG_THRESHOLD;
  comm->ceColl.agMulticastThreshold = (int64_t)ncclParamCeCollAgMulticastThreshold();
  comm->ceColl.initialized = true;
  NCCLCHECKGOTO(ncclCudaMemcpy(comm->ceColl.ceSeqNumDev + 1, (uint32_t*)&GRAPH_SYNC_VALUE, 1), ret, fail);
  INFO(NCCL_INIT, "Init CE, rank %d baseUCSymReadyPtr %p, baseUCSymComplPtr %p, seq num %d", comm->rank,
       comm->ceColl.baseUCSymReadyPtr, comm->ceColl.baseUCSymComplPtr, comm->ceColl.ceSeqNum);

exit:
  return ret;
fail:
  comm->ceColl.initialized = false;
  ncclCudaFree(comm->ceColl.ceSeqNumDev, comm->memManager);
  // Clean up partial initialization - both functions handle null safely
  ncclCommWindowDeregister(comm, ceWinDev);
  ncclCudaFree(ceDevBase, comm->memManager);
  goto exit;
}

ncclResult_t ncclCeFinalize(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;

  // Clean up ceInitTaskQueue
  while (!ncclIntruQueueEmpty(&comm->ceInitTaskQueue)) {
    struct ncclCeInitTask* task = ncclIntruQueueDequeue(&comm->ceInitTaskQueue);
    free(task);
  }

  // Clean up CE resources - continue cleanup even on errors to avoid leaks
  // Note: both functions handle null safely
  NCCLCHECKIGNORE(ncclCommWindowDeregister(comm, comm->ceColl.ceSyncWin ? comm->ceColl.ceSyncWin->vidmem : nullptr),
                  ret);
  NCCLCHECKIGNORE(ncclCudaFree(comm->ceColl.baseUCSymReadyPtr, comm->memManager), ret);
  NCCLCHECKIGNORE(ncclCudaFree(comm->ceColl.ceSeqNumDev, comm->memManager), ret);

  comm->ceColl.ceSeqNumDev = nullptr;
  comm->ceColl.baseUCSymReadyPtr = nullptr;
  comm->ceColl.baseUCSymComplPtr = nullptr;
  comm->ceColl.ceSyncWin = nullptr;
  comm->ceColl.initialized = false;

  return ret;
}

bool ncclCeImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
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

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                     ncclSymRegType_t winRegType, struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not implemented");
    return false;
  }
  if (ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin)) {
    TRACE(NCCL_TUNING, "Skipping CE collective: host-backed cuMem segments are not supported");
    return false;
  }
  if (ncclTeamLsa(comm).nRanks < comm->nRanks) {
    TRACE(NCCL_TUNING, "Skipping CE collective: not all ranks have NVLink connectivity");
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

ncclResult_t ncclPrepMCSync(struct ncclComm* comm, bool isComplete, CUstreamBatchMemOpParams* batchParams,
                            size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Wait value is either the constant graph sync value or the sequence number
  uint32_t waitValue = capturing ? GRAPH_SYNC_VALUE : currentSeq;

  // Use multi-cast address as destination pointer
  void* mcDstPtr;
  void* dstPtr = isComplete ? (void*)&completePtrs[myLsaRank] : (void*)&readyPtrs[myLsaRank];
  size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
  NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, comm->ceColl.ceSyncWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);

  // Store the updated sequence number in the device buffer.
  if (!capturing) {
    CUCHECKGOTO(cuStreamWriteValue32(stream, (CUdeviceptr)comm->ceColl.ceSeqNumDev, currentSeq,
                                     CU_STREAM_WRITE_VALUE_DEFAULT),
                ret, fail);
  }

  // Write our own ready/complete flag to the multi-cast address
  CUDACHECKGOTO(cudaMemcpyAsync(mcDstPtr, comm->ceColl.ceSeqNumDev + capturing, sizeof(uint32_t),
                                cudaMemcpyDeviceToDevice, stream),
                ret, fail);

  // Add local wait operations for every other rank
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
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

ncclResult_t ncclPrepUCSync(struct ncclComm* comm, bool isComplete, CUstreamBatchMemOpParams* batchParams,
                            size_t* opIdx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  bool capturing = ncclCudaGraphValid(comm->planner.capturingGraph);
  uint32_t currentSeq = ++comm->ceColl.ceSeqNum;

  // Store the updated sequence number in the device buffer.
  if (!capturing) {
    CUCHECKGOTO(cuStreamWriteValue32(stream, (CUdeviceptr)comm->ceColl.ceSeqNumDev, currentSeq,
                                     CU_STREAM_WRITE_VALUE_DEFAULT),
                ret, fail);
  }
  // Write our own ready/complete flag to remote ranks using cudaMemcpyAsync
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
    void* peerDstPtr;
    void* dstPtr = isComplete ? (void*)&completePtrs[myLsaRank] : (void*)&readyPtrs[myLsaRank];
    size_t offset = (uint8_t*)dstPtr - (uint8_t*)comm->ceColl.ceSyncWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, comm->ceColl.ceSyncWin, offset, r, &peerDstPtr), ret, fail);
    CUDACHECKGOTO(cudaMemcpyAsync(peerDstPtr, comm->ceColl.ceSeqNumDev + capturing, sizeof(uint32_t),
                                  cudaMemcpyDeviceToDevice, stream),
                  ret, fail);
  }

  // Add local wait operations for every other rank
  for (int r = 0; r < lsaSize; ++r) {
    if (r == myLsaRank) continue;
    batchParams[*opIdx] = {};
    batchParams[*opIdx].waitValue.operation = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batchParams[*opIdx].waitValue.address = (CUdeviceptr)(isComplete ? (void*)&completePtrs[r] : (void*)&readyPtrs[r]);
    batchParams[*opIdx].waitValue.value = capturing ? GRAPH_SYNC_VALUE : currentSeq;
    batchParams[*opIdx].waitValue.flags = CU_STREAM_WAIT_VALUE_EQ;
    (*opIdx)++;
  }

exit:
  return ret;
fail:
  goto exit;
}

// Intra-LSA-rank synchronization through memory operations.
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, struct ncclCeCollArgs* profilerArgs) {
  ncclResult_t ret = ncclSuccess;
  void* ceSyncHandle = NULL;
  int lsaSize = comm->devrState.lsaSize;

  // Get pointers to the ready and complete synchronization arrays
  uint32_t* readyPtrs = (uint32_t*)comm->ceColl.baseUCSymReadyPtr;
  uint32_t* completePtrs = (uint32_t*)comm->ceColl.baseUCSymComplPtr;

  // Allocate enough slots for all possible ops
  // We follow the built-in symmetric kernels on whether to use NVLS or not.
  bool useMCSync = comm->symkState.hasLsaMultimem;
  size_t batchSize = (useMCSync ? NCCL_CE_SYNC_OPS_PER_RANK_MC : NCCL_CE_SYNC_OPS_PER_RANK_UC) * lsaSize;
  size_t opIdx = 0;
  CUstreamBatchMemOpParams* batchParams = nullptr;

  // Start CE sync profiling (no-op if profilerArgs is nullptr)
  NCCLCHECKGOTO(ncclProfilerStartCeSyncEvent(comm, profilerArgs, stream, &ceSyncHandle), ret, fail);

  // Prepare batch memory operations for synchronization
  NCCLCHECKGOTO(ncclCalloc(&batchParams, batchSize), ret, fail);

  if (useMCSync) {
    NCCLCHECKGOTO(ncclPrepMCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  } else {
    NCCLCHECKGOTO(ncclPrepUCSync(comm, comm->ceColl.useCompletePtr, batchParams, &opIdx, stream), ret, fail);
  }

  // For CUDA graph capture, add reset operation
  if (ncclCudaGraphValid(comm->planner.capturingGraph)) {
    for (int i = 0; i < lsaSize; i++) {
      batchParams[opIdx] = {};
      batchParams[opIdx].writeValue.operation = CU_STREAM_MEM_OP_WRITE_VALUE_32;
      batchParams[opIdx].writeValue.address =
        (CUdeviceptr)(comm->ceColl.useCompletePtr ? (void*)&completePtrs[i] : (void*)&readyPtrs[i]);
      batchParams[opIdx].writeValue.value = 0;
      batchParams[opIdx].writeValue.flags = CU_STREAM_WRITE_VALUE_DEFAULT;
      opIdx++;
    }
  }

  // Execute all memory operations in a single batch
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, opIdx, batchParams), ret, fail);

  // Toggle the flag for next call
  comm->ceColl.useCompletePtr = !comm->ceColl.useCompletePtr;

exit:
  // Stop CE sync profiling - always attempt if started, even on error
  ncclProfilerStopCeSyncEvent(comm, ceSyncHandle, stream);
  if (batchParams) free(batchParams);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity) {
  ncclResult_t ret = ncclSuccess;

  void** srcs = nullptr;
  void** dsts = nullptr;
  size_t* sizes = nullptr;
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes* attrs = nullptr;
  size_t* attrIdxs = nullptr;
#endif

  NCCLCHECKGOTO(ncclCalloc(&srcs, capacity), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&dsts, capacity), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&sizes, capacity), ret, fail);
#if CUDART_VERSION >= 12080
  NCCLCHECKGOTO(ncclCalloc(&attrs, capacity), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&attrIdxs, capacity), ret, fail);
#endif

exit:
  params->srcs = srcs;
  params->dsts = dsts;
  params->sizes = sizes;
  params->numOps = 0;
  params->intraBatchSync = false;
#if CUDART_VERSION >= 12080
  params->attrs = attrs;
  params->attrIdxs = attrIdxs;
  params->numAttrs = 0;
#endif
  return ret;
fail:
  if (srcs) free(srcs);
  srcs = nullptr;
  if (dsts) free(dsts);
  dsts = nullptr;
  if (sizes) free(sizes);
  sizes = nullptr;
#if CUDART_VERSION >= 12080
  if (attrs) free(attrs);
  attrs = nullptr;
  if (attrIdxs) free(attrIdxs);
  attrIdxs = nullptr;
#endif
  goto exit;
}

void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params) {
  if (params->srcs) free(params->srcs);
  params->srcs = nullptr;
  if (params->dsts) free(params->dsts);
  params->dsts = nullptr;
  if (params->sizes) free(params->sizes);
  params->sizes = nullptr;
  params->numOps = 0;
  params->intraBatchSync = false;
#if CUDART_VERSION >= 12080
  if (params->attrs) free(params->attrs);
  params->attrs = nullptr;
  if (params->attrIdxs) free(params->attrIdxs);
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif
}

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* profilerArgs) {
  ncclResult_t ret = ncclSuccess;
  bool capturing;
  int driverVersion;
  void* ceBatchHandle = NULL;

  // cudaMemcpyBatchAsync does not accept the legacy null stream (e.g. PyTorch null stream).
  // Fall back to cudaMemcpyAsync per-op when stream is NULL.
  bool isLegacyStream;
  NCCLCHECKGOTO(ncclCudaStreamIsLegacyNull(stream, &isLegacyStream), ret, fail);

  // Start CE batch profiling (no-op if profilerArgs is nullptr)
  NCCLCHECKGOTO(ncclProfilerStartCeBatchEvent(comm, profilerArgs, params, stream, &ceBatchHandle), ret, fail);

  // Check if there are any operations to perform
  if (params->numOps == 0) goto exit;

  // Check if we are in a CUDA graph capture
  capturing = ncclCudaGraphValid(comm->planner.capturingGraph);

  NCCLCHECKGOTO(ncclCudaDriverVersion(&driverVersion), ret, fail);

  //--------------Graph capture / legacy stream--------------
  // cudaMemcpyBatchAsync is not supported during CUDA graph capture or with legacy stream
  if (capturing || isLegacyStream) {
    for (int i = 0; i < params->numOps; i++) {
      CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                    cudaMemcpyDeviceToDevice, stream),
                    ret, fail);

      if (params->intraBatchSync && ((i + 1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i + 1) < params->numOps)) {
        NCCLCHECKGOTO(ncclMemOpSync(comm, stream, profilerArgs), ret, fail);
      }
    }
    // WORKAROUND: This is a workaround to ensure that there is always an even number of intra-batch
    // synchronization operations.
    if (params->intraBatchSync &&
        ((params->numOps + comm->ceColl.intraBatchSyncFreq - 1) / comm->ceColl.intraBatchSyncFreq) % 2 == 0) {
      NCCLCHECKGOTO(ncclMemOpSync(comm, stream, profilerArgs), ret, fail);
    }
  }
  //--------------No graph capture / not legacy stream--------------
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

        size_t numTmpOps = params->numOps * numRounds;

      // Allocate temporary arrays for all chunked operations
      // Use ncclUniqueArrayPtr for automatic cleanup on any exit path
        ncclUniqueArrayPtr<void*> tmpDsts{nullptr};
        ncclUniqueArrayPtr<void*> tmpSrcs{nullptr};
        ncclUniqueArrayPtr<size_t> tmpSizes{nullptr};

        NCCLCHECKGOTO(ncclCalloc(tmpDsts, numTmpOps), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(tmpSrcs, numTmpOps), ret, fail);
        NCCLCHECKGOTO(ncclCalloc(tmpSizes, numTmpOps), ret, fail);

        int opIdx = 0;
        for (int round = 0; round < numRounds; round++) {
          size_t offset = round * chunkSize;
        // Prepare chunk transfers for this round
          for (int i = 0; i < params->numOps; i++) {
            int index = (i + round) % params->numOps;
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
          CUDACHECKGOTO(cudaMemcpyBatchAsync(tmpDsts.get(), tmpSrcs.get(), tmpSizes.get(), opIdx, params->attrs,
                                             params->attrIdxs, params->numAttrs, stream),
                        ret, fail);
#else
          CUDACHECKGOTO(cudaMemcpyBatchAsync(tmpDsts.get(), tmpSrcs.get(), tmpSizes.get(), opIdx, params->attrs,
                                             params->attrIdxs, params->numAttrs, nullptr, stream),
                        ret, fail);
#endif
        }
      } else {
      // Use single batch for all operations
#if CUDART_VERSION >= 13000
        CUDACHECKGOTO(cudaMemcpyBatchAsync(params->dsts, params->srcs, params->sizes, params->numOps, params->attrs,
                                           params->attrIdxs, params->numAttrs, stream),
                      ret, fail);
#else
        CUDACHECKGOTO(cudaMemcpyBatchAsync(params->dsts, params->srcs, params->sizes, params->numOps, params->attrs,
                                           params->attrIdxs, params->numAttrs, nullptr, stream),
                      ret, fail);
#endif
      }
#endif
    } else {
      // For older CUDA versions, fall back to individual transfers
      for (int i = 0; i < params->numOps; i++) {
        CUDACHECKGOTO(cudaMemcpyAsync((void*)params->dsts[i], (void*)params->srcs[i], params->sizes[i],
                                      cudaMemcpyDeviceToDevice, stream),
                      ret, fail);

        if (params->intraBatchSync && ((i + 1) % comm->ceColl.intraBatchSyncFreq == 0) && ((i + 1) < params->numOps)) {
          NCCLCHECKGOTO(ncclMemOpSync(comm, stream, profilerArgs), ret, fail);
        }
      }
    }
  }

exit:
  // Stop CE batch profiling - always attempt if started, even on error
  ncclProfilerStopCeBatchEvent(comm, ceBatchHandle, stream);
  return ret;
fail:
  goto exit;
}

// AllGather across the LSA team (intra-node only).
ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff + myLsaRank * chunkBytes;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};

  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // declare-then-assign: NCCLCHECKGOTO's goto can't cross a scalar initialization.
  bool agUseMulticast;
  agUseMulticast = ncclCeAllGatherUseMulticast(comm, chunkBytes, ncclCudaGraphValid(comm->planner.capturingGraph),
                                               mySendBuff == myRecvBuff);

  if (agUseMulticast) {
    // Multicast path: a single write to the multicast pointer covers
    // every rank in the LSA team (including self), so no self-copy and
    // no incast — incast never happens for multicast.
    void* mcDstPtr;
    offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, args->recvWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)mcDstPtr;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
    batchOpsParams.intraBatchSync = false;
  } else {
    // Unicast path (original behaviour).
    // Copy own data to receive buffer if operation is out-of-place.
    if (myRecvBuff != mySendBuff) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)myRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
    // Copy data to other ranks.
    for (int r = 1; r < lsaSize; r++) {
      int targetRank = (myLsaRank + r) % lsaSize;
      offset = myRecvBuff - (uint8_t*)args->recvWin->userPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, targetRank, &peerRecvBuff), ret, fail);
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
    batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                     chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// AlltoAll across the LSA team (intra-node only).
ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // Copy data to other ranks: send data chunk for each destination rank
  for (int r = 0; r < lsaSize; r++) {
    int dstRank = (myLsaRank + r) % lsaSize;
    uint8_t* srcPtr = mySendBuff + dstRank * chunkBytes;
    uint8_t* dstPtr = myRecvBuff + myLsaRank * chunkBytes;

    if (dstRank == myLsaRank) {
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
  batchOpsParams.intraBatchSync = (batchOpsParams.numOps > comm->ceColl.intraBatchSyncFreq &&
                                   chunkBytes * batchOpsParams.numOps >= comm->ceColl.intraBatchSyncMsgThreshold);

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// Scatter across the LSA team (intra-node only).
ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootLsaRank;
  void* peerDstPtr;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, lsaSize), ret, fail);
  NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, args->rootRank, &rootLsaRank), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (myLsaRank == rootLsaRank) {
    // Check if this is an in-place scatter operation
    bool isInPlace = (myRecvBuff == mySendBuff + myLsaRank * chunkBytes);

    // Copy root's own data first if not in-place
    if (!isInPlace) {
      uint8_t* srcPtr = mySendBuff + myLsaRank * chunkBytes;
      uint8_t* dstPtr = myRecvBuff;
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)srcPtr;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }

    // Root rank distributes data to other ranks
    for (int r = 1; r < lsaSize; r++) {
      int dstRank = (myLsaRank + r) % lsaSize;
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
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

// Gather across the LSA team (intra-node only).
ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int myLsaRank = comm->devrState.lsaSelf;
  // Calculate the size of data each rank sends to every other rank
  const size_t chunkBytes = args->nElts * args->eltSize;
  uint8_t* mySendBuff = (uint8_t*)args->sendBuff;
  uint8_t* myRecvBuff = (uint8_t*)args->recvBuff;
  int rootLsaRank;
  void* peerRecvBuff;
  size_t offset;
  struct ncclCeBatchOpsParams batchOpsParams = {};
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&batchOpsParams, 1), ret, fail);
  NCCLCHECKGOTO(ncclDevrWorldToLsaRank(comm, args->rootRank, &rootLsaRank), ret, fail);

  // Ensure all ranks are ready before starting transfers
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (myLsaRank == rootLsaRank) {
    // Root rank copies its own data to the correct position in receive buffer
    uint8_t* dstPtr = myRecvBuff + myLsaRank * chunkBytes;
    if (mySendBuff != dstPtr) {
      batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
      batchOpsParams.dsts[batchOpsParams.numOps] = (void*)dstPtr;
      batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
      batchOpsParams.numOps++;
    }
  } else {
    // Non-root ranks send their data to root's receive buffer
    uint8_t* rootRecvPtr = (uint8_t*)args->recvBuff + myLsaRank * chunkBytes;
    offset = rootRecvPtr - (uint8_t*)args->recvWin->userPtr;
    NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, args->recvWin, offset, rootLsaRank, &peerRecvBuff), ret, fail);
    batchOpsParams.srcs[batchOpsParams.numOps] = (void*)mySendBuff;
    batchOpsParams.dsts[batchOpsParams.numOps] = (void*)peerRecvBuff;
    batchOpsParams.sizes[batchOpsParams.numOps] = chunkBytes;
    batchOpsParams.numOps++;
  }

  // Launch the batch operations
  NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &batchOpsParams, stream, args), ret, fail);

  // Ensure all transfers are complete across all ranks
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&batchOpsParams);
  return ret;
fail:
  goto exit;
}

bool ncclHierCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                         ncclSymRegType_t winRegType, struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin) {
  if (!ncclCeImplemented(coll, red, ty)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: not implemented");
    return false;
  }
  if (ncclDevrWindowHasSysmemSegment(sendWin) || ncclDevrWindowHasSysmemSegment(recvWin)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: host-backed cuMem segments are not supported");
    return false;
  }
  if (coll != ncclFuncAllGather && coll != ncclFuncAlltoAll) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: only AllGather and AlltoAll are supported");
    return false;
  }

  // Must be multi-node (single-node uses the regular CE path)
  if (comm->nNodes <= 1) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: not multi-node");
    return false;
  }
  // If LSA already spans the whole comm, use CE path instead
  if (ncclDevrIsOneLsaTeam(comm)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: LSA spans the comm; use CE path instead");
    return false;
  }
  // Intra-node CE scatter writes via LSA pointers
  if (ncclTeamLsa(comm).nRanks < comm->localRanks) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: LSA team does not cover all local ranks");
    return false;
  }
  // Need symmetric support
  if (!comm->symmetricSupport) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: symmetric support is not enabled");
    return false;
  }
  // Need RMA proxy for inter-node puts, and the internal RMA contexts that back
  // the rail step (provisioned under exactly this path's guard conditions). This
  // is independent of the user's config.numRmaCtx -- the rail uses the internal
  // context range, so numRmaCtx == 0 is fine.
  if (!comm->hostRmaSupport || !ncclRmaWantInternalCtx(comm)) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: RMA proxy not available");
    return false;
  }
  // Need registered windows for both send and recv buffers
  if (winRegType != ncclSymSendRegRecvReg) {
    TRACE(NCCL_TUNING, "Skipping hierarchical CE collective: window registration type %d not supported", winRegType);
    return false;
  }
  return true;
}

// Per-(peer, chunk) chunking plan in flat form. Peer p's chunks
// span [chunkStart[p], chunkStart[p+1]); total chunks = chunkStart[nPeers].
struct ncclHierChunkPlan {
  int nPeers;
  int* chunkStart;   // [nPeers + 1]  -- prefix sums
  size_t* chunkBytes;   // [chunkStart[nPeers]]  -- per-chunk byte size
  size_t* chunkOff;     // [chunkStart[nPeers]]  -- per-chunk offset within
                         //                          peer's perRankBytes slice
};

// Build a uniform chunking plan
// Every peer gets the same chunk list, last chunk per peer absorbs the remainder.
static ncclResult_t ncclHierCollBuildChunk(size_t perRankBytes, int nPeers, size_t maxChunk,
                                           struct ncclHierChunkPlan* outPlan) {
  ncclResult_t ret = ncclSuccess;
  const size_t align = HIER_COLL_CHUNK_ALIGN;

  outPlan->nPeers = nPeers;
  outPlan->chunkStart = nullptr;
  outPlan->chunkBytes = nullptr;
  outPlan->chunkOff = nullptr;

  int numChunks;
  size_t uniformSize, lastChunk;
  if (perRankBytes == 0 || maxChunk == 0 || perRankBytes <= maxChunk) {
    numChunks = 1;
    uniformSize = perRankBytes;
    lastChunk = perRankBytes;
  } else {
    numChunks = (int)((perRankBytes + maxChunk - 1) / maxChunk);
    uniformSize = (perRankBytes / numChunks / align) * align;
    if (uniformSize < align) uniformSize = align;
    lastChunk = perRankBytes - uniformSize * (numChunks - 1);
  }

  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkStart, nPeers + 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkBytes, nPeers * numChunks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&outPlan->chunkOff, nPeers * numChunks), ret, fail);

  for (int p = 0; p <= nPeers; p++) {
    outPlan->chunkStart[p] = p * numChunks;
  }
  for (int p = 0; p < nPeers; p++) {
    size_t off = 0;
    for (int c = 0; c < numChunks; c++) {
      int idx = p * numChunks + c;
      size_t sz = (c == numChunks - 1) ? lastChunk : uniformSize;
      outPlan->chunkBytes[idx] = sz;
      outPlan->chunkOff[idx] = off;
      off += sz;
    }
  }
exit:
  return ret;
fail:
  free(outPlan->chunkStart);
  outPlan->chunkStart = nullptr;
  free(outPlan->chunkBytes);
  outPlan->chunkBytes = nullptr;
  free(outPlan->chunkOff);
  outPlan->chunkOff = nullptr;
  goto exit;
}

static void ncclHierCollFreeChunkPlan(struct ncclHierChunkPlan* plan) {
  if (plan == nullptr) return;
  free(plan->chunkStart);
  free(plan->chunkBytes);
  free(plan->chunkOff);
  plan->chunkStart = nullptr;
  plan->chunkBytes = nullptr;
  plan->chunkOff = nullptr;
  plan->nPeers = 0;
}

// Effective number of internal contexts for a hierarchical collective's rail
static int ncclHierCollNumCtx(struct ncclRmaProxyState* rmaProxyState, size_t perPeerBytes, bool persistent) {
  int numCtx = rmaProxyState->numIntCtx;
  int64_t numCtxOverride = ncclParamHierCeCollNumCtx();
  if (numCtxOverride > 0) {
    if (numCtxOverride > numCtx) {
      WARN("NCCL_HIER_CE_COLL_NUM_CTX=%lld exceeds provisioned contexts; using %d. "
           "Increase NCCL_NUM_RMA_INT_CTX before init.",
           (long long)numCtxOverride, numCtx);
    }
    return numCtxOverride < numCtx ? (int)numCtxOverride : numCtx;
  }
  if (persistent) return 1;
  int64_t threshold = ncclParamRmaMultiCtxThreshold();
  if (threshold < 0) threshold = HIER_COLL_MULTI_CTX_THRESHOLD_DEFAULT;
  if (perPeerBytes < (size_t)threshold) numCtx = 1;
  return numCtx;
}

// Max chunk width for the inter-node put-signal-group. With numCtx > 1, size
// the chunks so each peer's transfer spreads evenly across all contexts: the
// chunk count is a whole multiple of numCtx (one chunk per context, or more
// when a per-context share would exceed HIER_COLL_MAX_CHUNK_SIZE — capping the
// width alone would leave the round-robin stacking every peer's extra chunks
// on the first contexts). Fewer chunks only when the HIER_COLL_CHUNK_ALIGN
// floor binds. With numCtx == 1 this is the plain chunk width. Deterministic
// in (perPeerBytes, numCtx), so sender and receiver derive the same chunking.
static size_t ncclHierCollChunkWidth(size_t perPeerBytes, int numCtx) {
  size_t maxChunk = HIER_COLL_MAX_CHUNK_SIZE;
  if (numCtx > 1) {
    size_t perCtx = DIVUP(perPeerBytes, (size_t)numCtx);
    size_t chunksPerCtx = DIVUP(perCtx, HIER_COLL_MAX_CHUNK_SIZE);  // 1 unless the cap binds
    size_t target = alignUp(DIVUP(perPeerBytes, numCtx * chunksPerCtx), HIER_COLL_CHUNK_ALIGN);
    if (target < maxChunk) maxChunk = target;
  }
  return maxChunk;
}

// Cross-node rail-sync entry barrier for the hierarchical CE collectives.
static ncclResult_t ncclRailSync(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                 struct ncclKernelPlan* plan, int ctx, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  bool persistent = plan->persistent;

  // No remote nodes -> nothing to barrier across; fast-path no-op.
  if (nRemoteNodes <= 0) return ncclSuccess;

  int* railPeers = nullptr;
  int* railSigOnes = nullptr;
  int* railSignalIdxs = nullptr;
  // One signal-only put op per rail peer, packed into a single group desc.
  struct ncclRmaPutSignalOp* groupOps = nullptr;
  struct ncclRmaProxyDesc* groupDesc = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* putBatch = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&railPeers, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&railSigOnes, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&railSignalIdxs, nRemoteNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, nRemoteNodes), ret, fail);

  // Build one signal-only put op per rail peer
  {
    int idx = 0;
    for (int n = 0; n < nNodes; n++) {
      if (n == comm->node) continue;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      railPeers[idx] = railPeer;
      railSigOnes[idx] = 1;

      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, rmaProxyCtx, ctx, persistent,
                                           /*srcWin=*/nullptr, /*srcOff=*/0,
                                           /*peerWin=*/nullptr, /*peerOff=*/0,
                                           /*size=*/0, railPeer, /*signalIdx=*/0, NCCL_SIGNAL, &groupOps[idx]),
                    ret, fail);
      idx++;
    }
  }

  // Build the group put desc
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, rmaProxyCtx, plan, nRemoteNodes, &groupOps, ctx, groupDesc), ret,
                fail);

  // Build one wait descriptor that covers all nRemoteNodes inbound signals.
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, nRemoteNodes, &railPeers, &railSigOnes,
                                          &railSignalIdxs, waitDesc),
                ret, fail);

  // ------------------------------------------------------------------
  // Stage 1: issue the group put (start + done) as one batch.
  // ------------------------------------------------------------------
  {
    int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
    int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
    int putBatchOps = startOps + doneOps;

    NCCLCHECKGOTO(ncclCalloc(&putBatch, putBatchOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc, &putBatch[0]), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc, &putBatch[startOps]), ret, fail);

    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &groupDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, putBatchOps, putBatch), ret, fail);
  }

  // ------------------------------------------------------------------
  // Stage 2: issue the inbound-signal wait as a separate batch.
  // ------------------------------------------------------------------
  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(putBatch);
  free(waitBatch);
  if (groupDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(groupOps);
  free(railPeers);
  free(railSigOnes);
  free(railSignalIdxs);
  return ret;
fail:
  goto exit;
}

// Helper function to wait for a single peer's signals.
static ncclResult_t ncclProxyWaitOnePeer(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                         struct ncclKernelPlan* plan, int ctx, cudaStream_t stream, int peer,
                                         int nsignals) {
  ncclResult_t ret = ncclSuccess;

  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  int* waitSignalIdxs = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, 1), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSignalIdxs, 1), ret, fail);
  waitPeers[0] = peer;
  waitSigCounts[0] = nsignals;

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, 1, &waitPeers, &waitSigCounts, &waitSignalIdxs,
                                          waitDesc),
                ret, fail);

  {
    int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc);
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOps), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyWaitParams(rmaProxyCtx, waitDesc, waitBatch), ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(rmaProxyCtx, &waitDesc), ret, fail);
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOps, waitBatch), ret, fail);
  }

exit:
  free(waitBatch);
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  free(waitSignalIdxs);
  return ret;
fail:
  goto exit;
}

// Hierarchical AllGather: railed all-to-all inter-node + intra-node CE scatter.
// Each per-rank slice is split into chunks. A single PutGroup descriptor
// bundles all nRemoteNodes * nChunks puts.
//
// DAG on the user stream:
//   RailSync                    // cross-node entry barrier (net + wait)
//   PutGroupSubmit              // one memop fires all network puts in parallel
//   IntraNodeBarrier            // gates LSA peers' recvbuf writes; runs while proxy is in flight
//   SelfBcast                   // CE scatter of own slice to LSA peers
//   for (peer, chunk) in shift order:
//     wait for chunk's signal; CE-scatter it to local peers via LSA
//   PutGroupDone                // one memop blocks until all network puts complete
//   IntraNodeBarrier            // gates user code reading recvbuf

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Distribute the cross-node rail puts/waits across the NCCL-internal RMA proxy
  // contexts [numRmaCtx, numRmaCtx + numIntCtx); the user-addressable contexts
  // [0, numRmaCtx) are never touched by the collective. baseCtx is the first
  // internal context, and chunk-local index j maps to context baseCtx + j%numCtx.
  // Both ranks build the same chunk plan, so sender and receiver agree on each
  // chunk's context (a signal raised on context c is awaited on context c).
  // RailSync (the entry barrier) stays on the first internal context. Small
  // transfers stay on a single context (see ncclHierCollNumCtx).
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  int baseCtx = comm->config.numRmaCtx;
  int railCtx = baseCtx;
  int myRank = comm->rank;
  int localRank = comm->localRank;
  int nNodes = comm->nNodes;
  int nRemoteNodes = nNodes - 1;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  bool persistent = plan->persistent;

  struct ncclCeCollArgs* args = plan->ceCollArgs;
  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perRankBytes = args->nElts * args->eltSize;
  int numCtx = ncclHierCollNumCtx(rmaProxyState, perRankBytes, persistent);

  struct ncclRmaProxyCtx* railProxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[railCtx];

  // Per-(peer, chunk) plan.
  struct ncclHierChunkPlan chunkPlan = {};
  // Inter-node put-signal-group: one descriptor per RMA proxy context. The per-ctx
  // arrays are calloc'd (null-initialized) so the exit path frees them uniformly.
  int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
  int* ctxOps = nullptr;                                 // [numCtx] ops assigned per ctx
  int* ctxFill = nullptr;                                // [numCtx] running fill index
  struct ncclRmaProxyDesc** groupDesc = nullptr;         // [numCtx]
  struct ncclRmaPutSignalOp** groupOps = nullptr;        // [numCtx]
  // Start/done memop params for all active contexts, one contiguous slice per
  // context, so each phase fires a single stream batch instead of one per ctx.
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams = nullptr;  // [nActiveCtx * startOps]
  CUstreamBatchMemOpParams* groupDoneParams = nullptr;   // [nActiveCtx * doneOps]
  // Batch-ops scratch for intra-node broadcast.
  struct ncclCeBatchOpsParams ceBcastOps = {};
  // Batch-ops scratch for per-chunk intra-node CE scatter.
  struct ncclCeBatchOpsParams ceScatterOps = {};

  NCCLCHECKGOTO(ncclCalloc(&ctxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ctxFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, numCtx), ret, fail);

  // ====================================================================
  // Phase 1: Rail sync (cross-node entry barrier)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, railProxyCtx, plan, railCtx, stream), ret, fail);

  // ====================================================================
  // Phase 2: Start all inter-node puts (one group descriptor per context, chunked)
  // ====================================================================
  {
    size_t maxChunk = ncclHierCollChunkWidth(perRankBytes, numCtx);
    NCCLCHECKGOTO(ncclHierCollBuildChunk(perRankBytes, nRemoteNodes, maxChunk, &chunkPlan), ret, fail);

    // Window-relative offsets
    size_t srcWinOffset = (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr;
    size_t peerWinOffset = ((const uint8_t*)recvbuff + myRank * perRankBytes) - (const uint8_t*)recvWin->userPtr;

    // Pass 1: count ops per context. Each chunk is assigned to ctx (j % numCtx) by
    // its peer-local chunk index j, so every rank agrees on a chunk's context.
    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;
      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        ctxOps[(c - chunkPlan.chunkStart[p]) % numCtx]++;
      }
    }

    // Allocate ops array + descriptor for each active context, then one memop
    // param slice per active context.
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      NCCLCHECKGOTO(ncclCalloc(&groupOps[k], ctxOps[k]), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&groupDesc[k], 1), ret, fail);
      nActiveCtx++;
    }
    NCCLCHECKGOTO(ncclCalloc(&groupStartParams, (size_t)nActiveCtx * startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParams, (size_t)nActiveCtx * doneOps), ret, fail);

    // Pass 2: build each chunk's put op into its context's ops array.
    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;                                 // peer index in plan
      int n = (comm->node + s) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        int k = (c - chunkPlan.chunkStart[p]) % numCtx;
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k],
                                             baseCtx + k, persistent, sendWin, srcWinOffset + off, recvWin,
                                             peerWinOffset + off, subBytes, railPeer, /*signalIdx=*/0, NCCL_SIGNAL,
                                             &groupOps[k][ctxFill[k]++]),
                      ret, fail);
      }
    }

    // Build and enqueue each active context's group descriptor, then fire all
    // the start memops as one stream batch.
    int a = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, pc, plan, ctxOps[k], &groupOps[k], baseCtx + k, groupDesc[k]),
                    ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc[k], groupStartParams + a * startOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc[k], groupDoneParams + a * doneOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &groupDesc[k]), ret, fail);
      a++;
    }
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * startOps, groupStartParams), ret, fail);
  }

  // ====================================================================
  // Phase 3: Initial intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // Multicast/unicast switch for the intra-node scatters in Phase 4 + Phase 5.
  // Same heuristic as ncclCeAllGather: short transfers fit nicely in a single
  // multicast write to the LSA team, longer ones stay on the original N-1
  // unicast loop (which is better at hiding tail latency).
  // declare-then-assign: NCCLCHECKGOTO's goto can't cross a scalar initialization.
  bool agUseMulticast;
  agUseMulticast =
    ncclCeAllGatherUseMulticast(comm, perRankBytes, ncclCudaGraphValid(comm->planner.capturingGraph),
                                (const uint8_t*)sendbuff == (const uint8_t*)recvbuff + myRank * perRankBytes);

  // ====================================================================
  // Phase 4: Self-broadcast (intra-node CE Broadcast of own chunk)
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    if (agUseMulticast) {
      // Single multicast write covers self + all LSA peers; no self-copy needed.
      void* mcDstPtr;
      NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, recvWin, offset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = (void*)mcDstPtr;
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    } else {
      // Out-of-place: copy own data to own recvbuf slot
      if (myRecvSlot != (const uint8_t*)sendbuff) {
        ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
        ceBcastOps.dsts[ceBcastOps.numOps] = (void*)myRecvSlot;
        ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
        ceBcastOps.numOps++;
      }

      // Broadcast to all other LSA peers
      for (int r = 1; r < lsaSize; r++) {
        int targetLsaRank = (myLsaRank + r) % lsaSize;
        void* peerBuf;
        NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, offset, targetLsaRank, &peerBuf), ret, fail);
        ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
        ceBcastOps.dsts[ceBcastOps.numOps] = peerBuf;
        ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
        ceBcastOps.numOps++;
      }
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  // ====================================================================
  // Phase 5: Wait for each (peer, chunk) + intra-node CE scatter (pipelined)
  // ====================================================================
  {
    for (int s = 1; s < nNodes; s++) {
      int p = s - 1;                                 // peer index in plan
      int n = (comm->node - s + nNodes) % nNodes;
      int railPeer = comm->nodeRanks[n].localRankToRank[localRank];
      size_t peerSliceOffset = railPeer * perRankBytes;

      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        size_t subBytes = chunkPlan.chunkBytes[c];
        size_t off = chunkPlan.chunkOff[c];

        uint8_t* chunkSlot = (uint8_t*)recvbuff + peerSliceOffset + off;
        size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

        // ----- Wait for this sub-chunk's signal from railPeer on the chunk's context -----
        int k = (c - chunkPlan.chunkStart[p]) % numCtx;
        NCCLCHECKGOTO(ncclProxyWaitOnePeer(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k],
                                           plan, baseCtx + k, stream, railPeer, /*nsignals=*/1),
                      ret, fail);

        // ----- CE scatter this sub-chunk to all other LSA peers -----
        NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, lsaSize), ret, fail);
        if (agUseMulticast) {
          // One multicast write covers all LSA peers (including self, which
          // already has the data — harmless self-store).
          void* mcDstPtr;
          NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, recvWin, winOffset, ncclTeamLsa(comm), &mcDstPtr), ret, fail);
          ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
          ceScatterOps.dsts[ceScatterOps.numOps] = (void*)mcDstPtr;
          ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
          ceScatterOps.numOps++;
        } else {
          for (int r = 1; r < lsaSize; r++) {
            int targetLsaRank = (myLsaRank + r) % lsaSize;
            void* peerBuf;
            NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, winOffset, targetLsaRank, &peerBuf), ret, fail);
            ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
            ceScatterOps.dsts[ceScatterOps.numOps] = peerBuf;
            ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
            ceScatterOps.numOps++;
          }
        }

        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
        ncclCeFreeBatchOpsParams(&ceScatterOps);
      }
    }
  }

  // ====================================================================
  // Phase 6: Wait for all outgoing data puts to complete (all contexts, one batch)
  // ====================================================================
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * doneOps, groupDoneParams), ret, fail);

  // ====================================================================
  // Phase 7: Final intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  for (int k = 0; groupOps && k < numCtx; k++) free(groupOps[k]);
  if (groupDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (groupDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc[k]);
    }
  }
  free(groupOps);
  free(groupStartParams);
  free(groupDoneParams);
  free(groupDesc);
  free(ctxOps);
  free(ctxFill);
  ncclHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

// Hierarchical AlltoAll: alltoall inter-node + intra-node CE alltoall.
// DAG on the user stream:
//   RailSync                    // rail-only entry barrier
//   IntraNodeBarrier #1         // all ranks in sync
//   PutGroupSubmit              // one memop fires all put operations
//   IntraNodeAlltoAll           // batched CE alltoall
//   AggregateWait               // single multi-peer wait descriptor covering all remote peers
//   PutGroupDone                // one memop blocks until outbound puts done
//   IntraNodeBarrier #2         // all ranks in sync

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  // Distribute the cross-node rail puts/waits across the NCCL-internal RMA proxy
  // contexts [numRmaCtx, numRmaCtx + numIntCtx); the user-addressable contexts
  // [0, numRmaCtx) are never touched by the collective. baseCtx is the first
  // internal context, and chunk-local index j maps to context baseCtx + j%numCtx.
  // Both ranks build the same chunk plan, so sender and receiver agree on each
  // chunk's context. RailSync (the entry barrier) stays on the first internal
  // context. Small transfers stay on a single context (see ncclHierCollNumCtx).
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  int baseCtx = comm->config.numRmaCtx;
  int railCtx = baseCtx;
  int myRank = comm->rank;
  int myNode = comm->node;
  int nNodes = comm->nNodes;
  int localRanks = comm->localRanks;
  int myLsaRank = comm->devrState.lsaSelf;
  int lsaSize = comm->devrState.lsaSize;
  int numRemotePeers = (nNodes - 1) * localRanks;
  bool persistent = plan->persistent;

  struct ncclCeCollArgs* args = plan->ceCollArgs;
  const void* sendbuff = args->sendBuff;
  void* recvbuff = args->recvBuff;
  struct ncclDevrWindow* sendWin = args->sendWin;
  struct ncclDevrWindow* recvWin = args->recvWin;
  size_t perPeerBytes = args->nElts * args->eltSize;
  int numCtx = ncclHierCollNumCtx(rmaProxyState, perPeerBytes, persistent);
  bool inPlace = (sendbuff == recvbuff);

  struct ncclRmaProxyCtx* railProxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[railCtx];

  // Chunk plan for the inter-node put-signal-group.
  struct ncclHierChunkPlan chunkPlan = {};
  // Inter-node put-signal-group + inbound wait: one descriptor per RMA proxy context.
  // All per-ctx arrays are calloc'd (null-initialized) so the exit path frees them
  // uniformly; the ownership-transferring builders (PutGroupBuildDesc, WaitBuildDesc)
  // null the slots they consume.
  int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
  int* ctxOps = nullptr;                                 // [numCtx] ops assigned per ctx
  int* ctxFill = nullptr;                                // [numCtx] running fill index
  struct ncclRmaProxyDesc** groupDesc = nullptr;         // [numCtx]
  struct ncclRmaPutSignalOp** groupOps = nullptr;        // [numCtx]
  // Start/done memop params for all active contexts, one contiguous slice per
  // context, so each phase fires a single stream batch instead of one per ctx.
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams = nullptr;  // [nActiveCtx * startOps]
  CUstreamBatchMemOpParams* groupDoneParams = nullptr;   // [nActiveCtx * doneOps]
  // Per-context inbound wait descriptors (each covers the peers/counts whose chunks
  // landed on that context); their stream memops are likewise fired as one batch.
  int** waitPeers = nullptr;                             // [numCtx][]
  int** waitSigCounts = nullptr;                         // [numCtx][]
  int** waitSignalIdxs = nullptr;                        // [numCtx][] all-zero (hier-CE uses signal 0)
  struct ncclRmaProxyDesc** waitDesc = nullptr;          // [numCtx]
  CUstreamBatchMemOpParams* waitBatch = nullptr;         // [sum of per-ctx wait ops]
  // Intra-node alltoall scratch.
  struct ncclCeBatchOpsParams ceLocalA2A = {};

  NCCLCHECKGOTO(ncclCalloc(&ctxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ctxFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitPeers, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSignalIdxs, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, numCtx), ret, fail);

  // ====================================================================
  // Phase 1: Rail sync (rail-only cross-node entry barrier)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, railProxyCtx, plan, railCtx, stream), ret, fail);

  // ====================================================================
  // Phase 2: Intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  // ====================================================================
  // Phase 3: Build & submit put-signal-group (start memop).
  // ====================================================================
  {
    size_t maxChunk = ncclHierCollChunkWidth(perPeerBytes, numCtx);
    NCCLCHECKGOTO(ncclHierCollBuildChunk(perPeerBytes, numRemotePeers, maxChunk, &chunkPlan), ret, fail);

    // Pass 1: count ops per context (chunk j -> ctx j % numCtx by peer-local index).
    for (int p = 0; p < numRemotePeers; p++) {
      for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
        ctxOps[(c - chunkPlan.chunkStart[p]) % numCtx]++;
      }
    }

    // Allocate ops array + descriptor for each active context, then one memop
    // param slice per active context.
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      NCCLCHECKGOTO(ncclCalloc(&groupOps[k], ctxOps[k]), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&groupDesc[k], 1), ret, fail);
      nActiveCtx++;
    }
    NCCLCHECKGOTO(ncclCalloc(&groupStartParams, (size_t)nActiveCtx * startOps), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&groupDoneParams, (size_t)nActiveCtx * doneOps), ret, fail);

    // Pass 2: build each chunk's put op into its context's ops array.
    int p = 0;  // chunk plan slot index
    for (int s = 1; s < nNodes; s++) {
      int n = (myNode + s) % nNodes;
      for (int lr = 0; lr < localRanks; lr++) {
        int peer = comm->nodeRanks[n].localRankToRank[lr];
        size_t srcWinOffset =
          ((const uint8_t*)sendbuff + (size_t)peer * perPeerBytes) - (const uint8_t*)sendWin->userPtr;
        size_t peerWinOffset =
          ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

        for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
          int k = (c - chunkPlan.chunkStart[p]) % numCtx;
          size_t subBytes = chunkPlan.chunkBytes[c];
          size_t off = chunkPlan.chunkOff[c];

          NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k],
                                               baseCtx + k, persistent, sendWin, srcWinOffset + off, recvWin,
                                               peerWinOffset + off, subBytes, peer, /*signalIdx=*/0, NCCL_SIGNAL,
                                               &groupOps[k][ctxFill[k]++]),
                        ret, fail);
        }
        p++;
      }
    }

    // Build and enqueue each active context's group descriptor, then fire all
    // the start memops as one stream batch.
    int a = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, pc, plan, ctxOps[k], &groupOps[k], baseCtx + k, groupDesc[k]),
                    ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(groupDesc[k], groupStartParams + a * startOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(groupDesc[k], groupDoneParams + a * doneOps), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &groupDesc[k]), ret, fail);
      a++;
    }
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * startOps, groupStartParams), ret, fail);
  }

  // ====================================================================
  // Phase 4: Intra-node alltoall (batched CE memcpy over LSA).
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceLocalA2A, lsaSize), ret, fail);
  {
    size_t myRecvOffset = ((const uint8_t*)recvbuff + (size_t)myRank * perPeerBytes) - (const uint8_t*)recvWin->userPtr;

    for (int k = 0; k < lsaSize; k++) {
      int targetLsa = (myLsaRank + k) % lsaSize;
      int targetWorldRank = comm->nodeRanks[myNode].localRankToRank[targetLsa];

      if (inPlace && targetLsa == myLsaRank) continue;

      void* peerRecvSlot;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, myRecvOffset, targetLsa, &peerRecvSlot), ret, fail);

      ceLocalA2A.srcs[ceLocalA2A.numOps] = (void*)((const uint8_t*)sendbuff + (size_t)targetWorldRank * perPeerBytes);
      ceLocalA2A.dsts[ceLocalA2A.numOps] = peerRecvSlot;
      ceLocalA2A.sizes[ceLocalA2A.numOps] = perPeerBytes;
      ceLocalA2A.numOps++;
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceLocalA2A, stream, args), ret, fail);
  }

  // ====================================================================
  // Phase 5: Aggregate wait for all remote peers.
  // ====================================================================
  {
    // Per context, build an inbound wait covering the peers whose chunks landed on
    // that context, with per-peer signal count = number of that peer's chunks on the
    // context. The chunk->ctx assignment mirrors Phase 3 exactly, so the signal
    // accounting matches what the senders raised.
    int waitOpsTotal = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;

      NCCLCHECKGOTO(ncclCalloc(&waitPeers[k], numRemotePeers), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&waitSigCounts[k], numRemotePeers), ret, fail);
      // Signal indices default to 0 for hier-CE; allocated so WaitBuildDesc can index it.
      NCCLCHECKGOTO(ncclCalloc(&waitSignalIdxs[k], numRemotePeers), ret, fail);

      int wp = 0;
      int p = 0;
      for (int s = 1; s < nNodes; s++) {
        int n = (myNode - s + nNodes) % nNodes;
        for (int lr = 0; lr < localRanks; lr++) {
          int sig = 0;
          for (int c = chunkPlan.chunkStart[p]; c < chunkPlan.chunkStart[p + 1]; c++) {
            if ((c - chunkPlan.chunkStart[p]) % numCtx == k) sig++;
          }
          if (sig > 0) {
            waitPeers[k][wp] = comm->nodeRanks[n].localRankToRank[lr];
            waitSigCounts[k][wp] = sig;
            wp++;
          }
          p++;
        }
      }

      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      NCCLCHECKGOTO(ncclCalloc(&waitDesc[k], 1), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, pc, plan, wp, &waitPeers[k], &waitSigCounts[k], &waitSignalIdxs[k],
                                              waitDesc[k]),
                    ret, fail);
      waitOpsTotal += ncclRmaProxyWaitNumStreamOps(waitDesc[k]);
    }

    // Fill each context's wait memops into one contiguous array, enqueue the
    // descriptors, and fire all the waits as one stream batch.
    NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOpsTotal), ret, fail);
    int off = 0;
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
      int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc[k]);
      NCCLCHECKGOTO(ncclRmaProxyWaitParams(pc, waitDesc[k], waitBatch + off), ret, fail);
      NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &waitDesc[k]), ret, fail);
      off += waitOps;
    }
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOpsTotal, waitBatch), ret, fail);
  }

  // ====================================================================
  // Phase 6: PutGroupDone memop (outbound puts complete on the wire), all
  // contexts in one batch.
  // ====================================================================
  NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * doneOps, groupDoneParams), ret, fail);

  // ====================================================================
  // Phase 7: Intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceLocalA2A);
  for (int k = 0; groupOps && k < numCtx; k++) free(groupOps[k]);
  if (groupDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (groupDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc[k]);
    }
  }
  if (waitDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (waitDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc[k]);
    }
  }
  for (int k = 0; waitPeers && k < numCtx; k++) free(waitPeers[k]);
  for (int k = 0; waitSigCounts && k < numCtx; k++) free(waitSigCounts[k]);
  for (int k = 0; waitSignalIdxs && k < numCtx; k++) free(waitSignalIdxs[k]);
  free(groupOps);
  free(groupStartParams);
  free(groupDoneParams);
  free(groupDesc);
  free(waitBatch);
  free(waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  free(waitSignalIdxs);
  free(ctxOps);
  free(ctxFill);
  ncclHierCollFreeChunkPlan(&chunkPlan);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;
  struct ncclCeCollArgs* args = plan->ceCollArgs;

  // Start CE collective profiling
  NCCLCHECKGOTO(ncclProfilerStartCeCollEvent(comm, args, stream), ret, fail);

  // Hierarchical path: inter-node RMA + intra-node CE
  // Use ncclDevrIsOneLsaTeam instead of comm->nNodes as multi-clique single-NVLD should use CE path
  if (!ncclDevrIsOneLsaTeam(comm)) {
    switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(ncclHierCeAllGather(comm, plan, stream), ret, fail);
      break;
    case ncclFuncAlltoAll:
      NCCLCHECKGOTO(ncclHierCeAlltoAll(comm, plan, stream), ret, fail);
      break;
    default:
      WARN("Hierarchical CE collective not supported for %s", ncclFuncToString(args->func));
      ret = ncclInvalidUsage;
    }
  }
  // LSA-local CE path
  else {
    switch (args->func) {
    case ncclFuncAllGather:
      NCCLCHECKGOTO(ncclCeAllGather(comm, args, stream), ret, fail);
      break;
    case ncclFuncAlltoAll:
      NCCLCHECKGOTO(ncclCeAlltoAll(comm, args, stream), ret, fail);
      break;
    case ncclFuncScatter:
      NCCLCHECKGOTO(ncclCeScatter(comm, args, stream), ret, fail);
      break;
    case ncclFuncGather:
      NCCLCHECKGOTO(ncclCeGather(comm, args, stream), ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
    }
  }

exit:
  // Stop CE collective profiling - always attempt if started, even on error
  ncclProfilerStopCeCollEvent(comm, args, stream);
  return ret;
fail:
  goto exit;
}

ncclResult_t scheduleCeCollTaskToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  struct ncclKernelPlanner* planner = &comm->planner;
  struct ncclTaskColl* task = ncclIntruQueueHead(&planner->collCeTaskQueue);

  plan->isCeColl = true;
  plan->ceCollArgs = ncclMemoryStackAlloc<struct ncclCeCollArgs>(&comm->memScoped);
  plan->ceCollArgs->rootRank = task->root;
  plan->ceCollArgs->datatype = task->datatype;
  plan->ceCollArgs->nElts = task->count;
  plan->ceCollArgs->eltSize = ncclTypeSize(task->datatype);
  plan->ceCollArgs->sendBuff = (uint8_t*)task->sendbuff;
  plan->ceCollArgs->recvBuff = (uint8_t*)task->recvbuff;
  plan->ceCollArgs->func = task->func;
  plan->ceCollArgs->sendWin = task->sendWin;
  plan->ceCollArgs->recvWin = task->recvWin;
  plan->ceCollArgs->collApiEventHandle = task->collApiEventHandle;
  plan->ceCollArgs->userTag = task->profilerTag;

  if (comm->rank == 0) {
    if (!ncclDevrIsOneLsaTeam(comm)) {
      INFO(NCCL_TUNING, "%s [Hierarchical CE]: %ld Bytes -> RMA proxy + CE", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype));
    } else {
      const char* nvlsSync = comm->symkState.hasLsaMultimem ? "; CE synchronization with NVLS" : "";
      INFO(NCCL_TUNING, "%s [Copy Engine]: %ld Bytes -> cudaMemcpy%s", ncclFuncToString(task->func),
           task->count * ncclTypeSize(task->datatype), nvlsSync);
    }
  }

  ncclIntruQueueEnqueue(&planner->planQueue, plan);
  ncclIntruQueueDequeue(&planner->collCeTaskQueue);
  ncclMemoryPoolFree(&comm->memPool_ncclTaskColl, task);

  return ncclSuccess;
}
