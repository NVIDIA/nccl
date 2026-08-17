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
#include "enqueue.h"
#include "tuning.h"

// User override: when set (>= 0) the cost model is bypassed and this byte
// threshold decides multicast (sendSize <= threshold). -1 (unset) -> cost model.
NCCL_PARAM(CeCollAgMulticastThreshold, "CE_COLL_AG_MULTICAST_THRESHOLD", -1);

// Chunk size used when a cudaMemcpyBatchAsync CE data batch requests
// round-robin chunking. Chunking is opt-in per batch.
NCCL_PARAM(CeChunkSize, "CE_CHUNK_SIZE", 8 * 1024 * 1024);

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

// --------------------------------------------------------------------
// Hierarchical CE AllGather: Ring selection and chunking
// --------------------------------------------------------------------

// The retained ring implementation only changes chunking once messages are
// clearly bandwidth-dominated.
static constexpr size_t HIER_COLL_AG_RING_HALF_AWARE_THRESHOLD = 128 * 1024 * 1024;
static constexpr size_t HIER_COLL_AG_RING_MIN_HALF_AWARE_CHUNK = 8 * 1024 * 1024;

// Ring is opt-in in this MR: a positive value enables it, while zero or a
// negative value keeps the Direct path. The stacked tuning MR gives -1 its
// automatic-selection semantics.
NCCL_PARAM(HierCeCollAgRailRingEnable, "HIER_CE_COLL_AG_RAIL_RING_ENABLE", -1);

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
  params->chunking = false;
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
  params->chunking = false;
  params->intraBatchSync = false;
#if CUDART_VERSION >= 12080
  if (params->attrs) free(params->attrs);
  params->attrs = nullptr;
  if (params->attrIdxs) free(params->attrIdxs);
  params->attrIdxs = nullptr;
  params->numAttrs = 0;
#endif
}

#if CUDART_VERSION >= 12080
static ncclResult_t ncclCeLaunchChunkedMemcpyBatchAsync(struct ncclCeBatchOpsParams* params, size_t chunkSize,
                                                        cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;
  size_t maxSize = 0;
  for (int i = 0; i < params->numOps; i++) {
    maxSize = std::max(maxSize, params->sizes[i]);
  }
  size_t numRounds = maxSize == 0 ? 0 : 1 + (maxSize - 1) / chunkSize;

  ncclUniqueArrayPtr<void*> tmpDsts{nullptr};
  ncclUniqueArrayPtr<void*> tmpSrcs{nullptr};
  ncclUniqueArrayPtr<size_t> tmpSizes{nullptr};
  NCCLCHECKGOTO(ncclCalloc(tmpDsts, params->numOps), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(tmpSrcs, params->numOps), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(tmpSizes, params->numOps), ret, fail);

  // Submit one batch per round. CUDA guarantees stream ordering between
  // batches, but does not guarantee the execution order of copies within
  // a batch, so flattening every chunk into one batch would not pace the
  // destinations in round-robin waves.
  for (size_t round = 0; round < numRounds; round++) {
    size_t offset = round * chunkSize;
    int nWaveOps = 0;
    for (int i = 0; i < params->numOps; i++) {
      if (offset >= params->sizes[i]) continue;
      size_t bytes = std::min(params->sizes[i] - offset, chunkSize);
      tmpDsts[nWaveOps] = (uint8_t*)params->dsts[i] + offset;
      tmpSrcs[nWaveOps] = (uint8_t*)params->srcs[i] + offset;
      tmpSizes[nWaveOps] = bytes;
      nWaveOps++;
    }

    if (nWaveOps == 0) continue;
#if CUDART_VERSION >= 13000
    CUDACHECKGOTO(cudaMemcpyBatchAsync(tmpDsts.get(), tmpSrcs.get(), tmpSizes.get(), nWaveOps, params->attrs,
                                       params->attrIdxs, params->numAttrs, stream),
                  ret, fail);
#else
    CUDACHECKGOTO(cudaMemcpyBatchAsync(tmpDsts.get(), tmpSrcs.get(), tmpSizes.get(), nWaveOps, params->attrs,
                                       params->attrIdxs, params->numAttrs, nullptr, stream),
                  ret, fail);
#endif
  }

exit:
  return ret;
fail:
  goto exit;
}
#endif

ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* profilerArgs) {
  ncclResult_t ret = ncclSuccess;
  bool capturing;
  int driverVersion;
  int64_t chunkSizeParam = ncclParamCeChunkSize();
  size_t ceChunkSize = chunkSizeParam > 0 ? (size_t)chunkSizeParam : 0;
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

      if (params->chunking && ceChunkSize > 0) {
        NCCLCHECKGOTO(ncclCeLaunchChunkedMemcpyBatchAsync(params, ceChunkSize, stream), ret, fail);
      } else if (params->intraBatchSync) {
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
      // For older CUDA versions, fall back to individual transfers.
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
  agUseMulticast =
    ncclCeAllGatherUseMulticast(comm, chunkBytes, ncclCudaGraphValid(comm->planner.capturingGraph),
                                ncclAllGatherIsInPlace(args->sendBuff, args->recvBuff, myLsaRank, chunkBytes));

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

static size_t ncclHierCollRingChunkWidth(size_t perRankBytes, size_t cwBytes, size_t ccwBytes, int numCtx) {
  size_t maxChunk = ncclHierCollChunkWidth(perRankBytes, numCtx);
  if (numCtx < 4 || perRankBytes < HIER_COLL_AG_RING_HALF_AWARE_THRESHOLD) {
    return maxChunk;
  }

  size_t halfBytes = std::max(cwBytes, ccwBytes);
  if (halfBytes == 0) return maxChunk;

  size_t halfAware = alignUp(DIVUP(halfBytes, (size_t)numCtx), HIER_COLL_CHUNK_ALIGN);
  halfAware = std::min(halfAware, HIER_COLL_MAX_CHUNK_SIZE);
  halfAware = std::max(halfAware, HIER_COLL_AG_RING_MIN_HALF_AWARE_CHUNK);
  return std::min(maxChunk, halfAware);
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

// Helper function to wait for one or more distinct peers' signals.
static ncclResult_t ncclProxyWaitPeers(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                       struct ncclKernelPlan* plan, cudaStream_t stream, int npeers, const int* peersIn,
                                       const int* nsignalsIn) {
  ncclResult_t ret = ncclSuccess;

  int realPeers = 0;
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  int* waitSignalIdxs = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;
  CUstreamBatchMemOpParams* waitBatch = nullptr;

  if (npeers <= 0) return ncclSuccess;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, npeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, npeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSignalIdxs, npeers), ret, fail);
  for (int i = 0; i < npeers; i++) {
    if (nsignalsIn[i] <= 0) continue;
    waitPeers[realPeers] = peersIn[i];
    waitSigCounts[realPeers] = nsignalsIn[i];
    realPeers++;
  }
  if (realPeers == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, realPeers, &waitPeers, &waitSigCounts,
                                          &waitSignalIdxs, waitDesc),
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

// Variant specialized for the 1-peer / 2-peer wait cases used by the retained
// hierarchical ring allgather implementation. It merges the clockwise and
// counterclockwise entries when they resolve to the same peer.
static ncclResult_t ncclProxyBuildWaitPeersRing(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                                struct ncclKernelPlan* plan, int npeers, const int* peersIn,
                                                const int* nsignalsIn, struct ncclRmaProxyDesc** waitDescOut) {
  ncclResult_t ret = ncclSuccess;

  int realPeers = 0;
  int mergedPeers[2] = {0, 0};
  int mergedSignals[2] = {0, 0};
  int* waitPeers = nullptr;
  int* waitSigCounts = nullptr;
  int* waitSignalIdxs = nullptr;
  struct ncclRmaProxyDesc* waitDesc = nullptr;

  *waitDescOut = nullptr;
  if (npeers <= 0) return ncclSuccess;
  if (npeers > 2) return ncclInternalError;

  if (npeers == 1) {
    if (nsignalsIn[0] > 0) {
      mergedPeers[0] = peersIn[0];
      mergedSignals[0] = nsignalsIn[0];
      realPeers = 1;
    }
  } else if (npeers == 2) {
    int sig0 = nsignalsIn[0];
    int sig1 = nsignalsIn[1];
    if (sig0 > 0) {
      mergedPeers[0] = peersIn[0];
      mergedSignals[0] = sig0;
      realPeers = 1;
    }
    if (sig1 > 0) {
      if (realPeers > 0 && mergedPeers[0] == peersIn[1]) {
        mergedSignals[0] += sig1;
      } else {
        mergedPeers[realPeers] = peersIn[1];
        mergedSignals[realPeers] = sig1;
        realPeers++;
      }
    }
  }
  if (realPeers == 0) goto exit;

  NCCLCHECKGOTO(ncclCalloc(&waitPeers, realPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSigCounts, realPeers), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSignalIdxs, realPeers), ret, fail);
  for (int i = 0; i < realPeers; i++) {
    waitPeers[i] = mergedPeers[i];
    waitSigCounts[i] = mergedSignals[i];
  }

  NCCLCHECKGOTO(ncclCalloc(&waitDesc, 1), ret, fail);
  NCCLCHECKGOTO(ncclRmaProxyWaitBuildDesc(comm, rmaProxyCtx, plan, realPeers, &waitPeers, &waitSigCounts,
                                          &waitSignalIdxs, waitDesc),
                ret, fail);
  *waitDescOut = waitDesc;
  waitDesc = nullptr;

exit:
  if (waitDesc != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc);
  free(waitPeers);
  free(waitSigCounts);
  free(waitSignalIdxs);
  return ret;
fail:
  goto exit;
}

// Helper function to wait for a single peer's signals.
static ncclResult_t ncclProxyWaitOnePeer(struct ncclComm* comm, struct ncclRmaProxyCtx* rmaProxyCtx,
                                         struct ncclKernelPlan* plan, cudaStream_t stream, int peer, int nsignals) {
  int peers[1] = {peer};
  int sigCounts[1] = {nsignals};
  return ncclProxyWaitPeers(comm, rmaProxyCtx, plan, stream, 1, peers, sigCounts);
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

static ncclResult_t ncclHierCeAllGatherDirect(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
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
  agUseMulticast = ncclCeAllGatherUseMulticast(comm, perRankBytes, ncclCudaGraphValid(comm->planner.capturingGraph),
                                               ncclAllGatherIsInPlace(sendbuff, recvbuff, myRank, perRankBytes));

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
                                           plan, stream, railPeer, /*nsignals=*/1),
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

struct ncclHierCeAllGatherRingRound {
  int originRankSendCw;
  int originRankSendCcw;
  int originRankRecvCw;
  int originRankRecvCcw;
  struct ncclDevrWindow* srcWinHostCw;
  struct ncclDevrWindow* srcWinHostCcw;
  size_t srcBaseOffsetCw;
  size_t dstBaseOffsetCw;
  size_t srcBaseOffsetCcw;
  size_t dstBaseOffsetCcw;
};

// State consumed by the round-issue helper. The main ring routine owns the
// arrays and chunk plans; grouping the round-specific fields here keeps
// descriptor construction independent from setup, receive, and cleanup.
struct ncclHierCeAllGatherRingIssueState {
  int numCtx;
  int baseCtx;
  int nActiveCtx;
  int sendRailPeerCw;
  int sendRailPeerCcw;
  int* ctxOps;
  int* ctxFill;
  int* activeCtxs;
  struct ncclHierChunkPlan* cwChunkPlan;
  struct ncclHierChunkPlan* ccwChunkPlan;
  struct ncclRmaProxyDesc** groupDesc;
  struct ncclRmaPutSignalOp** groupOps;
  CUstreamBatchMemOpParams** groupStartParams;
  CUstreamBatchMemOpParams** groupDoneParams;
  struct ncclHierCeAllGatherRingRound* rounds;
};

// Build and submit one bidirectional forwarding round. Two parameter slots are
// alternated by the caller so the next round can be submitted before the
// current round's local CE fanout and completion retirement.
static ncclResult_t ncclHierCeAllGatherRingIssueRound(struct ncclComm* comm, struct ncclKernelPlan* plan,
                                                      cudaStream_t stream, struct ncclHierCeAllGatherRingIssueState* st,
                                                      int step, int slot) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  struct ncclDevrWindow* recvWin = plan->ceCollArgs->recvWin;
  struct ncclHierCeAllGatherRingRound* round = &st->rounds[step];
  int startOps = ncclRmaProxyPutGroupStartNumOps(plan->persistent);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(plan->persistent);

  memset(st->ctxFill, 0, (size_t)st->numCtx * sizeof(int));
  for (int ai = 0; ai < st->nActiveCtx; ai++) {
    int k = st->activeCtxs[ai];
    NCCLCHECKGOTO(ncclCalloc(&st->groupOps[k], st->ctxOps[k]), ret, fail);
    NCCLCHECKGOTO(ncclCalloc(&st->groupDesc[k], 1), ret, fail);
  }

  if (st->cwChunkPlan->chunkStart != nullptr) {
    for (int c = st->cwChunkPlan->chunkStart[0]; c < st->cwChunkPlan->chunkStart[1]; c++) {
      int k = c % st->numCtx;
      size_t subBytes = st->cwChunkPlan->chunkBytes[c];
      size_t off = st->cwChunkPlan->chunkOff[c];
      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[st->baseCtx + k],
                                           st->baseCtx + k, plan->persistent, round->srcWinHostCw,
                                           round->srcBaseOffsetCw + off, recvWin, round->dstBaseOffsetCw + off,
                                           subBytes, st->sendRailPeerCw,
                                           /*signalIdx=*/0, NCCL_SIGNAL, &st->groupOps[k][st->ctxFill[k]++]),
                    ret, fail);
    }
  }
  if (st->ccwChunkPlan->chunkStart != nullptr) {
    for (int c = st->ccwChunkPlan->chunkStart[0]; c < st->ccwChunkPlan->chunkStart[1]; c++) {
      int k = c % st->numCtx;
      size_t subBytes = st->ccwChunkPlan->chunkBytes[c];
      size_t off = st->ccwChunkPlan->chunkOff[c];
      NCCLCHECKGOTO(ncclRmaProxyPutBuildOp(comm, (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[st->baseCtx + k],
                                           st->baseCtx + k, plan->persistent, round->srcWinHostCcw,
                                           round->srcBaseOffsetCcw + off, recvWin, round->dstBaseOffsetCcw + off,
                                           subBytes, st->sendRailPeerCcw,
                                           /*signalIdx=*/0, NCCL_SIGNAL, &st->groupOps[k][st->ctxFill[k]++]),
                    ret, fail);
    }
  }

  for (int ai = 0; ai < st->nActiveCtx; ai++) {
    int k = st->activeCtxs[ai];
    struct ncclRmaProxyCtx* proxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[st->baseCtx + k];
    NCCLCHECKGOTO(ncclRmaProxyPutGroupBuildDesc(comm, proxyCtx, plan, st->ctxOps[k], &st->groupOps[k], st->baseCtx + k,
                                                st->groupDesc[k]),
                  ret, fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupStartParams(st->groupDesc[k], st->groupStartParams[slot] + ai * startOps), ret,
                  fail);
    NCCLCHECKGOTO(ncclRmaProxyPutGroupDoneParams(st->groupDesc[k], st->groupDoneParams[slot] + ai * doneOps), ret,
                  fail);
    NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(proxyCtx, &st->groupDesc[k]), ret, fail);
  }
  if (st->nActiveCtx > 0)
    NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, st->nActiveCtx * startOps, st->groupStartParams[slot]), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

// Bidirectional ring variant of hierarchical CE allgather.
//
// DAG on the user stream:
//   RailSync                    // cross-node entry barrier (net + wait)
//   IntraNodeBarrier            // gates LSA peers' recvbuf writes before local fanout starts
//   SelfBcast                   // CE scatter of own slice to LSA peers
//   Round1PutGroupSubmit        // prime the pipeline with both clockwise and counterclockwise halves
//   for each ring round:
//     wait per active context   // one wait covers that context's cw + ccw arrivals for the current round
//     NextRoundPutGroupSubmit   // submit the next round after current arrivals are known, before local fanout
//     CE-scatter arrived halves // local fanout of the current round's newly received sub-chunks via LSA
//     PutGroupDone              // retire the current round's outbound puts
//   IntraNodeBarrier            // gates user code reading recvbuf
static ncclResult_t ncclHierCeAllGatherRing(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  ncclResult_t ret = ncclSuccess;

  struct ncclRmaProxyState* rmaProxyState = &comm->rmaState.rmaProxyState;
  int baseCtx = comm->config.numRmaCtx;
  int railCtx = baseCtx;
  int myRank = comm->rank;
  int localRank = comm->localRank;
  int myNode = comm->node;
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
  size_t cwBytes = perRankBytes <= 1 ? perRankBytes : perRankBytes / 2;
  size_t ccwBytes = perRankBytes - cwBytes;
  bool agUseMulticast =
    ncclCeAllGatherUseMulticast(comm, perRankBytes, ncclCudaGraphValid(comm->planner.capturingGraph),
                                (const uint8_t*)sendbuff == (const uint8_t*)recvbuff + myRank * perRankBytes);
  int numCtx = ncclHierCollNumCtx(rmaProxyState, perRankBytes, persistent);

  struct ncclRmaProxyCtx* railProxyCtx = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[railCtx];
  struct ncclHierChunkPlan cwChunkPlan = {};
  struct ncclHierChunkPlan ccwChunkPlan = {};
  int startOps = ncclRmaProxyPutGroupStartNumOps(persistent);
  int doneOps = ncclRmaProxyPutGroupDoneNumOps(persistent);
  int* ctxOps = nullptr;
  int* cwCtxOps = nullptr;
  int* ccwCtxOps = nullptr;
  int* ctxFill = nullptr;
  int* cwChunkFill = nullptr;
  int* ccwChunkFill = nullptr;
  int* activeCtxs = nullptr;
  int** cwChunksByCtx = nullptr;
  int** ccwChunksByCtx = nullptr;
  uint8_t** scatterPeerBases = nullptr;
  int* waitNPeersByActiveCtx = nullptr;
  int** waitPeersByActiveCtx = nullptr;
  int** waitSignalsByActiveCtx = nullptr;
  struct ncclRmaProxyDesc** waitDesc = nullptr;
  struct ncclRmaProxyDesc** groupDesc = nullptr;
  struct ncclRmaPutSignalOp** groupOps = nullptr;
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams[2] = {nullptr, nullptr};
  CUstreamBatchMemOpParams* groupDoneParams[2] = {nullptr, nullptr};
  CUstreamBatchMemOpParams* waitBatch = nullptr;
  struct ncclCeBatchOpsParams ceBcastOps = {};
  struct ncclCeBatchOpsParams ceScatterOps = {};
  uint8_t* scatterMcBase = nullptr;
  struct ncclHierCeAllGatherRingRound* roundMeta = nullptr;
  struct ncclHierCeAllGatherRingIssueState issueState = {};

  NCCLCHECKGOTO(ncclCalloc(&ctxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&cwCtxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ccwCtxOps, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ctxFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&cwChunkFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ccwChunkFill, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&activeCtxs, numCtx), ret, fail);
  if (lsaSize > 1) {
    NCCLCHECKGOTO(ncclCalloc(&scatterPeerBases, lsaSize - 1), ret, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&cwChunksByCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ccwChunksByCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitNPeersByActiveCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitPeersByActiveCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitSignalsByActiveCtx, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&waitDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupDesc, numCtx), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&groupOps, numCtx), ret, fail);

  // ====================================================================
  // Phase 1: Rail sync (cross-node entry barrier)
  // ====================================================================
  NCCLCHECKGOTO(ncclRailSync(comm, railProxyCtx, plan, railCtx, stream), ret, fail);

  // Build one-peer chunking that is reused by every ring round.
  if (nRemoteNodes > 0) {
    size_t maxChunk = ncclHierCollRingChunkWidth(perRankBytes, cwBytes, ccwBytes, numCtx);
    if (cwBytes > 0) {
      NCCLCHECKGOTO(ncclHierCollBuildChunk(cwBytes, 1, maxChunk, &cwChunkPlan), ret, fail);
      for (int c = cwChunkPlan.chunkStart[0]; c < cwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        ctxOps[k]++;
        cwCtxOps[k]++;
      }
    }
    if (ccwBytes > 0) {
      NCCLCHECKGOTO(ncclHierCollBuildChunk(ccwBytes, 1, maxChunk, &ccwChunkPlan), ret, fail);
      for (int c = ccwChunkPlan.chunkStart[0]; c < ccwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        ctxOps[k]++;
        ccwCtxOps[k]++;
      }
    }
    for (int k = 0; k < numCtx; k++) {
      if (ctxOps[k] == 0) continue;
      activeCtxs[nActiveCtx] = k;
      nActiveCtx++;
    }

    for (int k = 0; k < numCtx; k++) {
      if (cwCtxOps[k] > 0) NCCLCHECKGOTO(ncclCalloc(&cwChunksByCtx[k], cwCtxOps[k]), ret, fail);
      if (ccwCtxOps[k] > 0) NCCLCHECKGOTO(ncclCalloc(&ccwChunksByCtx[k], ccwCtxOps[k]), ret, fail);
    }
    if (cwBytes > 0) {
      for (int c = cwChunkPlan.chunkStart[0]; c < cwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        cwChunksByCtx[k][cwChunkFill[k]++] = c;
      }
    }
    if (ccwBytes > 0) {
      for (int c = ccwChunkPlan.chunkStart[0]; c < ccwChunkPlan.chunkStart[1]; c++) {
        int k = c % numCtx;
        ccwChunksByCtx[k][ccwChunkFill[k]++] = c;
      }
    }

    for (int i = 0; i < 2; i++) {
      NCCLCHECKGOTO(ncclCalloc(&groupStartParams[i], (size_t)nActiveCtx * startOps), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&groupDoneParams[i], (size_t)nActiveCtx * doneOps), ret, fail);
    }
  }

  // ====================================================================
  // Phase 2: Initial intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

  if (agUseMulticast) {
    void* mcBase = nullptr;
    NCCLCHECKGOTO(ncclDevrGetLsaTeamPtrMC(comm, recvWin, 0, ncclTeamLsa(comm), &mcBase), ret, fail);
    scatterMcBase = (uint8_t*)mcBase;
  } else if (lsaSize > 1) {
    for (int r = 1; r < lsaSize; r++) {
      int targetLsaRank = (myLsaRank + r) % lsaSize;
      void* peerBase = nullptr;
      NCCLCHECKGOTO(ncclDevrGetLsaRankPtr(comm, recvWin, 0, targetLsaRank, &peerBase), ret, fail);
      scatterPeerBases[r - 1] = (uint8_t*)peerBase;
    }
  }

  // ====================================================================
  // Phase 3: Self-broadcast (intra-node CE broadcast of own chunk)
  // ====================================================================
  NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceBcastOps, lsaSize), ret, fail);
  {
    uint8_t* myRecvSlot = (uint8_t*)recvbuff + myRank * perRankBytes;
    size_t offset = myRecvSlot - (uint8_t*)recvWin->userPtr;

    if (agUseMulticast) {
      ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
      ceBcastOps.dsts[ceBcastOps.numOps] = (void*)(scatterMcBase + offset);
      ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
      ceBcastOps.numOps++;
    } else {
      if (myRecvSlot != (const uint8_t*)sendbuff) {
        ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
        ceBcastOps.dsts[ceBcastOps.numOps] = (void*)myRecvSlot;
        ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
        ceBcastOps.numOps++;
      }

      for (int r = 1; r < lsaSize; r++) {
        ceBcastOps.srcs[ceBcastOps.numOps] = (void*)sendbuff;
        ceBcastOps.dsts[ceBcastOps.numOps] = (void*)(scatterPeerBases[r - 1] + offset);
        ceBcastOps.sizes[ceBcastOps.numOps] = perRankBytes;
        ceBcastOps.numOps++;
      }
    }

    NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceBcastOps, stream, args), ret, fail);
  }

  // ====================================================================
  // Phase 4: Ring rounds (next-node send, previous-node receive)
  // ====================================================================
  if (nRemoteNodes > 0 && nActiveCtx > 0) {
    int nRounds = nNodes - 1;
    int nextNode = (myNode + 1) % nNodes;
    int prevNode = (myNode - 1 + nNodes) % nNodes;
    int sendRailPeerCw = comm->nodeRanks[nextNode].localRankToRank[localRank];
    int recvRailPeerCw = comm->nodeRanks[prevNode].localRankToRank[localRank];
    int sendRailPeerCcw = comm->nodeRanks[prevNode].localRankToRank[localRank];
    int recvRailPeerCcw = comm->nodeRanks[nextNode].localRankToRank[localRank];
    int nRoundChunks = (cwChunkPlan.chunkStart ? (cwChunkPlan.chunkStart[1] - cwChunkPlan.chunkStart[0]) : 0) +
                       (ccwChunkPlan.chunkStart ? (ccwChunkPlan.chunkStart[1] - ccwChunkPlan.chunkStart[0]) : 0);
    int scatterCapacity = agUseMulticast ? nRoundChunks : nRoundChunks * (lsaSize - 1);

    for (int ai = 0; ai < nActiveCtx; ai++) {
      int k = activeCtxs[ai];
      int waitNPeers = 0;
      NCCLCHECKGOTO(ncclCalloc(&waitPeersByActiveCtx[k], 2), ret, fail);
      NCCLCHECKGOTO(ncclCalloc(&waitSignalsByActiveCtx[k], 2), ret, fail);
      if (cwCtxOps[k] > 0) {
        waitPeersByActiveCtx[k][waitNPeers] = recvRailPeerCw;
        waitSignalsByActiveCtx[k][waitNPeers] = cwCtxOps[k];
        waitNPeers++;
      }
      if (ccwCtxOps[k] > 0) {
        waitPeersByActiveCtx[k][waitNPeers] = recvRailPeerCcw;
        waitSignalsByActiveCtx[k][waitNPeers] = ccwCtxOps[k];
        waitNPeers++;
      }
      waitNPeersByActiveCtx[k] = waitNPeers;
    }

    NCCLCHECKGOTO(ncclCalloc(&roundMeta, nRounds + 1), ret, fail);
    for (int step = 1; step <= nRounds; step++) {
      int originNodeSendCw = (myNode - (step - 1) + nNodes) % nNodes;
      int originNodeSendCcw = (myNode + (step - 1)) % nNodes;
      int originNodeRecvCw = (myNode - step + nNodes) % nNodes;
      int originNodeRecvCcw = (myNode + step) % nNodes;
      roundMeta[step].originRankSendCw = comm->nodeRanks[originNodeSendCw].localRankToRank[localRank];
      roundMeta[step].originRankSendCcw = comm->nodeRanks[originNodeSendCcw].localRankToRank[localRank];
      roundMeta[step].originRankRecvCw = comm->nodeRanks[originNodeRecvCw].localRankToRank[localRank];
      roundMeta[step].originRankRecvCcw = comm->nodeRanks[originNodeRecvCcw].localRankToRank[localRank];
      roundMeta[step].srcWinHostCw = step == 1 ? sendWin : recvWin;
      roundMeta[step].srcWinHostCcw = step == 1 ? sendWin : recvWin;
      roundMeta[step].srcBaseOffsetCw = step == 1 ?
                                          (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr :
                                          ((const uint8_t*)recvbuff + roundMeta[step].originRankSendCw * perRankBytes) -
                                            (const uint8_t*)recvWin->userPtr;
      roundMeta[step].dstBaseOffsetCw =
        ((const uint8_t*)recvbuff + roundMeta[step].originRankSendCw * perRankBytes) - (const uint8_t*)recvWin->userPtr;
      roundMeta[step].srcBaseOffsetCcw =
        (step == 1 ? (const uint8_t*)sendbuff - (const uint8_t*)sendWin->userPtr :
                     ((const uint8_t*)recvbuff + roundMeta[step].originRankSendCcw * perRankBytes) -
                       (const uint8_t*)recvWin->userPtr) +
        cwBytes;
      roundMeta[step].dstBaseOffsetCcw = ((const uint8_t*)recvbuff + roundMeta[step].originRankSendCcw * perRankBytes) -
                                         (const uint8_t*)recvWin->userPtr + cwBytes;
    }

    issueState.numCtx = numCtx;
    issueState.baseCtx = baseCtx;
    issueState.nActiveCtx = nActiveCtx;
    issueState.sendRailPeerCw = sendRailPeerCw;
    issueState.sendRailPeerCcw = sendRailPeerCcw;
    issueState.ctxOps = ctxOps;
    issueState.ctxFill = ctxFill;
    issueState.activeCtxs = activeCtxs;
    issueState.cwChunkPlan = &cwChunkPlan;
    issueState.ccwChunkPlan = &ccwChunkPlan;
    issueState.groupDesc = groupDesc;
    issueState.groupOps = groupOps;
    issueState.groupStartParams = groupStartParams;
    issueState.groupDoneParams = groupDoneParams;
    issueState.rounds = roundMeta;

    if (scatterCapacity > 0) NCCLCHECKGOTO(ncclCeInitBatchOpsParams(&ceScatterOps, scatterCapacity), ret, fail);

    NCCLCHECKGOTO(ncclHierCeAllGatherRingIssueRound(comm, plan, stream, &issueState, /*step=*/1, /*slot=*/0), ret,
                  fail);

    for (int step = 1; step <= nRounds; step++) {
      auto* meta = &roundMeta[step];
      int slot = (step - 1) & 1;

      // Wait for every active context before forwarding data read from recvBuff.
      // Collect all contexts' waits into one stream memop batch.
      int waitOpsTotal = 0;
      for (int ai = 0; ai < nActiveCtx; ai++) {
        int k = activeCtxs[ai];
        struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
        NCCLCHECKGOTO(ncclProxyBuildWaitPeersRing(comm, pc, plan, waitNPeersByActiveCtx[k], waitPeersByActiveCtx[k],
                                                  waitSignalsByActiveCtx[k], &waitDesc[k]),
                      ret, fail);
        if (waitDesc[k] != nullptr) waitOpsTotal += ncclRmaProxyWaitNumStreamOps(waitDesc[k]);
      }

      // The active contexts and their wait peer sets are invariant across rounds.
      if (waitBatch == nullptr) NCCLCHECKGOTO(ncclCalloc(&waitBatch, waitOpsTotal), ret, fail);
      int waitOff = 0;
      for (int ai = 0; ai < nActiveCtx; ai++) {
        int k = activeCtxs[ai];
        if (waitDesc[k] == nullptr) continue;
        struct ncclRmaProxyCtx* pc = (struct ncclRmaProxyCtx*)rmaProxyState->rmaProxyCtxs[baseCtx + k];
        int waitOps = ncclRmaProxyWaitNumStreamOps(waitDesc[k]);
        NCCLCHECKGOTO(ncclRmaProxyWaitParams(pc, waitDesc[k], waitBatch + waitOff), ret, fail);
        NCCLCHECKGOTO(ncclRmaProxyEnqueueDesc(pc, &waitDesc[k]), ret, fail);
        waitOff += waitOps;
      }
      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, waitOpsTotal, waitBatch), ret, fail);

      // Enqueue the next round as soon as its inputs are known to be ready,
      // before spending host time constructing this round's CE scatter batch.
      if (step < nRounds)
        NCCLCHECKGOTO(ncclHierCeAllGatherRingIssueRound(comm, plan, stream, &issueState, step + 1, slot ^ 1), ret,
                      fail);

      ceScatterOps.numOps = 0;
      for (int ai = 0; ai < nActiveCtx; ai++) {
        int k = activeCtxs[ai];
        if (cwBytes > 0) {
          for (int ci = 0; ci < cwCtxOps[k]; ci++) {
            int c = cwChunksByCtx[k][ci];
            size_t subBytes = cwChunkPlan.chunkBytes[c];
            size_t off = cwChunkPlan.chunkOff[c];
            uint8_t* chunkSlot = (uint8_t*)recvbuff + meta->originRankRecvCw * perRankBytes + off;
            size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

            if (agUseMulticast) {
              ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
              ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterMcBase + winOffset);
              ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
              ceScatterOps.numOps++;
            } else {
              for (int r = 1; r < lsaSize; r++) {
                ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
                ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterPeerBases[r - 1] + winOffset);
                ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
                ceScatterOps.numOps++;
              }
            }
          }
        }
        if (ccwBytes > 0) {
          for (int ci = 0; ci < ccwCtxOps[k]; ci++) {
            int c = ccwChunksByCtx[k][ci];
            size_t subBytes = ccwChunkPlan.chunkBytes[c];
            size_t off = ccwChunkPlan.chunkOff[c];
            uint8_t* chunkSlot = (uint8_t*)recvbuff + meta->originRankRecvCcw * perRankBytes + cwBytes + off;
            size_t winOffset = chunkSlot - (uint8_t*)recvWin->userPtr;

            if (agUseMulticast) {
              ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
              ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterMcBase + winOffset);
              ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
              ceScatterOps.numOps++;
            } else {
              for (int r = 1; r < lsaSize; r++) {
                ceScatterOps.srcs[ceScatterOps.numOps] = chunkSlot;
                ceScatterOps.dsts[ceScatterOps.numOps] = (void*)(scatterPeerBases[r - 1] + winOffset);
                ceScatterOps.sizes[ceScatterOps.numOps] = subBytes;
                ceScatterOps.numOps++;
              }
            }
          }
        }
      }

      if (ceScatterOps.numOps > 0) {
        NCCLCHECKGOTO(ncclCeLaunchBatchOps(comm, &ceScatterOps, stream, args), ret, fail);
      }

      NCCLCHECKGOTO(ncclCuStreamBatchMemOp(stream, nActiveCtx * doneOps, groupDoneParams[slot]), ret, fail);
    }
  }

  // ====================================================================
  // Phase 5: Final intra-node barrier
  // ====================================================================
  NCCLCHECKGOTO(ncclMemOpSync(comm, stream, args), ret, fail);

exit:
  ncclCeFreeBatchOpsParams(&ceBcastOps);
  ncclCeFreeBatchOpsParams(&ceScatterOps);
  if (groupDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (groupDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &groupDesc[k]);
    }
  }
  if (groupOps) {
    for (int k = 0; k < numCtx; k++) free(groupOps[k]);
  }
  free(groupOps);
  for (int i = 0; i < 2; i++) {
    free(groupStartParams[i]);
    free(groupDoneParams[i]);
  }
  free(groupDesc);
  free(ctxOps);
  free(cwCtxOps);
  free(ccwCtxOps);
  free(ctxFill);
  free(cwChunkFill);
  free(ccwChunkFill);
  free(activeCtxs);
  free(scatterPeerBases);
  free(roundMeta);
  if (waitPeersByActiveCtx) {
    for (int k = 0; k < numCtx; k++) free(waitPeersByActiveCtx[k]);
  }
  if (waitSignalsByActiveCtx) {
    for (int k = 0; k < numCtx; k++) free(waitSignalsByActiveCtx[k]);
  }
  free(waitNPeersByActiveCtx);
  free(waitPeersByActiveCtx);
  free(waitSignalsByActiveCtx);
  if (waitDesc) {
    for (int k = 0; k < numCtx; k++) {
      if (waitDesc[k] != nullptr) (void)ncclRmaProxyDestroyDesc(comm, &waitDesc[k]);
    }
  }
  free(waitDesc);
  free(waitBatch);
  if (cwChunksByCtx) {
    for (int k = 0; k < numCtx; k++) free(cwChunksByCtx[k]);
  }
  if (ccwChunksByCtx) {
    for (int k = 0; k < numCtx; k++) free(ccwChunksByCtx[k]);
  }
  free(cwChunksByCtx);
  free(ccwChunksByCtx);
  ncclHierCollFreeChunkPlan(&cwChunkPlan);
  ncclHierCollFreeChunkPlan(&ccwChunkPlan);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream) {
  size_t perRankBytes = plan->ceCollArgs->nElts * plan->ceCollArgs->eltSize;
  bool useRing = ncclParamHierCeCollAgRailRingEnable() > 0;
  const char* railAlgo = useRing ? "Ring" : "Direct";
  if (comm->rank == 0)
    INFO(NCCL_TUNING, "AllGather [Hierarchical CE]: %zu Bytes -> Rail %s RMA proxy + CE", perRankBytes, railAlgo);

  if (useRing) return ncclHierCeAllGatherRing(comm, plan, stream);
  return ncclHierCeAllGatherDirect(comm, plan, stream);
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
  int* ctxOps = nullptr; // [numCtx] ops assigned per ctx
  int* ctxFill = nullptr; // [numCtx] running fill index
  struct ncclRmaProxyDesc** groupDesc = nullptr; // [numCtx]
  struct ncclRmaPutSignalOp** groupOps = nullptr; // [numCtx]
  // Start/done memop params for all active contexts, one contiguous slice per
  // context, so each phase fires a single stream batch instead of one per ctx.
  int nActiveCtx = 0;
  CUstreamBatchMemOpParams* groupStartParams = nullptr; // [nActiveCtx * startOps]
  CUstreamBatchMemOpParams* groupDoneParams = nullptr; // [nActiveCtx * doneOps]
  // Per-context inbound wait descriptors (each covers the peers/counts whose chunks
  // landed on that context); their stream memops are likewise fired as one batch.
  int** waitPeers = nullptr; // [numCtx][]
  int** waitSigCounts = nullptr; // [numCtx][]
  int** waitSignalIdxs = nullptr; // [numCtx][] all-zero (hier-CE uses signal 0)
  struct ncclRmaProxyDesc** waitDesc = nullptr; // [numCtx]
  CUstreamBatchMemOpParams* waitBatch = nullptr; // [sum of per-ctx wait ops]
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
    int p = 0; // chunk plan slot index
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
      if (task->func != ncclFuncAllGather)
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
