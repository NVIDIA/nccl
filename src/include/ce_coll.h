/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CE_COLL_H_
#define NCCL_CE_COLL_H_

#include "nccl.h"
#include "nccl_common.h"
#include "bitops.h"
#include "profiler_event.h"

// Memory operations per rank for different synchronization protocols
#define NCCL_CE_SYNC_OPS_PER_RANK_MC 2
#define NCCL_CE_SYNC_OPS_PER_RANK_UC 3

enum ncclCeMethodId {
  ncclCeMethodId_Auto = -1, // Legacy CE tasks defer UC/MC selection until execution.
  ncclCeMethodId_AllGather_UC,
  ncclCeMethodId_AllGather_MC,
  ncclCeMethodId_Coll_P2P_UC,
  ncclCeMethodId_Count // Number of concrete methods; also means no CE method in tuning results.
};

struct ncclCeColl {
  bool initialized;
  uint8_t* baseUCSymReadyPtr;
  uint8_t* baseUCSymComplPtr;
  size_t baseUCSymReadyOffset;
  size_t baseUCSymComplOffset;
  uint32_t ceSeqNum;
  bool useCompletePtr;
  struct ncclDevrWindow* ceSyncWin;
  uint32_t* ceSeqNumDev; // Index 0 for the current value, 1 for GRAPH_SYNC_VALUE
};

struct ncclCeInitTask {
  struct ncclCeInitTask* next;
  struct ncclComm* comm;
};

struct alignas(16) ncclCeCollArgs {
  ncclFunc_t func;
  int method; // Selected CE method; Auto defers legacy UC/MC selection.
  int rootRank;
  ncclDataType_t datatype;
  size_t nElts;
  size_t eltSize;
  uint8_t* sendBuff;
  uint8_t* recvBuff;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  struct ncclProfilerEventRef apiEventHandle;
  struct ncclProfilerEventRef ceCollProfHandle;
  uint64_t userTag;           // Per-call profiler annotation (0 == untagged)
};

struct ncclCeBatchOpsParams {
  void** dsts;
  void** srcs;
  size_t* sizes;
  size_t numOps;
  // Split operations into stream-ordered round-robin waves.
  bool chunking;
  bool intraBatchSync;
  // Prefer batching even under graph capture; false retains the existing API policy.
  // Unsupported CUDA versions and legacy streams always use individual copies.
  bool preferBatch;
#if CUDART_VERSION >= 12080
  cudaMemcpyAttributes* attrs;
  size_t* attrIdxs;
  size_t numAttrs;
#endif
};

bool ncclCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                     ncclSymRegType_t winRegType, struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin);

bool ncclHierCeAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                         ncclSymRegType_t winRegType, struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin);

ncclResult_t ncclCeInit(struct ncclComm* comm);

ncclResult_t ncclCeFinalize(struct ncclComm* comm);

// Intra-LSA-rank barrier.
ncclResult_t ncclMemOpSync(struct ncclComm* comm, cudaStream_t stream, struct ncclCeCollArgs* profilerArgs = nullptr);

// Allocate / free internal arrays for a batch-ops parameter struct.
ncclResult_t ncclCeInitBatchOpsParams(struct ncclCeBatchOpsParams* params, int capacity);
void ncclCeFreeBatchOpsParams(struct ncclCeBatchOpsParams* params);

// Capability checks are shared by prediction and execution. A batching preference
// never overrides CUDA support or legacy-stream restrictions.
inline bool ncclCeMemcpyBatchSupported(int driverVersion, bool captured, bool legacyStream) {
  return CUDART_VERSION >= 12080 && driverVersion >= (captured ? 13050 : 12080) && !legacyStream;
}

// Submit a copy list using the supported CUDA API.
ncclResult_t ncclCeLaunchBatchOps(struct ncclComm* comm, struct ncclCeBatchOpsParams* params, cudaStream_t stream,
                                  struct ncclCeCollArgs* profilerArgs = nullptr);

ncclResult_t ncclLaunchCeColl(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t scheduleCeCollTaskToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan);

ncclResult_t ncclCeAllGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAllGatherUseMulticast(struct ncclComm* comm, size_t perRankBytes, int captured, int inPlace,
                                         bool* multicast);

ncclResult_t ncclCeScatter(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeGather(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclCeAlltoAll(struct ncclComm* comm, struct ncclCeCollArgs* args, cudaStream_t stream);

ncclResult_t ncclHierCeAllGather(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);

ncclResult_t ncclHierCeAlltoAll(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream);
#endif /* NCCL_CE_COLL_H_ */
