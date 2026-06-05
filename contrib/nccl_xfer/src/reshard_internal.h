/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Internal Function Declarations
 *
 * Cross-TU function declarations shared by the split library modules.
 * Each module (reshard_config, reshard_cache, reshard_mesh, etc.)
 * exports its public-to-the-library functions here.
 *
 * Not part of the public C API — intentionally kept inside src/.
 ************************************************************************/

#ifndef NCCLXFER_RESHARD_INTERNAL_H_
#define NCCLXFER_RESHARD_INTERNAL_H_

#include "nccl_xfer.h"
#include "reshard_types.h"
#include "reshard_log.h"

struct ncclDevComm;

/* ======================================================================
 * Global configuration (inline — getters fold into a single load).
 *
 * Initial values are library defaults.  ncclXferReshardInit applies the
 * ncclXferReshardConfig_t (if non-NULL) and then env vars in
 * reshard_config.cc.  Env vars always win.
 * ====================================================================*/

inline int gReshardGpusPerNode = 8;
inline int gReshardSrcDomainSize = 0;
inline int gReshardDstDomainSize = 0;
inline ReshardAlgorithm gReshardAlgorithm = RESHARD_ALGO_AUTO;
inline ReshardLoadBalanceMode gReshardLbMode = RESHARD_LB_UNIFORM;

/* Upper bound on pickNumCtas() output.  0 = unset (use DEFAULT_NUM_CTAS). */
inline int gReshardMaxCta = 0;

/* Resolved CTA count, computed once at ncclXferReshardInit from
 * gReshardMaxCta + DEFAULT_NUM_CTAS.  pickNumCtas reads this directly -
 * no per-call branch. */
inline int gReshardNumCtas = DEFAULT_NUM_CTAS;

/* Stream pool size populated at ncclXferReshardInit from
 *   NCCLXFER_RESHARD_STREAM_POOL_SIZE   (int, default 4)
 * Maximum number of distinct (ncclComm_t, cuda device) pairs the
 * pool will hold a stream+event for.  1:1 mapping — one stream and
 * one back-edge event per entry.  Values <= 0 disable the pool
 * entirely (default-stream callers run on the user's default stream
 * directly).  Values > STREAM_POOL_MAX_SIZE are capped (with a
 * warning).  Applies only to default-stream callers; explicit-stream
 * callers are unaffected. */
inline int gReshardStreamPoolSize = 4;

/* Byte-level chunk size used by the RING prepare path. Default is
 * CHUNK_SIZE_BYTES; overridable via NCCLXFER_RESHARD_CHUNK_SIZE.
 * Parsed once at init-time in applyReshardEnv — keeps prepareReshardParams
 * off the getenv path on every call. 0 means "use the compile-time default". */
inline size_t gReshardChunkSizeBytes = 0;

inline ReshardAlgorithm reshardGetAlgorithm() {
  return gReshardAlgorithm;
}
inline int reshardGetGpusPerNode() {
  return gReshardGpusPerNode;
}
inline int reshardGetSrcDomainSize() {
  return gReshardSrcDomainSize;
}
inline int reshardGetDstDomainSize() {
  return gReshardDstDomainSize;
}
inline ReshardLoadBalanceMode reshardGetLoadBalanceMode() {
  return gReshardLbMode;
}
inline int reshardGetStreamPoolSize() {
  return gReshardStreamPoolSize;
}

/* ======================================================================
 * reshard_config.cc — configuration appliers
 *
 * Applied in order from ncclXferReshardInit; env always overrides config.
 * ====================================================================*/
ncclResult_t applyReshardConfig(const ncclXferReshardConfig_t* config);
void applyReshardEnv();

/* Element-size lookup for the dtypes accepted by ncclXferReshardWithWindow.
 * Returns 0 for unsupported dtypes (the API rejects them at call time). */
inline size_t getNcclDtSize(ncclDataType_t t) {
  switch (t) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return 0;
  }
}

/* ======================================================================
 * Picker stubs for numCtas / elementsPerChunk
 *
 * Currently constant — return the value resolved once at
 * ncclXferReshardInit.  Signature intentionally future-aware
 * (`bytesPerRank`, `algo`) so an input-aware heuristic can drop in
 * without a caller change.
 * ====================================================================*/

inline int pickNumCtas(size_t bytesPerRank, ReshardAlgorithm algo) {
  (void)bytesPerRank;
  (void)algo;
  return gReshardNumCtas;
}

inline size_t pickElementsPerChunk(size_t bytesPerRank, ReshardAlgorithm algo) {
  (void)bytesPerRank;
  (void)algo;
  return DEFAULT_ELEMENTS_PER_CHUNK;
}

/* ======================================================================
 * reshard_cache.cc — DevComm and Window caches
 * ====================================================================*/

ncclDevComm* findCachedDevComm(ncclComm_t comm, int numCtas, int signalCount, cudaStream_t stream = nullptr);

ncclResult_t cacheDevComm(ncclComm_t comm, int numCtas, int signalCount, const ncclDevComm* devComm,
                          cudaStream_t stream = nullptr);

ncclWindow_t* findCachedInternalWindowByPtr(ncclComm_t comm, void* buffer, size_t size);

ncclResult_t cacheInternalWindow(ncclComm_t comm, void* buffer, size_t size, ncclWindow_t window);

/* Acquire a library-owned (stream, event) pair for callers that pass
 * the default stream (nullptr / cudaStreamLegacy / cudaStreamPerThread).
 * 1:1 mapping per (comm, dev) — lazy-creates the pair on first use;
 * subsequent calls for the same pair return the same handles.  Both
 * objects are owned by the cache and freed by cacheFinalize().  The
 * event is reused across calls so we don't pay cudaEvent{Create,
 * Destroy} per reshard.
 *
 * Pool-full fall-through: if a new (comm, dev) entry would exceed
 * NCCLXFER_RESHARD_STREAM_POOL_SIZE, returns ncclSuccess with *outStream
 * and *outEvent both set to nullptr (warns once).  Callers should
 * check that and run on the caller's default stream directly. */
ncclResult_t streamPoolAcquire(ncclComm_t comm, int dev, cudaStream_t* outStream, cudaEvent_t* outEvent);

void cacheFinalize();

/* ======================================================================
 * reshard_mesh.cc — Mesh analysis helpers
 * ====================================================================*/

void computeStrides(const size_t dims[], int ndims, size_t strides[]);

void computeMeshGroupInfo(const ncclXferReshardMesh_t* mesh, int worldRank, ncclXferMeshGroupInfo* info);

int getMeshRank(const ncclXferReshardMesh_t* mesh, const ncclXferMeshGroupInfo* info, int shardIdx, int repIdx);

void computeGlobalRange(const size_t localDims[], int ndims, int shardTensorDim, int shardIdx, size_t globalStart[],
                        size_t globalEnd[]);

bool computeOverlap(const size_t srcStart[], const size_t srcEnd[], const size_t dstStart[], const size_t dstEnd[],
                    int ndims, size_t overlapStart[], size_t overlapEnd[]);

void computeTransferPlan(const size_t srcDims[], const size_t srcStrides[], int srcShardDim, int srcShardIdx,
                         const size_t dstDims[], const size_t dstStrides[], int dstShardDim, int dstShardIdx, int ndims,
                         size_t elementsPerChunk, ncclXferTransferPlan* plan);

/* ======================================================================
 * reshard_loadbalance.cc — Replication load balancer
 * ====================================================================*/

int getNodeOfDestRep(const ncclXferRepLoadBalancer* lb, int dstRepIdx);
int getNumDestNodes(const ncclXferRepLoadBalancer* lb);

void getDestRepsOnNode(const ncclXferRepLoadBalancer* lb, int targetNode, int* repStart, int* repEnd);

void getDestRepsOnNodeRange(const ncclXferRepLoadBalancer* lb, int firstNode, int lastNode, int* repStart, int* repEnd);

void getTargetRepRange(const ncclXferRepLoadBalancer* lb, int srcRepIdx, int* repStart, int* repEnd);

int getSourceRepForDest(const ncclXferRepLoadBalancer* lb, int dstRepIdx);

/* ======================================================================
 * reshard_prepare.cc — Kernel parameter builders
 * ====================================================================*/

ncclXferReshardParams prepareReshardParams(
  int worldRank, const void* srcBuffer, const size_t srcTensorDims[], int ndims, const ncclXferReshardMesh_t* srcMesh,
  const void* dstBuffer, const size_t dstTensorDims[], const ncclXferReshardMesh_t* dstMesh, ncclWindow_t window,
  size_t elementsPerChunk, int numCtas, int srcGpusPerDomain, int dstGpusPerDomain, const size_t* allWindowOffsets);

ncclXferDirectReshardParams prepareDirectReshardParams(
  int worldRank, const size_t srcTensorDims[], const size_t dstTensorDims[], int ndims,
  const ncclXferReshardMesh_t* srcMesh, const ncclXferReshardMesh_t* dstMesh, ncclWindow_t window,
  size_t elementsPerChunk, int numCtas, const size_t* allWindowOffsets);

/* ======================================================================
 * reshard_transpose.cc — Cross-dim transpose buffer
 * ====================================================================*/

bool shouldTransposeForCrossDim(const size_t* srcDimsBytes, const size_t* dstDimsBytes, int ndims, int srcShardDim,
                                int dstShardDim, int srcShardCount, int dstShardCount, int* swapDimA, int* swapDimB);

ncclResult_t ensureTransposeBuffer(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream);
void* getTransposeBuffer(ncclComm_t comm);
size_t getTransposeBufferCapacity(ncclComm_t comm);
void transposeBufferFinalize();
ncclResult_t transposeBufferRecordEvent(ncclComm_t comm, cudaStream_t stream);

#endif /* NCCLXFER_RESHARD_INTERNAL_H_ */
