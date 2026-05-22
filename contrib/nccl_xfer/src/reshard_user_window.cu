/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — User-Window Path
 *
 * Single-shot resharding entry point that takes a caller-registered
 * ncclWindow_t and runs the RING or DIRECT algorithm.
 *
 * Caller contract (single-offset, symmetric, single window):
 *   - On every rank that participates, src/dst buffers must lie within
 *     the registered window, and srcBuffer and dstBuffer must share
 *     the same offset within it (single-offset assumption — the kernel
 *     uses one params.myWindowOffset field per rank).
 *   - All ranks must agree on that offset (symmetric assumption).
 *   - Window registered on the input comm (NOT a node-local sub-comm).
 *   - LSA fan-out walks the input comm's LSA team.
 *
 * Algorithm selection follows the NCCLXFER_RESHARD_ALGORITHM env var:
 *   RING   -> reshardKernelUserWindow
 *   DIRECT -> directReshardKernelUserWindow
 *   AUTO   -> RING
 ************************************************************************/

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cuda_runtime.h"
#include "nccl.h"
#include "nccl_device.h"

#include "nccl_xfer.h"
#include "reshard_types.h"
#include "reshard_checks.h"
#include "reshard_log.h"
#include "reshard_internal.h"
#include "reshard_kernels.cuh"

/* Error macros — aliases to the unified definitions in reshard_checks.h. */
#define UW_NCCLCHECK NCCLXFER_CHECK
#define UW_CUDACHECK NCCLXFER_CUDACHECK

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
#define NCCLXFER_RESHARD_GIN_FINAL_FENCE (ncclGinFenceLevel::Put | ncclGinFenceLevel::Get)
#else
#define NCCLXFER_RESHARD_GIN_FINAL_FENCE ncclGinFenceLevel::Relaxed
#endif

// ============================================================================
// Byte-level transpose kernel: [D0, D1, D2] -> [D0, D2, D1]  (row-major)
// ============================================================================

__global__ void uwTranspose2DInnerKernel(const char* __restrict__ in, char* __restrict__ out, size_t D0, size_t D1,
                                         size_t D2) {
  size_t total = D0 * D1 * D2;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;

  for (size_t i = idx; i < total; i += stride) {
    size_t d2 = i % D2;
    size_t rem = i / D2;
    size_t d1 = rem % D1;
    size_t d0 = rem / D1;

    size_t j = d0 * (D2 * D1) + d2 * D1 + d1;
    out[j] = in[i];
  }
}

static void uwLaunchTranspose2DInner(const void* in, void* out, size_t D0, size_t D1, size_t D2, cudaStream_t stream) {
  size_t total = D0 * D1 * D2;
  int blockSize = 256;
  size_t needed = (total + blockSize - 1) / blockSize;
  int numBlocks = (int)std::min(needed, (size_t)65535);
  uwTranspose2DInnerKernel<<<numBlocks, blockSize, 0, stream>>>((const char*)in, (char*)out, D0, D1, D2);
}

// ============================================================================
// RING (hierarchical) Kernel — User-Window Variant
//
// Uses the GLOBAL window for local-buffer access and LSA fan-out, resolving
// peer pointers via world-rank arithmetic.
// ============================================================================

__global__ __launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) void reshardKernelUserWindow(ncclXferReshardParams params,
                                                                                          struct ncclDevComm devComm) {
  int numContexts = min((int)gridDim.x, (int)devComm.ginContextCount);
  int ctasPerContext = (int)gridDim.x / numContexts;
  int ginContext = (int)blockIdx.x / ctasPerContext;
  ncclGin gin{devComm, ginContext};

  ncclTeam world = ncclTeamWorld(devComm);
  ncclTeam lsa = ncclTeamLsa(devComm);

  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  // [USER-WINDOW] Local pointer comes from the GLOBAL window.  Offset is
  // zero by contract but
  // we still pass params.myWindowOffset for symmetry.
  char* localBuffer = (char*)ncclGetLocalPointer(params.window, params.myWindowOffset);

  __shared__ uint64_t initialSignals[MAX_SOURCES];
  // Compile-time guard: shared-array dim must be at least the prep-side cap
  // on params.numSources.
  static_assert(sizeof(initialSignals) / sizeof(initialSignals[0]) >= MAX_SOURCES,
                "initialSignals[] must be sized at least MAX_SOURCES — "
                "kernel reads initialSignals[i] for i < params.numSources, "
                "and prepareReshardParams caps params.numSources at MAX_SOURCES");

  // Read initial signals (dest ranks only)
  if (params.isDest && params.numSources > 0) {
    if (threadIdx.x < params.numSources) {
      unsigned int signalIdx = params.sources[threadIdx.x].signalBase + blockIdx.x;
      initialSignals[threadIdx.x] = gin.readSignal(signalIdx);
    }
  }
  __syncthreads();

  // Initial barrier
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  // SOURCE: send data to targets
  if (params.isSource) {
    int activeSrcWarps = min(MAX_SRC_WARPS, (int)(blockDim.x / 32));
    if (warpId < activeSrcWarps) {
      for (int t = warpId; t < params.numTargets; t += activeSrcWarps) {
        ncclXferTargetInfo& target = params.targets[t];
        ncclXferTransferPlan& plan = target.plan;

        if (plan.totalInnerTransfers == 0) continue;

        unsigned int signalIdx = params.myWorldRank * params.totalCtas + blockIdx.x;

        if (target.isContiguous) {
          size_t totalSize = target.totalBytes;
          size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
          size_t myStart = blockIdx.x * bytesPerCta;
          size_t myEnd = min(myStart + bytesPerCta, totalSize);

          if (myStart < totalSize && laneId == 0) {
            gin.put(world, target.dstWorldRank, params.window, target.windowOffset + plan.dstBaseOffset + myStart,
                    params.window, params.myWindowOffset + plan.srcBaseOffset + myStart, myEnd - myStart,
                    ncclGin_SignalInc{signalIdx});
          }
        } else {
          size_t itersPerCta = (plan.totalInnerTransfers + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, plan.totalInnerTransfers);
          size_t myTotalBytes = (myIterEnd - myIterStart) * plan.innerSize;
          const size_t csb = params.chunkSizeBytes;
          size_t numChunks = (myTotalBytes + csb - 1) / csb;

          for (size_t chunk = 0; chunk < numChunks; chunk++) {
            size_t byteStart = chunk * csb;
            size_t remaining = myTotalBytes - byteStart;
            size_t thisBytes = (csb < remaining) ? csb : remaining;

            if (laneId == 0) {
              emitStridedChunkPuts(gin, world, target.dstWorldRank, params.window, plan, params.ndims, myIterStart,
                                   byteStart, thisBytes, signalIdx,
                                   /*useDstAsSrc=*/false, params.myWindowOffset, target.windowOffset);
            }
          }
        }
      }
    }
  }

  // DEST: receive and replicate
  if (params.isDest && params.numSources > 0) {
    int warpsPerCta = blockDim.x / 32;
    int activeSources = min(params.numSources, min(warpsPerCta, (int)MAX_WARP_GROUPS));
    int warpsPerSource = warpsPerCta / activeSources;
    if (warpsPerSource < 1) warpsPerSource = 1;

    int mySourceGroup = warpId / warpsPerSource;
    int warpInGroup = warpId % warpsPerSource;
    int groupStartWarp = mySourceGroup * warpsPerSource;
    bool isActive = (mySourceGroup < activeSources);

    // [USER-WINDOW] lsaStartRank = world rank of LSA-rank-0 on this
    // rank's LSA team within the input comm.  Used to translate world
    // ranks to LSA ranks for ncclGetLsaPointer on the global window.
    const int lsaStartRank = world.rank - lsa.rank;

    for (int srcOffset = mySourceGroup; srcOffset < params.numSources && isActive; srcOffset += activeSources) {
      ncclXferSourceInfo& source = params.sources[srcOffset];
      ncclXferTransferPlan& plan = source.plan;

      int barrierId = mySourceGroup;
      ncclCoopWarpSpan warps(groupStartWarp, warpsPerSource, barrierId);

      unsigned int signalIdx = source.signalBase + blockIdx.x;

      if (source.isContiguous) {
        size_t totalSize = source.totalBytes;
        size_t bytesPerCta = (totalSize + params.totalCtas - 1) / params.totalCtas;
        size_t myStart = blockIdx.x * bytesPerCta;
        size_t myEnd = min(myStart + bytesPerCta, totalSize);

        if (warpInGroup == 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + 1);
        warps.sync();

        // Ring forward
        if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
          if (myStart < totalSize) {
            gin.put(world, params.ringNextWorldRank, params.window,
                    params.ringNextWindowOffset + plan.dstBaseOffset + myStart, params.window,
                    params.myWindowOffset + plan.dstBaseOffset + myStart, myEnd - myStart,
                    ncclGin_SignalInc{signalIdx});
          }
        }

        // [USER-WINDOW] LSA fan-out via the global window keyed by
        // world-rank arithmetic.
        if (params.numLocalFollowers > 0 && myStart < totalSize) {
          int threadsInGroup = warpsPerSource * 32;
          int threadInGroup = warpInGroup * 32 + laneId;

          char* srcPtr = localBuffer + plan.dstBaseOffset + myStart;
          size_t chunkSize = myEnd - myStart;
          size_t dstByteOffset = plan.dstBaseOffset + myStart;

          lsaReplicateChunk(srcPtr, chunkSize, dstByteOffset,
                            /*fallbackWindow=*/params.window, params.localFollowerWorldRanks,
                            params.localFollowerWindowOffsets, lsaStartRank, params.numLocalFollowers, threadsInGroup,
                            threadInGroup);
        }

        warps.sync();
      } else {
        size_t itersPerCta = (plan.totalInnerTransfers + params.totalCtas - 1) / params.totalCtas;
        size_t myIterStart = blockIdx.x * itersPerCta;
        size_t myIterEnd = min(myIterStart + itersPerCta, plan.totalInnerTransfers);
        size_t myTotalBytes = (myIterEnd - myIterStart) * plan.innerSize;
        const size_t csb = params.chunkSizeBytes;
        size_t numChunks = (myTotalBytes + csb - 1) / csb;

        for (size_t chunk = 0; chunk < numChunks; chunk++) {
          size_t byteStart = chunk * csb;
          size_t remaining = myTotalBytes - byteStart;
          size_t thisBytes = (csb < remaining) ? csb : remaining;

          if (warpInGroup == 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignals[srcOffset] + chunk + 1);
          warps.sync();

          // Ring forward this chunk
          if (!params.isRingLast && warpInGroup == 0 && laneId == 0) {
            emitStridedChunkPuts(gin, world, params.ringNextWorldRank, params.window, plan, params.ndims, myIterStart,
                                 byteStart, thisBytes, signalIdx,
                                 /*useDstAsSrc=*/true, params.myWindowOffset, params.ringNextWindowOffset);
          }

          // [USER-WINDOW] LSA fan-out via the global window for
          // strided chunks.
          if (params.numLocalFollowers > 0 && thisBytes > 0) {
            int threadsInGroup = warpsPerSource * 32;
            int threadInGroup = warpInGroup * 32 + laneId;

            const size_t inner = plan.innerSize;
            size_t lsaIter = byteStart / inner;
            size_t lsaOffInIter = byteStart % inner;
            size_t lsaRemaining = thisBytes;

            while (lsaRemaining > 0) {
              size_t avail = inner - lsaOffInIter;
              size_t piece = (avail < lsaRemaining) ? avail : lsaRemaining;

              size_t srcOff, dstOff;
              computeTransferOffset(plan, myIterStart + lsaIter, params.ndims, &srcOff, &dstOff);

              char* piecePtr = localBuffer + dstOff + lsaOffInIter;
              size_t dstByteOffset = dstOff + lsaOffInIter;

              lsaReplicateChunk(piecePtr, piece, dstByteOffset,
                                /*fallbackWindow=*/params.window, params.localFollowerWorldRanks,
                                params.localFollowerWindowOffsets, lsaStartRank, params.numLocalFollowers,
                                threadsInGroup, threadInGroup);

              lsaRemaining -= piece;
              lsaIter++;
              lsaOffInIter = 0;
            }
          }
        }
      }
    }
  }

  __threadfence_system();
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 5)
  __syncthreads();
#else
  // Older GIN barriers do not drain puts; flush also synchronizes the CTA.
  gin.flush(ncclCoopCta());
#endif

  // Final barrier
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCLXFER_RESHARD_GIN_FINAL_FENCE);
}

// ============================================================================
// DIRECT Algorithm Kernel — User-Window Variant
// ============================================================================

// clang-format off
__global__
__launch_bounds__(DEFAULT_KERNEL_MAX_NTHREADS, 1) void
directReshardKernelUserWindow(
    ncclXferDirectReshardParams params,
    struct ncclDevComm      devComm)
// clang-format on
{
  int numContexts = min((int)gridDim.x, (int)devComm.ginContextCount);
  int ctasPerContext = (int)gridDim.x / numContexts;
  int ginContext = (int)blockIdx.x / ctasPerContext;
  ncclGin gin{devComm, ginContext};

  ncclTeam world = ncclTeamWorld(devComm);

  int warpId = threadIdx.x / 32;
  int laneId = threadIdx.x % 32;

  __shared__ uint64_t initialSignals[MAX_DIRECT_SOURCES];
  // Compile-time guard: shared-array dim must be at least the prep-side cap
  // on params.numSources.
  static_assert(sizeof(initialSignals) / sizeof(initialSignals[0]) >= MAX_DIRECT_SOURCES,
                "initialSignals[] must be sized at least MAX_DIRECT_SOURCES — "
                "kernel reads initialSignals[i] for i < params.numSources, "
                "and prepareDirectReshardParams caps params.numSources at "
                "MAX_DIRECT_SOURCES");

  if (params.isDest && params.numSources > 0) {
    if (threadIdx.x < params.numSources) {
      unsigned int signalIdx = params.sources[threadIdx.x].signalBase + blockIdx.x;
      initialSignals[threadIdx.x] = gin.readSignal(signalIdx);
    }
  }
  __syncthreads();

  // Initial barrier
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
  ncclGinBarrierSession<ncclCoopCta> bar{ncclCoopCta(), gin, ncclTeamTagWorld(), blockIdx.x};
#else
  ncclBarrierSession<ncclCoopCta> bar{ncclCoopCta(), ncclTeamTagWorld(), gin, blockIdx.x};
#endif
  bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

  if (params.isSource && params.numTargets > 0) {
    bool isGinWarp = (warpId == 0);

    if (isGinWarp) {
      for (int t = 0; t < params.numTargets; t++) {
        ncclXferDirectTargetInfo& target = params.targets[t];

        if (target.isContiguous) {
          size_t totalBytes = target.totalBytes;
          size_t bytesPerCta = (totalBytes + params.totalCtas - 1) / params.totalCtas;
          size_t myByteStart = blockIdx.x * bytesPerCta;
          size_t myByteEnd = min(myByteStart + bytesPerCta, totalBytes);

          if (myByteStart < totalBytes) {
            size_t myBytes = myByteEnd - myByteStart;
            size_t srcOffset = params.myWindowOffset + target.plan.srcBaseOffset + myByteStart;
            size_t dstOffset = target.windowOffset + target.plan.dstBaseOffset + myByteStart;

            unsigned int signalIdx = params.myWorldRank * params.totalCtas + blockIdx.x;

            if (laneId == 0) {
              gin.put(world, target.dstWorldRank, params.window, dstOffset, params.window, srcOffset, myBytes,
                      ncclGin_SignalInc{signalIdx});
            }
          }
        } else {
          size_t totalIters = target.plan.totalInnerTransfers;
          size_t itersPerCta = (totalIters + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, totalIters);

          if (myIterStart < totalIters) {
            for (size_t iter = myIterStart; iter < myIterEnd; iter++) {
              size_t srcOffset, dstOffset;
              computeTransferOffset(target.plan, iter, params.ndims, &srcOffset, &dstOffset);
              srcOffset += params.myWindowOffset;
              dstOffset += target.windowOffset;

              unsigned int signalIdx = params.myWorldRank * params.totalCtas + blockIdx.x;

              if (laneId == 0) {
                gin.put(world, target.dstWorldRank, params.window, dstOffset, params.window, srcOffset,
                        target.plan.innerSize, ncclGin_SignalInc{signalIdx});
              }
            }
          }
        }
      }
    }
  }

  if (params.isDest && params.numSources > 0) {
    if (warpId == 0) {
      for (int s = 0; s < params.numSources; s++) {
        ncclXferDirectSourceInfo& source = params.sources[s];
        unsigned int signalIdx = source.signalBase + blockIdx.x;
        uint64_t initialSignal = initialSignals[s];

        size_t mySignals;
        if (source.isContiguous) {
          size_t totalBytes = source.totalBytes;
          size_t bytesPerCta = (totalBytes + params.totalCtas - 1) / params.totalCtas;
          size_t myByteStart = blockIdx.x * bytesPerCta;
          mySignals = (myByteStart < totalBytes) ? 1 : 0;
        } else {
          size_t totalIters = source.plan.totalInnerTransfers;
          size_t itersPerCta = (totalIters + params.totalCtas - 1) / params.totalCtas;
          size_t myIterStart = blockIdx.x * itersPerCta;
          size_t myIterEnd = min(myIterStart + itersPerCta, totalIters);
          mySignals = (myIterStart < totalIters) ? (myIterEnd - myIterStart) : 0;
        }

        if (mySignals > 0) gin.waitSignal(ncclCoopWarp(), signalIdx, initialSignal + mySignals);
      }
    }
  }

  __syncthreads();
  gin.flush(ncclCoopCta());

  // Final barrier
  bar.sync(ncclCoopCta(), cuda::memory_order_acquire, NCCLXFER_RESHARD_GIN_FINAL_FENCE);
}

// ============================================================================
// Host: ncclXferReshardWithWindow
// ============================================================================

ncclResult_t ncclXferReshardWithWindow(ncclComm_t comm, ncclWindow_t window, const ncclXferDistTensor_t* src,
                                       const ncclXferDistTensor_t* dst, cudaStream_t stream) {
  /* Required handles. */
  if (comm == nullptr || window == nullptr) {
    fprintf(stderr, "[ncclXferReshardWithWindow] comm and window must both be "
                    "non-null\n");
    return ncclInvalidArgument;
  }
  /* Both descriptors required — each carries one side's mesh and the
     library reads both meshes on every rank.  A rank that does not
     have data on a given side still passes a fully-formed descriptor
     with dataPtr=NULL (the same convention PyTorch DTensor uses with
     a size-0 local tensor on non-participating ranks). */
  if (src == nullptr || dst == nullptr) {
    fprintf(stderr, "[ncclXferReshardWithWindow] src and dst tensor descriptors "
                    "must both be non-null on every rank (use dataPtr=NULL "
                    "on the side this rank doesn't participate in)\n");
    return ncclInvalidArgument;
  }
  if (src->mesh == nullptr || dst->mesh == nullptr) {
    fprintf(stderr, "[ncclXferReshardWithWindow] src->mesh and dst->mesh must "
                    "both be non-null on every rank\n");
    return ncclInvalidArgument;
  }
  if (src->ndims != dst->ndims) {
    fprintf(stderr,
            "[ncclXferReshardWithWindow] src->ndims (%d) and dst->ndims (%d) "
            "must match\n",
            src->ndims, dst->ndims);
    return ncclInvalidArgument;
  }
  if (src->dtype != dst->dtype) {
    fprintf(stderr,
            "[ncclXferReshardWithWindow] src->dtype (%d) and dst->dtype (%d) "
            "must match\n",
            (int)src->dtype, (int)dst->dtype);
    return ncclInvalidArgument;
  }
  int ndims = src->ndims;
  ncclDataType_t dtype = src->dtype;
  void* srcBuffer = src->dataPtr;
  void* dstBuffer = dst->dataPtr;
  const size_t* srcTensorDims = src->localShape;
  const size_t* dstTensorDims = dst->localShape;
  const ncclXferReshardMesh_t* srcMesh = src->mesh;
  const ncclXferReshardMesh_t* dstMesh = dst->mesh;
  if (ndims < 1 || ndims > MAX_TENSOR_DIMS) {
    fprintf(stderr, "[ncclXferReshardWithWindow] ndims (%d) out of range [1, %d]\n", ndims, MAX_TENSOR_DIMS);
    return ncclInvalidArgument;
  }
  size_t elementSize = getNcclDtSize(dtype);
  if (elementSize == 0) {
    fprintf(stderr, "[ncclXferReshardWithWindow] unsupported data type %d\n", (int)dtype);
    return ncclInvalidArgument;
  }

  // Workaround: when both mesh dims have REPLICATE placement,
  // computeMeshGroupInfo loses track of the replication dimension.
  // Collapse dims to [total, 1] with placement[1] = SHARD(0) (a no-op
  // shard with count 1) so repMeshDim=0 is always well-defined.
  auto fixFullyReplicated = [](ncclXferReshardMesh_t* mesh) {
    if (mesh->placement[0] == NCCLXFER_RESHARD_REPLICATE && mesh->placement[1] == NCCLXFER_RESHARD_REPLICATE) {
      int total = mesh->dims[0] * mesh->dims[1];
      mesh->dims[0] = total;
      mesh->dims[1] = 1;
      mesh->placement[1] = NCCLXFER_RESHARD_SHARD(0);
    }
  };

  ncclXferReshardMesh_t srcMeshLocal = *srcMesh;
  ncclXferReshardMesh_t dstMeshLocal = *dstMesh;
  fixFullyReplicated(&srcMeshLocal);
  fixFullyReplicated(&dstMeshLocal);
  srcMesh = &srcMeshLocal;
  dstMesh = &dstMeshLocal;

  UW_NCCLCHECK(ncclXferReshardInit(nullptr));

  int worldRank, worldSize;
  UW_NCCLCHECK(ncclCommUserRank(comm, &worldRank));
  UW_NCCLCHECK(ncclCommCount(comm, &worldSize));

  // Match the comm's CUDA device.
  int currentCudaDev;
  UW_CUDACHECK(cudaGetDevice(&currentCudaDev));
  ncclCommProperties commProps = NCCL_COMM_PROPERTIES_INITIALIZER;
  ncclResult_t propsResult = ncclCommQueryProperties(comm, &commProps);
  if (propsResult == ncclSuccess && currentCudaDev != commProps.cudaDev) UW_CUDACHECK(cudaSetDevice(commProps.cudaDev));

  // Default-stream callers run on a library-owned non-blocking
  // stream from the pool; back-edge below makes subsequent default-
  // stream work observe our completion.  See nccl_xfer.h for the
  // full contract.  NCCLXFER_RESHARD_STREAM_POOL_SIZE=0 disables this
  // (forces legacy synchronizing-default-stream behavior).
  const bool isDefaultStream = (stream == nullptr || stream == cudaStreamLegacy || stream == cudaStreamPerThread);
  const bool wantPool = isDefaultStream && reshardGetStreamPoolSize() > 0;
  cudaStream_t workStream = stream;
  cudaEvent_t poolEvent = nullptr;
  if (wantPool) {
    const int dev = (propsResult == ncclSuccess) ? commProps.cudaDev : currentCudaDev;
    UW_NCCLCHECK(streamPoolAcquire(comm, dev, &workStream, &poolEvent));
    if (workStream == nullptr) {
      /* Pool full — streamPoolAcquire warned.  Run on the
       * caller's default stream directly for this call. */
      workStream = stream;
    }
  }
  /* True iff we got a pool slot (stream + event).  Drives the
   * back-edge cudaEventRecord/Wait below.  False both when the
   * caller passed an explicit stream and when the pool was full. */
  const bool acquiredPoolSlot = (poolEvent != nullptr);

  // ------------------------------------------------------------------
  // Single-offset contract.  Each rank verifies that srcBuffer and
  // dstBuffer (when both present) share the same offset within the
  // registered window — the kernel uses one params.myWindowOffset
  // field per rank, so the two sides must agree.  Cross-rank symmetry
  // (every rank computes the same offset) is trusted under this
  // release; the opt-in §1d diagnostic verifies it when needed.
  //
  // localOffset stays in scope through the rest of the function and
  // gets threaded into the kernel params at launch.
  // ------------------------------------------------------------------
  intptr_t localOffset = 0;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2)
  {
    void* winUserPtr = nullptr;
    UW_NCCLCHECK(ncclWinGetUserPtr(comm, window, &winUserPtr));
    if (winUserPtr == nullptr) {
      RESHARD_FATAL(worldRank, "ncclWinGetUserPtr returned nullptr for the supplied window; "
                               "ncclXferReshardWithWindow requires a symmetric-memory window "
                               "(NCCL_WIN_COLL_SYMMETRIC).");
    }

    const bool hasSrc = (srcBuffer != nullptr);
    const bool hasDst = (dstBuffer != nullptr);
    intptr_t srcOff = hasSrc ? (intptr_t)((char*)srcBuffer - (char*)winUserPtr) : 0;
    intptr_t dstOff = hasDst ? (intptr_t)((char*)dstBuffer - (char*)winUserPtr) : 0;

    if (hasSrc && hasDst && srcOff != dstOff) {
      RESHARD_FATAL(worldRank,
                    "srcBuffer and dstBuffer must share the same offset within "
                    "the registered window (single-offset contract); got "
                    "srcOff=%lld, dstOff=%lld (srcBuffer=%p, dstBuffer=%p, "
                    "window_user_ptr=%p).",
                    (long long)srcOff, (long long)dstOff, srcBuffer, dstBuffer, winUserPtr);
    }
    if (hasSrc && srcOff < 0) {
      RESHARD_FATAL(worldRank,
                    "srcBuffer lies before the window base (srcOff=%lld < 0); "
                    "srcBuffer=%p, window_user_ptr=%p.",
                    (long long)srcOff, srcBuffer, winUserPtr);
    }
    if (hasDst && dstOff < 0) {
      RESHARD_FATAL(worldRank,
                    "dstBuffer lies before the window base (dstOff=%lld < 0); "
                    "dstBuffer=%p, window_user_ptr=%p.",
                    (long long)dstOff, dstBuffer, winUserPtr);
    }

    // For fully-inactive ranks (both buffers null), 0 is a safe
    // placeholder — the kernel won't read params.myWindowOffset.
    localOffset = hasSrc ? srcOff : (hasDst ? dstOff : 0);
  }
#endif // NCCL_VERSION_CODE >= 2.29.2

  // Resolve algorithm.  AUTO falls through to RING.
  ReshardAlgorithm algo = reshardGetAlgorithm();
  if (algo == RESHARD_ALGO_AUTO) algo = RESHARD_ALGO_RING;

  // Per-rank byte volume for the picker — max of src/dst local shapes
  // (whichever side this rank participates in; both zero on inactive
  // ranks, which the picker handles by falling through to its default).
  auto localBytes = [&](void* buf, const size_t* dims) -> size_t {
    if (buf == nullptr) return 0;
    size_t b = elementSize;
    for (int d = 0; d < ndims; d++) b *= dims[d];
    return b;
  };
  size_t srcBytes = localBytes(srcBuffer, srcTensorDims);
  size_t dstBytes = localBytes(dstBuffer, dstTensorDims);
  size_t bytesPerRank = (srcBytes > dstBytes) ? srcBytes : dstBytes;

  int numCtas = pickNumCtas(bytesPerRank, algo);
  size_t elementsPerChunk = pickElementsPerChunk(bytesPerRank, algo);

  int srcTotal = srcMesh->dims[0] * srcMesh->dims[1];
  int ginSignalCount = srcTotal * numCtas;

  // ------------------------------------------------------------------
  // Get-or-create the global devComm on this stream.  This both gives us
  // the LSA team size (used as gpusPerDomain below) and is what gets
  // passed to the kernel launch.
  // ------------------------------------------------------------------
  ncclDevComm* devCommPtr = findCachedDevComm(comm, numCtas, ginSignalCount, workStream);
  ncclDevComm localDevComm;
  if (devCommPtr == nullptr) {
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 0)
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
#else
    ncclDevCommRequirements reqs = {};
#endif
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 30, 0)
    if (algo == RESHARD_ALGO_DIRECT) reqs.worldGinBarrierCount = numCtas;
    else reqs.barrierCount = numCtas;
#else
    reqs.lsaBarrierCount = numCtas;
    reqs.railGinBarrierCount = numCtas;
#endif
    reqs.ginSignalCount = ginSignalCount;
#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 3)
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
#else
    reqs.ginForceEnable = true;
#endif
    reqs.ginContextCount = DEFAULT_GIN_CONTEXT_COUNT;

    memset(&localDevComm, 0, sizeof(localDevComm));
    UW_NCCLCHECK(ncclDevCommCreate(comm, &reqs, &localDevComm));
    UW_NCCLCHECK(cacheDevComm(comm, numCtas, ginSignalCount, &localDevComm, workStream));
    devCommPtr = findCachedDevComm(comm, numCtas, ginSignalCount, workStream);
    if (devCommPtr == nullptr) devCommPtr = &localDevComm;
  }

  // Domain-size resolution: override > LSA team size > gpusPerNode.
  const int lsaSizeFromComm = (devCommPtr->lsaSize > 0) ? devCommPtr->lsaSize : 0;
  const int gpusPerNode = reshardGetGpusPerNode();
  const int srcOverride = reshardGetSrcDomainSize();
  const int dstOverride = reshardGetDstDomainSize();

  int srcGpusPerDomain;
  if (srcOverride > 0) srcGpusPerDomain = srcOverride;
  else if (lsaSizeFromComm > 0) srcGpusPerDomain = lsaSizeFromComm;
  else srcGpusPerDomain = (gpusPerNode > 0) ? gpusPerNode : 1;

  int dstGpusPerDomain;
  if (dstOverride > 0) dstGpusPerDomain = dstOverride;
  else if (lsaSizeFromComm > 0) dstGpusPerDomain = lsaSizeFromComm;
  else dstGpusPerDomain = (gpusPerNode > 0) ? gpusPerNode : 1;

  RESHARD_INFO(worldRank,
               "algo=%s, lsa_size=%d, srcGpusPerDomain=%d, dstGpusPerDomain=%d, "
               "(srcOverride=%d, dstOverride=%d, gpusPerNode=%d), numCtas=%d, "
               "ginSignalCount=%d",
               algo == RESHARD_ALGO_RING ? "RING" : "DIRECT", lsaSizeFromComm, srcGpusPerDomain, dstGpusPerDomain,
               srcOverride, dstOverride, gpusPerNode, numCtas, ginSignalCount);

  // ------------------------------------------------------------------
  // Convert dims to bytes (matches ncclXferReshardWithWindow's contract for
  // prepareReshardParams / prepareDirectReshardParams).
  // ------------------------------------------------------------------
  size_t srcDimsBytes[MAX_TENSOR_DIMS] = {0};
  size_t dstDimsBytes[MAX_TENSOR_DIMS] = {0};
  for (int d = 0; d < ndims; d++) {
    srcDimsBytes[d] = srcTensorDims ? srcTensorDims[d] : 1;
    dstDimsBytes[d] = dstTensorDims ? dstTensorDims[d] : 1;
  }
  srcDimsBytes[ndims - 1] *= elementSize;
  dstDimsBytes[ndims - 1] *= elementSize;

  // ------------------------------------------------------------------
  // Cross-dim transpose optimisation (3D only).
  //
  // When src and dst shard different dimensions and the dst shard dim
  // is innermost, each GIN put is tiny.  Transposing the last two
  // tensor dims makes the large unshard dim innermost, boosting RDMA
  // throughput.  The kernel itself is unchanged — only the buffer
  // layout and mesh placements are rewritten around it.
  // ------------------------------------------------------------------
  int srcShardTensorDim = -1, dstShardTensorDim = -1;
  int srcShardCountForXpose = 1, dstShardCountForXpose = 1;
  for (int i = 0; i < 2; i++) {
    if (IS_SHARD_PLACEMENT(srcMesh->placement[i])) {
      srcShardTensorDim = GET_SHARD_TENSOR_DIM(srcMesh->placement[i]);
      srcShardCountForXpose = srcMesh->dims[i];
    }
    if (IS_SHARD_PLACEMENT(dstMesh->placement[i])) {
      dstShardTensorDim = GET_SHARD_TENSOR_DIM(dstMesh->placement[i]);
      dstShardCountForXpose = dstMesh->dims[i];
    }
  }

  int swapA = -1, swapB = -1;
  bool doTranspose = shouldTransposeForCrossDim(srcDimsBytes, dstDimsBytes, ndims, srcShardTensorDim, dstShardTensorDim,
                                                srcShardCountForXpose, dstShardCountForXpose, &swapA, &swapB);

  // Effective dims / meshes / buffers / window — may be overwritten by
  // the transpose path below, otherwise equal to the originals.
  size_t effSrcDims[MAX_TENSOR_DIMS], effDstDims[MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) {
    effSrcDims[d] = srcDimsBytes[d];
    effDstDims[d] = dstDimsBytes[d];
  }
  ncclXferReshardMesh_t effSrcMesh = *srcMesh;
  ncclXferReshardMesh_t effDstMesh = *dstMesh;
  void* effSrcBuffer = srcBuffer;
  void* effDstBuffer = dstBuffer;
  ncclWindow_t effWindow = window;

  int srcMeshSize = srcMesh->dims[0] * srcMesh->dims[1];
  int dstMeshSize = dstMesh->dims[0] * dstMesh->dims[1];
  bool isSource = (worldRank >= srcMesh->startRank && worldRank < srcMesh->startRank + srcMeshSize);
  bool isDest = (worldRank >= dstMesh->startRank && worldRank < dstMesh->startRank + dstMeshSize);

  if (doTranspose) {
    RESHARD_INFO(worldRank,
                 "Cross-dim transpose: swapping dims %d and %d "
                 "(srcShard=%d, dstShard=%d)",
                 swapA, swapB, srcShardTensorDim, dstShardTensorDim);

    // 1. Swap the last two dims in effective dims
    std::swap(effSrcDims[swapA], effSrcDims[swapB]);
    std::swap(effDstDims[swapA], effDstDims[swapB]);

    // 2. Rewrite mesh placements to match swapped layout
    for (int i = 0; i < 2; i++) {
      if (IS_SHARD_PLACEMENT(effSrcMesh.placement[i])) {
        int td = GET_SHARD_TENSOR_DIM(effSrcMesh.placement[i]);
        if (td == swapA) effSrcMesh.placement[i] = NCCLXFER_RESHARD_SHARD(swapB);
        else if (td == swapB) effSrcMesh.placement[i] = NCCLXFER_RESHARD_SHARD(swapA);
      }
      if (IS_SHARD_PLACEMENT(effDstMesh.placement[i])) {
        int td = GET_SHARD_TENSOR_DIM(effDstMesh.placement[i]);
        if (td == swapA) effDstMesh.placement[i] = NCCLXFER_RESHARD_SHARD(swapB);
        else if (td == swapB) effDstMesh.placement[i] = NCCLXFER_RESHARD_SHARD(swapA);
      }
    }

    // 3. Allocate / grow the transpose buffer.
    //    ncclCommWindowRegister is collective — all ranks must hit or
    //    miss the cache together.  Reconstruct global dims (uniform
    //    across all ranks) then derive both local sizes so every rank
    //    requests the same buffer size regardless of src/dst role.
    //    (Same pattern as prepareReshardParams globalDims logic.)
    size_t globalDims[MAX_TENSOR_DIMS];
    for (int d = 0; d < ndims; d++) {
      if (isSource && srcDimsBytes[d] > 0) {
        globalDims[d] = srcDimsBytes[d];
        if (d == srcShardTensorDim) globalDims[d] *= srcShardCountForXpose;
      } else if (isDest && dstDimsBytes[d] > 0) {
        globalDims[d] = dstDimsBytes[d];
        if (d == dstShardTensorDim) globalDims[d] *= dstShardCountForXpose;
      } else {
        globalDims[d] = 1;
      }
    }
    size_t srcLocal = 1, dstLocal = 1;
    for (int d = 0; d < ndims; d++) {
      srcLocal *=
        (d == srcShardTensorDim && srcShardCountForXpose > 1) ? globalDims[d] / srcShardCountForXpose : globalDims[d];
      dstLocal *=
        (d == dstShardTensorDim && dstShardCountForXpose > 1) ? globalDims[d] / dstShardCountForXpose : globalDims[d];
    }
    size_t myLocalSize = std::max(srcLocal, dstLocal);
    UW_NCCLCHECK(ensureTransposeBuffer(comm, myLocalSize, workStream));

    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr,
                "[nccl-reshard][Rank %d] CUDA error after "
                "ensureTransposeBuffer: %s\n",
                worldRank, cudaGetErrorString(err));
        return ncclSystemError;
      }
      RESHARD_DEBUG(worldRank, "ensureTransposeBuffer: size=%zu, buf=%p", myLocalSize, getTransposeBuffer(comm));
    }

    // 4. PACK (source side): transpose user buffer -> transpose buffer
    if (isSource) {
      if (ndims == 2) {
        uwLaunchTranspose2DInner(srcBuffer, getTransposeBuffer(comm), 1, srcDimsBytes[0], srcDimsBytes[1], workStream);
      } else {
        uwLaunchTranspose2DInner(srcBuffer, getTransposeBuffer(comm), srcDimsBytes[0], srcDimsBytes[swapA],
                                 srcDimsBytes[swapB], workStream);
      }

      {
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          fprintf(stderr,
                  "[nccl-reshard][Rank %d] CUDA error after "
                  "transpose pack: %s\n",
                  worldRank, cudaGetErrorString(err));
          return ncclSystemError;
        }
        RESHARD_DEBUG(worldRank, "transpose pack: ndims=%d, D0=%zu, D1=%zu, D2=%zu", ndims, srcDimsBytes[0],
                      srcDimsBytes[swapA], srcDimsBytes[swapB]);
      }

      effSrcBuffer = getTransposeBuffer(comm);
    }

    // 5. DEST: kernel writes into transpose buffer; unpack afterwards
    if (isDest) effDstBuffer = getTransposeBuffer(comm);

    // 6. Register the transpose buffer as a window on comm (cached).
    //    This is collective — all ranks reach it because
    //    ncclXferReshardWithWindow is itself collective.
    ncclWindow_t* cached =
      findCachedInternalWindowByPtr(comm, getTransposeBuffer(comm), getTransposeBufferCapacity(comm));
    if (cached != nullptr) {
      effWindow = *cached;
    } else {
      ncclWindow_t xposeWin;
      UW_NCCLCHECK(ncclCommWindowRegister(comm, getTransposeBuffer(comm), getTransposeBufferCapacity(comm), &xposeWin,
                                          NCCL_WIN_COLL_SYMMETRIC));
      UW_NCCLCHECK(cacheInternalWindow(comm, getTransposeBuffer(comm), getTransposeBufferCapacity(comm), xposeWin));
      effWindow = xposeWin;
    }

    {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        fprintf(stderr,
                "[nccl-reshard][Rank %d] CUDA error after "
                "window register: %s\n",
                worldRank, cudaGetErrorString(err));
        return ncclSystemError;
      }
      RESHARD_DEBUG(worldRank, "window register: buf=%p, cap=%zu, effWindow=%p", getTransposeBuffer(comm),
                    getTransposeBufferCapacity(comm), (void*)effWindow);
    }

    RESHARD_DEBUG(worldRank,
                  "Transpose: buf=%p, cap=%zu, effSrcDims=[%zu,%zu,%zu], "
                  "effDstDims=[%zu,%zu,%zu]",
                  getTransposeBuffer(comm), getTransposeBufferCapacity(comm), effSrcDims[0],
                  ndims >= 2 ? effSrcDims[1] : (size_t)0, ndims >= 3 ? effSrcDims[2] : (size_t)0, effDstDims[0],
                  ndims >= 2 ? effDstDims[1] : (size_t)0, ndims >= 3 ? effDstDims[2] : (size_t)0);
  }

  int threadsPerCta = DEFAULT_KERNEL_MAX_NTHREADS;

  {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "[nccl-reshard][Rank %d] CUDA error pre-launch: %s\n", worldRank, cudaGetErrorString(err));
      return ncclSystemError;
    }
    RESHARD_DEBUG(worldRank,
                  "pre-launch: doTranspose=%d, algo=%s, numCtas=%d, threads=%d, "
                  "effWindow=%p, eff_src=%p, eff_dst=%p",
                  (int)doTranspose, algo == RESHARD_ALGO_RING ? "RING" : "DIRECT", numCtas, threadsPerCta,
                  (void*)effWindow, effSrcBuffer, effDstBuffer);
  }

  // ------------------------------------------------------------------
  // Build params + launch.  Under the single-offset symmetric contract,
  // every rank's per-peer windowOffset equals its own localOffset, so
  // we fill allWindowOffsets[] uniformly and the prep helpers index
  // into it the same way they do for the multi-window paths.
  //
  // doTranspose path: effWindow is the internal transpose-buffer
  // window whose base IS the transpose buffer, so the effective offset
  // is always 0 regardless of the user's localOffset.
  //
  // For the RING path, all LSA fan-out uses the global user window.
  // ------------------------------------------------------------------
  const size_t kernelOffset = doTranspose ? 0 : (size_t)localOffset;
  std::vector<size_t> allWindowOffsets(worldSize, kernelOffset);
  if (algo == RESHARD_ALGO_DIRECT) {
    ncclXferDirectReshardParams directParams =
      prepareDirectReshardParams(worldRank, effSrcDims, effDstDims, ndims, &effSrcMesh, &effDstMesh, effWindow,
                                 elementsPerChunk, numCtas, allWindowOffsets.data());
    directParams.myWindowOffset = kernelOffset;

    directReshardKernelUserWindow<<<numCtas, threadsPerCta, 0, workStream>>>(directParams, *devCommPtr);
  } else {
    ncclXferReshardParams ringParams = prepareReshardParams(
      worldRank, effSrcBuffer, effSrcDims, ndims, &effSrcMesh, effDstBuffer, effDstDims, &effDstMesh, effWindow,
      elementsPerChunk, numCtas, srcGpusPerDomain, dstGpusPerDomain, allWindowOffsets.data());

    ringParams.myWindowOffset = kernelOffset;
    ringParams.ringNextWindowOffset = kernelOffset;
    for (int f = 0; f < ringParams.numLocalFollowers; f++) ringParams.localFollowerWindowOffsets[f] = kernelOffset;

    reshardKernelUserWindow<<<numCtas, threadsPerCta, 0, workStream>>>(ringParams, *devCommPtr);
  }

  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    fprintf(stderr,
            "[nccl-reshard][Rank %d] kernel launch failed: %s "
            "[algo=%s, numCtas=%d]\n",
            worldRank, cudaGetErrorString(launchErr), algo == RESHARD_ALGO_RING ? "RING" : "DIRECT", numCtas);
    return ncclSystemError;
  }

  // ------------------------------------------------------------------
  // Transpose UNPACK (dest side): reverse-transpose from the transpose
  // buffer back into the user's dstBuffer.
  // ------------------------------------------------------------------
  if (doTranspose && isDest && dstBuffer != nullptr) {
    if (ndims == 2) {
      uwLaunchTranspose2DInner(getTransposeBuffer(comm), dstBuffer, 1, effDstDims[0], effDstDims[1], workStream);
    } else {
      uwLaunchTranspose2DInner(getTransposeBuffer(comm), dstBuffer, effDstDims[0], effDstDims[swapA], effDstDims[swapB],
                               workStream);
    }
  }

  if (doTranspose) UW_NCCLCHECK(transposeBufferRecordEvent(comm, workStream));

  // Pool back-edge: caller's default stream waits for our internal
  // stream so subsequent default-stream work sees the result.  The
  // event is pool-owned and reused across calls.  Skipped when the
  // pool was full and we fell through to the caller's stream.
  if (acquiredPoolSlot) {
    UW_CUDACHECK(cudaEventRecord(poolEvent, workStream));
    UW_CUDACHECK(cudaStreamWaitEvent(stream, poolEvent, 0));
  }

  return ncclSuccess;
}
