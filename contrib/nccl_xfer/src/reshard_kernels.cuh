/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Shared Device Kernel Helpers
 *
 * Header-only device helpers used by reshardKernelUserWindow (RING)
 * and directReshardKernelUserWindow (DIRECT) in reshard_user_window.cu.
 *
 * All helpers are `__device__ static inline` so each translation unit
 * gets its own copy without ODR violations.
 ************************************************************************/

#ifndef NCCLXFER_RESHARD_KERNELS_CUH
#define NCCLXFER_RESHARD_KERNELS_CUH

#include "reshard_types.h"

#include "nccl.h"
#include "nccl_device.h"

static_assert(sizeof(uint4) == 16, "uint4 must be 16 bytes for the aligned LSA replication path");
static_assert(sizeof(uint32_t) == 4, "uint32_t must be 4 bytes for the aligned LSA replication path");

// ============================================================================
// Compute strided source / dest byte offsets for a single transfer iteration.
// ============================================================================
__device__ static inline void computeTransferOffset(const ncclXferTransferPlan& plan, size_t iterIdx, int ndims,
                                                    size_t* srcOffset, size_t* dstOffset) {
  *srcOffset = plan.srcBaseOffset;
  *dstOffset = plan.dstBaseOffset;

  size_t remaining = iterIdx;
  for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
    size_t idx = remaining % plan.outerCounts[d];
    remaining /= plan.outerCounts[d];
    *srcOffset += idx * plan.outerSrcStrides[d];
    *dstOffset += idx * plan.outerDstStrides[d];
  }
}

// ============================================================================
// Emit gin.put() calls for a contiguous byte range within a strided plan.
// Maps the byte range to the strided source/dest offsets, emitting one
// gin.put per (partial or full) inner transfer.  Handles partial inner
// transfers at both ends of the byte range.
// ============================================================================
__device__ static inline void emitStridedChunkPuts(ncclGin& gin, ncclTeam world, int dstWorldRank, ncclWindow_t window,
                                                   const ncclXferTransferPlan& plan, int ndims, size_t ctaIterStart,
                                                   size_t byteStart, size_t numBytes, unsigned int signalIdx,
                                                   bool useDstAsSrc, size_t srcWindowOffset, size_t dstWindowOffset) {
  const size_t inner = plan.innerSize;
  size_t firstIter = byteStart / inner;
  size_t firstOffset = byteStart % inner;

  size_t bytesRemaining = numBytes;
  size_t iter = firstIter;
  size_t offsetInIter = firstOffset;

  while (bytesRemaining > 0) {
    size_t avail = inner - offsetInIter;
    size_t thisBytes = (avail < bytesRemaining) ? avail : bytesRemaining;

    size_t srcOff, dstOff;
    computeTransferOffset(plan, ctaIterStart + iter, ndims, &srcOff, &dstOff);

    size_t readOff = useDstAsSrc ? dstOff : srcOff;

    bool isLast = (bytesRemaining == thisBytes);

    if (isLast) {
      gin.put(world, dstWorldRank, window, dstWindowOffset + dstOff + offsetInIter, window,
              srcWindowOffset + readOff + offsetInIter, thisBytes, ncclGin_SignalInc{signalIdx});
    } else {
      gin.put(world, dstWorldRank, window, dstWindowOffset + dstOff + offsetInIter, window,
              srcWindowOffset + readOff + offsetInIter, thisBytes, ncclGin_None{});
    }

    bytesRemaining -= thisBytes;
    iter++;
    offsetInIter = 0;
  }
}

// ============================================================================
// LSA peer-pointer resolution for the user-window path.
//
// Peer pointers come from the global window keyed by
// `(followerWorldRank - lsaStartRank)` — the LSA-rank of that follower
// within the input comm's LSA team.
//
// TODO: If a future path registers a separate LSA-only window, reintroduce an
// explicit host-side producer for that window and the node-local follower rank
// map before adding a second pointer-resolution branch here.
// ============================================================================
__device__ static inline char* lsaResolveFollowerPtr(ncclWindow_t fallbackWindow, size_t fOff, int followerWorldRank,
                                                     int lsaStartRank) {
  return (char*)ncclGetLsaPointer(fallbackWindow, fOff, followerWorldRank - lsaStartRank);
}

// ============================================================================
// Templated vector-width LSA replication step.
//
// Loads `chunkBytes / sizeof(T)` elements of T from srcPtr (cast) and
// stores each to every follower's LSA mapping.  T may be uint4 (16 B),
// uint32_t (4 B), or char (1 B for the byte path / unaligned tail).
// ============================================================================
template <typename T>
__device__ static inline void lsaReplicateChunkVec(const char* srcPtr, size_t chunkBytes, size_t dstByteOffset,
                                                   ncclWindow_t fallbackWindow, const int* followerWorldRanks,
                                                   const size_t* followerWindowOffsets, int lsaStartRank,
                                                   int numFollowers, int threadsInGroup, int threadInGroup) {
  constexpr size_t W = sizeof(T);
  const size_t n = chunkBytes / W;
  const T* srcT = (const T*)srcPtr;
  for (size_t p = threadInGroup; p < n; p += threadsInGroup) {
    T data = srcT[p];
    size_t off = dstByteOffset + p * W;
    for (int f = 0; f < numFollowers; f++) {
      char* dst =
        lsaResolveFollowerPtr(fallbackWindow, followerWindowOffsets[f] + off, followerWorldRanks[f], lsaStartRank);
      *(T*)dst = data;
    }
  }
}

// ============================================================================
// Alignment-aware LSA replication.
//
// Replicates a contiguous chunk from the leader's local buffer to all
// followers via LSA stores.  Picks the widest safe vector width at runtime
// (uint4 / uint32_t / byte) based on pointer alignment, matching the
// staging_memcpy pattern from staging_primitives.cuh.  The byte-sized tail
// (when a vector path leaves remainder bytes) is handled by reusing the
// templated loop with T=char.
// ============================================================================
__device__ static inline void lsaReplicateChunk(const char* srcPtr, size_t chunkBytes, size_t dstByteOffset,
                                                ncclWindow_t fallbackWindow, const int* followerWorldRanks,
                                                const size_t* followerWindowOffsets, int lsaStartRank, int numFollowers,
                                                int threadsInGroup, int threadInGroup) {
  auto replicateTail = [&](const char* p, size_t bytes, size_t off) {
    if (bytes == 0) return;
    lsaReplicateChunkVec<char>(p, bytes, off, fallbackWindow, followerWorldRanks, followerWindowOffsets, lsaStartRank,
                               numFollowers, threadsInGroup, threadInGroup);
  };

  uintptr_t baseAlign = (uintptr_t)srcPtr | (uintptr_t)dstByteOffset;

  if ((baseAlign & 0xF) == 0) {
    size_t aligned = (chunkBytes / 16) * 16;
    lsaReplicateChunkVec<uint4>(srcPtr, aligned, dstByteOffset, fallbackWindow, followerWorldRanks,
                                followerWindowOffsets, lsaStartRank, numFollowers, threadsInGroup, threadInGroup);
    replicateTail(srcPtr + aligned, chunkBytes - aligned, dstByteOffset + aligned);
  } else if ((baseAlign & 0x3) == 0) {
    size_t aligned = (chunkBytes / 4) * 4;
    lsaReplicateChunkVec<uint32_t>(srcPtr, aligned, dstByteOffset, fallbackWindow, followerWorldRanks,
                                   followerWindowOffsets, lsaStartRank, numFollowers, threadsInGroup, threadInGroup);
    replicateTail(srcPtr + aligned, chunkBytes - aligned, dstByteOffset + aligned);
  } else {
    replicateTail(srcPtr, chunkBytes, dstByteOffset);
  }
}

#endif // NCCLXFER_RESHARD_KERNELS_CUH
