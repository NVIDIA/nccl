/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_
#define _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_

#include "../reduce_copy.h"

#if NCCL_CHECK_CUDACC
#if defined(__CUDACC_EXTENDED_LAMBDA__)

#include "reduce_copy__impl.h"

// ============================================================================
// UNROLL Parameter Documentation
// ============================================================================
//
// All public API functions in this file accept an UNROLL template parameter
// that specifies the number of ELEMENTS to unroll (not packs).
//
// Default: UNROLL = 4*16/sizeof(T) (matches all_reduce.cuh UnrollPacks=4)
//   - For float/int32 (4 bytes): 16 elements
//   - For half/int16 (2 bytes): 32 elements
//   - For int8 (1 byte): 64 elements
// Note: UNROLL_PACKS is capped at 4 to match all_reduce.cuh behavior
//
// Note: Internally, this is converted to UNROLL_PACKS based on the vectorization
// strategy (Pack type). The distinction is:
//   - UNROLL (user-facing): Number of elements to process per iteration
//   - UNROLL_PACKS (internal): Number of vector packs to process per iteration
//
// ============================================================================

// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    RedOp redOp,
    IntCount count) {
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  nccl::utility::reduceCopy<T, RedOp, Coop, false, false, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, redOp, count, alignment.alignOffset, alignment.maxPackBytes);
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    RedOp redOp,
    IntCount count) {
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  nccl::utility::reduceCopy<T, RedOp, Coop, false, true, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, redOp, count, alignment.alignOffset, alignment.maxPackBytes);
}

// SERIES 2.x - Sum-Specific ReduceCopy

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  ncclLsaReduceLsaCopy<T, Coop, SrcLambda, DstLambda, nccl::utility::OpSum<T>, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count);
}

// [ID 2.2] LSA <-> Multimem ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  ncclLsaReduceMultimemCopy<T, Coop, SrcLambda, DstLambda, nccl::utility::OpSum<T>, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count);
}

// [ID 2.3] Multimem <-> LSA ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  // Multimem source only supports Sum
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, false, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignment.alignOffset, alignment.maxPackBytes);
}

// [ID 2.4] Multimem <-> Multimem ReduceSum
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  // Compute alignment for opaque lambdas with fallback to smaller pack sizes
  auto alignment = nccl::utility::computeLambdaAlignmentOffsetWithFallback<T>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);

  // Both multimem - only supports Sum
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, true, SrcLambda, DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignment.alignOffset, alignment.maxPackBytes);
}

// ============================================================================
// SERIES 3.x - ReduceSum (N->1)
// ============================================================================

// [ID 3.1] LSA ReduceSum: N sources -> 1 local destination (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                            SrcLambda srcLambda, int nSrc,
                                            T* dstPtr,
                                            IntCount count) {
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nDst = 1;  // Reduce has single destination
  ncclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.2a] LSA ReduceSum: N sources from team -> 1 local destination (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                            ncclSymPtr<T> src,
                                            T* dstPtr,
                                            IntCount count,
                                            ncclTeam team) {
  // Create lambda: N sources from team
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(team, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  // LSA translation ensures all peers have the same alignment
  // Only need to check the first pointer (much cheaper than checking all N)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcLambda(0), dstLambda(0), count, alignOffset, maxPackBytes);
  }

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, team.nRanks, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 3.2b] LSA ReduceSum: N sources from devComm -> 1 local destination (with ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                            ncclSymPtr<T> src,
                                            T* dstPtr,
                                            IntCount count,
                                            ncclDevComm_t devComm) {
  // Extract team from devComm
  ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2c] LSA ReduceSum: N sources from window+offset -> 1 local destination (with ncclWindow_t + ncclTeam)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                            ncclWindow_t window,
                                            size_t offset,
                                            T* dstPtr,
                                            IntCount count,
                                            ncclTeam team) {
  // Construct ncclSymPtr from window and offset using direct initialization
  ncclSymPtr<T> src{window, offset};

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, team);
}

// [ID 3.2d] LSA ReduceSum: N sources from window+offset -> 1 local destination (with ncclWindow_t + ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop coop,
                                            ncclWindow_t window,
                                            size_t offset,
                                            T* dstPtr,
                                            IntCount count,
                                            ncclDevComm_t devComm) {
  // Construct ncclSymPtr from window and offset using direct initialization
  ncclSymPtr<T> src{window, offset};

  ncclLsaReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, devComm);
}

// [ID 3.3a] Multimem ReduceSum: 1 multimem source -> 1 local destination (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop,
                                                 ncclSymPtr<T> src,
                                                 T* dstPtr,
                                                 IntCount count,
                                                 ncclMultimemHandle multimemHandle) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(multimemHandle);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  // Use the lambda-based function - alignment check is cheap for nSrc=1, nDst=1
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3b] Multimem ReduceSum: 1 multimem source -> 1 local destination (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop,
                                                 T* mcSrcPtr,
                                                 T* dstPtr,
                                                 IntCount count) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  // Use the lambda-based function - alignment check is cheap for nSrc=1, nDst=1
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumLsaCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.3c] Multimem ReduceSum: 1 multimem source -> 1 local destination (with ncclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop coop,
                                                 ncclWindow_t window,
                                                 size_t offset,
                                                 T* dstPtr,
                                                 IntCount count,
                                                 ncclMultimemHandle multimemHandle) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> src{window, offset};

  ncclMultimemReduceSum<T, Coop, IntCount, UNROLL>(coop, src, dstPtr, count, multimemHandle);
}

// [ID 3.4] Local ReduceSum: N local chunks -> 1 local destination (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop coop,
                                              SrcLambda srcLambda, int nSrc,
                                              T* dstPtr,
                                              IntCount count) {
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };
  constexpr int nDst = 1;  // Reduce has single destination
  ncclLsaReduceSumLsaCopy<T, Coop, SrcLambda, decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 3.5] Local ReduceSum: reduce n chunks separated by displacement
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop coop,
                                              int nSrc,
                                              T* basePtr,
                                              size_t displ,
                                              T* dstPtr,
                                              IntCount count) {
  // Fast alignment computation for strided addressing with fallback
  IntCount alignOffset;
  int maxPackBytes;
  nccl::utility::computeStridedAlignmentWithFallback<T>(
      basePtr, displ * sizeof(T), count, alignOffset, maxPackBytes);

  // Create lambda: n local sources separated by displ
  auto srcLambda = [=] __device__ (int i) -> T* {
    return basePtr + i * displ;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dstPtr;
  };

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// ============================================================================
// SERIES 4.x - Copy (Broadcast) (1->N)
// ============================================================================

// [ID 4.1] Lambda-based version
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       DstLambda dstLambda, int nDst,
                                       IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  constexpr int nSrc = 1;  // Copy has single source
  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.2a] LSA Copy: 1 local source -> N destinations (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       ncclSymPtr<T> dst,
                                       IntCount count,
                                       ncclTeam team) {
  // Create lambda: N destinations to team
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(team, i);
  };

  // LSA translation ensures all peers have the same alignment
  // Only need to check the first pointer (much cheaper than checking all N)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcLambda(0), dstLambda(0), count, alignOffset, maxPackBytes);
  }

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, team.nRanks, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 4.2b] LSA Copy: 1 local source -> N destinations (with ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       ncclSymPtr<T> dst,
                                       IntCount count,
                                       ncclDevComm_t devComm) {
  // Extract team from devComm
  ncclTeam team = ncclTeamLsa(devComm);

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2c] LSA Copy: 1 local source -> N destinations (with ncclWindow_t + ncclTeam)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       ncclWindow_t window,
                                       size_t offset,
                                       IntCount count,
                                       ncclTeam team) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> dst{window, offset};

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, team);
}

// [ID 4.2d] LSA Copy: 1 local source -> N destinations (with ncclWindow_t + ncclDevComm_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop coop,
                                       T* srcPtr,
                                       ncclWindow_t window,
                                       size_t offset,
                                       IntCount count,
                                       ncclDevComm_t devComm) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> dst{window, offset};

  ncclLsaCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, devComm);
}

// [ID 4.3a] Multimem Copy: 1 local source -> 1 multimem destination (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop,
                                            T* srcPtr,
                                            ncclSymPtr<T> dst,
                                            IntCount count,
                                            ncclMultimemHandle multimemHandle) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(multimemHandle);
  };

  // Use the lambda-based function - alignment check is cheap for nSrc=1, nDst=1
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3b] Multimem Copy: 1 local source -> 1 multimem destination (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop,
                                            T* srcPtr,
                                            T* mcDstPtr,
                                            IntCount count) {
  // Create lambdas for 1 source and 1 destination
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };

  // Use the lambda-based function - alignment check is cheap for nSrc=1, nDst=1
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclLsaReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.3c] Multimem Copy: 1 local source -> 1 multimem destination (with ncclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop coop,
                                            T* srcPtr,
                                            ncclWindow_t window,
                                            size_t offset,
                                            IntCount count,
                                            ncclMultimemHandle multimemHandle) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> dst{window, offset};

  ncclMultimemCopy<T, Coop, IntCount, UNROLL>(coop, srcPtr, dst, count, multimemHandle);
}

// [ID 4.4] Local Copy: 1 source -> N local destinations (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop coop,
                                         T* srcPtr,
                                         DstLambda dstLambda, int nDst,
                                         IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  constexpr int nSrc = 1;  // Copy has single source
  ncclLsaReduceSumLsaCopy<T, Coop, decltype(srcLambda), DstLambda, IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 4.5] Local Copy: copy to n chunks separated by displacement
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop coop,
                                         T* srcPtr,
                                         int nDst,
                                         T* basePtr,
                                         size_t displ,
                                         IntCount count) {
  // Fast alignment computation for strided addressing with fallback
  IntCount alignOffset;
  int maxPackBytes;
  nccl::utility::computeStridedAlignmentWithFallback<T>(
      basePtr, displ * sizeof(T), count, alignOffset, maxPackBytes);

  // Create lambda: n local destinations separated by displ
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return srcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return basePtr + i * displ;
  };

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// ============================================================================
// SERIES 5.x - ReduceSumCopy (N->M)
// ============================================================================

// [ID 5.1a] LSA ReduceSumCopy: same team for src and dst (most common case)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                                ncclSymPtr<T> src,
                                                ncclSymPtr<T> dst,
                                                IntCount count,
                                                ncclTeam team) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(team, i);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(team, i);
  };

  // LSA translation ensures all peers have the same alignment
  // Only need to check the first src and dst pointers (much cheaper than checking all N)
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && team.nRanks > 0) {
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcLambda(0), dstLambda(0), count, alignOffset, maxPackBytes);
  }

  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, team.nRanks, dstLambda, team.nRanks, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.1b] LSA ReduceSumCopy: with ncclDevComm_t (extract team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                                ncclSymPtr<T> src,
                                                ncclSymPtr<T> dst,
                                                IntCount count,
                                                ncclDevComm_t devComm) {
  ncclTeam team = ncclTeamLsa(devComm);
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1c] LSA ReduceSumCopy: with windows + ncclTeam
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                                ncclWindow_t srcWindow, size_t srcOffset,
                                                ncclWindow_t dstWindow, size_t dstOffset,
                                                IntCount count,
                                                ncclTeam team) {
  // Construct ncclSymPtr from window and offset using direct initialization
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, team);
}

// [ID 5.1d] LSA ReduceSumCopy: with windows + ncclDevComm_t
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                                ncclWindow_t srcWindow, size_t srcOffset,
                                                ncclWindow_t dstWindow, size_t dstOffset,
                                                IntCount count,
                                                ncclDevComm_t devComm) {
  // Construct ncclSymPtr from window and offset using direct initialization
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};
  ncclLsaReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, dst, count, devComm);
}

// [ID 5.1e] LSA ReduceSumCopy: different teams for src and dst (advanced use case)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop coop,
                                                ncclSymPtr<T> src, ncclTeam srcTeam,
                                                ncclSymPtr<T> dst, ncclTeam dstTeam,
                                                IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  // LSA translation ensures all peers have the same alignment within each team
  // Always check both src and dst pointers together for relative alignment when both are available
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, srcTeam.nRanks, dstLambda, dstTeam.nRanks, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.2a] Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop,
                                                     ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
                                                     ncclSymPtr<T> dst, ncclMultimemHandle dstHandle,
                                                     IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(srcHandle);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(dstHandle);
  };

  // Use the lambda-based function - alignment check is cheap for nSrc=1, nDst=1
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop,
                                                     T* mcSrcPtr,
                                                     T* mcDstPtr,
                                                     IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };

  // Use the lambda-based function - alignment check is cheap for nSrc=1, nDst=1
  constexpr int nSrc = 1;
  constexpr int nDst = 1;
  ncclMultimemReduceSumMultimemCopy<T, Coop, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, count);
}

// [ID 5.2c] Multimem ReduceSumCopy (with ncclWindow_t)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop coop,
                                                     ncclWindow_t srcWindow, size_t srcOffset, ncclMultimemHandle srcHandle,
                                                     ncclWindow_t dstWindow, size_t dstOffset, ncclMultimemHandle dstHandle,
                                                     IntCount count) {
  // Construct ncclSymPtr from window and offset
  ncclSymPtr<T> src{srcWindow, srcOffset};
  ncclSymPtr<T> dst{dstWindow, dstOffset};

  ncclMultimemReduceSumCopy<T, Coop, IntCount, UNROLL>(coop, src, srcHandle, dst, dstHandle, count);
}

// [ID 5.3a] LSA source -> Multimem destination
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop,
                                                        ncclSymPtr<T> src, ncclTeam srcTeam,
                                                        ncclSymPtr<T> dst, ncclMultimemHandle dstHandle,
                                                        IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return dst.multimemPtr(dstHandle);
  };

  // LSA and Multimem translation ensure consistent alignment
  // Always check both src and dst pointers together for relative alignment
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, true, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, srcTeam.nRanks, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3b] LSA source -> Multimem destination (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop coop,
                                                        ncclSymPtr<T> src, ncclTeam srcTeam,
                                                        T* mcDstPtr,
                                                        IntCount count) {
  auto srcLambda = [=] __device__ (int i) -> T* {
    return src.peerPtr(srcTeam, i);
  };
  auto dstLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcDstPtr;
  };

  // LSA and Multimem translation ensure consistent alignment
  // Always check both src and dst pointers together for relative alignment
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && srcTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nDst = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, true, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, srcTeam.nRanks, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3c] Multimem source -> LSA destination
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop,
                                                        ncclSymPtr<T> src, ncclMultimemHandle srcHandle,
                                                        ncclSymPtr<T> dst, ncclTeam dstTeam,
                                                        IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return src.multimemPtr(srcHandle);
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  // Multimem and LSA translation ensure consistent alignment
  // Always check both src and dst pointers together for relative alignment
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.3d] Multimem source -> LSA destination (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop coop,
                                                        T* mcSrcPtr,
                                                        ncclSymPtr<T> dst, ncclTeam dstTeam,
                                                        IntCount count) {
  auto srcLambda = [=] __device__ (int /*ignored*/) -> T* {
    return mcSrcPtr;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dst.peerPtr(dstTeam, i);
  };

  // Multimem and LSA translation ensure consistent alignment
  // Always check both src and dst pointers together for relative alignment
  IntCount alignOffset = 0;
  int maxPackBytes = 16;
  if (count > 0 && dstTeam.nRanks > 0) {
    void* srcPtr = srcLambda(0);
    void* dstPtr = dstLambda(0);
    nccl::utility::computePointerPairAlignmentWithFallback<T>(
        srcPtr, dstPtr, count, alignOffset, maxPackBytes);
  }

  constexpr int nSrc = 1;
  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, true, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, dstTeam.nRanks, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}

// [ID 5.4] Local ReduceSumCopy: N local sources -> M local destinations
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(Coop coop,
                                                  int nSrc, T* srcBasePtr, size_t srcDispl,
                                                  int nDst, T* dstBasePtr, size_t dstDispl,
                                                  IntCount count) {
  // Fast alignment computation for strided addressing with fallback
  IntCount alignOffset = count;
  int maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // Try each pack size from largest to smallest using explicit template instantiations
  if (nccl::utility::tryComplexStridedAlignmentForPackSize<T, 16>(
          srcBasePtr, srcDispl * sizeof(T), dstBasePtr, dstDispl * sizeof(T), alignOffset, maxPackBytes)) {
    // Found working pack size
  } else if (nccl::utility::tryComplexStridedAlignmentForPackSize<T, 4>(
          srcBasePtr, srcDispl * sizeof(T), dstBasePtr, dstDispl * sizeof(T), alignOffset, maxPackBytes)) {
    // Found working pack size
  }

  auto srcLambda = [=] __device__ (int i) -> T* {
    return srcBasePtr + i * srcDispl;
  };
  auto dstLambda = [=] __device__ (int i) -> T* {
    return dstBasePtr + i * dstDispl;
  };

  nccl::utility::reduceCopy<T, nccl::utility::OpSum<T>, Coop, false, false, decltype(srcLambda), decltype(dstLambda), IntCount, UNROLL>(
      coop, srcLambda, nSrc, dstLambda, nDst, nccl::utility::OpSum<T>{}, count, alignOffset, maxPackBytes);
}
#else // __CUDACC_EXTENDED_LAMBDA__

// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(
    Coop, SrcLambda, int, DstLambda, int, RedOp, IntCount) {
  // C++11 - C++17 considers this a template invalid cannot have a valid specialation.
  // By making this a dependent static_assert, there may exists an overload of always_false that is true, it is valid.
  // "The validity of a template checked prior to any instantiation."
  // C++20+ may make an exception for static_assert for this exact situation. https://eel.is/c++draft/temp.res#general-6
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename RedOp, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(
    Coop, SrcLambda, int, DstLambda, int, RedOp, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 2.x - Sum-Specific ReduceCopy (lambda-based foundation)
template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template <typename T, typename Coop, typename SrcLambda, typename DstLambda,
          typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(
    Coop, SrcLambda, int, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 3.x - ReduceSum (N->1)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop, SrcLambda, int, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop, ncclSymPtr<T>, T*, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop, ncclSymPtr<T>, T*, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop, ncclWindow_t, size_t, T*, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(
    Coop, ncclWindow_t, size_t, T*, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(
    Coop, ncclSymPtr<T>, T*, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 3.3b] Multimem ReduceSum (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(
    Coop, T*, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(
    Coop, ncclWindow_t, size_t, T*, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 3.4] Local ReduceSum (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(
    Coop, SrcLambda, int, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 3.5] Local ReduceSum (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(
    Coop, int, T*, size_t, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 4.x - Copy/Broadcast (1->N)

// 4.1] LSA Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop, T*, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2a] LSA Copy (with ncclSymPtr + team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop, T*, ncclSymPtr<T>, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2b] LSA Copy (with ncclSymPtr + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop, T*, ncclSymPtr<T>, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2c] LSA Copy (with window + offset + team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop, T*, ncclWindow_t, size_t, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.2d] LSA Copy (with window + offset + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaCopy(
    Coop, T*, ncclWindow_t, size_t, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.3a] Multimem Copy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(
    Coop, T*, ncclSymPtr<T>, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.3b] Multimem Copy (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(
    Coop, T*, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.3c] Multimem Copy (with window + offset)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemCopy(
    Coop, T*, ncclWindow_t, size_t, IntCount, ncclMultimemHandle) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.4] Local Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(
    Coop, T*, DstLambda, int, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 4.5] Local Copy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalCopy(
    Coop, T*, int, T*, size_t, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// SERIES 5.x - ReduceSumCopy (N->M)

// 5.1a] LSA ReduceSumCopy (same team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1b] LSA ReduceSumCopy (with devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1c] LSA ReduceSumCopy (with windows + team)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclTeam) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1d] LSA ReduceSumCopy (with windows + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclDevComm_t) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.1e] LSA ReduceSumCopy (different teams)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(
    Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclTeam, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.2a] Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(
    Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclMultimemHandle, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(
    Coop, T*, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.2c] Multimem ReduceSumCopy (with windows)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(
    Coop, ncclWindow_t, size_t, ncclMultimemHandle, ncclWindow_t, size_t, ncclMultimemHandle, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3a] LSA -> Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclMultimemHandle, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(
    Coop, ncclSymPtr<T>, ncclTeam, T*, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3c] Multimem -> LSA ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclTeam, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(
    Coop, T*, ncclSymPtr<T>, ncclTeam, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

// 5.4] Local ReduceSumCopy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(
    Coop, int, T*, size_t, int, T*, size_t, IntCount) {
  static_assert(nccl::utility::always_false<T>::value,
     "NCCL device API reduce/Copy functions require device side lambdas, please use '--extended-lambda' as compilation flag to enable that API.");
}

#endif // __CUDACC_EXTENDED_LAMBDA__
#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_REDUCE_COPY__FUNCS_H_
