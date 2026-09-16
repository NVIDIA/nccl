/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_REDUCE_COPY__IMPL_H_
#define _NCCL_DEVICE_REDUCE_COPY__IMPL_H_

#include "reduce_copy__types.h"
#include "multimem__funcs.h"
#include "vector__types.h"
#include "vector__funcs.h"
#include "../coop.h"
#include <cassert>
#include <type_traits>

#ifndef NCCL_DEVICE_DEBUG_CHECKS
#define NCCL_DEVICE_DEBUG_CHECKS 0
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
// including <cuda/barrier>/<cuda/ptx> can fail unless <cuda/std/ranges> is included. workaround introduced in src/device/symmetric/primitives.cuh
#include <cuda/std/ranges>
#endif

#if __CUDA_ARCH__ >= 1000
// These CCCL headers are needed for the TMA load/store API (sm100+)
#include <cuda/barrier>
#include <cuda/ptx>
#include <utility>

namespace ptx = cuda::ptx;
#endif

#if defined(__CUDACC__) && defined(__CUDACC_EXTENDED_LAMBDA__)

namespace nccl {
namespace utility {

// Core Loop Implementation

template <int UNROLL_PACKS, int UNROLL_SOURCE, typename T, typename Pack, typename RedOp, typename IntCount,
          typename Coop, bool srcMultimem, bool dstMultimem, typename SrcLambda, typename DstLambda, bool CHECK_BOUNDS,
          bool SINGLE_SRC>
NCCL_DEVICE_INLINE IntCount reduceCopyLoopCoreImpl(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda,
                                                   int nDst, RedOp const& redOp, IntCount totalPacks,
                                                   IntCount basePackIdx) {
  static_assert(!SINGLE_SRC || UNROLL_SOURCE == 1, "UNROLL_SOURCE must be 1 when SINGLE_SRC is set");

  constexpr int warpSize = 32;
  constexpr int coopStride = CoopStride<Coop>::value;
  constexpr int stride = (coopStride != 0) ? coopStride : warpSize;
  const int threadRank = coop.thread_rank();
  const int coopSize = coop.size();
  const int runtimeStride = (coopStride != 0) ? coopStride : min(coopSize, warpSize);
  const int laneId = threadRank % runtimeStride;
  const int groupId = threadRank / runtimeStride;

  IntCount groupBasePackIdx = basePackIdx + groupId * (stride * UNROLL_PACKS);
  IntCount groupLanePackIdx = groupBasePackIdx + laneId;

  using PackEltType = typename Pack::EltType;
  using BaseAccEltType = typename AccumulateType<RedOp>::Type;
  using AccEltType = std::conditional_t<SINGLE_SRC, PackEltType, BaseAccEltType>;
  using AccPackType = EltPack<AccEltType, Pack::Count>;

  using AccRedOpType = typename AccRedOp<RedOp, AccEltType>::Type;

  AccPackType acc[UNROLL_PACKS];

  // Reduce phase
  if NCCL_IF_CONSTEXPR (SINGLE_SRC) {
    Pack* srcPtr0 = (Pack*)srcLambda(0);
    NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
    for (int u = 0; u < UNROLL_PACKS; u++) {
      IntCount packIdx = groupLanePackIdx + u * runtimeStride;
      if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
        if (packIdx >= totalPacks) break;
      }

      Pack loaded = load<Pack, srcMultimem, RedOp>(srcPtr0 + packIdx);
      acc[u] = castPack<AccEltType, PackEltType, Pack::Count>(loaded);
    }
  } else {
    AccRedOpType accRedOp{};
    Pack loaded[UNROLL_SOURCE][UNROLL_PACKS];

    // Preseed acc[] with source 0 to avoid inner-loop branching.
    Pack* srcPtr = (Pack*)srcLambda(0);
    NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
    for (int u = 0; u < UNROLL_PACKS; u++) {
      IntCount packIdx = groupLanePackIdx + u * runtimeStride;
      if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
        if (packIdx >= totalPacks) break;
      }
      loaded[0][u] = load<Pack, srcMultimem, RedOp>(srcPtr + packIdx);
      AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[0][u]);
      acc[u] = val;
    }

    constexpr int srcCount = UNROLL_SOURCE;
    NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
    for (int srcOffset = 1; srcOffset < srcCount; srcOffset++) {
      Pack* srcPtr = (Pack*)srcLambda(srcOffset);
      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
          if (packIdx >= totalPacks) break;
        }
        loaded[srcOffset][u] = load<Pack, srcMultimem, RedOp>(srcPtr + packIdx);
      }
    }

    NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
    for (int u = 0; u < UNROLL_PACKS; u++) {
      if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
        IntCount packIdx = groupLanePackIdx + u * runtimeStride;
        if (packIdx >= totalPacks) break;
      }
      NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
      for (int srcOffset = 1; srcOffset < srcCount; srcOffset++) {
        AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[srcOffset][u]);
        acc[u] = reducePack(accRedOp, acc[u], val);
      }
    }

    // Finish remaining sources.
    for (int srcBase = UNROLL_SOURCE; srcBase < nSrc; srcBase += UNROLL_SOURCE) {
      Pack loaded[UNROLL_SOURCE][UNROLL_PACKS];
      NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
      for (int srcOffset = 0; srcOffset < srcCount; srcOffset++) {
        Pack* srcPtr = (Pack*)srcLambda(srcBase + srcOffset);
        NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
        for (int u = 0; u < UNROLL_PACKS; u++) {
          IntCount packIdx = groupLanePackIdx + u * runtimeStride;
          if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
            if (packIdx >= totalPacks) break;
          }
          loaded[srcOffset][u] = load<Pack, srcMultimem, RedOp>(srcPtr + packIdx);
        }
      }

      NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
      for (int u = 0; u < UNROLL_PACKS; u++) {
        if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
          IntCount packIdx = groupLanePackIdx + u * runtimeStride;
          if (packIdx >= totalPacks) break;
        }
        NVCC_PRAGMA_UNROLL(UNROLL_SOURCE)
        for (int srcOffset = 0; srcOffset < srcCount; srcOffset++) {
          AccPackType val = castPack<AccEltType, PackEltType, Pack::Count>(loaded[srcOffset][u]);
          acc[u] = reducePack(accRedOp, acc[u], val);
        }
      }
    }
  }

  // Broadcast phase
  NVCC_PRAGMA_UNROLL(4)
  for (int dstIdx = 0; dstIdx < nDst; dstIdx++) {
    Pack* dstPtr = (Pack*)dstLambda(dstIdx);
    NVCC_PRAGMA_UNROLL(UNROLL_PACKS)
    for (int u = 0; u < UNROLL_PACKS; u++) {
      IntCount packIdx = groupLanePackIdx + u * runtimeStride;
      if NCCL_IF_CONSTEXPR (CHECK_BOUNDS) {
        if (packIdx >= totalPacks) break;
      }

      Pack result = castPack<PackEltType, AccEltType, Pack::Count>(acc[u]);

      store<Pack, dstMultimem>(dstPtr + packIdx, result);
    }
  }
  const int numGroups = (coopSize + runtimeStride - 1) / runtimeStride;
  const IntCount packsPerIteration = numGroups * (runtimeStride * UNROLL_PACKS);
  const IntCount remainingPacks = (basePackIdx < totalPacks) ? (totalPacks - basePackIdx) : 0;
  const IntCount processedPacks = (remainingPacks < packsPerIteration) ? remainingPacks : packsPerIteration;
  return processedPacks * Pack::Count;
}

template <int UNROLL_PACKS, typename T, typename Pack, typename RedOp, typename IntCount, typename Coop,
          bool srcMultimem, bool dstMultimem, typename SrcLambda, typename DstLambda, bool CHECK_BOUNDS>
NCCL_DEVICE_INLINE IntCount reduceCopyLoopCore(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                               RedOp const& redOp, IntCount totalPacks, IntCount basePackIdx) {
  if (nSrc == 1) {
    return reduceCopyLoopCoreImpl<UNROLL_PACKS, /*nSrc=*/1, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                  SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/true>(
      coop, srcLambda, 1, dstLambda, nDst, redOp, totalPacks, basePackIdx);
  } else {
    if (nSrc >= 4 && nSrc % 4 == 0) {
      return reduceCopyLoopCoreImpl<UNROLL_PACKS, /*nSrc=*/4, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                    SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/false>(
        coop, srcLambda, nSrc, dstLambda, nDst, redOp, totalPacks, basePackIdx);
    }
    // NOTE: nSrc % 3 and nSrc % 2 specializations add code to this runtime-dispatched device function. Keep them
    // disabled unless performance data warrants the extra compile/code size cost.
    return reduceCopyLoopCoreImpl<UNROLL_PACKS, /*nSrc=*/1, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                  SrcLambda, DstLambda, CHECK_BOUNDS, /*singleSrc=*/false>(
      coop, srcLambda, nSrc, dstLambda, nDst, redOp, totalPacks, basePackIdx);
  }
}

// Helper struct to calculate loop iteration counts
template <int UNROLL_PACKS, typename Pack, typename IntCount>
struct ReduceCopyLoopParams {
  IntCount totalPacks;
  IntCount packsPerIteration;
  int effectiveUnrollPacks;
  IntCount numFullChunks; // Number of unchecked rounds
  IntCount remainingPacks; // Number of packs in checked round
  IntCount processedElts; // Number of elements processed (full packs only)

  NCCL_DEVICE_INLINE ReduceCopyLoopParams(IntCount count, int coopSize, int stride) {
    if NCCL_IF_CONSTEXPR (Pack::Count > 0) {
      totalPacks = safeDiv<IntCount>(count, Pack::Count);
    } else {
      totalPacks = 0;
    }

    effectiveUnrollPacks = UNROLL_PACKS;

    // Compute packs per iteration: numGroups * (stride * UNROLL_PACKS)
    const int numGroups = (coopSize + stride - 1) / stride;
    packsPerIteration = numGroups * (stride * effectiveUnrollPacks);

    // Calculate number of unchecked and checked rounds
    if NCCL_IF_CONSTEXPR (Pack::Count > 0) {
      numFullChunks = totalPacks / packsPerIteration;
      remainingPacks = totalPacks - numFullChunks * packsPerIteration;
      processedElts = numFullChunks * packsPerIteration * Pack::Count;
    } else {
      numFullChunks = 0;
      remainingPacks = 0;
      processedElts = 0;
    }
  }
};

template <int UNROLL_PACKS, typename T, typename Pack, typename RedOp, typename IntCount, typename Coop,
          bool srcMultimem, bool dstMultimem, typename SrcLambda, typename DstLambda, bool SkipTail>
NCCL_DEVICE_INLINE IntCount reduceCopyLoop(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                           RedOp const& redOp, IntCount count) {
  const int coopSize = coop.size();
  constexpr int warpSize = 32;
  constexpr int defaultStride = CoopStride<Coop>::value;
  const int stride = (defaultStride != 0) ? defaultStride : min(coopSize, warpSize);

  // Calculate loop parameters
  ReduceCopyLoopParams<UNROLL_PACKS, Pack, IntCount> params(count, coopSize, stride);
  if (params.totalPacks == 0) {
    return 0;
  }

  IntCount processedElts = 0;
  IntCount basePackIdx = 0;
  while (basePackIdx + params.packsPerIteration <= params.totalPacks) {
    processedElts +=
      reduceCopyLoopCore<UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem, SrcLambda, DstLambda,
                         false>(coop, srcLambda, nSrc, dstLambda, nDst, redOp, params.totalPacks, basePackIdx);
    basePackIdx += params.packsPerIteration;
  }

  if NCCL_IF_CONSTEXPR (!SkipTail) {
    constexpr int TAIL_UNROLL_PACKS = (UNROLL_PACKS > 2) ? 2 : UNROLL_PACKS;
    const int numGroups = (coopSize + stride - 1) / stride;
    const IntCount tailPacksPerIteration = numGroups * (stride * TAIL_UNROLL_PACKS);
    while (basePackIdx + tailPacksPerIteration <= params.totalPacks) {
      processedElts += reduceCopyLoopCore<TAIL_UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                          SrcLambda, DstLambda, false>(coop, srcLambda, nSrc, dstLambda, nDst, redOp,
                                                                       params.totalPacks, basePackIdx);
      basePackIdx += tailPacksPerIteration;
    }

    if (basePackIdx < params.totalPacks) {
      processedElts += reduceCopyLoopCore<TAIL_UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                          SrcLambda, DstLambda, true>(coop, srcLambda, nSrc, dstLambda, nDst, redOp,
                                                                      params.totalPacks, basePackIdx);
    }
  }
  return processedElts;
}

// Scalar Loop Implementation (for scalar remainder sections)
// Uses reduceCopyLoop with EltPack<T, 1> as the Pack type and UNROLL_PACKS=1
template <typename T, typename RedOp, typename IntCount, typename Coop, bool srcMultimem, bool dstMultimem,
          typename SrcLambda, typename DstLambda>
NCCL_DEVICE_INLINE void reduceCopyScalarLoop(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                             RedOp const& redOp, IntCount count) {
  if (count == 0) return;

  // Default scalar path: one element per pack.
  using Pack = EltPack<T, 1>;
  auto srcScalarLambda = [=] __device__(int i) -> Pack* {
    T* basePtr = srcLambda(i);
    return reinterpret_cast<Pack*>(basePtr);
  };
  auto dstScalarLambda = [=] __device__(int i) -> Pack* {
    T* basePtr = dstLambda(i);
    return reinterpret_cast<Pack*>(basePtr);
  };

  // Use reduceCopyLoop with EltPack<T, 1> as Pack and UNROLL_PACKS=1
  // This handles chunking and bounds checking properly
  constexpr int UNROLL_PACKS = 1;

  reduceCopyLoop<UNROLL_PACKS, T, Pack, RedOp, IntCount, Coop, srcMultimem, dstMultimem, decltype(srcScalarLambda),
                 decltype(dstScalarLambda), /*skipTail=*/false>(coop, srcScalarLambda, nSrc, dstScalarLambda, nDst,
                                                                redOp, count);
}

// Main Entry Point (Internal - Not Public API)

template <typename T, typename RedOp, typename Coop, bool srcMultimem, bool dstMultimem, typename SrcLambda,
          typename DstLambda, typename IntCount, int UNROLL_ELTS>
NCCL_DEVICE_INLINE void reduceCopy(Coop coop, SrcLambda srcLambda, int nSrc, DstLambda dstLambda, int nDst,
                                   RedOp const& redOp, IntCount count, IntCount alignOffset = 0,
                                   int maxPackBytes = 16) {
  // Step 1: Process scalar prefix to achieve alignment (if needed)
  // alignOffset is already computed by the alignment functions - use it directly
  IntCount processedElts = 0;
  if (alignOffset > 0 && alignOffset < count) {
    reduceCopyScalarLoop<T, RedOp, IntCount, Coop, srcMultimem, dstMultimem>(coop, srcLambda, nSrc, dstLambda, nDst,
                                                                             redOp, alignOffset);
    processedElts = alignOffset;
  }

  // Step 2: Process aligned bulk - match all_reduce.cuh strategy: check relative alignment and try pack sizes
  // sequentially
  IntCount remainingElts = count - processedElts;
  if (remainingElts == 0) {
    return;
  }

  // Create lambdas for remaining work
  auto srcRemaining = [=] __device__(int i) -> T* { return srcLambda(i) + processedElts; };
  auto dstRemaining = [=] __device__(int i) -> T* { return dstLambda(i) + processedElts; };

  // Check relative alignment of first source and destination pointers (like all_reduce.cuh)
  // all_reduce.cuh checks: (input.offset - output.offset)%16 == 0
  // This determines which pack sizes we can use
  void* srcPtr0 = (nSrc > 0) ? (void*)srcRemaining(0) : nullptr;
  void* dstPtr0 = (nDst > 0) ? (void*)dstRemaining(0) : nullptr;
  uintptr_t srcOffset = (srcPtr0 != nullptr) ? reinterpret_cast<uintptr_t>(srcPtr0) : 0;
  uintptr_t dstOffset = (dstPtr0 != nullptr) ? reinterpret_cast<uintptr_t>(dstPtr0) : 0;
  // Calculate relative alignment: (srcOffset - dstOffset) mod packSize
  // Note: We need signed difference to match all_reduce.cuh behavior
  intptr_t relOffset16 = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

  IntCount vectorizedElts = 0;
  constexpr int scalarSize = sizeof(T);

  // Step 2a: Try 16-byte packs first if relative alignment is good (matching all_reduce.cuh)
  // all_reduce.cuh checks: (input.offset - output.offset)%16 == 0
  if (maxPackBytes >= 16 && relOffset16 % 16 == 0 && remainingElts * scalarSize >= 16) {
    using Pack16 = nccl::utility::EltPackForBytes<T, 16>;
    if NCCL_IF_CONSTEXPR (Pack16::Count > 0) {
      constexpr int UNROLL_PACKS16_RAW = static_cast<int>(safeDiv(UNROLL_ELTS + Pack16::Count - 1, Pack16::Count));
      constexpr int UNROLL_PACKS16 = (UNROLL_PACKS16_RAW > 0) ? UNROLL_PACKS16_RAW : 1;
      if NCCL_IF_CONSTEXPR (UNROLL_PACKS16_RAW > 0) {
        IntCount vectorizableElts16 = safeDiv<IntCount>(remainingElts, Pack16::Count) * Pack16::Count;
        if (vectorizableElts16 > 0) {
          vectorizedElts += reduceCopyLoop<UNROLL_PACKS16, T, Pack16, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                           decltype(srcRemaining), decltype(dstRemaining), /*skipTail=*/false>(
            coop, srcRemaining, nSrc, dstRemaining, nDst, redOp, vectorizableElts16);
        }
      }
    }
  }

  // Step 2b: Try 4-byte packs on remainder (if 16-byte worked) or all remaining (if 16-byte didn't work)
  // all_reduce.cuh checks: sizeof(T) == 4 || (sizeof(T) < 4 && (input.offset - output.offset)%4 == 0)
  IntCount remainingAfter16 = remainingElts - vectorizedElts;
  if (maxPackBytes >= 4 && remainingAfter16 > 0) {
    // Recalculate alignment for Pack4 after Pack16 processing
    void* srcPtrAfter16 = (nSrc > 0) ? (void*)(srcRemaining(0) + vectorizedElts) : nullptr;
    void* dstPtrAfter16 = (nDst > 0) ? (void*)(dstRemaining(0) + vectorizedElts) : nullptr;
    uintptr_t srcOffsetAfter16 = (srcPtrAfter16 != nullptr) ? reinterpret_cast<uintptr_t>(srcPtrAfter16) : 0;
    uintptr_t dstOffsetAfter16 = (dstPtrAfter16 != nullptr) ? reinterpret_cast<uintptr_t>(dstPtrAfter16) : 0;
    intptr_t relOffset4After16 = static_cast<intptr_t>(srcOffsetAfter16) - static_cast<intptr_t>(dstOffsetAfter16);

    // Check individual pointer alignment for Pack4 (always 4-byte alignment requirement)
    // getAlignment returns bytes to next aligned address (0 = already aligned)
    using Pack4 = nccl::utility::EltPackForBytes<T, 4>;
    constexpr unsigned pack4Align = 4; // Pack4 always requires 4-byte alignment
    bool srcAligned4 = (srcPtrAfter16 == nullptr) || (nccl::utility::getAlignment(srcPtrAfter16, pack4Align) == 0);
    bool dstAligned4 = (dstPtrAfter16 == nullptr) || (nccl::utility::getAlignment(dstPtrAfter16, pack4Align) == 0);

    // Check if Pack4 can be used: relative alignment must be divisible by 4, and individual pointers must be aligned
    if (sizeof(T) == 4 || (sizeof(T) < 4 && relOffset4After16 % 4 == 0 && srcAligned4 && dstAligned4)) {
      if (remainingAfter16 * scalarSize >= 4) {
        if NCCL_IF_CONSTEXPR (Pack4::Count > 0) {
          constexpr int UNROLL_PACKS4_RAW = static_cast<int>(safeDiv(UNROLL_ELTS + Pack4::Count - 1, Pack4::Count));
          constexpr int UNROLL_PACKS4 = (UNROLL_PACKS4_RAW > 0) ? UNROLL_PACKS4_RAW : 1;
          if NCCL_IF_CONSTEXPR (UNROLL_PACKS4_RAW > 0) {
            IntCount vectorizableElts4 = safeDiv<IntCount>(remainingAfter16, Pack4::Count) * Pack4::Count;
            if (vectorizableElts4 > 0) {
              auto srcAfter16 = [=] __device__(int i) -> T* { return srcRemaining(i) + vectorizedElts; };
              auto dstAfter16 = [=] __device__(int i) -> T* { return dstRemaining(i) + vectorizedElts; };
              vectorizedElts += reduceCopyLoop<UNROLL_PACKS4, T, Pack4, RedOp, IntCount, Coop, srcMultimem, dstMultimem,
                                               decltype(srcAfter16), decltype(dstAfter16), /*skipTail=*/false>(
                coop, srcAfter16, nSrc, dstAfter16, nDst, redOp, vectorizableElts4);
            }
          }
        }
      }
    }
  }

  // Step 3: Scalar remainder
  IntCount scalarRemainder = remainingElts - vectorizedElts;
  if (scalarRemainder > 0) {
    auto srcScalar = [=] __device__(int i) -> T* { return srcRemaining(i) + vectorizedElts; };
    auto dstScalar = [=] __device__(int i) -> T* { return dstRemaining(i) + vectorizedElts; };

    // Process scalar remainder - always use scalar loop with EltPack<T, 1>
    reduceCopyScalarLoop<T, RedOp, IntCount, Coop, srcMultimem, dstMultimem>(coop, srcScalar, nSrc, dstScalar, nDst,
                                                                             redOp, scalarRemainder);
  }
}

#if __CUDA_ARCH__ >= 1000
// TMA-based Copy (Broadcast) Loop (sm100+ only). SmemBytesTotal is the size of the caller's staging
// buffer for the whole coop. It is split into as many 16B-aligned [data tile][mbarrier] slots as
// fit (each at least 32B). Extra warps stay idle and rejoin at coop.sync().
template <int SmemBytesTotal, typename T, typename IntCount, typename Coop, typename DstLambda>
NCCL_DEVICE_INLINE void lsaCopyTma(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count, char* smemPtr) {
  using Pack = EltPackForBytes<T, 16>;
  using Bar = cuda::barrier<cuda::thread_scope_block>;
  constexpr int warpSize = 32;
  constexpr int barFootprint = (int)((sizeof(Bar) + 15) & ~size_t(15)); // rounds sizeof(Bar) up to a multiple of 16
  constexpr int minSlotBytes = barFootprint + 16;

  static_assert(SmemBytesTotal >= minSlotBytes, "SmemBytesTotal must be at least 32 bytes");

  const IntCount totalPacks = safeDiv<IntCount>(count, Pack::Count);
  const int nWarps = (coop.size() + warpSize - 1) / warpSize;
  const int laneId = coop.thread_rank() % warpSize;
  const int groupId = coop.thread_rank() / warpSize;

  // We try to use as many warps as we can to get a minSlotBytes slice
  const int nActiveWarps = (nWarps < SmemBytesTotal / minSlotBytes) ? nWarps : (SmemBytesTotal / minSlotBytes);
  const int smemBytesPerWarp = (SmemBytesTotal / nActiveWarps) & ~15;
  const size_t tileSize = (size_t)(smemBytesPerWarp - barFootprint);
  const IntCount packsPerWarpTile = (IntCount)(tileSize / sizeof(Pack));
  const IntCount packsPerIter = (IntCount)nActiveWarps * packsPerWarpTile;

  if (groupId < nActiveWarps && laneId == 0) {
    char* slot = smemPtr + groupId * smemBytesPerWarp;
    Pack* tmaBuff = reinterpret_cast<Pack*>(slot);
    Bar* tmaBar = reinterpret_cast<Bar*>(slot + tileSize);
    init(tmaBar, 1);

    Pack* srcPack = (Pack*)srcPtr;
    // nBytes should always be a multiple of 16
    auto loadTile = [&](IntCount packIdx, size_t nBytes) {
      cuda::device::memcpy_async_tx(tmaBuff, srcPack + packIdx, cuda::aligned_size_t<16>(nBytes), *tmaBar);
      typename Bar::arrival_token token = cuda::device::barrier_arrive_tx(*tmaBar, 1, nBytes);
      tmaBar->wait(std::move(token));
    };
    auto storeTile = [&](IntCount packIdx, size_t nBytes) {
      NVCC_PRAGMA_UNROLL(4)
      for (int dstIdx = 0; dstIdx < nDst; dstIdx++) {
        Pack* dstPtr = (Pack*)dstLambda(dstIdx);
        ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, dstPtr + packIdx, tmaBuff, nBytes);
      }
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    };

    IntCount basePackIdx = 0;
    IntCount groupBasePackIdx = groupId * packsPerWarpTile;

    // PHASE 1: bulk load GMEM -> SMEM, then wait for it to land.
    if (packsPerIter <= totalPacks) {
      loadTile(groupBasePackIdx, tileSize);
    }

    // PHASE 2: bulk store SMEM -> peer GMEM for each destination, then wait for the buffer to drain.
    while (basePackIdx + packsPerIter <= totalPacks) {
      storeTile(groupBasePackIdx, tileSize);

      basePackIdx += packsPerIter;
      groupBasePackIdx += packsPerIter;
      if (basePackIdx + packsPerIter > totalPacks) break;

      loadTile(groupBasePackIdx, tileSize); // (back to) PHASE 1 for the next tile
    }

    // PHASE 3: at the end the tail warp may be less than packsPerIter. In that case only a
    // prefix of the active warps participate, and the last of those transfers "avail" packs.
    const IntCount tailPacks = totalPacks - basePackIdx;
    const IntCount myTailOffset = (IntCount)groupId * packsPerWarpTile;
    if (myTailOffset < tailPacks) {
      const IntCount avail = tailPacks - myTailOffset;
      const IntCount myTailPacks = (avail < packsPerWarpTile) ? avail : packsPerWarpTile;
      const size_t tailBytes = (size_t)myTailPacks * sizeof(Pack);

      loadTile(groupBasePackIdx, tailBytes);
      storeTile(groupBasePackIdx, tailBytes);
    }
  }
  coop.sync();
}
#endif // __CUDA_ARCH__ >= 1000

} // namespace utility
} // namespace nccl

#endif // __CUDACC__ && __CUDACC_EXTENDED_LAMBDA__

#endif // _NCCL_DEVICE_REDUCE_COPY__IMPL_H_
