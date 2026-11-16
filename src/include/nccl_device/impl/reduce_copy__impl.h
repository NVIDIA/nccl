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

// CCCL headers declare constants referenced by host registration.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 1000
#include <cuda/barrier>
#include <cuda/ptx>
#endif

#if __CUDA_ARCH__ >= 1000
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
  const int coopSize = coop.size();
  const int runtimeStride = (coopStride != 0) ? coopStride : min(coopSize, warpSize);
  const int laneId = coop.thread_rank() % runtimeStride;
  const int groupId = coop.thread_rank() / runtimeStride;

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
// cuda::ptx::elect_sync is absent from the CUDA 12.9 CCCL used by the cu12 CI build
NCCL_DEVICE_INLINE bool electSync() {
  const unsigned mask = __activemask();
  int elected = 0;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "elect.sync _|p, %1;\n\t"
               "selp.b32 %0, 1, 0, p;\n\t"
               "}"
               : "=r"(elected)
               : "r"(mask));
  return elected != 0;
}

// Serial TMA copy for coops with fewer than 2 warps (Thread, Warp, or a single-warp CTA).
// One elected thread loops over one 16B-aligned [mbarrier][data tile] slot.
template <typename T, typename IntCount, typename Coop, typename DstLambda>
NCCL_DEVICE_INLINE void lsaCopyTmaLeaderWarp(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count,
                                             char* smemPtr, int smemBytesTotal) {
  constexpr int barFootprint = 16;
  constexpr int minSlotBytes = barFootprint + 16;
  constexpr int warpSize = 32;
  const int warpId = coop.thread_rank() / warpSize;

  if (smemBytesTotal < minSlotBytes) {
#if NCCL_DEVICE_DEBUG_CHECKS
    if (warpId == 0 && electSync()) {
      assert(false && "smemBytesTotal must be at least 32 bytes");
    }
#endif
    return;
  }

  bool isLeader;
  if NCCL_IF_CONSTEXPR (std::is_same<Coop, ncclCoopThread>::value) {
    // if coop is thread, electSync() would pick one lane per warp and drop the other ncclCoopThread coops in the warp.
    isLeader = coop.thread_rank() == 0;
  } else {
    isLeader = (warpId == 0 && electSync());
  }

  if (isLeader) {
    using Bar = cuda::barrier<cuda::thread_scope_block>;
    Bar* bar = reinterpret_cast<Bar*>(smemPtr);
    char* buf = smemPtr + barFootprint;
    const size_t totalBytes = (size_t)count * sizeof(T); // total amount of data we must copy
    const size_t smemStageTile = (size_t)((smemBytesTotal - barFootprint) & ~15); // amount of staging smem available
    char* src = reinterpret_cast<char*>(srcPtr);

    init(bar, 1);

    for (size_t offset = 0; offset < totalBytes; offset += smemStageTile) {
      const uint32_t nBytes = (uint32_t)((totalBytes - offset < smemStageTile) ? (totalBytes - offset) : smemStageTile);

      // load tile GMEM -> SMEM, then wait for it to land
      cuda::device::memcpy_async_tx(buf, src + offset, cuda::aligned_size_t<16>(nBytes), *bar);
      typename Bar::arrival_token token = cuda::device::barrier_arrive_tx(*bar, 1, nBytes);
      bar->wait(std::move(token));

      // store tile SMEM -> GMEM to every destination, then wait for the buffer to drain
      for (int dstIdx = 0; dstIdx < nDst; dstIdx++) {
        char* dst = reinterpret_cast<char*>(dstLambda(dstIdx));
        ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, dst + offset, buf, nBytes);
      }
      ptx::cp_async_bulk_commit_group();
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
  }
  coop.sync();
}

// TMA-based Copy (Broadcast) Loop (sm100+ only): warp-specialized pipeline over nStages smem tiles.
// Warp 0 does TMA load of tiles from GMEM -> SMEM over nStages, warp 1 stores each landed tile SMEM -> GMEM over nStages.
// Each stage has a "full" mbarrier (load landed, tracked via tx-count) and an "empty" mbarrier (stores finished reading the tile).
template <typename T, typename IntCount, typename Coop, typename DstLambda>
NCCL_DEVICE_INLINE void lsaCopyTma(Coop coop, T* srcPtr, DstLambda dstLambda, int nDst, IntCount count, char* smemPtr,
                                   int smemBytesTotal) {
  constexpr int nStages = 4;
  constexpr int warpSize = 32;
  constexpr int barBytes = (int)((2 * nStages * sizeof(uint64_t) + 15) & ~size_t(15));
  // 8 barriers * 8 bytes/barrier + 4 stages * 16 bytes/stage = 128 byte minimum for the pipelined path.
  constexpr int minSmemBytes = barBytes + nStages * 16;

  // Coops with fewer than 2 warps (thread, warp, lanes, or a small warp span / CTA), or too little smem for nStages
  // tiles, run the serialized single-thread TMA load-store path.
  if (ncclCoopWithinWarp(coop) || coop.size() < 2 * warpSize || smemBytesTotal < minSmemBytes) {
    lsaCopyTmaLeaderWarp<T, IntCount, Coop, DstLambda>(coop, srcPtr, dstLambda, nDst, count, smemPtr, smemBytesTotal);
    return;
  }

  // smem layout: [fullBar x nStages][emptyBar x nStages] (padded to 16B) then nStages data tiles ("buf").
  using Bar = cuda::barrier<cuda::thread_scope_block>;
  Bar* fullBar = reinterpret_cast<Bar*>(smemPtr);
  Bar* emptyBar = fullBar + nStages;
  char* buf = smemPtr + barBytes;
  const size_t smemStageTile =
    (size_t)(((smemBytesTotal - barBytes) / nStages) & ~15); // size of a single SMEM stage tile

  const int warpId = coop.thread_rank() / warpSize;
  const int laneId = coop.thread_rank() % warpSize;

  static_assert(nStages <= warpSize, "stage pipeline length must be less than 32.");
  if (warpId == 0 && laneId < nStages) {
    init(&fullBar[laneId], 1);
    init(&emptyBar[laneId], 1);
    ptx::fence_mbarrier_init(ptx::sem_release, ptx::scope_cluster);
  }
  coop.sync();

  if (warpId < 2 && electSync()) {
    const size_t totalBytes = (size_t)count * sizeof(T);
    const size_t nTiles = (totalBytes + smemStageTile - 1) / smemStageTile;
    char* src = reinterpret_cast<char*>(srcPtr);

    if (warpId == 0) { // loader warp
      for (size_t i = 0; i < nTiles; i++) {
        // every data tile i is pre-assigned a stage, phase the loader waits for, and byte offset.
        const int stage = (int)(i % nStages);
        const uint32_t phase = (uint32_t)((i / nStages) & 1);
        const size_t offset = i * smemStageTile;
        const uint32_t nBytes =
          (uint32_t)((totalBytes - offset < smemStageTile) ? (totalBytes - offset) : smemStageTile);
        char* tile = buf + stage * smemStageTile;

        // wait until the store warp has drained this stage before issuing the next load, notifying, and moving on.
        while (!ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&emptyBar[stage]), phase ^ 1)) {
        }
        cuda::device::memcpy_async_tx(tile, src + offset, cuda::aligned_size_t<16>(nBytes), fullBar[stage]);
        (void)cuda::device::barrier_arrive_tx(fullBar[stage], 1, nBytes);
      }
    } else { // storer warp
      constexpr int K = (nStages >= 3) ? 1 : 0; // max store-group-operations that can get left outstanding

      for (size_t i = 0; i < nTiles; i++) {
        // every data tile i is pre-assigned a stage, phase the loader waits for, and byte offset.
        const int stage = (int)(i % nStages);
        const uint32_t phase = (uint32_t)((i / nStages) & 1);
        const size_t offset = i * smemStageTile;
        const uint32_t nBytes =
          (uint32_t)((totalBytes - offset < smemStageTile) ? (totalBytes - offset) : smemStageTile);
        char* tile = buf + stage * smemStageTile;

        // wait until this storer warp's stage is available
        while (!ptx::mbarrier_try_wait_parity(reinterpret_cast<uint64_t*>(&fullBar[stage]), phase)) {
        }

        // issue TMA stores to each dst and commit the group
        for (int dstIdx = 0; dstIdx < nDst; dstIdx++) {
          char* dst = reinterpret_cast<char*>(dstLambda(dstIdx));
          ptx::cp_async_bulk(ptx::space_global, ptx::space_shared, dst + offset, tile, nBytes);
        }
        ptx::cp_async_bulk_commit_group();

        if (i >= (size_t)K) {
          // Only the K most recently enqueued tiles may still be unfinished (not yet read out of smem).
          ptx::cp_async_bulk_wait_group_read(ptx::n32_t<K>());
          // After enqueueing tile i, wait until tile (i - K) is read out, then free its stage.
          (void)ptx::mbarrier_arrive(reinterpret_cast<uint64_t*>(&emptyBar[(i - K) % nStages]));
        }
      }
      ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
    }
  }
  coop.sync();
}
#endif // __CUDA_ARCH__ >= 1000

} // namespace utility
} // namespace nccl

#endif // __CUDACC__ && __CUDACC_EXTENDED_LAMBDA__

#endif // _NCCL_DEVICE_REDUCE_COPY__IMPL_H_
