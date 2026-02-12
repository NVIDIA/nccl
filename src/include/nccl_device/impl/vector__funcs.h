/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_VECTOR__FUNCS_H_
#define _NCCL_DEVICE_VECTOR__FUNCS_H_

#include "vector__types.h"
#include "reduce_copy__types.h"
#include "../utility.h"
#include "../coop.h"
#if defined(__CUDA_FP4_TYPES_EXIST__)
#include <cuda_fp4.h>
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__)
#include <cuda_fp8.h>
#endif

#if NCCL_CHECK_CUDACC

namespace nccl {
namespace utility {

// ============================================================================
// Alignment Utilities
// ============================================================================

// Compute alignment of a pointer relative to a modulo value
// Returns the number of bytes from ptr to the next aligned address
// Returns 0 if ptr is already aligned
NCCL_DEVICE_INLINE unsigned getAlignment(void* ptr, unsigned modulo) {
  return (modulo - reinterpret_cast<uintptr_t>(ptr)) % modulo;
}

// Safe division helper (returns 0 when denominator is 0)
template <typename Int>
NCCL_DEVICE_INLINE constexpr Int safeDiv(Int numerator, Int denominator) {
  return (denominator == 0) ? 0 : (numerator / denominator);
}

// Compute common alignment across multiple pointers using warp reduce
// This is more efficient than checking each pointer sequentially when nPtrs is large
// Returns the common alignment offset in bytes (0 means aligned)
template <typename Pack, typename Coop, typename Lambda>
NCCL_DEVICE_INLINE unsigned computeCommonAlignment(Coop coop, Lambda ptrLambda, int nPtrs) {
#if __CUDA_ARCH__ >= 800
  // Use efficient warp reduce
  auto lanes = ncclCoopCoalesced(coop);
  unsigned commonAlign = 0;
  #pragma unroll 1
  for (int i = lanes.thread_rank(); i < nPtrs; i += lanes.size()) {
    unsigned align = getAlignment(ptrLambda(i), sizeof(Pack));
    commonAlign = 1 + min(commonAlign - 1, align - 1);
  }
  commonAlign = 1 + __reduce_min_sync(ncclCoopGetLaneMask(lanes), commonAlign - 1);
  return commonAlign;
#else
  // Fall back to simple sequential check for older architectures
  unsigned commonAlign = 0;
  for (int i = 0; i < nPtrs; i++) {
    unsigned align = getAlignment(ptrLambda(i), sizeof(Pack));
    commonAlign = 1 + min(commonAlign - 1, align - 1);
  }
  return commonAlign;
#endif
}


// Helper to compute alignment for a specific pack size
// Returns alignment offset in bytes (0 means aligned)
template <typename T, int PackBytes, typename Coop, typename SrcLambda, typename DstLambda>
NCCL_DEVICE_INLINE unsigned computeAlignmentForPackSize(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst) {
  using Pack = EltPackForBytes<T, PackBytes>;
  unsigned srcAlign = (nSrc > 0) ? computeCommonAlignment<Pack>(coop, srcLambda, nSrc) : 0;
  unsigned dstAlign = (nDst > 0) ? computeCommonAlignment<Pack>(coop, dstLambda, nDst) : 0;

  // Compute common alignment using the 1+min(x-1,y-1) trick
  unsigned commonAlign = 0;
  if (nSrc > 0 && nDst > 0) {
    commonAlign = 1 + min(srcAlign - 1, dstAlign - 1);
  } else if (nSrc > 0) {
    commonAlign = srcAlign;
  } else if (nDst > 0) {
    commonAlign = dstAlign;
  }
  return commonAlign;
}

// Helper to try alignment for a specific pack size with lambdas
// Matches all_reduce.cuh: checks both individual alignment and relative alignment
template <typename T, int PackBytes, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount>
NCCL_DEVICE_INLINE bool tryLambdaAlignmentForPackSize(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count,
    IntCount& alignOffset,
    int& maxPackBytes) {
  constexpr IntCount eltPerPack = (PackBytes * 8) / bitSizeOf<T>();

  if (eltPerPack == 0 || count < eltPerPack) {
    return false;  // Too small to vectorize with this pack size
  }

  // Compute individual alignment for this pack size
  unsigned commonAlign = computeAlignmentForPackSize<T, PackBytes>(coop, srcLambda, nSrc, dstLambda, nDst);

  // Check relative alignment (matching all_reduce.cuh: (input.offset - output.offset) % PackBytes == 0)
  // After processing prefix, both pointers advance by the same amount, so relative offset doesn't change
  // We need to ensure the relative offset is divisible by PackBytes
  if (nSrc > 0 && nDst > 0) {
    void* srcPtr0 = srcLambda(0);
    void* dstPtr0 = dstLambda(0);
    uintptr_t srcOffset = reinterpret_cast<uintptr_t>(srcPtr0);
    uintptr_t dstOffset = reinterpret_cast<uintptr_t>(dstPtr0);
    intptr_t relOffset = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

    // Check if relative alignment is achievable (relative offset must be divisible by PackBytes)
    unsigned relOffsetAbs = (relOffset < 0) ? static_cast<unsigned>(-relOffset) : static_cast<unsigned>(relOffset);
    if (relOffsetAbs % PackBytes != 0) {
      // Relative alignment not achievable with this pack size
      return false;
    }
  }

  // Relative alignment is OK - use individual alignment
  unsigned totalAlignBytes = commonAlign;

  // Check if alignment is valid (must be divisible by element size)
  if (totalAlignBytes % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = totalAlignBytes / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// Result structure for alignment computation
template <typename IntCount>
struct AlignmentResult {
  IntCount alignOffset;  // number of scalar elements to skip before vectorized processing
  int maxPackBytes;      // maximum pack size (in bytes) that can be used after alignment
};

// Compute alignment offset with fallback to smaller pack sizes
// Tries 16, 4 bytes and returns both the offset and the max pack size that works
template <typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount>
NCCL_DEVICE_INLINE AlignmentResult<IntCount> computeLambdaAlignmentOffsetWithFallback(
    Coop coop,
    SrcLambda srcLambda, int nSrc,
    DstLambda dstLambda, int nDst,
    IntCount count) {
  AlignmentResult<IntCount> result;
  result.alignOffset = 0;
  result.maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // Try each pack size from largest to smallest using explicit template instantiations
  if (tryLambdaAlignmentForPackSize<T, 16>(coop, srcLambda, nSrc, dstLambda, nDst, count, result.alignOffset, result.maxPackBytes)) {
    return result;
  }
  if (tryLambdaAlignmentForPackSize<T, 4>(coop, srcLambda, nSrc, dstLambda, nDst, count, result.alignOffset, result.maxPackBytes)) {
    return result;
  }

  // If we get here, no pack size worked - process all as scalars
  result.alignOffset = count;
  result.maxPackBytes = static_cast<int>(sizeof(T));
  return result;
}

// Helper to compute alignment for a specific pack size (template parameter)
// Matches all_reduce.cuh: checks both individual alignment and relative alignment
template <typename T, int PackBytes, typename IntCount>
NCCL_DEVICE_INLINE bool tryPointerPairAlignmentForPackSize(
    void* srcPtr,
    void* dstPtr,
    IntCount& alignOffset,
    int& maxPackBytes) {
  using Pack = EltPackForBytes<T, PackBytes>;

  // Check individual alignments
  unsigned srcAlign = getAlignment(srcPtr, sizeof(Pack));
  unsigned dstAlign = getAlignment(dstPtr, sizeof(Pack));
  unsigned commonAlign = 1 + min(srcAlign - 1, dstAlign - 1);

  // Check relative alignment (matching all_reduce.cuh: (input.offset - output.offset) % PackBytes == 0)
  // After processing prefix, both pointers advance by the same amount, so relative offset doesn't change
  // We need to ensure the relative offset is divisible by PackBytes
  uintptr_t srcOffset = reinterpret_cast<uintptr_t>(srcPtr);
  uintptr_t dstOffset = reinterpret_cast<uintptr_t>(dstPtr);
  intptr_t relOffset = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

  // Check if relative alignment is achievable (relative offset must be divisible by PackBytes after prefix)
  // Since prefix advances both pointers equally, relative offset stays the same
  unsigned relOffsetAbs = (relOffset < 0) ? static_cast<unsigned>(-relOffset) : static_cast<unsigned>(relOffset);
  if (relOffsetAbs % PackBytes != 0) {
    // Relative alignment not achievable with this pack size - would need different prefix for src vs dst
    return false;
  }

  // Relative alignment is OK - use individual alignment
  unsigned totalAlignBytes = commonAlign;

  // Check if alignment is valid (must be divisible by element size)
  if (totalAlignBytes % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = totalAlignBytes / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}


// Compute alignment for two pointers with fallback to smaller pack sizes
// Tries 16, 4 bytes and returns both the offset and the max pack size that works
template <typename T, typename IntCount>
NCCL_DEVICE_INLINE void computePointerPairAlignmentWithFallback(
    void* srcPtr,
    void* dstPtr,
    IntCount count,
    IntCount& alignOffset,
    int& maxPackBytes) {
  alignOffset = 0;
  maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // Try each pack size from largest to smallest using explicit template instantiations
  if (tryPointerPairAlignmentForPackSize<T, 16>(srcPtr, dstPtr, alignOffset, maxPackBytes)) {
    return;
  }
  if (tryPointerPairAlignmentForPackSize<T, 4>(srcPtr, dstPtr, alignOffset, maxPackBytes)) {
    return;
  }

  // If we get here, no pack size worked - process all as scalars
  alignOffset = count;
  maxPackBytes = static_cast<int>(sizeof(T));
}

// Helper to compute strided alignment for a specific pack size (template parameter)
template <typename T, int PackBytes, typename IntCount>
NCCL_DEVICE_INLINE bool tryStridedAlignmentForPackSize(
    void* basePtr,
    size_t displ,
    IntCount& alignOffset,
    int& maxPackBytes) {
  using Pack = EltPackForBytes<T, PackBytes>;

  unsigned baseAlign = getAlignment(basePtr, sizeof(Pack));
  unsigned displAlign = getAlignment(reinterpret_cast<void*>(displ), sizeof(Pack));
  unsigned commonAlign = 1 + min(baseAlign - 1, displAlign - 1);

  // Check if alignment is valid (must be divisible by element size)
  if (commonAlign % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = commonAlign / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// Helper to try complex strided alignment (src and dst) for a specific pack size
template <typename T, int PackBytes, typename IntCount>
NCCL_DEVICE_INLINE bool tryComplexStridedAlignmentForPackSize(
    void* srcBasePtr,
    size_t srcDispl,
    void* dstBasePtr,
    size_t dstDispl,
    IntCount& alignOffset,
    int& maxPackBytes) {
  using Pack = EltPackForBytes<T, PackBytes>;

  // Compute alignment for source and destination strides
  unsigned srcAlign = getAlignment(srcBasePtr, sizeof(Pack));
  unsigned srcDisplAlign = getAlignment(reinterpret_cast<void*>(srcDispl), sizeof(Pack));
  unsigned srcCommonAlign = 1 + min(srcAlign - 1, srcDisplAlign - 1);

  unsigned dstAlign = getAlignment(dstBasePtr, sizeof(Pack));
  unsigned dstDisplAlign = getAlignment(reinterpret_cast<void*>(dstDispl), sizeof(Pack));
  unsigned dstCommonAlign = 1 + min(dstAlign - 1, dstDisplAlign - 1);

  // Compute common alignment across source and destination
  unsigned commonAlign = 1 + min(srcCommonAlign - 1, dstCommonAlign - 1);

  // Check relative alignment (matching all_reduce.cuh: (input.offset - output.offset) % PackBytes == 0)
  // After processing prefix, both pointers advance by the same amount, so relative offset doesn't change
  // We need to ensure the relative offset is divisible by PackBytes
  uintptr_t srcOffset = reinterpret_cast<uintptr_t>(srcBasePtr);
  uintptr_t dstOffset = reinterpret_cast<uintptr_t>(dstBasePtr);
  intptr_t relOffset = static_cast<intptr_t>(srcOffset) - static_cast<intptr_t>(dstOffset);

  // Check if relative alignment is achievable (relative offset must be divisible by PackBytes)
  unsigned relOffsetAbs = (relOffset < 0) ? static_cast<unsigned>(-relOffset) : static_cast<unsigned>(relOffset);
  if (relOffsetAbs % PackBytes != 0) {
    // Relative alignment not achievable with this pack size
    return false;
  }

  // Relative alignment is OK - use individual alignment
  unsigned totalAlignBytes = commonAlign;

  // Check if alignment is valid (must be divisible by element size)
  if (totalAlignBytes % static_cast<unsigned int>(sizeof(T)) == 0) {
    alignOffset = totalAlignBytes / static_cast<unsigned int>(sizeof(T));
    maxPackBytes = PackBytes;
    return true;  // Found a working pack size
  }
  return false;
}

// Compute strided alignment with fallback to smaller pack sizes
// Tries 16, 4 bytes and returns both the offset and the max pack size that works
template <typename T, typename IntCount>
NCCL_DEVICE_INLINE void computeStridedAlignmentWithFallback(
    void* basePtr,
    size_t displ,
    IntCount count,
    IntCount& alignOffset,
    int& maxPackBytes) {
  alignOffset = 0;
  maxPackBytes = static_cast<int>(sizeof(T));  // Default to scalar if nothing works

  // Try each pack size from largest to smallest using explicit template instantiations
  if (tryStridedAlignmentForPackSize<T, 16>(basePtr, displ, alignOffset, maxPackBytes)) {
    return;
  }
  if (tryStridedAlignmentForPackSize<T, 4>(basePtr, displ, alignOffset, maxPackBytes)) {
    return;
  }

  // If we get here, no pack size worked - process all as scalars
  alignOffset = count;
  maxPackBytes = static_cast<int>(sizeof(T));
}

// ============================================================================
// Pack casting and reduction operations
// ============================================================================
// These functions provide a simpler interface for working with typed packs.
// castPack: Cast pack from element type X to element type Y
// reducePack: Reduce two packs using a reduction operator

template <typename T, int n>
struct PackAccess {
  union {
    EltPack<T, n> pack;
    struct { EltPack<T, n / 2> lo; EltPack<T, n / 2> hi; };
  };
};

template <typename T>
struct PackAccess<T, 1> {
  union {
    EltPack<T, 1> pack;
    struct { EltPack<T, 1> lo; };
    struct { EltPack<T, 1> hi; };
  };
};

template <typename T>
struct PackAccess<T, 0> {
  union {
    EltPack<T, 0> pack;
    struct { EltPack<T, 0> lo; };
    struct { EltPack<T, 0> hi; };
  };
};

// Cast pack from element type X to element type Y
// Works with EltPack types
template<typename Y, typename X, int n>
NCCL_DEVICE_INLINE EltPack<Y, n> castPack(EltPack<X, n> x) {
  static_assert((n & (n - 1)) == 0, "EltPack requires power-of-two element count");

  PackAccess<X, n> in;
  PackAccess<Y, n> out;
  in.pack = x;
  if NCCL_IF_CONSTEXPR (n == 1) {
    out.pack.elts()[0] = static_cast<Y>(in.pack.elts()[0]);
  } else {
    out.lo = castPack<Y>(in.lo);
    out.hi = castPack<Y>(in.hi);
  }
  return out.pack;
}

// Specialization for zero-sized packs
template<typename Y, typename X>
NCCL_DEVICE_INLINE EltPack<Y, 0> castPack(EltPack<X, 0> /* x */) {
  EltPack<Y, 0> result{};
  return result;
}

// ============================================================================
// CastPack Specializations
// ============================================================================

// Specialization for half -> float conversion (upcast to accumulation type)
template<>
NCCL_DEVICE_INLINE EltPack<float, 1> castPack(EltPack<half, 1> x) {
  EltPack<float, 1> out{};
  out.elts()[0] = __half2float(x.elts()[0]);
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<float, 2> castPack(EltPack<half, 2> x) {
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  union Float2PackAccess {
    EltPack<float, 2> pack;
    float2 pair;
  };
  Half2PackAccess in;
  Float2PackAccess out;
  in.pack = x;
  out.pair = __half22float2(in.pair);
  return out.pack;
}

// Specialization for float -> half conversion (downcast from accumulation type)
template<>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<float, 1> x) {
  EltPack<half, 1> out{};
  out.elts()[0] = __float2half_rn(x.elts()[0]);  // Round to nearest
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<float, 2> x) {
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Half2PackAccess out;
  out.pair = __floats2half2_rn(x.elts()[0], x.elts()[1]);
  return out.pack;
}

#if defined(__CUDA_BF16_TYPES_EXIST__)

// Specialization for __nv_bfloat16 -> float conversion (upcast to accumulation type)
template<>
NCCL_DEVICE_INLINE EltPack<float, 1> castPack(EltPack<__nv_bfloat16, 1> x) {
  EltPack<float, 1> out{};
  out.elts()[0] = __bfloat162float(x.elts()[0]);
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<float, 2> castPack(EltPack<__nv_bfloat16, 2> x) {
  union Bf162PackAccess {
    EltPack<__nv_bfloat16, 2> pack;
    __nv_bfloat162 pair;
  };
  union Float2PackAccess {
    EltPack<float, 2> pack;
    float2 pair;
  };
  Bf162PackAccess in;
  Float2PackAccess out;
  in.pack = x;
  out.pair = __bfloat1622float2(in.pair);
  return out.pack;
}

// Specialization for float -> __nv_bfloat16 conversion (downcast from accumulation type)
template<>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 1> castPack(EltPack<float, 1> x) {
  EltPack<__nv_bfloat16, 1> out{};
  out.elts()[0] = __float2bfloat16_rn(x.elts()[0]);  // Round to nearest
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 2> castPack(EltPack<float, 2> x) {
  union Bf162PackAccess {
    EltPack<__nv_bfloat16, 2> pack;
    __nv_bfloat162 pair;
  };
  union Float2PackAccess {
    EltPack<float, 2> pack;
    float2 pair;
  };
  Float2PackAccess in;
  Bf162PackAccess out;
  in.pack = x;
  out.pair = __float22bfloat162_rn(in.pair);
  return out.pack;
}
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)

// Specialization for __nv_fp8_e4m3 -> half conversion (upcast to accumulation type)
// Uses vectorized fp8x2 -> half2 when available for SIMD performance
template<>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<__nv_fp8_e4m3, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E4m3Access1 {
    EltPack<__nv_fp8_e4m3, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E4m3Access1 in;
  EltPack<half, 1> out{};
  in.pack = x;
  HalfRawAccess h;
  h.raw = __nv_cvt_fp8_to_halfraw(in.storage, __NV_E4M3);
  out.elts()[0] = h.val;
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<__nv_fp8_e4m3, 2> x) {
  #if __CUDA_ARCH__ >= 900
    union Half2RawAccess {
      __half2_raw raw;
      half2 val;
    };
    union Half2PackAccess {
      EltPack<half, 2> pack;
      half2 pair;
    };
    union Fp8E4m3Access2 {
      EltPack<__nv_fp8_e4m3, 2> pack;
      __nv_fp8x2_storage_t storage2;
    };
    Fp8E4m3Access2 in;
    Half2PackAccess out;
    in.pack = x;
    Half2RawAccess h2;
    h2.raw = __nv_cvt_fp8x2_to_halfraw2(in.storage2, __NV_E4M3);
    out.pair = h2.val;
    return out.pack;
  #else
    PackAccess<__nv_fp8_e4m3, 2> in;
    PackAccess<half, 2> out;
    in.pack = x;
    out.lo = castPack<half>(in.lo);
    out.hi = castPack<half>(in.hi);
    return out.pack;
  #endif
}

// Specialization for __nv_fp8_e5m2 -> half conversion (upcast to accumulation type)
// Uses vectorized fp8x2 -> half2 when available for SIMD performance
template<>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<__nv_fp8_e5m2, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E5m2Access1 {
    EltPack<__nv_fp8_e5m2, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E5m2Access1 in;
  EltPack<half, 1> out{};
  in.pack = x;
  HalfRawAccess h;
  h.raw = __nv_cvt_fp8_to_halfraw(in.storage, __NV_E5M2);
  out.elts()[0] = h.val;
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<__nv_fp8_e5m2, 2> x) {
  #if __CUDA_ARCH__ >= 900
    union Half2RawAccess {
      __half2_raw raw;
      half2 val;
    };
    union Half2PackAccess {
      EltPack<half, 2> pack;
      half2 pair;
    };
    union Fp8E5m2Access2 {
      EltPack<__nv_fp8_e5m2, 2> pack;
      __nv_fp8x2_storage_t storage2;
    };
    Fp8E5m2Access2 in;
    Half2PackAccess out;
    in.pack = x;
    Half2RawAccess h2;
    h2.raw = __nv_cvt_fp8x2_to_halfraw2(in.storage2, __NV_E5M2);
    out.pair = h2.val;
    return out.pack;
  #else
    PackAccess<__nv_fp8_e5m2, 2> in;
    PackAccess<half, 2> out;
    in.pack = x;
    out.lo = castPack<half>(in.lo);
    out.hi = castPack<half>(in.hi);
    return out.pack;
  #endif
}

// Specialization for half -> __nv_fp8_e4m3 conversion (downcast from accumulation type)
// Uses vectorized half2 -> fp8x2 when available for SIMD performance
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 1> castPack(EltPack<half, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E4m3Access1 {
    EltPack<__nv_fp8_e4m3, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E4m3Access1 out;
  HalfRawAccess h;
  h.val = x.elts()[0];
  out.storage = __nv_cvt_halfraw_to_fp8(h.raw, __NV_SATFINITE, __NV_E4M3);
  return out.pack;
}

template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e4m3, 2> castPack(EltPack<half, 2> x) {
  #if __CUDA_ARCH__ >= 900
    union Half2RawAccess {
      __half2_raw raw;
      half2 val;
    };
    union Half2PackAccess {
      EltPack<half, 2> pack;
      half2 pair;
    };
    Half2PackAccess in;
    union Fp8E4m3Access2 {
      EltPack<__nv_fp8_e4m3, 2> pack;
      __nv_fp8x2_storage_t storage2;
    };
    Fp8E4m3Access2 out;
    in.pack = x;
    Half2RawAccess h2;
    h2.val = in.pair;
    out.storage2 = __nv_cvt_halfraw2_to_fp8x2(h2.raw, __NV_SATFINITE, __NV_E4M3);
    return out.pack;
  #else
    PackAccess<half, 2> in;
    PackAccess<__nv_fp8_e4m3, 2> out;
    in.pack = x;
    out.lo = castPack<__nv_fp8_e4m3>(in.lo);
    out.hi = castPack<__nv_fp8_e4m3>(in.hi);
    return out.pack;
  #endif
}

// Specialization for half -> __nv_fp8_e5m2 conversion (downcast from accumulation type)
// Uses vectorized half2 -> fp8x2 when available for SIMD performance
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 1> castPack(EltPack<half, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  union Fp8E5m2Access1 {
    EltPack<__nv_fp8_e5m2, 1> pack;
    __nv_fp8_storage_t storage;
  };
  Fp8E5m2Access1 out;
  HalfRawAccess h;
  h.val = x.elts()[0];
  out.storage = __nv_cvt_halfraw_to_fp8(h.raw, __NV_SATFINITE, __NV_E5M2);
  return out.pack;
}

template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp8_e5m2, 2> castPack(EltPack<half, 2> x) {
  #if __CUDA_ARCH__ >= 900
    union Half2RawAccess {
      __half2_raw raw;
      half2 val;
    };
    union Half2PackAccess {
      EltPack<half, 2> pack;
      half2 pair;
    };
    Half2PackAccess in;
    union Fp8E5m2Access2 {
      EltPack<__nv_fp8_e5m2, 2> pack;
      __nv_fp8x2_storage_t storage2;
    };
    Fp8E5m2Access2 out;
    in.pack = x;
    Half2RawAccess h2;
    h2.val = in.pair;
    out.storage2 = __nv_cvt_halfraw2_to_fp8x2(h2.raw, __NV_SATFINITE, __NV_E5M2);
    return out.pack;
  #else
    PackAccess<half, 2> in;
    PackAccess<__nv_fp8_e5m2, 2> out;
    in.pack = x;
    out.lo = castPack<__nv_fp8_e5m2>(in.lo);
    out.hi = castPack<__nv_fp8_e5m2>(in.hi);
    return out.pack;
  #endif
}

#endif

#if defined(__CUDA_FP4_TYPES_EXIST__)
template<int n>
struct Fp4E2m1Access {
  union {
    EltPack<__nv_fp4_e2m1, n> pack;
    __nv_fp4_e2m1 elts[n];
    __nv_fp4_storage_t storage[n];
    __nv_fp4x2_e2m1 pairs[n / 2];
    __nv_fp4x2_storage_t storage2[n / 2];
  };
};

template<>
struct Fp4E2m1Access<1> {
  union {
    EltPack<__nv_fp4_e2m1, 1> pack;
    __nv_fp4_e2m1 elts[1];
    __nv_fp4_storage_t storage[1];
  };
  static constexpr __nv_fp4x2_e2m1* pairs = nullptr;
  static constexpr __nv_fp4x2_storage_t* storage2 = nullptr;
};

// Specialization for __nv_fp4_e2m1 -> half conversion (upcast to accumulation type)
// Uses CUDA's vectorized fp4x2 types and conversion intrinsics for better SIMD performance
// Note: fp4 is 4 bits, so 2 fp4 values are packed per byte
template<>
NCCL_DEVICE_INLINE EltPack<half, 1> castPack(EltPack<__nv_fp4_e2m1, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  Fp4E2m1Access<1> in;
  EltPack<half, 1> out{};
  in.pack = x;
  HalfRawAccess h;
  h.raw = __nv_cvt_fp4_to_halfraw(in.storage[0], __NV_E2M1);
  out.elts()[0] = h.val;
  return out;
}

template<>
NCCL_DEVICE_INLINE EltPack<half, 2> castPack(EltPack<__nv_fp4_e2m1, 2> x) {
  union Half2RawAccess {
    __half2_raw raw;
    half2 val;
  };
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Fp4E2m1Access<2> in;
  Half2PackAccess out;
  in.pack = x;
  Half2RawAccess h2;
  h2.raw = __nv_cvt_fp4x2_to_halfraw2(in.storage2[0], __NV_E2M1);
  out.pair = h2.val;
  return out.pack;
}

// Specialization for half -> __nv_fp4_e2m1 conversion (downcast from accumulation type)
// Uses CUDA's vectorized fp4x2 types and conversion intrinsics for better SIMD performance
// Note: fp4 is 4 bits, so 2 fp4 values are packed per byte
template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp4_e2m1, 1> castPack(EltPack<half, 1> x) {
  union HalfRawAccess {
    __half_raw raw;
    half val;
  };
  Fp4E2m1Access<1> out;
  HalfRawAccess h;
  h.val = x.elts()[0];
  out.storage[0] = __nv_cvt_halfraw_to_fp4(h.raw, __NV_E2M1, cudaRoundNearest);
  return out.pack;
}

template<>
NCCL_DEVICE_INLINE EltPack<__nv_fp4_e2m1, 2> castPack(EltPack<half, 2> x) {
  union Half2RawAccess {
    __half2_raw raw;
    half2 val;
  };
  union Half2PackAccess {
    EltPack<half, 2> pack;
    half2 pair;
  };
  Half2PackAccess in;
  Fp4E2m1Access<2> out;
  in.pack = x;
  Half2RawAccess h2;
  h2.val = in.pair;
  out.storage2[0] = __nv_cvt_halfraw2_to_fp4x2(h2.raw, __NV_E2M1, cudaRoundNearest);
  return out.pack;
}
#endif



// ============================================================================
// ReducePack Base and Specializations
// ============================================================================

// Reduce pack using reduction operator
// Works with EltPack types
template<template<typename> typename Red, typename T, int n>
NCCL_DEVICE_INLINE EltPack<T, n> reducePack(Red<T> red, EltPack<T, n> a, EltPack<T, n> b) {
  static_assert((n & (n - 1)) == 0, "EltPack requires power-of-two element count");

  PackAccess<T, n> aa;
  PackAccess<T, n> bb;
  PackAccess<T, n> out;
  aa.pack = a;
  bb.pack = b;
  if NCCL_IF_CONSTEXPR (n == 1) {
    out.pack.elts()[0] = red(aa.pack.elts()[0], bb.pack.elts()[0]);
  } else {
    out.lo = reducePack(red, aa.lo, bb.lo);
    out.hi = reducePack(red, aa.hi, bb.hi);
  }
  return out.pack;
}

// Specialization for zero-sized packs
template<template<typename> typename Red, typename T>
NCCL_DEVICE_INLINE EltPack<T, 0> reducePack(Red<T> /* red */, EltPack<T, 0> /* a */, EltPack<T, 0> /* b */) {
  EltPack<T, 0> result{};
  return result;
}

// Specialization for int8_t with OpSum - uses __vadd4 SIMD intrinsic for performance
// Processes EltPack<int8_t, 4> as unsigned int chunks
// Note: __vadd4 is only valid for sum reduction, so this specialization is OpSum-specific
template<>
NCCL_DEVICE_INLINE EltPack<int8_t, 4> reducePack(OpSum<int8_t> /* red */, EltPack<int8_t, 4> a, EltPack<int8_t, 4> b) {
  union Int8PackAccess4 {
    EltPack<int8_t, 4> pack;
    unsigned int word;
  };
  Int8PackAccess4 aa;
  Int8PackAccess4 bb;
  Int8PackAccess4 out;
  aa.pack = a;
  bb.pack = b;
  out.word = __vadd4(aa.word, bb.word);
  return out.pack;
}

// Specialization for uint8_t with OpSum - reuses int8_t implementation via union access
template<int n>
NCCL_DEVICE_INLINE EltPack<uint8_t, n> reducePack(OpSum<uint8_t> /* red */, EltPack<uint8_t, n> a, EltPack<uint8_t, n> b) {
  static_assert((n & (n - 1)) == 0, "EltPack<uint8_t, n> requires power-of-two element count");
  union PackU8 {
    EltPack<uint8_t, n> u;
    EltPack<int8_t, n> s;
  };
  PackU8 aa;
  PackU8 bb;
  aa.u = a;
  bb.u = b;
  OpSum<int8_t> intRed{};
  PackU8 out;
  out.s = reducePack(intRed, aa.s, bb.s);
  return out.u;
}

// Specialization for half with OpSum - uses __hadd2 SIMD intrinsic for performance
// Architecture check: __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
template<>
NCCL_DEVICE_INLINE EltPack<half, 2> reducePack(OpSum<half> /* red */, EltPack<half, 2> a, EltPack<half, 2> b) {
  #if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    union Half2PackAccess {
      EltPack<half, 2> pack;
      half2 pair;
    };
    Half2PackAccess aa;
    Half2PackAccess bb;
    Half2PackAccess out;
    aa.pack = a;
    bb.pack = b;
    out.pair = __hadd2(aa.pair, bb.pair);
    return out.pack;
  #else
    EltPack<half, 2> out{};
    OpSum<half> red{};
    out.elts()[0] = red(a.elts()[0], b.elts()[0]);
    out.elts()[1] = red(a.elts()[1], b.elts()[1]);
    return out;
  #endif
}

#if defined(__CUDA_BF16_TYPES_EXIST__)
// Specialization for __nv_bfloat16 with OpSum - uses __hadd2 SIMD intrinsic
// Architecture check: __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
template<>
NCCL_DEVICE_INLINE EltPack<__nv_bfloat16, 2> reducePack(OpSum<__nv_bfloat16> /* red */, EltPack<__nv_bfloat16, 2> a, EltPack<__nv_bfloat16, 2> b) {
  #if __CUDA_ARCH__ >= 530 && __CUDA_ARCH__ != 610
    union Bf16PackAccess2 {
      EltPack<__nv_bfloat16, 2> pack;
      half2 pair;
    };
    Bf16PackAccess2 aa;
    Bf16PackAccess2 bb;
    Bf16PackAccess2 out;
    aa.pack = a;
    bb.pack = b;
    out.pair = __hadd2(aa.pair, bb.pair);
    return out.pack;
  #else
    EltPack<__nv_bfloat16, 2> out{};
    OpSum<__nv_bfloat16> red{};
    out.elts()[0] = red(a.elts()[0], b.elts()[0]);
    out.elts()[1] = red(a.elts()[1], b.elts()[1]);
    return out;
  #endif
}
#endif

} // namespace utility
} // namespace nccl

#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_VECTOR__FUNCS_H_

