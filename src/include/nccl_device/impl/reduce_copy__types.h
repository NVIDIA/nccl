/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
#define _NCCL_DEVICE_REDUCE_COPY__TYPES_H_

#include "vector__types.h"
#include "../utility.h"
#include "../coop.h"
#include <type_traits>

namespace nccl {
namespace utility {

// Reduction Operators

template <typename T>
struct OpSum {
  using EltType = T;
  NCCL_DEVICE_INLINE T operator()(const T& a, const T& b) const {
    return a + b;
  }
};

// Helper trait to create accumulator reduction operator from RedOp
// Maps RedOp (e.g., OpSum<T>) to accumulator reduction operator (e.g., OpSum<AccEltType>)
template<typename RedOp, typename AccEltType>
struct AccRedOp {
  // Default: try to extract template and reapply
  // For template template parameters like OpSum<T>, we need to extract OpSum and apply to AccEltType
  // This is a simplified approach - assumes RedOp follows pattern OpXXX<T>
  using Type = OpSum<AccEltType>;  // Fallback to OpSum
};

// Specialization for OpSum<T> -> OpSum<AccEltType>
template<typename T, typename AccEltType>
struct AccRedOp<OpSum<T>, AccEltType> {
  using Type = OpSum<AccEltType>;
};

// Cooperation Level Helpers for compile-time stride resolution
template <typename Coop>
struct CoopStride {
  // Default: runtime determined (use sentinel 0 to indicate runtime)
  static constexpr int value = 0;
};

#if NCCL_CHECK_CUDACC
// Specialization for warp: always 32
template <>
struct CoopStride<ncclCoopWarp> {
  static constexpr int value = 32;
};

// Specialization for CTA: use 32 for warp coalescing
template <>
struct CoopStride<ncclCoopCta> {
  static constexpr int value = 32;
};

// Specialization for thread: use 1
template <>
struct CoopStride<ncclCoopThread> {
  static constexpr int value = 1;
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
// Specialization for FP8 types - convert to half, add, convert back
template<>
struct OpSum<__nv_fp8_e4m3> {
  using EltType = __nv_fp8_e4m3;
  NCCL_DEVICE_INLINE __nv_fp8_e4m3 operator()(const __nv_fp8_e4m3& a, const __nv_fp8_e4m3& b) const {
    #if __CUDA_ARCH__ >= 800
      // Use native half addition on architectures that support it
      return __nv_fp8_e4m3(__hadd(__half(a), __half(b)));
    #else
      // Fallback: convert to float, add, convert back
      return __nv_fp8_e4m3(float(a) + float(b));
    #endif
  }
};

template<>
struct OpSum<__nv_fp8_e5m2> {
  using EltType = __nv_fp8_e5m2;
  NCCL_DEVICE_INLINE __nv_fp8_e5m2 operator()(const __nv_fp8_e5m2& a, const __nv_fp8_e5m2& b) const {
    #if __CUDA_ARCH__ >= 800
      // Use native half addition on architectures that support it
      return __nv_fp8_e5m2(__hadd(__half(a), __half(b)));
    #else
      // Fallback: convert to float, add, convert back
      return __nv_fp8_e5m2(float(a) + float(b));
    #endif
  }
};
#endif

#if defined(__CUDA_FP4_TYPES_EXIST__)
// Specialization for FP4 type - convert to half, add, convert back (LSA only)
template<>
struct OpSum<__nv_fp4_e2m1> {
  using EltType = __nv_fp4_e2m1;
  NCCL_DEVICE_INLINE __nv_fp4_e2m1 operator()(const __nv_fp4_e2m1& a, const __nv_fp4_e2m1& b) const {
    #if __CUDA_ARCH__ >= 800
      // Use native half addition on architectures that support it
      return __nv_fp4_e2m1(__hadd(__half(a), __half(b)));
    #else
      // Fallback: convert to float, add, convert back
      return __nv_fp4_e2m1(float(a) + float(b));
    #endif
  }
};
#endif

} // namespace utility
} // namespace nccl

#endif // _NCCL_DEVICE_REDUCE_COPY__TYPES_H_
