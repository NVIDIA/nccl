/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_VECTOR__TYPES_H_
#define _NCCL_DEVICE_VECTOR__TYPES_H_

#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <algorithm>

// Forward declaration for sum reduction operator (defined in reduce_copy__types.h)
namespace nccl {
namespace utility {
template<typename T> struct OpSum;
} // namespace utility
} // namespace nccl

namespace nccl {
namespace utility {

// ============================================================================
// Typed pack types
// ============================================================================
// EltPack<T, n>: Typed pack containing n elements of type T (T must be at least 1 byte).
// Provides both typed element access and untyped byte access for load/store.

template<typename T, int n>
struct EltPack {
  using EltType = T;  // Element type
  static constexpr int Count = n;  // Number of elements
  static constexpr int Bytes = n * static_cast<int>(sizeof(T));
  // Impose most generous alignment possible (greatest pow2 factor)
  static constexpr int Alignment = (Bytes & -Bytes);
  alignas(Alignment) char bytes[Bytes];

  // Element access via reinterpret_cast
  NCCL_DEVICE_INLINE T* elts() { return reinterpret_cast<T*>(bytes); }
  NCCL_DEVICE_INLINE const T* elts() const { return reinterpret_cast<const T*>(bytes); }
};

// Specialization for zero-sized packs
template<typename T>
struct EltPack<T, 0> {
  using EltType = T;  // Element type
  static constexpr int Count = 0;
  static constexpr int Bytes = 0;
  static constexpr int Alignment = 1;
  static constexpr char* bytes = nullptr;

  NCCL_DEVICE_INLINE T* elts() { return nullptr; }
  NCCL_DEVICE_INLINE const T* elts() const { return nullptr; }
};



// Helper: Create EltPack for a given byte size
// Computes the number of elements that fit in the specified byte size (element size >= 1 byte)
template<typename T, int Bytes>
using EltPackForBytes = EltPack<T, Bytes / static_cast<int>(sizeof(T))>;

// ============================================================================
// Accumulation type determination
// ============================================================================
// AccumulateType<Red>: Maps reduction operators to their accumulation type.
// Red combines the element type (scalar, not pack) and operation.
// The accumulation type depends on both the element type and the operation
// (e.g., min/max don't need wider types, but sum may benefit from wider types).

// Primary template - extracts element type from reduction operator
template<typename Red>
struct AccumulateType {
  using Type = typename Red::EltType;  // Default: use operator's element type
};

// Partial specialization for template template parameters (e.g., OpSum<T>)
// For most operators, accumulation type equals element type
template<template<typename> typename Red, typename T>
struct AccumulateType<Red<T>> {
  using Type = T;  // Default: same type
};

// Specialize for sum operations that benefit from wider accumulation
template<>
struct AccumulateType<OpSum<half>> {
  using Type = float;  // half accumulates into float for better precision
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template<>
struct AccumulateType<OpSum<__nv_bfloat16>> {
  using Type = float;  // bfloat16 accumulates into float for better precision
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template<>
struct AccumulateType<OpSum<__nv_fp8_e4m3>> {
  using Type = half;  // fp8 accumulates into half precision (matches .acc::f16 in multimem)
};
template<>
struct AccumulateType<OpSum<__nv_fp8_e5m2>> {
  using Type = half;  // fp8 accumulates into half precision (matches .acc::f16 in multimem)
};
#endif

// ============================================================================
// MinMultimemType: Maps scalar types to their minimum multimem-compatible type
// ============================================================================
// For types with multimem resolution requirements, this defines the smallest unit
// that can be used in multimem operations. Default is the type itself.
template <typename T>
struct MinMultimemType {
  using Type = T;  // Default: type itself
};

template <>
struct MinMultimemType<half> {
  using Type = EltPack<half, 2>;  // Minimum multimem type for half is EltPack<half, 2> (32 bits)
};

#if defined(__CUDA_BF16_TYPES_EXIST__)
template <>
struct MinMultimemType<__nv_bfloat16> {
  using Type = EltPack<__nv_bfloat16, 2>;  // Minimum multimem type for bfloat16 is EltPack<__nv_bfloat16, 2> (32 bits)
};
#endif

#if defined(__CUDA_FP8_TYPES_EXIST__)
template <>
struct MinMultimemType<__nv_fp8_e4m3> {
  using Type = EltPack<__nv_fp8_e4m3, 4>;  // Minimum multimem type for fp8_e4m3 is EltPack<__nv_fp8_e4m3, 4> (32 bits)
};
template <>
struct MinMultimemType<__nv_fp8_e5m2> {
  using Type = EltPack<__nv_fp8_e5m2, 4>;  // Minimum multimem type for fp8_e5m2 is EltPack<__nv_fp8_e5m2, 4> (32 bits)
};
#endif

} // namespace utility
} // namespace nccl

#endif // _NCCL_DEVICE_VECTOR__TYPES_H_
