/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_PROXY_ATOMICS_API_H_
#define NIIN_GPUNETIO_PROXY_ATOMICS_API_H_

#include "niin/gpunetio/proxy_atomics/protocol.h"

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__CUDACC__)
#define NIIN_GPUNETIO_PROXY_HOST_DEVICE __host__ __device__
#else
#define NIIN_GPUNETIO_PROXY_HOST_DEVICE
#endif

namespace niin {
namespace gpunetio {
namespace proxy {

// Map public C++ values to fixed-width protocol representations. Integral
// aliases (including ptrdiff_t) are represented only by their actual width,
// never by the spelling used in an application header.
template <typename T>
struct AtomicValueTraits {
  static constexpr bool kIsIntegral = std::is_integral<T>::value;
  static constexpr bool kSupported = kIsIntegral && (sizeof(T) == 4 || sizeof(T) == 8);
  static constexpr uint8_t kWireType =
      !kIsIntegral ? UINT8_MAX
                   : (sizeof(T) == 4 ? NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL32
                                     : (sizeof(T) == 8 ? NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL64
                                                       : UINT8_MAX));
  static constexpr size_t kBytes = kSupported ? sizeof(T) : 0;
  static constexpr bool kHasBaseRawOps = kSupported;
  static constexpr bool kHasAddOps = kSupported;
};

template <>
struct AtomicValueTraits<float> {
  static constexpr bool kIsIntegral = false;
  static constexpr bool kSupported = true;
  static constexpr uint8_t kWireType = NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT32;
  static constexpr size_t kBytes = sizeof(float);
  static constexpr bool kHasBaseRawOps = true;
  static constexpr bool kHasAddOps = true;
};

template <>
struct AtomicValueTraits<double> {
  static constexpr bool kIsIntegral = false;
  static constexpr bool kSupported = true;
  static constexpr uint8_t kWireType = NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT64;
  static constexpr size_t kBytes = sizeof(double);
  static constexpr bool kHasBaseRawOps = true;
  static constexpr bool kHasAddOps = true;
};

template <>
struct AtomicValueTraits<__half> {
  static constexpr bool kIsIntegral = false;
  static constexpr bool kSupported = true;
  static constexpr uint8_t kWireType = NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT16;
  static constexpr size_t kBytes = sizeof(__half);
  static constexpr bool kHasBaseRawOps = false;
  static constexpr bool kHasAddOps = true;
};

static_assert(sizeof(float) == 4 && sizeof(double) == 8,
              "NIIN proxy requires binary32/binary64 storage sizes");
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559,
              "NIIN proxy requires IEC 559 float/double semantics");
static_assert(sizeof(__half) == 2, "NIIN proxy requires binary16 storage");
static_assert(sizeof(ptrdiff_t) == 4 || sizeof(ptrdiff_t) == 8,
              "NIIN proxy supports only 32- or 64-bit ptrdiff_t");

// This intentionally narrows the generic transport grammar to NVSHMEM's
// public surface: floating values support raw fetch/swap/set and the six
// nvshmemx add/fetch-add extensions; half is extension-only.
template <typename T>
NIIN_GPUNETIO_PROXY_HOST_DEVICE constexpr bool isPublicAtomicOperation(uint8_t op) {
  using Traits = AtomicValueTraits<T>;
  if constexpr (!Traits::kSupported) {
    return false;
  } else if constexpr (Traits::kIsIntegral) {
    return niinGpunetioProxyAtomicOpSupportsType(op, Traits::kWireType);
  } else if constexpr (std::is_same<T, __half>::value) {
    return op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD ||
           op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD;
  } else {
    return op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD ||
           op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD ||
           op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP ||
           op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH ||
           op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET;
  }
}

template <typename T>
NIIN_GPUNETIO_PROXY_HOST_DEVICE inline uint64_t atomicValueBits(T value) {
  static_assert(AtomicValueTraits<T>::kSupported,
                "NIIN proxy AMOs require 4/8-byte integral, float, double, or __half values");
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(T));
  return bits;
}

template <typename T>
NIIN_GPUNETIO_PROXY_HOST_DEVICE inline T atomicValueFromBits(uint64_t bits) {
  static_assert(AtomicValueTraits<T>::kSupported,
                "NIIN proxy AMOs require 4/8-byte integral, float, double, or __half values");
  T value;
  memcpy(&value, &bits, sizeof(T));
  return value;
}

}  // namespace proxy
}  // namespace gpunetio
}  // namespace niin

#undef NIIN_GPUNETIO_PROXY_HOST_DEVICE

#endif  // NIIN_GPUNETIO_PROXY_ATOMICS_API_H_
