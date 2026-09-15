/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_PROXY_ATOMICS_PROTOCOL_H_
#define NIIN_GPUNETIO_PROXY_ATOMICS_PROTOCOL_H_

// Wire ABI for NIIN's optional software-atomic sidecar.
//
// This is deliberately independent of GPUNetIO and GIN. A GPU producer
// publishes a request into CUDA-mapped host memory; a NIIN-owned host proxy
// SENDs it on a raw-verbs RC QP to a target-owned SRQ. The target proxy
// performs a CPU-mapped RMW and writes a completion body followed by its
// ticket to the source completion ring.

#include <cstddef>
#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE __host__ __device__
#else
#define NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE
#endif

#define NIIN_GPUNETIO_PROXY_ATOMIC_PROTOCOL_MAGIC UINT32_C(0x4e495041)  // "NIPA"
#define NIIN_GPUNETIO_PROXY_ATOMIC_PROTOCOL_VERSION UINT16_C(1)
#define NIIN_GPUNETIO_PROXY_ATOMIC_UNPUBLISHED_TICKET UINT64_C(0)

// The two modes are mutually exclusive. ProxyAll routes self, LSA, and
// network AMOs through the CPU RMW service, avoiding a split linearization
// domain between CPU RMW and NIC AMOs.
enum niinGpunetioProxyAtomicExecutionMode : uint32_t {
  NIIN_GPUNETIO_PROXY_ATOMIC_EXECUTION_NATIVE_ONLY = 0,
  NIIN_GPUNETIO_PROXY_ATOMIC_EXECUTION_PROXY_ALL = 1,
};

enum niinGpunetioProxyAtomicOp : uint8_t {
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD = 0,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD = 1,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_COMPARE_SWAP = 2,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP = 3,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH = 4,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET = 5,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_INC = 6,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_INC = 7,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_AND = 8,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_AND = 9,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_OR = 10,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_OR = 11,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_XOR = 12,
  NIIN_GPUNETIO_PROXY_ATOMIC_OP_XOR = 13,
};

// Integral values are width-only: signed and unsigned aliases have identical
// raw representations. Floating values use IEEE raw bits.
enum niinGpunetioProxyAtomicType : uint8_t {
  NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL32 = 0,
  NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL64 = 1,
  NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT16 = 2,
  NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT32 = 3,
  NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT64 = 4,
};

enum niinGpunetioProxyAtomicCompletionStatus : uint32_t {
  NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_SUCCESS = 0,
  NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_BAD_REQUEST = 1,
  NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_BAD_ADDRESS = 2,
  NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_UNSUPPORTED = 3,
  NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_INTERNAL_ERROR = 4,
};

enum niinGpunetioProxyAtomicRequestFlags : uint32_t {
  NIIN_GPUNETIO_PROXY_ATOMIC_REQUEST_FLAG_NONE = 0,
};

// `ticket` is physically last and is published last by the GPU producer after
// a system-scope release. The target receives this fixed-width payload through
// its SRQ; it contains no host pointers, size_t values, or provider state.
struct alignas(uint64_t) niinGpunetioProxyAtomicRequest {
  uint32_t magic;
  uint16_t version;
  uint8_t op;
  uint8_t type;
  uint32_t sourcePe;
  uint32_t targetPe;
  uint32_t sourceSlot;
  uint32_t flags;
  uint64_t remoteOffset;
  uint64_t operandBits;
  uint64_t compareBits;
  uint64_t ticket;
};

// The target writes bytes [0, ticket) first, then ticket in a second ordered
// RC RDMA write. A GPU accepts a completion only after observing its ticket.
struct alignas(uint64_t) niinGpunetioProxyAtomicCompletion {
  uint64_t resultBits;
  uint32_t status;
  uint32_t reserved;
  uint64_t ticket;
};

static_assert(std::is_standard_layout<niinGpunetioProxyAtomicRequest>::value,
              "NIIN proxy request must remain a plain wire record");
static_assert(std::is_trivially_copyable<niinGpunetioProxyAtomicRequest>::value,
              "NIIN proxy request must be sent without serialization");
static_assert(sizeof(niinGpunetioProxyAtomicRequest) == 56,
              "NIIN proxy request ABI changed");
static_assert(offsetof(niinGpunetioProxyAtomicRequest, ticket) == 48,
              "NIIN proxy request ticket must remain the final wire field");
static_assert(std::is_standard_layout<niinGpunetioProxyAtomicCompletion>::value,
              "NIIN proxy completion must remain a plain wire record");
static_assert(std::is_trivially_copyable<niinGpunetioProxyAtomicCompletion>::value,
              "NIIN proxy completion must be written without serialization");
static_assert(sizeof(niinGpunetioProxyAtomicCompletion) == 24,
              "NIIN proxy completion ABI changed");
static_assert(offsetof(niinGpunetioProxyAtomicCompletion, ticket) == 16,
              "NIIN proxy completion ticket must remain the final wire field");

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr size_t
niinGpunetioProxyAtomicCompletionBodyBytes() {
  return offsetof(niinGpunetioProxyAtomicCompletion, ticket);
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicIsKnownOp(uint8_t op) {
  return op <= NIIN_GPUNETIO_PROXY_ATOMIC_OP_XOR;
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicIsKnownType(uint8_t type) {
  return type <= NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT64;
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicTypeIsIntegral(uint8_t type) {
  return type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL32 ||
         type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL64;
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicTypeIsFloating(uint8_t type) {
  return type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT16 ||
         type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT32 ||
         type == NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT64;
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr size_t
niinGpunetioProxyAtomicElementBytes(uint8_t type) {
  switch (type) {
    case NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL32:
    case NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT32:
      return 4;
    case NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_INTEGRAL64:
    case NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT64:
      return 8;
    case NIIN_GPUNETIO_PROXY_ATOMIC_TYPE_FLOAT16:
      return 2;
    default:
      return 0;
  }
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicOpReturnsPrevious(uint8_t op) {
  switch (op) {
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_COMPARE_SWAP:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_INC:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_AND:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_OR:
    case NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_XOR:
      return true;
    default:
      return false;
  }
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicOpIsBitwise(uint8_t op) {
  return op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_AND ||
         op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_AND ||
         op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_OR ||
         op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_OR ||
         op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_XOR ||
         op == NIIN_GPUNETIO_PROXY_ATOMIC_OP_XOR;
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool
niinGpunetioProxyAtomicOpSupportsType(uint8_t op, uint8_t type) {
  if (!niinGpunetioProxyAtomicIsKnownOp(op) || !niinGpunetioProxyAtomicIsKnownType(type))
    return false;
  if (niinGpunetioProxyAtomicTypeIsIntegral(type)) return true;
  // The wire can represent raw float/double/half fetch/swap/set and floating
  // add/fetch-add. The public adapter narrows this to actual NVSHMEM spellings.
  return !niinGpunetioProxyAtomicOpIsBitwise(op) &&
         op != NIIN_GPUNETIO_PROXY_ATOMIC_OP_INC &&
         op != NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_INC &&
         op != NIIN_GPUNETIO_PROXY_ATOMIC_OP_COMPARE_SWAP;
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr bool niinGpunetioProxyAtomicRequestIsWellFormed(
    const niinGpunetioProxyAtomicRequest& request) {
  const size_t elementBytes = niinGpunetioProxyAtomicElementBytes(request.type);
  return request.magic == NIIN_GPUNETIO_PROXY_ATOMIC_PROTOCOL_MAGIC &&
         request.version == NIIN_GPUNETIO_PROXY_ATOMIC_PROTOCOL_VERSION &&
         request.flags == NIIN_GPUNETIO_PROXY_ATOMIC_REQUEST_FLAG_NONE && request.ticket != 0 &&
         elementBytes != 0 && (request.remoteOffset % elementBytes) == 0 &&
         niinGpunetioProxyAtomicOpSupportsType(request.op, request.type);
}

NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE constexpr uint64_t
niinGpunetioProxyAtomicNextTicket(uint64_t previous) {
  return previous == UINT64_MAX ? UINT64_C(1) : previous + UINT64_C(1);
}

#undef NIIN_GPUNETIO_PROXY_PROTOCOL_HOST_DEVICE

#endif  // NIIN_GPUNETIO_PROXY_ATOMICS_PROTOCOL_H_
