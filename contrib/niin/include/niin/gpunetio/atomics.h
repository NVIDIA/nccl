/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_ATOMICS_H_
#define NIIN_GPUNETIO_ATOMICS_H_

#include "niin/gpunetio/device.h"
#include "niin/gpunetio/proxy_atomics/device.h"

#include <cstring>
#include <type_traits>

// All currently native NIIN integer AMOs.  The SRQ fallback owns operations
// which cannot be represented by GPUNetIO atomics (for example float add).
enum class niinGpunetioAtomicOp : uint8_t {
  FetchAdd,
  Add,
  CompareSwap,
  Swap,
  Fetch,
  Set,
  Inc,
  FetchInc,
  FetchAnd,
  And,
  FetchOr,
  Or,
  FetchXor,
  Xor,
};

// Keep the direct-QP opcode namespace private to that path, but map it
// explicitly to NIIN's proxy wire enum when a bound ProxyAll provider owns
// the whole AMO linearization domain. Do not rely on matching enum ordinals.
__host__ __device__ constexpr enum niinGpunetioProxyAtomicOp niin_gpunetio_proxy_op(
    enum niinGpunetioAtomicOp op) {
  switch (op) {
    case niinGpunetioAtomicOp::FetchAdd: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_ADD;
    case niinGpunetioAtomicOp::Add: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_ADD;
    case niinGpunetioAtomicOp::CompareSwap: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_COMPARE_SWAP;
    case niinGpunetioAtomicOp::Swap: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_SWAP;
    case niinGpunetioAtomicOp::Fetch: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH;
    case niinGpunetioAtomicOp::Set: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_SET;
    case niinGpunetioAtomicOp::Inc: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_INC;
    case niinGpunetioAtomicOp::FetchInc: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_INC;
    case niinGpunetioAtomicOp::FetchAnd: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_AND;
    case niinGpunetioAtomicOp::And: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_AND;
    case niinGpunetioAtomicOp::FetchOr: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_OR;
    case niinGpunetioAtomicOp::Or: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_OR;
    case niinGpunetioAtomicOp::FetchXor: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH_XOR;
    case niinGpunetioAtomicOp::Xor: return NIIN_GPUNETIO_PROXY_ATOMIC_OP_XOR;
  }
  return NIIN_GPUNETIO_PROXY_ATOMIC_OP_FETCH;
}

#if NIIN_GPUNETIO_HAS_DEVICE_API

namespace niin {
namespace gpunetio {
namespace detail {

template <typename T>
__device__ __forceinline__ uint64_t atomicBits(T value) {
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "GPUNetIO atomics require 4- or 8-byte operands");
  uint64_t bits = 0;
  memcpy(&bits, &value, sizeof(T));
  return bits;
}

template <typename T>
__device__ __forceinline__ T atomicValue(uint64_t bits) {
  T value;
  memcpy(&value, &bits, sizeof(T));
  return value;
}

template <typename T>
__device__ __forceinline__ bool getResponseSlot(const niinGpunetioAtomicContext* context, int pe,
                                                size_t* slotOffset) {
  // One local result slot belongs to every destination PE.  The endpoint lock
  // below holds that slot from WQE construction through completion, so callers
  // do not need to allocate or manage response-buffer tickets themselves.
  if (context == nullptr || context->responseBase == nullptr || context->endpointLocks == nullptr ||
      context->nPes <= 0 || context->responseStride < sizeof(uint64_t))
    return false;

  const size_t nPes = static_cast<size_t>(context->nPes);
  if (pe < 0 || pe >= context->nPes || nPes > context->responseBytes / context->responseStride) return false;

  const size_t offset = static_cast<size_t>(pe) * context->responseStride;
  if (offset > context->responseBytes || sizeof(T) > context->responseBytes - offset) return false;

  const uintptr_t base = reinterpret_cast<uintptr_t>(context->responseBase);
  const uintptr_t address = base + offset;
  if (address < base || (address & (alignof(T) - 1)) != 0) return false;

  *slotOffset = offset;
  return true;
}

template <typename T>
__device__ __forceinline__ bool directEndpointAvailable(const niinGpunetioAtomicContext* context, int pe) {
  constexpr uint32_t commonFlags = NIIN_GPUNETIO_ATOMIC_CONTEXT_READY |
                                   NIIN_GPUNETIO_ATOMIC_CONTEXT_DIRECT |
                                   NIIN_GPUNETIO_ATOMIC_CONTEXT_EXTENDED;
  constexpr uint32_t endpointFlags = NIIN_GPUNETIO_ATOMIC_ENDPOINT_CONNECTED |
                                     NIIN_GPUNETIO_ATOMIC_ENDPOINT_EXTENDED;
  if (context == nullptr || context->version != NIIN_GPUNETIO_ATOMIC_CONTEXT_VERSION ||
      (context->flags & commonFlags) != commonFlags || context->endpoints == nullptr || pe < 0 ||
      pe >= context->nPes)
    return false;

  const niinGpunetioAtomicEndpoint& endpoint = context->endpoints[pe];
  uint32_t requiredEndpointFlags = endpointFlags |
                                   (sizeof(T) == 4 ? NIIN_GPUNETIO_ATOMIC_ENDPOINT_ATOMIC_4B
                                                   : NIIN_GPUNETIO_ATOMIC_ENDPOINT_ATOMIC_8B);
  return endpoint.deviceQp != nullptr && (endpoint.flags & requiredEndpointFlags) == requiredEndpointFlags;
}

template <typename T>
__device__ __forceinline__ bool atomicConsumesTwoWqebbs(niinGpunetioAtomicOp op) {
  // A masked 64-bit compare-and-swap WQE occupies two WQEBBs.  The set/swap
  // and bitwise AND/OR operations use that encoding.  It has no control
  // segment in the second WQEBB, so a final signaled NOP supplies the CQE that
  // makes the whole chain visible to the caller.
  if constexpr (sizeof(T) != 8) {
    return false;
  } else {
    return op == niinGpunetioAtomicOp::Swap || op == niinGpunetioAtomicOp::Set ||
           op == niinGpunetioAtomicOp::FetchAnd || op == niinGpunetioAtomicOp::And ||
           op == niinGpunetioAtomicOp::FetchOr || op == niinGpunetioAtomicOp::Or;
  }
}

__device__ __forceinline__ bool validCstScratch(const niinGpunetioAtomicContext* context) {
  return context != nullptr && context->cstScratchOffset >= sizeof(uint64_t) &&
         context->responseStride >= sizeof(uint8_t) &&
         context->cstScratchOffset <= context->responseStride - sizeof(uint8_t);
}

__device__ __forceinline__ bool atomicReturnsPrevious(niinGpunetioAtomicOp op) {
  switch (op) {
    case niinGpunetioAtomicOp::FetchAdd:
    case niinGpunetioAtomicOp::CompareSwap:
    case niinGpunetioAtomicOp::Swap:
    case niinGpunetioAtomicOp::Fetch:
    case niinGpunetioAtomicOp::FetchInc:
    case niinGpunetioAtomicOp::FetchAnd:
    case niinGpunetioAtomicOp::FetchOr:
    case niinGpunetioAtomicOp::FetchXor:
      return true;
    case niinGpunetioAtomicOp::Add:
    case niinGpunetioAtomicOp::Set:
    case niinGpunetioAtomicOp::Inc:
    case niinGpunetioAtomicOp::And:
    case niinGpunetioAtomicOp::Or:
    case niinGpunetioAtomicOp::Xor:
      return false;
  }
  return false;
}

__device__ __forceinline__ bool lockEndpoint(int* endpointLock) {
  // `0` is available, `1` is held, and a negative value permanently poisons
  // the endpoint after a CQ error.  The latter prevents a second AMO from
  // submitting to a QP whose first error has already been observed.
  for (;;) {
    const int previous = atomicCAS(endpointLock, 0, 1);
    if (previous == 0) return true;
    if (previous < 0) return false;
  }
}

__device__ __forceinline__ void unlockEndpoint(int* endpointLock) {
  atomicExch(endpointLock, 0);
}

__device__ __forceinline__ void failEndpoint(int* endpointLock) {
  atomicExch(endpointLock, -1);
}

__device__ __forceinline__ bool isNativeOp(niinGpunetioAtomicOp op) {
  switch (op) {
    case niinGpunetioAtomicOp::FetchAdd:
    case niinGpunetioAtomicOp::Add:
    case niinGpunetioAtomicOp::CompareSwap:
    case niinGpunetioAtomicOp::Swap:
    case niinGpunetioAtomicOp::Fetch:
    case niinGpunetioAtomicOp::Set:
    case niinGpunetioAtomicOp::Inc:
    case niinGpunetioAtomicOp::FetchInc:
    case niinGpunetioAtomicOp::FetchAnd:
    case niinGpunetioAtomicOp::And:
    case niinGpunetioAtomicOp::FetchOr:
    case niinGpunetioAtomicOp::Or:
    case niinGpunetioAtomicOp::FetchXor:
    case niinGpunetioAtomicOp::Xor:
      return true;
  }
  return false;
}

template <typename T>
__device__ __forceinline__ void prepareAtomicWqe(
    struct doca_gpu_dev_verbs_qp* qp, struct doca_gpu_dev_verbs_wqe* wqe0,
    struct doca_gpu_dev_verbs_wqe* wqe1, uint16_t wqeIndex,
    enum doca_gpu_dev_verbs_wqe_ctrl_flags ctrlFlags, uint64_t remoteAddress, uint32_t remoteRkey,
    uint64_t localAddress, uint32_t localLkey, niinGpunetioAtomicOp op, T operand, T compare) {
  const uint64_t operandBits = atomicBits(operand);
  const uint64_t compareBits = atomicBits(compare);
  const uint64_t mask = sizeof(T) == 4 ? uint64_t(UINT32_MAX) : UINT64_MAX;

  if constexpr (sizeof(T) == 4) {
    switch (op) {
      case niinGpunetioAtomicOp::FetchAdd:
      case niinGpunetioAtomicOp::Add:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, operandBits, 0, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::Inc:
      case niinGpunetioAtomicOp::FetchInc:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 1, 0, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::Fetch:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::FetchXor:
      case niinGpunetioAtomicOp::Xor:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, operandBits, mask, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::Swap:
      case niinGpunetioAtomicOp::Set:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, 0, mask, 0);
        break;
      case niinGpunetioAtomicOp::FetchAnd:
      case niinGpunetioAtomicOp::And:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, 0, ~operandBits & mask, 0);
        break;
      case niinGpunetioAtomicOp::FetchOr:
      case niinGpunetioAtomicOp::Or:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, 0, operandBits, 0);
        break;
      case niinGpunetioAtomicOp::CompareSwap:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_4>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, compareBits, mask, mask);
        break;
    }
  } else {
    switch (op) {
      case niinGpunetioAtomicOp::FetchAdd:
      case niinGpunetioAtomicOp::Add:
        // The normal 64-bit fetch-add WQE is preferable when its exact
        // semantics are sufficient; the masked form remains available below
        // for increment, fetch, and xor.
        doca_gpu_dev_verbs_wqe_prepare_atomic(qp, wqe0, wqeIndex,
                                              DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA, ctrlFlags,
                                              remoteAddress, remoteRkey, localAddress, localLkey, 8,
                                              operandBits, 0);
        break;
      case niinGpunetioAtomicOp::CompareSwap:
        doca_gpu_dev_verbs_wqe_prepare_atomic(qp, wqe0, wqeIndex,
                                              DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_CS, ctrlFlags,
                                              remoteAddress, remoteRkey, localAddress, localLkey, 8,
                                              compareBits, operandBits);
        break;
      case niinGpunetioAtomicOp::Inc:
      case niinGpunetioAtomicOp::FetchInc:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 1, 0, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::Fetch:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::FetchXor:
      case niinGpunetioAtomicOp::Xor:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, operandBits, mask, 0, 0, 0, 0);
        break;
      case niinGpunetioAtomicOp::Swap:
      case niinGpunetioAtomicOp::Set:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, 0, mask, 0);
        break;
      case niinGpunetioAtomicOp::FetchAnd:
      case niinGpunetioAtomicOp::And:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, 0, ~operandBits, 0);
        break;
      case niinGpunetioAtomicOp::FetchOr:
      case niinGpunetioAtomicOp::Or:
        doca_gpu_dev_verbs_wqe_prepare_atomic_ext<DOCA_GPUNETIO_VERBS_ATOMIC_EXT_BYTES_8>(
            qp, wqe0, wqe1, wqeIndex, DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS, ctrlFlags,
            remoteAddress, remoteRkey, localAddress, localLkey, 0, 0, operandBits, 0, operandBits, 0);
        break;
    }
  }
}

template <typename T>
__device__ __forceinline__ T readAtomicResult(const niinGpunetioAtomicContext* context, void* address) {
  uint64_t bits;
  if constexpr (sizeof(T) == 4) {
    // RC atomics place 32-bit responses in network byte order.
    bits = doca_gpu_dev_verbs_bswap32(*reinterpret_cast<volatile uint32_t*>(address));
  } else {
    bits = *reinterpret_cast<volatile uint64_t*>(address);
    if ((context->flags & NIIN_GPUNETIO_ATOMIC_CONTEXT_8B_RESULT_HOST_ENDIAN) == 0)
      bits = doca_gpu_dev_verbs_bswap64(bits);
  }
  return atomicValue<T>(bits);
}

}  // namespace detail
}  // namespace gpunetio
}  // namespace niin

// Try the native GPUNetIO path.  It returns false without issuing a WQE when
// the independent NIIN provider is absent, not ready, lacks the required
// capability, or its per-PE response slot is invalid.  The context owns one
// response slot and one lock per destination PE, so callers do not need to
// manage response-buffer tickets.  `previous` is valid only for a fetching
// operation and may be null when the caller wants to discard that value.
template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_try(
    const niinGpunetioAtomicContext* context, niinGpunetioAtomicOp op, size_t remoteOffset, T operand,
    T compare, int pe, T* previous = nullptr) {
  static_assert(std::is_integral<T>::value, "native GPUNetIO atomics support integral types only");
  static_assert(sizeof(T) == 4 || sizeof(T) == 8, "native GPUNetIO atomics require 4- or 8-byte operands");

  // ProxyAll is deliberately checked before direct endpoint availability.
  // A CPU RMW service cannot be mixed with a NIC AMO against the same heap
  // location, even for self/LSA destinations selected by the public adapter.
  if (niin_gpunetio_proxy_atomic_active(context))
    return niin_gpunetio_proxy_atomic_try(context, niin_gpunetio_proxy_op(op), remoteOffset, operand,
                                           compare, pe, previous);

  using namespace niin::gpunetio::detail;
  if (!directEndpointAvailable<T>(context, pe) || !isNativeOp(op))
    return false;

  const bool returnsPrevious = atomicReturnsPrevious(op);
  if (previous != nullptr && !returnsPrevious) return false;

  size_t responseSlotOffset = 0;
  if (!getResponseSlot<T>(context, pe, &responseSlotOffset)) return false;

  const niinGpunetioAtomicEndpoint& endpoint = context->endpoints[pe];
  if (remoteOffset > endpoint.remoteHeapBytes || sizeof(T) > endpoint.remoteHeapBytes - remoteOffset) return false;
  // Mellanox RC AMOs require naturally aligned operands.  Do this validation
  // before taking the per-PE lock or reserving a WQE so a bad user pointer
  // cannot submit an invalid operation and poison the dedicated QP.
  if ((remoteOffset & (alignof(T) - 1)) != 0) return false;

  struct doca_gpu_dev_verbs_qp* qp = reinterpret_cast<struct doca_gpu_dev_verbs_qp*>(endpoint.deviceQp);
  const uint64_t remoteAddress = endpoint.remoteHeapBase + remoteOffset;
  if (remoteAddress < endpoint.remoteHeapBase || (remoteAddress & (alignof(T) - 1)) != 0) return false;

  void* localResult = static_cast<char*>(context->responseBase) + responseSlotOffset;
  const bool atomicUsesTwoWqebbs = atomicConsumesTwoWqebbs<T>(op);
  // Match NVSHMEM's GPUNetIO completion rule.  A fetching AMO needs a fenced
  // DUMP unless the provider has established that its completion mode makes
  // the response visible without one.  A 64-bit masked CS always needs a
  // terminal WQE because its atomic encoding spans two WQEBBs.
  const bool needsCst = returnsPrevious &&
                        (context->flags & NIIN_GPUNETIO_ATOMIC_CONTEXT_MAY_SKIP_CST) == 0;
  if (needsCst && !validCstScratch(context)) return false;
  const bool needsTerminalWqe = atomicUsesTwoWqebbs || needsCst;
  const uint32_t wqeCount = 1 + (atomicUsesTwoWqebbs ? 1 : 0) + (needsTerminalWqe ? 1 : 0);

  // The one-slot-per-PE ring makes this an intentionally serialized path per
  // destination.  Hold the lock through CQ completion so another caller cannot
  // reuse either the response location or this dedicated QP prematurely.
  int* endpointLock = context->endpointLocks + pe;
  if (!lockEndpoint(endpointLock)) return false;
  const uint64_t firstWqe =
      doca_gpu_dev_verbs_reserve_wq_slots<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, wqeCount);
  const uint64_t completionWqe = firstWqe + wqeCount - 1;
  const uint16_t atomicWqeIndex = static_cast<uint16_t>(firstWqe);

  struct doca_gpu_dev_verbs_wqe* wqe0 = doca_gpu_dev_verbs_get_wqe_ptr(qp, atomicWqeIndex);
  struct doca_gpu_dev_verbs_wqe* wqe1 = atomicUsesTwoWqebbs
                                            ? doca_gpu_dev_verbs_get_wqe_ptr(
                                                  qp, static_cast<uint16_t>(firstWqe + 1))
                                            : nullptr;
  enum doca_gpu_dev_verbs_wqe_ctrl_flags atomicCtrl =
      needsTerminalWqe ? DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE
                       : DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE;

  prepareAtomicWqe(qp, wqe0, wqe1, atomicWqeIndex, atomicCtrl, remoteAddress, endpoint.remoteHeapRkey,
                   reinterpret_cast<uint64_t>(localResult), context->responseLkey, op, operand, compare);

  if (needsTerminalWqe) {
    struct doca_gpu_dev_verbs_wqe* completion =
        doca_gpu_dev_verbs_get_wqe_ptr(qp, static_cast<uint16_t>(completionWqe));
    if (needsCst) {
      void* cstScratch = static_cast<char*>(context->responseBase) + responseSlotOffset +
                         context->cstScratchOffset;
      doca_gpu_dev_verbs_wqe_prepare_dump(qp, completion, static_cast<uint16_t>(completionWqe),
                                          DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE_FENCE,
                                          reinterpret_cast<uint64_t>(cstScratch), context->responseLkey,
                                          sizeof(uint8_t));
    } else {
      doca_gpu_dev_verbs_wqe_prepare_nop(qp, completion, static_cast<uint16_t>(completionWqe),
                                         DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE);
    }
  }

  doca_gpu_dev_verbs_mark_wqes_ready<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(qp, firstWqe,
                                                                                       completionWqe);
  // CPU_PROXY requires its producer index to be explicitly published after
  // ringing the GPU-to-host proxy doorbell.  GPUNetIO's default submission
  // intentionally omits that update, which is correct only for direct GPU
  // doorbells.  The host records the handler choice in this NIIN-owned flag.
  const uint32_t submitOptions =
      (context->flags & NIIN_GPUNETIO_ATOMIC_CONTEXT_CPU_PROXY_DOORBELL) != 0
          ? DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_CPU_PROXY_UPDATE_PI
          : DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT;
  doca_gpu_dev_verbs_submit<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
      qp, completionWqe + 1, submitOptions);
  const int completionStatus =
      doca_gpu_dev_verbs_poll_cq_at<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
          qp, completionWqe);
  if (completionStatus != 0) {
    failEndpoint(endpointLock);
    return false;
  }
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);

  if (previous != nullptr) *previous = readAtomicResult<T>(context, localResult);
  unlockEndpoint(endpointLock);
  return true;
}

#else  // NIIN_GPUNETIO_HAS_DEVICE_API

// Keep callers compileable in non-GPUNetIO builds.  The public NIIN layer can
// then select its existing fail-closed behavior or a future proxy fallback.
template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_try(
    const niinGpunetioAtomicContext* context, niinGpunetioAtomicOp op, size_t remoteOffset, T operand,
    T compare, int pe, T* previous = nullptr) {
  // The proxy device dispatcher has no GPUNetIO header dependency. This lets
  // a ProxyAll client compile its CUDA AMO call sites without enabling the
  // direct WQE path, while an unbound context retains fail-closed behavior.
  return niin_gpunetio_proxy_atomic_try(context, niin_gpunetio_proxy_op(op), remoteOffset, operand,
                                         compare, pe, previous);
}

#endif  // NIIN_GPUNETIO_HAS_DEVICE_API

// Thin typed helpers keep the eventual public NIIN atomics header free of
// GPUNetIO opcode knowledge.  The context owns the response ring, so no
// caller-visible result slot is part of this interface.
template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_fetch_add_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::FetchAdd, offset, value, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_add_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Add, offset, value, T{}, pe);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_compare_swap_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T compare, T value, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::CompareSwap, offset, value, compare, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_swap_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Swap, offset, value, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_fetch_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Fetch, offset, T{}, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_set_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Set, offset, value, T{}, pe);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_inc_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, int pe) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Inc, offset, T{}, T{}, pe);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_fetch_inc_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::FetchInc, offset, T{}, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_fetch_and_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::FetchAnd, offset, value, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_and_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::And, offset, value, T{}, pe);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_fetch_or_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::FetchOr, offset, value, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_or_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Or, offset, value, T{}, pe);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_fetch_xor_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe, T* previous) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::FetchXor, offset, value, T{}, pe, previous);
}

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_atomic_xor_try(
    const niinGpunetioAtomicContext* ctx, size_t offset, T value, int pe) {
  return niin_gpunetio_atomic_try(ctx, niinGpunetioAtomicOp::Xor, offset, value, T{}, pe);
}

#endif  // NIIN_GPUNETIO_ATOMICS_H_
