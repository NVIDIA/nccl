/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_PROXY_ATOMICS_DEVICE_H_
#define NIIN_GPUNETIO_PROXY_ATOMICS_DEVICE_H_

#include "niin/gpunetio/context.h"
#include "niin/gpunetio/proxy_atomics/api.h"

#if defined(__CUDACC__)
#include <cuda/atomic>
#endif

// One source slot has a GPU-published request and a target-published
// completion. The allocation is cudaHostAllocMapped memory: `slots` in the
// device context is its GPU alias, while the NIIN proxy threads retain the CPU
// alias. Only the ticket fields are concurrently accessed by CPU/NIC and GPU;
// their acquire/release protocol makes the surrounding body visible.
struct alignas(64) niinGpunetioProxyAtomicSlot {
  struct niinGpunetioProxyAtomicRequest request;
  struct niinGpunetioProxyAtomicCompletion completion;
};

// This device ABI stays separate from the direct GPUNetIO WQE context. It is
// deliberately made of NIIN-owned fixed-layout state; no GIN, GPUNetIO, or
// ibverbs object is exposed to device code.
#define NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_VERSION 2u

enum niinGpunetioProxyAtomicDeviceContextFlags : uint32_t {
  NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_READY = 1u << 0,
  NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_PROXY_ALL = 1u << 1,
};

struct niinGpunetioProxyAtomicDeviceContext {
  uint32_t version;
  uint32_t flags;
  int nPes;
  int localPe;
  size_t heapBytes;
  uint32_t slotsPerPe;
  uint32_t reserved;
  struct niinGpunetioProxyAtomicSlot* slots;
  int* endpointLocks;
  // CUDA-mapped host memory. The progress thread publishes a nonzero value
  // before it stops on an unrecoverable transport error, allowing a GPU
  // waiter to fail closed rather than spin forever on its completion ticket.
  uint32_t* progressFailure;
};

#if defined(__CUDACC__)

namespace niin {
namespace gpunetio {
namespace proxy {
namespace detail {

__device__ __forceinline__ bool proxyContextAvailable(
    const struct niinGpunetioAtomicContext* atomicContext,
    const struct niinGpunetioProxyAtomicDeviceContext** proxyContext) {
  constexpr uint32_t requiredAtomicFlags = NIIN_GPUNETIO_ATOMIC_CONTEXT_READY |
                                           NIIN_GPUNETIO_ATOMIC_CONTEXT_PROXY;
  if (atomicContext == nullptr || atomicContext->version != NIIN_GPUNETIO_ATOMIC_CONTEXT_VERSION ||
      (atomicContext->flags & requiredAtomicFlags) != requiredAtomicFlags ||
      atomicContext->proxyContext == nullptr)
    return false;
  const auto* context =
      static_cast<const struct niinGpunetioProxyAtomicDeviceContext*>(atomicContext->proxyContext);
  constexpr uint32_t requiredProxyFlags = NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_READY |
                                         NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_PROXY_ALL;
  if (context->version != NIIN_GPUNETIO_PROXY_ATOMIC_DEVICE_CONTEXT_VERSION ||
      (context->flags & requiredProxyFlags) != requiredProxyFlags || context->nPes <= 0 ||
      context->slotsPerPe == 0 || context->slots == nullptr || context->endpointLocks == nullptr ||
      context->progressFailure == nullptr)
    return false;
  *proxyContext = context;
  return true;
}

__device__ __forceinline__ bool lockEndpoint(int* lock) {
  for (;;) {
    const int old = atomicCAS(lock, 0, 1);
    if (old == 0) return true;
    if (old < 0) return false;
  }
}

__device__ __forceinline__ void unlockEndpoint(int* lock) { atomicExch(lock, 0); }

__device__ __forceinline__ void failEndpoint(int* lock) { atomicExch(lock, -1); }

template <typename T>
__device__ __forceinline__ bool issue(const struct niinGpunetioAtomicContext* atomicContext,
                                      enum niinGpunetioProxyAtomicOp op, size_t remoteOffset,
                                      T operand, T compare, int pe, T* previous) {
  using Traits = AtomicValueTraits<T>;
  static_assert(Traits::kSupported,
                "NIIN proxy AMOs require 4/8-byte integral, float, double, or __half values");
  if (!isPublicAtomicOperation<T>(static_cast<uint8_t>(op))) return false;

  const struct niinGpunetioProxyAtomicDeviceContext* proxyContext = nullptr;
  if (!proxyContextAvailable(atomicContext, &proxyContext) || pe < 0 || pe >= proxyContext->nPes ||
      remoteOffset > proxyContext->heapBytes || Traits::kBytes > proxyContext->heapBytes - remoteOffset ||
      (remoteOffset & (Traits::kBytes - 1)) != 0)
    return false;

  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> progressFailure(
      *proxyContext->progressFailure);
  if (progressFailure.load(cuda::memory_order_acquire) != 0) return false;

  // The first implementation intentionally leases one slot per destination.
  // The wire carries sourceSlot and supports a future ticketed ring, but using
  // one locked slot first provides strict completion ordering without a GPU
  // allocator or an ABA-sensitive free list.
  const uint64_t slotIndex = static_cast<uint64_t>(pe) * proxyContext->slotsPerPe;
  const uint64_t slotCount = static_cast<uint64_t>(proxyContext->nPes) * proxyContext->slotsPerPe;
  if (slotIndex >= slotCount) return false;
  struct niinGpunetioProxyAtomicSlot* slot = proxyContext->slots + slotIndex;
  int* endpointLock = proxyContext->endpointLocks + pe;
  if (!lockEndpoint(endpointLock)) return false;
  if (progressFailure.load(cuda::memory_order_acquire) != 0) {
    failEndpoint(endpointLock);
    return false;
  }

  // The slot cannot be reused until the prior caller consumed completion and
  // released this lock, so a nonzero monotonically incremented ticket is
  // enough to distinguish this request from every earlier use.
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> requestTicket(slot->request.ticket);
  cuda::atomic_ref<uint64_t, cuda::thread_scope_system> completionTicket(slot->completion.ticket);
  const uint64_t ticket = niinGpunetioProxyAtomicNextTicket(
      requestTicket.load(cuda::memory_order_relaxed));

  slot->request.magic = NIIN_GPUNETIO_PROXY_ATOMIC_PROTOCOL_MAGIC;
  slot->request.version = NIIN_GPUNETIO_PROXY_ATOMIC_PROTOCOL_VERSION;
  slot->request.op = static_cast<uint8_t>(op);
  slot->request.type = Traits::kWireType;
  slot->request.sourcePe = static_cast<uint32_t>(proxyContext->localPe);
  slot->request.targetPe = static_cast<uint32_t>(pe);
  slot->request.sourceSlot = static_cast<uint32_t>(slotIndex);
  slot->request.flags = NIIN_GPUNETIO_PROXY_ATOMIC_REQUEST_FLAG_NONE;
  slot->request.remoteOffset = remoteOffset;
  slot->request.operandBits = atomicValueBits(operand);
  slot->request.compareBits = atomicValueBits(compare);
  slot->completion.resultBits = 0;
  slot->completion.status = NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_INTERNAL_ERROR;
  slot->completion.reserved = 0;
  completionTicket.store(NIIN_GPUNETIO_PROXY_ATOMIC_UNPUBLISHED_TICKET,
                         cuda::memory_order_relaxed);

  // Ticket is physically last in the request. A system-scope release makes
  // every preceding request field visible before the host proxy observes it.
  requestTicket.store(ticket, cuda::memory_order_release);

  while (completionTicket.load(cuda::memory_order_acquire) != ticket) {
    if (progressFailure.load(cuda::memory_order_acquire) != 0) {
      failEndpoint(endpointLock);
      return false;
    }
  }

  const uint32_t status = slot->completion.status;
  if (status != NIIN_GPUNETIO_PROXY_ATOMIC_COMPLETION_SUCCESS) {
    // A non-success completion means a potentially fatal transport/target
    // condition. Poison only this destination endpoint; a later AMO cannot
    // overwrite its slot while the caller is reporting failure.
    failEndpoint(endpointLock);
    return false;
  }
  if (previous != nullptr && niinGpunetioProxyAtomicOpReturnsPrevious(static_cast<uint8_t>(op)))
    *previous = atomicValueFromBits<T>(slot->completion.resultBits);
  unlockEndpoint(endpointLock);
  return true;
}

}  // namespace detail
}  // namespace proxy
}  // namespace gpunetio
}  // namespace niin

template <typename T>
__device__ __forceinline__ bool niin_gpunetio_proxy_atomic_try(
    const struct niinGpunetioAtomicContext* context, enum niinGpunetioProxyAtomicOp op,
    size_t remoteOffset, T operand, T compare, int pe, T* previous = nullptr) {
  return niin::gpunetio::proxy::detail::issue(context, op, remoteOffset, operand, compare, pe,
                                                previous);
}

__device__ __forceinline__ bool niin_gpunetio_proxy_atomic_active(
    const struct niinGpunetioAtomicContext* context) {
  const struct niinGpunetioProxyAtomicDeviceContext* proxyContext = nullptr;
  return niin::gpunetio::proxy::detail::proxyContextAvailable(context, &proxyContext);
}

#else

template <typename T>
inline bool niin_gpunetio_proxy_atomic_try(const struct niinGpunetioAtomicContext*,
                                           enum niinGpunetioProxyAtomicOp, size_t, T, T, int,
                                           T* = nullptr) {
  return false;
}

inline bool niin_gpunetio_proxy_atomic_active(const struct niinGpunetioAtomicContext*) {
  return false;
}

#endif  // defined(__CUDACC__)

#endif  // NIIN_GPUNETIO_PROXY_ATOMICS_DEVICE_H_
