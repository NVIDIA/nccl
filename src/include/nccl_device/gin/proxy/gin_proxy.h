/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_PROXY_H_
#define _NCCL_DEVICE_GIN_PROXY_H_

//#include <config.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "nccl.h"
#include "nccl_device/utility.h"
#include "../gin_device_host_common.h"
#include "gin_proxy_device_host_common.h"

namespace nccl {
namespace gin {
namespace proxy {
NCCL_DEVICE_INLINE void flush(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, cuda::memory_order ord) {
  using nccl::utility::loadConst;
  using nccl::utility::rollingLessEq;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);

  // The PI and CI can keep moving because of concurrent threads posting GFDs to this queue, and the CPU consuming them.
  // Therefore, to prevent overflow issues in the while statement, we need to use a special comparison function.
  uint32_t p = pi.load(cuda::memory_order_relaxed);
#pragma unroll 1
  while (!rollingLessEq<uint32_t>(p, ci.load(ord))) continue;
}

template <typename Coop>
NCCL_DEVICE_INLINE void postGfd(Coop coop, ncclGinProxyGpuCtx_t* proxyCtx, ncclGinProxyGfd_t* gfd,
                                uint32_t pe) {
  using nccl::utility::loadConst;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  ncclGinProxyGfd_t* q = &loadConst(&proxyCtx->queues)[pe * proxyCtx->queueSize];
  uint32_t queueSize = loadConst(&proxyCtx->queueSize);

  if (coop.thread_rank() == 0) {
    // claim a slot in the gfd queue
    uint32_t idx = pi.fetch_add(1, cuda::memory_order_relaxed);
    // wait for credits
    while (queueSize <= idx - ci.load(cuda::memory_order_relaxed)) {
    }
    idx &= queueSize - 1;
// 4x16 byte store with the write-through cache hint
#pragma unroll
    for (uint8_t i = 0; i < 4; i++) {
      __stwt((uint4*)&q[idx] + i, ((uint4*)gfd)[i]);
    }
  }
}

template <typename T>
// Descriptor must be at least GWQ_GFD_SIZE bytes and it should be aligned
__device__ __forceinline__ void buildGfd(ncclGinProxyGfd_t* gfd, ncclGinProxyOp_t op, T srcVal,
                                         bool hasInline, size_t srcOff, ncclGinWindow_t srcHandle,
                                         size_t dstOff, ncclGinWindow_t dstHandle, size_t size,
                                         ncclGinCounter_t counterId, ncclGinSignal_t signalId,
                                         uint64_t signalVal) {
  gfd->qword[ncclGinProxyGfdHeader].header.flag = 1;
  gfd->qword[ncclGinProxyGfdHeader].header.op = op;
  gfd->qword[ncclGinProxyGfdHeader].header.size = (uint64_t)size;

  if (hasInline) {
    gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.flag = 1;
    gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow = (uint32_t)srcVal;
    gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.flag = 1;
    if (sizeof(T) > 4)
      gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 = (uint64_t)srcVal >> 32;
    if (sizeof(T) > 6)
      gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh = (uint64_t)srcVal >> 48;
  } else {
    gfd->qword[ncclGinProxyGfdSrcOff].srcOff.flag = 1;
    gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff = (uint64_t)srcOff;
    gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.flag = 1;
    gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle = (uint64_t)srcHandle;
  }

  gfd->qword[ncclGinProxyGfdDstOff].dstOff.flag = 1;
  gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff = (uint64_t)dstOff;
  gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.flag = 1;
  gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle = (uint64_t)dstHandle;

  gfd->qword[ncclGinProxyGfdCompletion].completion.flag = 1;
  gfd->qword[ncclGinProxyGfdCompletion].completion.counterId = (uint16_t)counterId;
  gfd->qword[ncclGinProxyGfdCompletion].completion.signalId = (uint16_t)signalId;

  // The signal value is split between two qwords, as the signal value is a full 64 bits
  gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow = (uint16_t)signalVal;
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.flag = 1;
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 = (uint16_t)(signalVal >> 16);
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh = (uint32_t)(signalVal >> 32);

  gfd->qword[ncclGinProxyGfdReserved].flag.v = 1;
}

__device__ __forceinline__ void constructProxyOp(ncclGinProxyOp_t& op, bool hasInline,
                                                 bool hasSignal, ncclGinSignalOp_t signalOp,
                                                 bool hasCounter) {
  op = ncclGinProxyOpPut;
  if (hasInline)
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint8_t>(op) |
                                       static_cast<uint8_t>(ncclGinProxyOpWithInline));
  if (hasCounter)
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint8_t>(op) |
                                       static_cast<uint8_t>(ncclGinProxyOpWithCounter));
  if (hasSignal) {
    switch (signalOp) {
      case ncclGinSignalInc:
        op = static_cast<ncclGinProxyOp_t>(static_cast<uint8_t>(op) |
                                           static_cast<uint8_t>(ncclGinProxyOpWithSignalInc));
        break;
      case ncclGinSignalAdd:
        op = static_cast<ncclGinProxyOp_t>(static_cast<uint8_t>(op) |
                                           static_cast<uint8_t>(ncclGinProxyOpWithSignalAdd));
        break;
      default:
        __builtin_unreachable();
    }
  }
}

template <typename Coop, typename T>
NCCL_DEVICE_INLINE void put(Coop coop, ncclGinProxyGfd_t* gfd, ncclGinProxyGpuCtx_t* proxyCtx,
                            int peer, ncclGinWindow_t dstWnd, size_t dstOff, T srcVal,
                            bool hasInline, ncclGinWindow_t srcWnd, size_t srcOff, size_t bytes,
                            bool hasSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                            uint64_t signalVal, bool hasCounter, ncclGinCounter_t counterId,
                            cuda::thread_scope required, cuda::thread_scope given) {
  if ((int)given > (int)cuda::thread_scope_system) {
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  }
  constexpr size_t chunkSize = 1ULL << 30;
  while (bytes > chunkSize) {
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*hasInline*/false, /*hasSignal*/false, signalOp, /*hasCounter*/false);
    nccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/0, /*hasInline*/false, srcOff, srcWnd,
                               dstOff, dstWnd, chunkSize, /*counterId*/0, /*signalId*/0,
                               /*signalVal*/0);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
    bytes -= chunkSize;
    srcOff += chunkSize;
    dstOff += chunkSize;
  }
  ncclGinProxyOp_t op;
  constructProxyOp(op, hasInline, hasSignal, signalOp, hasCounter);
  nccl::gin::proxy::buildGfd(gfd, op, srcVal, hasInline, srcOff, srcWnd, dstOff, dstWnd, bytes,
                             hasCounter ? counterId : 0, hasSignal ? signalId : 0, signalVal);
  nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
}
}  // namespace proxy
}  // namespace gin
}  // namespace nccl

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinProxyGpuCtx_t* proxyCtx = (ncclGinProxyGpuCtx_t*)ctx.handle;
    uint64_t* counter = nccl::utility::loadConst(&proxyCtx->counters) + counterId;
    return counter;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinProxyGpuCtx_t* proxyCtx = (ncclGinProxyGpuCtx_t*)ctx.handle;
    uint64_t* counter = nccl::utility::loadConst(&proxyCtx->counters) + counterId;
    *counter = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinProxyGpuCtx_t* proxyCtx = (ncclGinProxyGpuCtx_t*)ctx.handle;
    uint64_t* signal = nccl::utility::loadConst(&proxyCtx->signals) + signalId;
    return signal;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinProxyGpuCtx_t* proxyCtx = (ncclGinProxyGpuCtx_t*)ctx.handle;
    uint64_t* signal = nccl::utility::loadConst(&proxyCtx->signals) + signalId;
    *signal = 0;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord) {
    ncclGinProxyGpuCtx_t* proxyCtx = (ncclGinProxyGpuCtx_t*)ctx.handle;
#pragma unroll 1
    for (int pe = coop.thread_rank(); pe < ctx.nRanks; pe += coop.size()) {
      nccl::gin::proxy::flush(proxyCtx, pe, ord);
    }
  }
};

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    nccl::gin::proxy::put<Coop, uint64_t>(
      coop, desc, (ncclGinProxyGpuCtx_t*)ctx.handle, peer, dstWin, dstOff, 0, false, srcWin, srcOff,
      bytes, hasSignal, signalId, signalOp, signalOpArg, hasCounter, counterId, required, given);
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    nccl::gin::proxy::put<Coop, T>(coop, desc, (ncclGinProxyGpuCtx_t*)ctx.handle, peer, dstWin,
                                   dstOff, srcVal, true, nullptr, 0, sizeof(T), hasSignal, signalId,
                                   signalOp, signalOpArg, false, 0, required, given);
  }
};

#endif
