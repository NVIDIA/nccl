/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_PROXY_H_
#define _NCCL_DEVICE_GIN_PROXY_H_

// #include <config.h>

#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include "nccl.h"
#include "nccl_device/gin/gin_device_common.h"
#include "nccl_device/utility.h"
#include "../gin_device_host_common.h"
#include "gin_proxy_device_host_common.h"

struct ncclGinCpuProxyRequest {
  int peer;
  uint32_t nextGfdIdx;
  uint32_t lastIssuedGet;
};
static_assert(sizeof(ncclGinCpuProxyRequest) <= sizeof(ncclGinRequest_t),
              "ncclGinCpuProxyRequest must fit in ncclGinRequest_t");

// Clang's CUDA mode skips sm_32_intrinsics.hpp; clang <= 20 has no __stwt
// declarations.
#if defined(__clang__) && defined(__CUDA__) && (__clang_major__ < 21)
NCCL_DEVICE_INLINE void __stwt(uint4* addr, const uint4& val) {
  asm("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(addr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)
      : "memory");
}
#endif

namespace nccl {
namespace gin {
namespace proxy {

// Chunk size for Gin Proxy GFD operations
static constexpr size_t DataChunkSize = 1ULL << 30;  // 1 GB

template <bool HasTimeout>
NCCL_DEVICE_INLINE ncclResult_t waitForGfdCompleteCore(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, uint32_t nextGfdIdx,
                                                       cuda::memory_order ord, uint32_t* abortFlag, uint64_t startCycle,
                                                       uint64_t timeoutCycles) {
  using nccl::utility::loadConst;
  using nccl::utility::rollingLessEq;
  using nccl::utility::testAbort;
  (void)startCycle;
  (void)timeoutCycles; // referenced only when HasTimeout is true
  // The PI and CI can keep moving because of concurrent threads posting GFDs to this queue, and the CPU consuming
  // them. Therefore, to prevent overflow issues in the while statement, we need to use a special comparison function.
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> ci(loadConst(&proxyCtx->cis)[pe]);
  uint32_t steps = 0;
  NVCC_PRAGMA_UNROLL_DISABLED
  while (true) {
    if (rollingLessEq<uint32_t>(nextGfdIdx, ci.load(ord))) return ncclSuccess;
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
    }
    if (testAbort(abortFlag, steps)) return ncclSuccess;
  }
}

NCCL_DEVICE_INLINE void waitForGfdComplete(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, uint32_t nextGfdIdx,
                                           cuda::memory_order ord, uint32_t* abortFlag) {
  (void)waitForGfdCompleteCore</*HasTimeout=*/false>(proxyCtx, pe, nextGfdIdx, ord, abortFlag, 0, 0);
}

NCCL_DEVICE_INLINE ncclResult_t waitForGfdComplete(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, uint32_t nextGfdIdx,
                                                   cuda::memory_order ord, uint32_t* abortFlag, uint64_t startCycle,
                                                   uint64_t timeoutCycles) {
  return waitForGfdCompleteCore</*HasTimeout=*/true>(proxyCtx, pe, nextGfdIdx, ord, abortFlag, startCycle,
                                                     timeoutCycles);
}

template <bool HasTimeout>
NCCL_DEVICE_INLINE ncclResult_t flushCore(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, cuda::memory_order ord,
                                          uint32_t* abortFlag, uint64_t startCycle, uint64_t timeoutCycles) {
  using nccl::utility::loadConst;
  cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[pe]);
  uint32_t p = pi.load(cuda::memory_order_relaxed);
  return waitForGfdCompleteCore<HasTimeout>(proxyCtx, pe, p, ord, abortFlag, startCycle, timeoutCycles);
}

NCCL_DEVICE_INLINE void flush(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, cuda::memory_order ord,
                              uint32_t* abortFlag) {
  (void)flushCore</*HasTimeout=*/false>(proxyCtx, pe, ord, abortFlag, 0, 0);
}

NCCL_DEVICE_INLINE ncclResult_t flush(ncclGinProxyGpuCtx_t* proxyCtx, uint32_t pe, cuda::memory_order ord,
                                      uint32_t* abortFlag, uint64_t startCycle, uint64_t timeoutCycles) {
  return flushCore</*HasTimeout=*/true>(proxyCtx, pe, ord, abortFlag, startCycle, timeoutCycles);
}

template <typename Coop>
NCCL_DEVICE_INLINE void postGfd(Coop coop, ncclGinProxyGpuCtx_t* proxyCtx, ncclGinProxyGfd_t* gfd, uint32_t pe,
                                bool isGet = false) {
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
    uint32_t gfdIdx = idx & (queueSize - 1);
    // 16-byte vector stores with the write-through cache hint. Both sides cast
    // through (uint4*), which emits v4.b32 PTX requiring 16-byte alignment.
    // ncclGinProxyGfd_t is declared __attribute__((packed, aligned(16))) in
    // gin_proxy_device_host_common.h; static_asserts there enforce the contract.
    NVCC_PRAGMA_UNROLL_AUTO
    for (uint8_t i = 0; i < sizeof(ncclGinProxyGfd_t) / sizeof(uint4); i++) {
      __stwt((uint4*)&q[gfdIdx] + i, ((uint4*)gfd)[i]);
    }
    if (isGet) {
      // Atomic max with rolling logic.
      cuda::atomic_ref<uint32_t, cuda::thread_scope_device> lastIssuedGet(
        nccl::utility::loadConst(&proxyCtx->lastIssuedGet)[pe]);
      uint32_t current = lastIssuedGet.load(cuda::memory_order_relaxed);
      while (nccl::utility::rollingLessThan<uint32_t>(current, idx + 1)) {
        if (lastIssuedGet.compare_exchange_weak(current, idx + 1, cuda::memory_order_release,
                                                cuda::memory_order_relaxed))
          break;
        current = lastIssuedGet.load(cuda::memory_order_relaxed);
      }
    }
  }
}

template <typename T>
// Descriptor must be at least GWQ_GFD_SIZE bytes and it should be aligned
// Assumes little-endian, which is okay.
__device__ __forceinline__ void buildGfd(ncclGinProxyGfd_t* gfd, ncclGinProxyOp_t op, T srcVal, bool hasInline,
                                         size_t srcOff, ncclGinWindow_t srcHandle, size_t dstOff,
                                         ncclGinWindow_t dstHandle, size_t size, ncclGinCounter_t counterId,
                                         ncclGinSignal_t signalId, uint64_t signalVal, ncclGinWindow_t signalWindow,
                                         size_t signalOff, bool isStrongSignal = false) {
  for (int i = 0; i < ncclGinProxyGfdQwords; i++) {
    gfd->qword[i].flag.v = 1;
  }

  gfd->qword[ncclGinProxyGfdHeader].header.version = (uint64_t)NCCL_GIN_PROXY_GFD_VERSION;
  gfd->qword[ncclGinProxyGfdHeader].header.size = (uint64_t)size;
  gfd->qword[ncclGinProxyGfdHeaderExt].headerExt.op = (uint16_t)op;

  if (op & ncclGinProxyOpFlush) {
    return;
  }

  if (hasInline) {
    uint64_t srcValBits = 0;
    memcpy(&srcValBits, &srcVal, sizeof(T));
    gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow = (uint32_t)srcValBits;
    if (sizeof(T) > 4) gfd->qword[ncclGinProxyGfdInlineLow].inlineLow.inlineValLow2 = (uint64_t)srcValBits >> 32;
    if (sizeof(T) > 6) gfd->qword[ncclGinProxyGfdInlineHigh].inlineHigh.inlineValHigh = (uint64_t)srcValBits >> 48;
  } else if (op & ncclGinProxyOpVASignal) {
    gfd->qword[ncclGinProxyGfdVASignalOff].vaSignalOff.vaSignalOff = (uint64_t)signalOff;
    gfd->qword[ncclGinProxyGfdVASignalHandle].vaSignalHandle.vaSignalHandle = (uint64_t)signalWindow;
  } else {
    gfd->qword[ncclGinProxyGfdSrcOff].srcOff.srcOff = (uint64_t)srcOff;
    gfd->qword[ncclGinProxyGfdSrcHandle].srcHandle.srcHandle = (uint64_t)srcHandle;
  }

  gfd->qword[ncclGinProxyGfdDstOff].dstOff.dstOff = (uint64_t)dstOff;
  gfd->qword[ncclGinProxyGfdDstHandle].dstHandle.dstHandle = (uint64_t)dstHandle;

  gfd->qword[ncclGinProxyGfdCompletion].completion.counterId = counterId;
  gfd->qword[ncclGinProxyGfdCompletion].completion.signalId = signalId;

  // The signal value is split between two qwords, as the signal value is a full 64 bits
  gfd->qword[ncclGinProxyGfdCompletion].completion.signalValLow = (uint16_t)signalVal;
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValLow2 = (uint16_t)(signalVal >> 16);
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.signalValHigh = (uint32_t)(signalVal >> 32);
  gfd->qword[ncclGinProxyGfdSignalVal].signalVal.isStrongSignal = isStrongSignal ? 1 : 0;
}

__device__ __forceinline__ void constructProxyOp(ncclGinProxyOp_t& op, bool isGet, bool isFlush, bool hasInline,
                                                 ncclGinSignalType signalType, ncclGinSignalOp_t signalOp,
                                                 bool hasCounter) {
  op = (ncclGinProxyOp_t)(0);
  if (isGet) {
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpGet));
    return;
  }

  if (isFlush) {
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpFlush));
    return;
  }

  if (signalType != NCCL_GIN_SIGNAL_TYPE_NONE) {
    switch (signalOp) {
    case ncclGinSignalInc:
      op =
        static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpWithSignalInc));
      break;
    case ncclGinSignalAdd:
      op =
        static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpWithSignalAdd));
      break;
    default:
      __builtin_unreachable();
    }
  }
  if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpVASignal));
    return;
  }
  op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpPut));
  if (hasInline)
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpWithInline));
  if (hasCounter)
    op = static_cast<ncclGinProxyOp_t>(static_cast<uint16_t>(op) | static_cast<uint16_t>(ncclGinProxyOpWithCounter));
}

template <typename Coop>
NCCL_DEVICE_INLINE void get(Coop coop, ncclGinProxyGpuCtx_t* proxyCtx, int peer, ncclGinWindow_t remoteWnd,
                            size_t remoteOff, ncclGinWindow_t localWnd, size_t localOff, size_t bytes,
                            ncclGinProxyGfd_t* desc) {
  using nccl::gin::proxy::DataChunkSize;
  while (bytes > 0) {
    size_t sendSize = min(bytes, DataChunkSize);
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/ true, /*isFlush*/ false, /*hasInline*/ false, NCCL_GIN_SIGNAL_TYPE_NONE,
                     ncclGinSignalInc, /*hasCounter*/ false);
    nccl::gin::proxy::buildGfd(desc, op, /*srcVal*/ 0, /*hasInline*/ false, remoteOff, remoteWnd, localOff, localWnd,
                               sendSize, /*counterId*/ 0, /*signalId*/ 0,
                               /*signalVal*/ 0, nullptr, 0);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, desc, peer, /*isGet*/ true);
    bytes -= sendSize;
    remoteOff += sendSize;
    localOff += sendSize;
  }
}

template <typename Coop, typename T>
NCCL_DEVICE_INLINE void put(Coop coop, ncclGinProxyGfd_t* gfd, ncclGinProxyGpuCtx_t* proxyCtx, int peer,
                            ncclGinWindow_t dstWnd, size_t dstOff, T srcVal, bool hasInline, ncclGinWindow_t srcWnd,
                            size_t srcOff, size_t bytes, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                            uint64_t signalVal, bool hasCounter, ncclGinCounter_t counterId,
                            cuda::thread_scope required, cuda::thread_scope given) {
  if ((int)given > (int)cuda::thread_scope_system) {
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  }

  using nccl::gin::proxy::DataChunkSize;
  while (bytes > DataChunkSize) {
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/ false, /*isFlush*/ false, /*hasInline*/ false, NCCL_GIN_SIGNAL_TYPE_NONE, signalOp,
                     /*hasCounter*/ false);
    nccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/ 0, /*hasInline*/ false, srcOff, srcWnd, dstOff, dstWnd,
                               DataChunkSize, /*counterId*/ 0, /*signalId*/ 0,
                               /*signalVal*/ 0, nullptr, 0);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
    bytes -= DataChunkSize;
    srcOff += DataChunkSize;
    dstOff += DataChunkSize;
  }

  ncclGinSignalType putSignalType;
  uint64_t putSignalVal;
  ncclGinSignal_t putSignalId;
  switch (signal.type) {
  case NCCL_GIN_SIGNAL_TYPE_INDEXED:
    putSignalType = NCCL_GIN_SIGNAL_TYPE_INDEXED;
    putSignalVal = signalVal;
    putSignalId = signal.indexedSignal.signalId;
    break;
  case NCCL_GIN_SIGNAL_TYPE_VA: // VA signals must be in a separate GFD. Use no signal during first put.
  case NCCL_GIN_SIGNAL_TYPE_NONE:
    putSignalType = NCCL_GIN_SIGNAL_TYPE_NONE;
    putSignalVal = 0;
    putSignalId = 0;
    break;
  default:
    __builtin_unreachable();
  }
  if (hasInline || hasCounter || srcWnd != nullptr || putSignalType != NCCL_GIN_SIGNAL_TYPE_NONE) {
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/ false, /*isFlush*/ false, hasInline, putSignalType, signalOp, hasCounter);
    nccl::gin::proxy::buildGfd(gfd, op, srcVal, hasInline, srcOff, srcWnd, dstOff, dstWnd, bytes,
                               hasCounter ? counterId : 0, putSignalId, putSignalVal, nullptr, 0, signal.isStrong);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
  }

  // Handle additional GFD for VA signals.
  if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
    ncclGinProxyOp_t op;
    constructProxyOp(op, /*isGet*/ false, /*isFlush*/ false, /*hasInline*/ false, NCCL_GIN_SIGNAL_TYPE_VA, signalOp,
                     /*hasCounter*/ false);
    nccl::gin::proxy::buildGfd(gfd, op, /*srcVal*/ 0, /*hasInline*/ false, 0, nullptr, 0, nullptr, 0, 0, 0, signalVal,
                               signal.vaSignal.signalWindow, signal.vaSignal.signalOffset, signal.isStrong);
    nccl::gin::proxy::postGfd<Coop>(coop, proxyCtx, gfd, peer);
  }
}

template <bool HasTimeout>
NCCL_DEVICE_INLINE static ncclResult_t waitImplCore(ncclGinCtx ctx, ncclGinRequest_t& request, cuda::memory_order ord,
                                                    uint32_t* abortFlag, uint64_t timeoutCycles) {
  using nccl::utility::loadConst;
  using nccl::utility::rollingLessThan;
  (void)timeoutCycles; // referenced only when HasTimeout is true
  ncclGinCpuProxyRequest& req = reinterpret_cast<ncclGinCpuProxyRequest&>(request);
  ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true

  // Wait for our posted GFDs to be consumed by the proxy.
  if NCCL_IF_CONSTEXPR (HasTimeout) {
    startCycle = clock64();
    ncclResult_t ret = nccl::gin::proxy::waitForGfdComplete(
      proxyCtx, req.peer, req.nextGfdIdx, cuda::memory_order_relaxed, abortFlag, startCycle, timeoutCycles);
    if (ret != ncclSuccess) return ret;
  } else {
    nccl::gin::proxy::waitForGfdComplete(proxyCtx, req.peer, req.nextGfdIdx, cuda::memory_order_relaxed, abortFlag);
  }

  // Ensure gets are visible by issuing a local flush, but only if there are gets that haven't been flushed yet.
  uint32_t* visibleGets = loadConst(&proxyCtx->lastVisibleGet);
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> lastVisibleGet(visibleGets[req.peer]);
  uint32_t visible = lastVisibleGet.load(cuda::memory_order_relaxed);
  if (rollingLessThan<uint32_t>(visible, req.lastIssuedGet)) {
    ncclGinProxyGfd_t gfd;
    ncclGinProxyOp_t op;
    nccl::gin::proxy::constructProxyOp(op, /*isGet*/ false, /*isFlush*/ true, /*hasInline*/ false,
                                       NCCL_GIN_SIGNAL_TYPE_NONE, ncclGinSignalInc, /*hasCounter*/ false);
    nccl::gin::proxy::buildGfd(&gfd, op, /*srcVal*/ 0, /*hasInline*/ false, 0, nullptr, 0, nullptr, 0, 0, 0, 0, nullptr,
                               0);
    int flushPeer = ctx.rank; // A flush GFD can be posted to any queue. We choose the local queue.
    nccl::gin::proxy::postGfd(ncclCoopThread(), proxyCtx, &gfd, flushPeer);
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      ncclResult_t ret = nccl::gin::proxy::flush(proxyCtx, flushPeer, ord, abortFlag, startCycle, timeoutCycles);
      if (ret != ncclSuccess) return ret;
    } else {
      nccl::gin::proxy::flush(proxyCtx, flushPeer, ord, abortFlag);
    }
    // may move backward in case of concurrent flushes. That's okay.
    lastVisibleGet.store(req.lastIssuedGet, cuda::memory_order_relaxed);
  }
  return ncclSuccess;
}
} // namespace proxy
} // namespace gin
} // namespace nccl

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    nccl::gin::proxy::get<Coop>(coop, proxyCtx, peer, remoteWin, remoteOff, localWin, localOff, bytes, desc);
  }
};

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    (void)hasDescriptor;
    (void)descriptor;
    using nccl::utility::loadConst;
    ncclGinCpuProxyRequest* req = reinterpret_cast<ncclGinCpuProxyRequest*>(outRequest);
    req->peer = peer;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    cuda::atomic_ref<uint32_t, cuda::thread_scope_device> lastIssuedGet(loadConst(&proxyCtx->lastIssuedGet)[peer]);
    // Must be before pi is loaded in case of concurrent gets
    req->lastIssuedGet = lastIssuedGet.load(cuda::memory_order_acquire);

    cuda::atomic_ref<uint32_t, cuda::thread_scope_system> pi(loadConst(&proxyCtx->pis)[peer]);
    req->nextGfdIdx = pi.load(cuda::memory_order_relaxed);
  }
};

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)hasDescriptor;
    (void)descriptor;
    (void)nccl::gin::proxy::waitImplCore</*HasTimeout=*/false>(ctx, request, ord, abortFlag, 0);
  }

  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    return nccl::gin::proxy::waitImplCore</*HasTimeout=*/true>(ctx, request, ord, abortFlag, timeoutCycles);
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    return {nccl::utility::loadConst(&proxyCtx->counters) + counterId, 0};
  }
};

template <>
struct ncclGinApi_SupportsStrongSignal<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return true;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t* counter = nccl::utility::loadConst(&proxyCtx->counters) + counterId;
    *counter = 0;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    return {nccl::utility::loadConst(&proxyCtx->signals) + signalId,
            nccl::utility::loadConst(&proxyCtx->signalOffsets)[signalId]};
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_PROXY> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t* signalPtr = (uint64_t*)ncclGetLocalPointer(signal.vaSignal.ncclWindow, signal.vaSignal.signalOffset);
      *signalPtr = 0;
    } else {
      uint64_t* signalPtr = nccl::utility::loadConst(&proxyCtx->signals) + signal.indexedSignal.signalId;
      uint64_t* offsetPtr = nccl::utility::loadConst(&proxyCtx->signalOffsets) + signal.indexedSignal.signalId;
      *offsetPtr = cuda::atomic_ref<uint64_t>{*signalPtr}.load(cuda::memory_order_relaxed);
    }
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int pe = coop.thread_rank(); pe < ctx.nRanks; pe += coop.size()) {
      ncclGinRequest_t request;
      ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, pe, &request, hasDescriptor, descriptor,
                                                             ncclGinOptFlagsDefault);
      // This is slightly inefficient. If there are prior gets, there will be one flush GFD per peer even though 1
      // suffices.
      ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, request, hasDescriptor, descriptor, ord, abortFlag);
    }
  }

  template <typename Coop>
  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    (void)hasDescriptor;
    (void)descriptor;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    uint64_t startCycle = clock64();
    NVCC_PRAGMA_UNROLL_DISABLED
    for (int pe = coop.thread_rank(); pe < ctx.nRanks; pe += coop.size()) {
      ncclResult_t ret = nccl::gin::proxy::flush(proxyCtx, pe, ord, abortFlag, startCycle, timeoutCycles);
      if (ret != ncclSuccess) return ret;
    }
    return ncclSuccess;
  }
};

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    nccl::gin::proxy::put<Coop, uint64_t>(coop, desc, proxyCtx, peer, dstWin, dstOff, 0, false, srcWin, srcOff, bytes,
                                          signal, signalOp, signalOpArg, hasCounter, counterId, required, given);
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_PROXY> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                      T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    ncclGinProxyGfd_t tmpDesc;
    ncclGinProxyGfd_t* desc = hasDescriptor ? (ncclGinProxyGfd_t*)descriptor : &tmpDesc;
    ncclGinProxyGpuCtx_t* proxyCtx = &((ncclGinProxyGpuCtx_t*)ctx.handle)[ctx.contextId];
    nccl::gin::proxy::put<Coop, T>(coop, desc, proxyCtx, peer, dstWin, dstOff, srcVal, true, nullptr, 0, sizeof(T),
                                   signal, signalOp, signalOpArg, false, 0, required, given);
  }
};

#endif
