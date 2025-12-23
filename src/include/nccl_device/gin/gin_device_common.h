/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_GIN_DEVICE_COMMON_H_
#define _NCCL_GIN_DEVICE_COMMON_H_

#include "../net_device.h"
#include "../utility.h"
#include "gin_device_host_common.h"

#if CUDA_VERSION >= 12080 && __CUDA_ARCH__ >= 900
#define NCCL_GIN_HAS_FENCE_ACQUIRE_RELEASE_PTX 1
#endif

#ifndef NCCL_GIN_PROXY_ENABLE
#define NCCL_GIN_PROXY_ENABLE 1
#endif

#ifndef NCCL_GIN_GDAKI_ENABLE
#if CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700
#define NCCL_GIN_GDAKI_ENABLE 1
#else
#define NCCL_GIN_GDAKI_ENABLE 0
#endif
#endif

#define NCCL_GIN_BACKEND_MASK_ALL                                               \
  (((NCCL_GIN_PROXY_ENABLE) ? 1u : 0u) << (unsigned)NCCL_NET_DEVICE_GIN_PROXY | \
   ((NCCL_GIN_GDAKI_ENABLE) ? 1u : 0u) << (unsigned)NCCL_NET_DEVICE_GIN_GDAKI)

struct ncclGinCtx {
  unsigned backendMask;
  ncclNetDeviceType backend;
  int rank;
  int nRanks;
  void* handle;
};

template <unsigned backendMask>
struct ncclGinCtx_M : ncclGinCtx {};

struct ncclGinDescriptorSmem {
  alignas(16) char space[64];
};

#if NCCL_CHECK_CUDACC
template <ncclNetDeviceType backend>
struct ncclGinApi_Put {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_PutValue {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcData, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_GetSignalPtr {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx, int peer, ncclGinSignal_t signalId);
};
template <ncclNetDeviceType backend>
struct ncclGinApi_GetCounterPtr {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx, int peer, ncclGinCounter_t counterId);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_ResetSignal {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinSignal_t signalId);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_ResetCounter {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, ncclGinCounter_t counterId);
};

template <ncclNetDeviceType backend>
struct ncclGinApi_Flush {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx, Coop, cuda::memory_order ord);
};
#endif

#if NCCL_CHECK_CUDACC
template <template <ncclNetDeviceType> typename ApiFn, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCallImpl(unsigned beMask, ncclGinCtx ctx, Arg&&... arg) {
  bool singleton = (beMask & (beMask - 1)) == 0;  // Only one bit set
  switch (singleton ? __popc(beMask - 1) : (int)ctx.backend) {
#if NCCL_GIN_PROXY_ENABLE
    case (int)NCCL_NET_DEVICE_GIN_PROXY:
      if (!(1 & (beMask >> (int)NCCL_NET_DEVICE_GIN_PROXY))) __builtin_unreachable();
      return ApiFn<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, static_cast<Arg&&>(arg)...);
#endif
#if NCCL_GIN_GDAKI_ENABLE
    case (int)NCCL_NET_DEVICE_GIN_GDAKI:
      if (!(1 & (beMask >> (int)NCCL_NET_DEVICE_GIN_GDAKI))) __builtin_unreachable();
      return ApiFn<NCCL_NET_DEVICE_GIN_GDAKI>::call(ctx, static_cast<Arg&&>(arg)...);
#endif
    default:
      __builtin_unreachable();
  }
}

template <template <ncclNetDeviceType> typename ApiFn, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCall(ncclGinCtx ctx, Arg&&... arg) {
  return ncclGinCallImpl<ApiFn>(ctx.backendMask, ctx, static_cast<Arg&&>(arg)...);
}

template <template <ncclNetDeviceType> typename ApiFn, unsigned beMask, typename... Arg>
NCCL_DEVICE_INLINE static decltype(auto) ncclGinCall(ncclGinCtx_M<beMask> ctx, Arg&&... arg) {
  return ncclGinCallImpl<ApiFn>(beMask, ctx, static_cast<Arg&&>(arg)...);
}
#endif

#endif
