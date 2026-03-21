/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_GDAKI_H_
#define _NCCL_DEVICE_GIN_GDAKI_H_

#include <cstdint>
#ifndef DOCA_VERBS_USE_CUDA_WRAPPER
#define DOCA_VERBS_USE_CUDA_WRAPPER
#endif

#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
#define DOCA_GPUNETIO_VERBS_ENABLE_DEBUG 1
#endif

#include "../gin_device_common.h"
#include "gin_gdaki_device_host_common.h"
#include "doca_gpunetio/doca_gpunetio_device.h"

struct ncclGinGdakiRequest {
  int peer;
  doca_gpu_dev_verbs_ticket_t docaTicket;
};
static_assert(sizeof(ncclGinGdakiRequest) <= sizeof(ncclGinRequest_t), "ncclGinGdakiRequest must fit in ncclGinRequest_t");

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
#include <stdio.h>
#endif

namespace nccl {
namespace gin {
namespace gdaki {

NCCL_DEVICE_INLINE uint32_t docaOptFlagsFromGinOptFlags(uint32_t ginOptFlags) {
  return DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT
    | (!!(ginOptFlags & ncclGinOptFlagsMaySkipCreditCheck) * DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_AVAILABILITY_CHECK)
    | (!!(ginOptFlags & ncclGinOptFlagsAggregateRequests) * DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_SKIP_DB_RINGING);
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                              ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                              size_t srcOff, size_t bytes, bool hasSignal,
                                              size_t signalOffset, __be32 signalKey, ncclGinSignalOp_t signalOp,
                                              uint64_t signalOpArg, bool hasCounter,
                                              ncclGinCounter_t counterId, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor,
                                              cuda::thread_scope required, cuda::thread_scope given,
                                              uint32_t optFlags) {
  using nccl::utility::loadConst;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
    doca_gpu_dev_verbs_qp* companion_qp;
    ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;
    ncclGinGdakiMemHandle* srcMh = (ncclGinGdakiMemHandle*)srcWin;
    uint32_t codeOpt = nccl::gin::gdaki::docaOptFlagsFromGinOptFlags(optFlags);

    doca_gpu_dev_verbs_addr raddr, laddr;
    if (hasWins) {
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
      laddr.addr = srcOff, laddr.key = loadConst(&srcMh->lkey);
    }

    doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
    if (hasSignal) {
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      sig_raddr.addr = signalOffset;
      sig_raddr.key = signalKey;
      sig_laddr.addr = 0;
      sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
    }

    doca_gpu_dev_verbs_addr counter_raddr, counter_laddr;
    if (hasCounter) {
      companion_qp = loadConst(&gdaki->companion_gdqp) + peer;
      counter_raddr.addr = sizeof(uint64_t) * (counterId + loadConst(&gdaki->counters_table.offset));
      counter_raddr.key = loadConst(loadConst(&gdaki->counters_table.rkeys) + ctx.rank);
      counter_laddr.addr = 0;
      counter_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
    }

    // cuda::thread_scope_system has the lowest value
    if ((required == cuda::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }

    if (hasWins) {
      if (hasSignal && hasCounter) {
        doca_gpu_dev_verbs_put_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr,
          counter_laddr, 1, codeOpt);
      } else if (hasSignal) {
        doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg, codeOpt);
      } else if (hasCounter) {
        doca_gpu_dev_verbs_put_counter<resource_sharing_mode>(
          qp, raddr, laddr, bytes, companion_qp, counter_raddr, counter_laddr, 1, codeOpt);
      } else {
        doca_gpu_dev_verbs_put<resource_sharing_mode>(qp, raddr, laddr, bytes, codeOpt);
      }
    } else {
      if (hasCounter) {
        doca_gpu_dev_verbs_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr, counter_laddr, 1, codeOpt);
      } else {
        doca_gpu_dev_verbs_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
          qp, sig_raddr, sig_laddr, signalOpArg, codeOpt);
      }
    }

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
    doca_gpu_dev_verbs_wait(qp);
    if (hasCounter) doca_gpu_dev_verbs_wait(companion_qp);
#endif
  }
  coop.sync();
}

template <typename Coop>
NCCL_DEVICE_INLINE static void putImpl(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                              ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                              size_t srcOff, size_t bytes, bool hasSignal,
                                              size_t signalOffset, __be32 signalKey, ncclGinSignalOp_t signalOp,
                                              uint64_t signalOpArg, bool hasCounter,
                                              ncclGinCounter_t counterId, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor,
                                              cuda::thread_scope required, cuda::thread_scope given,
                                              uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasCounter, counterId, hasDescriptor, descriptor,
        required, given, optFlags);
      break;
    default:
      putImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasCounter, counterId, hasDescriptor, descriptor,
        required, given, optFlags);
      break;
  }
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcData, bool hasSignal,
                                      size_t signalOffset, __be32 signalKey, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags) {
  using nccl::utility::loadConst;

  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
    ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;
    uint32_t codeOpt = nccl::gin::gdaki::docaOptFlagsFromGinOptFlags(optFlags);

    doca_gpu_dev_verbs_addr raddr;
    raddr.addr = dstOff;
    raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);

    doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
    if (hasSignal) {
      if (signalOp == ncclGinSignalInc) signalOpArg = 1;
      sig_raddr.addr = signalOffset;
      sig_raddr.key = signalKey;
      sig_laddr.addr = 0;
      sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
    }

    // cuda::thread_scope_system has the lowest value
    if ((required == cuda::thread_scope_system) && (given > required)) {
      doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
    }

    if (hasSignal) {
      doca_gpu_dev_verbs_p_signal<T, DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD, resource_sharing_mode>(
        qp, raddr, srcData, sig_raddr, sig_laddr, signalOpArg, codeOpt);
    } else {
      doca_gpu_dev_verbs_p<T, resource_sharing_mode>(qp, raddr, srcData, codeOpt);
    }

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
    doca_gpu_dev_verbs_wait(qp);
#endif
  }
  coop.sync();
}

template <typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImpl(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcData, bool hasSignal,
                                      size_t signalOffset, __be32 signalKey, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags) {
  switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA>(
        ctx, coop, peer, dstWin, dstOff, srcData,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
    default:
      putValueImplMode<DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(
        ctx, coop, peer, dstWin, dstOff, srcData,
        hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
        hasDescriptor, descriptor, required, given, optFlags);
      break;
  }
}

} // namespace gdaki
} // namespace gin
} // namespace nccl


template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t* outRequest) {
    using nccl::utility::loadConst;
    ncclGinGdakiRequest* req = (ncclGinGdakiRequest*)outRequest;
    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + req->peer;
    doca_gpu_dev_verbs_addr uninitialized_daddr{};
    doca_gpu_dev_verbs_get_wait(qp, uninitialized_daddr, &req->docaTicket);
  }
};

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes,
                                      bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      ncclGinRequest_t* outRequest, uint32_t optFlags) {
    using nccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
      doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
      ncclGinGdakiMemHandle* remoteMh = (ncclGinGdakiMemHandle*)remoteWin;
      ncclGinGdakiMemHandle* localMh = (ncclGinGdakiMemHandle*)localWin;
      doca_gpu_dev_verbs_addr raddr, laddr;
      raddr.addr = remoteOff;
      raddr.key = loadConst(loadConst(&remoteMh->rkeys) + peer);
      laddr.addr = localOff;
      laddr.key = loadConst(&localMh->lkey);
      doca_gpu_dev_verbs_addr uninitialized_daddr{};
      doca_gpu_dev_verbs_ticket_t doca_out_ticket;
      uint32_t codeOpt = nccl::gin::gdaki::docaOptFlagsFromGinOptFlags(optFlags);
      doca_gpu_dev_verbs_get(
          qp, raddr, laddr, bytes, uninitialized_daddr, &doca_out_ticket, codeOpt);

      if (outRequest) {
        ncclGinGdakiRequest* req = (ncclGinGdakiRequest*)outRequest;
        req->peer = peer;
        req->docaTicket = doca_out_ticket;
      }
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    size_t signalOffset = 0;
    __be32 signalKey = 0;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
      ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
      signalOffset = sizeof(uint64_t) * (signal.indexedSignal.signalId + loadConst(&gdaki->signals_table.offset));
      signalKey = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
    } else if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
      ncclGinGdakiMemHandle* signalMh = (ncclGinGdakiMemHandle*)signal.vaSignal.signalWindow;
      signalKey = loadConst(loadConst(&signalMh->rkeys) + peer);
      signalOffset = signal.vaSignal.signalOffset;
    }
    nccl::gin::gdaki::putImpl(
      ctx, coop, peer, hasWins, dstWin, dstOff, srcWin, srcOff, bytes,
      hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
      hasCounter, counterId, hasDescriptor, descriptor,
      required, given, optFlags
    );
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    size_t signalOffset = 0;
    __be32 signalKey = 0;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
      ncclGinGdakiMemHandle* signalMh = (ncclGinGdakiMemHandle*)signal.vaSignal.signalWindow;
      signalKey = loadConst(loadConst(&signalMh->rkeys) + peer);
      signalOffset = signal.vaSignal.signalOffset;
    } else if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
      ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
      signalOffset = sizeof(uint64_t) * (signal.indexedSignal.signalId + loadConst(&gdaki->signals_table.offset));
      signalKey = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
    }
    nccl::gin::gdaki::putValueImpl(
      ctx, coop, peer, dstWin, dstOff, srcVal,
      hasSignal, signalOffset, signalKey, signalOp, signalOpArg,
      hasDescriptor, descriptor, required, given, optFlags
    );
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    loadConst(&gdaki->counters_table.buffer)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    using nccl::utility::loadConst;
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_VA) {
      uint64_t* signalPtr = (uint64_t*)ncclGetLocalPointer(signal.vaSignal.ncclWindow, signal.vaSignal.signalOffset);
      *signalPtr = 0;
    } else {
      ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
      loadConst(&gdaki->signals_table.buffer)[signal.indexedSignal.signalId] = 0;
    }
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    return loadConst(&gdaki->counters_table.buffer) + counterId;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    return loadConst(&gdaki->signals_table.buffer) + signalId;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord, uint32_t* abortFlag) {
    using nccl::utility::loadConst;
    using nccl::utility::testAbort;

    ncclGinGdakiGPUContext* gdaki = &((struct ncclGinGdakiGPUContext*)ctx.handle)[ctx.contextId];
    doca_gpu_dev_verbs_qp* qps = loadConst(&gdaki->gdqp);

    if (abortFlag) {
      uint32_t steps = 0;
      #pragma unroll 1
      for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
        int status = EBUSY;
        uint64_t ticket = doca_gpu_dev_verbs_atomic_read<uint64_t, DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU>(&qps[peer].sq_rsvd_index);
        if (ticket == 0)
          return;
        --ticket;
        while (status != 0 && !testAbort(abortFlag, steps)) {
          status = doca_gpu_dev_verbs_poll_one_cq_at(&qps[peer].cq_sq, ticket);
        }
      }
    } else {
      #pragma unroll 1
      for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
        doca_gpu_dev_verbs_wait(qps + peer);
      }
    }
  }
};

#endif /* _NCCL_DEVICE_GIN_GDAKI_H_ */
