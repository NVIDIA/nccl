/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_GDAKI_H_
#define _NCCL_DEVICE_GIN_GDAKI_H_

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

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
#include <stdio.h>
#endif

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins,
                                      ncclGinWindow_t dstWin, size_t dstOff, ncclGinWindow_t srcWin,
                                      size_t srcOff, size_t bytes, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasCounter,
                                      ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given) {
    using nccl::utility::loadConst;

    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinGdakiGPUContext* gdaki = (struct ncclGinGdakiGPUContext*)ctx.handle;
      doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
      doca_gpu_dev_verbs_qp* companion_qp;
      ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;
      ncclGinGdakiMemHandle* srcMh = (ncclGinGdakiMemHandle*)srcWin;

      doca_gpu_dev_verbs_addr raddr, laddr;
      if (hasWins) {
        raddr.addr = dstOff;
        raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);
        laddr.addr = srcOff, laddr.key = loadConst(&srcMh->lkey);
      }

      doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
      if (hasSignal) {
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        sig_raddr.addr = sizeof(uint64_t) * signalId;
        sig_raddr.key = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
        sig_laddr.addr = 0;
        sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
      }

      doca_gpu_dev_verbs_addr counter_raddr, counter_laddr;
      if (hasCounter) {
        companion_qp = loadConst(&gdaki->companion_gdqp) + peer;
        counter_raddr.addr = sizeof(uint64_t) * counterId;
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
          doca_gpu_dev_verbs_put_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(
            qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr,
            counter_laddr, 1);
        } else if (hasSignal) {
          doca_gpu_dev_verbs_put_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(
            qp, raddr, laddr, bytes, sig_raddr, sig_laddr, signalOpArg);
        } else if (hasCounter) {
          doca_gpu_dev_verbs_put_counter(qp, raddr, laddr, bytes, companion_qp, counter_raddr,
                                              counter_laddr, 1);
        } else {
          doca_gpu_dev_verbs_put(qp, raddr, laddr, bytes);
        }
      } else {
        if (hasCounter) {
          doca_gpu_dev_verbs_signal_counter<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(
            qp, sig_raddr, sig_laddr, signalOpArg, companion_qp, counter_raddr, counter_laddr, 1);
        } else {
          doca_gpu_dev_verbs_signal<DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(
            qp, sig_raddr, sig_laddr, signalOpArg);
        }
      }

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
      doca_gpu_dev_verbs_wait(qp);
      if (hasCounter) doca_gpu_dev_verbs_wait(companion_qp);
#endif
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                      size_t dstOff, T srcVal, bool hasSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given) {
    using nccl::utility::loadConst;

    coop.sync();
    if (coop.thread_rank() == 0) {
      ncclGinGdakiGPUContext* gdaki = (struct ncclGinGdakiGPUContext*)ctx.handle;
      doca_gpu_dev_verbs_qp* qp = loadConst(&gdaki->gdqp) + peer;
      ncclGinGdakiMemHandle* dstMh = (ncclGinGdakiMemHandle*)dstWin;

      doca_gpu_dev_verbs_addr raddr;
      raddr.addr = dstOff;
      raddr.key = loadConst(loadConst(&dstMh->rkeys) + peer);

      doca_gpu_dev_verbs_addr sig_raddr, sig_laddr;
      if (hasSignal) {
        if (signalOp == ncclGinSignalInc) signalOpArg = 1;
        sig_raddr.addr = sizeof(uint64_t) * signalId;
        sig_raddr.key = loadConst(loadConst(&gdaki->signals_table.rkeys) + peer);
        sig_laddr.addr = 0;
        sig_laddr.key = loadConst(&gdaki->sink_buffer_lkey);
      }

      // cuda::thread_scope_system has the lowest value
      if ((required == cuda::thread_scope_system) && (given > required)) {
        doca_gpu_dev_verbs_fence_release<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS>();
      }

      if (hasSignal) {
        doca_gpu_dev_verbs_p_signal<T, DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD>(
          qp, raddr, srcVal, sig_raddr, sig_laddr, signalOpArg);
      } else {
        doca_gpu_dev_verbs_p(qp, raddr, srcVal);
      }

#ifdef NCCL_DEVICE_GIN_GDAKI_ENABLE_DEBUG
      doca_gpu_dev_verbs_wait(qp);
#endif
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = (ncclGinGdakiGPUContext*)ctx.handle;
    loadConst(&gdaki->counters_table.buffer)[counterId] = 0;
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = (ncclGinGdakiGPUContext*)ctx.handle;
    loadConst(&gdaki->signals_table.buffer)[signalId] = 0;
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = (ncclGinGdakiGPUContext*)ctx.handle;
    return loadConst(&gdaki->counters_table.buffer) + counterId;
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_GDAKI> {
  NCCL_DEVICE_INLINE static uint64_t* call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = (ncclGinGdakiGPUContext*)ctx.handle;
    return loadConst(&gdaki->signals_table.buffer) + signalId;
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_GDAKI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, cuda::memory_order ord) {
    using nccl::utility::loadConst;
    ncclGinGdakiGPUContext* gdaki = (ncclGinGdakiGPUContext*)ctx.handle;
    doca_gpu_dev_verbs_qp* qps = loadConst(&gdaki->gdqp);
#pragma unroll 1
    for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
      doca_gpu_dev_verbs_wait(qps + peer);
    }
  }
};

#endif /* _NCCL_DEVICE_GIN_GDAKI_H_ */
