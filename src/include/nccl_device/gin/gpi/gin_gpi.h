/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_GPI_H_
#define _NCCL_DEVICE_GIN_GPI_H_

#ifndef DOCA_GPUNETIO_USE_CUDA_WRAPPER
#define DOCA_GPUNETIO_USE_CUDA_WRAPPER
#endif

#ifndef DOCA_VERBS_USE_NET_WRAPPER
#define DOCA_VERBS_USE_NET_WRAPPER
#endif

#ifdef NCCL_DEVICE_GIN_GPI_ENABLE_DEBUG
#define GPI_ENABLE_DEBUG 1
#endif

#include "../gin_device_common.h"
#include "gin_gpi_device_host_common.h"

#ifdef NCCL_DEVICE_GIN_GPI_ENABLE_DEBUG
#include <stdio.h>
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define GPI_USE_TMA_ 1
#endif

struct ncclGinGpiRequest {
  uint64_t waitValue;
  uint64_t* flushCounterPtr;
};
static_assert(sizeof(ncclGinGpiRequest) <= sizeof(ncclGinRequest_t), "ncclGinGpiRequest must fit in ncclGinRequest_t");

namespace nccl {
namespace gin {
namespace gpi {

__device__ static inline gpi_gpu_channel_t* gpi_gpu_channel_get_ptr(ncclGinCtx ctx) {
  const uint64_t ctx_offset = ctx.contextId * (sizeof(gpi_gpu_channel_t) + (ctx.nRanks * sizeof(uint64_t)));
  return (gpi_gpu_channel_t*)(((char*)ctx.handle) + ctx_offset);
}
__device__ static inline uint64_t* gpi_flush_tickets_get_ptr(gpi_gpu_channel_t* gpi_ctx) {
  return (uint64_t*)(((char*)gpi_ctx) + sizeof(gpi_gpu_channel_t));
}

template <typename T, enum gpi_resource_sharing_mode resource_sharing_mode>
__device__ static inline T gpi_atomic_add(T* ptr, T value) {
  if (resource_sharing_mode == GPI_RESOURCE_SHARING_MODE_EXCLUSIVE) {
    T old_val = *ptr;
    *ptr += value;
    return old_val;
  } else if (resource_sharing_mode == GPI_RESOURCE_SHARING_MODE_CTA) {
    cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
    return ptr_aref.fetch_add(value, cuda::std::memory_order_relaxed);
  } else {
    cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
    return ptr_aref.fetch_add(value, cuda::std::memory_order_relaxed);
  }
}

template <enum gpi_resource_sharing_mode resource_sharing_mode>
__device__ static inline uint64_t gpi_gpu_channel_read_ci_shadow(gpi_gpu_channel_t* ch) {
  if (resource_sharing_mode == GPI_RESOURCE_SHARING_MODE_EXCLUSIVE) {
    return ch->queue_.ci_value_;
  } else if (resource_sharing_mode == GPI_RESOURCE_SHARING_MODE_CTA) {
    uint64_t val;
    asm volatile("ld.relaxed.cta.u64 %0, [%1];" : "=l"(val) : "l"(&ch->queue_.ci_value_));
    return val;
  } else {
    uint64_t val;
    asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(val) : "l"(&ch->queue_.ci_value_));
    return val;
  }
}

template <enum gpi_resource_sharing_mode resource_sharing_mode>
__device__ static inline void gpi_gpu_channel_write_ci_shadow(gpi_gpu_channel_t* ch, uint64_t val) {
  if (resource_sharing_mode == GPI_RESOURCE_SHARING_MODE_EXCLUSIVE) {
    ch->queue_.ci_value_ = val;
  } else if (resource_sharing_mode == GPI_RESOURCE_SHARING_MODE_CTA) {
    asm volatile("st.relaxed.cta.u64 [%0], %1;" : : "l"(&ch->queue_.ci_value_), "l"(val));
  } else {
    asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(&ch->queue_.ci_value_), "l"(val));
  }
}

template <enum gpi_resource_sharing_mode resource_sharing_mode>
__device__ static inline uint64_t gpi_gpu_channel_get_pi(gpi_gpu_channel_t* ch, uint64_t slots, uint32_t optFlags) {
  using nccl::utility::loadConst;
  uint64_t pi = gpi_atomic_add<uint64_t, resource_sharing_mode>((uint64_t*)&(ch->queue_.pi_), slots);
  if (optFlags & ncclGinOptFlagsMaySkipCreditCheck) {
    return pi;
  }
  const size_t size = 1UL << loadConst(&ch->queue_.log_depth);
  uint64_t ci_shadow = gpi_gpu_channel_read_ci_shadow<resource_sharing_mode>(ch);

  if ((pi + slots) - ci_shadow > size) {
    ci_shadow = GPI_READ_ONCE(loadConst(&ch->queue_.ci_)->value);
    while ((pi + slots) - ci_shadow > size) {
      ci_shadow = GPI_READ_ONCE(loadConst(&ch->queue_.ci_)->value);
    }

    gpi_gpu_channel_write_ci_shadow<resource_sharing_mode>(ch, ci_shadow);
  }
  return pi;
}
__device__ static inline uint64_t gpi_gpu_channel_get_idx(gpi_gpu_channel_t* ch, uint64_t pi) {
  using nccl::utility::loadConst;
  const size_t size = 1UL << loadConst(&ch->queue_.log_depth);
  return pi & (size - 1);
}
__device__ static inline void gpi_gpu_channel_set_gfd_flag(gpi_gpu_channel_t* ch, gpi_gfd_t* gfd, uint64_t pi) {
  using nccl::utility::loadConst;
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < GPI_GFD_SEG_MAX; i += 1) {
    gfd->segments[i].flag.owner = pi >> loadConst(&ch->queue_.log_depth);
  }
}

__device__ static inline uint64_t gpi_gpu_channel_get_counter_value(gpi_gpu_channel_t* ch, uint16_t idx) {
  return GPI_READ_ONCE(ch->gpu_counter_ptr_[idx].value);
}

__device__ static inline void gpi_gpu_channel_reset_signal(gpi_gpu_channel_t* ch, uint16_t idx) {
  using nccl::utility::loadConst;
  gpi_signal_t* ptr = loadConst(&ch->gpu_signal_ptr_) + idx;
  GPI_WRITE_ONCE(ptr->value, 0);
}
__device__ static inline void gpi_gpu_channel_reset_signal_flag(gpi_gpu_channel_t* ch, uint16_t idx) {
  using nccl::utility::loadConst;
  gpi_signal_t* ptr = loadConst(&ch->gpu_signal_ptr_) + idx;
  GPI_WRITE_ONCE(ptr->flags, 0);
}

__device__ static inline void gpi_gpu_channel_reset_counter(gpi_gpu_channel_t* ch, uint16_t idx) {
  using nccl::utility::loadConst;
  uint64_t* ptr = (uint64_t*)(loadConst(&ch->gpu_counter_ptr_) + idx);
  GPI_WRITE_ONCE(*ptr, 0);
}

__device__ static inline uint64_t gpi_gpu_channel_get_signal_value(gpi_gpu_channel_t* ch, uint16_t idx) {
  using nccl::utility::loadConst;
  gpi_signal_t* sig = loadConst(&ch->gpu_signal_ptr_);
  return GPI_READ_ONCE(sig[idx].value);
}
__device__ static inline bool gpi_gpu_channel_is_signal_flags(gpi_gpu_channel_t* ch, uint16_t idx) {
  using nccl::utility::loadConst;
  gpi_signal_t* sig = loadConst(&ch->gpu_signal_ptr_);
  return (GPI_READ_ONCE(sig[idx].flags) & GPI_SIGNAL_COUNTED_FLAG) == GPI_SIGNAL_COUNTED_FLAG;
}

#ifdef GPI_USE_TMA_
__device__ __inline__ uint32_t __as_ptr_smem(const void* __ptr) {
  // Consider adding debug asserts here.
  return static_cast<uint32_t>(__cvta_generic_to_shared(__ptr));
}

__device__ __inline__ uint64_t __as_ptr_gmem(const void* __ptr) {
  // Consider adding debug asserts here.
  return static_cast<uint64_t>(__cvta_generic_to_global(__ptr));
}

__device__ __inline__ void TmaCopy(void* dstMem, const void* srcMem, uint32_t& size) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2; // 3. "
               :
               : "l"(__as_ptr_gmem(dstMem)), "r"(__as_ptr_smem(srcMem)), "r"(size)
               : "memory");
}

__device__ __inline__ void TmaWait() {
  asm volatile("cp.async.bulk.wait_group 0;" : : :);
}

template <enum gpi_resource_sharing_mode resource_sharing_mode>
__device__ static inline void gpi_gpu_channel_post_gfd_tma(gpi_gpu_channel_t* ch, gpi_gfd_t* gfd, uint32_t optFlags) {
  using nccl::utility::loadConst;
  uint32_t size = sizeof(gpi_gfd_t);
  uint64_t pi = gpi_gpu_channel_get_pi<resource_sharing_mode>(ch, 1, optFlags);
  uint64_t idx = gpi_gpu_channel_get_idx(ch, pi);
  gpi_gpu_channel_set_gfd_flag(ch, gfd, pi);
  void* dst = (void*)&((uint8_t*)loadConst(&ch->queue_.gpu_memic_ptr))[idx * 64];
  TmaCopy(dst, (const void*)gfd, size);
  TmaWait();
}
#else
#define gpi_gpu_channel_post_gfd_tma gpi_gpu_channel_post_gfd_thread
#endif
/**
 * @brief Post a GFD using a single thread.
 *
 * @param [in] ch The GPU channel the GFD is associated with.
 * @param [in] gfd The GFD to post.
 */
template <enum gpi_resource_sharing_mode resource_sharing_mode>
__device__ static inline void gpi_gpu_channel_post_gfd_thread(gpi_gpu_channel_t* ch, gpi_gfd_t* gfd,
                                                              uint32_t optFlags) {
  using nccl::utility::loadConst;
  uint64_t pi = gpi_gpu_channel_get_pi<resource_sharing_mode>(ch, 1, optFlags);
  uint64_t idx = gpi_gpu_channel_get_idx(ch, pi);
  gpi_gpu_channel_set_gfd_flag(ch, gfd, pi);
  void* dst = (void*)&((uint8_t*)loadConst(&ch->queue_.gpu_memic_ptr))[idx * 64];
  gpi_gfd_t* queue_entry = (gpi_gfd_t*)dst;
  NVCC_PRAGMA_UNROLL_AUTO
  for (int i = 0; i < GPI_GFD_SEG_MAX; i += 2) {
    gpi_gfd_segment_t* segment = &gfd->segments[i];
    gpi_gfd_segment_t* queue_entry_segment = &queue_entry->segments[i];
    // Manual PTX for MMIO 128-bit store (b128 needs CUDA 12.3+ / PTX 8.3)
    uint64_t val_lo = segment[0].raw;
    uint64_t val_hi = segment[1].raw;
#if CUDART_VERSION >= 12030
    asm volatile(R"YYY(
      .reg .b128 _v%=;
      mov.b128 _v%=, {%1, %2};
      st.relaxed.sys.global.b128 [%0], _v%=;
    )YYY" ::"l"(queue_entry_segment),
                 "l"(val_lo), "l"(val_hi)
                 : "memory");
#else
    asm volatile("st.relaxed.sys.global.v2.b64 [%0], {%1, %2};" ::"l"(queue_entry_segment), "l"(val_lo), "l"(val_hi)
                 : "memory");
#endif
  }
}

__device__ static inline void gpi_gpu_build_data_transfer_gfd(
  gpi_gfd_t* gfd, uint8_t op, uint8_t op_flags, uint32_t size, uint32_t pe, uint16_t src_handle, uint64_t src_offset,
  uint16_t dst_handle, uint64_t dst_offset, uint16_t counter, uint16_t signal, int64_t signal_value) {
  gfd->segments[GPI_GFD_SEG_HEADER].header.op = op;
  gfd->segments[GPI_GFD_SEG_HEADER].header.op_flags = op_flags;
  gfd->segments[GPI_GFD_SEG_HEADER].header.counter = counter;
  gfd->segments[GPI_GFD_SEG_HEADER].header.signal = signal;
  gfd->segments[GPI_GFD_DATA_DST].dst.pe = pe;
  gfd->segments[GPI_GFD_DATA_DST].dst.size = size;
  gfd->segments[GPI_GFD_DATA_SRC_MEM_HANDLE].src_handle.handle = src_handle;
  gfd->segments[GPI_GFD_DATA_SRC_MEM_HANDLE].src_handle.signal_value_high = signal_value >> 32;
  gfd->segments[GPI_GFD_DATA_SRC_MEM_HANDLE_OFFSET].handle_offset.offset = src_offset;
  gfd->segments[GPI_GFD_DATA_DST_MEM_HANDLE].dst_handle.handle = dst_handle;
  gfd->segments[GPI_GFD_DATA_DST_MEM_HANDLE].dst_handle.signal_value_low = signal_value;
  gfd->segments[GPI_GFD_DATA_DST_MEM_HANDLE_OFFSET].handle_offset.offset = dst_offset;
}

__device__ static inline void gpi_gpu_build_inline_data_transfer_gfd(
  gpi_gfd_t* gfd, uint8_t op, uint8_t op_flags, uint32_t size, uint32_t pe, uint64_t src_data, uint16_t dst_handle,
  uint64_t dst_offset, uint16_t counter, uint16_t signal, int64_t signal_value) {
  gfd->segments[GPI_GFD_SEG_HEADER].header.op = op;
  gfd->segments[GPI_GFD_SEG_HEADER].header.op_flags = op_flags;
  gfd->segments[GPI_GFD_SEG_HEADER].header.counter = counter;
  gfd->segments[GPI_GFD_SEG_HEADER].header.signal = signal;
  gfd->segments[GPI_GFD_DATA_DST].dst.pe = pe;
  gfd->segments[GPI_GFD_DATA_DST].dst.size = size;
  gfd->segments[GPI_GFD_DATA_SRC_MEM_HANDLE].src_handle.signal_value_high = signal_value >> 32;
  gfd->segments[GPI_GFD_DATA_INLINE_DATA_LOW].inline_data.data = src_data;
  gfd->segments[GPI_GFD_DATA_INLINE_DATA_HIGH].inline_data.data = src_data >> 32;
  gfd->segments[GPI_GFD_DATA_DST_MEM_HANDLE].dst_handle.handle = dst_handle;
  gfd->segments[GPI_GFD_DATA_DST_MEM_HANDLE].dst_handle.signal_value_low = signal_value;
  gfd->segments[GPI_GFD_DATA_DST_MEM_HANDLE_OFFSET].handle_offset.offset = dst_offset;
}

__device__ static inline void gpi_gpu_build_control_gfd(gpi_gfd_t* gfd, uint8_t op, uint8_t op_flags, uint16_t counter,
                                                        uint16_t signal) {
  gfd->segments[GPI_GFD_SEG_HEADER].header.op = op | GPI_GFD_OP_CTRL;
  gfd->segments[GPI_GFD_SEG_HEADER].header.op_flags = op_flags;
  gfd->segments[GPI_GFD_SEG_HEADER].header.counter = counter;
  gfd->segments[GPI_GFD_SEG_HEADER].header.signal = signal;
}
__device__ static inline void gpi_gpu_build_pe_flush_gfd(gpi_gfd_t* gfd, uint8_t op_flags, uint32_t pe,
                                                         uint16_t counter) {
  gfd->segments[GPI_GFD_SEG_HEADER].header.op = GPI_GFD_DATA_OP_PE_FLUSH;
  gfd->segments[GPI_GFD_SEG_HEADER].header.op_flags = op_flags;
  gfd->segments[GPI_GFD_SEG_HEADER].header.counter = counter;
  gfd->segments[GPI_GFD_SEG_HEADER].header.signal = 0;
  gfd->segments[GPI_GFD_DATA_DST].dst.pe = pe;
  gfd->segments[GPI_GFD_DATA_DST].dst.size = 0;
}

template <enum gpi_resource_sharing_mode resource_sharing_mode = GPI_RESOURCE_SHARING_MODE_GPU,
          enum gpi_post_mode post_mode = GPI_POST_MODE_THREAD>
__device__ static inline void gpi_gpu_channel_post_gfd(gpi_gpu_channel_t* ch, gpi_gfd_t* gfd, uint32_t optFlags) {
  if (post_mode == GPI_POST_MODE_THREAD) {
    gpi_gpu_channel_post_gfd_thread<resource_sharing_mode>(ch, gfd, optFlags);
  } else if (post_mode == GPI_POST_MODE_TMA) {
    gpi_gpu_channel_post_gfd_tma<resource_sharing_mode>(ch, gfd, optFlags);
  }
}

__device__ inline void GpiFenceAcquireSys() {
  uint32_t dummy;
  const uint32_t val = 0;
  asm volatile("ld.acquire.sys.b32 %0, [%1];" : : "r"(val), "l"(&dummy));
}
__device__ inline void GpiFenceRelease(cuda::thread_scope scope) {
  uint32_t dummy;
  const uint32_t val = 0;

  if (scope == cuda::thread_scope_block) asm volatile("st.release.cta.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
  else if (scope == cuda::thread_scope_device) asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
  else if (scope == cuda::thread_scope_system) asm volatile("st.release.sys.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
  else if (scope == cuda::thread_scope_thread)
    ;  // no-op
}

template <enum gpi_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void putImplMode(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                           size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                           ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                           uint64_t signalOpArg, bool hasCounter, ncclGinCounter_t counterId,
                                           bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                           cuda::thread_scope required, cuda::thread_scope given, uint32_t optFlags) {
  using nccl::utility::loadConst;
  coop.sync();
  if (coop.thread_rank() == 0) {
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;
    gpi_gfd_op_t op = GPI_GFD_DATA_OP_WRITE;
    uint16_t counterId_ = 0;
    uint16_t signalId_ = 0;
    uint64_t signalVal_ = 0;
    uint64_t signalOffset_ = 0;
    uint8_t op_flags = 0;
    uint8_t op_flags_signal = 0;
    if (hasCounter) {
      op_flags = GPI_GFD_DATA_OP_WITH_COUNTER_COUNTED | GPI_GFD_DATA_OP_WITH_COUNTER_WRITEBACK;
      counterId_ = counterId;
    }
    if (hasSignal) {
      signalVal_ = signalOpArg;
      if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
        if (signalOp == ncclGinSignalAdd) {
          op = GPI_GFD_DATA_OP_WRITE_SIGNAL_ADD;
        } else {
          op = GPI_GFD_DATA_OP_WRITE_SIGNAL_COUNTED;
        }
        signalId_ = signal.indexedSignal.signalId;
      } else {
        op_flags_signal = op_flags;
        op_flags = 0;
        signalId_ = (uint16_t)((uint64_t)signal.vaSignal.signalWindow);
        signalOffset_ = signal.vaSignal.signalOffset;
      }
    }
    uint64_t gfd_local[GPI_GFD_SEG_MAX];
    gpi_gfd_t* gfd = hasDescriptor ? (gpi_gfd_t*)descriptor : (gpi_gfd_t*)gfd_local;
    uint16_t gpiSrcHandle = (uint16_t)((uint64_t)srcWin);
    uint16_t gpiDstHandle = (uint16_t)((uint64_t)dstWin);
    if ((required == cuda::thread_scope_system) && (given > required)) {
      GpiFenceRelease(cuda::thread_scope_system);
    } else {
      if (given == cuda::thread_scope_thread) {
        GpiFenceRelease(cuda::thread_scope_device);
      }
    }
    gpi_gpu_channel_t* gpi_ctx = gpi_gpu_channel_get_ptr(ctx);
    if (!hasSignal || (hasSignal && signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)) {
      gpi_gpu_build_data_transfer_gfd(gfd, op, op_flags, bytes, peer, gpiSrcHandle, srcOff, gpiDstHandle, dstOff,
                                      counterId_, signalId_, signalVal_);
      if (hasDescriptor) {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_TMA>(gpi_ctx, gfd, optFlags);
      } else {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, optFlags);
      }
    } else {
      if (hasWins) {
        gpi_gpu_build_data_transfer_gfd(gfd, op, 0, bytes, peer, gpiSrcHandle, srcOff, gpiDstHandle, dstOff, 0, 0, 0);
        if (hasDescriptor) {
          gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_TMA>(gpi_ctx, gfd, optFlags);
        } else {
          gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, optFlags);
        }
      }
      // build signal gfd
      op = GPI_GFD_DATA_OP_AMO_ADD;
      gpi_gpu_build_inline_data_transfer_gfd(gfd, op, op_flags_signal, sizeof(uint64_t), peer, (uint64_t)signalVal_,
                                             signalId_, signalOffset_, counterId_, 0, 0);
      if (hasDescriptor) {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_TMA>(gpi_ctx, gfd, optFlags);
      } else {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, optFlags);
      }
    }
  }
  coop.sync();
}

template <enum gpi_resource_sharing_mode resource_sharing_mode, typename Coop, typename T>
NCCL_DEVICE_INLINE static void putValueImplMode(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin,
                                                size_t dstOff, T srcVal, ncclGinSignalDescriptor signal,
                                                ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool hasDescriptor,
                                                ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                                cuda::thread_scope given, uint32_t optFlags) {
  coop.sync();
  if (coop.thread_rank() == 0) {
    gpi_gfd_op_t op = GPI_GFD_DATA_OP_WRITE_INLINE;
    bool hasSignal = signal.type != NCCL_GIN_SIGNAL_TYPE_NONE;
    uint16_t signalId_ = 0;
    uint64_t signalVal_ = 0;
    uint64_t signalOffset_ = 0;
    if (hasSignal) {
      signalVal_ = signalOpArg;
      if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
        if (signalOp == ncclGinSignalAdd) {
          op = GPI_GFD_DATA_OP_WRITE_INLINE_SIGNAL_ADD;
        } else {
          op = GPI_GFD_DATA_OP_WRITE_INLINE_SIGNAL_COUNTED;
        }
        signalId_ = signal.indexedSignal.signalId;
      } else {
        signalId_ = (uint16_t)((uint64_t)signal.vaSignal.signalWindow);
        signalOffset_ = signal.vaSignal.signalOffset;
      }
    }
    uint64_t gfd_local[GPI_GFD_SEG_MAX];
    gpi_gfd_t* gfd = hasDescriptor ? (gpi_gfd_t*)descriptor : (gpi_gfd_t*)gfd_local;

    uint16_t gpiDstHandle = (uint16_t)((uint64_t)dstWin);
    if (given > required) {
      GpiFenceRelease(required);
    }
    uint64_t src_val_bits = 0;
    __builtin_memcpy(&src_val_bits, &srcVal, sizeof(T));
    gpi_gpu_channel_t* gpi_ctx = gpi_gpu_channel_get_ptr(ctx);
    if (!hasSignal || (hasSignal && signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED)) {
      gpi_gpu_build_inline_data_transfer_gfd(gfd, op, 0, sizeof(T), peer, src_val_bits, gpiDstHandle, dstOff, 0,
                                             signalId_, signalVal_);
      if (hasDescriptor) {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_TMA>(gpi_ctx, gfd, optFlags);
      } else {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, optFlags);
      }
    } else {
      gpi_gpu_build_inline_data_transfer_gfd(gfd, op, 0, sizeof(T), peer, src_val_bits, gpiDstHandle, dstOff, 0, 0, 0);
      if (hasDescriptor) {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_TMA>(gpi_ctx, gfd, optFlags);
      } else {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, optFlags);
      }
      // build signal gfd
      op = GPI_GFD_DATA_OP_AMO_ADD;
      gpi_gpu_build_inline_data_transfer_gfd(gfd, op, 0, sizeof(uint64_t), peer, (uint64_t)signalVal_, signalId_,
                                             signalOffset_, 0, 0, 0);
      if (hasDescriptor) {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_TMA>(gpi_ctx, gfd, optFlags);
      } else {
        gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, optFlags);
      }
    }
  }
  coop.sync();
}

template <bool HasTimeout, enum gpi_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImplModeCore(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                                         ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                                         uint32_t* abortFlag, uint64_t timeoutCycles) {
  using nccl::utility::loadConst;
  using nccl::utility::testAbort;
  (void)timeoutCycles; // referenced only when HasTimeout is true
  uint64_t gfd_local[GPI_GFD_SEG_MAX];
  gpi_gfd_t* gfd = (gpi_gfd_t*)gfd_local;
  gpi_gpu_channel_t* gpi_ctx = gpi_gpu_channel_get_ptr(ctx);
  uint64_t* flush_tickets_ctx = gpi_flush_tickets_get_ptr(gpi_ctx);
  int16_t flush_counter_idx =
    (int16_t)(((uint64_t*)loadConst(&gpi_ctx->gpu_signal_ptr_) - (uint64_t*)loadConst(&gpi_ctx->gpu_counter_ptr_)) /
              sizeof(uint64_t)) -
    ctx.nRanks;
  uint32_t steps = 0;
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();
  NVCC_PRAGMA_UNROLL_DISABLED
  for (int peer = coop.thread_rank(); peer < ctx.nRanks; peer += coop.size()) {
    uint16_t flush_counter_peer_idx = (uint16_t)(peer + flush_counter_idx);
    uint64_t* ticket_peer = &flush_tickets_ctx[peer];
    uint64_t ticket_value = gpi_atomic_add<uint64_t, resource_sharing_mode>(ticket_peer, (uint64_t)1);
    gpi_gpu_build_pe_flush_gfd(gfd, GPI_GFD_DATA_OP_WITH_COUNTER_COUNTED | GPI_GFD_DATA_OP_WITH_COUNTER_WRITEBACK, peer,
                               flush_counter_peer_idx);
    gpi_gpu_channel_post_gfd<resource_sharing_mode, GPI_POST_MODE_THREAD>(gpi_ctx, gfd, 0);
    NVCC_PRAGMA_UNROLL_DISABLED
    while (true) {
      if (GPI_READ_ONCE((gpi_ctx->gpu_counter_ptr_[flush_counter_peer_idx].value)) > ticket_value) break;
      if NCCL_IF_CONSTEXPR (HasTimeout) {
        if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
      }
      if (testAbort(abortFlag, steps)) {
        if NCCL_IF_CONSTEXPR (HasTimeout) {
          cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
          return ncclSuccess;
        } else {
          break; // Original non-timeout path advances to the next peer on abort.
        }
      }
    }
  }
  cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
  return ncclSuccess;
}

template <enum gpi_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static void flushImplMode(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                             ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                             uint32_t* abortFlag) {
  (void)flushImplModeCore</*HasTimeout=*/false, resource_sharing_mode>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                       abortFlag, 0);
}

template <enum gpi_resource_sharing_mode resource_sharing_mode, typename Coop>
NCCL_DEVICE_INLINE static ncclResult_t flushImplMode(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                                     ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                                     uint32_t* abortFlag, uint64_t timeoutCycles) {
  return flushImplModeCore</*HasTimeout=*/true, resource_sharing_mode>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                       abortFlag, timeoutCycles);
}

template <bool HasTimeout>
NCCL_DEVICE_INLINE static ncclResult_t waitImplCore(ncclGinRequest_t& request, cuda::memory_order ord,
                                                    uint32_t* abortFlag, uint64_t timeoutCycles) {
  using nccl::utility::testAbort;
  (void)timeoutCycles; // referenced only when HasTimeout is true
  ncclGinGpiRequest& req = reinterpret_cast<ncclGinGpiRequest&>(request);
  uint32_t steps = 0;
  uint64_t startCycle = 0;
  (void)startCycle; // referenced only when HasTimeout is true
  if NCCL_IF_CONSTEXPR (HasTimeout) startCycle = clock64();
  NVCC_PRAGMA_UNROLL_DISABLED
  while (true) {
    if (GPI_READ_ONCE(req.flushCounterPtr[0]) > req.waitValue) break;
    if NCCL_IF_CONSTEXPR (HasTimeout) {
      if (clock64() - startCycle >= timeoutCycles) return ncclTimeout;
    }
    if (testAbort(abortFlag, steps)) {
      if NCCL_IF_CONSTEXPR (HasTimeout) {
        cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
        return ncclSuccess;
      } else {
        break; // Original non-timeout path falls through to the trailing fence.
      }
    }
  }
  cuda::atomic_thread_fence(ord, cuda::thread_scope_system);
  return ncclSuccess;
}

} // namespace gpi
} // namespace gin
} // namespace nccl

template <>
struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GPI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, bool hasWins, ncclGinWindow_t dstWin,
                                      size_t dstOff, ncclGinWindow_t srcWin, size_t srcOff, size_t bytes,
                                      ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool hasCounter, ncclGinCounter_t counterId, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::thread_scope required,
                                      cuda::thread_scope given, uint32_t optFlags = ncclGinOptFlagsDefault) {
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::gpi::putImplMode<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin,
                                                                       srcOff, bytes, signal, signalOp, signalOpArg,
                                                                       hasCounter, counterId, hasDescriptor, descriptor,
                                                                       required, given, optFlags);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::gpi::putImplMode<GPI_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin,
                                                                 srcOff, bytes, signal, signalOp, signalOpArg,
                                                                 hasCounter, counterId, hasDescriptor, descriptor,
                                                                 required, given, optFlags);
      break;
    default:
      nccl::gin::gpi::putImplMode<GPI_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, hasWins, dstWin, dstOff, srcWin,
                                                                 srcOff, bytes, signal, signalOp, signalOpArg,
                                                                 hasCounter, counterId, hasDescriptor, descriptor,
                                                                 required, given, optFlags);
      break;
    }
  }
};

template <>
struct ncclGinApi_PutValue<NCCL_NET_DEVICE_GIN_GPI> {
  template <typename Coop, typename T>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t dstWin, size_t dstOff,
                                      T srcVal, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                      uint64_t signalOpArg, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope required, cuda::thread_scope given,
                                      uint32_t optFlags = ncclGinOptFlagsDefault) {
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::gpi::putValueImplMode<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, peer, dstWin, dstOff, srcVal,
                                                                            signal, signalOp, signalOpArg,
                                                                            hasDescriptor, descriptor, required, given,
                                                                            optFlags);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::gpi::putValueImplMode<GPI_RESOURCE_SHARING_MODE_CTA>(ctx, coop, peer, dstWin, dstOff, srcVal, signal,
                                                                      signalOp, signalOpArg, hasDescriptor, descriptor,
                                                                      required, given, optFlags);
      break;
    default:
      nccl::gin::gpi::putValueImplMode<GPI_RESOURCE_SHARING_MODE_GPU>(ctx, coop, peer, dstWin, dstOff, srcVal, signal,
                                                                      signalOp, signalOpArg, hasDescriptor, descriptor,
                                                                      required, given, optFlags);
      break;
    }
  }
};

template <>
struct ncclGinApi_SupportsStrongSignal<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static bool call(ncclGinCtx) {
    return true;
  }
};

template <>
struct ncclGinApi_ResetCounter<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    gpi_gpu_channel_t* gpi_ctx = nccl::gin::gpi::gpi_gpu_channel_get_ptr(ctx);
    gpi_gfd_t gfd;
    nccl::gin::gpi::gpi_gpu_build_control_gfd(&gfd, GPI_GFD_CTRL_OP_COUNTER_RESET, 0, counterId, 0);
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE, GPI_POST_MODE_THREAD>(gpi_ctx, &gfd,
                                                                                                          0);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_CTA, GPI_POST_MODE_THREAD>(gpi_ctx, &gfd, 0);
      break;
    default:
      nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_GPU, GPI_POST_MODE_THREAD>(gpi_ctx, &gfd, 0);
      break;
    }
    nccl::gin::gpi::gpi_gpu_channel_reset_counter(gpi_ctx, counterId);
  }
};

template <>
struct ncclGinApi_ResetSignal<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinSignalDescriptor signal) {
    gpi_gpu_channel_t* gpi_ctx = nccl::gin::gpi::gpi_gpu_channel_get_ptr(ctx);
    if (signal.type == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
      uint16_t signalId = signal.indexedSignal.signalId;
      if (nccl::gin::gpi::gpi_gpu_channel_is_signal_flags(gpi_ctx, signalId) == true) {
        gpi_gfd_t gfd;
        nccl::gin::gpi::gpi_gpu_build_control_gfd(&gfd, GPI_GFD_CTRL_OP_SIGNAL_RESET, 0, 0, signalId);
        switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
        case NCCL_GIN_RESOURCE_SHARING_THREAD:
          nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE, GPI_POST_MODE_THREAD>(gpi_ctx,
                                                                                                              &gfd, 0);
          break;
        case NCCL_GIN_RESOURCE_SHARING_CTA:
          nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_CTA, GPI_POST_MODE_THREAD>(gpi_ctx, &gfd,
                                                                                                        0);
          break;
        default:
          nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_GPU, GPI_POST_MODE_THREAD>(gpi_ctx, &gfd,
                                                                                                        0);
          break;
        }
        while (nccl::gin::gpi::gpi_gpu_channel_get_signal_value(gpi_ctx, signalId) != 0) {
        }
        nccl::gin::gpi::gpi_gpu_channel_reset_signal_flag(gpi_ctx, signalId);
      } else {
        nccl::gin::gpi::gpi_gpu_channel_reset_signal(gpi_ctx, signalId);
      }
    } else {
      uint64_t* signalPtr = (uint64_t*)ncclGetLocalPointer(signal.vaSignal.ncclWindow, signal.vaSignal.signalOffset);
      *signalPtr = 0;
    }
  }
};

template <>
struct ncclGinApi_GetCounterPtr<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinCounter_t counterId) {
    using nccl::utility::loadConst;
    gpi_gpu_channel_t* gpi_ctx = nccl::gin::gpi::gpi_gpu_channel_get_ptr(ctx);
    return {(uint64_t*)(loadConst(&gpi_ctx->gpu_counter_ptr_) + counterId), 0};
  }
};

template <>
struct ncclGinApi_GetSignalPtr<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static ncclGinOffsetPtr call(ncclGinCtx ctx, ncclGinSignal_t signalId) {
    using nccl::utility::loadConst;
    gpi_gpu_channel_t* gpi_ctx = nccl::gin::gpi::gpi_gpu_channel_get_ptr(ctx);
    return {(uint64_t*)(loadConst(&gpi_ctx->gpu_signal_ptr_) + signalId), 0};
  }
};

template <>
struct ncclGinApi_Flush<NCCL_NET_DEVICE_GIN_GPI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, bool hasDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::memory_order ord, uint32_t* abortFlag) {
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::gpi::flushImplMode<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                         abortFlag);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::gpi::flushImplMode<GPI_RESOURCE_SHARING_MODE_CTA>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                   abortFlag);
      break;
    default:
      nccl::gin::gpi::flushImplMode<GPI_RESOURCE_SHARING_MODE_GPU>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                   abortFlag);
      break;
    }
  }

  template <typename Coop>
  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, Coop coop, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      return nccl::gin::gpi::flushImplMode<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE>(ctx, coop, hasDescriptor, descriptor,
                                                                                ord, abortFlag, timeoutCycles);
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      return nccl::gin::gpi::flushImplMode<GPI_RESOURCE_SHARING_MODE_CTA>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                          abortFlag, timeoutCycles);
    default:
      return nccl::gin::gpi::flushImplMode<GPI_RESOURCE_SHARING_MODE_GPU>(ctx, coop, hasDescriptor, descriptor, ord,
                                                                          abortFlag, timeoutCycles);
    }
  }
};

template <>
struct ncclGinApi_Get<NCCL_NET_DEVICE_GIN_GPI> {
  template <typename Coop>
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, Coop coop, int peer, ncclGinWindow_t remoteWin, size_t remoteOff,
                                      ncclGinWindow_t localWin, size_t localOff, size_t bytes, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags = ncclGinOptFlagsDefault) {
    using nccl::utility::loadConst;
    coop.sync();
    if (coop.thread_rank() == 0) {
      gpi_gpu_channel_t* gpi_ctx = nccl::gin::gpi::gpi_gpu_channel_get_ptr(ctx);
      uint64_t gfd_local[GPI_GFD_SEG_MAX];
      gpi_gfd_t* gfd = hasDescriptor ? (gpi_gfd_t*)descriptor : (gpi_gfd_t*)gfd_local;

      uint16_t gpiSrcHandle = (uint16_t)((uint64_t)localWin);
      uint16_t gpiDstHandle = (uint16_t)((uint64_t)remoteWin);
      nccl::gin::gpi::gpi_gpu_build_data_transfer_gfd(gfd, GPI_GFD_DATA_OP_READ, 0, bytes, peer, gpiSrcHandle, localOff,
                                                      gpiDstHandle, remoteOff, 0, 0, 0);
      switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
      case NCCL_GIN_RESOURCE_SHARING_THREAD:
        nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE, GPI_POST_MODE_THREAD>(
          gpi_ctx, gfd, optFlags);
        break;
      case NCCL_GIN_RESOURCE_SHARING_CTA:
        nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_CTA, GPI_POST_MODE_THREAD>(gpi_ctx, gfd,
                                                                                                      optFlags);
        break;
      default:
        nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_GPU, GPI_POST_MODE_THREAD>(gpi_ctx, gfd,
                                                                                                      optFlags);
        break;
      }
    }
    coop.sync();
  }
};

template <>
struct ncclGinApi_FlushAsync<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, int peer, ncclGinRequest_t* outRequest, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
    using nccl::utility::loadConst;
    uint64_t gfd_local[GPI_GFD_SEG_MAX];
    gpi_gfd_t* gfd = hasDescriptor ? (gpi_gfd_t*)descriptor : (gpi_gfd_t*)gfd_local;
    ncclGinGpiRequest* req = reinterpret_cast<ncclGinGpiRequest*>(outRequest);
    gpi_gpu_channel_t* gpi_ctx = nccl::gin::gpi::gpi_gpu_channel_get_ptr(ctx);
    uint64_t* flush_tickets_ctx = nccl::gin::gpi::gpi_flush_tickets_get_ptr(gpi_ctx);
    int16_t flush_counter_idx =
      (int16_t)(((uint64_t*)loadConst(&gpi_ctx->gpu_signal_ptr_) - (uint64_t*)loadConst(&gpi_ctx->gpu_counter_ptr_)) /
                sizeof(uint64_t)) -
      ctx.nRanks;
    uint16_t flush_counter_peer_idx = (uint16_t)(peer + flush_counter_idx);
    // GPI_READ_ONCE(ticket_peer);
    uint64_t flush_ticket_value =
      nccl::gin::gpi::gpi_atomic_add<uint64_t, GPI_RESOURCE_SHARING_MODE_GPU>(&flush_tickets_ctx[peer], (uint64_t)1);
    req->flushCounterPtr = (uint64_t*)(loadConst(&gpi_ctx->gpu_counter_ptr_) + flush_counter_peer_idx);
    req->waitValue = flush_ticket_value;
    nccl::gin::gpi::gpi_gpu_build_pe_flush_gfd(
      gfd, GPI_GFD_DATA_OP_WITH_COUNTER_COUNTED | GPI_GFD_DATA_OP_WITH_COUNTER_WRITEBACK, peer, flush_counter_peer_idx);
    switch ((ncclGinResourceSharingMode)ctx.resourceSharingMode) {
    case NCCL_GIN_RESOURCE_SHARING_THREAD:
      nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_EXCLUSIVE, GPI_POST_MODE_THREAD>(gpi_ctx, gfd,
                                                                                                          optFlags);
      break;
    case NCCL_GIN_RESOURCE_SHARING_CTA:
      nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_CTA, GPI_POST_MODE_THREAD>(gpi_ctx, gfd,
                                                                                                    optFlags);
      break;
    default:
      nccl::gin::gpi::gpi_gpu_channel_post_gfd<GPI_RESOURCE_SHARING_MODE_GPU, GPI_POST_MODE_THREAD>(gpi_ctx, gfd,
                                                                                                    optFlags);
      break;
    }
  }
};

template <>
struct ncclGinApi_Wait<NCCL_NET_DEVICE_GIN_GPI> {
  NCCL_DEVICE_INLINE static void call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                      ncclGinDescriptorSmem* descriptor, cuda::memory_order ord, uint32_t* abortFlag) {
    (void)nccl::gin::gpi::waitImplCore</*HasTimeout=*/false>(request, ord, abortFlag, 0);
  }

  NCCL_DEVICE_INLINE static ncclResult_t call(ncclGinCtx ctx, ncclGinRequest_t& request, bool hasDescriptor,
                                              ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                              uint32_t* abortFlag, uint64_t timeoutCycles) {
    return nccl::gin::gpi::waitImplCore</*HasTimeout=*/true>(request, ord, abortFlag, timeoutCycles);
  }
};

#endif /* _NCCL_DEVICE_GIN_GPI_H_ */
