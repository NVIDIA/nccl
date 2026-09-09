/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_CFT__FUNCS_H_
#define _NCCL_DEVICE_CFT__FUNCS_H_

#include "cft__types.h"
#include "../coop.h"
#include <assert.h>

#ifdef __CUDACC__
#include <type_traits>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(CUDART_VERSION) && CUDART_VERSION >= 13030
#define NCCL_CFT_ENABLE 1
// #define NCCL_DEVICE_CFT_ENABLE_DEBUG
#else
#define NCCL_CFT_ENABLE 0
#endif

namespace nccl {
namespace cft {
namespace internal {

template <typename Coop>
NCCL_DEVICE_INLINE bool elected(Coop coop) {
  return coop.thread_rank() == 0;
}

template <>
NCCL_DEVICE_INLINE bool elected(ncclCoopThread coop) {
  return true;
}

template <>
NCCL_DEVICE_INLINE bool elected(ncclCoopWarp coop) {
  int chosen;
  asm volatile("{\n\t"
               ".reg .pred p;\n\t"
               "elect.sync _|p, %1;\n\t"
               "selp.b32 %0, 1, 0, p;\n"
               "}"
               : "=r"(chosen)
               : "r"(0xFFFFFFFFu));
  return chosen;
}

NCCL_DEVICE_INLINE void unsupported() {
  assert(false && "CFT device helpers require CUDA Toolkit support for fabric PTX instructions.");
}

#if NCCL_CFT_ENABLE
NCCL_DEVICE_INLINE uint32_t smemAddr(void* ptr) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
  assert(__isShared(ptr));
#endif
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

NCCL_DEVICE_INLINE uint32_t smemAddr(ncclCftSmem& cftSmem) {
  return smemAddr(&cftSmem.bar);
}

NCCL_DEVICE_INLINE void waitMbarrier(ncclCftSmem& cftSmem, uint32_t phaseParity, int* hasReport, uint32_t* report) {
  int ready = 0;
  *hasReport = 0;
  *report = 0;
  do {
    // clang-format off
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 ".reg .pred hasReport;\n\t"
                 ".reg .u8 s8;\n\t"
                 "mbarrier.try_wait.parity.phase_type::primary.acquire.cta.shared::cta.b64 p|hasReport, s8, [%3], %4;\n\t"
                 "selp.b32 %0, 1, 0, p;\n\t"
                 "@p cvt.u32.u8 %2, s8;\n\t"
                 "@p selp.b32 %1, 1, 0, hasReport;\n\t"
                 "}"
                 : "=r"(ready), "+r"(*hasReport), "+r"(*report)
                 : "r"(smemAddr(cftSmem)), "r"(phaseParity)
                 : "memory");
    // clang-format on
  } while (!ready);
}

#endif

NCCL_DEVICE_INLINE const char* redOpUnsupported() {
  return "CFT reductions need a RedOp-to-PTX mapping before this overload can be instantiated.";
}

#if NCCL_CFT_ENABLE
#define NCCL_CFT_DEFINE_RED(NAME, PTX_MULTIMEM, PTX_OP, PTX_TYPE) \
  NCCL_DEVICE_INLINE void red_##NAME(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                     ncclCftSmem& cftSmem) { \
    uint32_t srcSmemPtr = smemAddr(src); \
    uint32_t mbarPtr = smemAddr(cftSmem); \
    asm volatile("fabric.try_red.async" PTX_MULTIMEM \
                 ".shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys." PTX_OP "." PTX_TYPE \
                 " [%0, %1], [%2], %3, [%4];" \
                 : \
                 : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr) \
                 : "memory"); \
  }

#define NCCL_CFT_DEFINE_RED_CP_MASK(NAME, PTX_OP, PTX_TYPE) \
  NCCL_DEVICE_INLINE void red_cp_mask_##NAME(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                             ncclCftSmem& cftSmem, uint16_t cpMask) { \
    uint32_t srcSmemPtr = smemAddr(src); \
    uint32_t mbarPtr = smemAddr(cftSmem); \
    asm volatile("fabric.try_red.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask" \
                 ".relaxed.sys." PTX_OP "." PTX_TYPE " [%0, %1], [%2], %3, [%4], %5;" \
                 : \
                 : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr), "h"(cpMask) \
                 : "memory"); \
  }

#define NCCL_CFT_DEFINE_RED_MULTIMEM_CP_MASK(NAME, PTX_OP, PTX_TYPE) \
  NCCL_DEVICE_INLINE void red_multimem_cp_mask_##NAME(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                                      ncclCftSmem& cftSmem, uint16_t cpMask) { \
    uint32_t srcSmemPtr = smemAddr(src); \
    uint32_t mbarPtr = smemAddr(cftSmem); \
    asm volatile("fabric.try_red.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric." \
                 "cp_mask.relaxed.sys." PTX_OP "." PTX_TYPE " [%0, %1], [%2], %3, [%4], %5;" \
                 : \
                 : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr), "h"(cpMask) \
                 : "memory"); \
  }

#define NCCL_CFT_DEFINE_PULLRED(NAME, PTX_OP, PTX_TYPE) \
  NCCL_DEVICE_INLINE void pullred_##NAME(ncclCftLeId leId, size_t leOffset, void* dst, uint32_t bytes, \
                                         ncclCftSmem& cftSmem) { \
    uint32_t dstSmemPtr = smemAddr(dst); \
    uint32_t mbarPtr = smemAddr(cftSmem); \
    asm volatile("fabric.try_pullred.async.multimem.shared::cta.mbarrier::complete_tx::bytes" \
                 ".mbarrier::report::fabric." PTX_TYPE "." PTX_OP ".sync.relaxed.sys " \
                 "[%0], [%1, %2], %3, [%4], 0xffffffff;" \
                 : \
                 : "r"(dstSmemPtr), "r"(leId), "l"(leOffset), "r"(bytes), "r"(mbarPtr) \
                 : "memory"); \
  }

#define NCCL_CFT_DEFINE_RED_FAMILY(NAME, OP, TYPE) \
  NCCL_CFT_DEFINE_RED(NAME, "", OP, TYPE) \
  NCCL_CFT_DEFINE_RED(multimem_##NAME, ".multimem", OP, TYPE) \
  NCCL_CFT_DEFINE_RED_CP_MASK(NAME, OP, TYPE) \
  NCCL_CFT_DEFINE_RED_MULTIMEM_CP_MASK(NAME, OP, TYPE)

#define NCCL_CFT_DEFINE_ALL_FAMILY(NAME, OP, TYPE) \
  NCCL_CFT_DEFINE_RED_FAMILY(NAME, OP, TYPE) \
  NCCL_CFT_DEFINE_PULLRED(NAME, OP, TYPE)

NCCL_CFT_DEFINE_ALL_FAMILY(and_b32, "and", "b32")
NCCL_CFT_DEFINE_ALL_FAMILY(and_b64, "and", "b64")
NCCL_CFT_DEFINE_ALL_FAMILY(xor_b32, "xor", "b32")
NCCL_CFT_DEFINE_ALL_FAMILY(xor_b64, "xor", "b64")
NCCL_CFT_DEFINE_ALL_FAMILY(or_b32, "or", "b32")
NCCL_CFT_DEFINE_ALL_FAMILY(or_b64, "or", "b64")
NCCL_CFT_DEFINE_ALL_FAMILY(min_u32, "min", "u32")
NCCL_CFT_DEFINE_ALL_FAMILY(min_s32, "min", "s32")
NCCL_CFT_DEFINE_ALL_FAMILY(min_u64, "min", "u64")
NCCL_CFT_DEFINE_ALL_FAMILY(min_s64, "min", "s64")
NCCL_CFT_DEFINE_ALL_FAMILY(min_bf16, "min", "bf16")
NCCL_CFT_DEFINE_ALL_FAMILY(min_f16, "min", "f16")
NCCL_CFT_DEFINE_ALL_FAMILY(max_u32, "max", "u32")
NCCL_CFT_DEFINE_ALL_FAMILY(max_s32, "max", "s32")
NCCL_CFT_DEFINE_ALL_FAMILY(max_u64, "max", "u64")
NCCL_CFT_DEFINE_ALL_FAMILY(max_s64, "max", "s64")
NCCL_CFT_DEFINE_ALL_FAMILY(max_bf16, "max", "bf16")
NCCL_CFT_DEFINE_ALL_FAMILY(max_f16, "max", "f16")
NCCL_CFT_DEFINE_ALL_FAMILY(add_u32, "add", "u32")
NCCL_CFT_DEFINE_ALL_FAMILY(add_u64, "add", "u64")
NCCL_CFT_DEFINE_ALL_FAMILY(add_bf16, "add", "bf16")
NCCL_CFT_DEFINE_ALL_FAMILY(add_f16, "add", "f16")
NCCL_CFT_DEFINE_ALL_FAMILY(add_f32, "add", "f32")
NCCL_CFT_DEFINE_RED_FAMILY(add_f64, "add", "f64")

#undef NCCL_CFT_DEFINE_ALL_FAMILY
#undef NCCL_CFT_DEFINE_RED_FAMILY
#undef NCCL_CFT_DEFINE_PULLRED
#undef NCCL_CFT_DEFINE_RED_MULTIMEM_CP_MASK
#undef NCCL_CFT_DEFINE_RED_CP_MASK
#undef NCCL_CFT_DEFINE_RED

template <typename RedOp>
struct Red;

#define NCCL_CFT_RED_TRAIT(OP_TAG, TYPE, NAME) \
  template <> \
  struct Red<OP_TAG<TYPE>> { \
    static NCCL_DEVICE_INLINE void red(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                       ncclCftSmem& cftSmem) { \
      red_##NAME(leId, leOffset, src, bytes, cftSmem); \
    } \
    static NCCL_DEVICE_INLINE void redCpMask(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                             ncclCftSmem& cftSmem, uint16_t cpMask) { \
      red_cp_mask_##NAME(leId, leOffset, src, bytes, cftSmem, cpMask); \
    } \
    static NCCL_DEVICE_INLINE void redMultimem(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                               ncclCftSmem& cftSmem) { \
      red_multimem_##NAME(leId, leOffset, src, bytes, cftSmem); \
    } \
    static NCCL_DEVICE_INLINE void redMultimemCpMask(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                                     ncclCftSmem& cftSmem, uint16_t cpMask) { \
      red_multimem_cp_mask_##NAME(leId, leOffset, src, bytes, cftSmem, cpMask); \
    } \
    static NCCL_DEVICE_INLINE void pullred(ncclCftLeId leId, size_t leOffset, void* dst, uint32_t bytes, \
                                           ncclCftSmem& cftSmem) { \
      pullred_##NAME(leId, leOffset, dst, bytes, cftSmem); \
    } \
  };

#define NCCL_CFT_RED_TRAIT_NO_PULL(OP_TAG, TYPE, NAME) \
  template <> \
  struct Red<OP_TAG<TYPE>> { \
    static NCCL_DEVICE_INLINE void red(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                       ncclCftSmem& cftSmem) { \
      red_##NAME(leId, leOffset, src, bytes, cftSmem); \
    } \
    static NCCL_DEVICE_INLINE void redCpMask(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                             ncclCftSmem& cftSmem, uint16_t cpMask) { \
      red_cp_mask_##NAME(leId, leOffset, src, bytes, cftSmem, cpMask); \
    } \
    static NCCL_DEVICE_INLINE void redMultimem(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                               ncclCftSmem& cftSmem) { \
      red_multimem_##NAME(leId, leOffset, src, bytes, cftSmem); \
    } \
    static NCCL_DEVICE_INLINE void redMultimemCpMask(ncclCftLeId leId, size_t leOffset, void* src, uint32_t bytes, \
                                                     ncclCftSmem& cftSmem, uint16_t cpMask) { \
      red_multimem_cp_mask_##NAME(leId, leOffset, src, bytes, cftSmem, cpMask); \
    } \
  };

#define NCCL_CFT_LOGICAL_TRAITS(OP_TAG, NAME) \
  NCCL_CFT_RED_TRAIT(OP_TAG, uint32_t, NAME##_b32) \
  NCCL_CFT_RED_TRAIT(OP_TAG, int32_t, NAME##_b32) \
  NCCL_CFT_RED_TRAIT(OP_TAG, uint64_t, NAME##_b64) \
  NCCL_CFT_RED_TRAIT(OP_TAG, int64_t, NAME##_b64)

NCCL_CFT_LOGICAL_TRAITS(ncclCftOpAnd, and)
NCCL_CFT_LOGICAL_TRAITS(ncclCftOpXor, xor)
NCCL_CFT_LOGICAL_TRAITS(ncclCftOpOr, or)

#define NCCL_CFT_MINMAX_TRAITS(OP_TAG, NAME) \
  NCCL_CFT_RED_TRAIT(OP_TAG, uint32_t, NAME##_u32) \
  NCCL_CFT_RED_TRAIT(OP_TAG, int32_t, NAME##_s32) \
  NCCL_CFT_RED_TRAIT(OP_TAG, uint64_t, NAME##_u64) \
  NCCL_CFT_RED_TRAIT(OP_TAG, int64_t, NAME##_s64) \
  NCCL_CFT_RED_TRAIT(OP_TAG, __nv_bfloat16, NAME##_bf16) \
  NCCL_CFT_RED_TRAIT(OP_TAG, half, NAME##_f16)

NCCL_CFT_MINMAX_TRAITS(ncclCftOpMin, min)
NCCL_CFT_MINMAX_TRAITS(ncclCftOpMax, max)

NCCL_CFT_RED_TRAIT(ncclCftOpSum, uint32_t, add_u32)
NCCL_CFT_RED_TRAIT(ncclCftOpSum, uint64_t, add_u64)
NCCL_CFT_RED_TRAIT(ncclCftOpSum, __nv_bfloat16, add_bf16)
NCCL_CFT_RED_TRAIT(ncclCftOpSum, half, add_f16)
NCCL_CFT_RED_TRAIT(ncclCftOpSum, float, add_f32)
NCCL_CFT_RED_TRAIT_NO_PULL(ncclCftOpSum, double, add_f64)

#undef NCCL_CFT_MINMAX_TRAITS
#undef NCCL_CFT_LOGICAL_TRAITS
#undef NCCL_CFT_RED_TRAIT_NO_PULL
#undef NCCL_CFT_RED_TRAIT
#endif

} // namespace internal
} // namespace cft
} // namespace nccl

template <typename Coop>
NCCL_DEVICE_INLINE void ncclMemFence(Coop coop, cuda::memory_order order, ncclMemProxyType producer,
                                     ncclMemProxyType consumer, ncclMemFenceScope scope) {
  if (order == cuda::memory_order_relaxed) return;
  if (nccl::utility::releaseOrderOf(order) == cuda::memory_order_release ||
      nccl::utility::releaseOrderOf(order) == cuda::memory_order_seq_cst) {
    coop.sync();
  }
#if NCCL_CFT_ENABLE
  if (nccl::cft::internal::elected(coop)) {
    if (scope == ncclMemFenceScope::Cta) {
      if (nccl::utility::releaseOrderOf(order) == cuda::memory_order_release ||
          nccl::utility::releaseOrderOf(order) == cuda::memory_order_seq_cst) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
    } else {
      if (producer == ncclMemProxyType::Generic && consumer == ncclMemProxyType::Fabric) {
        if (order == cuda::memory_order_acquire) {
          asm volatile("fence.proxy.fabric::generic.alias.acquire.sys;" ::: "memory");
        } else if (order == cuda::memory_order_release) {
          asm volatile("fence.proxy.fabric::generic.alias.release.sys;" ::: "memory");
        } else if (order == cuda::memory_order_acq_rel || order == cuda::memory_order_seq_cst) {
          asm volatile("fence.proxy.fabric::generic.alias.acquire.sys;" ::: "memory");
          asm volatile("fence.proxy.fabric::generic.alias.release.sys;" ::: "memory");
        }
      } else if (producer == ncclMemProxyType::Fabric && consumer == ncclMemProxyType::Fabric) {
        if (order == cuda::memory_order_acquire) {
          asm volatile("fence.proxy.fabric::fabric.alias.acquire.sys;" ::: "memory");
        } else if (order == cuda::memory_order_release) {
          asm volatile("fence.proxy.fabric::fabric.alias.release.sys;" ::: "memory");
        } else if (order == cuda::memory_order_acq_rel || order == cuda::memory_order_seq_cst) {
          asm volatile("fence.proxy.fabric::fabric.alias.acquire.sys;" ::: "memory");
          asm volatile("fence.proxy.fabric::fabric.alias.release.sys;" ::: "memory");
        }
      } else if (producer == ncclMemProxyType::Fabric && consumer == ncclMemProxyType::Generic) {
        if (order == cuda::memory_order_acquire) {
          asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
        } else if (order == cuda::memory_order_release) {
          asm volatile("fence.proxy.generic::fabric.alias.release.sys;" ::: "memory");
        } else if (order == cuda::memory_order_acq_rel || order == cuda::memory_order_seq_cst) {
          asm volatile("fence.proxy.generic::fabric.alias.acquire.sys;" ::: "memory");
          asm volatile("fence.proxy.generic::fabric.alias.release.sys;" ::: "memory");
        }
      } else { // Generic to Generic proxy sync
        if (order == cuda::memory_order_acquire) {
          asm volatile("fence.acquire.sys;" ::: "memory");
        } else if (order == cuda::memory_order_release) {
          asm volatile("fence.release.sys;" ::: "memory");
        } else if (order == cuda::memory_order_acq_rel || order == cuda::memory_order_seq_cst) {
          asm volatile("membar.sys;" ::: "memory");
        }
      }
    }
  }
  if (nccl::utility::acquireOrderOf(order) == cuda::memory_order_acquire ||
      nccl::utility::acquireOrderOf(order) == cuda::memory_order_seq_cst) {
    coop.sync();
  }
#else
  (void)coop;
  (void)order;
  (void)producer;
  (void)consumer;
  (void)scope;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
NCCL_DEVICE_INLINE ncclCft<Coop>::ncclCft(Coop coop, ncclCftSmem& cftSmem)
  : ncclCft_internal<Coop>{coop, cftSmem, /*txCount=*/0, /*phaseParity=*/0} {
#if NCCL_CFT_ENABLE
  if (nccl::cft::internal::elected(coop)) {
    asm volatile("mbarrier.init.shared.layout::v1.b64 [%0], %1;"
                 :
                 : "r"(nccl::cft::internal::smemAddr(cftSmem)), "r"(1)
                 : "memory");
  }
  // Every thread in coop must wait for mbarrier to be initialize before using it
  coop.sync();
#else
  (void)coop;
  (void)cftSmem;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
NCCL_DEVICE_INLINE ncclCft<Coop>::~ncclCft() {
#if NCCL_CFT_ENABLE
  // Elected thread must wait for all threads using mbarrier to be done before invalidating it
  this->coop.sync();
  if (nccl::cft::internal::elected(this->coop)) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                 :
                 : "r"(nccl::cft::internal::smemAddr(this->cftSmem))
                 : "memory");
  }
#else
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::submit(OpCoop coop) {
#if NCCL_CFT_ENABLE
  if (nccl::cft::internal::elected(coop)) {
    asm volatile("fabric.submit;" ::: "memory");
    asm volatile("mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(nccl::cft::internal::smemAddr(this->cftSmem)), "r"(this->txCount)
                 : "memory");
    this->txCount = 0;
  }
#else
  (void)coop;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::flushSmem(OpCoop coop) {
#if NCCL_CFT_ENABLE
  asm volatile("fabric.wait.sync_restrict::reads;" ::: "memory");
  // Every thread in the coop must wait for fabric operations to consume shared memory.
  coop.sync();
#else
  (void)coop;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::flush(OpCoop coop, bool* hasReport, uint32_t* report) {
#if NCCL_CFT_ENABLE
  if (nccl::cft::internal::elected(coop)) {
    asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(nccl::cft::internal::smemAddr(this->cftSmem)), "r"(1)
                 : "memory");
  }

  int hasReportValue = 0;
  uint32_t reportValue = 0;
  nccl::cft::internal::waitMbarrier(this->cftSmem, this->phaseParity, &hasReportValue, &reportValue);
  this->phaseParity ^= 1;

  if (hasReport) *hasReport = (hasReportValue != 0);
  if (report) *report = reportValue;
#else
  (void)coop;
  if (hasReport) *hasReport = false;
  if (report) *report = 0;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::put(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource,
                                           uint32_t bytes) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::put requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::put requires 'bytes' to be a multiple of 16.");
#endif
    uint32_t srcSmemPtr = nccl::cft::internal::smemAddr(smemSource);
    uint32_t mbarPtr = nccl::cft::internal::smemAddr(this->cftSmem);
    asm volatile(
      "fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
      "[%0, %1], [%2], %3, [%4];"
      :
      : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr)
      : "memory");
    this->txCount += (bytes / 16);
  }
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)smemSource;
  (void)bytes;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::putCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource,
                                                 uint32_t bytes, uint16_t cpMask) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::putCpMask requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::putCpMask requires 'bytes' to be a multiple of 16.");
#endif
    uint32_t srcSmemPtr = nccl::cft::internal::smemAddr(smemSource);
    uint32_t mbarPtr = nccl::cft::internal::smemAddr(this->cftSmem);
    asm volatile("fabric.try_put.async.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask."
                 "relaxed.sys.b128 [%0, %1], [%2], %3, [%4], %5;"
                 :
                 : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr), "h"(cpMask)
                 : "memory");
    this->txCount += (bytes / 16);
  }
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)smemSource;
  (void)bytes;
  (void)cpMask;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::putMultimem(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemSource,
                                                   uint32_t bytes) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::putMultimem requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::putMultimem requires 'bytes' to be a multiple of 16.");
#endif
    uint32_t srcSmemPtr = nccl::cft::internal::smemAddr(smemSource);
    uint32_t mbarPtr = nccl::cft::internal::smemAddr(this->cftSmem);
    asm volatile(
      "fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.relaxed.sys.b128 "
      "[%0, %1], [%2], %3, [%4];"
      :
      : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr)
      : "memory");
    this->txCount += (bytes / 16);
  }
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)smemSource;
  (void)bytes;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::putMultimemCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset,
                                                         void* smemSource, uint32_t bytes, uint16_t cpMask) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::putMultimemCpMask requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::putMultimemCpMask requires 'bytes' to be a multiple of 16.");
#endif
    uint32_t srcSmemPtr = nccl::cft::internal::smemAddr(smemSource);
    uint32_t mbarPtr = nccl::cft::internal::smemAddr(this->cftSmem);
    asm volatile(
      "fabric.try_put.async.multimem.shared::cta.mbarrier::complete_tx::16B.mbarrier::report::fabric.cp_mask."
      "relaxed.sys.b128 [%0, %1], [%2], %3, [%4], %5;"
      :
      : "r"(leId), "l"(leOffset), "r"(srcSmemPtr), "r"(bytes), "r"(mbarPtr), "h"(cpMask)
      : "memory");
    this->txCount += (bytes / 16);
  }
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)smemSource;
  (void)bytes;
  (void)cpMask;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::get(OpCoop coop, ncclCftLeId leId, size_t leOffset, void* smemDestination,
                                           uint32_t bytes) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemDestination) % 16 == 0 &&
           "ncclCft::get requires 'smemDestination' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::get requires 'bytes' to be a multiple of 16.");
#endif
    uint32_t dstSmemPtr = nccl::cft::internal::smemAddr(smemDestination);
    uint32_t mbarPtr = nccl::cft::internal::smemAddr(this->cftSmem);
    asm volatile(
      "fabric.try_get.async.shared::cta.mbarrier::complete_tx::bytes.mbarrier::report::fabric.relaxed.sys.b128 "
      "[%0], [%1, %2], %3, [%4];"
      :
      : "r"(dstSmemPtr), "r"(leId), "l"(leOffset), "r"(bytes), "r"(mbarPtr)
      : "memory");
    this->txCount += bytes;
  }
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)smemDestination;
  (void)bytes;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename RedOp, typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::red(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                           void* smemSource, uint32_t bytes) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::red requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::red requires 'bytes' to be a multiple of 16.");
#endif
    nccl::cft::internal::Red<RedOp>::red(leId, leOffset, smemSource, bytes, this->cftSmem);
    this->txCount += (bytes / 16);
  }
  (void)red;
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)red;
  (void)smemSource;
  (void)bytes;
  assert(false && nccl::cft::internal::redOpUnsupported());
#endif
}

template <typename Coop>
template <typename RedOp, typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::redCpMask(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                                 void* smemSource, uint32_t bytes, uint16_t cpMask) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::redCpMask requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::redCpMask requires 'bytes' to be a multiple of 16.");
#endif
    nccl::cft::internal::Red<RedOp>::redCpMask(leId, leOffset, smemSource, bytes, this->cftSmem, cpMask);
    this->txCount += (bytes / 16);
  }
  (void)red;
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)red;
  (void)smemSource;
  (void)bytes;
  (void)cpMask;
  assert(false && nccl::cft::internal::redOpUnsupported());
#endif
}

template <typename Coop>
template <typename RedOp, typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::redMultimem(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                                   void* smemSource, uint32_t bytes) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::redMultimem requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::redMultimem requires 'bytes' to be a multiple of 16.");
#endif
    nccl::cft::internal::Red<RedOp>::redMultimem(leId, leOffset, smemSource, bytes, this->cftSmem);
    this->txCount += (bytes / 16);
  }
  (void)red;
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)red;
  (void)smemSource;
  (void)bytes;
  assert(false && nccl::cft::internal::redOpUnsupported());
#endif
}

template <typename Coop>
template <typename RedOp, typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::redMultimemCpMask(
  OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red, void* smemSource, uint32_t bytes, uint16_t cpMask) {
#if NCCL_CFT_ENABLE
  coop.sync();
  if (nccl::cft::internal::elected(coop)) {
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
    assert(reinterpret_cast<uintptr_t>(smemSource) % 16 == 0 &&
           "ncclCft::redMultimemCpMask requires 'smemSource' to be 16 bytes aligned.");
    assert(bytes % 16 == 0 && "ncclCft::redMultimemCpMask requires 'bytes' to be a multiple of 16.");
#endif
    nccl::cft::internal::Red<RedOp>::redMultimemCpMask(leId, leOffset, smemSource, bytes, this->cftSmem, cpMask);
    this->txCount += (bytes / 16);
  }
  (void)red;
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)red;
  (void)smemSource;
  (void)bytes;
  (void)cpMask;
  nccl::cft::internal::unsupported();
#endif
}

template <typename Coop>
template <typename RedOp, typename OpCoop>
NCCL_DEVICE_INLINE void ncclCft<Coop>::pullRed(OpCoop coop, ncclCftLeId leId, size_t leOffset, RedOp const& red,
                                               void* smemDestination, uint32_t bytes) {
  static_assert(std::is_same<OpCoop, ncclCoopWarp>::value, "ncclCft::pullRed requires ncclCoopWarp");
#if NCCL_CFT_ENABLE
  coop.sync();
#ifdef NCCL_DEVICE_CFT_ENABLE_DEBUG
  assert(reinterpret_cast<uintptr_t>(smemDestination) % 16 == 0 &&
         "ncclCft::pullRed requires 'smemDestination' to be 16 bytes aligned.");
  assert(bytes % 16 == 0 && "ncclCft::pullRed requires 'bytes' to be a multiple of 16.");
#endif
  nccl::cft::internal::Red<RedOp>::pullred(leId, leOffset, smemDestination, bytes, this->cftSmem);
  asm volatile("fabric.submit.op_restrict::fetching;" ::: "memory");
  this->txCount += nccl::cft::internal::elected(coop) ? bytes : 0;
  (void)red;
#else
  (void)coop;
  (void)leId;
  (void)leOffset;
  (void)red;
  (void)smemDestination;
  (void)bytes;
  assert(false && nccl::cft::internal::redOpUnsupported());
#endif
}

#undef NCCL_CFT_ENABLE

#endif // __CUDACC__

#endif // _NCCL_DEVICE_CFT__FUNCS_H_
