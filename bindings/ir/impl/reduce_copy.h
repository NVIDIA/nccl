/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_IMPL_REDUCE_COPY_H_
#define _NCCL_DEVICE_WRAPPER_IMPL_REDUCE_COPY_H_

// Internal implementation fragment; include only from nccl_device_wrapper.cu
// after its prerequisite headers.

#define NCCL_IR_DEFINE_ncclIrLsaReduceSum(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrLsaReduceSum_##suffix(ncclIrCoop const* coop, ncclWindow_t srcWindow, \
                                                      size_t srcOffset, type* dst, size_t count, ncclTeam team) { \
    ncclLsaReduceSum<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), srcWindow, srcOffset, \
                                                dst, count, team); \
  }
#define NCCL_IR_DEFINE_ncclIrMultimemReduceSum(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrMultimemReduceSum_##suffix(ncclIrCoop const* coop, type* mcSrc, type* dst, \
                                                           size_t count) { \
    ncclMultimemReduceSum<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), mcSrc, dst, count); \
  }
#define NCCL_IR_DEFINE_ncclIrLsaCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrLsaCopy_##suffix(ncclIrCoop const* coop, type* src, ncclWindow_t dstWindow, \
                                                 size_t dstOffset, size_t count, ncclTeam team) { \
    ncclLsaCopy<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), src, dstWindow, dstOffset, \
                                           count, team); \
  }
#define NCCL_IR_DEFINE_ncclIrMultimemCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrMultimemCopy_##suffix(ncclIrCoop const* coop, type* src, type* mcDst, size_t count) { \
    ncclMultimemCopy<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), src, mcDst, count); \
  }
#define NCCL_IR_DEFINE_ncclIrLsaReduceSumCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrLsaReduceSumCopy_##suffix(ncclIrCoop const* coop, ncclWindow_t srcWindow, \
                                                          size_t srcOffset, ncclWindow_t dstWindow, size_t dstOffset, \
                                                          size_t count, ncclTeam team) { \
    ncclLsaReduceSumCopy<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), srcWindow, srcOffset, \
                                                    dstWindow, dstOffset, count, team); \
  }
#define NCCL_IR_DEFINE_ncclIrMultimemReduceSumCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrMultimemReduceSumCopy_##suffix(ncclIrCoop const* coop, type* mcSrc, type* mcDst, \
                                                               size_t count) { \
    ncclMultimemReduceSumCopy<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), mcSrc, mcDst, \
                                                         count); \
  }
#define NCCL_IR_DEFINE_ncclIrLocalReduceSumCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrLocalReduceSumCopy_##suffix(ncclIrCoop const* coop, int nSrc, type* srcBase, \
                                                            size_t srcDispl, int nDst, type* dstBase, size_t dstDispl, \
                                                            size_t count) { \
    ncclLocalReduceSumCopy<type, ncclCoopAny, size_t>(*reinterpret_cast<ncclCoopAny const*>(coop), nSrc, srcBase, \
                                                      srcDispl, nDst, dstBase, dstDispl, count); \
  }
#define NCCL_IR_DEFINE_ncclIrLsaCopyTma(suffix, type) \
  NCCL_DEVICE_INLINE void ncclIrLsaCopyTma_##suffix(ncclIrCoop const* coop, type* src, ncclWindow_t dstWindow, \
                                                    size_t dstOffset, size_t count, ncclTeam team, char* smemPtr, \
                                                    int smemBytesTotal) { \
    auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop); \
    if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopThread>()) { \
      ncclLsaCopyTma<type, ncclCoopThread, size_t>(*reinterpret_cast<ncclCoopThread const*>(&coopImpl.storage), src, \
                                                   dstWindow, dstOffset, count, team, smemPtr, smemBytesTotal); \
    } else if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopWarp>()) { \
      ncclLsaCopyTma<type, ncclCoopWarp, size_t>(*reinterpret_cast<ncclCoopWarp const*>(&coopImpl.storage), src, \
                                                 dstWindow, dstOffset, count, team, smemPtr, smemBytesTotal); \
    } else if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopCta>()) { \
      ncclLsaCopyTma<type, ncclCoopCta, size_t>(*reinterpret_cast<ncclCoopCta const*>(&coopImpl.storage), src, \
                                                dstWindow, dstOffset, count, team, smemPtr, smemBytesTotal); \
    } else if (NCCL_DEVICE_DEBUG_CHECKS) { \
      assert(false && "ncclIrLsaCopyTma requires Thread, Warp, or CTA"); \
    } \
  }

NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaReduceSum)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclIrMultimemReduceSum)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaCopy)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclIrMultimemCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaReduceSumCopy)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclIrMultimemReduceSumCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLocalReduceSumCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclIrLsaCopyTma)

#endif // _NCCL_DEVICE_WRAPPER_IMPL_REDUCE_COPY_H_
