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

#define NCCL_IR_DEFINE_ncclLsaReduceSum(suffix, type) \
  NCCL_DEVICE_INLINE void ncclLsaReduceSum_##suffix(ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset, \
                                                    type* dst, size_t count, ncclTeam team) { \
    ncclLsaReduceSum<type, ncclCoopAny, size_t>(coop, srcWindow, srcOffset, dst, count, team); \
  }
#define NCCL_IR_DEFINE_ncclMultimemReduceSum(suffix, type) \
  NCCL_DEVICE_INLINE void ncclMultimemReduceSum_##suffix(ncclCoopAny coop, type* mcSrc, type* dst, size_t count) { \
    ncclMultimemReduceSum<type, ncclCoopAny, size_t>(coop, mcSrc, dst, count); \
  }
#define NCCL_IR_DEFINE_ncclLsaCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclLsaCopy_##suffix(ncclCoopAny coop, type* src, ncclWindow_t dstWindow, size_t dstOffset, \
                                               size_t count, ncclTeam team) { \
    ncclLsaCopy<type, ncclCoopAny, size_t>(coop, src, dstWindow, dstOffset, count, team); \
  }
#define NCCL_IR_DEFINE_ncclMultimemCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclMultimemCopy_##suffix(ncclCoopAny coop, type* src, type* mcDst, size_t count) { \
    ncclMultimemCopy<type, ncclCoopAny, size_t>(coop, src, mcDst, count); \
  }
#define NCCL_IR_DEFINE_ncclLsaReduceSumCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy_##suffix(ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset, \
                                                        ncclWindow_t dstWindow, size_t dstOffset, size_t count, \
                                                        ncclTeam team) { \
    ncclLsaReduceSumCopy<type, ncclCoopAny, size_t>(coop, srcWindow, srcOffset, dstWindow, dstOffset, count, team); \
  }
#define NCCL_IR_DEFINE_ncclMultimemReduceSumCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy_##suffix(ncclCoopAny coop, type* mcSrc, type* mcDst, \
                                                             size_t count) { \
    ncclMultimemReduceSumCopy<type, ncclCoopAny, size_t>(coop, mcSrc, mcDst, count); \
  }
#define NCCL_IR_DEFINE_ncclLocalReduceSumCopy(suffix, type) \
  NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy_##suffix(ncclCoopAny coop, int nSrc, type* srcBase, size_t srcDispl, \
                                                          int nDst, type* dstBase, size_t dstDispl, size_t count) { \
    ncclLocalReduceSumCopy<type, ncclCoopAny, size_t>(coop, nSrc, srcBase, srcDispl, nDst, dstBase, dstDispl, count); \
  }

NCCL_IR_DEFINE_API_ALL_TYPES(ncclLsaReduceSum)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclMultimemReduceSum)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLsaCopy)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclMultimemCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLsaReduceSumCopy)
NCCL_IR_DEFINE_API_MULTIMEM_TYPES(ncclMultimemReduceSumCopy)
NCCL_IR_DEFINE_API_ALL_TYPES(ncclLocalReduceSumCopy)

#endif // _NCCL_DEVICE_WRAPPER_IMPL_REDUCE_COPY_H_
