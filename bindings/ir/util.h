/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_IR_UTIL_H_
#define _NCCL_IR_UTIL_H_

// BF16 requires CUDA 11.0; FP8 requires CUDA 11.8.
#define NCCL_IR_BF16_TYPES(F) \
  F(BF16, __nv_bfloat16)

#define NCCL_IR_FP8_TYPES(F) \
  F(F8E4M3, __nv_fp8_e4m3) \
  F(F8E5M2, __nv_fp8_e5m2)

#define NCCL_IR_MULTIMEM_TYPES(F) \
  F(I32, int32_t) \
  F(U32, uint32_t) \
  F(I64, int64_t) \
  F(U64, uint64_t) \
  F(F16, half) \
  F(F32, float) \
  F(F64, double) \
  NCCL_IR_BF16_TYPES(F) \
  NCCL_IR_FP8_TYPES(F)

#define NCCL_IR_TYPES(F) \
  F(I8, int8_t) \
  F(U8, uint8_t) \
  F(I32, int32_t) \
  F(U32, uint32_t) \
  F(I64, int64_t) \
  F(U64, uint64_t) \
  F(F16, half) \
  F(F32, float) \
  F(F64, double) \
  NCCL_IR_BF16_TYPES(F) \
  NCCL_IR_FP8_TYPES(F)

#define NCCL_IR_DEFINE_API_ALL_TYPES(func) \
  NCCL_IR_TYPES(NCCL_IR_DEFINE_##func)

#define NCCL_IR_DEFINE_API_MULTIMEM_TYPES(func) \
  NCCL_IR_MULTIMEM_TYPES(NCCL_IR_DEFINE_##func)

#endif // _NCCL_IR_UTIL_H_
