/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * NCCL Device API C ABI implementation for LLVM bitcode and NVCC LTO IR.
 */

// Omit C++ inline so unused entry points remain available for linking.
// NVCC may use Clang as its host preprocessor.
// https://llvm.org/docs/CompileCudaWithLLVM.html#detecting-clang-vs-nvcc-from-code
#if defined(__NVCC__)
// always_inline lost entry points during LTO IR generation; use a hint (!2936).
#define NCCL_IR_DEVICE_INLINE __device__ __inline_hint__
#elif defined(__clang__) && defined(__CUDA__)
// Request inlining without the inline semantics of __forceinline__.
#define NCCL_IR_DEVICE_INLINE __device__ __attribute__((always_inline))
#else
#error "NCCL device IR requires Clang CUDA or NVCC"
#endif

// Work around libNVVM failing to lower Clang's __activemask intrinsic.
// Must precede CUDA/NCCL includes so native coop call sites expand it.
#if defined(__clang__) && defined(__CUDA__) && defined(__CUDA_ARCH__)
__attribute__((device)) __attribute__((always_inline)) static unsigned ncclIrActivemask() {
  unsigned mask;
  asm volatile("activemask.b32 %0;" : "=r"(mask));
  return mask;
}
#undef __activemask
#define __activemask ncclIrActivemask
#endif

#include "nccl_device_ir.h"
// The bitcode library is device-only: suppress "nccl_device/host.h" by
// pre-defining its include guard, so the host entrypoints never enter this TU.
#define _NCCL_DEVICE_HOST_H_
#include "nccl_device.h"
#ifdef NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
#error "nccl_device/host.h leaked in: its include guard was renamed, update the #define above"
#endif
#include "util.h"

#include <new>

#include "core.h"
#include "coop.h"
#include "gin.h"
#include "barrier.h"
#include "reduce_copy.h"
