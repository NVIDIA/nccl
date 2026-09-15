/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Compile-only coverage for the public NIIN atomic surface.  Do not launch
// this kernel: its purpose is to instantiate the direct GPUNetIO WQE paths
// without requiring a particular fabric in a build environment.

#include <nvshmem.h>

#include <cstddef>
#include <cstdint>

extern "C" __global__ void niin_gpunetio_atomic_compile_surface(
    niinContext* context, int* i32, uint32_t* u32, uint64_t* u64, ptrdiff_t* pdiff,
    float* f32, double* f64, __half* f16) {
  niin_g_ctx = context;

  int oldI32 = 0;
  uint32_t oldU32 = 0;
  uint64_t oldU64 = 0;
  ptrdiff_t oldPdiff = 0;

  oldI32 = nvshmem_int_atomic_fetch_add(i32, 1, 1);
  nvshmem_int_atomic_add(i32, 1, 1);
  oldI32 = nvshmem_int_atomic_compare_swap(i32, 0, 1, 1);
  oldI32 = nvshmem_int_atomic_swap(i32, 1, 1);
  oldI32 = nvshmem_int_atomic_fetch(i32, 1);
  nvshmem_int_atomic_set(i32, 1, 1);
  nvshmem_int_atomic_inc(i32, 1);
  oldI32 = nvshmem_int_atomic_fetch_inc(i32, 1);

  oldU32 = nvshmem_uint32_atomic_fetch_and(u32, UINT32_C(0xff), 1);
  nvshmem_uint32_atomic_and(u32, UINT32_C(0xff), 1);
  oldU32 = nvshmem_uint32_atomic_fetch_or(u32, UINT32_C(1), 1);
  nvshmem_uint32_atomic_or(u32, UINT32_C(1), 1);
  oldU32 = nvshmem_uint32_atomic_fetch_xor(u32, UINT32_C(1), 1);
  nvshmem_uint32_atomic_xor(u32, UINT32_C(1), 1);

  oldU64 = nvshmem_uint64_atomic_fetch_add(u64, UINT64_C(1), 1);
  nvshmem_uint64_atomic_add(u64, UINT64_C(1), 1);
  oldU64 = nvshmem_uint64_atomic_compare_swap(u64, UINT64_C(0), UINT64_C(1), 1);
  oldU64 = nvshmem_uint64_atomic_swap(u64, UINT64_C(1), 1);
  oldU64 = nvshmem_uint64_atomic_fetch(u64, 1);
  nvshmem_uint64_atomic_set(u64, UINT64_C(1), 1);
  nvshmem_uint64_atomic_inc(u64, 1);
  oldU64 = nvshmem_uint64_atomic_fetch_inc(u64, 1);
  oldU64 = nvshmem_uint64_atomic_fetch_and(u64, UINT64_C(0xff), 1);
  nvshmem_uint64_atomic_and(u64, UINT64_C(0xff), 1);
  oldU64 = nvshmem_uint64_atomic_fetch_or(u64, UINT64_C(1), 1);
  nvshmem_uint64_atomic_or(u64, UINT64_C(1), 1);
  oldU64 = nvshmem_uint64_atomic_fetch_xor(u64, UINT64_C(1), 1);
  nvshmem_uint64_atomic_xor(u64, UINT64_C(1), 1);

  // ptrdiff is part of NVSHMEM's standard integer AMO spelling. It is a
  // 4- or 8-byte integral type on supported CUDA platforms and therefore
  // belongs on the native direct-QP path alongside size_t.
  oldPdiff = nvshmem_ptrdiff_atomic_fetch_add(pdiff, ptrdiff_t{1}, 1);
  nvshmem_ptrdiff_atomic_add(pdiff, ptrdiff_t{1}, 1);
  oldPdiff = nvshmem_ptrdiff_atomic_compare_swap(pdiff, ptrdiff_t{0}, ptrdiff_t{1}, 1);
  oldPdiff = nvshmem_ptrdiff_atomic_swap(pdiff, ptrdiff_t{1}, 1);
  oldPdiff = nvshmem_ptrdiff_atomic_fetch(pdiff, 1);
  nvshmem_ptrdiff_atomic_set(pdiff, ptrdiff_t{1}, 1);
  nvshmem_ptrdiff_atomic_inc(pdiff, 1);
  oldPdiff = nvshmem_ptrdiff_atomic_fetch_inc(pdiff, 1);

  // Base floating fetch/swap/set preserve raw representations. The extended
  // NVSHMEMX add/fetch-add calls instantiate the NIIN SRQ proxy adapter; they
  // intentionally do not instantiate a GPUNetIO integer AMO WQE.
  float oldF32 = nvshmem_float_atomic_swap(f32, 1.0f, 1);
  oldF32 = nvshmem_float_atomic_fetch(f32, 1);
  nvshmem_float_atomic_set(f32, 2.0f, 1);
  double oldF64 = nvshmem_double_atomic_swap(f64, 1.0, 1);
  oldF64 = nvshmem_double_atomic_fetch(f64, 1);
  nvshmem_double_atomic_set(f64, 2.0, 1);
  const __half oneHalf = __float2half_rn(1.0f);
  __half oldF16 = nvshmemx_half_atomic_fetch_add(f16, oneHalf, 1);
  nvshmemx_half_atomic_add(f16, oneHalf, 1);
  oldF32 = nvshmemx_float_atomic_fetch_add(f32, 1.0f, 1);
  nvshmemx_float_atomic_add(f32, 1.0f, 1);
  oldF64 = nvshmemx_double_atomic_fetch_add(f64, 1.0, 1);
  nvshmemx_double_atomic_add(f64, 1.0, 1);

  // Preserve the temporary values so device compilation cannot erase every
  // call before the WQE template bodies are instantiated.
  if (oldI32 == -1) *i32 = oldI32;
  if (oldU32 == UINT32_MAX) *u32 = oldU32;
  if (oldU64 == UINT64_MAX) *u64 = oldU64;
  if (oldPdiff == -1) *pdiff = oldPdiff;
  if (oldF32 == -1.0f) *f32 = oldF32;
  if (oldF64 == -1.0) *f64 = oldF64;
  if (__half2float(oldF16) == -1.0f) *f16 = oldF16;
}
