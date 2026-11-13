/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NVLS_TMA_SCATTER_CUH_
#define NCCL_NVLS_TMA_SCATTER_CUH_

#include "device.h"

#if defined(CUDART_VERSION) && CUDART_VERSION >= 13010
// NVCC host registration stubs need the CCCL globals declared in both passes.
#include <cuda/std/ranges>
#include <cuda/std/utility>
#endif

struct ncclNvlsTmaScatterSpan {
  size_t prefix, bulk, suffix;
};

__host__ __device__ inline ncclNvlsTmaScatterSpan ncclNvlsTmaScatterSplit(uintptr_t dst, uintptr_t src, size_t bytes) {
  if ((src & 15) != (dst & 15)) return {bytes, 0, 0};
  size_t prefix = (-src) & 15;
  if (prefix > bytes) prefix = bytes;
  size_t bulk = (bytes - prefix) & ~size_t(15);
  return {prefix, bulk, bytes - prefix - bulk};
}

#if NCCL_NVLS_TMA_SCATTER_ARCH(__CUDA_ARCH__)
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda/std/utility>

struct alignas(16) ncclNvlsTmaScatterHeader {
  cuda::barrier<cuda::thread_scope_block> bar;
  char padding[ncclNvlsTmaScatterHeaderBytes - sizeof(cuda::barrier<cuda::thread_scope_block>)];
};
static_assert(sizeof(ncclNvlsTmaScatterHeader) == ncclNvlsTmaScatterHeaderBytes, "TMA header size mismatch");

__device__ __forceinline__ bool ncclNvlsTmaScatterSetup(void* scratch, bool mayIssue) {
  bool issuer = mayIssue && cuda::ptx::elect_sync(0xffffffff);
  if (issuer) init(&static_cast<ncclNvlsTmaScatterHeader*>(scratch)->bar, 1);
  __syncwarp();
  return issuer;
}

__device__ __forceinline__ void ncclNvlsTmaScatterCopy(void* scratch, uint32_t tileBytes, char* dst, const char* src,
                                                       size_t bytes) {
  auto& bar = static_cast<ncclNvlsTmaScatterHeader*>(scratch)->bar;
  char* tile = static_cast<char*>(scratch) + ncclNvlsTmaScatterHeaderBytes;
  while (bytes != 0) {
    uint32_t n = bytes < tileBytes ? (uint32_t)bytes : tileBytes;
    cuda::device::memcpy_async_tx(tile, src, cuda::aligned_size_t<16>(n), bar);
    auto token = cuda::device::barrier_arrive_tx(bar, 1, n);
    bar.wait(cuda::std::move(token));
    cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared, dst, tile, n);
    cuda::ptx::cp_async_bulk_commit_group();
    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
    src += n;
    dst += n;
    bytes -= n;
  }
}

__device__ __forceinline__ void ncclNvlsTmaScatterWait() {
  cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>());
}

__device__ __forceinline__ void ncclNvlsTmaScatterRelease(void* scratch) {
  asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(
                 (uint32_t)__cvta_generic_to_shared(&static_cast<ncclNvlsTmaScatterHeader*>(scratch)->bar))
               : "memory");
}
#endif

#endif // NCCL_NVLS_TMA_SCATTER_CUH_
