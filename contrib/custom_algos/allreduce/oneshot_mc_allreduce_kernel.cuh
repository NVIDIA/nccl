/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// oneshot_mc_allreduce_kernel.cuh
//
// One-shot NVLink multicast AllReduce using multimem.ld_reduce.
//
// Each GPU independently issues multimem.ld_reduce.add.v4.f32 which reads
// ALL ranks' sendBuff[i] simultaneously via the multicast VA and returns
// the hardware-reduced sum.  The result is stored directly to recvLocal[i].
//
// No barrier or sentinel is needed: the caller synchronises all ranks
// (e.g. via MPI_Barrier) before launching so all sendBuff[i] are valid.
//
// Requirements:
//   - sm >= 90 (Hopper / Blackwell) for multimem PTX.
//   - ncclDevComm with lsaMultimem=true (ncclGetLsaMultimemPointer).
//   - sendBuff registered with NCCL_WIN_COLL_SYMMETRIC (provides mc VA).

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include "nccl_device.h"

#define MC_AR_MAXTHREADS   1024

// ---------------------------------------------------------------------------
// oneshot_mc_allreduce_kernel
//
// Parameters:
//   recvLocal  — output buffer (float4, at least nlines elements)
//   sendWin    — symmetric window over sendBuff (multicast VA source)
//   nlines     — float4 elements per AllReduce (msgBytes / 16)
//   devComm    — NCCL device communicator (provides lsaMultimem base ptr)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(MC_AR_MAXTHREADS)
    oneshot_mc_allreduce_kernel(float4*      recvLocal,
                                ncclWindow_t sendWin,
                                size_t       nlines,
                                ncclDevComm  devComm)
{
#if __CUDA_ARCH__ >= 900
    const int tid    = threadIdx.x + blockDim.x * blockIdx.x;
    const int stride = blockDim.x * gridDim.x;

    void *mc_ptr = ncclGetLsaMultimemPointer(sendWin, 0, devComm);

    for (size_t i = tid; i < nlines; i += stride) {
        float4 result;
        asm volatile(
            "multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(result.x), "=f"(result.y),
              "=f"(result.z), "=f"(result.w)
            : "l"((uintptr_t)mc_ptr + (uintptr_t)i * sizeof(float4))
            : "memory");
        recvLocal[i] = result;
    }
#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[oneshot_mc_allreduce] ERROR: requires sm >= 90 (Hopper/Blackwell)\n");
    assert(false);
#endif
}
