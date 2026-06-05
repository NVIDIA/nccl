/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// twoshot_mc_simple_allreduce_kernel.cuh
//
// Two-shot NVLink multicast AllReduce:
//   Shot 1 — ReduceScatter: each rank reduces its assigned chunk of the
//             send buffer using multimem.ld_reduce.add.v4.f32.  The
//             hardware reads ALL ranks' sendBuff[i] simultaneously and
//             returns the sum.
//   Shot 2 — AllGather: each rank broadcasts its reduced chunk to ALL
//             ranks' receive buffers via multimem.st.global.v4.f32.
//   Barrier — ncclLsaBarrierSession (release/acquire) ensures all
//             AllGather stores are globally visible before the kernel
//             returns.
//
// Shots 1 and 2 are fused into a single element-wise loop: each element
// is reduced and immediately broadcast without intermediate buffering.
//
// Buffer layout:
//   sendBuff[0 .. nlines-1]   — symmetric input; each rank's own data.
//   recvBuff[0 .. nlines-1]   — symmetric output; written via multicast.
//   Rank r's chunk: [r * (nlines/nRanks) .. (r+1) * (nlines/nRanks) - 1]
//
// Requirements:
//   - sm >= 90 (Hopper / Blackwell) for multimem PTX.
//   - Both sendBuff and recvBuff registered as NCCL_WIN_COLL_SYMMETRIC.
//   - nlines divisible by lsaTeam.nRanks.
//   - Caller barriers all ranks before launch (sendBuff fully written).
//   - devComm created with lsaBarrierCount >= gridDim.x.

#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include "nccl_device.h"

#define TWOSHOT_MC_MAXTHREADS  1024

// ---------------------------------------------------------------------------
// twoshot_mc_simple_allreduce_kernel
//
// Parameters:
//   sendWin  — symmetric window over sendBuff (multicast VA for ld_reduce)
//   recvWin  — symmetric window over recvBuff (multicast VA for st)
//   nlines   — total float4 elements; must be divisible by lsaTeam.nRanks
//   devComm  — NCCL device comm (lsaMultimem=true; lsaBarrierCount >= gridDim.x)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TWOSHOT_MC_MAXTHREADS)
    twoshot_mc_simple_allreduce_kernel(ncclWindow_t sendWin,
                                       ncclWindow_t recvWin,
                                       size_t       nlines,
                                       ncclDevComm  devComm)
{
#if __CUDA_ARCH__ >= 900
    ncclTeam lsaTeam = ncclTeamLsa(devComm);
    int nRanks = lsaTeam.nRanks;
    int myRank = lsaTeam.rank;

    size_t chunkLines = nlines / nRanks;
    size_t myStart    = (size_t)myRank * chunkLines;

    // Divide chunkLines across blocks; each block uses barrier slot blockIdx.x.
    size_t blockChunkSize = (chunkLines + gridDim.x - 1) / gridDim.x;
    size_t blockStart     = (size_t)blockIdx.x * blockChunkSize;
    size_t blockEnd       = min(blockStart + blockChunkSize, chunkLines);
    if (blockStart >= chunkLines) return;

    void* mc_sendPtr = ncclGetLsaMultimemPointer(sendWin, 0, devComm);
    void* mc_recvPtr = ncclGetLsaMultimemPointer(recvWin, 0, devComm);

    // -----------------------------------------------------------------------
    // Fused ReduceScatter + AllGather.
    //
    // Each thread processes a strided range within this block's sub-chunk.
    //   ld_reduce: atomically reduces all ranks' sendBuff[myStart + idx]
    //              and returns the sum — no intermediate storage needed.
    //   multimem.st: broadcasts the result to ALL ranks' recvBuff[myStart + idx]
    //                in a single NVLink store.
    // -----------------------------------------------------------------------
    for (size_t i = threadIdx.x; i < (blockEnd - blockStart); i += blockDim.x) {
        size_t idx = myStart + blockStart + i;

        float4 result;
        asm volatile(
            "multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
            : "=f"(result.x), "=f"(result.y),
              "=f"(result.z), "=f"(result.w)
            : "l"((uintptr_t)mc_sendPtr + idx * sizeof(float4))
            : "memory");

        asm volatile(
            "multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};"
            :: "l"((uintptr_t)mc_recvPtr + idx * sizeof(float4)),
               "f"(result.x), "f"(result.y),
               "f"(result.z), "f"(result.w)
            : "memory");
    }

    // -----------------------------------------------------------------------
    // Barrier: wait for all ranks' AllGather stores to be globally visible.
    // -----------------------------------------------------------------------
    ncclLsaBarrierSession<ncclCoopCta> bar {
        ncclCoopCta(), devComm, lsaTeam, devComm.lsaBarrier, (uint32_t)blockIdx.x
    };
    bar.arrive(ncclCoopCta(), cuda::memory_order_release);
    bar.wait  (ncclCoopCta(), cuda::memory_order_acquire);

#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[twoshot_mc_simple_allreduce] ERROR: requires sm >= 90 (Hopper/Blackwell)\n");
    assert(false);
#endif
}
