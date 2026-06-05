/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// twoshot_mc_poison_allreduce_kernel.cuh
//
// Two-shot NVLink multicast AllReduce with Lamport sentinel completion:
//   Shot 1 — ReduceScatter: each rank reduces its assigned chunk via
//             multimem.ld_reduce.add.v4.f32 (hardware reduction).
//             AllGather write: result immediately broadcast to ALL ranks'
//             recvBuf via multimem.st (single NVLink multicast store).
//   Shot 2 — Poll: each rank spins on its local recvBuf until all other
//             ranks' chunks arrive (.w transitions from LSA_POISON to data).
//   Clear  — Re-poison clearBuf (this rank's chunk of the next-next recvBuf)
//             inline, eliminating a separate cudaMemset call.
//
// No explicit exit barrier: completion is detected via the Lamport sentinel
// (.w == LSA_POISON before arrival; .w == data after multimem.st).
// An entry barrier (skip_barrier=false for the first 2 launches per handle)
// ensures all ranks have finished re-poisoning before any rank overwrites
// with new data.
//
// Buffer layout (each of 3 symmetric recvBufs, length = nlines float4):
//   recvBuf[r * chunkLines .. (r+1)*chunkLines - 1]  — chunk written by rank r
//   chunkLines = nlines / nRanks
//
// Triple-buffer rotation (managed by caller):
//   Iter k: recvBuf = rotBuf[k%3],  clearBuf = rotBuf[(k+2)%3]
//   skip_barrier = false for k=0,1,2; true thereafter (encoded in handle by lib).
//
// Requirements:
//   - sm >= 90 (Hopper / Blackwell) for multimem PTX.
//   - sendBuff and all 3 rotBufs registered as NCCL_WIN_COLL_SYMMETRIC.
//   - nlines divisible by nRanks; count (floats) divisible by 4*nRanks.
//   - Caller barriers all ranks before launch (sendBuff fully written).
//   - devComm created with lsaMultimem=true; lsaBarrierCount >= gridDim.x.

#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include "nccl_device.h"

// LSA_POISON is defined in lsa_poison_alltoall_kernel.cuh (included before this).
// Re-declare guard to avoid requiring include order.
#ifndef LSA_POISON
#define LSA_POISON  0xFFFAFFFAu
#endif

#define TWOSHOT_MC_POISON_MAXTHREADS  1024

// ---------------------------------------------------------------------------
// twoshot_mc_poison_allreduce_kernel
//
// Parameters:
//   sendWin      — symmetric window over sendBuff (multicast VA for ld_reduce)
//   recvBuf      — local pointer to this rank's current recvBuf (for polling)
//   recvWin      — symmetric window over recvBuf   (multicast VA for st)
//   clearBuf     — local pointer to this rank's next-next recvBuf (.w re-poisoned)
//   nlines       — total float4 elements; must be divisible by nRanks
//   devComm      — ncclDevComm (lsaMultimem=true; lsaBarrierCount >= gridDim.x)
//   skip_barrier — false for first 2 launches per handle; true in steady state
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TWOSHOT_MC_POISON_MAXTHREADS)
    twoshot_mc_poison_allreduce_kernel(ncclWindow_t sendWin,
                                       float4*      recvBuf,
                                       ncclWindow_t recvWin,
                                       float4*      clearBuf,
                                       size_t       nlines,
                                       ncclDevComm  devComm,
                                       bool         skip_barrier)
{
#if __CUDA_ARCH__ >= 900
    ncclTeam lsaTeam = ncclTeamLsa(devComm);
    int nRanks = lsaTeam.nRanks;
    int myRank = lsaTeam.rank;

    size_t chunkLines = nlines / nRanks;
    size_t myStart    = (size_t)myRank * chunkLines;

    // Divide my chunk across active blocks; each active block uses barrier slot blockIdx.x.
    size_t blockChunkSize = (chunkLines + gridDim.x - 1) / gridDim.x;
    size_t blockStart     = (size_t)blockIdx.x * blockChunkSize;
    size_t blockEnd       = min(blockStart + blockChunkSize, chunkLines);
    if (blockStart >= chunkLines) return;

    // Number of active blocks: ceil(chunkLines / blockChunkSize).
    size_t nActiveBlocks = (chunkLines + blockChunkSize - 1) / blockChunkSize;

    // -----------------------------------------------------------------------
    // Entry barrier (skipped in steady state).
    //
    //   Ensures all ranks have finished re-poisoning this buffer (via the
    //   inline clear 2 iterations ago, or the initial pre-poisoning) before
    //   any rank begins writing new data into it.
    // -----------------------------------------------------------------------
    if (!skip_barrier) {
        ncclLsaBarrierSession<ncclCoopCta> bar {
            ncclCoopCta(), devComm, lsaTeam, devComm.lsaBarrier, (uint32_t)blockIdx.x
        };
        bar.arrive(ncclCoopCta(), cuda::memory_order_release);
        bar.wait  (ncclCoopCta(), cuda::memory_order_acquire);
    }

    void* mc_sendPtr = ncclGetLsaMultimemPointer(sendWin, 0, devComm);
    void* mc_recvPtr = ncclGetLsaMultimemPointer(recvWin, 0, devComm);

    // -----------------------------------------------------------------------
    // Phase 1: ReduceScatter + AllGather write (fused per element).
    //
    //   ld_reduce: hardware-reduces all ranks' sendBuff[myStart+idx] in one
    //              NVLink operation and returns the sum.
    //   multimem.st: broadcasts the reduced value to ALL ranks' recvBuf[myStart+idx]
    //                in a single NVLink multicast store.  After this store,
    //                .w of recvBuf[myStart+idx] on every rank transitions from
    //                LSA_POISON to the real data value.
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
    // Phase 2: Poll local recvBuf until all other ranks' chunks arrive.
    //
    //   Distribute all nlines elements evenly across the active blocks.
    //   My own chunk (myStart..myStart+chunkLines-1) was written in Phase 1
    //   and will pass the sentinel check immediately.
    //
    //   ld.relaxed.sys: bypasses L1 (same cache behaviour as ld.acquire.sys)
    //   but emits no per-load fence.  NVLink issues float4/uint4 stores as
    //   single 128-bit transactions, so .w != poison implies the full float4
    //   has landed in L2.
    // -----------------------------------------------------------------------
    size_t pollChunk = (nlines + nActiveBlocks - 1) / nActiveBlocks;
    size_t pollStart = min((size_t)blockIdx.x * pollChunk, nlines);
    size_t pollEnd   = min(pollStart + pollChunk, nlines);

    for (size_t i = threadIdx.x; i < (pollEnd - pollStart); i += blockDim.x) {
        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> sentinel(
            reinterpret_cast<uint32_t&>(recvBuf[pollStart + i].w));
        while (sentinel.load(cuda::memory_order_relaxed) == LSA_POISON) {}
    }

    // -----------------------------------------------------------------------
    // Phase 3: Re-poison clearBuf (my chunk only — each rank handles its own).
    //
    //   No fence between Phase 2 and Phase 3: recvBuf (read in Phase 2) and
    //   clearBuf (written here) are independent buffers.
    // -----------------------------------------------------------------------
    for (size_t i = threadIdx.x; i < (blockEnd - blockStart); i += blockDim.x) {
        clearBuf[myStart + blockStart + i].w = __int_as_float(LSA_POISON);
    }

#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[twoshot_mc_poison_allreduce] ERROR: requires sm >= 90 (Hopper/Blackwell)\n");
    assert(false);
#endif
}
