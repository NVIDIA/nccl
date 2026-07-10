/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// lsa_poison_alltoall_kernel.cuh
//
// LSA AllToAll using Lamport-clock completion detection instead of an explicit
// barrier.  Only the .w field of each uint4 slot serves as the sentinel; it is
// re-poisoned in a separate pass on a DIFFERENT buffer (clearBuf), leaving
// recvBuff intact and readable by the caller after the kernel returns.
//
// Design (inspired by userbuffers_a2a_lamport):
//
//   Sentinel: .w == LSA_SENTINEL_POISON (0xFFFFFFFF, set by cudaMemset 0xFF).
//     Valid data must have .w != poison. NVLink issues uint4 stores as single
//     128-bit transactions, so .w != poison implies the full uint4 has arrived.
//
//   Separate clear buffer: after polling all slots in recvBuff, the kernel
//     re-poisons .w of every slot in clearBuf (the buffer used 2 iterations
//     ago).  recvBuff is left untouched so the caller can read the output
//     after the kernel returns without a copy.
//
//   Entry barrier (skip_barrier=false for first 2 calls): ensures all ranks
//     have finished pre-poisoning this iteration's buffer before any rank
//     begins writing. In steady state (skip_barrier=true) this is skipped.
//
//   Triple buffering: three symmetric windows rotate each iteration.
//     At iteration si, the host passes:
//       recvBuff / memWin  = bufs[si % 3]           (receives into this)
//       clearBuf           = bufs[(si + 2) % 3]     (re-poisons this)
//     A peer at most 1 iteration ahead writes to bufs[(si+1) % 3], so
//     clearBuf (bufs[(si+2)%3]) is never written concurrently.
//     By the time a buffer is reused (3 iterations later), it was cleared
//     2 iterations prior and all ranks have completed that clear.
//
// Buffer layout (each of the 3 symmetric windows, length = lsaSize * count):
//   recvBuff[r * count .. (r+1)*count - 1]  — slot written by LSA rank r
//
// Host setup:
//   Allocate 3 symmetric buffers; cudaMemset all with 0xFF before first call.
//   Pass bufs[si%3] as recvBuff/memWin, bufs[(si+2)%3] as clearBuf.
//   skip_barrier = (iter >= 2).

#pragma once

#include <cuda_runtime.h>
#include <cuda/atomic>
#include "nccl_device.h"

// Poison value for .w field.  Valid data must never produce this value.
// 0xFFFAFFFA matches the value used by userbuffers_a2a_lamport.
#define LSA_SENTINEL_POISON  0xFFFAFFFAu

// lsa_poison_alltoall_kernel
//
// Parameters:
//   sendBuff     — Local source. sendBuff[dest*count .. (dest+1)*count-1] → dest.
//   recvBuff     — Symmetric output buffer, lsaSize * count uint4.
//                  Registered as ncclWindow. .w must equal poison on entry.
//                  Left intact after return so the caller can read the result.
//   memWin       — ncclWindow_t handle for recvBuff.
//   count        — uint4 elements per rank-pair.
//   devComm      — ncclDevComm. lsaBarrier used only when skip_barrier=false.
//   clearBuf     — Buffer to re-poison after polling (bufs[(iter+2)%3] on host).
//                  Its .w fields are set to LSA_SENTINEL_POISON after all slots
//                  in recvBuff have been confirmed received.
//   skip_barrier — false for iter 0 and 1; true for all subsequent iterations.
//
__global__ void lsa_poison_alltoall_kernel(
    const uint4* sendBuff,
    uint4*       recvBuff,
    ncclWindow_t memWin,
    int          count,
    ncclDevComm  devComm,
    uint4*       clearBuf,
    bool         skip_barrier)
{
#if __CUDA_ARCH__ >= 700
    ncclTeam lsaTeam = ncclTeamLsa(devComm);
    int lsaSize = lsaTeam.nRanks;
    int myRank  = lsaTeam.rank;

    int chunkSize  = (count + gridDim.x - 1) / gridDim.x;
    int chunkStart = (int)blockIdx.x * chunkSize;
    int chunkEnd   = min(chunkStart + chunkSize, count);
    int chunkLen   = chunkEnd - chunkStart;

    if (chunkStart >= count) return;

    // ------------------------------------------------------------------
    // Entry barrier (skipped in steady state).
    //
    //   Ensures all ranks have finished poisoning recvBuff (via the inline
    //   clear from 2 iterations ago, or the initial cudaMemset) before any
    //   rank starts writing new data into it.
    // ------------------------------------------------------------------
    if (!skip_barrier) {
        ncclLsaBarrierSession<ncclCoopCta> bar {
            ncclCoopCta(), devComm, lsaTeam, devComm.lsaBarrier, (uint32_t)blockIdx.x
        };
        bar.arrive(ncclCoopCta(), cuda::memory_order_release);
        bar.wait  (ncclCoopCta(), cuda::memory_order_acquire);
    }

    // ------------------------------------------------------------------
    // Push: write this block's chunk to every peer's recvBuff slot.
    // Warps distributed round-robin across destinations keep all NVLink
    // links active simultaneously.
    // ------------------------------------------------------------------
    const int warpIdx      = threadIdx.x / 32;
    const int laneIdx      = threadIdx.x % 32;
    const int numWarps     = blockDim.x  / 32;
    const int warpsPerDest = max(1, numWarps / lsaSize);
    const int myDest       = warpIdx % lsaSize;
    const int myDestWarp   = (warpIdx / lsaSize) % warpsPerDest;

    int subLen   = (chunkLen + warpsPerDest - 1) / warpsPerDest;
    int subStart = myDestWarp * subLen;
    int subEnd   = min(subStart + subLen, chunkLen);

    for (int destOff = myDest; destOff < lsaSize; destOff += numWarps) {
        if (subStart < chunkLen) {
            uint4* peerSlot = (uint4*)ncclGetLsaPointer(
                memWin,
                (size_t)(myRank * count + chunkStart + subStart) * sizeof(uint4),
                destOff);
            const uint4* src = sendBuff + destOff * count + chunkStart + subStart;
            for (int i = laneIdx; i < (subEnd - subStart); i += 32)
                peerSlot[i] = src[i];
        }
    }

    // ------------------------------------------------------------------
    // Pass 1: Poll recvBuff.
    //
    //   Spin on recvBuff[...].w via cuda::atomic_ref with
    //   memory_order_relaxed and thread_scope_system.
    //   thread_scope_system relaxed atomics compile to ld.relaxed.sys,
    //   which bypasses L1 (L1 is not coherent at system scope).  Remote
    //   NVLink writes land in L2, so bypassing L1 is all that is needed
    //   to observe them.  No acquire fence is required: NVLink delivers
    //   the full uint4 as a single 128-bit transaction, so .x/.y/.z are
    //   already in L2 once .w != poison; and the caller reads the result
    //   after cudaStreamSynchronize which provides a full system barrier.
    //   recvBuff is left intact so the caller can read the result.
    // ------------------------------------------------------------------
    const int total = lsaSize * chunkLen;
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        int r   = t / chunkLen;
        int idx = t % chunkLen;

        cuda::atomic_ref<uint32_t, cuda::thread_scope_system> sentinel(
            (recvBuff + r * count + chunkStart + idx)->w);
        while (sentinel.load(cuda::memory_order_relaxed) == LSA_SENTINEL_POISON) {}
    }

    // ------------------------------------------------------------------
    // Pass 2: Re-poison clearBuf.
    //
    //   clearBuf is bufs[(iter+2)%3] on the host — the buffer used 2
    //   iterations ago.  Re-poisoning it here prepares it for reuse 1
    //   iteration later.  No fence between Pass 1 and Pass 2: they
    //   operate on independent buffers (recvBuff vs clearBuf).
    //   Relaxed stores suffice; triple-buffering guarantees clearBuf is
    //   not being read or written by any peer while we clear it.
    // ------------------------------------------------------------------
    for (int t = threadIdx.x; t < total; t += blockDim.x) {
        int r   = t / chunkLen;
        int idx = t % chunkLen;
        clearBuf[r * count + chunkStart + idx].w = LSA_SENTINEL_POISON;
    }

#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[custom_algos] ERROR: LSA requires sm >= 70\n");
    assert(false);
#endif
}
