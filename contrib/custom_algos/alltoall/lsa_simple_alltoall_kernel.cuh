/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// lsa_simple_alltoall_kernel.cuh
//
// Sample custom CUDA kernel demonstrating the public NCCL LSA (Load-Store
// Access) device API to perform AllToAll within the LSA team (same NVLink
// island / node).
//
// What it does:
//   Every LSA rank sends `count` uint4 (128-bit) elements to every other LSA
//   rank.  After the kernel:
//     recvBuff[peer * count + i] == sendBuff[peer * count + i]
//   as seen on rank peer, for all peer in [0, lsaSize), i in [0, count).
//
// Source buffer (sendBuff, local — not peer-accessible):
//   sendBuff[dest * count .. (dest+1)*count - 1]  — data to send to dest.
//
// Buffer layout (recvBuff, symmetric window, length >= lsaSize * count uint4):
//   recvBuff[r * count .. (r+1)*count - 1]  — slot for data written by LSA rank r
//   Layout mirrors the UB convention: sender fills contiguous per-dest slabs.
//   Must be 16-byte aligned; count must be a multiple of 1 (uint4 units).
//
// Push design (reads src HBM → writes to remote dest HBM via NVLink):
//   Each rank reads from its local ptr_in slab and writes directly into the
//   corresponding slot on every peer's buffer using ncclGetLsaPointer.
//   NVLink writes are one-directional (no round-trip ACK), giving higher
//   throughput than pulling (loads) from remote memory.
//
// Multi-CTA, warp-parallel design:
//   The element range [0, count) is divided across gridDim.x blocks.  Each
//   block handles its chunk [chunkStart, chunkEnd) for ALL destinations, using
//   barrier slot blockIdx.x.
//
//   Within each block, warps are distributed across destinations in round-robin
//   order (warp w → destination w % lsaSize).  Multiple warps cover the same
//   destination when blockDim.x/32 > lsaSize; they split the chunk evenly.
//   This keeps all lsaSize NVLink links active simultaneously and hides HBM
//   load latency with sufficient warp parallelism.
//
//   Recommended blockDim.x: lsaSize * 32 * k (k ≥ 1), capped at 1024.
//   For lsaSize = 8: 256 (k=1) → 1 warp/dest; 512 (k=2) → 2 warps/dest.
//
//   Host must set lsaBarrierCount >= gridDim.x.

#pragma once

#include <cuda_runtime.h>
#include "nccl_device.h"

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

// lsa_simple_alltoall_kernel
//
// Launch with gridDim.x blocks and blockDim.x threads per block.
// Each block covers a contiguous chunk of the count elements and uses
// barrier slot blockIdx.x.  Blocks where chunkStart >= count return
// immediately (safe for any grid size >= 1).
//
// Parameters:
//   sendBuff — Local source buffer. sendBuff[dest * count .. (dest+1)*count-1]
//              is the data this rank sends to dest. Must hold at least
//              lsaSize * count uint4 (only entries for lsaSize peers are read).
//              Must be 16-byte aligned.
//   recvBuff — Symmetric device buffer of at least lsaSize * count uint4,
//              registered as an ncclWindow with ncclCommWindowRegister.
//              Must be 16-byte aligned.
//   memWin   — ncclWindow_t handle for recvBuff.
//   count    — Number of uint4 (128-bit) elements to exchange per rank-pair.
//   devComm  — ncclDevComm passed by value. Must be created with
//              lsaBarrierCount >= gridDim.x.
//
__global__ void lsa_simple_alltoall_kernel(
    const uint4* sendBuff,
    uint4*       recvBuff,
    ncclWindow_t memWin,
    int          count,
    ncclDevComm  devComm)
{
#if __CUDA_ARCH__ >= 700
    ncclTeam lsaTeam = ncclTeamLsa(devComm);
    int lsaSize = lsaTeam.nRanks;
    int myRank  = lsaTeam.rank;

    // Divide [0, count) evenly across blocks; last block may get fewer elements.
    int chunkSize  = (count + gridDim.x - 1) / gridDim.x;
    int chunkStart = (int)blockIdx.x * chunkSize;
    int chunkEnd   = min(chunkStart + chunkSize, count);
    int chunkLen   = chunkEnd - chunkStart;

    if (chunkStart >= count) return;   // surplus block — nothing to do

    // ------------------------------------------------------------------
    // Step 1: Push this block's chunk into every peer's buffer.
    //
    //   Warps are assigned to destinations in round-robin order so all
    //   lsaSize NVLink links are active simultaneously.  When there are
    //   more warps than destinations (warpsPerDest > 1), each destination's
    //   chunk is split further across warps to maximise HBM parallelism.
    //   Within each warp, 32 lanes stride over elements (coalesced access).
    // ------------------------------------------------------------------
    const int warpIdx    = threadIdx.x / 32;
    const int laneIdx    = threadIdx.x % 32;
    const int numWarps   = blockDim.x  / 32;

    // How many warps service each destination (≥ 1).
    // If numWarps < lsaSize some warps handle multiple destinations (outer
    // loop below); if numWarps ≥ lsaSize use all extra warps for the same
    // destination to improve HBM parallelism.
    const int warpsPerDest = max(1, numWarps / lsaSize);

    // Which destination this warp is responsible for.
    const int myDest = warpIdx % lsaSize;

    // Which sub-warp this is within its destination (0 .. warpsPerDest-1).
    const int myDestWarp = (warpIdx / lsaSize) % warpsPerDest;

    // Element sub-range within [0, chunkLen) owned by this warp.
    int subLen   = (chunkLen + warpsPerDest - 1) / warpsPerDest;
    int subStart = myDestWarp * subLen;
    int subEnd   = min(subStart + subLen, chunkLen);

    // Outer loop: needed only when numWarps < lsaSize (each warp covers
    // multiple destinations sequentially).  In the common case (numWarps
    // >= lsaSize) this executes exactly once.
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
    // Step 2: Inter-rank barrier for this chunk.
    //
    //   Small-message mode (numWarps == lsaSize, one warp per dest):
    //     All threads issue a system-scope release fence to make their
    //     NVLink stores globally visible, then only warp 0 runs the
    //     barrier session via ncclCoopWarpSpan.  The three coop.sync()
    //     calls inside arrive/wait/destructor become __barrier_sync_count
    //     (trivially fast for a lockstep warp).  arrive() is relaxed
    //     because the explicit fence already provides release ordering.
    //     A trailing __syncthreads() broadcasts warp 0's acquire to the
    //     whole CTA.
    //
    //   Large-message mode (numWarps > lsaSize, multiple warps per dest):
    //     Use ncclCoopCta so all warps' fences are folded into arrive's
    //     own coop.sync()+fence, avoiding the extra named-barrier tracking
    //     overhead that ncclCoopWarpSpan incurs at high warp counts.
    // ------------------------------------------------------------------
    if (numWarps == lsaSize) {
        // Small-message path.
        __syncthreads();
        cuda::atomic_thread_fence(cuda::memory_order_release);  // system scope

        if (warpIdx == 0) {
            ncclCoopWarpSpan barCoop{0, 1, 0};
            ncclLsaBarrierSession<ncclCoopWarpSpan> bar {
                barCoop, devComm, lsaTeam, devComm.lsaBarrier, (uint32_t)blockIdx.x
            };
            bar.arrive(barCoop, cuda::memory_order_relaxed);
            bar.wait  (barCoop, cuda::memory_order_acquire);
        }

        __syncthreads();
    } else {
        // Large-message path: all threads participate via ncclCoopCta.
        ncclLsaBarrierSession<ncclCoopCta> bar {
            ncclCoopCta(), devComm, lsaTeam, devComm.lsaBarrier, (uint32_t)blockIdx.x
        };
        bar.arrive(ncclCoopCta(), cuda::memory_order_release);
        bar.wait  (ncclCoopCta(), cuda::memory_order_acquire);
    }

    // After this point: recvBuff[r * count + chunkStart..chunkEnd-1] holds the
    // data that rank r sent to us for this chunk.

#else
    if (threadIdx.x == 0 && blockIdx.x == 0)
        printf("[sample_kernel] ERROR: LSA requires sm >= 70\n");
    assert(false);
#endif
}
