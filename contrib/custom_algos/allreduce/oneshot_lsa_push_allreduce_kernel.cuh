/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// oneshot_lsa_push_allreduce_kernel.cuh
//
// One-shot "push" AllReduce over LSA (load/store accessible) peer memory.
// Written for GPUs that reach each other through PCIe peer-to-peer and have
// no NVLink multicast, e.g. two T4s behind a PCIe host bridge.
//
// Each rank stores its own input into every peer's scratch window with plain
// st.global through ncclGetLsaPointer.  Remote stores over PCIe are posted
// writes, so the sender never waits for a round trip.  One LSA barrier per
// CTA (release on arrive, acquire on wait) publishes the stores.  Each rank
// then reduces its own input with the copies it received, reading only local
// memory.  Remote reads (a "pull" one-shot) cost a PCIe round trip per load,
// which is why this kernel pushes instead.
//
// Algorithm (per CTA, which owns one contiguous chunk of the message):
//   1. Push    - copy sendLocal[chunk] into scratch[slot][myRank][chunk] on
//                every peer.
//   2. Barrier - ncclLsaBarrierSession::sync(acq_rel).
//   3. Reduce  - recvLocal[i] = sum over ranks r (in rank order) of
//                (r == myRank ? sendLocal[i] : scratch[slot][r][i]).
//
// Scratch is double-buffered by 'slot'.  Consecutive calls must alternate
// slot (0, 1, 0, ...).  This is safe with a single barrier per call: a rank
// can only start call k+2 after its call k+1 barrier, and that barrier needs
// every peer to have started call k+1, which runs after the peer's call k
// (stream order) finished reading the slot.
//
// T is the element type (float, __half or __nv_bfloat16).  Data moves as
// 16-byte lines (float4) for every type; only the local add depends on T
// (lsa_push_reduce.cuh).  nlines counts 16-byte lines.
//
// Summation order is the same on every rank, so all ranks get bit-identical
// results.  sendLocal may alias recvLocal (in place).
//
// Requirements:
//   - ncclDevComm with lsaBarrierCount >= gridDim.x.
//   - Every rank launches the same grid with the same nlines.
//   - scratchWin: symmetric window of at least
//     2 * lsaSize * slotLines * sizeof(float4) bytes, nlines <= slotLines.
//   - CUDA P2P between all ranks of the LSA team (no NVLink needed).

#pragma once

#include <cuda_runtime.h>
#include "nccl_device.h"
#include "lsa_push_reduce.cuh"

#define LSA_PUSH_MAXTHREADS 1024

// Lines of one CTA's chunk.  Rounded up to 32 lines (512 B) so that chunks
// start on a 512 B boundary.
__device__ __forceinline__ size_t lsaPushChunkLines(size_t nlines, int nBlocks) {
    size_t c = (nlines + nBlocks - 1) / nBlocks;
    return (c + 31) & ~(size_t)31;
}

template <typename T = float>
__global__ void __launch_bounds__(LSA_PUSH_MAXTHREADS)
    oneshot_lsa_push_allreduce_kernel(const float4* sendLocal,
                                      float4*       recvLocal,
                                      ncclWindow_t  scratchWin,
                                      size_t        slotLines,
                                      size_t        nlines,
                                      int           slot,
                                      ncclDevComm   devComm)
{
    const ncclTeam team = ncclTeamLsa(devComm);
    if (nlines > slotLines) __trap();        // scratch too small for this message
    const size_t chunk = lsaPushChunkLines(nlines, gridDim.x);
    const size_t lo = (size_t)blockIdx.x * chunk;
    if (lo >= nlines) return;                // same on every rank: barrier slot unused
    const size_t hi = (lo + chunk < nlines) ? lo + chunk : nlines;
    const size_t slotBase = (size_t)slot * team.nRanks;

    ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x };

    // 1. Push my chunk into every peer's scratch.
    const size_t myOff = (slotBase + team.rank) * slotLines * sizeof(float4);
    for (int p = 1; p < team.nRanks; p++) {
        int peer = team.rank + p;
        if (peer >= team.nRanks) peer -= team.nRanks;
        float4* dst = (float4*)ncclGetLsaPointer(scratchWin, myOff, peer);
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x)
            dst[i] = sendLocal[i];
    }

    // 2. Release my stores to the peers, acquire theirs.
    bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);

    // 3. Reduce from local memory only, in rank order.
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float4 acc;
        for (int r = 0; r < team.nRanks; r++) {
            const float4* src = (r == team.rank)
                ? sendLocal
                : (const float4*)ncclGetLocalPointer(scratchWin, (slotBase + r) * slotLines * sizeof(float4));
            acc = (r == 0) ? src[i] : LsaPushSum<T>::add(acc, src[i]);
        }
        recvLocal[i] = acc;
    }
}
