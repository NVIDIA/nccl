/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// oneshot_lsa_push_ll_allreduce_kernel.cuh
//
// One-shot "push" AllReduce over LSA peer memory with LL (low-latency) flags
// instead of a barrier.  Same data path as oneshot_lsa_push_allreduce_kernel,
// for PCIe peer-to-peer GPUs without NVLink multicast.
//
// It uses the device API's ncclLLA2ASession.  send() writes each 8 bytes of
// payload as one 16-byte store {data0, epoch, data1, epoch} (st.relaxed.sys)
// into the peer's LL buffer; recv() polls the local buffer until both epoch
// words match.  Each 8-byte half carries its own flag, so the protocol only
// relies on 8-byte store atomicity, which also holds over PCIe (NCCL's LL
// protocol makes the same assumption).  The cost is 2x the bytes on the wire.
// In exchange there is no memory fence, no barrier and no round trip: each
// thread proceeds as soon as its own elements have arrived.
//
// The session keeps its epoch in device memory and double-buffers the slots
// by epoch parity, so there is no host-side slot or flag argument and the
// kernel can be replayed from a CUDA graph.
//
// Algorithm (per CTA, which owns one contiguous chunk of the message):
//   1. Push   - send(peer, idx, sendLocal[i]) for every peer.  idx is
//               (my index among the peer's peers) * chunk + (i - lo).
//   2. Reduce - recvLocal[i] = sum over ranks r (in rank order) of
//               (r == myRank ? sendLocal[i] : recv(idx of r)).
//   3. endEpoch.
//
// T is the element type (float, __half or __nv_bfloat16), as in the
// barrier variant.  The LL lines carry 16-byte payloads for every type.
//
// Requirements:
//   - ll: from ncclLLA2ACreateRequirement(nBlocks >= gridDim.x, nSlots) with
//     nSlots >= ncclLLA2ACalcSlots((lsaSize - 1) * chunkLines, sizeof(float4)),
//     where chunkLines = lsaPushLLChunkLines(nlines, gridDim.x).
//   - Every rank launches the same grid with the same nlines.
//   - sendLocal may alias recvLocal (in place).

#pragma once

#include <cuda_runtime.h>
#include "nccl_device.h"
#include "lsa_push_reduce.cuh"

#define LSA_PUSH_LL_MAXTHREADS 1024

__host__ __device__ __forceinline__ size_t lsaPushLLChunkLines(size_t nlines, int nBlocks) {
    size_t c = (nlines + nBlocks - 1) / nBlocks;
    return (c + 31) & ~(size_t)31;
}

template <typename T = float>
__global__ void __launch_bounds__(LSA_PUSH_LL_MAXTHREADS)
    oneshot_lsa_push_ll_allreduce_kernel(const float4*     sendLocal,
                                         float4*           recvLocal,
                                         ncclLLA2AHandle_t ll,
                                         size_t            nlines,
                                         ncclDevComm       devComm)
{
    const ncclTeam team = ncclTeamLsa(devComm);
    const size_t chunk = lsaPushLLChunkLines(nlines, gridDim.x);
    const size_t lo = (size_t)blockIdx.x * chunk;
    if (lo >= nlines) return;                // same on every rank: block epoch unchanged
    const int n = (int)(((lo + chunk < nlines) ? lo + chunk : nlines) - lo);
    const int pitch = (team.nRanks - 1) * (int)chunk;  // elements this CTA receives per call
    if (2 * pitch > (int)ll.nSlots) __trap();          // LL buffer too small for this chunk

    ncclLLA2ASession<ncclCoopCta> ses { ncclCoopCta(), devComm, team, ll, blockIdx.x, pitch };

    // 1. Push my chunk to every peer.
    for (int p = 1; p < team.nRanks; p++) {
        int peer = team.rank + p;
        if (peer >= team.nRanks) peer -= team.nRanks;
        const int idx = (team.rank < peer ? team.rank : team.rank - 1) * (int)chunk;
        for (int j = threadIdx.x; j < n; j += blockDim.x)
            ses.send(peer, idx + j, sendLocal[lo + j]);
    }

    // 2. Receive and reduce in rank order.
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float4 acc;
        for (int r = 0; r < team.nRanks; r++) {
            float4 v = (r == team.rank)
                ? sendLocal[lo + j]
                : ses.recv<float4>((r < team.rank ? r : r - 1) * (int)chunk + j);
            acc = (r == 0) ? v : LsaPushSum<T>::add(acc, v);
        }
        recvLocal[lo + j] = acc;
    }

    ses.endEpoch(ncclCoopCta());
}
