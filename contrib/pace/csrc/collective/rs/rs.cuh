#ifndef PACE_KERNELS_INCLUDE_RS_ALL_RING_CUH
#define PACE_KERNELS_INCLUDE_RS_ALL_RING_CUH

#include "device/common.cuh"
#include "collective/rs/rs_defs.cuh"

namespace pace {

// All-ring reduce_scatter (intra-node + inter-node both use ring reduce).
//
// The sole RS implementation: RSComm dispatches this unconditionally for both
// single-machine and multi-machine cases.
//
// Buffer layout requirements (set up in RSComm constructor under multi-node
// branch):
//   per SM: nvl_ring + 2 * rdma_ring slots of (rdma_unroll * GIN_CTA_THREADS) f4.
//   region 0: nvl ring (single track per SM, used for intra-ring forwarding).
//   region 1: rdma send ring (per-SM, holds outbound partials for inter ring).
//   region 2: rdma recv ring (per-SM, receives inbound partials from prev node).
//
// Intra-ring signals live in gmem_barrier (sig_ptr scheme); inter-ring
// signals are GIN signals indexed per (sm, ring_step).
//
// Reuses RSComm's scratch buffers (gmem_barrier, gin_win_ptr) unchanged; the
// all-ring path uses ring reduce end-to-end.
//
// pipe_depth (D, default 1): seg-level pipeline depth.
//   D == 1: legacy serial schedule (intra ring all c -> inter ring all k -> next seg).
//   D >= 2: seg-level pipeline. rdma_send / rdma_recv buffers + GIN signals
//     for the inter-ring phase are organized along an extra (seg_slot=seg%D)
//     axis so that intra(seg) can overlap with inter(seg-1..seg-D+1).
//     Caller MUST guarantee rdma_ring_size >= D * num_nodes and gin_sigs
//     allocation is scaled by D (RSComm constructor handles both).
void reduce_scatter_func(const int& type, const int& red_op,
    const float extra_mul, const float extra_post_mul, const int& out_cast_type,
    const int& rdma_unroll,
    uint64_t *args_cpu, uint64_t *args, size_t nargs,
    void* out_ptr, ncclWindow_t gin_win, void *gin_win_ptr,
    int *gmem_barrier, const int nvl_ring, const int rdma_ring,
    int round_n, bool use_wg, int rank, int num_local_ranks, int num_ranks,
    ncclDevComm devComm, const int num_sms, cudaStream_t stream,
    int arrival_sig_base = 0, int cuda_device_id = 0);

}

constexpr int kRSRingWarpGroups = 4;

#endif
