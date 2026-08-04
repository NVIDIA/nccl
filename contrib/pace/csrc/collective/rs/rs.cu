// =============================================================================
// rs.cu
//
// All-ring reduce_scatter: BOTH intra-node and inter-node phases use ring
// reduce. Selected by RSComm host dispatch when num_nodes > 1.
//
// Design (see plan: RS All Ring Kernel):
//   - intra-node phase: chunk-shift ring across nvl_local_ranks. For each
//     inter-chunk c ∈ [0, N_n), one intra ring run produces this rank's
//     intra_partial for chunk c (i.e. sum over local ranks of input chunks
//     destined for global rank target_node*N_l + local_rank, where
//     target_node = (c - 1 + N_n) % N_n). The result is staged into
//     rdma_send[c] for the inter ring.
//   - inter-node phase: per-rail node-level chunk-shift ring identical to
//     rs_inter_ring. After (N_n - 1) ring steps, rdma_send[(node_id+1) % N_n]
//     holds the full sum for this rank's shard; written out via the
//     out_cast_if_needed epilog store.
//
// Buffer layout (per rank, byte-compatible with the reduced data-ring):
//   nvl region (gin_win base): per SM, (num_local_ranks + 1) slots.
//     Slots [c_intra ∈ [0, N_l)] hold partial chunks (rank-local), and the
//     last slot is a single recv buffer reused across intra steps via the
//     data/ack handshake.
//   rdma_send region: per SM, num_nodes slots (one per inter chunk c).
//   rdma_recv region: per SM, num_nodes slots (one per inter ring step k).
//
// Signals:
//   - intra_data_sig / intra_ack_sig per (sm): gmem_barrier sig area
//     (sig_ptr scheme), monotonic across launches via intra_gstart_ptr.
//   - inter_data_sig / inter_ack_sig per (sm, k_inter): GIN signals,
//     aligned with rs_inter_ring's namespace.
//
// Note: RS dispatches unconditionally to this kernel for both single-machine
// and multi-machine cases.
// =============================================================================

#include "util/error.hpp"
#include "collective/rs/rs.cuh"
#include "device/comm.cuh"
#include "util/math.hpp"
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <nccl_device/coop.h>
#include <nccl_device/core.h>
#include <nccl_device/gin.h>
#include <type_traits>
#include <cooperative_groups.h>
#include <nccl_device/gin/gdaki/gin_gdaki.h>

#include "collective/rs/rs_kernels.cuh"  // device kernels (extern template — see rs_extern_decls.cuh)
#include "rs_extern_decls.cuh"  // generated: extern template decls (symbols from build/gen/rs/*.cu)

namespace pace {

// =============================================================================
// Host launcher
// =============================================================================
void reduce_scatter_func(const int& type, const int& red_op,
    const float extra_mul, const float extra_post_mul, const int& out_cast_type, const int& rdma_unroll,
    uint64_t *args_cpu, uint64_t *args, size_t nargs,
    void *out_ptr, ncclWindow_t gin_win, void *gin_win_ptr,
    int *gmem_barrier, const int nvl_ring, const int rdma_ring,
    int round_n, bool use_wg, int rank, int num_local_ranks, int num_ranks,
    ncclDevComm devComm, const int num_sms, cudaStream_t stream,
    int arrival_sig_base, int cuda_device_id)
{
    bool data_aligned = true;
    for (size_t i = 0; i < nargs; ++i) {
        if (args_cpu[4 * i] % (type == RS_TYPE_FLOAT32 ? sizeof(float4) : sizeof(float2)) != 0) {
            data_aligned = false; break;
        }
        if (args_cpu[4 * i + 2] % 4 != 0) {
            data_aligned = false; break;
        }
    }
    int out_mode;
    if (red_op == RS_RED_OP_SUM && out_cast_type == RS_OUT_CAST_NONE) {
        out_mode = RS_OUT_MODE_DIRECT;
    } else if (red_op == RS_RED_OP_AVG && out_cast_type == RS_OUT_CAST_NONE) {
        out_mode = RS_OUT_MODE_AVG;
    } else if (red_op == RS_RED_OP_SUM && out_cast_type == RS_OUT_CAST_BF16) {
        out_mode = RS_OUT_MODE_CAST_BF16;
    } else {
        out_mode = RS_OUT_MODE_AVG_CAST_BF16;
    }
#define base_macro(is_aligned, p_out_mode, use_mul, data_type) {\
    auto func = reduce_scatter<data_type, use_mul, p_out_mode, is_aligned>;\
    func<<<num_sms, kRSRingThreads, 0, stream>>>(extra_mul, extra_post_mul, args, nargs, out_ptr,\
        gin_win, gin_win_ptr, gmem_barrier, nvl_ring, rdma_ring, rdma_unroll,\
        round_n, use_wg, rank, num_local_ranks, num_ranks, devComm, arrival_sig_base, cuda_device_id);\
}
#define L1_align_macro(p_out_mode, use_mul, data_type) {\
    SWITCH_ALIGN(data_aligned, base_macro, p_out_mode, use_mul, data_type);\
}
#define L2_macro(use_mul, data_type) {\
    SWITCH_OUT_MODE(out_mode, L1_align_macro, use_mul, data_type);\
}
#define L3_macro(data_type) {\
    SWITCH_MUL(extra_mul, extra_post_mul, L2_macro, data_type);\
}
    SWITCH_TYPE(type, L3_macro);
#undef base_macro
#undef L1_align_macro
#undef L2_macro
#undef L3_macro
    CUDACHECK(cudaGetLastError());
}

} // namespace pace
