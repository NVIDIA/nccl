/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "nccl_device.h"
#include "hybridep_adapter.cuh"
#include "hybridep_configs.cuh"
#include "hybrid_ep.cuh"
#include "include/common.hpp"
#include "jit/combine_jit.cuh"
#include "jit/dispatch_jit.cuh"
#include "jit/preprocess_jit.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace nccl_ep {
namespace hybridep {

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
__global__ void convert_topk_to_routing_map_kernel(
    const int64_t* __restrict__ topk_idx,    // [num_tokens, num_topk]
    uint8_t* __restrict__ routing_bitmap,     // [num_tokens, num_experts_packed]
    int64_t* __restrict__ cached_topk_idx,    // [num_tokens, num_topk]; nullable
    int num_tokens,
    int num_topk,
    int num_experts_packed                    // = ceil(num_experts / 8)
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Each thread exclusively owns its row -- no atomics needed.
    // Zero the row before OR-ing in bits; the caller does not pre-zero.
    uint8_t* row = routing_bitmap + token * num_experts_packed;
    for (int b = 0; b < num_experts_packed; b++) row[b] = 0;
    const int64_t* in_row  = topk_idx + token * num_topk;
    int64_t*       out_row = cached_topk_idx ? cached_topk_idx + token * num_topk : nullptr;
    for (int k = 0; k < num_topk; k++) {
        int64_t expert = in_row[k];
        if (out_row) out_row[k] = expert;
        if (expert >= 0) {
            row[expert / 8] |= (1u << (expert % 8));
        }
    }
}

// ============================================================================
// Convert topk to bitmap routing map
// ============================================================================
void convert_topk_to_routing_map(
    const int64_t* topk_idx,
    uint8_t* routing_bitmap,
    int64_t* cached_topk_idx,
    int num_tokens,
    int num_topk,
    int num_experts_packed,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    convert_topk_to_routing_map_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, routing_bitmap, cached_topk_idx, num_tokens, num_topk, num_experts_packed);
}

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob
// ============================================================================
__global__ void sparse_to_dense_prob_kernel(
    const int64_t* __restrict__ topk_idx,      // [num_tokens, topk]
    const float* __restrict__ topk_weights,    // [num_tokens, topk]
    float* __restrict__ dense_prob,            // [num_tokens, num_experts]
    int num_tokens,
    int num_topk,
    int num_experts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token = tid / num_topk;
    int k = tid % num_topk;

    if (token >= num_tokens) return;

    int64_t expert = topk_idx[token * num_topk + k];
    float weight = topk_weights[token * num_topk + k];

    // Scatter weight to the correct expert position
    if (expert >= 0 && expert < num_experts) {
        dense_prob[token * num_experts + expert] = weight;
    }
}

// ============================================================================
// Convert sparse to dense prob
// ============================================================================
void sparse_to_dense_prob(
    const int64_t* topk_idx,
    const float* topk_weights,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream
) {
    int total_elements = num_tokens * num_topk;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    sparse_to_dense_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, topk_weights, dense_prob, num_tokens, num_topk, num_experts);
}

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob for combine input
// ============================================================================
// Used for combine backward pass. Uses local_expert_routing_map to determine
// which experts each token is routed to, matching the order from dispatch output.
// Each thread handles one token.
__global__ void sparse_to_dense_prob_combine_kernel(
    const float* __restrict__ topk_weights,           // [num_tokens, topk]
    const bool* __restrict__ local_expert_routing_map, // [num_tokens, experts_per_rank]
    float* __restrict__ dense_prob,                   // [num_tokens, experts_per_node]
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Scan local experts in order (matches dense_to_sparse_prob output order)
    int k_in = 0;
    for (int e = 0; e < experts_per_rank && k_in < num_topk; e++) {
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // This expert is active for this token - take next weight from sparse input
            float weight = topk_weights[token * num_topk + k_in];

            // Place at correct position in dense matrix
            // Local expert e on local_rank maps to: local_rank * experts_per_rank + e
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
            dense_prob[dense_idx] = weight;

            k_in++;
        }
    }
}

// ============================================================================
// Convert sparse to dense prob for combine input
// ============================================================================
void sparse_to_dense_prob_combine(
    const float* topk_weights,
    const bool* local_expert_routing_map,
    float* dense_prob,
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    sparse_to_dense_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_weights, local_expert_routing_map, dense_prob, num_tokens, num_topk,
        experts_per_rank, experts_per_node, local_rank);
}

// ============================================================================
// Kernel: Convert dense prob output to sparse format
// ============================================================================
// One thread per token. Output by layout:
//   FLAT/RM: recv_topk_weights[token, k_out] zero-filled tail; recv_topk_idx parallel.
//   EM:      recv_topk_weights[token] (single scalar; slot = (token, local_expert)); recv_topk_idx unused.
__global__ void dense_to_sparse_prob_kernel(
    const float* __restrict__ dense_prob,              // [num_recv_tokens, experts_per_node]
    const bool* __restrict__ local_expert_routing_map, // [num_recv_tokens, experts_per_rank]
    float* __restrict__ recv_topk_weights,             // EM: [N]; FLAT/RM: [N, topk]
    int64_t* __restrict__ recv_topk_idx,               // [num_recv_tokens, topk]; nullptr under EM
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    bool expert_major
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_recv_tokens) return;

    if (expert_major) {
        // Each slot has at most one matching local expert (the one defining the slot).
        // Write the single scalar weight at recv_topk_weights[token]; default 0.
        float weight = 0.0f;
        for (int e = 0; e < experts_per_rank; e++) {
            if (local_expert_routing_map[token * experts_per_rank + e]) {
                int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
                weight = dense_prob[dense_idx];
                break;
            }
        }
        recv_topk_weights[token] = weight;
        return;
    }

    int k_out = 0;

    // Scan local experts (the ones this rank is responsible for)
    for (int e = 0; e < experts_per_rank && k_out < topk; e++) {
        // Check if this token is routed to expert e
        if (local_expert_routing_map[token * experts_per_rank + e]) {
            // Use local expert id for NCCL API compatibility (expects 0-based local indices)
            int64_t local_expert = static_cast<int64_t>(e);

            // Get weight from dense output (indexed by local expert within node)
            // dense_prob layout: [token, experts_per_node] where experts_per_node = experts_per_rank * ranks_per_node
            // Local rank's experts are at offset: local_rank * experts_per_rank
            int dense_idx = token * experts_per_node + local_rank * experts_per_rank + e;
            float weight = dense_prob[dense_idx];

            // Write outputs
            if (recv_topk_idx != nullptr) {
                recv_topk_idx[token * topk + k_out] = local_expert;
            }
            recv_topk_weights[token * topk + k_out] = weight;
            k_out++;
        }
    }

    // Zero-fill remaining topk slots if fewer than topk experts found
    for (; k_out < topk; k_out++) {
        if (recv_topk_idx != nullptr) {
            recv_topk_idx[token * topk + k_out] = -1;  // Invalid expert marker
        }
        recv_topk_weights[token * topk + k_out] = 0.0f;
    }
}

// O(top_k) lookup from cached_topk_idx; k-slot order preserves FWD input.
__global__ void dense_to_sparse_prob_combine_kernel(
    const float* __restrict__ dense_prob,         // [num_tokens, num_experts]
    const int64_t* __restrict__ cached_topk_idx,  // [num_tokens, topk]
    float* __restrict__ combined_topk_weights,    // [num_tokens, topk]
    int num_tokens,
    int topk,
    int num_experts
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    for (int k = 0; k < topk; k++) {
        int64_t e = cached_topk_idx[token * topk + k];
        float weight = (e >= 0 && e < num_experts)
            ? dense_prob[token * num_experts + e]
            : 0.0f;
        combined_topk_weights[token * topk + k] = weight;
    }
}

void dense_to_sparse_prob_combine(
    const float* dense_prob,
    const int64_t* cached_topk_idx,
    float* combined_topk_weights,
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, cached_topk_idx, combined_topk_weights,
        num_tokens, topk, num_experts);
}


// ============================================================================
// Dense to sparse prob
// ============================================================================
void dense_to_sparse_prob(
    const float* dense_prob,
    const bool* local_expert_routing_map,
    float* recv_topk_weights,
    int64_t* recv_topk_idx,
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank,
    bool expert_major,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_recv_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, local_expert_routing_map, recv_topk_weights, recv_topk_idx,
        num_recv_tokens, topk, experts_per_rank, experts_per_node, local_rank,
        expert_major);
}

// ============================================================================
// Call metadata preprocessing
// ============================================================================
void call_metadata_preprocessing(
    const uint8_t* global_routing_map,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    void* token_rank_mask,
    int32_t* num_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    void* scan_tmp,
    int node_rank,
    int local_rank,
    int num_tokens_per_rank,
    int num_nodes,
    int num_ranks_per_node,
    int experts_per_rank,
    bool     expert_major,
    int64_t* internal_offsets,
    void*    padded_out_counts,
    void*    out_offsets,
    size_t alignment,
    int32_t* actual_counts_out,
    int s2d_inner_dim,
    void*    recv_total_counter,
    bool     out_is_int64,
    int      max_recv_tokens_per_rank,
    int      num_blocks,
    void*    scan_gscratch,
    cudaStream_t stream
) {
    if (expert_major && per_expert_token_counts == nullptr) {
        EP_HOST_ASSERT(false && "EXPERT_MAJOR remap requires per_expert_token_counts != nullptr");
    }
    if (expert_major) {
        EP_HOST_ASSERT(scan_gscratch != nullptr &&
                       "EM scan requires scan_gscratch != nullptr");
    }

    if (per_expert_token_counts != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(per_expert_token_counts, 0, experts_per_rank * sizeof(int32_t), stream));
    }

    constexpr int NUM_THREADS_PER_BLOCK = HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING;
    const int NUM_OF_BLOCKS = num_blocks;
    constexpr int NUM_OF_WARPS_PER_BLOCK_SCAN = NUM_THREADS_PER_BLOCK / 32;

    const size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(scan_tmp, 0, preprocessing_tmp_sz, stream));

    const size_t scan_smem_size =
        (2 * NUM_OF_WARPS_PER_BLOCK_SCAN * num_ranks_per_node * sizeof(int32_t)) +
        (num_ranks_per_node * sizeof(int32_t)) +
        (per_expert_token_counts != nullptr ? experts_per_rank * sizeof(int32_t) : 0);
    const size_t remap_smem_size = expert_major
        ? (static_cast<size_t>(experts_per_rank) * sizeof(int64_t) +
           static_cast<size_t>(NUM_OF_WARPS_PER_BLOCK_SCAN) * experts_per_rank * sizeof(int32_t))
        : 0;
    const int dynamic_smem_bytes = static_cast<int>(
        scan_smem_size > remap_smem_size ? scan_smem_size : remap_smem_size);

    ::hybrid_ep::scan_kernel_param_t sp;
    sp.input_routing_map = global_routing_map;
    sp.tmp = reinterpret_cast<::hybrid_ep::tmp_state_t*>(scan_tmp);
    sp.sparse_to_dense_map = sparse_to_dense_map;
    sp.rdma_to_attn_map = rdma_to_attn_map;
    sp.attn_to_rdma_map = attn_to_rdma_map;
    sp.token_rank_mask = token_rank_mask;
    sp.num_of_tokens_for_experts = num_tokens_for_experts;
    sp.local_expert_routing_map = local_expert_routing_map;
    sp.per_expert_token_counts = per_expert_token_counts;
    sp.node_rank = node_rank;
    sp.local_rank = local_rank;
    sp.num_of_tokens_per_rank = num_tokens_per_rank;
    sp.num_of_ranks_per_node = num_ranks_per_node;
    sp.experts_per_rank = experts_per_rank;
    sp.expert_major = expert_major;
    sp.remap_alignment = alignment;
    sp.remap_internal_offsets = internal_offsets;
    sp.remap_padded_out_counts = padded_out_counts;
    sp.remap_out_offsets = out_offsets;
    sp.remap_actual_counts_out = actual_counts_out;
    sp.s2d_inner_dim = s2d_inner_dim;
    sp.recv_total_counter = recv_total_counter;
    sp.out_is_int64 = out_is_int64;
    sp.max_recv_tokens_per_rank = max_recv_tokens_per_rank;

    jit::launch_scan(
        NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, num_nodes, num_ranks_per_node,
        per_expert_token_counts != nullptr, sp, dynamic_smem_bytes, stream);

    // EM cooperative scan: produces S2D, LERM, em offsets/counts from the AG'd bitmap.
    if (expert_major) {
        const int num_mask_words = (num_ranks_per_node + 63) / 64;
        const int num_total_attn_tokens = num_tokens_per_rank * num_ranks_per_node * num_nodes;
        launch_em_scan(
            global_routing_map,
            token_rank_mask,
            num_mask_words,
            num_total_attn_tokens,
            num_tokens_per_rank,
            num_ranks_per_node,
            experts_per_rank,
            num_nodes,
            node_rank,
            local_rank,
            s2d_inner_dim,
            max_recv_tokens_per_rank,
            static_cast<int>(alignment),
            sparse_to_dense_map,
            local_expert_routing_map,
            num_tokens_for_experts,
            internal_offsets,
            padded_out_counts,
            out_offsets,
            actual_counts_out,
            recv_total_counter,
            out_is_int64,
            static_cast<int32_t*>(scan_gscratch),
            NUM_OF_BLOCKS,
            stream);
    }
}

size_t get_preprocessing_scan_tmp_size(int num_ranks_per_node) {
    return HYBRIDEP_NUM_BLOCKS_PREPROCESSING * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
}

size_t get_rank_mask_elem_size(int lsa_team_size) {
    return ((lsa_team_size + 63) / 64) * sizeof(uint64_t);
}

namespace cg = cooperative_groups;

static constexpr int kEmScanBlockDim = 256;

// Load 64 bits at a byte offset; memcpy folds to LDG.E.64 (or split if unaligned).
__device__ __forceinline__ uint64_t em_ld64(const uint8_t* row, int byte_off) {
    uint64_t v;
    memcpy(&v, row + byte_off, sizeof(v));
    return v;
}

// Cooperative EM scan from the AG'd bitmap.
//
// gscratch: block_count[num_blocks][n_dle] int32. Each block writes its own
// row, then a single grid.sync separates count from emit. Block-prefix and
// padded offsets are recomputed locally in smem so no second sync is needed.
//
// Slot assignment is deterministic:
//   em_slot = s_offsets[dle]              // padded prefix over le
//           + s_warp_state[warp][dle]     // block_prefix + earlier-warp counts
//           + __popc(ballot & lower_lanes) // earlier lanes in tile
__global__ void em_scan_kernel(
    const uint8_t* __restrict__ input_routing_map,
    const uint64_t* __restrict__ token_rank_mask_words,
    int num_mask_words,
    int num_total_attn_tokens,
    int num_tokens_per_rank,
    int num_ranks_per_node,
    int experts_per_rank,
    int num_lsa_teams,
    int node_rank,
    int local_rank,
    int s2d_inner_dim,
    int max_recv_tokens_per_rank,
    int em_alignment,
    int32_t* __restrict__ sparse_to_dense_map,
    bool*    __restrict__ local_expert_routing_map,
    int32_t* __restrict__ num_tokens_for_experts,
    int64_t* __restrict__ em_internal_offsets,
    int32_t* __restrict__ em_padded_out_counts_i32,
    int64_t* __restrict__ em_padded_out_counts_i64,
    int32_t* __restrict__ em_out_offsets_i32,
    int64_t* __restrict__ em_out_offsets_i64,
    int32_t* __restrict__ em_actual_counts_out,
    int32_t* __restrict__ recv_total_counter_i32,
    int64_t* __restrict__ recv_total_counter_i64,
    bool out_is_int64,
    int32_t* __restrict__ gscratch)
{
    extern __shared__ int32_t s_smem[];

    const int epr   = experts_per_rank;
    const int nrpn  = num_ranks_per_node;
    const int n_dle = nrpn * epr;
    const int packed_row_bytes =
        ((num_lsa_teams * nrpn * epr) + 7) / 8;

    int32_t* g_block_count = gscratch;

    const int local_per_node_bytes = ((nrpn * epr) + 7) / 8;
    const int tid       = threadIdx.x;
    const int lane      = tid & 31;
    const int warp      = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const int B         = blockIdx.x;
    const int N         = gridDim.x;
    const int s_warp_stride = n_dle;
    int32_t* s_warp_state = s_smem;
    int32_t* s_offsets    = s_smem + num_warps * n_dle;

    auto grid = cg::this_grid();

    const int tpb = (num_total_attn_tokens + N - 1) / N;
    const int bs  = B * tpb;
    const int be  = min(bs + tpb, num_total_attn_tokens);
    const int tpw = (be - bs + num_warps - 1) / num_warps;
    const int ws  = bs + warp * tpw;
    const int we  = min(ws + tpw, be);

    // Phase 1: per-warp count into smem; later aggregated to block_count[B][dle].
    for (int i = tid; i < num_warps * n_dle; i += blockDim.x) s_warp_state[i] = 0;
    __syncthreads();

    const int n_local_bits_ph1  = nrpn * epr;
    const int n_local_words_ph1 = (n_local_bits_ph1 + 63) / 64;
    for (int tok = ws + lane; tok < we; tok += 32) {
        const uint64_t* mw = token_rank_mask_words + (size_t)tok * num_mask_words;
        const uint64_t mw0 = mw[0];
        const uint64_t mw1 = (num_mask_words >= 2) ? mw[1] : 0;
        if (mw0 == 0 && mw1 == 0) continue;
        const uint8_t* row = input_routing_map + (size_t)tok * packed_row_bytes
                           + (size_t)node_rank * local_per_node_bytes;
        for (int wi = 0; wi < n_local_words_ph1; wi++) {
            const int word_bit_base = wi * 64;
            const int remaining = n_local_bits_ph1 - word_bit_base;
            const int word_bits = remaining >= 64 ? 64 : remaining;
            uint64_t s = em_ld64(row, wi * 8);
            if (word_bits < 64) s &= (uint64_t{1} << word_bits) - 1;
            while (s) {
                const int b = __ffsll(static_cast<long long>(s)) - 1;
                atomicAdd(&s_warp_state[warp * s_warp_stride + word_bit_base + b], 1);
                s &= s - 1;
            }
        }
    }
    __syncthreads();

    for (int dle = tid; dle < n_dle; dle += blockDim.x) {
        int sum = 0;
        for (int w = 0; w < num_warps; w++) sum += s_warp_state[w * s_warp_stride + dle];
        g_block_count[(size_t)B * n_dle + dle] = sum;
    }

    grid.sync();

    // Phase 2 prep: per dle compute total + this block's prefix; convert
    // s_warp_state from counts to per-warp prior offsets.
    for (int dle = tid; dle < n_dle; dle += blockDim.x) {
        int my_prefix = 0;
        int total = 0;
        for (int b = 0; b < N; b++) {
            const int c = g_block_count[(size_t)b * n_dle + dle];
            if (b < B) my_prefix += c;
            total += c;
        }
        s_offsets[dle] = total;
        int cum = my_prefix;
        for (int w = 0; w < num_warps; w++) {
            const int c = s_warp_state[w * s_warp_stride + dle];
            s_warp_state[w * s_warp_stride + dle] = cum;
            cum += c;
        }
    }
    __syncthreads();

    // Per-dest padded-prefix scan over le (warp-per-dest), in place on s_offsets.
    // The warp handling d=local_rank in block 0 also writes EM-layout outputs.
    {
        const int align = (em_alignment > 1) ? em_alignment : 1;
        for (int d = warp; d < nrpn; d += num_warps) {
            if (lane == 0) {
                const bool write_em = (B == 0 && d == local_rank);
                int cum = 0;
                for (int k = 0; k < epr; k++) {
                    const int c = s_offsets[d * epr + k];
                    const int padded = (align > 1 && c > 0) ? ((c + align - 1) / align) * align : c;
                    s_offsets[d * epr + k] = cum;
                    if (write_em) {
                        if (em_internal_offsets) em_internal_offsets[k] = cum;
                        if (em_actual_counts_out) em_actual_counts_out[k] = c;
                        if (out_is_int64) {
                            if (em_padded_out_counts_i64) em_padded_out_counts_i64[k] = (int64_t)padded;
                            if (em_out_offsets_i64) em_out_offsets_i64[k] = (int64_t)cum;
                        } else {
                            if (em_padded_out_counts_i32) em_padded_out_counts_i32[k] = (int32_t)padded;
                            if (em_out_offsets_i32) em_out_offsets_i32[k] = (int32_t)cum;
                        }
                    }
                    cum += padded;
                }
                if (cum > max_recv_tokens_per_rank) {
                    printf("em_scan_kernel: dest %d padded slots %d > "
                           "max_recv_tokens_per_rank %d\n", d, cum, max_recv_tokens_per_rank);
                    __trap();
                }
                if (write_em) {
                    if (em_internal_offsets) em_internal_offsets[epr] = (int64_t)cum;
                    if (num_tokens_for_experts) *num_tokens_for_experts = (int32_t)cum;
                    if (out_is_int64) {
                        if (recv_total_counter_i64) *recv_total_counter_i64 = (int64_t)cum;
                    } else {
                        if (recv_total_counter_i32) *recv_total_counter_i32 = (int32_t)cum;
                    }
                }
            }
        }
    }
    __syncthreads();

    // Phase 2 emit: warp-tile bit-popcount scan over the local-node slice.
    // Requires epr to be a power of two.
    const int n_local_bits  = nrpn * epr;
    const int n_local_words = (n_local_bits + 63) / 64;
    const int num_tiles     = (we - ws + 31) / 32;
    const int epr_l2        = __ffs(epr) - 1;
    const int epr_mask      = epr - 1;
    for (int tile = 0; tile < num_tiles; tile++) {
        const int tok = ws + tile * 32 + lane;
        const bool valid = (tok < we);
        bool any_hit = false;
        if (valid) {
            const uint64_t* mw = token_rank_mask_words + (size_t)tok * num_mask_words;
            const uint64_t mw0 = mw[0];
            const uint64_t mw1 = (num_mask_words >= 2) ? mw[1] : 0;
            any_hit = (mw0 != 0) || (mw1 != 0);
        }

        const uint8_t* row_local = nullptr;
        int send_idx = 0;
        bool is_our_send = false;
        if (any_hit) {
            row_local = input_routing_map + (size_t)tok * packed_row_bytes
                      + (size_t)node_rank * local_per_node_bytes;
            const int sgr = tok / num_tokens_per_rank;
            const int sn  = sgr / nrpn;
            const int slr = sgr % nrpn;
            const int lti = tok % num_tokens_per_rank;
            send_idx = sn * num_tokens_per_rank + lti;
            is_our_send = (slr == local_rank);
        }

        int my_packed_idx = 0;

        for (int wi = 0; wi < n_local_words; wi++) {
            const int word_bit_base = wi * 64;
            const int remaining = n_local_bits - word_bit_base;
            const int word_bits = remaining >= 64 ? 64 : remaining;
            uint64_t my_slice = 0;
            if (any_hit) {
                my_slice = em_ld64(row_local, wi * 8);
                if (word_bits < 64) my_slice &= (uint64_t{1} << word_bits) - 1;
            }
            const uint32_t any_lo = __reduce_or_sync(0xffffffff, (uint32_t)my_slice);
            const uint32_t any_hi = (word_bits > 32)
                ? __reduce_or_sync(0xffffffff, (uint32_t)(my_slice >> 32)) : 0u;
            uint64_t union_slice = ((uint64_t)any_hi << 32) | (uint64_t)any_lo;

            while (union_slice) {
                const int b = __ffsll(static_cast<long long>(union_slice)) - 1;
                const int dle = word_bit_base + b;
                const int d = dle >> epr_l2;
                const int le = dle & epr_mask;
                const bool my_hit = (my_slice >> b) & 1ull;
                const uint32_t mask = __ballot_sync(0xffffffff, my_hit);
                if (my_hit) {
                    const int within = __popc(mask & ((1u << lane) - 1u));
                    const int em_slot = s_offsets[dle] + s_warp_state[warp * s_warp_stride + dle] + within;
                    if (d == local_rank) local_expert_routing_map[em_slot * epr + le] = true;
                    if (is_our_send) {
                        sparse_to_dense_map[(size_t)send_idx * s2d_inner_dim + my_packed_idx] =
                            ::hybrid_ep::em_s2d_pack(d, em_slot);
                        my_packed_idx++;
                    }
                }
                if (lane == 0) {
                    s_warp_state[warp * s_warp_stride + dle] += __popc(mask);
                }
                union_slice &= union_slice - 1;
            }
        }
    }
}

void launch_em_scan(
    const uint8_t* input_routing_map,
    const void*    token_rank_mask,
    int num_mask_words,
    int num_total_attn_tokens,
    int num_tokens_per_rank,
    int num_ranks_per_node,
    int experts_per_rank,
    int num_lsa_teams,
    int node_rank,
    int local_rank,
    int s2d_inner_dim,
    int max_recv_tokens_per_rank,
    int em_alignment,
    int32_t* sparse_to_dense_map,
    bool*    local_expert_routing_map,
    int32_t* num_tokens_for_experts,
    int64_t* em_internal_offsets,
    void*    em_padded_out_counts,
    void*    em_out_offsets,
    int32_t* em_actual_counts_out,
    void*    recv_total_counter,
    bool     out_is_int64,
    int32_t* gscratch,
    int      num_sms,
    cudaStream_t stream)
{
    if (num_total_attn_tokens <= 0 || num_ranks_per_node <= 0 || experts_per_rank <= 0) return;
    assert((experts_per_rank & (experts_per_rank - 1)) == 0 && "experts_per_rank must be a power of two");
    assert(num_mask_words >= 1 && num_mask_words <= 2 && "lsa_team_size must be <= 128");
    assert(num_sms > 0 && "launch_em_scan requires num_sms > 0");
    const int n_dle = num_ranks_per_node * experts_per_rank;

    constexpr int kNumWarps = kEmScanBlockDim / 32;
    const size_t smem_bytes = static_cast<size_t>(kNumWarps + 1) * n_dle * sizeof(int32_t);

    SETUP_LAUNCH_CONFIG(num_sms, kEmScanBlockDim, stream);
    cfg.dynamicSmemBytes = smem_bytes;
    LAUNCH_KERNEL(&cfg, em_scan_kernel,
        input_routing_map,
        static_cast<const uint64_t*>(token_rank_mask),
        num_mask_words,
        num_total_attn_tokens,
        num_tokens_per_rank,
        num_ranks_per_node,
        experts_per_rank,
        num_lsa_teams,
        node_rank,
        local_rank,
        s2d_inner_dim,
        max_recv_tokens_per_rank,
        em_alignment,
        sparse_to_dense_map,
        local_expert_routing_map,
        num_tokens_for_experts,
        em_internal_offsets,
        out_is_int64 ? nullptr : static_cast<int32_t*>(em_padded_out_counts),
        out_is_int64 ? static_cast<int64_t*>(em_padded_out_counts) : nullptr,
        out_is_int64 ? nullptr : static_cast<int32_t*>(em_out_offsets),
        out_is_int64 ? static_cast<int64_t*>(em_out_offsets) : nullptr,
        em_actual_counts_out,
        out_is_int64 ? nullptr : static_cast<int32_t*>(recv_total_counter),
        out_is_int64 ? static_cast<int64_t*>(recv_total_counter) : nullptr,
        out_is_int64,
        gscratch);
}

size_t get_scan_gscratch_size(int num_ranks_per_node, int experts_per_rank, int num_sms) {
    assert(num_sms > 0);
    return static_cast<size_t>(num_sms) * num_ranks_per_node * experts_per_rank * sizeof(int32_t);
}

int get_device_max_dynamic_smem() {
    int device = 0;
    int max_smem = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    return max_smem;
}

void check_dispatch_smem_limit(
    const ::hybrid_ep::dispatch_config_t& config,
    size_t smem_size) {
    const int max_smem = get_device_max_dynamic_smem();
    if (smem_size <= static_cast<size_t>(max_smem)) return;

    std::fprintf(
        stderr,
        "[nccl_ep] dispatch dynamic shared memory exceeds device limit: requested=%zu bytes, "
        "limit=%d bytes. Tune dispatch stages/pipelines; current stages=%d, pipelines=%d.\n",
        smem_size,
        max_smem,
        config.num_of_stages,
        config.num_pipelines);
    std::abort();
}

// ============================================================================
// Dispatch wrapper implementation
// ============================================================================

// Helper to populate the fixed-size dispatch parameter fields from DispatchParams.
template<typename TOKEN_DATA_TYPE>
::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>
build_dispatch_param_base(const DispatchParams& params) {
    ::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE> kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User input buffers
    kp.attn_input_token = reinterpret_cast<const TOKEN_DATA_TYPE*>(params.attn_input_token);
    kp.attn_input_prob = params.attn_input_prob;
    kp.attn_input_token_scaling_factor = params.attn_input_scaling_factor;

    // Metadata and sync flags
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.pad_actual_counts = params.pad_actual_counts;
    kp.pad_expert_token_offsets = params.pad_expert_token_offsets;
    kp.pad_alignment = params.pad_alignment;
    kp.expected_rdma_flag_value = params.expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.intra_node_write_completion_flags;
    kp.dispatch_grid_barrier_counter = params.dispatch_grid_barrier_counter;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.token_window = params.nccl_token_window;
    kp.prob_window = params.nccl_prob_window;
    kp.sf_window = params.nccl_sf_window;
    kp.dest_window = params.nccl_internal_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .attn_input_token_offset = params.mr_info.attn_input_token_offset,
               .attn_input_prob_offset = params.mr_info.attn_input_prob_offset,
               .attn_input_scaling_factor_offset = params.mr_info.attn_input_scaling_factor_offset,
               // Batched staging parameters (packed layout)
               .rdma_send_staging_offset = params.mr_info.rdma_send_staging_offset,
               .rdma_inter_node_group_packed_offset = params.mr_info.rdma_inter_node_group_packed_offset,
               .bytes_per_entry = params.mr_info.bytes_per_entry,
               .max_tokens_per_dest = params.mr_info.max_tokens_per_dest,
               // Streaming signal parameters
               .signals_tail_base = params.mr_info.signals_tail_base,
               .num_max_rdma_chunked_send_tokens = params.mr_info.num_max_rdma_chunked_send_tokens
            };

    return kp;
}

template<typename TOKEN_DATA_TYPE>
std::vector<uint8_t> build_dispatch_arg_buffer(
    const ::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>& kp,
    const DispatchParams& params) {
    using ParamBase = ::hybrid_ep::dispatch_kernel_param_base_t<TOKEN_DATA_TYPE>;
    static_assert(sizeof(ParamBase) % alignof(void*) == 0);

    const size_t base_size = sizeof(ParamBase);
    const size_t token_offset = base_size;
    const size_t prob_offset = token_offset + params.num_ranks_per_node * sizeof(TOKEN_DATA_TYPE*);
    const size_t sf_offset = prob_offset + params.num_ranks_per_node * sizeof(float*);
    const size_t total_size = sf_offset + params.num_ranks_per_node * sizeof(float*);

    std::vector<uint8_t> arg(total_size);
    std::memcpy(arg.data(), &kp, sizeof(kp));

    auto* token_ptrs = reinterpret_cast<TOKEN_DATA_TYPE**>(arg.data() + token_offset);
    auto* prob_ptrs = reinterpret_cast<float**>(arg.data() + prob_offset);
    auto* sf_ptrs = reinterpret_cast<float**>(arg.data() + sf_offset);
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        token_ptrs[i] = reinterpret_cast<TOKEN_DATA_TYPE*>(params.expert_output_token_ptrs[i]);
        prob_ptrs[i] = params.expert_output_prob_ptrs ? params.expert_output_prob_ptrs[i] : nullptr;
        sf_ptrs[i] = params.expert_output_scaling_factor_ptrs ?
            params.expert_output_scaling_factor_ptrs[i] : nullptr;
    }

    return arg;
}

// Template dispatch launcher for forward/backward and sync modes
template<bool FORWARD_DISPATCH>
void dispatch_impl(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    int num_blocks,
    cudaStream_t stream
) {
    HYBRIDEP_SWITCH_DATATYPE(use_fp8, {
        // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
        // Check alignment at runtime now that experts_per_rank is dynamic
        const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
        assert((experts_per_node * sizeof(float)) % 16 == 0 &&
               "experts_per_node must be multiple of 4 for TMA alignment");
        // 16B cp.async.bulk alignment for the S2D map fetch; matters when s2d_inner_dim < 4.
        assert((static_cast<int64_t>(params.num_tokens_per_rank) * params.s2d_inner_dim) % 4 == 0 &&
               "Dispatch S2D cp.async.bulk: num_tokens_per_rank * s2d_inner_dim must be a "
               "multiple of 4 (flat layout with lsa_team_size <= 3 requires even num_tokens_per_rank)");

        auto kp = build_dispatch_param_base<TOKEN_DATA_TYPE>(params);
        constexpr bool kUseFp8 = std::is_same_v<TOKEN_DATA_TYPE, uint8_t>;

        // Compute dynamic SMEM size at host (was done inside hybrid_ep::dispatch).
        ::hybrid_ep::dispatch_config_t d_config;
        ::hybrid_ep::model_config_t   d_model;
        d_config.num_of_stages           = HYBRIDEP_DISPATCH_NUM_OF_STAGES;
        d_config.num_of_in_flight_s2g    = HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G;
        d_config.num_of_tokens_per_chunk = HT_OF_NUM_TOKENS_PER_CHUNK;
        d_config.num_of_blocks           = num_blocks;
        d_config.forward_dispatch        = FORWARD_DISPATCH;
        d_config.token_data_type         = std::is_same_v<TOKEN_DATA_TYPE, uint16_t> ? 1 : 0;
        d_config.num_pipelines           = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
        d_config.stages_per_pipeline     = HYBRIDEP_DISPATCH_NUM_OF_STAGES / HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
        d_config.s2d_inner_dim           = kp.s2d_inner_dim;
        d_model.hidden_dim               = kp.hidden_dim;
        d_model.max_num_of_tokens_per_rank = MAX_SUPPORTED_TOKENS_PER_RANK;
        d_model.num_of_experts_per_rank  = kp.experts_per_rank;
        d_model.num_of_ranks_per_node    = kp.num_of_ranks_per_node;
        d_model.num_of_nodes             = num_nodes;

        const int smem_size = (params.layout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? ::hybrid_ep::calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_EXPERT_MAJOR>(d_config, d_model)
            : ::hybrid_ep::calculate_dispatch_smem_layout_size<NCCL_EP_LAYOUT_FLAT>(d_config, d_model);
        check_dispatch_smem_limit(d_config, smem_size);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        const jit::dispatch_warp_layout_t dispatch_layout =
            jit::compute_dispatch_warp_layout(num_nodes, params.layout);
        const int dispatch_wt_total = num_blocks * (dispatch_layout.block_dim / 32);
        ::hybrid_ep::dispatch_warp_timing_entry_t* d_wt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_wt, dispatch_wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t)));
        CUDA_CHECK(cudaMemsetAsync(d_wt, 0, dispatch_wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t), stream));
        kp.warp_timing = d_wt;
#endif

        std::vector<uint8_t> kernel_arg = build_dispatch_arg_buffer(kp, params);
        jit::launch_dispatch(
            HYBRIDEP_DISPATCH_NUM_OF_STAGES,
            HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G,
            HT_OF_NUM_TOKENS_PER_CHUNK,
            num_blocks,
            FORWARD_DISPATCH,
            num_nodes,
            params.num_ranks_per_node,
            params.layout,
            kUseFp8,
            kp.hidden_dim,
            kernel_arg.data(),
            kernel_arg.size(),
            smem_size,
            stream);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        jit::dispatch_dump_warp_timing(dispatch_layout, num_blocks, d_wt, stream);
        CUDA_CHECK(cudaFree(d_wt));
#endif
    });
}

void call_dispatch(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    bool forward_dispatch,
    int num_blocks,
    cudaStream_t stream
) {
    // Dispatch based on forward/backward and sync mode
    if (forward_dispatch) {
        dispatch_impl<true>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, use_fp8, num_blocks, stream);

    } else {
        dispatch_impl<false>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, use_fp8, num_blocks, stream);

    }
}

// ============================================================================
// Combine wrapper implementation
// ============================================================================

// Helper to populate the fixed-size combine parameter fields from CombineParams.
::hybrid_ep::combine_kernel_param_base_t
build_combine_param_base(const CombineParams& params) {
    ::hybrid_ep::combine_kernel_param_base_t kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User output buffers
    kp.attn_output_token = reinterpret_cast<uint16_t*>(params.attn_output_token);
    kp.attn_output_prob = params.attn_output_prob;

    // RDMA buffers (multi-node only)
    kp.rdma_intra_node_red_token = params.rdma_intra_node_red_token;
    kp.rdma_intra_node_red_prob = params.rdma_intra_node_red_prob;
    kp.rdma_inter_node_group_token = params.combine_rdma_inter_node_group_token;
    kp.rdma_inter_node_group_prob = params.combine_rdma_inter_node_group_prob;

    // Metadata
    kp.sparse_to_dense_map = params.sparse_to_dense_map;
    kp.s2d_inner_dim = params.s2d_inner_dim;
    kp.rdma_to_attn_map = params.rdma_to_attn_map;
    kp.attn_to_rdma_map = params.attn_to_rdma_map;

    // Sync flags
    kp.expected_rdma_flag_value = params.combine_expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.combine_expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.combine_rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.combine_intra_node_write_completion_flags;
    kp.combine_grid_barrier_counter = params.combine_grid_barrier_counter;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;
    kp.num_real_tokens        = params.num_real_tokens;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.token_window = params.nccl_token_window;
    kp.prob_window = params.nccl_prob_window;
    kp.dest_window = params.nccl_internal_window;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    kp.combine_signal_offset = params.combine_signal_offset;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .rdma_intra_node_red_token_offset = params.mr_info.rdma_intra_node_red_token_offset,
               .combine_rdma_inter_node_group_token_offset = params.mr_info.combine_rdma_inter_node_group_token_offset,
               .rdma_intra_node_red_prob_offset = params.mr_info.rdma_intra_node_red_prob_offset,
               .combine_rdma_inter_node_group_prob_offset = params.mr_info.combine_rdma_inter_node_group_prob_offset
    };

    return kp;
}

std::vector<uint8_t> build_combine_arg_buffer(
    const ::hybrid_ep::combine_kernel_param_base_t& kp,
    const CombineParams& params) {
    using ParamBase = ::hybrid_ep::combine_kernel_param_base_t;
    static_assert(sizeof(ParamBase) % alignof(void*) == 0);

    const size_t base_size = sizeof(ParamBase);
    const size_t token_offset = base_size;
    const size_t prob_offset = token_offset + params.num_ranks_per_node * sizeof(uint16_t*);
    const size_t total_size = prob_offset + params.num_ranks_per_node * sizeof(float*);

    std::vector<uint8_t> arg(total_size);
    std::memcpy(arg.data(), &kp, sizeof(kp));

    auto* token_ptrs = reinterpret_cast<uint16_t**>(arg.data() + token_offset);
    auto* prob_ptrs = reinterpret_cast<float**>(arg.data() + prob_offset);
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        token_ptrs[i] = params.expert_input_token_ptrs[i];
        prob_ptrs[i] = params.expert_input_prob_ptrs ? params.expert_input_prob_ptrs[i] : nullptr;
    }

    return arg;
}


// Template combine launcher for forward/backward
template<bool BACKWARD_COMBINE>
void combine_impl(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    int num_blocks,
    cudaStream_t stream
) {
    // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
    const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
    assert((experts_per_node * sizeof(float)) % 16 == 0 &&
           "experts_per_node must be multiple of 4 for TMA alignment");

    auto kp = build_combine_param_base(params);

    // Select config based on num_nodes (single-node: 12 stages/2 pipelines, multi-node: 5 stages/1 pipeline)
    const int num_stages_g2s = (num_nodes == 1)
        ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S
        : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S;
    const int num_stages_s2g = (num_nodes == 1)
        ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G
        : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G;

    ::hybrid_ep::model_config_t model;
    model.hidden_dim = kp.hidden_dim;
    model.max_num_of_tokens_per_rank = MAX_SUPPORTED_TOKENS_PER_RANK;
    model.num_of_experts_per_rank = kp.experts_per_rank;
    model.num_of_ranks_per_node = kp.num_of_ranks_per_node;
    model.num_of_nodes = num_nodes;
    const int smem_size = ::hybrid_ep::calculate_combine_smem_layout_size(
        num_stages_g2s,
        num_stages_s2g,
        HT_OF_NUM_TOKENS_PER_CHUNK,
        MAX_SUPPORTED_TOKENS_PER_RANK,
        num_nodes,
        BACKWARD_COMBINE,
        model);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    const jit::combine_warp_layout_t combine_layout = jit::compute_combine_warp_layout(num_nodes);
    const int combine_wt_total = num_blocks * (combine_layout.block_dim / 32);
    ::hybrid_ep::combine_warp_timing_entry_t* d_wt = nullptr;
    ::hybrid_ep::combine_block_timing_entry_t* d_bt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wt, combine_wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t)));
    CUDA_CHECK(cudaMalloc(&d_bt, num_blocks * sizeof(::hybrid_ep::combine_block_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, combine_wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_bt, 0, num_blocks * sizeof(::hybrid_ep::combine_block_timing_entry_t), stream));
    kp.warp_timing = d_wt;
    kp.block_timing = d_bt;
#endif

    std::vector<uint8_t> kernel_arg = build_combine_arg_buffer(kp, params);
    jit::launch_combine(
        num_stages_g2s,
        num_stages_s2g,
        HT_OF_NUM_TOKENS_PER_CHUNK,
        HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
        num_blocks,
        HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
        BACKWARD_COMBINE,
        num_nodes,
        params.num_ranks_per_node,
        params.layout,
        kp.hidden_dim,
        kernel_arg.data(),
        kernel_arg.size(),
        smem_size,
        stream);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    jit::combine_dump_warp_timing(combine_layout, num_blocks, d_wt, d_bt, stream);
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_bt));
#endif
}

void call_combine(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank,
    int num_nodes,
    bool backward_combine,
    int num_blocks,
    cudaStream_t stream
) {
    if (backward_combine) {
        combine_impl<true>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, num_blocks, stream);
    } else {
        combine_impl<false>(
            params, max_dispatch_tokens_per_rank,
            num_nodes, num_blocks, stream);
    }
}

} // namespace hybridep
} // namespace nccl_ep
