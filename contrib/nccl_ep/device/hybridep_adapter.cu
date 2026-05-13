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

namespace nccl_ep {
namespace hybridep {

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
__global__ void convert_topk_to_routing_map_kernel(
    const int64_t* __restrict__ topk_idx,    // [num_tokens, num_topk]
    uint8_t* __restrict__ routing_bitmap,     // [num_tokens, num_experts_packed]
    int num_tokens,
    int num_topk,
    int num_experts_packed                    // = ceil(num_experts / 8)
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    // Buffer is pre-zeroed by per-iteration memset; just OR in set bits.
    // Each thread exclusively owns its row -- no atomics needed.
    uint8_t* row = routing_bitmap + token * num_experts_packed;
    for (int k = 0; k < num_topk; k++) {
        int expert = static_cast<int>(topk_idx[token * num_topk + k]);
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
    int num_tokens,
    int num_topk,
    int num_experts_packed,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    convert_topk_to_routing_map_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, routing_bitmap, num_tokens, num_topk, num_experts_packed);
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

// ============================================================================
// Kernel: Convert dense prob output to sparse format for combine output
// ============================================================================
// Used for combine backward pass. Converts kernel's dense output to sparse format
// with GLOBAL expert indices (matching original dispatch input format).
// Each thread handles one token.
__global__ void dense_to_sparse_prob_combine_kernel(
    const float* __restrict__ dense_prob,         // [num_tokens, num_experts]
    const uint8_t* __restrict__ routing_bitmap,   // [num_tokens, ceil(num_experts / 8)]
    float* __restrict__ combined_topk_weights,    // [num_tokens, topk]
    int64_t* __restrict__ combined_topk_idx,      // [num_tokens, topk] (optional, can be nullptr)
    int num_tokens,
    int topk,
    int num_experts
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    int packed_cols = (num_experts + 7) / 8;
    int k_out = 0;

    // Scan all experts in order (matches original dispatch input order)
    for (int e = 0; e < num_experts && k_out < topk; e++) {
        if ((routing_bitmap[token * packed_cols + e / 8] >> (e % 8)) & 1) {
            // This expert is active for this token
            float weight = dense_prob[token * num_experts + e];

            combined_topk_weights[token * topk + k_out] = weight;
            if (combined_topk_idx != nullptr) {
                combined_topk_idx[token * topk + k_out] = static_cast<int64_t>(e);  // GLOBAL expert ID
            }
            k_out++;
        }
    }

    // Zero-fill remaining topk slots
    for (; k_out < topk; k_out++) {
        combined_topk_weights[token * topk + k_out] = 0.0f;
        if (combined_topk_idx != nullptr) {
            combined_topk_idx[token * topk + k_out] = -1;
        }
    }
}

// ============================================================================
// Convert dense prob output to sparse format for combine output
// ============================================================================
void dense_to_sparse_prob_combine(
    const float* dense_prob,
    const uint8_t* routing_bitmap,
    float* combined_topk_weights,
    int64_t* combined_topk_idx,
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_combine_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, routing_bitmap, combined_topk_weights, combined_topk_idx,
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
    int hidden_dim,
    int num_nodes,
    int num_ranks_per_node,
    int experts_per_rank,
    int64_t* internal_offsets,
    void*    padded_out_counts,
    void*    out_offsets,
    size_t alignment,
    int32_t* actual_counts_out,
    int s2d_inner_dim,
    void*    recv_total_counter,
    bool     out_is_int64,
    int      max_recv_token_slots_per_rank,
    cudaStream_t stream
) {
    if (alignment > 0 && per_expert_token_counts == nullptr) {
        EP_HOST_ASSERT(false && "EXPERT_MAJOR remap requires per_expert_token_counts != nullptr");
    }

    if (per_expert_token_counts != nullptr) {
        CUDA_CHECK(cudaMemsetAsync(per_expert_token_counts, 0, experts_per_rank * sizeof(int32_t), stream));
    }

    // MNNVL configurations (> 32 GPUs per LSA domain) are not yet supported: the scan
    // kernel uses warp-reduction (LSA_TEAM_SIZE <= 32). Extend when adding MNNVL support.
    EP_HOST_ASSERT(num_ranks_per_node <= 32 && "metadata_preprocessing: LSA team size > 32 not yet supported (MNNVL)");

    constexpr int NUM_THREADS_PER_BLOCK = HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING;
    constexpr int NUM_OF_BLOCKS = HYBRIDEP_NUM_BLOCKS_PREPROCESSING;
    constexpr int NUM_OF_WARPS_PER_BLOCK_SCAN = NUM_THREADS_PER_BLOCK / 32;

    const size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(scan_tmp, 0, preprocessing_tmp_sz, stream));

    const size_t scan_smem_size =
        (NUM_OF_WARPS_PER_BLOCK_SCAN * num_ranks_per_node * sizeof(int32_t)) +
        (num_ranks_per_node * sizeof(int32_t)) +
        (per_expert_token_counts != nullptr ? experts_per_rank * sizeof(int32_t) : 0);
    const size_t remap_smem_size = (alignment > 0)
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
    sp.remap_alignment = alignment;
    sp.remap_internal_offsets = internal_offsets;
    sp.remap_padded_out_counts = padded_out_counts;
    sp.remap_out_offsets = out_offsets;
    sp.remap_actual_counts_out = actual_counts_out;
    sp.s2d_inner_dim = s2d_inner_dim;
    sp.recv_total_counter = recv_total_counter;
    sp.out_is_int64 = out_is_int64;
    sp.max_recv_token_slots_per_rank = max_recv_token_slots_per_rank;

    jit::launch_scan(
        NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, num_nodes, num_ranks_per_node,
        per_expert_token_counts != nullptr, sp, dynamic_smem_bytes, stream);
}

size_t get_preprocessing_scan_tmp_size(int num_ranks_per_node) {
    return HYBRIDEP_NUM_BLOCKS_PREPROCESSING * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
}

size_t get_rank_mask_elem_size(int lsa_team_size) {
    // RankMask<N> picks the smallest unsigned int type that holds N bits.
    if (lsa_team_size <= 8)  return sizeof(uint8_t);
    if (lsa_team_size <= 16) return sizeof(uint16_t);
    if (lsa_team_size <= 32) return sizeof(uint32_t);
    if (lsa_team_size <= 64) return sizeof(uint64_t);
    assert(false && "lsa_team_size > 64 is not supported");
    return 0;
}

// ============================================================================
// Dispatch wrapper implementation
// ============================================================================

// Helper to populate dispatch_kernel_param_t from DispatchParams
template<typename TOKEN_DATA_TYPE>
::hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE>
build_dispatch_param(const DispatchParams& params) {
    ::hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE> kp{};
    // Model configuration
    kp.hidden_dim = params.hidden_dim;
    kp.experts_per_rank = params.experts_per_rank;
    kp.num_of_ranks_per_node = params.num_ranks_per_node;
    // User input buffers
    kp.attn_input_token = reinterpret_cast<const TOKEN_DATA_TYPE*>(params.attn_input_token);
    kp.attn_input_prob = params.attn_input_prob;
    kp.attn_input_token_scaling_factor = params.attn_input_scaling_factor;

    // Copy IPC buffer pointers from HOST arrays into embedded param struct arrays.
    // This allows fast __grid_constant__ access in the kernel (vs slow global memory indirection).
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        kp.expert_output_token[i] =
            reinterpret_cast<TOKEN_DATA_TYPE*>(params.expert_output_token_ptrs[i]);
        kp.expert_output_prob[i] = params.expert_output_prob_ptrs ?
            params.expert_output_prob_ptrs[i] : nullptr;
        kp.expert_output_scaling_factor[i] = params.expert_output_scaling_factor_ptrs ?
            params.expert_output_scaling_factor_ptrs[i] : nullptr;
    }

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

// Template dispatch launcher for forward/backward and sync modes
template<bool FORWARD_DISPATCH>
void dispatch_impl(
    const DispatchParams& params,
    int max_send_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    cudaStream_t stream
) {
    HYBRIDEP_SWITCH_DATATYPE(use_fp8, {
        // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
        // Check alignment at runtime now that experts_per_rank is dynamic
        const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
        assert((experts_per_node * sizeof(float)) % 16 == 0 &&
               "experts_per_node must be multiple of 4 for TMA alignment");
        assert(params.num_ranks_per_node <= ::hybrid_ep::HYBRIDEP_MAX_LSA_TEAM_SIZE &&
               "num_ranks_per_node exceeds HYBRIDEP_MAX_LSA_TEAM_SIZE");

        auto kp = build_dispatch_param<TOKEN_DATA_TYPE>(params);
        constexpr bool kUseFp8 = std::is_same_v<TOKEN_DATA_TYPE, uint8_t>;

        // Compute dynamic SMEM size at host (was done inside hybrid_ep::dispatch).
        ::hybrid_ep::dispatch_config_t d_config;
        ::hybrid_ep::model_config_t   d_model;
        d_config.num_of_stages           = HYBRIDEP_DISPATCH_NUM_OF_STAGES;
        d_config.num_of_in_flight_s2g    = HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G;
        d_config.num_of_tokens_per_chunk = HT_OF_NUM_TOKENS_PER_CHUNK;
        d_config.num_of_blocks           = HYBRIDEP_DISPATCH_NUM_OF_BLOCKS;
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

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        const jit::dispatch_warp_layout_t dispatch_layout =
            jit::compute_dispatch_warp_layout(num_nodes, params.layout);
        const int dispatch_wt_total = HYBRIDEP_DISPATCH_NUM_OF_BLOCKS * (dispatch_layout.block_dim / 32);
        ::hybrid_ep::dispatch_warp_timing_entry_t* d_wt = nullptr;
        CUDA_CHECK(cudaMalloc(&d_wt, dispatch_wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t)));
        CUDA_CHECK(cudaMemsetAsync(d_wt, 0, dispatch_wt_total * sizeof(::hybrid_ep::dispatch_warp_timing_entry_t), stream));
        kp.warp_timing = d_wt;
#endif

        jit::launch_dispatch(
            HYBRIDEP_DISPATCH_NUM_OF_STAGES,
            HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G,
            HT_OF_NUM_TOKENS_PER_CHUNK,
            HYBRIDEP_DISPATCH_NUM_OF_BLOCKS,
            FORWARD_DISPATCH,
            num_nodes,
            params.num_ranks_per_node,
            params.layout,
            kUseFp8,
            kp.hidden_dim,
            &kp,
            smem_size,
            stream);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        jit::dispatch_dump_warp_timing(dispatch_layout, HYBRIDEP_DISPATCH_NUM_OF_BLOCKS, d_wt, stream);
        CUDA_CHECK(cudaFree(d_wt));
#endif
    });
}

void call_dispatch(
    const DispatchParams& params,
    int max_send_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    bool forward_dispatch,
    cudaStream_t stream
) {
    // Dispatch based on forward/backward and sync mode
    if (forward_dispatch) {
        dispatch_impl<true>(
            params, max_send_tokens_per_rank,
            num_nodes, use_fp8, stream);

    } else {
        dispatch_impl<false>(
            params, max_send_tokens_per_rank,
            num_nodes, use_fp8, stream);

    }
}

// ============================================================================
// Combine wrapper implementation
// ============================================================================

// Helper to populate combine_kernel_param_t from CombineParams
::hybrid_ep::combine_kernel_param_t
build_combine_param(const CombineParams& params) {
    ::hybrid_ep::combine_kernel_param_t kp{};

    // Copy IPC buffer pointers from HOST arrays into embedded param struct arrays.
    // This allows fast __grid_constant__ access in the kernel (vs slow global memory indirection).
    for (int i = 0; i < params.num_ranks_per_node; i++) {
        kp.expert_input_token[i] = params.expert_input_token_ptrs[i];
        kp.expert_input_prob[i] = params.expert_input_prob_ptrs ?
            params.expert_input_prob_ptrs[i] : nullptr;
    }

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

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;

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


// Template combine launcher for forward/backward
template<bool BACKWARD_COMBINE>
void combine_impl(
    const CombineParams& params,
    int max_send_tokens_per_rank,
    int num_nodes,
    cudaStream_t stream
) {
    // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
    const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
    assert((experts_per_node * sizeof(float)) % 16 == 0 &&
           "experts_per_node must be multiple of 4 for TMA alignment");
    assert(params.num_ranks_per_node <= ::hybrid_ep::HYBRIDEP_MAX_LSA_TEAM_SIZE &&
           "num_ranks_per_node exceeds HYBRIDEP_MAX_LSA_TEAM_SIZE");

    auto kp = build_combine_param(params);

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
    const int combine_wt_total = HYBRIDEP_COMBINE_NUM_OF_BLOCKS * (combine_layout.block_dim / 32);
    ::hybrid_ep::combine_warp_timing_entry_t* d_wt = nullptr;
    ::hybrid_ep::combine_block_timing_entry_t* d_bt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wt, combine_wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t)));
    CUDA_CHECK(cudaMalloc(&d_bt, HYBRIDEP_COMBINE_NUM_OF_BLOCKS * sizeof(::hybrid_ep::combine_block_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, combine_wt_total * sizeof(::hybrid_ep::combine_warp_timing_entry_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_bt, 0, HYBRIDEP_COMBINE_NUM_OF_BLOCKS * sizeof(::hybrid_ep::combine_block_timing_entry_t), stream));
    kp.warp_timing = d_wt;
    kp.block_timing = d_bt;
#endif

    jit::launch_combine(
        num_stages_g2s,
        num_stages_s2g,
        HT_OF_NUM_TOKENS_PER_CHUNK,
        HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
        HYBRIDEP_COMBINE_NUM_OF_BLOCKS,
        HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
        BACKWARD_COMBINE,
        num_nodes,
        params.num_ranks_per_node,
        params.layout,
        kp.hidden_dim,
        &kp,
        smem_size,
        stream);

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    jit::combine_dump_warp_timing(combine_layout, HYBRIDEP_COMBINE_NUM_OF_BLOCKS, d_wt, d_bt, stream);
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_bt));
#endif
}

void call_combine(
    const CombineParams& params,
    int max_send_tokens_per_rank,
    int num_nodes,
    bool backward_combine,
    cudaStream_t stream
) {
    if (backward_combine) {
        combine_impl<true>(
            params, max_send_tokens_per_rank,
            num_nodes, stream);
    } else {
        combine_impl<false>(
            params, max_send_tokens_per_rank,
            num_nodes, stream);
    }
}

} // namespace hybridep
} // namespace nccl_ep
