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

namespace nccl_ep {
namespace hybridep {

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
__global__ void convert_topk_to_routing_map_kernel(
    const int64_t* __restrict__ topk_idx,  // [num_tokens, num_topk]
    bool* __restrict__ routing_map,         // [num_tokens, num_experts]
    int num_tokens,
    int num_topk,
    int num_experts // column count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int token = tid / num_experts;
    int expert = tid % num_experts;
    if (token >= num_tokens) return;

    bool selected = false;
    for (int k = 0; k < num_topk; k++) {
        if (topk_idx[token * num_topk + k] == expert) {
            selected = true;
            break;
        }
    }
    routing_map[token * num_experts + expert] = selected;
}

// ============================================================================
// Convert topk to dense routing map
// ============================================================================
void convert_topk_to_routing_map(
    const int64_t* topk_idx,
    bool* routing_map,
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream
) {
    int total_elements = num_tokens * num_experts;
    int block_size = 256;
    int grid_size = (total_elements + block_size - 1) / block_size;

    convert_topk_to_routing_map_kernel<<<grid_size, block_size, 0, stream>>>(
        topk_idx, routing_map, num_tokens, num_topk, num_experts);
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
// Each thread handles one token, scans for non-zero experts
__global__ void dense_to_sparse_prob_kernel(
    const float* __restrict__ dense_prob,              // [num_recv_tokens, experts_per_node]
    const bool* __restrict__ local_expert_routing_map, // [num_recv_tokens, experts_per_rank]
    float* __restrict__ recv_topk_weights,             // [num_recv_tokens, topk]
    int64_t* __restrict__ recv_topk_idx,               // [num_recv_tokens, topk]
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,
    int local_rank
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_recv_tokens) return;

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

            // Write both outputs
            recv_topk_idx[token * topk + k_out] = local_expert;
            recv_topk_weights[token * topk + k_out] = weight;
            k_out++;
        }
    }

    // Zero-fill remaining topk slots if fewer than topk experts found
    for (; k_out < topk; k_out++) {
        recv_topk_idx[token * topk + k_out] = -1;  // Invalid expert marker
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
    const bool* __restrict__ local_routing_map,   // [num_tokens, num_experts]
    float* __restrict__ combined_topk_weights,    // [num_tokens, topk]
    int64_t* __restrict__ combined_topk_idx,      // [num_tokens, topk] (optional, can be nullptr)
    int num_tokens,
    int topk,
    int num_experts
) {
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    int k_out = 0;

    // Scan all experts in order (matches original dispatch input order)
    for (int e = 0; e < num_experts && k_out < topk; e++) {
        if (local_routing_map[token * num_experts + e]) {
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
    const bool* local_routing_map,
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
        dense_prob, local_routing_map, combined_topk_weights, combined_topk_idx,
        num_tokens, topk, num_experts);
}

// ============================================================================
// Kernel: Compute per-expert token counts from local_expert_routing_map
// ============================================================================
// Uses shared memory reduction to minimize global memory atomics
__global__ void compute_per_expert_counts_kernel(
    const bool* local_expert_routing_map,   // [max_tokens, num_experts]
    int32_t* per_expert_counts,            // [num_experts]
    const int32_t* total_tokens,           // [1]
    int num_experts
) {
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id >= num_experts) return;

    int total = *total_tokens;
    int count = 0;

    // Count tokens routed to this expert
    for (int token_idx = 0; token_idx < total; token_idx++) {
        if (local_expert_routing_map[token_idx * num_experts + expert_id]) {
            count++;
        }
    }

    per_expert_counts[expert_id] = count;
}

// ============================================================================
// Compute per-expert token counts from local_expert_routing_map
// ============================================================================
void compute_per_expert_counts(
    const bool* local_expert_routing_map,
    int32_t* per_expert_counts,
    const int32_t* total_tokens,
    int num_experts,
    cudaStream_t stream
) {
    // Initialize counts to 0
    CUDA_CHECK(cudaMemsetAsync(per_expert_counts, 0, num_experts * sizeof(int32_t), stream));

    // Launch kernel to count tokens per expert
    int block_size = 256;
    int grid_size = (num_experts + block_size - 1) / block_size;

    compute_per_expert_counts_kernel<<<grid_size, block_size, 0, stream>>>(
        local_expert_routing_map, per_expert_counts, total_tokens, num_experts);
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
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (num_recv_tokens + block_size - 1) / block_size;

    dense_to_sparse_prob_kernel<<<grid_size, block_size, 0, stream>>>(
        dense_prob, local_expert_routing_map, recv_topk_weights, recv_topk_idx,
        num_recv_tokens, topk, experts_per_rank, experts_per_node, local_rank);
}

// ============================================================================
// Call metadata preprocessing
// ============================================================================
void call_metadata_preprocessing(
    const bool* global_routing_map,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    int32_t* num_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    int node_rank,
    int local_rank,
    int num_tokens_per_rank,
    int hidden_dim,
    int num_nodes,
    int num_ranks_per_node,
    int experts_per_rank,
    cudaStream_t stream
) {
    void* scan_tmp = nullptr;
    size_t scan_tmp_size = HYBRIDEP_NUM_BLOCKS_PREPROCESSING * num_ranks_per_node * sizeof(::hybrid_ep::tmp_state_t);
    CUDA_CHECK(cudaMalloc(&scan_tmp, scan_tmp_size));

    HYBRIDEP_SWITCH_NUM_NODES(num_nodes, {
            using HybridEPType = ::hybrid_ep::hybrid_ep<MAX_SUPPORTED_TOKENS_PER_RANK, NUM_NODES>;
            HybridEPType::template metadata_preprocessing<
                HYBRIDEP_NUM_THREADS_PER_BLOCK_PREPROCESSING, HYBRIDEP_NUM_BLOCKS_PREPROCESSING>(
                global_routing_map,
                reinterpret_cast<::hybrid_ep::tmp_state_t*>(scan_tmp),
                sparse_to_dense_map,
                rdma_to_attn_map,
                attn_to_rdma_map,
                num_tokens_for_experts,
                local_expert_routing_map,
                node_rank,
                local_rank,
                num_tokens_per_rank,
                num_ranks_per_node,
                experts_per_rank,
                stream
            );
        });

    CUDA_CHECK(cudaFree(scan_tmp));

    // Compute per-expert token counts for NCCL API compatibility if requested
    // This is done after metadata_preprocessing because we need the populated local_expert_routing_map
    if (per_expert_token_counts != nullptr) {
        compute_per_expert_counts(
            local_expert_routing_map,
            per_expert_token_counts,
            num_tokens_for_experts,
            experts_per_rank,
            stream);
    }
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
    kp.expected_rdma_flag_value = params.expected_rdma_flag_value;
    kp.expected_intra_node_flag_value = params.expected_intra_node_flag_value;
    kp.rdma_inter_node_group_flags = params.rdma_inter_node_group_flags;
    kp.intra_node_write_completion_flags = params.intra_node_write_completion_flags;

    // Runtime config
    kp.local_rank = params.local_rank;
    kp.node_rank = params.node_rank;
    kp.num_of_tokens_per_rank = params.num_tokens_per_rank;

    // Pass device communicators and windows
    kp.dcomms = params.dcomms;
    kp.nccl_windows = params.nccl_windows;
    kp.num_gin_comms = params.num_gin_comms;
    kp.num_ctx_per_comm = params.num_ctx_per_comm;
    kp.gin_base_ptr = params.gin_base_ptr;
    kp.signals_base = params.signals_base;
    // Use offsets relative to gin_base_ptr
    kp.mr_info = {
               .attn_input_token_offset = params.mr_info.attn_input_token_offset,
               .rdma_inter_node_group_token_offset = params.mr_info.rdma_inter_node_group_token_offset,
               .attn_input_prob_offset = params.mr_info.attn_input_prob_offset,
               .rdma_inter_node_group_prob_offset = params.mr_info.rdma_inter_node_group_prob_offset,
               .attn_input_scaling_factor_offset = params.mr_info.attn_input_scaling_factor_offset,
               .rdma_inter_node_group_scaling_factor_offset = params.mr_info.rdma_inter_node_group_scaling_factor_offset,
               // Batched staging parameters
               .rdma_send_staging_offset = params.mr_info.rdma_send_staging_offset,
               .rdma_inter_node_group_packed_offset = params.mr_info.rdma_inter_node_group_packed_offset,
               .rdma_batch_size = params.mr_info.rdma_batch_size,
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
    int max_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    cudaStream_t stream
) {
    HYBRIDEP_SWITCH_DATATYPE(use_fp8, {
            HYBRIDEP_SWITCH_NUM_NODES(num_nodes, {
                // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
                // Check alignment at runtime now that experts_per_rank is dynamic
                const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
                assert((experts_per_node * sizeof(float)) % 16 == 0 &&
                       "experts_per_node must be multiple of 4 for TMA alignment");

                using HybridEPType = ::hybrid_ep::hybrid_ep<
                    MAX_SUPPORTED_TOKENS_PER_RANK,  // MAX_NUM_OF_TOKENS_PER_RANK - use fixed max for template
                    NUM_NODES>;

                auto kp = build_dispatch_param<TOKEN_DATA_TYPE>(params);

                HybridEPType::template dispatch<
                    TOKEN_DATA_TYPE,
                    HYBRIDEP_DISPATCH_NUM_OF_STAGES,
                    HYBRIDEP_DISPATCH_NUM_OF_IN_FLIGHT_S2G,
                    HT_OF_NUM_TOKENS_PER_CHUNK,
                    HYBRIDEP_DISPATCH_NUM_OF_BLOCKS,
                    FORWARD_DISPATCH>(kp, stream);
            });
    });
}

void call_dispatch(
    const DispatchParams& params,
    int max_tokens_per_rank,
    int num_nodes,
    bool use_fp8,
    bool forward_dispatch,
    cudaStream_t stream
) {
    // Dispatch based on forward/backward and sync mode
    if (forward_dispatch) {
        dispatch_impl<true>(
            params, max_tokens_per_rank,
            num_nodes, use_fp8, stream);

    } else {
        dispatch_impl<false>(
            params, max_tokens_per_rank,
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
    kp.nccl_windows = params.nccl_windows;
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
    int max_tokens_per_rank,
    int num_nodes,
    cudaStream_t stream
) {
    // HT combine doesn't support FP8, only BF16
    using TOKEN_DATA_TYPE = uint16_t;

        HYBRIDEP_SWITCH_NUM_NODES(num_nodes, {
            // TMA requires prob buffer (experts_per_node * sizeof(float)) to be 16B aligned
            const int experts_per_node = params.experts_per_rank * params.num_ranks_per_node;
            assert((experts_per_node * sizeof(float)) % 16 == 0 &&
                   "experts_per_node must be multiple of 4 for TMA alignment");

            using HybridEPType = ::hybrid_ep::hybrid_ep<
                MAX_SUPPORTED_TOKENS_PER_RANK,  // MAX_NUM_OF_TOKENS_PER_RANK - use fixed max for template
                NUM_NODES>;

            auto kp = build_combine_param(params);

            // Select config based on NUM_NODES (single-node: 12 stages/2 pipelines, multi-node: 5 stages/1 pipeline)
            constexpr int num_stages_g2s = (NUM_NODES == 1)
                ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_G2S
                : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_G2S;
            constexpr int num_stages_s2g = (NUM_NODES == 1)
                ? HYBRIDEP_COMBINE_SINGLENODE_NUM_OF_STAGES_S2G
                : HYBRIDEP_COMBINE_MULTINODE_NUM_OF_STAGES_S2G;

            HybridEPType::template combine<
                num_stages_g2s,
                num_stages_s2g,
                HT_OF_NUM_TOKENS_PER_CHUNK,
                HYBRIDEP_COMBINE_NUM_OF_TOKENS_PER_GROUP,
                HYBRIDEP_COMBINE_NUM_OF_BLOCKS,
                HYBRIDEP_COMBINE_NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                BACKWARD_COMBINE>(kp, stream);
        });
}

void call_combine(
    const CombineParams& params,
    int max_tokens_per_rank,
    int num_nodes,
    bool backward_combine,
    cudaStream_t stream
) {
    if (backward_combine) {
        combine_impl<true>(
            params, max_tokens_per_rank,
            num_nodes, stream);
    } else {
        combine_impl<false>(
            params, max_tokens_per_rank,
            num_nodes, stream);
    }
}

} // namespace hybridep
} // namespace nccl_ep
