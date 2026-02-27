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

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <nccl.h>
#include "nccl_device.h"

namespace nccl_ep {
namespace hybridep {

// ============================================================================
// Maximum constants from configs.cuh (for array sizing/validation only):
//   NUM_MAX_NVL_PEERS = 8       (max ranks per node)
//   NUM_MAX_RDMA_PEERS = 20     (max number of nodes)
//   NUM_MAX_LOCAL_EXPERTS = 1024 (max experts per rank)
//
// Actual runtime values come from ep_group:
//   num_nvl_ranks, rdma_ranks, num_local_experts, hidden
// Use HYBRIDEP_SWITCH_* macros to instantiate templates with runtime values.
//
// ============================================================================

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
// NCCL API uses sparse format: topk_idx[token][k] = expert_id
// HT uses dense format: routing_map[token][expert] = true/false
void convert_topk_to_routing_map(
    const int64_t* topk_idx,
    bool* routing_map,
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob (for dispatch input)
// ============================================================================
// NCCL API uses sparse format: topk_weights[token][k] = weight, topk_idx[token][k] = expert_id
// HT uses dense format: prob[token][expert] = weight (0 for non-selected)
void sparse_to_dense_prob(
    const int64_t* topk_idx,      // [num_tokens, topk] - which experts
    const float* topk_weights,    // [num_tokens, topk] - weights for those experts
    float* dense_prob,            // [num_tokens, num_experts] - output dense prob (pre-zeroed)
    int num_tokens,
    int num_topk,
    int num_experts,
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob (for combine input)
// ============================================================================
// Used for combine backward pass. Unlike dispatch which has explicit topk_idx,
// combine uses local_expert_routing_map to determine expert positions.
// The order matches dispatch output (dense_to_sparse_prob): experts scanned in order 0,1,2...
// NOTE: No topk_idx needed - expert mapping comes from local_expert_routing_map
void sparse_to_dense_prob_combine(
    const float* topk_weights,                // [num_tokens, topk] - sparse weights from combine input
    const bool* local_expert_routing_map,     // [num_tokens, experts_per_rank] - from handle
    float* dense_prob,                        // [num_tokens, experts_per_node] - output (pre-zeroed)
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node,                     // = experts_per_rank * ranks_per_node
    int local_rank,                           // rank within node
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert dense prob output to sparse format (for dispatch output)
// ============================================================================
// HT outputs: dense_prob[recv_token][experts_per_node]
// HT expects: recv_topk_weights[recv_token][topk], recv_topk_idx[recv_token][topk]
// Uses local_expert_routing_map to know which experts each token is routed to.
// NOTE: recv_topk_idx contains LOCAL expert indices (0-based within this rank)
//       to match NCCL API expectations.
void dense_to_sparse_prob(
    const float* dense_prob,              // [num_recv_tokens, experts_per_node] - from IPC buffer
    const bool* local_expert_routing_map, // [num_recv_tokens, experts_per_rank] - from preprocessing
    float* recv_topk_weights,             // [num_recv_tokens, topk] - output sparse weights
    int64_t* recv_topk_idx,               // [num_recv_tokens, topk] - output LOCAL expert indices (0-based)
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node,                 // = experts_per_rank * ranks_per_node
    int local_rank,                       // rank within node
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert dense prob output to sparse format (for combine output)
// ============================================================================
// Used for combine backward pass output. Converts kernel's dense output back to
// sparse format with GLOBAL expert indices (matching original dispatch input format).
// Uses local_routing_map (original tokens → global experts) for the conversion.
void dense_to_sparse_prob_combine(
    const float* dense_prob,              // [num_tokens, num_experts] - from kernel output
    const bool* local_routing_map,        // [num_tokens, num_experts] - original routing map
    float* combined_topk_weights,         // [num_tokens, topk] - output sparse weights
    int64_t* combined_topk_idx,           // [num_tokens, topk] - output GLOBAL expert indices (optional, can be nullptr)
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream);

// ============================================================================
// Switch macros for compile-time template instantiation
// Uses constants from configs.cuh where applicable
// ============================================================================



// Switch on number of nodes (up to NUM_MAX_RDMA_PEERS from configs.cuh)
#define HYBRIDEP_SWITCH_NUM_NODES(num_nodes_val, ...) \
    do { \
        switch (num_nodes_val) { \
            case 1:  { constexpr int NUM_NODES = 1;  __VA_ARGS__; } break; \
            case 2:  { constexpr int NUM_NODES = 2;  __VA_ARGS__; } break; \
            case 4:  { constexpr int NUM_NODES = 4;  __VA_ARGS__; } break; \
            case 8:  { constexpr int NUM_NODES = 8;  __VA_ARGS__; } break; \
            default: \
                assert(false && "Unsupported node count for HT (max=" \
                       "NUM_MAX_RDMA_PEERS)"); \
        } \
    } while(0)

// Switch on token data type (BF16 = uint16_t, FP8 = uint8_t)
// use_fp8: false = BF16 (uint16_t), true = FP8 (uint8_t)
#define HYBRIDEP_SWITCH_DATATYPE(use_fp8, ...) \
    do { \
        if (use_fp8) { \
            using TOKEN_DATA_TYPE = uint8_t; \
            __VA_ARGS__; \
        } else { \
            using TOKEN_DATA_TYPE = uint16_t; \
            __VA_ARGS__; \
        } \
    } while(0)

// ============================================================================
// Preprocessing wrapper with template parameter resolution
// ============================================================================

// Helper to call metadata_preprocessing with resolved template parameters
// Note: MAX_NUM_OF_TOKENS_PER_RANK is not used by metadata_preprocessing,
// so we use a fixed dummy value for class instantiation.
// Note: Temp buffer for scan is allocated/freed internally - caller doesn't need to manage it.
// Also computes per-expert token counts for NCCL API compatibility when requested.
void call_metadata_preprocessing(
    const bool* global_routing_map,     // Already allgathered routing map
    int32_t* sparse_to_dense_map,       // Output: token→rank→position mapping
    bool* rdma_to_attn_map,             // Output: which tokens come from RDMA
    bool* attn_to_rdma_map,             // Output: which tokens go to RDMA
    int32_t* num_tokens_for_experts,    // Output: total tokens for local experts
    bool* local_expert_routing_map,     // Output: per-expert routing for local tokens
    int32_t* per_expert_token_counts,   // Optional output: per-expert counts (nullptr to skip)
    int node_rank,                      // This node's rank (0 to num_nodes-1)
    int local_rank,                     // Rank within node (0 to num_ranks_per_node-1)
    int num_tokens_per_rank,            // Actual tokens per rank this iteration (runtime)
    int hidden_dim,                     // Model hidden dimension
    int num_nodes,                      // Number of nodes (RDMA domain size)
    int num_ranks_per_node,             // Ranks per node (NVLink domain size, 1-8)
    int experts_per_rank,               // Experts per GPU
    cudaStream_t stream);

// ============================================================================
// Memory region info structs for GIN
// All buffers are part of a single large gin_base_ptr buffer
// Offsets are relative to gin_base_ptr (stored as size_t for offset calculation)
// ============================================================================

struct dispatch_memory_region_info_t {
    // Offsets relative to gin_base_ptr for RDMA operations
    size_t attn_input_token_offset;            // Offset of token staging buffer from gin_base_ptr
    size_t rdma_inter_node_group_token_offset; // Offset of rdma token buffer from gin_base_ptr
    size_t attn_input_prob_offset;             // Offset of prob staging buffer from gin_base_ptr
    size_t rdma_inter_node_group_prob_offset;  // Offset of rdma prob buffer from gin_base_ptr
    size_t attn_input_scaling_factor_offset;   // Offset of scaling factor staging buffer
    size_t rdma_inter_node_group_scaling_factor_offset; // Offset of rdma scaling factor buffer
    // Batched RDMA staging
    size_t rdma_send_staging_offset;           // Offset of per-destination staging buffer
    size_t rdma_inter_node_group_packed_offset;// Offset of packed receive buffer (token+prob+sf per entry)
    int rdma_batch_size;                       // Tokens per batch (default: 6)
    size_t bytes_per_entry;                    // Size of packed entry (token + prob + sf)
    size_t max_tokens_per_dest;                // Max tokens that can be staged per destination
    // Streaming RDMA signals
    unsigned signals_tail_base;               // Base signal ID for tail tracking (sender -> receiver)
    // Streaming buffer configuration
    int num_max_rdma_chunked_send_tokens;     // Batch size per RDMA put (default: 6)
};

struct combine_memory_region_info_t {
    // Offsets relative to gin_base_ptr for RDMA operations
    size_t rdma_intra_node_red_token_offset;        // Offset of intra-node reduced token buffer
    size_t combine_rdma_inter_node_group_token_offset; // Offset of combine rdma token buffer
    size_t rdma_intra_node_red_prob_offset;         // Offset of intra-node reduced prob buffer
    size_t combine_rdma_inter_node_group_prob_offset;  // Offset of combine rdma prob buffer
};

// ============================================================================
// Dispatch wrapper with template parameter resolution
// ============================================================================

// All parameters needed for dispatch kernel, needed redifine because those in hybrid_ep.cuh are with template on data type.
struct DispatchParams {
    // User inputs
    int hidden_dim;             // Model hidden dimension
    int experts_per_rank;       // Experts per GPU
    int num_ranks_per_node;     // Ranks per node (NVLink domain size, 1-8)
    const void* attn_input_token;
    const float* attn_input_prob;            // Forward dispatch only
    const float* attn_input_scaling_factor;  // FP8 only

    // IPC-mapped output buffers (from ep_group)
    // Pointer tables are expected to be device-resident (d_* arrays).
    void* const* expert_output_token_ptrs;   // Array[num_ranks_per_node]
    float* const* expert_output_prob_ptrs;   // Forward only
    float* const* expert_output_scaling_factor_ptrs; // FP8 only

    // Metadata (from handle->hybridep preprocessing outputs)
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;

    // Sync state (from ep_group->doca_config, group-level monotonic counters)
    uint64_t* expected_rdma_flag_value;
    uint32_t* expected_intra_node_flag_value;
    uint64_t* rdma_inter_node_group_flags;
    uint32_t* intra_node_write_completion_flags;

    // GIN context (from ep_group, multi-node only)
    ncclDevComm_t* dcomms;           // Device communicators array
    ncclWindow_t* nccl_windows;      // Windows array (one per comm)
    int num_gin_comms;               // Number of GIN communicators
    int num_ctx_per_comm;            // Number of contexts per communicator
    void* gin_base_ptr;              // Base pointer for offset calculations
    unsigned signals_base;           // Base signal ID
    dispatch_memory_region_info_t mr_info;

    // Runtime config
    int local_rank;
    int node_rank;
    int num_tokens_per_rank;
};

// Call dispatch kernel with runtime template parameter resolution
void call_dispatch(
    const DispatchParams& params,
    int max_tokens_per_rank,    // Max tokens for buffer sizing
    int num_nodes,              // Number of nodes (RDMA domain size)
    bool use_fp8,               // false = BF16 (uint16_t), true = FP8 (uint8_t)
    bool forward_dispatch,      // True for forward, false for backward
    cudaStream_t stream);

// ============================================================================
// Combine wrapper with template parameter resolution
// ============================================================================

// All parameters needed for combine kernel
struct CombineParams {
    // IPC-mapped input buffers (expert outputs from MLP)
    // NOTE: Pass HOST arrays containing device pointers (not device arrays).
    // These pointers are copied into the kernel param struct for fast __grid_constant__ access.
    int hidden_dim;             // Model hidden dimension
    int experts_per_rank;       // Experts per GPU
    int num_ranks_per_node;     // Ranks per node (NVLink domain size, 1-8)
    uint16_t* const* expert_input_token_ptrs;    // HOST array[num_ranks_per_node], BF16 only
    float* const* expert_input_prob_ptrs;        // HOST array, backward only

    // User output buffers
    void* attn_output_token;
    float* attn_output_prob;                           // Backward only (dense format for kernel)

    // RDMA buffers (multi-node only, from ep_group)
    uint16_t* rdma_intra_node_red_token;
    float* rdma_intra_node_red_prob;                   // Backward only
    const uint16_t* combine_rdma_inter_node_group_token;
    const float* combine_rdma_inter_node_group_prob;   // Backward only

    // Metadata (from handle->hybridep preprocessing outputs)
    const int32_t* sparse_to_dense_map;
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;                      // For multi-node RDMA routing
    const bool* local_expert_routing_map;              // For backward gradient routing

    // Sync state (from ep_group, group-level monotonic counters)
    uint64_t* combine_expected_rdma_flag_value;
    uint32_t* combine_expected_intra_node_flag_value;
    uint64_t* combine_rdma_inter_node_group_flags;
    uint32_t* combine_intra_node_write_completion_flags;

    // GIN context (multi-node only)
    ncclDevComm_t* dcomms;           // Device communicators array
    ncclWindow_t* nccl_windows;      // Windows array (one per comm)
    int num_gin_comms;               // Number of GIN communicators
    int num_ctx_per_comm;            // Number of contexts per communicator
    void* gin_base_ptr;              // Base pointer for offset calculations
    unsigned signals_base;           // Base signal ID
    unsigned combine_signal_offset;  // Signal offset for combine operations
    combine_memory_region_info_t mr_info;

    // Runtime config
    int local_rank;
    int node_rank;
    int num_tokens_per_rank;    // Original token count from dispatch
    int num_recv_tokens;        // Actual received tokens this rank
};

// Call combine kernel with runtime template parameter resolution
// Note: HT combine doesn't support FP8, only BF16
void call_combine(
    const CombineParams& params,
    int max_tokens_per_rank,    // Max tokens for buffer sizing
    int num_nodes,              // Number of nodes (RDMA domain size)
    bool backward_combine,      // True for backward (training), false for forward
    cudaStream_t stream);


} // namespace hybridep
} // namespace nccl_ep
