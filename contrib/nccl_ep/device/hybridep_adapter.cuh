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
#include "ep_enums.h"
#include "common.hpp"
#include "nccl_ep_env.h"

namespace nccl_ep {
namespace hybridep {

// ============================================================================
// Runtime constants (from ep_group, not hardcoded):
//   lsa_team_size  - ncclTeamLsa(comm).nRanks  (1..HYBRIDEP_MAX_LSA_TEAM_SIZE)
//   rdma_team_size - ncclTeamRail(comm).nRanks
//   num_local_experts, hidden
//
// Kernels are JIT-compiled per (lsa_team_size, num_lsa_teams, ...); templates
// size their register/SMEM arrays exactly to the JIT lsa_team_size, so any
// value in [1, HYBRIDEP_MAX_LSA_TEAM_SIZE] works.  The only runtime
// precondition is the S2D cp.async.bulk alignment asserted in dispatch_impl
// (matters when s2d_inner_dim is not a multiple of 4).
// ============================================================================

// ============================================================================
// Kernel: Convert sparse topk_idx to dense routing map
// ============================================================================
// NCCL API uses sparse format: topk_idx[token][k] = expert_id
// HT uses bitmap format: routing_bitmap[token][expert/8] has bit (expert%8) set
// cached_topk_idx mirrors topk_idx in its native width (int32 or int64).
template <typename TopkIdxT>
void convert_topk_to_routing_map(
    const TopkIdxT* topk_idx,
    uint8_t* routing_bitmap,
    TopkIdxT* cached_topk_idx,  // nullable; when non-null, mirrors topk_idx in the same pass
    int num_tokens,
    int max_tokens,            // tail bound; rows [num_tokens, max_tokens) are zeroed
    int num_topk,
    int num_experts_packed,    // = ceil(num_experts / 8)
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert sparse topk_weights to dense prob (for dispatch input)
// ============================================================================
// NCCL API uses sparse format: topk_weights[token][k] = weight, topk_idx[token][k] = expert_id
// HT uses dense format: prob[token][expert] = weight (0 for non-selected)
template <typename TopkIdxT>
void sparse_to_dense_prob(
    const TopkIdxT* topk_idx,     // [num_tokens, topk] - which experts
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
    const float* topk_weights, // [num_tokens, topk] - sparse weights from combine input
    const bool* local_expert_routing_map, // [num_tokens, experts_per_rank] - from handle
    float* dense_prob, // [num_tokens, experts_per_node] - output (pre-zeroed)
    int num_tokens,
    int num_topk,
    int experts_per_rank,
    int experts_per_node, // = experts_per_rank * ranks_per_node
    int local_rank, // rank within node
    cudaStream_t stream);

// ============================================================================
// Kernel: Convert dense prob output to sparse format (for dispatch output)
// ============================================================================
// HT outputs: dense_prob[recv_token][experts_per_node]
// HT expects: recv_topk_weights[recv_token][topk], recv_topk_idx[recv_token][topk]
// Uses local_expert_routing_map to know which experts each token is routed to.
// recv_topk_idx numbering is selected by recv_topk_idx_kind:
//   LOCAL  -- per-rank local expert id (0-based within this rank); default.
//   GLOBAL -- wire-format global expert id (= global_expert_offset + local_id),
//             matching the LL rank-major GLOBAL contract.
// recv_topk_idx_kind must be resolved (no AUTO) by the caller.
void dense_to_sparse_prob(
    const float* dense_prob, // [num_recv_tokens, experts_per_node] - from IPC buffer
    const bool* local_expert_routing_map, // [num_recv_tokens, experts_per_rank] - from preprocessing
    float* recv_topk_weights, // EM: [N]; FLAT/RM: [N, topk]
    int64_t* recv_topk_idx, // [num_recv_tokens, topk] - LOCAL or GLOBAL expert id (see kind); nullptr under EM
    int num_recv_tokens,
    int topk,
    int experts_per_rank,
    int experts_per_node, // = experts_per_rank * ranks_per_node
    int local_rank, // rank within node
    int global_expert_offset, // = group_rank * experts_per_rank; added to local id under GLOBAL
    ncclEpExpertIdKind_t recv_topk_idx_kind,
    bool expert_major, // true = 1D recv_topk_weights, false = 2D
    cudaStream_t stream);

// Combine BWD output: sparse k-slot writeback keyed by FWD-input cached_topk_idx.
template <typename TopkIdxT>
void dense_to_sparse_prob_combine(
    const float* dense_prob, // [num_tokens, num_experts]
    const TopkIdxT* cached_topk_idx, // [num_tokens, topk]
    float* combined_topk_weights, // [num_tokens, topk]
    int num_tokens,
    int topk,
    int num_experts,
    cudaStream_t stream);

// ============================================================================
// Preprocessing wrapper with template parameter resolution
// ============================================================================

// Helper to call metadata_preprocessing with resolved template parameters.
// Caller must provide a pre-allocated scan temp buffer (see get_preprocessing_scan_tmp_size).
// When alignment > 0, the kernel also fuses the expert-major remap pass.
ncclResult_t call_metadata_preprocessing(
    const uint8_t* global_routing_map, // Already allgathered bitmap routing map
    int32_t*
        sparse_to_dense_map, // Output: unified S2D — flat stores token->rank->slot; expert-major stores packed (rank, slot) entries
    bool* rdma_to_attn_map, // Output: which tokens come from RDMA
    bool* attn_to_rdma_map, // Output: which tokens go to RDMA
    void* token_rank_mask, // Scratch rank masks, sized by get_rank_mask_elem_size().
    int32_t* num_tokens_for_experts, // Output: total tokens for local experts
    bool* local_expert_routing_map, // Output: per-expert routing for local tokens
    int32_t* per_expert_token_counts, // Optional output: per-expert counts (nullptr to skip)
    void* scan_tmp, // Pre-allocated temp buffer (from get_preprocessing_scan_tmp_size)
    int node_rank, // This node's rank (0 to num_nodes-1)
    int local_rank, // Rank within node (0 to num_ranks_per_node-1)
    int num_tokens_per_rank, // Actual tokens per rank this iteration (runtime)
    int num_nodes, // Number of nodes (RDMA domain size)
    int num_ranks_per_node, // Ranks per node (NVLink domain size)
    int experts_per_rank, // Experts per GPU
    bool expert_major = false, // true = expert-major layout (gates fused remap)
    int64_t* internal_offsets = nullptr, // Expert-major: per-expert zone offsets consumed by dispatch
    void* padded_out_counts =
        nullptr, // Expert-major: per-expert padded counts (caller tensor, nullable; int32 or int64)
    void* out_offsets = nullptr, // Expert-major: per-expert offsets (caller tensor, nullable; int32 or int64)
    size_t alignment =
        0, // Per-expert zone alignment in tokens (pow2; 0/1 = no padding). Ignored when expert_major=false.
    int32_t* actual_counts_out = nullptr, // Expert-major: authoritative per-expert dispatch counts
    int s2d_inner_dim = 0, // 0 = flat (n_ranks_per_node); >0 = expert-major (top_k)
    void* recv_total_counter = nullptr, // Optional scalar: total recv tokens (int32 or int64; nullable)
    bool out_is_int64 = true, // Shared dtype for the 3 int output tensors above
    int max_recv_tokens_per_rank = 0, // HT recv-budget; __trap on overflow.
    int32_t* emuf_group_buf = nullptr, // Local-fanout dup-groups (nullable; EM only): [num_groups, group_stride]
    int32_t* emuf_group_count = nullptr, // Scalar group counter (zeroed by caller)
    int emuf_group_stride = 0, // Row width (= experts_per_rank)
    int emuf_max_groups = 0, // Row capacity; kernel __trap on overflow.
    int num_blocks = 16, // Number of SMs for the kernel grid
    void* scan_gscratch = nullptr, // EM cooperative scan scratch (required when expert_major)
    // EM-permute mode: when true, FLAT scan replaces scan_em and writes the
    // unified (FLAT-shaped) sparse_to_dense_map + FLAT-shape LERM;
    // em_scan_kernel produces only EM offsets / counts / em_slot table.
    bool em_permute = false,
    int32_t* token_to_recv_slot = nullptr,
    int32_t* flat2em_slot_map = nullptr,
    int em_top_k = 0,
    cudaStream_t stream = 0);

// Returns required size in bytes for the scan temp buffer used by call_metadata_preprocessing.
// Caller must allocate at least this many bytes and pass the pointer to call_metadata_preprocessing.
// num_blocks must be >= the block count the scan is launched with (see the
// preprocessing SM count resolved in ncclEpCreateGroup); size with the device
// SM count to cover any NCCL_EP_PREPROCESS_NUM_SMS override.
size_t get_preprocessing_scan_tmp_size(int num_blocks, int num_ranks_per_node);

// Returns sizeof(rank_mask_t<ceil(lsa_team_size/64)>) for the given lsa_team_size.
// Formula: ceil(lsa_team_size / 64) * sizeof(uint64_t).
size_t get_rank_mask_elem_size(int lsa_team_size);

// Cooperative kernel that consumes the preprocessing scan outputs (rank mask)
// and builds the EM-layout index tables: S2D, LERM, per-expert counts/offsets,
// em_slot. Used when expert_major=true. Requires lsa_team_size <= 128
// (fits in 2 x uint64 mask words) and experts_per_rank to be a power of two.
void launch_build_em_tables(
    const uint8_t* input_routing_map,
    const void* token_rank_mask,
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
    bool* local_expert_routing_map,
    int32_t* num_tokens_for_experts,
    int64_t* em_internal_offsets,
    void* em_padded_out_counts,
    void* em_out_offsets,
    int32_t* em_actual_counts_out,
    void* recv_total_counter,
    bool out_is_int64,
    int32_t*
        emuf_group_buf, // Local-fanout dup-groups (nullable): [num_groups, group_stride] = [primary, sec0, ..., -1]
    int32_t* emuf_group_count, // Scalar group counter
    int emuf_group_stride, // Row width (= experts_per_rank)
    int emuf_max_groups, // Row capacity; kernel __trap on overflow.
    int32_t* gscratch,
    int num_sms,
    const int32_t* token_to_recv_slot,
    int32_t* flat2em_slot_map,
    int em_top_k,
    cudaStream_t stream);

// Returns required size (bytes) for the gscratch buffer used by launch_build_em_tables.
size_t get_scan_gscratch_size(int num_ranks_per_node, int experts_per_rank, int num_sms);

// Scatter FLAT staging rows into EM zones using flat2em_slot_map (written by
// em_scan_kernel during UpdateHandle); zero-fill per-expert pad rows.
// prolog_epilog_sms=0 picks sm_count.
void launch_dispatch_permute(
    void* recv_x_em,
    float* recv_topk_weights_em,
    const void* flat_staging,
    const float* recv_topk_weights_flat,
    const int32_t* flat2em_slot_map,
    const int32_t* num_recv_tokens_dev,
    const int64_t* expert_token_offsets,
    const int32_t* per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int row_bytes,
    int sm_count,
    unsigned int prolog_epilog_sms,
    cudaStream_t stream);

// Inverse of launch_dispatch_permute: gather caller EM combine input into
// FLAT staging by summing the top_k EM rows per FLAT recv slot. Optional
// em_weights_in / flat_weights_out pair fuses a 1D EM to [num_flat, top_k]
// FLAT weight gather (BWD combine); pass nullptr for both on FWD.
void launch_combine_reduce(
    void* flat_staging,
    const void* recv_x_em,
    const int32_t* flat2em_slot_map,
    const int32_t* num_recv_tokens_dev,
    const float* em_weights_in,
    float* flat_weights_out,
    int top_k,
    int row_bytes,
    int sm_count,
    unsigned int prolog_epilog_sms,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16);

// ============================================================================
// Memory region info structs for GIN
// All buffers are part of a single large gin_base_ptr buffer
// Offsets are relative to gin_base_ptr (stored as size_t for offset calculation)
// ============================================================================

struct dispatch_memory_region_info_t {
    // Offsets relative to gin_base_ptr for RDMA operations
    size_t attn_input_token_offset; // Offset of token staging buffer from gin_base_ptr
    size_t attn_input_prob_offset; // Offset of prob staging buffer from gin_base_ptr
    size_t attn_input_scaling_factor_offset; // Offset of scaling factor staging buffer
    // Batched RDMA staging (packed layout: token+prob+sf per entry)
    size_t rdma_send_staging_offset; // Offset of per-destination staging buffer
    size_t rdma_inter_node_group_packed_offset; // Offset of packed receive buffer (token+prob+sf per entry)
    size_t guard_offset; // Offset of RDMA sync-guard readiness flags (NUM_LSA_TEAMS uint64 slots)
    size_t bytes_per_entry; // Size of packed entry (token + prob + sf)
    size_t max_tokens_per_dest; // Max tokens that can be staged per destination
    // Streaming RDMA signals
    unsigned signals_tail_base; // Base signal ID for tail tracking (sender -> receiver)
    // Streaming buffer configuration
    int num_max_rdma_chunked_send_tokens; // Batch size per RDMA put (default: 6)
};

struct combine_memory_region_info_t {
    // Offsets relative to gin_base_ptr for RDMA operations
    size_t rdma_intra_node_red_token_offset; // Offset of intra-node reduced token buffer
    size_t combine_rdma_inter_node_group_token_offset; // Offset of combine rdma token buffer
    size_t rdma_intra_node_red_prob_offset; // Offset of intra-node reduced prob buffer
    size_t combine_rdma_inter_node_group_prob_offset; // Offset of combine rdma prob buffer
    size_t guard_offset; // Offset of combine's RDMA sync-guard readiness flags
};

// ============================================================================
// Dispatch wrapper with template parameter resolution
// ============================================================================

// All parameters needed for dispatch kernel, needed redifine because those in hybrid_ep.cuh are with template on data type.
struct DispatchParams {
    // User inputs
    int hidden_dim; // Model hidden dimension
    int experts_per_rank; // Experts per GPU
    int num_ranks_per_node; // Ranks per node (NVLink domain size)
    const void* attn_input_token;
    const float* attn_input_prob; // Forward dispatch only
    const uint8_t* attn_input_scaling_factor; // FP8 only (float* for FP32, uint8_t* for UE8M0)

    // IPC-mapped output buffers (from ep_group)
    // Pointer tables are expected to be device-resident (d_* arrays).
    void* const* expert_output_token_ptrs; // Array[num_ranks_per_node]
    float* const* expert_output_prob_ptrs; // Forward only
    uint8_t* const* expert_output_scaling_factor_ptrs; // FP8 only

    // Metadata (from handle->hybridep preprocessing outputs)
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // Inner dim of unified S2D (num_topk in expert-major, n_ranks_per_node in flat).
    ncclEpLayout_t layout; // Output layout (selects kernel template specialization).

    // EM zero-padding inputs for the in-kernel PAD warp (alignment==0 disables; ptrs may be null).
    const int32_t* pad_actual_counts;
    const int64_t* pad_expert_token_offsets;
    int pad_alignment;

    // Device pointers to the expected counters; initialized at bootstrap and
    // bumped by the dispatch kernel tail so CUDA-graph replays self-sequence.
    uint64_t* expected_rdma_flag_value;
    uint32_t* expected_intra_node_flag_value;
    uint64_t* rdma_inter_node_group_flags;
    uint32_t* intra_node_write_completion_flags;
    // Grid barrier counter for fused device_sync in dispatch tail
    uint32_t* dispatch_grid_barrier_counter;

    // GIN context (from ep_group, multi-node only)
    ncclDevComm dcomm; // Device communicator (single comm, by value)
    ncclWindow_t nccl_token_window; // Source window handle for token data
    ncclWindow_t nccl_prob_window; // Registered window handle for probability data
    ncclWindow_t nccl_sf_window; // Registered window handle for scaling-factor data
    ncclWindow_t nccl_internal_window; // Internal destination window handle
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    dispatch_memory_region_info_t mr_info;

    // Runtime config
    int local_rank;
    int node_rank;
    int num_tokens_per_rank;

    // EM local-fanout: > 0 enables sender S2G dedup + a receiver local_dup kernel.
    int local_dup_num_sms = 0;

    bool guard_enabled = false; // RDMA + LSA buffer guard on/off
};

// Call dispatch kernel with runtime template parameter resolution.
// Returns ncclInvalidArgument if the dispatch SMEM exceeds the device limit.
ncclResult_t call_dispatch(
    const DispatchParams& params,
    int max_dispatch_tokens_per_rank, // Max tokens for buffer sizing (chunk-aligned)
    int num_tokens_per_chunk, // Dispatch/combine tokens per chunk (resolved per group)
    int num_nodes, // Number of nodes (RDMA domain size)
    bool use_fp8, // true = FP8 (uint8_t); mutually exclusive with token_dtype != BF16
    bool forward_dispatch, // True for forward, false for backward
    int num_blocks, // Number of SMs/blocks for the kernel grid
    int sf_bytes_per_token, // Total scale bytes per token (pre-computed on host)
    const ncclEpEnvConfig* env, // Group env config; gates rank-0 verbose param dump (may be null)
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16);

// ============================================================================
// Combine wrapper with template parameter resolution
// ============================================================================

// All parameters needed for combine kernel
struct CombineParams {
    // IPC-mapped input buffers (expert outputs from MLP)
    // NOTE: Pass HOST arrays containing device pointers (not device arrays).
    // These pointers are copied into the kernel param struct for fast __grid_constant__ access.
    int hidden_dim; // Model hidden dimension
    int experts_per_rank; // Experts per GPU
    int num_ranks_per_node; // Ranks per node (NVLink domain size)
    uint16_t* const* expert_input_token_ptrs; // HOST array[num_ranks_per_node], BF16 only
    float* const* expert_input_prob_ptrs; // HOST array, backward only

    // User output buffers
    void* attn_output_token;
    float* attn_output_prob; // Backward only (dense format for kernel)

    // RDMA buffers (multi-node only, from ep_group)
    uint16_t* rdma_intra_node_red_token;
    float* rdma_intra_node_red_prob; // Backward only
    const uint16_t* combine_rdma_inter_node_group_token;
    const float* combine_rdma_inter_node_group_prob; // Backward only

    // Metadata (from handle->hybridep preprocessing outputs)
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // Inner dim of unified S2D (num_topk in expert-major, n_ranks_per_node in flat).
    ncclEpLayout_t layout; // Output layout (selects kernel template specialization).
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map; // For multi-node RDMA routing
    const bool* local_expert_routing_map; // For backward gradient routing

    // Device pointers to the expected counters; initialized at bootstrap and
    // bumped by the combine kernel tail so CUDA-graph replays self-sequence.
    uint64_t* combine_expected_rdma_flag_value;
    uint32_t* combine_expected_intra_node_flag_value;
    uint64_t* combine_rdma_inter_node_group_flags;
    uint32_t* combine_intra_node_write_completion_flags;
    // Per-rank grid-barrier counter that elects the last block at the combine tail.
    uint32_t* combine_grid_barrier_counter;

    // GIN context (multi-node only)
    ncclDevComm_t* dcomms; // Device communicators array
    ncclWindow_t nccl_token_window; // Source window handle for token data
    ncclWindow_t nccl_prob_window; // Source window handle for probability data
    ncclWindow_t nccl_internal_window; // Internal destination window handle
    int num_gin_comms; // Number of GIN communicators
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    unsigned signals_base; // Base signal ID
    unsigned combine_signal_offset; // Signal offset for combine operations
    combine_memory_region_info_t mr_info;

    // Runtime config
    int local_rank;
    int node_rank;
    int num_tokens_per_rank; // Stride for map indexing (= max_tokens_per_rank)
    int num_real_tokens; // Actual token count for output write gate
    int num_recv_tokens; // Actual received tokens this rank

    // EM unfused-combine: combine skips secondary em_slots; primaries hold the
    // pre-reduced sum written by local_reduce.
    bool combine_local_reduce_enabled = false;
    ncclDataType_t token_dtype = ncclBfloat16; // BF16/FP16/FP32 wire (NONE mode); ignored when use_fp8

    bool guard_enabled = false; // RDMA + LSA buffer guard on/off
};

// Call combine kernel with runtime template parameter resolution
void call_combine(
    const CombineParams& params,
    int max_dispatch_tokens_per_rank, // Max tokens for buffer sizing (chunk-aligned)
    int num_tokens_per_chunk, // Dispatch/combine tokens per chunk (resolved per group)
    int num_nodes, // Number of nodes (RDMA domain size)
    bool backward_combine, // True for backward (training), false for forward
    int num_blocks, // Number of SMs/blocks for the kernel grid
    const ncclEpEnvConfig* env, // Group env config; gates rank-0 verbose param dump (may be null)
    cudaStream_t stream);

// EM local-fanout: fill secondary em_slots from primaries after dispatch.
void call_local_dup(
    void* expert_output_token, // [recv_total, hidden]
    float* expert_output_prob, // forward: [recv_total, epr * nranks_per_node]
    const int32_t* emuf_group_buf, // device [num_groups, group_stride]
    const int32_t* emuf_group_count, // device scalar (read by kernel)
    int emuf_group_stride,
    const uint32_t* intra_node_write_completion_flag,
    uint32_t* expected_intra_node_flag_value,
    uint32_t* grid_barrier_counter,
    int hidden_dim,
    int experts_per_rank,
    int num_of_ranks_per_node,
    bool forward_dispatch,
    int num_blocks,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16);

// EM local-fanout: pre-sum secondaries into primaries before combine.
void call_local_reduce(
    void* expert_input_token, // [recv_total, hidden]
    float* expert_input_prob, // backward: [recv_total, epr * nranks_per_node]
    const int32_t* emuf_group_buf, // device [num_groups, group_stride]
    const int32_t* emuf_group_count, // device scalar (read by kernel)
    int emuf_group_stride, // row width (= experts_per_rank)
    int hidden_dim,
    int experts_per_rank,
    int num_of_ranks_per_node,
    bool backward_combine,
    int num_blocks,
    cudaStream_t stream,
    ncclDataType_t token_dtype = ncclBfloat16);

} // namespace hybridep
} // namespace nccl_ep
