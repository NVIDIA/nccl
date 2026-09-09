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
#include "common.hpp"
#include "device_primitives.cuh"
#include "hybridep_configs.cuh"
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda/ptx>
#include "nccl_device.h"
#include "cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x

namespace hybrid_ep {

template <int NUM_OF_BOOL_TO_REDUCE>
using Reduce_t = typename std::conditional<
    NUM_OF_BOOL_TO_REDUCE % 8 == 0,
    uint64_t,
    typename std::conditional<
        NUM_OF_BOOL_TO_REDUCE % 4 == 0,
        uint32_t,
        typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 2 == 0, uint16_t, uint8_t>::type>::type>::type;

template <int NUM_OF_BYTES_TO_COPY>
using Copy_t = typename std::conditional<
    NUM_OF_BYTES_TO_COPY % 16 == 0,
    uint4,
    typename std::conditional<
        NUM_OF_BYTES_TO_COPY % 8 == 0,
        uint2,
        typename std::conditional<
            NUM_OF_BYTES_TO_COPY % 4 == 0,
            uint32_t,
            typename std::conditional<NUM_OF_BYTES_TO_COPY % 2 == 0, uint16_t, uint8_t>::type>::type>::type>::type;

// Conditionally allocate compile-time arrays only when enabled.
template <bool ENABLE, int N>
struct acc_prob_storage_t {};

template <int N>
struct acc_prob_storage_t<true, N> {
    float data[N];
};

// Generic warp group for warp-specializaion.
template <int NUM_WARPS, int STARTING_WARPS>
struct warp_group {
    __host__ __device__ static constexpr int size() {
        return 32 * NUM_WARPS;
    }
    __host__ __device__ static constexpr int warp_size() {
        return NUM_WARPS;
    }

    __host__ __device__ static int thread_rank() {
        return threadIdx.x - (32 * STARTING_WARPS);
    }
    __host__ __device__ static int warp_rank() {
        return thread_rank() / 32;
    }
};

// Memory region info structs for GIN (gin-deepep style)
// All buffers are part of a single large gin_base_ptr buffer
// Offsets are relative to gin_base_ptr (stored as size_t for offset calculation)
struct dispatch_memory_region_info_t {
    size_t attn_input_token_offset;           // Offset of token staging buffer from gin_base_ptr
    size_t attn_input_prob_offset;             // Offset of prob staging buffer from gin_base_ptr
    size_t attn_input_scaling_factor_offset;   // Offset of scaling factor staging buffer
  // Batched RDMA staging (packed layout: token+prob+sf per entry)
    size_t rdma_send_staging_offset;           // Offset of per-destination staging buffer
    size_t rdma_inter_node_group_packed_offset; // Offset of packed receive buffer (token+prob+sf per entry)
    size_t guard_offset; // Offset of RDMA sync-guard readiness flags (NUM_LSA_TEAMS uint64 slots)
    size_t bytes_per_entry; // Size of packed entry (token + prob + sf)
    size_t max_tokens_per_dest; // Max tokens that can be staged per destination
    // Streaming RDMA signals
    unsigned signals_tail_base; // Base signal ID for tail tracking (sender -> receiver)
    // Streaming buffer configuration
    int num_max_rdma_chunked_send_tokens; // Batch size per RDMA put (default: 6)
} __attribute__((__aligned__(8)));

// Tail-signal id for the (src_node -> dst_node) edge of (local_rank, chunk); namespace [src][dst][local_rank][chunk].
__forceinline__ __device__ unsigned dispatch_tail_signal_id(
    unsigned signals_tail_base,
    int src_node,
    int dst_node,
    int local_rank,
    int chunk_idx,
    int num_lsa_teams,
    int ranks_per_node,
    int max_chunks_per_rank) {
    return signals_tail_base +
           ((src_node * num_lsa_teams + dst_node) * ranks_per_node + local_rank) * max_chunks_per_rank + chunk_idx;
}

// Byte offset (from the packed inter-node receive region start) of one source slot's chunk.
// Packed layout is [remote_only_node_id][token-in-slot], each entry bytes_per_entry (token + prob + sf).
__forceinline__ __device__ size_t
dispatch_packed_entry_offset(const dispatch_memory_region_info_t* mr, int remote_only_node_id, int chunk_first_token) {
    return mr->rdma_inter_node_group_packed_offset +
           (static_cast<size_t>(remote_only_node_id) * mr->max_tokens_per_dest +
            static_cast<size_t>(chunk_first_token)) *
               mr->bytes_per_entry;
}

struct combine_memory_region_info_t {
    size_t rdma_intra_node_red_token_offset; // Offset of intra-node reduced token buffer
    size_t combine_rdma_inter_node_group_token_offset; // Offset of combine rdma token buffer
    size_t rdma_intra_node_red_prob_offset; // Offset of intra-node reduced prob buffer
    size_t combine_rdma_inter_node_group_prob_offset; // Offset of combine rdma prob buffer
    size_t guard_offset; // RDMA sync-guard: offset of combine's internal-buffer readiness flags
} __attribute__((__aligned__(8)));

// ============================================================================
// Warp-parallel memory copy helper for RDMA staging
// All 32 threads participate using int4 (16-byte) loads/stores for maximum bandwidth
// ============================================================================
template <int STRIDE = 32>
__device__ __forceinline__ void
warp_copy_int4(void* __restrict__ dst, const void* __restrict__ src, size_t bytes, int lane_id) {
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    const int count = bytes / sizeof(int4);

#pragma unroll 4
    for (int i = lane_id; i < count; i += STRIDE) {
        dst4[i] = __ldg(src4 + i);
    }
    __syncwarp();
}

// Acquire/release lock helpers for shared memory coordination
__device__ __forceinline__ void acquire_lock(int* lock) {
    while (atomicCAS(lock, 0, 1) != 0) {
    }
    __threadfence_block();
}

__device__ __forceinline__ void release_lock(int* lock) {
    __threadfence_block();
    atomicExch(lock, 0);
}

struct dispatch_config_t {
    int num_of_stages;
    int num_of_in_flight_s2g;
    int num_of_tokens_per_chunk;
    int num_of_blocks;
    bool forward_dispatch;
    bool device_side_sync;
    int s2d_inner_dim; // flat: num_ranks_per_node, expert-major: num_topk
    int num_pipelines;
    int stages_per_pipeline;
    int sf_bytes_per_token; // total scale bytes per token (pre-computed on host)
};

struct combine_config_t {
    int num_of_stages_g2s;
    int num_of_stages_s2g;
    int num_of_tokens_per_chunk;
    int num_of_tokens_per_group;
    int num_of_blocks;
    int num_of_additional_in_flight_s2g;
    bool backward_combine;
    bool device_side_sync;
};

struct model_config_t {
    int hidden_dim;
    int max_num_of_tokens_per_rank;
    int num_of_experts_per_rank;
    int num_of_ranks_per_node;
    int num_of_nodes;
};

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
struct dispatch_warp_timing_entry_t {
    long long start_clock;
    long long end_clock;
};
struct combine_warp_timing_entry_t {
    long long work_start_clock;
    long long work_end_clock;
};
struct combine_block_timing_entry_t {
    long long head_sync_start_clock;
    long long head_sync_end_clock;
};
#endif

// Expert-major S2D entry: int32 packed as [31:22]=rank_id (10b, <= 1024), [21:0]=slot (22b).
// -1 = no entry.  Only used when expert-major layout is active.
constexpr int EM_S2D_RANK_BITS = 10;
constexpr int EM_S2D_SLOT_BITS = 32 - EM_S2D_RANK_BITS;
constexpr int EM_S2D_MAX_RANKS = 1 << EM_S2D_RANK_BITS; // 1024
constexpr uint32_t EM_S2D_SLOT_MASK = (1u << EM_S2D_SLOT_BITS) - 1u;
// Slot field must hold values up to MAX_SUPPORTED_TOKENS_PER_RANK; -1 sentinel must not collide.
static_assert(
    MAX_SUPPORTED_TOKENS_PER_RANK < (1 << EM_S2D_SLOT_BITS),
    "MAX_SUPPORTED_TOKENS_PER_RANK exceeds em_s2d slot field width");

// s2d-map double-buffer depth: consume one stage while prefetching the next (chunk, node) row.
constexpr int S2D_MAP_RING_STAGES = 2;

__host__ __device__ __forceinline__ int32_t em_s2d_pack(int rank_id, int slot) {
    return static_cast<int32_t>(
        (static_cast<uint32_t>(rank_id) << EM_S2D_SLOT_BITS) | (static_cast<uint32_t>(slot) & EM_S2D_SLOT_MASK));
}
__host__ __device__ __forceinline__ int em_s2d_unpack_rank(int32_t v) {
    return static_cast<int>(static_cast<uint32_t>(v) >> EM_S2D_SLOT_BITS);
}
__host__ __device__ __forceinline__ int em_s2d_unpack_slot(int32_t v) {
    return static_cast<int>(static_cast<uint32_t>(v) & EM_S2D_SLOT_MASK);
}

// EM unfused-combine path: s2d entries are lex-sorted (dest_rank, k), so duplicates
// within a row are adjacent. Returns true when this lane's entry has the same
// dest_rank as the lane immediately upstream -- the peer rank's local_reduce_kernel
// has already merged it into the primary slot, so this entry should be skipped.
// Must be called with all 32 lanes active (uses __shfl_up_sync(0xffffffff, ...)).
template <ncclEpLayout_t kLayout>
__device__ __forceinline__ bool is_em_secondary_entry(int32_t s2d_val, int lane_id, bool enabled) {
    if constexpr (kLayout != NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        return false;
    }
    const int32_t prev = __shfl_up_sync(0xffffffff, s2d_val, 1);
    return enabled && lane_id > 0 && s2d_val != -1 && prev != -1 &&
           em_s2d_unpack_rank(s2d_val) == em_s2d_unpack_rank(prev);
}

// Popcount of row[start_bit, end_bit). Step 4 uses this to derive em_s2d slot indices atomic-free.
__device__ __forceinline__ int popcount_bit_range(const uint8_t* row, int start_bit, int end_bit) {
    if (end_bit <= start_bit) return 0;
    const int byte_lo = start_bit >> 3;
    const int byte_hi = (end_bit + 7) >> 3;
    const int lo_off = start_bit & 7;
    const int hi_keep = end_bit & 7; // 0 → keep full last byte
    int total = 0;
    for (int b = byte_lo; b < byte_hi; b++) {
        unsigned byte = row[b];
        if (b == byte_lo) byte &= (0xFFu << lo_off) & 0xFFu;
        if (b == byte_hi - 1 && hi_keep != 0) byte &= (1u << hi_keep) - 1u;
        total += __popc(byte);
    }
    return total;
}

// Extract up to 64 contiguous bits from a bit-packed row.
__device__ __forceinline__ uint64_t extract_bits64(const uint8_t* row, int start_bit, int nbits) {
    if (nbits <= 0) return 0;
    const int byte_lo = start_bit >> 3;
    const int byte_hi = (start_bit + nbits + 7) >> 3;
    const int lo_off = start_bit & 7;
    uint64_t out = 0;
    for (int b = byte_lo; b < byte_hi && (b - byte_lo) < 9; b++) {
        out |= static_cast<uint64_t>(row[b]) << ((b - byte_lo) * 8);
    }
    out >>= lo_off;
    if (nbits < 64) out &= (static_cast<uint64_t>(1) << nbits) - 1;
    return out;
}

struct combine_smem_layout_t {
    uint16_t* intra_node_token_G2S_buffer;
    uint16_t* intra_node_token_S2G_buffer;
    uint16_t* inter_node_token_G2S_buffer;
    uint16_t* inter_node_token_S2G_buffer;
    float* intra_node_prob_G2S_buffer;
    float* intra_node_prob_S2G_buffer;
    float* inter_node_prob_G2S_buffer;
    float* inter_node_prob_S2G_buffer;
    uint64_t* intra_node_mbarrier_G2S_buffer;
    uint64_t* inter_node_mbarrier_G2S_buffer;
    uint64_t* intra_node_to_rdma_mbarrier_buffer;
    bool* intra_node_flag_G2S_buffer;
    bool* inter_node_flag_G2S_buffer;

    int token_G2S_stage_stride; // elements (not bytes)
    int token_S2G_stage_stride; // elements (not bytes)
    int prob_G2S_stage_stride; // elements (not bytes)
    int prob_S2G_stage_stride; // intra-node elements (not bytes)
    int prob_S2G_inter_stage_stride; // inter-node elements (not bytes)
    combine_memory_region_info_t* combine_memory_region_info;

    // Streaming overlap: reduction warp -> RDMA warp within a chunk
    uint32_t* rdma_streaming_counter; // [1] cumulative tokens produced for the current chunk

    int s2d_inner_dim; // Inner dimension of unified S2D map (n_ranks_per_node or num_topk)

    // Accessor methods for staged buffers
    __device__ __forceinline__ uint16_t* get_intra_node_token_G2S(int stage) const {
        return intra_node_token_G2S_buffer + stage * token_G2S_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_intra_node_token_S2G(int stage) const {
        return intra_node_token_S2G_buffer + stage * token_S2G_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_inter_node_token_G2S(int stage) const {
        return inter_node_token_G2S_buffer + stage * token_G2S_stage_stride;
    }
    __device__ __forceinline__ uint16_t* get_inter_node_token_S2G(int stage) const {
        return inter_node_token_S2G_buffer + stage * token_S2G_stage_stride;
    }
    __device__ __forceinline__ float* get_intra_node_prob_G2S(int stage) const {
        return intra_node_prob_G2S_buffer + stage * prob_G2S_stage_stride;
    }
    __device__ __forceinline__ float* get_intra_node_prob_S2G(int stage) const {
        return intra_node_prob_S2G_buffer + stage * prob_S2G_stage_stride;
    }
    __device__ __forceinline__ float* get_inter_node_prob_G2S(int stage) const {
        return inter_node_prob_G2S_buffer + stage * prob_G2S_stage_stride;
    }
    __device__ __forceinline__ float* get_inter_node_prob_S2G(int stage) const {
        return inter_node_prob_S2G_buffer + stage * prob_S2G_inter_stage_stride;
    }
    // Accessor methods for mbarrier buffers (producer = stage*2, consumer = stage*2+1)
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_G2S_producer(int stage) const {
        return intra_node_mbarrier_G2S_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_G2S_consumer(int stage) const {
        return intra_node_mbarrier_G2S_buffer + stage * 2 + 1;
    }
    __device__ __forceinline__ uint64_t* get_inter_node_mbarrier_G2S_producer(int stage) const {
        return inter_node_mbarrier_G2S_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_inter_node_mbarrier_G2S_consumer(int stage) const {
        return inter_node_mbarrier_G2S_buffer + stage * 2 + 1;
    }
};

struct dispatch_smem_layout_t {
    void* intra_node_token_buffer;
    float* intra_node_prob_buffer;
    uint8_t* intra_node_scaling_factor_buffer;
    int32_t* sparse_to_dense_map_buffer;
    bool* attn_to_rdma_map_buffer;
    uint64_t* intra_node_mbarrier_buffer;
    uint64_t* sparse_to_dense_map_mbarrier_buffer;
    uint64_t* S2G_group_mbarrier_buffer;
    // Single TMA staging slot used by the PAD warp to broadcast a zeroed token
    // row to padding slots (expert-major only; nullptr otherwise).
    void* pad_tma_buffer;

    int token_buffer_stage_stride; // bytes
    int prob_buffer_stage_stride; // bytes
    int sf_buffer_stage_stride; // bytes
    int s2d_map_stage_stride; // bytes (flat: tokens * ranks, expert-major: tokens * topk)
    int pad_tma_slot_bytes; // bytes (= padded hidden_dim * sizeof(token))
    int s2d_inner_dim; // flat: num_ranks_per_node, expert-major: num_topk
    int num_pipelines;
    int stages_per_pipeline;
    dispatch_memory_region_info_t* dispatch_memory_region_info;

    // Flat stage accessors (used when pipeline_id is already folded into stage)
    __device__ __forceinline__ void* get_token_buffer(int stage) const {
        return reinterpret_cast<void*>(
            reinterpret_cast<uint8_t*>(intra_node_token_buffer) + stage * token_buffer_stage_stride);
    }
    __device__ __forceinline__ float* get_prob_buffer(int stage) const {
        return reinterpret_cast<float*>(
            reinterpret_cast<uint8_t*>(intra_node_prob_buffer) + stage * prob_buffer_stage_stride);
    }
    __device__ __forceinline__ void* get_sf_buffer(int stage) const {
        return reinterpret_cast<void*>(intra_node_scaling_factor_buffer + stage * sf_buffer_stage_stride);
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_producer(int stage) const {
        return intra_node_mbarrier_buffer + stage * 2;
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_consumer(int stage) const {
        return intra_node_mbarrier_buffer + stage * 2 + 1;
    }

    // Pipeline-indexed stage accessors: translate (pipeline_id, local_stage) to absolute stage
    __device__ __forceinline__ void* get_token_buffer(int pipeline_id, int local_stage) const {
        return get_token_buffer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ float* get_prob_buffer(int pipeline_id, int local_stage) const {
        return get_prob_buffer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ void* get_sf_buffer(int pipeline_id, int local_stage) const {
        return get_sf_buffer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_producer(int pipeline_id, int local_stage) const {
        return get_intra_node_mbarrier_producer(pipeline_id * stages_per_pipeline + local_stage);
    }
    __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_consumer(int pipeline_id, int local_stage) const {
        return get_intra_node_mbarrier_consumer(pipeline_id * stages_per_pipeline + local_stage);
    }

    // Per-pipeline s2d_map accessors: each pipeline has its own S2D_MAP_RING_STAGES ping-pong stages
    __device__ __forceinline__ int32_t* get_s2d_map_buffer(int pipeline_id, int stage, int token_idx) const {
        int abs_stage = pipeline_id * S2D_MAP_RING_STAGES + stage;
        return reinterpret_cast<int32_t*>(
                   reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + abs_stage * s2d_map_stage_stride) +
               token_idx * s2d_inner_dim;
    }
    __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int pipeline_id, int stage) const {
        int abs_stage = pipeline_id * S2D_MAP_RING_STAGES + stage;
        return reinterpret_cast<int32_t*>(
            reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + abs_stage * s2d_map_stage_stride);
    }
    // Legacy s2d accessors (pipeline_id=0)
    __device__ __forceinline__ int32_t* get_s2d_map_buffer(int stage, int token_idx) const {
        return get_s2d_map_buffer(0, stage, token_idx);
    }
    __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int stage) const {
        return get_s2d_map_buffer_base(0, stage);
    }

    // Per-pipeline s2d_map mbarrier: each pipeline has S2D_MAP_RING_STAGES ping-pong mbarriers
    __device__ __forceinline__ uint64_t* get_s2d_map_mbar(int pipeline_id, int stage) const {
        return sparse_to_dense_map_mbarrier_buffer + pipeline_id * S2D_MAP_RING_STAGES + stage;
    }
    // Per-pipeline S2G group mbarrier
    __device__ __forceinline__ uint64_t* get_S2G_group_mbar(int pipeline_id) const {
        return S2G_group_mbarrier_buffer + pipeline_id;
    }

    __device__ __forceinline__ void* get_pad_tma_slot() const {
        return pad_tma_buffer;
    }
};

template <ncclEpLayout_t kLayout, ncclDataType_t kTokenDtype>
__device__ dispatch_smem_layout_t create_dispatch_smem_layout(
    dispatch_smem_layout_t& layout,
    void* smem_base,
    const dispatch_config_t& config,
    const model_config_t& model) {
    size_t offset = 0;
    const int num_pipelines = config.num_pipelines;
    layout.num_pipelines = num_pipelines;
    layout.stages_per_pipeline = config.stages_per_pipeline;

    // Token buffer (aligned to 128B for TMA) -- total stages unchanged
    const int token_size = nccl_ep::size_u8<kTokenDtype>();
    layout.token_buffer_stage_stride = model.hidden_dim * token_size;
    layout.token_buffer_stage_stride = (layout.token_buffer_stage_stride + 127) & ~127;
    layout.intra_node_token_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * layout.token_buffer_stage_stride;

    // Sparse to dense map buffer: S2D_MAP_RING_STAGES ping-pong stages PER PIPELINE (128B aligned)
    // Inner dim is mode-dependent: flat = num_ranks_per_node, expert-major = num_topk.
    layout.s2d_inner_dim = config.s2d_inner_dim;
    layout.s2d_map_stage_stride = config.num_of_tokens_per_chunk * config.s2d_inner_dim * sizeof(int32_t);
    layout.s2d_map_stage_stride = (layout.s2d_map_stage_stride + 127) & ~127;
    layout.sparse_to_dense_map_buffer = reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += S2D_MAP_RING_STAGES * num_pipelines * layout.s2d_map_stage_stride;

    // Prob buffer (only if forward dispatch, 16B aligned) -- total stages unchanged
    if (config.forward_dispatch) {
        layout.prob_buffer_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);
        layout.prob_buffer_stage_stride = (layout.prob_buffer_stage_stride + 15) & ~15;
        layout.intra_node_prob_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_stages * layout.prob_buffer_stage_stride;
    } else {
        layout.intra_node_prob_buffer = nullptr;
        layout.prob_buffer_stage_stride = 0;
    }

    // Scaling factor buffer (only if quantized, 16B aligned) -- total stages unchanged
    if (config.sf_bytes_per_token > 0) {
        layout.sf_buffer_stage_stride = config.sf_bytes_per_token;
        layout.sf_buffer_stage_stride = (layout.sf_buffer_stage_stride + 15) & ~15;
        layout.intra_node_scaling_factor_buffer = reinterpret_cast<uint8_t*>(smem_base) + offset;
        offset += config.num_of_stages * layout.sf_buffer_stage_stride;
    } else {
        layout.intra_node_scaling_factor_buffer = nullptr;
        layout.sf_buffer_stage_stride = 0;
    }

    // attn_to_rdma_map buffer (16B aligned, only if multinode, shared across pipelines)
    if (model.num_of_nodes > 1) {
        offset = (offset + 15) & ~15;
        layout.attn_to_rdma_map_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
    } else {
        layout.attn_to_rdma_map_buffer = nullptr;
    }

    // Mbarrier buffers (8B aligned) -- total stages unchanged (producer+consumer per stage)
    offset = (offset + 7) & ~7;
    layout.intra_node_mbarrier_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * 2 * sizeof(uint64_t);

    // Per-pipeline s2d_map mbarriers: 2 per pipeline
    layout.sparse_to_dense_map_mbarrier_buffer =
        reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += 2 * num_pipelines * sizeof(uint64_t);

    // Per-pipeline S2G group mbarrier: 1 per pipeline
    layout.S2G_group_mbarrier_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_pipelines * sizeof(uint64_t);

    if (model.num_of_nodes > 1) {
        offset = (offset + 7) & ~7;
        layout.dispatch_memory_region_info =
            reinterpret_cast<dispatch_memory_region_info_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
    } else {
        layout.dispatch_memory_region_info = nullptr;
    }

    // PAD warp TMA slot: one zeroed token row, broadcast to padding slots.
    // Only allocated for expert-major; flat leaves the pointer null.
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        int pad_bytes = model.hidden_dim * nccl_ep::size_u8<kTokenDtype>();
        layout.pad_tma_slot_bytes = (pad_bytes + 127) & ~127;
        offset = (offset + 127) & ~127;
        layout.pad_tma_buffer = reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += layout.pad_tma_slot_bytes;
    } else {
        layout.pad_tma_buffer = nullptr;
        layout.pad_tma_slot_bytes = 0;
    }

    return layout;
}
template <ncclEpLayout_t kLayout, ncclDataType_t kTokenDtype>
static size_t calculate_dispatch_smem_layout_size(const dispatch_config_t& config, const model_config_t& model) {
    size_t total_size = 0;
    const int num_pipelines = config.num_pipelines;
    const int token_size = nccl_ep::size_u8<kTokenDtype>();

    // Token buffer (aligned to 128B for TMA) -- total stages unchanged
    int token_buffer_stage_stride = model.hidden_dim * token_size;
    token_buffer_stage_stride = (token_buffer_stage_stride + 127) & ~127;
    total_size += config.num_of_stages * token_buffer_stage_stride;

    // Sparse to dense map buffer: S2D_MAP_RING_STAGES ping-pong stages PER PIPELINE (128B aligned)
    // Inner dim is mode-dependent: flat = num_ranks_per_node, expert-major = num_topk.
    int s2d_map_stage_stride = config.num_of_tokens_per_chunk * config.s2d_inner_dim * sizeof(int32_t);
    s2d_map_stage_stride = (s2d_map_stage_stride + 127) & ~127;
    total_size += S2D_MAP_RING_STAGES * num_pipelines * s2d_map_stage_stride;

    // Prob buffer (16B aligned per stage) -- total stages unchanged
    if (config.forward_dispatch) {
        int prob_buffer_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);
        prob_buffer_stage_stride = (prob_buffer_stage_stride + 15) & ~15;
        total_size += config.num_of_stages * prob_buffer_stage_stride;
    }

    // Scaling factor buffer (16B aligned per stage, only if quantized) -- total stages unchanged
    if (config.sf_bytes_per_token > 0) {
        int sf_buffer_stage_stride = config.sf_bytes_per_token;
        sf_buffer_stage_stride = (sf_buffer_stage_stride + 15) & ~15;
        total_size += config.num_of_stages * sf_buffer_stage_stride;
    }
    // attn_to_rdma_map buffer (aligned to 16B, only if multinode, shared)
    if (model.num_of_nodes > 1) {
        total_size = (total_size + 15) & ~15;
        total_size += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
    }
    // Mbarrier buffers (aligned to 8B) -- total stages unchanged
    total_size = (total_size + 7) & ~7;
    total_size += config.num_of_stages * 2 * sizeof(uint64_t);
    // Per-pipeline s2d_map mbarriers: 2 per pipeline
    total_size = (total_size + 7) & ~7;
    total_size += 2 * num_pipelines * sizeof(uint64_t);
    // Per-pipeline S2G group mbarrier: 1 per pipeline
    total_size = (total_size + 7) & ~7;
    total_size += num_pipelines * sizeof(uint64_t);
    // Dispatch memory region info buffer (aligned to 8B, only if multinode)
    if (model.num_of_nodes > 1) {
        total_size = (total_size + 7) & ~7;
        total_size += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
    }
    // PAD warp TMA slot (expert-major only, 128B aligned)
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        int pad_bytes = model.hidden_dim * nccl_ep::size_u8<kTokenDtype>();
        pad_bytes = (pad_bytes + 127) & ~127;
        total_size = (total_size + 127) & ~127;
        total_size += pad_bytes;
    }
    // Add padding for alignment
    total_size = (total_size + 127) & ~127;
    return total_size;
}

// kTokenDtype drives the per-stage token-buffer stride via the derived element width
// (2 B for BF16/FP16, 4 B for FP32); everything else (prob, mbarriers, scales) is
// element-width-invariant.
template <ncclDataType_t kTokenDtype = ncclBfloat16>
__device__ combine_smem_layout_t create_combine_smem_layout(
    combine_smem_layout_t& layout,
    void* smem_base,
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_chunk,
    bool backward_combine,
    const model_config_t& model) {
    size_t offset = 0;
    const uintptr_t smem_base_addr = reinterpret_cast<uintptr_t>(smem_base);
    auto align_offset = [&](size_t alignment) {
        const size_t mask = alignment - 1;
        const size_t misalignment = (smem_base_addr + offset) & mask;
        if (misalignment != 0) {
            offset += alignment - misalignment;
        }
    };

    // In the single-node case (num_of_nodes == 1), the combine kernel does not use the
    // intra-node staging buffers. Skipping these buffers can cut SMEM roughly in half.
    const bool multinode = (model.num_of_nodes > 1);

    // Per-token wire size: bytes for buffer offsets, uint16_t units for stage strides (the
    // token buffer base is uint16_t*). FP32 doubles both vs BF16/FP16.
    const int token_bytes = model.hidden_dim * nccl_ep::size_u8<kTokenDtype>();
    const int token_stride_u16 = model.hidden_dim * nccl_ep::size_u16<kTokenDtype>();

    // Stage strides in uint16_t units, so FP32 stages advance 2× and don't overlap.
    layout.token_G2S_stage_stride = token_stride_u16;
    layout.token_S2G_stage_stride = token_stride_u16;
    layout.prob_G2S_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node;
    layout.prob_S2G_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node;
    layout.prob_S2G_inter_stage_stride = layout.prob_S2G_stage_stride * model.num_of_nodes;

    // intra_node_token_* buffers (128B aligned, multi-node only). Stage stride scales
    // with the on-wire element width (2 B for BF16/FP16, 4 B for FP32).
    if (multinode) {
        align_offset(128);
        layout.intra_node_token_G2S_buffer =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * token_bytes;

        align_offset(128);
        layout.intra_node_token_S2G_buffer =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * token_bytes;
    } else {
        layout.intra_node_token_G2S_buffer = nullptr;
        layout.intra_node_token_S2G_buffer = nullptr;
    }

    // inter_node_token_G2S_buffer (128B aligned)
    align_offset(128);
    layout.inter_node_token_G2S_buffer = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * token_bytes;

    // inter_node_token_S2G_buffer (128B aligned)
    align_offset(128);
    layout.inter_node_token_S2G_buffer = reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_s2g * token_bytes;

    // Prob buffers (only if backward_combine, 16B aligned)
    if (backward_combine) {
        if (multinode) {
            // intra_node_prob_G2S_buffer
            align_offset(16);
            layout.intra_node_prob_G2S_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_g2s * model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);

            // intra_node_prob_S2G_buffer
            align_offset(16);
            layout.intra_node_prob_S2G_buffer =
                reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
            offset += num_of_stages_s2g * model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);
        } else {
            layout.intra_node_prob_G2S_buffer = nullptr;
            layout.intra_node_prob_S2G_buffer = nullptr;
        }

        // inter_node_prob_G2S_buffer
        align_offset(16);
        layout.inter_node_prob_G2S_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);

        // inter_node_prob_S2G_buffer
        align_offset(16);
        layout.inter_node_prob_S2G_buffer = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_s2g * model.num_of_experts_per_rank * model.num_of_ranks_per_node * model.num_of_nodes *
                  sizeof(float);
    } else {
        layout.intra_node_prob_G2S_buffer = nullptr;
        layout.intra_node_prob_S2G_buffer = nullptr;
        layout.inter_node_prob_G2S_buffer = nullptr;
        layout.inter_node_prob_S2G_buffer = nullptr;
    }

    // Mbarrier buffers (8B aligned)
    // intra_node_mbarrier_G2S_buffer (multi-node only)
    if (multinode) {
        align_offset(8);
        layout.intra_node_mbarrier_G2S_buffer =
            reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * 2 * sizeof(uint64_t);
    } else {
        layout.intra_node_mbarrier_G2S_buffer = nullptr;
    }

    // inter_node_mbarrier_G2S_buffer
    align_offset(8);
    layout.inter_node_mbarrier_G2S_buffer = reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * 2 * sizeof(uint64_t);

    // intra_node_to_rdma_mbarrier_buffer (only if multi-node)
    if (model.num_of_nodes > 1) {
        int max_num_of_chunks_per_rank =
            (model.max_num_of_tokens_per_rank + num_of_tokens_per_chunk - 1) / num_of_tokens_per_chunk;
        align_offset(8);
        layout.intra_node_to_rdma_mbarrier_buffer =
            reinterpret_cast<uint64_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_of_nodes - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
    } else {
        layout.intra_node_to_rdma_mbarrier_buffer = nullptr;
    }

    if (model.num_of_nodes > 1) {
        align_offset(8);
        layout.combine_memory_region_info =
            reinterpret_cast<combine_memory_region_info_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += (model.num_of_nodes - 1) * sizeof(combine_memory_region_info_t);
    } else {
        layout.combine_memory_region_info = nullptr;
    }

    // Flag buffers (no special alignment needed)
    if (multinode) {
        layout.intra_node_flag_G2S_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += num_of_stages_g2s * sizeof(bool);
    } else {
        layout.intra_node_flag_G2S_buffer = nullptr;
    }

    layout.inter_node_flag_G2S_buffer = reinterpret_cast<bool*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * sizeof(bool);

    // Streaming overlap fields (multi-node only, 4B aligned)
    if (multinode) {
        align_offset(4);
        layout.rdma_streaming_counter = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_base) + offset);
        offset += sizeof(uint32_t);
    } else {
        layout.rdma_streaming_counter = nullptr;
    }

    return layout;
}

template <ncclDataType_t kTokenDtype = ncclBfloat16>
static size_t calculate_combine_smem_layout_size(
    int num_of_stages_g2s,
    int num_of_stages_s2g,
    int num_of_tokens_per_chunk,
    int max_num_of_tokens_per_rank,
    int num_lsa_teams,
    bool backward_combine,
    const model_config_t& model) {
    // Dynamically computes the size required for combine shared memory layout,
    // mirroring the logic from create_combine_smem_layout
    size_t total_size = 0;

    // Compute max number of chunks per rank
    const int hidden_dim = model.hidden_dim;
    const int token_bytes = hidden_dim * nccl_ep::size_u8<kTokenDtype>(); // per-token wire bytes
    const int max_num_of_chunks_per_rank =
        (max_num_of_tokens_per_rank + num_of_tokens_per_chunk - 1) / num_of_tokens_per_chunk;
    const bool multinode = (num_lsa_teams > 1);

    // Token buffers (128B aligned for TMA). Stage stride scales with the wire element
    // width (2 B for BF16/FP16, 4 B for FP32).
    // intra_node_token_* buffers (multi-node only)
    if (multinode) {
        total_size = (total_size + 127) & ~127;
        total_size += num_of_stages_g2s * token_bytes;

        total_size = (total_size + 127) & ~127;
        total_size += num_of_stages_s2g * token_bytes;
    }

    // inter_node_token_G2S_buffer
    total_size = (total_size + 127) & ~127;
    total_size += num_of_stages_g2s * token_bytes;

    // inter_node_token_S2G_buffer
    total_size = (total_size + 127) & ~127;
    total_size += num_of_stages_s2g * token_bytes;

    // Prob buffers (16B aligned, only if backward_combine)
    if (backward_combine) {
        if (multinode) {
            // intra_node_prob_G2S_buffer
            total_size = (total_size + 15) & ~15;
            total_size +=
                num_of_stages_g2s * model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);

            // intra_node_prob_S2G_buffer
            total_size = (total_size + 15) & ~15;
            total_size +=
                num_of_stages_s2g * model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);
        }

        // inter_node_prob_G2S_buffer
        total_size = (total_size + 15) & ~15;
        total_size += num_of_stages_g2s * model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);

        // inter_node_prob_S2G_buffer
        total_size = (total_size + 15) & ~15;
        total_size += num_of_stages_s2g * model.num_of_experts_per_rank * model.num_of_ranks_per_node * num_lsa_teams *
                      sizeof(float);
    }

    // Mbarrier buffers (8B aligned)
    // intra_node_mbarrier_G2S_buffer [stages][2] (multi-node only)
    if (multinode) {
        total_size = (total_size + 7) & ~7;
        total_size += num_of_stages_g2s * 2 * sizeof(uint64_t);
    }

    // inter_node_mbarrier_G2S_buffer [stages][2]
    total_size = (total_size + 7) & ~7;
    total_size += num_of_stages_g2s * 2 * sizeof(uint64_t);

    // intra_node_to_rdma_mbarrier_buffer [(nodes-1)][chunks] (only if multi-node)
    if (multinode) {
        total_size = (total_size + 7) & ~7;
        total_size += (num_lsa_teams - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
    }

    // combine_memory_region_info [(nodes-1)] (align 8B, only if multi-node)
    if (multinode) {
        total_size = (total_size + 7) & ~7;
        total_size += (num_lsa_teams - 1) * sizeof(combine_memory_region_info_t);
    }

    // Flag buffers (no special alignment needed)
    if (multinode) {
        total_size += num_of_stages_g2s * sizeof(bool);
    }
    total_size += num_of_stages_g2s * sizeof(bool);

    // Streaming overlap fields (multi-node only, 4B aligned)
    if (multinode) {
        total_size = (total_size + 3) & ~3;
        total_size += sizeof(uint32_t); // rdma_streaming_counter
    }

    return total_size;
}
// Fixed-size part of dispatch kernel parameters. Peer pointer arrays are appended
// by dispatch_kernel_param_t<..., LSA_TEAM_SIZE> for JIT-specialized kernels.
template <typename TOKEN_DATA_TYPE>
struct dispatch_kernel_param_base_t {
    int hidden_dim;
    int experts_per_rank;
    int num_of_ranks_per_node;
    // Input buffers. These buffers are local buffers.
    const TOKEN_DATA_TYPE* attn_input_token;
    const float* attn_input_prob; // Needed by expert layer, so only valid in forward dispatch.
    const uint8_t*
        attn_input_token_scaling_factor; // FP8 EXTERN: per-token scales (float* for FP32, uint8_t* for UE8M0 — pure byte transport).
    // Internal temp buffers. These buffers are local buffers.
    uint64_t* rdma_inter_node_group_flags; // For RDMA Atomic flags.
    uint32_t* intra_node_write_completion_flags; // For intra-node S2G write completion notification.
    // Metadata buffers. These buffers are local buffers.
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // flat: num_ranks_per_node, expert-major: num_topk
    // Expert-major zero-padding inputs for the PAD warp.
    // PAD warp is a no-op when pad_alignment <= 1; pointers may be null then.
    const int32_t* pad_actual_counts; // [experts_per_rank] unpadded token counts
    const int64_t* pad_expert_token_offsets; // [experts_per_rank] zone start offsets
    int pad_alignment; // per-expert zone alignment in tokens (<=1 = no padding)
    // Device-resident expected counters. Initialized at bootstrap; bumped in
    // this kernel's tail so CUDA-graph capture+replay self-sequences.
    uint64_t* expected_rdma_flag_value;
    uint32_t* expected_intra_node_flag_value;
    int local_rank;
    int node_rank;
    // The number of token output by attn layer on a rank/GPU.
    int num_of_tokens_per_rank;
    // NCCL GIN context
    ncclDevComm dcomm; // Device communicator
    ncclWindow_t token_window; // Source window handle for token data
    ncclWindow_t prob_window; // Source window handle for probability data
    ncclWindow_t sf_window; // Source window handle for scaling-factor data
    ncclWindow_t dest_window; // Destination window handle
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    // Memory Region info
    struct dispatch_memory_region_info_t mr_info;
    // Grid barrier counter for fused device_sync in dispatch tail (per-rank, not IPC-shared)
    uint32_t* dispatch_grid_barrier_counter;
    // When true (EM layout with local fanout enabled),
    // sender S2G dedups consecutive same-dest entries; secondary slots are filled
    // afterwards by the local_dup kernel.
    bool local_dup_enabled;
    // Cross-round WAR sync-guards: LSA (intra-node staging) uses the NCCL LSA barrier; RDMA
    // (inter-node staging) is hand-rolled. Only the enable flags are needed on the device now.
    bool guard_enabled; // cross-round WAR guard (LSA + RDMA share one enable)
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    dispatch_warp_timing_entry_t* warp_timing;
#endif
};

// Data structure for JIT dispatch kernel parameters.
template <typename TOKEN_DATA_TYPE, int LSA_TEAM_SIZE>
struct dispatch_kernel_param_t : dispatch_kernel_param_base_t<TOKEN_DATA_TYPE> {
    // Output buffers. These buffers are both local and remote buffers.
    // Keep embedded arrays here to avoid device-side pointer-table indirection.
    TOKEN_DATA_TYPE* expert_output_token[LSA_TEAM_SIZE];
    float* expert_output_prob[LSA_TEAM_SIZE]; // Only valid in forward dispatch.
    uint8_t* expert_output_scaling_factor[LSA_TEAM_SIZE]; // Only valid for FP8 token type.
};

// Fixed-size part of combine kernel parameters. Peer pointer arrays are appended
// by combine_kernel_param_t<LSA_TEAM_SIZE> for JIT-specialized kernels.
struct combine_kernel_param_base_t {
    int hidden_dim;
    int experts_per_rank;
    int num_of_ranks_per_node;
    // EM unfused-combine: when true, the inter-node G2S warp group skips local-dup
    // secondary em_slots (primaries already carry the pre-reduced weighted sum
    // written by the local_reduce kernel). Default false => fused fanout.
    bool combine_local_reduce_enabled;
    // Output buffers. These buffers are local buffers.
    uint16_t* attn_output_token;
    float* attn_output_prob;
    // Internal temp buffers. These buffers are local buffers.
    uint16_t* rdma_intra_node_red_token;
    float* rdma_intra_node_red_prob;
    const uint16_t* rdma_inter_node_group_token;
    const float* rdma_inter_node_group_prob;
    uint64_t* rdma_inter_node_group_flags;
    uint32_t* intra_node_write_completion_flags; // For intra-node src ready notification.
    // Metadata buffers. These buffers are local buffers.
    const bool* rdma_to_attn_map;
    const bool* attn_to_rdma_map;
    const int32_t* sparse_to_dense_map;
    int s2d_inner_dim; // flat: num_ranks_per_node, expert-major: num_topk
    // Device-resident expected counters. Initialized at bootstrap; bumped in
    // this kernel's tail so CUDA-graph capture+replay self-sequences.
    uint64_t* expected_rdma_flag_value;
    uint32_t* expected_intra_node_flag_value;
    int local_rank;
    int node_rank;
    // Stride for routing-map indexing (= max_tokens_per_rank).
    int num_of_tokens_per_rank;
    // Actual token count; gates the inter_node_red TMA store.
    int num_real_tokens;
    // Per-rank grid-barrier counter that elects the last block at the combine tail.
    uint32_t* combine_grid_barrier_counter;
    // NCCL GIN context
    ncclDevComm_t* dcomms; // Device communicators array (1 element, on device)
    ncclWindow_t token_window; // Source window handle for token data
    ncclWindow_t prob_window; // Source window handle for probability data
    ncclWindow_t dest_window; // Destination window handle
    int num_gin_comms; // Number of GIN communicators (1)
    int num_ctx_per_comm; // Number of contexts per communicator
    void* gin_base_ptr; // Base pointer for offset calculations
    unsigned signals_base; // Base signal ID
    unsigned combine_signal_offset; // Signal offset for combine operations
    // qp info and mr info
    struct combine_memory_region_info_t mr_info;
    // Cross-round WAR sync-guards: LSA (intra-node staging) uses the NCCL LSA barrier; RDMA
    // (inter-node staging) is hand-rolled. Only the enable flags are needed on the device now.
    bool guard_enabled; // cross-round WAR guard (LSA + RDMA share one enable)
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    combine_warp_timing_entry_t* warp_timing;
    combine_block_timing_entry_t* block_timing;
#endif
};

// Data structure for JIT combine kernel parameters.
template <int LSA_TEAM_SIZE>
struct combine_kernel_param_t : combine_kernel_param_base_t {
    // Input buffers. These buffers are both local and remote buffers.
    // Keep embedded arrays here to avoid device-side pointer-table indirection.
    uint16_t* expert_input_token[LSA_TEAM_SIZE];
    float* expert_input_prob[LSA_TEAM_SIZE];
};

// Each CUDA block has sixteen named barriers numbered 0..15.
// __syncthreads(); will use the 0 named barriers, so we want to avoid that.
// We want to use 1 for intra-node reduction warp group, >= 2 for inter-node reduction warp group,
// RDMA warp group currently only contains 1 warp so does not use named bar yet, if it need to use, it should use 2 + NUM_OF_DATA_PIPELINE_PER_BLOCK.
__forceinline__ __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id = 0) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

// Helper to compute communicator index and context index from global channel
// Used for 6-comm x 4-ctx GIN configuration (6 communicators with 4 contexts each = 24 total channels)
__forceinline__ __device__ void get_comm_ctx(int global_channel, int num_ctx_per_comm, int& comm_idx, int& ctx_idx) {
    comm_idx = global_channel / num_ctx_per_comm;
    ctx_idx = global_channel % num_ctx_per_comm;
}

// Advance a ring-buffer slot; on wrap (slot == num_slots) reset to 0 and flip phase parity.
// Shared by producer/consumer FIFO traversals.
template <typename SlotT>
__forceinline__ __device__ void ring_advance(SlotT& slot, uint32_t& parity, int num_slots) {
    if (++slot == static_cast<SlotT>(num_slots)) {
        slot = 0;
        parity ^= 1;
    }
}

// Spin until an mbarrier reaches the expected phase parity.
__forceinline__ __device__ void mbarrier_wait(uint64_t* mbar, uint32_t parity) {
    while (!cuda::ptx::mbarrier_try_wait_parity(mbar, parity)) {
    }
}

// Put one token's bundle (token, +prob if FWD, +sf if quantized) to a remote node, packed from dst_offset.
template <typename TOKEN_DATA_TYPE, bool FORWARD_DISPATCH, bool HAS_SF, int NUM_LSA_TEAMS>
__forceinline__ __device__ void dispatch_n2n_put_token(
    ncclGin& net,
    const ncclTeam& rail,
    int remote_node_id,
    ncclWindow_t internal_window,
    size_t dst_offset,
    const struct dispatch_memory_region_info_t* mr_info,
    ncclWindow_t token_window,
    ncclWindow_t prob_window,
    ncclWindow_t sf_window,
    int token_idx,
    size_t token_bytes,
    size_t prob_bytes,
    int sf_bytes_per_token) {
    size_t token_src = mr_info->attn_input_token_offset + token_idx * token_bytes;
    net.put(
        rail,
        remote_node_id,
        internal_window,
        dst_offset,
        token_window,
        token_src,
        token_bytes,
        ncclGin_None{},
        ncclGin_None{},
        ncclCoopThread(),
        ncclGin_None{},
        cuda::thread_scope_thread,
        cuda::thread_scope_device,
        ncclGinOptFlagsAggregateRequests);
    if constexpr (FORWARD_DISPATCH) {
        size_t prob_src = mr_info->attn_input_prob_offset + (token_idx * NUM_LSA_TEAMS + remote_node_id) * prob_bytes;
        net.put(
            rail,
            remote_node_id,
            internal_window,
            dst_offset + token_bytes,
            prob_window,
            prob_src,
            prob_bytes,
            ncclGin_None{},
            ncclGin_None{},
            ncclCoopThread(),
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
    }
    if constexpr (HAS_SF) {
        size_t sf_src = mr_info->attn_input_scaling_factor_offset + token_idx * sf_bytes_per_token;
        net.put(
            rail,
            remote_node_id,
            internal_window,
            dst_offset + token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0),
            sf_window,
            sf_src,
            sf_bytes_per_token,
            ncclGin_None{},
            ncclGin_None{},
            ncclCoopThread(),
            ncclGin_None{},
            cuda::thread_scope_thread,
            cuda::thread_scope_device,
            ncclGinOptFlagsAggregateRequests);
    }
}

// Resolved G2S source for one (node, chunk): packed-remote entry base, or strided-local base pointers.
template <typename TOKEN_DATA_TYPE>
struct g2s_source_t {
    bool use_packed;
    const uint8_t* packed_base;
    const TOKEN_DATA_TYPE* token_base;
    const float* prob_base;
    const uint8_t* sf_base;
};

// Resolve a (node, chunk) source: packed-remote base (after waiting the RDMA arrival signal) or strided-local bases.
template <
    typename TOKEN_DATA_TYPE,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_OF_BLOCKS,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ g2s_source_t<TOKEN_DATA_TYPE> dispatch_g2s_resolve_source(
    const TOKEN_DATA_TYPE* attn_input_token,
    const float* attn_input_prob,
    const uint8_t* attn_input_token_scaling_factor,
    const int node_id,
    const int node_rank,
    const int local_rank,
    const int chunk_idx,
    const uint64_t expected_flag_value,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    const struct dispatch_memory_region_info_t* mr_info) {
    g2s_source_t<TOKEN_DATA_TYPE> src;
    src.use_packed = false;
    src.packed_base = nullptr;
    src.token_base = nullptr;
    src.prob_base = nullptr;
    src.sf_base = nullptr;

    if (node_id != node_rank) {
        // Remote: wait for the RDMA arrival signal, then point at this tile+chunk in the packed buffer.
        constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
        unsigned tail_signal_id = dispatch_tail_signal_id(
            mr_info->signals_tail_base,
            node_id,
            node_rank,
            local_rank,
            chunk_idx,
            NUM_LSA_TEAMS,
            LSA_TEAM_SIZE,
            MAX_CHUNKS_PER_RANK);
        constexpr int N2N_WARPS = (NUM_LSA_TEAMS == 1) ? 1 : HYBRIDEP_DISPATCH_N2N_WARPS;
        int signal_channel = chunk_idx % (NUM_OF_BLOCKS * N2N_WARPS);

        int ctx_idx = signal_channel % num_ctx_per_comm;
        ncclGin net(dcomm, ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
        net.waitSignal(ncclCoopThread(), tail_signal_id, expected_flag_value);

        const int remote_only_node_id = node_id > node_rank ? node_id - 1 : node_id;
        const int chunk_first_token = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;

        src.use_packed = true;
        src.packed_base = reinterpret_cast<const uint8_t*>(gin_base_ptr) +
                          dispatch_packed_entry_offset(mr_info, remote_only_node_id, chunk_first_token);
    } else {
        // Local: strided bases into this rank's own global mem arrays for this chunk.
        int chunk_first_token = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
        src.token_base = attn_input_token + chunk_first_token * HIDDEN_DIM;
        if constexpr (FORWARD_DISPATCH) {
            // attn_input_prob is laid out per token as NUM_LSA_TEAMS blocks of experts_per_node floats.
            const int experts_per_node = experts_per_rank * LSA_TEAM_SIZE;
            const int prob_row_stride = experts_per_node * NUM_LSA_TEAMS;
            src.prob_base = attn_input_prob + chunk_first_token * prob_row_stride;
        }
        if constexpr (HAS_SF) {
            src.sf_base = attn_input_token_scaling_factor + chunk_first_token * sf_bytes_per_token;
        }
    }
    return src;
}

enum class copy_dir {
    to_smem,
    to_gmem
}; // load into SMEM / store to gmem

// One field copy: to_smem loads (gmem->smem, mbar form), to_gmem stores (smem->gmem). Returns bytes copied.
template <copy_dir DIR>
__forceinline__ __device__ uint32_t bulk_copy(void* smem_ptr, const void* gmem_ptr, uint32_t bytes, uint64_t* mbar) {
    if constexpr (DIR == copy_dir::to_smem) {
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            /*dst=*/smem_ptr,
            /*src=*/gmem_ptr,
            bytes,
            mbar);
    } else {
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_global,
            cuda::ptx::space_shared,
            /*dst=*/const_cast<void*>(gmem_ptr),
            /*src=*/smem_ptr,
            bytes);
    }
    return bytes;
}

// Copy a token bundle (token, +prob if FWD, +sf if quantized) between this stage's SMEM and gmem. Returns total bytes.
template <typename TOKEN_DATA_TYPE, typename SMEM_TYPE, bool FORWARD_DISPATCH, bool HAS_SF, copy_dir DIR>
__forceinline__ __device__ uint32_t copy_token_bundle(
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const int stage,
    const void* token_gmem_ptr,
    const void* prob_gmem_ptr,
    const void* sf_gmem_ptr,
    const uint32_t token_bytes,
    const uint32_t prob_bytes,
    const uint32_t sf_bytes,
    uint64_t* mbar) {
    uint32_t tx =
        bulk_copy<DIR>(smem_buffer_ptr->get_token_buffer(pipeline_rank, stage), token_gmem_ptr, token_bytes, mbar);
    if constexpr (FORWARD_DISPATCH) {
        tx += bulk_copy<DIR>(
            reinterpret_cast<void*>(smem_buffer_ptr->get_prob_buffer(pipeline_rank, stage)),
            prob_gmem_ptr,
            prob_bytes,
            mbar);
    }
    if constexpr (HAS_SF) {
        tx += bulk_copy<DIR>(
            reinterpret_cast<void*>(smem_buffer_ptr->get_sf_buffer(pipeline_rank, stage)),
            sf_gmem_ptr,
            sf_bytes,
            mbar);
    }
    return tx;
}

// TMA-copy one token (+prob if FWD, +sf if quantized) from a packed-remote or strided-local source into its SMEM stage, then publish.
template <
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ void dispatch_g2s_issue_token(
    const g2s_source_t<TOKEN_DATA_TYPE>& src,
    const int current_token_id,
    const int packed_dense_idx,
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const int stage,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const int node_rank,
    const struct dispatch_memory_region_info_t* mr_info) {
    uint64_t* mbar = smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage);
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SIZE) * sizeof(float));
    const uint32_t sf_bytes = (uint32_t)sf_bytes_per_token;
    uint32_t tx_bytes;

    if (src.use_packed) {
        // Packed entry is contiguous [token | prob | sf].
        const uint8_t* packed_src_base = src.packed_base + packed_dense_idx * mr_info->bytes_per_entry;
        const void* token_src = packed_src_base;
        const void* prob_src = packed_src_base + token_bytes;
        const void* sf_src = packed_src_base + token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0);
        tx_bytes = copy_token_bundle<TOKEN_DATA_TYPE, SMEM_TYPE, FORWARD_DISPATCH, HAS_SF, copy_dir::to_smem>(
            smem_buffer_ptr,
            pipeline_rank,
            stage,
            token_src,
            prob_src,
            sf_src,
            token_bytes,
            prob_bytes,
            sf_bytes,
            mbar);
    } else {
        // Strided-local: each field has its own base.
        const void* token_src = src.token_base + (current_token_id * HIDDEN_DIM);
        const void* prob_src = nullptr;
        const void* sf_src = nullptr;
        if constexpr (FORWARD_DISPATCH) {
            // Advance by whole token rows, then pick this node's expert slice within the row.
            const int experts_per_node = experts_per_rank * LSA_TEAM_SIZE;
            const int prob_row_stride = experts_per_node * NUM_LSA_TEAMS;
            prob_src = src.prob_base + current_token_id * prob_row_stride + node_rank * experts_per_node;
        }
        if constexpr (HAS_SF) {
            sf_src = src.sf_base + current_token_id * sf_bytes_per_token;
        }
        tx_bytes = copy_token_bundle<TOKEN_DATA_TYPE, SMEM_TYPE, FORWARD_DISPATCH, HAS_SF, copy_dir::to_smem>(
            smem_buffer_ptr,
            pipeline_rank,
            stage,
            token_src,
            prob_src,
            sf_src,
            token_bytes,
            prob_bytes,
            sf_bytes,
            mbar);
    }

    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release,
        cuda::ptx::scope_cta,
        cuda::ptx::space_shared,
        mbar,
        tx_bytes);
}

// One destination decoded from a single s2d-map entry.
struct s2g_dest_t {
    bool issue; // false for an empty entry or an EM secondary duplicate
    int remote_rank_id;
    int output_buffer_index;
};

// TMA-load one (node, chunk)'s s2d-map slice into the SMEM stage and publish. Caller elects one lane.
template <typename SMEM_TYPE, int NUM_OF_TOKENS_PER_CHUNK>
__forceinline__ __device__ void dispatch_s2g_prefetch_s2d_map(
    const int32_t* sparse_to_dense_map,
    SMEM_TYPE* smem_buffer_ptr,
    const int pipeline_rank,
    const uint32_t s2d_map_stage,
    const int node_id,
    const int chunk_id,
    const int chunk_size,
    const int num_of_tokens_per_rank,
    const int s2d_inner_dim) {
    const int32_t* s2d_base =
        sparse_to_dense_map + (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * s2d_inner_dim;
    void* smem_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_s2d_map_buffer_base(pipeline_rank, s2d_map_stage));
    uint64_t* mbar = smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, s2d_map_stage);
    // cp.async.bulk needs a 16B-multiple size: round up. Safe because the source S2D buffer
    // is over-allocated for max_tokens_per_rank and the smem dest stage is padded to 128B.
    uint32_t copy_bytes = (uint32_t)(chunk_size * s2d_inner_dim * sizeof(int32_t));
    copy_bytes = (copy_bytes + 15u) & ~15u;
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        smem_dst,
        reinterpret_cast<const void*>(s2d_base),
        copy_bytes,
        mbar);
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release,
        cuda::ptx::scope_cta,
        cuda::ptx::space_shared,
        mbar,
        copy_bytes);
}

// Decode one s2d-map entry into its remote destination. `issue` is false for an empty entry (-1)
// or an EM secondary duplicate (the receiver's local_dup kernel fills it from the primary slot).
template <ncclEpLayout_t kLayout>
__forceinline__ __device__ s2g_dest_t
dispatch_s2g_resolve_dest(const int32_t* s2d_row, const int flat_idx, const bool local_dup_enabled) {
    s2g_dest_t dst;
    dst.issue = false;
    dst.remote_rank_id = -1;
    dst.output_buffer_index = -1;

    const int32_t entry_val = s2d_row[flat_idx];
    if (entry_val == -1) {
        return dst;
    }

    // Rank-major: entry_idx=rank, value=slot. Expert-major: value packs (rank,slot).
    if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
        dst.remote_rank_id = em_s2d_unpack_rank(entry_val);
        dst.output_buffer_index = em_s2d_unpack_slot(entry_val);
        if (local_dup_enabled && flat_idx > 0) {
            const int32_t prev_val = s2d_row[flat_idx - 1];
            if (prev_val != -1 && em_s2d_unpack_rank(prev_val) == dst.remote_rank_id) {
                return dst; // secondary dup: issue stays false
            }
        }
    } else {
        dst.remote_rank_id = flat_idx;
        dst.output_buffer_index = entry_val;
    }
    dst.issue = true;
    return dst;
}

// TMA-store one token (+prob if FWD, +sf if quantized) from this stage's SMEM to one resolved remote destination.
template <typename TOKEN_DATA_TYPE, typename SMEM_TYPE, int LSA_TEAM_SIZE, bool FORWARD_DISPATCH, bool HAS_SF>
__forceinline__ __device__ void dispatch_s2g_issue_token(
    const s2g_dest_t& dst,
    SMEM_TYPE* smem_buffer_ptr,
    TOKEN_DATA_TYPE* const* remote_expert_output_token,
    float* const* remote_expert_output_prob,
    uint8_t* const* remote_expert_output_scaling_factor,
    const int pipeline_rank,
    const int stage,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank) {
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * LSA_TEAM_SIZE) * sizeof(float));
    const uint32_t sf_bytes = (uint32_t)sf_bytes_per_token;

    // Remote fan-out: each field has its own output array, indexed [rank]+slot*stride.
    const void* token_dst = remote_expert_output_token[dst.remote_rank_id] + (dst.output_buffer_index * HIDDEN_DIM);
    const void* prob_dst = nullptr;
    const void* sf_dst = nullptr;
    if constexpr (FORWARD_DISPATCH) {
        prob_dst = remote_expert_output_prob[dst.remote_rank_id] +
                   (dst.output_buffer_index * (experts_per_rank * LSA_TEAM_SIZE));
    }
    if constexpr (HAS_SF) {
        sf_dst = remote_expert_output_scaling_factor[dst.remote_rank_id] + dst.output_buffer_index * sf_bytes_per_token;
    }
    copy_token_bundle<TOKEN_DATA_TYPE, SMEM_TYPE, FORWARD_DISPATCH, HAS_SF, copy_dir::to_gmem>(
        smem_buffer_ptr,
        pipeline_rank,
        stage,
        token_dst,
        prob_dst,
        sf_dst,
        token_bytes,
        prob_bytes,
        sf_bytes,
        /*mbar=*/nullptr);
}

// Device function for inter-node node2node(RDMA) warp for dispatch kernel.
template <
    typename INTER_NODE_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_OF_STAGES,
    int NUM_OF_TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    int NUM_OF_BLOCKS,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ void dispatch_N2N_warp(
    // INPUT
    const bool* attn_to_rdma_map,
    // CONFIG
    const int local_rank,
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    ncclWindow_t nccl_token_window,
    ncclWindow_t nccl_prob_window,
    ncclWindow_t nccl_sf_window,
    ncclWindow_t nccl_internal_window,
    const struct dispatch_memory_region_info_t* mr_info,
    SMEM_TYPE* smem_buffer_ptr) {
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, NUM_OF_TOKENS_PER_CHUNK);

    static_assert(INTER_NODE_GROUP::size() >= NUM_LSA_TEAMS - 1, "mr_info should be loaded at once.");
    static_assert(NUM_OF_TOKENS_PER_CHUNK % 32 == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
    static_assert(
        MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0,
        "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");

    // Load mr_info into shared memory for faster access in Put calls.
    int lane_id = INTER_NODE_GROUP::thread_rank() % 32;
    struct dispatch_memory_region_info_t* smem_mr_info_ptr = nullptr;
    if constexpr (NUM_LSA_TEAMS != 1) {
        smem_mr_info_ptr = smem_buffer_ptr->dispatch_memory_region_info;
        if (lane_id == 0) {
            smem_mr_info_ptr[0] = mr_info[0];
        }
        __syncwarp();
    }

    constexpr int N2N_WARPS = INTER_NODE_GROUP::size() / 32;
    int n2n_warp_id = INTER_NODE_GROUP::thread_rank() / 32;
    size_t token_bytes = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
    size_t prob_bytes = (experts_per_rank * LSA_TEAM_SIZE) * sizeof(float);
    constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
    constexpr int NUM_REMOTE_NODES = NUM_LSA_TEAMS - 1;

    // GIN device side setup. Single communicator; ctx_idx spreads QP traffic.
    int global_channel = blockIdx.x * N2N_WARPS + n2n_warp_id;
    int ctx_idx = global_channel % num_ctx_per_comm;

    ncclGin net(dcomm, ctx_idx, NCCL_GIN_RESOURCE_SHARING_CTA);
    ncclTeam rail = ncclTeamRail(dcomm);

    for (int chunk_idx = blockIdx.x * N2N_WARPS + n2n_warp_id; chunk_idx < MAX_CHUNKS_PER_RANK;
         chunk_idx += NUM_OF_BLOCKS * N2N_WARPS) {
        int chunk_first_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
        int current_chunk_size = 0;
        if (chunk_idx < num_of_chunks_per_rank) {
            current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
            if (chunk_first_token_idx + current_chunk_size > num_of_tokens_per_rank) {
                current_chunk_size = num_of_tokens_per_rank - chunk_first_token_idx;
            }
        }

        for (int j = 0; j < NUM_REMOTE_NODES; ++j) {
            // Skip-self ring over the other nodes, load-balanced start at node_rank.
            int remote_idx = (j + node_rank) % NUM_REMOTE_NODES;
            int remote_node_id = remote_idx < node_rank ? remote_idx : remote_idx + 1;
            int remote_only_node_id = remote_idx < node_rank ? node_rank - 1 : node_rank;

            size_t dense_dst_offset =
                dispatch_packed_entry_offset(smem_mr_info_ptr, remote_only_node_id, chunk_first_token_idx);
            const size_t entry_bytes = smem_mr_info_ptr->bytes_per_entry;

            // Create a bitmask of tokens that need to be written. One word per 32 tokens.
            int32_t need_write_bitmask[NUM_OF_TOKENS_PER_CHUNK / 32];
            for (int i = 0; i < NUM_OF_TOKENS_PER_CHUNK / 32; i++) {
                int token_idx_in_chunk = i * 32 + ncclCoopWarp().thread_rank();
                bool need_write =
                    token_idx_in_chunk < current_chunk_size &&
                    attn_to_rdma_map[((token_idx_in_chunk + chunk_first_token_idx) * NUM_REMOTE_NODES) + remote_idx];
                need_write_bitmask[i] = __ballot_sync(~0u, need_write);
            }

            for (int token_idx_in_chunk = ncclCoopWarp().thread_rank(); token_idx_in_chunk < current_chunk_size;
                 token_idx_in_chunk += ncclCoopWarp().size()) {
                const int word = token_idx_in_chunk / 32;
                const int lane_in_word = token_idx_in_chunk % 32;
                size_t dst_offset = dense_dst_offset;

                // Compact: skip space for earlier tokens in this word that aren't written.
                if (lane_in_word > 0) {
                    uint32_t writes_before = need_write_bitmask[word] & ((1u << lane_in_word) - 1);
                    dst_offset += __popc(writes_before) * entry_bytes;
                }

                bool need_write =
                    attn_to_rdma_map[((chunk_first_token_idx + token_idx_in_chunk) * NUM_REMOTE_NODES) + remote_idx];
                if (need_write) {
                    int token_idx = chunk_first_token_idx + token_idx_in_chunk;
                    dispatch_n2n_put_token<TOKEN_DATA_TYPE, FORWARD_DISPATCH, HAS_SF, NUM_LSA_TEAMS>(
                        net,
                        rail,
                        remote_node_id,
                        nccl_internal_window,
                        dst_offset,
                        smem_mr_info_ptr,
                        nccl_token_window,
                        nccl_prob_window,
                        nccl_sf_window,
                        token_idx,
                        token_bytes,
                        prob_bytes,
                        sf_bytes_per_token);
                }
                // Advance the compacted base past all written tokens in this word.
                dense_dst_offset += __popc(need_write_bitmask[word]) * entry_bytes;
            }

            // Signal chunk completion on the SAME put comm: same-QP ordering makes all
            // preceding puts visible at the remote before this signal arrives.
            unsigned tail_signal_id = dispatch_tail_signal_id(
                smem_mr_info_ptr->signals_tail_base,
                node_rank,
                remote_node_id,
                local_rank,
                chunk_idx,
                NUM_LSA_TEAMS,
                LSA_TEAM_SIZE,
                MAX_CHUNKS_PER_RANK);
            net.signal(
                rail,
                remote_node_id,
                ncclGin_SignalAdd{tail_signal_id, 1},
                ncclCoopWarp(),
                ncclGin_None{},
                cuda::thread_scope_thread,
                cuda::thread_scope_thread,
                ncclGinOptFlagsDefault);
        }
    }
    // GIN flush with coopWarp includes syncwarp at the end
    net.flush(ncclCoopWarp(), cuda::memory_order_acquire);
}

// Dispatch intra-node S2G warp group. With NUM_PIPELINES > 1, each warp is an
// independent pipeline consumer paired with the G2S warp of the same pipeline_rank.
template <
    typename INTRA_NODE_S2G_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_OF_STAGES,
    int NUM_OF_IN_FLIGHT_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    int NUM_OF_BLOCKS,
    int NUM_PIPELINES,
    bool FORWARD_DISPATCH,
    bool HAS_SF,
    ncclEpLayout_t kLayout>
__forceinline__ __device__ void dispatch_S2G_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const int32_t* sparse_to_dense_map,
    // OUTPUT
    TOKEN_DATA_TYPE* const* remote_expert_output_token,
    float* const* remote_expert_output_prob,
    uint8_t* const* remote_expert_output_scaling_factor,
    // CONFIG
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const bool local_dup_enabled,
    SMEM_TYPE* smem_buffer_ptr) {
    constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;
    static_assert(
        NUM_OF_IN_FLIGHT_S2G < STAGES_PER_PIPELINE,
        "NUM_OF_IN_FLIGHT_S2G must be smaller than STAGES_PER_PIPELINE.");
    using routing_loads_t = uint4;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % sizeof(routing_loads_t) == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of routing_loads_t.");
    constexpr int TOKENS_PER_ROUTING_LOAD = sizeof(routing_loads_t) / sizeof(bool);
    constexpr int ROUTING_LOADS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / TOKENS_PER_ROUTING_LOAD;

    // S2D inner dim: mode-dependent, carried by SMEM layout struct.
    const int s2d_inner_dim = smem_buffer_ptr->s2d_inner_dim;

    const int pipeline_rank = INTRA_NODE_S2G_GROUP::warp_rank();
    const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, NUM_OF_TOKENS_PER_CHUNK);
    // TOKENS_PER_ROUTING_LOAD must match the producer's pad in scan_kernel.cuh
    const int routing_map_node_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
    int in_flight_s2g = 0;
    int stage = 0;
    uint32_t producer_parity = 0;
    uint32_t s2d_stage = 0;
    uint32_t s2d_parity = 0;

    // S2G on all 32 lanes (warp-uniform state); cp_async_bulk striped by lane=flat_idx (up to s2d_inner_dim stores/token).
    const int s2g_lane = INTRA_NODE_S2G_GROUP::thread_rank() % 32;

    // Each pipeline prefetches its own first s2d map for its first chunk (single TMA load, lane 0 only).
    if (s2g_lane == 0) {
        int chunk_iter = 0;
        for (int chunk_idx = blockIdx.x; chunk_idx < num_of_chunks_per_rank; chunk_idx += NUM_OF_BLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) == pipeline_rank) {
                int current_chunk_size;
                if (remainder_chunk_size != 0 && chunk_idx == num_of_chunks_per_rank - 1) {
                    current_chunk_size = remainder_chunk_size;
                } else {
                    current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
                }
                dispatch_s2g_prefetch_s2d_map<SMEM_TYPE, NUM_OF_TOKENS_PER_CHUNK>(
                    sparse_to_dense_map,
                    smem_buffer_ptr,
                    pipeline_rank,
                    s2d_stage,
                    node_rank,
                    chunk_idx,
                    current_chunk_size,
                    num_of_tokens_per_rank,
                    s2d_inner_dim);
                break;
            }
        }
    }
    __syncwarp();

    {
        int chunk_iter = 0;
        for (int chunk_idx = blockIdx.x; chunk_idx < num_of_chunks_per_rank; chunk_idx += NUM_OF_BLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) continue;

            int routing_loads_in_chunk;
            int current_chunk_size;
            if (remainder_chunk_size != 0 && chunk_idx == num_of_chunks_per_rank - 1) {
                routing_loads_in_chunk = nccl_ep::ceil_div(remainder_chunk_size, (int)sizeof(routing_loads_t));
                current_chunk_size = remainder_chunk_size;
            } else {
                routing_loads_in_chunk = ROUTING_LOADS_PER_CHUNK;
                current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
            }
            for (int j = 0; j < NUM_LSA_TEAMS; j++) {
                // Per-pipeline self-sync (arrival count = 1, trivially satisfied); lane 0 only.
                if (s2g_lane == 0) {
                    uint64_t state_token =
                        cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_S2G_group_mbar(pipeline_rank));
                    while (!cuda::ptx::mbarrier_try_wait(
                        smem_buffer_ptr->get_S2G_group_mbar(pipeline_rank),
                        state_token)) {
                    }
                }
                __syncwarp();

                // Prefetch next (chunk, node) s2d map for THIS pipeline (single TMA load, lane 0 only).
                if (s2g_lane == 0) {
                    int next_chunk_id;
                    int next_node_id;
                    int next_node_iter = j + 1;
                    if (next_node_iter < NUM_LSA_TEAMS) {
                        next_chunk_id = chunk_idx;
                        next_node_id = (node_rank + NUM_LSA_TEAMS - next_node_iter) % NUM_LSA_TEAMS;
                    } else {
                        // Find the next chunk this pipeline will process
                        int future_chunk_iter = chunk_iter; // chunk_iter was already incremented for current chunk
                        next_chunk_id = -1;
                        for (int fi = chunk_idx + NUM_OF_BLOCKS; fi < num_of_chunks_per_rank; fi += NUM_OF_BLOCKS) {
                            if ((future_chunk_iter++ % NUM_PIPELINES) == pipeline_rank) {
                                next_chunk_id = fi;
                                break;
                            }
                        }
                        next_node_id = node_rank;
                    }

                    if (next_chunk_id >= 0 && next_chunk_id < num_of_chunks_per_rank) {
                        int next_chunk_size;
                        if (remainder_chunk_size != 0 && next_chunk_id == num_of_chunks_per_rank - 1) {
                            next_chunk_size = remainder_chunk_size;
                        } else {
                            next_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
                        }
                        dispatch_s2g_prefetch_s2d_map<SMEM_TYPE, NUM_OF_TOKENS_PER_CHUNK>(
                            sparse_to_dense_map,
                            smem_buffer_ptr,
                            pipeline_rank,
                            s2d_stage ^ 1,
                            next_node_id,
                            next_chunk_id,
                            next_chunk_size,
                            num_of_tokens_per_rank,
                            s2d_inner_dim);
                    }
                }

                // Walk nodes backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int node_id = (node_rank + NUM_LSA_TEAMS - j) % NUM_LSA_TEAMS;
                const routing_loads_t* routing_map_ptr = reinterpret_cast<const routing_loads_t*>(
                    rdma_to_attn_map + (node_id * routing_map_node_stride + chunk_idx * NUM_OF_TOKENS_PER_CHUNK));

                {
                    uint64_t* wait_mbar = smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, s2d_stage);
                    mbarrier_wait(wait_mbar, s2d_parity);
                }

                for (int load_idx = 0; load_idx < routing_loads_in_chunk; load_idx++) {
                    routing_loads_t routing_flags = routing_map_ptr[load_idx];
#pragma unroll
                    for (int token_in_load = 0; token_in_load < TOKENS_PER_ROUTING_LOAD; token_in_load++) {
                        int current_token_id = load_idx * TOKENS_PER_ROUTING_LOAD + token_in_load;
                        if (current_token_id >= current_chunk_size) {
                            break;
                        }
                        bool token_needed = *(reinterpret_cast<bool*>(&routing_flags) + token_in_load);
                        if (token_needed) {
                            const int32_t* s2d_smem_row =
                                smem_buffer_ptr->get_s2d_map_buffer(pipeline_rank, s2d_stage, current_token_id);
                            mbarrier_wait(
                                smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage),
                                producer_parity);

                            // Per-entry parallel issue (lane handles flat_idx=lane,lane+32,...); empty/EM-dup entries resolve to issue=false.
                            for (int flat_idx = s2g_lane; flat_idx < s2d_inner_dim; flat_idx += 32) {
                                s2g_dest_t dst =
                                    dispatch_s2g_resolve_dest<kLayout>(s2d_smem_row, flat_idx, local_dup_enabled);
                                if (dst.issue) {
                                    dispatch_s2g_issue_token<
                                        TOKEN_DATA_TYPE,
                                        SMEM_TYPE,
                                        LSA_TEAM_SIZE,
                                        FORWARD_DISPATCH,
                                        HAS_SF>(
                                        dst,
                                        smem_buffer_ptr,
                                        remote_expert_output_token,
                                        remote_expert_output_prob,
                                        remote_expert_output_scaling_factor,
                                        pipeline_rank,
                                        stage,
                                        HIDDEN_DIM,
                                        sf_bytes_per_token,
                                        experts_per_rank);
                                }
                            }
                            // S1: only issuing lanes commit/wait — idle lanes skip the empty pair.
                            if (s2g_lane < s2d_inner_dim) {
                                cuda::ptx::cp_async_bulk_commit_group();
                            }
                            in_flight_s2g += 1;
                            if (in_flight_s2g > NUM_OF_IN_FLIGHT_S2G) {
                                if (s2g_lane < s2d_inner_dim) {
                                    cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_IN_FLIGHT_S2G>{});
                                }
                                __syncwarp();
                                in_flight_s2g -= 1;
                                int notify_stage = (stage - NUM_OF_IN_FLIGHT_S2G) >= 0 ?
                                                       (stage - NUM_OF_IN_FLIGHT_S2G) :
                                                       (stage - NUM_OF_IN_FLIGHT_S2G + STAGES_PER_PIPELINE);
                                if (s2g_lane == 0) {
                                    cuda::ptx::mbarrier_arrive(
                                        smem_buffer_ptr->get_intra_node_mbarrier_consumer(pipeline_rank, notify_stage));
                                }
                            }

                            ring_advance(stage, producer_parity, STAGES_PER_PIPELINE);
                        }
                    }
                }
                ring_advance(s2d_stage, s2d_parity, S2D_MAP_RING_STAGES);
            }
        }
        // Drain in-flight TMA S2G writes before returning (each lane drains its own commit groups).
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        // Drain TMA stores into the generic memory path so the dispatch-tail barrier sees them.
        nccl_ep::fence_proxy_async();
    }
}

// Dispatch intra-node G2S warp group. With NUM_PIPELINES > 1, each warp is an independent
// pipeline processing disjoint chunks through its own partition of the shared-memory FIFO.
template <
    typename INTRA_NODE_G2S_GROUP,
    typename TOKEN_DATA_TYPE,
    typename SMEM_TYPE,
    int NUM_OF_STAGES,
    int NUM_OF_TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    int NUM_OF_BLOCKS,
    int NUM_PIPELINES,
    bool FORWARD_DISPATCH,
    bool HAS_SF>
__forceinline__ __device__ void dispatch_G2S_warp(
    // INPUT
    const bool* rdma_to_attn_map,
    const TOKEN_DATA_TYPE* attn_input_token,
    const float* attn_input_prob,
    const uint8_t* attn_input_token_scaling_factor,
    // OUTPUT
    uint64_t* rdma_inter_node_group_flags,
    // CONFIG
    const int local_rank,
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int HIDDEN_DIM,
    const int sf_bytes_per_token,
    const int experts_per_rank,
    const uint64_t expected_flag_value,
    const ncclDevComm& dcomm,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    const struct dispatch_memory_region_info_t* mr_info,
    SMEM_TYPE* smem_buffer_ptr) {
    using routing_loads_t = uint4;

    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % sizeof(routing_loads_t) == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of routing_loads_t.");
    static_assert(
        MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0,
        "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");

    constexpr int TOKENS_PER_ROUTING_LOAD = sizeof(routing_loads_t) / sizeof(bool);
    constexpr int ROUTING_LOADS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / TOKENS_PER_ROUTING_LOAD;
    constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;

    const int pipeline_rank = INTRA_NODE_G2S_GROUP::warp_rank();
    const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
    const int num_of_chunks_per_rank = nccl_ep::ceil_div(num_of_tokens_per_rank, NUM_OF_TOKENS_PER_CHUNK);
    const int max_num_of_chunks_per_rank = nccl_ep::ceil_div(MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_TOKENS_PER_CHUNK);
    // TOKENS_PER_ROUTING_LOAD must match the producer's pad in scan_kernel.cuh
    const int routing_map_node_stride = nccl_ep::align(num_of_tokens_per_rank, TOKENS_PER_ROUTING_LOAD);
    int stage = 0;
    uint32_t consumer_parity = 1;
    int tokens_produced = 0;

    if (cuda::ptx::elect_sync(~0)) {
        int chunk_iter = 0;
        for (int chunk_idx = blockIdx.x; chunk_idx < num_of_chunks_per_rank; chunk_idx += NUM_OF_BLOCKS) {
            if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) continue;

            int routing_loads_in_chunk;
            int current_chunk_size;
            if (remainder_chunk_size != 0 && chunk_idx == num_of_chunks_per_rank - 1) {
                routing_loads_in_chunk = nccl_ep::ceil_div(remainder_chunk_size, (int)sizeof(routing_loads_t));
                current_chunk_size = remainder_chunk_size;
            } else {
                routing_loads_in_chunk = ROUTING_LOADS_PER_CHUNK;
                current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
            }

            for (int j = 0; j < NUM_LSA_TEAMS; j++) {
                // Walk nodes backward from self around the ring (j=0 -> self, j>=1 -> remote)
                int node_id = (node_rank + NUM_LSA_TEAMS - j) % NUM_LSA_TEAMS;

                g2s_source_t<TOKEN_DATA_TYPE> src = dispatch_g2s_resolve_source<
                    TOKEN_DATA_TYPE,
                    NUM_LSA_TEAMS,
                    LSA_TEAM_SIZE,
                    MAX_NUM_OF_TOKENS_PER_RANK,
                    NUM_OF_TOKENS_PER_CHUNK,
                    NUM_OF_BLOCKS,
                    FORWARD_DISPATCH,
                    HAS_SF>(
                    attn_input_token,
                    attn_input_prob,
                    attn_input_token_scaling_factor,
                    node_id,
                    node_rank,
                    local_rank,
                    chunk_idx,
                    expected_flag_value,
                    HIDDEN_DIM,
                    sf_bytes_per_token,
                    experts_per_rank,
                    dcomm,
                    num_ctx_per_comm,
                    gin_base_ptr,
                    mr_info);

                const routing_loads_t* routing_map_ptr = reinterpret_cast<const routing_loads_t*>(
                    rdma_to_attn_map + (node_id * routing_map_node_stride) + (chunk_idx * NUM_OF_TOKENS_PER_CHUNK));

                int packed_dense_idx = 0;
                for (int load_idx = 0; load_idx < routing_loads_in_chunk; load_idx++) {
                    routing_loads_t routing_flags = routing_map_ptr[load_idx];

#pragma unroll
                    for (int token_in_load = 0; token_in_load < TOKENS_PER_ROUTING_LOAD; token_in_load++) {
                        int current_token_id = load_idx * TOKENS_PER_ROUTING_LOAD + token_in_load;
                        if (current_token_id >= current_chunk_size) {
                            break;
                        }

                        bool token_needed = *(reinterpret_cast<bool*>(&routing_flags) + token_in_load);
                        if (token_needed) {
                            if (tokens_produced >= STAGES_PER_PIPELINE) {
                                uint64_t* mbar =
                                    smem_buffer_ptr->get_intra_node_mbarrier_consumer(pipeline_rank, stage);
                                mbarrier_wait(mbar, consumer_parity);
                            }

                            dispatch_g2s_issue_token<
                                TOKEN_DATA_TYPE,
                                SMEM_TYPE,
                                NUM_LSA_TEAMS,
                                LSA_TEAM_SIZE,
                                FORWARD_DISPATCH,
                                HAS_SF>(
                                src,
                                current_token_id,
                                packed_dense_idx,
                                smem_buffer_ptr,
                                pipeline_rank,
                                stage,
                                HIDDEN_DIM,
                                sf_bytes_per_token,
                                experts_per_rank,
                                node_rank,
                                mr_info);

                            if (src.use_packed) {
                                packed_dense_idx++;
                            }

                            tokens_produced += 1;
                            ring_advance(stage, consumer_parity, STAGES_PER_PIPELINE);
                        }
                    }
                }
            }
        }
    }
    // Update residue flags (only pipeline 0 does this to avoid duplicate writes).
    if (INTRA_NODE_G2S_GROUP::warp_rank() == 0) {
        int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;

        for (int node_id = blockIdx.x; node_id < NUM_LSA_TEAMS - 1; node_id += gridDim.x) {
            uint64_t* residue_flag_base_ptr =
                rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank) + num_of_chunks_per_rank;
            if (INTRA_NODE_G2S_GROUP::thread_rank() < residue_flag_count) {
                residue_flag_base_ptr[INTRA_NODE_G2S_GROUP::thread_rank()] = expected_flag_value;
            }
        }
    }
}

// Shared single-entry G2S issue for both the intra- and inter-node combine
// G2S warps. INTER_NODE selects the inter-node staged-buffer accessors and
// flag buffer; otherwise the intra-node ones (both sets live on the same
// combine smem layout, so one helper covers both tiers).
//   - derive stage_idx + parity from (global_offset + rank_in_batch)
//   - wait for consumer to free the stage (mbarrier_try_wait_parity)
//   - cp_async_bulk the token (and the prob under BACKWARD_COMBINE)
//   - optionally write <tier>_flag_G2S_buffer[stage_idx]
//   - mbarrier_arrive_expect_tx with the cumulative tx size
//
// The caller has already computed the token (and prob) source pointers
// and the `is_last_entry` boolean (only consulted when WRITE_LAST_FLAG).
template <bool INTER_NODE, bool BACKWARD_COMBINE, bool WRITE_LAST_FLAG, typename SMEM_TYPE>
__forceinline__ __device__ void issue_g2s_entry(
    SMEM_TYPE* smem_buffer_ptr,
    int global_offset,
    int rank_in_batch,
    int starting_G2S_index,
    int ring_len,
    const uint16_t* token_src,
    uint32_t token_bytes,
    const float* prob_src,
    uint32_t prob_bytes,
    bool is_last_entry) {
    const int my_abs_offset = global_offset + rank_in_batch;
    const int stage_idx = starting_G2S_index + (my_abs_offset % ring_len);
    const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

    uint64_t* consumer_mbar;
    uint64_t* producer_mbar;
    void* token_dst;
    if constexpr (INTER_NODE) {
        consumer_mbar = smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(stage_idx);
        producer_mbar = smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx);
        token_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(stage_idx));
    } else {
        consumer_mbar = smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(stage_idx);
        producer_mbar = smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(stage_idx);
        token_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_token_G2S(stage_idx));
    }

    while (!cuda::ptx::mbarrier_try_wait_parity(consumer_mbar, parity)) {
    }

    uint32_t total_tx_size = 0;
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared,
        cuda::ptx::space_global,
        token_dst,
        reinterpret_cast<const void*>(token_src),
        token_bytes,
        producer_mbar);
    total_tx_size += token_bytes;

    if constexpr (BACKWARD_COMBINE) {
        void* prob_dst;
        if constexpr (INTER_NODE) {
            prob_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(stage_idx));
        } else {
            prob_dst = reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_prob_G2S(stage_idx));
        }
        cuda::ptx::cp_async_bulk(
            cuda::ptx::space_shared,
            cuda::ptx::space_global,
            prob_dst,
            reinterpret_cast<const void*>(prob_src),
            prob_bytes,
            producer_mbar);
        total_tx_size += prob_bytes;
    }

    if constexpr (WRITE_LAST_FLAG) {
        if constexpr (INTER_NODE) {
            smem_buffer_ptr->inter_node_flag_G2S_buffer[stage_idx] = is_last_entry;
        } else {
            smem_buffer_ptr->intra_node_flag_G2S_buffer[stage_idx] = is_last_entry;
        }
    }

    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_release,
        cuda::ptx::scope_cta,
        cuda::ptx::space_shared,
        producer_mbar,
        total_tx_size);
}

// Warp-cooperative scan of one sparse_to_dense_map row plus its inline
// broadcast-issue, shared by the intra-node G2S warp and the LOCAL tier of
// the inter-node G2S warp (the two differ only in INTER_NODE, starting_G2S_index,
// and which s2d row / FIFO slice they target -- all passed in).
//
//  * Process s2d entries in WARP_SIZE steps
//  * Each lane loads an s2d entry and participates in valid entry filtering
//  * Next a single lane performs TMA loads of the token and probability data
//    * NOTE: Using single lane allows to ensure the ordering of TMA loads so
//      they are aligned with the RED process.
template <
    bool INTER_NODE,
    bool BACKWARD_COMBINE,
    ncclEpLayout_t kLayout,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype,
    typename SMEM_TYPE>
__forceinline__ __device__ void issue_local_g2s_row(
    SMEM_TYPE* smem_buffer_ptr,
    const int32_t* sparse_to_dense_row,
    int s2d_entries,
    int& global_offset,
    int starting_G2S_index,
    int ring_len,
    int lane_id,
    uint16_t* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    uint32_t token_bytes,
    uint32_t prob_bytes,
    int experts_per_rank,
    int num_of_ranks_per_node,
    bool combine_local_reduce_enabled) {
    constexpr int WARP_SIZE = 32;
    int total_valid_count = 0;
    int valid_seen = 0;
    bool have_pending = false;
    int32_t pending_s2d = -1;
    int pending_entry_idx = -1;

    // Resolve the s2d entry and issue TMA load
    auto issue_pending = [&](bool is_last_entry) {
        int rank_id;
        int slot;
        if constexpr (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) {
            rank_id = em_s2d_unpack_rank(pending_s2d);
            slot = em_s2d_unpack_slot(pending_s2d);
        } else {
            rank_id = pending_entry_idx;
            slot = pending_s2d;
        }
        const uint16_t* token_src =
            remote_expert_input_token[rank_id] + (slot * HIDDEN_DIM * nccl_ep::size_u16<kTokenDtype>());
        const float* prob_src = nullptr;
        if constexpr (BACKWARD_COMBINE) {
            prob_src = remote_expert_input_prob[rank_id] + (slot * (experts_per_rank * num_of_ranks_per_node));
        }
        issue_g2s_entry<INTER_NODE, BACKWARD_COMBINE, /*WRITE_LAST_FLAG=*/true>(
            smem_buffer_ptr,
            global_offset,
            valid_seen,
            starting_G2S_index,
            ring_len,
            token_src,
            token_bytes,
            prob_src,
            prob_bytes,
            is_last_entry);
    };

    for (int entry_base = 0; entry_base < s2d_entries; entry_base += WARP_SIZE) {
        const int entry_idx = entry_base + lane_id;
        const bool lane_active = (entry_idx < s2d_entries);
        const int32_t s2d_val = lane_active ? sparse_to_dense_row[entry_idx] : -1;
        const bool is_secondary = is_em_secondary_entry<kLayout>(s2d_val, lane_id, combine_local_reduce_enabled);

        const unsigned mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1 && !is_secondary);
        total_valid_count += __popc(mask);

        // All 32 lanes iterate set bits of this batch's mask in lockstep. The lane
        // at `src_lane` broadcasts its s2d_val to every lane via __shfl_sync; lane
        // 0 takes the value and issues the TMA. The issue itself is deferred by one
        // entry (kept in pending_*) so the final issue knows it's last.
        unsigned m = mask;
        while (m != 0) {
            const int src_lane = __ffs((int)m) - 1; // 0..31
            m &= (m - 1);
            const int32_t bcast_s2d = __shfl_sync(0xffffffff, s2d_val, src_lane);

            if (lane_id == 0) {
                if (have_pending) {
                    // Flush the previously-buffered entry; not last because we just
                    // discovered another to issue.
                    issue_pending(/*is_last_entry=*/false);
                    valid_seen++;
                }
                pending_s2d = bcast_s2d;
                pending_entry_idx = entry_base + src_lane;
                have_pending = true;
            }
        }
    }

    // Final flush: issue the last pending entry (if any) with is_last_entry=true.
    // Only lane 0 ever sets have_pending, so this is a single-thread tail call.
    if (lane_id == 0 && have_pending) {
        issue_pending(/*is_last_entry=*/true);
    }

    global_offset += total_valid_count;
}

// Device function for intra-node G2S warp for combine kernel. There can be only 1 such warp per CUDA block!
template <
    typename SMEM_TYPE,
    int NUM_OF_STAGES_G2S,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclEpLayout_t kLayout,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void combine_warps_G2S_intra(
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int num_of_ranks_per_node,
    const bool* rdma_to_attn_map,
    const int32_t* sparse_to_dense_map,
    uint16_t* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    SMEM_TYPE* smem_buffer_ptr,
    const int experts_per_rank,
    const bool combine_local_reduce_enabled) {
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // The intra node reduction warp group of each CUDA block produce a chunk at a time.
    // The chunk order is: first produce the same chunk id for all other nodes id, then produce following chunk id.
    // (i.e. chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, node + 2, ... node - 1)
    // The RDMA warp group of a CUDA block will consume the chunk by the same order. So each CUDA block will produce and consume the same set of chunks id.
    // The reason to distribute chunk in this order is that the inter-node reduction will need the same chunk id from all other nodes, so we need to produce and send chunks in this order.

    const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
    // How many chunks per rank. Including full chunks and the remainder chunk.
    const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
    // Total number of chunks to produce for RDMA warps to consume.
    const int total_num_of_chunks = (NUM_LSA_TEAMS - 1) * num_of_chunks_per_rank;
    // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
    // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
    // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_LSA_TEAMS.
    const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
    // Warp-cooperative G2S: all lanes participate in parallel TMA.
    // Lanes load sparse_to_dense_map in parallel (up to 32 ranks per warp pass),
    // then valid lanes issue TMA to different stages simultaneously.
    // RED processes stages sequentially.
    //
    // Parity protocol: G2S tracks a "global_offset" counting total stages filled.
    // For a lane with global rank R among valid entries, its stage and parity are:
    //   stage_idx = (global_offset + R) % ring_len
    //   parity    = 1 ^ ((global_offset + R) / ring_len) & 1
    // This matches RED's sequential consumption exactly.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    constexpr int ring_len = NUM_OF_STAGES_G2S;
    // Wire token width: 2 B for BF16/FP16 (default), 4 B for FP32 NONE.
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>()));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));

    // EM unfused-combine dedup uses __shfl_up_sync(1); requires s2d_inner_dim <= WARP_SIZE.
    if (combine_local_reduce_enabled && lane_id == 0 && smem_buffer_ptr->s2d_inner_dim > WARP_SIZE) {
        __trap();
    }

    // Track total stages filled across all tokens.
    int global_offset = 0;

    // Iterate through all chunks assigned to this block.
    for (int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS) {
        // Which node this chunk will be sent to.
        int node_id = (i % (NUM_LSA_TEAMS - 1) + (node_rank + 1)) % NUM_LSA_TEAMS;
        // What is the chunk id of this chunk for the node it will be sent to.
        int chunk_id = i / (NUM_LSA_TEAMS - 1);
        // How many token for this chunk.
        int current_chunk_size;
        if (remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1) {
            current_chunk_size = remainder_chunk_size;
        } else {
            current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
        }

        const bool* rdma_to_attn_map_load_base_addr =
            rdma_to_attn_map + (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK);

        // S2D inner dim: mode-dependent, carried by SMEM layout struct.
        const int s2d_entries_g2s = smem_buffer_ptr->s2d_inner_dim;
        const int32_t* sparse_to_dense_map_load_base_addr =
            sparse_to_dense_map +
            (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * s2d_entries_g2s;

        // Iterate through all dst tokens within this chunk.
        for (int current_token_id = 0; current_token_id < current_chunk_size; current_token_id++) {
            // Check whether this dst token is needed by this node. If not needed, just skip.
            bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
            if (!token_needed_by_this_node) {
                continue;
            }

            const int32_t* sparse_to_dense_row =
                sparse_to_dense_map_load_base_addr + current_token_id * s2d_entries_g2s;

            // Warp-cooperative s2d-row scan with inline broadcast-issue; advances
            // global_offset by the number of entries issued. See issue_local_g2s_row.
            issue_local_g2s_row</*INTER_NODE=*/false, BACKWARD_COMBINE, kLayout, HIDDEN_DIM, kTokenDtype>(
                smem_buffer_ptr,
                sparse_to_dense_row,
                s2d_entries_g2s,
                global_offset,
                /*starting_G2S_index=*/0,
                ring_len,
                lane_id,
                remote_expert_input_token,
                remote_expert_input_prob,
                token_bytes,
                prob_bytes,
                experts_per_rank,
                num_of_ranks_per_node,
                combine_local_reduce_enabled);
        }
    }
}

// Device function for intra-node reduction warp group for combine kernel.
template <
    typename INTRA_NODE_RED_GROUP,
    typename SMEM_TYPE,
    int NUM_OF_STAGES_G2S,
    int NUM_OF_STAGES_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SIZE,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void intra_node_red_warp_group_device_function(
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int num_of_ranks_per_node,
    const bool* rdma_to_attn_map,
    uint16_t* rdma_intra_node_red_token,
    float* rdma_intra_node_red_prob,
    SMEM_TYPE* smem_buffer_ptr,
    const int experts_per_rank) {
    // Vectorized loads from rdma_to_attn_map. Each destination token contributes one bool.
    using rdma_to_attn_map_load_t = uint4;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
    constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
    constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

    // Token values are processed as BF16x2 and accumulated in FP32. HIDDEN_DIM must be even.

    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
    constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
    constexpr int NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA =
        ((NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
    // Backward-combine probability vectors stay in float (no BF16 packing).
    const int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
        ((experts_per_rank * num_of_ranks_per_node - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
    // Compile-time upper bound sized exactly to this instantiation's LSA team.
    constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
        ((NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SIZE - 1) / INTRA_NODE_RED_GROUP::size()) + 1;

    // This warp group emits chunks in the same per-destination order consumed by the RDMA warp group:
    // chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, ...
    // That ordering lets the downstream inter-node stage observe matching chunk IDs across peers.

    const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
    // Number of chunks for one rank, including the tail chunk if present.
    const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
    // Total chunks emitted by this node across all remote destinations.
    const int total_num_of_chunks = (NUM_LSA_TEAMS - 1) * num_of_chunks_per_rank;
    // Pad each node's rdma_to_attn_map slice to one vector-load granularity (16 bytes / 16 bools).
    const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
    // G2S FIFO cursor and producer parity for source-token consumption.
    int token_stage = 0;
    uint32_t token_producer_parity = 0;

    // S2G FIFO cursor for reduced destination tokens.
    int dst_token_stage = 0;

    // Streaming overlap: drain + signal every STREAMING_BATCH dst tokens.
    // The counter is CUMULATIVE across all chunks (never reset), avoiding inter-chunk races.
    constexpr int STREAMING_BATCH = HYBRIDEP_COMBINE_RDMA_STREAMING_BATCH;
    int streaming_pending = 0; // tokens TMA-committed but not yet signaled to counter
    uint32_t cumulative_produced = 0; // total active tokens whose TMA S2G is complete (across all chunks)

    // Iterate through all chunks assigned to this block.
    for (int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS) {
        // Destination node for this emitted chunk.
        int node_id = (i % (NUM_LSA_TEAMS - 1) + (node_rank + 1)) % NUM_LSA_TEAMS;
        // Chunk index within that destination node's stream.
        int chunk_id = i / (NUM_LSA_TEAMS - 1);
        // Compact destination-slot index in the RDMA reduction buffers.
        int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
        // Token offset of this chunk inside the per-destination reduction buffer.
        int rdma_intra_node_red_id =
            rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
        // Number of vector loads needed for the routing flags of this chunk.
        int num_of_routing_info_load_iter_for_current_chunk;
        // Number of valid tokens in this chunk.
        int current_chunk_size;
        if (remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1) {
            num_of_routing_info_load_iter_for_current_chunk =
                ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
            current_chunk_size = remainder_chunk_size;
        } else {
            num_of_routing_info_load_iter_for_current_chunk = NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK;
            current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
        }

        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr =
            reinterpret_cast<const rdma_to_attn_map_load_t*>(
                rdma_to_attn_map + (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK));

        // Per-token stride in uint16_t units: HIDDEN_DIM for BF16/FP16, 2*HIDDEN_DIM for FP32.
        uint16_t* rdma_intra_node_red_token_base_ptr =
            rdma_intra_node_red_token + rdma_intra_node_red_id * HIDDEN_DIM * nccl_ep::size_u16<kTokenDtype>();
        float* rdma_intra_node_red_prob_base_ptr;
        if constexpr (BACKWARD_COMBINE) {
            const int experts_per_node = experts_per_rank * num_of_ranks_per_node;
            rdma_intra_node_red_prob_base_ptr = rdma_intra_node_red_prob + rdma_intra_node_red_id * experts_per_node;
        }

        // Cumulative counter: no handshake or reset needed between chunks.
        // The counter monotonically increases across all chunks, so the consumer
        // always sees a valid threshold and never races with a reset.
        streaming_pending = 0;

        // Number of destination-token S2G copies committed for this chunk.
        int additional_in_flight_s2g = 0;
        // Iterate through all destination tokens within this chunk.
        for (int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++) {
            rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
#pragma unroll
            for (int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++) {
                int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
                // Tail chunk: stop once we step past the real token count.
                if (current_token_id >= current_chunk_size) {
                    break;
                }
                // Check whether the destination node for this chunk needs this token.
                bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
                // If so, one or more contributing source tokens are already being staged through G2S.
                if (token_needed_by_this_node) {
                    // FP32 accumulator for the register-resident token.
                    float2 acc_token_fp32[NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA];
                    // Optional FP32 accumulator for probability data in backward combine.
                    // This storage is instantiated only in backward specializations.
                    using acc_prob_storage_type =
                        acc_prob_storage_t<BACKWARD_COMBINE, MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD>;
                    [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
                    [[maybe_unused]] float* acc_prob_ptr = nullptr;
                    if constexpr (BACKWARD_COMBINE) {
                        acc_prob_ptr = acc_prob_storage.data;
                    }
                    // Producer marks the final contributor for this destination token with this flag.
                    bool last_src_token = false;
#pragma unroll
                    for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA; n++) {
                        acc_token_fp32[n].x = 0.0f;
                        acc_token_fp32[n].y = 0.0f;
                    }
                    if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                        for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
                            acc_prob_ptr[n] = 0.0f;
                        }
                    }
                    // Consume source tokens for this destination token until the producer marks the last one.
                    do {
                        // Current source token / optional prob slice in the G2S FIFO stage.
                        __nv_bfloat162* load_token_base_ptr =
                            reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_G2S(token_stage));
                        float* load_prob_base_ptr;
                        if constexpr (BACKWARD_COMBINE) {
                            load_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_G2S(token_stage);
                        }

                        // Warp 0 waits for the producer; then the whole reduction group can read this stage.
                        if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                            if (cuda::ptx::elect_sync(~0)) {
                                while (!cuda::ptx::mbarrier_try_wait_parity(
                                    smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage),
                                    token_producer_parity)) {
                                }
                            }
                        }
                        arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

// Accumulate the register-resident token. NONE-FP16 reinterprets the same 4
// SMEM bytes as __half2 instead of __nv_bfloat162. NONE-FP32 reads a float2
// (8 SMEM bytes per slot) and skips precision conversion. Predicates are
// launch-uniform so branching costs nothing per warp.
#pragma unroll
                        for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA; n++) {
                            int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
                            if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                                float2 src_data_fp32 =
                                    nccl_ep::ld_token_pair<kTokenDtype>(load_token_base_ptr, element_id);
                                acc_token_fp32[n].x += src_data_fp32.x;
                                acc_token_fp32[n].y += src_data_fp32.y;
                            }
                        }
                        // Accumulate the token tail in shared memory to cap register usage for large hidden dims.
                        if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                            for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
                                int prob_element_id =
                                    INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
                                if (prob_element_id < experts_per_rank * num_of_ranks_per_node) {
                                    float src_data = load_prob_base_ptr[prob_element_id];
                                    acc_prob_ptr[n] += src_data;
                                }
                            }
                        }

                        // Producer sets this on the last source token for the current destination token.
                        last_src_token = smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage];

                        // All reduction threads must finish consuming this G2S stage before the producer reuses it.
                        arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);
                        if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                            if (cuda::ptx::elect_sync(~0)) {
                                cuda::ptx::mbarrier_arrive(
                                    smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(token_stage));
                            }
                        }

                        // Advance to the next G2S stage, toggling parity on wraparound.
                        token_stage += 1;
                        if (token_stage == NUM_OF_STAGES_G2S) {
                            token_stage = 0;
                            token_producer_parity ^= 1;
                        }

                    } while (!last_src_token);

                    // Current reduced destination token / optional prob slice in the S2G FIFO stage.
                    __nv_bfloat162* store_token_base_ptr =
                        reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage));
                    float* store_prob_base_ptr;
                    if constexpr (BACKWARD_COMBINE) {
                        store_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_S2G(dst_token_stage);
                    }

                    // Ensure any earlier TMA read from this S2G stage has completed before we overwrite it.
                    if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                        if (cuda::ptx::elect_sync(~0)) {
                            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G - 1>{});
                        }
                    }
                    // All reduction threads wait here before storing new data into this stage.
                    arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

// Store the register-resident token.
//   NONE-FP16: pack via __float22half2_rn into the same 4 SMEM bytes.
//   NONE-FP32: write float2 verbatim (8 SMEM bytes per slot) — no precision
//   conversion. TMA later copies bytes verbatim to global.
#pragma unroll
                    for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD_INTRA; n++) {
                        int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
                        if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                            nccl_ep::st_token_pair<kTokenDtype>(store_token_base_ptr, element_id, acc_token_fp32[n]);
                        }
                    }
                    // Store the token tail from the SMEM accumulator, or zeros if no tail element was touched.
                    // Store the prob(optional).
                    if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                        for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
                            int prob_element_id =
                                INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
                            if (prob_element_id < experts_per_rank * num_of_ranks_per_node) {
                                store_prob_base_ptr[prob_element_id] = acc_prob_ptr[n];
                            }
                        }
                    }

                    // Publish these shared-memory writes to the async copy engine.
                    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

                    // All threads must finish populating this S2G stage before the TMA thread launches the copy.
                    arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

                    // Warp 0 issues the S2G copies for this reduced destination token.
                    if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                        if (cuda::ptx::elect_sync(~0)) {
                            // Wire token width: 4 B for FP32, 2 B for BF16/FP16. The global buffer base is
                            // uint16_t*, so the per-token stride is scaled into uint16_t units.
                            const size_t red_token_bytes = HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>());
                            uint16_t* current_token_addr =
                                rdma_intra_node_red_token_base_ptr +
                                (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * red_token_bytes / sizeof(uint16_t);
                            // Copy the reduced token from the S2G shared stage to the per-destination global buffer.
                            cuda::ptx::cp_async_bulk(
                                cuda::ptx::space_global,
                                cuda::ptx::space_shared,
                                reinterpret_cast<void*>(current_token_addr),
                                reinterpret_cast<const void*>(
                                    smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage)),
                                (uint32_t)(red_token_bytes));

                            // Store the prob from shared to global(Optional).
                            if constexpr (BACKWARD_COMBINE) {
                                float* current_prob_addr = rdma_intra_node_red_prob_base_ptr +
                                                           (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) *
                                                               (experts_per_rank * num_of_ranks_per_node);
                                cuda::ptx::cp_async_bulk(
                                    cuda::ptx::space_global,
                                    cuda::ptx::space_shared,
                                    reinterpret_cast<void*>(current_prob_addr),
                                    reinterpret_cast<const void*>(
                                        smem_buffer_ptr->get_intra_node_prob_S2G(dst_token_stage)),
                                    (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float)));
                            }
                            // Group the token/prob copies for this destination token into one async-copy commit.
                            cuda::ptx::cp_async_bulk_commit_group();
                        }
                    }

                    // Advance to the next S2G stage.
                    dst_token_stage += 1;
                    if (dst_token_stage == NUM_OF_STAGES_S2G) {
                        dst_token_stage = 0;
                    }

                    // Another token entry's S2G in-flight.
                    additional_in_flight_s2g += 1;

                    // Streaming: periodic drain + counter update
                    streaming_pending++;
                    if constexpr (STREAMING_BATCH > 0) {
                        if (streaming_pending >= STREAMING_BATCH) {
                            if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                                if (cuda::ptx::elect_sync(~0)) {
                                    // Drain ALL outstanding TMA S2G writes
                                    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
                                    // Signal tokens ready to RDMA warp (cumulative, never reset).
                                    // Volatile store instead of atomicExch: same block, shared memory
                                    // is not cached on sm_90. __threadfence_block() ensures TMA S2G
                                    // writes are visible before the counter update.
                                    cumulative_produced += streaming_pending;
                                    __threadfence_block();
                                    *((volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter) =
                                        cumulative_produced;
                                }
                            }
                            additional_in_flight_s2g = 0;
                            streaming_pending = 0;
                        }
                    }
                }
            }
        }
        // End of chunk: drain remaining TMA writes + signal streaming counter
        if (streaming_pending > 0 || additional_in_flight_s2g > 0) {
            if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                if (cuda::ptx::elect_sync(~0)) {
                    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
                    if constexpr (STREAMING_BATCH > 0) {
                        cumulative_produced += streaming_pending;
                        __threadfence(); // device scope: flush L2→VRAM for NIC visibility (GDR)
                        *((volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter) = cumulative_produced;
                    }
                }
            }
            streaming_pending = 0;
            additional_in_flight_s2g = 0;
        }

        // Signal chunk-complete mbarrier unconditionally (for parity tracking)
        if constexpr (NUM_LSA_TEAMS != 1) {
            if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
                if (cuda::ptx::elect_sync(~0)) {
                    cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer
                                                    [rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id]);
                }
            }
        }
    }

    // No post-loop cleanup needed: every chunk is fully drained and signaled at chunk end.
}

// Device function for inter-node node2node(RDMA) warp for combine kernel. There can be only 1 inter-node warp per CUDA block!
// Uses ncclGin API (net.put, net.signal)
template <
    typename INTER_NODE_RDMA_GROUP,
    typename SMEM_TYPE,
    int NUM_OF_STAGES_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void inter_node_N2N_warp_group_device_function(
    const int local_rank,
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int num_of_ranks_per_node,
    const bool* rdma_to_attn_map,
    ncclDevComm_t* dcomms,
    ncclWindow_t nccl_token_window,
    ncclWindow_t nccl_prob_window,
    ncclWindow_t nccl_internal_window,
    int num_gin_comms,
    int num_ctx_per_comm,
    void* gin_base_ptr,
    unsigned signals_base,
    unsigned combine_signal_offset,
    const struct combine_memory_region_info_t* mr_info,
    SMEM_TYPE* smem_buffer_ptr,
    const int experts_per_rank) {
    // Token RDMA offsets/sizes below scale by size_u8 (4 B for FP32, 2 B for BF16/FP16);
    // prob is always float and is unaffected.
    // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
    using rdma_to_attn_map_load_t = uint4;
    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
    static_assert(INTER_NODE_RDMA_GROUP::size() == 32, "INTER_NODE_RDMA_GROUP should be 1 warp.");
    static_assert(INTER_NODE_RDMA_GROUP::size() >= NUM_LSA_TEAMS - 1, "mr_info should be loaded at once.");
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_RDMA_GROUP::size() == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of sizeof(rdma_to_attn_map_load_t).");
    // The (NUM_LSA_TEAMS - 1) queue pairs of one block were arranged together.
    // int block_offset = blockIdx.x * (NUM_LSA_TEAMS - 1);
    // Mr_infos and rdma_mbarrier_buffer in shared memory.
    struct combine_memory_region_info_t* smem_mr_info_ptr = nullptr;
    uint64_t* intra_node_to_rdma_mbarrier_buffer_ptr = nullptr;
    constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
    if constexpr (NUM_LSA_TEAMS != 1) {
        smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
        if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
            smem_mr_info_ptr[0] = mr_info[0];
        }
        intra_node_to_rdma_mbarrier_buffer_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;
    }
    __syncwarp();

    // Total number of chunks to produce for RDMA warps to consume.
    int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
    int TOTAL_NUM_OF_CHUNKS = (NUM_LSA_TEAMS - 1) * MAX_NUM_OF_CHUNKS_PER_RANK;
    // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
    // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
    // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_LSA_TEAMS.
    const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
    // INTRA_NODE_RED_GROUP should be 1 warp.
    // The inter_node_N2N_warp should process the same chunk as intra_node_red_warp(They belong to the same block.)
    uint32_t token_consumer_parity = 0;
    uint32_t cumulative_sent = 0; // Cumulative count of active tokens RDMA-put across all chunks (never reset)
    // Loop for every chunks.
    for (int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS) {
        // Which node this chunk will be sent to.
        int node_id = (i % (NUM_LSA_TEAMS - 1) + (node_rank + 1)) % NUM_LSA_TEAMS;
        // With rail-scoped GIN comms, node index maps to rail-team rank.
        int rank_in_remote = node_id < node_rank ? node_rank - 1 : node_rank;
        // What is the chunk id of this chunk for the node it will be sent to.
        int chunk_id = i / (NUM_LSA_TEAMS - 1);
        bool is_residue = (chunk_id >= NUM_OF_CHUNKS_PER_RANK);

        // Distribute chunks across comms for parallelism
        // Use chunk_id to select comm - both sender (here) and receiver (G2S)
        // use the same formula so signals match
        int total_channels = num_gin_comms * num_ctx_per_comm;
        int global_channel = chunk_id % total_channels;
        int comm_idx, ctx_idx;
        get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
        ncclGin net(dcomms[comm_idx], ctx_idx);
        ncclTeam rail = ncclTeamRail(dcomms[comm_idx]);
        int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
        int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
        int token_range = 0;
        if (!is_residue) {
            token_range = NUM_OF_TOKENS_PER_CHUNK;
            if (chunk_id * NUM_OF_TOKENS_PER_CHUNK + token_range > num_of_tokens_per_rank) {
                token_range = num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK;
            }
        }
        constexpr int STREAMING_BATCH = HYBRIDEP_COMBINE_RDMA_STREAMING_BATCH;
        // Per-token wire bytes (compile-time): hidden x element width.
        constexpr size_t token_bytes = static_cast<size_t>(HIDDEN_DIM) * nccl_ep::size_u8<kTokenDtype>();
        if constexpr (STREAMING_BATCH > 0) {
            // ---- STREAMING PATH: process tokens as reduction warp produces them ----
            // cumulative_sent tracks total active tokens across all chunks (no reset).

            if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
                int batch_start_in_chunk = -1;
                int batch_count = 0;

                for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
                    bool need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
                    bool is_last = (token_idx_in_chunk == token_range - 1);

                    if (need_write) {
                        if (batch_count == 0) batch_start_in_chunk = token_idx_in_chunk;
                        batch_count++;
                    }

                    bool should_flush = batch_count > 0 && (!need_write || is_last || batch_count >= STREAMING_BATCH);

                    if (should_flush) {
                        while (*((volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter) <
                               (cumulative_sent + batch_count)) {
                        }

                        int batch_start_token = batch_start_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
                        size_t token_src_offset =
                            smem_mr_info_ptr->rdma_intra_node_red_token_offset +
                            (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * token_bytes;
                        size_t token_dst_offset =
                            smem_mr_info_ptr->combine_rdma_inter_node_group_token_offset +
                            (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * token_bytes;
                        net.put(
                            rail,
                            node_id,
                            nccl_internal_window,
                            token_dst_offset,
                            nccl_token_window,
                            token_src_offset,
                            batch_count * token_bytes,
                            ncclGin_None{},
                            ncclGin_None{},
                            ncclCoopThread());

                        if constexpr (BACKWARD_COMBINE) {
                            size_t prob_src_offset =
                                smem_mr_info_ptr->rdma_intra_node_red_prob_offset +
                                (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                                    (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
                            size_t prob_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_prob_offset +
                                                     (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                                                         (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
                            net.put(
                                rail,
                                node_id,
                                nccl_internal_window,
                                prob_dst_offset,
                                nccl_prob_window,
                                prob_src_offset,
                                batch_count * (experts_per_rank * num_of_ranks_per_node) * sizeof(float),
                                ncclGin_None{},
                                ncclGin_None{},
                                ncclCoopThread());
                        }

                        cumulative_sent += batch_count;
                        batch_count = 0;
                        batch_start_in_chunk = -1;
                    }
                }
            }

            // Wait for mbarrier (parity tracking -- reduction warp always arrives)
            if (!is_residue) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    &intra_node_to_rdma_mbarrier_buffer_ptr
                        [rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id],
                    token_consumer_parity)) {
                }
            }

            // Signal remote
            __syncwarp();
            if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
                constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
                unsigned signal_id = signals_base + combine_signal_offset +
                                     local_rank * (NUM_LSA_TEAMS * MAX_CHUNKS_PER_RANK) +
                                     node_rank * MAX_CHUNKS_PER_RANK + chunk_id;
                net.signal(
                    rail,
                    node_id,
                    ncclGin_SignalAdd{signal_id, 1},
                    ncclCoopThread(),
                    ncclGin_None{},
                    cuda::thread_scope_thread,
                    cuda::thread_scope_thread);
            }
            __syncwarp();

            // No consumed handshake needed: cumulative counter never resets.

        } else {
            // ---- FALLBACK PATH (STREAMING_BATCH == 0): original mbarrier-first ----
            if (!is_residue) {
                while (!cuda::ptx::mbarrier_try_wait_parity(
                    &intra_node_to_rdma_mbarrier_buffer_ptr
                        [rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id],
                    token_consumer_parity)) {
                }
            }

            if (!is_residue && INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
                constexpr int max_batch = HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE;
                int batch_start_in_chunk = -1;
                int batch_count = 0;

                for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
                    bool need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];
                    bool is_last = (token_idx_in_chunk == token_range - 1);

                    if (need_write) {
                        if (batch_count == 0) batch_start_in_chunk = token_idx_in_chunk;
                        batch_count++;
                    }

                    bool should_flush = batch_count > 0 && (!need_write || is_last || batch_count >= max_batch);

                    if (should_flush) {
                        int batch_start_token = batch_start_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
                        size_t token_src_offset =
                            smem_mr_info_ptr->rdma_intra_node_red_token_offset +
                            (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * token_bytes;
                        size_t token_dst_offset =
                            smem_mr_info_ptr->combine_rdma_inter_node_group_token_offset +
                            (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) * token_bytes;
                        net.put(
                            rail,
                            node_id,
                            nccl_internal_window,
                            token_dst_offset,
                            nccl_token_window,
                            token_src_offset,
                            batch_count * token_bytes,
                            ncclGin_None{},
                            ncclGin_None{},
                            ncclCoopThread());

                        if constexpr (BACKWARD_COMBINE) {
                            size_t prob_src_offset =
                                smem_mr_info_ptr->rdma_intra_node_red_prob_offset +
                                (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                                    (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
                            size_t prob_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_prob_offset +
                                                     (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                                                         (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
                            net.put(
                                rail,
                                node_id,
                                nccl_internal_window,
                                prob_dst_offset,
                                nccl_prob_window,
                                prob_src_offset,
                                batch_count * (experts_per_rank * num_of_ranks_per_node) * sizeof(float),
                                ncclGin_None{},
                                ncclGin_None{},
                                ncclCoopThread());
                        }

                        batch_count = 0;
                        batch_start_in_chunk = -1;
                    }
                }
            }

            // Signal remote
            __syncwarp();
            if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
                constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
                unsigned signal_id = signals_base + combine_signal_offset +
                                     local_rank * (NUM_LSA_TEAMS * MAX_CHUNKS_PER_RANK) +
                                     node_rank * MAX_CHUNKS_PER_RANK + chunk_id;
                net.signal(
                    rail,
                    node_id,
                    ncclGin_SignalAdd{signal_id, 1},
                    ncclCoopThread(),
                    ncclGin_None{},
                    cuda::thread_scope_thread,
                    cuda::thread_scope_thread);
            }
            __syncwarp();
        }
    }
    token_consumer_parity ^= 1;
}

// Device function for inter-node G2S warp for combine kernel.
template <
    typename SMEM_TYPE,
    typename INTER_NODE_G2S_GROUP,
    int NUM_OF_STAGES_G2S,
    int NUM_OF_TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    int NUM_OF_TOKENS_PER_GROUP,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SIZE,
    ncclEpLayout_t kLayout,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void combine_warps_G2S_inter(
    const int local_rank,
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int num_of_ranks_per_node,
    const uint64_t expected_flag_value,
    const bool* rdma_to_attn_map,
    const bool* attn_to_rdma_map,
    const int32_t* sparse_to_dense_map,
    uint16_t* const* remote_expert_input_token,
    float* const* remote_expert_input_prob,
    const uint16_t* rdma_inter_node_group_token,
    const float* rdma_inter_node_group_prob,
    ncclDevComm_t* dcomms,
    unsigned signals_base,
    unsigned combine_signal_offset,
    int num_gin_comms,
    int num_ctx_per_comm,
    uint64_t* rdma_inter_node_group_flags,
    SMEM_TYPE* smem_buffer_ptr,
    const int experts_per_rank,
    const bool combine_local_reduce_enabled) {
    // The warps from inter-node G2S warp group will be divided into multiple independent pipeline.
    // Each pipeline can only have 1 warp, so INTER_NODE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_BLOCK and warp has the same meaning as pipeline in inter-node G2S warp group.
    // Number of pipeline should match inter-node red warp group, so they can coupled into multiple independent data pipeline within a CUDA block.
    // Evenly distribute the inter-node G2S FIFO to every pipeline(warp) within the inter-node G2S warp group.
    // When inter-node G2S warp group only has 1 warp, then the algorith is the same as old version(1 pipeline per CUDA block).
    static_assert(
        NUM_OF_STAGES_G2S % INTER_NODE_G2S_GROUP::warp_size() == 0,
        "NUM_OF_STAGES_G2S must be multiple of inter-node G2S warp group warp size.");
    constexpr int NUM_OF_STAGES_G2S_PER_WARP = NUM_OF_STAGES_G2S / INTER_NODE_G2S_GROUP::warp_size();
    // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks.
    // This is different than other functions where chunks are assigned to different CUDA blocks.
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
    constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;

    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // The inter node reduction warp group of each CUDA block produce a token group of a chunk at a time. Token groups of each chunk assigned to each CUDA block in interleave pattern.
    // The chunk order is: i.e. chunk 0, then chunk 1, ... the last chunk of attn output buffer.
    // The RDMA network for current rank will produce the same chunk id from node - 1, node - 2 ... node + 1.
    // So inter node reduction warp group will consume the src chunk in the same order.

    const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
    // How many chunks per rank. Including full chunks and the remainder chunk.
    const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
    const int max_num_of_chunks_per_rank = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
    // Total number of chunks to process in the output buffer(attn buffer). output buffer(attn buffer) will only have 1 rank's tokens.
    const int total_num_of_chunks = num_of_chunks_per_rank;
    // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
    // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
    // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_LSA_TEAMS.
    const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
    // Starting and ending index within G2S FIFO for this warp(pipeline).
    const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * INTER_NODE_G2S_GROUP::warp_rank();
    const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * (INTER_NODE_G2S_GROUP::warp_rank() + 1);
    // Unified inter-node G2S body. One flow handles both NVLink-only
    // (NUM_LSA_TEAMS==1) and RDMA+NVLink (>1) configurations.
    //
    // LOCAL tier (always, when the dst token has any local contribution):
    //   two-pass warp-cooperative scan of sparse_to_dense_row, then
    //   parallel TMA issue with ring_len batching. is_em_secondary_entry
    //   (no-op under FLAT, __shfl_up_sync(1) dedup under EM) filters the
    //   ballot mask uniformly across both configurations.
    //
    // RDMA tier (NUM_LSA_TEAMS > 1 only):
    //   chunk-level ncclGin signal pre-wait, per-token warp-cooperative
    //   parallel-issue of RDMA-buffer TMAs, post-loop residue-flag update.
    //   Unchanged from the prior multi-domain branch except for being
    //   nested inside the unified chunk loop instead of a separate `else`.
    //
    // Parity protocol (unchanged): global_offset counts stages filled so
    // far this warp; entry rank R in the current batch lands at
    //   stage_idx = starting + (global_offset + R) % ring_len
    //   parity    = 1 ^ ((global_offset + R) / ring_len) & 1
    // which matches RED's sequential consumption.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    const int ring_len = ending_G2S_index - starting_G2S_index;
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>()));
    const uint32_t prob_bytes = (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));

    // EM unfused-combine dedup uses __shfl_up_sync(1); requires
    // s2d_inner_dim <= WARP_SIZE. The runtime predicate
    // `combine_local_reduce_enabled` is only true under EM unfused
    // combine, so this trap is a no-op for FLAT.
    if (combine_local_reduce_enabled && lane_id == 0 && smem_buffer_ptr->s2d_inner_dim > WARP_SIZE) {
        __trap();
    }

    // Total stages filled across all tokens (local + RDMA).
    int global_offset = 0;

    for (int i = 0; i < total_num_of_chunks; i++) {
        int num_of_token_groups_for_current_chunk;
        int current_chunk_size;
        if (remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1) {
            num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
            current_chunk_size = remainder_chunk_size;
        } else {
            num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
            current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
        }

        const bool* rdma_to_attn_map_load_base_addr =
            rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
        const int s2d_entries = smem_buffer_ptr->s2d_inner_dim;
        const int32_t* sparse_to_dense_map_load_base_addr =
            sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * s2d_entries;

        // Chunk-level RDMA state. Pre-wait ncclGin signals so the per-token
        // RDMA tier can issue without per-token signal waits. Only multi-domain.
        const bool* attn_to_rdma_map_load_base_addr = nullptr;
        bool rdma_flag_clear[NUM_LSA_TEAMS];
        if constexpr (NUM_LSA_TEAMS > 1) {
            attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_LSA_TEAMS - 1);

            if (lane_id == 0) {
                constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
                int total_channels = num_gin_comms * num_ctx_per_comm;
                int global_channel = i % total_channels;
                int comm_idx, ctx_idx;
                get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
                ncclGin net(dcomms[comm_idx], ctx_idx);
                for (int n = 1; n < NUM_LSA_TEAMS; n++) {
                    int node_id_for_signal = node_rank >= n ? node_rank - n : node_rank + NUM_LSA_TEAMS - n;
                    unsigned signal_id = signals_base + combine_signal_offset +
                                         local_rank * (NUM_LSA_TEAMS * MAX_CHUNKS_PER_RANK) +
                                         node_id_for_signal * MAX_CHUNKS_PER_RANK + i;
                    net.waitSignal(ncclCoopThread(), signal_id, expected_flag_value);
                }
            }
            __syncwarp(0xffffffff);

#pragma unroll
            for (int jj = 0; jj < NUM_LSA_TEAMS; ++jj) {
                rdma_flag_clear[jj] = true;
            }
        }

        for (int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS) {
            for (int k = INTER_NODE_G2S_GROUP::warp_rank(); k < NUM_OF_TOKENS_PER_GROUP;
                 k += INTER_NODE_G2S_GROUP::warp_size()) {
                int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
                if (current_token_id >= current_chunk_size) {
                    break;
                }

                // Whether the local LSA team has any contribution to this token.
                // Uniform across the warp. NOTE: no early `continue` -- the RDMA
                // tier (NUM_LSA_TEAMS > 1) still runs when this is false.
                bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];

                // ===== LOCAL TIER (always, when token_needed_by_this_node) =====
                // Warp-cooperative s2d-row scan with inline broadcast-issue:
                // each WARP_SIZE-wide batch loads 32 s2d entries (one per lane),
                // builds a 32-bit ballot mask, then all 32 lanes step through
                // the set bits of that mask in lockstep. For each set bit the
                // owning lane's s2d_val is broadcast to the whole warp via
                // __shfl_sync; lane 0 captures the broadcast and issues the
                // per-entry TMA serially. The s2d value crosses lanes via the
                // shuffle, never through shmem; gmem is read exactly once.
                //
                // The issue is deferred by one entry (kept in pending_*) so
                // that `is_last_entry` can be marked correctly without knowing
                // `total_valid_count` up front (it accumulates across batches).
                // `__ballot_sync` at the head of each batch is the cross-iter
                // barrier that keeps lanes 1..31 from racing ahead of lane 0's
                // serialised issues.
                if (token_needed_by_this_node) {
                    const int32_t* sparse_to_dense_row =
                        sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * s2d_entries;

                    // Warp-cooperative s2d-row scan with inline broadcast-issue; advances
                    // global_offset by the number of entries issued, keeping subsequent
                    // reads (e.g. the RDMA tier below at NUM_LSA_TEAMS > 1) consistent on
                    // every lane. See issue_local_g2s_row.
                    issue_local_g2s_row</*INTER_NODE=*/true, BACKWARD_COMBINE, kLayout, HIDDEN_DIM, kTokenDtype>(
                        smem_buffer_ptr,
                        sparse_to_dense_row,
                        s2d_entries,
                        global_offset,
                        starting_G2S_index,
                        ring_len,
                        lane_id,
                        remote_expert_input_token,
                        remote_expert_input_prob,
                        token_bytes,
                        prob_bytes,
                        experts_per_rank,
                        num_of_ranks_per_node,
                        combine_local_reduce_enabled);
                } // end LOCAL TIER

                // ===== RDMA TIER (NUM_LSA_TEAMS > 1 only) =====
                if constexpr (NUM_LSA_TEAMS > 1) {
                    // Warp-cooperative RDMA: each lane maps to a remote node
                    // (lane_id -> n = lane_id + 1). Valid lanes issue TMAs in
                    // parallel to different stages. Signal waits remain on
                    // lane 0 only (ncclGin safety). Since NUM_LSA_TEAMS <= 33,
                    // a single warp pass covers all remote nodes.
                    static_assert(
                        NUM_LSA_TEAMS <= 33,
                        "NUM_LSA_TEAMS must fit in a single warp pass for RDMA parallelization.");

                    // Ensure all local TMAs are committed before RDMA (memory fence).
                    __syncwarp(0xffffffff);

                    const bool* attn_to_rdma_map_load_addr =
                        attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_LSA_TEAMS - 1);

                    // Each lane maps to one remote node.
                    const int rdma_n = lane_id + 1;
                    const bool rdma_lane_active = (rdma_n < NUM_LSA_TEAMS);

                    int rdma_node_id = 0, rdma_buffer_tile_id = 0;
                    bool rdma_entry_valid = false;
                    if (rdma_lane_active) {
                        rdma_node_id = node_rank >= rdma_n ? node_rank - rdma_n : node_rank + NUM_LSA_TEAMS - rdma_n;
                        rdma_buffer_tile_id = rdma_node_id > node_rank ? rdma_node_id - 1 : rdma_node_id;
                        rdma_entry_valid = attn_to_rdma_map_load_addr[rdma_buffer_tile_id];
                    }

                    // Count valid RDMA entries -- identical for all 32 lanes.
                    const unsigned rdma_valid_mask = __ballot_sync(0xffffffff, rdma_lane_active && rdma_entry_valid);
                    const int rdma_valid_count = __popc(rdma_valid_mask);

                    if (rdma_valid_count > 0) {
                        // Signal already pre-waited at chunk start (before per-token loop).
                        // Proceed directly to warp-cooperative TMA load.
                        // Each valid lane computes its rank among valid entries (ascending lane_id = ascending n order).
                        const int rdma_local_rank = __popc(rdma_valid_mask & ((1u << lane_id) - 1));

                        // Issue TMAs in batches of ring_len to prevent stage collisions
                        // when rdma_valid_count > ring_len (e.g. 8-node multinode config
                        // with ring_len=4). Mirrors the local-producer batching above;
                        // without this the parity protocol cannot resolve cleanly because
                        // RED consumes stages sequentially while overflow lanes wait on
                        // parity-flipped consumer arrives that depend on RED progress.
                        int rdma_ranks_issued = 0;
                        while (rdma_ranks_issued < rdma_valid_count) {
                            const int batch_end = (rdma_ranks_issued + ring_len < rdma_valid_count) ?
                                                      rdma_ranks_issued + ring_len :
                                                      rdma_valid_count;

                            const bool in_batch = rdma_lane_active && rdma_entry_valid &&
                                                  rdma_local_rank >= rdma_ranks_issued && rdma_local_rank < batch_end;
                            if (in_batch) {
                                const int rank_in_batch = rdma_local_rank - rdma_ranks_issued;

                                const uint16_t* token_src =
                                    rdma_inter_node_group_token +
                                    (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK + i * NUM_OF_TOKENS_PER_CHUNK +
                                     j * NUM_OF_TOKENS_PER_GROUP + k) *
                                        HIDDEN_DIM * nccl_ep::size_u16<kTokenDtype>();
                                const float* prob_src = nullptr;
                                if constexpr (BACKWARD_COMBINE) {
                                    prob_src = rdma_inter_node_group_prob +
                                               (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK +
                                                i * NUM_OF_TOKENS_PER_CHUNK + j * NUM_OF_TOKENS_PER_GROUP + k) *
                                                   (experts_per_rank * num_of_ranks_per_node);
                                }
                                // RDMA tier does not set inter_node_flag_G2S_buffer:
                                // the RED warp group also reads attn_to_rdma_map to
                                // demarcate RDMA entries.
                                issue_g2s_entry</*INTER_NODE=*/true, BACKWARD_COMBINE, /*WRITE_LAST_FLAG=*/false>(
                                    smem_buffer_ptr,
                                    global_offset,
                                    rank_in_batch,
                                    starting_G2S_index,
                                    ring_len,
                                    token_src,
                                    token_bytes,
                                    prob_src,
                                    prob_bytes,
                                    /*is_last_entry=*/false);
                            }

                            global_offset += (batch_end - rdma_ranks_issued);
                            rdma_ranks_issued = batch_end;
                            // Prevent non-in_batch lanes from racing into the next iteration
                            // before in_batch lanes finish their wait_parity + TMA + arrive.
                            __syncwarp(0xffffffff);
                        }
                    }
                } // end RDMA TIER
            }
        }
    }
    if constexpr (NUM_LSA_TEAMS > 1) {
        // Update residue flags.
        int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;
        for (int node_id = blockIdx.x; node_id < NUM_LSA_TEAMS - 1; node_id += gridDim.x) {
            uint64_t* residue_flag_base_ptr =
                rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank + num_of_chunks_per_rank);
            for (int flag_id = INTER_NODE_G2S_GROUP::thread_rank(); flag_id < residue_flag_count;
                 flag_id += INTER_NODE_G2S_GROUP::size()) {
                residue_flag_base_ptr[flag_id] = expected_flag_value;
            }
        }
    }
}

// Device function for inter-node reduction warp group for combine kernel.
template <
    typename SMEM_TYPE,
    typename INTER_NODE_RED_GROUP,
    int NUM_OF_DATA_PIPELINE_PER_BLOCK,
    int NUM_OF_STAGES_G2S,
    int NUM_OF_STAGES_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    int NUM_OF_TOKENS_PER_GROUP,
    bool BACKWARD_COMBINE,
    int HIDDEN_DIM,
    int LSA_TEAM_SIZE,
    ncclDataType_t kTokenDtype>
__forceinline__ __device__ void inter_node_red_warp_group_device_function(
    const int node_rank,
    const int num_of_tokens_per_rank,
    const int num_real_tokens,
    const int num_of_ranks_per_node,
    const bool* rdma_to_attn_map,
    const bool* attn_to_rdma_map,
    uint16_t* attn_output_token,
    float* attn_output_prob,
    SMEM_TYPE* smem_buffer_ptr,
    const int experts_per_rank) {
    // The warps from inter-node red warp group will be divided into multiple independent pipeline. Each pipeline has INTER_NODE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK warps.
    // Number of pipeline should match inter-node G2S warp group, so they can coupled into multiple independent data pipeline within a CUDA block.
    static_assert(
        INTER_NODE_RED_GROUP::warp_size() % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0,
        "The warp count of inter-node red warp group must be multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_OF_THREADS_PER_PIPELINE =
        (INTER_NODE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK) * WARP_SIZE;
    // Evenly distribute the inter-node G2S FIFO to every pipeline within the inter-node red warp group.
    // When NUM_OF_DATA_PIPELINE_PER_BLOCK = 1 and INTER_NODE_RED_GROUP::warp_size() = 4, then the algorith is the same as old version(1 pipeline w/ 4 warps per CUDA block).
    static_assert(
        NUM_OF_STAGES_G2S % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0,
        "NUM_OF_STAGES_G2S must be multiple of data pipeline per CUDA block.");
    constexpr int NUM_OF_STAGES_G2S_PER_PIPELINE = NUM_OF_STAGES_G2S / NUM_OF_DATA_PIPELINE_PER_BLOCK;
    // Evenly distribute the inter-node S2G FIFO to every pipeline within the inter-node red warp group.
    static_assert(
        NUM_OF_STAGES_S2G % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0,
        "NUM_OF_STAGES_S2G must be multiple of data pipeline per CUDA block.");
    constexpr int NUM_OF_STAGES_S2G_PER_PIPELINE = NUM_OF_STAGES_S2G / NUM_OF_DATA_PIPELINE_PER_BLOCK;
    // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks.
    // This is different than other functions where chunks are assigned to different CUDA blocks.
    static_assert(
        NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0,
        "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
    constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;

    static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

    // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
    constexpr int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
    constexpr int NUM_OF_ACC_ELEMENTS_PER_THREAD =
        ((NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;

    // Processing prob using fp32.
    const int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
        ((experts_per_rank * num_of_ranks_per_node - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;

    // Compile-time upper bound sized exactly to this instantiation's LSA team.
    constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
        ((NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SIZE - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;

    // The inter node reduction warp group of each CUDA block produce a token group of a chunk at a time. Token groups of each chunk assigned to each CUDA block in interleave pattern.
    // The chunk order is: i.e. chunk 0, then chunk 1, ... the last chunk of attn output buffer.
    // The RDMA network for current rank will produce the same chunk id from node - 1, node - 2 ... node + 1.
    // So inter node reduction warp group will consume the src chunk in the same order.

    const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
    // How many chunks per rank. Including full chunks and the remainder chunk.
    const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
    // Total number of chunks to process in the output buffer(attn buffer). output buffer(attn buffer) will only have 1 rank's tokens.
    const int total_num_of_chunks = num_of_chunks_per_rank;
    // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
    // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
    // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_LSA_TEAMS.
    const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
    // Pipeline rank and thread/warp rank within the pipeline for this thread.
    const int pipeline_rank = INTER_NODE_RED_GROUP::thread_rank() / NUM_OF_THREADS_PER_PIPELINE;
    const int thread_rank_within_pipeline = INTER_NODE_RED_GROUP::thread_rank() % NUM_OF_THREADS_PER_PIPELINE;
    const int warp_rank_within_pipeline = thread_rank_within_pipeline / WARP_SIZE;
    // Starting and ending index within G2S FIFO for this pipeline.
    const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * pipeline_rank;
    const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_PIPELINE * (pipeline_rank + 1);
    // Src token stage id and phase.
    int token_stage = starting_G2S_index;
    uint32_t token_producer_parity = 0;

    // Starting and ending index within S2G FIFO for this pipeline.
    const int starting_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * pipeline_rank;
    const int ending_S2G_index = NUM_OF_STAGES_S2G_PER_PIPELINE * (pipeline_rank + 1);
    // Dst token stage id.
    int dst_token_stage = starting_S2G_index;

    // Iterate through all chunks. All chunks will assign to all CUDA block.
    for (int i = 0; i < total_num_of_chunks; i++) {
        // How many rdma_to_attn load iter(a.k.a token group) for this chunk.
        int num_of_token_groups_for_current_chunk;
        // How many token for this chunk.
        int current_chunk_size;
        if (remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1) { // tail processing
            num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
            current_chunk_size = remainder_chunk_size;
        } else {
            num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
            current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
        }

        const bool* rdma_to_attn_map_load_base_addr =
            rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
        const bool* attn_to_rdma_map_load_base_addr = nullptr;
        if constexpr (NUM_LSA_TEAMS > 1) {
            attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_LSA_TEAMS - 1);
        }
        // Per-token stride: HIDDEN_DIM uint16_t units (BF16/FP16) or HIDDEN_DIM*2 (FP32).
        // out_token_stride_u16 is the dtype-aware stride captured here.
        const size_t elem_bytes = nccl_ep::size_u8<kTokenDtype>();
        const size_t out_token_stride_u16 = (size_t)HIDDEN_DIM * elem_bytes / sizeof(uint16_t);
        uint16_t* attn_output_token_base_ptr =
            attn_output_token + (size_t)(i * NUM_OF_TOKENS_PER_CHUNK) * out_token_stride_u16;
        float* attn_output_prob_base_ptr;
        if constexpr (BACKWARD_COMBINE) {
            attn_output_prob_base_ptr =
                attn_output_prob +
                (i * NUM_OF_TOKENS_PER_CHUNK) * (experts_per_rank * num_of_ranks_per_node * NUM_LSA_TEAMS);
        }
        // Iterate through all token groups within this chunk which assign to this CUDA block.
        for (int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS) {
            // Iterate through all dst(output) tokens within this token group.
            // Assign each dst token to each pipeline using a round-robin fasion.
            for (int k = pipeline_rank; k < NUM_OF_TOKENS_PER_GROUP; k += NUM_OF_DATA_PIPELINE_PER_BLOCK) {
                int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
                // If the current token is out-of-bound, then just end this load iter.
                if (current_token_id >= current_chunk_size) {
                    break;
                }
                // Each dst token need to accumulate src tokens from local node's ranks(this part is the same as intra-node reduction), and src tokens from rdma inter-node buffers.
                // Accumulate local tokens first, then rdma tokens.
                // Accumulator for this dst token. Token must be accumulated in FP32.
                float2 acc_token_fp32[NUM_OF_ACC_ELEMENTS_PER_THREAD];
                // Optional Accumulator for this dst token prob.
                // Different node's prob need to be gathered together to output.
                // 0 used for local node's prob, [1, NUM_LSA_TEAMS - 1] used for remote node's prob.
                // Flattened array: acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m] for 2D access
                // Use MAX size for compile-time array allocation, actual size determined by runtime experts_per_rank
                using acc_prob_storage_type =
                    acc_prob_storage_t<BACKWARD_COMBINE, NUM_LSA_TEAMS * MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD>;
                [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
                [[maybe_unused]] float* acc_prob_ptr = nullptr;
                if constexpr (BACKWARD_COMBINE) {
                    acc_prob_ptr = acc_prob_storage.data;
                }
// Init accumulator.
#pragma unroll
                for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD; n++) {
                    acc_token_fp32[n].x = 0.0f;
                    acc_token_fp32[n].y = 0.0f;
                }
                if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                    for (int n = 0; n < NUM_LSA_TEAMS; n++) {
                        for (int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++) {
                            acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m] = 0.0f;
                        }
                    }
                }
                // Check whether this dst token is needed by this(local) node. If not needed, just skip local accumulation.
                bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
                // If this dst token is needed by this node, load the local src token from shared memory and accumulate them.
                if (token_needed_by_this_node) {
                    // End reduction group flag.
                    bool last_local_node_src_token = false;
                    // Continue loading local src token for this dst token and reduce them to accumulator until all local src token for this dst token have been accumulated.
                    do {
                        // Base address for current token and prob(optional) in shared memory.
                        __nv_bfloat162* load_token_base_ptr =
                            reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_G2S(token_stage));
                        float* load_prob_base_ptr;
                        if constexpr (BACKWARD_COMBINE) {
                            load_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_G2S(token_stage);
                        }

                        // Wait until current src token ready in shared memory.
                        if (warp_rank_within_pipeline == 0) {
                            if (cuda::ptx::elect_sync(~0)) {
                                while (!cuda::ptx::mbarrier_try_wait_parity(
                                    smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage),
                                    token_producer_parity)) {
                                }
                            }
                        }
                        // named barrier: we wait for number of threads(all threads in the pipline) that must arrive before any can proceed
                        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

// Accumulate token and prob(optional). NONE-FP16: reinterpret as __half2;
// NONE-FP32: read float2 directly (8 SMEM bytes per slot).
#pragma unroll
                        for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD; n++) {
                            int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
                            if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                                float2 src_data_fp32 =
                                    nccl_ep::ld_token_pair<kTokenDtype>(load_token_base_ptr, element_id);
                                acc_token_fp32[n].x += src_data_fp32.x;
                                acc_token_fp32[n].y += src_data_fp32.y;
                            }
                        }
                        if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                            for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
                                int element_id = thread_rank_within_pipeline + n * NUM_OF_THREADS_PER_PIPELINE;
                                if (element_id < experts_per_rank * num_of_ranks_per_node) {
                                    float src_data = load_prob_base_ptr[element_id];
                                    acc_prob_ptr[0 * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + n] += src_data;
                                }
                            }
                        }

                        // Check flag for last src token.
                        last_local_node_src_token = smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage];

                        // Make sure all threads within the pipeline have finished loading the token entry and accumulate it to the register accumulator.
                        // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
                        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
                        if (warp_rank_within_pipeline == 0) {
                            if (cuda::ptx::elect_sync(~0)) {
                                cuda::ptx::mbarrier_arrive(
                                    smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage));
                            }
                        }

                        // Goto next src token entry.
                        token_stage += 1;
                        if (token_stage == ending_G2S_index) {
                            token_stage = starting_G2S_index;
                            token_producer_parity ^= 1;
                        }

                    } while (!last_local_node_src_token);
                }

                if constexpr (NUM_LSA_TEAMS > 1) {
                    // Then accumulate from rdma inter-node buffers. There are total NUM_LSA_TEAMS - 1 (possible) src tokens from rdma buffer to reduce.
                    const bool* attn_to_rdma_map_load_addr =
                        attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_LSA_TEAMS - 1);
#pragma unroll
                    for (int n = 1; n < NUM_LSA_TEAMS; n++) {
                        // The current node been processed. For each chunk id, node_id order is
                        // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
                        int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_LSA_TEAMS - n;
                        // The tile id within the rdma buffers(include attn_to_rdma map) for the current node id. Because these rdma buffers only have NUM_LSA_TEAMS - 1 tile or element.
                        int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
                        // Check wether current dst token need src token from this (remote) node.
                        if (attn_to_rdma_map_load_addr[rdma_buffer_tile_id]) {
                            // Base address for current token and prob(optional) in shared memory.
                            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(
                                smem_buffer_ptr->get_inter_node_token_G2S(token_stage));
                            float* load_prob_base_ptr;
                            if constexpr (BACKWARD_COMBINE) {
                                load_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_G2S(token_stage);
                            }
                            // Wait until current src token ready in shared memory.
                            if (warp_rank_within_pipeline ==
                                0) { // this means that only wrap 0 in the pipeline participates
                                if (cuda::ptx::elect_sync(~0)) {
                                    while (!cuda::ptx::mbarrier_try_wait_parity(
                                        smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage),
                                        token_producer_parity)) {
                                    }
                                }
                            }
                            arrive_and_wait(
                                NUM_OF_THREADS_PER_PIPELINE,
                                2 + pipeline_rank); // named barrier, we wait for number of threads that must arrive before any can proceed

// Accumulate token and prob(optional). NONE-FP16: reinterpret as __half2;
// NONE-FP32: read float2 directly (8 SMEM bytes per slot).
#pragma unroll
                            for (int m = 0; m < NUM_OF_ACC_ELEMENTS_PER_THREAD; m++) {
                                int element_id = (m * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
                                if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                                    float2 src_data_fp32 =
                                        nccl_ep::ld_token_pair<kTokenDtype>(load_token_base_ptr, element_id);
                                    acc_token_fp32[m].x += src_data_fp32.x;
                                    acc_token_fp32[m].y += src_data_fp32.y;
                                }
                            }
                            if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                                for (int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++) {
                                    int element_id = thread_rank_within_pipeline + m * NUM_OF_THREADS_PER_PIPELINE;
                                    if (element_id < experts_per_rank * num_of_ranks_per_node) {
                                        acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m] =
                                            load_prob_base_ptr[element_id];
                                    }
                                }
                            }

                            // Make sure all threads within the pipeline have finished loading the token entry and accumulate it to the register accumulator.
                            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
                            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
                            if (warp_rank_within_pipeline == 0) {
                                if (cuda::ptx::elect_sync(~0)) {
                                    cuda::ptx::mbarrier_arrive(
                                        smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage));
                                }
                            }

                            // Goto next src token entry.
                            token_stage += 1;
                            if (token_stage == ending_G2S_index) {
                                token_stage = starting_G2S_index;
                                token_producer_parity ^= 1;
                            }
                        }
                    }
                }

                // Store the dst token back to share memory.
                // Because each attn token must have go to TOPK rank in dispatch, so it must have been reduced in combine. So each attn dst token must be written back.
                // Base address for current dst token and prob(optional) in shared memory.
                __nv_bfloat162* store_token_base_ptr =
                    reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_S2G(dst_token_stage));
                float* store_prob_base_ptr;
                if constexpr (BACKWARD_COMBINE) {
                    store_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_S2G(dst_token_stage);
                }

                // Select the TMA thread within the pipeline to wait for previously issued TMA S2G operations finish reading this entry.
                if (warp_rank_within_pipeline == 0) {
                    if (cuda::ptx::elect_sync(~0)) {
                        cuda::ptx::cp_async_bulk_wait_group_read(
                            cuda::ptx::n32_t<NUM_OF_STAGES_S2G_PER_PIPELINE - 1>{});
                    }
                }
                // Make sure all threads within the pipeline have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
                arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

// Store the token.
//   NONE-FP16: pack via __float22half2_rn into the same 4 SMEM bytes.
//   NONE-FP32: write float2 verbatim (8 SMEM bytes per slot).
// TMA later copies bytes verbatim to global so the wire dtype matches the kernel pack.
#pragma unroll
                for (int n = 0; n < NUM_OF_ACC_ELEMENTS_PER_THREAD; n++) {
                    int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
                    if (element_id < NUM_OF_BF16X2_ELEMENTS_PER_TOKEN) {
                        nccl_ep::st_token_pair<kTokenDtype>(store_token_base_ptr, element_id, acc_token_fp32[n]);
                    }
                }
                // Store the prob(optional).
                if constexpr (BACKWARD_COMBINE) {
#pragma unroll
                    for (int n = 0; n < NUM_LSA_TEAMS; n++) {
                        int attn_prob_output_node_id =
                            (node_rank - n) >= 0 ? node_rank - n : node_rank + NUM_LSA_TEAMS - n;
                        int element_base_id = attn_prob_output_node_id * (experts_per_rank * num_of_ranks_per_node);
#pragma unroll
                        for (int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++) {
                            int element_id = thread_rank_within_pipeline + m * NUM_OF_THREADS_PER_PIPELINE;
                            if (element_id < experts_per_rank * num_of_ranks_per_node) {
                                store_prob_base_ptr[element_base_id + element_id] =
                                    acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m];
                            }
                        }
                    }
                }

                // Make sure the shared memory stored by current thread is visible by async proxy.
                cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

                // Make sure all threads within the pipeline have finished storing the current token entry and making it visible to async proxy.
                arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

                // Select the TMA thread within the pipeline to issue S2G TMA operations for current token entry.
                if (warp_rank_within_pipeline == 0) {
                    int absolute_token_id = i * NUM_OF_TOKENS_PER_CHUNK + j * NUM_OF_TOKENS_PER_GROUP + k;
                    if (cuda::ptx::elect_sync(~0) && absolute_token_id < num_real_tokens) {
                        // Per-token stride: HIDDEN_DIM uint16_t units (BF16/FP16) or HIDDEN_DIM*2 (FP32).
                        // out_token_stride_u16 is the dtype-aware stride captured above.
                        uint16_t* current_token_addr = attn_output_token_base_ptr +
                                                       (size_t)(j * NUM_OF_TOKENS_PER_GROUP + k) * out_token_stride_u16;
                        // Store the token from shared to global output. Bytes scale with elem width.
                        cuda::ptx::cp_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            reinterpret_cast<void*>(current_token_addr),
                            reinterpret_cast<const void*>(smem_buffer_ptr->get_inter_node_token_S2G(dst_token_stage)),
                            (uint32_t)(HIDDEN_DIM * (nccl_ep::size_u8<kTokenDtype>())));

                        // Store the prob from shared to global output.
                        if constexpr (BACKWARD_COMBINE) {
                            float* current_prob_addr = attn_output_prob_base_ptr +
                                                       (j * NUM_OF_TOKENS_PER_GROUP + k) *
                                                           (experts_per_rank * num_of_ranks_per_node * NUM_LSA_TEAMS);
                            cuda::ptx::cp_async_bulk(
                                cuda::ptx::space_global,
                                cuda::ptx::space_shared,
                                reinterpret_cast<void*>(current_prob_addr),
                                reinterpret_cast<const void*>(
                                    smem_buffer_ptr->get_inter_node_prob_S2G(dst_token_stage)),
                                (uint32_t)((experts_per_rank * num_of_ranks_per_node * NUM_LSA_TEAMS) * sizeof(float)));
                        }
                        // Commit S2G TMA operations for this dst token into a bulk async copy group.
                        cuda::ptx::cp_async_bulk_commit_group();
                    }
                }

                // Goto next dst token entry.
                dst_token_stage += 1;
                if (dst_token_stage == ending_S2G_index) {
                    dst_token_stage = starting_S2G_index;
                }
            }
        }
    }
    // Because the attn output buffers will only be produced by local combine kernel, not by the combine kernels on other ranks,
    // so we only need to wait for local combine kernel to finish writing all token data back to output buffer before we can exit.
    // Also, a kernel will be considered completed from CUDA stream's perspective if and only if all the threads are exit and all memory operations(including TMA operations)
    // issued by all threads have been completed and made visible to sys scope.
    // So the CUDA stream's kernel boundary implicit synchronization should be enough to sync with all TMA operations issued in the combine kernel.
    // So we can directly exit w/o any explicit synchronization with TMA operations.
}

// __launch_bounds__(1, 1)
// __global__ void device_sync_kernel(uint32_t* intra_node_remote_flags, const uint32_t* expected_flag_value)
// {
//   // Atomically reduce add 1 to the u32 flag on rank #0 in current NVLink domain.
//   // Need a strong system-scope red to make sure all ranks from current NVLink domain can see the side effect.
//   // But no memory fence(i.e. .release) needed since CUDA stream already do that for us.
//   // red.relaxed.sys.global.add.u32          [a], 1;
//   asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
//                 :
//                 : "l"(__cvta_generic_to_global(intra_node_remote_flags)), "n"(1)
//                 : "memory");

//   // Polling flag value from the u32 flag on rank #0 in current NVLink domain.
//   // Keep polling until reach the expected value.
//   uint32_t flag_data = 0;
//   do {
//       flag_data = 0;
//       // Need a strong system-scope load to observe other ranks' Atomic result.
//       // But no no memory fence(i.e. .aquired) needed since no memory operation behind this.
//       asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
//                     : "=r"(flag_data)
//                     : "l"(__cvta_generic_to_global(intra_node_remote_flags))
//                     : "memory");
//     } while (flag_data != *expected_flag_value);
// }

// ============================================================================
// PAD warp device function (expert-major zero padding, fused into dispatch_kernel)
// ============================================================================

// Zero-init EM alignment padding slots; one warp inside dispatch_kernel, concurrent with N2N/G2S/S2G.
// Zeroes one SMEM row, then 32 lanes cp_async_bulk it to padding slots striped by (block,lane).
template <typename PAD_GROUP, typename TOKEN_DATA_TYPE, typename SMEM_TYPE>
__forceinline__ __device__ void PAD_warp_group_device_function(
    TOKEN_DATA_TYPE* __restrict__ local_buf,
    const int32_t* __restrict__ actual_counts,
    const int64_t* __restrict__ zone_offsets,
    const int experts_per_rank,
    const int alignment,
    const int hidden_dim,
    const int num_blocks,
    SMEM_TYPE* smem) {
    // Caller zeroes alignment when not expert-major; alignment<=1 ⇒ no padding work.
    if (alignment <= 1 || local_buf == nullptr || actual_counts == nullptr || zone_offsets == nullptr) return;

    const int lane = PAD_GROUP::thread_rank();
    constexpr int warp_size = 32;
    const uint32_t token_bytes = static_cast<uint32_t>(hidden_dim * sizeof(TOKEN_DATA_TYPE));

    // Cooperatively zero the SMEM staging slot once per kernel invocation.
    auto* smem_u4 = reinterpret_cast<uint4*>(smem->get_pad_tma_slot());
    const int vec_n = token_bytes / sizeof(uint4);
    const uint4 zero4{0, 0, 0, 0};
    for (int i = lane; i < vec_n; i += warp_size) smem_u4[i] = zero4;
    __syncwarp();

    // Flatten padding rows across all experts and stripe across (blocks × lanes).
    const int global_id = static_cast<int>(blockIdx.x) * warp_size + lane;
    const int global_stride = num_blocks * warp_size;

    int row_idx = 0;
    int my_in_flight = 0;
    for (int e = 0; e < experts_per_rank; e++) {
        int32_t count = actual_counts[e];
        // Empty experts reserve no zone slot; never pad them.
        if (count == 0) continue;
        int32_t rem = count % alignment;
        int32_t pad = rem ? (alignment - rem) : 0;
        for (int p = 0; p < pad; p++, row_idx++) {
            if ((row_idx % global_stride) == global_id) {
                void* dst = reinterpret_cast<void*>(local_buf + (zone_offsets[e] + count + p) * hidden_dim);
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst,
                    smem->get_pad_tma_slot(),
                    token_bytes);
                my_in_flight++;
            }
        }
    }
    if (my_in_flight > 0) {
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
    }
    __syncwarp();
}

// Inter-node RDMA (GIN) cross-round WAR guard, warp-collective (lane i -> peer i), relaxed (WAR only).

// Wait until every rail peer's flag reaches the expected round.
__device__ __forceinline__ void
warp_rdma_guard_wait(const uint64_t* peer_flags, int node_rank, int num_lsa_teams, uint64_t expected) {
    for (int peer = (threadIdx.x & 31); peer < num_lsa_teams; peer += 32) {
        if (peer == node_rank) continue;
        while (nccl_ep::ld_relaxed_sys_global(&peer_flags[peer]) + 1ull < expected) { /* busy-wait */
        }
    }
}

// Publish the expected round into this rank's slot (my_slot) of every rail peer's window.
__device__ __forceinline__ void warp_rdma_guard_publish(
    ncclDevComm dcomm,
    ncclWindow_t dest_window,
    size_t my_slot,
    int node_rank,
    int num_lsa_teams,
    uint64_t expected) {
    ncclGin net(dcomm, /*contextIndex=*/0, NCCL_GIN_RESOURCE_SHARING_THREAD);
    ncclTeam rail = ncclTeamRail(dcomm);
    for (int peer = (threadIdx.x & 31); peer < num_lsa_teams; peer += 32) {
        if (peer == node_rank) continue;
        net.putValue(rail, peer, dest_window, my_slot, expected, ncclGin_None{}, ncclCoopThread());
    }
}

// Elect the last block to arrive at *counter (result broadcast to all threads in the block).
__device__ __forceinline__ bool elect_last_block(const int* counter, int num_blocks) {
    __syncthreads();
    int arrived = -1;
    if (threadIdx.x == 0) arrived = static_cast<int>(nccl_ep::atomic_add_acqrel_global(counter, 1));
    return __syncthreads_or(arrived == num_blocks - 1);
}

template <
    ncclDataType_t kTokenDtype,
    typename INTER_NODE_GROUP,
    typename INTRA_NODE_G2S_GROUP,
    typename INTRA_NODE_S2G_GROUP,
    typename PAD_GROUP,
    int NUM_OF_STAGES,
    int NUM_OF_IN_FLIGHT_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int MAX_NUM_OF_TOKENS_PER_RANK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    bool FORWARD_DISPATCH,
    int NUM_PIPELINES,
    int LSA_TEAM_SIZE,
    ncclEpLayout_t kLayout,
    int HIDDEN_DIM,
    int SF_BYTES_PER_TOKEN>
__device__ __forceinline__ void dispatch_kernel_impl(
    const dispatch_kernel_param_t<nccl_ep::wire_t<kTokenDtype>, LSA_TEAM_SIZE>& param,
    uint8_t* smem_bytes) {
    using TOKEN_DATA_TYPE = nccl_ep::wire_t<kTokenDtype>;
    if constexpr (NUM_LSA_TEAMS != 1) {
        static_assert(
            INTER_NODE_GROUP::size() % 32 == 0 && INTER_NODE_GROUP::size() <= 64,
            "Dispatch kernel supports 1 or 2 N2N warps.");
    }
    static_assert(NUM_OF_STAGES % NUM_PIPELINES == 0, "NUM_OF_STAGES must be divisible by NUM_PIPELINES.");
    constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;

    using cur_smem_t = dispatch_smem_layout_t;

    cur_smem_t smem_layout;
    dispatch_config_t d_config;
    model_config_t d_model;
    d_config.num_of_stages = NUM_OF_STAGES;
    d_config.num_of_in_flight_s2g = NUM_OF_IN_FLIGHT_S2G;
    d_config.num_of_tokens_per_chunk = NUM_OF_TOKENS_PER_CHUNK;
    d_config.num_of_blocks = NUM_OF_BLOCKS;
    d_config.forward_dispatch = FORWARD_DISPATCH;
    d_config.sf_bytes_per_token = SF_BYTES_PER_TOKEN;
    d_config.num_pipelines = NUM_PIPELINES;
    d_config.stages_per_pipeline = STAGES_PER_PIPELINE;
    d_config.s2d_inner_dim = param.s2d_inner_dim;
    d_model.hidden_dim = HIDDEN_DIM;
    d_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    d_model.num_of_experts_per_rank = param.experts_per_rank;
    d_model.num_of_ranks_per_node = param.num_of_ranks_per_node;
    d_model.num_of_nodes = NUM_LSA_TEAMS;
    create_dispatch_smem_layout<kLayout, kTokenDtype>(smem_layout, smem_bytes, d_config, d_model);
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    using head_init_warp = warp_group<1, 0>;
    using head_rdma_warp = warp_group<1, 1>;
    using head_lsa_warp = warp_group<1, 2>;
    static_assert(
        INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size() + PAD_GROUP::size() >=
            3 * 32,
        "dispatch head needs 3 warps");
    const int head_tid = (int)threadIdx.x;
    if (head_tid < head_init_warp::size()) {
        // warp 0: per-pipeline mbarrier init (both producer/consumer arrival counts = 1).
        if (head_tid == 0) {
            for (int p = 0; p < NUM_PIPELINES; p++) {
                for (int s = 0; s < STAGES_PER_PIPELINE; s++) {
                    int abs_stage = p * STAGES_PER_PIPELINE + s;
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_buffer + 2 * abs_stage, 1);
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_buffer + 2 * abs_stage + 1, 1);
                }
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_s2d_map_mbar(p, 0), 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_s2d_map_mbar(p, 1), 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->get_S2G_group_mbar(p), 1);
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size()) {
        // warp 1: inter-node RDMA guard wait.
        if constexpr (NUM_LSA_TEAMS != 1) {
            if (param.guard_enabled)
                warp_rdma_guard_wait(
                    reinterpret_cast<const uint64_t*>(
                        reinterpret_cast<const uint8_t*>(param.gin_base_ptr) + param.mr_info.guard_offset),
                    param.node_rank,
                    NUM_LSA_TEAMS,
                    *param.expected_rdma_flag_value);
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size() + head_lsa_warp::size()) {
        // warp 2: intra-node LSA barrier.
        if constexpr (LSA_TEAM_SIZE != 1) {
            if (param.guard_enabled) {
                ncclLsaBarrierSession<ncclCoopWarp> bar(
                    ncclCoopWarp(),
                    param.dcomm,
                    ncclTeamTagLsa(),
                    (uint32_t)blockIdx.x);
                bar.sync(ncclCoopWarp(), cuda::memory_order_relaxed);
            }
        }
    }

    __syncthreads();

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    long long _wt_start = 0;
    if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif
    constexpr bool HAS_SF = (SF_BYTES_PER_TOKEN > 0);
    int threadIdx_x_int = (int)threadIdx.x;
    if (threadIdx_x_int < INTER_NODE_GROUP::size()) {
        if constexpr (NUM_LSA_TEAMS != 1) {
            dispatch_N2N_warp<
                INTER_NODE_GROUP,
                TOKEN_DATA_TYPE,
                cur_smem_t,
                NUM_OF_STAGES,
                NUM_OF_TOKENS_PER_CHUNK,
                MAX_NUM_OF_TOKENS_PER_RANK,
                NUM_LSA_TEAMS,
                LSA_TEAM_SIZE,
                NUM_OF_BLOCKS,
                FORWARD_DISPATCH,
                HAS_SF>(
                param.attn_to_rdma_map,
                param.local_rank,
                param.node_rank,
                param.num_of_tokens_per_rank,
                HIDDEN_DIM,
                SF_BYTES_PER_TOKEN,
                param.experts_per_rank,
                param.dcomm,
                param.num_ctx_per_comm,
                param.token_window,
                param.prob_window,
                param.sf_window,
                param.dest_window,
                &param.mr_info,
                smem_buffer_ptr);
        }
    } else if (threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size()) {
        dispatch_G2S_warp<
            INTRA_NODE_G2S_GROUP,
            TOKEN_DATA_TYPE,
            cur_smem_t,
            NUM_OF_STAGES,
            NUM_OF_TOKENS_PER_CHUNK,
            MAX_NUM_OF_TOKENS_PER_RANK,
            NUM_LSA_TEAMS,
            LSA_TEAM_SIZE,
            NUM_OF_BLOCKS,
            NUM_PIPELINES,
            FORWARD_DISPATCH,
            HAS_SF>(
            param.rdma_to_attn_map,
            param.attn_input_token,
            param.attn_input_prob,
            param.attn_input_token_scaling_factor,
            param.rdma_inter_node_group_flags,
            param.local_rank,
            param.node_rank,
            param.num_of_tokens_per_rank,
            HIDDEN_DIM,
            SF_BYTES_PER_TOKEN,
            param.experts_per_rank,
            *param.expected_rdma_flag_value,
            param.dcomm,
            param.num_ctx_per_comm,
            param.gin_base_ptr,
            &param.mr_info,
            smem_buffer_ptr);
    } else if (
        threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()) {
        dispatch_S2G_warp<
            INTRA_NODE_S2G_GROUP,
            TOKEN_DATA_TYPE,
            cur_smem_t,
            NUM_OF_STAGES,
            NUM_OF_IN_FLIGHT_S2G,
            NUM_OF_TOKENS_PER_CHUNK,
            NUM_LSA_TEAMS,
            LSA_TEAM_SIZE,
            NUM_OF_BLOCKS,
            NUM_PIPELINES,
            FORWARD_DISPATCH,
            HAS_SF,
            kLayout>(
            param.rdma_to_attn_map,
            param.sparse_to_dense_map,
            param.expert_output_token,
            param.expert_output_prob,
            param.expert_output_scaling_factor,
            param.node_rank,
            param.num_of_tokens_per_rank,
            HIDDEN_DIM,
            SF_BYTES_PER_TOKEN,
            param.experts_per_rank,
            param.local_dup_enabled,
            smem_buffer_ptr);
    } else if (
        PAD_GROUP::size() > 0 && threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() +
                                                       INTRA_NODE_S2G_GROUP::size() + PAD_GROUP::size()) {
        // PAD warp: zero-init expert-major alignment padding slots concurrently with S2G.
        // No barrier needed against S2G — padding rows live past the actual token rows
        // in each expert's zone, so the two warps target disjoint global memory.
        PAD_warp_group_device_function<PAD_GROUP, TOKEN_DATA_TYPE>(
            param.expert_output_token[param.local_rank],
            param.pad_actual_counts,
            param.pad_expert_token_offsets,
            param.experts_per_rank,
            param.pad_alignment,
            HIDDEN_DIM,
            NUM_OF_BLOCKS,
            smem_buffer_ptr);
    }
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    if (threadIdx.x % 32 == 0) {
        constexpr int _WT_WARPS = (INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() +
                                   INTRA_NODE_S2G_GROUP::size() + PAD_GROUP::size()) /
                                  32;
        int _warp_id = threadIdx.x / 32;
        int _idx = blockIdx.x * _WT_WARPS + _warp_id;
        param.warp_timing[_idx].start_clock = _wt_start;
        param.warp_timing[_idx].end_clock = clock64();
    }
#endif

    // ===== FUSED DEVICE SYNC (dispatch tail) =====
    if (elect_last_block(reinterpret_cast<const int*>(param.dispatch_grid_barrier_counter), NUM_OF_BLOCKS)) {
        using tail_completion_warp = warp_group<1, 0>; // warp 0
        using tail_rdma_warp = warp_group<1, 1>; // warp 1
        const int tail_tid = (int)threadIdx.x;
        if (tail_tid < tail_completion_warp::size()) {
            // warp 0 (thread 0): inter-rank completion barrier, then reset + bump the intra-node round
            // (local_dup defers that bump to its own tail).
            if (tail_tid == 0) {
                const uint32_t expected_val = *param.expected_intra_node_flag_value;
                nccl_ep::red_add_release_sys_global(param.intra_node_write_completion_flags, 1u);
                uint32_t flag_data;
                do {
                    flag_data = nccl_ep::ld_relaxed_sys_global(param.intra_node_write_completion_flags);
                } while (flag_data != expected_val);
                nccl_ep::memory_fence();
                atomicExch((unsigned int*)param.dispatch_grid_barrier_counter, 0u);
                if (!param.local_dup_enabled)
                    *param.expected_intra_node_flag_value += static_cast<uint32_t>(param.num_of_ranks_per_node);
            }
        } else if (tail_tid < tail_completion_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the inter-node RDMA guard + bump the RDMA round.
            if constexpr (NUM_LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_rdma_flag_value;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        param.dcomm,
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(param.node_rank) * sizeof(uint64_t),
                        param.node_rank,
                        NUM_LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_rdma_flag_value = expected + 1ull;
            }
        }
    }
}

template < // This type represent intra-node reduction warp group.
  typename INTRA_NODE_RED_GROUP,
  // This type represent inter-node reduction warp group.
  typename INTER_NODE_RED_GROUP,
  // This type represent intra-node G2S warp group.
  typename INTRA_NODE_G2S_GROUP,
  // This type represent inter-node G2S warp group.
  typename INTER_NODE_G2S_GROUP,
  // This type represent inter-node rdma warp group.
  typename INTER_NODE_RDMA_GROUP,
  // Number of independent data pipeline per CUDA block.
  int NUM_OF_DATA_PIPELINE_PER_BLOCK,
  // Number of token entry in the shared memory for G2S operations.
  int NUM_OF_STAGES_G2S,
  // Number of token entry in the shared memory for S2G operations.
  int NUM_OF_STAGES_S2G,
  // Number of token per group in the inter-node reduction/G2S warp group.
  int NUM_OF_TOKENS_PER_GROUP,
  // Size of each chunk.
  int NUM_OF_TOKENS_PER_CHUNK,
  // Model configuration.
  int MAX_NUM_OF_TOKENS_PER_RANK, int NUM_LSA_TEAMS,
  // Number of CUDA block running dispatch kernel.
  int NUM_OF_BLOCKS,
  // Number of fully in-flight S2G in intra-node reduction warp group.
  int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
  // Whether the combine kernel is used in backward process. If so, need to transfer the prob for each token as well.
  bool BACKWARD_COMBINE, int HIDDEN_DIM, int LSA_TEAM_SIZE, ncclEpLayout_t kLayout,
  // NONE output dtype, resolved at compile time (JIT literal) so the per-element
  // reduction branches fold away — BF16 (default) pays zero dtype-branch cost.
  ncclDataType_t kTokenDtype = ncclBfloat16>
// Each CUDA block of combine kernel has 5 warp groups and has the following layout:
// 1. intra-node reduction warp group(4 warps, only valid for multinode scenario). 2. inter-node reduction warp group(4 warps, 1 pipeline for multinode scenario, 2 pipeline otherwise).
// 3. intra-node G2S warp group(1 warp, only valid for multinode scenario). 4. inter-node G2S warp group(1 warp for multinode scenario, 2 warps otherwise). 5. inter-node N2N rdma warp group(1 warp, only valid for multinode scenario).
// Total 6(single-node) or 11(multi-node) warps per CUDA block/SM.
__device__ __forceinline__ void combine_kernel_impl(const combine_kernel_param_t<LSA_TEAM_SIZE>& param,
                                                    uint8_t* smem_bytes) {
    // Compile-time check (only enforce for multi-node layout).
    if constexpr (NUM_LSA_TEAMS != 1) {
        static_assert(
            INTRA_NODE_G2S_GROUP::size() == 32,
            "Combine kernel only support 1 INTRA_NODE_G2S warp currently.");
        static_assert(
            INTER_NODE_G2S_GROUP::size() == 32,
            "Combine kernel only support 1 INTER_NODE_G2S warp currently.");
    }
    // The token and its properties should meet size and alignment requirement.
    // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
    // We need to add padding or not using TMA for prob, if we want to support other scenario.
    // assert((param.experts_per_rank * param.num_of_ranks_per_node * sizeof(float)) % 16 == 0);
    static_assert((HIDDEN_DIM % 2) == 0, "HIDDEN_DIM must be even for BF16x2.");
    static_assert((HIDDEN_DIM * sizeof(uint16_t)) % 16 == 0, "HIDDEN_DIM must satisfy TMA alignment.");
    static_assert(
        MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0,
        "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
    constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    constexpr int _WT_WARPS =
        (INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() +
         INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size()) /
        32;
    long long _wt_head_start = 0;
    long long _wt_head_end = 0;
#endif

    // Shared memory used over 48KB, should use dynamic shared memory.
    using cur_smem_t = combine_smem_layout_t;

    // Initialize the layout struct (each thread has its own copy in registers)
    cur_smem_t smem_layout;
    model_config_t c_model;
    c_model.hidden_dim = HIDDEN_DIM;
    c_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    c_model.num_of_experts_per_rank = param.experts_per_rank;
    c_model.num_of_ranks_per_node = param.num_of_ranks_per_node;
    c_model.num_of_nodes = NUM_LSA_TEAMS;
    // Layout derives the element width from kTokenDtype (FP32 doubles the per-stage
    // token-buffer bytes vs BF16/FP16).
    create_combine_smem_layout<kTokenDtype>(
        smem_layout,
        smem_bytes,
        NUM_OF_STAGES_G2S,
        NUM_OF_STAGES_S2G,
        NUM_OF_TOKENS_PER_CHUNK,
        BACKWARD_COMBINE,
        c_model);
    smem_layout.s2d_inner_dim = param.s2d_inner_dim;
    cur_smem_t* smem_buffer_ptr = &smem_layout;

    // ===== FUSED DEVICE SYNC (combine head) =====
    using head_init_warp = warp_group<1, 0>; // warp 0
    using head_rdma_warp = warp_group<1, 1>; // warp 1
    const int head_tid = (int)threadIdx.x;
    if (head_tid < head_init_warp::size()) {
        if (head_tid == 0) {
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
            _wt_head_start = clock64();
#endif
            // Inter-rank completion barrier: block 0 signals (red.release orders prior stores), every block polls.
            if (blockIdx.x == 0) nccl_ep::red_add_release_sys_global(param.intra_node_write_completion_flags, 1u);
            const uint32_t expected_val = *param.expected_intra_node_flag_value;
            uint32_t flag_data;
            do {
                flag_data = nccl_ep::ld_relaxed_sys_global(param.intra_node_write_completion_flags);
            } while (flag_data != expected_val);
            nccl_ep::memory_fence();
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
            _wt_head_end = clock64();
            param.block_timing[blockIdx.x].head_sync_start_clock = _wt_head_start;
            param.block_timing[blockIdx.x].head_sync_end_clock = _wt_head_end;
#endif
            // mbarrier init (both producer/consumer arrival counts = 1).
            for (int i = 0; i < NUM_OF_STAGES_G2S; i++) {
                if constexpr (NUM_LSA_TEAMS != 1) {
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_G2S_buffer + 2 * i, 1);
                    cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_G2S_buffer + 2 * i + 1, 1);
                }
                cuda::ptx::mbarrier_init(smem_buffer_ptr->inter_node_mbarrier_G2S_buffer + 2 * i, 1);
                cuda::ptx::mbarrier_init(smem_buffer_ptr->inter_node_mbarrier_G2S_buffer + 2 * i + 1, 1);
            }
            if constexpr (NUM_LSA_TEAMS != 1) {
                for (int i = 0; i < NUM_LSA_TEAMS - 1; i++)
                    for (int j = 0; j < MAX_NUM_OF_CHUNKS_PER_RANK; j++)
                        cuda::ptx::mbarrier_init(
                            smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer + i * MAX_NUM_OF_CHUNKS_PER_RANK + j,
                            1);
                *(smem_buffer_ptr->rdma_streaming_counter) = 0u;
            }
            cuda::ptx::fence_proxy_async(
                cuda::ptx::space_shared); // make mbarrier init visible to the async (TMA) proxy
        }
    } else if (head_tid < head_init_warp::size() + head_rdma_warp::size()) {
        // warp 1: inter-node RDMA guard wait.
        if constexpr (NUM_LSA_TEAMS != 1) {
            if (param.guard_enabled)
                warp_rdma_guard_wait(
                    reinterpret_cast<const uint64_t*>(
                        reinterpret_cast<const uint8_t*>(param.gin_base_ptr) + param.mr_info.guard_offset),
                    param.node_rank,
                    NUM_LSA_TEAMS,
                    *param.expected_rdma_flag_value);
        }
    }

    // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
    __syncthreads();

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    // Measure warp-group work only (starts after combine-head sync and setup barriers).
    long long _wt_start = 0;
    if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif

    // Now warps can become specialized.
    // The input warp group data type must match the warp groups layout.
    // To prevent compiler generate pointless comparison warning.
    int threadIdx_x_int = (int)threadIdx.x;
    if (threadIdx_x_int < INTRA_NODE_RED_GROUP::size()) {
        if constexpr (NUM_LSA_TEAMS != 1) {
            // Intra-node reduction warp group.
            intra_node_red_warp_group_device_function<
                INTRA_NODE_RED_GROUP,
                cur_smem_t,
                NUM_OF_STAGES_G2S,
                NUM_OF_STAGES_S2G,
                NUM_OF_TOKENS_PER_CHUNK,
                MAX_NUM_OF_TOKENS_PER_RANK,
                NUM_LSA_TEAMS,
                NUM_OF_BLOCKS,
                NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
                BACKWARD_COMBINE,
                HIDDEN_DIM,
                LSA_TEAM_SIZE,
                kTokenDtype>(
                param.node_rank,
                param.num_of_tokens_per_rank,
                param.num_of_ranks_per_node,
                param.rdma_to_attn_map,
                param.rdma_intra_node_red_token,
                param.rdma_intra_node_red_prob,
                smem_buffer_ptr,
                param.experts_per_rank);
        }
    } else if (threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size()) {
        // Inter-node reduction warp group.
        inter_node_red_warp_group_device_function<
            cur_smem_t,
            INTER_NODE_RED_GROUP,
            NUM_OF_DATA_PIPELINE_PER_BLOCK,
            NUM_OF_STAGES_G2S,
            NUM_OF_STAGES_S2G,
            NUM_OF_TOKENS_PER_CHUNK,
            NUM_LSA_TEAMS,
            NUM_OF_BLOCKS,
            NUM_OF_TOKENS_PER_GROUP,
            BACKWARD_COMBINE,
            HIDDEN_DIM,
            LSA_TEAM_SIZE,
            kTokenDtype>(
            param.node_rank,
            param.num_of_tokens_per_rank,
            param.num_real_tokens,
            param.num_of_ranks_per_node,
            param.rdma_to_attn_map,
            param.attn_to_rdma_map,
            param.attn_output_token,
            param.attn_output_prob,
            smem_buffer_ptr,
            param.experts_per_rank);
    } else if (
        threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size()) {
        // Intra-node G2S warp group.
        if constexpr (NUM_LSA_TEAMS != 1) {
            combine_warps_G2S_intra<
                cur_smem_t,
                NUM_OF_STAGES_G2S,
                NUM_OF_TOKENS_PER_CHUNK,
                NUM_LSA_TEAMS,
                NUM_OF_BLOCKS,
                BACKWARD_COMBINE,
                HIDDEN_DIM,
                kLayout,
                kTokenDtype>(
                param.node_rank,
                param.num_of_tokens_per_rank,
                param.num_of_ranks_per_node,
                param.rdma_to_attn_map,
                param.sparse_to_dense_map,
                param.expert_input_token,
                param.expert_input_prob,
                smem_buffer_ptr,
                param.experts_per_rank,
                param.combine_local_reduce_enabled);
        }
    } else if (
        threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() +
                              INTER_NODE_G2S_GROUP::size()) {
        // Inter-node G2S warp group.
        combine_warps_G2S_inter<
            cur_smem_t,
            INTER_NODE_G2S_GROUP,
            NUM_OF_STAGES_G2S,
            NUM_OF_TOKENS_PER_CHUNK,
            MAX_NUM_OF_TOKENS_PER_RANK,
            NUM_LSA_TEAMS,
            NUM_OF_BLOCKS,
            NUM_OF_TOKENS_PER_GROUP,
            BACKWARD_COMBINE,
            HIDDEN_DIM,
            LSA_TEAM_SIZE,
            kLayout,
            kTokenDtype>(
            param.local_rank,
            param.node_rank,
            param.num_of_tokens_per_rank,
            param.num_of_ranks_per_node,
            *param.expected_rdma_flag_value,
            param.rdma_to_attn_map,
            param.attn_to_rdma_map,
            param.sparse_to_dense_map,
            param.expert_input_token,
            param.expert_input_prob,
            param.rdma_inter_node_group_token,
            param.rdma_inter_node_group_prob,
            param.dcomms,
            param.signals_base,
            param.combine_signal_offset,
            param.num_gin_comms,
            param.num_ctx_per_comm,
            param.rdma_inter_node_group_flags,
            smem_buffer_ptr,
            param.experts_per_rank,
            param.combine_local_reduce_enabled);
    } else if (
        threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() +
                              INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size()) {
        // Inter-node rdma warp group.
        if constexpr (NUM_LSA_TEAMS != 1) {
            inter_node_N2N_warp_group_device_function<
                INTER_NODE_RDMA_GROUP,
                cur_smem_t,
                NUM_OF_STAGES_S2G,
                NUM_OF_TOKENS_PER_CHUNK,
                MAX_NUM_OF_TOKENS_PER_RANK,
                NUM_LSA_TEAMS,
                NUM_OF_BLOCKS,
                BACKWARD_COMBINE,
                HIDDEN_DIM,
                kTokenDtype>(
                param.local_rank,
                param.node_rank,
                param.num_of_tokens_per_rank,
                param.num_of_ranks_per_node,
                param.rdma_to_attn_map,
                param.dcomms,
                param.token_window,
                param.prob_window,
                param.dest_window,
                param.num_gin_comms,
                param.num_ctx_per_comm,
                param.gin_base_ptr,
                param.signals_base,
                param.combine_signal_offset,
                &param.mr_info,
                smem_buffer_ptr,
                param.experts_per_rank);
        }
    } else {
        // Too many threads, should not goes here.
    }
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    if (threadIdx.x % 32 == 0) {
        int _warp_id = threadIdx.x / 32;
        int _idx = blockIdx.x * _WT_WARPS + _warp_id;
        param.warp_timing[_idx].work_start_clock = _wt_start;
        param.warp_timing[_idx].work_end_clock = clock64();
    }
#endif

    if (elect_last_block(reinterpret_cast<const int*>(param.combine_grid_barrier_counter), NUM_OF_BLOCKS)) {
        using tail_reset_warp = warp_group<1, 0>; // warp 0
        using tail_rdma_warp = warp_group<1, 1>; // warp 1
        using tail_lsa_warp = warp_group<1, 2>; // warp 2
        static_assert(
            INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() +
                    INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size() >=
                3 * 32,
            "combine tail needs 3 warps");
        const int tail_tid = (int)threadIdx.x;
        if (tail_tid < tail_reset_warp::size()) {
            // warp 0: reset the grid counter + bump the intra-node round.
            if (tail_tid == 0) {
                atomicExch((unsigned int*)param.combine_grid_barrier_counter, 0u);
                *param.expected_intra_node_flag_value += static_cast<uint32_t>(param.num_of_ranks_per_node);
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size()) {
            // warp 1: publish the inter-node RDMA guard, then bump the RDMA round.
            if constexpr (NUM_LSA_TEAMS != 1) {
                const uint64_t expected = *param.expected_rdma_flag_value;
                if (param.guard_enabled)
                    warp_rdma_guard_publish(
                        param.dcomms[0],
                        param.dest_window,
                        param.mr_info.guard_offset + static_cast<size_t>(param.node_rank) * sizeof(uint64_t),
                        param.node_rank,
                        NUM_LSA_TEAMS,
                        expected);
                if (tail_rdma_warp::thread_rank() == 0) *param.expected_rdma_flag_value = expected + 1ull;
            }
        } else if (tail_tid < tail_reset_warp::size() + tail_rdma_warp::size() + tail_lsa_warp::size()) {
            // warp 2: intra-node LSA WAR barrier. Relaxed -- the tail __syncthreads already drained this
            // round's reads. Index = dispatch's block count (disjoint from dispatch's per-block [0, NB)).
            if constexpr (LSA_TEAM_SIZE != 1) {
                if (param.guard_enabled) {
                    ncclLsaBarrierSession<ncclCoopWarp> bar(
                        ncclCoopWarp(),
                        param.dcomms[0],
                        ncclTeamTagLsa(),
                        (uint32_t)HYBRIDEP_DISPATCH_NUM_OF_BLOCKS);
                    bar.sync(ncclCoopWarp(), cuda::memory_order_relaxed);
                }
            }
        }
    }
}

// ============================================================================
// Fills secondary EM slots from the primary slot in this rank's recv token
// buffer after dispatch (EM-unfused mode only).
// ============================================================================

template <typename T>
struct local_dup_kernel_param_t {
    T* expert_output_token; // [max_recv_tokens, hidden]
    float* expert_output_prob; // [max_recv_tokens, epr * num_of_ranks_per_node]; valid iff FORWARD_DISPATCH
    const int32_t* emuf_group_buf; // [num_groups, group_stride] = [primary, sec0, ..., -1]
    const int32_t* emuf_group_count; // scalar (produced by scan)
    int emuf_group_stride; // = experts_per_rank
    // S2G-completion flag dispatch polls; local_dup re-polls before reading primaries.
    const uint32_t* intra_node_write_completion_flag;
    // Shared with dispatch. When local_dup_enabled, dispatch defers the bump
    // here so peers only observe the flag move after secondaries are filled.
    uint32_t* expected_intra_node_flag_value;
    // Reused from dispatch_grid_barrier_counter (dispatch leaves it at 0).
    uint32_t* grid_barrier_counter;
    int experts_per_rank;
    int num_of_ranks_per_node;
};

// Dynamic shared-memory bytes required by local_dup_kernel_impl for the given
// hidden_dim and token element size (token dtype is BF16/uint16_t).
inline int local_dup_dynamic_smem_bytes(
    int hidden_dim,
    int pipe_depth,
    bool forward_dispatch,
    int experts_per_rank,
    int num_of_ranks_per_node,
    size_t token_elem_bytes) {
    const int token_bytes = hidden_dim * static_cast<int>(token_elem_bytes);
    const int prob_bytes =
        forward_dispatch ? experts_per_rank * num_of_ranks_per_node * static_cast<int>(sizeof(float)) : 0;
    const int rings = pipe_depth * (token_bytes + prob_bytes);
    const int mbar_bytes = pipe_depth * 2 * static_cast<int>(sizeof(uint64_t)) + 8;
    return rings + mbar_bytes;
}

// TODO: FP8 token duplication is not yet supported.
template <typename T, int HIDDEN_DIM, int PIPE_DEPTH, bool FORWARD_DISPATCH>
__device__ __forceinline__ void local_dup_kernel_impl(const local_dup_kernel_param_t<T>& p) {
    // Wait until all peers have signaled S2G completion on this rank's recv buffer.
    // Use >= rather than == so a future code path that overshoots the counter
    // (e.g. extra peer arrivals) doesn't hang.
    if (threadIdx.x == 0) {
        const uint32_t expected_val = *p.expected_intra_node_flag_value;
        uint32_t v;
        do {
            v = nccl_ep::ld_relaxed_sys_global(p.intra_node_write_completion_flag);
        } while (v < expected_val);
        nccl_ep::memory_fence();
    }
    __syncthreads();

    constexpr int kTokenBytes = HIDDEN_DIM * sizeof(T);
    const int prob_floats = FORWARD_DISPATCH ? (p.experts_per_rank * p.num_of_ranks_per_node) : 0;
    const int prob_bytes = prob_floats * static_cast<int>(sizeof(float));

    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* smem_ptr = smem_raw;
    T* smem_token[PIPE_DEPTH];
    float* smem_prob[PIPE_DEPTH];
#pragma unroll
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        smem_token[s] = reinterpret_cast<T*>(smem_ptr);
        smem_ptr += kTokenBytes;
    }
    if constexpr (FORWARD_DISPATCH) {
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            smem_prob[s] = reinterpret_cast<float*>(smem_ptr);
            smem_ptr += prob_bytes;
        }
    }
    smem_ptr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_ptr) + 7) & ~uintptr_t(7));
    uint64_t* prod_mbar = reinterpret_cast<uint64_t*>(smem_ptr);
    uint64_t* cons_mbar = prod_mbar + PIPE_DEPTH;

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    if (threadIdx.x == 0) {
#pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            cuda::ptx::mbarrier_init(&prod_mbar[s], 1);
            cuda::ptx::mbarrier_init(&cons_mbar[s], 1);
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    __shared__ int s_group_count;
    if (threadIdx.x == 0) s_group_count = *p.emuf_group_count;
    __syncthreads();

    const int N = s_group_count;
    if (N == 0) {
        // Dispatch deferred the bump; still owe peers the flag advance.
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t arrived = static_cast<uint32_t>(
                nccl_ep::atomic_add_acqrel_global(reinterpret_cast<const int*>(p.grid_barrier_counter), 1));
            if (arrived == gridDim.x - 1) {
                atomicExch((unsigned int*)p.grid_barrier_counter, 0u);
                *p.expected_intra_node_flag_value += static_cast<uint32_t>(p.num_of_ranks_per_node);
            }
        }
        return;
    }

    const int block_id = blockIdx.x;
    const int n_blocks = gridDim.x;
    const int group_stride = p.emuf_group_stride;
    const uint32_t total_tx = static_cast<uint32_t>(kTokenBytes + prob_bytes);

    if (warp_id == 0) {
        // Producer (G2S): 1 TMA load of the primary token per group.
        int stage = 0;
        uint32_t consumer_parity = 1;
        int iters_done = 0;
        for (int i = block_id; i < N; i += n_blocks) {
            if (iters_done >= PIPE_DEPTH) {
                while (!cuda::ptx::mbarrier_try_wait_parity(&cons_mbar[stage], consumer_parity)) {
                }
            }
            if (lane == 0) {
                const int primary_em = p.emuf_group_buf[i * group_stride + 0];
                const T* src_token = p.expert_output_token + static_cast<size_t>(primary_em) * HIDDEN_DIM;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_shared,
                    cuda::ptx::space_global,
                    smem_token[stage],
                    src_token,
                    kTokenBytes,
                    &prod_mbar[stage]);
                if constexpr (FORWARD_DISPATCH) {
                    const float* src_prob = p.expert_output_prob + static_cast<size_t>(primary_em) *
                                                                       (p.experts_per_rank * p.num_of_ranks_per_node);
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_shared,
                        cuda::ptx::space_global,
                        smem_prob[stage],
                        src_prob,
                        prob_bytes,
                        &prod_mbar[stage]);
                }
                cuda::ptx::mbarrier_arrive_expect_tx(
                    cuda::ptx::sem_release,
                    cuda::ptx::scope_cta,
                    cuda::ptx::space_shared,
                    &prod_mbar[stage],
                    total_tx);
            }
            iters_done++;
            stage++;
            if (stage == PIPE_DEPTH) {
                stage = 0;
                consumer_parity ^= 1;
            }
        }
    } else if (warp_id == 1) {
        // Consumer (S2G): fan the primary stage out to every secondary in the row.
        int stage = 0;
        uint32_t producer_parity = 0;
        for (int i = block_id; i < N; i += n_blocks) {
            while (!cuda::ptx::mbarrier_try_wait_parity(&prod_mbar[stage], producer_parity)) {
            }
            if (lane == 0) {
                const int32_t* row = p.emuf_group_buf + static_cast<size_t>(i) * group_stride;
                for (int s = 1; s < group_stride; s++) {
                    const int sec = row[s];
                    if (sec < 0) break;
                    T* dst_token = p.expert_output_token + static_cast<size_t>(sec) * HIDDEN_DIM;
                    cuda::ptx::cp_async_bulk(
                        cuda::ptx::space_global,
                        cuda::ptx::space_shared,
                        dst_token,
                        smem_token[stage],
                        kTokenBytes);
                    if constexpr (FORWARD_DISPATCH) {
                        float* dst_prob = p.expert_output_prob +
                                          static_cast<size_t>(sec) * (p.experts_per_rank * p.num_of_ranks_per_node);
                        cuda::ptx::cp_async_bulk(
                            cuda::ptx::space_global,
                            cuda::ptx::space_shared,
                            dst_prob,
                            smem_prob[stage],
                            prob_bytes);
                    }
                }
                cuda::ptx::cp_async_bulk_commit_group();
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<PIPE_DEPTH - 1>{});
                cuda::ptx::mbarrier_arrive(&cons_mbar[stage]);
            }
            stage++;
            if (stage == PIPE_DEPTH) {
                stage = 0;
                producer_parity ^= 1;
            }
        }
        // Drain S2G before peers observe the flag bump below.
        if (lane == 0) {
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            nccl_ep::fence_proxy_async();
        }
    }

    // Last block owns the flag bump so peers only see it after secondaries land.
    __syncthreads();
    __threadfence();
    if (threadIdx.x == 0) {
        uint32_t arrived = static_cast<uint32_t>(
            nccl_ep::atomic_add_acqrel_global(reinterpret_cast<const int*>(p.grid_barrier_counter), 1));
        if (arrived == gridDim.x - 1) {
            atomicExch((unsigned int*)p.grid_barrier_counter, 0u);
            *p.expected_intra_node_flag_value += static_cast<uint32_t>(p.num_of_ranks_per_node);
        }
    }
}

// ============================================================================
// Pre-sums secondary EM slots into the primary slot in expert_input_token
// (plus expert_input_prob for BACKWARD_COMBINE). Runs before combine in
// EM-unfused mode.
// ============================================================================

template <typename T>
struct local_reduce_kernel_param_t {
    T* expert_input_token; // [max_recv_tokens, hidden]
    float* expert_input_prob; // [max_recv_tokens, epr * num_of_ranks_per_node]; valid iff BACKWARD_COMBINE
    const int32_t* emuf_group_buf; // [num_groups, group_stride] = [primary, sec0, ..., -1]
    const int32_t* emuf_group_count; // scalar
    int emuf_group_stride; // = experts_per_rank
    int experts_per_rank;
    int num_of_ranks_per_node;
};

// Dynamic shared-memory bytes required by local_reduce_kernel_impl for the
// given hidden_dim (token dtype is BF16/uint16_t).
inline int local_reduce_dynamic_smem_bytes(int hidden_dim, int token_elem_bytes) {
    constexpr int kPipeDepth = NCCLEP_LOCAL_REDUCE_PIPE_DEPTH;
    constexpr int kOutStages = NCCLEP_LOCAL_REDUCE_OUT_STAGES;
    return (kPipeDepth + kOutStages) * hidden_dim * token_elem_bytes +
           2 * kPipeDepth * static_cast<int>(sizeof(uint64_t)) + 8;
}

template <typename T, int HIDDEN_DIM, int BLOCK_DIM, bool BACKWARD_COMBINE, ncclDataType_t kTokenDtype = ncclBfloat16>
__device__ __forceinline__ void local_reduce_kernel_impl(const local_reduce_kernel_param_t<T>& p) {
    static_assert(HIDDEN_DIM % 8 == 0, "HIDDEN_DIM must be a multiple of 8 (uint4 = 8 elems @2 B / 4 @4 B per vector)");
    static_assert(BLOCK_DIM % 32 == 0 && BLOCK_DIM >= 64, "BLOCK_DIM must be a multiple of 32 and at least 2 warps");

    // 1 producer warp (parallel G2S over lanes 0..n_src-1) + (W-1) consumer warps
    // (FP32 accumulate, BF16 cast, S2G).
    constexpr int kProdWarpCount = 1;
    constexpr int kWarpCount = BLOCK_DIM / 32;
    constexpr int kConsWarpCount = kWarpCount - kProdWarpCount;
    constexpr int kConsThreads = kConsWarpCount * 32;
    constexpr int kConsBarId = 1;

    // uint4 (16 B) holds 8 elems for 2-byte dtypes, 4 for FP32.
    constexpr int VEC_DIM = HIDDEN_DIM * static_cast<int>(sizeof(T)) / 16;
    constexpr int VEC_PER_THREAD = (VEC_DIM + kConsThreads - 1) / kConsThreads;
    constexpr int kTokenBytes = HIDDEN_DIM * sizeof(T);

    constexpr int PIPE_DEPTH = NCCLEP_LOCAL_REDUCE_PIPE_DEPTH;
    constexpr int kOutStages = NCCLEP_LOCAL_REDUCE_OUT_STAGES;

    const int N = *p.emuf_group_count;
    if (N == 0) return;

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int stride = p.emuf_group_stride;

    // n_src per group must be <= PIPE_DEPTH (also <= 32 since PIPE_DEPTH <= 32);
    // enforced by __trap in the producer loop once n_src is known.
    const int PROB_DIM = BACKWARD_COMBINE ? (p.experts_per_rank * p.num_of_ranks_per_node) : 0;

    // Shmem layout:
    //   s_in[PIPE_DEPTH] x kTokenBytes        (G2S ring)
    //   s_out[kOutStages] x kTokenBytes        (S2G ring)
    //   prod_mbar[PIPE_DEPTH], cons_mbar[PIPE_DEPTH]
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint8_t* smem_ptr = smem_raw;
    T* s_in[PIPE_DEPTH];
#pragma unroll
    for (int s = 0; s < PIPE_DEPTH; ++s) {
        s_in[s] = reinterpret_cast<T*>(smem_ptr);
        smem_ptr += kTokenBytes;
    }
    T* s_out[kOutStages];
#pragma unroll
    for (int s = 0; s < kOutStages; ++s) {
        s_out[s] = reinterpret_cast<T*>(smem_ptr);
        smem_ptr += kTokenBytes;
    }
    smem_ptr = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_ptr) + 7) & ~uintptr_t(7));
    uint64_t* prod_mbar = reinterpret_cast<uint64_t*>(smem_ptr);
    uint64_t* cons_mbar = prod_mbar + PIPE_DEPTH;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < PIPE_DEPTH; ++s) {
            cuda::ptx::mbarrier_init(&prod_mbar[s], 1);
            cuda::ptx::mbarrier_init(&cons_mbar[s], 1);
        }
        cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    }
    __syncthreads();

    const T* token_base = reinterpret_cast<const T*>(p.expert_input_token);
    T* token_base_w = reinterpret_cast<T*>(p.expert_input_token);
    const int n_blocks = gridDim.x;
    const int my_block = blockIdx.x;

    if (warp_id == 0) {
        // PRODUCER WARP (cooperative G2S): lanes 0..n_src-1 issue TMA in parallel for
        // the current group's primary+secondaries. After each group, advance the global
        // stage offset by n_src so the next group's lanes target the next set of stages.
        int global_offset = 0;
        for (int i = my_block; i < N; i += n_blocks) {
            const int32_t* row = p.emuf_group_buf + static_cast<size_t>(i) * stride;
            // Lane 0 scans the row terminator and broadcasts n_src.
            int n_src;
            if (lane == 0) {
                int n = 1; // primary
                for (int s = 1; s < stride; s++) {
                    if (row[s] < 0) break;
                    n++;
                }
                n_src = n;
            }
            n_src = __shfl_sync(0xffffffff, n_src, 0);
            if (lane == 0 && n_src > PIPE_DEPTH) {
                __trap();
            }

            if (lane < n_src) {
                const int absolute = global_offset + lane;
                const int stage = absolute % PIPE_DEPTH;
                const uint32_t parity = 1u ^ (static_cast<uint32_t>(absolute / PIPE_DEPTH) & 1u);
                while (!cuda::ptx::mbarrier_try_wait_parity(&cons_mbar[stage], parity)) {
                }
                const int slot = row[lane]; // row[0]=primary, row[1..n_sec]=secondaries
                const T* src = token_base + static_cast<size_t>(slot) * HIDDEN_DIM;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_shared,
                    cuda::ptx::space_global,
                    s_in[stage],
                    src,
                    kTokenBytes,
                    &prod_mbar[stage]);
                cuda::ptx::mbarrier_arrive_expect_tx(
                    cuda::ptx::sem_release,
                    cuda::ptx::scope_cta,
                    cuda::ptx::space_shared,
                    &prod_mbar[stage],
                    kTokenBytes);
            }
            global_offset += n_src;
        }
    } else {
        // Consumer warps: FP32-accumulate n_src sources, BF16-cast, S2G to primary.
        int stage = 0;
        int absolute = 0;
        int out_slot = 0;
        const int cons_tid = tid - 32;

        for (int i = my_block; i < N; i += n_blocks) {
            const int32_t* row = p.emuf_group_buf + static_cast<size_t>(i) * stride;
            const int primary = row[0];
            int n_sec = 0;
            int secondaries[HYBRIDEP_MAX_LOCAL_EXPERTS_PER_RANK];
#pragma unroll 1
            for (int s = 1; s < stride; s++) {
                int v = row[s];
                if (v < 0) break;
                secondaries[n_sec++] = v;
            }
            const int n_src = 1 + n_sec;

            // FP32 accumulator in registers (4 float2's per uint4 slot -> 8 floats).
            float2 acc[VEC_PER_THREAD][4];
#pragma unroll
            for (int n = 0; n < VEC_PER_THREAD; n++) {
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    acc[n][k].x = 0.f;
                    acc[n][k].y = 0.f;
                }
            }

            for (int k = 0; k < n_src; k++) {
                const uint32_t prod_parity = static_cast<uint32_t>(absolute / PIPE_DEPTH) & 1u;
                while (!cuda::ptx::mbarrier_try_wait_parity(&prod_mbar[stage], prod_parity)) {
                }
                // Consumer-only barrier: producer is racing ahead on later stages.
                arrive_and_wait(kConsThreads, kConsBarId);

                const uint4* in_vec = reinterpret_cast<const uint4*>(s_in[stage]);
#pragma unroll
                for (int n = 0; n < VEC_PER_THREAD; n++) {
                    const int e = n * kConsThreads + cons_tid;
                    if (e < VEC_DIM) {
                        uint4 v = in_vec[e];
                        // uint4 holds 2 FP32 pairs or 4 packed 16-bit pairs; decode each pair
                        // to FP32 and accumulate (the helper picks the per-dtype unpack).
                        constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                        for (int kk = 0; kk < kPairs; kk++) {
                            float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&v, kk);
                            acc[n][kk].x += f.x;
                            acc[n][kk].y += f.y;
                        }
                    }
                }
                arrive_and_wait(kConsThreads, kConsBarId);
                if (tid == 32) {
                    cuda::ptx::mbarrier_arrive(&cons_mbar[stage]);
                }
                absolute++;
                stage++;
                if (stage == PIPE_DEPTH) stage = 0;
            }

            // Drain prior S2G before reusing s_out[out_slot].
            if (tid == 32) {
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<kOutStages - 1>{});
            }
            arrive_and_wait(kConsThreads, kConsBarId);

            uint4* out_vec = reinterpret_cast<uint4*>(s_out[out_slot]);
#pragma unroll
            for (int n = 0; n < VEC_PER_THREAD; n++) {
                const int e = n * kConsThreads + cons_tid;
                if (e < VEC_DIM) {
                    uint4 out;
                    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                    for (int kk = 0; kk < kPairs; kk++) nccl_ep::st_token_pair<kTokenDtype>(&out, kk, acc[n][kk]);
                    out_vec[e] = out;
                }
            }
            cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
            arrive_and_wait(kConsThreads, kConsBarId);

            if (tid == 32) {
                T* dst = token_base_w + static_cast<size_t>(primary) * HIDDEN_DIM;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst,
                    s_out[out_slot],
                    kTokenBytes);
                cuda::ptx::cp_async_bulk_commit_group();
            }
            out_slot ^= 1;

            if constexpr (BACKWARD_COMBINE) {
                float* prim_prob = p.expert_input_prob + static_cast<size_t>(primary) * PROB_DIM;
                for (int e = cons_tid; e < PROB_DIM; e += kConsThreads) {
                    float a = prim_prob[e];
#pragma unroll 1
                    for (int s = 0; s < n_sec; s++) {
                        const float* sec_prob = p.expert_input_prob + static_cast<size_t>(secondaries[s]) * PROB_DIM;
                        a += sec_prob[e];
                    }
                    prim_prob[e] = a;
                }
            }
        }
        if (tid == 32) {
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
            nccl_ep::fence_proxy_async();
        }
    }
}

// Local EM permute kernel (HT + EM + zero_copy != ON). Scatters FLAT staging
// rows into per-expert EM zones with zero padding for inactive slots.
//
// Two concurrent warp groups in each block:
//   - kPermuteWarps token-permute warps: load each FLAT row once into
//     registers and scatter to its EM destinations (load-once-scatter-many,
//     one HBM read amortized over up to top_k STGs). The int4 unroll length
//     HiddenVec is JIT'd per HiddenInt4 (see pick_dup_hidden_vec) to trade
//     ILP against register pressure.
//   - kPadWarps pad-fill warps: cp.async.bulk S2G from a smem zero buffer to
//     per-expert pad rows. TMA path doesn't compete with LDG/STG queues.
// Disjoint output rows + disjoint memory paths => the two groups overlap.
//
// kLocalPermuteMaxExpertsPerRank bounds the per-warp smem active-EM list.
constexpr int kLocalPermuteMaxExpertsPerRank = 256;
constexpr int kLocalPermuteThreadsPerSlot = 32;
constexpr int kLocalPermutePermuteWarps = 8;
constexpr int kLocalPermutePadWarps = 1;
constexpr int kLocalPermuteTokensPerBlock = kLocalPermutePermuteWarps;
constexpr int kLocalPermuteThreads =
    kLocalPermuteThreadsPerSlot * (kLocalPermutePermuteWarps + kLocalPermutePadWarps); // 288
// local_permute_reduce: 128 threads per slot, S slots per block. 8 slots
// (1024 threads) puts 8 independent token loads in flight per block for
// better HBM/L2 latency hiding; single block per SM at this size.
constexpr int kLocalPermuteReduceSlotsPerBlock = 8;
constexpr int kLocalPermuteReduceThreads = 128 * kLocalPermuteReduceSlotsPerBlock;
constexpr int kLocalPermuteReduceBlocksPerSM = 1;

struct local_permute_dup_param_t {
    void* recv_x_em;
    float* recv_topk_weights_em;
    const void* flat_staging;
    const float* recv_topk_weights_flat;
    const int32_t* flat2em_slot_map;
    const int32_t* num_recv_tokens_dev;
    const int64_t* expert_token_offsets;
    const int32_t* per_expert_counts_active;
    int top_k;
    int experts_per_rank;
    int row_bytes;
};

template <int HiddenInt4, int HiddenVec>
__device__ __forceinline__ void local_permute_dup(
    uint8_t* __restrict__ recv_x_em,
    float* __restrict__ recv_topk_weights_em,
    const uint8_t* __restrict__ flat_staging,
    const float* __restrict__ recv_topk_weights_flat,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const int64_t* __restrict__ expert_token_offsets,
    const int32_t* __restrict__ per_expert_counts_active,
    int top_k,
    int experts_per_rank,
    int /*row_bytes*/) {
    constexpr int kThreadsPerSlot = kLocalPermuteThreadsPerSlot;
    constexpr int kHiddenVec = HiddenVec;
    constexpr int kPermuteWarps = kLocalPermutePermuteWarps;
    constexpr int kPadWarps = kLocalPermutePadWarps;
    // Per-warp cap on active EM slots; bounded by top_k and experts_per_rank.
    constexpr int kMaxActivePerToken = kLocalPermuteMaxExpertsPerRank;

    const int warp_id = threadIdx.x / kThreadsPerSlot;
    const int lane = threadIdx.x & (kThreadsPerSlot - 1);
    const bool is_pad = warp_id >= kPermuteWarps;
    const int pad_idx = warp_id - kPermuteWarps;

    const int num_recv = *num_recv_tokens_dev;
    constexpr int row_int4 = HiddenInt4;
    constexpr int row_bytes = HiddenInt4 * 16;

    const int4* src_int4 = reinterpret_cast<const int4*>(flat_staging);
    int4* dst_int4 = reinterpret_cast<int4*>(recv_x_em);

    __shared__ int32_t s_active[kPermuteWarps][kMaxActivePerToken];
    __shared__ int s_count[kPermuteWarps];
    // Source row for the pad warp's cp.async.bulk S2G.
    extern __shared__ int4 s_zero[];

    for (int j = threadIdx.x; j < row_int4; j += blockDim.x) {
        s_zero[j] = int4{0, 0, 0, 0};
    }
    __syncthreads();

    if (is_pad) {
        // 32 lanes stripe pad slots across the grid; each lane issues
        // cp.async.bulk S2G from s_zero to its assigned slots.
        const int total_pad_lanes = kThreadsPerSlot * kPadWarps * static_cast<int>(gridDim.x);
        const int my_pad_lane = (static_cast<int>(blockIdx.x) * kPadWarps + pad_idx) * kThreadsPerSlot + lane;
        // Host validates row_bytes % 16 == 0; pass it straight to cp.async.bulk.
        assert((row_bytes & 15) == 0);
        const bool zero_weights = (recv_topk_weights_em != nullptr);
        for (int e = 0; e < experts_per_rank; ++e) {
            const int64_t zone_start = expert_token_offsets[e];
            const int64_t zone_end = expert_token_offsets[e + 1];
            const int32_t active = per_expert_counts_active[e];
            const int64_t pad_begin = zone_start + active;
            const int64_t pad_count = zone_end - pad_begin;
            for (int64_t offs = my_pad_lane; offs < pad_count; offs += total_pad_lanes) {
                const int64_t slot = pad_begin + offs;
                uint8_t* dst_g = recv_x_em + static_cast<size_t>(slot) * row_bytes;
                cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    dst_g,
                    reinterpret_cast<uint8_t*>(s_zero),
                    row_bytes);
                if (zero_weights) {
                    recv_topk_weights_em[slot] = 0.0f;
                }
            }
        }
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        nccl_ep::fence_proxy_async();
    } else {
        // One warp per token, grid-strided over tokens.
        for (int blk = static_cast<int>(blockIdx.x) * kPermuteWarps; blk < num_recv;
             blk += kPermuteWarps * static_cast<int>(gridDim.x)) {
            const int token = blk + warp_id;
            if (token >= num_recv) continue;

            // Lane 0 packs active em_slots into smem and folds in the
            // topk-weights copy (one scalar store per slot).
            if (lane == 0) {
                const int32_t* slot_row = flat2em_slot_map + static_cast<size_t>(token) * top_k;
                int c = 0;
                const bool copy_weights = recv_topk_weights_em != nullptr && recv_topk_weights_flat != nullptr;
                for (int k = 0; k < top_k; ++k) {
                    const int32_t es = __ldg(slot_row + k);
                    if (es < 0) continue;
                    if (c < kMaxActivePerToken) s_active[warp_id][c] = es;
                    if (copy_weights) {
                        recv_topk_weights_em[es] = recv_topk_weights_flat[static_cast<size_t>(token) * top_k + k];
                    }
                    ++c;
                }
                s_count[warp_id] = c;
            }
            __syncwarp();

            const int4* src = src_int4 + static_cast<size_t>(token) * row_int4;
            const int cnt = s_count[warp_id];

            constexpr int kStride = kThreadsPerSlot * kHiddenVec;
            constexpr int j_main_end = (row_int4 / kStride) * kStride;
            for (int j_base = 0; j_base < j_main_end; j_base += kStride) {
                int4 buf[kHiddenVec];
#pragma unroll
                for (int u = 0; u < kHiddenVec; ++u) {
                    buf[u] = src[j_base + u * kThreadsPerSlot + lane];
                }
                for (int a = 0; a < cnt; ++a) {
                    int4* dst = dst_int4 + static_cast<size_t>(s_active[warp_id][a]) * row_int4;
#pragma unroll
                    for (int u = 0; u < kHiddenVec; ++u) {
                        int4* p = dst + j_base + u * kThreadsPerSlot + lane;
                        nccl_ep::st_cg_global(p, buf[u]);
                    }
                }
            }
            if constexpr (j_main_end < row_int4) {
                for (int j = j_main_end + lane; j < row_int4; j += kThreadsPerSlot) {
                    const int4 v = src[j];
                    for (int a = 0; a < cnt; ++a) {
                        int4* dst = dst_int4 + static_cast<size_t>(s_active[warp_id][a]) * row_int4;
                        nccl_ep::st_cg_global(&dst[j], v);
                    }
                }
            }
            __syncwarp();
        }
    }

    __syncthreads(); // pad TMAs must complete before block exits.
}

// Local EM reduce kernel (inverse of local_permute_dup). Sums the top_k EM
// rows that share a FLAT recv slot and writes the bf16 result back into FLAT
// staging.
struct local_permute_reduce_param_t {
    void* flat_staging;
    const void* recv_x_em;
    const int32_t* flat2em_slot_map;
    const int32_t* num_recv_tokens_dev;
    // Optional fused EM to FLAT weight gather. Both null on FWD (token only).
    const float* em_weights_in;
    float* flat_weights_out;
    int top_k;
    int row_bytes;
};

// Direct-load reduce: each slot's row is reduced by a 128-thread sub-warp;
// with kSlotsPerBlock=8 a block computes 8 slots in parallel. For each int4
// lane the sub-warp's 128 threads accumulate across top_k contributors via
// direct cached global loads, then write the packed bf16 result back to
// flat_staging. HiddenInt4 = row_bytes / 16 is templated so the per-thread
// strided element loop is a compile-time bound.
template <int MaxTopK, int HiddenInt4, ncclDataType_t kTokenDtype = ncclBfloat16>
__device__ __forceinline__ void local_permute_reduce(
    uint8_t* __restrict__ flat_staging,
    const uint8_t* __restrict__ recv_x_em,
    const int32_t* __restrict__ flat2em_slot_map,
    const int32_t* __restrict__ num_recv_tokens_dev,
    const float* __restrict__ em_weights_in,
    float* __restrict__ flat_weights_out,
    int top_k,
    int /*row_bytes*/) {
    constexpr int kRowBytes = HiddenInt4 * 16;

    constexpr int kThreadsPerSlot = 128;
    constexpr int kSlotsPerBlock = kLocalPermuteReduceSlotsPerBlock;
    constexpr int kBlockDim = kThreadsPerSlot * kSlotsPerBlock;
    constexpr int kElemsPerThread = (HiddenInt4 + kThreadsPerSlot - 1) / kThreadsPerSlot;

    // Per-slot packed em_slot ids in smem: only valid contributors (the rest
    // of top_k are -1 from non-local experts). Lets the inner loop iterate
    // n_valid instead of top_k, which is the dominant win at top_k > EPR.
    __shared__ int32_t smem_flat2em_slot_map[kSlotsPerBlock][MaxTopK];
    __shared__ int s_nvalid[kSlotsPerBlock];

    const int tid = threadIdx.x;
    const int slot_in_block = tid / kThreadsPerSlot;
    const int lane = tid % kThreadsPerSlot;

    const int num_recv = *num_recv_tokens_dev;
    const int slot_stride = kSlotsPerBlock * static_cast<int>(gridDim.x);

    for (int s_base = static_cast<int>(blockIdx.x) * kSlotsPerBlock; s_base < num_recv; s_base += slot_stride) {
        const int slot = s_base + slot_in_block;
        const bool slot_valid = (slot < num_recv);

        // Cooperative pack: warp 0 of each slot reads slot_row[lane] in
        // parallel, ballots valid lanes, and packs via warp scan. Requires
        // MaxTopK <= 32 (true for all current configs).
        static_assert(MaxTopK <= 32, "cooperative pack assumes MaxTopK <= 32");
        if (slot_valid && lane < 32) {
            const int32_t* slot_row_global = flat2em_slot_map + static_cast<size_t>(slot) * top_k;
            const int32_t s = (lane < top_k) ? __ldg(slot_row_global + lane) : -1;
            if (em_weights_in != nullptr && lane < top_k) {
                flat_weights_out[static_cast<size_t>(slot) * top_k + lane] = (s >= 0) ? em_weights_in[s] : 0.0f;
            }
            const unsigned valid = __ballot_sync(0xFFFFFFFFu, s >= 0);
            const int my_pos = __popc(valid & ((1u << lane) - 1));
            if (s >= 0) smem_flat2em_slot_map[slot_in_block][my_pos] = s;
            if (lane == 0) s_nvalid[slot_in_block] = __popc(valid);
        }
        __syncthreads();

        if (slot_valid) {
            const int n = s_nvalid[slot_in_block];

            int4* dst_int4 = reinterpret_cast<int4*>(flat_staging + static_cast<size_t>(slot) * kRowBytes);

            // Process the per-thread hidden-dim int4 indices in groups of
            // kHiddenVec so each iter has kHiddenVec * n LDGs in flight per
            // thread, hiding per-LDG latency. Cap kHiddenVec at
            // kElemsPerThread (JIT-known from HiddenInt4) so at small hidden
            // the dead u-lanes and their float2 accumulators disappear:
            // H=2048 -> kHiddenVec=2 (vs 4) frees 16 float regs per thread.
            constexpr int kHiddenVec = (kElemsPerThread < 4) ? kElemsPerThread : 4;
            for (int nn_base = 0; nn_base < kElemsPerThread; nn_base += kHiddenVec) {
                int js[kHiddenVec];
                bool valid_u[kHiddenVec];
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    const int nn = nn_base + u;
                    js[u] = lane + nn * kThreadsPerSlot;
                    valid_u[u] = (nn < kElemsPerThread) && (js[u] < HiddenInt4);
                }

                float2 acc[kHiddenVec][4];
#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
#pragma unroll
                    for (int p = 0; p < 4; p++) {
                        acc[u][p].x = 0.0f;
                        acc[u][p].y = 0.0f;
                    }
                }

                for (int k = 0; k < n; k++) {
                    const int32_t em_slot = smem_flat2em_slot_map[slot_in_block][k];
                    const int4* src =
                        reinterpret_cast<const int4*>(recv_x_em + static_cast<size_t>(em_slot) * kRowBytes);
                    int4 buf[kHiddenVec];
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (valid_u[u]) buf[u] = src[js[u]];
                    }
#pragma unroll
                    for (int u = 0; u < kHiddenVec; u++) {
                        if (!valid_u[u]) continue;
                        // int4 holds 2 FP32 pairs or 4 packed 16-bit pairs; decode
                        // each pair to FP32 and accumulate.
                        constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                        for (int p = 0; p < kPairs; p++) {
                            float2 f = nccl_ep::ld_token_pair<kTokenDtype>(&buf[u], p);
                            acc[u][p].x += f.x;
                            acc[u][p].y += f.y;
                        }
                    }
                }

#pragma unroll
                for (int u = 0; u < kHiddenVec; u++) {
                    if (!valid_u[u]) continue;
                    int4 out;
                    constexpr int kPairs = nccl_ep::pairs_per_int4<kTokenDtype>();
#pragma unroll
                    for (int p = 0; p < kPairs; p++) {
                        nccl_ep::st_token_pair<kTokenDtype>(&out, p, acc[u][p]);
                    }
                    // Keep the FLAT recv row in L2 for the host-side D2D
                    // that reads it next.
                    nccl_ep::st_cg_global(&dst_int4[js[u]], out);
                }
            }
        }
        __syncthreads();
    }
}

} // namespace hybrid_ep

#include "scan_kernel.cuh"
