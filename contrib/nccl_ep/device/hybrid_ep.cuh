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
#include "hybridep_configs.cuh"
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda/ptx>
#include <nccl.h>
#include "nccl_device.h"
#include "cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x
#include "include/common.hpp"

namespace hybrid_ep{


// MAX_RANKS_PER_NODE is now the LSA_TEAM_SIZE template parameter on hybrid_ep.
// Removed as a free-standing constant; each instantiation sizes arrays exactly.

// Register head size for combine (BF16 tokens). Tail (if any) accumulates in shared memory.
constexpr int COMBINE_REG_HEAD_HIDDEN_DIM = 4096;
static_assert((COMBINE_REG_HEAD_HIDDEN_DIM % 2) == 0,
              "COMBINE_REG_HEAD_HIDDEN_DIM must be even for BF16x2.");
static_assert(COMBINE_REG_HEAD_HIDDEN_DIM <= MAX_HIDDEN_DIM,
              "COMBINE_REG_HEAD_HIDDEN_DIM must not exceed MAX_HIDDEN_DIM.");

template<int NUM_OF_BOOL_TO_REDUCE>
using Reduce_t =
  typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 8 == 0, uint64_t,
    typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 4 == 0, uint32_t,
      typename std::conditional<NUM_OF_BOOL_TO_REDUCE % 2 == 0, uint16_t, uint8_t
      >::type
    >::type
  >::type;

template<int NUM_OF_BYTES_TO_COPY>
using Copy_t =
  typename std::conditional<NUM_OF_BYTES_TO_COPY % 16 == 0, uint4,
    typename std::conditional<NUM_OF_BYTES_TO_COPY % 8 == 0, uint2,
      typename std::conditional<NUM_OF_BYTES_TO_COPY % 4 == 0, uint32_t,
        typename std::conditional<NUM_OF_BYTES_TO_COPY % 2 == 0, uint16_t, uint8_t
        >::type
      >::type
    >::type
  >::type;

// Conditionally allocate compile-time arrays only when enabled.
template<bool ENABLE, int N>
struct acc_prob_storage_t {};

template<int N>
struct acc_prob_storage_t<true, N> {
  float data[N];
};

enum scan_state{
  EMPTY = 0,
  PRIV_SUM = 1
};

struct tmp_state_t{
  scan_state state;
  int32_t value;
};

// Generic warp group for warp-specializaion.
template<int NUM_WARPS,
         int STARTING_WARPS>
struct warp_group{
  __host__ __device__ static constexpr int size() { return 32 * NUM_WARPS; }
  __host__ __device__ static constexpr int warp_size() { return NUM_WARPS; }

  __host__ __device__ static int thread_rank() { return threadIdx.x - (32 * STARTING_WARPS); }
  __host__ __device__ static int warp_rank() { return thread_rank() / 32; }
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
  size_t bytes_per_entry;                    // Size of packed entry (token + prob + sf)
  size_t max_tokens_per_dest;                // Max tokens that can be staged per destination
  // Streaming RDMA signals
  unsigned signals_tail_base;               // Base signal ID for tail tracking (sender -> receiver)
  // Streaming buffer configuration
  int num_max_rdma_chunked_send_tokens;     // Batch size per RDMA put (default: 6)
} __attribute__((__aligned__(8)));

struct combine_memory_region_info_t {
  size_t rdma_intra_node_red_token_offset;        // Offset of intra-node reduced token buffer
  size_t combine_rdma_inter_node_group_token_offset; // Offset of combine rdma token buffer
  size_t rdma_intra_node_red_prob_offset;         // Offset of intra-node reduced prob buffer
  size_t combine_rdma_inter_node_group_prob_offset;  // Offset of combine rdma prob buffer
} __attribute__((__aligned__(8)));

// ============================================================================
// Warp-parallel memory copy helper for RDMA staging
// All 32 threads participate using int4 (16-byte) loads/stores for maximum bandwidth
// ============================================================================
template<int STRIDE = 32>
__device__ __forceinline__ void warp_copy_int4(
    void* __restrict__ dst,
    const void* __restrict__ src,
    size_t bytes,
    int lane_id)
{
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
    while (atomicCAS(lock, 0, 1) != 0) {}
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
  int token_data_type;  // 0 = uint8_t (FP8), 1 = uint16_t (BF16)
  int num_pipelines;
  int stages_per_pipeline;
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
struct combine_smem_layout_t {
  uint16_t* intra_node_token_G2S_buffer;
  uint16_t* intra_node_token_S2G_buffer;
  float2* intra_node_token_tail_S2G_buffer;
  uint16_t* inter_node_token_G2S_buffer;
  uint16_t* inter_node_token_S2G_buffer;
  float2* inter_node_token_tail_S2G_buffer;
  float* intra_node_prob_G2S_buffer;
  float* intra_node_prob_S2G_buffer;
  float* inter_node_prob_G2S_buffer;
  float* inter_node_prob_S2G_buffer;
  uint64_t* intra_node_mbarrier_G2S_buffer;
  uint64_t* inter_node_mbarrier_G2S_buffer;
  uint64_t* intra_node_to_rdma_mbarrier_buffer;
  bool* intra_node_flag_G2S_buffer;
  bool* inter_node_flag_G2S_buffer;

  int token_G2S_stage_stride;  // elements (not bytes)
  int token_S2G_stage_stride;  // elements (not bytes)
  int token_tail_S2G_stage_stride;  // float2 elements (not bytes)
  int prob_G2S_stage_stride;        // elements (not bytes)
  int prob_S2G_stage_stride;        // intra-node elements (not bytes)
  int prob_S2G_inter_stage_stride;  // inter-node elements (not bytes)
  int hidden_dim;              // model hidden dimension
  combine_memory_region_info_t* combine_memory_region_info;

  // Streaming overlap: reduction warp -> RDMA warp within a chunk
  uint32_t* rdma_streaming_counter;   // [1] cumulative tokens produced for the current chunk

  // Accessor methods for staged buffers
  __device__ __forceinline__ uint16_t* get_intra_node_token_G2S(int stage) const {
    return intra_node_token_G2S_buffer + stage * token_G2S_stage_stride;
  }
  __device__ __forceinline__ uint16_t* get_intra_node_token_S2G(int stage) const {
    return intra_node_token_S2G_buffer + stage * token_S2G_stage_stride;
  }
  __device__ __forceinline__ float2* get_intra_node_token_tail_S2G(int stage) const {
    return intra_node_token_tail_S2G_buffer + stage * token_tail_S2G_stage_stride;
  }
  __device__ __forceinline__ uint16_t* get_inter_node_token_G2S(int stage) const {
    return inter_node_token_G2S_buffer + stage * token_G2S_stage_stride;
  }
  __device__ __forceinline__ uint16_t* get_inter_node_token_S2G(int stage) const {
    return inter_node_token_S2G_buffer + stage * token_S2G_stage_stride;
  }
  __device__ __forceinline__ float2* get_inter_node_token_tail_S2G(int stage) const {
    return inter_node_token_tail_S2G_buffer + stage * token_tail_S2G_stage_stride;
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
  float* intra_node_scaling_factor_buffer;
  int32_t* sparse_to_dense_map_buffer;
  bool* attn_to_rdma_map_buffer;
  uint64_t* intra_node_mbarrier_buffer;
  uint64_t* sparse_to_dense_map_mbarrier_buffer;
  uint64_t* S2G_group_mbarrier_buffer;

  int token_buffer_stage_stride;  // bytes
  int prob_buffer_stage_stride;   // bytes
  int sf_buffer_stage_stride;     // bytes
  int s2d_map_stage_stride;       // bytes
  int num_of_ranks_per_node;      // for sparse_to_dense map indexing
  int num_pipelines;
  int stages_per_pipeline;
  dispatch_memory_region_info_t* dispatch_memory_region_info;

  // Flat stage accessors (used when pipeline_id is already folded into stage)
  __device__ __forceinline__ void* get_token_buffer(int stage) const {
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(intra_node_token_buffer) + stage * token_buffer_stage_stride);
  }
  __device__ __forceinline__ float* get_prob_buffer(int stage) const {
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(intra_node_prob_buffer) + stage * prob_buffer_stage_stride);
  }
  __device__ __forceinline__ float* get_sf_buffer(int stage) const {
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(intra_node_scaling_factor_buffer) + stage * sf_buffer_stage_stride);
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
  __device__ __forceinline__ float* get_sf_buffer(int pipeline_id, int local_stage) const {
    return get_sf_buffer(pipeline_id * stages_per_pipeline + local_stage);
  }
  __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_producer(int pipeline_id, int local_stage) const {
    return get_intra_node_mbarrier_producer(pipeline_id * stages_per_pipeline + local_stage);
  }
  __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_consumer(int pipeline_id, int local_stage) const {
    return get_intra_node_mbarrier_consumer(pipeline_id * stages_per_pipeline + local_stage);
  }

  // Per-pipeline s2d_map accessors: each pipeline has its own 2 ping-pong stages
  __device__ __forceinline__ int32_t* get_s2d_map_buffer(int pipeline_id, int stage, int token_idx) const {
    int abs_stage = pipeline_id * 2 + stage;
    return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + abs_stage * s2d_map_stage_stride) + token_idx * num_of_ranks_per_node;
  }
  __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int pipeline_id, int stage) const {
    int abs_stage = pipeline_id * 2 + stage;
    return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + abs_stage * s2d_map_stage_stride);
  }
  // Legacy s2d accessors (pipeline_id=0)
  __device__ __forceinline__ int32_t* get_s2d_map_buffer(int stage, int token_idx) const {
    return get_s2d_map_buffer(0, stage, token_idx);
  }
  __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int stage) const {
    return get_s2d_map_buffer_base(0, stage);
  }

  // Per-pipeline s2d_map mbarrier: each pipeline has 2 ping-pong mbarriers
  __device__ __forceinline__ uint64_t* get_s2d_map_mbar(int pipeline_id, int stage) const {
    return sparse_to_dense_map_mbarrier_buffer + pipeline_id * 2 + stage;
  }
  // Per-pipeline S2G group mbarrier
  __device__ __forceinline__ uint64_t* get_S2G_group_mbar(int pipeline_id) const {
    return S2G_group_mbarrier_buffer + pipeline_id;
  }
};

__device__ dispatch_smem_layout_t create_dispatch_smem_layout(
  dispatch_smem_layout_t &layout,
  void* smem_base,
  const dispatch_config_t& config,
  const model_config_t& model)
{
  size_t offset = 0;
  const int num_pipelines = config.num_pipelines;
  layout.num_pipelines = num_pipelines;
  layout.stages_per_pipeline = config.stages_per_pipeline;

  // Token buffer (aligned to 128B for TMA) -- total stages unchanged
  int token_size = (config.token_data_type == 0) ? 1 : 2;
  layout.token_buffer_stage_stride = model.hidden_dim * token_size;
  layout.token_buffer_stage_stride = (layout.token_buffer_stage_stride + 127) & ~127;
  layout.intra_node_token_buffer = reinterpret_cast<void*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += config.num_of_stages * layout.token_buffer_stage_stride;

  // Sparse to dense map buffer: 2 ping-pong stages PER PIPELINE (128B aligned)
  layout.s2d_map_stage_stride = config.num_of_tokens_per_chunk *
                                model.num_of_ranks_per_node * sizeof(int32_t);
  layout.s2d_map_stage_stride = (layout.s2d_map_stage_stride + 127) & ~127;
  layout.sparse_to_dense_map_buffer = reinterpret_cast<int32_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += 2 * num_pipelines * layout.s2d_map_stage_stride;

  // Prob buffer (only if forward dispatch, 16B aligned) -- total stages unchanged
  if (config.forward_dispatch) {
    layout.prob_buffer_stage_stride = model.num_of_experts_per_rank *
                                      model.num_of_ranks_per_node * sizeof(float);
    layout.prob_buffer_stage_stride = (layout.prob_buffer_stage_stride + 15) & ~15;
    layout.intra_node_prob_buffer = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * layout.prob_buffer_stage_stride;
  } else {
    layout.intra_node_prob_buffer = nullptr;
    layout.prob_buffer_stage_stride = 0;
  }

  // Scaling factor buffer (only if FP8, 16B aligned) -- total stages unchanged
  if (config.token_data_type == 0) {
    layout.sf_buffer_stage_stride = (model.hidden_dim / 128) * sizeof(float);
    layout.sf_buffer_stage_stride = (layout.sf_buffer_stage_stride + 15) & ~15;
    layout.intra_node_scaling_factor_buffer = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_stages * layout.sf_buffer_stage_stride;
  } else {
    layout.intra_node_scaling_factor_buffer = nullptr;
    layout.sf_buffer_stage_stride = 0;
  }

  // attn_to_rdma_map buffer (16B aligned, only if multinode, shared across pipelines)
  if (model.num_of_nodes > 1) {
    offset = (offset + 15) & ~15;
    layout.attn_to_rdma_map_buffer = reinterpret_cast<bool*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
  } else {
    layout.attn_to_rdma_map_buffer = nullptr;
  }

  // Mbarrier buffers (8B aligned) -- total stages unchanged (producer+consumer per stage)
  offset = (offset + 7) & ~7;
  layout.intra_node_mbarrier_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += config.num_of_stages * 2 * sizeof(uint64_t);

  // Per-pipeline s2d_map mbarriers: 2 per pipeline
  layout.sparse_to_dense_map_mbarrier_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += 2 * num_pipelines * sizeof(uint64_t);

  // Per-pipeline S2G group mbarrier: 1 per pipeline
  layout.S2G_group_mbarrier_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += num_pipelines * sizeof(uint64_t);

  layout.num_of_ranks_per_node = model.num_of_ranks_per_node;
  if (model.num_of_nodes > 1) {
    offset = (offset + 7) & ~7;
    layout.dispatch_memory_region_info = reinterpret_cast<dispatch_memory_region_info_t*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
  } else {
    layout.dispatch_memory_region_info = nullptr;
  }
  return layout;
}
static size_t calculate_dispatch_smem_layout_size(
  const dispatch_config_t& config,
  const model_config_t& model)
{
  size_t total_size = 0;
  const int num_pipelines = config.num_pipelines;
  int token_size = (config.token_data_type == 0) ? 1 : 2;

  // Token buffer (aligned to 128B for TMA) -- total stages unchanged
  int token_buffer_stage_stride = model.hidden_dim * token_size;
  token_buffer_stage_stride = (token_buffer_stage_stride + 127) & ~127;
  total_size += config.num_of_stages * token_buffer_stage_stride;

  // Sparse to dense map buffer: 2 ping-pong stages PER PIPELINE (128B aligned)
  int s2d_map_stage_stride = config.num_of_tokens_per_chunk * model.num_of_ranks_per_node * sizeof(int32_t);
  s2d_map_stage_stride = (s2d_map_stage_stride + 127) & ~127;
  total_size += 2 * num_pipelines * s2d_map_stage_stride;

  // Prob buffer (16B aligned per stage) -- total stages unchanged
  if (config.forward_dispatch) {
    int prob_buffer_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);
    prob_buffer_stage_stride = (prob_buffer_stage_stride + 15) & ~15;
    total_size += config.num_of_stages * prob_buffer_stage_stride;
  }

  // Scaling factor buffer (16B aligned per stage, only if FP8) -- total stages unchanged
  if (config.token_data_type == 0) {
    int sf_buffer_stage_stride = (model.hidden_dim / 128) * sizeof(float);
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
  // Add padding for alignment
  total_size = (total_size + 127) & ~127;
  return total_size;
}

__device__ combine_smem_layout_t create_combine_smem_layout(
  combine_smem_layout_t &layout,
  void* smem_base,
  int num_of_stages_g2s,
  int num_of_stages_s2g,
  int num_of_tokens_per_chunk,
  bool backward_combine,
  const model_config_t& model)
{
  size_t offset = 0;
  const uintptr_t smem_base_addr = reinterpret_cast<uintptr_t>(smem_base);
  auto align_offset = [&](size_t alignment) {
    const size_t mask = alignment - 1;
    const size_t misalignment = (smem_base_addr + offset) & mask;
    if (misalignment != 0) {
      offset += alignment - misalignment;
    }
  };

  // Store hidden_dim in layout
  layout.hidden_dim = model.hidden_dim;

  // In the single-node case (num_of_nodes == 1), the combine kernel does not use the
  // intra-node staging buffers. Skipping these buffers can cut SMEM roughly in half.
  const bool multinode = (model.num_of_nodes > 1);

  // Calculate stage strides (in elements, not bytes)
  layout.token_G2S_stage_stride = model.hidden_dim;
  layout.token_S2G_stage_stride = model.hidden_dim;
  const int tail_hidden_dim =
      model.hidden_dim > COMBINE_REG_HEAD_HIDDEN_DIM
          ? (model.hidden_dim - COMBINE_REG_HEAD_HIDDEN_DIM)
          : 0;
  layout.token_tail_S2G_stage_stride = tail_hidden_dim / 2;
  layout.prob_G2S_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node;
  layout.prob_S2G_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node;
  layout.prob_S2G_inter_stage_stride = layout.prob_S2G_stage_stride * model.num_of_nodes;

  // intra_node_token_* buffers (128B aligned, multi-node only)
  if (multinode) {
    align_offset(128);
  layout.intra_node_token_G2S_buffer = reinterpret_cast<uint16_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * model.hidden_dim * sizeof(uint16_t);

    align_offset(128);
  layout.intra_node_token_S2G_buffer = reinterpret_cast<uint16_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_s2g * model.hidden_dim * sizeof(uint16_t);

    // intra_node_token_tail_S2G_buffer (FP32 tail accum, 16B aligned)
    if (tail_hidden_dim > 0) {
      align_offset(16);
      layout.intra_node_token_tail_S2G_buffer = reinterpret_cast<float2*>(
          reinterpret_cast<uint8_t*>(smem_base) + offset);
      offset += num_of_stages_s2g * layout.token_tail_S2G_stage_stride * sizeof(float2);
    } else {
      layout.intra_node_token_tail_S2G_buffer = nullptr;
    }
  } else {
    layout.intra_node_token_G2S_buffer = nullptr;
    layout.intra_node_token_S2G_buffer = nullptr;
    layout.intra_node_token_tail_S2G_buffer = nullptr;
  }

  // inter_node_token_G2S_buffer (128B aligned)
  align_offset(128);
  layout.inter_node_token_G2S_buffer = reinterpret_cast<uint16_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += num_of_stages_g2s * model.hidden_dim * sizeof(uint16_t);

  // inter_node_token_S2G_buffer (128B aligned)
  align_offset(128);
  layout.inter_node_token_S2G_buffer = reinterpret_cast<uint16_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += num_of_stages_s2g * model.hidden_dim * sizeof(uint16_t);

  // inter_node_token_tail_S2G_buffer (FP32 tail accum, 16B aligned)
  if (tail_hidden_dim > 0) {
    align_offset(16);
    layout.inter_node_token_tail_S2G_buffer = reinterpret_cast<float2*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_s2g * layout.token_tail_S2G_stage_stride * sizeof(float2);
  } else {
    layout.inter_node_token_tail_S2G_buffer = nullptr;
  }

  // Prob buffers (only if backward_combine, 16B aligned)
  if (backward_combine) {
    if (multinode) {
      // intra_node_prob_G2S_buffer
      align_offset(16);
      layout.intra_node_prob_G2S_buffer = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
      offset += num_of_stages_g2s * model.num_of_experts_per_rank *
             model.num_of_ranks_per_node * sizeof(float);

      // intra_node_prob_S2G_buffer
      align_offset(16);
      layout.intra_node_prob_S2G_buffer = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
      offset += num_of_stages_s2g * model.num_of_experts_per_rank *
             model.num_of_ranks_per_node * sizeof(float);
    } else {
      layout.intra_node_prob_G2S_buffer = nullptr;
      layout.intra_node_prob_S2G_buffer = nullptr;
    }

    // inter_node_prob_G2S_buffer
    align_offset(16);
    layout.inter_node_prob_G2S_buffer = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * model.num_of_experts_per_rank *
             model.num_of_ranks_per_node * sizeof(float);

    // inter_node_prob_S2G_buffer
    align_offset(16);
    layout.inter_node_prob_S2G_buffer = reinterpret_cast<float*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_s2g * model.num_of_experts_per_rank *
             model.num_of_ranks_per_node * model.num_of_nodes * sizeof(float);
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
  layout.intra_node_mbarrier_G2S_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * 2 * sizeof(uint64_t);
  } else {
    layout.intra_node_mbarrier_G2S_buffer = nullptr;
  }

  // inter_node_mbarrier_G2S_buffer
  align_offset(8);
  layout.inter_node_mbarrier_G2S_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += num_of_stages_g2s * 2 * sizeof(uint64_t);

  // intra_node_to_rdma_mbarrier_buffer (only if multi-node)
  if (model.num_of_nodes > 1) {
    int max_num_of_chunks_per_rank = (model.max_num_of_tokens_per_rank +
                                      num_of_tokens_per_chunk - 1) /
                                     num_of_tokens_per_chunk;
    align_offset(8);
    layout.intra_node_to_rdma_mbarrier_buffer = reinterpret_cast<uint64_t*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += (model.num_of_nodes - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
  } else {
    layout.intra_node_to_rdma_mbarrier_buffer = nullptr;
  }

  if (model.num_of_nodes > 1) {
    align_offset(8);
    layout.combine_memory_region_info = reinterpret_cast<combine_memory_region_info_t*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += (model.num_of_nodes - 1) * sizeof(combine_memory_region_info_t);
  } else {
    layout.combine_memory_region_info = nullptr;
  }


  // Flag buffers (no special alignment needed)
  if (multinode) {
    layout.intra_node_flag_G2S_buffer = reinterpret_cast<bool*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += num_of_stages_g2s * sizeof(bool);
  } else {
    layout.intra_node_flag_G2S_buffer = nullptr;
  }

  layout.inter_node_flag_G2S_buffer = reinterpret_cast<bool*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += num_of_stages_g2s * sizeof(bool);

  // Streaming overlap fields (multi-node only, 4B aligned)
  if (multinode) {
    align_offset(4);
    layout.rdma_streaming_counter = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += sizeof(uint32_t);
  } else {
    layout.rdma_streaming_counter = nullptr;
  }

  return layout;

}

template<int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         bool BACKWARD_COMBINE>
static size_t calculate_combine_smem_layout_size(
  const model_config_t& model)
{
  // Dynamically computes the size required for combine shared memory layout,
  // mirroring the logic from create_combine_smem_layout
  size_t total_size = 0;

  // Compute max number of chunks per rank
  const int hidden_dim = model.hidden_dim;
  const int max_num_of_chunks_per_rank = (MAX_NUM_OF_TOKENS_PER_RANK +
                                          NUM_OF_TOKENS_PER_CHUNK - 1) /
                                         NUM_OF_TOKENS_PER_CHUNK;
  constexpr bool multinode = (NUM_OF_NODES > 1);

  // Token buffers (128B aligned for TMA)
  // intra_node_token_* buffers (multi-node only)
  if constexpr (multinode) {
  total_size = (total_size + 127) & ~127;
    total_size += NUM_OF_STAGES_G2S * hidden_dim * sizeof(uint16_t);

  total_size = (total_size + 127) & ~127;
    total_size += NUM_OF_STAGES_S2G * hidden_dim * sizeof(uint16_t);
  }

  // token_tail_S2G buffers (FP32 tail accum):
  const int tail_hidden_dim =
      hidden_dim > COMBINE_REG_HEAD_HIDDEN_DIM
          ? (hidden_dim - COMBINE_REG_HEAD_HIDDEN_DIM)
          : 0;
  const int token_tail_S2G_stage_stride = tail_hidden_dim / 2;  // float2 elements
  if (tail_hidden_dim > 0) {
    if constexpr (multinode) {
      total_size = (total_size + 15) & ~15;
      total_size += NUM_OF_STAGES_S2G * token_tail_S2G_stage_stride * sizeof(float2);
    }
  }

  // inter_node_token_G2S_buffer
  total_size = (total_size + 127) & ~127;
  total_size += NUM_OF_STAGES_G2S * hidden_dim * sizeof(uint16_t);

  // inter_node_token_S2G_buffer
  total_size = (total_size + 127) & ~127;
  total_size += NUM_OF_STAGES_S2G * hidden_dim * sizeof(uint16_t);

  // inter_node_token_tail_S2G_buffer (FP32 tail accum)
  if (tail_hidden_dim > 0) {
    total_size = (total_size + 15) & ~15;
    total_size += NUM_OF_STAGES_S2G * token_tail_S2G_stage_stride * sizeof(float2);
  }

  // Prob buffers (16B aligned, only if backward_combine)
  if constexpr (BACKWARD_COMBINE) {
    if constexpr (multinode) {
    // intra_node_prob_G2S_buffer
    total_size = (total_size + 15) & ~15;
      total_size += NUM_OF_STAGES_G2S * model.num_of_experts_per_rank *
                 model.num_of_ranks_per_node * sizeof(float);

    // intra_node_prob_S2G_buffer
    total_size = (total_size + 15) & ~15;
      total_size += NUM_OF_STAGES_S2G * model.num_of_experts_per_rank *
                 model.num_of_ranks_per_node * sizeof(float);
    }

    // inter_node_prob_G2S_buffer
    total_size = (total_size + 15) & ~15;
    total_size += NUM_OF_STAGES_G2S * model.num_of_experts_per_rank *
                 model.num_of_ranks_per_node * sizeof(float);

    // inter_node_prob_S2G_buffer
    total_size = (total_size + 15) & ~15;
    total_size += NUM_OF_STAGES_S2G * model.num_of_experts_per_rank *
                 model.num_of_ranks_per_node * NUM_OF_NODES * sizeof(float);
  }

  // Mbarrier buffers (8B aligned)
  // intra_node_mbarrier_G2S_buffer [stages][2] (multi-node only)
  if constexpr (multinode) {
  total_size = (total_size + 7) & ~7;
    total_size += NUM_OF_STAGES_G2S * 2 * sizeof(uint64_t);
  }

  // inter_node_mbarrier_G2S_buffer [stages][2]
  total_size = (total_size + 7) & ~7;
  total_size += NUM_OF_STAGES_G2S * 2 * sizeof(uint64_t);

  // intra_node_to_rdma_mbarrier_buffer [(nodes-1)][chunks] (only if multi-node)
  if constexpr (multinode) {
    total_size = (total_size + 7) & ~7;
    total_size += (NUM_OF_NODES - 1) * max_num_of_chunks_per_rank * sizeof(uint64_t);
  }

  // combine_memory_region_info [(nodes-1)] (align 8B, only if multi-node)
  if constexpr (multinode) {
    total_size = (total_size + 7) & ~7;
    total_size += (NUM_OF_NODES - 1) * sizeof(combine_memory_region_info_t);
  }

  // Flag buffers (no special alignment needed)
  if constexpr (multinode) {
    total_size += NUM_OF_STAGES_G2S * sizeof(bool);
  }
  total_size += NUM_OF_STAGES_G2S * sizeof(bool);

  // Streaming overlap fields (multi-node only, 4B aligned)
  if constexpr (multinode) {
    total_size = (total_size + 3) & ~3;
    total_size += sizeof(uint32_t);  // rdma_streaming_counter
  }

  return total_size;
}
// Data structure for kernel parameter for dispatch kernel.
template<typename TOKEN_DATA_TYPE>
struct dispatch_kernel_param_t{
  int hidden_dim;
  int experts_per_rank;
  int num_of_ranks_per_node;
  // Input buffers. These buffers are local buffers.
  const TOKEN_DATA_TYPE* attn_input_token;
  const float* attn_input_prob; // Needed by expert layer, so only valid in forward dispatch.
  const float* attn_input_token_scaling_factor; // If input token is FP8 dtype, we need scaling factor for tokens.
  // Output buffers. These buffers are both local and remote buffers.
  // NOTE: The source pointer arrays are allocated with cudaHostAllocMapped on the host side.
  // Device dereferencing of host-mapped pointer tables is very slow.
  // Keep a fixed-size array here and copy pointers on the host into this param struct.
  TOKEN_DATA_TYPE* expert_output_token[MAX_RANKS_PER_NODE];
  float* expert_output_prob[MAX_RANKS_PER_NODE]; // Only valid in forward dispatch.
  float* expert_output_scaling_factor[MAX_RANKS_PER_NODE]; // Only valid for FP8 token type.
  // Internal temp buffers. These buffers are local buffers.
  uint64_t* rdma_inter_node_group_flags; // For RDMA Atomic flags.
  uint32_t* intra_node_write_completion_flags; // For intra-node S2G write completion notification.
  // Metadata buffers. These buffers are local buffers.
  const bool* rdma_to_attn_map;
  const bool* attn_to_rdma_map;
  const int32_t* sparse_to_dense_map;
  uint64_t expected_rdma_flag_value;
  uint32_t expected_intra_node_flag_value;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
  // NCCL GIN context
  ncclDevComm_t* dcomms;           // Device communicators array (1 element, on device)
  ncclWindow_t nccl_window;        // Single registered window handle (by value)
  int num_gin_comms;               // Number of GIN communicators (1)
  int num_ctx_per_comm;            // Number of contexts per communicator
  void* gin_base_ptr;              // Base pointer for offset calculations
  unsigned signals_base;           // Base signal ID
  // Memory Region info
  struct dispatch_memory_region_info_t mr_info;
  // Grid barrier counter for fused device_sync in dispatch tail (per-rank, not IPC-shared)
  uint32_t* dispatch_grid_barrier_counter;
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
  struct warp_timing_entry_t { long long start_clock; long long end_clock; };
  warp_timing_entry_t* warp_timing;
#endif
};

// Data structure for kernel parameter for combine kernel.
struct combine_kernel_param_t{
  int hidden_dim;
  int experts_per_rank;
  int num_of_ranks_per_node;
  // Input buffers. These buffers are both local and remote buffers.
  // NOTE: The source pointer arrays are allocated with cudaHostAllocMapped on the host side.
  // Device dereferencing of host-mapped pointer tables is very slow.
  // Keep a fixed-size array here and copy pointers on the host into this param struct.
  uint16_t* expert_input_token[MAX_RANKS_PER_NODE];
  float* expert_input_prob[MAX_RANKS_PER_NODE];
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
  uint64_t expected_rdma_flag_value;
  uint32_t expected_intra_node_flag_value;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
  // NCCL GIN context
  ncclDevComm_t* dcomms;           // Device communicators array (1 element, on device)
  ncclWindow_t nccl_window;        // Single registered window handle (by value)
  int num_gin_comms;               // Number of GIN communicators (1)
  int num_ctx_per_comm;            // Number of contexts per communicator
  void* gin_base_ptr;              // Base pointer for offset calculations
  unsigned signals_base;           // Base signal ID
  unsigned combine_signal_offset;  // Signal offset for combine operations
  // qp info and mr info
  struct combine_memory_region_info_t mr_info;
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
__forceinline__ __device__ void get_comm_ctx(int global_channel, int num_ctx_per_comm,
                                    int& comm_idx, int& ctx_idx) {
    comm_idx = global_channel / num_ctx_per_comm;
    ctx_idx = global_channel % num_ctx_per_comm;
}

// Device function for inter-node node2node(RDMA) warp for dispatch kernel. There can be only 1 inter-node warp per CUDA block!
// Uses ncclGin API (net.put, net.signal)
template<typename INTER_NODE_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
__forceinline__ __device__ void N2N_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const int num_of_ranks_per_node,
                                                      const bool *attn_to_rdma_map,
                                                      ncclDevComm_t* dcomms,
                                                      ncclWindow_t nccl_window,
                                                      int num_gin_comms,
                                                      int num_ctx_per_comm,
                                                      void* gin_base_ptr,
                                                      unsigned signals_base,
                                                      const struct dispatch_memory_region_info_t *mr_info,
                                                      SMEM_TYPE* smem_buffer_ptr,
                                                      const int HIDDEN_DIM,
                                                      const int experts_per_rank)
{
  // Load attn_to_rdma_map using LDG.128. Each token will need 1 bool from this map.
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  constexpr int GIN_NUM_RATIO = 1 + std::is_same<TOKEN_DATA_TYPE, uint8_t>::value + FORWARD_DISPATCH;

  static_assert(INTER_NODE_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % 32 == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  // int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Loading mr_infos to shared memory for faster access in Put calls.
  int lane_id = INTER_NODE_GROUP::thread_rank() % 32;
  struct dispatch_memory_region_info_t *smem_mr_info_ptr = nullptr;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->dispatch_memory_region_info;
    if (lane_id == 0) {
      smem_mr_info_ptr[0] = mr_info[0];
    }
    __syncwarp();
  }

  // Batched chunk processing: issue all puts + signals across all chunks,
  // then flush once at the end. Staging is per-chunk (no aliasing).
  // Signals use the same comm as puts (same-QP ordering: put before signal).
  // With multiple N2N warps, chunks are partitioned by warp_id for parallelism.
  constexpr int N2N_WARPS = INTER_NODE_GROUP::size() / 32;
  int n2n_warp_id = INTER_NODE_GROUP::thread_rank() / 32;
  size_t token_bytes = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
  size_t prob_bytes = (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
  size_t sf_bytes = (HIDDEN_DIM / 128) * sizeof(float);
  uint32_t all_used_comms_mask = 0;
  constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;

  for (int chunk_idx = blockIdx.x * N2N_WARPS + n2n_warp_id;
       chunk_idx < NUM_OF_CHUNKS_PER_RANK;
       chunk_idx += NUM_OF_BLOCKS * N2N_WARPS) {
    int chunk_base_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    if (chunk_base_token_idx + token_range > num_of_tokens_per_rank) {
      token_range = num_of_tokens_per_rank - chunk_base_token_idx;
    }

    for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
      int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);
      int total_channels = num_gin_comms * num_ctx_per_comm;
      int global_channel = (remote_idx + chunk_idx + local_rank) % total_channels;
      int comm_idx, ctx_idx;
      get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
      all_used_comms_mask |= (1u << comm_idx);
      ncclGin net(dcomms[comm_idx], ctx_idx);
      ncclTeam world = ncclTeamWorld(dcomms[comm_idx]);
      int remote_node_id = remote_idx < node_rank ? remote_idx : remote_idx + 1;
      int rank_in_remote = remote_idx < node_rank ? node_rank - 1 : node_rank;

      uint8_t* gin_base = reinterpret_cast<uint8_t*>(gin_base_ptr);
      int run_count = 0;
      int dense_dst_offset = 0;
      size_t staging_base = smem_mr_info_ptr->rdma_send_staging_offset +
                            remote_idx * smem_mr_info_ptr->max_tokens_per_dest *
                            smem_mr_info_ptr->bytes_per_entry +
                            static_cast<size_t>(chunk_idx * NUM_OF_TOKENS_PER_CHUNK) *
                            smem_mr_info_ptr->bytes_per_entry;
      size_t packed_dst_base = smem_mr_info_ptr->rdma_inter_node_group_packed_offset +
                               rank_in_remote * smem_mr_info_ptr->max_tokens_per_dest *
                               smem_mr_info_ptr->bytes_per_entry +
                               static_cast<size_t>(chunk_idx * NUM_OF_TOKENS_PER_CHUNK) *
                               smem_mr_info_ptr->bytes_per_entry;

      for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
        int token_idx = token_idx_in_chunk + chunk_base_token_idx;
        bool need_write = attn_to_rdma_map[chunk_base_token_idx * (NUM_OF_NODES - 1) + remote_idx + token_idx_in_chunk * (NUM_OF_NODES - 1)];
        bool is_last_token = (token_idx_in_chunk == token_range - 1);

        if (need_write) {
          uint8_t* staging_entry = gin_base + staging_base +
                                   (dense_dst_offset + run_count) * smem_mr_info_ptr->bytes_per_entry;

          const uint8_t* token_src = gin_base +
                                     smem_mr_info_ptr->attn_input_token_offset +
                                    token_idx * token_bytes;
          warp_copy_int4(staging_entry, token_src, token_bytes, lane_id);

          if constexpr(FORWARD_DISPATCH) {
            const uint8_t* prob_src = gin_base +
                                      smem_mr_info_ptr->attn_input_prob_offset +
                                      (token_idx * NUM_OF_NODES + remote_node_id) * prob_bytes;
            warp_copy_int4(staging_entry + token_bytes, prob_src, prob_bytes, lane_id);
          }

          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            const uint8_t* sf_src = gin_base +
                                    smem_mr_info_ptr->attn_input_scaling_factor_offset +
                                    token_idx * sf_bytes;
            size_t sf_offset_in_entry = token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0);
            warp_copy_int4(staging_entry + sf_offset_in_entry, sf_src, sf_bytes, lane_id);
          }

          run_count++;
        }

        constexpr int max_batch = HYBRIDEP_DISPATCH_RDMA_BATCH_SIZE;
        bool should_flush = run_count > 0 &&
                           (is_last_token || run_count >= max_batch);

        if (should_flush) {
          size_t packed_dst_offset = packed_dst_base +
                                     dense_dst_offset * smem_mr_info_ptr->bytes_per_entry;

          net.put(world, remote_node_id,
                  nccl_window, packed_dst_offset,
                  nccl_window, staging_base + dense_dst_offset * smem_mr_info_ptr->bytes_per_entry,
                  run_count * smem_mr_info_ptr->bytes_per_entry,
                  ncclGin_None{}, ncclGin_None{}, ncclCoopWarp());
          dense_dst_offset += run_count;
          run_count = 0;
        }
      }

      // Signal this chunk's completion on the SAME put comm.
      // Same-QP ordering guarantees all preceding puts are visible at the
      // remote before this signal arrives.
      if (lane_id == 0) {
        unsigned tail_signal_id = smem_mr_info_ptr->signals_tail_base +
                                   node_rank * (NUM_OF_NODES * num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                   remote_node_id * (num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                   local_rank * MAX_CHUNKS_PER_RANK +
                                   chunk_idx;
        net.signal(world, remote_node_id,
                   ncclGin_SignalAdd{tail_signal_id, 1},
                   ncclCoopThread(),
                   ncclGin_None{},
                   cuda::thread_scope_thread,
                   cuda::thread_scope_thread);
      }
    }
  }

  // Single flush covering all puts + signals across all chunks.
  for (int c = 0; c < num_gin_comms; ++c) {
    if (all_used_comms_mask & (1u << c)) {
      ncclGin net_flush(dcomms[c], 0);
      net_flush.flush(ncclCoopWarp(), cuda::std::memory_order_acquire);
    }
  }
  __syncwarp(0xffffffff);
}


// Device function for intra-node G2S warp group for dispatch kernel.
// With NUM_PIPELINES > 1, each warp within the group is an independent pipeline
// processing disjoint chunks through its own partition of the shared memory FIFO.
template<typename INTRA_NODE_G2S_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH,
         int NUM_PIPELINES>
__forceinline__ __device__ void G2S_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const int num_of_ranks_per_node,
                                                      const uint64_t expected_flag_value,
                                                      const int HIDDEN_DIM,
                                                      const bool* rdma_to_attn_map,
                                                      const TOKEN_DATA_TYPE* attn_input_token,
                                                      const float* attn_input_prob,
                                                      const float* attn_input_token_scaling_factor,
                                                      uint64_t* rdma_inter_node_group_flags,
                                                      ncclDevComm_t* dcomms,
                                                      unsigned signals_base,
                                                      int num_gin_comms,
                                                      int num_ctx_per_comm,
                                                      void* gin_base_ptr,
                                                      const struct dispatch_memory_region_info_t* mr_info,
                                                      SMEM_TYPE* smem_buffer_ptr,
                                                      const int experts_per_rank)
{
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);
  constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;

  const int pipeline_rank = INTRA_NODE_G2S_GROUP::warp_rank();
  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  const int max_num_of_chunks_per_rank = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  int stage = 0;
  uint32_t consumer_parity = 1;
  int tokens_produced = 0;

  if(cuda::ptx::elect_sync(~0)){
    int chunk_iter = 0;
    for(int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS){
      if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) continue;

      int num_of_routing_info_load_iter_for_current_chunk;
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for(int j = 0; j < NUM_OF_NODES; j++){
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
        if(node_id != node_rank){
          constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
          unsigned tail_signal_id = mr_info->signals_tail_base +
                                     node_id * (NUM_OF_NODES * num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                     node_rank * (num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                     local_rank * MAX_CHUNKS_PER_RANK +
                                     i;
          int sender_remote_idx = node_rank < node_id ? node_rank : node_rank - 1;
          int total_channels = num_gin_comms * num_ctx_per_comm;
          int signal_channel = (sender_remote_idx + i + local_rank) % total_channels;
          int comm_idx, ctx_idx;
          get_comm_ctx(signal_channel, num_ctx_per_comm, comm_idx, ctx_idx);
          ncclGin net(dcomms[comm_idx], ctx_idx);
          net.waitSignal(ncclCoopThread(), tail_signal_id, expected_flag_value);
        }
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map +
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));
        const TOKEN_DATA_TYPE* token_load_base_addr = nullptr;
        const float* prob_load_base_addr = nullptr;
        const float* scaling_factor_load_base_addr = nullptr;
        const uint8_t* packed_base = nullptr;
        bool use_packed_layout = false;

        if (node_id != node_rank) {
          use_packed_layout = true;
          packed_base = reinterpret_cast<const uint8_t*>(gin_base_ptr) +
                        mr_info->rdma_inter_node_group_packed_offset +
                        rdma_buffer_tile_id * mr_info->max_tokens_per_dest * mr_info->bytes_per_entry +
                        static_cast<size_t>(i * NUM_OF_TOKENS_PER_CHUNK) * mr_info->bytes_per_entry;
        } else {
          int chunk_first_token_id = i * NUM_OF_TOKENS_PER_CHUNK;
          token_load_base_addr = attn_input_token + chunk_first_token_id * HIDDEN_DIM;
          if constexpr(FORWARD_DISPATCH) {
            prob_load_base_addr = attn_input_prob + chunk_first_token_id * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES);
          }
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            scaling_factor_load_base_addr = attn_input_token_scaling_factor + chunk_first_token_id * (HIDDEN_DIM / 128);
          }
        }
        int packed_dense_idx = 0;
        for (int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++) {
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for (int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            if (current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            if (token_needed_by_this_node){
              if (tokens_produced >= STAGES_PER_PIPELINE){
                while(!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_consumer(pipeline_rank, stage), consumer_parity)){}
              }
              uint32_t total_tx_size = 0;
              size_t token_bytes = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
              size_t prob_bytes = (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
              size_t sf_bytes = (HIDDEN_DIM / 128) * sizeof(float);

              if (use_packed_layout) {
                const uint8_t* packed_entry = packed_base + packed_dense_idx * mr_info->bytes_per_entry;
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         smem_buffer_ptr->get_token_buffer(pipeline_rank, stage),
                                         reinterpret_cast<const void*>(packed_entry),
                                         (uint32_t)token_bytes,
                                         smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage));
                total_tx_size += (uint32_t)token_bytes;
                if constexpr(FORWARD_DISPATCH) {
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_prob_buffer(pipeline_rank, stage)),
                                           reinterpret_cast<const void*>(packed_entry + token_bytes),
                                           (uint32_t)prob_bytes,
                                           smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage));
                  total_tx_size += (uint32_t)prob_bytes;
                }
                if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
                  size_t sf_offset_in_entry = token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0);
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_sf_buffer(pipeline_rank, stage)),
                                           reinterpret_cast<const void*>(packed_entry + sf_offset_in_entry),
                                           (uint32_t)sf_bytes,
                                           smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage));
                  total_tx_size += (uint32_t)sf_bytes;
                }
                packed_dense_idx++;
              } else {
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         smem_buffer_ptr->get_token_buffer(pipeline_rank, stage),
                                         reinterpret_cast<const void*>(token_load_base_addr + (current_token_id * HIDDEN_DIM)),
                                         (uint32_t)token_bytes,
                                         smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage));
                total_tx_size += (uint32_t)token_bytes;
                if constexpr(FORWARD_DISPATCH) {
                  const float* prob_load_token_addr = prob_load_base_addr +
                      (current_token_id * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES)) +
                      (node_rank * (experts_per_rank * num_of_ranks_per_node));
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_prob_buffer(pipeline_rank, stage)),
                                           reinterpret_cast<const void*>(prob_load_token_addr),
                                           (uint32_t)prob_bytes,
                                           smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage));
                  total_tx_size += (uint32_t)prob_bytes;
                }
                if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_sf_buffer(pipeline_rank, stage)),
                                           reinterpret_cast<const void*>(scaling_factor_load_base_addr + (current_token_id * (HIDDEN_DIM / 128))),
                                           (uint32_t)sf_bytes,
                                           smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage));
                  total_tx_size += (uint32_t)sf_bytes;
                }
              }

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage),
                                                   total_tx_size);
              tokens_produced += 1;
              stage += 1;
              if (stage == STAGES_PER_PIPELINE) {
                stage = 0;
                consumer_parity ^= 1;
              }
            }
          }
        }
      }
    }
  }
  // Update residue flags (only pipeline 0 does this to avoid duplicate writes).
  if (INTRA_NODE_G2S_GROUP::warp_rank() == 0) {
    int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;
    for (int node_id = blockIdx.x; node_id < NUM_OF_NODES - 1; node_id += gridDim.x) {
      uint64_t *residue_flag_base_ptr = rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank + num_of_chunks_per_rank);
      if (INTRA_NODE_G2S_GROUP::thread_rank() < residue_flag_count) {
        residue_flag_base_ptr[INTRA_NODE_G2S_GROUP::thread_rank()] = expected_flag_value;
      }
    }
  }
}

// Device function for intra-node S2G warp group for dispatch kernel.
// With NUM_PIPELINES > 1, each warp is an independent pipeline consumer
// paired with the G2S warp of the same pipeline_rank.
template<typename INTRA_NODE_S2G_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int NUM_OF_IN_FLIGHT_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH,
         int NUM_PIPELINES>
__forceinline__ __device__ void S2G_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const int num_of_ranks_per_node,
                                                      const int HIDDEN_DIM,
                                                      const bool* rdma_to_attn_map,
                                                      const int32_t* sparse_to_dense_map,
                                                      TOKEN_DATA_TYPE* const* remote_expert_output_token,
                                                      float* const* remote_expert_output_prob,
                                                      float* const* remote_expert_output_scaling_factor,
                                                      SMEM_TYPE* smem_buffer_ptr,
                                                      const int experts_per_rank)
{
  constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;
  static_assert(NUM_OF_IN_FLIGHT_S2G < STAGES_PER_PIPELINE, "NUM_OF_IN_FLIGHT_S2G must be smaller than STAGES_PER_PIPELINE.");
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  constexpr int MAX_RANKS_PER_NODE = 8;
  using sparse_to_dense_map_load_t = Copy_t<MAX_RANKS_PER_NODE * sizeof(int32_t)>;
  const int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN = (num_of_ranks_per_node * sizeof(int32_t) + sizeof(sparse_to_dense_map_load_t) - 1) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

  const int pipeline_rank = INTRA_NODE_S2G_GROUP::warp_rank();
  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  int in_flight_s2g = 0;
  int stage = 0;
  uint32_t producer_parity = 0;
  uint32_t sparse_to_dense_map_stage = 0;
  uint32_t sparse_to_dense_map_parity = 0;

  if (cuda::ptx::elect_sync(~0)) {
    // Each pipeline's S2G prefetches its own first s2d map for its first chunk.
    // Find the first chunk this pipeline will process.
    {
      int chunk_iter = 0;
      for (int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS) {
        if ((chunk_iter++ % NUM_PIPELINES) == pipeline_rank) {
          int current_chunk_size;
          if (remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1) {
            current_chunk_size = remainder_chunk_size;
          } else {
            current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
          }
          const int32_t* s2d_base = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;
          cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                   cuda::ptx::space_global,
                                   reinterpret_cast<void*>(smem_buffer_ptr->get_s2d_map_buffer_base(pipeline_rank, sparse_to_dense_map_stage)),
                                   reinterpret_cast<const void*>(s2d_base),
                                   (uint32_t)(current_chunk_size * num_of_ranks_per_node * sizeof(int32_t)),
                                   smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, sparse_to_dense_map_stage));
          cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                               cuda::ptx::scope_cta,
                                               cuda::ptx::space_shared,
                                               smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, sparse_to_dense_map_stage),
                                               (uint32_t)(current_chunk_size * num_of_ranks_per_node * sizeof(int32_t)));
          break;
        }
      }
    }

    int chunk_iter = 0;
    for (int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS) {
      if ((chunk_iter++ % NUM_PIPELINES) != pipeline_rank) continue;

      int num_of_routing_info_load_iter_for_current_chunk;
      int current_chunk_size;
      if (remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1) {
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      } else {
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for (int j = 0; j < NUM_OF_NODES; j++) {
        // Per-pipeline self-sync (arrival count = 1, trivially satisfied)
        uint64_t state_token = cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_S2G_group_mbar(pipeline_rank));
        while (!cuda::ptx::mbarrier_try_wait(smem_buffer_ptr->get_S2G_group_mbar(pipeline_rank), state_token)) {}

        // Prefetch s2d map for next (chunk, node) pair for THIS pipeline
        {
          int next_chunk_id;
          int next_node_id;
          int next_node_iter = j + 1;
          if (next_node_iter < NUM_OF_NODES) {
            next_chunk_id = i;
            next_node_id = node_rank >= next_node_iter ? node_rank - next_node_iter : node_rank + NUM_OF_NODES - next_node_iter;
          } else {
            // Find the next chunk this pipeline will process
            int future_chunk_iter = chunk_iter;  // chunk_iter was already incremented for current chunk
            next_chunk_id = -1;
            for (int fi = i + NUM_OF_BLOCKS; fi < num_of_chunks_per_rank; fi += NUM_OF_BLOCKS) {
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
            const int32_t* s2d_base = sparse_to_dense_map + (next_node_id * num_of_tokens_per_rank + next_chunk_id * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                     cuda::ptx::space_global,
                                     reinterpret_cast<void*>(smem_buffer_ptr->get_s2d_map_buffer_base(pipeline_rank, sparse_to_dense_map_stage ^ 1)),
                                     reinterpret_cast<const void*>(s2d_base),
                                     (uint32_t)(next_chunk_size * num_of_ranks_per_node * sizeof(int32_t)),
                                     smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, sparse_to_dense_map_stage ^ 1));
            cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                 cuda::ptx::scope_cta,
                                                 cuda::ptx::space_shared,
                                                 smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, sparse_to_dense_map_stage ^ 1),
                                                 (uint32_t)(next_chunk_size * num_of_ranks_per_node * sizeof(int32_t)));
          }
        }

        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map +
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));

        while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_s2d_map_mbar(pipeline_rank, sparse_to_dense_map_stage), sparse_to_dense_map_parity)){}

        for (int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++){
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for (int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            if (current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            if (token_needed_by_this_node){
              const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                                (smem_buffer_ptr->get_s2d_map_buffer(pipeline_rank, sparse_to_dense_map_stage, k * NUM_OF_TOKENS_PER_LOAD_ITER + n));
              while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_producer(pipeline_rank, stage), producer_parity)){}

              // Each pipeline has 1 S2G warp, so this thread handles ALL destination ranks
              for (int m = 0; m < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; m++){
                sparse_to_dense_map_load_t sparse_to_dense_map_data = sparse_to_dense_map_load_addr[m];
                #pragma unroll
                for (int t = 0; t < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; t++){
                  int32_t output_buffer_index = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data) + t);
                  if (output_buffer_index != -1) {
                    int remote_rank_id = m * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + t;
                    TOKEN_DATA_TYPE* remote_token_addr = remote_expert_output_token[remote_rank_id] + (output_buffer_index * HIDDEN_DIM);
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                             cuda::ptx::space_shared,
                                             reinterpret_cast<void*>(remote_token_addr),
                                             smem_buffer_ptr->get_token_buffer(pipeline_rank, stage),
                                             (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)));
                    if constexpr(FORWARD_DISPATCH) {
                      float* remote_prob_addr = remote_expert_output_prob[remote_rank_id] + (output_buffer_index * (experts_per_rank * num_of_ranks_per_node));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_prob_addr),
                                               reinterpret_cast<const void*>(smem_buffer_ptr->get_prob_buffer(pipeline_rank, stage)),
                                               (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float)));
                    }
                    if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
                      float* remote_scaling_factor_addr = remote_expert_output_scaling_factor[remote_rank_id] + (output_buffer_index * (HIDDEN_DIM / 128));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_scaling_factor_addr),
                                               reinterpret_cast<const void*>(smem_buffer_ptr->get_sf_buffer(pipeline_rank, stage)),
                                               (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)));
                    }
                  }
                }
              }
              cuda::ptx::cp_async_bulk_commit_group();
              in_flight_s2g += 1;
              if (in_flight_s2g > NUM_OF_IN_FLIGHT_S2G) {
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_IN_FLIGHT_S2G>{});
                in_flight_s2g -= 1;
                int notify_stage = (stage - NUM_OF_IN_FLIGHT_S2G) >= 0 ? (stage - NUM_OF_IN_FLIGHT_S2G) : (stage - NUM_OF_IN_FLIGHT_S2G + STAGES_PER_PIPELINE);
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_intra_node_mbarrier_consumer(pipeline_rank, notify_stage));
              }

              stage += 1;
              if (stage == STAGES_PER_PIPELINE){
                stage = 0;
                producer_parity ^= 1;
              }
            }
          }
        }
        sparse_to_dense_map_stage += 1;
        if(sparse_to_dense_map_stage == 2){
          sparse_to_dense_map_stage = 0;
          sparse_to_dense_map_parity ^= 1;
        }
      }
    }
    // Drain all in-flight TMA S2G writes before returning.
    // Required because fused device_sync signals inter-rank completion
    // inside this kernel (no stream-ordering guarantee from a separate kernel).
    cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
  }
}

// Device function for intra-node G2S warp for combine kernel. There can be only 1 such warp per CUDA block!
template<typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
__forceinline__ __device__ void intra_node_G2S_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const bool* rdma_to_attn_map,
                                                                 const int32_t* sparse_to_dense_map,
                                                                 uint16_t* const* remote_expert_input_token,
                                                                 float* const* remote_expert_input_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr,
                                                                 const int HIDDEN_DIM,
                                                                 const int experts_per_rank)
{
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
  const int total_num_of_chunks = (NUM_OF_NODES - 1) * num_of_chunks_per_rank;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
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
  const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));
  const uint32_t prob_bytes =
      (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));

  // Track total stages filled across all tokens.
  int global_offset = 0;

  // Iterate through all chunks assigned to this block.
  for (int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS) {
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);
    // How many token for this chunk.
    int current_chunk_size;
    if (remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1) {
      current_chunk_size = remainder_chunk_size;
    } else {
      current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
    }

    const bool* rdma_to_attn_map_load_base_addr = rdma_to_attn_map +
        (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK);

    const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;

    // Iterate through all dst tokens within this chunk.
    for (int current_token_id = 0; current_token_id < current_chunk_size; current_token_id++) {
      // Check whether this dst token is needed by this node. If not needed, just skip.
      bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
      if (!token_needed_by_this_node) {
        continue;
      }

      const int32_t* sparse_to_dense_row = sparse_to_dense_map_load_base_addr + current_token_id * num_of_ranks_per_node;

      // First pass: count total valid ranks across all slices.
      int total_valid_count = 0;
      for (int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE) {
        const int rank_id = rank_base + lane_id;
        const bool lane_active = (rank_id < num_of_ranks_per_node);
        const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
        const unsigned mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
        total_valid_count += __popc(mask);
      }
      if (total_valid_count == 0) {
        continue;
      }

      // Second pass: issue TMA in batches of ring_len to prevent stage
      // collisions when total_valid_count > ring_len (e.g. top_k=8, ring_len=5).
      int ranks_issued = 0;
      while (ranks_issued < total_valid_count) {
        const int batch_end = (ranks_issued + ring_len < total_valid_count)
                                ? ranks_issued + ring_len : total_valid_count;

        int slice_offset = 0;
        for (int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE) {
          const int rank_id = rank_base + lane_id;
          const bool lane_active = (rank_id < num_of_ranks_per_node);
          const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
          const unsigned valid_mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
          const int slice_valid = __popc(valid_mask);
          const bool lane_valid = lane_active && s2d_val != -1;
          const int local_lane_rank = __popc(valid_mask & ((1u << lane_id) - 1));
          const int global_rank = slice_offset + local_lane_rank;
          const bool in_batch = lane_valid && global_rank >= ranks_issued && global_rank < batch_end;

          if (in_batch) {
            const int rank_in_batch = global_rank - ranks_issued;
            const int my_abs_offset = global_offset + rank_in_batch;
            const int stage_idx = my_abs_offset % ring_len;
            const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

            while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(stage_idx), parity)){}

            const uint16_t* rank_token_ptr = remote_expert_input_token[rank_id];
            uint32_t total_tx_size = 0;
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                     cuda::ptx::space_global,
                                     reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_token_G2S(stage_idx)),
                                     reinterpret_cast<const void*>(rank_token_ptr + (s2d_val * HIDDEN_DIM)),
                                     token_bytes,
                                     smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(stage_idx));

            total_tx_size += token_bytes;

            if constexpr(BACKWARD_COMBINE) {
              const float* rank_prob_ptr = remote_expert_input_prob[rank_id];
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_prob_G2S(stage_idx)),
                                       reinterpret_cast<const void*>(rank_prob_ptr + (s2d_val * (experts_per_rank * num_of_ranks_per_node))),
                                       prob_bytes,
                                       smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(stage_idx));

              total_tx_size += prob_bytes;
            }

            smem_buffer_ptr->intra_node_flag_G2S_buffer[stage_idx] = (global_rank == total_valid_count - 1);

            cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                 cuda::ptx::scope_cta,
                                                 cuda::ptx::space_shared,
                                                 smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(stage_idx),
                                                 total_tx_size);
          }

          slice_offset += slice_valid;
        }

        global_offset += (batch_end - ranks_issued);
        ranks_issued = batch_end;
      }
    }
  }
}

// Device function for intra-node reduction warp group for combine kernel.
template<typename INTRA_NODE_RED_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
         bool BACKWARD_COMBINE>
__forceinline__ __device__ void intra_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const bool* rdma_to_attn_map,
                                                                 uint16_t* rdma_intra_node_red_token,
                                                                 float* rdma_intra_node_red_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr,
                                                                 const int HIDDEN_DIM,
                                                                 const int experts_per_rank)
{
  // Vectorized loads from rdma_to_attn_map. Each destination token contributes one bool.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);


  // Token values are processed as BF16x2 and accumulated in FP32. HIDDEN_DIM must be even.

  const int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
  // Keep a fixed-size prefix of each token in registers; spill the remaining suffix to SMEM.
  constexpr int REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN = COMBINE_REG_HEAD_HIDDEN_DIM / 2;
  const int head_bf16x2 =
      NUM_OF_BF16X2_ELEMENTS_PER_TOKEN > REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN
          ? REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN
          : NUM_OF_BF16X2_ELEMENTS_PER_TOKEN;
  const int tail_bf16x2 = NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - head_bf16x2;
  const int tail_float4 = tail_bf16x2 >> 1;
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
  constexpr int MAX_NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD_INTRA =
      ((REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  // Backward-combine probability vectors stay in float (no BF16 packing).
  const int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
      ((experts_per_rank * num_of_ranks_per_node - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  // Compile-time upper bound sized exactly to this instantiation's LSA team.
  constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_MAX_LOCAL_EXPERTS * LSA_TEAM_SIZE - 1) / INTRA_NODE_RED_GROUP::size()) + 1;

  // This warp group emits chunks in the same per-destination order consumed by the RDMA warp group:
  // chunk 0 for node + 1, node + 2, ... node - 1, then chunk 1 for node + 1, ...
  // That ordering lets the downstream inter-node stage observe matching chunk IDs across peers.

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // Number of chunks for one rank, including the tail chunk if present.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // Total chunks emitted by this node across all remote destinations.
  const int total_num_of_chunks = (NUM_OF_NODES - 1) * num_of_chunks_per_rank;
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
  int streaming_pending = 0;        // tokens TMA-committed but not yet signaled to counter
  uint32_t cumulative_produced = 0; // total active tokens whose TMA S2G is complete (across all chunks)

  // Iterate through all chunks assigned to this block.
  for(int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS){
    // Destination node for this emitted chunk.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // Chunk index within that destination node's stream.
    int chunk_id = i / (NUM_OF_NODES - 1);
    // Compact destination-slot index in the RDMA reduction buffers.
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    // Token offset of this chunk inside the per-destination reduction buffer.
    int rdma_intra_node_red_id = rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    // Number of vector loads needed for the routing flags of this chunk.
    int num_of_routing_info_load_iter_for_current_chunk;
    // Number of valid tokens in this chunk.
    int current_chunk_size;
    if (remainder_chunk_size != 0 && chunk_id == num_of_chunks_per_rank - 1) {
      num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
      current_chunk_size = remainder_chunk_size;
    } else {
      num_of_routing_info_load_iter_for_current_chunk = NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK;
      current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
    }

    const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map +
                                                                      (node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK));

    uint16_t* rdma_intra_node_red_token_base_ptr = rdma_intra_node_red_token + rdma_intra_node_red_id * HIDDEN_DIM;
    float* rdma_intra_node_red_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE) {
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
          // FP32 accumulator for the register-resident token head.
          float2 acc_token_fp32[MAX_NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD_INTRA];
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
          for (int n = 0; n < MAX_NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD_INTRA; n++) {
            acc_token_fp32[n].x = 0.0f;
            acc_token_fp32[n].y = 0.0f;
          }
          if constexpr(BACKWARD_COMBINE) {
            #pragma unroll
            for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
              acc_prob_ptr[n] = 0.0f;
            }
          }
          float4* acc_token_tail_smem4 = nullptr;
          bool tail_initialized = false;
          if (tail_bf16x2 > 0) {
            // Tail accumulation uses the per-stage S2G scratch buffer in shared memory.
            acc_token_tail_smem4 = reinterpret_cast<float4*>(
                smem_buffer_ptr->get_intra_node_token_tail_S2G(dst_token_stage));
          }

          // Consume source tokens for this destination token until the producer marks the last one.
          do {
            // Current source token / optional prob slice in the G2S FIFO stage.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_G2S(token_stage));
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE) {
              load_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_G2S(token_stage);
            }

            // Warp 0 waits for the producer; then the whole reduction group can read this stage.
            if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
              if (cuda::ptx::elect_sync(~0)) {
                while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage), token_producer_parity)) {}
              }
            }
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

            // Accumulate the register-resident token head.
            #pragma unroll
            for (int n = 0; n < MAX_NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD_INTRA; n++) {
              int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
              if (element_id < head_bf16x2) {
                __nv_bfloat162 src_data = load_token_base_ptr[element_id];
                float2 src_data_fp32 = __bfloat1622float2(src_data);
                acc_token_fp32[n].x += src_data_fp32.x;
                acc_token_fp32[n].y += src_data_fp32.y;
              }
            }
            // Accumulate the token tail in shared memory to cap register usage for large hidden dims.
            if (tail_bf16x2 > 0) {
              if (!tail_initialized) {
                // First contributor initializes the SMEM tail accumulator.
                int p = INTRA_NODE_RED_GROUP::thread_rank();
                for (; p + 7 * INTRA_NODE_RED_GROUP::size() < tail_float4;
                     p += 8 * INTRA_NODE_RED_GROUP::size()) {
                  const int p0 = p, p1 = p + INTRA_NODE_RED_GROUP::size();
                  const int p2 = p + 2 * INTRA_NODE_RED_GROUP::size(), p3 = p + 3 * INTRA_NODE_RED_GROUP::size();
                  const int p4 = p + 4 * INTRA_NODE_RED_GROUP::size(), p5 = p + 5 * INTRA_NODE_RED_GROUP::size();
                  const int p6 = p + 6 * INTRA_NODE_RED_GROUP::size(), p7 = p + 7 * INTRA_NODE_RED_GROUP::size();
                  const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                  const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                  const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                  const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                  __nv_bfloat162 s0a = load_token_base_ptr[e0], s0b = load_token_base_ptr[e0 + 1];
                  __nv_bfloat162 s1a = load_token_base_ptr[e1], s1b = load_token_base_ptr[e1 + 1];
                  __nv_bfloat162 s2a = load_token_base_ptr[e2], s2b = load_token_base_ptr[e2 + 1];
                  __nv_bfloat162 s3a = load_token_base_ptr[e3], s3b = load_token_base_ptr[e3 + 1];
                  __nv_bfloat162 s4a = load_token_base_ptr[e4], s4b = load_token_base_ptr[e4 + 1];
                  __nv_bfloat162 s5a = load_token_base_ptr[e5], s5b = load_token_base_ptr[e5 + 1];
                  __nv_bfloat162 s6a = load_token_base_ptr[e6], s6b = load_token_base_ptr[e6 + 1];
                  __nv_bfloat162 s7a = load_token_base_ptr[e7], s7b = load_token_base_ptr[e7 + 1];
                  float2 f0a = __bfloat1622float2(s0a), f0b = __bfloat1622float2(s0b);
                  float2 f1a = __bfloat1622float2(s1a), f1b = __bfloat1622float2(s1b);
                  float2 f2a = __bfloat1622float2(s2a), f2b = __bfloat1622float2(s2b);
                  float2 f3a = __bfloat1622float2(s3a), f3b = __bfloat1622float2(s3b);
                  float2 f4a = __bfloat1622float2(s4a), f4b = __bfloat1622float2(s4b);
                  float2 f5a = __bfloat1622float2(s5a), f5b = __bfloat1622float2(s5b);
                  float2 f6a = __bfloat1622float2(s6a), f6b = __bfloat1622float2(s6b);
                  float2 f7a = __bfloat1622float2(s7a), f7b = __bfloat1622float2(s7b);
                  acc_token_tail_smem4[p0] = make_float4(f0a.x, f0a.y, f0b.x, f0b.y);
                  acc_token_tail_smem4[p1] = make_float4(f1a.x, f1a.y, f1b.x, f1b.y);
                  acc_token_tail_smem4[p2] = make_float4(f2a.x, f2a.y, f2b.x, f2b.y);
                  acc_token_tail_smem4[p3] = make_float4(f3a.x, f3a.y, f3b.x, f3b.y);
                  acc_token_tail_smem4[p4] = make_float4(f4a.x, f4a.y, f4b.x, f4b.y);
                  acc_token_tail_smem4[p5] = make_float4(f5a.x, f5a.y, f5b.x, f5b.y);
                  acc_token_tail_smem4[p6] = make_float4(f6a.x, f6a.y, f6b.x, f6b.y);
                  acc_token_tail_smem4[p7] = make_float4(f7a.x, f7a.y, f7b.x, f7b.y);
                }
                for (; p < tail_float4; p += INTRA_NODE_RED_GROUP::size()) {
                  const int e = head_bf16x2 + (p << 1);
                  __nv_bfloat162 sa = load_token_base_ptr[e], sb = load_token_base_ptr[e + 1];
                  float2 fa = __bfloat1622float2(sa), fb = __bfloat1622float2(sb);
                  acc_token_tail_smem4[p] = make_float4(fa.x, fa.y, fb.x, fb.y);
                }
                tail_initialized = true;
              } else {
                // Later contributors add into the existing SMEM tail accumulator.
                int p = INTRA_NODE_RED_GROUP::thread_rank();
                for (; p + 7 * INTRA_NODE_RED_GROUP::size() < tail_float4;
                     p += 8 * INTRA_NODE_RED_GROUP::size()) {
                  const int p0 = p, p1 = p + INTRA_NODE_RED_GROUP::size();
                  const int p2 = p + 2 * INTRA_NODE_RED_GROUP::size(), p3 = p + 3 * INTRA_NODE_RED_GROUP::size();
                  const int p4 = p + 4 * INTRA_NODE_RED_GROUP::size(), p5 = p + 5 * INTRA_NODE_RED_GROUP::size();
                  const int p6 = p + 6 * INTRA_NODE_RED_GROUP::size(), p7 = p + 7 * INTRA_NODE_RED_GROUP::size();
                  const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                  const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                  const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                  const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                  __nv_bfloat162 s0a = load_token_base_ptr[e0], s0b = load_token_base_ptr[e0 + 1];
                  __nv_bfloat162 s1a = load_token_base_ptr[e1], s1b = load_token_base_ptr[e1 + 1];
                  __nv_bfloat162 s2a = load_token_base_ptr[e2], s2b = load_token_base_ptr[e2 + 1];
                  __nv_bfloat162 s3a = load_token_base_ptr[e3], s3b = load_token_base_ptr[e3 + 1];
                  __nv_bfloat162 s4a = load_token_base_ptr[e4], s4b = load_token_base_ptr[e4 + 1];
                  __nv_bfloat162 s5a = load_token_base_ptr[e5], s5b = load_token_base_ptr[e5 + 1];
                  __nv_bfloat162 s6a = load_token_base_ptr[e6], s6b = load_token_base_ptr[e6 + 1];
                  __nv_bfloat162 s7a = load_token_base_ptr[e7], s7b = load_token_base_ptr[e7 + 1];
                  float2 f0a = __bfloat1622float2(s0a), f0b = __bfloat1622float2(s0b);
                  float2 f1a = __bfloat1622float2(s1a), f1b = __bfloat1622float2(s1b);
                  float2 f2a = __bfloat1622float2(s2a), f2b = __bfloat1622float2(s2b);
                  float2 f3a = __bfloat1622float2(s3a), f3b = __bfloat1622float2(s3b);
                  float2 f4a = __bfloat1622float2(s4a), f4b = __bfloat1622float2(s4b);
                  float2 f5a = __bfloat1622float2(s5a), f5b = __bfloat1622float2(s5b);
                  float2 f6a = __bfloat1622float2(s6a), f6b = __bfloat1622float2(s6b);
                  float2 f7a = __bfloat1622float2(s7a), f7b = __bfloat1622float2(s7b);
                  float4 acc0 = acc_token_tail_smem4[p0];
                  float4 acc1 = acc_token_tail_smem4[p1];
                  float4 acc2 = acc_token_tail_smem4[p2];
                  float4 acc3 = acc_token_tail_smem4[p3];
                  float4 acc4 = acc_token_tail_smem4[p4];
                  float4 acc5 = acc_token_tail_smem4[p5];
                  float4 acc6 = acc_token_tail_smem4[p6];
                  float4 acc7 = acc_token_tail_smem4[p7];
                  acc0.x += f0a.x; acc0.y += f0a.y; acc0.z += f0b.x; acc0.w += f0b.y;
                  acc1.x += f1a.x; acc1.y += f1a.y; acc1.z += f1b.x; acc1.w += f1b.y;
                  acc2.x += f2a.x; acc2.y += f2a.y; acc2.z += f2b.x; acc2.w += f2b.y;
                  acc3.x += f3a.x; acc3.y += f3a.y; acc3.z += f3b.x; acc3.w += f3b.y;
                  acc4.x += f4a.x; acc4.y += f4a.y; acc4.z += f4b.x; acc4.w += f4b.y;
                  acc5.x += f5a.x; acc5.y += f5a.y; acc5.z += f5b.x; acc5.w += f5b.y;
                  acc6.x += f6a.x; acc6.y += f6a.y; acc6.z += f6b.x; acc6.w += f6b.y;
                  acc7.x += f7a.x; acc7.y += f7a.y; acc7.z += f7b.x; acc7.w += f7b.y;
                  acc_token_tail_smem4[p0] = acc0;
                  acc_token_tail_smem4[p1] = acc1;
                  acc_token_tail_smem4[p2] = acc2;
                  acc_token_tail_smem4[p3] = acc3;
                  acc_token_tail_smem4[p4] = acc4;
                  acc_token_tail_smem4[p5] = acc5;
                  acc_token_tail_smem4[p6] = acc6;
                  acc_token_tail_smem4[p7] = acc7;
                }
                for (; p < tail_float4; p += INTRA_NODE_RED_GROUP::size()) {
                  const int e = head_bf16x2 + (p << 1);
                  __nv_bfloat162 sa = load_token_base_ptr[e], sb = load_token_base_ptr[e + 1];
                  float2 fa = __bfloat1622float2(sa), fb = __bfloat1622float2(sb);
                  float4 acc = acc_token_tail_smem4[p];
                  acc.x += fa.x; acc.y += fa.y; acc.z += fb.x; acc.w += fb.y;
                  acc_token_tail_smem4[p] = acc;
                }
              }
            }

            if constexpr(BACKWARD_COMBINE) {
              #pragma unroll
              for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
                int prob_element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
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
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(token_stage));
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
          __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage));
          float* store_prob_base_ptr;
          if constexpr(BACKWARD_COMBINE) {
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

          // Store the register-resident token head.
          #pragma unroll
          for (int n = 0; n < MAX_NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD_INTRA; n++) {
            int element_id = (n * INTRA_NODE_RED_GROUP::size()) + INTRA_NODE_RED_GROUP::thread_rank();
            if (element_id < head_bf16x2) {
              store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[n]);
            }
          }
          // Store the token tail from the SMEM accumulator, or zeros if no tail element was touched.
          if (tail_bf16x2 > 0) {
            const __nv_bfloat162 zero_bf16x2 = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
            if (!tail_initialized) {
              int p = INTRA_NODE_RED_GROUP::thread_rank();
              for (; p + 7 * INTRA_NODE_RED_GROUP::size() < tail_float4;
                   p += 8 * INTRA_NODE_RED_GROUP::size()) {
                const int p0 = p, p1 = p + INTRA_NODE_RED_GROUP::size();
                const int p2 = p + 2 * INTRA_NODE_RED_GROUP::size(), p3 = p + 3 * INTRA_NODE_RED_GROUP::size();
                const int p4 = p + 4 * INTRA_NODE_RED_GROUP::size(), p5 = p + 5 * INTRA_NODE_RED_GROUP::size();
                const int p6 = p + 6 * INTRA_NODE_RED_GROUP::size(), p7 = p + 7 * INTRA_NODE_RED_GROUP::size();
                const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                store_token_base_ptr[e0] = zero_bf16x2; store_token_base_ptr[e0 + 1] = zero_bf16x2;
                store_token_base_ptr[e1] = zero_bf16x2; store_token_base_ptr[e1 + 1] = zero_bf16x2;
                store_token_base_ptr[e2] = zero_bf16x2; store_token_base_ptr[e2 + 1] = zero_bf16x2;
                store_token_base_ptr[e3] = zero_bf16x2; store_token_base_ptr[e3 + 1] = zero_bf16x2;
                store_token_base_ptr[e4] = zero_bf16x2; store_token_base_ptr[e4 + 1] = zero_bf16x2;
                store_token_base_ptr[e5] = zero_bf16x2; store_token_base_ptr[e5 + 1] = zero_bf16x2;
                store_token_base_ptr[e6] = zero_bf16x2; store_token_base_ptr[e6 + 1] = zero_bf16x2;
                store_token_base_ptr[e7] = zero_bf16x2; store_token_base_ptr[e7 + 1] = zero_bf16x2;
              }
              for (; p < tail_float4; p += INTRA_NODE_RED_GROUP::size()) {
                const int e = head_bf16x2 + (p << 1);
                store_token_base_ptr[e] = zero_bf16x2;
                store_token_base_ptr[e + 1] = zero_bf16x2;
              }
            } else {
              int p = INTRA_NODE_RED_GROUP::thread_rank();
              for (; p + 7 * INTRA_NODE_RED_GROUP::size() < tail_float4;
                   p += 8 * INTRA_NODE_RED_GROUP::size()) {
                const int p0 = p, p1 = p + INTRA_NODE_RED_GROUP::size();
                const int p2 = p + 2 * INTRA_NODE_RED_GROUP::size(), p3 = p + 3 * INTRA_NODE_RED_GROUP::size();
                const int p4 = p + 4 * INTRA_NODE_RED_GROUP::size(), p5 = p + 5 * INTRA_NODE_RED_GROUP::size();
                const int p6 = p + 6 * INTRA_NODE_RED_GROUP::size(), p7 = p + 7 * INTRA_NODE_RED_GROUP::size();
                const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                float4 acc0 = acc_token_tail_smem4[p0];
                float4 acc1 = acc_token_tail_smem4[p1];
                float4 acc2 = acc_token_tail_smem4[p2];
                float4 acc3 = acc_token_tail_smem4[p3];
                float4 acc4 = acc_token_tail_smem4[p4];
                float4 acc5 = acc_token_tail_smem4[p5];
                float4 acc6 = acc_token_tail_smem4[p6];
                float4 acc7 = acc_token_tail_smem4[p7];
                store_token_base_ptr[e0] = __float22bfloat162_rn(make_float2(acc0.x, acc0.y));
                store_token_base_ptr[e0 + 1] = __float22bfloat162_rn(make_float2(acc0.z, acc0.w));
                store_token_base_ptr[e1] = __float22bfloat162_rn(make_float2(acc1.x, acc1.y));
                store_token_base_ptr[e1 + 1] = __float22bfloat162_rn(make_float2(acc1.z, acc1.w));
                store_token_base_ptr[e2] = __float22bfloat162_rn(make_float2(acc2.x, acc2.y));
                store_token_base_ptr[e2 + 1] = __float22bfloat162_rn(make_float2(acc2.z, acc2.w));
                store_token_base_ptr[e3] = __float22bfloat162_rn(make_float2(acc3.x, acc3.y));
                store_token_base_ptr[e3 + 1] = __float22bfloat162_rn(make_float2(acc3.z, acc3.w));
                store_token_base_ptr[e4] = __float22bfloat162_rn(make_float2(acc4.x, acc4.y));
                store_token_base_ptr[e4 + 1] = __float22bfloat162_rn(make_float2(acc4.z, acc4.w));
                store_token_base_ptr[e5] = __float22bfloat162_rn(make_float2(acc5.x, acc5.y));
                store_token_base_ptr[e5 + 1] = __float22bfloat162_rn(make_float2(acc5.z, acc5.w));
                store_token_base_ptr[e6] = __float22bfloat162_rn(make_float2(acc6.x, acc6.y));
                store_token_base_ptr[e6 + 1] = __float22bfloat162_rn(make_float2(acc6.z, acc6.w));
                store_token_base_ptr[e7] = __float22bfloat162_rn(make_float2(acc7.x, acc7.y));
                store_token_base_ptr[e7 + 1] = __float22bfloat162_rn(make_float2(acc7.z, acc7.w));
              }
              for (; p < tail_float4; p += INTRA_NODE_RED_GROUP::size()) {
                const int e = head_bf16x2 + (p << 1);
                float4 acc = acc_token_tail_smem4[p];
                store_token_base_ptr[e] = __float22bfloat162_rn(make_float2(acc.x, acc.y));
                store_token_base_ptr[e + 1] = __float22bfloat162_rn(make_float2(acc.z, acc.w));
              }
            }
          }

          // Store the prob(optional).
          if constexpr(BACKWARD_COMBINE) {
            #pragma unroll
            for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
              int prob_element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
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
              uint16_t* current_token_addr = rdma_intra_node_red_token_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * HIDDEN_DIM;
              // Copy the reduced token from the S2G shared stage to the per-destination global buffer.
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(current_token_addr),
                                       reinterpret_cast<const void*>(smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage)),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

              // Store the prob from shared to global(Optional).
              if constexpr(BACKWARD_COMBINE) {
                float* current_prob_addr = rdma_intra_node_red_prob_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * (experts_per_rank * num_of_ranks_per_node);
                cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                         cuda::ptx::space_shared,
                                         reinterpret_cast<void*>(current_prob_addr),
                                         reinterpret_cast<const void*>(smem_buffer_ptr->get_intra_node_prob_S2G(dst_token_stage)),
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
          if constexpr(STREAMING_BATCH > 0) {
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
                  *((volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter) = cumulative_produced;
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
          if constexpr(STREAMING_BATCH > 0) {
            cumulative_produced += streaming_pending;
            __threadfence();  // device scope: flush L2→VRAM for NIC visibility (GDR)
            *((volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter) = cumulative_produced;
          }
        }
      }
      streaming_pending = 0;
      additional_in_flight_s2g = 0;
    }

    // Signal chunk-complete mbarrier unconditionally (for parity tracking)
    if constexpr(NUM_OF_NODES != 1) {
      if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
          cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id]);
        }
      }
    }
  }

  // No post-loop cleanup needed: every chunk is fully drained and signaled at chunk end.
}

// Device function for inter-node node2node(RDMA) warp for combine kernel. There can be only 1 inter-node warp per CUDA block!
// Uses ncclGin API (net.put, net.signal)
template<typename INTER_NODE_RDMA_GROUP,
         typename SMEM_TYPE,
         int NUM_OF_STAGES_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
__forceinline__ __device__ void inter_node_N2N_warp_group_device_function(const int local_rank,
                                                                 const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const bool* rdma_to_attn_map,
                                                                 ncclDevComm_t* dcomms,
                                                                 ncclWindow_t nccl_window,
                                                                 int num_gin_comms,
                                                                 int num_ctx_per_comm,
                                                                 void* gin_base_ptr,
                                                                 unsigned signals_base,
                                                                 unsigned combine_signal_offset,
                                                                 const struct combine_memory_region_info_t *mr_info,
                                                                 SMEM_TYPE* smem_buffer_ptr,
                                                                 const int HIDDEN_DIM,
                                                                 const int experts_per_rank)
{
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(INTER_NODE_RDMA_GROUP::size() == 32, "INTER_NODE_RDMA_GROUP should be 1 warp.");
  static_assert(INTER_NODE_RDMA_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_RDMA_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of sizeof(rdma_to_attn_map_load_t).");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  // int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Mr_infos and rdma_mbarrier_buffer in shared memory.
  struct combine_memory_region_info_t *smem_mr_info_ptr = nullptr;
  uint64_t *intra_node_to_rdma_mbarrier_buffer_ptr = nullptr;
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      smem_mr_info_ptr[0] = mr_info[0];
    }
    intra_node_to_rdma_mbarrier_buffer_ptr = smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer;
  }
  __syncwarp();

  // Total number of chunks to produce for RDMA warps to consume.
  int NUM_OF_CHUNKS_PER_RANK = (num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK + 1;
  int TOTAL_NUM_OF_CHUNKS = (NUM_OF_NODES - 1) * NUM_OF_CHUNKS_PER_RANK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // INTRA_NODE_RED_GROUP should be 1 warp.
  // The inter_node_N2N_warp should process the same chunk as intra_node_red_warp(They belong to the same block.)
  uint32_t token_consumer_parity = 0;
  uint32_t cumulative_sent = 0;  // Cumulative count of active tokens RDMA-put across all chunks (never reset)
  // Loop for every chunks.
  for (int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS) {
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // With split comm (nNodes ranks), node index IS the rank in the communicator
    int remote_global_rank = node_id;
    int rank_in_remote = node_id < node_rank ? node_rank - 1 : node_rank;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);

    // Distribute chunks across comms for parallelism
    // Use chunk_id to select comm - both sender (here) and receiver (G2S)
    // use the same formula so signals match
    int total_channels = num_gin_comms * num_ctx_per_comm;
    int global_channel = chunk_id % total_channels;
    int comm_idx, ctx_idx;
    get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
    ncclGin net(dcomms[comm_idx], ctx_idx);
    ncclTeam world = ncclTeamWorld(dcomms[comm_idx]);
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    if (chunk_id * NUM_OF_TOKENS_PER_CHUNK + token_range > num_of_tokens_per_rank) {
      token_range = num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    }
    constexpr int STREAMING_BATCH = HYBRIDEP_COMBINE_RDMA_STREAMING_BATCH;
    if constexpr(STREAMING_BATCH > 0) {
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

          bool should_flush = batch_count > 0 &&
                              (!need_write || is_last || batch_count >= STREAMING_BATCH);

          if (should_flush) {
            while (*((volatile uint32_t*)smem_buffer_ptr->rdma_streaming_counter) < (cumulative_sent + batch_count)) {}

            int batch_start_token = batch_start_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
            size_t token_src_offset = smem_mr_info_ptr->rdma_intra_node_red_token_offset +
                (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                HIDDEN_DIM * sizeof(uint16_t);
            size_t token_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_token_offset +
                (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                HIDDEN_DIM * sizeof(uint16_t);
            net.put(world, remote_global_rank,
                    nccl_window, token_dst_offset,
                    nccl_window, token_src_offset,
                    batch_count * HIDDEN_DIM * sizeof(uint16_t),
                    ncclGin_None{}, ncclGin_None{}, ncclCoopThread());

            if constexpr(BACKWARD_COMBINE) {
              size_t prob_src_offset = smem_mr_info_ptr->rdma_intra_node_red_prob_offset +
                  (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                  (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
              size_t prob_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_prob_offset +
                  (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                  (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
              net.put(world, remote_global_rank,
                      nccl_window, prob_dst_offset,
                      nccl_window, prob_src_offset,
                      batch_count * (experts_per_rank * num_of_ranks_per_node) * sizeof(float),
                      ncclGin_None{}, ncclGin_None{}, ncclCoopThread());
            }

            cumulative_sent += batch_count;
            batch_count = 0;
            batch_start_in_chunk = -1;
          }
        }
      }

      // Wait for mbarrier (parity tracking -- reduction warp always arrives)
      while (!cuda::ptx::mbarrier_try_wait_parity(
          &intra_node_to_rdma_mbarrier_buffer_ptr[rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id],
          token_consumer_parity)) {}

      // Signal remote
      __syncwarp();
      if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
        constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
        unsigned signal_id = signals_base + combine_signal_offset +
            local_rank * (NUM_OF_NODES * MAX_CHUNKS_PER_RANK) +
            node_rank * MAX_CHUNKS_PER_RANK + chunk_id;
        net.signal(world, remote_global_rank,
                   ncclGin_SignalAdd{signal_id, 1},
                   ncclCoopThread(), ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread);
      }
      __syncwarp();

      // No consumed handshake needed: cumulative counter never resets.

    } else {
      // ---- FALLBACK PATH (STREAMING_BATCH == 0): original mbarrier-first ----
      while (!cuda::ptx::mbarrier_try_wait_parity(
          &intra_node_to_rdma_mbarrier_buffer_ptr[rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id],
          token_consumer_parity)) {}

      if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
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

          bool should_flush = batch_count > 0 &&
                              (!need_write || is_last || batch_count >= max_batch);

          if (should_flush) {
            int batch_start_token = batch_start_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
            size_t token_src_offset = smem_mr_info_ptr->rdma_intra_node_red_token_offset +
                (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                HIDDEN_DIM * sizeof(uint16_t);
            size_t token_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_token_offset +
                (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                HIDDEN_DIM * sizeof(uint16_t);
            net.put(world, remote_global_rank,
                    nccl_window, token_dst_offset,
                    nccl_window, token_src_offset,
                    batch_count * HIDDEN_DIM * sizeof(uint16_t),
                    ncclGin_None{}, ncclGin_None{}, ncclCoopThread());

            if constexpr(BACKWARD_COMBINE) {
              size_t prob_src_offset = smem_mr_info_ptr->rdma_intra_node_red_prob_offset +
                  (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                  (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
              size_t prob_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_prob_offset +
                  (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + batch_start_token) *
                  (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
              net.put(world, remote_global_rank,
                      nccl_window, prob_dst_offset,
                      nccl_window, prob_src_offset,
                      batch_count * (experts_per_rank * num_of_ranks_per_node) * sizeof(float),
                      ncclGin_None{}, ncclGin_None{}, ncclCoopThread());
            }

            batch_count = 0;
            batch_start_in_chunk = -1;
          }
        }
      }
      __syncwarp();
      if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
        constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
        unsigned signal_id = signals_base + combine_signal_offset +
            local_rank * (NUM_OF_NODES * MAX_CHUNKS_PER_RANK) +
            node_rank * MAX_CHUNKS_PER_RANK + chunk_id;
        net.signal(world, remote_global_rank,
                   ncclGin_SignalAdd{signal_id, 1},
                   ncclCoopThread(), ncclGin_None{},
                   cuda::thread_scope_thread, cuda::thread_scope_thread);
      }
      __syncwarp();
    }
  }
  token_consumer_parity ^= 1;
}

// Device function for inter-node G2S warp for combine kernel.
template<typename SMEM_TYPE,
         typename INTER_NODE_G2S_GROUP,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_TOKENS_PER_GROUP,
         bool BACKWARD_COMBINE>
__forceinline__ __device__ void inter_node_G2S_warp_group_device_function(const int local_rank,
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
                                                                 const int HIDDEN_DIM,
                                                                 const int experts_per_rank)
{
  // The warps from inter-node G2S warp group will be divided into multiple independent pipeline.
  // Each pipeline can only have 1 warp, so INTER_NODE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_BLOCK and warp has the same meaning as pipeline in inter-node G2S warp group.
  // Number of pipeline should match inter-node red warp group, so they can coupled into multiple independent data pipeline within a CUDA block.
  // Evenly distribute the inter-node G2S FIFO to every pipeline(warp) within the inter-node G2S warp group.
  // When inter-node G2S warp group only has 1 warp, then the algorith is the same as old version(1 pipeline per CUDA block).
  static_assert(NUM_OF_STAGES_G2S % INTER_NODE_G2S_GROUP::warp_size() == 0, "NUM_OF_STAGES_G2S must be multiple of inter-node G2S warp group warp size.");
  constexpr int NUM_OF_STAGES_G2S_PER_WARP = NUM_OF_STAGES_G2S / INTER_NODE_G2S_GROUP::warp_size();
  // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks.
  // This is different than other functions where chunks are assigned to different CUDA blocks.
  static_assert(NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
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
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // Starting and ending index within G2S FIFO for this warp(pipeline).
  const int starting_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * INTER_NODE_G2S_GROUP::warp_rank();
  const int ending_G2S_index = NUM_OF_STAGES_G2S_PER_WARP * (INTER_NODE_G2S_GROUP::warp_rank() + 1);
  if constexpr (NUM_OF_NODES == 1) {
    // Single-node optimized path: warp-cooperative G2S.
    // Lanes load sparse_to_dense_map in parallel (up to 32 ranks per warp pass),
    // then valid lanes issue TMA to different stages simultaneously.
    // RED processes stages sequentially.
    //
    // Parity protocol: G2S tracks a "global_offset" counting total stages filled.
    // For a lane with global rank R among valid entries, its stage and parity are:
    //   stage_idx = starting + (global_offset + R) % ring_len
    //   parity    = 1 ^ ((global_offset + R) / ring_len) & 1
    // This matches RED's sequential consumption exactly.
    //
    // For num_of_ranks_per_node > 32, ranks are processed in warp-sized slices.
    // Each slice's valid lanes get a global_rank = slice_offset + local_lane_rank.
    // The last-token flag is set only by the globally last valid lane.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    const int ring_len = ending_G2S_index - starting_G2S_index;
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));
    const uint32_t prob_bytes =
        (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));

    // Track total stages filled across all tokens.
    int global_offset = 0;

    // Iterate through all chunks.
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

      const bool* rdma_to_attn_map_load_base_addr = rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;

      for (int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS) {
        for (int k = INTER_NODE_G2S_GROUP::warp_rank(); k < NUM_OF_TOKENS_PER_GROUP; k += INTER_NODE_G2S_GROUP::warp_size()) {
          int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
          if (current_token_id >= current_chunk_size) {
            break;
          }
          bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
          if (!token_needed_by_this_node) {
            continue;
          }

          const int32_t* sparse_to_dense_row = sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * num_of_ranks_per_node;

          // First pass: count total valid ranks across all slices.
          int total_valid_count = 0;
          for (int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE) {
            const int rank_id = rank_base + lane_id;
            const bool lane_active = (rank_id < num_of_ranks_per_node);
            const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
            const unsigned mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
            total_valid_count += __popc(mask);
          }
          if (total_valid_count == 0) {
            continue;
          }

          // Second pass: issue TMA in batches of ring_len to prevent stage
          // collisions when total_valid_count > ring_len (e.g. top_k=8, ring_len=6).
          int ranks_issued = 0;
          while (ranks_issued < total_valid_count) {
            const int batch_end = (ranks_issued + ring_len < total_valid_count)
                                    ? ranks_issued + ring_len : total_valid_count;

            int slice_offset = 0;
            for (int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE) {
              const int rank_id = rank_base + lane_id;
              const bool lane_active = (rank_id < num_of_ranks_per_node);
              const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
              const unsigned valid_mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
              const int slice_valid = __popc(valid_mask);
              const bool lane_valid = lane_active && s2d_val != -1;
              const int local_lane_rank = __popc(valid_mask & ((1u << lane_id) - 1));
              const int global_rank = slice_offset + local_lane_rank;
              const bool in_batch = lane_valid && global_rank >= ranks_issued && global_rank < batch_end;

              if (in_batch) {
                const int rank_in_batch = global_rank - ranks_issued;
                const int my_abs_offset = global_offset + rank_in_batch;
                const int stage_idx = starting_G2S_index + (my_abs_offset % ring_len);
                const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

                while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(stage_idx), parity)){}

                const uint16_t* rank_token_ptr = remote_expert_input_token[rank_id];
                uint32_t total_tx_size = 0;
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(stage_idx)),
                                         reinterpret_cast<const void*>(rank_token_ptr + (s2d_val * HIDDEN_DIM)),
                                         token_bytes,
                                         smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx));

                total_tx_size += token_bytes;

                if constexpr(BACKWARD_COMBINE) {
                  const float* rank_prob_ptr = remote_expert_input_prob[rank_id];
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(stage_idx)),
                                           reinterpret_cast<const void*>(rank_prob_ptr + (s2d_val * (experts_per_rank * num_of_ranks_per_node))),
                                           prob_bytes,
                                           smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx));

                  total_tx_size += prob_bytes;
                }

                smem_buffer_ptr->inter_node_flag_G2S_buffer[stage_idx] = (global_rank == total_valid_count - 1);

                cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                     cuda::ptx::scope_cta,
                                                     cuda::ptx::space_shared,
                                                     smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx),
                                                     total_tx_size);
              }

              slice_offset += slice_valid;
            }

            global_offset += (batch_end - ranks_issued);
            ranks_issued = batch_end;
          }
        }
      }
    }
  } else {
    // Multi-node warp-cooperative path: local block uses full warp (same as NUM_OF_NODES==1),
    // then RDMA block runs on lane 0 only, sharing the same global_offset stream.
    // After RDMA, global_offset is broadcast back to all lanes via __shfl_sync.
    constexpr int WARP_SIZE = 32;
    const int lane_id = (int)(threadIdx.x & (WARP_SIZE - 1));
    const int ring_len = ending_G2S_index - starting_G2S_index;
    const uint32_t token_bytes = (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));
    const uint32_t prob_bytes =
        (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));

    // Track total stages filled across all tokens (local + RDMA).
    int global_offset = 0;

    // Iterate through all chunks. All chunks will assign to all CUDA block.
    for (int i = 0; i < total_num_of_chunks; i++) {
      // How many rdma_to_attn load iter(a.k.a token group) for this chunk.
      int num_of_token_groups_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if (remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1) {
        num_of_token_groups_for_current_chunk = ((remainder_chunk_size - 1) / NUM_OF_TOKENS_PER_GROUP) + 1;
        current_chunk_size = remainder_chunk_size;
      } else {
        num_of_token_groups_for_current_chunk = NUM_OF_TOKEN_GROUPS_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }

      const bool* rdma_to_attn_map_load_base_addr = rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + i * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;

      // RDMA-only state. rdma_flag_clear is per-chunk, only used by lane 0.
      const bool* attn_to_rdma_map_load_base_addr = nullptr;
      bool rdma_flag_clear[NUM_OF_NODES];
      if constexpr (NUM_OF_NODES > 1) {
        attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1);

        // Pre-wait for ALL remote signals for this chunk before processing any tokens.
        // This guarantees RDMA data is available for all tokens in this chunk,
        // eliminating the per-token signal-wait overhead inside the token loop.
        if (lane_id == 0) {
          constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
          int total_channels = num_gin_comms * num_ctx_per_comm;
          int global_channel = i % total_channels;
          int comm_idx, ctx_idx;
          get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
          ncclGin net(dcomms[comm_idx], ctx_idx);
          for (int n = 1; n < NUM_OF_NODES; n++) {
            int node_id_for_signal = node_rank >= n
                ? node_rank - n : node_rank + NUM_OF_NODES - n;
            unsigned signal_id = signals_base + combine_signal_offset
                + local_rank * (NUM_OF_NODES * MAX_CHUNKS_PER_RANK)
                + node_id_for_signal * MAX_CHUNKS_PER_RANK + i;
            net.waitSignal(ncclCoopThread(), signal_id, expected_flag_value);
          }
        }
        __syncwarp(0xffffffff);

        #pragma unroll
        for (int jj = 0; jj < NUM_OF_NODES; ++jj) {
          rdma_flag_clear[jj] = true;
        }
      }

      // Iterate through all token groups within this chunk which assign to this CUDA block.
      for (int j = blockIdx.x; j < num_of_token_groups_for_current_chunk; j += NUM_OF_BLOCKS) {
        // Iterate through all dst(output) tokens within this token group.
        // Assign each dst token to each G2S warp(pipeline) using a round-robin fasion.
        for (int k = INTER_NODE_G2S_GROUP::warp_rank(); k < NUM_OF_TOKENS_PER_GROUP; k += INTER_NODE_G2S_GROUP::warp_size()) {
          int current_token_id = j * NUM_OF_TOKENS_PER_GROUP + k;
          // If the current token is out-of-bound, then just end this load iter.
          if (current_token_id >= current_chunk_size) {
            break;
          }
          // Each dst token need to accumulate src tokens from local node's ranks, and src tokens from rdma inter-node buffers.
          // Accumulate local tokens first (warp-cooperative), then rdma tokens (lane 0 only).

          // Check whether this dst token is needed by this(local) node. If not needed, skip local accumulation only.
          // NOTE: Do NOT use 'continue' here -- RDMA block must still run even when token_needed_by_this_node is false.
          bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
          if (token_needed_by_this_node) {
            const int32_t* sparse_to_dense_row = sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * num_of_ranks_per_node;

            // First pass: count total valid ranks across all slices.
            int total_valid_count = 0;
            for (int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE) {
              const int rank_id = rank_base + lane_id;
              const bool lane_active = (rank_id < num_of_ranks_per_node);
              const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
              const unsigned mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
              total_valid_count += __popc(mask);
            }

            if (total_valid_count > 0) {
              // Second pass: issue TMA in batches of ring_len to prevent stage
              // collisions when total_valid_count > ring_len.
              int ranks_issued = 0;
              while (ranks_issued < total_valid_count) {
                const int batch_end = (ranks_issued + ring_len < total_valid_count)
                                        ? ranks_issued + ring_len : total_valid_count;

                int slice_offset = 0;
                for (int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE) {
                  const int rank_id = rank_base + lane_id;
                  const bool lane_active = (rank_id < num_of_ranks_per_node);
                  const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
                  const unsigned valid_mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
                  const int slice_valid = __popc(valid_mask);
                  const bool lane_valid = lane_active && s2d_val != -1;
                  const int local_lane_rank = __popc(valid_mask & ((1u << lane_id) - 1));
                  const int global_rank = slice_offset + local_lane_rank;
                  const bool in_batch = lane_valid && global_rank >= ranks_issued && global_rank < batch_end;

                  if (in_batch) {
                    const int rank_in_batch = global_rank - ranks_issued;
                    const int my_abs_offset = global_offset + rank_in_batch;
                    const int stage_idx = starting_G2S_index + (my_abs_offset % ring_len);
                    const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

                    while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(stage_idx), parity)){}

                    const uint16_t* rank_token_ptr = remote_expert_input_token[rank_id];
                    uint32_t total_tx_size = 0;
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(stage_idx)),
                                             reinterpret_cast<const void*>(rank_token_ptr + (s2d_val * HIDDEN_DIM)),
                                             token_bytes,
                                             smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx));

                    total_tx_size += token_bytes;

                    if constexpr(BACKWARD_COMBINE) {
                      const float* rank_prob_ptr = remote_expert_input_prob[rank_id];
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                               cuda::ptx::space_global,
                                               reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(stage_idx)),
                                               reinterpret_cast<const void*>(rank_prob_ptr + (s2d_val * (experts_per_rank * num_of_ranks_per_node))),
                                               prob_bytes,
                                               smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx));

                      total_tx_size += prob_bytes;
                    }

                    smem_buffer_ptr->inter_node_flag_G2S_buffer[stage_idx] = (global_rank == total_valid_count - 1);

                    cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                         cuda::ptx::scope_cta,
                                                         cuda::ptx::space_shared,
                                                         smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx),
                                                         total_tx_size);
                  }

                  slice_offset += slice_valid;
                }

                global_offset += (batch_end - ranks_issued);
                ranks_issued = batch_end;
              }
            }
          } // end if (token_needed_by_this_node)

          if constexpr (NUM_OF_NODES > 1) {
          // Warp-cooperative RDMA: each lane maps to a remote node (lane_id -> n = lane_id + 1).
          // Valid lanes issue TMAs in parallel to different stages.
          // Signal waits remain on lane 0 only (ncclGin safety).
          // Since NUM_OF_NODES <= 33, a single warp pass covers all remote nodes.
          static_assert(NUM_OF_NODES <= 33, "NUM_OF_NODES must fit in a single warp pass for RDMA parallelization.");

          // Ensure all local TMAs are committed before RDMA (memory fence).
          __syncwarp(0xffffffff);

          const bool* attn_to_rdma_map_load_addr =
              attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);

          // Each lane maps to one remote node.
          const int rdma_n = lane_id + 1;
          const bool rdma_lane_active = (rdma_n < NUM_OF_NODES);

          int rdma_node_id = 0, rdma_buffer_tile_id = 0;
          bool rdma_entry_valid = false;
          if (rdma_lane_active) {
            rdma_node_id = node_rank >= rdma_n ? node_rank - rdma_n : node_rank + NUM_OF_NODES - rdma_n;
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

            if (rdma_lane_active && rdma_entry_valid) {
              const int my_abs_offset = global_offset + rdma_local_rank;
              const int stage_idx = starting_G2S_index + (my_abs_offset % ring_len);
              const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

              // Wait for consumer to free this stage.
              // When rdma_valid_count > ring_len, overflow ranks block here until RED consumes earlier stages (parity protocol).
              while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(stage_idx), parity)){}

              // Load the src token from this rdma inter-node group buffer chunk to shared memory entry.
              uint32_t total_tx_size = 0;
              const uint16_t* rdma_inter_node_group_token_load_addr = rdma_inter_node_group_token +
                                                                      (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK +
                                                                      i * NUM_OF_TOKENS_PER_CHUNK +
                                                                      j * NUM_OF_TOKENS_PER_GROUP + k) * HIDDEN_DIM;
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(stage_idx)),
                                       reinterpret_cast<const void*>(rdma_inter_node_group_token_load_addr),
                                       token_bytes,
                                       smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx));

              total_tx_size += token_bytes;

              if constexpr (BACKWARD_COMBINE) {
                const float* rdma_inter_node_group_prob_load_addr = rdma_inter_node_group_prob +
                                                                    (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK +
                                                                    i * NUM_OF_TOKENS_PER_CHUNK +
                                                                    j * NUM_OF_TOKENS_PER_GROUP + k) * (experts_per_rank * num_of_ranks_per_node);

                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(stage_idx)),
                                         reinterpret_cast<const void*>(rdma_inter_node_group_prob_load_addr),
                                         prob_bytes,
                                         smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx));

                total_tx_size += prob_bytes;
              }

              // Inter-node token does not need flag since the red warp group will also read attn_to_rdma_map.
              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx),
                                                   total_tx_size);
            }

            // ALL lanes advance uniformly -- no __shfl_sync needed.
            // rdma_valid_count is identical for all 32 lanes (from __popc of same __ballot_sync result).
            global_offset += rdma_valid_count;
          }
          } // end if constexpr (NUM_OF_NODES > 1)
        }
      }
    }
  }
  if constexpr (NUM_OF_NODES > 1) {
    // Update residue flags.
    int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;
    for (int node_id = blockIdx.x; node_id < NUM_OF_NODES - 1; node_id += gridDim.x) {
      uint64_t *residue_flag_base_ptr = rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank + num_of_chunks_per_rank);
      for (int flag_id = INTER_NODE_G2S_GROUP::thread_rank(); flag_id < residue_flag_count; flag_id += INTER_NODE_G2S_GROUP::size()) {
        residue_flag_base_ptr[flag_id] = expected_flag_value;
      }
    }
  }
}

// Device function for inter-node reduction warp group for combine kernel.
template<typename SMEM_TYPE,
         typename INTER_NODE_RED_GROUP,
         int NUM_OF_DATA_PIPELINE_PER_BLOCK,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_STAGES_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         int NUM_OF_TOKENS_PER_GROUP,
         bool BACKWARD_COMBINE>
__forceinline__ __device__ void inter_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const bool* rdma_to_attn_map,
                                                                 const bool* attn_to_rdma_map,
                                                                 uint16_t* attn_output_token,
                                                                 float* attn_output_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr,
                                                                 const int HIDDEN_DIM,
                                                                 const int experts_per_rank)
{

  // The warps from inter-node red warp group will be divided into multiple independent pipeline. Each pipeline has INTER_NODE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK warps.
  // Number of pipeline should match inter-node G2S warp group, so they can coupled into multiple independent data pipeline within a CUDA block.
  static_assert(INTER_NODE_RED_GROUP::warp_size() % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "The warp count of inter-node red warp group must be multiple of NUM_OF_DATA_PIPELINE_PER_BLOCK.");
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_THREADS_PER_PIPELINE = (INTER_NODE_RED_GROUP::warp_size() / NUM_OF_DATA_PIPELINE_PER_BLOCK) * WARP_SIZE;
  // Evenly distribute the inter-node G2S FIFO to every pipeline within the inter-node red warp group.
  // When NUM_OF_DATA_PIPELINE_PER_BLOCK = 1 and INTER_NODE_RED_GROUP::warp_size() = 4, then the algorith is the same as old version(1 pipeline w/ 4 warps per CUDA block).
  static_assert(NUM_OF_STAGES_G2S % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "NUM_OF_STAGES_G2S must be multiple of data pipeline per CUDA block.");
  constexpr int NUM_OF_STAGES_G2S_PER_PIPELINE = NUM_OF_STAGES_G2S / NUM_OF_DATA_PIPELINE_PER_BLOCK;
  // Evenly distribute the inter-node S2G FIFO to every pipeline within the inter-node red warp group.
  static_assert(NUM_OF_STAGES_S2G % NUM_OF_DATA_PIPELINE_PER_BLOCK == 0, "NUM_OF_STAGES_S2G must be multiple of data pipeline per CUDA block.");
  constexpr int NUM_OF_STAGES_S2G_PER_PIPELINE = NUM_OF_STAGES_S2G / NUM_OF_DATA_PIPELINE_PER_BLOCK;
  // All chunks in output buffer(attn buffer) will be divided into token groups and assigned to different CUDA blocks.
  // This is different than other functions where chunks are assigned to different CUDA blocks.
  static_assert(NUM_OF_TOKENS_PER_CHUNK % NUM_OF_TOKENS_PER_GROUP == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of NUM_OF_TOKENS_PER_GROUP.");
  constexpr int NUM_OF_TOKEN_GROUPS_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / NUM_OF_TOKENS_PER_GROUP;

  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");

  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  constexpr int REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN = COMBINE_REG_HEAD_HIDDEN_DIM / 2;
  const int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;

  // Split the token into a register-resident "head" and a shared-memory "tail".
  // The head fits in registers per thread (bf16x2 lanes), giving a fixed per-thread
  // workload and avoiding large register arrays when HIDDEN_DIM is large. The tail
  // (if any) is handled separately (float4 = 2 bf16x2) from shared memory.
  constexpr int NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD =
      ((REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;
  const int head_bf16x2 = NUM_OF_BF16X2_ELEMENTS_PER_TOKEN > REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN
          ? REG_HEAD_BF16X2_ELEMENTS_PER_TOKEN
          : NUM_OF_BF16X2_ELEMENTS_PER_TOKEN;
  const int tail_bf16x2 = NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - head_bf16x2;
  const int tail_float4 = tail_bf16x2 >> 1;

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
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
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

    const bool* rdma_to_attn_map_load_base_addr = rdma_to_attn_map + (node_rank * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK);
    const bool* attn_to_rdma_map_load_base_addr = nullptr;
    if constexpr (NUM_OF_NODES > 1) {
      attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1);
    }
    uint16_t* attn_output_token_base_ptr = attn_output_token + (i * NUM_OF_TOKENS_PER_CHUNK) * HIDDEN_DIM;
    float* attn_output_prob_base_ptr;
    if constexpr(BACKWARD_COMBINE) {
      attn_output_prob_base_ptr = attn_output_prob + (i * NUM_OF_TOKENS_PER_CHUNK) * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES);
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
        float2 acc_token_fp32[NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD];
        // Optional Accumulator for this dst token prob.
        // Different node's prob need to be gathered together to output.
        // 0 used for local node's prob, [1, NUM_OF_NODES - 1] used for remote node's prob.
        // Flattened array: acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m] for 2D access
        // Use MAX size for compile-time array allocation, actual size determined by runtime experts_per_rank
        using acc_prob_storage_type =
            acc_prob_storage_t<BACKWARD_COMBINE,
                               NUM_OF_NODES * MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD>;
        [[maybe_unused]] acc_prob_storage_type acc_prob_storage;
        [[maybe_unused]] float* acc_prob_ptr = nullptr;
        if constexpr (BACKWARD_COMBINE) {
          acc_prob_ptr = acc_prob_storage.data;
        }
        // Init accumulator.
        #pragma unroll
        for (int n = 0; n < NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD; n++) {
          acc_token_fp32[n].x = 0.0f;
          acc_token_fp32[n].y = 0.0f;
        }
        if constexpr(BACKWARD_COMBINE) {
          #pragma unroll
          for (int n = 0; n < NUM_OF_NODES; n++) {
            for (int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++) {
              acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m] = 0.0f;
            }
          }
        }
        float4* acc_token_tail_smem4 = nullptr;
        bool tail_initialized = false;
        if (tail_bf16x2 > 0) {
          acc_token_tail_smem4 = reinterpret_cast<float4*>(
              smem_buffer_ptr->get_inter_node_token_tail_S2G(dst_token_stage));
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
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_G2S(token_stage));
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE) {
              load_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_G2S(token_stage);
            }

            // Wait until current src token ready in shared memory.
            if (warp_rank_within_pipeline == 0) {
              if (cuda::ptx::elect_sync(~0)) {
                while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage), token_producer_parity)) {}
              }
            }
            // named barrier: we wait for number of threads(all threads in the pipline) that must arrive before any can proceed
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

            // Accumulate token and prob(optional).
            #pragma unroll
            for (int n = 0; n < NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD; n++) {
              int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
              if (element_id < head_bf16x2) {
                __nv_bfloat162 src_data = load_token_base_ptr[element_id];
                float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[n].x += src_data_fp32.x;
              acc_token_fp32[n].y += src_data_fp32.y;
              }
            }
            if (tail_bf16x2 > 0) {
              if (!tail_initialized) {
                // First iteration: load source directly into smem (no accumulation)
                int p = thread_rank_within_pipeline;
                for (; p + 7 * NUM_OF_THREADS_PER_PIPELINE < tail_float4;
                    p += 8 * NUM_OF_THREADS_PER_PIPELINE) {
                  const int p0 = p, p1 = p + NUM_OF_THREADS_PER_PIPELINE;
                  const int p2 = p + 2 * NUM_OF_THREADS_PER_PIPELINE, p3 = p + 3 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p4 = p + 4 * NUM_OF_THREADS_PER_PIPELINE, p5 = p + 5 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p6 = p + 6 * NUM_OF_THREADS_PER_PIPELINE, p7 = p + 7 * NUM_OF_THREADS_PER_PIPELINE;
                  const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                  const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                  const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                  const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                  __nv_bfloat162 s0a = load_token_base_ptr[e0], s0b = load_token_base_ptr[e0 + 1];
                  __nv_bfloat162 s1a = load_token_base_ptr[e1], s1b = load_token_base_ptr[e1 + 1];
                  __nv_bfloat162 s2a = load_token_base_ptr[e2], s2b = load_token_base_ptr[e2 + 1];
                  __nv_bfloat162 s3a = load_token_base_ptr[e3], s3b = load_token_base_ptr[e3 + 1];
                  __nv_bfloat162 s4a = load_token_base_ptr[e4], s4b = load_token_base_ptr[e4 + 1];
                  __nv_bfloat162 s5a = load_token_base_ptr[e5], s5b = load_token_base_ptr[e5 + 1];
                  __nv_bfloat162 s6a = load_token_base_ptr[e6], s6b = load_token_base_ptr[e6 + 1];
                  __nv_bfloat162 s7a = load_token_base_ptr[e7], s7b = load_token_base_ptr[e7 + 1];
                  float2 f0a = __bfloat1622float2(s0a), f0b = __bfloat1622float2(s0b);
                  float2 f1a = __bfloat1622float2(s1a), f1b = __bfloat1622float2(s1b);
                  float2 f2a = __bfloat1622float2(s2a), f2b = __bfloat1622float2(s2b);
                  float2 f3a = __bfloat1622float2(s3a), f3b = __bfloat1622float2(s3b);
                  float2 f4a = __bfloat1622float2(s4a), f4b = __bfloat1622float2(s4b);
                  float2 f5a = __bfloat1622float2(s5a), f5b = __bfloat1622float2(s5b);
                  float2 f6a = __bfloat1622float2(s6a), f6b = __bfloat1622float2(s6b);
                  float2 f7a = __bfloat1622float2(s7a), f7b = __bfloat1622float2(s7b);
                  acc_token_tail_smem4[p0] = make_float4(f0a.x, f0a.y, f0b.x, f0b.y);
                  acc_token_tail_smem4[p1] = make_float4(f1a.x, f1a.y, f1b.x, f1b.y);
                  acc_token_tail_smem4[p2] = make_float4(f2a.x, f2a.y, f2b.x, f2b.y);
                  acc_token_tail_smem4[p3] = make_float4(f3a.x, f3a.y, f3b.x, f3b.y);
                  acc_token_tail_smem4[p4] = make_float4(f4a.x, f4a.y, f4b.x, f4b.y);
                  acc_token_tail_smem4[p5] = make_float4(f5a.x, f5a.y, f5b.x, f5b.y);
                  acc_token_tail_smem4[p6] = make_float4(f6a.x, f6a.y, f6b.x, f6b.y);
                  acc_token_tail_smem4[p7] = make_float4(f7a.x, f7a.y, f7b.x, f7b.y);
                }
                for (; p < tail_float4; p += NUM_OF_THREADS_PER_PIPELINE) {
                  const int e = head_bf16x2 + (p << 1);
                  __nv_bfloat162 sa = load_token_base_ptr[e], sb = load_token_base_ptr[e + 1];
                  float2 fa = __bfloat1622float2(sa), fb = __bfloat1622float2(sb);
                  acc_token_tail_smem4[p] = make_float4(fa.x, fa.y, fb.x, fb.y);
                }
                tail_initialized = true;
              } else {
                // Subsequent iterations: load, add to smem accumulator, store back
                int p = thread_rank_within_pipeline;
                for (; p + 7 * NUM_OF_THREADS_PER_PIPELINE < tail_float4;
                    p += 8 * NUM_OF_THREADS_PER_PIPELINE) {
                  const int p0 = p, p1 = p + NUM_OF_THREADS_PER_PIPELINE;
                  const int p2 = p + 2 * NUM_OF_THREADS_PER_PIPELINE, p3 = p + 3 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p4 = p + 4 * NUM_OF_THREADS_PER_PIPELINE, p5 = p + 5 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p6 = p + 6 * NUM_OF_THREADS_PER_PIPELINE, p7 = p + 7 * NUM_OF_THREADS_PER_PIPELINE;
                  const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                  const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                  const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                  const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                  __nv_bfloat162 s0a = load_token_base_ptr[e0], s0b = load_token_base_ptr[e0 + 1];
                  __nv_bfloat162 s1a = load_token_base_ptr[e1], s1b = load_token_base_ptr[e1 + 1];
                  __nv_bfloat162 s2a = load_token_base_ptr[e2], s2b = load_token_base_ptr[e2 + 1];
                  __nv_bfloat162 s3a = load_token_base_ptr[e3], s3b = load_token_base_ptr[e3 + 1];
                  __nv_bfloat162 s4a = load_token_base_ptr[e4], s4b = load_token_base_ptr[e4 + 1];
                  __nv_bfloat162 s5a = load_token_base_ptr[e5], s5b = load_token_base_ptr[e5 + 1];
                  __nv_bfloat162 s6a = load_token_base_ptr[e6], s6b = load_token_base_ptr[e6 + 1];
                  __nv_bfloat162 s7a = load_token_base_ptr[e7], s7b = load_token_base_ptr[e7 + 1];
                  float2 f0a = __bfloat1622float2(s0a), f0b = __bfloat1622float2(s0b);
                  float2 f1a = __bfloat1622float2(s1a), f1b = __bfloat1622float2(s1b);
                  float2 f2a = __bfloat1622float2(s2a), f2b = __bfloat1622float2(s2b);
                  float2 f3a = __bfloat1622float2(s3a), f3b = __bfloat1622float2(s3b);
                  float2 f4a = __bfloat1622float2(s4a), f4b = __bfloat1622float2(s4b);
                  float2 f5a = __bfloat1622float2(s5a), f5b = __bfloat1622float2(s5b);
                  float2 f6a = __bfloat1622float2(s6a), f6b = __bfloat1622float2(s6b);
                  float2 f7a = __bfloat1622float2(s7a), f7b = __bfloat1622float2(s7b);
                  float4 acc0 = acc_token_tail_smem4[p0];
                  float4 acc1 = acc_token_tail_smem4[p1];
                  float4 acc2 = acc_token_tail_smem4[p2];
                  float4 acc3 = acc_token_tail_smem4[p3];
                  float4 acc4 = acc_token_tail_smem4[p4];
                  float4 acc5 = acc_token_tail_smem4[p5];
                  float4 acc6 = acc_token_tail_smem4[p6];
                  float4 acc7 = acc_token_tail_smem4[p7];
                  acc0.x += f0a.x; acc0.y += f0a.y; acc0.z += f0b.x; acc0.w += f0b.y;
                  acc1.x += f1a.x; acc1.y += f1a.y; acc1.z += f1b.x; acc1.w += f1b.y;
                  acc2.x += f2a.x; acc2.y += f2a.y; acc2.z += f2b.x; acc2.w += f2b.y;
                  acc3.x += f3a.x; acc3.y += f3a.y; acc3.z += f3b.x; acc3.w += f3b.y;
                  acc4.x += f4a.x; acc4.y += f4a.y; acc4.z += f4b.x; acc4.w += f4b.y;
                  acc5.x += f5a.x; acc5.y += f5a.y; acc5.z += f5b.x; acc5.w += f5b.y;
                  acc6.x += f6a.x; acc6.y += f6a.y; acc6.z += f6b.x; acc6.w += f6b.y;
                  acc7.x += f7a.x; acc7.y += f7a.y; acc7.z += f7b.x; acc7.w += f7b.y;
                  acc_token_tail_smem4[p0] = acc0;
                  acc_token_tail_smem4[p1] = acc1;
                  acc_token_tail_smem4[p2] = acc2;
                  acc_token_tail_smem4[p3] = acc3;
                  acc_token_tail_smem4[p4] = acc4;
                  acc_token_tail_smem4[p5] = acc5;
                  acc_token_tail_smem4[p6] = acc6;
                  acc_token_tail_smem4[p7] = acc7;
                }
                for (; p < tail_float4; p += NUM_OF_THREADS_PER_PIPELINE) {
                  const int e = head_bf16x2 + (p << 1);
                  __nv_bfloat162 sa = load_token_base_ptr[e], sb = load_token_base_ptr[e + 1];
                  float2 fa = __bfloat1622float2(sa), fb = __bfloat1622float2(sb);
                  float4 acc = acc_token_tail_smem4[p];
                  acc.x += fa.x; acc.y += fa.y; acc.z += fb.x; acc.w += fb.y;
                  acc_token_tail_smem4[p] = acc;
                }
              }
            }

            if constexpr(BACKWARD_COMBINE) {
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
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage));
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

        if constexpr (NUM_OF_NODES > 1) {
        // Then accumulate from rdma inter-node buffers. There are total NUM_OF_NODES - 1 (possible) src tokens from rdma buffer to reduce.
        const bool* attn_to_rdma_map_load_addr = attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);
        #pragma unroll
        for (int n = 1; n < NUM_OF_NODES; n++) {
          // The current node been processed. For each chunk id, node_id order is
          // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
          int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_OF_NODES - n;
          // The tile id within the rdma buffers(include attn_to_rdma map) for the current node id. Because these rdma buffers only have NUM_OF_NODES - 1 tile or element.
          int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
          // Check wether current dst token need src token from this (remote) node.
          if (attn_to_rdma_map_load_addr[rdma_buffer_tile_id]) {
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_G2S(token_stage));
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE) {
              load_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_G2S(token_stage);
            }
            // Wait until current src token ready in shared memory.
            if (warp_rank_within_pipeline == 0) { // this means that only wrap 0 in the pipeline participates
              if (cuda::ptx::elect_sync(~0)) {
                while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage), token_producer_parity)){}
              }
            }
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank); // named barrier, we wait for number of threads that must arrive before any can proceed

            // Accumulate token and prob(optional).
            #pragma unroll
            for (int m = 0; m < NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD; m++){
              int element_id = (m * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
              if (element_id < head_bf16x2){
                __nv_bfloat162 src_data = load_token_base_ptr[element_id];
                float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[m].x += src_data_fp32.x;
              acc_token_fp32[m].y += src_data_fp32.y;
              }
            }
            if (tail_bf16x2 > 0) {
              if (!tail_initialized) {
                // First iteration: load source directly into smem (no accumulation)
                int p = thread_rank_within_pipeline;
                for (; p + 7 * NUM_OF_THREADS_PER_PIPELINE < tail_float4;
                    p += 8 * NUM_OF_THREADS_PER_PIPELINE) {
                  const int p0 = p, p1 = p + NUM_OF_THREADS_PER_PIPELINE;
                  const int p2 = p + 2 * NUM_OF_THREADS_PER_PIPELINE, p3 = p + 3 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p4 = p + 4 * NUM_OF_THREADS_PER_PIPELINE, p5 = p + 5 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p6 = p + 6 * NUM_OF_THREADS_PER_PIPELINE, p7 = p + 7 * NUM_OF_THREADS_PER_PIPELINE;
                  const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                  const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                  const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                  const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                  __nv_bfloat162 s0a = load_token_base_ptr[e0], s0b = load_token_base_ptr[e0 + 1];
                  __nv_bfloat162 s1a = load_token_base_ptr[e1], s1b = load_token_base_ptr[e1 + 1];
                  __nv_bfloat162 s2a = load_token_base_ptr[e2], s2b = load_token_base_ptr[e2 + 1];
                  __nv_bfloat162 s3a = load_token_base_ptr[e3], s3b = load_token_base_ptr[e3 + 1];
                  __nv_bfloat162 s4a = load_token_base_ptr[e4], s4b = load_token_base_ptr[e4 + 1];
                  __nv_bfloat162 s5a = load_token_base_ptr[e5], s5b = load_token_base_ptr[e5 + 1];
                  __nv_bfloat162 s6a = load_token_base_ptr[e6], s6b = load_token_base_ptr[e6 + 1];
                  __nv_bfloat162 s7a = load_token_base_ptr[e7], s7b = load_token_base_ptr[e7 + 1];
                  float2 f0a = __bfloat1622float2(s0a), f0b = __bfloat1622float2(s0b);
                  float2 f1a = __bfloat1622float2(s1a), f1b = __bfloat1622float2(s1b);
                  float2 f2a = __bfloat1622float2(s2a), f2b = __bfloat1622float2(s2b);
                  float2 f3a = __bfloat1622float2(s3a), f3b = __bfloat1622float2(s3b);
                  float2 f4a = __bfloat1622float2(s4a), f4b = __bfloat1622float2(s4b);
                  float2 f5a = __bfloat1622float2(s5a), f5b = __bfloat1622float2(s5b);
                  float2 f6a = __bfloat1622float2(s6a), f6b = __bfloat1622float2(s6b);
                  float2 f7a = __bfloat1622float2(s7a), f7b = __bfloat1622float2(s7b);
                  acc_token_tail_smem4[p0] = make_float4(f0a.x, f0a.y, f0b.x, f0b.y);
                  acc_token_tail_smem4[p1] = make_float4(f1a.x, f1a.y, f1b.x, f1b.y);
                  acc_token_tail_smem4[p2] = make_float4(f2a.x, f2a.y, f2b.x, f2b.y);
                  acc_token_tail_smem4[p3] = make_float4(f3a.x, f3a.y, f3b.x, f3b.y);
                  acc_token_tail_smem4[p4] = make_float4(f4a.x, f4a.y, f4b.x, f4b.y);
                  acc_token_tail_smem4[p5] = make_float4(f5a.x, f5a.y, f5b.x, f5b.y);
                  acc_token_tail_smem4[p6] = make_float4(f6a.x, f6a.y, f6b.x, f6b.y);
                  acc_token_tail_smem4[p7] = make_float4(f7a.x, f7a.y, f7b.x, f7b.y);
                }
                for (; p < tail_float4; p += NUM_OF_THREADS_PER_PIPELINE){
                  const int e = head_bf16x2 + (p << 1);
                  __nv_bfloat162 sa = load_token_base_ptr[e], sb = load_token_base_ptr[e + 1];
                  float2 fa = __bfloat1622float2(sa), fb = __bfloat1622float2(sb);
                  acc_token_tail_smem4[p] = make_float4(fa.x, fa.y, fb.x, fb.y);
                }
                tail_initialized = true;
              } else {
                // Subsequent iterations: load, add to smem accumulator, store back
                int p = thread_rank_within_pipeline;
                for (; p + 7 * NUM_OF_THREADS_PER_PIPELINE < tail_float4;
                    p += 8 * NUM_OF_THREADS_PER_PIPELINE){
                  const int p0 = p, p1 = p + NUM_OF_THREADS_PER_PIPELINE;
                  const int p2 = p + 2 * NUM_OF_THREADS_PER_PIPELINE, p3 = p + 3 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p4 = p + 4 * NUM_OF_THREADS_PER_PIPELINE, p5 = p + 5 * NUM_OF_THREADS_PER_PIPELINE;
                  const int p6 = p + 6 * NUM_OF_THREADS_PER_PIPELINE, p7 = p + 7 * NUM_OF_THREADS_PER_PIPELINE;
                  const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
                  const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
                  const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
                  const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
                  __nv_bfloat162 s0a = load_token_base_ptr[e0], s0b = load_token_base_ptr[e0 + 1];
                  __nv_bfloat162 s1a = load_token_base_ptr[e1], s1b = load_token_base_ptr[e1 + 1];
                  __nv_bfloat162 s2a = load_token_base_ptr[e2], s2b = load_token_base_ptr[e2 + 1];
                  __nv_bfloat162 s3a = load_token_base_ptr[e3], s3b = load_token_base_ptr[e3 + 1];
                  __nv_bfloat162 s4a = load_token_base_ptr[e4], s4b = load_token_base_ptr[e4 + 1];
                  __nv_bfloat162 s5a = load_token_base_ptr[e5], s5b = load_token_base_ptr[e5 + 1];
                  __nv_bfloat162 s6a = load_token_base_ptr[e6], s6b = load_token_base_ptr[e6 + 1];
                  __nv_bfloat162 s7a = load_token_base_ptr[e7], s7b = load_token_base_ptr[e7 + 1];
                  float2 f0a = __bfloat1622float2(s0a), f0b = __bfloat1622float2(s0b);
                  float2 f1a = __bfloat1622float2(s1a), f1b = __bfloat1622float2(s1b);
                  float2 f2a = __bfloat1622float2(s2a), f2b = __bfloat1622float2(s2b);
                  float2 f3a = __bfloat1622float2(s3a), f3b = __bfloat1622float2(s3b);
                  float2 f4a = __bfloat1622float2(s4a), f4b = __bfloat1622float2(s4b);
                  float2 f5a = __bfloat1622float2(s5a), f5b = __bfloat1622float2(s5b);
                  float2 f6a = __bfloat1622float2(s6a), f6b = __bfloat1622float2(s6b);
                  float2 f7a = __bfloat1622float2(s7a), f7b = __bfloat1622float2(s7b);
                  float4 acc0 = acc_token_tail_smem4[p0];
                  float4 acc1 = acc_token_tail_smem4[p1];
                  float4 acc2 = acc_token_tail_smem4[p2];
                  float4 acc3 = acc_token_tail_smem4[p3];
                  float4 acc4 = acc_token_tail_smem4[p4];
                  float4 acc5 = acc_token_tail_smem4[p5];
                  float4 acc6 = acc_token_tail_smem4[p6];
                  float4 acc7 = acc_token_tail_smem4[p7];
                  acc0.x += f0a.x; acc0.y += f0a.y; acc0.z += f0b.x; acc0.w += f0b.y;
                  acc1.x += f1a.x; acc1.y += f1a.y; acc1.z += f1b.x; acc1.w += f1b.y;
                  acc2.x += f2a.x; acc2.y += f2a.y; acc2.z += f2b.x; acc2.w += f2b.y;
                  acc3.x += f3a.x; acc3.y += f3a.y; acc3.z += f3b.x; acc3.w += f3b.y;
                  acc4.x += f4a.x; acc4.y += f4a.y; acc4.z += f4b.x; acc4.w += f4b.y;
                  acc5.x += f5a.x; acc5.y += f5a.y; acc5.z += f5b.x; acc5.w += f5b.y;
                  acc6.x += f6a.x; acc6.y += f6a.y; acc6.z += f6b.x; acc6.w += f6b.y;
                  acc7.x += f7a.x; acc7.y += f7a.y; acc7.z += f7b.x; acc7.w += f7b.y;
                  acc_token_tail_smem4[p0] = acc0;
                  acc_token_tail_smem4[p1] = acc1;
                  acc_token_tail_smem4[p2] = acc2;
                  acc_token_tail_smem4[p3] = acc3;
                  acc_token_tail_smem4[p4] = acc4;
                  acc_token_tail_smem4[p5] = acc5;
                  acc_token_tail_smem4[p6] = acc6;
                  acc_token_tail_smem4[p7] = acc7;
                }
                for (; p < tail_float4; p += NUM_OF_THREADS_PER_PIPELINE){
                  const int e = head_bf16x2 + (p << 1);
                  __nv_bfloat162 sa = load_token_base_ptr[e], sb = load_token_base_ptr[e + 1];
                  float2 fa = __bfloat1622float2(sa), fb = __bfloat1622float2(sb);
                  float4 acc = acc_token_tail_smem4[p];
                  acc.x += fa.x; acc.y += fa.y; acc.z += fb.x; acc.w += fb.y;
                  acc_token_tail_smem4[p] = acc;
                }
              }
            }

            if constexpr(BACKWARD_COMBINE) {
              #pragma unroll
              for (int m = 0; m < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; m++) {
                int element_id = thread_rank_within_pipeline + m * NUM_OF_THREADS_PER_PIPELINE;
                if (element_id < experts_per_rank * num_of_ranks_per_node) {
                  acc_prob_ptr[n * NUM_OF_PROB_VEC_ELEMENT_PER_THREAD + m] = load_prob_base_ptr[element_id];
                }
              }
            }

            // Make sure all threads within the pipeline have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);
            if (warp_rank_within_pipeline == 0) {
              if (cuda::ptx::elect_sync(~0)) {
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage));
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
        __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_inter_node_token_S2G(dst_token_stage));
        float* store_prob_base_ptr;
        if constexpr(BACKWARD_COMBINE) {
          store_prob_base_ptr = smem_buffer_ptr->get_inter_node_prob_S2G(dst_token_stage);
        }

        // Select the TMA thread within the pipeline to wait for previously issued TMA S2G operations finish reading this entry.
        if (warp_rank_within_pipeline == 0) {
          if (cuda::ptx::elect_sync(~0)) {
            cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G_PER_PIPELINE - 1>{});
          }
        }
        // Make sure all threads within the pipeline have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
        arrive_and_wait(NUM_OF_THREADS_PER_PIPELINE, 2 + pipeline_rank);

        // Store the token.
        #pragma unroll
        for (int n = 0; n < NUM_OF_HEAD_ACC_ELEMENTS_PER_THREAD; n++) {
          int element_id = (n * NUM_OF_THREADS_PER_PIPELINE) + thread_rank_within_pipeline;
          if (element_id < head_bf16x2) {
            // Convert accumulated token back to BF16 and store the result back to shared memory token entry.
            store_token_base_ptr[element_id] = __float22bfloat162_rn(acc_token_fp32[n]);
          }
        }
        if (tail_bf16x2 > 0) {
          const __nv_bfloat162 zero_bf16x2 = __float22bfloat162_rn(make_float2(0.0f, 0.0f));
          if (!tail_initialized) {
            // No sources contributed to this token's tail, write zeros
            int p = thread_rank_within_pipeline;
            for (; p + 7 * NUM_OF_THREADS_PER_PIPELINE < tail_float4;
                p += 8 * NUM_OF_THREADS_PER_PIPELINE) {
              const int p0 = p, p1 = p + NUM_OF_THREADS_PER_PIPELINE;
              const int p2 = p + 2 * NUM_OF_THREADS_PER_PIPELINE, p3 = p + 3 * NUM_OF_THREADS_PER_PIPELINE;
              const int p4 = p + 4 * NUM_OF_THREADS_PER_PIPELINE, p5 = p + 5 * NUM_OF_THREADS_PER_PIPELINE;
              const int p6 = p + 6 * NUM_OF_THREADS_PER_PIPELINE, p7 = p + 7 * NUM_OF_THREADS_PER_PIPELINE;
              const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
              const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
              const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
              const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
              store_token_base_ptr[e0] = zero_bf16x2; store_token_base_ptr[e0 + 1] = zero_bf16x2;
              store_token_base_ptr[e1] = zero_bf16x2; store_token_base_ptr[e1 + 1] = zero_bf16x2;
              store_token_base_ptr[e2] = zero_bf16x2; store_token_base_ptr[e2 + 1] = zero_bf16x2;
              store_token_base_ptr[e3] = zero_bf16x2; store_token_base_ptr[e3 + 1] = zero_bf16x2;
              store_token_base_ptr[e4] = zero_bf16x2; store_token_base_ptr[e4 + 1] = zero_bf16x2;
              store_token_base_ptr[e5] = zero_bf16x2; store_token_base_ptr[e5 + 1] = zero_bf16x2;
              store_token_base_ptr[e6] = zero_bf16x2; store_token_base_ptr[e6 + 1] = zero_bf16x2;
              store_token_base_ptr[e7] = zero_bf16x2; store_token_base_ptr[e7 + 1] = zero_bf16x2;
            }
            for (; p < tail_float4; p += NUM_OF_THREADS_PER_PIPELINE) {
              const int e = head_bf16x2 + (p << 1);
              store_token_base_ptr[e] = zero_bf16x2;
              store_token_base_ptr[e + 1] = zero_bf16x2;
            }
          } else {
            // Store accumulated tail from smem to global
            int p = thread_rank_within_pipeline;
            for (; p + 7 * NUM_OF_THREADS_PER_PIPELINE < tail_float4;
                p += 8 * NUM_OF_THREADS_PER_PIPELINE) {
              const int p0 = p, p1 = p + NUM_OF_THREADS_PER_PIPELINE;
              const int p2 = p + 2 * NUM_OF_THREADS_PER_PIPELINE, p3 = p + 3 * NUM_OF_THREADS_PER_PIPELINE;
              const int p4 = p + 4 * NUM_OF_THREADS_PER_PIPELINE, p5 = p + 5 * NUM_OF_THREADS_PER_PIPELINE;
              const int p6 = p + 6 * NUM_OF_THREADS_PER_PIPELINE, p7 = p + 7 * NUM_OF_THREADS_PER_PIPELINE;
              const int e0 = head_bf16x2 + (p0 << 1), e1 = head_bf16x2 + (p1 << 1);
              const int e2 = head_bf16x2 + (p2 << 1), e3 = head_bf16x2 + (p3 << 1);
              const int e4 = head_bf16x2 + (p4 << 1), e5 = head_bf16x2 + (p5 << 1);
              const int e6 = head_bf16x2 + (p6 << 1), e7 = head_bf16x2 + (p7 << 1);
              float4 acc0 = acc_token_tail_smem4[p0];
              float4 acc1 = acc_token_tail_smem4[p1];
              float4 acc2 = acc_token_tail_smem4[p2];
              float4 acc3 = acc_token_tail_smem4[p3];
              float4 acc4 = acc_token_tail_smem4[p4];
              float4 acc5 = acc_token_tail_smem4[p5];
              float4 acc6 = acc_token_tail_smem4[p6];
              float4 acc7 = acc_token_tail_smem4[p7];
              store_token_base_ptr[e0] = __float22bfloat162_rn(make_float2(acc0.x, acc0.y));
              store_token_base_ptr[e0 + 1] = __float22bfloat162_rn(make_float2(acc0.z, acc0.w));
              store_token_base_ptr[e1] = __float22bfloat162_rn(make_float2(acc1.x, acc1.y));
              store_token_base_ptr[e1 + 1] = __float22bfloat162_rn(make_float2(acc1.z, acc1.w));
              store_token_base_ptr[e2] = __float22bfloat162_rn(make_float2(acc2.x, acc2.y));
              store_token_base_ptr[e2 + 1] = __float22bfloat162_rn(make_float2(acc2.z, acc2.w));
              store_token_base_ptr[e3] = __float22bfloat162_rn(make_float2(acc3.x, acc3.y));
              store_token_base_ptr[e3 + 1] = __float22bfloat162_rn(make_float2(acc3.z, acc3.w));
              store_token_base_ptr[e4] = __float22bfloat162_rn(make_float2(acc4.x, acc4.y));
              store_token_base_ptr[e4 + 1] = __float22bfloat162_rn(make_float2(acc4.z, acc4.w));
              store_token_base_ptr[e5] = __float22bfloat162_rn(make_float2(acc5.x, acc5.y));
              store_token_base_ptr[e5 + 1] = __float22bfloat162_rn(make_float2(acc5.z, acc5.w));
              store_token_base_ptr[e6] = __float22bfloat162_rn(make_float2(acc6.x, acc6.y));
              store_token_base_ptr[e6 + 1] = __float22bfloat162_rn(make_float2(acc6.z, acc6.w));
              store_token_base_ptr[e7] = __float22bfloat162_rn(make_float2(acc7.x, acc7.y));
              store_token_base_ptr[e7 + 1] = __float22bfloat162_rn(make_float2(acc7.z, acc7.w));
            }
            for(; p < tail_float4; p += NUM_OF_THREADS_PER_PIPELINE){
              const int e = head_bf16x2 + (p << 1);
              float4 acc = acc_token_tail_smem4[p];
              store_token_base_ptr[e] = __float22bfloat162_rn(make_float2(acc.x, acc.y));
              store_token_base_ptr[e + 1] = __float22bfloat162_rn(make_float2(acc.z, acc.w));
            }
          }
        }

        // Store the prob(optional).
        if constexpr(BACKWARD_COMBINE) {
          #pragma unroll
          for (int n = 0; n < NUM_OF_NODES; n++) {
            int attn_prob_output_node_id = (node_rank - n) >= 0 ? node_rank - n : node_rank + NUM_OF_NODES - n;
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
          if (cuda::ptx::elect_sync(~0)) {
            uint16_t* current_token_addr = attn_output_token_base_ptr + (j * NUM_OF_TOKENS_PER_GROUP + k) * HIDDEN_DIM;
            // Store the token from shared to global output.
            cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                     cuda::ptx::space_shared,
                                     reinterpret_cast<void*>(current_token_addr),
                                     reinterpret_cast<const void*>(smem_buffer_ptr->get_inter_node_token_S2G(dst_token_stage)),
                                     (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)));

            // Store the prob from shared to global output.
            if constexpr(BACKWARD_COMBINE) {
              float* current_prob_addr = attn_output_prob_base_ptr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES);
              cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                       cuda::ptx::space_shared,
                                       reinterpret_cast<void*>(current_prob_addr),
                                       reinterpret_cast<const void*>(smem_buffer_ptr->get_inter_node_prob_S2G(dst_token_stage)),
                                       (uint32_t)((experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES) * sizeof(float)));

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

// This kernel will update expected_rdma_flag_value and expected_intra_node_flag_value in local device memory
// by increasing the expected_rdma_flag_value by 1 and expected_intra_node_flag_value by num_of_ranks_per_node.
template<int NUM_OF_NODES>
__launch_bounds__(1, 1)
__global__ void update_expected_value_kernel(uint64_t* expected_rdma_flag_value, uint32_t* expected_intra_node_flag_value, const int num_of_ranks_per_node)
{
  if constexpr(NUM_OF_NODES != 1) {
    (*expected_rdma_flag_value) += 1;
  }
  (*expected_intra_node_flag_value) += num_of_ranks_per_node;
}

template<typename TOKEN_DATA_TYPE,
         typename INTER_NODE_GROUP,
         typename INTRA_NODE_G2S_GROUP,
         typename INTRA_NODE_S2G_GROUP,
         int NUM_OF_STAGES,
         int NUM_OF_IN_FLIGHT_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH,
         int NUM_PIPELINES>
__launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size(), 1)
__global__ void dispatch_kernel(const __grid_constant__ dispatch_kernel_param_t<TOKEN_DATA_TYPE> param)
{
  if constexpr (NUM_OF_NODES != 1) {
    static_assert(INTER_NODE_GROUP::size() % 32 == 0 && INTER_NODE_GROUP::size() <= 64,
                  "Dispatch kernel supports 1 or 2 N2N warps.");
  }
  static_assert(NUM_OF_STAGES % NUM_PIPELINES == 0, "NUM_OF_STAGES must be divisible by NUM_PIPELINES.");
  constexpr int STAGES_PER_PIPELINE = NUM_OF_STAGES / NUM_PIPELINES;

  extern __shared__ uint8_t smem_bytes[];
  using cur_smem_t = dispatch_smem_layout_t;

  cur_smem_t smem_layout;
  dispatch_config_t d_config;
  model_config_t d_model;
  d_config.num_of_stages = NUM_OF_STAGES;
  d_config.num_of_in_flight_s2g = NUM_OF_IN_FLIGHT_S2G;
  d_config.num_of_tokens_per_chunk = NUM_OF_TOKENS_PER_CHUNK;
  d_config.num_of_blocks = NUM_OF_BLOCKS;
  d_config.forward_dispatch = FORWARD_DISPATCH;
  d_config.token_data_type = std::is_same_v<TOKEN_DATA_TYPE, uint16_t> ? 1 : 0;
  d_config.num_pipelines = NUM_PIPELINES;
  d_config.stages_per_pipeline = STAGES_PER_PIPELINE;
  d_model.hidden_dim = param.hidden_dim;
  d_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
  d_model.num_of_experts_per_rank = param.experts_per_rank;
  d_model.num_of_ranks_per_node = param.num_of_ranks_per_node;
  d_model.num_of_nodes = NUM_OF_NODES;
  create_dispatch_smem_layout(smem_layout, smem_bytes, d_config, d_model);
  cur_smem_t* smem_buffer_ptr = &smem_layout;

  if (threadIdx.x == 0) {
    // Per-pipeline mbarrier initialization.
    // CRITICAL: both producer and consumer arrival counts = 1 per pipeline.
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

  __syncthreads();

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
  long long _wt_start = 0;
  if (threadIdx.x % 32 == 0) _wt_start = clock64();
#endif
  int threadIdx_x_int = (int)threadIdx.x;
  if(threadIdx_x_int < INTER_NODE_GROUP::size()){
    if constexpr(NUM_OF_NODES != 1){
      N2N_warp_group_device_function
      <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.attn_to_rdma_map,
       param.dcomms, param.nccl_window, param.num_gin_comms, param.num_ctx_per_comm, param.gin_base_ptr, param.signals_base,
       &param.mr_info, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
    }
  } else if (threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    G2S_warp_group_device_function
    <INTRA_NODE_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK,
     MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH, NUM_PIPELINES>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.expected_rdma_flag_value, param.hidden_dim, param.rdma_to_attn_map, param.attn_input_token,
    param.attn_input_prob, param.attn_input_token_scaling_factor,
    param.rdma_inter_node_group_flags, param.dcomms, param.signals_base, param.num_gin_comms, param.num_ctx_per_comm,
    param.gin_base_ptr, &param.mr_info, smem_buffer_ptr, param.experts_per_rank);
  } else if (threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()){
    S2G_warp_group_device_function
    <INTRA_NODE_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_IN_FLIGHT_S2G, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH, NUM_PIPELINES>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.hidden_dim, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_output_token, param.expert_output_prob,
    param.expert_output_scaling_factor, smem_buffer_ptr, param.experts_per_rank);
  }
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
  if (threadIdx.x % 32 == 0) {
      constexpr int _WT_WARPS = (INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()) / 32;
      int _warp_id = threadIdx.x / 32;
      int _idx = blockIdx.x * _WT_WARPS + _warp_id;
      param.warp_timing[_idx].start_clock = _wt_start;
      param.warp_timing[_idx].end_clock = clock64();
  }
#endif

  // ===== FUSED DEVICE SYNC (dispatch tail) =====
  // All threads in this block must finish G2S/S2G/N2N work before we signal completion.
  __syncthreads();

  if (threadIdx.x == 0) {
      // Grid barrier: all blocks in this kernel must arrive before inter-rank signaling.
      // TMA S2G writes are already committed by cp_async_bulk_wait_group in S2G warp group.
      unsigned int arrived = atomicAdd(param.dispatch_grid_barrier_counter, 1);

      // Last block to arrive: all blocks' TMA writes are now system-visible.
      // Signal the inter-rank completion flag.
      if (arrived == NUM_OF_BLOCKS - 1) {
          asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                       :
                       : "l"(__cvta_generic_to_global(
                             param.intra_node_write_completion_flags)), "n"(1)
                       : "memory");
      }

      // All blocks poll the inter-rank flag until all ranks have signaled.
      uint32_t flag_data;
      do {
          asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                       : "=r"(flag_data)
                       : "l"(__cvta_generic_to_global(
                             param.intra_node_write_completion_flags))
                       : "memory");
      } while (flag_data != param.expected_intra_node_flag_value);

      // Last block resets grid counter for next invocation.
      if (arrived == NUM_OF_BLOCKS - 1) {
          atomicExch((unsigned int*)param.dispatch_grid_barrier_counter, 0u);
      }
  }
}

template<// This type represent intra-node reduction warp group.
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
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         // Number of CUDA block running dispatch kernel.
         int NUM_OF_BLOCKS,
         // Number of fully in-flight S2G in intra-node reduction warp group.
         int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
         // Whether the combine kernel is used in backward process. If so, need to transfer the prob for each token as well.
         bool BACKWARD_COMBINE>
// Each CUDA block of combine kernel has 5 warp groups and has the following layout:
// 1. intra-node reduction warp group(4 warps, only valid for multinode scenario). 2. inter-node reduction warp group(4 warps, 1 pipeline for multinode scenario, 2 pipeline otherwise).
// 3. intra-node G2S warp group(1 warp, only valid for multinode scenario). 4. inter-node G2S warp group(1 warp for multinode scenario, 2 warps otherwise). 5. inter-node N2N rdma warp group(1 warp, only valid for multinode scenario).
// Total 6(single-node) or 11(multi-node) warps per CUDA block/SM.
__launch_bounds__(INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size(), 1)
__global__ void combine_kernel(const __grid_constant__ combine_kernel_param_t param)
{
  // Compile-time check (only enforce for multi-node layout).
  if constexpr (NUM_OF_NODES != 1) {
  static_assert(INTRA_NODE_G2S_GROUP::size() == 32, "Combine kernel only support 1 INTRA_NODE_G2S warp currently.");
  static_assert(INTER_NODE_G2S_GROUP::size() == 32, "Combine kernel only support 1 INTER_NODE_G2S warp currently.");
  }
  // The token and its properties should meet size and alignment requirement.
  // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
  // We need to add padding or not using TMA for prob, if we want to support other scenario.
  //assert((param.experts_per_rank * param.num_of_ranks_per_node * sizeof(float)) % 16 == 0);
  assert((param.hidden_dim * sizeof(uint16_t)) % 16 == 0);
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;

  // Shared memory used over 48KB, should use dynamic shared memory.
  extern __shared__ uint8_t smem_bytes[];
    using cur_smem_t  = combine_smem_layout_t;

    // Initialize the layout struct (each thread has its own copy in registers)
    cur_smem_t smem_layout;
    model_config_t c_model;
    c_model.hidden_dim = param.hidden_dim;
    c_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    c_model.num_of_experts_per_rank = param.experts_per_rank;
    c_model.num_of_ranks_per_node = param.num_of_ranks_per_node;
    c_model.num_of_nodes = NUM_OF_NODES;
    create_combine_smem_layout(smem_layout,
                               smem_bytes,
                               NUM_OF_STAGES_G2S,
                               NUM_OF_STAGES_S2G,
                               NUM_OF_TOKENS_PER_CHUNK,
                               BACKWARD_COMBINE,
                               c_model);
    cur_smem_t* smem_buffer_ptr = &smem_layout;

  // ===== FUSED DEVICE SYNC (combine head) =====
  // Wait for all ranks' dispatch S2G writes to complete before reading them.
  // Block 0 signals this rank's arrival at the inter-rank barrier.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
      asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                   :
                   : "l"(__cvta_generic_to_global(
                         param.intra_node_write_completion_flags)), "n"(1)
                   : "memory");
  }
  // All blocks' thread 0 poll until all ranks have signaled.
  if (threadIdx.x == 0) {
      uint32_t flag_data;
      do {
          asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                       : "=r"(flag_data)
                       : "l"(__cvta_generic_to_global(
                             param.intra_node_write_completion_flags))
                       : "memory");
      } while (flag_data != param.expected_intra_node_flag_value);
  }
  // The __syncthreads() below (for mbarrier init) also ensures all threads
  // see the poll completion before proceeding with combine work.

  // Let first thread of each CUDA block initialize the mbarrier.
  if (threadIdx.x == 0) {
    for (int i = 0; i < NUM_OF_STAGES_G2S; i++) {
      // Initialize mbarrier
      if constexpr(NUM_OF_NODES != 1) {
        cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_G2S_buffer + 2 * i, 1);
        cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_G2S_buffer + 2 * i + 1, 1);
      }
      cuda::ptx::mbarrier_init(smem_buffer_ptr->inter_node_mbarrier_G2S_buffer + 2 * i, 1);
      cuda::ptx::mbarrier_init(smem_buffer_ptr->inter_node_mbarrier_G2S_buffer + 2 * i + 1, 1);
    }
    if constexpr(NUM_OF_NODES != 1) {
      // Initialize mbarrier
      for (int i = 0; i < NUM_OF_NODES - 1; i++) {
        for (int j = 0; j < MAX_NUM_OF_CHUNKS_PER_RANK; j++) {
          cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer + i * MAX_NUM_OF_CHUNKS_PER_RANK + j, 1);
        }
      }
    }
    // Initialize streaming overlap fields (cumulative counter, never reset between chunks)
    if constexpr(NUM_OF_NODES != 1) {
      *(smem_buffer_ptr->rdma_streaming_counter) = 0u;
    }
    // Make mbarriers initialization visible to async proxy(TMA).
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }

  // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
  __syncthreads();

  // Now warps can become specialized.
  // The input warp group data type must match the warp groups layout.
  // To prevent compiler generate pointless comparison warning.
  int threadIdx_x_int = (int)threadIdx.x;
  if (threadIdx_x_int < INTRA_NODE_RED_GROUP::size()) {
    if constexpr(NUM_OF_NODES != 1) {
    // Intra-node reduction warp group.
      intra_node_red_warp_group_device_function
      <INTRA_NODE_RED_GROUP, cur_smem_t, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK,
      NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.rdma_to_attn_map, param.rdma_intra_node_red_token, param.rdma_intra_node_red_prob, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
    }
  }else if (threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size()) {
    // Inter-node reduction warp group.
    inter_node_red_warp_group_device_function
    <cur_smem_t, INTER_NODE_RED_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, NUM_OF_STAGES_G2S, NUM_OF_STAGES_S2G, NUM_OF_TOKENS_PER_CHUNK,
    NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
    (param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.rdma_to_attn_map, param.attn_to_rdma_map, param.attn_output_token, param.attn_output_prob, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    // Intra-node G2S warp group.
    if constexpr(NUM_OF_NODES != 1) {
      intra_node_G2S_warp_group_device_function
      <cur_smem_t, NUM_OF_STAGES_G2S, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_input_token, param.expert_input_prob, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
    }
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size()){
    // Inter-node G2S warp group.
    inter_node_G2S_warp_group_device_function
    <cur_smem_t, INTER_NODE_G2S_GROUP, NUM_OF_STAGES_G2S, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES, NUM_OF_BLOCKS,
    NUM_OF_TOKENS_PER_GROUP, BACKWARD_COMBINE>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.expected_rdma_flag_value, param.rdma_to_attn_map, param.attn_to_rdma_map, param.sparse_to_dense_map, param.expert_input_token, param.expert_input_prob,
    param.rdma_inter_node_group_token, param.rdma_inter_node_group_prob, param.dcomms, param.signals_base, param.combine_signal_offset, param.num_gin_comms, param.num_ctx_per_comm, param.rdma_inter_node_group_flags, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
  }else if(threadIdx_x_int < INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size()){
    // Inter-node rdma warp group.
    if constexpr(NUM_OF_NODES != 1){
      inter_node_N2N_warp_group_device_function
      <INTER_NODE_RDMA_GROUP, cur_smem_t, NUM_OF_STAGES_S2G, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES, NUM_OF_BLOCKS, BACKWARD_COMBINE>
      (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.rdma_to_attn_map,
       param.dcomms, param.nccl_window, param.num_gin_comms, param.num_ctx_per_comm, param.gin_base_ptr, param.signals_base, param.combine_signal_offset,
       &param.mr_info, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
    }
  }else{
    // Too many threads, should not goes here.
  }
}

static __device__ __forceinline__ bool bitmap_range_has_set_bit(
    const uint8_t* bitmap_row, int bit_begin, int bit_count) {
  if (bit_count <= 0) return false;

  const int bit_end = bit_begin + bit_count;  // exclusive
  const int first_byte = bit_begin >> 3;
  const int last_byte = (bit_end - 1) >> 3;
  const int first_bit = bit_begin & 7;
  const int last_bit = (bit_end - 1) & 7;

  if (first_byte == last_byte) {
    const uint8_t left_mask = static_cast<uint8_t>(0xFFu << first_bit);
    const uint8_t right_mask = static_cast<uint8_t>((last_bit == 7) ? 0xFFu : ((1u << (last_bit + 1)) - 1u));
    return (bitmap_row[first_byte] & static_cast<uint8_t>(left_mask & right_mask)) != 0;
  }

  const uint8_t first_mask = static_cast<uint8_t>(0xFFu << first_bit);
  if ((bitmap_row[first_byte] & first_mask) != 0) {
    return true;
  }

  uint8_t middle_or = 0;
  for (int b = first_byte + 1; b < last_byte; b++) {
    middle_or |= bitmap_row[b];
  }
  if (middle_or != 0) {
    return true;
  }

  const uint8_t last_mask = static_cast<uint8_t>((last_bit == 7) ? 0xFFu : ((1u << (last_bit + 1)) - 1u));
  return (bitmap_row[last_byte] & last_mask) != 0;
}

static __device__ __forceinline__ uint8_t bitmap_row_to_rank_mask(
    const uint8_t* bitmap_row, int num_of_ranks_per_node, int experts_per_rank) {
  uint8_t rank_mask = 0;
  #pragma unroll
  for (int rank = 0; rank < MAX_RANKS_PER_NODE; rank++) {
    if (rank < num_of_ranks_per_node) {
      const int bit_begin = rank * experts_per_rank;
      if (bitmap_range_has_set_bit(bitmap_row, bit_begin, experts_per_rank)) {
        rank_mask |= static_cast<uint8_t>(1u << rank);
      }
    }
  }
  return rank_mask;
}

template<int NUM_THREADS_PER_BLOCK,
         int NUM_OF_BLOCKS,
         int NUM_OF_NODES,
         int LSA_TEAM_SIZE,
         bool ENABLE_PER_EXPERT_COUNTS>
__launch_bounds__(NUM_THREADS_PER_BLOCK, 1)
__global__ void scan(const uint8_t* input_routing_map,
                     tmp_state_t* tmp,
                     int32_t* sparse_to_dense_map,
                     bool* rdma_to_attn_map,
                     bool* attn_to_rdma_map,
                     uint8_t* token_rank_mask,
                     int32_t* num_of_tokens_for_experts,
                     bool* local_expert_routing_map,
                     int32_t* per_expert_token_counts,
                     const int node_rank,
                     const int local_rank,
                     const int num_of_tokens_per_rank,
                     const int num_of_ranks_per_node,
                     const int experts_per_rank)
{
  (void)per_expert_token_counts;
  // Calculate the warps per block.
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;

  // Calculate total threads count.
  constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;

  // Maximum ranks per node for compile-time sizing of per-thread arrays
  constexpr int MAX_RANKS_PER_NODE_SCAN = LSA_TEAM_SIZE;

  // Calculate the number of tokens belong to each CUDA block, warp and thread.
  // We assign 1 token(row in routing map) to 1 thread.
  const int num_of_total_attn_tokens = num_of_tokens_per_rank * num_of_ranks_per_node * NUM_OF_NODES;
  const int num_of_tokens_per_thread = ((num_of_total_attn_tokens - 1) / NUM_OF_TOTAL_THREADS) + 1;
  const int num_of_tokens_per_warp = num_of_tokens_per_thread * WARP_SIZE;
  const int num_of_tokens_per_block = num_of_tokens_per_warp * NUM_OF_WARPS_PER_BLOCK;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

  // Bitmap routing map constants: each expert is 1 bit, packed 8 per byte.
  // Any experts_per_rank is supported; last byte may have unused high bits (always 0).
  const int experts_per_node = experts_per_rank * num_of_ranks_per_node;
  const int experts_per_node_packed = (experts_per_node + 7) / 8;  // bytes per node section (ceil)
  const int packed_row_bytes = experts_per_node_packed * NUM_OF_NODES; // full bitmap row width

  // For each token, calculate how many bytes need to be store to sparse_to_dense_map.
  // Use maximum value for compile-time type selection
  constexpr int MAX_NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * MAX_RANKS_PER_NODE_SCAN;
  using write_t = Copy_t<MAX_NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN>;
  const int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * num_of_ranks_per_node;
  const int S2D_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN / sizeof(write_t);

  // How to convert per-rank routing info to per-node routing info.
  // Current HT constraint is max 8 ranks per node, so one lane can represent one rank.
  static_assert(MAX_RANKS_PER_NODE_SCAN <= WARP_SIZE,
                "scan assumes max ranks per node fits in one warp.");

  // Use dynamic shared memory for runtime-sized arrays
  extern __shared__ uint8_t smem_bytes[];

  // Sum of per-rank routing info of all warps within the block.
  int32_t* warp_token_routing_map_sum = reinterpret_cast<int32_t*>(smem_bytes);
  // Sum of previous blocks' per-rank routing info.
  int32_t* previous_block_sum = reinterpret_cast<int32_t*>(smem_bytes + NUM_OF_WARPS_PER_BLOCK * num_of_ranks_per_node * sizeof(int32_t));
  // Optional per-block local-expert counts scratch.
  [[maybe_unused]] int32_t* block_expert_token_counts = nullptr;
  if constexpr (ENABLE_PER_EXPERT_COUNTS) {
    block_expert_token_counts = reinterpret_cast<int32_t*>(
        smem_bytes + (NUM_OF_WARPS_PER_BLOCK * num_of_ranks_per_node + num_of_ranks_per_node) * sizeof(int32_t));
    for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
      block_expert_token_counts[e] = 0;
    }
    __syncthreads();
  }

  // We assign contiguous tokens called chunk to each CUDA block, each CUDA block get the same size of chunk.
  int block_starting_token = blockIdx.x * num_of_tokens_per_block;
  // warp id and lane id.
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  // We assign contiguous tokens called sub-chunk to each warp within a CUDA block, each warp within a CUDA block get the same size of sub-chunk.
  int warp_starting_token = block_starting_token + warp_id * num_of_tokens_per_warp;
  // Within a sub-chunk, we assign tokens to thread in a interleave pattern. So each thread process a token each time and each warp sum a tile of 32 tokens each time.
  int thread_starting_token = warp_starting_token + lane_id;

  // Step 0: Each warp sum the sub-chunk assigned to them and store the sum back to shared memory.
  // All warps within all CTA attend this step.
  // Also, some tokens need per-node info which store to rdma_to_attn_map, also processed here.

  // Sum of per-rank token routing map within a thread.
  // Use MAX size for compile-time array allocation, actual size determined by runtime num_of_ranks_per_node
  int32_t token_routing_map_sum[MAX_RANKS_PER_NODE_SCAN];
  #pragma unroll
  for(int i = 0; i < num_of_ranks_per_node; i++){
    token_routing_map_sum[i] = 0;
  }

  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then just end summing tokens assigned to this thread.
    if(current_token_id >= num_of_total_attn_tokens){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * num_of_ranks_per_node);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * num_of_ranks_per_node)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;
    // If the token belongs to the inter-node group.
    // We need to calculate the per-node routing info and save back to rdma_to_attn_map.
    bool per_node_routing_info = (current_token_local_rank == local_rank);
    int current_token_rdma_to_attn_map_id = current_token_node_rank * rdma_to_attn_map_size_per_node + current_token_local_id;
    // Load bitmap for this token's local-node section.
    const uint8_t* bitmap_row = input_routing_map +
                                current_token_id * packed_row_bytes +
                                node_rank * experts_per_node_packed;
    // Decode bitmap row once into compact per-token rank mask.
    uint8_t rank_mask = bitmap_row_to_rank_mask(bitmap_row, num_of_ranks_per_node, experts_per_rank);

    // Accumulate per-rank sums from rank_mask bits.
    #pragma unroll
    for(int j = 0; j < MAX_RANKS_PER_NODE_SCAN; j++){
      if (j < num_of_ranks_per_node) {
        token_routing_map_sum[j] += ((rank_mask >> j) & 1u);
      }
    }
    token_rank_mask[current_token_id] = rank_mask;

    // Save the per node routing info back to rdma_to_attn_map if needed.
    if(per_node_routing_info){
      rdma_to_attn_map[current_token_rdma_to_attn_map_id] = (rank_mask != 0);
    }
  }

  // Each warp reduces per-rank routing counts across lanes.
  // Lane `rank` stores the reduced sum for that rank.
  #pragma unroll
  for (int rank = 0; rank < num_of_ranks_per_node; rank++) {
    int32_t rank_sum = __reduce_add_sync(~0u, token_routing_map_sum[rank]);
    if (lane_id == rank) {
      warp_token_routing_map_sum[warp_id * num_of_ranks_per_node + rank] = rank_sum;
    }
  }

  // Sync within a CUDA block to make sure all warps have produced the per-rank sum data to the shared memory before any thread can consume them to produce CUDA block level's sum data.
  __syncthreads();

  // Step 1: Communication between CUDA blocks. Each CUDA block's threads need to produce and store the current block's per-rank sum data to global memory,
  // and load and accumulate previous blocks' per-rank sum data and save the result to shared memory.

  // Each thread within a CUDA block calculate the CUDA block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < num_of_ranks_per_node; i += NUM_THREADS_PER_BLOCK){
    int32_t rank_acc = 0;
    // Calculate the sum of current rank within this CUDA block.
    #pragma unroll
    for(int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++){
      rank_acc += warp_token_routing_map_sum[j * num_of_ranks_per_node + i];
    }

    // Store the sum of current rank within this CUDA block to global memory for later scan opeartions.
    // Strong(atomic) store is needed to be visible to strong(atomic) load from other blocks.
    tmp_state_t* tmp_dst = &tmp[blockIdx.x * num_of_ranks_per_node + i];
    tmp_state_t tmp_data{PRIV_SUM, rank_acc};
    uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
    asm volatile("st.relaxed.gpu.global.b64 [%0], %1;"
                  :
                  : "l"(__cvta_generic_to_global(tmp_dst)), "l"(data)
                  : "memory");
  }

  // Each thread within a CUDA block load previous blocks' block level sum for a single rank at a time.
  for(int i = threadIdx.x; i < num_of_ranks_per_node; i += NUM_THREADS_PER_BLOCK){
    int32_t previous_block_sum_for_current_rank = 0;
    for(int j = 0; j < blockIdx.x; j++){
      tmp_state_t tmp_data{EMPTY, 0};
      tmp_state_t* tmp_src = &tmp[j * num_of_ranks_per_node + i];
      do{
          // Load previous blocks' per-rank sum from global memory.
          // Strong(atomic) load is needed to view strong(atomic) store from other blocks.
          uint64_t data = 0;
          asm volatile("ld.relaxed.gpu.global.b64 %0, [%1];"
                        : "=l"(data)
                        : "l"(__cvta_generic_to_global(tmp_src))
                        : "memory");
          tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
      }while(tmp_data.state != PRIV_SUM);
      previous_block_sum_for_current_rank += tmp_data.value;
    }
    previous_block_sum[i] = previous_block_sum_for_current_rank;
  }

  // Sync within a CUDA block to make sure all previous blocks' per-rank sum have been produced to the shared memory before any thread can consume them in scan operation.
  __syncthreads();

  // Step 2: Each warp scan the sub-chunk assigned to them(the same sub-chunk as step 0) and produce sparse_to_dense_map, local_expert_routing_map and num_of_tokens_for_experts.
  // Use MAX size for compile-time array allocation
  int32_t previous_token_sum[MAX_RANKS_PER_NODE_SCAN];

  // Lane `lane_id` accumulates prefix for that same rank.
  int32_t lane_rank_prefix = 0;
  if (lane_id < num_of_ranks_per_node) {
    lane_rank_prefix = previous_block_sum[lane_id];
    for (int j = 0; j < warp_id; j++) {
      lane_rank_prefix += warp_token_routing_map_sum[j * num_of_ranks_per_node + lane_id];
    }
  }

  // Broadcast accumulated per-rank prefix sums from source lanes [0, num_of_ranks_per_node).
  #pragma unroll
  for (int rank = 0; rank < num_of_ranks_per_node; rank++) {
    previous_token_sum[rank] = __shfl_sync(~0u, lane_rank_prefix, rank);
  }

  // Each warp scan all the tiles within its sub-chunk.
  //#pragma unroll
  for(int i = 0; i < num_of_tokens_per_thread; i++){
    // The global token id conditions for current token.
    int current_token_id = thread_starting_token + i * WARP_SIZE;
    // If the current token is out-of-bound, then mark it as out-of-bound.
    int token_out_of_bound = 0;
    if(current_token_id >= num_of_total_attn_tokens){
      token_out_of_bound = 1;
    }
    // If the whole tiles are out-of-bound, the warp just finish and exit the scan loop together.
    if(__all_sync(~0, token_out_of_bound) != 0){
      break;
    }
    int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * num_of_ranks_per_node);
    int current_token_local_rank = (current_token_id % (num_of_tokens_per_rank * num_of_ranks_per_node)) / num_of_tokens_per_rank;
    int current_token_local_id = current_token_id % num_of_tokens_per_rank;

    uint8_t rank_mask = 0;
    if(token_out_of_bound == 0){
      rank_mask = token_rank_mask[current_token_id];
    }

    // Convert the routing map to per rank routing info for current token,
    // then produce the per-rank final exclusive scan within the warp for this tile.
    // Use MAX size for compile-time array allocation
    int32_t final_ex_scan[MAX_RANKS_PER_NODE_SCAN];
    bool token_needed_by_local_rank = false;
    int32_t local_rank_prefix_after_scan = 0;
    bool local_rank_seen = false;
    #pragma unroll
    for(int j = 0; j < num_of_ranks_per_node; j++){
      int32_t temp_scan = 0;
      bool token_needed_by_this_rank = ((rank_mask >> j) & 1u) != 0;
      if(token_out_of_bound == 0 && token_needed_by_this_rank){
        temp_scan = 1;
      } else {
        temp_scan = 0;
      }

      // Each warp perform a inclusive scan from all threads(lanes).
      #pragma unroll
      for(int k = 1; k < WARP_SIZE; k *= 2){
        int32_t temp = __shfl_up_sync(~0, temp_scan, k);
        if(lane_id >= k){
          temp_scan += temp;
        }
      }

      // The inclusive scan from last lane is the sum of this rank of this tile. Need to accumulate that for later tiles.
      int32_t temp_sum = __shfl_sync(~0, temp_scan, WARP_SIZE - 1);

      // Make scan exclusive.
      int32_t exclusive_scan = __shfl_up_sync(~0, temp_scan, 1);
      temp_scan = (lane_id >= 1) ? exclusive_scan : 0;

      // Calculate the final exclusive scan for current token. -1 represent that the current rank does not need the current token.
      final_ex_scan[j] = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;

      // Accumulate the sum to accumulator.
      previous_token_sum[j] += temp_sum;

      if (j == local_rank) {
        token_needed_by_local_rank = token_needed_by_this_rank;
        local_rank_prefix_after_scan = previous_token_sum[j];
        local_rank_seen = true;
      }
    }

    // Each thread saves local routing map for this token of the local rank if needed.
    // Optional per-expert counting is fused here to avoid a second pass over local_expert_routing_map.
    if (local_rank_seen) {
      bool lane_participates = (token_out_of_bound == 0) && token_needed_by_local_rank;
      bool* local_expert_routing_map_store_base_addr = nullptr;
      const uint8_t* local_rank_bitmap_row = nullptr;
      if (lane_participates) {
        local_rank_bitmap_row = input_routing_map +
                                current_token_id * packed_row_bytes +
                                node_rank * experts_per_node_packed;
        local_expert_routing_map_store_base_addr =
            local_expert_routing_map + (final_ex_scan[local_rank] * experts_per_rank);
      }

      const int local_expert_bit_base = local_rank * experts_per_rank;
      for (int k = 0; k < experts_per_rank; k++) {
        int expert_bit = local_expert_bit_base + k;
        bool routed_to_expert = false;
        if (lane_participates) {
          routed_to_expert =
              ((local_rank_bitmap_row[expert_bit / 8] >> (expert_bit % 8)) & 1u) != 0;
          local_expert_routing_map_store_base_addr[k] = routed_to_expert;
        }

        if constexpr (ENABLE_PER_EXPERT_COUNTS) {
          unsigned expert_mask = __ballot_sync(~0u, routed_to_expert);
          if (lane_id == 0) {
            int warp_expert_count = __popc(expert_mask);
            if (warp_expert_count > 0) {
              atomicAdd(block_expert_token_counts + k, warp_expert_count);
            }
          }
        }
      }
    }

    // The thread that processing the global last token save the final sum for current rank to num_of_tokens_for_experts.
    if(current_token_id == num_of_total_attn_tokens - 1 && local_rank_seen){
      *num_of_tokens_for_experts = local_rank_prefix_after_scan;
    }

    // Save final exclusive scan of this token back to sparse_to_dense_map if current token is not out-of-bound and is needed.
    if(token_out_of_bound == 0 && current_token_local_rank == local_rank){
      // sparse_to_dense_map store base addr for current token.
      write_t* sparse_to_dense_map_store_base_addr = reinterpret_cast<write_t*>(sparse_to_dense_map +
                                                                                (current_token_node_rank * num_of_tokens_per_rank + current_token_local_id) * num_of_ranks_per_node);
      #pragma unroll
      for(int j = 0; j < S2D_MAP_STORE_ITER; j++){
        sparse_to_dense_map_store_base_addr[j] = *(reinterpret_cast<write_t*>(final_ex_scan) + j);
      }
    }
  }

  if constexpr (ENABLE_PER_EXPERT_COUNTS) {
    __syncthreads();
    for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
      int32_t block_count = block_expert_token_counts[e];
      if (block_count > 0) {
        atomicAdd(per_expert_token_counts + e, block_count);
      }
    }
  }

  // Step 3: When NUM_OF_NODES > 1, we need to produce attn_to_rdma_map.
  // Since each token(row) is fully independent, each token(row) is assigned to each threads in a interleave pattern.
  if constexpr(NUM_OF_NODES != 1){
    const int num_of_total_token_rows = (NUM_OF_NODES - 1) * num_of_tokens_per_rank;
    const int num_of_token_rows_per_thread = ((num_of_total_token_rows - 1) / NUM_OF_TOTAL_THREADS) + 1;

    int tid = threadIdx.x + blockIdx.x * NUM_THREADS_PER_BLOCK;

    //#pragma unroll
    for(int i = 0; i < num_of_token_rows_per_thread; i++){
      int current_token_id = i * NUM_OF_TOTAL_THREADS + tid;
      // If the current token is out-of-bound, then just end processing token rows assigned to this thread.
      if(current_token_id >= num_of_total_token_rows){
        break;
      }
      int current_token_attn_to_rdma_map_node_id = current_token_id % (NUM_OF_NODES - 1);
      int current_token_node_id = current_token_attn_to_rdma_map_node_id < node_rank ? current_token_attn_to_rdma_map_node_id : current_token_attn_to_rdma_map_node_id + 1;
      int current_token_local_id = current_token_id / (NUM_OF_NODES - 1);

      // Load bitmap for this token's remote-node section (current_token_node_id, not node_rank).
      const uint8_t* bitmap_row = input_routing_map +
                                  ((node_rank * num_of_ranks_per_node + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                                  packed_row_bytes +
                                  current_token_node_id * experts_per_node_packed;

      bool* attn_to_rdma_map_base_addr = attn_to_rdma_map + (current_token_local_id * (NUM_OF_NODES - 1) + current_token_attn_to_rdma_map_node_id);

      // Any bit set in this remote-node section means this token must be sent via RDMA.
      bool token_needed_by_this_node = bitmap_range_has_set_bit(bitmap_row, 0, experts_per_node);
      *attn_to_rdma_map_base_addr = token_needed_by_this_node;
    }
  }
}

template<
        // The max num of attn tokens output by a rank/GPU. Used by combine API.
        int MAX_NUM_OF_TOKENS_PER_RANK,
        // Number of total NVLink domain, i.e. the size of RDMA domain.
        int NUM_OF_NODES,
        // LSA team size: number of ranks reachable via NVLink/LSA from this rank.
        // Used for compile-time register-file array sizing; queried at runtime via
        // ncclTeamLsa(comm).nRanks and dispatched via HYBRIDEP_SWITCH_LSA_TEAM_SIZE.
        int LSA_TEAM_SIZE>
class hybrid_ep{
public:

  // Processing metadata. Calculate routing info needed by dispatch and combine operations.
  // input_routing_map: IO: input, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES, NUM_OF_EXPERTS_PER_RANK * NUM_OF_RANKS_PER_NODE * NUM_OF_NODES].
  // Routing map which contain global routing info from all tokens to all expert. Allgather is needed before passing the routing map to this API.
  // preprocessing_tmp: IO: output/input, dtype: tmp_state_t, shape: [NUM_OF_BLOCKS for preprocessing kernel, NUM_OF_RANKS_PER_NODE].
  // The temp buffer needed by the preprocessing kernel.
  // sparse_to_dense_map: IO: output, dtype: int32_t, shape: [NUM_OF_TOKENS_PER_RANK * NUM_OF_NODES, NUM_OF_RANKS_PER_NODE].
  // The routing info needed by NVL warps(i.e. intra-node communication warps) during both dispatch and combine operation. Remains the same in a trainning iteration(FW+BP).
  // rdma_to_attn_map: IO: output, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK padded to 16 * NUM_OF_NODES]
  // The routing info mainly needed by RDMA warps during the combine operation. Remains the same in a trainning iteration(FW+BP).
  // attn_to_rdma_map: IO: output, dtype: bool, shape: [NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES - 1].
  // The routing info mainly needed by RDMA warps during the dispatch operation. Remains the same in a trainning iteration(FW+BP).
  // num_of_tokens_for_experts: IO: output, dtype: int32_t, shape: [1].
  // The total size of expert buffer on this rank(in number of tokens), according to the global routing map. If there are multiple expert on this rank, each token will only appear once.
  // Remains the same in a trainning iteration(FW+BP).
  // local_expert_routing_map: IO: output, dtype: bool, shape: [at least num_of_tokens_for_experts, NUM_OF_EXPERTS_PER_RANK].
  // The per-expert routing info for all tokens within the expert buffer of this rank. It is used by later layer to routing the tokens to different experts on this rank.
  // Remains the same in a trainning iteration(FW+BP).
  template<// Block size for preprocessing kernel.
           int NUM_THREADS_PER_BLOCK,
           // Grid size for preprocessing kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS>
  static void metadata_preprocessing(const uint8_t* input_routing_map,
                                     tmp_state_t* preprocessing_tmp,
                                     int32_t* sparse_to_dense_map,
                                     bool* rdma_to_attn_map,
                                     bool* attn_to_rdma_map,
                                     uint8_t* token_rank_mask,
                                     int32_t* num_of_tokens_for_experts,
                                     bool* local_expert_routing_map,
                                     int32_t* per_expert_token_counts,
                                     const int node_rank,
                                     const int local_rank,
                                     const int num_of_tokens_per_rank,
                                     const int num_of_ranks_per_node,
                                     const int experts_per_rank,
                                     cudaStream_t stream)
  {
    // Init preprocessing_tmp buffers.
    const size_t preprocessing_tmp_sz = NUM_OF_BLOCKS * num_of_ranks_per_node * sizeof(tmp_state_t);
    CUDA_CHECK(cudaMemsetAsync(preprocessing_tmp, 0, preprocessing_tmp_sz, stream));

    // Launch the preprocessing kernel to process the global routing map.
    // CRITICAL: Must use cudaLaunchKernelEx (via LAUNCH_KERNEL macro) instead of <<<>>> syntax.
    // The <<<>>> syntax fails at runtime for heavily templated kernels instantiated through
    // nested macros in separate device compilation mode. cudaLaunchKernelEx uses direct
    // function pointers, bypassing CUDA's fragile static kernel registration tables.
    SETUP_LAUNCH_CONFIG(NUM_OF_BLOCKS, NUM_THREADS_PER_BLOCK, stream);
    // Calculate dynamic shared memory size for scan kernel.
    constexpr int NUM_OF_WARPS_PER_BLOCK_SCAN = NUM_THREADS_PER_BLOCK / 32;
    const size_t scan_base_smem_size =
        (NUM_OF_WARPS_PER_BLOCK_SCAN * num_of_ranks_per_node * sizeof(int32_t)) +
        (num_of_ranks_per_node * sizeof(int32_t));
    if (per_expert_token_counts != nullptr) {
      cfg.dynamicSmemBytes = scan_base_smem_size + (experts_per_rank * sizeof(int32_t));
      auto scan_kernel_ptr = scan<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_OF_NODES, LSA_TEAM_SIZE, true>;
      LAUNCH_KERNEL(&cfg, scan_kernel_ptr,
                    input_routing_map, preprocessing_tmp, sparse_to_dense_map,
                    rdma_to_attn_map, attn_to_rdma_map, token_rank_mask, num_of_tokens_for_experts,
                    local_expert_routing_map, per_expert_token_counts, node_rank, local_rank,
                    num_of_tokens_per_rank, num_of_ranks_per_node, experts_per_rank);
    } else {
      cfg.dynamicSmemBytes = scan_base_smem_size;
      auto scan_kernel_ptr = scan<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_OF_NODES, LSA_TEAM_SIZE, false>;
      LAUNCH_KERNEL(&cfg, scan_kernel_ptr,
                    input_routing_map, preprocessing_tmp, sparse_to_dense_map,
                    rdma_to_attn_map, attn_to_rdma_map, token_rank_mask, num_of_tokens_for_experts,
                    local_expert_routing_map, static_cast<int32_t*>(nullptr), node_rank, local_rank,
                    num_of_tokens_per_rank, num_of_ranks_per_node, experts_per_rank);
    }
  }

  // Dispatch tokens or token gradient to expert MLPs.
  template<// Token data type. Only support uint16_t(represent for BF16) and uint8_t(represent for FP8) for now.
           typename TOKEN_DATA_TYPE,
           // Number of token entry in the shared memory.
           int NUM_OF_STAGES,
           // Number of in-flight S2G token entry in the shared memory, must be smaller than NUM_OF_STAGES.
           int NUM_OF_IN_FLIGHT_S2G,
           // The size of token chunk used in dispatch kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Grid size for dispatch kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS,
           // Whether the dispatch kernel is used in forward process.
           bool FORWARD_DISPATCH>
  static void dispatch(dispatch_kernel_param_t<TOKEN_DATA_TYPE> param, cudaStream_t stream)
  {
    constexpr bool multinode_layout = (NUM_OF_NODES != 1);
    constexpr int NUM_PIPELINES = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
    constexpr int INTER_NODE_GROUP_WARPS = multinode_layout ? 2 : 0;
    constexpr int INTER_NODE_GROUP_START = 0;
    constexpr int INTRA_NODE_G2S_GROUP_WARPS = NUM_PIPELINES;
    constexpr int INTRA_NODE_G2S_GROUP_START = multinode_layout ? 2 : 0;
    constexpr int INTRA_NODE_S2G_GROUP_WARPS = NUM_PIPELINES;
    constexpr int INTRA_NODE_S2G_GROUP_START = multinode_layout ? (2 + NUM_PIPELINES) : NUM_PIPELINES;
    using INTER_NODE_GROUP = warp_group<INTER_NODE_GROUP_WARPS, INTER_NODE_GROUP_START>;
    using INTRA_NODE_G2S_GROUP = warp_group<INTRA_NODE_G2S_GROUP_WARPS, INTRA_NODE_G2S_GROUP_START>;
    using INTRA_NODE_S2G_GROUP = warp_group<INTRA_NODE_S2G_GROUP_WARPS, INTRA_NODE_S2G_GROUP_START>;

    const auto dispatch_kernel_ptr = dispatch_kernel<TOKEN_DATA_TYPE,
                                                     INTER_NODE_GROUP,
                                                     INTRA_NODE_G2S_GROUP,
                                                     INTRA_NODE_S2G_GROUP,
                                                     NUM_OF_STAGES,
                                                     NUM_OF_IN_FLIGHT_S2G,
                                                     NUM_OF_TOKENS_PER_CHUNK,
                                                     MAX_NUM_OF_TOKENS_PER_RANK,
                                                     NUM_OF_NODES,
                                                     NUM_OF_BLOCKS,
                                                     FORWARD_DISPATCH,
                                                     NUM_PIPELINES>;

    dispatch_config_t config;
    model_config_t model;
    config.num_of_stages = NUM_OF_STAGES;
    config.num_of_in_flight_s2g = NUM_OF_IN_FLIGHT_S2G;
    config.num_of_tokens_per_chunk = NUM_OF_TOKENS_PER_CHUNK;
    config.num_of_blocks = NUM_OF_BLOCKS;
    config.forward_dispatch = FORWARD_DISPATCH;
    config.token_data_type = std::is_same_v<TOKEN_DATA_TYPE, uint16_t> ? 1 : 0;
    config.num_pipelines = NUM_PIPELINES;
    config.stages_per_pipeline = NUM_OF_STAGES / NUM_PIPELINES;
    model.hidden_dim = param.hidden_dim;
    model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    model.num_of_experts_per_rank = param.experts_per_rank;
    model.num_of_ranks_per_node = param.num_of_ranks_per_node;
    model.num_of_nodes = NUM_OF_NODES;
    const int SMEM_SIZE = calculate_dispatch_smem_layout_size(config, model);
    static int configured_smem = 0;
    if(SMEM_SIZE > configured_smem){
      CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
      configured_smem = SMEM_SIZE;
    }

    constexpr int BLOCK_DIM = INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size();
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    using warp_timing_entry_t = typename dispatch_kernel_param_t<TOKEN_DATA_TYPE>::warp_timing_entry_t;
    constexpr int WT_WARPS_PER_BLOCK = BLOCK_DIM / 32;
    constexpr int WT_TOTAL = NUM_OF_BLOCKS * WT_WARPS_PER_BLOCK;
    warp_timing_entry_t* d_wt;
    CUDA_CHECK(cudaMalloc(&d_wt, WT_TOTAL * sizeof(warp_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, WT_TOTAL * sizeof(warp_timing_entry_t), stream));
    param.warp_timing = d_wt;
#endif
    dispatch_kernel_ptr<<<NUM_OF_BLOCKS, BLOCK_DIM, SMEM_SIZE, stream>>>(param);
    CUDA_CHECK(cudaGetLastError());
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    CUDA_CHECK(cudaStreamSynchronize(stream));
    warp_timing_entry_t h_wt[WT_TOTAL];
    CUDA_CHECK(cudaMemcpy(h_wt, d_wt, WT_TOTAL * sizeof(warp_timing_entry_t), cudaMemcpyDeviceToHost));
    int _wt_clock_khz;
    CUDA_CHECK(cudaDeviceGetAttribute(&_wt_clock_khz, cudaDevAttrClockRate, 0));
    auto _wt_us = [&](long long cycles) { return (double)cycles * 1000.0 / _wt_clock_khz; };
    auto _wt_print_group = [&](const char* name, int warp_start, int warp_count) {
        if (warp_count == 0) return;
        long long mn = LLONG_MAX, mx = 0, sum = 0;
        int n = 0;
        for (int b = 0; b < NUM_OF_BLOCKS; b++) {
            for (int w = warp_start; w < warp_start + warp_count; w++) {
                long long d = h_wt[b * WT_WARPS_PER_BLOCK + w].end_clock
                            - h_wt[b * WT_WARPS_PER_BLOCK + w].start_clock;
                if (d < mn) mn = d;
                if (d > mx) mx = d;
                sum += d;
                n++;
            }
        }
        printf("  %-9s (%d warp%s x %d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
               name, warp_count, warp_count > 1 ? "s" : " ", NUM_OF_BLOCKS,
               _wt_us(mn), _wt_us(mx), _wt_us(sum / n));
    };
    printf("[DISPATCH WARP TIMING] (%d blocks, %d warps/block, %d pipelines, clock=%d kHz)\n",
           NUM_OF_BLOCKS, WT_WARPS_PER_BLOCK, NUM_PIPELINES, _wt_clock_khz);
    _wt_print_group("N2N",   0, INTER_NODE_GROUP_WARPS);
    _wt_print_group("G2S",   INTER_NODE_GROUP_WARPS, INTRA_NODE_G2S_GROUP_WARPS);
    _wt_print_group("S2G",   INTER_NODE_GROUP_WARPS + INTRA_NODE_G2S_GROUP_WARPS, INTRA_NODE_S2G_GROUP_WARPS);
    CUDA_CHECK(cudaFree(d_wt));
#endif
  }

  // Combine tokens or token gradient from expert MLPs.
  template<// Number of token entry in the shared memory for G2S TMA.
           int NUM_OF_STAGES_G2S,
           // Number of token entry in the shared memory for S2G TMA.
           int NUM_OF_STAGES_S2G,
           // The size of token chunk used in combine kernel.
           int NUM_OF_TOKENS_PER_CHUNK,
           // Number of token per group in the inter-node reduction/G2S warp group.
           int NUM_OF_TOKENS_PER_GROUP,
           // Grid size for combine kernel(1:1 block:SM mapping).
           int NUM_OF_BLOCKS,
           // Number of fully in-flight S2G in intra-node reduction warp group.
           int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
           // Whether the combine kernel is used in backward process.
           bool BACKWARD_COMBINE>
  static void combine(combine_kernel_param_t param, cudaStream_t stream)
  {
    // The warp groups data type for combine kernel, must match the warp groups layout required by the combine kernel.
    constexpr bool multinode_layout = (NUM_OF_NODES != 1);
    constexpr int INTRA_NODE_RED_GROUP_WARPS = multinode_layout ? 4 : 0;
    constexpr int INTRA_NODE_RED_GROUP_START = 0;
    constexpr int INTER_NODE_RED_GROUP_WARPS = 4;
    constexpr int INTER_NODE_RED_GROUP_START = multinode_layout ? 4 : 0;
    constexpr int INTRA_NODE_G2S_GROUP_WARPS = multinode_layout ? 1 : 0;
    constexpr int INTRA_NODE_G2S_GROUP_START = multinode_layout ? 8 : 4;
    constexpr int INTER_NODE_G2S_GROUP_WARPS = multinode_layout ? 1 : 2;
    constexpr int INTER_NODE_G2S_GROUP_START = multinode_layout ? 9 : 4;
    constexpr int INTER_NODE_RDMA_GROUP_WARPS = multinode_layout ? 1 : 0;
    constexpr int INTER_NODE_RDMA_GROUP_START = multinode_layout ? 10 : 6;
    using INTRA_NODE_RED_GROUP = warp_group<INTRA_NODE_RED_GROUP_WARPS, INTRA_NODE_RED_GROUP_START>;
    using INTER_NODE_RED_GROUP = warp_group<INTER_NODE_RED_GROUP_WARPS, INTER_NODE_RED_GROUP_START>;
    using INTRA_NODE_G2S_GROUP = warp_group<INTRA_NODE_G2S_GROUP_WARPS, INTRA_NODE_G2S_GROUP_START>;
    using INTER_NODE_G2S_GROUP = warp_group<INTER_NODE_G2S_GROUP_WARPS, INTER_NODE_G2S_GROUP_START>;
    using INTER_NODE_RDMA_GROUP = warp_group<INTER_NODE_RDMA_GROUP_WARPS, INTER_NODE_RDMA_GROUP_START>;
    constexpr int NUM_OF_DATA_PIPELINE_PER_BLOCK = multinode_layout ? 1 : 2;
    static_assert(INTER_NODE_G2S_GROUP::warp_size() == NUM_OF_DATA_PIPELINE_PER_BLOCK, "Inter-node G2S warp group pipeline and inter-node red warp group pipeline mismatch.");

    // The shared memory needed by the combine kernel.
    // The combine kernel to be launched.
    const auto combine_kernel_ptr = combine_kernel<INTRA_NODE_RED_GROUP, INTER_NODE_RED_GROUP, INTRA_NODE_G2S_GROUP, INTER_NODE_G2S_GROUP, INTER_NODE_RDMA_GROUP, NUM_OF_DATA_PIPELINE_PER_BLOCK, NUM_OF_STAGES_G2S,
                                                   NUM_OF_STAGES_S2G, NUM_OF_TOKENS_PER_GROUP, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK,
                                                  NUM_OF_NODES, NUM_OF_BLOCKS, NUM_OF_ADDITIONAL_IN_FLIGHT_S2G, BACKWARD_COMBINE>;

    // Configure dynamic shared memory for the combine kernel.
    model_config_t model;
    model.hidden_dim = param.hidden_dim;
    model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    model.num_of_experts_per_rank = param.experts_per_rank;
    model.num_of_ranks_per_node = param.num_of_ranks_per_node;
    model.num_of_nodes = NUM_OF_NODES;
    const int SMEM_SIZE = calculate_combine_smem_layout_size<NUM_OF_STAGES_G2S,
                                                                  NUM_OF_STAGES_S2G,
                                                                  NUM_OF_TOKENS_PER_CHUNK,
                                                                  MAX_NUM_OF_TOKENS_PER_RANK,
                                                                  NUM_OF_NODES,
                                                                  BACKWARD_COMBINE>(model);
    // Configure dynamic shared memory; reconfigure if size grows.
    static int configured_smem = 0;
    if(SMEM_SIZE > configured_smem){
      // If the dynamic shared memory requested is too large, we may need to modify the carveout.
      //CUDA_CHECK(cudaFuncSetAttribute(combine_kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      CUDA_CHECK(cudaFuncSetAttribute(combine_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
      configured_smem = SMEM_SIZE;
    }

    // Launch combine kernel (device_sync is fused into combine kernel head).
    constexpr int BLOCK_DIM = INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size();
    combine_kernel_ptr<<<NUM_OF_BLOCKS, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }

};
} // namespace hybrid_ep
