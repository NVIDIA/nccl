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
#include <assert.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda/ptx>
#include <nccl.h>
#include "nccl_device.h"
#include "cuda_compat_shims.cuh" // Compatibility shims for CUDA 12.x
#include "include/common.hpp"

namespace hybrid_ep{


// Single source of truth for "max ranks per node" used for compile-time sizing only.
// Runtime value is `param.num_of_ranks_per_node`.
constexpr int MAX_RANKS_PER_NODE = NUM_MAX_NVL_PEERS;

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
  size_t rdma_inter_node_group_token_offset; // Offset of rdma token buffer from gin_base_ptr
  size_t attn_input_prob_offset;             // Offset of prob staging buffer from gin_base_ptr
  size_t rdma_inter_node_group_prob_offset;  // Offset of rdma prob buffer from gin_base_ptr
  size_t attn_input_scaling_factor_offset;   // Offset of scaling factor staging buffer
  size_t rdma_inter_node_group_scaling_factor_offset; // Offset of rdma scaling factor buffer
  // Batched RDMA staging
  size_t rdma_send_staging_offset;           // Offset of per-destination staging buffer
  size_t rdma_inter_node_group_packed_offset; // Offset of packed receive buffer (token+prob+sf per entry)
  int rdma_batch_size;                       // Tokens per batch (default: 6)
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
  uint32_t* inter_node_num_of_write_per_node;

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
  dispatch_memory_region_info_t* dispatch_memory_region_info;
  uint32_t* inter_node_num_of_write_per_node;

  // Helper functions for accessing staged buffers
  __device__ __forceinline__ void* get_token_buffer(int stage) const {
    return reinterpret_cast<void*>(reinterpret_cast<uint8_t*>(intra_node_token_buffer) + stage * token_buffer_stage_stride);
  }
  __device__ __forceinline__ float* get_prob_buffer(int stage) const {
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(intra_node_prob_buffer) + stage * prob_buffer_stage_stride);
  }
  __device__ __forceinline__ float* get_sf_buffer(int stage) const {
    return reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(intra_node_scaling_factor_buffer) + stage * sf_buffer_stage_stride);
  }
  __device__ __forceinline__ int32_t* get_s2d_map_buffer(int stage, int token_idx) const {
    return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + stage * s2d_map_stage_stride) + token_idx * num_of_ranks_per_node;
  }
  __device__ __forceinline__ int32_t* get_s2d_map_buffer_base(int stage) const {
    return reinterpret_cast<int32_t*>(reinterpret_cast<uint8_t*>(sparse_to_dense_map_buffer) + stage * s2d_map_stage_stride);
  }
  // mbarrier: producer uses [stage*2], consumer uses [stage*2+1]
  __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_producer(int stage) const {
    return intra_node_mbarrier_buffer + stage * 2;
  }
  __device__ __forceinline__ uint64_t* get_intra_node_mbarrier_consumer(int stage) const {
    return intra_node_mbarrier_buffer + stage * 2 + 1;
  }
};

__device__ dispatch_smem_layout_t create_dispatch_smem_layout(
  dispatch_smem_layout_t &layout,
  void* smem_base,
  const dispatch_config_t& config,
  const model_config_t& model)
{
  size_t offset = 0;

  // Token buffer (aligned to 128B for TMA)
  int token_size = (config.token_data_type == 0) ? 1 : 2;
  layout.token_buffer_stage_stride = model.hidden_dim * token_size;

  layout.token_buffer_stage_stride = (layout.token_buffer_stage_stride + 127) & ~127;

  layout.intra_node_token_buffer = reinterpret_cast<void*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += config.num_of_stages * layout.token_buffer_stage_stride;

  // Sparse to dense map buffer (ping-pong, 128B aligned)
  layout.s2d_map_stage_stride = config.num_of_tokens_per_chunk *
                                model.num_of_ranks_per_node * sizeof(int32_t);
  layout.s2d_map_stage_stride = (layout.s2d_map_stage_stride + 127) & ~127;
  layout.sparse_to_dense_map_buffer = reinterpret_cast<int32_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += 2 * layout.s2d_map_stage_stride;

  // Prob buffer (only if forward dispatch, 16B aligned)
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

  // Scaling factor buffer (only if FP8, 16B aligned)
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

  // attn_to_rdma_map buffer (16B aligned, only if multinode)
  if (model.num_of_nodes > 1) {
    offset = (offset + 15) & ~15;
    layout.attn_to_rdma_map_buffer = reinterpret_cast<bool*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
  } else {
    layout.attn_to_rdma_map_buffer = nullptr;
  }

  // Mbarrier buffers (8B aligned)
  offset = (offset + 7) & ~7;
  layout.intra_node_mbarrier_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += config.num_of_stages * 2 * sizeof(uint64_t);

  layout.sparse_to_dense_map_mbarrier_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += 2 * sizeof(uint64_t);

  layout.S2G_group_mbarrier_buffer = reinterpret_cast<uint64_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);

  layout.num_of_ranks_per_node = model.num_of_ranks_per_node;
  layout.dispatch_memory_region_info = reinterpret_cast<dispatch_memory_region_info_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
  layout.inter_node_num_of_write_per_node = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uint8_t*>(smem_base) + offset);
  offset += (model.num_of_nodes - 1) * sizeof(uint32_t);
  return layout;
}
static size_t calculate_dispatch_smem_layout_size(
  const dispatch_config_t& config,
  const model_config_t& model)
{
  size_t total_size = 0;
  int token_size = (config.token_data_type == 0) ? 1 : 2;

  // Token buffer (aligned to 128B for TMA, with per-stage stride alignment)
  int token_buffer_stage_stride = model.hidden_dim * token_size;
  token_buffer_stage_stride = (token_buffer_stage_stride + 127) & ~127;  // 128B align each stage
  total_size += config.num_of_stages * token_buffer_stage_stride;

  // Sparse to dense map buffer (ping-pong, 128B aligned per stage)
  int s2d_map_stage_stride = config.num_of_tokens_per_chunk * model.num_of_ranks_per_node * sizeof(int32_t);
  s2d_map_stage_stride = (s2d_map_stage_stride + 127) & ~127;  // 128B align each stage
  total_size += 2 * s2d_map_stage_stride;

  // Prob buffer (16B aligned per stage)
  if (config.forward_dispatch) {
    int prob_buffer_stage_stride = model.num_of_experts_per_rank * model.num_of_ranks_per_node * sizeof(float);
    prob_buffer_stage_stride = (prob_buffer_stage_stride + 15) & ~15;  // 16B align each stage
    total_size += config.num_of_stages * prob_buffer_stage_stride;
  }

  // Scaling factor buffer (16B aligned per stage, only if FP8)
  if (config.token_data_type == 0) {
    int sf_buffer_stage_stride = (model.hidden_dim / 128) * sizeof(float);
    sf_buffer_stage_stride = (sf_buffer_stage_stride + 15) & ~15;  // 16B align each stage
    total_size += config.num_of_stages * sf_buffer_stage_stride;
  }
  // attn_to_rdma_map buffer (aligned to 16B, only if multinode)
  if (model.num_of_nodes > 1) {
    total_size = (total_size + 15) & ~15;
    total_size += config.num_of_tokens_per_chunk * (model.num_of_nodes - 1) * sizeof(bool);
  }
  // Mbarrier buffers (aligned to 8B)
  total_size = (total_size + 7) & ~7;
  total_size += config.num_of_stages * 2 * sizeof(uint64_t);           // intra_node_mbarrier_buffer
  total_size = (total_size + 7) & ~7;
  total_size += 2 * sizeof(uint64_t);                                  // sparse_to_dense_map mbarrier
  total_size = (total_size + 7) & ~7;
  total_size += sizeof(uint64_t);                                      // S2G group mbarrier
  total_size = (total_size + 7) & ~7;
  // Dispatch memory region info buffer (aligned to 8B, only if multinode)
  if (model.num_of_nodes > 1) {
    total_size = (total_size + 7) & ~7;
    total_size += (model.num_of_nodes - 1) * sizeof(dispatch_memory_region_info_t);
  }
  // Inter-node num of write per node buffer (no alignment needed, only if multinode)
  if (model.num_of_nodes > 1) {
    total_size = (total_size + 7) & ~7;
    total_size += (model.num_of_nodes - 1) * sizeof(uint32_t);
  }
  // inter_node_num_of_write_per_node [(nodes-1)]
  total_size = (total_size + 7) & ~7;
  total_size += (model.num_of_nodes - 1) * sizeof(uint32_t);


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
  } else {
    layout.intra_node_token_G2S_buffer = nullptr;
    layout.intra_node_token_S2G_buffer = nullptr;
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

    align_offset(8);
    layout.inter_node_num_of_write_per_node = reinterpret_cast<uint32_t*>(
        reinterpret_cast<uint8_t*>(smem_base) + offset);
    offset += (model.num_of_nodes - 1) * sizeof(uint32_t);
  } else {
    layout.combine_memory_region_info = nullptr;
    layout.inter_node_num_of_write_per_node = nullptr;
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

  // inter_node_token_G2S_buffer
  total_size = (total_size + 127) & ~127;
  total_size += NUM_OF_STAGES_G2S * hidden_dim * sizeof(uint16_t);

  // inter_node_token_S2G_buffer
  total_size = (total_size + 127) & ~127;
  total_size += NUM_OF_STAGES_S2G * hidden_dim * sizeof(uint16_t);

  // inter_node_token_tail_S2G_buffer (FP32 tail accum)
  const int tail_hidden_dim =
      hidden_dim > COMBINE_REG_HEAD_HIDDEN_DIM
          ? (hidden_dim - COMBINE_REG_HEAD_HIDDEN_DIM)
          : 0;
  if (tail_hidden_dim > 0) {
    total_size = (total_size + 15) & ~15;
    total_size += NUM_OF_STAGES_S2G * (tail_hidden_dim / 2) * sizeof(float2);
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

    // inter_node_num_of_write_per_node [(nodes-1)]
    total_size = (total_size + 7) & ~7;
    total_size += (NUM_OF_NODES - 1) * sizeof(uint32_t);
  }

  // Flag buffers (no special alignment needed)
  if constexpr (multinode) {
    total_size += NUM_OF_STAGES_G2S * sizeof(bool);
  }
  total_size += NUM_OF_STAGES_G2S * sizeof(bool);

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
  uint64_t* expected_rdma_flag_value;
  uint32_t* expected_intra_node_flag_value;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
  // NCCL GIN context
  ncclDevComm_t* dcomms;           // Device communicators array
  ncclWindow_t* nccl_windows;      // Windows array (one per comm)
  int num_gin_comms;               // Number of GIN communicators
  int num_ctx_per_comm;            // Number of contexts per communicator (4)
  void* gin_base_ptr;              // Base pointer for offset calculations
  unsigned signals_base;           // Base signal ID
  // Memory Region info
  struct dispatch_memory_region_info_t mr_info;
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
  uint64_t* expected_rdma_flag_value;
  uint32_t* expected_intra_node_flag_value;
  int local_rank;
  int node_rank;
  // The number of token output by attn layer on a rank/GPU.
  int num_of_tokens_per_rank;
  // NCCL GIN context
  ncclDevComm_t* dcomms;           // Device communicators array
  ncclWindow_t* nccl_windows;      // Windows array (one per comm)
  int num_gin_comms;               // Number of GIN communicators
  int num_ctx_per_comm;            // Number of contexts per communicator (4)
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
inline __device__ void arrive_and_wait(uint32_t num_threads, uint32_t barrier_id = 0) {
    asm volatile("bar.sync %0, %1;" : : "r"(barrier_id), "r"(num_threads));
}

// Helper to compute communicator index and context index from global channel
// Used for 6-comm x 4-ctx GIN configuration (6 communicators with 4 contexts each = 24 total channels)
inline __device__ void get_comm_ctx(int global_channel, int num_ctx_per_comm,
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
inline __device__ void N2N_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const int num_of_ranks_per_node,
                                                      const bool *attn_to_rdma_map,
                                                      ncclDevComm_t* dcomms,
                                                      ncclWindow_t* nccl_windows,
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

  static_assert(INTER_NODE_GROUP::size() == 32, "INTER_NODE_GROUP should be 1 warp.");
  static_assert(INTER_NODE_GROUP::size() >= NUM_OF_NODES - 1, "mr_info should be loaded at once.");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % INTER_NODE_GROUP::size() == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of 32.");
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  // The (NUM_OF_NODES - 1) queue pairs of one block were arranged together.
  int block_offset = blockIdx.x * (NUM_OF_NODES - 1);
  // Loading mr_infos to shared memory for faster access in Put calls.
  struct dispatch_memory_region_info_t *smem_mr_info_ptr = nullptr;
  uint32_t *smem_inter_node_num_of_write_per_node_ptr = nullptr;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->dispatch_memory_region_info;
    smem_inter_node_num_of_write_per_node_ptr = smem_buffer_ptr->inter_node_num_of_write_per_node;
    // Load mr_info[0] into shared memory (same handles for all blocks/remotes)
    // Thread 0 loads the mr_info, other threads initialize write counters
    if (INTER_NODE_GROUP::thread_rank() == 0) {
      smem_mr_info_ptr[0] = mr_info[0];
    }
    if (INTER_NODE_GROUP::thread_rank() < NUM_OF_NODES - 1) {
      smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_GROUP::thread_rank()] = 0;
    }
    __syncwarp();
  }

  // For each chunk.
  for (int chunk_idx = blockIdx.x; chunk_idx < NUM_OF_CHUNKS_PER_RANK; chunk_idx += NUM_OF_BLOCKS) {
    int chunk_base_token_idx = chunk_idx * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    // Attn_to_rdma_map cached in shared memory.
    bool *smem_attn_to_rdma_map_ptr = nullptr;
    // Reading one chunk of attn_to_rdma_map into shared memory.
    if constexpr(NUM_OF_NODES != 1) {
      smem_attn_to_rdma_map_ptr = smem_buffer_ptr->attn_to_rdma_map_buffer;
      if (chunk_base_token_idx + token_range > num_of_tokens_per_rank) {
        token_range = num_of_tokens_per_rank - chunk_base_token_idx;
      }
      for (int map_load_idx = INTER_NODE_GROUP::thread_rank();
           map_load_idx < token_range * (NUM_OF_NODES - 1);
           map_load_idx += INTER_NODE_GROUP::size()) {
        smem_attn_to_rdma_map_ptr[map_load_idx] = attn_to_rdma_map[chunk_base_token_idx * (NUM_OF_NODES - 1) + map_load_idx];
      }
      __syncwarp();
    }

    // Distribute chunks across comms for parallelism
    // Use chunk_idx to select comm - both sender (here) and receiver (G2S)
    // use the same formula so signals match

    // Common values used across all remote nodes
    size_t token_bytes = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
    size_t prob_bytes = (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
    size_t sf_bytes = (HIDDEN_DIM / 128) * sizeof(float);


    // Lane ID for warp-parallel operations
    int lane_id = INTER_NODE_GROUP::thread_rank() % 32;

    // =========================================================================
    // PHASE 1: Issue all RDMA puts to ALL remote nodes using WARP-PARALLEL copies
    // All threads in the warp collaborate to copy each token's data
    // =========================================================================
    for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
      int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);
      // Include chunk_idx in comm_id for better channel distribution
      int total_channels = num_gin_comms * num_ctx_per_comm;
      int global_channel = (remote_idx + block_offset + chunk_idx) % total_channels;
      int comm_idx, ctx_idx;
      get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
      ncclGin net(dcomms[comm_idx], ctx_idx);
      ncclTeam world = ncclTeamWorld(dcomms[comm_idx]);
      // Window array layout: [comm0_ctx0, comm0_ctx1, ..., comm0_ctx3, comm1_ctx0, ...]
      auto nccl_window = nccl_windows[comm_idx * num_ctx_per_comm + ctx_idx];
      int remote_node_id = remote_idx < node_rank ? remote_idx : remote_idx + 1;
      // gin-deepep style: With sub-communicators, each comm only has nNodes ranks
      // The rank within the sub-comm is just the node_id (not global rank)
      int rank_in_remote = remote_idx < node_rank ? node_rank - 1 : node_rank;

      // =========================================================================
      // WARP-COLLABORATIVE TOKEN PROCESSING:
      // All 32 threads work together on each token using warp_copy_int4
      // This achieves ~32x speedup over scalar memcpy
      // =========================================================================
      uint8_t* gin_base = reinterpret_cast<uint8_t*>(gin_base_ptr);

      // Track consecutive runs for batched RDMA
      int run_start = -1;
      int run_count = 0;

      // Iterate over ALL tokens in chunk - warp processes each token together
      for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
        int token_idx = token_idx_in_chunk + chunk_base_token_idx;
        bool need_write = smem_attn_to_rdma_map_ptr[remote_idx + token_idx_in_chunk * (NUM_OF_NODES - 1)];
        bool is_last_token = (token_idx_in_chunk == token_range - 1);

        // uint32_t write_map = __ballot_sync(0xffffffff, need_write);
        // uint32_t partial_write_map = ((1 << INTER_NODE_GROUP::thread_rank()) - 1) & write_map;
        // int write_cnt = __popc(write_map);
        // int write_idx = __popc(partial_write_map);
        if (need_write) {
          if (run_count == 0) {
            run_start = token_idx;
          }

          // Calculate staging entry address for this token
          size_t staging_base = smem_mr_info_ptr->rdma_send_staging_offset +
                                remote_idx * smem_mr_info_ptr->max_tokens_per_dest *
                                smem_mr_info_ptr->bytes_per_entry;
          uint8_t* staging_entry = gin_base + staging_base +
                                   token_idx * smem_mr_info_ptr->bytes_per_entry;

          // WARP-PARALLEL token copy (all 32 threads collaborate)
          const uint8_t* token_src = gin_base +
                                     smem_mr_info_ptr->attn_input_token_offset +
                                    token_idx * token_bytes;
          warp_copy_int4(staging_entry, token_src, token_bytes, lane_id);

          // WARP-PARALLEL prob copy
          if constexpr(FORWARD_DISPATCH) {
            const uint8_t* prob_src = gin_base +
                                      smem_mr_info_ptr->attn_input_prob_offset +
                                      (token_idx * NUM_OF_NODES + remote_node_id) * prob_bytes;
            warp_copy_int4(staging_entry + token_bytes, prob_src, prob_bytes, lane_id);
          }

          // WARP-PARALLEL SF copy
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            const uint8_t* sf_src = gin_base +
                                    smem_mr_info_ptr->attn_input_scaling_factor_offset +
                                    token_idx * sf_bytes;
            size_t sf_offset_in_entry = token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0);
            warp_copy_int4(staging_entry + sf_offset_in_entry, sf_src, sf_bytes, lane_id);
          }

          run_count++;
        }

        // Issue RDMA put when: run breaks, last token, or batch size reached
        int max_batch = smem_mr_info_ptr->num_max_rdma_chunked_send_tokens;
        if (max_batch <= 0) max_batch = 6;  // Default batch size

        bool should_flush = run_count > 0 &&
                           (!need_write || is_last_token || run_count >= max_batch);

        if (should_flush) {
          // Calculate staging buffer offset for this run
          // Tokens are always packed at positions 0, 1, 2, ... run_count-1 in staging
          size_t staging_base = smem_mr_info_ptr->rdma_send_staging_offset +
                                remote_idx * smem_mr_info_ptr->max_tokens_per_dest *
                                smem_mr_info_ptr->bytes_per_entry;
          size_t staging_src = staging_base +
                               run_start * smem_mr_info_ptr->bytes_per_entry;

          size_t packed_dst_offset = smem_mr_info_ptr->rdma_inter_node_group_packed_offset +
                                     rank_in_remote * smem_mr_info_ptr->max_tokens_per_dest *
                                     smem_mr_info_ptr->bytes_per_entry +
                                     run_start * smem_mr_info_ptr->bytes_per_entry;

          // WARP-LEVEL RDMA put (all 32 threads contribute to doorbell)
          net.put(world, remote_node_id,
                  nccl_window, packed_dst_offset,
                  nccl_window, staging_src,
                  run_count * smem_mr_info_ptr->bytes_per_entry,
                  ncclGin_None{}, ncclGin_None{}, ncclCoopWarp());

          run_start = -1;
          run_count = 0;
        } // if (should_flush)
      } // for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk)
    }  // End of Phase 1: all puts issued to all remote nodes

    // =========================================================================
    // PHASE 2: Flush ALL comms to ensure RDMA puts are visible before signals
    // Each comm has its own QP - puts on different comms need separate flushes
    // =========================================================================
    __syncwarp(0xffffffff);
    {
      for (int c = 0; c < num_gin_comms; ++c) {
        ncclGin net_flush(dcomms[c], 0);
        net_flush.flush(ncclCoopWarp(), cuda::std::memory_order_acquire);
      }
    }
    __syncwarp(0xffffffff);

    // =========================================================================
    // PHASE 3: Issue streaming tail signals to ALL remote nodes
    // Using new streaming signal infrastructure: signals_tail_base
    // Signal ID = signals_tail_base + src_node * NUM_NODES * n_ranks + dst_node * n_ranks + local_rank
    // Signal value = 1 per chunk (chunk-based for now, can be token count for full streaming)
    // =========================================================================
    if (INTER_NODE_GROUP::thread_rank() == 0) {
      constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
      for (int idx = 0; idx < NUM_OF_NODES - 1; ++idx) {
        int remote_idx = (idx + node_rank) % (NUM_OF_NODES - 1);

        int remote_node_id = remote_idx < node_rank ? remote_idx : remote_idx + 1;

        // For signals, use a FIXED channel per (local_rank, remote_idx) - NOT per chunk.
        // Signal counters are per-context, so sender and receiver must use the same context.
        int total_channels = num_gin_comms * num_ctx_per_comm;
        int signal_channel = (local_rank * (NUM_OF_NODES - 1) + remote_idx) % total_channels;
        int comm_idx, ctx_idx;
        get_comm_ctx(signal_channel, num_ctx_per_comm, comm_idx, ctx_idx);
        ncclGin net(dcomms[comm_idx], ctx_idx);
        ncclTeam world = ncclTeamWorld(dcomms[comm_idx]);

        // Per-chunk tail signal: sender (node_rank) -> receiver (remote_node_id)
        // Layout: [src_node][dst_node][local_rank][chunk]
        // Each chunk has its own signal ID to avoid out-of-order block completion races.
        unsigned tail_signal_id = smem_mr_info_ptr->signals_tail_base +
                                   node_rank * (NUM_OF_NODES * num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                   remote_node_id * (num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                   local_rank * MAX_CHUNKS_PER_RANK +
                                   chunk_idx;
        // Signal +1 per dispatch (cumulative across dispatches for cached mode)
        net.signal(world,
                   remote_node_id,
                   ncclGin_SignalAdd{tail_signal_id, 1},
                   ncclCoopThread(),
                   ncclGin_None{},
                   cuda::thread_scope_thread,
                   cuda::thread_scope_thread);
      }

      // Flush ALL comms AND ALL contexts to ensure per-chunk signals are visible.
      // Signals use channel = (local_rank * (NUM_OF_NODES-1) + remote_idx) % total_channels,
      // which maps to different (comm_idx, ctx_idx) pairs via get_comm_ctx.
      // Must flush the exact context each signal was posted on.
      for (int c = 0; c < num_gin_comms; ++c) {
        for (int ctx = 0; ctx < num_ctx_per_comm; ++ctx) {
          ncclGin net_signal_flush(dcomms[c], ctx);
          net_signal_flush.flush(ncclCoopThread(), cuda::std::memory_order_acquire);
        }
      }
    }
    __syncwarp(0xffffffff);
  }
}


// Device function for intra-node G2S warp for dispatch kernel. There can be only 1 intra-node G2S warp per CUDA block!
template<typename INTRA_NODE_G2S_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int NUM_OF_TOKENS_PER_CHUNK,
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void G2S_warp_group_device_function(const int local_rank,
                                                      const int node_rank,
                                                      const int num_of_tokens_per_rank,
                                                      const int num_of_ranks_per_node,
                                                      const uint64_t* expected_flag_value,
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

  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  static_assert(MAX_NUM_OF_TOKENS_PER_RANK % NUM_OF_TOKENS_PER_CHUNK == 0, "MAX_NUM_OF_TOKENS_PER_RANK must be multiple of NUM_OF_TOKENS_PER_CHUNK.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  const int max_num_of_chunks_per_rank = ((MAX_NUM_OF_TOKENS_PER_RANK - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  int stage = 0;
  uint32_t consumer_parity = 1;
  int tokens_produced = 0;  // Track how many tokens have been produced

  // Only 1 thread within the G2S warp will be active, other threads will just exit.
  if(cuda::ptx::elect_sync(~0)){
    // Loop through all data chunk. Data(chunk) parallel between multiple CUDA blocks.
    for(int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS){
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if(remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1){
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      }else{
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for(int j = 0; j < NUM_OF_NODES; j++){
        // The current node been processed. For each chunk id, node_id order is local_node, local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        // The tile id within the rdma buffers for the current node id. Because rdma buffers only have NUM_OF_NODES - 1 tile.
        int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
        // Check if the chunk of this node is ready to be consumed.
        // The chunks of local node is the attn input buffers, which are always ready to be consumed.
        // The chunks of remote node is the rdma_inter_node_group buffers, which is produced by remote RDMA Write operation. Should poll the flag produced by remote RDMA Atomic FA before consumed.
        if(node_id != node_rank){
          // =================================================================
          // PER-CHUNK TAIL SIGNAL: Poll for tokens from sender (node_id)
          // Signal ID layout: [src_node][dst_node][local_rank][chunk]
          // Each chunk has its own signal ID for independent completion tracking.
          // Sender (node_id) -> Receiver (node_rank)
          // =================================================================
          constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
          unsigned tail_signal_id = mr_info->signals_tail_base +
                                     node_id * (NUM_OF_NODES * num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                     node_rank * (num_of_ranks_per_node * MAX_CHUNKS_PER_RANK) +
                                     local_rank * MAX_CHUNKS_PER_RANK +
                                     i;  // i is the chunk index from the outer loop

          // Use FIXED channel per (local_rank, sender_remote_idx) to match N2N sender.
          // Signal counters are per-context, so sender and receiver must use the same context.
          int sender_remote_idx = node_rank < node_id ? node_rank : node_rank - 1;
          int total_channels = num_gin_comms * num_ctx_per_comm;
          int signal_channel = (local_rank * (NUM_OF_NODES - 1) + sender_remote_idx) % total_channels;
          int comm_idx, ctx_idx;
          get_comm_ctx(signal_channel, num_ctx_per_comm, comm_idx, ctx_idx);
          ncclGin net(dcomms[comm_idx], ctx_idx);

          // Wait for this specific chunk's signal to reach expected_flag_value.
          // update_expected_value_kernel increments this before each dispatch,
          // so after dispatch #0 value=1, #1 value=2, etc.
          net.waitSignal(ncclCoopThread(), tail_signal_id, *expected_flag_value);
        }
        // Load every token and its properties from Global to Shared. Only load tokens that is needed by this node.
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map +
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));
        const TOKEN_DATA_TYPE* token_load_base_addr = nullptr;
        const float* prob_load_base_addr = nullptr;
        const float* scaling_factor_load_base_addr = nullptr;

        // Packed buffer parameters for remote nodes
        const uint8_t* packed_base = nullptr;
        bool use_packed_layout = false;

        // For other node's attn token and properties, read from packed RDMA buffer.
        // For this node's attn token and properties, read from attn input buffers (separate).
        if (node_id != node_rank) {
          // Remote node: use packed buffer layout

          use_packed_layout = true;
          packed_base = reinterpret_cast<const uint8_t*>(gin_base_ptr) +
                        mr_info->rdma_inter_node_group_packed_offset +
                        rdma_buffer_tile_id * mr_info->max_tokens_per_dest * mr_info->bytes_per_entry +
                        static_cast<size_t>(i * NUM_OF_TOKENS_PER_CHUNK) * mr_info->bytes_per_entry;
        } else {
          // Local node: use separate buffers (unchanged)
          int chunk_first_token_id = i * NUM_OF_TOKENS_PER_CHUNK;
          token_load_base_addr = attn_input_token + chunk_first_token_id * HIDDEN_DIM;
          if constexpr(FORWARD_DISPATCH) {
            prob_load_base_addr = attn_input_prob + chunk_first_token_id * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES);
          }
          if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
            scaling_factor_load_base_addr = attn_input_token_scaling_factor + chunk_first_token_id * (HIDDEN_DIM / 128);
          }
        }
        for (int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++) {
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for (int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            // If the current token is out-of-bound, then just end this load iter.
            if (current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            // If a token is needed by this node(i.e. any expert of this node), load the token and its properties to shared memory entry.
            if (token_needed_by_this_node){
              // Wait until shared memory has free entry.
              // Skip waiting for the first NUM_OF_STAGES tokens since buffers are initially empty.
              if (tokens_produced >= NUM_OF_STAGES){
                while(!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_consumer(stage), consumer_parity)){}
              }
              // Issue TMA to load current token and its properties from global to shared memory.
              uint32_t total_tx_size = 0;

              // Calculate sizes for packed entry offsets
              size_t token_bytes = HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE);
              size_t prob_bytes = (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
              size_t sf_bytes = (HIDDEN_DIM / 128) * sizeof(float);

              if (use_packed_layout) {
                // =================================================================
                // PACKED LAYOUT: Remote node - read from packed buffer
                // Packed entry layout: [token | prob | sf]
                // =================================================================
                const uint8_t* packed_entry = packed_base + current_token_id * mr_info->bytes_per_entry;

                // Load token (offset 0 in packed entry)
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         smem_buffer_ptr->get_token_buffer(stage),
                                         reinterpret_cast<const void*>(packed_entry),
                                         (uint32_t)token_bytes,
                                         smem_buffer_ptr->get_intra_node_mbarrier_producer(stage));
                total_tx_size += (uint32_t)token_bytes;

                // Load prob (offset = token_bytes in packed entry)
                if constexpr(FORWARD_DISPATCH) {
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_prob_buffer(stage)),
                                           reinterpret_cast<const void*>(packed_entry + token_bytes),
                                           (uint32_t)prob_bytes,
                                           smem_buffer_ptr->get_intra_node_mbarrier_producer(stage));
                  total_tx_size += (uint32_t)prob_bytes;
                }

                // Load SF (offset = token_bytes + prob_bytes in packed entry)
                if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
                  size_t sf_offset_in_entry = token_bytes + (FORWARD_DISPATCH ? prob_bytes : 0);
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_sf_buffer(stage)),
                                           reinterpret_cast<const void*>(packed_entry + sf_offset_in_entry),
                                           (uint32_t)sf_bytes,
                                           smem_buffer_ptr->get_intra_node_mbarrier_producer(stage));
                  total_tx_size += (uint32_t)sf_bytes;
                }
              } else {
                // =================================================================
                // SEPARATE LAYOUT: Local node - read from separate buffers
                // =================================================================
                // Load token
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       smem_buffer_ptr->get_token_buffer(stage),
                                       reinterpret_cast<const void*>(token_load_base_addr + (current_token_id * HIDDEN_DIM)),
                                         (uint32_t)token_bytes,
                                       smem_buffer_ptr->get_intra_node_mbarrier_producer(stage));
                total_tx_size += (uint32_t)token_bytes;

                // Load prob (local node has different stride)
              if constexpr(FORWARD_DISPATCH) {
                  const float* prob_load_token_addr = prob_load_base_addr +
                      (current_token_id * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES)) +
                                                               (node_rank * (experts_per_rank * num_of_ranks_per_node));
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(smem_buffer_ptr->get_prob_buffer(stage)),
                                         reinterpret_cast<const void*>(prob_load_token_addr),
                                           (uint32_t)prob_bytes,
                                         smem_buffer_ptr->get_intra_node_mbarrier_producer(stage));
                  total_tx_size += (uint32_t)prob_bytes;
              }

                // Load scaling factor
              if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(smem_buffer_ptr->get_sf_buffer(stage)),
                                         reinterpret_cast<const void*>(scaling_factor_load_base_addr + (current_token_id * (HIDDEN_DIM / 128))),
                                           (uint32_t)sf_bytes,
                                         smem_buffer_ptr->get_intra_node_mbarrier_producer(stage));
                  total_tx_size += (uint32_t)sf_bytes;
                }
              }

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   smem_buffer_ptr->get_intra_node_mbarrier_producer(stage),
                                                   total_tx_size);

              tokens_produced += 1;
              stage += 1;
              if (stage == NUM_OF_STAGES) {
                stage = 0;
                consumer_parity ^= 1;
              }
            }
          }
        }
      }
    }
  }
  // Update residue flags.
  int residue_flag_count = max_num_of_chunks_per_rank - num_of_chunks_per_rank;
  for (int node_id = blockIdx.x; node_id < NUM_OF_NODES - 1; node_id += gridDim.x) {
    uint64_t *residue_flag_base_ptr = rdma_inter_node_group_flags + (node_id * max_num_of_chunks_per_rank + num_of_chunks_per_rank);
    for (int flag_id = INTRA_NODE_G2S_GROUP::thread_rank(); flag_id < residue_flag_count; flag_id += INTRA_NODE_G2S_GROUP::size()) {
      residue_flag_base_ptr[flag_id] = *expected_flag_value;
    }
  }
}

// Device function for intra-node S2G warp group for dispatch kernel.
template<typename INTRA_NODE_S2G_GROUP,
         typename TOKEN_DATA_TYPE,
         typename SMEM_TYPE,
         int NUM_OF_STAGES,
         int NUM_OF_IN_FLIGHT_S2G,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool FORWARD_DISPATCH>
inline __device__ void S2G_warp_group_device_function(const int local_rank,
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
  static_assert(NUM_OF_IN_FLIGHT_S2G < NUM_OF_STAGES, "NUM_OF_IN_FLIGHT_S2G must smaller than NUM_OF_STAGES.");
  // Load rdma_to_attn_map using LDG.128. Each token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the num_of_ranks_per_node.
  // Use max value for compile-time type selection, runtime value for actual iterations
  constexpr int MAX_RANKS_PER_NODE = 8;  // NUM_MAX_NVL_PEERS
  // Type must use MAX for compile-time template instantiation
  using sparse_to_dense_map_load_t = Copy_t<MAX_RANKS_PER_NODE * sizeof(int32_t)>;
  // Runtime calculation based on actual num_of_ranks_per_node
  const int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN = (num_of_ranks_per_node * sizeof(int32_t) + sizeof(sparse_to_dense_map_load_t) - 1) / sizeof(sparse_to_dense_map_load_t);
  // Compile-time constant for element size
  constexpr int NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

  const int remainder_chunk_size = num_of_tokens_per_rank % NUM_OF_TOKENS_PER_CHUNK;
  // How many chunks per rank. Including full chunks and the remainder chunk.
  const int num_of_chunks_per_rank = ((num_of_tokens_per_rank - 1) / NUM_OF_TOKENS_PER_CHUNK) + 1;
  // The rdma_to_attn_map need to be paded to multiple of rdma_to_attn_map_load_t per node.
  // The largest size of rdma_to_attn_map_load_t allowed in all Hybrid-EP kernels are 16B(16 bools), so need to be paded to 16B per node.
  // That means the size of rdma_to_attn_map should be rdma_to_attn_map_size_per_node * NUM_OF_NODES.
  const int rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;
  // How many S2G token entry of have been in-flight.
  int in_flight_s2g = 0;
  int stage = 0;
  uint32_t producer_parity = 0;
  // sparse_to_dense map stage for consuming.
  uint32_t sparse_to_dense_map_stage = 0;
  // sparse_to_dense map parity for consuming.
  uint32_t sparse_to_dense_map_parity = 0;

  // Only 1 thread per warp within the S2G warp group will be active, other threads will just exit.
  if (cuda::ptx::elect_sync(~0)) {
    // First warp(thread) will load the sparse_to_dense map for the first chunk for this CUDA block if any.
    if (INTRA_NODE_S2G_GROUP::warp_rank() == 0){
      if ((int)blockIdx.x < num_of_chunks_per_rank){
        // How many token for this chunk.
        int current_chunk_size;
        if (remainder_chunk_size != 0 && (int)blockIdx.x == num_of_chunks_per_rank - 1){
          current_chunk_size = remainder_chunk_size;
        } else {
          current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
        }
        // sparse_to_dense map load base addr.
        const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_rank * num_of_tokens_per_rank + (int)blockIdx.x * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;
        // Load the sparse_to_dense map for the first chunk.
        cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                 cuda::ptx::space_global,
                                 reinterpret_cast<void*>(smem_buffer_ptr->get_s2d_map_buffer_base(sparse_to_dense_map_stage)),
                                 reinterpret_cast<const void*>(sparse_to_dense_map_load_base_addr),
                                 (uint32_t)(current_chunk_size * num_of_ranks_per_node * sizeof(int32_t)),
                                 smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer + sparse_to_dense_map_stage);

        cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                             cuda::ptx::scope_cta,
                                             cuda::ptx::space_shared,
                                             smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer + sparse_to_dense_map_stage,
                                             (uint32_t)(current_chunk_size * num_of_ranks_per_node * sizeof(int32_t)));
      }
    }
    // Loop through all data chunk. Data(chunk) parallel between multiple CUDA blocks.
    for (int i = blockIdx.x; i < num_of_chunks_per_rank; i += NUM_OF_BLOCKS) {
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
      int current_chunk_size;
      if (remainder_chunk_size != 0 && i == num_of_chunks_per_rank - 1) {
        num_of_routing_info_load_iter_for_current_chunk = ((remainder_chunk_size - 1) / sizeof(rdma_to_attn_map_load_t)) + 1;
        current_chunk_size = remainder_chunk_size;
      } else {
        num_of_routing_info_load_iter_for_current_chunk = NUM_OF_ROUTING_INFO_LOAD_ITER_PER_CHUNK;
        current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
      }
      for (int j = 0; j < NUM_OF_NODES; j++) {
        // All S2G warps(threads) need to sync to make sure all of them have finished consuming the sparse_to_dense map for the last chunk before prefetching the sparse_to_dense map for next chunk.
        // Equal to arrive_and_wait. But arrive_and_wait can only used for whole warps.
        uint64_t state_token = cuda::ptx::mbarrier_arrive(smem_buffer_ptr->S2G_group_mbarrier_buffer);
        while (!cuda::ptx::mbarrier_try_wait(smem_buffer_ptr->S2G_group_mbarrier_buffer, state_token)) {}

        // First warp(thread) will prefetch sparse_to_dense map for next chunk.
        if (INTRA_NODE_S2G_GROUP::warp_rank() == 0) {
          // Calculate next chunk id for this CUDA block to prefetch sparse_to_dense map for next chunk.
          int next_chunk_id;
          int next_node_id;
          int next_node_iter = j + 1;
          if (next_node_iter < NUM_OF_NODES) {
            next_chunk_id = i;
            next_node_id = node_rank >= next_node_iter ? node_rank - next_node_iter : node_rank + NUM_OF_NODES - next_node_iter;
          } else {
            next_chunk_id = i + NUM_OF_BLOCKS;
            next_node_id = node_rank;
          }

          // If next chunk exist, load the sparse_to_dense map for next chunk.
          if (next_chunk_id < num_of_chunks_per_rank) {
            // How many token for this chunk.
            int current_chunk_size;
            if (remainder_chunk_size != 0 && next_chunk_id == num_of_chunks_per_rank - 1) {
              current_chunk_size = remainder_chunk_size;
            } else {
              current_chunk_size = NUM_OF_TOKENS_PER_CHUNK;
            }
            // sparse_to_dense map load base addr.
            const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (next_node_id * num_of_tokens_per_rank + next_chunk_id * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;
            // Load the sparse_to_dense map for the next chunk.
            cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                     cuda::ptx::space_global,
                                     reinterpret_cast<void*>(smem_buffer_ptr->get_s2d_map_buffer_base(sparse_to_dense_map_stage ^ 1)),
                                     reinterpret_cast<const void*>(sparse_to_dense_map_load_base_addr),
                                     (uint32_t)(current_chunk_size * num_of_ranks_per_node * sizeof(int32_t)),
                                     smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer + (sparse_to_dense_map_stage ^ 1));

            cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                 cuda::ptx::scope_cta,
                                                 cuda::ptx::space_shared,
                                                 smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer + (sparse_to_dense_map_stage ^ 1),
                                                 (uint32_t)(current_chunk_size * num_of_ranks_per_node * sizeof(int32_t)));
          }
        }

        // The current node been processed. For each chunk id, node_id order is local_node, local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
        int node_id = node_rank >= j ? node_rank - j : node_rank + NUM_OF_NODES - j;
        // Store every token and its properties from Shared to Global. Only store tokens that is needed by this node.
        const rdma_to_attn_map_load_t* rdma_to_attn_map_load_base_addr = reinterpret_cast<const rdma_to_attn_map_load_t*>(rdma_to_attn_map +
                                                                         (node_id * rdma_to_attn_map_size_per_node + i * NUM_OF_TOKENS_PER_CHUNK));

        // Wait for sparse_to_dense map ready in smem for current chunk.
        while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer + sparse_to_dense_map_stage, sparse_to_dense_map_parity)){}

        for (int k = 0; k < num_of_routing_info_load_iter_for_current_chunk; k++){
          rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[k];
          #pragma unroll
          for (int n = 0; n < NUM_OF_TOKENS_PER_LOAD_ITER; n++){
            int current_token_id = k * NUM_OF_TOKENS_PER_LOAD_ITER + n;
            // If the current token is out-of-bound, then just end this load iter.
            if (current_token_id >= current_chunk_size){
              break;
            }
            bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + n);
            if (token_needed_by_this_node){
              const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                                (smem_buffer_ptr->get_s2d_map_buffer(sparse_to_dense_map_stage, k * NUM_OF_TOKENS_PER_LOAD_ITER + n));
              // Wait until token entry within the shared memory has been produced.
              while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_producer(stage), producer_parity)){}

              // This token entry will be multicast to all ranks within this node which need this token and its properties.
              // The current implementation do the multicast by issue each unicast separately(we call it a unicast group). If NVLS can be used, we should use it here.
              // Multicast of a src token will be ditributed to multiple S2G threads.
              for (int m = INTRA_NODE_S2G_GROUP::warp_rank(); m < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_INPUT_TOKEN; m += INTRA_NODE_S2G_GROUP::warp_size()){
                // Load sparse_to_dense_map.
                sparse_to_dense_map_load_t sparse_to_dense_map_data = sparse_to_dense_map_load_addr[m];
                #pragma unroll
                for (int t = 0; t < NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER; t++){
                  int32_t output_buffer_index = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data) + t);
                  // Only unicast to this rank if it need the current token.
                  if (output_buffer_index != -1) {
                    int remote_rank_id = m * NUM_OF_OUTPUT_TOKENS_PER_LOAD_ITER + t;
                    // Store the token from shared to remote global.
                    TOKEN_DATA_TYPE* remote_token_addr = remote_expert_output_token[remote_rank_id] + (output_buffer_index * HIDDEN_DIM);
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                             cuda::ptx::space_shared,
                                             reinterpret_cast<void*>(remote_token_addr),
                                             smem_buffer_ptr->get_token_buffer(stage),
                                             (uint32_t)(HIDDEN_DIM * sizeof(TOKEN_DATA_TYPE)));

                    // Store the prob from shared to remote global for FW dispatch.
                    if constexpr(FORWARD_DISPATCH) {
                      float* remote_prob_addr = remote_expert_output_prob[remote_rank_id] + (output_buffer_index * (experts_per_rank * num_of_ranks_per_node));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_prob_addr),
                                               reinterpret_cast<const void*>(smem_buffer_ptr->get_prob_buffer(stage)),
                                               (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float)));

                    }

                    // Store the scaling factor from shared to remote global for FP8 tokens.
                    if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value) {
                      float* remote_scaling_factor_addr = remote_expert_output_scaling_factor[remote_rank_id] + (output_buffer_index * (HIDDEN_DIM / 128));
                      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                               cuda::ptx::space_shared,
                                               reinterpret_cast<void*>(remote_scaling_factor_addr),
                                               reinterpret_cast<const void*>(smem_buffer_ptr->get_sf_buffer(stage)),
                                               (uint32_t)((HIDDEN_DIM / 128) * sizeof(float)));

                    }
                  }
                }
              }
              // Commit the previous issued S2G TMA instructions for the same shared memory token entry to a bulk async copy group.
              cuda::ptx::cp_async_bulk_commit_group();
              // Add 1 more in-flight S2G token entry to the counter.
              in_flight_s2g += 1;
              // If in-flight S2G token entry count has exceeded the expectation, release the 1 oldest token entry for the producer.
              if (in_flight_s2g > NUM_OF_IN_FLIGHT_S2G) {
                // Wait for all TMA S2G instructions for the 1 oldest token entry to finish reading the shared memory, so the token entry can be reused by the producer.
                cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_IN_FLIGHT_S2G>{});
                // Reduce 1 in-flight S2G token entry from the counter.
                in_flight_s2g -= 1;
                // Notify the producer warp to load next token entry to the oldest token entry as the shared memory can be reused.
                int notify_stage = (stage - NUM_OF_IN_FLIGHT_S2G) >= 0 ? (stage - NUM_OF_IN_FLIGHT_S2G) : (stage - NUM_OF_IN_FLIGHT_S2G + NUM_OF_STAGES);
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_intra_node_mbarrier_consumer(notify_stage));
              }

              // Goto next token entry in shared memory.
              stage += 1;
              if (stage == NUM_OF_STAGES){
                stage = 0;
                producer_parity ^= 1;
              }
            }
          }
        }
        // Before goto next chunk, go to next sparse_to_dense map stage.
        sparse_to_dense_map_stage += 1;
        if(sparse_to_dense_map_stage == 2){
          sparse_to_dense_map_stage = 0;
          sparse_to_dense_map_parity ^= 1;
        }
      }
    }
  }
}

// Device function for intra-node G2S warp for combine kernel. There can be only 1 such warp per CUDA block!
template<typename SMEM_TYPE,
         int NUM_OF_STAGES_G2S,
         int NUM_OF_TOKENS_PER_CHUNK,
         int NUM_OF_NODES,
         int NUM_OF_BLOCKS,
         bool BACKWARD_COMBINE>
inline __device__ void intra_node_G2S_warp_group_device_function(const int node_rank,
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
  // Load rdma_to_attn_map using LDG.128. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);

  // Load sparse_to_dense_map according to the num_of_ranks_per_node.
  // Use max value for compile-time type selection and array sizing, runtime value for actual iterations
  constexpr int MAX_RANKS_PER_NODE = 8;  // NUM_MAX_NVL_PEERS
  using sparse_to_dense_map_load_t = Copy_t<MAX_RANKS_PER_NODE * sizeof(int32_t)>;
  // Constexpr for array sizing (use max possible value)
  constexpr int MAX_NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER = (MAX_RANKS_PER_NODE * sizeof(int32_t) + sizeof(sparse_to_dense_map_load_t) - 1) / sizeof(sparse_to_dense_map_load_t);
  // Runtime value for actual iteration count
  const int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (num_of_ranks_per_node * sizeof(int32_t) + sizeof(sparse_to_dense_map_load_t) - 1) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

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
  // Token stage id and phase.
  int token_stage = 0;
  uint32_t token_consumer_parity = 1;

  // Only 1 thread within the intra-node G2S warp will be active, other threads will just exit.
  if (cuda::ptx::elect_sync(~0)) {
    // Iterate through all chunks assigned to this block.
    for (int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS) {
      // Which node this chunk will be sent to.
      int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
      // What is the chunk id of this chunk for the node it will be sent to.
      int chunk_id = i / (NUM_OF_NODES - 1);
      // How many rdma_to_attn load iter for this chunk.
      int num_of_routing_info_load_iter_for_current_chunk;
      // How many token for this chunk.
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

      const int32_t* sparse_to_dense_map_load_base_addr = sparse_to_dense_map + (node_id * num_of_tokens_per_rank + chunk_id * NUM_OF_TOKENS_PER_CHUNK) * num_of_ranks_per_node;

      // Iterate through all dst tokens within this chunk.
      for (int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++) {
        rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
        #pragma unroll
        for (int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++) {
          int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
          // If the current token is out-of-bound, then just end this load iter.
          if (current_token_id >= current_chunk_size) {
            break;
          }
          // Check whether this dst token is needed by this node. If not needed, just skip.
          bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
          // If this dst token is needed by this node, load the sparse_to_dense map and load the src token for this dst token.
          if (token_needed_by_this_node) {
            const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                              (sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * num_of_ranks_per_node);
            // Load sparse_to_dense map for this dst token(i.e. a row in sparse_to_dense map).
            // Use MAX size for compile-time array allocation, runtime value for loop bounds
            sparse_to_dense_map_load_t sparse_to_dense_map_data[MAX_NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER];
            // First load sparse_to_dense map and decide the last src token within this row.
            int last_src_token_id = -1;
            for (int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++) {
              sparse_to_dense_map_data[n] = sparse_to_dense_map_load_addr[n];
              const int base_rank = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER;
              if (base_rank >= num_of_ranks_per_node) break;

              // Avoid creating a local array (can spill). Use lane scalars.
              // -1 sentinel is 0xFFFFFFFF in two's complement.
              const int32_t lane0 = sparse_to_dense_map_data[n].x;
              const int32_t lane1 = sparse_to_dense_map_data[n].y;
              const int32_t lane2 = sparse_to_dense_map_data[n].z;
              const int32_t lane3 = sparse_to_dense_map_data[n].w;

              const int r0 = base_rank + 0;
              if (r0 < num_of_ranks_per_node && lane0 != -1) last_src_token_id = r0;
              const int r1 = base_rank + 1;
              if (r1 < num_of_ranks_per_node && lane1 != -1) last_src_token_id = r1;
              const int r2 = base_rank + 2;
              if (r2 < num_of_ranks_per_node && lane2 != -1) last_src_token_id = r2;
              const int r3 = base_rank + 3;
              if (r3 < num_of_ranks_per_node && lane3 != -1) last_src_token_id = r3;
            }
            // Then issue all G2S TMA for this row.
            for (int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++) {
              for (int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++) {
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if (sparse_to_dense_map_value != -1) {
                  int current_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                  // Wait until current token entry within the shared memory has been consumed.
                  while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(token_stage), token_consumer_parity)){}
                  const void* tma_src_addr = reinterpret_cast<const void*>(remote_expert_input_token[current_src_token_id] + (sparse_to_dense_map_value * HIDDEN_DIM));
                  uint32_t total_tx_size = 0;
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_token_G2S(token_stage)),
                                           tma_src_addr,
                                           (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                           smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage));

                  total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

                  if constexpr(BACKWARD_COMBINE) {
                    const void* tma_prob_src_addr = reinterpret_cast<const void*>(remote_expert_input_prob[current_src_token_id] + (sparse_to_dense_map_value * (experts_per_rank * num_of_ranks_per_node)));
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(smem_buffer_ptr->get_intra_node_prob_G2S(token_stage)),
                                             tma_prob_src_addr,
                                             (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float)),
                                             smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage));

                    total_tx_size += (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));
                  }

                  smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage] = (current_src_token_id == last_src_token_id);

                  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                       cuda::ptx::scope_cta,
                                                       cuda::ptx::space_shared,
                                                       smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage),
                                                       total_tx_size);

                  // Goto next token entry in shared memory.
                  token_stage += 1;
                  if (token_stage == NUM_OF_STAGES_G2S) {
                    token_stage = 0;
                    token_consumer_parity ^= 1;
                  }
                }
              }
            }
          }
        }
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
inline __device__ void intra_node_red_warp_group_device_function(const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const bool* rdma_to_attn_map,
                                                                 uint16_t* rdma_intra_node_red_token,
                                                                 float* rdma_intra_node_red_prob,
                                                                 SMEM_TYPE* smem_buffer_ptr,
                                                                 const int HIDDEN_DIM,
                                                                 const int experts_per_rank)
{
  // Load rdma_to_attn_map using LDG.128. Each dst token will need 1 bool from this map.
  using rdma_to_attn_map_load_t = uint4;
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  static_assert(NUM_OF_TOKENS_PER_CHUNK % sizeof(rdma_to_attn_map_load_t) == 0, "NUM_OF_TOKENS_PER_CHUNK must be multiple of rdma_to_attn_map_load_t.");
  constexpr int NUM_OF_RDMA_TO_ATTN_LOAD_ITER_PER_CHUNK = NUM_OF_TOKENS_PER_CHUNK / sizeof(rdma_to_attn_map_load_t);
  constexpr int NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER = sizeof(rdma_to_attn_map_load_t) / sizeof(bool);


  // Processing token using BF16x2 intruction, HIDDEN_DIM must be multiple of 2.
  assert(HIDDEN_DIM % 2 == 0);
  assert(HIDDEN_DIM <= MAX_HIDDEN_DIM);
  const int NUM_OF_BF16X2_ELEMENTS_PER_TOKEN = HIDDEN_DIM / 2;
  const int NUM_OF_ELEMENT_PER_THREAD = ((NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  // Maximum elements per thread for array sizing (based on MAX_HIDDEN_DIM and the actual group size).
  // Using the real group size keeps the accumulator arrays tight and avoids register pressure/spills.
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
  constexpr int MAX_NUM_OF_BF16X2_ELEMENTS_PER_TOKEN_INTRA = MAX_HIDDEN_DIM / 2;
  constexpr int MAX_NUM_OF_ELEMENT_PER_THREAD_INTRA =
      ((MAX_NUM_OF_BF16X2_ELEMENTS_PER_TOKEN_INTRA - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  // Processing prob using fp32.
  const int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((experts_per_rank * num_of_ranks_per_node - 1) / INTRA_NODE_RED_GROUP::size()) + 1;
  // Max prob elements per thread for compile-time array sizing
  constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((NUM_MAX_LOCAL_EXPERTS * NUM_MAX_NVL_PEERS - 1) / INTRA_NODE_RED_GROUP::size()) + 1;

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
  // Src token stage id and phase.
  int token_stage = 0;
  uint32_t token_producer_parity = 0;

  // Dst token stage id.
  int dst_token_stage = 0;

  // Optimized loop parameters (similar to inter_node_red style)
  constexpr int step_size = 4;
  // Round *down* to keep (n+3) in-bounds.
  const int num_of_elements_per_thread_loop = (NUM_OF_ELEMENT_PER_THREAD / step_size) * step_size;
  const int num_of_elements_per_thread_loop_remainder = NUM_OF_ELEMENT_PER_THREAD - num_of_elements_per_thread_loop;

  // Split BF16x2 element processing into:
  // 1) full iterations (all threads in-bounds, no per-element bounds checks needed)
  // 2) one optional tail iteration (only lanes < tail_elems participate)
  const int full_bf16x2_iters = NUM_OF_BF16X2_ELEMENTS_PER_TOKEN / INTRA_NODE_RED_GROUP::size();
  const int tail_bf16x2_elems = NUM_OF_BF16X2_ELEMENTS_PER_TOKEN - full_bf16x2_iters * INTRA_NODE_RED_GROUP::size();
  // Round *down* to keep (m+3) in-bounds for the 4-way unrolled body over full iterations.
  const int full_bf16x2_iters_loop = (full_bf16x2_iters / step_size) * step_size;

  // Whether there are S2G TMA operations of a previous chunk's dst token in-flight(unfinished).
  bool outstanding_in_flight_chunk = false;

  // rdma_remote_node_id and chunk_id for previous chunk.
  int last_chunk_id;
  int last_rdma_remote_node_id;

  // Iterate through all chunks assigned to this block.
  for(int i = blockIdx.x; i < total_num_of_chunks; i += NUM_OF_BLOCKS){
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // What is the chunk id of this chunk for the node it will be sent to.
    int chunk_id = i / (NUM_OF_NODES - 1);
    // Which node this chunk belongs to in output rdma reduction buffers.
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    int rdma_intra_node_red_id = rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    // How many rdma_to_attn load iter for this chunk.
    int num_of_routing_info_load_iter_for_current_chunk;
    // How many token for this chunk.
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
      rdma_intra_node_red_prob_base_ptr = rdma_intra_node_red_prob + rdma_intra_node_red_id * (experts_per_rank * num_of_ranks_per_node);
    }

    // How many dst token entry of current chunk have been in-flight.
    int additional_in_flight_s2g = 0;
    // Iterate through all dst tokens within this chunk.
    for (int j = 0; j < num_of_routing_info_load_iter_for_current_chunk; j++) {
      rdma_to_attn_map_load_t rdma_to_attn_map_data = rdma_to_attn_map_load_base_addr[j];
      #pragma unroll
      for (int k = 0; k < NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER; k++) {
        // Check whether there is a previous chunk's dst token S2G in-flight and also current chunk already has NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token S2G in-flight.
        // If so, wait for previous chunk's S2G finish and notify the RDMA warp groups.
        if (outstanding_in_flight_chunk && (additional_in_flight_s2g == NUM_OF_ADDITIONAL_IN_FLIGHT_S2G)) {
          if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
              // Wait for previous chunk's S2G finish.
              cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<NUM_OF_ADDITIONAL_IN_FLIGHT_S2G>{});
              // Notify the rdma warp group.
              if constexpr(NUM_OF_NODES != 1) {
                cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + last_chunk_id]);
              }
            }
          }
          outstanding_in_flight_chunk = false;
        }
        int current_token_id = j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k;
        // If the current token is out-of-bound, then just end this load iter.
        if (current_token_id >= current_chunk_size) {
          break;
        }
        // Check whether this dst token is needed by this node. If not needed, just skip.
        bool token_needed_by_this_node = *(reinterpret_cast<bool*>(&rdma_to_attn_map_data) + k);
        // If this dst token is needed by this node, which means this dst token will have at least 1 src token within the shread memory.
        // Then, load the src token for this dst token from shared memory and accumulate it to the accumulator.
        if (token_needed_by_this_node) {
          // Accumulator for this dst token. Token must be accumulated in FP32.
          // Use MAX size for compile-time array allocation (stored in registers), actual size determined by runtime HIDDEN_DIM
          float2 acc_token_fp32[MAX_NUM_OF_ELEMENT_PER_THREAD_INTRA];
          // Optional Accumulator for this dst token prob.
          // Use MAX size for compile-time array allocation, actual iterations use runtime value
          float acc_prob[MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD];
          // End reduction group flag.
          bool last_src_token = false;
          // Init accumulator (optimized 4-way unrolled style like inter_node_red).
          for (int n = 0; n < num_of_elements_per_thread_loop; n += step_size) {
            acc_token_fp32[n].x = 0.0f;
            acc_token_fp32[n].y = 0.0f;
            acc_token_fp32[n + 1].x = 0.0f;
            acc_token_fp32[n + 1].y = 0.0f;
            acc_token_fp32[n + 2].x = 0.0f;
            acc_token_fp32[n + 2].y = 0.0f;
            acc_token_fp32[n + 3].x = 0.0f;
            acc_token_fp32[n + 3].y = 0.0f;
          }
          for (int n = 0; n < num_of_elements_per_thread_loop_remainder; n++) {
            acc_token_fp32[num_of_elements_per_thread_loop + n].x = 0.0f;
            acc_token_fp32[num_of_elements_per_thread_loop + n].y = 0.0f;
          }
          if constexpr(BACKWARD_COMBINE) {
            #pragma unroll
            for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
              acc_prob[n] = 0.0f;
            }
          }

          // Continue loading src token for this dst token and reduce them to accumulator until all src token for this dst token have been accumulated.
          do {
            // Base address for current token and prob(optional) in shared memory.
            __nv_bfloat162* load_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_G2S(token_stage));
            float* load_prob_base_ptr;
            if constexpr(BACKWARD_COMBINE) {
              load_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_G2S(token_stage);
            }

            // Wait until current src token ready in shared memory.
            if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
              if (cuda::ptx::elect_sync(~0)) {
                while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_intra_node_mbarrier_G2S_producer(token_stage), token_producer_parity)) {}
              }
            }
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

            // Accumulate token (optimized 4-way unrolled style like inter_node_red).
            int element_id = INTRA_NODE_RED_GROUP::thread_rank();
            int element_id_1 = element_id + INTRA_NODE_RED_GROUP::size();
            int element_id_2 = element_id + 2 * INTRA_NODE_RED_GROUP::size();
            int element_id_3 = element_id + 3 * INTRA_NODE_RED_GROUP::size();
            const int elementStepSize = step_size * INTRA_NODE_RED_GROUP::size();
            // Full iterations (no bounds checks).
            for (int n = 0; n < full_bf16x2_iters_loop; n += step_size) {
              const __nv_bfloat162 src0 = load_token_base_ptr[element_id];
              const __nv_bfloat162 src1 = load_token_base_ptr[element_id_1];
              const __nv_bfloat162 src2 = load_token_base_ptr[element_id_2];
              const __nv_bfloat162 src3 = load_token_base_ptr[element_id_3];

              float2 src_data_fp32 = __bfloat1622float2(src0);
              float2 src_data_fp32_1 = __bfloat1622float2(src1);
              float2 src_data_fp32_2 = __bfloat1622float2(src2);
              float2 src_data_fp32_3 = __bfloat1622float2(src3);
              acc_token_fp32[n].x += src_data_fp32.x;
              acc_token_fp32[n].y += src_data_fp32.y;
              acc_token_fp32[n + 1].x += src_data_fp32_1.x;
              acc_token_fp32[n + 1].y += src_data_fp32_1.y;
              acc_token_fp32[n + 2].x += src_data_fp32_2.x;
              acc_token_fp32[n + 2].y += src_data_fp32_2.y;
              acc_token_fp32[n + 3].x += src_data_fp32_3.x;
              acc_token_fp32[n + 3].y += src_data_fp32_3.y;

              element_id += elementStepSize;
              element_id_1 += elementStepSize;
              element_id_2 += elementStepSize;
              element_id_3 += elementStepSize;
            }
            // Remaining full iterations (0..3, no bounds checks).
            for (int m = full_bf16x2_iters_loop; m < full_bf16x2_iters; ++m) {
              const __nv_bfloat162 src_data = load_token_base_ptr[element_id];
              const float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[m].x += src_data_fp32.x;
              acc_token_fp32[m].y += src_data_fp32.y;
              element_id += INTRA_NODE_RED_GROUP::size();
            }
            // Tail iteration: only lanes < tail_bf16x2_elems participate.
            if (tail_bf16x2_elems != 0 && INTRA_NODE_RED_GROUP::thread_rank() < tail_bf16x2_elems) {
              const __nv_bfloat162 src_data = load_token_base_ptr[element_id];
              const float2 src_data_fp32 = __bfloat1622float2(src_data);
              acc_token_fp32[full_bf16x2_iters].x += src_data_fp32.x;
              acc_token_fp32[full_bf16x2_iters].y += src_data_fp32.y;
            }

            if constexpr(BACKWARD_COMBINE) {
              #pragma unroll
              for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
                int prob_element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
                if (prob_element_id < experts_per_rank * num_of_ranks_per_node) {
                  float src_data = load_prob_base_ptr[prob_element_id];
                  acc_prob[n] += src_data;
                }
              }
            }

            // Check flag for last src token.
            last_src_token = smem_buffer_ptr->intra_node_flag_G2S_buffer[token_stage];

            // Make sure all warp group have finished loading the token entry and accumulate it to the register accumulator.
            // Then notify the producer warp to load next token entry to the shared memory as the shared memory can be reused.
            arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);
            if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
              if (cuda::ptx::elect_sync(~0)) {
                cuda::ptx::mbarrier_arrive(smem_buffer_ptr->get_intra_node_mbarrier_G2S_consumer(token_stage));
              }
            }

            // Goto next src token entry.
            token_stage += 1;
            if (token_stage == NUM_OF_STAGES_G2S) {
              token_stage = 0;
              token_producer_parity ^= 1;
            }

          } while (!last_src_token); // do while accumulating src tokens

          // Base address for current dst token and prob(optional) in shared memory.
          __nv_bfloat162* store_token_base_ptr = reinterpret_cast<__nv_bfloat162*>(smem_buffer_ptr->get_intra_node_token_S2G(dst_token_stage));
          float* store_prob_base_ptr;
          if constexpr(BACKWARD_COMBINE) {
            store_prob_base_ptr = smem_buffer_ptr->get_intra_node_prob_S2G(dst_token_stage);
          }

          // Let the TMA thread to wait for previously issued TMA S2G operations finish reading this entry.
          if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
              cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<NUM_OF_STAGES_S2G - 1>{});
            }
          }
          // Make sure all threads within the red warp group have wait for previously issued TMA S2G operations finish reading this entry before storing new data to this entry.
          arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

          // Store the token (optimized 4-way unrolled style like inter_node_red).
          {
            int store_element_id = INTRA_NODE_RED_GROUP::thread_rank();
            const int store_elementStepSize = step_size * INTRA_NODE_RED_GROUP::size();
            // Full iterations (no bounds checks).
            for (int m = 0; m < full_bf16x2_iters_loop; m += step_size) {
              store_token_base_ptr[store_element_id] = __float22bfloat162_rn(acc_token_fp32[m + 0]);
              store_token_base_ptr[store_element_id + INTRA_NODE_RED_GROUP::size()] = __float22bfloat162_rn(acc_token_fp32[m + 1]);
              store_token_base_ptr[store_element_id + 2 * INTRA_NODE_RED_GROUP::size()] = __float22bfloat162_rn(acc_token_fp32[m + 2]);
              store_token_base_ptr[store_element_id + 3 * INTRA_NODE_RED_GROUP::size()] = __float22bfloat162_rn(acc_token_fp32[m + 3]);
              store_element_id += store_elementStepSize;
            }
            // Remaining full iterations (0..3, no bounds checks).
            for (int m = full_bf16x2_iters_loop; m < full_bf16x2_iters; ++m) {
              store_token_base_ptr[store_element_id] = __float22bfloat162_rn(acc_token_fp32[m]);
              store_element_id += INTRA_NODE_RED_GROUP::size();
            }
            // Tail iteration: only lanes < tail_bf16x2_elems participate.
            if (tail_bf16x2_elems != 0 && INTRA_NODE_RED_GROUP::thread_rank() < tail_bf16x2_elems) {
              store_token_base_ptr[store_element_id] = __float22bfloat162_rn(acc_token_fp32[full_bf16x2_iters]);
            }
          }

          // Store the prob(optional).
          if constexpr(BACKWARD_COMBINE) {
            #pragma unroll
            for (int n = 0; n < NUM_OF_PROB_VEC_ELEMENT_PER_THREAD; n++) {
              int prob_element_id = INTRA_NODE_RED_GROUP::thread_rank() + n * INTRA_NODE_RED_GROUP::size();
              if (prob_element_id < experts_per_rank * num_of_ranks_per_node) {
                store_prob_base_ptr[prob_element_id] = acc_prob[n];
              }
            }
          }

          // Make sure the shared memory stored by current thread is visible by async proxy.
          cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);

          // Make sure all threads within the red warp group have finished storing the current token entry and making it visible to async proxy.
          arrive_and_wait(INTRA_NODE_RED_GROUP::size(), 1);

          // Let the TMA thread to issue S2G TMA operations for current token entry.
          if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
            if (cuda::ptx::elect_sync(~0)) {
              uint16_t* current_token_addr = rdma_intra_node_red_token_base_ptr + (j * NUM_OF_TOKENS_PER_RDMA_TO_ATTN_LOAD_ITER + k) * HIDDEN_DIM;
              // Store the token from shared to global.
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
              // Commit S2G TMA operations for this dst token into a bulk async copy group.
              cuda::ptx::cp_async_bulk_commit_group();
            }
          }

          // Goto next dst token entry.
          dst_token_stage += 1;
          if (dst_token_stage == NUM_OF_STAGES_S2G) {
            dst_token_stage = 0;
          }

          // Another token entry's S2G in-flight.
          additional_in_flight_s2g += 1;
        }
      }
    }
    // If the current chunk does not have NUM_OF_ADDITIONAL_IN_FLIGHT_S2G dst token entry in-flight, which is possible of rdma_to_attn map is really sparse.
    // We need to wait for both previous and current chunks' dst token entry S2G to finish and notify the RDMA warp group.
    if (outstanding_in_flight_chunk) {
      if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
          // Wait for all previous chunk's(i.e. previous and current chunk) S2G finish.
          cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
          // Notify the rdma warp group for previous chunk.
          if constexpr(NUM_OF_NODES != 1) {
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK +last_chunk_id]);
          }
        }
      }
      outstanding_in_flight_chunk = false;
    }

    // Handle current chunk's mbarrier signaling
    if (additional_in_flight_s2g == 0) {
      // No data to send for this chunk - signal mbarrier immediately so inter_N2N can proceed
      if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
        if (cuda::ptx::elect_sync(~0)) {
          if constexpr(NUM_OF_NODES != 1) {
            cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id]);
          }
        }
      }
    } else {
      // Current chunk has data in-flight, mark it for signaling when done
      outstanding_in_flight_chunk = true;
    }

    // Always update last chunk's id for next iteration
    last_rdma_remote_node_id = rdma_remote_node_id;
    last_chunk_id = chunk_id;
  }

  // When all chunks have been processed, we need to check whether the last chunk is still in-flight.
  // If so, wait for it and notify RDMA warp group.
  if (outstanding_in_flight_chunk) {
    if (INTRA_NODE_RED_GROUP::warp_rank() == 0) {
      if (cuda::ptx::elect_sync(~0)) {
        // Wait for the last chunk's S2G finish.
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        // Notify the rdma warp group.
        if constexpr(NUM_OF_NODES != 1) {
          constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
          cuda::ptx::mbarrier_arrive(&smem_buffer_ptr->intra_node_to_rdma_mbarrier_buffer[last_rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + last_chunk_id]);
        }
      }
    }
  }
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
inline __device__ void inter_node_N2N_warp_group_device_function(const int local_rank,
                                                                 const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const bool* rdma_to_attn_map,
                                                                 ncclDevComm_t* dcomms,
                                                                 ncclWindow_t* nccl_windows,
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
  uint32_t *smem_inter_node_num_of_write_per_node_ptr = nullptr;
  uint64_t *intra_node_to_rdma_mbarrier_buffer_ptr = nullptr;
  constexpr int MAX_NUM_OF_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
  if constexpr(NUM_OF_NODES != 1) {
    smem_mr_info_ptr = smem_buffer_ptr->combine_memory_region_info;
    smem_inter_node_num_of_write_per_node_ptr = smem_buffer_ptr->inter_node_num_of_write_per_node;
    // Load mr_info[0] into shared memory (same handles for all blocks/remotes)
    // Thread 0 loads the mr_info, other threads initialize write counters
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      smem_mr_info_ptr[0] = mr_info[0];
    }
    if (INTER_NODE_RDMA_GROUP::thread_rank() < NUM_OF_NODES - 1) {
      smem_inter_node_num_of_write_per_node_ptr[INTER_NODE_RDMA_GROUP::thread_rank()] = 0;
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
  // Loop for every chunks.
  for (int i = blockIdx.x; i < TOTAL_NUM_OF_CHUNKS; i += NUM_OF_BLOCKS) {
    // Which node this chunk will be sent to.
    int node_id = (i % (NUM_OF_NODES - 1) + (node_rank + 1)) % NUM_OF_NODES;
    // With sub-communicators, each comm only has nNodes ranks
    // The rank within the sub-comm is just the node_id (not global rank)
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
    // Window array layout: [comm0_ctx0, comm0_ctx1, ..., comm0_ctx3, comm1_ctx0, ...]
    auto nccl_window = nccl_windows[comm_idx * num_ctx_per_comm + ctx_idx];
    int rdma_remote_node_id = node_id > node_rank ? node_id - 1 : node_id;
    int chunk_base_token_idx = node_id * rdma_to_attn_map_size_per_node + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    int token_range = NUM_OF_TOKENS_PER_CHUNK;
    if (chunk_id * NUM_OF_TOKENS_PER_CHUNK + token_range > num_of_tokens_per_rank) {
      token_range = num_of_tokens_per_rank - chunk_id * NUM_OF_TOKENS_PER_CHUNK;
    }
    // Try wait mbarrier.
    while (!cuda::ptx::mbarrier_try_wait_parity(&intra_node_to_rdma_mbarrier_buffer_ptr[rdma_remote_node_id * MAX_NUM_OF_CHUNKS_PER_RANK + chunk_id], token_consumer_parity)) {}

    // Simple sequential per-token RDMA (non-batched) - only thread 0 does the work
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      for (int token_idx_in_chunk = 0; token_idx_in_chunk < token_range; ++token_idx_in_chunk) {
        int token_idx = token_idx_in_chunk + chunk_id * NUM_OF_TOKENS_PER_CHUNK;
        bool need_write = rdma_to_attn_map[token_idx_in_chunk + chunk_base_token_idx];

        if (need_write) {
          // Calculate offsets relative to gin_base_ptr
          size_t token_src_offset = smem_mr_info_ptr->rdma_intra_node_red_token_offset +
                                    (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + token_idx) * HIDDEN_DIM * sizeof(uint16_t);
          size_t token_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_token_offset +
                                    (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + token_idx) * HIDDEN_DIM * sizeof(uint16_t);
          // Single token RDMA put
          net.put(world,
                  remote_global_rank,
                  nccl_window, token_dst_offset,
                  nccl_window, token_src_offset,
                  HIDDEN_DIM * sizeof(uint16_t),
                  ncclGin_None{},  // no signal
                  ncclGin_None{},  // no counter
                  ncclCoopThread());

          if constexpr(BACKWARD_COMBINE) {

            size_t prob_src_offset = smem_mr_info_ptr->rdma_intra_node_red_prob_offset +
                                     (rdma_remote_node_id * MAX_NUM_OF_TOKENS_PER_RANK + token_idx) *
                                     (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
            size_t prob_dst_offset = smem_mr_info_ptr->combine_rdma_inter_node_group_prob_offset +
                                     (rank_in_remote * MAX_NUM_OF_TOKENS_PER_RANK + token_idx) *
                                     (experts_per_rank * num_of_ranks_per_node) * sizeof(float);
            net.put(world,
                    remote_global_rank,
                    nccl_window, prob_dst_offset,
                    nccl_window, prob_src_offset,
                    (experts_per_rank * num_of_ranks_per_node) * sizeof(float),
                    ncclGin_None{},  // no signal
                    ncclGin_None{},  // no counter
                    ncclCoopThread());
          }
        }
      }
    }
    // Single sync before signal to ensure all threads finished issuing RDMA Puts
    __syncwarp();
    if (INTER_NODE_RDMA_GROUP::thread_rank() == 0) {
      if constexpr (BACKWARD_COMBINE) {
        // Ensure all RDMA puts for this chunk are visible before signaling completion.
        // Old DOCA path had implicit ordering via CQ/atomic progression; GIN path needs explicit flush.
        net.flush(ncclCoopThread(), cuda::std::memory_order_acquire);
      }

      // GIN - include local_rank in signal_id to avoid collision between GPUs on same node
      constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
      unsigned signal_id = signals_base + combine_signal_offset + local_rank * (NUM_OF_NODES * MAX_CHUNKS_PER_RANK) + node_rank * MAX_CHUNKS_PER_RANK + chunk_id;
      // Use net.signal()
      net.signal(world,
                 remote_global_rank,
                 ncclGin_SignalAdd{signal_id, 1},  // signal + value
                 ncclCoopThread(),
                 ncclGin_None{},  // no descriptor
                 cuda::thread_scope_thread,
                 cuda::thread_scope_thread);
    }
    __syncwarp();
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
inline __device__ void inter_node_G2S_warp_group_device_function(const int local_rank,
                                                                 const int node_rank,
                                                                 const int num_of_tokens_per_rank,
                                                                 const int num_of_ranks_per_node,
                                                                 const uint64_t* expected_flag_value,
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

  // Load sparse_to_dense_map according to the num_of_ranks_per_node.
  // Use max value for compile-time type selection, runtime value for actual iterations
  using sparse_to_dense_map_load_t = int4;
  const int NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (num_of_ranks_per_node * sizeof(int32_t) + sizeof(sparse_to_dense_map_load_t) - 1) / sizeof(sparse_to_dense_map_load_t);
  constexpr int NUM_OF_INPUT_TOKENS_PER_LOAD_ITER = sizeof(sparse_to_dense_map_load_t) / sizeof(int32_t);

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
  // Token stage id and phase.
  int token_stage = starting_G2S_index;
  uint32_t token_consumer_parity = 1;

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

          // Second pass: issue TMA for each valid rank across all slices.
          int slice_offset = 0;  // running count of valid ranks from previous slices
          for(int rank_base = 0; rank_base < num_of_ranks_per_node; rank_base += WARP_SIZE){
            const int rank_id = rank_base + lane_id;
            const bool lane_active = (rank_id < num_of_ranks_per_node);
            const int32_t s2d_val = lane_active ? sparse_to_dense_row[rank_id] : -1;
            const unsigned valid_mask = __ballot_sync(0xffffffff, lane_active && s2d_val != -1);
            const int slice_valid = __popc(valid_mask);
            const bool lane_valid = lane_active && s2d_val != -1;
            const int local_lane_rank = __popc(valid_mask & ((1u << lane_id) - 1));
            const int global_rank = slice_offset + local_lane_rank;

            if (lane_valid) {
              const int my_abs_offset = global_offset + global_rank;
              const int stage_idx = starting_G2S_index + (my_abs_offset % ring_len);
              const uint32_t parity = 1u ^ ((uint32_t)(my_abs_offset / ring_len) & 1u);

              // Wait for consumer to free this stage.
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

              // Last-token flag: only the globally last valid lane sets it.
              smem_buffer_ptr->inter_node_flag_G2S_buffer[stage_idx] = (global_rank == total_valid_count - 1);

              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(stage_idx),
                                                   total_tx_size);
            }

            slice_offset += slice_valid;
          }

          // Advance global offset by total valid count for this token.
          global_offset += total_valid_count;
        }
      }
    }
  } else {
    constexpr int MAX_RANKS_PER_NODE = 8;  // NUM_MAX_NVL_PEERS

    constexpr int MAX_NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN = (MAX_RANKS_PER_NODE * sizeof(int32_t) + sizeof(sparse_to_dense_map_load_t) - 1) / sizeof(sparse_to_dense_map_load_t);

  // Only 1 thread within each inter-node G2S warp will be active, other threads will just exit.
  if (cuda::ptx::elect_sync(~0)) {
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

      // RDMA-only state (compiled out for NUM_OF_NODES == 1).
      const bool* attn_to_rdma_map_load_base_addr = nullptr;
      bool rdma_flag_clear[NUM_OF_NODES];
      if constexpr (NUM_OF_NODES > 1) {
        attn_to_rdma_map_load_base_addr = attn_to_rdma_map + (i * NUM_OF_TOKENS_PER_CHUNK) * (NUM_OF_NODES - 1);

        // We still only use first NUM_OF_NODES - 1 flags; the last element is padding.
        #pragma unroll
        for (int jj = 0; jj < NUM_OF_NODES; ++jj) {
          rdma_flag_clear[jj] = false;
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
          // Each dst token need to accumulate src tokens from local node's ranks(this part is the same as intra-node reduction), and src tokens from rdma inter-node buffers.
          // Accumulate local tokens first, then rdma tokens.

          // Check whether this dst token is needed by this(local) node. If not needed, just skip local accumulation.
          bool token_needed_by_this_node = rdma_to_attn_map_load_base_addr[current_token_id];
          // If this dst token is needed by this node, load the sparse_to_dense map and load the local src token for this dst token.
          if (token_needed_by_this_node) {
            const sparse_to_dense_map_load_t* sparse_to_dense_map_load_addr = reinterpret_cast<const sparse_to_dense_map_load_t*>
                                                                              (sparse_to_dense_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * num_of_ranks_per_node);
            // Load sparse_to_dense map for this dst token(i.e. a row in sparse_to_dense map).
            // Use MAX size for compile-time array allocation, runtime value for loop bounds
            sparse_to_dense_map_load_t sparse_to_dense_map_data[MAX_NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN];
            // First load sparse_to_dense map and decide the last src token within this row.
            int last_src_token_id = -1;
            //#pragma unroll
            for (int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++) {
              sparse_to_dense_map_data[n] = sparse_to_dense_map_load_addr[n];
              const int base_rank = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER;
              if (base_rank >= num_of_ranks_per_node) break;

              // Avoid creating a local array (can spill). Use lane scalars.
              // -1 sentinel is 0xFFFFFFFF in two's complement.
              const int32_t lane0 = sparse_to_dense_map_data[n].x;
              const int32_t lane1 = sparse_to_dense_map_data[n].y;
              const int32_t lane2 = sparse_to_dense_map_data[n].z;
              const int32_t lane3 = sparse_to_dense_map_data[n].w;

              const int r0 = base_rank + 0;
              if (r0 < num_of_ranks_per_node && lane0 != -1) last_src_token_id = r0;
              const int r1 = base_rank + 1;
              if (r1 < num_of_ranks_per_node && lane1 != -1) last_src_token_id = r1;
              const int r2 = base_rank + 2;
              if (r2 < num_of_ranks_per_node && lane2 != -1) last_src_token_id = r2;
              const int r3 = base_rank + 3;
              if (r3 < num_of_ranks_per_node && lane3 != -1 ) last_src_token_id = r3;
            }
            // Then issue all G2S TMA for this row.
            for (int n = 0; n < NUM_OF_SPARSE_TO_DENSE_MAP_LOAD_ITER_PER_OUTPUT_TOKEN; n++) {
              for (int m = 0; m < NUM_OF_INPUT_TOKENS_PER_LOAD_ITER; m++) {
                int32_t sparse_to_dense_map_value = *(reinterpret_cast<int32_t*>(&sparse_to_dense_map_data[n]) + m);
                if (sparse_to_dense_map_value != -1) {
                  int current_src_token_id = n * NUM_OF_INPUT_TOKENS_PER_LOAD_ITER + m;
                  // Wait until current token entry within the shared memory has been consumed.
                  while (!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage), token_consumer_parity)){}
                  const void* tma_src_addr = reinterpret_cast<const void*>(remote_expert_input_token[current_src_token_id] + (sparse_to_dense_map_value * HIDDEN_DIM));
                  uint32_t total_tx_size = 0;
                  cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                           cuda::ptx::space_global,
                                           reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(token_stage)),
                                           tma_src_addr,
                                           (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                           smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage));

                  total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

                  if constexpr(BACKWARD_COMBINE) {
                    const void* tma_prob_src_addr = reinterpret_cast<const void*>(remote_expert_input_prob[current_src_token_id] + (sparse_to_dense_map_value * (experts_per_rank * num_of_ranks_per_node)));
                    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                             cuda::ptx::space_global,
                                             reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(token_stage)),
                                             tma_prob_src_addr,
                                             (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float)),
                                             smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage));

                    total_tx_size += (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));
                  }

                  smem_buffer_ptr->inter_node_flag_G2S_buffer[token_stage] = (current_src_token_id == last_src_token_id);


                  cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                       cuda::ptx::scope_cta,
                                                       cuda::ptx::space_shared,
                                                       smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage),
                                                       total_tx_size);

                  // Goto next token entry in shared memory.
                  token_stage += 1;
                  if (token_stage == ending_G2S_index) {
                    token_stage = starting_G2S_index;
                    token_consumer_parity ^= 1;
                  }
                }
              }
            }
          }
          if constexpr (NUM_OF_NODES > 1) {
          // Then accumulate from rdma inter-node buffers. There are total NUM_OF_NODES - 1 (possible) src tokens from rdma buffer to reduce.
            const bool* attn_to_rdma_map_load_addr =
                attn_to_rdma_map_load_base_addr + (j * NUM_OF_TOKENS_PER_GROUP + k) * (NUM_OF_NODES - 1);
          #pragma unroll
          for (int n = 1; n < NUM_OF_NODES; n++) {
            // The current node been processed. For each chunk id, node_id order is
            // (no local_node itself, which is already been accumulated above) local_node - 1, local_node - 2, ......, local_node + 1 and will wrap around.
            int node_id = node_rank >= n ? node_rank - n : node_rank + NUM_OF_NODES - n;
            // The tile id within the rdma buffers for the current node id. Because rdma buffers only have NUM_OF_NODES - 1 tile.
            int rdma_buffer_tile_id = node_id > node_rank ? node_id - 1 : node_id;
            // Check wether current dst token need src token from this node.
            if(attn_to_rdma_map_load_addr[rdma_buffer_tile_id]){
              // If the current chunk is not ready yet, wait for related rdma inter-node group buffer chunks ready first.
              if(rdma_flag_clear[n - 1] == false){
                // Include local_rank in signal_id to match sender's signal (peer GPUs have same local_rank)
                // Wait for COMBINE signal from remote node's combine N2N
                constexpr int MAX_CHUNKS_PER_RANK = MAX_NUM_OF_TOKENS_PER_RANK / NUM_OF_TOKENS_PER_CHUNK;
                unsigned signal_id = signals_base + combine_signal_offset + local_rank * (NUM_OF_NODES * MAX_CHUNKS_PER_RANK) + node_id * MAX_CHUNKS_PER_RANK + i;
                // Use same comm_id as sender: chunk_idx (i) % num_gin_comms
                // Sender (combine N2N) uses chunk_id % num_gin_comms, so receiver must match
                int total_channels = num_gin_comms * num_ctx_per_comm;
                int global_channel = i % total_channels;
                int comm_idx, ctx_idx;
                get_comm_ctx(global_channel, num_ctx_per_comm, comm_idx, ctx_idx);
                ncclGin net(dcomms[comm_idx], ctx_idx);
                net.waitSignal(ncclCoopThread(), signal_id, *expected_flag_value);
                // Mark the chunk from this node(tile) is already clear.
                rdma_flag_clear[n - 1] = true;
              }
              // Wait until current token entry within the shared memory has been consumed.
              while(!cuda::ptx::mbarrier_try_wait_parity(smem_buffer_ptr->get_inter_node_mbarrier_G2S_consumer(token_stage), token_consumer_parity)){}
              // Load the src token from this rdma inter-node group buffer chunk to shared memory entry.
              uint32_t total_tx_size = 0;
              const uint16_t* rdma_inter_node_group_token_load_addr = rdma_inter_node_group_token +
                                                                      (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK +
                                                                      i * NUM_OF_TOKENS_PER_CHUNK +
                                                                      j * NUM_OF_TOKENS_PER_GROUP + k) * HIDDEN_DIM;
              cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                       cuda::ptx::space_global,
                                       reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_token_G2S(token_stage)),
                                       reinterpret_cast<const void*>(rdma_inter_node_group_token_load_addr),
                                       (uint32_t)(HIDDEN_DIM * sizeof(uint16_t)),
                                       smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage));

              total_tx_size += (uint32_t)(HIDDEN_DIM * sizeof(uint16_t));

              if constexpr (BACKWARD_COMBINE){
                const float* rdma_inter_node_group_prob_load_addr = rdma_inter_node_group_prob +
                                                                    (rdma_buffer_tile_id * MAX_NUM_OF_TOKENS_PER_RANK +
                                                                    i * NUM_OF_TOKENS_PER_CHUNK +
                                                                    j * NUM_OF_TOKENS_PER_GROUP + k) * (experts_per_rank * num_of_ranks_per_node);

                cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
                                         cuda::ptx::space_global,
                                         reinterpret_cast<void*>(smem_buffer_ptr->get_inter_node_prob_G2S(token_stage)),
                                         reinterpret_cast<const void*>(rdma_inter_node_group_prob_load_addr),
                                         (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float)),
                                         smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage));

                total_tx_size += (uint32_t)((experts_per_rank * num_of_ranks_per_node) * sizeof(float));
              }

              // Inter-node token does not need flag since the red warp group will also read attn_to_rdma_map.
              cuda::ptx::mbarrier_arrive_expect_tx(cuda::ptx::sem_release,
                                                   cuda::ptx::scope_cta,
                                                   cuda::ptx::space_shared,
                                                   smem_buffer_ptr->get_inter_node_mbarrier_G2S_producer(token_stage),
                                                   total_tx_size);

              // Goto next token entry in shared memory.
              token_stage += 1;
              if (token_stage == ending_G2S_index) {
                token_stage = starting_G2S_index;
                token_consumer_parity ^= 1;
                }
              }
              }
            }
          }
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
        residue_flag_base_ptr[flag_id] = *expected_flag_value;
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
inline __device__ void inter_node_red_warp_group_device_function(const int node_rank,
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
  const int NUM_OF_PROB_VEC_ELEMENT_PER_THREAD = ((experts_per_rank * num_of_ranks_per_node - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;

  // Maximum prob vector elements per thread (worst case: max experts, max ranks, actual pipeline width)
  constexpr int MAX_NUM_OF_PROB_VEC_ELEMENT_PER_THREAD =
      ((NUM_MAX_LOCAL_EXPERTS * 8 - 1) / NUM_OF_THREADS_PER_PIPELINE) + 1;

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

__launch_bounds__(1, 1)
__global__ void device_sync_kernel(uint32_t* intra_node_remote_flags, const uint32_t* expected_flag_value)
{
  // Atomically reduce add 1 to the u32 flag on rank #0 in current NVLink domain.
  // Need a strong system-scope red to make sure all ranks from current NVLink domain can see the side effect.
  // But no memory fence(i.e. .release) needed since CUDA stream already do that for us.
  // red.relaxed.sys.global.add.u32          [a], 1;
  asm volatile("red.relaxed.sys.global.add.u32 [%0], %1;"
                :
                : "l"(__cvta_generic_to_global(intra_node_remote_flags)), "n"(1)
                : "memory");

  // Polling flag value from the u32 flag on rank #0 in current NVLink domain.
  // Keep polling until reach the expected value.
  uint32_t flag_data = 0;
  do {
      flag_data = 0;
      // Need a strong system-scope load to observe other ranks' Atomic result.
      // But no no memory fence(i.e. .aquired) needed since no memory operation behind this.
      asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
                    : "=r"(flag_data)
                    : "l"(__cvta_generic_to_global(intra_node_remote_flags))
                    : "memory");
    } while (flag_data != *expected_flag_value);
}

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
         // This type represent inter-node warp group.
         typename INTER_NODE_GROUP,
         // This type represent intra-node G2S warp group.
         typename INTRA_NODE_G2S_GROUP,
         // This type represent intra-node S2G warp group.
         typename INTRA_NODE_S2G_GROUP,
         // Number of token entry in the shared memory.
         int NUM_OF_STAGES,
         // Number of in-flight S2G token entry in the shared memory, must be smaller than NUM_OF_STAGES.
         int NUM_OF_IN_FLIGHT_S2G,
         // Size of each chunk.
         int NUM_OF_TOKENS_PER_CHUNK,
         // Model configuration.
         int MAX_NUM_OF_TOKENS_PER_RANK,
         int NUM_OF_NODES,
         // Number of CUDA block running dispatch kernel.
         int NUM_OF_BLOCKS,
         // Whether the dispatch kernel is used in forward process or backward process.
         bool FORWARD_DISPATCH>
// Each CUDA block of dispatch kernel has 3 warp groups and has the following layout:
// 1. inter-node warp group(i.e. RDMA N2N warp group, 1 warp, only valid for multinode scenario) 2. intra-node G2S warp group(i.e. NVL G2S warp group, 1 warp).
// 3. intra-node S2G warp group(i.e. NVL S2G warp group, 2(multinode scenario)-3(single-node scenario) warps). Total 4 warps per CUDA block/SM.
__launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size(), 1)
__global__ void dispatch_kernel(const __grid_constant__ dispatch_kernel_param_t<TOKEN_DATA_TYPE> param)
{
  // Compile-time check (only enforce for multi-node layout).
  if constexpr (NUM_OF_NODES != 1) {
  static_assert(INTER_NODE_GROUP::size() == 32, "Dispatch kernel only support 1 N2N warp currently.");
  }
  static_assert(INTRA_NODE_G2S_GROUP::size() == 32, "Dispatch kernel only support 1 G2S warp currently.");
  // The token and its properties should meet size and alignment requirement.
  // Currently, we use TMA to copy prob data, which need at least 16B size and alignment(which requires expert per node to be multiple of 4).
  // We need to add padding or not using TMA for prob, if we want to support other scenario.
  assert((param.experts_per_rank * param.num_of_ranks_per_node * sizeof(float)) % 16 == 0);//, "Currently, expert per node must be multiple of 4(So the prob for each token is multiple of 16B) to make TMA work.");
  assert((param.hidden_dim * sizeof(TOKEN_DATA_TYPE)) % 16 == 0);//, "Currently, the size of token must be multiple of 16B to make TMA work.");
  if constexpr(std::is_same<TOKEN_DATA_TYPE, uint8_t>::value){
    // If FP8 token is used, HIDDEN_DIM must be multiple of 128 for scaling factor usage.
    assert(param.hidden_dim % 128 == 0);//, "HIDDEN_DIM must be multiple of 128 for scaling factor");
    // If FP8 token is used, HIDDEN_DIM must be multiple of 512 to make scaling factor multiple of 16B to make TMA work.
    assert(((param.hidden_dim / 128) * sizeof(float)) % 16 == 0);//, "Currently, scaling factor per token must be multiple of 16B.");
  }


  // Shared memory used over 48KB, should use dynamic shared memory.
  extern __shared__ uint8_t smem_bytes[];
  // using cur_smem_t = dispatch_kernel_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_EXPERTS_PER_RANK, num_of_ranks_per_node, NUM_OF_NODES, FORWARD_DISPATCH>;
  using cur_smem_t  = dispatch_smem_layout_t;

  // Initialize the layout struct (each thread has its own copy in registers)
  cur_smem_t smem_layout;
  dispatch_config_t d_config;
  model_config_t d_model;
  d_config.num_of_stages = NUM_OF_STAGES;
  d_config.num_of_in_flight_s2g = NUM_OF_IN_FLIGHT_S2G;
  d_config.num_of_tokens_per_chunk = NUM_OF_TOKENS_PER_CHUNK;
  d_config.num_of_blocks = NUM_OF_BLOCKS;
  d_config.forward_dispatch = FORWARD_DISPATCH;
  d_config.token_data_type = std::is_same_v<TOKEN_DATA_TYPE, uint16_t> ? 1 : 0;
  d_model.hidden_dim = param.hidden_dim;
  d_model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
  d_model.num_of_experts_per_rank = param.experts_per_rank;
  d_model.num_of_ranks_per_node = param.num_of_ranks_per_node;
  d_model.num_of_nodes = NUM_OF_NODES;
  create_dispatch_smem_layout(smem_layout, smem_bytes, d_config, d_model);
  cur_smem_t* smem_buffer_ptr = &smem_layout;

  // Let first thread of each CUDA block initialize the mbarrier.
  if (threadIdx.x == 0) {
    for (int i = 0; i < NUM_OF_STAGES; i++) {
      // Initialize mbarrier
      cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_buffer + 2 * i, 1);
      cuda::ptx::mbarrier_init(smem_buffer_ptr->intra_node_mbarrier_buffer + 2 * i + 1, INTRA_NODE_S2G_GROUP::warp_size());
    }
    // Initialize sparse_to_dense map mbarrier.
    cuda::ptx::mbarrier_init(smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer, 1);
    cuda::ptx::mbarrier_init(smem_buffer_ptr->sparse_to_dense_map_mbarrier_buffer + 1, 1);
    // Initialize S2G warp group mbarrier.
    cuda::ptx::mbarrier_init(smem_buffer_ptr->S2G_group_mbarrier_buffer, INTRA_NODE_S2G_GROUP::warp_size());
    // Make mbarriers initialization visible to async proxy(TMA).
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }

  // Make sure all the warps wait for mbarriers to be initialized before producing/consuming data.
  __syncthreads();

  // Now warps can become specialized.
  // The input warp group data type must match the warp groups layout.
  // To prevent compiler generate pointless comparison warning.
  int threadIdx_x_int = (int)threadIdx.x;
  if(threadIdx_x_int < INTER_NODE_GROUP::size()){
    // Inter-node warps groups.
    if constexpr(NUM_OF_NODES != 1){
      N2N_warp_group_device_function
      <INTER_NODE_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK, MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
      (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.attn_to_rdma_map,
       param.dcomms, param.nccl_windows, param.num_gin_comms, param.num_ctx_per_comm, param.gin_base_ptr, param.signals_base,
       &param.mr_info, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
    }
  } else if (threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size()){
    // Intra-node G2S warp groups.
    G2S_warp_group_device_function
    <INTRA_NODE_G2S_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK,
     MAX_NUM_OF_TOKENS_PER_RANK, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.expected_rdma_flag_value, param.hidden_dim, param.rdma_to_attn_map, param.attn_input_token,
    param.attn_input_prob, param.attn_input_token_scaling_factor,
    param.rdma_inter_node_group_flags, param.dcomms, param.signals_base, param.num_gin_comms, param.num_ctx_per_comm,
    param.gin_base_ptr, &param.mr_info, smem_buffer_ptr, param.experts_per_rank);
  } else if (threadIdx_x_int < INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size()){
    // Intra-node S2G warp groups.
    S2G_warp_group_device_function
    <INTRA_NODE_S2G_GROUP, TOKEN_DATA_TYPE, cur_smem_t, NUM_OF_STAGES, NUM_OF_IN_FLIGHT_S2G, NUM_OF_TOKENS_PER_CHUNK, NUM_OF_NODES, NUM_OF_BLOCKS, FORWARD_DISPATCH>
    (param.local_rank, param.node_rank, param.num_of_tokens_per_rank, param.num_of_ranks_per_node, param.hidden_dim, param.rdma_to_attn_map, param.sparse_to_dense_map, param.expert_output_token, param.expert_output_prob,
    param.expert_output_scaling_factor, smem_buffer_ptr, param.experts_per_rank);
  } else {
    // Too many threads, should not goes here.
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
  assert((param.experts_per_rank * param.num_of_ranks_per_node * sizeof(float)) % 16 == 0);
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
       param.dcomms, param.nccl_windows, param.num_gin_comms, param.num_ctx_per_comm, param.gin_base_ptr, param.signals_base, param.combine_signal_offset,
       &param.mr_info, smem_buffer_ptr, param.hidden_dim, param.experts_per_rank);
    }
  }else{
    // Too many threads, should not goes here.
  }
}

template<int NUM_THREADS_PER_BLOCK,
         int NUM_OF_BLOCKS,
         int NUM_OF_NODES>
__launch_bounds__(NUM_THREADS_PER_BLOCK, 1)
__global__ void scan(const bool* input_routing_map,
                     tmp_state_t* tmp,
                     int32_t* sparse_to_dense_map,
                     bool* rdma_to_attn_map,
                     bool* attn_to_rdma_map,
                     int32_t* num_of_tokens_for_experts,
                     bool* local_expert_routing_map,
                     const int node_rank,
                     const int local_rank,
                     const int num_of_tokens_per_rank,
                     const int num_of_ranks_per_node,
                     const int experts_per_rank)
{
  // Calculate the warps per block.
  constexpr int WARP_SIZE = 32;
  constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;

  // Calculate total threads count.
  constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;

  // Maximum ranks per node for compile-time sizing of per-thread arrays
  constexpr int MAX_RANKS_PER_NODE_SCAN = 8;  // NUM_MAX_NVL_PEERS

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

  // For each token(row in routing map), calculate how many bytes need to be loaded from the routing map and how to load them.
  static_assert(sizeof(bool) == 1, "Routing map loads assume sizeof(bool) == 1");
  const int NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN = experts_per_rank * num_of_ranks_per_node;
  // Use uint4 (16 bytes) for vectorized loads when possible, otherwise fall back to byte-by-byte
  using copy_t = uint4;
  const int ROUTING_MAP_LOAD_ITER = (NUM_OF_BYTES_TO_LOAD_FOR_EACH_TOKEN + sizeof(copy_t) - 1) / sizeof(copy_t);
  (void)ROUTING_MAP_LOAD_ITER;  // May be used in future optimizations

  // For each token, calculate how many bytes need to be store to sparse_to_dense_map.
  // Use maximum value for compile-time type selection
  constexpr int MAX_NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * MAX_RANKS_PER_NODE_SCAN;
  using write_t = Copy_t<MAX_NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN>;
  const int NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN = sizeof(int32_t) * num_of_ranks_per_node;
  const int S2D_MAP_STORE_ITER = NUM_OF_BYTES_TO_STORE_FOR_EACH_TOKEN / sizeof(write_t);

  // How to convert per-expert routing info to per-rank routing info. We support any number of expert per rank.
  // Use uint8_t for runtime reduction (check each expert individually)
  using expert_to_rank_t = uint8_t;
  const int EXPERTS_TO_RANK_REDUCE_ITER = experts_per_rank;
  (void)EXPERTS_TO_RANK_REDUCE_ITER;  // May be used in future optimizations

  // How to convert per-rank routing info to per-node routing info. We support any number of ranks per node(nvl domain).

  // How do a warp save per-rank routing info back to shared memory. What's the max number of elements does each thread save back.
  const int NUM_OF_RANKS_PER_THREAD = ((num_of_ranks_per_node - 1) / WARP_SIZE) + 1;

  // Use dynamic shared memory for runtime-sized arrays
  extern __shared__ uint8_t smem_bytes[];

  // Sum of per-rank routing info of all warps within the block.
  int32_t* warp_token_routing_map_sum = reinterpret_cast<int32_t*>(smem_bytes);
  // Sum of previous blocks' per-rank routing info.
  int32_t* previous_block_sum = reinterpret_cast<int32_t*>(smem_bytes + NUM_OF_WARPS_PER_BLOCK * num_of_ranks_per_node * sizeof(int32_t));

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
    // Global routing map load base addr for current token.
    const bool* routing_map_load_base_addr = input_routing_map +
                                              current_token_id * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES) +
                                              node_rank * (experts_per_rank * num_of_ranks_per_node);

    // Load the routing map for current token.
    // Use MAX size for array allocation but only access runtime-determined elements
    bool token_routing_map[NUM_MAX_LOCAL_EXPERTS * 8];  // 8 is max num_of_ranks_per_node
    const int actual_routing_map_size = experts_per_rank * num_of_ranks_per_node;
    // Load routing map byte-by-byte for simplicity with runtime size
    for(int j = 0; j < actual_routing_map_size; j++){
      token_routing_map[j] = routing_map_load_base_addr[j];
    }

    // Convert the routing map to per rank routing info and accumulate to accumulator.
    // Also convert the per rank routing info to per node routing info.
    bool token_needed_by_this_node = false;
    #pragma unroll
    for(int j = 0; j < num_of_ranks_per_node; j++){
      bool token_needed_by_this_rank = false;
      // Check each expert for this rank
      for(int k = 0; k < experts_per_rank; k++){
        int expert_idx = j * experts_per_rank + k;
        if(token_routing_map[expert_idx]){
          token_needed_by_this_rank = true;
          break;
        }
      }
      if(token_needed_by_this_rank){
        token_routing_map_sum[j] += 1;
        token_needed_by_this_node = true;
      }
    }

    // Save the per node routing info back to rdma_to_attn_map if needed.
    if(per_node_routing_info){
      rdma_to_attn_map[current_token_rdma_to_attn_map_id] = token_needed_by_this_node;
    }
  }

  // Each warp sum the per-rank routing info from all its threads.
  #pragma unroll
  for(int i = 0; i < num_of_ranks_per_node; i++){
    int dst_tid = i % WARP_SIZE;
    int dst_id = i / WARP_SIZE;
    int32_t temp_sum = __reduce_add_sync(~0, token_routing_map_sum[i]);
    if(lane_id == dst_tid){
      token_routing_map_sum[dst_id] = temp_sum;
    }
  }

  // Each warp store the sum of per-rank routing info back to shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < num_of_ranks_per_node){
      warp_token_routing_map_sum[warp_id * num_of_ranks_per_node + element_id] = token_routing_map_sum[i];
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

  // Each warp load the previous blocks' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < num_of_ranks_per_node){
      previous_token_sum[i] = previous_block_sum[element_id];
    }
  }

  // Each warp accumulate the previous warps' per-rank sum from shared memory.
  #pragma unroll
  for(int i = 0; i < NUM_OF_RANKS_PER_THREAD; i++){
    int element_id = i * WARP_SIZE + lane_id;
    if(element_id < num_of_ranks_per_node){
      for(int j = 0; j < warp_id; j++){
        previous_token_sum[i] += warp_token_routing_map_sum[j * num_of_ranks_per_node + element_id];
      }
    }
  }

  // Each warp broadcast the accumulated previous per-rank routing info to all its threads.
  // Exact reverse of warp reduce operation.
  #pragma unroll
  for(int i = num_of_ranks_per_node - 1; i >= 0 ; i--){
    int src_tid = i % WARP_SIZE;
    int src_id = i / WARP_SIZE;
    previous_token_sum[i] = __shfl_sync(~0, previous_token_sum[src_id], src_tid);
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

    // Global routing map load base addr for current token.
    const bool* routing_map_load_base_addr = input_routing_map +
                                              current_token_id * (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES) +
                                              node_rank * (experts_per_rank * num_of_ranks_per_node);

    // Load the routing map for current token. Only load when the token is not out-of-bound.
    // Use MAX size for array allocation but only access runtime-determined elements
    bool token_routing_map[NUM_MAX_LOCAL_EXPERTS * 8];  // 8 is max num_of_ranks_per_node
    const int actual_routing_map_size = experts_per_rank * num_of_ranks_per_node;
    if(token_out_of_bound == 0){
      // Load routing map byte-by-byte for simplicity with runtime size
      for(int j = 0; j < actual_routing_map_size; j++){
        token_routing_map[j] = routing_map_load_base_addr[j];
      }
    }

    // Convert the routing map to per rank routing info for current token,
    // then produce the per-rank final exclusive scan within the warp for this tile.
    // Use MAX size for compile-time array allocation
    int32_t final_ex_scan[MAX_RANKS_PER_NODE_SCAN];
    #pragma unroll
    for(int j = 0; j < num_of_ranks_per_node; j++){
      int32_t temp_scan = 0;
      bool token_needed_by_this_rank = false;
      // If the token is not out-of-bound, check whether this rank need this token.
      if(token_out_of_bound == 0){
        // Check each expert for this rank
        for(int k = 0; k < experts_per_rank; k++){
          int expert_idx = j * experts_per_rank + k;
          if(token_routing_map[expert_idx]){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          temp_scan = 1;
        }else{
          temp_scan = 0;
        }
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

      // Each thread save local routing map for this token of the local rank to local_expert_routing_map if this token is needed by the local rank.
      if(j == local_rank && token_needed_by_this_rank){
        bool* local_expert_routing_map_store_base_addr = local_expert_routing_map + (final_ex_scan[j] * experts_per_rank);
        // Store the expert routing info for this token
        for(int k = 0; k < experts_per_rank; k++){
          int expert_idx = j * experts_per_rank + k;
          local_expert_routing_map_store_base_addr[k] = token_routing_map[expert_idx];
        }
      }

      // The thread that processing the global last token save the final sum for current rank to num_of_tokens_for_experts.
      if(current_token_id == num_of_total_attn_tokens - 1 && j == local_rank){
        *num_of_tokens_for_experts = previous_token_sum[j];
      }
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

      const bool* routing_map_load_base_addr = input_routing_map +
                                                ((node_rank * num_of_ranks_per_node + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                                                (experts_per_rank * num_of_ranks_per_node * NUM_OF_NODES) +
                                                (current_token_node_id * experts_per_rank * num_of_ranks_per_node);

      bool* attn_to_rdma_map_base_addr = attn_to_rdma_map + (current_token_local_id * (NUM_OF_NODES - 1) + current_token_attn_to_rdma_map_node_id);

      // Load the routing map for current token row.
      // Use MAX size for array allocation but only access runtime-determined elements
      bool token_routing_map[NUM_MAX_LOCAL_EXPERTS * 8];  // 8 is max num_of_ranks_per_node
      const int actual_routing_map_size_step3 = experts_per_rank * num_of_ranks_per_node;
      // Load routing map byte-by-byte for simplicity with runtime size
      for(int j = 0; j < actual_routing_map_size_step3; j++){
        token_routing_map[j] = routing_map_load_base_addr[j];
      }

      // Convert the routing map to per rank routing info and then to per node routing info.
      bool token_needed_by_this_node = false;
      #pragma unroll
      for(int j = 0; j < num_of_ranks_per_node; j++){
        bool token_needed_by_this_rank = false;
        // Check each expert for this rank
        for(int k = 0; k < experts_per_rank; k++){
          int expert_idx = j * experts_per_rank + k;
          if(token_routing_map[expert_idx]){
            token_needed_by_this_rank = true;
            break;
          }
        }
        if(token_needed_by_this_rank){
          token_needed_by_this_node = true;
          break;
        }
      }

      *attn_to_rdma_map_base_addr = token_needed_by_this_node;
    }
  }
}

template<
        // The max num of attn tokens output by a rank/GPU. Used by combine API.
        int MAX_NUM_OF_TOKENS_PER_RANK,
        // Number of total NVLink domain, i.e. the size of RDMA domain.
        int NUM_OF_NODES>
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
  static void metadata_preprocessing(const bool* input_routing_map,
                                     tmp_state_t* preprocessing_tmp,
                                     int32_t* sparse_to_dense_map,
                                     bool* rdma_to_attn_map,
                                     bool* attn_to_rdma_map,
                                     int32_t* num_of_tokens_for_experts,
                                     bool* local_expert_routing_map,
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
    // Calculate dynamic shared memory size for scan kernel
    constexpr int NUM_OF_WARPS_PER_BLOCK_SCAN = NUM_THREADS_PER_BLOCK / 32;
    const size_t scan_smem_size = (NUM_OF_WARPS_PER_BLOCK_SCAN * num_of_ranks_per_node * sizeof(int32_t)) +
                                    (num_of_ranks_per_node * sizeof(int32_t));
    cfg.dynamicSmemBytes = scan_smem_size;  // Set dynamic shared memory size
    auto scan_kernel_ptr = scan<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_OF_NODES>;
    LAUNCH_KERNEL(&cfg, scan_kernel_ptr,
                  input_routing_map, preprocessing_tmp, sparse_to_dense_map,
                  rdma_to_attn_map, attn_to_rdma_map, num_of_tokens_for_experts,
                  local_expert_routing_map, node_rank, local_rank, num_of_tokens_per_rank, num_of_ranks_per_node, experts_per_rank);
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
    // The warp groups data type for dispatch kernel, must match the warp groups layout required by the dispatch kernel.
    constexpr bool multinode_layout = (NUM_OF_NODES != 1);
    constexpr int INTER_NODE_GROUP_WARPS = multinode_layout ? 1 : 0;
    constexpr int INTER_NODE_GROUP_START = 0;
    constexpr int INTRA_NODE_G2S_GROUP_WARPS = 1;
    constexpr int INTRA_NODE_G2S_GROUP_START = multinode_layout ? 1 : 0;
    constexpr int INTRA_NODE_S2G_GROUP_WARPS = multinode_layout ? 2 : 3;
    constexpr int INTRA_NODE_S2G_GROUP_START = multinode_layout ? 2 : 1;
    using INTER_NODE_GROUP = warp_group<INTER_NODE_GROUP_WARPS, INTER_NODE_GROUP_START>;
    using INTRA_NODE_G2S_GROUP = warp_group<INTRA_NODE_G2S_GROUP_WARPS, INTRA_NODE_G2S_GROUP_START>;
    using INTRA_NODE_S2G_GROUP = warp_group<INTRA_NODE_S2G_GROUP_WARPS, INTRA_NODE_S2G_GROUP_START>;
    // The shared memory needed by the dispatch kernel.
    // using dispatch_kernel_smem_t = dispatch_kernel_dynamic_shared_memory_buffer_t<TOKEN_DATA_TYPE, NUM_OF_STAGES, NUM_OF_TOKENS_PER_CHUNK,
    //                                                                               NUM_OF_EXPERTS_PER_RANK, num_of_ranks_per_node, NUM_OF_NODES, FORWARD_DISPATCH>;
    // using dispatch_kernel_smem_t = dispatch_smem_layout_t;
    // The dispatch kernel to be launched.
    const auto dispatch_kernel_ptr = dispatch_kernel<TOKEN_DATA_TYPE,
                                                     INTER_NODE_GROUP, //
                                                     INTRA_NODE_G2S_GROUP,
                                                     INTRA_NODE_S2G_GROUP,
                                                     NUM_OF_STAGES,
                                                     NUM_OF_IN_FLIGHT_S2G,
                                                     NUM_OF_TOKENS_PER_CHUNK,
                                                     MAX_NUM_OF_TOKENS_PER_RANK,
                                                     NUM_OF_NODES,
                                                     NUM_OF_BLOCKS,
                                                     FORWARD_DISPATCH>;

    // Configure dynamic shared memory for the dispatch kernel.
    dispatch_config_t config;
    model_config_t model;
    config.num_of_stages = NUM_OF_STAGES;
    config.num_of_in_flight_s2g = NUM_OF_IN_FLIGHT_S2G;
    config.num_of_tokens_per_chunk = NUM_OF_TOKENS_PER_CHUNK;
    config.num_of_blocks = NUM_OF_BLOCKS;
    config.forward_dispatch = FORWARD_DISPATCH;
    config.token_data_type = std::is_same_v<TOKEN_DATA_TYPE, uint16_t> ? 1 : 0;
    model.hidden_dim = param.hidden_dim;
    model.max_num_of_tokens_per_rank = MAX_NUM_OF_TOKENS_PER_RANK;
    model.num_of_experts_per_rank = param.experts_per_rank;
    model.num_of_ranks_per_node = param.num_of_ranks_per_node;
    model.num_of_nodes = NUM_OF_NODES;
    const int SMEM_SIZE = calculate_dispatch_smem_layout_size(config, model);
    // Configure dynamic shared memory; reconfigure if size grows.
    static int configured_smem = 0;
    if(SMEM_SIZE > configured_smem){
      // If the dynamic shared memory requested is too large, we may need to modify the carveout.
      //CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel_ptr, cudaFuncAttributePreferredSharedMemoryCarveout, 100));
      CUDA_CHECK(cudaFuncSetAttribute(dispatch_kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE));
      configured_smem = SMEM_SIZE;
    }

    // Launch update_expected_value_kernel to update expected flag value.
    update_expected_value_kernel<NUM_OF_NODES>
    <<<1, 1, 0, stream>>>(param.expected_rdma_flag_value, param.expected_intra_node_flag_value, param.num_of_ranks_per_node);

    // Launch dispatch kernel.
    constexpr int BLOCK_DIM = INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size();
    dispatch_kernel_ptr<<<NUM_OF_BLOCKS, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Launch device sync kernel.
    device_sync_kernel<<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value);
    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
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

    // Launch update_expected_value_kernel to update expected flag value.
    update_expected_value_kernel<NUM_OF_NODES>
    <<<1, 1, 0, stream>>>(param.expected_rdma_flag_value, param.expected_intra_node_flag_value, param.num_of_ranks_per_node);

    // Launch device sync kernel.
    device_sync_kernel<<<1, 1, 0, stream>>>(param.intra_node_write_completion_flags, param.expected_intra_node_flag_value);

    // Launch combine kernel.
    constexpr int BLOCK_DIM = INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size();
    combine_kernel_ptr<<<NUM_OF_BLOCKS, BLOCK_DIM, SMEM_SIZE, stream>>>(param);

    // Check if there is any CUDA error.
    CUDA_CHECK(cudaGetLastError());
  }

};
} // namespace hybrid_ep
