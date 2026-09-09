/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once
#include "common.hpp"
#include "device_primitives.cuh"

namespace hybrid_ep {

enum scan_state {
    EMPTY = 0,
    PRIV_SUM = 1
};

struct tmp_state_t {
    scan_state state;
    int32_t value;
};

static constexpr int WARP_SIZE = 32;

template <int NUM_MASK_WORDS>
struct rank_mask_t {
    uint64_t words[NUM_MASK_WORDS];

    __device__ __forceinline__ bool test(int r) const {
        return (words[r >> 6] >> (r & 63)) & 1ull;
    }
    __device__ __forceinline__ void set(int r) {
        words[r >> 6] |= 1ull << (r & 63);
    }
    __device__ __forceinline__ bool any() const {
        uint64_t v = 0;
#pragma unroll
        for (int i = 0; i < NUM_MASK_WORDS; i++) v |= words[i];
        return v != 0;
    }
    __device__ __forceinline__ void clear() {
#pragma unroll
        for (int i = 0; i < NUM_MASK_WORDS; i++) words[i] = 0;
    }
};

struct scan_smem_t {
    int32_t* warp_rank_sums;
    int32_t* block_prefix;
    int32_t* warp_prefix;
    int32_t* expert_counts;

    __device__ static scan_smem_t from_raw(uint8_t* smem, int num_warps, int num_ranks, bool has_expert_counts) {
        scan_smem_t s;
        s.warp_rank_sums = reinterpret_cast<int32_t*>(smem);
        s.block_prefix = s.warp_rank_sums + num_warps * num_ranks;
        s.warp_prefix = s.block_prefix + num_ranks;
        s.expert_counts = has_expert_counts ? s.warp_prefix + num_warps * num_ranks : nullptr;
        return s;
    }
};

static __device__ __forceinline__ bool
bitmap_range_has_set_bit(const uint8_t* bitmap_row, int bit_begin, int bit_count) {
    if (bit_count <= 0) return false;

    const int bit_end = bit_begin + bit_count;
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
    if ((bitmap_row[first_byte] & first_mask) != 0) return true;

    uint8_t middle_or = 0;
    for (int b = first_byte + 1; b < last_byte; b++) middle_or |= bitmap_row[b];
    if (middle_or != 0) return true;

    const uint8_t last_mask = static_cast<uint8_t>((last_bit == 7) ? 0xFFu : ((1u << (last_bit + 1)) - 1u));
    return (bitmap_row[last_byte] & last_mask) != 0;
}

template <int NUM_MASK_WORDS, int LSA_TEAM_SIZE>
static __device__ __forceinline__ rank_mask_t<NUM_MASK_WORDS>
bitmap_row_to_rank_mask(const uint8_t* bitmap_row, int num_of_ranks_per_node, int experts_per_rank) {
    rank_mask_t<NUM_MASK_WORDS> rank_mask;
    rank_mask.clear();
#pragma unroll
    for (int rank = 0; rank < LSA_TEAM_SIZE; rank++) {
        if (rank < num_of_ranks_per_node) {
            const int bit_begin = rank * experts_per_rank;
            if (bitmap_range_has_set_bit(bitmap_row, bit_begin, experts_per_rank)) {
                rank_mask.set(rank);
            }
        }
    }
    return rank_mask;
}

template <int LSA_TEAM_SIZE, int NUM_MASK_WORDS>
__device__ __forceinline__ void tally_ranks(
    const uint8_t* input_routing_map,
    rank_mask_t<NUM_MASK_WORDS>* token_rank_mask,
    bool* rdma_to_attn_map,
    int32_t* rank_sums,
    int thread_starting_token,
    int num_of_tokens_per_thread,
    int num_of_total_attn_tokens,
    int num_of_tokens_per_rank,
    int num_of_ranks_per_node,
    int experts_per_rank,
    int packed_row_bytes,
    int experts_per_node_packed,
    int node_rank,
    int local_rank,
    int rdma_to_attn_map_size_per_node) {
#pragma unroll
    for (int i = 0; i < LSA_TEAM_SIZE; i++) rank_sums[i] = 0;

    for (int i = 0; i < num_of_tokens_per_thread; i++) {
        int current_token_id = thread_starting_token + i * WARP_SIZE;
        if (current_token_id >= num_of_total_attn_tokens) break;

        int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * num_of_ranks_per_node);
        int current_token_local_rank =
            (current_token_id % (num_of_tokens_per_rank * num_of_ranks_per_node)) / num_of_tokens_per_rank;
        int current_token_local_id = current_token_id % num_of_tokens_per_rank;
        int rdma_map_id = current_token_node_rank * rdma_to_attn_map_size_per_node + current_token_local_id;

        const uint8_t* bitmap_row =
            input_routing_map + current_token_id * packed_row_bytes + node_rank * experts_per_node_packed;
        rank_mask_t<NUM_MASK_WORDS> rank_mask =
            bitmap_row_to_rank_mask<NUM_MASK_WORDS, LSA_TEAM_SIZE>(bitmap_row, num_of_ranks_per_node, experts_per_rank);

#pragma unroll
        for (int j = 0; j < LSA_TEAM_SIZE; j++) {
            if (j < num_of_ranks_per_node) rank_sums[j] += rank_mask.test(j);
        }
        token_rank_mask[current_token_id] = rank_mask;

        if (current_token_local_rank == local_rank) {
            rdma_to_attn_map[rdma_map_id] = rank_mask.any();
        }
    }
}

template <int LSA_TEAM_SIZE>
__device__ __forceinline__ void
reduce_warp_tally(const int32_t* rank_sums, scan_smem_t& smem, int num_of_ranks_per_node, int warp_id, int lane_id) {
#pragma unroll
    for (int rank = 0; rank < LSA_TEAM_SIZE; rank++) {
        if (rank >= num_of_ranks_per_node) break;
        int32_t rank_sum = __reduce_add_sync(~0u, rank_sums[rank]);
        if (lane_id == 0) {
            smem.warp_rank_sums[warp_id * num_of_ranks_per_node + rank] = rank_sum;
        }
    }
}

template <int NUM_THREADS_PER_BLOCK, int NUM_OF_WARPS_PER_BLOCK>
__device__ __forceinline__ void
cross_block_prefix_scan(scan_smem_t& smem, tmp_state_t* tmp, int num_of_ranks_per_node, int block_id) {
    for (int i = threadIdx.x; i < num_of_ranks_per_node; i += NUM_THREADS_PER_BLOCK) {
        int32_t rank_acc = 0;
#pragma unroll
        for (int j = 0; j < NUM_OF_WARPS_PER_BLOCK; j++) {
            rank_acc += smem.warp_rank_sums[j * num_of_ranks_per_node + i];
        }

        tmp_state_t tmp_data{PRIV_SUM, rank_acc};
        uint64_t data = *reinterpret_cast<uint64_t*>(&tmp_data);
        nccl_ep::st_relaxed_gpu_global(reinterpret_cast<uint64_t*>(&tmp[block_id * num_of_ranks_per_node + i]), data);
    }

    for (int i = threadIdx.x; i < num_of_ranks_per_node; i += NUM_THREADS_PER_BLOCK) {
        int32_t previous_block_sum_for_current_rank = 0;
        for (int j = 0; j < block_id; j++) {
            tmp_state_t tmp_data{EMPTY, 0};
            tmp_state_t* tmp_src = &tmp[j * num_of_ranks_per_node + i];
            do {
                uint64_t data = nccl_ep::ld_relaxed_gpu_global(reinterpret_cast<const uint64_t*>(tmp_src));
                tmp_data = *reinterpret_cast<tmp_state_t*>(&data);
            } while (tmp_data.state != PRIV_SUM);
            previous_block_sum_for_current_rank += tmp_data.value;
        }
        smem.block_prefix[i] = previous_block_sum_for_current_rank;
    }
}

template <int NUM_OF_WARPS_PER_BLOCK>
__device__ __forceinline__ void init_warp_rank_prefixes(scan_smem_t& smem, int num_of_ranks_per_node) {
    if (threadIdx.x < num_of_ranks_per_node) {
        const int rank = threadIdx.x;
        int32_t prefix = smem.block_prefix[rank];
#pragma unroll
        for (int warp = 0; warp < NUM_OF_WARPS_PER_BLOCK; warp++) {
            smem.warp_prefix[warp * num_of_ranks_per_node + rank] = prefix;
            prefix += smem.warp_rank_sums[warp * num_of_ranks_per_node + rank];
        }
    }
}

__device__ __forceinline__ int32_t warp_excl_scan(bool participates, int lane_id, int32_t& tile_sum_out) {
    int32_t temp_scan = participates ? 1 : 0;
#pragma unroll
    for (int k = 1; k < WARP_SIZE; k *= 2) {
        int32_t temp = __shfl_up_sync(~0u, temp_scan, k);
        if (lane_id >= k) temp_scan += temp;
    }

    tile_sum_out = __shfl_sync(~0u, temp_scan, WARP_SIZE - 1);
    int32_t exclusive_scan = __shfl_up_sync(~0u, temp_scan, 1);
    return (lane_id >= 1) ? exclusive_scan : 0;
}

template <bool ENABLE_PER_EXPERT_COUNTS>
__device__ __forceinline__ void write_local_routing(
    const uint8_t* input_routing_map,
    bool* local_expert_routing_map,
    int32_t* block_expert_token_counts,
    int current_token_id,
    int token_out_of_bound,
    bool token_needed_by_local_rank,
    int32_t local_rank_slot,
    int packed_row_bytes,
    int experts_per_node_packed,
    int node_rank,
    int local_rank,
    int experts_per_rank,
    int lane_id,
    bool expert_major) {
    bool lane_participates = (token_out_of_bound == 0) && token_needed_by_local_rank;
    const uint8_t* local_rank_bitmap_row = nullptr;
    bool* local_expert_routing_map_store_base_addr = nullptr;

    if (lane_participates) {
        local_rank_bitmap_row =
            input_routing_map + current_token_id * packed_row_bytes + node_rank * experts_per_node_packed;
        if (!expert_major) {
            local_expert_routing_map_store_base_addr = local_expert_routing_map + local_rank_slot * experts_per_rank;
        }
    }

    const int local_expert_bit_base = local_rank * experts_per_rank;
    for (int k = 0; k < experts_per_rank; k++) {
        int expert_bit = local_expert_bit_base + k;
        bool routed_to_expert = false;
        if (lane_participates) {
            routed_to_expert = ((local_rank_bitmap_row[expert_bit / 8] >> (expert_bit % 8)) & 1u) != 0;
            if (!expert_major) {
                local_expert_routing_map_store_base_addr[k] = routed_to_expert;
            }
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

template <int NUM_RANK_TILES, int LSA_TEAM_SIZE, int NUM_MASK_WORDS, bool ENABLE_PER_EXPERT_COUNTS>
__device__ __forceinline__ void assign_recv_slots(
    const uint8_t* input_routing_map,
    int32_t* sparse_to_dense_map,
    rank_mask_t<NUM_MASK_WORDS>* token_rank_mask,
    bool* local_expert_routing_map,
    int32_t* block_expert_token_counts,
    int32_t* num_of_tokens_for_experts,
    void* recv_total_counter,
    scan_smem_t& smem,
    int thread_starting_token,
    int num_of_tokens_per_thread,
    int num_of_total_attn_tokens,
    int num_of_tokens_per_rank,
    int num_of_ranks_per_node,
    int experts_per_rank,
    int packed_row_bytes,
    int experts_per_node_packed,
    int node_rank,
    int local_rank,
    int warp_id,
    int lane_id,
    bool expert_major,
    bool out_is_int64,
    int max_recv_tokens_per_rank,
    int32_t* token_to_recv_slot = nullptr) {
    for (int i = 0; i < num_of_tokens_per_thread; i++) {
        int current_token_id = thread_starting_token + i * WARP_SIZE;
        int token_out_of_bound = 0;
        if (current_token_id >= num_of_total_attn_tokens) token_out_of_bound = 1;
        if (__all_sync(~0u, token_out_of_bound) != 0) break;

        int current_token_node_rank = current_token_id / (num_of_tokens_per_rank * num_of_ranks_per_node);
        int current_token_local_rank =
            (current_token_id % (num_of_tokens_per_rank * num_of_ranks_per_node)) / num_of_tokens_per_rank;
        int current_token_local_id = current_token_id % num_of_tokens_per_rank;

        rank_mask_t<NUM_MASK_WORDS> rank_mask;
        rank_mask.clear();
        if (token_out_of_bound == 0) rank_mask = token_rank_mask[current_token_id];

        bool token_needed_by_local_rank = false;
        int32_t local_rank_slot = -1;
        int32_t local_rank_prefix_after_scan = 0;

#pragma unroll NUM_RANK_TILES
        for (int tile = 0; tile < NUM_RANK_TILES; tile++) {
            const int rank_base = tile * 32;
            const int tile_width = (LSA_TEAM_SIZE - rank_base < 32) ? (LSA_TEAM_SIZE - rank_base) : 32;

            int32_t previous_token_sum[32];
#pragma unroll
            for (int j = 0; j < tile_width; j++) {
                previous_token_sum[j] = smem.warp_prefix[warp_id * num_of_ranks_per_node + rank_base + j];
            }

#pragma unroll
            for (int j = 0; j < tile_width; j++) {
                const int rank = rank_base + j;
                bool token_needed_by_this_rank = (rank < num_of_ranks_per_node) && rank_mask.test(rank);
                int32_t temp_sum = 0;
                int32_t temp_scan =
                    warp_excl_scan(token_out_of_bound == 0 && token_needed_by_this_rank, lane_id, temp_sum);
                int32_t final_ex_scan = token_needed_by_this_rank ? previous_token_sum[j] + temp_scan : -1;
                previous_token_sum[j] += temp_sum;

                if (!expert_major && token_out_of_bound == 0 && current_token_local_rank == local_rank &&
                    rank < num_of_ranks_per_node) {
                    sparse_to_dense_map
                        [(current_token_node_rank * num_of_tokens_per_rank + current_token_local_id) *
                             num_of_ranks_per_node +
                         rank] = final_ex_scan;
                }

                if (rank == local_rank) {
                    token_needed_by_local_rank = token_needed_by_this_rank;
                    local_rank_slot = final_ex_scan;
                    local_rank_prefix_after_scan = previous_token_sum[j];
                }
            }

            if (lane_id == 0) {
#pragma unroll
                for (int j = 0; j < tile_width; j++) {
                    smem.warp_prefix[warp_id * num_of_ranks_per_node + rank_base + j] = previous_token_sum[j];
                }
            }
            __syncwarp();
        }

        write_local_routing<ENABLE_PER_EXPERT_COUNTS>(
            input_routing_map,
            local_expert_routing_map,
            block_expert_token_counts,
            current_token_id,
            token_out_of_bound,
            token_needed_by_local_rank,
            local_rank_slot,
            packed_row_bytes,
            experts_per_node_packed,
            node_rank,
            local_rank,
            experts_per_rank,
            lane_id,
            expert_major);

    // em-permute: persist per-global-token recv slot (-1 = no local-rank hit).
        if (token_to_recv_slot != nullptr && token_out_of_bound == 0) {
            token_to_recv_slot[current_token_id] = token_needed_by_local_rank ? local_rank_slot : -1;
        }

        if (!expert_major && current_token_id == num_of_total_attn_tokens - 1) {
            if (local_rank_prefix_after_scan > max_recv_tokens_per_rank) {
                printf(
                    "ncclEpUpdateHandle: HT FLAT actual recv tokens %d > "
                    "max_recv_tokens_per_rank %d on (node %d local %d); "
                    "increase ncclEpGroupConfig_t::max_recv_tokens_per_rank\n",
                    local_rank_prefix_after_scan,
                    max_recv_tokens_per_rank,
                    node_rank,
                    local_rank);
                __trap();
            }
            *num_of_tokens_for_experts = local_rank_prefix_after_scan;
            if (recv_total_counter) {
                if (out_is_int64) {
                    *static_cast<int64_t*>(recv_total_counter) = static_cast<int64_t>(local_rank_prefix_after_scan);
                } else {
                    *static_cast<int32_t*>(recv_total_counter) = local_rank_prefix_after_scan;
                }
            }
        }
    }
}

template <int NUM_LSA_TEAMS, int NUM_THREADS_PER_BLOCK, int NUM_OF_BLOCKS>
__device__ __forceinline__ void fill_attn_to_rdma(
    const uint8_t* input_routing_map,
    bool* attn_to_rdma_map,
    int num_of_tokens_per_rank,
    int num_of_ranks_per_node,
    int experts_per_rank,
    int node_rank,
    int local_rank,
    int packed_row_bytes,
    int experts_per_node_packed) {
    if constexpr (NUM_LSA_TEAMS == 1) return;

    constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;
    const int num_of_total_token_rows = (NUM_LSA_TEAMS - 1) * num_of_tokens_per_rank;
    const int num_of_token_rows_per_thread = ((num_of_total_token_rows - 1) / NUM_OF_TOTAL_THREADS) + 1;
    int tid = threadIdx.x + blockIdx.x * NUM_THREADS_PER_BLOCK;
    const int experts_per_node = experts_per_rank * num_of_ranks_per_node;

    for (int i = 0; i < num_of_token_rows_per_thread; i++) {
        int current_token_id = i * NUM_OF_TOTAL_THREADS + tid;
        if (current_token_id >= num_of_total_token_rows) break;

        int attn_node_id = current_token_id % (NUM_LSA_TEAMS - 1);
        int current_token_node_id = attn_node_id < node_rank ? attn_node_id : attn_node_id + 1;
        int current_token_local_id = current_token_id / (NUM_LSA_TEAMS - 1);

        const uint8_t* bitmap_row =
            input_routing_map +
            ((node_rank * num_of_ranks_per_node + local_rank) * num_of_tokens_per_rank + current_token_local_id) *
                packed_row_bytes +
            current_token_node_id * experts_per_node_packed;

        bool* attn_to_rdma_map_base_addr =
            attn_to_rdma_map + (current_token_local_id * (NUM_LSA_TEAMS - 1) + attn_node_id);
        *attn_to_rdma_map_base_addr = bitmap_range_has_set_bit(bitmap_row, 0, experts_per_node);
    }
}

// Shared geometry used by both FLAT and EM scan entries.
struct scan_geometry_t {
    int num_of_total_attn_tokens;
    int num_of_tokens_per_thread;
    int num_of_tokens_per_warp;
    int num_of_tokens_per_block;
    int rdma_to_attn_map_size_per_node;
    int experts_per_node_packed;
    int packed_row_bytes;
    int thread_starting_token;
    int warp_id;
    int lane_id;
};

template <int NUM_THREADS_PER_BLOCK, int NUM_OF_BLOCKS, int NUM_LSA_TEAMS>
__device__ __forceinline__ scan_geometry_t
compute_scan_geometry(int num_of_tokens_per_rank, int num_of_ranks_per_node, int experts_per_rank) {
    constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;
    constexpr int NUM_OF_TOTAL_THREADS = NUM_THREADS_PER_BLOCK * NUM_OF_BLOCKS;

    scan_geometry_t g;
    g.num_of_total_attn_tokens = num_of_tokens_per_rank * num_of_ranks_per_node * NUM_LSA_TEAMS;
    g.num_of_tokens_per_thread = ((g.num_of_total_attn_tokens - 1) / NUM_OF_TOTAL_THREADS) + 1;
    g.num_of_tokens_per_warp = g.num_of_tokens_per_thread * WARP_SIZE;
    g.num_of_tokens_per_block = g.num_of_tokens_per_warp * NUM_OF_WARPS_PER_BLOCK;
    g.rdma_to_attn_map_size_per_node = (((num_of_tokens_per_rank - 1) / 16) + 1) * 16;

    const int experts_per_node = experts_per_rank * num_of_ranks_per_node;
    g.experts_per_node_packed = (experts_per_node + 7) / 8;
    g.packed_row_bytes = g.experts_per_node_packed * NUM_LSA_TEAMS;

    const int block_starting_token = blockIdx.x * g.num_of_tokens_per_block;
    g.warp_id = threadIdx.x / WARP_SIZE;
    g.lane_id = threadIdx.x % WARP_SIZE;
    const int warp_starting_token = block_starting_token + g.warp_id * g.num_of_tokens_per_warp;
    g.thread_starting_token = warp_starting_token + g.lane_id;
    return g;
}

// FLAT-layout scan: produces token_rank_mask, rdma/attn maps, sparse_to_dense
// (rank-major), recv-slot prefixes, and (optionally) per-expert token counts.
template <
    int NUM_THREADS_PER_BLOCK,
    int NUM_OF_BLOCKS,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    bool ENABLE_PER_EXPERT_COUNTS>
__device__ __forceinline__ void scan_impl_flat(
    const uint8_t* input_routing_map,
    tmp_state_t* tmp,
    int32_t* sparse_to_dense_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    rank_mask_t<(LSA_TEAM_SIZE + 63) / 64>* token_rank_mask,
    int32_t* num_of_tokens_for_experts,
    bool* local_expert_routing_map,
    int32_t* per_expert_token_counts,
    const int node_rank,
    const int local_rank,
    const int num_of_tokens_per_rank,
    const int num_of_ranks_per_node,
    const int experts_per_rank,
    void* recv_total_counter,
    bool out_is_int64,
    int max_recv_tokens_per_rank,
    int32_t* token_to_recv_slot,
    uint8_t* smem_bytes) {
    static_assert(
        LSA_TEAM_SIZE <= EM_S2D_MAX_RANKS,
        "em_s2d_pack rank field is 10 bits; LSA team size must fit in 1024");
    static_assert(
        LSA_TEAM_SIZE <= NUM_THREADS_PER_BLOCK,
        "scan_impl_flat requires one block thread per LSA rank to initialize warp prefixes");

    constexpr int NUM_OF_WARPS_PER_BLOCK = NUM_THREADS_PER_BLOCK / WARP_SIZE;
    constexpr int NUM_MASK_WORDS = (LSA_TEAM_SIZE + 63) / 64;
    constexpr int NUM_RANK_TILES = (LSA_TEAM_SIZE + 31) / 32;

    const scan_geometry_t g = compute_scan_geometry<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_LSA_TEAMS>(
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank);

    scan_smem_t smem =
        scan_smem_t::from_raw(smem_bytes, NUM_OF_WARPS_PER_BLOCK, num_of_ranks_per_node, ENABLE_PER_EXPERT_COUNTS);

    if constexpr (ENABLE_PER_EXPERT_COUNTS) {
        for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
            smem.expert_counts[e] = 0;
        }
        __syncthreads();
    }

    int32_t token_routing_map_sum[LSA_TEAM_SIZE];
    tally_ranks<LSA_TEAM_SIZE, NUM_MASK_WORDS>(
        input_routing_map,
        token_rank_mask,
        rdma_to_attn_map,
        token_routing_map_sum,
        g.thread_starting_token,
        g.num_of_tokens_per_thread,
        g.num_of_total_attn_tokens,
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank,
        g.packed_row_bytes,
        g.experts_per_node_packed,
        node_rank,
        local_rank,
        g.rdma_to_attn_map_size_per_node);

    reduce_warp_tally<LSA_TEAM_SIZE>(token_routing_map_sum, smem, num_of_ranks_per_node, g.warp_id, g.lane_id);
    __syncthreads();

    cross_block_prefix_scan<NUM_THREADS_PER_BLOCK, NUM_OF_WARPS_PER_BLOCK>(
        smem,
        tmp,
        num_of_ranks_per_node,
        blockIdx.x);
    __syncthreads();

    init_warp_rank_prefixes<NUM_OF_WARPS_PER_BLOCK>(smem, num_of_ranks_per_node);
    __syncthreads();

    assign_recv_slots<NUM_RANK_TILES, LSA_TEAM_SIZE, NUM_MASK_WORDS, ENABLE_PER_EXPERT_COUNTS>(
        input_routing_map,
        sparse_to_dense_map,
        token_rank_mask,
        local_expert_routing_map,
        smem.expert_counts,
        num_of_tokens_for_experts,
        recv_total_counter,
        smem,
        g.thread_starting_token,
        g.num_of_tokens_per_thread,
        g.num_of_total_attn_tokens,
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank,
        g.packed_row_bytes,
        g.experts_per_node_packed,
        node_rank,
        local_rank,
        g.warp_id,
        g.lane_id,
        /*expert_major=*/false,
        out_is_int64,
        max_recv_tokens_per_rank,
        token_to_recv_slot);

    if constexpr (ENABLE_PER_EXPERT_COUNTS) {
        __syncthreads();
        for (int e = threadIdx.x; e < experts_per_rank; e += NUM_THREADS_PER_BLOCK) {
            int32_t block_count = smem.expert_counts[e];
            if (block_count > 0) atomicAdd(per_expert_token_counts + e, block_count);
        }
    }

    fill_attn_to_rdma<NUM_LSA_TEAMS, NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS>(
        input_routing_map,
        attn_to_rdma_map,
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank,
        node_rank,
        local_rank,
        g.packed_row_bytes,
        g.experts_per_node_packed);
}

// EM-layout pre-scan: produces only what em_scan_kernel (see hybridep_adapter.cu)
// and the rest of the EM pipeline still need from the global routing bitmap --
// the per-token rank mask, rdma_to_attn_map (filled by tally_ranks), and
// attn_to_rdma_map. The FLAT-only prefix scan / sparse_to_dense / per-expert
// counts are skipped entirely.
template <int NUM_THREADS_PER_BLOCK, int NUM_OF_BLOCKS, int NUM_LSA_TEAMS, int LSA_TEAM_SIZE>
__device__ __forceinline__ void scan_impl_em(
    const uint8_t* input_routing_map,
    bool* rdma_to_attn_map,
    bool* attn_to_rdma_map,
    rank_mask_t<(LSA_TEAM_SIZE + 63) / 64>* token_rank_mask,
    const int node_rank,
    const int local_rank,
    const int num_of_tokens_per_rank,
    const int num_of_ranks_per_node,
    const int experts_per_rank) {
    static_assert(
        LSA_TEAM_SIZE <= EM_S2D_MAX_RANKS,
        "em_s2d_pack rank field is 10 bits; LSA team size must fit in 1024");

    constexpr int NUM_MASK_WORDS = (LSA_TEAM_SIZE + 63) / 64;

    const scan_geometry_t g = compute_scan_geometry<NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS, NUM_LSA_TEAMS>(
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank);

    int32_t token_routing_map_sum[LSA_TEAM_SIZE];
    tally_ranks<LSA_TEAM_SIZE, NUM_MASK_WORDS>(
        input_routing_map,
        token_rank_mask,
        rdma_to_attn_map,
        token_routing_map_sum,
        g.thread_starting_token,
        g.num_of_tokens_per_thread,
        g.num_of_total_attn_tokens,
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank,
        g.packed_row_bytes,
        g.experts_per_node_packed,
        node_rank,
        local_rank,
        g.rdma_to_attn_map_size_per_node);
    (void)token_routing_map_sum;

    fill_attn_to_rdma<NUM_LSA_TEAMS, NUM_THREADS_PER_BLOCK, NUM_OF_BLOCKS>(
        input_routing_map,
        attn_to_rdma_map,
        num_of_tokens_per_rank,
        num_of_ranks_per_node,
        experts_per_rank,
        node_rank,
        local_rank,
        g.packed_row_bytes,
        g.experts_per_node_packed);
}

// Parameter packs for the scan JIT entries. Non-templated so the host can build
// them once without knowing LSA_TEAM_SIZE at compile time; the JIT-emitted
// wrappers reinterpret_cast token_rank_mask to rank_mask_t<(LSA_TEAM_SIZE+63)/64>*
// before calling the corresponding scan_impl.
struct scan_flat_kernel_param_t {
    const uint8_t* input_routing_map;
    tmp_state_t* tmp;
    int32_t* sparse_to_dense_map;
    bool* rdma_to_attn_map;
    bool* attn_to_rdma_map;
    void* token_rank_mask;
    int32_t* num_of_tokens_for_experts;
    bool* local_expert_routing_map;
    int32_t* per_expert_token_counts;
    int node_rank;
    int local_rank;
    int num_of_tokens_per_rank;
    int num_of_ranks_per_node;
    int experts_per_rank;
    void* recv_total_counter;
    bool out_is_int64;
    int max_recv_tokens_per_rank;
    int32_t* token_to_recv_slot;  // em-permute scratch; null otherwise.
};

struct scan_em_kernel_param_t {
    const uint8_t* input_routing_map;
    bool* rdma_to_attn_map;
    bool* attn_to_rdma_map;
    void* token_rank_mask;
    int node_rank;
    int local_rank;
    int num_of_tokens_per_rank;
    int num_of_ranks_per_node;
    int experts_per_rank;
};

} // namespace hybrid_ep
