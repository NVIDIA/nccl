/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/hybrid_ep.cuh"
#include "device/jit/jit_runtime.hpp"

#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace hybridep {
namespace jit {

constexpr const char* kCombineJitEntryName = "nccl_ep_jit_ht_combine_kernel";

inline const char* bool_literal(bool value) {
    return value ? "true" : "false";
}

template <
    int INTRA_NODE_RED_GROUP_WARPS,
    int INTRA_NODE_RED_GROUP_START,
    int INTER_NODE_RED_GROUP_WARPS,
    int INTER_NODE_RED_GROUP_START,
    int INTRA_NODE_G2S_GROUP_WARPS,
    int INTRA_NODE_G2S_GROUP_START,
    int INTER_NODE_G2S_GROUP_WARPS,
    int INTER_NODE_G2S_GROUP_START,
    int INTER_NODE_RDMA_GROUP_WARPS,
    int INTER_NODE_RDMA_GROUP_START,
    int NUM_OF_DATA_PIPELINE_PER_BLOCK,
    int NUM_OF_STAGES_G2S,
    int NUM_OF_STAGES_S2G,
    int NUM_OF_TOKENS_PER_GROUP,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
    bool BACKWARD_COMBINE,
    int LSA_TEAM_SIZE,
    ncclEpLayout_t kLayout>
std::string combine_jit_source(int hidden_dim) {
    const char* layout_literal =
        (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? "NCCL_EP_LAYOUT_EXPERT_MAJOR"
            : "NCCL_EP_LAYOUT_FLAT";
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "using INTRA_NODE_RED_GROUP = hybrid_ep::warp_group<" << INTRA_NODE_RED_GROUP_WARPS << ", " << INTRA_NODE_RED_GROUP_START << ">;\n"
        << "using INTER_NODE_RED_GROUP = hybrid_ep::warp_group<" << INTER_NODE_RED_GROUP_WARPS << ", " << INTER_NODE_RED_GROUP_START << ">;\n"
        << "using INTRA_NODE_G2S_GROUP = hybrid_ep::warp_group<" << INTRA_NODE_G2S_GROUP_WARPS << ", " << INTRA_NODE_G2S_GROUP_START << ">;\n"
        << "using INTER_NODE_G2S_GROUP = hybrid_ep::warp_group<" << INTER_NODE_G2S_GROUP_WARPS << ", " << INTER_NODE_G2S_GROUP_START << ">;\n"
        << "using INTER_NODE_RDMA_GROUP = hybrid_ep::warp_group<" << INTER_NODE_RDMA_GROUP_WARPS << ", " << INTER_NODE_RDMA_GROUP_START << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(INTRA_NODE_RED_GROUP::size() + INTER_NODE_RED_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTER_NODE_G2S_GROUP::size() + INTER_NODE_RDMA_GROUP::size(), 1)\n"
        << "__global__ void " << kCombineJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::combine_kernel_param_t<" << LSA_TEAM_SIZE << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::combine_kernel_impl<\n"
        << "      INTRA_NODE_RED_GROUP,\n"
        << "      INTER_NODE_RED_GROUP,\n"
        << "      INTRA_NODE_G2S_GROUP,\n"
        << "      INTER_NODE_G2S_GROUP,\n"
        << "      INTER_NODE_RDMA_GROUP,\n"
        << "      " << NUM_OF_DATA_PIPELINE_PER_BLOCK << ",\n"
        << "      " << NUM_OF_STAGES_G2S << ",\n"
        << "      " << NUM_OF_STAGES_S2G << ",\n"
        << "      " << NUM_OF_TOKENS_PER_GROUP << ",\n"
        << "      " << NUM_OF_TOKENS_PER_CHUNK << ",\n"
        << "      " << MAX_SUPPORTED_TOKENS_PER_RANK << ",\n"
        << "      " << NUM_LSA_TEAMS << ",\n"
        << "      " << NUM_OF_BLOCKS << ",\n"
        << "      " << NUM_OF_ADDITIONAL_IN_FLIGHT_S2G << ",\n"
        << "      " << bool_literal(BACKWARD_COMBINE) << ",\n"
        << "      " << hidden_dim << ",\n"
        << "      " << LSA_TEAM_SIZE << ",\n"
        << "      " << layout_literal << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

template <
    int NUM_OF_STAGES_G2S,
    int NUM_OF_STAGES_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_OF_TOKENS_PER_GROUP,
    int NUM_OF_BLOCKS,
    int NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
    bool BACKWARD_COMBINE,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    ncclEpLayout_t kLayout>
void launch_combine(
    ::hybrid_ep::combine_kernel_param_t<LSA_TEAM_SIZE>& param,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    constexpr bool multinode_layout = (NUM_LSA_TEAMS != 1);
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
    constexpr int NUM_OF_DATA_PIPELINE_PER_BLOCK = multinode_layout ? 1 : 2;
    constexpr int BLOCK_DIM = 32 * (
        INTRA_NODE_RED_GROUP_WARPS +
        INTER_NODE_RED_GROUP_WARPS +
        INTRA_NODE_G2S_GROUP_WARPS +
        INTER_NODE_G2S_GROUP_WARPS +
        INTER_NODE_RDMA_GROUP_WARPS);

    const int hidden_dim = param.hidden_dim;
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name
            << "combine"
            << "_nodes" << NUM_LSA_TEAMS
            << "_lsa" << LSA_TEAM_SIZE
            << "_hdim" << hidden_dim
            << "_g2s" << NUM_OF_STAGES_G2S
            << "_s2g" << NUM_OF_STAGES_S2G
            << "_chunk" << NUM_OF_TOKENS_PER_CHUNK
            << "_group" << NUM_OF_TOKENS_PER_GROUP
            << "_blocks" << NUM_OF_BLOCKS
            << "_extra" << NUM_OF_ADDITIONAL_IN_FLIGHT_S2G
            << (BACKWARD_COMBINE ? "_bwd" : "_fwd")
            << (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "_em" : "_fl");
        return name.str();
    }();
    const std::string source = combine_jit_source<
        INTRA_NODE_RED_GROUP_WARPS,
        INTRA_NODE_RED_GROUP_START,
        INTER_NODE_RED_GROUP_WARPS,
        INTER_NODE_RED_GROUP_START,
        INTRA_NODE_G2S_GROUP_WARPS,
        INTRA_NODE_G2S_GROUP_START,
        INTER_NODE_G2S_GROUP_WARPS,
        INTER_NODE_G2S_GROUP_START,
        INTER_NODE_RDMA_GROUP_WARPS,
        INTER_NODE_RDMA_GROUP_START,
        NUM_OF_DATA_PIPELINE_PER_BLOCK,
        NUM_OF_STAGES_G2S,
        NUM_OF_STAGES_S2G,
        NUM_OF_TOKENS_PER_GROUP,
        NUM_OF_TOKENS_PER_CHUNK,
        NUM_LSA_TEAMS,
        NUM_OF_BLOCKS,
        NUM_OF_ADDITIONAL_IN_FLIGHT_S2G,
        BACKWARD_COMBINE,
        LSA_TEAM_SIZE,
        kLayout>(hidden_dim);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_combine";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kCombineJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(hidden_dim);
    variant.num_blocks = NUM_OF_BLOCKS;
    variant.block_dim = BLOCK_DIM;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    using warp_timing_entry_t = typename ::hybrid_ep::combine_kernel_param_t<LSA_TEAM_SIZE>::warp_timing_entry_t;
    using block_timing_entry_t = typename ::hybrid_ep::combine_kernel_param_t<LSA_TEAM_SIZE>::block_timing_entry_t;
    constexpr int WT_WARPS_PER_BLOCK = BLOCK_DIM / 32;
    constexpr int WT_TOTAL = NUM_OF_BLOCKS * WT_WARPS_PER_BLOCK;
    warp_timing_entry_t* d_wt = nullptr;
    block_timing_entry_t* d_bt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wt, WT_TOTAL * sizeof(warp_timing_entry_t)));
    CUDA_CHECK(cudaMalloc(&d_bt, NUM_OF_BLOCKS * sizeof(block_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, WT_TOTAL * sizeof(warp_timing_entry_t), stream));
    CUDA_CHECK(cudaMemsetAsync(d_bt, 0, NUM_OF_BLOCKS * sizeof(block_timing_entry_t), stream));
    param.warp_timing = d_wt;
    param.block_timing = d_bt;
#endif

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        CUDA_CHECK(cudaFree(d_wt));
        CUDA_CHECK(cudaFree(d_bt));
#endif
        std::fprintf(
            stderr,
            "[nccl_ep jit] fatal combine JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    char* pmix_rank_str = std::getenv("PMIX_RANK");
    int pmix_rank = pmix_rank_str ? std::atoi(pmix_rank_str) : -1;
    static int iter_count = 0;
    iter_count++;
    if (pmix_rank == 0 && iter_count == 40) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        warp_timing_entry_t h_wt[WT_TOTAL];
        block_timing_entry_t h_bt[NUM_OF_BLOCKS];
        CUDA_CHECK(cudaMemcpy(h_wt, d_wt, WT_TOTAL * sizeof(warp_timing_entry_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_bt, d_bt, NUM_OF_BLOCKS * sizeof(block_timing_entry_t), cudaMemcpyDeviceToHost));
        int _wt_clock_khz;
        CUDA_CHECK(cudaDeviceGetAttribute(&_wt_clock_khz, cudaDevAttrClockRate, 0));
        auto _wt_us = [&](long long cycles) { return (double)cycles * 1000.0 / _wt_clock_khz; };
        auto _wt_print_head_sync = [&]() {
            long long mn = LLONG_MAX, mx = 0, sum = 0;
            for (int b = 0; b < NUM_OF_BLOCKS; b++) {
                long long d = h_bt[b].head_sync_end_clock - h_bt[b].head_sync_start_clock;
                if (d < mn) mn = d;
                if (d > mx) mx = d;
                sum += d;
            }
            std::printf("[COMBINE HEAD SYNC TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                        NUM_OF_BLOCKS, _wt_us(mn), _wt_us(mx), _wt_us(sum / NUM_OF_BLOCKS));
        };
        auto _wt_print_work_group = [&](const char* name, int warp_start, int warp_count) {
            if (warp_count == 0) return;
            long long mn = LLONG_MAX, mx = 0, sum = 0;
            int n = 0;
            for (int b = 0; b < NUM_OF_BLOCKS; b++) {
                for (int w = warp_start; w < warp_start + warp_count; w++) {
                    long long d = h_wt[b * WT_WARPS_PER_BLOCK + w].work_end_clock -
                                  h_wt[b * WT_WARPS_PER_BLOCK + w].work_start_clock;
                    if (d < mn) mn = d;
                    if (d > mx) mx = d;
                    sum += d;
                    n++;
                }
            }
            std::printf("  %-9s (%d warp%s x %d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                        name, warp_count, warp_count > 1 ? "s" : " ", NUM_OF_BLOCKS,
                        _wt_us(mn), _wt_us(mx), _wt_us(sum / n));
        };
        auto _wt_print_block_span = [&]() {
            long long mn = LLONG_MAX, mx = 0, sum = 0;
            for (int b = 0; b < NUM_OF_BLOCKS; b++) {
                long long blk_start = LLONG_MAX;
                long long blk_end = 0;
                for (int w = 0; w < WT_WARPS_PER_BLOCK; w++) {
                    const auto& e = h_wt[b * WT_WARPS_PER_BLOCK + w];
                    if (e.work_start_clock < blk_start) blk_start = e.work_start_clock;
                    if (e.work_end_clock > blk_end) blk_end = e.work_end_clock;
                }
                long long d = blk_end - blk_start;
                if (d < mn) mn = d;
                if (d > mx) mx = d;
                sum += d;
            }
            std::printf("[COMBINE BLOCK SPAN TIMING] (%d blocks):  min=%8.2f us  max=%8.2f us  avg=%8.2f us\n",
                        NUM_OF_BLOCKS, _wt_us(mn), _wt_us(mx), _wt_us(sum / NUM_OF_BLOCKS));
        };
        _wt_print_head_sync();
        std::printf("[COMBINE WORK WARP TIMING] (%d blocks, %d warps/block, %d pipelines, clock=%d kHz)\n",
                    NUM_OF_BLOCKS, WT_WARPS_PER_BLOCK, NUM_OF_DATA_PIPELINE_PER_BLOCK, _wt_clock_khz);
        _wt_print_work_group("INTRA_RED", INTRA_NODE_RED_GROUP_START, INTRA_NODE_RED_GROUP_WARPS);
        _wt_print_work_group("INTER_RED", INTER_NODE_RED_GROUP_START, INTER_NODE_RED_GROUP_WARPS);
        _wt_print_work_group("INTRA_G2S", INTRA_NODE_G2S_GROUP_START, INTRA_NODE_G2S_GROUP_WARPS);
        _wt_print_work_group("INTER_G2S", INTER_NODE_G2S_GROUP_START, INTER_NODE_G2S_GROUP_WARPS);
        _wt_print_work_group("INTER_N2N", INTER_NODE_RDMA_GROUP_START, INTER_NODE_RDMA_GROUP_WARPS);
        _wt_print_block_span();
    }
    CUDA_CHECK(cudaFree(d_wt));
    CUDA_CHECK(cudaFree(d_bt));
#endif
}

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
