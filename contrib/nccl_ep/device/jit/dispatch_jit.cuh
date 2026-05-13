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

constexpr const char* kDispatchJitEntryName = "nccl_ep_jit_ht_dispatch_kernel";

inline const char* dispatch_bool_literal(bool value) { return value ? "true" : "false"; }

inline const char* dispatch_token_data_type_literal(bool use_fp8) {
    return use_fp8 ? "uint8_t" : "uint16_t";
}

template <
    int INTER_NODE_GROUP_WARPS,
    int INTER_NODE_GROUP_START,
    int INTRA_NODE_G2S_GROUP_WARPS,
    int INTRA_NODE_G2S_GROUP_START,
    int INTRA_NODE_S2G_GROUP_WARPS,
    int INTRA_NODE_S2G_GROUP_START,
    int PAD_GROUP_WARPS,
    int PAD_GROUP_START,
    int NUM_OF_STAGES,
    int NUM_OF_IN_FLIGHT_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_LSA_TEAMS,
    int NUM_OF_BLOCKS,
    bool FORWARD_DISPATCH,
    int NUM_PIPELINES,
    int LSA_TEAM_SIZE,
    ncclEpLayout_t kLayout>
std::string dispatch_jit_source(bool use_fp8, int hidden_dim) {
    const char* layout_literal =
        (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? "NCCL_EP_LAYOUT_EXPERT_MAJOR"
            : "NCCL_EP_LAYOUT_FLAT";
    const char* token_type_literal = dispatch_token_data_type_literal(use_fp8);
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "using TOKEN_DATA_TYPE = " << token_type_literal << ";\n"
        << "using INTER_NODE_GROUP     = hybrid_ep::warp_group<" << INTER_NODE_GROUP_WARPS    << ", " << INTER_NODE_GROUP_START    << ">;\n"
        << "using INTRA_NODE_G2S_GROUP = hybrid_ep::warp_group<" << INTRA_NODE_G2S_GROUP_WARPS << ", " << INTRA_NODE_G2S_GROUP_START << ">;\n"
        << "using INTRA_NODE_S2G_GROUP = hybrid_ep::warp_group<" << INTRA_NODE_S2G_GROUP_WARPS << ", " << INTRA_NODE_S2G_GROUP_START << ">;\n"
        << "using PAD_GROUP            = hybrid_ep::warp_group<" << PAD_GROUP_WARPS            << ", " << PAD_GROUP_START            << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size() + PAD_GROUP::size(), 1)\n"
        << "__global__ void " << kDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, " << LSA_TEAM_SIZE << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::dispatch_kernel_impl<\n"
        << "      TOKEN_DATA_TYPE,\n"
        << "      INTER_NODE_GROUP,\n"
        << "      INTRA_NODE_G2S_GROUP,\n"
        << "      INTRA_NODE_S2G_GROUP,\n"
        << "      PAD_GROUP,\n"
        << "      " << NUM_OF_STAGES << ",\n"
        << "      " << NUM_OF_IN_FLIGHT_S2G << ",\n"
        << "      " << NUM_OF_TOKENS_PER_CHUNK << ",\n"
        << "      " << MAX_SUPPORTED_TOKENS_PER_RANK << ",\n"
        << "      " << NUM_LSA_TEAMS << ",\n"
        << "      " << NUM_OF_BLOCKS << ",\n"
        << "      " << dispatch_bool_literal(FORWARD_DISPATCH) << ",\n"
        << "      " << NUM_PIPELINES << ",\n"
        << "      " << LSA_TEAM_SIZE << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << hidden_dim << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

template <
    int NUM_OF_STAGES,
    int NUM_OF_IN_FLIGHT_S2G,
    int NUM_OF_TOKENS_PER_CHUNK,
    int NUM_OF_BLOCKS,
    bool FORWARD_DISPATCH,
    int NUM_LSA_TEAMS,
    int LSA_TEAM_SIZE,
    ncclEpLayout_t kLayout,
    typename TOKEN_DATA_TYPE>
void launch_dispatch(
    ::hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, LSA_TEAM_SIZE>& param,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    constexpr bool multinode_layout = (NUM_LSA_TEAMS != 1);
    constexpr int NUM_PIPELINES = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
    constexpr int INTER_NODE_GROUP_WARPS    = multinode_layout ? HYBRIDEP_DISPATCH_N2N_WARPS : 0;
    constexpr int INTER_NODE_GROUP_START    = 0;
    constexpr int INTRA_NODE_G2S_GROUP_WARPS = NUM_PIPELINES;
    constexpr int INTRA_NODE_G2S_GROUP_START = multinode_layout ? HYBRIDEP_DISPATCH_N2N_WARPS : 0;
    constexpr int INTRA_NODE_S2G_GROUP_WARPS = NUM_PIPELINES;
    constexpr int INTRA_NODE_S2G_GROUP_START = multinode_layout
        ? (HYBRIDEP_DISPATCH_N2N_WARPS + NUM_PIPELINES)
        : NUM_PIPELINES;
    constexpr int PAD_GROUP_WARPS = (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? 1 : 0;
    constexpr int PAD_GROUP_START = INTRA_NODE_S2G_GROUP_START + INTRA_NODE_S2G_GROUP_WARPS;
    constexpr int BLOCK_DIM = 32 * (
        INTER_NODE_GROUP_WARPS +
        INTRA_NODE_G2S_GROUP_WARPS +
        INTRA_NODE_S2G_GROUP_WARPS +
        PAD_GROUP_WARPS);
    constexpr bool USE_FP8 = std::is_same_v<TOKEN_DATA_TYPE, uint8_t>;

    const int hidden_dim = param.hidden_dim;
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name
            << "dispatch"
            << "_nodes" << NUM_LSA_TEAMS
            << "_lsa" << LSA_TEAM_SIZE
            << "_hdim" << hidden_dim
            << "_stages" << NUM_OF_STAGES
            << "_inflt" << NUM_OF_IN_FLIGHT_S2G
            << "_chunk" << NUM_OF_TOKENS_PER_CHUNK
            << "_blocks" << NUM_OF_BLOCKS
            << (FORWARD_DISPATCH ? "_fwd" : "_bwd")
            << (kLayout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "_em" : "_fl")
            << (USE_FP8 ? "_fp8" : "_bf16");
        return name.str();
    }();
    const std::string source = dispatch_jit_source<
        INTER_NODE_GROUP_WARPS,
        INTER_NODE_GROUP_START,
        INTRA_NODE_G2S_GROUP_WARPS,
        INTRA_NODE_G2S_GROUP_START,
        INTRA_NODE_S2G_GROUP_WARPS,
        INTRA_NODE_S2G_GROUP_START,
        PAD_GROUP_WARPS,
        PAD_GROUP_START,
        NUM_OF_STAGES,
        NUM_OF_IN_FLIGHT_S2G,
        NUM_OF_TOKENS_PER_CHUNK,
        NUM_LSA_TEAMS,
        NUM_OF_BLOCKS,
        FORWARD_DISPATCH,
        NUM_PIPELINES,
        LSA_TEAM_SIZE,
        kLayout>(USE_FP8, hidden_dim);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_dispatch";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kDispatchJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(hidden_dim);
    variant.num_blocks = NUM_OF_BLOCKS;
    variant.block_dim = BLOCK_DIM;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    using warp_timing_entry_t = typename ::hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, LSA_TEAM_SIZE>::warp_timing_entry_t;
    constexpr int WT_WARPS_PER_BLOCK = BLOCK_DIM / 32;
    constexpr int WT_TOTAL = NUM_OF_BLOCKS * WT_WARPS_PER_BLOCK;
    warp_timing_entry_t* d_wt = nullptr;
    CUDA_CHECK(cudaMalloc(&d_wt, WT_TOTAL * sizeof(warp_timing_entry_t)));
    CUDA_CHECK(cudaMemsetAsync(d_wt, 0, WT_TOTAL * sizeof(warp_timing_entry_t), stream));
    param.warp_timing = d_wt;
#endif

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, &param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
#ifdef HYBRIDEP_ENABLE_WARP_TIMING
        CUDA_CHECK(cudaFree(d_wt));
#endif
        std::fprintf(
            stderr,
            "[nccl_ep jit] fatal dispatch JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }

#ifdef HYBRIDEP_ENABLE_WARP_TIMING
    CUDA_CHECK(cudaFree(d_wt));
#endif
}

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
