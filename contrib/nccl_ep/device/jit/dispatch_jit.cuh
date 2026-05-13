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

inline std::string dispatch_jit_source(
    int inter_node_group_warps,
    int inter_node_group_start,
    int intra_node_g2s_group_warps,
    int intra_node_g2s_group_start,
    int intra_node_s2g_group_warps,
    int intra_node_s2g_group_start,
    int pad_group_warps,
    int pad_group_start,
    int num_of_stages,
    int num_of_in_flight_s2g,
    int num_of_tokens_per_chunk,
    int num_lsa_teams,
    int num_of_blocks,
    bool forward_dispatch,
    int num_pipelines,
    int lsa_team_size,
    ncclEpLayout_t layout,
    bool use_fp8,
    int hidden_dim) {
    const char* layout_literal =
        (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR)
            ? "NCCL_EP_LAYOUT_EXPERT_MAJOR"
            : "NCCL_EP_LAYOUT_FLAT";
    const char* token_type_literal = dispatch_token_data_type_literal(use_fp8);
    std::ostringstream src;
    src
        << "#include \"device/hybrid_ep.cuh\"\n"
        << "\n"
        << "using TOKEN_DATA_TYPE = " << token_type_literal << ";\n"
        << "using INTER_NODE_GROUP     = hybrid_ep::warp_group<" << inter_node_group_warps    << ", " << inter_node_group_start    << ">;\n"
        << "using INTRA_NODE_G2S_GROUP = hybrid_ep::warp_group<" << intra_node_g2s_group_warps << ", " << intra_node_g2s_group_start << ">;\n"
        << "using INTRA_NODE_S2G_GROUP = hybrid_ep::warp_group<" << intra_node_s2g_group_warps << ", " << intra_node_s2g_group_start << ">;\n"
        << "using PAD_GROUP            = hybrid_ep::warp_group<" << pad_group_warps            << ", " << pad_group_start            << ">;\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(INTER_NODE_GROUP::size() + INTRA_NODE_G2S_GROUP::size() + INTRA_NODE_S2G_GROUP::size() + PAD_GROUP::size(), 1)\n"
        << "__global__ void " << kDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ hybrid_ep::dispatch_kernel_param_t<TOKEN_DATA_TYPE, " << lsa_team_size << "> param) {\n"
        << "  extern __shared__ uint8_t smem_bytes[];\n"
        << "  hybrid_ep::dispatch_kernel_impl<\n"
        << "      TOKEN_DATA_TYPE,\n"
        << "      INTER_NODE_GROUP,\n"
        << "      INTRA_NODE_G2S_GROUP,\n"
        << "      INTRA_NODE_S2G_GROUP,\n"
        << "      PAD_GROUP,\n"
        << "      " << num_of_stages << ",\n"
        << "      " << num_of_in_flight_s2g << ",\n"
        << "      " << num_of_tokens_per_chunk << ",\n"
        << "      " << MAX_SUPPORTED_TOKENS_PER_RANK << ",\n"
        << "      " << num_lsa_teams << ",\n"
        << "      " << num_of_blocks << ",\n"
        << "      " << dispatch_bool_literal(forward_dispatch) << ",\n"
        << "      " << num_pipelines << ",\n"
        << "      " << lsa_team_size << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << hidden_dim << ">(param, smem_bytes);\n"
        << "}\n";
    return src.str();
}

inline void launch_dispatch(
    int num_of_stages,
    int num_of_in_flight_s2g,
    int num_of_tokens_per_chunk,
    int num_of_blocks,
    bool forward_dispatch,
    int num_lsa_teams,
    int lsa_team_size,
    ncclEpLayout_t layout,
    bool use_fp8,
    int hidden_dim,
    void* param,
    int dynamic_smem_bytes,
    cudaStream_t stream) {
    const bool multinode_layout = (num_lsa_teams != 1);
    const int num_pipelines = HYBRIDEP_DISPATCH_NUM_OF_PIPELINES_PER_BLOCK;
    const int inter_node_group_warps    = multinode_layout ? HYBRIDEP_DISPATCH_N2N_WARPS : 0;
    const int inter_node_group_start    = 0;
    const int intra_node_g2s_group_warps = num_pipelines;
    const int intra_node_g2s_group_start = multinode_layout ? HYBRIDEP_DISPATCH_N2N_WARPS : 0;
    const int intra_node_s2g_group_warps = num_pipelines;
    const int intra_node_s2g_group_start = multinode_layout
        ? (HYBRIDEP_DISPATCH_N2N_WARPS + num_pipelines)
        : num_pipelines;
    const int pad_group_warps = (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? 1 : 0;
    const int pad_group_start = intra_node_s2g_group_start + intra_node_s2g_group_warps;
    const int block_dim = 32 * (
        inter_node_group_warps +
        intra_node_g2s_group_warps +
        intra_node_s2g_group_warps +
        pad_group_warps);

    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name
            << "dispatch"
            << "_nodes" << num_lsa_teams
            << "_lsa" << lsa_team_size
            << "_hdim" << hidden_dim
            << "_stages" << num_of_stages
            << "_inflt" << num_of_in_flight_s2g
            << "_chunk" << num_of_tokens_per_chunk
            << "_blocks" << num_of_blocks
            << (forward_dispatch ? "_fwd" : "_bwd")
            << (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "_em" : "_fl")
            << (use_fp8 ? "_fp8" : "_bf16");
        return name.str();
    }();
    const std::string source = dispatch_jit_source(
        inter_node_group_warps,
        inter_node_group_start,
        intra_node_g2s_group_warps,
        intra_node_g2s_group_start,
        intra_node_s2g_group_warps,
        intra_node_s2g_group_start,
        pad_group_warps,
        pad_group_start,
        num_of_stages,
        num_of_in_flight_s2g,
        num_of_tokens_per_chunk,
        num_lsa_teams,
        num_of_blocks,
        forward_dispatch,
        num_pipelines,
        lsa_team_size,
        layout,
        use_fp8,
        hidden_dim);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ht_dispatch";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kDispatchJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(hidden_dim);
    variant.num_blocks = num_of_blocks;
    variant.block_dim = block_dim;
    variant.dynamic_smem_bytes = dynamic_smem_bytes;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, param, stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(
            stderr,
            "[nccl_ep jit] fatal dispatch JIT launch failure for %s: %s%s%s\n",
            variant_name.c_str(),
            ::nccl_ep::jit::jit_kernel_status_name(status),
            error.empty() ? "" : ": ",
            error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace hybridep
} // namespace nccl_ep
