/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include "device/ll_ep_adapter.cuh"
#include "device/jit/jit_runtime.hpp"

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <sstream>
#include <string>

namespace nccl_ep {
namespace internode_ll {
namespace jit {

constexpr const char* kLlDispatchJitEntryName = "nccl_ep_jit_ll_dispatch_kernel";

inline const char* ll_dispatch_bool_literal(bool value) {
    return value ? "true" : "false";
}

inline std::string ll_dispatch_jit_source(
    bool useFp8,
    bool useUe8m0,
    bool useExternQuant,
    int hidden,
    ncclEpLayout_t layout,
    bool nvlinkOnly,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype) {
    const char* layout_literal =
        (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR) ? "NCCL_EP_LAYOUT_EXPERT_MAJOR" : "NCCL_EP_LAYOUT_RANK_MAJOR";
    const char* topk_type = topkIdxIsInt64 ? "int64_t" : "int32_t";
    const char* token_dtype_literal = ll_token_dtype_template_literal(tokenDtype);

    // The JIT source must reference the same arg struct definition that the
    // host packs. Including device/ll_ep_adapter.cuh keeps the layout in sync;
    // ll_ep.cuh pulls in all kernel templates and helpers.
    std::ostringstream src;
    src << "#include \"device/ll_ep.cuh\"\n"
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(1024, 1)\n"
        << "__global__ void " << kLlDispatchJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::internode_ll::dispatch_kernel_args_t p) {\n"
        << "  nccl_ep::internode_ll::dispatch_kernel_impl<\n"
        << "      " << ll_dispatch_bool_literal(useFp8) << ",\n"
        << "      " << ll_dispatch_bool_literal(useUe8m0) << ",\n"
        << "      " << ll_dispatch_bool_literal(useExternQuant) << ",\n"
        << "      " << hidden << ",\n"
        << "      " << layout_literal << ",\n"
        << "      " << ll_dispatch_bool_literal(nvlinkOnly) << ",\n"
        << "      " << topk_type << ",\n"
        << "      " << token_dtype_literal << ">(\n"
        << "      p.inData, p.inScalesBuf,\n"
        << "      static_cast<const " << topk_type << "*>(p.inTopkIdx), p.inTopkWeights,\n"
        << "      p.rankMask, p.asyncErrorFlag,\n"
        << "      p.outDataBuf, p.outScalesBuf, p.outSrcInfo,\n"
        << "      p.outRecvRankCounter, p.outLayout, p.outCnt,\n"
        << "      p.outRecvTopkWeights, p.outRecvTopkIdx,\n"
        << "      p.sendBuf, p.recvBuf, p.recvCntBuf,\n"
        << "      p.sendOff, p.recvOff, p.recvCntOff,\n"
        << "      p.rankCountersBase, p.rankDone,\n"
        << "      p.nextRecvCntBuf, p.nextRecvCntBufSize,\n"
        << "      p.recvStats, p.waitStats,\n"
        << "      p.numTokens, p.scalesPerToken, p.maxTokensPerRank, p.numTopk, p.numExperts,\n"
        << "      p.currRank, p.numRanks,\n"
        << "      p.numWarpGroups, p.numWarpsPerGroup,\n"
        << "      p.roundScale, p.recvTopkIdxKind, p.phases, p.numComms,\n"
        << "      p.devComms, p.windows, p.signalsBase, p.timeoutCycles,\n"
        << "      p.recvDataWindow, p.recvDataOffset);\n"
        << "}\n";
    return src.str();
}

inline void launch_ll_dispatch(
    bool useFp8,
    bool useUe8m0,
    bool useExternQuant,
    int hidden,
    ncclEpLayout_t layout,
    bool nvlinkOnly,
    bool topkIdxIsInt64,
    ncclDataType_t tokenDtype,
    int numSms,
    int numWarps,
    const dispatch_kernel_args_t& args,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = [&] {
        std::ostringstream name;
        name << "ll_dispatch"
             << "_hdim" << hidden << (layout == NCCL_EP_LAYOUT_EXPERT_MAJOR ? "_em" : "_rm")
             << (useFp8 ? "_fp8" : "_bf16") << (useUe8m0 ? "_ue8m0" : "") << (useExternQuant ? "_extern" : "")
             << (nvlinkOnly ? "_nvlinkonly" : "") << (topkIdxIsInt64 ? "_topk64" : "_topk32")
             << ll_token_dtype_name_tag(tokenDtype);
        return name.str();
    }();
    const std::string source = ll_dispatch_jit_source(
        useFp8,
        useUe8m0,
        useExternQuant,
        hidden,
        layout,
        nvlinkOnly,
        topkIdxIsInt64,
        tokenDtype);

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ll_dispatch";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLlDispatchJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = numSms;
    variant.block_dim = numWarps * 32;
    // Dispatch uses only statically allocated shared memory.
    variant.dynamic_smem_bytes = 0;
    // Cooperative launch is required for the cg::this_grid().sync() between the
    // SEND and RECV phases.
    variant.cooperative = true;
    // Pair SMs into clusters of 2 when possible to share distributed SMEM.
    variant.cluster_dim_x = (numSms % 2 == 0) ? 2 : 1;

    std::string error;
    // sizeof = 0 means "kernel_param points to a single fixed-size arg struct".
    const ::nccl_ep::jit::JitKernelStatus status =
        ::nccl_ep::jit::launch_jit_kernel(variant, const_cast<dispatch_kernel_args_t*>(&args), stream, &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal LL dispatch JIT launch failure for %s: %s%s%s\n",
                     variant_name.c_str(), ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace internode_ll
} // namespace nccl_ep
