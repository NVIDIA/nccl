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

constexpr const char* kLlCleanJitEntryName = "nccl_ep_jit_ll_clean_kernel";
constexpr int kLlCleanNumThreads = 256;

inline std::string ll_clean_jit_source() {
    std::ostringstream src;
    src << "#include \"device/ll_ep.cuh\"\n"
        << "#include \"device/ll_ep_adapter.cuh\"\n"
        << "\n"
        << "extern \"C\" __launch_bounds__(" << kLlCleanNumThreads << ", 1)\n"
        << "__global__ void " << kLlCleanJitEntryName << "(\n"
        << "    const __grid_constant__ nccl_ep::internode_ll::clean_low_latency_buffer_kernel_args_t p) {\n"
        << "  nccl_ep::internode_ll::clean_low_latency_buffer_kernel_impl<" << kLlCleanNumThreads << ">(\n"
        << "      p.clean_0, p.num_clean_int_0,\n"
        << "      p.clean_1, p.num_clean_int_1,\n"
        << "      p.rankMask,\n"
        << "      p.syncBuffer, p.syncWindow,\n"
        << "      p.devComms, p.barrierSignalBase, p.timeoutCycles);\n"
        << "}\n";
    return src.str();
}

inline void launch_ll_clean_low_latency_buffer(
    const clean_low_latency_buffer_kernel_args_t& args,
    cudaStream_t stream) {
    static const int variant_identity = 0;
    const std::string variant_name = "ll_clean";
    const std::string source = ll_clean_jit_source();

    ::nccl_ep::jit::JitKernelVariant variant;
    variant.kernel_family = "ll_clean";
    variant.variant_name = variant_name;
    variant.source = source;
    variant.entry_name = kLlCleanJitEntryName;
    variant.identity = &variant_identity;
    variant.runtime_key = static_cast<std::uint64_t>(std::hash<std::string>{}(variant_name));
    variant.num_blocks = 1;
    variant.block_dim = kLlCleanNumThreads;
    variant.dynamic_smem_bytes = 0;
    // Cooperative launch is required for the grid-wide barrier in the kernel.
    // A single block means no clustering.
    variant.cooperative = true;
    variant.cluster_dim_x = 1;

    std::string error;
    const ::nccl_ep::jit::JitKernelStatus status = ::nccl_ep::jit::launch_jit_kernel(
        variant,
        const_cast<clean_low_latency_buffer_kernel_args_t*>(&args),
        stream,
        &error);

    if (status != ::nccl_ep::jit::JitKernelStatus::kLaunched) {
        std::fprintf(stderr, "[nccl_ep jit] fatal LL clean JIT launch failure: %s%s%s\n",
                     ::nccl_ep::jit::jit_kernel_status_name(status), error.empty() ? "" : ": ",
                     error.empty() ? "" : error.c_str());
        std::abort();
    }
}

} // namespace jit
} // namespace internode_ll
} // namespace nccl_ep
