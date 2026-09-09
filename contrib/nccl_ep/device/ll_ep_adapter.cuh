/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "nccl_device.h"
#include "ep_enums.h"
#include "common.hpp"

namespace nccl_ep {
namespace internode_ll {

// Token-wire-dtype helpers shared by the LL dispatch/combine JIT generators.
// `*_template_literal` is the ncclDataType_t enumerator emitted into the JIT
// source as the kTokenDtype template argument; `*_name_tag` is a short suffix
// folded into the variant_name so distinct wire dtypes get distinct cache keys.
inline const char* ll_token_dtype_template_literal(ncclDataType_t dt) {
    switch (dt) {
    case ncclFloat32:
        return "ncclFloat32";
    case ncclFloat16:
        return "ncclFloat16";
    default:
        return "ncclBfloat16";
    }
}
inline const char* ll_token_dtype_name_tag(ncclDataType_t dt) {
    switch (dt) {
    case ncclFloat32:
        return "_tfp32";
    case ncclFloat16:
        return "_tfp16";
    default:
        return "_tbf16";
    }
}

// ============================================================================
// Packed kernel parameter structs.
//
// These structs are written by the host-side adapters and consumed by the
// JIT-compiled kernel entry points. The JIT entry receives the struct as
// `const __grid_constant__ <struct> p`, then unpacks `p.foo` into the
// templated `*_kernel_impl(...)` device function defined in ll_ep.cuh.
//
// The struct layouts must match what the JIT source expects, so any field
// added here must also be threaded through the corresponding JIT entry
// declared in device/jit/ll_dispatch_jit.cuh, device/jit/ll_combine_jit.cuh,
// or device/jit/ll_clean_jit.cuh.
// ============================================================================

struct dispatch_kernel_args_t {
    // INPUT
    const void* inData;
    const uint8_t* inScalesBuf;   // non-null for EXTERN FP8 quant (kExternQuant)
    const void* inTopkIdx;        // cast to const TopkIdxT* by the JIT entry
    const float* inTopkWeights;
    int* rankMask;
    int* asyncErrorFlag;
    // OUTPUT
    void* outDataBuf;
    void* outScalesBuf;
    int* outSrcInfo;
    int* outRecvRankCounter;
    int64_t* outLayout;
    int* outCnt;
    float* outRecvTopkWeights;
    int32_t* outRecvTopkIdx;
    // INTERMEDIATE
    void* sendBuf;
    void* recvBuf;
    int* recvCntBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvCntOff;
    int* rankCountersBase;
    int* rankDone;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int* recvStats;
    int64_t* waitStats;
    // CONFIG
    int numTokens;
    int scalesPerToken;           // EXTERN FP8 input scale count per token
    int maxTokensPerRank;
    int numTopk;
    int numExperts;
    int currRank;
    int numRanks;
    int numWarpGroups;
    int numWarpsPerGroup;
    bool roundScale;
    // recv_topk_idx numbering (LOCAL/GLOBAL); resolved on the host (never AUTO).
    ncclEpExpertIdKind_t recvTopkIdxKind;
    int phases;
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;
    uint64_t timeoutCycles;
    // Zero-copy dispatch output: when recvDataWindow != {}, sender writes the
    // payload directly into the peer's recv_x and the receiver skips the copy.
    ncclWindow_t recvDataWindow;
    size_t recvDataOffset;
};

struct combine_kernel_args_t {
    // INPUT
    const void* inData;
    const int* srcInfo;
    const int64_t* layoutRange;
    const void* inTopkIdx;        // cast to const TopkIdxT* by the JIT entry
    const float* topkWeights;
    int* rankMask;
    int* asyncErrorFlag;
    // OUTPUT
    void* outData;
    // INTERMEDIATE
    void* sendBuf;
    void* recvBuf;
    int* recvFlagBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvFlagOff;
    int* atomicCleanFlag;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int64_t* waitStats;
    // CONFIG
    int numCombinedTokens;
    int hidden;
    int numTopk;
    int maxTokensPerRank;
    int numExperts;
    int currRank;
    int numRanks;
    int numWarpGroups;
    int numWarpsPerGroup;
    int phases;
    bool zeroCopy;
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;
    uint64_t timeoutCycles;
};

struct clean_low_latency_buffer_kernel_args_t {
    int* clean_0;
    int num_clean_int_0;
    int* clean_1;
    int num_clean_int_1;
    int* rankMask;
    int* syncBuffer;
    ncclWindow_t* syncWindow;
    ncclDevComm* devComms;
    unsigned barrierSignalBase;
    uint64_t timeoutCycles;
};

// ============================================================================
// Public host-side parameter structs.
//
// Callers fill in a *Params struct once and pass it to call_dispatch /
// call_combine / call_clean_low_latency_buffer along with the few flags that
// vary per call (template-affecting bools, phases, stream). Grouping the many
// arguments into a struct keeps the call sites readable and lets fields default.
// ============================================================================

struct DispatchParams {
    // User inputs
    const void* inData;
    const uint8_t* inScalesBuf = nullptr;  // non-null for EXTERN FP8 quant
    const void* inTopkIdx;                  // int32_t* or int64_t*; see topkIdxIsInt64
    bool topkIdxIsInt64 = true;             // selects the TopkIdxT kernel specialization
    int scalesPerToken = 0;                 // EXTERN FP8 input scale count per token
    const float* inTopkWeights;

    // User / pre-allocated output buffers
    void* outDataBuf;
    void* outScalesBuf;
    int* outSrcInfo;
    int* outRecvRankCounter;       // rank-major only; nullptr otherwise
    int64_t* outLayout;
    int* outCnt;
    float* outRecvTopkWeights;     // rank-major only; nullptr otherwise
    int32_t* outRecvTopkIdx;       // rank-major only; nullptr otherwise

    // Intermediate RDMA buffers + window-relative offsets
    void* sendBuf;
    void* recvBuf;
    int* recvCntBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvCntOff;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int* recvStats;
    int64_t* waitStats;

    // Sizes / identifiers
    int numTokens;
    int hidden;
    int maxTokensPerRank;
    int numTopk;
    int numExperts;
    int currRank;
    int numRanks;
    ncclEpLayout_t layout;

    // GIN / NCCL device context
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;

    // Runtime workspace + error tracking
    void* workspace;
    int numDeviceSms;
    int* rankMask = nullptr;
    int* asyncErrorFlag = nullptr;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;

    // Per-call behavior toggles that select the JIT kernel specialization.
    // useFp8 stays outside the struct so callers see, at the call site,
    // which token-data-type variant is being launched.
    bool useUe8m0 = false;
    bool roundScale = false;
    bool nvlinkOnly = false;
    // recv_topk_idx numbering; the host wrapper resolves AUTO -> LOCAL before
    // launch, so the kernel only ever sees LOCAL or GLOBAL.
    ncclEpExpertIdKind_t recvTopkIdxKind = NCCL_EP_EXPERT_ID_LOCAL;
    int phases = 0;

    // Zero-copy dispatch output (rank-major + nvlinkOnly + bf16). When
    // recvDataWindow != {}, the sender writes payload directly into the peer's
    // recv_x buffer and the receiver skips the staging->recv_x copy.
    ncclWindow_t recvDataWindow = ncclWindow_t{};
    size_t recvDataOffset = 0;

    // Token wire dtype (unquantized payload width). Selects the kTokenDtype
    // kernel specialization: dispatch is a byte-copy so FP16 folds onto the
    // BF16 kernel and only FP32 is distinct; FP8 paths always wire BF16.
    ncclDataType_t tokenDtype = ncclBfloat16;
};

struct CombineParams {
    // User inputs
    const void* inData;
    const int* srcInfo;
    const int64_t* layoutRange;
    const void* inTopkIdx;        // int32_t* or int64_t*; see topkIdxIsInt64
    bool topkIdxIsInt64 = true;   // selects the TopkIdxT kernel specialization
    const float* topkWeights;

    // User output
    void* outData;

    // Intermediate RDMA buffers + window-relative offsets
    void* sendBuf;
    void* recvBuf;
    int* recvFlagBuf;
    size_t sendOff;
    size_t recvOff;
    size_t recvFlagOff;
    int* nextRecvCntBuf;
    int nextRecvCntBufSize;
    int64_t* waitStats;

    // Sizes / identifiers
    int numCombinedTokens;
    int hidden;
    int maxTokensPerRank;
    int numTopk;
    int numExperts;
    int currRank;
    int numRanks;
    ncclEpLayout_t layout;

    // GIN / NCCL device context
    int numComms;
    ncclDevComm* devComms;
    const ncclWindow_t* windows;
    unsigned signalsBase;

    // Runtime workspace + error tracking
    void* workspace;
    int numDeviceSms;
    int* rankMask = nullptr;
    int* asyncErrorFlag = nullptr;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;

    // Per-call behavior toggles that select the JIT kernel specialization.
    bool useLogFmt = false;
    bool zeroCopy = false;
    int phases = 0;

    // Token wire dtype (unquantized payload width). Combine decodes + reduces,
    // so BF16/FP16/FP32 are three distinct kernel specializations (FP32 also
    // halves kMaxNumGroups to stay within the dynamic-SMEM cap).
    ncclDataType_t tokenDtype = ncclBfloat16;
};

struct CleanLowLatencyBufferParams {
    int* clean_0;
    int num_clean_int_0;
    int* clean_1;
    int num_clean_int_1;
    int* rankMask;
    int* syncBuffer;
    ncclWindow_t* syncWindow;
    ncclDevComm* devComms;
    unsigned barrierSignalBase;
    uint64_t timeoutCycles = NUM_TIMEOUT_CYCLES;
};

// ============================================================================
// Host-side wrappers.
//
// Each wrapper resolves all runtime template parameters (hidden, layout,
// useFp8/useUe8m0, nvlinkOnly, useLogFmt) and dispatches to a per-variant
// JIT-compiled kernel.
// ============================================================================

void call_dispatch(const DispatchParams& params, bool useFp8, cudaStream_t stream = 0);

void call_combine(const CombineParams& params, cudaStream_t stream = 0);

void call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream = 0);

} // namespace internode_ll
} // namespace nccl_ep
