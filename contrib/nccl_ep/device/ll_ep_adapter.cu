/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#include "device/ll_ep_adapter.cuh"
#include "common.hpp"
#include "jit/ll_dispatch_jit.cuh"
#include "jit/ll_combine_jit.cuh"
#include "jit/ll_clean_jit.cuh"

#include <algorithm>

namespace nccl_ep {
namespace internode_ll {

// Forward-declare the host-side `ceil_div` helper used here. `common.hpp`
// provides templated ceil_div in namespace nccl_ep; we reuse it via ADL.
using ::nccl_ep::ceil_div;

// ============================================================================
// LL dispatch wrapper
//
//   - validates the workspace + numTopk/numExperts/numDeviceSms constraints
//   - chooses (numSms, numWarps) from the per-rank expert count
//   - packs the params + per-call flags into dispatch_kernel_args_t
//   - hands off to launch_ll_dispatch(), which JIT-compiles the kernel
//     specialised for (useFp8, useUe8m0, hidden, layout, nvlinkOnly)
// ============================================================================
void call_dispatch(const DispatchParams& params, bool useFp8, cudaStream_t stream) {
    constexpr int kNumMaxTopK = 9;
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= numWarpGroups * numWarpsPerGroup);

    const int numWarps = numWarpGroups * numWarpsPerGroup;
    const int numSms = ceil_div(params.numExperts, numWarpGroups);
    EP_HOST_ASSERT(params.numTopk <= kNumMaxTopK);

    // Workspace: [rankSentCnt | rankArrivedCnt | rankDone(=expertDone)].
    auto rankCountersBase = static_cast<int*>(params.workspace);
    auto rankDone = rankCountersBase + 2 * params.numRanks;
    EP_HOST_ASSERT((2 * params.numRanks + params.numExperts) * sizeof(int) <= NUM_WORKSPACE_BYTES);

    if (params.useUe8m0) EP_HOST_ASSERT(params.roundScale and "UE8M0 SF requires `round_scale=True`");

    dispatch_kernel_args_t args{};
    args.inData = params.inData;
    args.inScalesBuf = params.inScalesBuf;
    args.inTopkIdx = params.inTopkIdx;
    args.inTopkWeights = params.inTopkWeights;
    args.rankMask = params.rankMask;
    args.asyncErrorFlag = params.asyncErrorFlag;
    args.outDataBuf = params.outDataBuf;
    args.outScalesBuf = params.outScalesBuf;
    args.outSrcInfo = params.outSrcInfo;
    args.outRecvRankCounter = params.outRecvRankCounter;
    args.outLayout = params.outLayout;
    args.outCnt = params.outCnt;
    args.outRecvTopkWeights = params.outRecvTopkWeights;
    args.outRecvTopkIdx = params.outRecvTopkIdx;
    args.sendBuf = params.sendBuf;
    args.recvBuf = params.recvBuf;
    args.recvCntBuf = params.recvCntBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvCntOff = params.recvCntOff;
    args.rankCountersBase = rankCountersBase;
    args.rankDone = rankDone;
    args.nextRecvCntBuf = params.nextRecvCntBuf;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
    args.recvStats = params.recvStats;
    args.waitStats = params.waitStats;
    args.numTokens = params.numTokens;
    args.scalesPerToken = params.scalesPerToken;
    args.maxTokensPerRank = params.maxTokensPerRank;
    args.numTopk = params.numTopk;
    args.numExperts = params.numExperts;
    args.currRank = params.currRank;
    args.numRanks = params.numRanks;
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.roundScale = params.roundScale;
    args.recvTopkIdxKind = params.recvTopkIdxKind;
    args.phases = params.phases;
    args.numComms = params.numComms;
    args.devComms = params.devComms;
    args.windows = params.windows;
    args.signalsBase = params.signalsBase;
    args.timeoutCycles = params.timeoutCycles;
    args.recvDataWindow = params.recvDataWindow;
    args.recvDataOffset = params.recvDataOffset;

    // EXTERN FP8 (kExternQuant) is signalled by a non-null input scale buffer.
    const bool useExternQuant = (params.inScalesBuf != nullptr);

    // Dispatch is a byte-copy, so FP16 folds onto the BF16 kernel (no redundant
    // FP16 instantiation) and any FP8 path always wires BF16. Only non-FP8 FP32
    // input gets the distinct 4-byte kernel. Mirrors the legacy host launch
    // selection exactly (see ll_ep.cuh dispatch DISPATCH_LAUNCH_CASE_IMPL).
    const ncclDataType_t kernelTokenDtype =
        (!useFp8 && !useExternQuant && params.tokenDtype == ncclFloat32) ? ncclFloat32 : ncclBfloat16;

    jit::launch_ll_dispatch(
        useFp8,
        params.useUe8m0,
        useExternQuant,
        params.hidden,
        params.layout,
        params.nvlinkOnly,
        params.topkIdxIsInt64,
        kernelTokenDtype,
        numSms,
        numWarps,
        args,
        stream);
}

// ============================================================================
// LL combine wrapper
//
// Resolves (numSms, numWarps), computes the dynamic SMEM budget, packs args,
// and hands off to launch_ll_combine() for JIT compile + launch.
// ============================================================================
void call_combine(const CombineParams& params, cudaStream_t stream) {
    const int numWarpGroups = ceil_div(params.numExperts, params.numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    const int numRecvPerSm = ceil_div(params.numCombinedTokens, params.numDeviceSms);
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0 and numRecvPerSm >= 0);

    const int numWarps = numWarpGroups * numWarpsPerGroup;
    const int numSms = std::max(
        ceil_div(params.numExperts, numWarpGroups),
        numRecvPerSm == 0 ? 1 : ceil_div(params.numCombinedTokens, numRecvPerSm));

    auto atomicCleanFlag = static_cast<int*>(params.workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(params.numTopk <= jit::kLlCombineMaxTopk);

    // Online cast (LogFMT) is incompatible with zero-copy.
    EP_HOST_ASSERT(not(params.zeroCopy and params.useLogFmt));

    // Per-block SMEM = max(send-side TMA staging, recv-side TMA staging).
    // Send side: numWarps × kNumStages TMA buffers + per-warp LogFMT metadata.
    // Recv side: kMaxNumGroups × (kNumStages TMA buffers + decoded output +
    //            kNumStages × LogFMT decode metadata).
    constexpr int kNumStages = 3;
    // Must mirror the kernel's group count exactly (see combine_kernel_impl
    // kMaxNumGroups): FP32 doubles per-stage token bytes, so it runs 1 group to
    // stay within the device dynamic-SMEM cap; BF16/FP16 run 2. Computing 2 for
    // FP32 would over-request SMEM and fail the func attribute at large hidden.
    const int elemBytes = (params.tokenDtype == ncclFloat32) ? 4 : 2;
    const int kMaxNumGroups = (elemBytes == 2) ? 2 : 1;
    const int hidden = params.hidden;
    const int numMetaBytes = hidden / 128 * 4;
    const int numSendTmaBytes = 32 * static_cast<int>(sizeof(int4)) * jit::kLlCombineMaxUnrolls + 16;
    const int smemSendSize = numWarps * (kNumStages * numSendTmaBytes + numMetaBytes);
    const int numRecvTmaBytes = 16 + hidden * elemBytes;
    const int smemRecvSize =
        kMaxNumGroups * (kNumStages * numRecvTmaBytes + hidden * elemBytes + kNumStages * numMetaBytes * 3);
    const int smem_size = std::max(smemSendSize, smemRecvSize);

    combine_kernel_args_t args{};
    args.inData = params.inData;
    args.srcInfo = params.srcInfo;
    args.layoutRange = params.layoutRange;
    args.inTopkIdx = params.inTopkIdx;
    args.topkWeights = params.topkWeights;
    args.rankMask = params.rankMask;
    args.asyncErrorFlag = params.asyncErrorFlag;
    args.outData = params.outData;
    args.sendBuf = params.sendBuf;
    args.recvBuf = params.recvBuf;
    args.recvFlagBuf = params.recvFlagBuf;
    args.sendOff = params.sendOff;
    args.recvOff = params.recvOff;
    args.recvFlagOff = params.recvFlagOff;
    args.atomicCleanFlag = atomicCleanFlag;
    args.nextRecvCntBuf = params.nextRecvCntBuf;
    args.nextRecvCntBufSize = params.nextRecvCntBufSize;
    args.waitStats = params.waitStats;
    args.numCombinedTokens = params.numCombinedTokens;
    args.hidden = hidden;
    args.numTopk = params.numTopk;
    args.maxTokensPerRank = params.maxTokensPerRank;
    args.numExperts = params.numExperts;
    args.currRank = params.currRank;
    args.numRanks = params.numRanks;
    args.numWarpGroups = numWarpGroups;
    args.numWarpsPerGroup = numWarpsPerGroup;
    args.phases = params.phases;
    args.zeroCopy = params.zeroCopy;
    args.numComms = params.numComms;
    args.devComms = params.devComms;
    args.windows = params.windows;
    args.signalsBase = params.signalsBase;
    args.timeoutCycles = params.timeoutCycles;

    jit::launch_ll_combine(
        params.useLogFmt,
        hidden,
        params.layout,
        params.topkIdxIsInt64,
        params.tokenDtype,
        numSms,
        numWarps,
        smem_size,
        args,
        stream);
}

// ============================================================================
// LL buffer-clean wrapper
// ============================================================================
void call_clean_low_latency_buffer(const CleanLowLatencyBufferParams& params, cudaStream_t stream) {
    clean_low_latency_buffer_kernel_args_t args{};
    args.clean_0 = params.clean_0;
    args.num_clean_int_0 = params.num_clean_int_0;
    args.clean_1 = params.clean_1;
    args.num_clean_int_1 = params.num_clean_int_1;
    args.rankMask = params.rankMask;
    args.syncBuffer = params.syncBuffer;
    args.syncWindow = params.syncWindow;
    args.devComms = params.devComms;
    args.barrierSignalBase = params.barrierSignalBase;
    args.timeoutCycles = params.timeoutCycles;

    jit::launch_ll_clean_low_latency_buffer(args, stream);
}

} // namespace internode_ll
} // namespace nccl_ep
