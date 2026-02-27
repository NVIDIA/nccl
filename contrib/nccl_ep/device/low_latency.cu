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

#include <cooperative_groups.h>
#include "nccl_device.h"
#include "device_primitives.cuh"

namespace cg = cooperative_groups;

namespace nccl_ep {

namespace internode_ll {
template <bool useWarpSync = false>
__forceinline__ __device__ bool isRankMasked(int* rankMask, int rank) {
    if (rankMask == nullptr) {
        return false;
    }
    if constexpr (useWarpSync) {
        return __shfl_sync(0xffffffff, ld_acquire_global(rankMask + rank), 0) != 0;
    } else {
        return ld_acquire_global(rankMask + rank) != 0;
    }
}
__device__ __constant__ bool dP2pDisabled = false;

__forceinline__ __device__ uint64_t ncclGetP2pPtr(const uint64_t& dstPtr,
    const size_t& offset,
    const int& rank,
    const int& dstRank,
    const int& expertIdx,
    const ncclWindow_t* ncclWindows,
    ncclDevComm* devComms) {
    // Local rank, no need for peer mapping
    if (rank == dstRank) {
        return dstPtr;
    }

    // If P2P is globally disabled, always use RDMA path
    if (dP2pDisabled) {
        return 0;
    }

    // P2P/NVLink only works between ranks on the same node (LSA team)
    // Use NCCL team APIs to check if dstRank is in the same LSA team
    auto commId = expertIdx / MAX_NCCL_GIN_CTX_PER_COMM;
    ncclTeam lsa = ncclTeamLsa(devComms[commId]);
    ncclTeam world = ncclTeamWorld(devComms[commId]);
    if (!ncclTeamRankIsMember(lsa, world, dstRank))
        return 0;  // Different nodes (not in same LSA team), must use RDMA

    auto const p2pPtr = reinterpret_cast<uint64_t>(ncclGetPeerPointer(ncclWindows[commId],
                                                                       offset, dstRank));

    return p2pPtr ? p2pPtr : 0;
}

// ============================================================================
// Helper functions for dispatch kernel modularization
// ============================================================================

// Cast BF16 data to FP8 and write to send buffer, or copy BF16 data directly
template<bool kUseFP8>
__forceinline__ __device__ void castAndWriteToSendBuf(
    const int4* srcDataInt4,
    typename std::conditional<kUseFP8, int2, int4>::type* sendBufVec,
    float* sendBufScales,
    int threadId,
    int numThreads,
    int laneId,
    int hiddenBf16Int4,
    bool roundScale) {
    constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int kNumPerChannels = 128;

    EP_DEVICE_ASSERT(hiddenBf16Int4 % 32 == 0);
    #pragma unroll
    for (int i = threadId; i < hiddenBf16Int4; i += numThreads) {
        auto dataInt4 = __ldg(srcDataInt4 + i);

        if constexpr (kUseFP8) {
            // Calculate local amax
            auto bf16Data = reinterpret_cast<nv_bfloat16*>(&dataInt4);
            float fp32Data[kNumElemsPerRead];
            float amax = kFP8Margin, scale, scaleInv;
            #pragma unroll
            for (int j = 0; j < kNumElemsPerRead; ++j) {
                fp32Data[j] = static_cast<float>(bf16Data[j]);
                amax = fmaxf(amax, fabsf(fp32Data[j]));
            }

            // Reduce amax and scale
            EP_STATIC_ASSERT(kNumElemsPerRead * 32 / kNumPerChannels == 2, "Invalid vectorization");
            amax = warp_reduce_max<16>(amax);
            calculate_fp8_scales(amax, scale, scaleInv, roundScale);
            if (laneId == 0 or laneId == 16)
                sendBufScales[i * kNumElemsPerRead / 128] = scaleInv;

            // Cast into send buffer
            typename std::conditional<kUseFP8, int2, int4>::type dataInt2;
            auto fp8x2Data = reinterpret_cast<__nv_fp8x2_storage_t*>(&dataInt2);
            #pragma unroll
            for (int j = 0; j < kNumElemsPerRead; j += 2) {
                float2 fp32x2 = {fp32Data[j] * scale, fp32Data[j + 1] * scale};
                fp8x2Data[j / 2] = __nv_cvt_float2_to_fp8x2(fp32x2, __NV_SATFINITE, __NV_E4M3);
            }
            sendBufVec[i] = dataInt2;
        } else {
            // Reinterpret-cast is for C++14 compatibility
            sendBufVec[i] = *reinterpret_cast<typename std::conditional<kUseFP8, int2, int4>::type*>(&dataInt4);
        }
    }
}

// Send a token via RDMA or P2P
__forceinline__ __device__ void sendToken(
    const int* sendBufSrcIdx,
    uint64_t recvPtr,
    size_t expectedDstOffset,
    int tokenIdx,
    size_t sendOff,
    size_t numBytesPerMsg,
    int dstRank,
    int dstExpertLocalIdx,
    int currRank,
    int* rankMask,
    const ncclWindow_t* windows,
    ncclDevComm* devComms,
    int laneId,
    size_t numInt4PerMsg) {
    const auto dstP2pPtr = ncclGetP2pPtr(recvPtr, expectedDstOffset, currRank,
                                         dstRank, dstExpertLocalIdx, windows, devComms);

    if (not isRankMasked<true>(rankMask, dstRank)) {
        if (dstP2pPtr == 0) {
            if (laneId == 0) {
                size_t expectedSrcOffset = sendOff + tokenIdx * numBytesPerMsg;
                auto commId = dstExpertLocalIdx / MAX_NCCL_GIN_CTX_PER_COMM;
                auto ctxId = dstExpertLocalIdx % MAX_NCCL_GIN_CTX_PER_COMM;
                ncclGin net(devComms[commId], ctxId);
                ncclTeam world = ncclTeamWorld(devComms[commId]);
                auto ncclWindow = windows[commId];
                net.put(world,
                        dstRank,
                        ncclWindow,
                        expectedDstOffset,
                        ncclWindow,
                        expectedSrcOffset,
                        numBytesPerMsg,
                        ncclGin_None{},  // no signal
                        ncclGin_None{},  // no counter
                        ncclCoopThread());
            }
        } else {
            // NOTES: only 2 load iterations for 7K hidden with 8 unrolls
            // Copy entire message including index from sendBufSrcIdx to dstP2pPtr
            const auto* sendDataInt4 = reinterpret_cast<const int4*>(sendBufSrcIdx);
            const auto* recvDataInt4 = reinterpret_cast<int4*>(dstP2pPtr);
            UNROLLED_WARP_COPY(8, laneId, numInt4PerMsg, recvDataInt4, sendDataInt4, ld_nc_global, st_na_global);
        }
    }
}

// Clean next receive count buffer
__forceinline__ __device__ void cleanNextRecvCntBuf(
    int* nextRecvCntBuf,
    int nextRecvCntBufSize,
    int laneId) {
    if (!dP2pDisabled) {
        #pragma unroll
        for (int i = laneId; i < nextRecvCntBufSize; i += 32)
            nextRecvCntBuf[i] = 0;
    }
}

// Count tokens per expert from topk indices
__forceinline__ __device__ void countTokensPerExpert(
    const int64_t* inTopkIdx,
    int numTokens,
    int numTopk,
    int expertBeginIdx,
    int expertEndIdx,
    int* expertCount,
    int* sharedNumTokensSentPerExpert,
    int* expertDone,
    int laneId) {

    // Per lane count
    #pragma unroll 8
    for (int i = laneId; i < numTokens * numTopk; i += 32) {
        auto idx = static_cast<int>(__ldg(inTopkIdx + i));
        if (idx >= expertBeginIdx and idx < expertEndIdx)
            expertCount[idx - expertBeginIdx]++;
    }

    // Warp reduce
    #pragma unroll
    for (int i = expertBeginIdx; i < expertEndIdx; ++i) {
        auto sum = warp_reduce_sum(expertCount[i - expertBeginIdx]);
        if (laneId == 0) {
            sharedNumTokensSentPerExpert[i - expertBeginIdx] = sum;
            atomic_add_release_global(expertDone + i, FINISHED_SUM_TAG - sum);
        }
    }
}

// Send expert count to destination rank
__forceinline__ __device__ void sendExpertCount(
    int numTokensSent,
    int dstRank,
    int dstExpertLocalIdx,
    int currRank,
    int numRanks,
    uint64_t recvCntPtr,
    size_t recvCntOffset,
    int* recvCntBuf,
    int* rankMask,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms) {
    const auto dstP2pPtr = ncclGetP2pPtr(recvCntPtr, recvCntOffset, currRank, dstRank,
                                          dstExpertLocalIdx, windows, devComms);

    if (not isRankMasked(rankMask, dstRank)) {
        if (dstP2pPtr == 0) {
            auto commId = dstExpertLocalIdx / MAX_NCCL_GIN_CTX_PER_COMM;
            auto ctxId = dstExpertLocalIdx % MAX_NCCL_GIN_CTX_PER_COMM;
            auto signalId = signalsBase + dstExpertLocalIdx * numRanks + currRank;
            ncclGin net(devComms[commId], ctxId);
            ncclTeam world = ncclTeamWorld(devComms[commId]);
            auto ncclWindow = windows[commId];
            net.put(world,
                    dstRank,
                    ncclWindow,
                    recvCntOffset,
                    ncclWindow,
                    0,
                    0,  // 0 bytes transfer
                    ncclGin_SignalAdd{signalId, static_cast<uint64_t>(numTokensSent) + 1},
                    ncclGin_None{},  // no counter
                    ncclCoopThread());
        } else {
            st_release_sys_global(reinterpret_cast<int*>(dstP2pPtr), -numTokensSent - 1);
        }
    }
}

// Wait for receive tokens to arrive and return count
__forceinline__ __device__ int waitForRecvTokens(
    int srcRank,
    int localExpertIdx,
    int currRank,
    int numRanks,
    size_t recvCntOff,
    int* recvCntBuf,
    int* rankMask,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms,
    int* recvStats,
    int64_t* waitStats) {
    auto startTime = clock64();
    uint64_t waitRecvCost = 0;
    int numRecvTokens = 0;

    if (not isRankMasked(rankMask, srcRank)) {
        size_t srcOffset = recvCntOff + (localExpertIdx * numRanks + srcRank) * sizeof(int);
        auto srcP2pPtr = ncclGetP2pPtr(0x01, srcOffset, currRank, srcRank, localExpertIdx, windows, devComms);

        if (srcP2pPtr == 0) {
            auto commId = localExpertIdx / MAX_NCCL_GIN_CTX_PER_COMM;
            auto ctxId = localExpertIdx % MAX_NCCL_GIN_CTX_PER_COMM;
            ncclGin net(devComms[commId], ctxId);
            uint64_t curValue;
            do {
                curValue = net.readSignal(signalsBase + localExpertIdx * numRanks + srcRank);
            } while (curValue < 1                                                       // data not arrived
                     && (waitRecvCost = clock64() - startTime) <= NUM_TIMEOUT_CYCLES  // not timeout
            );
            net.resetSignal(signalsBase + localExpertIdx * numRanks + srcRank);
            numRecvTokens = -(int)curValue;
        } else {
            while ((numRecvTokens = ld_acquire_sys_global((recvCntBuf + localExpertIdx * numRanks + srcRank))) ==
                       0                                                               // data not arrived
                   && (waitRecvCost = clock64() - startTime) <= NUM_TIMEOUT_CYCLES  // not timeout
                  );
        }
    }

    // Do not receive tokens if rank timeout or masked
    if (numRecvTokens == 0)
        numRecvTokens = -1;

    // Mask rank if timeout
    if (waitRecvCost > NUM_TIMEOUT_CYCLES) {
        printf("Warning: NCCL EP timeout for dispatch receive, rank %d, local_expert_idx %d, src_rank %d\n",
               currRank, localExpertIdx, srcRank);
        if (rankMask == nullptr)
            trap();
        atomicExch(rankMask + srcRank, 1);
    }

    numRecvTokens = -numRecvTokens - 1;

    // Add stats for diagnosis
    if (recvStats != nullptr)
        atomicAdd(recvStats + localExpertIdx, numRecvTokens);
    if (waitStats != nullptr)
        atomicAdd(reinterpret_cast<unsigned long long*>(waitStats + srcRank), waitRecvCost);

    return numRecvTokens;
}

// Copy received token data and scales
template<bool kUseFP8, bool kUseUE8M0>
__forceinline__ __device__ void copyRecvTokenData(
    const uint8_t* recvBufUint8,
    int tokenIdx,
    int recvTokenBeginIdx,
    int4* outDataInt4,
    int* outSrcInfo,
    typename std::conditional<kUseUE8M0, uint8_t, float>::type* outScales,
    int hiddenInt4,
    int hiddenBytes,
    int numScales,
    int numBytesPerMsg,
    int maxTokensPerRank,
    int numRanks,
    int laneId) {
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;

    // Copy source info
    const auto recvBufSrcIdx = reinterpret_cast<const int*>(recvBufUint8 + tokenIdx * numBytesPerMsg);
    if (laneId == 0)
        outSrcInfo[recvTokenBeginIdx + tokenIdx] = ld_nc_global(recvBufSrcIdx);
    __syncwarp();

    // Copy data
    // NOTES: only 2 load iterations for 7K hidden with 7 unrolls
    const auto recvDataInt4 = reinterpret_cast<const int4*>(reinterpret_cast<const uint8_t*>(recvBufSrcIdx) + sizeof(int4));
    const auto outDataInt4Ptr = outDataInt4 + (recvTokenBeginIdx + tokenIdx) * hiddenInt4;
    UNROLLED_WARP_COPY(7, laneId, hiddenInt4, outDataInt4Ptr, recvDataInt4, ld_nc_global, st_na_global);

    // Copy scales
    if constexpr (kUseFP8) {
        // Equivalent CuTe layout:
        //   (num_tokens, (num_packed, num_elems_per_pack)):(num_elems_per_pack, (num_tokens * num_elems_per_pack, 1))
        const auto recvScales = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(recvDataInt4) + hiddenBytes);
        const auto numElemsPerPack = static_cast<int>(sizeof(packed_t) / sizeof(scale_t));
        const auto tokenIdxFinal = recvTokenBeginIdx + tokenIdx;
        const auto tokenStride = numElemsPerPack;
        const auto packStride = numRanks * maxTokensPerRank * numElemsPerPack;
        if (laneId < numScales) {
            const auto packIdx = laneId / numElemsPerPack;
            const auto elemIdx = laneId % numElemsPerPack;
            auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(recvScales + laneId));
            outScales[tokenIdxFinal * tokenStride + packIdx * packStride + elemIdx] = scale;
        }
        if (laneId + 32 < numScales) {
            const auto packIdx = (laneId + 32) / numElemsPerPack;
            const auto elemIdx = (laneId + 32) % numElemsPerPack;
            auto scale = extract_required_scale_format<kUseUE8M0>(ld_nc_global(recvScales + laneId + 32));
            outScales[tokenIdxFinal * tokenStride + packIdx * packStride + elemIdx] = scale;
        }
    }
}

template <bool kUseFP8, bool kUseUE8M0, int kHidden>
__global__ __launch_bounds__(1024, 1) void dispatch(// INPUT
                                                    const void* inData,
                                                    const int64_t* inTopkIdx,
                                                    int* rankMask,
                                                    // OUTPUT
                                                    void* outDataBuf,
                                                    void* outScalesBuf,
                                                    int* outSrcInfo,
                                                    int64_t* outLayout,
                                                    int* outCnt,
                                                    // INTERMEDIATE
                                                    void* sendBuf,
                                                    void* recvBuf,
                                                    int* recvCntBuf,
                                                    size_t sendOff,
                                                    size_t recvOff,
                                                    size_t recvCntOff,
                                                    int* expertCnt,
                                                    int* expertDone,
                                                    int* nextRecvCntBuf,
                                                    int nextRecvCntBufSize,
                                                    int* recvStats,
                                                    int64_t* waitStats,
                                                    // CONFIG
                                                    int numTokens,
                                                    int maxTokensPerRank,
                                                    int numTopk,
                                                    int numExperts,
                                                    int currRank,
                                                    int numRanks,
                                                    int numWarpGroups,
                                                    int numWarpsPerGroup,
                                                    bool roundScale,
                                                    int phases,
                                                    int numComms,
                                                    ncclDevComm* devComms,
                                                    const ncclWindow_t* windows,
                                                    unsigned signalsBase) {
    const auto smId = static_cast<int>(blockIdx.x);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto warpId = threadId / 32, laneId = get_lane_id();
    const auto numSms = static_cast<int>(gridDim.x);
    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto numLocalExperts = numExperts / numRanks;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    // May extract UE8M0 from the scales
    using scale_t = std::conditional_t<kUseUE8M0, uint8_t, float>;
    using packed_t = std::conditional_t<kUseUE8M0, uint32_t, float>;
    EP_STATIC_ASSERT(sizeof(packed_t) % sizeof(scale_t) == 0, "Invalid vector length");

    // FP8 staffs
    constexpr int kNumPerChannels = 128;
    const int numScales = kHidden / kNumPerChannels;
    const size_t hiddenBytes = kHidden * (kUseFP8 ? sizeof(__nv_fp8_storage_t) : sizeof(nv_bfloat16));
    const size_t hiddenInt4 = hiddenBytes / sizeof(int4);

    // Message package: index at source (int), 3 reserved int fields, hidden data, FP8 scales
    // NOTES: currently we have 3 reserved int fields for future use
    using vec_t = std::conditional_t<kUseFP8, int2, int4>;
    const size_t numBytesPerMsg = sizeof(int4) + (kUseFP8 ? (kHidden + numScales * sizeof(float)) : (kHidden * sizeof(nv_bfloat16)));
    const size_t numInt4PerMsg = numBytesPerMsg / sizeof(int4);
    EP_DEVICE_ASSERT(numBytesPerMsg % sizeof(int4) == 0);

    // Expert counts
    constexpr int kNumMaxWarpGroups = 32;
    __shared__ int sharedNumTokensSentPerExpert[kNumMaxWarpGroups];

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_DISPATCH_RECV;

    // There are 2 kinds of warps in this part:
    // 1. The first-kind warps for FP8 cast and sending top-k tokens
    // 2. The last warp for reading `topk_idx` and count for per-expert information
    if (warpId < numWarps - 1) {
        constexpr int kNumElemsPerRead = sizeof(int4) / sizeof(nv_bfloat16);
        EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerRead) == 0, "Invalid hidden");
        EP_STATIC_ASSERT(kNumElemsPerRead * 32 % kNumPerChannels == 0, "Invalid vectorization");
        const auto numThreads = (numWarps - 1) * 32;
        const size_t hiddenBf16Int4 = kHidden / kNumElemsPerRead;

        for (int tokenIdx = smId; tokenIdx < numTokens; tokenIdx += numSms) {
            const auto srcDataInt4 = static_cast<const int4*>(inData) + tokenIdx * hiddenBf16Int4;
            const auto sendBufSrcIdx = reinterpret_cast<int*>(static_cast<uint8_t*>(sendBuf) + tokenIdx * numBytesPerMsg);
            const auto sendBufVec = reinterpret_cast<vec_t*>(reinterpret_cast<uint8_t*>(sendBufSrcIdx) + sizeof(int4));
            const auto sendBufScales = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(sendBufVec) + hiddenBytes);

            // Overlap top-k index read and source token index writes
            auto dstExpertIdx = warpId < numTopk ? static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + warpId)) : -1;
            threadId == 0 ? (*sendBufSrcIdx = tokenIdx) : 0;

            // Cast and write data to send buffer
            castAndWriteToSendBuf<kUseFP8>(
                srcDataInt4, sendBufVec, sendBufScales,
                threadId, numThreads, laneId, hiddenBf16Int4, roundScale);
            asm volatile("bar.sync 1, %0;" :: "r"(numThreads));

            // Issue IBGDA sends
            if (dstExpertIdx >= 0) {
                int slotIdx = laneId == 0 ? atomicAdd(expertCnt + dstExpertIdx, 1) : 0;
                slotIdx = __shfl_sync(0xffffffff, slotIdx, 0);
                const auto dstRank = dstExpertIdx / numLocalExperts;
                const auto dstExpertLocalIdx = dstExpertIdx % numLocalExperts;
                const auto recvPtr = reinterpret_cast<uint64_t>(recvBuf) +
                                     dstExpertLocalIdx * numRanks * maxTokensPerRank * numBytesPerMsg +
                                     currRank * maxTokensPerRank * numBytesPerMsg +
                                     slotIdx * numBytesPerMsg;
                size_t expectedDstOffset = recvOff +
                                             dstExpertLocalIdx * numRanks * maxTokensPerRank * numBytesPerMsg +
                                             currRank * maxTokensPerRank * numBytesPerMsg + slotIdx * numBytesPerMsg;

                sendToken(
                    sendBufSrcIdx, recvPtr, expectedDstOffset, tokenIdx,
                    sendOff, numBytesPerMsg, dstRank, dstExpertLocalIdx,
                    currRank, rankMask, windows, devComms, laneId, numInt4PerMsg);

                // Increase counter after finishing
                __syncwarp();
                laneId == 0 ? atomic_add_release_global(expertDone + dstExpertIdx, 1) : 0;
            }
        }
    } else if (warpId == numWarps - 1) {
        EP_DEVICE_ASSERT(numSms > 1);
        if (smId == 0) {
            cleanNextRecvCntBuf(nextRecvCntBuf, nextRecvCntBufSize, laneId);
            // Notify before executing `int_p`
            __syncwarp();
            #pragma unroll
            for (int i = laneId; i < numExperts; i += 32)
                atomic_add_release_global(expertDone + i, FINISHED_SUM_TAG);
        }

        // Count tokens per expert from topk indices
        int expertCount[kNumMaxWarpGroups] = {0};
        const auto expertBeginIdx = smId * numWarpGroups;
        const auto expertEndIdx = min(expertBeginIdx + numWarpGroups, numExperts);
        countTokensPerExpert(
            inTopkIdx, numTokens, numTopk, expertBeginIdx, expertEndIdx,
            expertCount, sharedNumTokensSentPerExpert, expertDone, laneId);
    }
    __syncthreads();

    // Issue count sends
    if (responsibleExpertIdx < numExperts and subWarpId == 0 and laneId == 0) {
        const auto dstRank = responsibleExpertIdx / numLocalExperts;
        const auto dstExpertLocalIdx = responsibleExpertIdx % numLocalExperts;
        const auto numTokensSent = sharedNumTokensSentPerExpert[responsibleExpertIdx - smId * numWarpGroups];

        // Wait local sends issued and send expert counts
        while (ld_acquire_global(expertDone + responsibleExpertIdx) != FINISHED_SUM_TAG * 2);
        auto recvCntPtr = reinterpret_cast<uint64_t>(recvCntBuf + dstExpertLocalIdx * numRanks + currRank);
        size_t recvCntOffset = recvCntOff + (dstExpertLocalIdx * numRanks + currRank) * sizeof(int);

        sendExpertCount(
            numTokensSent, dstRank, dstExpertLocalIdx, currRank, numRanks,
            recvCntPtr, recvCntOffset, recvCntBuf, rankMask, signalsBase,
            windows, devComms);

        // Clean workspace for next use
        expertCnt[responsibleExpertIdx] = 0;
        expertDone[responsibleExpertIdx] = 0;

        // Clean `packed_recv_count`
        if (dstRank == 0)
            outCnt[dstExpertLocalIdx] = 0;
    }
    __syncwarp();

// Receiving phase
LOW_LATENCY_DISPATCH_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // For send-and-recv kernels, we need a grid sync for making `packed_recv_count` visible
    if (phases & LOW_LATENCY_SEND_PHASE)
        cg::this_grid().sync();

    // Receiving and packing
    if (responsibleExpertIdx < numExperts) {
        const auto srcRank = responsibleExpertIdx / numLocalExperts;
        const auto localExpertIdx = responsibleExpertIdx % numLocalExperts;
        const auto recvBufUint8 = static_cast<uint8_t*>(recvBuf) +
                localExpertIdx * numRanks * maxTokensPerRank * numBytesPerMsg +
                srcRank * maxTokensPerRank * numBytesPerMsg;
        const auto outDataInt4 = static_cast<int4*>(outDataBuf) +
                localExpertIdx * numRanks * maxTokensPerRank * hiddenInt4;
        const auto recvSrcInfo = outSrcInfo + localExpertIdx * numRanks * maxTokensPerRank;
        const auto recvRange = outLayout + localExpertIdx * numRanks;
        const auto numAlignedScales = align<int>(numScales, sizeof(float) / sizeof(scale_t));
        const auto outScales = static_cast<scale_t*>(outScalesBuf) + localExpertIdx * numRanks * maxTokensPerRank * numAlignedScales;

        // Shared between sub-warps in warp groups
        __shared__ int sharedNumRecvTokens[kNumMaxWarpGroups], sharedRecvTokenBeginIdx[kNumMaxWarpGroups];

        // Wait tokens to arrive
        // NOTES: using sub-warp 1 to overlap with sub-warp 0
        int numRecvTokens, recvTokenBeginIdx;
        EP_DEVICE_ASSERT(numWarpsPerGroup > 1 and numWarpGroups < 15);
        if (subWarpId == 1 and laneId == 0) {
            numRecvTokens = waitForRecvTokens(
                srcRank, localExpertIdx, currRank, numRanks, recvCntOff,
                recvCntBuf, rankMask, signalsBase, windows, devComms,
                recvStats, waitStats);
            recvTokenBeginIdx = atomicAdd(outCnt + localExpertIdx, numRecvTokens);
            sharedNumRecvTokens[warpGroupId] = numRecvTokens;
            sharedRecvTokenBeginIdx[warpGroupId] = recvTokenBeginIdx;
            recvRange[srcRank] = pack2<int, int64_t>(numRecvTokens, recvTokenBeginIdx);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(warpGroupId + 2), "r"(numWarpsPerGroup * 32));
        numRecvTokens = sharedNumRecvTokens[warpGroupId];
        recvTokenBeginIdx = sharedRecvTokenBeginIdx[warpGroupId];

        // Copy tokens
        EP_DEVICE_ASSERT(numScales <= 64);
        for (int i = subWarpId; i < numRecvTokens; i += numWarpsPerGroup) {
            copyRecvTokenData<kUseFP8, kUseUE8M0>(
                recvBufUint8, i, recvTokenBeginIdx, outDataInt4, recvSrcInfo, outScales,
                hiddenInt4, hiddenBytes, numScales, numBytesPerMsg,
                maxTokensPerRank, numRanks, laneId);
        }
    }
}

void dispatch(const void* inData,
              const int64_t* inTopkIdx,
              void* outDataBuf,
              void* outScalesBuf,
              int* outSrcInfo,
              int64_t* outLayout,
              int* outCnt,
              void* sendBuf,
              void* recvBuf,
              int* recvCntBuf,
              size_t sendOff,
              size_t recvOff,
              size_t recvCntOff,
              int* nextRecvCntBuf,
              int nextRecvCntBufSize,
              int* recvStats,
              int64_t* waitStats,
              int numTokens,
              int hidden,
              int maxTokensPerRank,
              int numTopk,
              int numExperts,
              int currRank,
              int numRanks,
              bool useFp8,
              bool roundScale,
              bool useUe8m0,
              int phases,
              int numComms,
              ncclDevComm* devComms,
              const ncclWindow_t* windows,
              unsigned signalsBase,
              void* workspace,
              int numDeviceSms,
              cudaStream_t stream) {
    constexpr int kNumMaxTopK = 9;
    const int numWarpGroups = ceil_div(numExperts, numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0);
    EP_HOST_ASSERT(kNumMaxTopK + 1 <= numWarpGroups * numWarpsPerGroup);

    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto numSms = ceil_div(numExperts, numWarpGroups);
    EP_HOST_ASSERT(numTopk <= kNumMaxTopK);

    // Workspace checks
    auto expertCnt = static_cast<int*>(workspace);
    auto expertDone = expertCnt + numExperts;
    EP_HOST_ASSERT(numExperts * sizeof(int) * 2 <= NUM_WORKSPACE_BYTES);

    // FP8 checks
    if (useUe8m0)
        EP_HOST_ASSERT(roundScale and "UE8M0 SF requires `round_scale=True`");

#define DISPATCH_LAUNCH_CASE(hidden) { \
auto dispatchFunc = dispatch<false, false, hidden>; \
if (useFp8 and not useUe8m0) \
    dispatchFunc = dispatch<true, false, hidden>; \
if (useFp8 and useUe8m0) \
    dispatchFunc = dispatch<true, true, hidden>; \
LAUNCH_KERNEL(&cfg, dispatchFunc, \
              inData, \
              inTopkIdx, \
              /*rankMask=*/nullptr, \
              outDataBuf, \
              outScalesBuf, \
              outSrcInfo, \
              outLayout, \
              outCnt, \
              sendBuf, \
              recvBuf, \
              recvCntBuf, \
              sendOff, \
              recvOff, \
              recvCntOff, \
              expertCnt, \
              expertDone, \
              nextRecvCntBuf, \
              nextRecvCntBufSize, \
              recvStats, \
              waitStats, \
              numTokens, \
              maxTokensPerRank, \
              numTopk, \
              numExperts, \
              currRank, \
              numRanks, \
              numWarpGroups, \
              numWarpsPerGroup, \
              roundScale, \
              phases, \
              numComms, \
              devComms, \
              windows, \
              signalsBase); } break

    SETUP_LAUNCH_CONFIG(numSms, numWarps * 32, stream);
    SWITCH_HIDDEN(DISPATCH_LAUNCH_CASE);
#undef DISPATCH_LAUNCH_CASE
}

template <int kNumSendUnrolls>
__forceinline__ __device__ int logfmtEncode(void* buffer, nv_bfloat162 *sharedAmaxmin, const int& laneId) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`
    constexpr int kNumBits = 10;
    constexpr int kNumValues = 1 << (kNumBits - 1);

    int4 int4Data[kNumSendUnrolls];
    const auto& uint32Data = reinterpret_cast<uint32_t*>(int4Data);
    const auto& bf162Data = reinterpret_cast<nv_bfloat162*>(int4Data);

    // Calculate lane offset
    const auto& loadBuf = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + laneId * (kNumSendUnrolls * sizeof(int4)));
    const auto& storeBuf = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(buffer) + laneId * (kNumSendUnrolls * sizeof(int4) * 10 / 16));

    // Local log amax
    auto bf162Amax = __nv_bfloat162(CUDART_ZERO_BF16, CUDART_ZERO_BF16);
    auto bf162Amin = __nv_bfloat162(CUDART_INF_BF16, CUDART_INF_BF16);
    uint32_t localSigns = 0;
    #pragma unroll
    for (int k = 0; k < kNumSendUnrolls * kNumElemsPerInt4 / 2; ++ k) {
        uint32Data[k] = loadBuf[k];
        localSigns |= ((uint32Data[k] >> 15) & 1) << (k * 2);
        localSigns |= ((uint32Data[k] >> 31) & 1) << (k * 2 + 1);
        uint32Data[k] &= 0x7fff7fff;

        bf162Amax = __hmax2(bf162Amax, bf162Data[k]);
        bf162Amin = __hmin2(bf162Amin, bf162Data[k]);
    }

    // Reduce per 128 channels
    auto amax = std::max(static_cast<float>(bf162Amax.x), static_cast<float>(bf162Amax.y));
    auto amin = std::min(static_cast<float>(bf162Amin.x), static_cast<float>(bf162Amin.y));
    constexpr static int kNumLanesToReduce = 128 * sizeof(nv_bfloat16) / (kNumSendUnrolls * sizeof(int4));
    amax = warp_reduce_max<kNumLanesToReduce>(amax);
    amin = warp_reduce_min<kNumLanesToReduce>(amin);

    // Write min/max into the shared memory
    if (sharedAmaxmin != nullptr)
        *sharedAmaxmin = __nv_bfloat162(amax, amin);
    __syncwarp();

    // Calculate log amin/amax float
    const auto& logAmax = log2f_approx(amax);
    const auto& logAmin = fmaxf(log2f_approx(amin), logAmax - kMinClip);
    const bool& enableCast = warp_reduce_and<kNumLanesToReduce, true>(logAmax < kLogThreshold and logAmin < logAmax);

    // Case into LogFMT-10 if satisfied
    if (enableCast) {
        const auto step = (logAmax - logAmin) / static_cast<float>(kNumValues - 2);
        const auto stepInv = 1.0f / step;
        const auto rounding = 2.0f - log2f_approx((1.0f + exp2f_approx(step)) * 0.5f) * stepInv;
        const auto fusedRounding = rounding - logAmin * stepInv;

        // Pack every 256 bits into 160 bits
        EP_STATIC_ASSERT(kNumSendUnrolls == 2 or kNumSendUnrolls == 4, "kNumSendUnrolls == 2 or 4 only");
        uint32_t encodedData[kNumElemsPerInt4 * 2];
        #pragma unroll 1
        for (int i = 0; i < kNumSendUnrolls / 2; ++ i) {
            #pragma unroll
            for (int k = 0; k < kNumElemsPerInt4; ++ k) {
                const auto& [x, y] = __bfloat1622float2(bf162Data[i * kNumElemsPerInt4 + k]);
                encodedData[k * 2 + 0] = __float2uint_rd(fmaxf(log2f_approx(x) * stepInv + fusedRounding, 0));
                encodedData[k * 2 + 1] = __float2uint_rd(fmaxf(log2f_approx(y) * stepInv + fusedRounding, 0));
            }
            storeBuf[i * 5 + 0] = (encodedData[ 0] >> 0) | (encodedData[ 1] << 9) | (encodedData[ 2] << 18) | (encodedData[ 3] << 27);
            storeBuf[i * 5 + 1] = (encodedData[ 3] >> 5) | (encodedData[ 4] << 4) | (encodedData[ 5] << 13) | (encodedData[ 6] << 22) | (encodedData[7]  << 31);
            storeBuf[i * 5 + 2] = (encodedData[ 7] >> 1) | (encodedData[ 8] << 8) | (encodedData[ 9] << 17) | (encodedData[10] << 26);
            storeBuf[i * 5 + 3] = (encodedData[10] >> 6) | (encodedData[11] << 3) | (encodedData[12] << 12) | (encodedData[13] << 21) | (encodedData[14] << 30);
            storeBuf[i * 5 + 4] = (encodedData[14] >> 2) | (encodedData[15] << 7) | ((i == 0) ? (localSigns << 16) : (localSigns & 0xffff0000u));
        }
        tma_store_fence();
        __syncwarp();
    }

    // Return TMA copy bytes
    return enableCast ? (32 * (kNumSendUnrolls * sizeof(int4) * 8 * 10 / 16 / 8)):
                         (32 * (kNumSendUnrolls * sizeof(int4)));
}

template <int kNumLanes, int kNumSendUnrolls, int kNumRecvUnrolls>
__forceinline__ __device__ void logfmtCheckAmaxmin(uint8_t* metaBuffer, float2* sharedLogAmax,
                                                   float2* sharedLogAmin, int* sharedCastInfo,
                                                   const int laneId) {
    constexpr float kLogThreshold = 0;
    constexpr float kMinClip = 32; // `== log_2(2 ^ (2 ^ 5))`

    bool enableCast = true;
    if (laneId < kNumLanes) {
        // Calculate log amin/amax float
        auto amaxminData = reinterpret_cast<uint64_t*>(metaBuffer)[laneId];
        const auto& bf162Amaxmin = reinterpret_cast<__nv_bfloat162*>(&amaxminData);
        float logAmax[2], logAmin[2];
        #pragma unroll
        for (int i = 0; i < 2; ++ i) {
            auto amax = static_cast<float>(bf162Amaxmin[i].x);
            auto amin = static_cast<float>(bf162Amaxmin[i].y);
            logAmax[i] = log2f_approx(amax);
            logAmin[i] = amin == 0 ? logAmax[i] - kMinClip : fmaxf(log2f_approx(amin), logAmax[i] - kMinClip);
            enableCast = enableCast and logAmax[i] < kLogThreshold and logAmin[i] < logAmax[i];
        }
        sharedLogAmax[laneId] = make_float2(logAmax[0], logAmax[1]);
        sharedLogAmin[laneId] = make_float2(logAmin[0], logAmin[1]);
    }

    const auto& casted = warp_reduce_and<kNumSendUnrolls>(enableCast) ? 1u << (laneId / kNumRecvUnrolls): 0u;
    const auto& numCastedPrefix = __popc(warp_reduce_or<kNumRecvUnrolls, true>(casted) & ((1u << (laneId / kNumRecvUnrolls)) - 1));

    if (laneId < kNumLanes and laneId % kNumRecvUnrolls == 0)
        sharedCastInfo[laneId / kNumRecvUnrolls] = (numCastedPrefix << 1) | (casted ? 1u : 0u);
    __syncwarp();
}

template <int kNumRecvUnrolls>
__forceinline__ __device__ void decodeAndAccumulate(uint32_t* ldBuffer, float* accum,
                                                    const float& logAmax, const float& logAmin,
                                                    const bool& enableCast, const float& weight) {
    if (enableCast) {
        constexpr int kNumBits = 10;
        constexpr int kNumValues = 1 << (kNumBits - 1);

        const auto& step = (logAmax - logAmin) / static_cast<float>(kNumValues - 2);
        auto decode = [=](const uint32_t &encoded, const uint32_t &sign) {
            const auto decoded = encoded == 0 ? .0f : exp2f_approx((encoded - 1) * step + logAmin);
            return sign ? -decoded : decoded;
        };

        EP_STATIC_ASSERT(kNumRecvUnrolls == 2 or kNumRecvUnrolls == 4, "kNumRecvUnrolls == 2 or 4 only");
        #pragma unroll
        for (int i = 0; i < kNumRecvUnrolls / 2; ++ i) {
            uint32_t concatData[6];
            concatData[0] = ldBuffer[i * 5];
            #pragma unroll
            for (int k = 1; k < 5; ++ k)
                concatData[k] = (ldBuffer[i * 5 + k - 1] >> (32 - k * 5)) | (ldBuffer[i * 5 + k] << (k * 5));
            concatData[5] = ldBuffer[i * 5 + 4] >> 7;

            const uint32_t& localSigns = ldBuffer[i * 5 + 4] >> 16;
            #pragma unroll
            for (int k = 0; k < 5; ++ k) {
                accum[i * 16 + k * 3 + 0] += decode((concatData[k] >>  0) & 0x1ff, (localSigns >> (k * 3 + 0)) & 1) * weight;
                accum[i * 16 + k * 3 + 1] += decode((concatData[k] >>  9) & 0x1ff, (localSigns >> (k * 3 + 1)) & 1) * weight;
                accum[i * 16 + k * 3 + 2] += decode((concatData[k] >> 18) & 0x1ff, (localSigns >> (k * 3 + 2)) & 1) * weight;
            }
            accum[i * 16 + 15] += decode(concatData[5] & 0x1ff, (localSigns >> 15) & 1) * weight;
        }
    } else {
        #pragma unroll
        for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
            auto bf16Pack = *reinterpret_cast<__nv_bfloat162*>(ldBuffer + k);
            accum[k * 2 + 0] += static_cast<float>(bf16Pack.x) * weight;
            accum[k * 2 + 1] += static_cast<float>(bf16Pack.y) * weight;
        }
    }
}

// Clean next receive count buffer and notify atomic clean flag
__forceinline__ __device__ void cleanNextRecvCntBufAndNotify(
    int* nextRecvCntBuf,
    int nextRecvCntBufSize,
    int* atomicCleanFlag,
    int numExperts,
    int laneId) {
    if (!dP2pDisabled) {
        #pragma unroll
        for (int i = laneId; i < nextRecvCntBufSize; i += 32)
            nextRecvCntBuf[i] = 0;
    }
    // Notify before executing `int_p`
    __syncwarp();
    if (laneId == 0)
        atomic_add_release_global(atomicCleanFlag, numExperts);
}

// Send finish flag to destination rank
__forceinline__ __device__ void sendFinishFlag(
    int globalExpertIdx,
    int dstRank,
    int localExpertIdx,
    int currRank,
    int numLocalExperts,
    uint64_t recvFlagPtr,
    size_t recvFlagOffset,
    int* recvFlagBuf,
    int* rankMask,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms) {
    auto dstP2pPtr = ncclGetP2pPtr(
        recvFlagPtr, recvFlagOffset, currRank, dstRank, localExpertIdx, windows, devComms);

    if (not isRankMasked(rankMask, dstRank)) {
        if (dstP2pPtr == 0) {
            auto signalId = signalsBase + globalExpertIdx;
            auto localExpertIdxFlag = localExpertIdx;
            auto commId = localExpertIdxFlag / MAX_NCCL_GIN_CTX_PER_COMM;
            auto ctxId = localExpertIdxFlag % MAX_NCCL_GIN_CTX_PER_COMM;

            ncclGin net(devComms[commId], ctxId);
            ncclTeam world = ncclTeamWorld(devComms[commId]);
            auto ncclWindow = windows[commId];

            net.put(world,
                    dstRank,
                    ncclWindow,
                    recvFlagOffset,
                    ncclWindow,
                    0,
                    0,  // 0 bytes transfer
                    ncclGin_SignalAdd{signalId, 1},
                    ncclGin_None{},  // no counter
                    ncclCoopThread());
        } else {
            st_release_sys_global(reinterpret_cast<int*>(dstP2pPtr), 1);
        }
    }
}

// Wait for receive flag to arrive
__forceinline__ __device__ void waitForRecvFlag(
    int responsibleExpertIdx,
    int srcRank,
    int currRank,
    int numLocalExperts,
    size_t recvFlagOff,
    int* recvFlagBuf,
    int* rankMask,
    unsigned signalsBase,
    const ncclWindow_t* windows,
    ncclDevComm* devComms,
    int64_t* waitStats) {
    auto startTime = clock64();
    uint64_t waitRecvCost = 0;

    size_t srcOffset = recvFlagOff + responsibleExpertIdx * sizeof(int);
    auto srcP2pPtr = ncclGetP2pPtr(
        0x01, srcOffset, currRank, srcRank, (responsibleExpertIdx % numLocalExperts), windows, devComms);
    if (not isRankMasked(rankMask, srcRank)) {
        if (srcP2pPtr == 0) {
            uint64_t curValue;
            auto localExpertIdxWait = responsibleExpertIdx % numLocalExperts;
            auto commIdWait = localExpertIdxWait / MAX_NCCL_GIN_CTX_PER_COMM;
            auto ctxIdWait = localExpertIdxWait % MAX_NCCL_GIN_CTX_PER_COMM;
            ncclGin net(devComms[commIdWait], ctxIdWait);
            do {
                curValue = net.readSignal(signalsBase + responsibleExpertIdx);
            } while (curValue < 1                                                       // signal not arrived
                     && (waitRecvCost = clock64() - startTime) <= NUM_TIMEOUT_CYCLES  // not timeout
            );
            net.resetSignal(signalsBase + responsibleExpertIdx);
        } else {
            while (ld_acquire_sys_global(recvFlagBuf + responsibleExpertIdx) == 0  // recv not ready
                   && (waitRecvCost = clock64() - startTime) <= NUM_TIMEOUT_CYCLES   // not timeout
            );
        }
    }
    // Mask rank if timeout
    if (waitRecvCost > NUM_TIMEOUT_CYCLES) {
        printf("Warning: NCCL EP timeout for combine receive, rank %d, local_expert_idx %d, src_rank %d\n",
               currRank,
               responsibleExpertIdx % numLocalExperts,
               srcRank);
        if (rankMask == nullptr)
            trap();
        atomicExch(rankMask + srcRank, 1);
    }

    if (waitStats != nullptr) {
        atomicAdd(reinterpret_cast<unsigned long long*>(waitStats + srcRank), waitRecvCost);
    }
}

// Send token via RDMA
__forceinline__ __device__ void sendTokenViaRdma(
    int globalExpertIdx,
    int dstRank,
    int localExpertIdx,
    int currRank,
    int numRanks,
    int maxTokensPerRank,
    int tokenIdx,
    int srcIdx,
    size_t sendOff,
    size_t recvOff,
    size_t numBytesPerSlot,
    int hidden,
    ncclDevComm* devComms,
    const ncclWindow_t* windows) {
    const auto expectedDstOffset = recvOff + (globalExpertIdx * maxTokensPerRank + srcIdx) * numBytesPerSlot;
    const auto expectedBufOffset = sendOff +
        (localExpertIdx * numRanks * maxTokensPerRank * numBytesPerSlot) +
        tokenIdx * numBytesPerSlot;

    auto commId = localExpertIdx / MAX_NCCL_GIN_CTX_PER_COMM;
    auto ctxId = localExpertIdx % MAX_NCCL_GIN_CTX_PER_COMM;
    ncclGin net(devComms[commId], ctxId);
    ncclTeam world = ncclTeamWorld(devComms[commId]);
    auto ncclWindow = windows[commId];
    net.put(world,
            dstRank,
            ncclWindow,
            expectedDstOffset,
            ncclWindow,
            expectedBufOffset,
            hidden * sizeof(nv_bfloat16),
            ncclGin_None{},  // no signal
            ncclGin_None{},  // no counter
            ncclCoopThread());
}

// Process and send a single token with TMA copy and optional RDMA
template<bool kUseLogFMT, int kHidden, int kNumSendUnrolls, int kNumStages, int kNumPrefetch,
         typename TmaBuffersT, typename FullBarriersT, typename TmaLoadAndArriveT, typename GetNumTmaBytesT>
__forceinline__ __device__ void processAndSendToken(
    int tokenIdx,
    int srcIdx,
    const int4* srcDataInt4Ptr,
    int64_t sendBufPtr,
    uint64_t recvPtr,
    size_t recvOff,
    size_t sendOff,
    int globalExpertIdx,
    int dstRank,
    int localExpertIdx,
    int currRank,
    int numRanks,
    int maxTokensPerRank,
    size_t numBytesPerSlot,
    int hidden,
    int hiddenBf16Int4,
    int hiddenBf16Int4Pad,
    int kNumMetaBytes,
    int kNumTMABufferBytes,
    bool zeroCopy,
    uint64_t dstP2pPtr,
    TmaBuffersT tmaBuffers,
    FullBarriersT fullBarriers,
    nv_bfloat162* metaBuffers,
    uint32_t& tmaPhase,
    TmaLoadAndArriveT& tmaLoadAndArrive,
    GetNumTmaBytesT& getNumTmaBytes,
    int laneId,
    ncclDevComm* devComms,
    const ncclWindow_t* windows) {
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    const int kNumIters = hiddenBf16Int4Pad / (32 * kNumSendUnrolls);

    if (not zeroCopy or dstP2pPtr != 0) {
        // Read from `copySrcPtr` and copy into `copyDstPtr`
        const auto copySrcPtr = zeroCopy ? reinterpret_cast<int4*>(sendBufPtr) : srcDataInt4Ptr;
        const auto copyDstPtr =
            dstP2pPtr == 0 ? reinterpret_cast<int4*>(sendBufPtr) : reinterpret_cast<int4*>(dstP2pPtr);

        // Prefetch
        if (elect_one_sync())
            tmaLoadAndArrive(0, copySrcPtr, getNumTmaBytes(0));
        __syncwarp();

        int tmaOffsetBytes = kNumMetaBytes;
        #pragma unroll
        for (int i = laneId * kNumSendUnrolls, iterIdx = 0; i < hiddenBf16Int4Pad; i += 32 * kNumSendUnrolls, ++iterIdx) {
            // Load the next iteration
            const int& stageIdx = iterIdx % kNumStages;
            const int& nextStageIdx = (iterIdx + 1) % kNumStages;
            if (iterIdx + 1 < kNumIters and elect_one_sync()) {
                tma_store_wait<kNumStages - kNumPrefetch - 1>();
                const auto& offsetInt4 = i + 32 * kNumSendUnrolls;
                tmaLoadAndArrive(nextStageIdx, copySrcPtr + offsetInt4, getNumTmaBytes(offsetInt4));
            }
            __syncwarp();

            // Wait the current TMA arrival
            EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
            mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);
            if constexpr (kUseLogFMT) {
                // Cast if possible
                constexpr int kNumInt4PerDivision = 128 / kNumElemsPerInt4;
                int numTmaBytes = logfmtEncode<kNumSendUnrolls>(
                    tmaBuffers[stageIdx],
                    // NOTES: only the leader lane will write the result
                    (i % kNumInt4PerDivision == 0) ? metaBuffers + i / kNumInt4PerDivision : nullptr,
                    laneId);
                if (elect_one_sync())
                    tma_store_1d(
                        tmaBuffers[stageIdx], reinterpret_cast<uint8_t*>(copyDstPtr) + tmaOffsetBytes, numTmaBytes);
                tmaOffsetBytes += numTmaBytes;
            } else {
                // BF16 original values
                if (elect_one_sync())
                    tma_store_1d(tmaBuffers[stageIdx], copyDstPtr + i, getNumTmaBytes(i));
            }
            __syncwarp();
        }

        // Store metadata (min/max values) for LogFMT
        if constexpr (kUseLogFMT) {
            if (elect_one_sync())
                tma_store_1d(metaBuffers, copyDstPtr, kNumMetaBytes);
        }

        // Flush all stores
        tma_store_wait<0>();
        __syncwarp();
    }

    // Issue RDMA
    // NOTES: for zero-copy mode, we assume the data is already in the send buffer
    if (dstP2pPtr == 0) {
        if (laneId == 0) {
            sendTokenViaRdma(
                globalExpertIdx, dstRank, localExpertIdx, currRank, numRanks,
                maxTokensPerRank, tokenIdx, srcIdx, sendOff, recvOff,
                numBytesPerSlot, hidden, devComms, windows);
        }
    }
}

template <bool kUseLogFMT, int kHidden, int kNumMaxTopk, int kNumMaxUnrolls>
__global__ __launch_bounds__(1024, 1) void combine(// INPUT
                                                   const void* inData,
                                                   const int* srcInfo,
                                                   const int64_t* layoutRange,
                                                   const int64_t* inTopkIdx,
                                                   const float* topkWeights,
                                                   int* rankMask,
                                                   // OUTPUT
                                                   void* outData,
                                                   // INTERMEDIATE
                                                   void* sendBuf,
                                                   void* recvBuf,
                                                   int* recvFlagBuf,
                                                   size_t sendOff,
                                                   size_t recvOff,
                                                   size_t recvFlagOff,
                                                   int* atomicCleanFlag,
                                                   int* nextRecvCntBuf,
                                                   int nextRecvCntBufSize,
                                                   int64_t* waitStats,
                                                   // CONFIG
                                                   int numCombinedTokens,
                                                   int hidden,
                                                   int numTopk,
                                                   int maxTokensPerRank,
                                                   int numExperts,
                                                   int currRank,
                                                   int numRanks,
                                                   int numWarpGroups,
                                                   int numWarpsPerGroup,
                                                   int phases,
                                                   bool zeroCopy,
                                                   int numComms,
                                                   ncclDevComm* devComms,
                                                   const ncclWindow_t* windows,
                                                   unsigned signalsBase) {
    const auto smId = __shfl_sync(0xffffffff, static_cast<int>(blockIdx.x), 0);
    const auto numSms = __shfl_sync(0xffffffff, static_cast<int>(gridDim.x), 0);
    const auto threadId = static_cast<int>(threadIdx.x);
    const auto numThreads = __shfl_sync(0xffffffff, static_cast<int>(blockDim.x), 0);
    const auto warpId = __shfl_sync(0xffffffff, threadId / 32, 0), laneId = get_lane_id();
    const auto numLocalExperts = numExperts / numRanks;
    const auto warpGroupId = warpId / numWarpsPerGroup;
    const auto subWarpId = warpId % numWarpsPerGroup;
    const auto responsibleExpertIdx = smId * numWarpGroups + warpGroupId;

    extern __shared__ __align__(1024) uint8_t smemBuffer[];

    // Data type staffs
    constexpr int kNumElemsPerInt4 = sizeof(int4) / sizeof(nv_bfloat16);
    constexpr int64_t hiddenBf16Int4 = kHidden / kNumElemsPerInt4;

    // Use different unroll factors for send and recv phases
    constexpr int kNumSendUnrolls = kHidden % (32 * 4 * sizeof(int4) / sizeof(nv_bfloat16)) == 0 ? 4 : 2;
    constexpr int kNumRecvUnrolls = 2;
    constexpr int hiddenBf16Int4Pad = align(static_cast<int>(hiddenBf16Int4), 32 * kNumSendUnrolls);
    EP_STATIC_ASSERT(kHidden % (32 * 2 * sizeof(int4) / sizeof(nv_bfloat16)) == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls <= kNumMaxUnrolls and kNumRecvUnrolls <= kNumMaxUnrolls, "Invalid unrolls");
    EP_STATIC_ASSERT(hiddenBf16Int4 % kNumSendUnrolls == 0, "Invalid hidden");
    EP_STATIC_ASSERT(kNumSendUnrolls >= kNumRecvUnrolls, "Invalid unroll factors");

    // Message package
    EP_STATIC_ASSERT(kHidden % 128 == 0, "Invalid hidden");
    constexpr int kNumDivisions = kHidden / 128;
    constexpr int kNumMetaBytes = kNumDivisions * sizeof(nv_bfloat162);
    constexpr size_t numBytesPerSlot = kHidden * sizeof(nv_bfloat16) + kNumMetaBytes;
    EP_STATIC_ASSERT(numBytesPerSlot % sizeof(int4) == 0, "Invalid vectorization");

    // Sending phase
    if ((phases & LOW_LATENCY_SEND_PHASE) == 0)
        goto LOW_LATENCY_COMBINE_RECV;

    // Clean up next buffer
    if (smId == 0 and warpGroupId == 0 and subWarpId == 0) {
        cleanNextRecvCntBufAndNotify(
            nextRecvCntBuf, nextRecvCntBufSize, atomicCleanFlag, numExperts, laneId);
    }

    // Issue IBGDA sends
    if (responsibleExpertIdx < numExperts) {
        const auto dstRank = responsibleExpertIdx / numLocalExperts;
        const auto localExpertIdx = responsibleExpertIdx % numLocalExperts;
        const auto globalExpertIdx = currRank * numLocalExperts + localExpertIdx;
        const auto layout = __ldg(layoutRange + localExpertIdx * numRanks + dstRank);
        const auto srcDataInt4 = static_cast<const int4*>(inData) +
                localExpertIdx * numRanks * maxTokensPerRank * hiddenBf16Int4;
        const auto localSrcInfo = srcInfo + localExpertIdx * numRanks * maxTokensPerRank;
        const auto sendBufUint8 = static_cast<uint8_t*>(sendBuf) +
                localExpertIdx * numRanks * maxTokensPerRank * numBytesPerSlot;

        // Unpack layout
        int offset, numTokensToSend;
        unpack2(layout, numTokensToSend, offset);

        // TMA stuffs
        constexpr int kNumTMABufferBytes = sizeof(int4) * 32 * kNumSendUnrolls;
        constexpr int kNumStages = 3;
        constexpr int kNumPrefetch = 1;
        EP_STATIC_ASSERT(kNumStages == 3 and kNumPrefetch == 1, "Invalid stages");

        auto smemPtr = smemBuffer + warpId * (kNumStages * (kNumTMABufferBytes + 16) + kNumMetaBytes);
        uint32_t tmaPhase = 0;
        auto tmaBuffers   = PatternVisitor([=](const int& i) { return reinterpret_cast<int4*>(smemPtr + i * (kNumTMABufferBytes + 16)); });
        auto fullBarriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smemPtr + i * (kNumTMABufferBytes + 16) + kNumTMABufferBytes); });
        auto metaBuffers  = kUseLogFMT ? reinterpret_cast<nv_bfloat162*>(smemPtr + kNumStages * (kNumTMABufferBytes + 16)) : nullptr;
        EP_STATIC_ASSERT(kNumSendUnrolls * kNumStages <= 12, "TMA buffer size exceed limit");

        // Initialize m-barriers
        if (laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            fence_barrier_init();
        }
        __syncwarp();

        auto tmaLoadAndArrive = [&](const int& stageIdx, const int4* gmemPtr, const int& numBytes) {
            tma_load_1d(tmaBuffers[stageIdx], gmemPtr, fullBarriers[stageIdx], numBytes);
            mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numBytes);
        };
        auto getNumTmaBytes = [&](const int& offsetInt4) {
            return min(kNumTMABufferBytes, static_cast<int>((hiddenBf16Int4 - offsetInt4) * sizeof(int4)));
        };

        // Issue IBGDA send
        if (not isRankMasked<true>(rankMask, dstRank)) {
            for (int tokenIdx = offset + subWarpId; tokenIdx < offset + numTokensToSend; tokenIdx += numWarpsPerGroup) {
                const auto srcDataInt4Ptr = srcDataInt4 + tokenIdx * hiddenBf16Int4;
                const auto sendBufTypeRow = reinterpret_cast<int*>(sendBufUint8 + tokenIdx * numBytesPerSlot);
                const auto sendBufDataRow = reinterpret_cast<uint8_t*>(sendBufTypeRow);

                // Copy directly to local rank, or copy to buffer and issue RDMA
                const auto srcIdx = __shfl_sync(0xffffffff, __ldg(localSrcInfo + tokenIdx), 0);
                const auto sendBufPtr = reinterpret_cast<int64_t>(sendBufDataRow);
                const auto recvPtr = reinterpret_cast<uint64_t>(recvBuf) +
                    (globalExpertIdx * maxTokensPerRank + srcIdx) * numBytesPerSlot;

                const auto expectedDstOffset =
                    recvOff + (globalExpertIdx * maxTokensPerRank + srcIdx) * numBytesPerSlot;
                const auto dstP2pPtr =
                    ncclGetP2pPtr(recvPtr, expectedDstOffset, currRank, dstRank, localExpertIdx, windows, devComms);

                processAndSendToken<kUseLogFMT, kHidden, kNumSendUnrolls, kNumStages, kNumPrefetch>(
                    tokenIdx, srcIdx, srcDataInt4Ptr, sendBufPtr, recvPtr,
                    recvOff, sendOff, globalExpertIdx, dstRank, localExpertIdx,
                    currRank, numRanks, maxTokensPerRank, numBytesPerSlot,
                    hidden, hiddenBf16Int4, hiddenBf16Int4Pad, kNumMetaBytes,
                    kNumTMABufferBytes, zeroCopy,
                    dstP2pPtr, tmaBuffers, fullBarriers, metaBuffers, tmaPhase,
                    tmaLoadAndArrive, getNumTmaBytes, laneId, devComms, windows);
            }
        }

        // Put the finishing flag
        EP_DEVICE_ASSERT(numWarpsPerGroup > 1 and numWarpGroups < 16);
        asm volatile("bar.sync %0, %1;" :: "r"(warpGroupId + 1), "r"(numWarpsPerGroup * 32));
        if (subWarpId == 1 and laneId == 0) {
            while (ld_acquire_global(atomicCleanFlag) == 0);
            auto recvFlagPtr = reinterpret_cast<uint64_t>(recvFlagBuf + globalExpertIdx);
            size_t recvFlagOffset = recvFlagOff + globalExpertIdx * sizeof(int);

            sendFinishFlag(
                globalExpertIdx, dstRank, localExpertIdx, currRank, numLocalExperts,
                recvFlagPtr, recvFlagOffset, recvFlagBuf, rankMask, signalsBase,
                windows, devComms);
            atomic_add_release_global(atomicCleanFlag, -1);
        }
        __syncwarp();

        // Destroy m-barriers
        if (laneId < kNumStages) {
            mbarrier_inval(fullBarriers[laneId]);
            fence_barrier_init();
        }
        __syncwarp();
    }

// Receiving phase
LOW_LATENCY_COMBINE_RECV:
    if ((phases & LOW_LATENCY_RECV_PHASE) == 0)
        return;

    // Wait all ranks to arrive
    if (responsibleExpertIdx < numExperts) {
        EP_DEVICE_ASSERT(numWarpsPerGroup > 1);
        if (subWarpId == 0 and laneId == 0) {
            const auto srcRank = responsibleExpertIdx / numLocalExperts;
            waitForRecvFlag(
                responsibleExpertIdx, srcRank, currRank, numLocalExperts,
                recvFlagOff, recvFlagBuf, rankMask, signalsBase,
                windows, devComms, waitStats);
        }
    }
    cg::this_grid().sync();

    // Reassign warp groups
    constexpr int kMaxNumGroups = 2;
    const int numDecodeWarps = hiddenBf16Int4Pad / (kNumRecvUnrolls * 32);
    const int numGroups = min(kMaxNumGroups, (numThreads / 32) / (numDecodeWarps + 1));
    const int decodeWarpIdx = __shfl_sync(0xffffffff, warpId % (numDecodeWarps + 1), 0);
    const int groupIdx = __shfl_sync(0xffffffff, warpId / (numDecodeWarps + 1), 0);
    EP_STATIC_ASSERT(kHidden % (32 * kNumElemsPerInt4) == 0, "Invalid vectorization");
    EP_DEVICE_ASSERT(numTopk <= 32);
    EP_DEVICE_ASSERT(numGroups > 0);

    if (groupIdx < numGroups) {
        constexpr int kNumStages = 3;
        constexpr int kNumTMABufferBytes = 16 * 2 + kHidden * 2;
        constexpr int kNumBF16PerWarpBytes = 32 * kNumRecvUnrolls * kNumElemsPerInt4 * 2;
        constexpr int kNumLogFMTPerWarpBytes = kNumBF16PerWarpBytes / 16 * 10;
        constexpr int kNumDivisionBytes = kNumDivisions * sizeof(uint32_t);
        constexpr int kNumBytesPerGroup = kNumStages * kNumTMABufferBytes + kHidden * 2 + kNumStages * kNumDivisionBytes * 3;

        // Reallocate shared memory
        const auto smemGroupBuffer = smemBuffer + kNumBytesPerGroup * groupIdx;
        auto fullBarriers  = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes); });
        auto emptyBarriers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint64_t*>(smemGroupBuffer + i * kNumTMABufferBytes + 8); });
        auto tmaLdBuffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint8_t* >(smemGroupBuffer + i * kNumTMABufferBytes + 16); });
        auto tmaStBuffers = PatternVisitor([=](const int& i) { return reinterpret_cast<uint32_t*>(smemGroupBuffer + kNumStages * kNumTMABufferBytes + i * kNumBF16PerWarpBytes); });

        // Redundant when logfmt is disabled
        const auto smemGroupPtr = smemGroupBuffer + kNumStages * kNumTMABufferBytes + kHidden * 2;
        auto logAmaxBuffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smemGroupPtr + i * kNumDivisionBytes); });
        auto logAminBuffers  = PatternVisitor([=](const int& i) { return reinterpret_cast<float*>(smemGroupPtr + kNumStages * kNumDivisionBytes + i * kNumDivisionBytes); });
        auto castInfoBuffers = PatternVisitor([=](const int& i) { return reinterpret_cast<int*>  (smemGroupPtr + kNumStages * kNumDivisionBytes * 2 + i * kNumDivisionBytes); });

        uint32_t tmaPhase = 0;
        EP_STATIC_ASSERT(kNumStages < 32, "Too many stages");
        if (decodeWarpIdx == numDecodeWarps)
            tmaPhase = (1 << kNumStages) - 1;

        // Initialize m-barriers
        if (decodeWarpIdx == numDecodeWarps and laneId < kNumStages) {
            mbarrier_init(fullBarriers[laneId], 1);
            mbarrier_init(emptyBarriers[laneId], numDecodeWarps);
        }
        asm volatile("bar.sync %0, %1;" :: "r"(groupIdx + 1), "r"((numDecodeWarps + 1) * 32));

        int stageIdx = 0, topkIdxByLane = 0;
        EP_STATIC_ASSERT(kNumMaxTopk <= 32, "Invalid number of topks");
        if (decodeWarpIdx == numDecodeWarps) {
            // TMA load warp
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens; tokenIdx += numSms * numGroups) {
                if (laneId < numTopk)
                    topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));
                for (int i = 0; i < numTopk; ++ i) {
                    int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                    if (topkIdxReg < 0)
                        continue;
                    if (isRankMasked(rankMask, topkIdxReg / numLocalExperts))
                        continue;

                    mbarrier_wait<true>(emptyBarriers[stageIdx], tmaPhase, stageIdx);
                    auto recvBufData = static_cast<uint8_t*>(recvBuf) + (topkIdxReg * maxTokensPerRank + tokenIdx) * numBytesPerSlot;
                    if constexpr (kUseLogFMT) {
                        logfmtCheckAmaxmin<kNumDivisions / 2, kNumSendUnrolls, kNumRecvUnrolls>(
                            recvBufData,
                            reinterpret_cast<float2*>(logAmaxBuffers[stageIdx]),
                            reinterpret_cast<float2*>(logAminBuffers[stageIdx]),
                            castInfoBuffers[stageIdx],
                            laneId);
                    }
                    if (elect_one_sync()) {
                        int numCasted = 0;
                        if constexpr (kUseLogFMT) {
                            const auto& info = castInfoBuffers[stageIdx][numDecodeWarps - 1];
                            numCasted = (info >> 1) + (info & 1);
                        }
                        int numTmaBytes = numCasted * kNumLogFMTPerWarpBytes + (numDecodeWarps - numCasted) * kNumBF16PerWarpBytes;
                        tma_load_1d(tmaLdBuffers[stageIdx], recvBufData + (kUseLogFMT ? kNumMetaBytes : 0), fullBarriers[stageIdx], numTmaBytes);
                        mbarrier_arrive_and_expect_tx(fullBarriers[stageIdx], numTmaBytes);
                    }
                    __syncwarp();
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
            }
        } else {
            // Reduction warps
            float topkWeightsByLane;
            for (int tokenIdx = smId + numSms * groupIdx; tokenIdx < numCombinedTokens; tokenIdx += numSms * numGroups) {
                if (laneId < numTopk) {
                    topkIdxByLane = static_cast<int>(__ldg(inTopkIdx + tokenIdx * numTopk + laneId));
                    topkWeightsByLane = __ldg(topkWeights + tokenIdx * numTopk + laneId);
                }
                __syncwarp();

                float combinedData[kNumElemsPerInt4 * kNumRecvUnrolls] = {0.0f};
                for (int i = 0; i < numTopk; ++ i) {
                    int topkIdxReg = __shfl_sync(0xffffffff, topkIdxByLane, i);
                    if (topkIdxReg < 0)
                        continue;
                    if (isRankMasked(rankMask, topkIdxReg / numLocalExperts))
                        continue;
                    const auto& topkWeight = __shfl_sync(0xffffffff, topkWeightsByLane, i);

                    mbarrier_wait<true>(fullBarriers[stageIdx], tmaPhase, stageIdx);
                    if constexpr (kUseLogFMT) {
                        const auto& info = castInfoBuffers[stageIdx][decodeWarpIdx];
                        bool enableCast = info & 1;
                        int numCastedPrefix = info >> 1;
                        int tmaOffset = kNumLogFMTPerWarpBytes * numCastedPrefix + kNumBF16PerWarpBytes * (decodeWarpIdx - numCastedPrefix);
                        int divisionIdx = decodeWarpIdx * (kNumRecvUnrolls * 2) + laneId * kNumRecvUnrolls / 16;
                        decodeAndAccumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tmaLdBuffers[stageIdx] + tmaOffset +
                                                        (enableCast ? kNumLogFMTPerWarpBytes : kNumBF16PerWarpBytes) / 32 * laneId),
                            combinedData,
                            logAmaxBuffers[stageIdx][divisionIdx],
                            logAminBuffers[stageIdx][divisionIdx],
                            enableCast,
                            topkWeight);
                    } else {
                        int tmaOffset = kNumBF16PerWarpBytes * decodeWarpIdx;
                        decodeAndAccumulate<kNumRecvUnrolls>(
                            reinterpret_cast<uint32_t*>(tmaLdBuffers[stageIdx] + tmaOffset + kNumBF16PerWarpBytes / 32 * laneId),
                            combinedData,
                            0,
                            0,
                            false,
                            topkWeight);
                    }

                    if (elect_one_sync())
                        mbarrier_arrive(emptyBarriers[stageIdx]);
                    stageIdx = (stageIdx + 1) % kNumStages;
                }
                tma_store_wait<0>();

                #pragma unroll
                for (int k = 0; k < kNumRecvUnrolls * 4; ++ k) {
                    auto combinedPack = __nv_bfloat162(combinedData[k * 2], combinedData[k * 2 + 1]);
                    tmaStBuffers[decodeWarpIdx][kNumRecvUnrolls * 4 * laneId + k] = *reinterpret_cast<uint32_t*>(&combinedPack);
                }
                tma_store_fence();
                if (elect_one_sync()) {
                    tma_store_1d(tmaStBuffers[decodeWarpIdx],
                                 static_cast<int4*>(outData) + tokenIdx * hiddenBf16Int4 + decodeWarpIdx * kNumRecvUnrolls * 32,
                                 kNumBF16PerWarpBytes);
                }
                __syncwarp();
            }
        }
    }
}

// Constants for combine kernel template parameters
constexpr int kCombineMaxTopk = 9;
constexpr int kCombineMaxUnrolls = 4;

void combine(const void* inData,
             const int* srcInfo,
             const int64_t* layoutRange,
             const int64_t* inTopkIdx,
             const float* topkWeights,
             void* outData,
             void* sendBuf,
             void* recvBuf,
             int* recvFlagBuf,
             size_t sendOff,
             size_t recvOff,
             size_t recvFlagOff,
             int* nextRecvCntBuf,
             int nextRecvCntBufSize,
             int64_t* waitStats,
             int numCombinedTokens,
             int hidden,
             int maxTokensPerRank,
             int numTopk,
             int numExperts,
             int currRank,
             int numRanks,
             bool useLogfmt,
             int phases,
             bool zeroCopy,
             int numComms,
             ncclDevComm* devComms,
             const ncclWindow_t* windows,
             unsigned signalsBase,
             void* workspace,
             int numDeviceSms,
             cudaStream_t stream) {
    const int numWarpGroups = ceil_div(numExperts, numDeviceSms);
    const int numWarpsPerGroup = 32 / numWarpGroups;
    const int numRecvPerSm = ceil_div(numCombinedTokens, numDeviceSms);
    EP_HOST_ASSERT(numWarpGroups > 0 and numWarpsPerGroup > 0 and numRecvPerSm >= 0);

    const auto numWarps = numWarpGroups * numWarpsPerGroup;
    const auto numSms = max(ceil_div(numExperts, numWarpGroups),
                             numRecvPerSm == 0 ? 1 : ceil_div(numCombinedTokens, numRecvPerSm));

    // Check workspace
    auto atomicCleanFlag = static_cast<int*>(workspace);
    EP_HOST_ASSERT(sizeof(int) <= NUM_WORKSPACE_BYTES);
    EP_HOST_ASSERT(numTopk <= kCombineMaxTopk);

    // Online cast cannot use zero-copy
    EP_HOST_ASSERT(not (zeroCopy and useLogfmt));

    constexpr int kNumStages = 3;
    constexpr int kMaxNumGroups = 2;

    // Send buffer size
    const int numMetaBytes = hidden / 128 * 4;
    const int numSendTmaBytes = 32 * sizeof(int4) * kCombineMaxUnrolls + 16;
    const int smemSendSize = numWarps * (kNumStages * numSendTmaBytes + numMetaBytes);

    // Receive buffer size
    const int numRecvTmaBytes = 16 + hidden * 2;
    const int smemRecvSize = kMaxNumGroups * (kNumStages * numRecvTmaBytes + hidden * 2 + kNumStages * numMetaBytes * 3);

    // Total requirement
    const int smem_size = max(smemSendSize, smemRecvSize);

#define COMBINE_LAUNCH_CASE(hidden) { \
if (useLogfmt) { \
    SET_SHARED_MEMORY_FOR_TMA((combine<true, hidden, kCombineMaxTopk, kCombineMaxUnrolls>)); \
    LAUNCH_KERNEL(&cfg, combine<true, hidden, kCombineMaxTopk, kCombineMaxUnrolls>, \
                  inData, \
                  srcInfo, \
                  layoutRange, \
                  inTopkIdx, \
                  topkWeights, \
                  /*rankMask=*/nullptr, \
                  outData, \
                  sendBuf, \
                  recvBuf, \
                  recvFlagBuf, \
                  sendOff, \
                  recvOff, \
                  recvFlagOff, \
                  atomicCleanFlag, \
                  nextRecvCntBuf, \
                  nextRecvCntBufSize, \
                  waitStats, \
                  numCombinedTokens, \
                  hidden, \
                  numTopk, \
                  maxTokensPerRank, \
                  numExperts, \
                  currRank, \
                  numRanks, \
                  numWarpGroups, \
                  numWarpsPerGroup, \
                  phases, \
                  zeroCopy, \
                  numComms, \
                  devComms, \
                  windows, \
                  signalsBase); \
} else { \
    SET_SHARED_MEMORY_FOR_TMA((combine<false, hidden, kCombineMaxTopk, kCombineMaxUnrolls>)); \
    LAUNCH_KERNEL(&cfg, combine<false, hidden, kCombineMaxTopk, kCombineMaxUnrolls>, \
                  inData, \
                  srcInfo, \
                  layoutRange, \
                  inTopkIdx, \
                  topkWeights, \
                  /*rankMask=*/nullptr, \
                  outData, \
                  sendBuf, \
                  recvBuf, \
                  recvFlagBuf, \
                  sendOff, \
                  recvOff, \
                  recvFlagOff, \
                  atomicCleanFlag, \
                  nextRecvCntBuf, \
                  nextRecvCntBufSize, \
                  waitStats, \
                  numCombinedTokens, \
                  hidden, \
                  numTopk, \
                  maxTokensPerRank, \
                  numExperts, \
                  currRank, \
                  numRanks, \
                  numWarpGroups, \
                  numWarpsPerGroup, \
                  phases, \
                  zeroCopy, \
                  numComms, \
                  devComms, \
                  windows, \
                  signalsBase); \
} } break

    SETUP_LAUNCH_CONFIG(numSms, numWarps * 32, stream);
    SWITCH_HIDDEN(COMBINE_LAUNCH_CASE);
#undef COMBINE_LAUNCH_CASE
}

} // namespace internode_ll

} // namespace nccl_ep
