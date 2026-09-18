/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_GENERAL_GENK_H_
#define NCCL_DEVICE_GENERAL_GENK_H_

#include "sym_kernels.h"
#include "bitops.h"
#include "flow_reduce.cuh"
#include "../work.cuh"
#include "../common_kernel.h"

template <ncclFlowProtocol Protocol>
struct ncclGenkProtocolTraits;

template <>
struct ncclGenkProtocolTraits<ncclFlowProtocolSimple> {
  static constexpr bool SupportsDirect = false; // Temporarily disabled until the synchronization is sorted out.
  using DirectSend = ncclFlowDirectSend;
  using DirectRecv = ncclFlowDirectRecv;
  static constexpr int SlotsPerRingProcess = ncclGenkRingSlotsPerProcess;
  static constexpr int RingFifoSlots = ncclGenkRingSimpleFifoSlots;
  static constexpr size_t RingChannelWireBytes = ncclGenkRingSimpleChannelSize;
  static constexpr size_t TreeChannelWireBytes = ncclGenkTreeSimpleChannelSize;
  static constexpr size_t RingChunkPayloadBytes = ncclGenkRingSimpleChannelSize / RingFifoSlots * SlotsPerRingProcess;
  static constexpr size_t TreeChunkPayloadBytes =
    ncclGenkTreeSimpleChannelSize / (ncclGenkTreeFifoSlots * ncclGenkTreeParts * ncclGenkTreeSlots);

  NCCL_DEVICE_INLINE static int treeReduceWorkerWarps(int nWorkerWarps) {
    int reduceWorkerWarps = nWorkerWarps / 2;
    if (reduceWorkerWarps >= 8) reduceWorkerWarps += 2;
    return max(2, min(reduceWorkerWarps, nWorkerWarps - 2));
  }

  NCCL_DEVICE_INLINE static ncclDevResourceHandle ringBuffer(ncclGenkDevComm const& genk) {
    return genk.ringBuffer;
  }
  NCCL_DEVICE_INLINE static ncclDevResourceHandle treeBuffer(ncclGenkDevComm const& genk) {
    return genk.treeBuffer;
  }
};

template <>
struct ncclGenkProtocolTraits<ncclFlowProtocolLL> {
  static constexpr bool SupportsDirect = false;
  using DirectSend = ncclFlowSend;
  using DirectRecv = ncclFlowRecv;
  static constexpr int SlotsPerRingProcess = 1;
  static constexpr int RingFifoSlots = ncclGenkRingLLFifoSlots;
  static constexpr size_t RingChannelWireBytes = ncclGenkRingLLChannelWireBytes;
  static constexpr size_t TreeChannelWireBytes = ncclGenkTreeLLChannelWireBytes;
  static constexpr size_t RingChunkPayloadBytes = ncclGenkRingLLChunkPayloadBytes;
  static constexpr size_t TreeChunkPayloadBytes = ncclGenkTreeLLChunkPayloadBytes;

  NCCL_DEVICE_INLINE static int treeReduceWorkerWarps(int nWorkerWarps) {
    return max(2, min(nWorkerWarps * 7 / 10, nWorkerWarps - 2));
  }

  NCCL_DEVICE_INLINE static ncclDevResourceHandle ringBuffer(ncclGenkDevComm const& genk) {
    return genk.ringLLBuffer;
  }
  NCCL_DEVICE_INLINE static ncclDevResourceHandle treeBuffer(ncclGenkDevComm const& genk) {
    return genk.treeLLBuffer;
  }
};

template <>
struct ncclGenkProtocolTraits<ncclFlowProtocolLL128> {
  static constexpr bool SupportsDirect = false;
  using DirectSend = ncclFlowSend;
  using DirectRecv = ncclFlowRecv;
  static constexpr int SlotsPerRingProcess = 1;
  static constexpr int RingFifoSlots = ncclGenkRingLL128FifoSlots;
  static constexpr size_t RingChannelWireBytes = ncclGenkRingLL128ChannelWireBytes;
  static constexpr size_t TreeChannelWireBytes = ncclGenkTreeLL128ChannelWireBytes;
  static constexpr size_t RingChunkPayloadBytes = ncclGenkRingLL128ChunkPayloadBytes;
  static constexpr size_t TreeChunkPayloadBytes = ncclGenkTreeLL128ChunkPayloadBytes;

  NCCL_DEVICE_INLINE static int treeReduceWorkerWarps(int nWorkerWarps) {
    return max(2, min(nWorkerWarps * 7 / 10, nWorkerWarps - 2));
  }

  NCCL_DEVICE_INLINE static ncclDevResourceHandle ringBuffer(ncclGenkDevComm const& genk) {
    return genk.ringLL128Buffer;
  }
  NCCL_DEVICE_INLINE static ncclDevResourceHandle treeBuffer(ncclGenkDevComm const& genk) {
    return genk.treeLL128Buffer;
  }
};

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE ncclFlowConfig<0> ncclGenkRingLsaConfig(ncclGenkDevComm const& genk) {
  using Traits = ncclGenkProtocolTraits<Protocol>;
  return {ncclGetResourceBuffer(genk.devComm, Traits::ringBuffer(genk)),
          ncclGetResourceBuffer(genk.devComm, genk.connState), Traits::RingChannelWireBytes, Traits::RingFifoSlots,
          ncclGenkRingConnections};
}

NCCL_DEVICE_INLINE ncclSymPtr<ncclFlowConnState> ncclGenkTreeConnState(ncclGenkDevComm const& genk) {
  ncclSymPtr<ncclFlowConnState> state = ncclGetResourceBuffer(genk.devComm, genk.connState);
  return state + genk.nResourceChannels * ncclGenkRingConnections;
}

template <ncclFlowProtocol Protocol>
NCCL_DEVICE_INLINE ncclFlowConfig<0> ncclGenkTreeLsaConfig(ncclGenkDevComm const& genk) {
  using Traits = ncclGenkProtocolTraits<Protocol>;
  return {ncclGetResourceBuffer(genk.devComm, Traits::treeBuffer(genk)), ncclGenkTreeConnState(genk),
          Traits::TreeChannelWireBytes, ncclGenkTreeFifoSlots, ncclGenkTreeConnections};
}

#if !defined(NCCL_OS_WINDOWS)
template <ncclFlowProtocol Protocol, unsigned GinBackendMask>
NCCL_DEVICE_INLINE ncclFlowConfig<GinBackendMask> ncclGenkRingGinConfig(ncclGenkDevComm const& genk, int channelId) {
  using Traits = ncclGenkProtocolTraits<Protocol>;
  ncclDevComm const& comm = genk.devComm;
  int const contextIndex = channelId / genk.nRingChannels;
  ncclSymPtr<char> ginBuffer;
  if (genk.ringGinChannelMask & (1UL << channelId)) {
    size_t constexpr ringGinChannelStride = alignUp(ncclGenkRingSimpleChannelSize, (size_t)128);
    uint64_t lowerBits = genk.ringGinChannelMask & ((1UL << channelId) - 1);
    ginBuffer = ncclGetResourceBuffer(comm, genk.ginFlowBuffer) + __popcll(lowerBits) * ringGinChannelStride;
  }
  return {ncclGetResourceBuffer(comm, Traits::ringBuffer(genk)),
          ncclGetResourceBuffer(comm, genk.connState),
          Traits::RingChannelWireBytes,
          Traits::RingFifoSlots,
          ncclGenkRingConnections,
          ncclTeamWorld(comm),
          genk.ginSignal0,
          ginBuffer,
          contextIndex};
}

template <ncclFlowProtocol Protocol, unsigned GinBackendMask>
NCCL_DEVICE_INLINE ncclFlowConfig<GinBackendMask> ncclGenkTreeGinConfig(ncclGenkDevComm const& genk, int channelId) {
  using Traits = ncclGenkProtocolTraits<Protocol>;
  ncclDevComm const& comm = genk.devComm;
  int const contextIndex = channelId / genk.nTreeSearchChannels;
  ncclSymPtr<char> ginBuffer;
  if (genk.treeGinChannelMask & (1UL << channelId)) {
    size_t constexpr ringGinChannelStride = alignUp(ncclGenkRingSimpleChannelSize, (size_t)128);
    size_t constexpr treeGinChannelStride = alignUp(ncclGenkTreeSimpleChannelSize, (size_t)128);
    uint64_t lowerBits = genk.treeGinChannelMask & ((1UL << channelId) - 1);
    ginBuffer = ncclGetResourceBuffer(comm, genk.ginFlowBuffer) +
                __popcll(genk.ringGinChannelMask) * ringGinChannelStride + __popcll(lowerBits) * treeGinChannelStride;
  }
  return {ncclGetResourceBuffer(comm, Traits::treeBuffer(genk)),
          ncclGenkTreeConnState(genk),
          Traits::TreeChannelWireBytes,
          ncclGenkTreeFifoSlots,
          ncclGenkTreeConnections,
          ncclTeamWorld(comm),
          genk.ginSignal0 + 2 * genk.nResourceChannels * ncclGenkRingConnections,
          ginBuffer,
          contextIndex};
}
#endif

namespace {
struct ncclGenkArgsHandler : ncclSymkWorkArgsHandler {
  __device__ ncclGenkArgsHandler(ncclGenkDevWorkArgs const* args)
    : ncclSymkWorkArgsHandler(args->kcomm.devComm, ncclSymkGetWorkRange(args),
                              ncclSymkGetWorks(args, args->nMaxChannels)) {}
};
} // namespace

#endif // NCCL_DEVICE_GENERAL_GENK_H_
