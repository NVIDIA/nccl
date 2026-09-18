/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "kernel.cuh"
#include "genk.cuh"

template <ncclFlowProtocol Protocol, bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_Flow(ncclGenkDevWorkArgs const* args) {
  using ProtocolTraits = ncclGenkProtocolTraits<Protocol>;
  ncclGenkArgsHandler handler{args};
  ncclDevComm const& comm = handler.comm;
  ncclTeam const world = ncclTeamWorld(comm);
  int const nRanks = world.nRanks;
  int const channelId = blockIdx.x;
  ncclGenkRing const& ring = args->kcomm.rings[channelId % args->kcomm.nRingChannels];
  int const* ringRanks = ring.userRanks;
  assert(ringRanks != nullptr && ringRanks[0] == world.rank);

  constexpr int maxPeers = 1;
  int const nPeers = nRanks == 1 ? 0 : 1;
  int const sendPeers[maxPeers] = {ring.next};
  int const recvPeers[maxPeers] = {ring.prev};
  constexpr int nSlotsPerProcess = ProtocolTraits::SlotsPerRingProcess;
  constexpr int signalWarpsPerPart = ncclFlowGetSignalWarpsPerPart(Protocol);
  int const nWorkerWarps = blockDim.x / WARP_SIZE - signalWarpsPerPart;
  __shared__ ncclFlowShmem flowShmem;
#if !defined(NCCL_OS_WINDOWS)
  ncclFlowConfig<ncclGenkGinBackendMask> const flowConfig =
    ncclGenkRingGinConfig<Protocol, ncclGenkGinBackendMask>(args->kcomm, channelId);
  ncclFlowProcessor<ncclGenkGinBackendMask, nSlotsPerProcess, maxPeers, Protocol> flow(
    comm, flowConfig, sendPeers, nPeers, recvPeers, nPeers, channelId, nWorkerWarps, flowShmem);
#else
  ncclFlowConfig<0> const flowConfig = ncclGenkRingLsaConfig<Protocol>(args->kcomm);
  ncclFlowProcessor<0, nSlotsPerProcess, maxPeers, Protocol> flow(comm, flowConfig, sendPeers, nPeers, recvPeers,
                                                                  nPeers, channelId, nWorkerWarps, flowShmem);
#endif
  int const minChunkElts = args->minChunkPayloadBytes;
  constexpr int maxChunkElts = ProtocolTraits::RingChunkPayloadBytes;
  using DirectSend = typename ProtocolTraits::DirectSend;
  using DirectRecv = typename ProtocolTraits::DirectRecv;
  FuncCopy<uint8_t> copyOp;

  handler.forEachWork<uint8_t>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts,
                                              ncclSymPtr<uint8_t> input, ncclSymPtr<uint8_t> output,
                                              uint64_t /*redOpArg*/) {
    uint8_t* const localIn = input.localPtr();
    ncclFlowNullPtrFn<uint8_t> nullPtr;
    bool const direct = ProtocolTraits::SupportsDirect && output.window != nullptr;
    bool const sendLsa = nRanks > 1 && ncclDevCommCanGetPeerPointer(comm, ring.next);
    size_t gridOffset = 0;

    while (gridOffset < nElts) {
      // Rebalance the final revolution across CTAs instead of leaving full
      // chunks followed by partial or empty chunks.
      size_t const remainingElts = nElts - gridOffset;
      size_t chunkElts = alignUp(divUp(remainingElts, (size_t)nBlocks), 16);
      if (chunkElts < minChunkElts) chunkElts = minChunkElts;
      if (chunkElts > maxChunkElts) chunkElts = maxChunkElts;
      size_t const loopOffset = gridOffset + block * chunkElts;
      if (loopOffset >= nElts) return;
      size_t const elts = min(chunkElts, nElts - loopOffset);
      uint8_t* const src = localIn + loopOffset;
      ncclSymPtr<uint8_t> directOutput = output + (size_t)ringRanks[0] * nAllElts + loopOffset;
      uint8_t* dst = directOutput.localPtr();
      bool inPlace = (src == dst);

      if (nRanks == 1) {
        if (!inPlace) {
          flow.process(
            ncclFlowNoSend{}, ncclFlowNoRecv{}, input + loopOffset, directOutput,
            [=] __device__(int) -> uint8_t* { return src; }, 1, [=] __device__(int) -> uint8_t* { return dst; }, 1,
            copyOp, elts);
        }
        gridOffset += nBlocks * chunkElts;
        continue;
      }

      // Publish this rank's contribution while copying it into its world-rank-ordered output segment.
      int const nLocalDsts = !inPlace;
      int const nDirectDsts = direct && sendLsa;
      uint8_t* const peerDst = nDirectDsts != 0 ? directOutput.peerPtr(ring.next) : nullptr;
      flow.process(
        DirectSend{direct}, ncclFlowNoRecv{}, input + loopOffset, directOutput,
        [=] __device__(int) -> uint8_t* { return src; }, 1,
        [=] __device__(int i) -> uint8_t* { return i < nLocalDsts ? dst : peerDst; }, nLocalDsts + nDirectDsts, copyOp,
        elts);

      // Forward each contribution received from the previous ring rank and retain a local copy.
      for (int step = 1; step < nRanks - 1; step++) {
        directOutput = output + (size_t)ringRanks[nRanks - step] * nAllElts + loopOffset;
        dst = directOutput.localPtr();
        bool const directRecv = direct;
        bool const directSend = direct && sendLsa;
        uint8_t* const peerDst = directSend ? directOutput.peerPtr(ring.next) : nullptr;
        flow.process(
          DirectSend{direct}, DirectRecv{direct}, input + loopOffset, directOutput,
          [=] __device__(int) -> uint8_t* { return dst; }, directRecv,
          [=] __device__(int i) -> uint8_t* { return !directRecv && i == 0 ? dst : peerDst; }, !directRecv + directSend,
          copyOp, elts);
      }

      directOutput = output + (size_t)ringRanks[1] * nAllElts + loopOffset;
      dst = directOutput.localPtr();
      bool const directRecv = direct;
      flow.process(
        ncclFlowNoSend{}, DirectRecv{direct}, input + loopOffset, directOutput, nullPtr, 0,
        [=] __device__(int) -> uint8_t* { return dst; }, !directRecv, copyOp, elts);

      gridOffset += nBlocks * chunkElts;
    }
  });

  flow.close();
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_Simple(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllGather_Ring_Flow<ncclFlowProtocolSimple, EnableProfiler>(args);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_LL(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllGather_Ring_Flow<ncclFlowProtocolLL, EnableProfiler>(args);
}

template <bool EnableProfiler>
__device__ __forceinline__ void ncclGenkRun_AllGather_Ring_LL128(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllGather_Ring_Flow<ncclFlowProtocolLL128, EnableProfiler>(args);
}
