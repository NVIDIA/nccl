/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "kernel.cuh"
#include "genk.cuh"

template <ncclFlowProtocol Protocol, bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_Flow(ncclGenkDevWorkArgs const* args) {
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
  int const minChunkElts = args->minChunkPayloadBytes / sizeof(T);
  constexpr int maxChunkElts = ProtocolTraits::RingChunkPayloadBytes / sizeof(T);

  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t nAllElts, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output, uint64_t redOpArg) {
    Red<T> red(redOpArg);
    T* const localOut = output.localPtr();
    ncclFlowNullPtrFn<T> nullPtr;
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
      ncclSymPtr<T> sourceInput = input + (size_t)ringRanks[nRanks - 1] * nAllElts + loopOffset;
      T* src = sourceInput.localPtr();

      if (nRanks == 1) {
        flow.process(
          ncclFlowNoSend{}, ncclFlowNoRecv{}, sourceInput, output + loopOffset,
          [=] __device__(int) -> T* { return src; }, 1, [=] __device__(int) -> T* { return localOut + loopOffset; }, 1,
          red, elts);
        gridOffset += nBlocks * chunkElts;
        continue;
      }

      // Start the reduction of the segment owned by the previous rank in the ring.
      flow.process(
        ncclFlowSend{}, ncclFlowNoRecv{}, sourceInput, output + loopOffset, [=] __device__(int) -> T* { return src; },
        1, nullPtr, 0, red, elts);

      // Reduce every in-flight segment with this rank's corresponding input and forward it.
      for (int step = 2; step < nRanks; step++) {
        sourceInput = input + (size_t)ringRanks[nRanks - step] * nAllElts + loopOffset;
        src = sourceInput.localPtr();
        flow.process(
          ncclFlowSend{}, ncclFlowRecv{}, sourceInput, output + loopOffset, [=] __device__(int) -> T* { return src; },
          1, nullPtr, 0, red, elts);
      }

      // Finish the segment owned by this rank and write it in recv-buffer order.
      sourceInput = input + (size_t)ringRanks[0] * nAllElts + loopOffset;
      src = sourceInput.localPtr();
      flow.process(
        ncclFlowNoSend{}, ncclFlowRecv{}, sourceInput, output + loopOffset, [=] __device__(int) -> T* { return src; },
        1, [=] __device__(int) -> T* { return localOut + loopOffset; }, 1, red, elts);

      gridOffset += nBlocks * chunkElts;
    }
  });

  flow.close();
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_Simple(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_ReduceScatter_Ring_Flow<ncclFlowProtocolSimple, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_LL(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_ReduceScatter_Ring_Flow<ncclFlowProtocolLL, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_ReduceScatter_Ring_LL128(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_ReduceScatter_Ring_Flow<ncclFlowProtocolLL128, EnableProfiler, Red, T>(args);
}
