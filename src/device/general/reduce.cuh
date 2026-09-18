/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

#include "sym_kernels.h"
#include "kernel.cuh"
#include "genk.cuh"

template <ncclFlowProtocol Protocol, bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_Flow(ncclGenkDevWorkArgs const* args) {
  using ProtocolTraits = ncclGenkProtocolTraits<Protocol>;
  ncclGenkArgsHandler handler{args};
  ncclDevComm const& comm = handler.comm;
  ncclTeam const world = ncclTeamWorld(comm);
  int const nRanks = world.nRanks;
  int const channelId = blockIdx.x;
  ncclGenkRing const& ring = args->kcomm.rings[channelId % args->kcomm.nRingChannels];

  // Keep both directions available because fused works may use different roots.
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

  handler.forEachWorkWithRoot<T>([&] __device__(int root, int block, int nBlocks, size_t nElts, size_t /*nAllElts*/,
                                                ncclSymPtr<T> input, ncclSymPtr<T> output, uint64_t redOpArg) {
    Red<T> red(redOpArg);
    T* const localIn = input.localPtr();
    T* const localOut = output.localPtr();
    ncclFlowNullPtrFn<T> nullPtr;
    size_t chunkElts = alignUp(divUp(nElts, (size_t)nBlocks), 16);
    if (chunkElts < minChunkElts) chunkElts = minChunkElts;
    if (chunkElts > maxChunkElts) chunkElts = maxChunkElts;

    for (size_t loopOffset = block * chunkElts; loopOffset < nElts; loopOffset += nBlocks * chunkElts) {
      size_t const elts = min(chunkElts, nElts - loopOffset);
      T* const src = localIn + loopOffset;
      T* const dst = world.rank == root ? localOut + loopOffset : nullptr;

      // The root's ring successor injects, the root terminates, and every other rank reduces and forwards.
      if (nRanks == 1) {
        flow.process(
          ncclFlowNoSend{}, ncclFlowNoRecv{}, input + loopOffset, output + loopOffset,
          [=] __device__(int) -> T* { return src; }, 1, [=] __device__(int) -> T* { return dst; }, 1, red, elts);
      } else if (ring.prev == root) {
        flow.process(
          ncclFlowSend{}, ncclFlowNoRecv{}, input + loopOffset, output + loopOffset,
          [=] __device__(int) -> T* { return src; }, 1, nullPtr, 0, red, elts);
      } else if (world.rank == root) {
        flow.process(
          ncclFlowNoSend{}, ncclFlowRecv{}, input + loopOffset, output + loopOffset,
          [=] __device__(int) -> T* { return src; }, 1, [=] __device__(int) -> T* { return dst; }, 1, red, elts);
      } else {
        flow.process(
          ncclFlowSend{}, ncclFlowRecv{}, input + loopOffset, output + loopOffset,
          [=] __device__(int) -> T* { return src; }, 1, nullPtr, 0, red, elts);
      }
    }
  });

  flow.close();
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_Simple(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_Reduce_Ring_Flow<ncclFlowProtocolSimple, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_LL(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_Reduce_Ring_Flow<ncclFlowProtocolLL, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_Reduce_Ring_LL128(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_Reduce_Ring_Flow<ncclFlowProtocolLL128, EnableProfiler, Red, T>(args);
}
