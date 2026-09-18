/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "nccl_device.h"
#include "kernel.cuh"
#include "genk.cuh"
#include <stdio.h>

template <ncclFlowProtocol Protocol, bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_Flow(ncclGenkDevWorkArgs const* args) {
  using ProtocolTraits = ncclGenkProtocolTraits<Protocol>;
  ncclGenkArgsHandler handler{args};
  ncclDevComm const& comm = handler.comm;
  ncclTeam const world = ncclTeamWorld(comm);
  int const nRanks = world.nRanks;
  int const channelId = blockIdx.x;
  ncclGenkRing const& ring = args->kcomm.rings[channelId % args->kcomm.nRingChannels];
  int const sendWorld = ring.next;
  int const recvWorld = ring.prev;
  constexpr int maxPeers = 1;
  constexpr int nSendPeers = 1;
  constexpr int nRecvPeers = 1;
  int const sendPeers[nSendPeers] = {sendWorld};
  int const recvPeers[nRecvPeers] = {recvWorld};
  constexpr int nSlotsPerProcess = ProtocolTraits::SlotsPerRingProcess;
  constexpr int signalWarpsPerPart = ncclFlowGetSignalWarpsPerPart(Protocol);
  int const nWorkerWarps = blockDim.x / WARP_SIZE - signalWarpsPerPart;
  __shared__ ncclFlowShmem flowShmem;
#if !defined(NCCL_OS_WINDOWS)
  ncclFlowConfig<ncclGenkGinBackendMask> const flowConfig =
    ncclGenkRingGinConfig<Protocol, ncclGenkGinBackendMask>(args->kcomm, channelId);
  ncclFlowProcessor<ncclGenkGinBackendMask, nSlotsPerProcess, maxPeers, Protocol> flow(
    comm, flowConfig, sendPeers, nSendPeers, recvPeers, nRecvPeers, channelId, nWorkerWarps, flowShmem);
#else
  ncclFlowConfig<0> const flowConfig = ncclGenkRingLsaConfig<Protocol>(args->kcomm);
  ncclFlowProcessor<0, nSlotsPerProcess, maxPeers, Protocol> flow(comm, flowConfig, sendPeers, nSendPeers, recvPeers,
                                                                  nRecvPeers, channelId, nWorkerWarps, flowShmem);
#endif
  int const minChunkElts = args->minChunkPayloadBytes / sizeof(T);
  constexpr int maxChunkElts = ProtocolTraits::RingChunkPayloadBytes / sizeof(T);
  using DirectSend = typename ProtocolTraits::DirectSend;
  using DirectRecv = typename ProtocolTraits::DirectRecv;

  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t /*nAllElts*/, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output, uint64_t redOpArg) {
    Red<T> red(redOpArg);
    T* localIn = (T*)input.localPtr();
    T* localOut = (T*)output.localPtr();
    ncclFlowNullPtrFn<T> nullPtr;
    bool const direct = ProtocolTraits::SupportsDirect && output.window != nullptr;
    bool const sendLsa = ncclDevCommCanGetPeerPointer(comm, sendWorld);
    // Balance all required revolutions across CTAs and ranks instead of
    // filling the first revolutions and leaving a lightly populated tail.
    size_t const chunksPerRevolution = (size_t)nBlocks * nRanks;
    size_t const maxRevolutionElts = chunksPerRevolution * maxChunkElts;
    size_t const nRevolutions = divUp(nElts, maxRevolutionElts);
    size_t chunkElts = alignUp(divUp(nElts, chunksPerRevolution * nRevolutions), 16);
    if (chunkElts < minChunkElts) chunkElts = minChunkElts;
    if (chunkElts > maxChunkElts) chunkElts = maxChunkElts;
    size_t gridOffset = 0;
    size_t loopOffset, offset, elts;
    int dataChunk = ring.index;
    T *src, *dst;

  nextround:
    if (gridOffset >= nElts) return;
    loopOffset = gridOffset + block * chunkElts * nRanks;
    if (loopOffset >= nElts) return;
    dataChunk = ring.index;

    // Send the first chunk. 1 source (input), 1 dest (sendbuf)
    offset = loopOffset + dataChunk * chunkElts;
    elts = offset < nElts ? min(chunkElts, nElts - offset) : 0;
    src = localIn + offset;
    flow.process(
      ncclFlowSend{}, ncclFlowNoRecv{}, input + offset, output + offset, [=] __device__(int) -> T* { return src; }, 1,
      nullPtr, 0, red, elts);

    // Reduce+Send the next (nranks-2) chunks. 2 sources (input+recvbuf), 1 dest (sendbuf)
    for (int s = 1; s < nRanks - 1; s++) {
      dataChunk = (ring.index - s + nRanks) % nRanks;
      offset = loopOffset + dataChunk * chunkElts;
      elts = (offset < nElts) ? min(chunkElts, nElts - offset) : 0;
      src = localIn + offset;
      flow.process(
        ncclFlowSend{}, ncclFlowRecv{}, input + offset, output + offset, [=] __device__(int) -> T* { return src; }, 1,
        nullPtr, 0, red, elts);
    }

    // Last step of Reduce Scatter + First step of allgather. 2 sources (input+recvbuf), 2 dests (sendbuf + output)
    dataChunk = (ring.index + 1) % nRanks;
    offset = loopOffset + dataChunk * chunkElts;
    elts = (offset < nElts) ? min(chunkElts, nElts - offset) : 0;
    src = localIn + offset;
    dst = localOut + offset;
    ncclSymPtr<T> directOutput = output + offset;
    bool directSend = direct && sendLsa;
    T* peerDst = directSend ? directOutput.peerPtr(sendWorld) : nullptr;
    flow.process(
      DirectSend{direct}, ncclFlowRecv{}, input + offset, directOutput, [=] __device__(int) -> T* { return src; }, 1,
      [=] __device__(int i) -> T* { return i == 0 ? dst : peerDst; }, 1 + directSend, red, elts);

    // Recv and copy+send the next (nranks-2) chunks. 1 source (recvbuf), 2 dests (output+sendbuf)
    for (int s = 0; s < nRanks - 2; s++) {
      dataChunk = (ring.index - s + nRanks) % nRanks;
      offset = loopOffset + dataChunk * chunkElts;
      elts = (offset < nElts) ? min(chunkElts, nElts - offset) : 0;
      dst = localOut + offset;
      directOutput = output + offset;
      bool const directRecv = direct;
      directSend = direct && sendLsa;
      peerDst = directSend ? directOutput.peerPtr(sendWorld) : nullptr;
      flow.process(
        DirectSend{direct}, DirectRecv{direct}, input + offset, directOutput, [=] __device__(int) -> T* { return dst; },
        directRecv, [=] __device__(int i) -> T* { return !directRecv && i == 0 ? dst : peerDst; },
        !directRecv + directSend, red, elts);
    }

    // Final copy. 1 source (recvbuf), 1 dest (output)
    dataChunk = (ring.index + 2) % nRanks;
    offset = loopOffset + dataChunk * chunkElts;
    elts = (offset < nElts) ? min(chunkElts, nElts - offset) : 0;
    dst = localOut + offset;
    bool const directRecv = direct;
    flow.process(
      ncclFlowNoSend{}, DirectRecv{direct}, input + offset, output + offset, nullPtr, 0,
      [=] __device__(int) -> T* { return dst; }, !directRecv, red, elts);

    gridOffset += chunkElts * nRanks * nBlocks;
    if (gridOffset < nElts) goto nextround;
  });

  flow.close();
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_Simple(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllReduce_Ring_Flow<ncclFlowProtocolSimple, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_LL(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllReduce_Ring_Flow<ncclFlowProtocolLL, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Ring_LL128(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllReduce_Ring_Flow<ncclFlowProtocolLL128, EnableProfiler, Red, T>(args);
}

template <ncclFlowProtocol Protocol, bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_Flow(ncclGenkDevWorkArgs const* args) {
  using ProtocolTraits = ncclGenkProtocolTraits<Protocol>;
  ncclGenkArgsHandler handler{args};
  ncclDevComm const& comm = handler.comm;
  int const channelId = blockIdx.x;
  ncclGenkTree const& tree = args->kcomm.trees[channelId % args->kcomm.nTreeChannels];
  int const upPeer = tree.upLsa != -1 ? tree.upLsa : tree.upGin;
  bool const isRoot = upPeer == -1;

  constexpr int reducePart = 0;
  constexpr int bcastPart = 1;
  constexpr int nParts = ncclGenkTreeParts;
  constexpr int maxPeers = ncclGenkTreeSlots;
  constexpr int signalWarpsPerPart = ncclFlowGetSignalWarpsPerPart(Protocol);
  int const nWorkerWarps = blockDim.x / WARP_SIZE - nParts * signalWarpsPerPart;
  int const reduceWorkerWarps = ProtocolTraits::treeReduceWorkerWarps(nWorkerWarps);
  int const partWorkerWarps[nParts] = {isRoot ? nWorkerWarps : reduceWorkerWarps,
                                       isRoot ? 0 : nWorkerWarps - reduceWorkerWarps};
  int const part = ncclFlowThreadPart(nWorkerWarps, nParts, partWorkerWarps, signalWarpsPerPart);
  assert(part == reducePart || part == bcastPart);

  int downPeers[NCCL_MAX_TREE_ARITY];
  int downSlots[NCCL_MAX_TREE_ARITY];
  int nDownPeers = 0;
  if (tree.downLsa != -1) {
    downPeers[nDownPeers] = tree.downLsa;
    downSlots[nDownPeers++] = tree.downLsaSlot - ncclFlowTreeSlotChild0;
  }
  for (int i = 0; i < 2; i++) {
    if (tree.downGin[i] != -1) {
      downPeers[nDownPeers] = tree.downGin[i];
      downSlots[nDownPeers++] = tree.downGinSlots[i] - ncclFlowTreeSlotChild0;
    }
  }

  int const upSlot = tree.upSlot - ncclFlowTreeSlotChild0;
  int const* sendPeers = part == reducePart && !isRoot ? &upPeer : downPeers;
  int const* sendSlots = part == reducePart && !isRoot ? &upSlot : downSlots;
  int const nSendPeers = part == reducePart ? (isRoot ? nDownPeers : 1) : (isRoot ? 0 : nDownPeers);
  int const* recvPeers = part == reducePart ? downPeers : &upPeer;
  int const* recvSlots = part == reducePart ? downSlots : &upSlot;
  int const nRecvPeers = part == reducePart ? nDownPeers : !isRoot;
  // The fused root receives from the reduce-part FIFOs and publishes into the broadcast-part FIFOs.
  int const sendPart = isRoot && part == reducePart ? bcastPart : part;
  int const recvPart = part;

  __shared__ ncclFlowShmem flowShmem;
#if !defined(NCCL_OS_WINDOWS)
  ncclFlowConfig<ncclGenkGinBackendMask> const flowConfig =
    ncclGenkTreeGinConfig<Protocol, ncclGenkGinBackendMask>(args->kcomm, channelId);
  ncclFlowProcessor<ncclGenkGinBackendMask, /*SlotsPerProcess=*/1, maxPeers, Protocol> flow(
    comm, flowConfig, sendPeers, nSendPeers, recvPeers, nRecvPeers, channelId, nWorkerWarps, flowShmem, nParts,
    sendPart, recvPart, sendSlots, recvSlots, partWorkerWarps);
#else
  ncclFlowConfig<0> const flowConfig = ncclGenkTreeLsaConfig<Protocol>(args->kcomm);
  ncclFlowProcessor<0, /*SlotsPerProcess=*/1, maxPeers, Protocol> flow(comm, flowConfig, sendPeers, nSendPeers,
                                                                       recvPeers, nRecvPeers, channelId, nWorkerWarps,
                                                                       flowShmem, nParts, sendPart, recvPart, sendSlots,
                                                                       recvSlots, partWorkerWarps);
#endif

  int const minChunkElts = args->minChunkPayloadBytes / sizeof(T);
  constexpr int maxChunkElts = ProtocolTraits::TreeChunkPayloadBytes / sizeof(T);
  using DirectSend = typename ProtocolTraits::DirectSend;
  using DirectRecv = typename ProtocolTraits::DirectRecv;

  handler.forEachWork<T>([&] __device__(int block, int nBlocks, size_t nElts, size_t /*nAllElts*/, ncclSymPtr<T> input,
                                        ncclSymPtr<T> output, uint64_t redOpArg) {
    Red<T> red(redOpArg);
    T* const localIn = (T*)input.localPtr();
    ncclFlowNullPtrFn<T> nullPtr;
    bool const direct = ProtocolTraits::SupportsDirect && output.window != nullptr;
    bool const directUp = direct && !isRoot;
    bool const directDownLsa = direct && tree.downLsa != -1;
    // Avoid a lightly populated final round when one chunk cannot cover this
    // CTA's entire share of the operation.
    size_t const chunksPerRound = (size_t)nBlocks;
    size_t const maxRoundElts = chunksPerRound * maxChunkElts;
    size_t const nRounds = divUp(nElts, maxRoundElts);
    size_t chunkElts = alignUp(divUp(nElts, chunksPerRound * nRounds), 16);
    if (chunkElts < minChunkElts) chunkElts = minChunkElts;
    if (chunkElts > maxChunkElts) chunkElts = maxChunkElts;

    for (size_t offset = block * chunkElts; offset < nElts; offset += nBlocks * chunkElts) {
      size_t const elts = min(chunkElts, nElts - offset);
      T* const src = localIn + offset;
      ncclSymPtr<T> const directOutput = output + offset;
      T* const dst = directOutput.localPtr();

      if (part == reducePart) {
        if (isRoot) {
          if (nDownPeers == 0) {
            flow.process(
              ncclFlowNoSend{}, ncclFlowNoRecv{}, input + offset, directOutput,
              [=] __device__(int) -> T* { return src; }, 1, [=] __device__(int) -> T* { return dst; }, 1, red, elts);
          } else {
            T* const peerDst = directDownLsa ? directOutput.peerPtr(tree.downLsa) : nullptr;
            flow.process(
              DirectSend{direct}, ncclFlowRecv{}, input + offset, directOutput,
              [=] __device__(int) -> T* { return src; }, 1,
              [=] __device__(int i) -> T* { return i == 0 ? dst : peerDst; }, 1 + directDownLsa, red, elts);
          }
        } else if (nDownPeers == 0) {
          flow.process(
            ncclFlowSend{}, ncclFlowNoRecv{}, input + offset, directOutput, [=] __device__(int) -> T* { return src; },
            1, nullPtr, 0, red, elts);
        } else {
          flow.process(
            ncclFlowSend{}, ncclFlowRecv{}, input + offset, directOutput, [=] __device__(int) -> T* { return src; }, 1,
            nullPtr, 0, red, elts);
        }
      } else if (!isRoot && nDownPeers == 0) {
        flow.process(
          ncclFlowNoSend{}, DirectRecv{direct}, input + offset, directOutput, nullPtr, 0,
          [=] __device__(int) -> T* { return dst; }, !directUp, red, elts);
      } else if (!isRoot) {
        T* const peerDst = directDownLsa ? directOutput.peerPtr(tree.downLsa) : nullptr;
        flow.process(
          DirectSend{direct}, DirectRecv{direct}, input + offset, directOutput,
          [=] __device__(int) -> T* { return dst; }, directUp,
          [=] __device__(int i) -> T* { return !directUp && i == 0 ? dst : peerDst; }, !directUp + directDownLsa, red,
          elts);
      }
    }
  });

  flow.close();
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_Simple(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllReduce_Tree_Flow<ncclFlowProtocolSimple, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_LL(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllReduce_Tree_Flow<ncclFlowProtocolLL, EnableProfiler, Red, T>(args);
}

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclGenkRun_AllReduce_Tree_LL128(ncclGenkDevWorkArgs const* args) {
  ncclGenkRun_AllReduce_Tree_Flow<ncclFlowProtocolLL128, EnableProfiler, Red, T>(args);
}
