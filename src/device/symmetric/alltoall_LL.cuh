/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "kernel.cuh"
#include "symk.cuh"

struct ncclSymkAlltoAllLLLineRange {
  size_t begin;
  size_t end;
};

static __device__ __forceinline__ ncclSymkAlltoAllLLLineRange ncclSymkAlltoAllLLPartition(size_t count, int partCount,
                                                                                          int part) {
  size_t countPerPart = count / (size_t)partCount;
  size_t extra = count % (size_t)partCount;
  size_t extraBefore = (size_t)part < extra ? (size_t)part : extra;
  size_t begin = (size_t)part * countPerPart + extraBefore;
  size_t partSize = countPerPart + ((size_t)part < extra ? 1 : 0);
  return {begin, begin + partSize};
}

static __device__ __forceinline__ ncclSymkAlltoAllLLScratchState* ncclSymkAlltoAllLLGetScratchState(
  ncclDevComm const& comm, ncclSymkAlltoAllLLScratchHandle handle) {
  auto* state = (ncclSymkAlltoAllLLScratchState*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
#if CUDART_VERSION < 13030
  // Work around some compilers passing the wrong pointer type to cuda::atomic_ref.
  // The empty asm hides how the pointer was computed and emits no instructions.
  asm("" : "+l"(state));
#endif
  return state;
}

struct ncclSymkAlltoAllLLScratchEpoch {
  ncclSymkAlltoAllLLScratchState* state;
  uint32_t epoch;
  ncclSymPtr<char> receivePlaneSym;
  ncclSymPtr<char> sendPlaneSym;
  char* receivePlaneLocal;
  char* sendPlaneLocal;
};

// Any CTA may win the publication CAS. It advances the single communicator
// epoch once; one lane per warp then acquires that published value. This works
// unchanged under CUDA graph replay because the epoch lives in device memory.
static __device__ __forceinline__ ncclSymkAlltoAllLLScratchEpoch ncclSymkAlltoAllLLOpenScratchEpoch(
  ncclDevComm const& comm, ncclSymkAlltoAllLLScratchHandle handle) {
  using nccl::utility::atomicLoad;
  using nccl::utility::atomicStore;
  using nccl::utility::testAbort;

  ncclSymkAlltoAllLLScratchState* state = ncclSymkAlltoAllLLGetScratchState(comm, handle);
  if (threadIdx.x == 0 && atomicCAS(&state->publishedEpoch, 0u, 1u) == 0u) {
    uint32_t epoch = NCCL_LL_FLAG(state->epoch + 1);
    if (epoch < 2) epoch = 2;
    state->epoch = epoch;
    atomicStore(&state->publishedEpoch, epoch, cuda::memory_order_release, cuda::thread_scope_device);
  }

  uint32_t epoch = 0;
  int lane = (int)threadIdx.x % WARP_SIZE;
  if (lane == 0) {
    uint32_t steps = 0;
    do {
      epoch = atomicLoad(&state->publishedEpoch, cuda::memory_order_acquire, cuda::thread_scope_device);
      if (testAbort(comm.abortFlag, steps)) break;
    } while (epoch < 2);
  }
  epoch = __shfl_sync(0xffffffffu, epoch, 0);

  ncclSymPtr<char> scratchSym = ncclGetResourceBuffer(comm, handle.bufHandle) + sizeof(ncclSymkAlltoAllLLScratchState);
  char* scratchLocal =
    (char*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle) + sizeof(ncclSymkAlltoAllLLScratchState);
  size_t receiveOffset = ncclSymkAlltoAllLLReceivePlaneOffset(epoch, handle.receivePlaneBytes);
  size_t sendOffset = ncclSymkAlltoAllLLSendPlaneOffset(epoch, handle.receivePlaneBytes, handle.sendPlaneBytes);
  return {state,
          epoch,
          scratchSym + receiveOffset,
          scratchSym + sendOffset,
          scratchLocal + receiveOffset,
          scratchLocal + sendOffset};
}

static __device__ __forceinline__ int ncclSymkAlltoAllLLPeerForOffset(ncclTeam team, int peerOffset, bool sendNotRecv) {
  int sign = sendNotRecv ? 1 : -1;
  int peer = team.rank + sign * (1 + peerOffset);
  if (unsigned(team.nRanks) <= unsigned(peer)) peer += -sign * team.nRanks;
  return peer;
}

// Store one LL fifo line as {low32, flag, high32, flag}. GIN packing only
// needs a local volatile store; direct NVLink publication uses system scope.
static __device__ __forceinline__ void ncclSymkAlltoAllLLStoreLocalLine(ncclLLFifoLine* dst, uint64_t value,
                                                                        uint32_t flag) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%2};" ::"l"(&dst->i4), "r"((uint32_t)value), "r"(flag),
               "r"((uint32_t)(value >> 32))
               : "memory");
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLStoreRemoteLine(ncclLLFifoLine* dst, uint64_t value,
                                                                         uint32_t flag) {
#if __CUDA_ARCH__ >= 700
  asm volatile("st.relaxed.sys.v4.u32 [%0], {%1,%2,%3,%2};" ::"l"(&dst->i4), "r"((uint32_t)value), "r"(flag),
               "r"((uint32_t)(value >> 32))
               : "memory");
#else
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%2};" ::"l"(&dst->i4), "r"((uint32_t)value), "r"(flag),
               "r"((uint32_t)(value >> 32))
               : "memory");
#endif
}

// Poll one LL fifo line until both flags match, then return the interleaved
// payload words. No separate GIN signal is needed for data readiness.
static __device__ __forceinline__ uint64_t ncclSymkAlltoAllLLWaitLine(ncclDevComm const& comm, ncclLLFifoLine* src,
                                                                      uint32_t flag) {
  using nccl::utility::testAbort;
  uint32_t data1, flag1, data2, flag2;
  uint32_t steps = 0;
  do {
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.relaxed.sys.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                 : "l"(&src->i4)
                 : "memory");
#else
    asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(data1), "=r"(flag1), "=r"(data2), "=r"(flag2)
                 : "l"(&src->i4)
                 : "memory");
#endif
    if (testAbort(comm.abortFlag, steps)) break;
  } while (flag1 != flag || flag2 != flag);

  return (uint64_t)data1 | ((uint64_t)data2 << 32);
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLResetReceivePlane(char* receivePlane, size_t planeBytes,
                                                                           int workBlock, int totalBlocks) {
  ncclSymkAlltoAllLLLineRange range = ncclSymkAlltoAllLLPartition(planeBytes / sizeof(uint4), totalBlocks, workBlock);
  uint4* lines = (uint4*)receivePlane;
  for (size_t line = range.begin + threadIdx.x; line < range.end; line += blockDim.x) {
    lines[line] = uint4{0, 0, 0, 0};
  }
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLWaitForValue(ncclDevComm const& comm, uint32_t* value,
                                                                      uint32_t expected) {
  using nccl::utility::atomicLoad;
  using nccl::utility::testAbort;

  uint32_t steps = 0;
  while (atomicLoad(value, cuda::memory_order_acquire, cuda::thread_scope_device) != expected) {
    if (testAbort(comm.abortFlag, steps)) break;
  }
}

// Rollover must synchronize every rank before and after clearing receive
// scratch. Pure-LSA communicators use their existing LSA barrier. Mixed
// communicators use the single hybrid world-barrier slot reserved for this
// path; payload puts have already been consumed, so the barrier needs no GIN
// visibility fence.
static __device__ __forceinline__ void ncclSymkAlltoAllLLRolloverBarrier(ncclSymkArgsHandler const& handler,
                                                                         ncclCoopCta cta) {
  if (handler.comm.nRanks == ncclTeamLsa(handler.comm).nRanks) {
    ncclLsaBarrierSession<ncclCoopCta> barrier(cta, handler.comm, ncclTeamTagLsa(), /*index=*/0);
    barrier.sync(cta, cuda::memory_order_acq_rel);
  } else {
    ncclGin gin(handler.comm, /*context=*/0);
    ncclBarrierSession<ncclCoopCta> barrier(cta, ncclTeamTagWorld(), gin, /*index=*/0);
    barrier.sync(cta, cuda::memory_order_acq_rel, ncclGinFenceLevel::None);
  }
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLEndScratchEpoch(
  ncclSymkArgsHandler const& handler, ncclCoopCta cta, ncclSymkAlltoAllLLScratchEpoch const& scratch, size_t planeBytes,
  int workBlock, int totalBlocks) {
  using nccl::utility::atomicStore;

  cta.sync();
  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> completed(scratch.state->completedCtas);
  if (__builtin_expect(scratch.epoch < NCCL_LL_FLAG(-2u), true)) {
    if (threadIdx.x != 0) return;
    uint32_t completedBefore = completed.fetch_add(1, cuda::memory_order_acq_rel);
    if (completedBefore + 1 == (uint32_t)totalBlocks) {
      completed.store(0, cuda::memory_order_relaxed);
      // Only a later kernel consumes idle; its entry dependency waits for this
      // kernel to complete and flush global memory.
      atomicStore(&scratch.state->publishedEpoch, 0u, cuda::memory_order_relaxed, cuda::thread_scope_device);
    }
    return;
  }

  // Rollover cleanup has two distinct races to exclude:
  // 1. A CTA must not clear lines that another local CTA is still consuming.
  // 2. A faster rank must not start the next operation and publish into either
  //    plane while a slower rank is still clearing it.
  //
  // AlltoAll_LL launches no more than the kernel's resident-grid capacity, so
  // CTA 0 can rendezvous the local grid without preventing another CTA from
  // running. Only CTA 0 enters the world barriers; all CTAs clear disjoint
  // ranges between them.
  if (threadIdx.x == 0) completed.fetch_add(1, cuda::memory_order_release);

  if (workBlock == 0) {
    if (threadIdx.x == 0) {
      ncclSymkAlltoAllLLWaitForValue(handler.comm, &scratch.state->completedCtas, (uint32_t)totalBlocks);
    }
    cta.sync();
    ncclSymkAlltoAllLLRolloverBarrier(handler, cta);
    if (threadIdx.x == 0) {
      atomicStore(&scratch.state->resetReady, 1u, cuda::memory_order_release, cuda::thread_scope_device);
    }
  } else if (threadIdx.x == 0) {
    ncclSymkAlltoAllLLWaitForValue(handler.comm, &scratch.state->resetReady, 1u);
  }
  cta.sync();

  // Clear both complete receive planes, not just this invocation's used lines.
  // A later, larger AlltoAll can therefore never match stale flags.
  char* receivePlaneBase = scratch.receivePlaneLocal - (scratch.epoch & 1) * planeBytes;
  ncclSymkAlltoAllLLResetReceivePlane(receivePlaneBase, planeBytes, workBlock, totalBlocks);
  ncclSymkAlltoAllLLResetReceivePlane(receivePlaneBase + planeBytes, planeBytes, workBlock, totalBlocks);
  cta.sync();

  cuda::atomic_ref<uint32_t, cuda::thread_scope_device> resetCompleted(scratch.state->resetCompletedCtas);
  if (threadIdx.x == 0) resetCompleted.fetch_add(1, cuda::memory_order_release);

  if (workBlock == 0) {
    if (threadIdx.x == 0) {
      ncclSymkAlltoAllLLWaitForValue(handler.comm, &scratch.state->resetCompletedCtas, (uint32_t)totalBlocks);
    }
    cta.sync();
    ncclSymkAlltoAllLLRolloverBarrier(handler, cta);
    if (threadIdx.x == 0) {
      resetCompleted.store(0, cuda::memory_order_relaxed);
      atomicStore(&scratch.state->resetReady, 0u, cuda::memory_order_relaxed, cuda::thread_scope_device);
      completed.store(0, cuda::memory_order_relaxed);
      atomicStore(&scratch.state->publishedEpoch, 0u, cuda::memory_order_relaxed, cuda::thread_scope_device);
    }
  }
}

struct ncclSymkAlltoAllLLChunk {
  int peerOffset;
  int firstPeerLine;
  int lineCount;
  size_t firstGlobalLine;
};

static __device__ __forceinline__ ncclSymkAlltoAllLLChunk ncclSymkAlltoAllLLGetChunk(
  size_t chunkIndex, int canonicalPeerBase, int linesPerPeer, int chunksPerPeer, int linesPerWarp) {
  int peerOffset = (int)(chunkIndex / (size_t)chunksPerPeer);
  int peerChunk = (int)(chunkIndex - (size_t)peerOffset * chunksPerPeer);
  int firstPeerLine = peerChunk * linesPerWarp;
  int lineCount = linesPerPeer - firstPeerLine;
  if (lineCount > linesPerWarp) lineCount = linesPerWarp;
  return {peerOffset, firstPeerLine, lineCount,
          (size_t)(canonicalPeerBase + peerOffset) * linesPerPeer + firstPeerLine};
}

// A GIN warp formats one peer-local chunk and posts one put. Each lane formats
// one or more lines; the warp barrier publishes them before lane 0 posts GIN.
static __device__ __forceinline__ void ncclSymkAlltoAllLLSendGinChunk(
  ncclSymkArgsHandler const& handler, ncclTeam world, ncclTeam lsa, ncclSymkAlltoAllLLScratchEpoch const& scratch,
  size_t chunkIndex, int linesPerPeer, int chunksPerPeer, int linesPerWarp, size_t bytesPerPeer, size_t globalWarp,
  ncclSymPtr<char> input) {
  int lane = (int)threadIdx.x % WARP_SIZE;
  int lsaPeerCount = lsa.nRanks > 1 ? lsa.nRanks - 1 : 0;
  ncclSymkAlltoAllLLChunk chunk =
    ncclSymkAlltoAllLLGetChunk(chunkIndex, lsaPeerCount, linesPerPeer, chunksPerPeer, linesPerWarp);
  int sendPeer = ncclSymkAlltoAllLLRemotePeer(world.rank, world.nRanks, lsa.rank, lsa.nRanks, chunk.peerOffset,
                                              /*sendNotRecv=*/true);

  size_t firstPayloadOffset = (size_t)chunk.firstPeerLine * ncclSymkAlltoAllLLPayloadBytesPerLine;
  char* sourceBase = input.localPtr() + (size_t)sendPeer * bytesPerPeer + firstPayloadOffset;
  // Receive scratch keeps the LSA-prefix canonical layout, but local send
  // staging omits LSA peers because they never use a GIN source buffer.
  size_t firstSendLine = (size_t)chunk.peerOffset * linesPerPeer + chunk.firstPeerLine;
  ncclLLFifoLine* sendScratchBase = (ncclLLFifoLine*)scratch.sendPlaneLocal + firstSendLine;
  size_t payloadBytesFromChunkStart = bytesPerPeer - firstPayloadOffset;

  for (int chunkLine = lane; chunkLine < chunk.lineCount; chunkLine += WARP_SIZE) {
    size_t chunkPayloadOffset = (size_t)chunkLine * ncclSymkAlltoAllLLPayloadBytesPerLine;
    int lineBytes = (int)(payloadBytesFromChunkStart - chunkPayloadOffset < ncclSymkAlltoAllLLPayloadBytesPerLine ?
                            payloadBytesFromChunkStart - chunkPayloadOffset :
                            ncclSymkAlltoAllLLPayloadBytesPerLine);
    BytePack<8> value = loadPack<BytePack<8>>(sourceBase + chunkPayloadOffset, /*ix=*/0, /*end=*/lineBytes);
    ncclSymkAlltoAllLLStoreLocalLine(sendScratchBase + chunkLine, value.native, scratch.epoch);
  }
  __syncwarp();

  if (lane == 0) {
    int ginContext = (int)(globalWarp % (size_t)handler.comm.ginContextCount);
    ncclGin gin(handler.comm, ginContext);
    size_t receiveScratchOffset = chunk.firstGlobalLine * ncclSymkAlltoAllLLWireBytesPerLine;
    size_t sendScratchOffset = firstSendLine * ncclSymkAlltoAllLLWireBytesPerLine;
    size_t wireBytes = (size_t)chunk.lineCount * ncclSymkAlltoAllLLWireBytesPerLine;
    gin.put(world, sendPeer, scratch.receivePlaneSym + receiveScratchOffset, scratch.sendPlaneSym + sendScratchOffset,
            wireBytes, ncclGin_None{}, ncclGin_None{}, ncclCoopThread{}, ncclGin_None{}, cuda::thread_scope_thread,
            cuda::thread_scope_system);
  }
}

// LSA needs no staging: lanes write one or more interleaved data/flag lines
// directly to the destination GPU over NVLink.
static __device__ __forceinline__ void ncclSymkAlltoAllLLSendLsaChunk(
  ncclSymkArgsHandler const& handler, ncclTeam lsa, ncclSymkAlltoAllLLScratchEpoch const& scratch, size_t chunkIndex,
  int linesPerPeer, int chunksPerPeer, int linesPerWarp, size_t bytesPerPeer, ncclSymPtr<char> input) {
  int lane = (int)threadIdx.x % WARP_SIZE;
  ncclSymkAlltoAllLLChunk chunk =
    ncclSymkAlltoAllLLGetChunk(chunkIndex, /*canonicalPeerBase=*/0, linesPerPeer, chunksPerPeer, linesPerWarp);
  int sendLsaRank = ncclSymkAlltoAllLLPeerForOffset(lsa, chunk.peerOffset, /*sendNotRecv=*/true);
  int sendWorldRank = ncclTeamRankToWorld(handler.comm, lsa, sendLsaRank);

  size_t firstPayloadOffset = (size_t)chunk.firstPeerLine * ncclSymkAlltoAllLLPayloadBytesPerLine;
  char* sourceBase = input.localPtr() + (size_t)sendWorldRank * bytesPerPeer + firstPayloadOffset;
  ncclLLFifoLine* receiveScratchBase =
    (ncclLLFifoLine*)scratch.receivePlaneSym.peerPtr(lsa, sendLsaRank) + chunk.firstGlobalLine;
  size_t payloadBytesFromChunkStart = bytesPerPeer - firstPayloadOffset;

  for (int chunkLine = lane; chunkLine < chunk.lineCount; chunkLine += WARP_SIZE) {
    size_t chunkPayloadOffset = (size_t)chunkLine * ncclSymkAlltoAllLLPayloadBytesPerLine;
    int lineBytes = (int)(payloadBytesFromChunkStart - chunkPayloadOffset < ncclSymkAlltoAllLLPayloadBytesPerLine ?
                            payloadBytesFromChunkStart - chunkPayloadOffset :
                            ncclSymkAlltoAllLLPayloadBytesPerLine);
    BytePack<8> value = loadPack<BytePack<8>>(sourceBase + chunkPayloadOffset, /*ix=*/0, /*end=*/lineBytes);
    ncclSymkAlltoAllLLStoreRemoteLine(receiveScratchBase + chunkLine, value.native, scratch.epoch);
  }
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLReceiveChunk(
  ncclSymkArgsHandler const& handler, ncclSymkAlltoAllLLScratchEpoch const& scratch,
  ncclSymkAlltoAllLLChunk const& chunk, int receiveWorldRank, size_t bytesPerPeer, ncclSymPtr<char> output) {
  int lane = (int)threadIdx.x % WARP_SIZE;
  size_t firstPayloadOffset = (size_t)chunk.firstPeerLine * ncclSymkAlltoAllLLPayloadBytesPerLine;
  ncclLLFifoLine* receiveScratchBase = (ncclLLFifoLine*)scratch.receivePlaneLocal + chunk.firstGlobalLine;
  char* destinationBase = output.localPtr() + (size_t)receiveWorldRank * bytesPerPeer + firstPayloadOffset;
  size_t payloadBytesFromChunkStart = bytesPerPeer - firstPayloadOffset;

  for (int chunkLine = lane; chunkLine < chunk.lineCount; chunkLine += WARP_SIZE) {
    BytePack<8> value;
    value.native = ncclSymkAlltoAllLLWaitLine(handler.comm, receiveScratchBase + chunkLine, scratch.epoch);
    size_t chunkPayloadOffset = (size_t)chunkLine * ncclSymkAlltoAllLLPayloadBytesPerLine;
    int lineBytes = (int)(payloadBytesFromChunkStart - chunkPayloadOffset < ncclSymkAlltoAllLLPayloadBytesPerLine ?
                            payloadBytesFromChunkStart - chunkPayloadOffset :
                            ncclSymkAlltoAllLLPayloadBytesPerLine);
    storePack(destinationBase + chunkPayloadOffset, /*ix=*/0, /*end=*/lineBytes, value);
  }
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLReceiveGinChunk(
  ncclSymkArgsHandler const& handler, ncclTeam world, ncclTeam lsa, ncclSymkAlltoAllLLScratchEpoch const& scratch,
  size_t chunkIndex, int linesPerPeer, int chunksPerPeer, int linesPerWarp, size_t bytesPerPeer,
  ncclSymPtr<char> output) {
  int lsaPeerCount = lsa.nRanks > 1 ? lsa.nRanks - 1 : 0;
  ncclSymkAlltoAllLLChunk chunk =
    ncclSymkAlltoAllLLGetChunk(chunkIndex, lsaPeerCount, linesPerPeer, chunksPerPeer, linesPerWarp);
  int receivePeer = ncclSymkAlltoAllLLRemotePeer(world.rank, world.nRanks, lsa.rank, lsa.nRanks, chunk.peerOffset,
                                                 /*sendNotRecv=*/false);
  ncclSymkAlltoAllLLReceiveChunk(handler, scratch, chunk, receivePeer, bytesPerPeer, output);
}

static __device__ __forceinline__ void ncclSymkAlltoAllLLReceiveLsaChunk(
  ncclSymkArgsHandler const& handler, ncclTeam lsa, ncclSymkAlltoAllLLScratchEpoch const& scratch, size_t chunkIndex,
  int linesPerPeer, int chunksPerPeer, int linesPerWarp, size_t bytesPerPeer, ncclSymPtr<char> output) {
  ncclSymkAlltoAllLLChunk chunk =
    ncclSymkAlltoAllLLGetChunk(chunkIndex, /*canonicalPeerBase=*/0, linesPerPeer, chunksPerPeer, linesPerWarp);
  int receiveLsaRank = ncclSymkAlltoAllLLPeerForOffset(lsa, chunk.peerOffset, /*sendNotRecv=*/false);
  int receiveWorldRank = ncclTeamRankToWorld(handler.comm, lsa, receiveLsaRank);
  ncclSymkAlltoAllLLReceiveChunk(handler, scratch, chunk, receiveWorldRank, bytesPerPeer, output);
}

// Small/medium-message one-shot LL kernel. There are no payload waves, credits,
// or acknowledgements: the full expanded image must fit scratch. Grid and world
// rendezvous are confined to the rare LL-flag rollover cleanup.
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AlltoAll_LL(struct ncclSymkDevWorkArgs const* args) {
  ncclSymkArgsHandler handler{args};
  ncclTeam world = ncclTeamWorld(handler.comm);
  ncclTeam lsa = ncclTeamLsa(handler.comm);

  // AlltoAll is isolated to one work descriptor. Its channels are CTA work
  // partitions, not generic byte-range channels.
  ncclSymkDevWork const& work = handler.devWork[0];
  if (blockIdx.x < work.sChannelId || blockIdx.x >= work.sChannelId + work.nChannels) return;
  int workBlock = (int)blockIdx.x - (int)work.sChannelId;
  int totalBlocks = (int)work.nChannels;
  size_t bytesPerPeer = work.nElts;
  if (bytesPerPeer == 0 || totalBlocks <= 0) return;

  int lsaPeerCount = lsa.nRanks > 1 ? lsa.nRanks - 1 : 0;
  int ginPeerCount = world.nRanks - lsa.nRanks;
  constexpr int lsaLinesPerWarp = ncclSymkAlltoAllLLLsaLinesPerWarp;
  int ginLinesPerWarp = ncclSymkAlltoAllLLGinLinesPerWarp(bytesPerPeer);
  int linesPerPeer = ncclSymkAlltoAllLLLineCount(bytesPerPeer);
  int lsaChunksPerPeer = ncclSymkAlltoAllLLChunksPerPeer(bytesPerPeer, lsaLinesPerWarp);
  int ginChunksPerPeer = ncclSymkAlltoAllLLChunksPerPeer(bytesPerPeer, ginLinesPerWarp);
  size_t ginChunkCount = ncclSymkAlltoAllLLChunkCount(ginPeerCount, bytesPerPeer, ginLinesPerWarp);
  size_t lsaChunkCount = ncclSymkAlltoAllLLChunkCount(lsaPeerCount, bytesPerPeer, lsaLinesPerWarp);
  size_t totalLines = ncclSymkAlltoAllLLTotalLineCount(world.nRanks, bytesPerPeer);
  size_t ginLines = (size_t)ginPeerCount * linesPerPeer;
  if (totalLines * ncclSymkAlltoAllLLWireBytesPerLine > handler.alltoAllLLScratch.receivePlaneBytes ||
      ginLines * ncclSymkAlltoAllLLWireBytesPerLine > handler.alltoAllLLScratch.sendPlaneBytes) {
    return;
  }

  ncclSymkAlltoAllLLScratchEpoch scratch = ncclSymkAlltoAllLLOpenScratchEpoch(handler.comm, handler.alltoAllLLScratch);
  if (scratch.epoch < 2) return;

  int warpInBlock = (int)threadIdx.x / WARP_SIZE;
  int warpsPerBlock = (int)blockDim.x / WARP_SIZE;
  size_t globalWarp = (size_t)warpInBlock * totalBlocks + workBlock;
  size_t totalWarps = (size_t)warpsPerBlock * totalBlocks;

  // All warps in the grid form one round-robin pool. Interleaving CTA indices
  // spreads adjacent chunks across CTAs. Work IDs place every GIN chunk before
  // every LSA chunk, so enough warps give the domains disjoint owners.
  //
  // For N=32, LSA=4, B=8 KiB, automatic GIN chunks have 512 lines:
  // 28 remote peers x 2 GIN chunks plus 3 local peers x 32 LSA chunks =
  // 152 useful warps. Ten 16-warp CTAs give every chunk a distinct owner;
  // work IDs 0-55 post GIN and IDs 56-151 issue direct NVLink stores.
  ncclSymPtr<char> input{work.inputWin, work.inputOff};
  ncclSymPtr<char> output{work.outputWin, work.outputOff};

  for (size_t chunkIndex = globalWarp; chunkIndex < ginChunkCount; chunkIndex += totalWarps) {
    ncclSymkAlltoAllLLSendGinChunk(handler, world, lsa, scratch, chunkIndex, linesPerPeer, ginChunksPerPeer,
                                   ginLinesPerWarp, bytesPerPeer, globalWarp, input);
  }

  size_t firstLsaWork = globalWarp;
  if (firstLsaWork < ginChunkCount) {
    size_t skippedRounds = (ginChunkCount - firstLsaWork + totalWarps - 1) / totalWarps;
    firstLsaWork += skippedRounds * totalWarps;
  }
  size_t totalChunkCount = ginChunkCount + lsaChunkCount;
  for (size_t workIndex = firstLsaWork; workIndex < totalChunkCount; workIndex += totalWarps) {
    ncclSymkAlltoAllLLSendLsaChunk(handler, lsa, scratch, workIndex - ginChunkCount, linesPerPeer, lsaChunksPerPeer,
                                   lsaLinesPerWarp, bytesPerPeer, input);
  }

  // Split the self transfer across all CTAs in 16-byte units. Each warp reaches it after
  // its assigned sends, overlapping the local copy with remote data arrival.
  size_t selfPackCount = (bytesPerPeer + sizeof(uint4) - 1) / sizeof(uint4);
  ncclSymkAlltoAllLLLineRange selfRange = ncclSymkAlltoAllLLPartition(selfPackCount, totalBlocks, workBlock);
  size_t selfBegin = selfRange.begin * sizeof(uint4);
  size_t selfEnd = selfRange.end * sizeof(uint4);
  if (selfEnd > bytesPerPeer) selfEnd = bytesPerPeer;
  if (selfBegin < selfEnd) {
    uint8_t* selfInput = (uint8_t*)input.localPtr() + (size_t)world.rank * bytesPerPeer + selfBegin;
    uint8_t* selfOutput = (uint8_t*)output.localPtr() + (size_t)world.rank * bytesPerPeer + selfBegin;
    ncclLocalCopy<uint8_t>(ncclCoopCta{}, selfInput, /*nDst=*/1, selfOutput, /*displ=*/0, selfEnd - selfBegin);
  }

  // Consume the faster LSA arrivals first, then the GIN arrivals. The inverse
  // peer mapping addresses the same canonical scratch chunks used by senders.
  for (size_t workIndex = firstLsaWork; workIndex < totalChunkCount; workIndex += totalWarps) {
    ncclSymkAlltoAllLLReceiveLsaChunk(handler, lsa, scratch, workIndex - ginChunkCount, linesPerPeer, lsaChunksPerPeer,
                                      lsaLinesPerWarp, bytesPerPeer, output);
  }
  for (size_t chunkIndex = globalWarp; chunkIndex < ginChunkCount; chunkIndex += totalWarps) {
    ncclSymkAlltoAllLLReceiveGinChunk(handler, world, lsa, scratch, chunkIndex, linesPerPeer, ginChunksPerPeer,
                                      ginLinesPerWarp, bytesPerPeer, output);
  }

  // Send staging alternates between two epoch-selected planes, so no explicit
  // GIN flush/wait is needed here. Before operation k+2 can start locally, every
  // peer must have completed k to post its k+1 data back to this rank. Completing
  // k means that peer observed every LL line from this rank, so its GIN reads of
  // the operation-k send plane have already finished.
  ncclSymkAlltoAllLLEndScratchEpoch(handler, ncclCoopCta{}, scratch, handler.alltoAllLLScratch.receivePlaneBytes,
                                    workBlock, totalBlocks);
}
