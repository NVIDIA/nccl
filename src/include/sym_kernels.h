/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_SYM_KERNELS_H_
#define NCCL_SYM_KERNELS_H_
#include "nccl.h"
#include "nccl_device.h"
#include "nccl_common.h"
#include "device.h"
#if !defined(NCCL_OS_WINDOWS)
#include "../device/symmetric/gin_scratch.h"
#else
#include "nccl_device/gin_win_stub.h"
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclSymk[Foo]: Specialized symmetric kernels and shared dispatch machinery
// ncclGenk[Foo]: General Ring/Tree kernels built on the device API

#define NCCL_SYM_KERNEL_CELL_SIZE 1024 // no less than 16 bytes minimal cell size

constexpr int ncclSymkMaxBlocks = 64;
constexpr int ncclSymkMaxThreads = 512;
constexpr int ncclSymkLLMaxEltSize = 8;
constexpr int ncclGenkRingMaxWorkerWarps = ncclSymkMaxThreads / WARP_SIZE;
constexpr int ncclGenkRingSlicesPerChunk = 2;
// Simple Ring groups its FIFO slots into two slices per chunk.
constexpr size_t ncclGenkRingSimpleChannelSize = 1 << 22;
constexpr int ncclGenkRingSimpleFifoSlices = 2 * ncclGenkRingSlicesPerChunk;
constexpr int ncclGenkRingLLFifoSlices = ncclFlowFifoSlots;
constexpr int ncclGenkRingLL128FifoSlices = ncclFlowFifoSlots;
constexpr int ncclGenkTreeMaxWorkerWarps = ncclSymkMaxThreads / WARP_SIZE;
constexpr int ncclGenkTreeParts = 2;
constexpr int ncclGenkTreeFifoSlices = ncclFlowFifoSlots;
constexpr int ncclGenkTreeConnectionsPerPart = 3;
constexpr int ncclGenkRingConnections = 1;
constexpr int ncclGenkTreeConnections = ncclGenkTreeParts * ncclGenkTreeConnectionsPerPart;
constexpr size_t ncclGenkTreeSimpleSliceBytes = 1 << 19;
constexpr size_t ncclGenkTreeSimpleChannelSize =
  ncclGenkTreeSimpleSliceBytes * ncclGenkTreeFifoSlices * ncclGenkTreeConnections;
constexpr size_t ncclGenkRingLLChunkPayloadBytes = NCCL_LL_LINES_PER_THREAD * ncclSymkMaxThreads * sizeof(uint64_t);
constexpr size_t ncclGenkTreeLLChunkPayloadBytes = ncclGenkRingLLChunkPayloadBytes;
constexpr size_t ncclGenkRingLLChannelWireBytes =
  NCCL_LL_LINES_PER_THREAD * ncclSymkMaxThreads * ncclFlowFifoSlots * sizeof(ncclLLFifoLine);
constexpr size_t ncclGenkTreeLLChannelWireBytes =
  NCCL_LL_LINES_PER_THREAD * ncclSymkMaxThreads * ncclFlowFifoSlots * ncclGenkTreeConnections * sizeof(ncclLLFifoLine);
// Ring and Tree use the same LL128 wire slot size.
constexpr size_t ncclGenkLL128SlotSize = 256 << 10;
constexpr size_t ncclGenkRingLL128ChunkPayloadBytes =
  ncclGenkLL128SlotSize * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
constexpr size_t ncclGenkTreeLL128ChunkPayloadBytes = ncclGenkRingLL128ChunkPayloadBytes;
constexpr size_t ncclGenkRingLL128ChannelWireBytes = ncclGenkLL128SlotSize * ncclFlowFifoSlots;
constexpr size_t ncclGenkTreeLL128ChannelWireBytes =
  ncclGenkLL128SlotSize * ncclFlowFifoSlots * ncclGenkTreeConnections;
// General kernels use a dedicated device communicator whose selected GIN backend matches this mask.
constexpr unsigned ncclGenkGinBackendMask = 1u << (unsigned)NCCL_NET_DEVICE_GIN_PROXY;

// AlltoAll LL work is expressed as canonical 8B payload lines. Warps own
// domain-specific peer-local chunks and chunks are assigned round-robin over the
// complete grid, so CTA boundaries do not constrain peer or domain ownership.
constexpr size_t ncclSymkAlltoAllLLPayloadBytesPerLine = 8;
constexpr size_t ncclSymkAlltoAllLLWireBytesPerLine = 16;
constexpr size_t ncclSymkAlltoAllLLScratchAlignment = 128;
constexpr size_t ncclSymkAlltoAllLLMaxPayloadBytes = 1 << 20;
constexpr int ncclSymkAlltoAllLLReceivePlaneCount = 2;
constexpr int ncclSymkAlltoAllLLSendPlaneCount = 2;
constexpr int ncclSymkAlltoAllLLProvisionPlaneCount =
  ncclSymkAlltoAllLLReceivePlaneCount + ncclSymkAlltoAllLLSendPlaneCount;
constexpr int ncclSymkAlltoAllLLLsaLinesPerWarp = WARP_SIZE;
constexpr size_t ncclSymkAlltoAllLLGinSingleChunkMaxPayloadBytes = 2 << 10;
constexpr int ncclSymkAlltoAllLLGinMaxLinesPerWarp = 512;

// The fixed scratch allocation contains one global state followed by two
// receive planes and two GIN send-staging planes. Receive planes cover every
// non-self peer, while send planes only cover GIN peers because LSA publishes
// directly to remote receive scratch. Both alternate by epoch, so operation k
// can leave its GIN reads in flight while operation k+1 uses the other send
// plane.
struct ncclSymkAlltoAllLLScratchHandle {
  ncclDevResourceHandle bufHandle;
  uint32_t receivePlaneBytes;
  uint32_t sendPlaneBytes;
};

struct alignas(128) ncclSymkAlltoAllLLScratchState {
  // One epoch advances per AlltoAll_LL invocation and is shared by every CTA.
  // publishedEpoch is zero while idle, one while a CTA is publishing, and the
  // active LL flag otherwise. Epochs zero and one are never used as LL flags.
  uint32_t epoch;
  uint32_t publishedEpoch;
  uint32_t completedCtas;
  // Rollover-only local grid rendezvous. CTA 0 publishes resetReady after the
  // pre-reset world barrier; resetCompletedCtas lets it wait for every CTA's
  // disjoint clear before entering the post-reset world barrier.
  uint32_t resetReady;
  uint32_t resetCompletedCtas;
};
static_assert(sizeof(ncclSymkAlltoAllLLScratchState) == ncclSymkAlltoAllLLScratchAlignment,
              "Keep the global scratch state 128B aligned");

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLReceivePlaneOffset(uint32_t epoch, size_t receivePlaneBytes) {
  return (epoch & 1) * receivePlaneBytes;
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLSendPlaneOffset(uint32_t epoch, size_t receivePlaneBytes,
                                                                       size_t sendPlaneBytes) {
  return ncclSymkAlltoAllLLReceivePlaneCount * receivePlaneBytes + (epoch & 1) * sendPlaneBytes;
}

constexpr __host__ __device__ int ncclSymkAlltoAllLLLineCount(size_t payloadBytes) {
  return (int)(((payloadBytes > 0 ? payloadBytes : 1) + ncclSymkAlltoAllLLPayloadBytesPerLine - 1) /
               ncclSymkAlltoAllLLPayloadBytesPerLine);
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLWireBytes(size_t payloadBytes) {
  return (size_t)ncclSymkAlltoAllLLLineCount(payloadBytes) * ncclSymkAlltoAllLLWireBytesPerLine;
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLTotalLineCount(int worldRanks, size_t payloadBytes) {
  return worldRanks > 1 ? (size_t)(worldRanks - 1) * ncclSymkAlltoAllLLLineCount(payloadBytes) : 0;
}

// For small payloads, one chunk covers the exact LL-rounded peer payload. At
// larger sizes, grow power-of-two, whole-warp chunks while retaining at least
// two chunks per peer.
// The 512-line automatic cap preserves enough independent GPU packing and GIN
// posting work as payloads grow.
constexpr __host__ __device__ int ncclSymkAlltoAllLLGinLinesPerWarp(size_t payloadBytes) {
  int linesPerPeer = ncclSymkAlltoAllLLLineCount(payloadBytes);
  if (payloadBytes <= ncclSymkAlltoAllLLGinSingleChunkMaxPayloadBytes) return linesPerPeer;

  int linesPerWarp = WARP_SIZE;
  while (linesPerWarp < ncclSymkAlltoAllLLGinMaxLinesPerWarp && 2 * linesPerWarp < linesPerPeer) {
    linesPerWarp *= 2;
  }
  return linesPerWarp;
}

constexpr __host__ __device__ int ncclSymkAlltoAllLLChunksPerPeer(size_t payloadBytes, int linesPerWarp) {
  return (ncclSymkAlltoAllLLLineCount(payloadBytes) + linesPerWarp - 1) / linesPerWarp;
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLChunkCount(int peerCount, size_t payloadBytes,
                                                                  int linesPerWarp) {
  return peerCount > 0 ? (size_t)peerCount * ncclSymkAlltoAllLLChunksPerPeer(payloadBytes, linesPerWarp) : 0;
}

// LSA keeps small chunks for NVLink store parallelism while GIN uses larger
// chunks to reduce put posting. Their chunk counts share one global warp pool.
constexpr __host__ __device__ int ncclSymkAlltoAllLLUsefulWarpCount(int lsaPeerCount, int ginPeerCount,
                                                                    size_t payloadBytes, int lsaLinesPerWarp,
                                                                    int ginLinesPerWarp, int warpsPerBlock) {
  size_t chunks = ncclSymkAlltoAllLLChunkCount(lsaPeerCount, payloadBytes, lsaLinesPerWarp) +
                  ncclSymkAlltoAllLLChunkCount(ginPeerCount, payloadBytes, ginLinesPerWarp);
  size_t maxWarps = (size_t)ncclSymkMaxBlocks * warpsPerBlock;
  return (int)(chunks == 0 ? 1 : chunks < maxWarps ? chunks : maxWarps);
}

constexpr __host__ __device__ int ncclSymkAlltoAllLLBlockCount(int lsaPeerCount, int ginPeerCount, size_t payloadBytes,
                                                               int lsaLinesPerWarp, int ginLinesPerWarp,
                                                               int warpsPerBlock) {
  int warps = ncclSymkAlltoAllLLUsefulWarpCount(lsaPeerCount, ginPeerCount, payloadBytes, lsaLinesPerWarp,
                                                ginLinesPerWarp, warpsPerBlock);
  return (warps + warpsPerBlock - 1) / warpsPerBlock;
}

constexpr __host__ __device__ int ncclSymkAlltoAllLLSelectedBlockCount(
  int lsaPeerCount, int ginPeerCount, size_t payloadBytes, int lsaLinesPerWarp, int ginLinesPerWarp, int warpsPerBlock,
  int residentBlockLimit, int resolvedBlockLimit, int userBlockLimit) {
  int limit = residentBlockLimit > 0 ? residentBlockLimit : 1;
  if (resolvedBlockLimit > 0 && resolvedBlockLimit < limit) limit = resolvedBlockLimit;
  if (userBlockLimit > 0 && userBlockLimit < limit) limit = userBlockLimit;
  int automaticBlocks = ncclSymkAlltoAllLLBlockCount(lsaPeerCount, ginPeerCount, payloadBytes, lsaLinesPerWarp,
                                                     ginLinesPerWarp, warpsPerBlock);
  return automaticBlocks < limit ? automaticBlocks : limit;
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLRequiredPeerPlaneBytes(int peerCount, size_t payloadBytes) {
  size_t bytes = peerCount > 0 ? (size_t)peerCount * ncclSymkAlltoAllLLWireBytes(payloadBytes) : 0;
  return (bytes + ncclSymkAlltoAllLLScratchAlignment - 1) & ~(ncclSymkAlltoAllLLScratchAlignment - 1);
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLRequiredReceivePlaneBytes(int worldRanks, size_t payloadBytes) {
  return ncclSymkAlltoAllLLRequiredPeerPlaneBytes(worldRanks > 1 ? worldRanks - 1 : 0, payloadBytes);
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLRequiredSendPlaneBytes(int worldRanks, int lsaRanks,
                                                                              size_t payloadBytes) {
  return ncclSymkAlltoAllLLRequiredPeerPlaneBytes(worldRanks > lsaRanks ? worldRanks - lsaRanks : 0, payloadBytes);
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLRequiredScratchBytes(int worldRanks, int lsaRanks,
                                                                            size_t payloadBytes) {
  return sizeof(ncclSymkAlltoAllLLScratchState) +
         ncclSymkAlltoAllLLReceivePlaneCount * ncclSymkAlltoAllLLRequiredReceivePlaneBytes(worldRanks, payloadBytes) +
         ncclSymkAlltoAllLLSendPlaneCount *
           ncclSymkAlltoAllLLRequiredSendPlaneBytes(worldRanks, lsaRanks, payloadBytes);
}

// Preserve a topology-independent payload provision by treating the configured
// budget as four worst-case planes. The actual allocation compacts GIN send
// planes according to the topology and can therefore be smaller than the budget.
constexpr __host__ __device__ size_t ncclSymkAlltoAllLLPlaneCapacityForBudget(size_t scratchBytes) {
  if (scratchBytes <= sizeof(ncclSymkAlltoAllLLScratchState)) return 0;
  size_t planeBytes = (scratchBytes - sizeof(ncclSymkAlltoAllLLScratchState)) / ncclSymkAlltoAllLLProvisionPlaneCount;
  return planeBytes & ~(ncclSymkAlltoAllLLScratchAlignment - 1);
}

constexpr __host__ __device__ size_t ncclSymkAlltoAllLLRequiredScratchBudgetBytes(int worldRanks, size_t payloadBytes) {
  return sizeof(ncclSymkAlltoAllLLScratchState) +
         ncclSymkAlltoAllLLProvisionPlaneCount * ncclSymkAlltoAllLLRequiredReceivePlaneBytes(worldRanks, payloadBytes);
}

constexpr __host__ __device__ bool ncclSymkAlltoAllLLScratchFits(int worldRanks, size_t payloadBytes,
                                                                 size_t scratchBytes) {
  return payloadBytes != 0 && ncclSymkAlltoAllLLRequiredReceivePlaneBytes(worldRanks, payloadBytes) <=
                                ncclSymkAlltoAllLLPlaneCapacityForBudget(scratchBytes);
}

// Map a compact remote-peer offset to its direct world peer. LSA is the
// inner factor of world, so (worldRank / lsaRanks, lsaRank) are the group/lane
// coordinates. Applying opposite deltas at the receiver makes the mapping its
// own inverse for a given offset.
constexpr __host__ __device__ int ncclSymkAlltoAllLLRemotePeer(int worldRank, int worldRanks, int lsaRank, int lsaRanks,
                                                               int remotePeerOffset, bool sendNotRecv) {
  int sign = sendNotRecv ? 1 : -1;
  int lsaGroupCount = worldRanks / lsaRanks;
  int groupDelta = 1 + remotePeerOffset / lsaRanks;
  int laneDelta = remotePeerOffset % lsaRanks;
  int peerGroup = worldRank / lsaRanks + sign * groupDelta;
  int peerLane = lsaRank + sign * laneDelta;
  if (peerGroup < 0) peerGroup += lsaGroupCount;
  if (peerGroup >= lsaGroupCount) peerGroup -= lsaGroupCount;
  if (peerLane < 0) peerLane += lsaRanks;
  if (peerLane >= lsaRanks) peerLane -= lsaRanks;
  return peerGroup * lsaRanks + peerLane;
}

constexpr __host__ __device__ int ncclSymkLLMaxSlots(int eltSize = ncclSymkLLMaxEltSize) {
  return ncclSymkMaxThreads * ncclSymkLLMaxEltSize / eltSize;
}

enum ncclSymkKernelId {
  ncclSymkKernelId_AllReduce_AGxLL_R,
  ncclSymkKernelId_AllReduce_AGxLLMC_R,
  ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST,
  ncclSymkKernelId_AllReduce_RSxLD_AGxST,
  ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC,

  ncclSymkKernelId_AllGather_LL,
  ncclSymkKernelId_AllGather_LLMC,
  ncclSymkKernelId_AllGather_TmaST,
  ncclSymkKernelId_AllGather_ST,
  ncclSymkKernelId_AllGather_TmaSTMC,
  ncclSymkKernelId_AllGather_STMC,
  ncclSymkKernelId_AllGather_RailRing_LsaSTMC,

  ncclSymkKernelId_ReduceScatter_LL,
  ncclSymkKernelId_ReduceScatter_TmaLD,
  ncclSymkKernelId_ReduceScatter_LD,
  ncclSymkKernelId_ReduceScatter_LDMC,
  ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD,
  ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC,

  // Append Genk IDs so the original Symk tuning bits remain unchanged.
  ncclSymkKernelId_AllReduce_Ring_Simple,
  ncclSymkKernelId_AllReduce_Tree_Simple,
  ncclSymkKernelId_AllGather_Ring_Simple,
  ncclSymkKernelId_ReduceScatter_Ring_Simple,

  ncclSymkKernelId_Broadcast_Ring_Simple,
  ncclSymkKernelId_Reduce_Ring_Simple,

  ncclSymkKernelId_AllReduce_Ring_LL,
  ncclSymkKernelId_AllReduce_Tree_LL,
  ncclSymkKernelId_AllGather_Ring_LL,
  ncclSymkKernelId_ReduceScatter_Ring_LL,
  ncclSymkKernelId_Broadcast_Ring_LL,
  ncclSymkKernelId_Reduce_Ring_LL,

  ncclSymkKernelId_AllReduce_Ring_LL128,
  ncclSymkKernelId_AllReduce_Tree_LL128,
  ncclSymkKernelId_AllGather_Ring_LL128,
  ncclSymkKernelId_ReduceScatter_Ring_LL128,
  ncclSymkKernelId_Broadcast_Ring_LL128,
  ncclSymkKernelId_Reduce_Ring_LL128,

  // Append new kernels so existing tuning IDs remain stable.
  ncclSymkKernelId_AlltoAll_LL,

  ncclSymkKernelId_Count
};

using ncclSymkKernelMask = uint64_t;
static_assert(ncclSymkKernelId_Count <= 8 * sizeof(ncclSymkKernelMask), "Symmetric kernel mask is too small");

constexpr char const* ncclSymKernelStr[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R", "AllReduce_AGxLLMC_R", "AllReduce_RSxTmaLD_AGxTmaST", "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC", "AllGather_LL", "AllGather_LLMC", "AllGather_TmaST", "AllGather_ST", "AllGather_TmaSTMC",
  "AllGather_STMC", "AllGather_RailRing_LsaSTMC", "ReduceScatter_LL", "ReduceScatter_TmaLD", "ReduceScatter_LD",
  "ReduceScatter_LDMC", "ReduceScatter_RailA2A_LsaLD", "ReduceScatter_RailA2A_LsaLDMC",
  // Genk kernels
  "AllReduce_Ring_Simple", "AllReduce_Tree_Simple", "AllGather_Ring_Simple", "ReduceScatter_Ring_Simple",
  "Broadcast_Ring_Simple", "Reduce_Ring_Simple", "AllReduce_Ring_LL", "AllReduce_Tree_LL", "AllGather_Ring_LL",
  "ReduceScatter_Ring_LL", "Broadcast_Ring_LL", "Reduce_Ring_LL", "AllReduce_Ring_LL128", "AllReduce_Tree_LL128",
  "AllGather_Ring_LL128", "ReduceScatter_Ring_LL128", "Broadcast_Ring_LL128", "Reduce_Ring_LL128", "AlltoAll_LL"
};

// Per-channel view of the finalized topology ring for this rank. index is this rank's position in the ring when rank
// zero is assigned index zero; prev, next, and userRanks entries are world ranks. userRanks is ordered from this rank,
// so userRanks[0] is self, userRanks[1] is next, and userRanks[nRanks-1] is prev.
struct ncclGenkRing {
  int prev;
  int next;
  int index;
  int const* userRanks;
};

// Per-channel view of the finalized topology tree for this rank: at most one LSA child and two GIN children. All
// peers are world ranks. Indices identify the fixed ncclFlowProcessor FIFO and state used at both ends of an edge.
// A -1 entry is unused.
struct ncclGenkTree {
  int upLsa;
  int upGin;
  int upIndex;
  int downLsa;
  int downLsaIndex;
  int downGin[2];
  int downGinIndices[2];
};

// Device state for the general Ring/Tree kernels built on ncclFlowProcessor.
struct ncclGenkDevComm {
  struct ncclDevComm devComm;
#if !defined(NCCL_OS_WINDOWS)
  ncclGinSignal_t ginSignal0;
  // GIN send staging is packed by active logical CTA channel: Ring buffers first, then Tree buffers.
  // Each mask marks channels with a staging buffer.
  ncclDevResourceHandle ginFlowBuffer;
  uint64_t ringGinChannelMask;
  uint64_t treeGinChannelMask;
  static_assert(ncclSymkMaxBlocks <= sizeof(ringGinChannelMask) * 8);
  static_assert(ncclSymkMaxBlocks <= sizeof(treeGinChannelMask) * 8);
#endif
  // Resources in devComm's window.
  ncclDevResourceHandle ringBuffer;
  ncclDevResourceHandle treeBuffer;
  ncclDevResourceHandle ringLLBuffer;
  ncclDevResourceHandle treeLLBuffer;
  ncclDevResourceHandle ringLL128Buffer;
  ncclDevResourceHandle treeLL128Buffer;
  ncclDevResourceHandle connState; // Ring states followed by Tree states.
  // Profiler counters (host-pinned), indexed by channel id and per-channel entry.
  // workPhases holds per-phase timestamps published behind a fence.
  struct ncclDevProfiler* workStarted;
  struct ncclDevProfiler* workCompleted;
  struct ncclDevProfilerPhases* workPhases;
  int nResourceChannels; // Per-channel FIFOs, connection states, and signals allocated at initialization.
  int nRingChannels; // Distinct ring topologies before copyChannels expansion.
  int nTreeChannels; // Distinct tree topologies before copyChannels expansion.
  int nTreeSearchChannels; // Tree channels before double-tree expansion.
  struct ncclGenkRing* rings;
  struct ncclGenkTree* trees;
};

// Device state for the specialized symmetric kernels.
struct ncclSymkDevComm {
  struct ncclDevComm devComm;
  struct ncclLLA2AHandle lsaLLA2A;
  struct ncclGinOutboxHandle ginOutbox;
  struct ncclGinInboxA2AHandle ginInboxRail;
  struct ncclSymkAlltoAllLLScratchHandle alltoAllLLScratch;
  struct ncclGinSyncHandle ginSyncHandle;
  ncclDevResourceHandle rsGinAccumBuf;
  uint32_t rsGinAccumBytesPerBlock;
  // Profiler counters (host-pinned), indexed by channel id and per-channel entry.
  // workPhases holds per-phase timestamps published behind a fence.
  struct ncclDevProfiler* workStarted;
  struct ncclDevProfiler* workCompleted;
  struct ncclDevProfilerPhases* workPhases;
};

struct ncclSymkState {
  bool initialized;
  bool genkInitStarted;
  bool genkInitialized;
  bool hasLsaMultimem;
  int maxGinInboxBlocks;
  int alltoAllLLMaxResidentCtas;
#if !defined(NCCL_OS_WINDOWS)
  struct {
    ncclDevResourceHandle bufHandle;
    ncclGinSignal_t signal0;
  } genkGinFlow;
#endif
  struct ncclSymkDevComm kcomm;
  struct ncclGenkDevComm genkComm;
  uint32_t alltoAllLLProvisionBytes;
};

struct ncclSymkChannelWorkRange {
  uint16_t workHi; // inclusive index of my ending work
  uint16_t fracHi; // 16-bit fraction in (0.0, 1.0] indicating where my part ends
};

// 16 bytes aligned
struct alignas(16) ncclSymkDevWork {
  uint64_t redOpArg; // must be collectively uniform
  size_t nElts;
  struct ncclWindow_vidmem *inputWin, *outputWin;
  size_t inputOff, outputOff; // these = origUserOffset + cbdPartOffset
  int rootRank;
  uint64_t sChannelId:16, nChannels:16, padding:32;
};

// Device-side profiling requested for a launch; KernelPhase implies KernelCh. Only the
// symmetric kernels stamp phases, and they read these bits at runtime.
enum ncclDevProfilerMode : uint8_t {
  ncclDevProfilerModeNone = 0,
  ncclDevProfilerModeKernelCh = 1 << 0,
  ncclDevProfilerModeKernelPhase = 1 << 1,
};

struct alignas(16) ncclSymkDevWorkArgs {
  struct ncclSymkDevComm kcomm;
  int nMaxChannels;
  int maxDynamicSmem;
  uint8_t profilerMode; // ncclDevProfilerMode bits; nonzero means profilerWorkCounters[nMaxChannels] follows
};

struct alignas(16) ncclGenkDevWorkArgs {
  struct ncclGenkDevComm kcomm;
  int nMaxChannels;
  int maxDynamicSmem;
  uint8_t profilerMode; // ncclDevProfilerMode bits; nonzero means profilerWorkCounters[nMaxChannels] follows
  int minChunkPayloadBytes;
};

// Both kernel families use the same variable-length trailing data layout:
//   if profilerMode: uint64_t profilerWorkCounters[nMaxChannels] (aligned to 16)
//   ncclSymkChannelWorkRange[nChannels] (aligned to 16)
//   ncclSymkDevWork[nWorks]
template <typename Args>
__host__ constexpr size_t ncclSymkDevWorkArgsSize(int nChannels, int nWorks, bool profiler = false) {
  return alignUp(sizeof(Args), 16) + (profiler ? alignUp(nChannels * sizeof(uint64_t), 16) : size_t(0)) +
         alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16) + nWorks * sizeof(struct ncclSymkDevWork);
}

template <typename Args>
__host__ __device__ uint64_t* ncclSymkGetProfilerCounters(Args const* args) {
  return (uint64_t*)((uint8_t*)args + alignUp(sizeof(Args), 16));
}

template <typename Args>
__host__ __device__ struct ncclSymkChannelWorkRange* ncclSymkGetWorkRange(Args const* args) {
  size_t off = alignUp(sizeof(Args), 16);
  if (args->profilerMode) off += alignUp(args->nMaxChannels * sizeof(uint64_t), 16);
  return (struct ncclSymkChannelWorkRange*)((uint8_t*)args + off);
}

template <typename Args>
__host__ __device__ struct ncclSymkDevWork* ncclSymkGetWorks(Args const* args, int nChannels) {
  return (struct ncclSymkDevWork*)((uint8_t*)ncclSymkGetWorkRange(args) +
                                   alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16));
}

union ncclSymkDevWorkArgs4K {
  struct ncclSymkDevWorkArgs args;
  char buf4K[4096];
};

union ncclGenkDevWorkArgs4K {
  struct ncclGenkDevWorkArgs args;
  char buf4K[4096];
};

typedef enum {
  ncclSymSendNonregRecvNonreg = 0,
  ncclSymSendNonregRecvReg = 1,
  ncclSymSendRegRecvNonreg = 2,
  ncclSymSendRegRecvReg = 3,
  ncclNumSymRegTypes = 4
} ncclSymRegType_t;

// We assume ncclComm contains a field: `ncclSymkState symkState`
ncclResult_t ncclSymkInitOnce(struct ncclComm* comm);
ncclResult_t ncclGenkInitStart(struct ncclComm* comm, bool* needGenkDevComm);
ncclResult_t ncclGenkInitEnd(struct ncclComm* comm);
ncclResult_t ncclSymkFinalize(struct ncclComm* comm);

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                       size_t nElts);
bool ncclSymkAlltoAllLLForced(struct ncclComm* comm);
ncclSymkKernelMask ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                                size_t nElts, bool symAligned16B = true);

ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork);
bool ncclSymkTmaAvailable(struct ncclComm* comm);
bool ncclSymkTmaDeepEligible(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, int nBlocks);

// Generated by src/device/symmetric/generate.py
extern int const ncclSymkKernelCount;
extern void* ncclSymkKernelList[/*ncclSymkKernelCount*/];
// Instrumented variants, indexed identically to ncclSymkKernelList. Selected
// (instead of ncclSymkKernelList) only when kernel-channel profiling is active.
extern void* ncclSymkKernelListProfile[/*ncclSymkKernelCount*/];
extern int ncclSymkKernelRequirements[/*ncclSymkKernelCount*/];
extern int ncclSymkKernelMaxDynamicSmem[/*ncclSymkKernelCount*/]; // initialized by ncclInitKernelsForDevice()
extern int ncclSymkKernelMaxThreads[/*ncclSymkKernelCount*/]; // initialized by ncclInitKernelsForDevice()
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);
int ncclSymkKernelMaxWarps(ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);

// Generated by src/device/general/generate.py
extern int const ncclGenkKernelCount;
extern void* ncclGenkKernelList[/*ncclGenkKernelCount*/];
extern void* ncclGenkKernelListProfile[/*ncclGenkKernelCount*/];
extern int ncclGenkKernelRequirements[/*ncclGenkKernelCount*/];
extern int ncclGenkKernelMaxDynamicSmem[/*ncclGenkKernelCount*/];
int ncclGenkGetKernelIndex(ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);

const char* ncclSymkKernelIdToString(int kernelId);
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                               ncclSymRegType_t* winRegType);

ncclSymkKernelMask ncclSymkLLKernelMask();
ncclSymkKernelMask ncclSymkDynamicSmemKernelMask();
ncclSymkKernelMask ncclSymkTmaKernelMask();
ncclSymkKernelMask ncclSymkGinKernelMask();
ncclSymkKernelMask ncclGenkKernelMask();
ncclSymkKernelMask ncclSymkAGKernelMask();
ncclSymkKernelMask ncclSymkARKernelMask();
ncclSymkKernelMask ncclSymkRSKernelMask();
ncclSymkKernelMask ncclSymkNonGenkKernelMask();
size_t ncclSymkRsGinChunkBytes();

constexpr int ncclSymkAllGather_RailRing_ChunkSize = 1 << 20;

constexpr int ncclSymkMinWarpsPerBlock = 4;
constexpr int ncclSymkBytePerPack = 16;
constexpr __host__ __device__ int ncclSymkGetBytesPerChunk(int nWarps, int unrollPacks,
                                                           int bytePerPack = ncclSymkBytePerPack) {
  return nWarps * unrollPacks * WARP_SIZE * bytePerPack;
}

// SM kernel unroll packs
constexpr int ncclSymkUnrollPacks = 4;
constexpr int ncclSymkBytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkUnrollPacks);

// TMA kernel unroll packs
constexpr int ncclSymkDeepUnrollPacks = 8;
constexpr int ncclSymkDeepBytePerChunk = ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkDeepUnrollPacks);

// ReduceScatter LD/TmaLD small chunks use ordinary loads. Keep the kernel and
// host model on these shared sizes; kernel changes also require model validation.
constexpr int ncclSymkSmallBytePerPack = 4;
constexpr int ncclSymkSmallUnrollPacks = 4;
constexpr int ncclSymkSmallBytePerChunk =
  ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkSmallUnrollPacks, ncclSymkSmallBytePerPack);

// Multimem bcast deep loop (single warp; shares unroll with ncclSymkDeepUnrollPacks)
constexpr int ncclSymkMultimemDeepBytePerChunk = ncclSymkGetBytesPerChunk(1, ncclSymkDeepUnrollPacks);

// Spread concurrent MC operations across different addresses to avoid contention.
constexpr int ncclSymkMcPerRankOffsetBytes = 32 * 1024 * 1024;
static_assert(ncclSymkMcPerRankOffsetBytes % ncclSymkMultimemDeepBytePerChunk == 0);

// Large-tile TMA deep loop (AllGather: 16 B relatively aligned, store side 256 B)
constexpr int ncclSymkAlign256BDeepUnrollPacks = 16;
constexpr int ncclSymkAlign256BDeepBytePerChunk =
  ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkAlign256BDeepUnrollPacks);

int64_t ncclParamSymGenkEnable();
#endif
