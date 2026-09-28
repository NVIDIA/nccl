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
constexpr int ncclGenkRingSlotsPerProcess = 2;
// Simple Ring uses four 1 MiB published slices in its 4 MiB FIFO and processes two slices per 2 MiB chunk.
constexpr size_t ncclGenkRingSimpleChannelSize = 1 << 22;
constexpr int ncclGenkRingSimpleFifoSlots = 2 * ncclGenkRingSlotsPerProcess;
constexpr int ncclGenkRingLLFifoSlots = NCCL_STEPS;
constexpr int ncclGenkRingLL128FifoSlots = NCCL_STEPS;
constexpr int ncclGenkTreeMaxWorkerWarps = ncclSymkMaxThreads / WARP_SIZE;
constexpr int ncclGenkTreeParts = 2;
constexpr int ncclGenkTreeFifoSlots = 8;
constexpr int ncclGenkTreeSlots = 3;
constexpr int ncclGenkRingConnections = 1;
constexpr int ncclGenkTreeConnections = ncclGenkTreeParts * ncclGenkTreeSlots;
constexpr size_t ncclGenkTreeSimpleSliceBytes = 1 << 19;
constexpr size_t ncclGenkTreeSimpleChannelSize =
  ncclGenkTreeSimpleSliceBytes * ncclGenkTreeFifoSlots * ncclGenkTreeConnections;
constexpr size_t ncclGenkRingLLChunkPayloadBytes = NCCL_LL_LINES_PER_THREAD * ncclSymkMaxThreads * sizeof(uint64_t);
constexpr size_t ncclGenkTreeLLChunkPayloadBytes = ncclGenkRingLLChunkPayloadBytes;
constexpr size_t ncclGenkRingLLChannelWireBytes =
  NCCL_LL_LINES_PER_THREAD * ncclSymkMaxThreads * ncclGenkRingLLFifoSlots * sizeof(ncclLLFifoLine);
constexpr size_t ncclGenkTreeLLChannelWireBytes = NCCL_LL_LINES_PER_THREAD * ncclSymkMaxThreads *
                                                  ncclGenkTreeFifoSlots * ncclGenkTreeConnections *
                                                  sizeof(ncclLLFifoLine);
// Ring and Tree use the same LL128 wire slot size.
constexpr size_t ncclGenkLL128SlotSize = 256 << 10;
constexpr size_t ncclGenkRingLL128ChunkPayloadBytes =
  ncclGenkLL128SlotSize * NCCL_LL128_DATAELEMS / NCCL_LL128_LINEELEMS;
constexpr size_t ncclGenkTreeLL128ChunkPayloadBytes = ncclGenkRingLL128ChunkPayloadBytes;
constexpr size_t ncclGenkRingLL128ChannelWireBytes = ncclGenkLL128SlotSize * ncclGenkRingLL128FifoSlots;
constexpr size_t ncclGenkTreeLL128ChannelWireBytes =
  ncclGenkLL128SlotSize * ncclGenkTreeFifoSlots * ncclGenkTreeConnections;
// General kernels use a dedicated device communicator whose selected GIN backend matches this mask.
constexpr unsigned ncclGenkGinBackendMask = 1u << (unsigned)NCCL_NET_DEVICE_GIN_PROXY;

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

  ncclSymkKernelId_Count
};

constexpr char const* ncclSymKernelStr[] = {
  // Must align with enum ncclSymkKernelId definition in src/include/sym_kernels.h
  "AllReduce_AGxLL_R",
  "AllReduce_AGxLLMC_R",
  "AllReduce_RSxTmaLD_AGxTmaST",
  "AllReduce_RSxLD_AGxST",
  "AllReduce_RSxLDMC_AGxSTMC",
  "AllGather_LL",
  "AllGather_LLMC",
  "AllGather_TmaST",
  "AllGather_ST",
  "AllGather_TmaSTMC",
  "AllGather_STMC",
  "AllGather_RailRing_LsaSTMC",
  "ReduceScatter_LL",
  "ReduceScatter_TmaLD",
  "ReduceScatter_LD",
  "ReduceScatter_LDMC",
  "ReduceScatter_RailA2A_LsaLD",
  "ReduceScatter_RailA2A_LsaLDMC"
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
// peers are world ranks. Slots identify the fixed ncclFlowProcessor FIFO region used at both ends of an edge. A -1
// entry is unused.
struct ncclGenkTree {
  int upLsa;
  int upGin;
  int upSlot;
  int downLsa;
  int downLsaSlot;
  int downGin[2];
  int downGinSlots[2];
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
  // Profiler counters (host-pinned), indexed by channel id and per-channel slot.
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
  struct ncclGinSyncHandle ginSyncHandle;
  ncclDevResourceHandle rsGinAccumBuf;
  uint32_t rsGinAccumBytesPerBlock;
  // Profiler counters (host-pinned), indexed by channel id and per-channel slot.
  // workPhases holds per-phase timestamps published behind a fence.
  struct ncclDevProfiler* workStarted;
  struct ncclDevProfiler* workCompleted;
  struct ncclDevProfilerPhases* workPhases;
};

struct ncclSymkState {
  bool initialized;
  bool genkInitialized;
  bool hasLsaMultimem;
  int maxGinInboxBlocks;
  struct ncclSymkDevComm kcomm;
  struct ncclGenkDevComm genkComm;
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
  // Variable-length trailing data layout:
  //   if profilerMode: uint64_t profilerWorkCounters[nMaxChannels] (aligned to 16)
  //   ncclSymkChannelWorkRange[nChannels] (aligned to 16)
  //   ncclSymkDevWork[nWorks]
  // aux functions
  __host__ static constexpr size_t calcArgsSize(int nChannels, int nWorks, bool profiler = false) {
    return alignUp(sizeof(struct ncclSymkDevWorkArgs), 16) +
           (profiler ? alignUp(nChannels * sizeof(uint64_t), 16) : size_t(0)) +
           alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16) + nWorks * sizeof(struct ncclSymkDevWork);
  }
  __host__ __device__ uint64_t* getProfilerCounters() const {
    return (uint64_t*)((uint8_t*)this + alignUp(sizeof(struct ncclSymkDevWorkArgs), 16));
  }
  __host__ __device__ struct ncclSymkChannelWorkRange* getWorkRange() const {
    size_t off = alignUp(sizeof(struct ncclSymkDevWorkArgs), 16);
    if (profilerMode) off += alignUp(nMaxChannels * sizeof(uint64_t), 16);
    return (struct ncclSymkChannelWorkRange*)((uint8_t*)this + off);
  }
  __host__ __device__ struct ncclSymkDevWork* getWorks(int nChannels) const {
    return (struct ncclSymkDevWork*)((uint8_t*)this->getWorkRange() +
                                     alignUp(nChannels * sizeof(struct ncclSymkChannelWorkRange), 16));
  }
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
ncclResult_t ncclGenkInitOnce(struct ncclComm* comm);
ncclResult_t ncclSymkFinalize(struct ncclComm* comm);

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                       size_t nElts);
uint32_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
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
int ncclSymkGetKernelIndex(ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty);

const char* ncclSymkKernelIdToString(int kernelId);
ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                               ncclSymRegType_t* winRegType);

int ncclSymkLLKernelMask();
int ncclSymkDynamicSmemKernelMask();
int ncclSymkTmaKernelMask();
int ncclSymkGinKernelMask();
int ncclSymkAGKernelMask();
int ncclSymkARKernelMask();
int ncclSymkRSKernelMask();
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

// Deep loop when input/output are 256 B-aligned
constexpr int ncclSymkAlign256BDeepUnrollPacks = 16;
constexpr int ncclSymkAlign256BDeepBytePerChunk =
  ncclSymkGetBytesPerChunk(ncclSymkMinWarpsPerBlock, ncclSymkAlign256BDeepUnrollPacks);

#endif
