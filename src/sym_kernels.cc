/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "alloc.h"
#include "comm.h"
#include "device.h"
#include "nccl_device/core.h"
#include "transport.h"
#include "tuning.h"
#include <cmath>
#include <cfloat>

constexpr ncclSymkKernelMask kernelMask_STMC =
  1ull << ncclSymkKernelId_AllGather_LLMC | 1ull << ncclSymkKernelId_AllGather_STMC |
  1ull << ncclSymkKernelId_AllGather_TmaSTMC | 1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC | 1ull << ncclSymkKernelId_ReduceScatter_LDMC |
  1ull << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr ncclSymkKernelMask kernelMask_LDMC = 1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                               1ull << ncclSymkKernelId_ReduceScatter_LDMC |
                                               1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;

constexpr ncclSymkKernelMask kernelMask_GenkLL128 =
  1ull << ncclSymkKernelId_AllReduce_Ring_LL128 | 1ull << ncclSymkKernelId_AllReduce_Tree_LL128 |
  1ull << ncclSymkKernelId_AllGather_Ring_LL128 | 1ull << ncclSymkKernelId_ReduceScatter_Ring_LL128 |
  1ull << ncclSymkKernelId_Broadcast_Ring_LL128 | 1ull << ncclSymkKernelId_Reduce_Ring_LL128;
constexpr ncclSymkKernelMask kernelMask_GenkLL128Reduce =
  1ull << ncclSymkKernelId_AllReduce_Ring_LL128 | 1ull << ncclSymkKernelId_AllReduce_Tree_LL128 |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL128 | 1ull << ncclSymkKernelId_Reduce_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_LL =
  1ull << ncclSymkKernelId_AllReduce_AGxLL_R | 1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1ull << ncclSymkKernelId_AllGather_LL | 1ull << ncclSymkKernelId_AllGather_LLMC |
  1ull << ncclSymkKernelId_ReduceScatter_LL | 1ull << ncclSymkKernelId_AllReduce_Ring_LL |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL | 1ull << ncclSymkKernelId_AllGather_Ring_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL | 1ull << ncclSymkKernelId_Broadcast_Ring_LL |
  1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128;

constexpr ncclSymkKernelMask kernelMask_Bcast = 1ull << ncclSymkKernelId_Broadcast_Ring_Simple |
                                                1ull << ncclSymkKernelId_Broadcast_Ring_LL |
                                                1ull << ncclSymkKernelId_Broadcast_Ring_LL128;
constexpr ncclSymkKernelMask kernelMask_Reduce = 1ull << ncclSymkKernelId_Reduce_Ring_Simple |
                                                 1ull << ncclSymkKernelId_Reduce_Ring_LL |
                                                 1ull << ncclSymkKernelId_Reduce_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_AG =
  1ull << ncclSymkKernelId_AllGather_LL | 1ull << ncclSymkKernelId_AllGather_LLMC |
  1ull << ncclSymkKernelId_AllGather_ST | 1ull << ncclSymkKernelId_AllGather_STMC |
  1ull << ncclSymkKernelId_AllGather_TmaST | 1ull << ncclSymkKernelId_AllGather_TmaSTMC |
  1ull << ncclSymkKernelId_AllGather_RailRing_LsaSTMC | 1ull << ncclSymkKernelId_AllGather_Ring_Simple |
  1ull << ncclSymkKernelId_AllGather_Ring_LL | 1ull << ncclSymkKernelId_AllGather_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_AR =
  1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R | 1ull << ncclSymkKernelId_AllReduce_AGxLL_R |
  1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC | 1ull << ncclSymkKernelId_AllReduce_RSxLD_AGxST |
  1ull << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST | 1ull << ncclSymkKernelId_AllReduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Tree_Simple | 1ull << ncclSymkKernelId_AllReduce_Ring_LL |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL | 1ull << ncclSymkKernelId_AllReduce_Ring_LL128 |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL128;

constexpr ncclSymkKernelMask kernelMask_RS =
  1ull << ncclSymkKernelId_ReduceScatter_LD | 1ull << ncclSymkKernelId_ReduceScatter_LDMC |
  1ull << ncclSymkKernelId_ReduceScatter_TmaLD | 1ull << ncclSymkKernelId_ReduceScatter_LL |
  1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD | 1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple | 1ull << ncclSymkKernelId_ReduceScatter_Ring_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL128;

constexpr ncclSymkKernelMask kernelMask_LSA =
  1ull << ncclSymkKernelId_AllReduce_AGxLL_R | 1ull << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1ull << ncclSymkKernelId_AllReduce_RSxLD_AGxST | 1ull << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
  1ull << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST | 1ull << ncclSymkKernelId_AllReduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Tree_Simple | 1ull << ncclSymkKernelId_AllGather_LL |
  1ull << ncclSymkKernelId_AllGather_LLMC | 1ull << ncclSymkKernelId_AllGather_ST |
  1ull << ncclSymkKernelId_AllGather_STMC | 1ull << ncclSymkKernelId_AllGather_TmaST |
  1ull << ncclSymkKernelId_AllGather_TmaSTMC | 1ull << ncclSymkKernelId_AllGather_Ring_Simple |
  1ull << ncclSymkKernelId_ReduceScatter_LL | 1ull << ncclSymkKernelId_ReduceScatter_LD |
  1ull << ncclSymkKernelId_ReduceScatter_LDMC | 1ull << ncclSymkKernelId_ReduceScatter_TmaLD |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple | 1ull << ncclSymkKernelId_Broadcast_Ring_Simple |
  1ull << ncclSymkKernelId_Reduce_Ring_Simple | 1ull << ncclSymkKernelId_AllReduce_Ring_LL |
  1ull << ncclSymkKernelId_AllReduce_Tree_LL | 1ull << ncclSymkKernelId_AllGather_Ring_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL | 1ull << ncclSymkKernelId_Broadcast_Ring_LL |
  1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128;

constexpr ncclSymkKernelMask kernelMask_GinOnly = 1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                                  1ull << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
                                                  1ull << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr ncclSymkKernelMask kernelMask_Genk =
  1ull << ncclSymkKernelId_AllReduce_Ring_Simple | 1ull << ncclSymkKernelId_AllReduce_Tree_Simple |
  1ull << ncclSymkKernelId_AllGather_Ring_Simple | 1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple |
  1ull << ncclSymkKernelId_Broadcast_Ring_Simple | 1ull << ncclSymkKernelId_Reduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Ring_LL | 1ull << ncclSymkKernelId_AllReduce_Tree_LL |
  1ull << ncclSymkKernelId_AllGather_Ring_LL | 1ull << ncclSymkKernelId_ReduceScatter_Ring_LL |
  1ull << ncclSymkKernelId_Broadcast_Ring_LL | 1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128;
constexpr ncclSymkKernelMask kernelMask_GenkReduce =
  1ull << ncclSymkKernelId_AllReduce_Ring_Simple | 1ull << ncclSymkKernelId_AllReduce_Tree_Simple |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_Simple | 1ull << ncclSymkKernelId_Reduce_Ring_Simple |
  1ull << ncclSymkKernelId_AllReduce_Ring_LL | 1ull << ncclSymkKernelId_AllReduce_Tree_LL |
  1ull << ncclSymkKernelId_ReduceScatter_Ring_LL | 1ull << ncclSymkKernelId_Reduce_Ring_LL | kernelMask_GenkLL128Reduce;
constexpr ncclSymkKernelMask kernelMask_GinCapable = kernelMask_GinOnly | kernelMask_Genk;

constexpr ncclSymkKernelMask kernelMask_Tma =
  1ull << ncclSymkKernelId_AllGather_TmaST | 1ull << ncclSymkKernelId_AllGather_TmaSTMC |
  1ull << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST | 1ull << ncclSymkKernelId_ReduceScatter_TmaLD;

constexpr ncclSymkKernelMask kernelMask_DynamicSmem = kernelMask_Tma;

constexpr ncclSymkKernelMask kernelMask_Symk = (kernelMask_AG | kernelMask_AR | kernelMask_RS) & ~kernelMask_Genk;

ncclSymkKernelMask ncclSymkLLKernelMask() {
  return kernelMask_LL;
}
ncclSymkKernelMask ncclSymkDynamicSmemKernelMask() {
  return kernelMask_DynamicSmem;
}
ncclSymkKernelMask ncclSymkTmaKernelMask() {
  return kernelMask_Tma;
}

ncclSymkKernelMask ncclSymkGinKernelMask() {
  return kernelMask_GinOnly;
}

ncclSymkKernelMask ncclGenkKernelMask() {
  return kernelMask_Genk;
}

ncclSymkKernelMask ncclSymkAGKernelMask() {
  return kernelMask_AG;
}

ncclSymkKernelMask ncclSymkARKernelMask() {
  return kernelMask_AR;
}

ncclSymkKernelMask ncclSymkRSKernelMask() {
  return kernelMask_RS;
}

ncclSymkKernelMask ncclSymkNonGenkKernelMask() {
  return kernelMask_Symk;
}

// Host picker: true when nBytes is large enough for this kernel's TMA deep loop at nBlocks.
bool ncclSymkTmaDeepEligible(struct ncclComm* comm, ncclSymkKernelId k, size_t nBytes, int nBlocks) {
  int bytePerChunk = 0;
  int chunkMod = 0; // imodFast32 divisor on deep-loop chunk count (see src/device/symmetric/*.cuh)
  switch (k) {
  case ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST:
  case ncclSymkKernelId_ReduceScatter_TmaLD:
    bytePerChunk = ncclSymkDeepBytePerChunk;
    chunkMod = comm->nRanks * nBlocks;
    break;
  case ncclSymkKernelId_AllGather_TmaST:
    // Picker bar uses the 16 B-aligned deep tier; the 256 B TMA tier may still be skipped.
    bytePerChunk = ncclSymkBytePerChunk;
    chunkMod = nBlocks;
    break;
  case ncclSymkKernelId_AllGather_TmaSTMC:
    bytePerChunk = ncclSymkMultimemDeepBytePerChunk;
    chunkMod = 1;
    break;
  default:
    WARN("Unexpected kernel id %d in ncclSymkTmaDeepEligible", (int)k);
    return false;
  }
  if (bytePerChunk == 0 || chunkMod == 0) return false;
  return nBytes >= (size_t)bytePerChunk * (size_t)chunkMod;
}

static ncclSymkKernelMask kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
  case ncclFuncBroadcast:
    return kernelMask_Bcast;
  case ncclFuncReduce:
    return kernelMask_Reduce;
  case ncclFuncAllGather:
    return kernelMask_AG;
  case ncclFuncAllReduce:
    return kernelMask_AR;
  case ncclFuncReduceScatter:
    return kernelMask_RS;
  default:
    return 0;
  }
}

NCCL_PARAM(SymGinKernelsEnable, "SYM_GIN_KERNELS_ENABLE", 1)
NCCL_PARAM(SymRsGinChunkSize, "SYM_RS_GIN_CHUNK_SIZE", -1)
NCCL_PARAM(SymTmaEnable, "SYM_TMA_ENABLE", 1)
NCCL_PARAM(SymGenkEnable, "SYM_GENK_ENABLE", 0)

static bool ncclGenkGinAvailable(struct ncclComm* comm) {
#if !defined(NCCL_OS_WINDOWS)
  if (comm->globalGinSupport != NCCL_GIN_CONNECTION_FULL) return false;

  struct ncclGinState const& ginState = comm->sharedRes->ginState;
  for (int i = 0; i < ginState.numActiveBackends; i++) {
    if (ginState.backends[i].ginType == NCCL_GIN_TYPE_PROXY) return true;
  }
#else
  (void)comm;
#endif
  return false;
}

bool ncclSymkTmaAvailable(struct ncclComm* comm) {
  // TMA requires up to (8KB data + 8B mbarrier + alignment) x 16 warps SMEM.
  // SMEM is partitioned across the 16 warps such that each warp gets ncclTmaShmemScratchWarpSize() bytes.
  if (comm->maxSharedMemOptin < ncclTmaShmemScratchWarpSize() * 16) {
    return false;
  }
  return comm->minCompCap >= 100 && ncclParamSymTmaEnable();
}

static constexpr size_t ncclSymkRsGinDefaultChunkBytes = 128 << 10;
static constexpr size_t ncclSymkRsGinMinChunkBytes = 128;
static constexpr size_t ncclSymkRsGinMaxChunkBytes = size_t(1) << 30;

size_t ncclSymkRsGinChunkBytes() {
  int64_t param = ncclParamSymRsGinChunkSize();
  size_t chunkBytes = param > 0 ? (size_t)param : ncclSymkRsGinDefaultChunkBytes;
  chunkBytes = std::max(ncclSymkRsGinMinChunkBytes, std::min(chunkBytes, ncclSymkRsGinMaxChunkBytes));
  return pow2Down(chunkBytes);
}

static uint32_t ncclSymkRsGinAccumBytesPerBlock() {
  return (uint32_t)alignUp(2 * ncclSymkRsGinChunkBytes(), 128);
}

static void getRequirements_gin(struct ncclComm* comm, int* out_nBlocks, size_t* out_bufSize) {
  *out_nBlocks = 0;
  *out_bufSize = 0;
  for (int ldmc = 0; ldmc <= 1; ldmc++) {
    double lsaBw = ncclTuningGetLsaBw(comm);
    double ginBw = ncclTuningGetGinBw(comm);
    double ginLat = ncclTuningGetGinLat(comm);
    double smLat = ncclTuningGetSmLatReduceScatterRailA2A(comm, ldmc);
    double smMul, lsaMul, ginMul;
    ncclTuningGetBusMulReduceScatterRailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
    // GIN could be throttled by LSA work
    double ginBwRenorm = std::min(lsaBw / lsaMul, ginBw / ginMul) * ginMul;
    size_t bufSize = ginBwRenorm * (ginLat + smLat);
    int nBlocks = ncclTuningCalcSatBlocksReduceScatterRailA2A(comm, ldmc);
    if (comm->rank == 0) {
      double minLsaGinEffBw = std::min(lsaBw / lsaMul, ginBw / ginMul);
      INFO(NCCL_TUNING, "ReduceScatter_RailA2A_Lsa%s : satblocks=%d bufsize=%d effbw=%g", ldmc ? "LDMC" : "LD", nBlocks,
           (int)bufSize, minLsaGinEffBw * smMul);
    }
    *out_nBlocks = std::max(*out_nBlocks, nBlocks);
    *out_bufSize = std::max(*out_bufSize, bufSize);
  }
}

extern int64_t ncclParamSymCTAs();

ncclResult_t ncclSymkInitOnce(struct ncclComm* comm) {
  // ncclTeamLsa() below calls this internally but drops the error code so we do it here.
  NCCLCHECK(ncclDevrInitOnce(comm));

  struct ncclSymkState* symk = &comm->symkState;
  if (!symk->initialized) {
    symk->initialized = true;
    struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    // Disable LSA multicast for cross-clique since NVLS isn't available across cliques
    symk->hasLsaMultimem =
      ncclNvlsSymmetricMultimemEnabled(comm) && ncclTeamLsa(comm).nRanks > 2 && !comm->p2pCrossClique;
    reqs.lsaMultimem = symk->hasLsaMultimem;
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;
    reqs.ginStrongSignalsRequired = false;
    reqs.ginVaSignalsRequired = false;

    struct ncclDevResourceRequirements lla2aReq;
    ncclLLA2ACreateRequirement(ncclSymkMaxBlocks,
                               ncclLLA2ACalcSlots(ncclTeamLsa(comm).nRanks * ncclSymkMaxThreads, ncclSymkLLMaxEltSize),
                               &symk->kcomm.lsaLLA2A, &lla2aReq);
    lla2aReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &lla2aReq;

    struct ncclDevResourceRequirements ginInboxRailReq = {};
    struct ncclDevResourceRequirements ginOutboxReq = {};
    struct ncclDevResourceRequirements rsGinAccumReq = {};
    struct ncclDevResourceRequirements railSignalReq = {};
    if (ncclParamSymGinKernelsEnable() && ncclTeamLsa(comm).nRanks < comm->nRanks) {
      int maxBlocks;
      size_t bufSize;
      getRequirements_gin(comm, &maxBlocks, &bufSize);

      maxBlocks = std::max(maxBlocks, comm->config.minCTAs);
      maxBlocks = std::min(maxBlocks, comm->config.maxCTAs);
      if (ncclParamSymCTAs() >= 1) maxBlocks = ncclParamSymCTAs();
      maxBlocks = std::min(maxBlocks, ncclSymkMaxBlocks);
      symk->maxGinInboxBlocks = maxBlocks;
      symk->kcomm.rsGinAccumBytesPerBlock = ncclSymkRsGinAccumBytesPerBlock();

      rsGinAccumReq.bufferSize = (size_t)maxBlocks * symk->kcomm.rsGinAccumBytesPerBlock;
      rsGinAccumReq.bufferAlign = 128;
      rsGinAccumReq.outBufferHandle = &symk->kcomm.rsGinAccumBuf;
      rsGinAccumReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &rsGinAccumReq;

      ncclGinInboxA2ACreateRequirement(ncclTeamRail(comm), maxBlocks, log2Up(bufSize), &symk->kcomm.ginInboxRail,
                                       &ginInboxRailReq);
      ginInboxRailReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &ginInboxRailReq;

      ncclGinOutboxCreateRequirement(maxBlocks, log2Up(bufSize), &symk->kcomm.ginOutbox, &ginOutboxReq);
      ginOutboxReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &ginOutboxReq;

      uint32_t railSignalCount = ncclTeamRail(comm).nRanks * ncclSymkMaxBlocks;

      railSignalReq.bufferSize = 0;
      railSignalReq.bufferAlign = 0;
      railSignalReq.outBufferHandle = nullptr;
      railSignalReq.ginSignalCount = railSignalCount;
      railSignalReq.outGinSignalStart = &symk->kcomm.ginSyncHandle.railSignals;
      railSignalReq.next = reqs.resourceRequirementsList;
      reqs.resourceRequirementsList = &railSignalReq;
      reqs.barrierCount = ncclSymkMaxBlocks;
      reqs.ginConnectionType = NCCL_GIN_CONNECTION_RAIL;
      reqs.ginStrongSignalsRequired = true;
      reqs.ginVaSignalsRequired = true;
    }

    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm, /*isInternal=*/true,
                                         /*deviceCodeVersion=*/NCCL_VERSION_CODE));
    // Dedicated sym profiler buffers, kept separate from the regular kernels' so the
    // sym workCounter never interleaves with device channels[].workCounter.
    symk->kcomm.workStarted = comm->profiler.symWorkStarted;
    symk->kcomm.workCompleted = comm->profiler.symWorkCompleted;
    symk->kcomm.workPhases = comm->profiler.symWorkPhases;
  }
  return ncclSuccess;
}

ncclResult_t ncclSymkFinalize(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->genkInitialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->genkComm.devComm));
    symk->genkInitialized = false;
  }
  if (symk->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->kcomm.devComm));
  }
  return ncclSuccess;
}

static bool ncclGenkReduceImplemented(int /*ncclDevRedOp_t*/ red) {
  return red == ncclDevSum || red == ncclDevProd || red == ncclDevMinMax;
}

static bool ncclSymkOtherReduceImplemented(int /*ncclDevRedOp_t*/ red, ncclDataType_t ty, ncclFunc_t coll) {
  bool isFloat;
  switch (ty) {
  case ncclFloat64:
  case ncclFloat32:
  case ncclFloat16:
  case ncclBfloat16:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    isFloat = true;
    break;
  default:
    isFloat = false;
    break;
  }
  if (coll == ncclFuncReduceScatter) {
    return (red == ncclDevSum || red == ncclDevSumPostDiv) && isFloat && ty != ncclFloat64;
  }
  return (red == ncclDevSum) && isFloat && ty != ncclFloat64;
}

static bool ncclSymkImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
  switch (coll) {
  case ncclFuncBroadcast:
  case ncclFuncAllGather:
    return true;
  case ncclFuncReduce:
  case ncclFuncAllReduce:
  case ncclFuncReduceScatter:
    return ncclGenkReduceImplemented(red) || ncclSymkOtherReduceImplemented(red, ty, coll);
  default:
    return false;
  }
}

ncclSymkKernelMask ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                                size_t nElts, bool symAligned16B) {
  ncclSymkKernelMask kmask = kernelMask_coll(coll);

  bool const isReduction = coll == ncclFuncReduce || coll == ncclFuncAllReduce || coll == ncclFuncReduceScatter;
  if (isReduction) {
    if (!ncclGenkReduceImplemented(red)) kmask &= ~kernelMask_GenkReduce;
    if (!ncclSymkOtherReduceImplemented(red, ty, coll)) kmask &= kernelMask_GenkReduce;
  }

  bool hasSTMC = comm->symkState.hasLsaMultimem;
  bool hasLDMC = false;
  if (comm->symkState.hasLsaMultimem) {
    switch (ty) {
    case ncclInt32:
    case ncclUint32:
    case ncclInt64:
    case ncclUint64:
    case ncclFloat16:
    case ncclBfloat16:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax || red == ncclDevSumPostDiv;
      break;
    case ncclFloat8e4m3:
    case ncclFloat8e5m2:
      hasLDMC = red == ncclDevSum || red == ncclDevMinMax || red == ncclDevSumPostDiv;
      hasLDMC &= comm->compCap >= 100;
      break;
    case ncclFloat:
    case ncclDouble:
      hasLDMC = red == ncclDevSum || red == ncclDevSumPostDiv;
      break;
    default:
      break;
    }
  }
  if (!hasSTMC) kmask &= ~kernelMask_STMC;
  if (!hasLDMC) kmask &= ~kernelMask_LDMC;

  size_t nBytes = alignUp(nElts * ncclTypeSize(ty), NCCL_SYM_KERNEL_CELL_SIZE);
  size_t nBusBytes = (coll == ncclFuncAllGather || coll == ncclFuncReduceScatter ? comm->nRanks : 1) * nBytes;
  // LL kernels use 32-bit ints to track element counts and indices.
  if (nBusBytes >= (size_t(2) << 30)) kmask &= ~kernelMask_LL;
  // Any kernel might use 32-bit int to track unrolled loop chunks (which are going
  // to be at least 32 bytes per chunk)
  if (nBusBytes >= 32 * (size_t(2) << 30)) kmask = 0;

  if (!ncclSymkTmaAvailable(comm)) kmask &= ~kernelMask_Tma;
  if (!symAligned16B) kmask &= ~kernelMask_Tma;
  // Specialized kernels still require direct NVLink; Genk can also use LSA over CUDA P2P or GIN.
  if (!comm->isAllDirectNvlink) kmask &= kernelMask_Genk;
  if (ncclParamSymGenkEnable() == 0 || comm->minCompCap < 90) kmask &= ~kernelMask_Genk;

  bool const hasGin = ncclParamSymGinKernelsEnable() != 0;
  bool const hasMultipleLsaTeams = ncclTeamLsa(comm).nRanks < comm->nRanks;
  if (hasMultipleLsaTeams) {
    kmask &= kernelMask_GinCapable;
    if (!hasGin) kmask &= ~kernelMask_GinOnly;
  } else {
    kmask &= ~kernelMask_GinOnly;
  }
  if (comm->nNodes > 1 && (!hasGin || !ncclGenkGinAvailable(comm))) kmask &= ~kernelMask_Genk;
  return kmask;
}

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                       size_t nElts) {
  if (!comm->symmetricSupport) return false;
  if (!ncclSymkImplemented(coll, red, ty)) return false;

  return (ncclSymkMask(comm, coll, red, ty, nElts) != 0);
}

const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) {
    return "Unknown";
  }
  return ncclSymKernelStr[kernelId];
}

int ncclSymkMaxChunkElts(struct ncclComm* comm, ncclSymkKernelId kernelId, int /*ncclDevRedOp_t*/ red,
                         ncclDataType_t ty) {
  bool isReduce = 1 & ((kernelMask_Reduce | kernelMask_AR | kernelMask_RS) >> (int)kernelId);
  int eltSize = ncclTypeSize(ty);
  int accMult = !isReduce ? 1 : eltSize < 4 ? 2 : 1;
  bool isGenk = 1 & (kernelMask_Genk >> (int)kernelId);
  int kernelIndex = isGenk ? ncclGenkGetKernelIndex(kernelId, red, ty) : ncclSymkGetKernelIndex(kernelId, red, ty);
  if (kernelIndex < 0) return 0;
  int maxDynamicSmem = isGenk ? ncclGenkKernelMaxDynamicSmem[kernelIndex] : ncclSymkKernelMaxDynamicSmem[kernelIndex];
  return maxDynamicSmem / (eltSize * accMult);
}

/* this function fills in the devWork except nextWorkOffset */
ncclResult_t ncclSymkMakeDevWork(struct ncclComm* comm, struct ncclTaskColl* task, struct ncclSymkDevWork* outDevWork) {
  outDevWork->rootRank = task->root;
  outDevWork->redOpArg = task->opDev.scalarArg;
  outDevWork->nElts = task->count;
  outDevWork->inputWin = task->sendWin ? task->sendWin->vidmem : nullptr;
  outDevWork->inputOff =
    task->sendWin ? (uint8_t*)task->sendbuff - (uint8_t*)task->sendWin->userPtr : (size_t)task->sendbuff;
  outDevWork->outputWin = task->recvWin ? task->recvWin->vidmem : nullptr;
  outDevWork->outputOff =
    task->recvWin ? (uint8_t*)task->recvbuff - (uint8_t*)task->recvWin->userPtr : (size_t)task->recvbuff;
  outDevWork->sChannelId = 0xffff;
  outDevWork->nChannels = 0;
  return ncclSuccess;
}

ncclResult_t ncclGetSymRegType(struct ncclDevrWindow* sendWin, struct ncclDevrWindow* recvWin,
                               ncclSymRegType_t* winRegType) {
  bool isSendSymmReg = false;
  bool isRecvSymmReg = false;
  if (sendWin && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isSendSymmReg = true;
  if (recvWin && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) isRecvSymmReg = true;
  // determine the registration type
  if (!isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvNonreg;
  } else if (isSendSymmReg && !isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvNonreg;
  } else if (!isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendNonregRecvReg;
  } else if (isSendSymmReg && isRecvSymmReg) {
    *winRegType = ncclSymSendRegRecvReg;
  }
  return ncclSuccess;
}
