/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "comm.h"
#include "device.h"
#include "nccl_device/core.h"
#include "transport.h"
#include "tuning.h"
#include <cmath>
#include <cfloat>

constexpr uint32_t kernelMask_STMC =
  1 << ncclSymkKernelId_AllGather_LLMC | 1 << ncclSymkKernelId_AllGather_STMC |
  1 << ncclSymkKernelId_AllGather_TmaSTMC | 1 << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1 << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC | 1 << ncclSymkKernelId_ReduceScatter_LDMC |
  1 << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr uint32_t kernelMask_LDMC = 1 << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                     1 << ncclSymkKernelId_ReduceScatter_LDMC |
                                     1 << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;

constexpr uint32_t kernelMask_LL = 1 << ncclSymkKernelId_AllReduce_AGxLL_R | 1 << ncclSymkKernelId_AllReduce_AGxLLMC_R |
                                   1 << ncclSymkKernelId_AllGather_LL | 1 << ncclSymkKernelId_AllGather_LLMC |
                                   1 << ncclSymkKernelId_ReduceScatter_LL;

constexpr uint32_t kernelMask_AG = 1 << ncclSymkKernelId_AllGather_LL | 1 << ncclSymkKernelId_AllGather_LLMC |
                                   1 << ncclSymkKernelId_AllGather_ST | 1 << ncclSymkKernelId_AllGather_STMC |
                                   1 << ncclSymkKernelId_AllGather_TmaST | 1 << ncclSymkKernelId_AllGather_TmaSTMC |
                                   1 << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr uint32_t kernelMask_AR = 1 << ncclSymkKernelId_AllReduce_AGxLLMC_R | 1 << ncclSymkKernelId_AllReduce_AGxLL_R |
                                   1 << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
                                   1 << ncclSymkKernelId_AllReduce_RSxLD_AGxST |
                                   1 << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST;

constexpr uint32_t kernelMask_RS = 1 << ncclSymkKernelId_ReduceScatter_LD | 1 << ncclSymkKernelId_ReduceScatter_LDMC |
                                   1 << ncclSymkKernelId_ReduceScatter_TmaLD | 1 << ncclSymkKernelId_ReduceScatter_LL |
                                   1 << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                   1 << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;

constexpr uint32_t kernelMask_LSA =
  1 << ncclSymkKernelId_AllReduce_AGxLL_R | 1 << ncclSymkKernelId_AllReduce_AGxLLMC_R |
  1 << ncclSymkKernelId_AllReduce_RSxLD_AGxST | 1 << ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC |
  1 << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST | 1 << ncclSymkKernelId_AllGather_LL |
  1 << ncclSymkKernelId_AllGather_LLMC | 1 << ncclSymkKernelId_AllGather_ST | 1 << ncclSymkKernelId_AllGather_STMC |
  1 << ncclSymkKernelId_AllGather_TmaST | 1 << ncclSymkKernelId_AllGather_TmaSTMC |
  1 << ncclSymkKernelId_ReduceScatter_LL | 1 << ncclSymkKernelId_ReduceScatter_LD |
  1 << ncclSymkKernelId_ReduceScatter_LDMC | 1 << ncclSymkKernelId_ReduceScatter_TmaLD;

constexpr uint32_t kernelMask_Gin = 1 << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD |
                                    1 << ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC |
                                    1 << ncclSymkKernelId_AllGather_RailRing_LsaSTMC;

constexpr uint32_t kernelMask_Tma = 1 << ncclSymkKernelId_AllGather_TmaST | 1 << ncclSymkKernelId_AllGather_TmaSTMC |
                                    1 << ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST |
                                    1 << ncclSymkKernelId_ReduceScatter_TmaLD;

constexpr uint32_t kernelMask_DynamicSmem = kernelMask_Tma;

int ncclSymkLLKernelMask() {
  return kernelMask_LL;
}
int ncclSymkDynamicSmemKernelMask() {
  return kernelMask_DynamicSmem;
};

int ncclSymkGinKernelMask() {
  return kernelMask_Gin;
}

int ncclSymkAGKernelMask() {
  return kernelMask_AG;
}

int ncclSymkARKernelMask() {
  return kernelMask_AR;
}

static uint32_t kernelMask_coll(ncclFunc_t coll) {
  switch (coll) {
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
NCCL_PARAM(SymTmaEnable, "SYM_TMA_ENABLE", 0)

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
    symk->hasLsaMultimem = comm->nvlsSupport && ncclTeamLsa(comm).nRanks > 2 && !comm->p2pCrossClique;
    reqs.lsaMultimem = symk->hasLsaMultimem;
    reqs.lsaBarrierCount = ncclSymkMaxBlocks;

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
    }

    NCCLCHECK(ncclDevrCommCreateInternal(comm, &reqs, &symk->kcomm.devComm, true));
  }
  return ncclSuccess;
}

ncclResult_t ncclSymkFinalize(struct ncclComm* comm) {
  struct ncclSymkState* symk = &comm->symkState;
  if (symk->initialized) {
    NCCLCHECK(ncclDevCommDestroy(comm, &symk->kcomm.devComm));
  }
  return ncclSuccess;
}

static bool ncclSymkImplemented(ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty) {
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

  switch (coll) {
  case ncclFuncAllGather:
    return true;
  case ncclFuncAllReduce:
  case ncclFuncReduceScatter:
    if (red == ncclDevSum || red == ncclDevSumPostDiv) {
      return isFloat && ty != ncclFloat64;
    }
  default:
    return false;
  }
}

uint32_t ncclSymkMask(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                      size_t nElts) {
  uint32_t kmask = kernelMask_coll(coll);

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

  size_t nBytes = nElts * ncclTypeSize(ty);
  size_t nBusBytes = (coll == ncclFuncAllReduce ? 1 : comm->nRanks) * nBytes;
  // LL kernels use 32-bit ints to track element counts and indices.
  if (nBusBytes >= (size_t(2) << 30)) kmask &= ~kernelMask_LL;
  // Any kernel might use 32-bit int to track unrolled loop chunks (which are going
  // to be at least 32 bytes per chunk)
  if (nBusBytes >= 32 * (size_t(2) << 30)) kmask = 0;

  bool hasTma = comm->minCompCap >= 100 && ncclParamSymTmaEnable();
  if (!hasTma) kmask &= ~kernelMask_Tma;

  bool hasGin = ncclParamSymGinKernelsEnable() != 0;
  if (!hasGin) kmask &= ~kernelMask_Gin;
  bool needGin = ncclTeamLsa(comm).nRanks < comm->nRanks;
  kmask &= needGin ? kernelMask_Gin : ~kernelMask_Gin;
  return kmask;
}

bool ncclSymkAvailable(struct ncclComm* comm, ncclFunc_t coll, int /*ncclDevRedOp_t*/ red, ncclDataType_t ty,
                       size_t nElts) {
  if (!comm->isAllDirectNvlink) return false;
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
  bool isReduce = 1 & ((kernelMask_AR | kernelMask_RS) >> (int)kernelId);
  int eltSize = ncclTypeSize(ty);
  int accMult = !isReduce ? 1 : eltSize < 4 ? 2 : 1;
  int kernelIndex = ncclSymkGetKernelIndex(kernelId, red, ty);
  return kernelIndex < 0 ? 0 : ncclSymkKernelMaxDynamicSmem[kernelIndex] / (eltSize * accMult);
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
