/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "sym_kernels.h"
#include "tuning_int.h"
#include "core.h"
#include "comm.h"
#include "transport.h"
#include <cfloat>

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)

#define NCCL_NVLINK_BW_IDX_HOPPER 0
#define NCCL_NVLINK_BW_IDX_BLACKWELL 1
#define NCCL_NVLINK_BW_IDX_NUM 2
// NVLS max bws NCCL can achieve
static const float nvlinkBws[NCCL_NVLINK_BW_IDX_NUM] = {
  360.0f, // Hopper
  720.0f, // Blackwell
};

double ncclTuningGetLsaBw(struct ncclComm* comm) {
  int compCapIndex = comm->minCompCap >= 100 ? NCCL_NVLINK_BW_IDX_BLACKWELL : NCCL_NVLINK_BW_IDX_HOPPER;
  return (/*byte/sec*/ 1.e9) * nvlinkBws[compCapIndex];
}

double ncclTuningGetGinLat(struct ncclComm* comm) {
  return (/*sec/usec*/ 1.e-6) *
         comm->tuningContext.tuningConstants.hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE];
}

double ncclTuningGetGinBw(struct ncclComm* comm) {
  return (/*byte/sec*/ 1.e9) * comm->minNetBw;
}

static double softmin(double x, double ceiling, double softness) {
  // looks like a smooth version of: min(x, ceiling)
  return ceiling - softness * std::log1p((std::exp(ceiling / softness) - 1) * std::exp(-x / softness));
}

static double softplus(double x, double softness) {
  // looks like a smooth version of: max(0, x)
  double z = x / softness;
  return 100.0 <= z ? x : softness * std::log1p(std::exp(z));
}

static double model(double busBytes, double baseLat, int nSMs, double smBw, double busMultiplier, double peakBw) {
  double bw = softmin(nSMs * smBw * busMultiplier, peakBw, smBw);
  return baseLat + softplus(busBytes / bw - 1, 1);
}

// Bus multipliers count number of times data is sent through that widget.
void ncclTuningGetBusMulReduceScatterRailA2A(struct ncclComm* comm, bool ldmc,
    // Bus multipliers per bottleneck
                                             double* out_smMul, double* out_lsaMul, double* out_ginMul) {
  int lsaRanks = ncclTeamLsa(comm).nRanks;
  int railRanks = ncclTeamRail(comm).nRanks;
  // LSA
  *out_lsaMul = std::max(
    /*inbound*/ (ldmc ? lsaRanks : lsaRanks - 1) * railRanks,
    /*outbound*/ (lsaRanks - 1) * railRanks);
  // GIN
  *out_ginMul = railRanks - 1; // inbound == outbound
  // SM. Inbound (reads) only because it dominates outbound (writes).
  *out_smMul =
    /*stage 0*/ (lsaRanks == 1 ? 0 : (ldmc ? 1 : lsaRanks) * (railRanks - 1)) +
    /*stage 1*/ (ldmc ? 1 : lsaRanks) + (railRanks - 1);
}

static double getSmBw_ReduceScatter_RailA2A(struct ncclComm* comm, bool ldmc) {
  // Empirically calculated as effbw/nctas where effbw is reported by TUNING
  // debug logging (from getRequirements_gin()) and nctas is the number of ctas
  // that appear to saturate bandwidth.
  if (100 <= comm->minCompCap) {
    return ldmc ? 8.44e9 : 26.6e9;
  } else {
    return ldmc ? 4.22e9 : 13.7e9;
  }
}

double ncclTuningGetSmLatReduceScatterRailA2A(struct ncclComm* comm, bool ldmc) {
  // Processing delay. Larger value means bigger network buffers.
  return 10.e-6;
}

// Calculate saturation block count:
int ncclTuningCalcSatBlocksReduceScatterRailA2A(struct ncclComm* comm, bool ldmc) {
  double lsaBw = ncclTuningGetLsaBw(comm);
  double ginBw = ncclTuningGetGinBw(comm);
  double smBw = getSmBw_ReduceScatter_RailA2A(comm, ldmc);
  double smMul, lsaMul, ginMul;
  ncclTuningGetBusMulReduceScatterRailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
  // Effective Bandwidth: EffBw = Bw/Mul
  // Let smsEffBw = smEffBw*nBlocks
  // Set smsEffBw = min(lsaEffBw, ginEffBw)
  // Solve for nBlocks:
  double minLsaGinEffBw = std::min(lsaBw / lsaMul, ginBw / ginMul);
  return std::ceil(std::min(double(1 << 30), minLsaGinEffBw / (smBw / smMul)));
}

// Given the kernel and bytes, return the minimum number of blocks to run on such that
// perf is 99% of running at max blocks, and return the estimate runtime for that
// block count.
static void queryModel_gin(struct ncclTuningInput_t* input, ncclSymkKernelId k, size_t nBytes, float* timeUs,
                           int* nBlocks);
static void queryModel_lsa(struct ncclTuningInput_t* input, ncclSymkKernelId k, size_t nBytes, float* timeUs,
                           int* nBlocks);

static void queryModel(struct ncclTuningInput_t* input, ncclSymkKernelId k, size_t nBytes, float* timeUs,
                       int* nBlocks) {
  if (ncclSymkGinKernelMask() >> k & 1) {
    queryModel_gin(input, k, nBytes, timeUs, nBlocks);
  } else {
    queryModel_lsa(input, k, nBytes, timeUs, nBlocks);
  }
}

static void queryModel_gin(struct ncclTuningInput_t* input, ncclSymkKernelId k, size_t nBytes, float* timeUs,
                           int* nBlocks) {
  struct ncclComm* comm = input->comm;
  struct ncclSymkState* symk = &comm->symkState;
  // ncclTeam world = ncclTeamWorld(comm);
  // ncclTeam lsa = ncclTeamLsa(comm);
  ncclTeam rail = ncclTeamRail(comm);
  double lsaBw = ncclTuningGetLsaBw(comm);
  double ginLat = ncclTuningGetGinLat(comm);
  double ginBw = ncclTuningGetGinBw(comm);
  // maxCTAs/minCTAs are resolved (env > per-call > comm) at task-append time.
  int nMaxBlocks = std::min<int>(input->maxCTAs, ncclSymkMaxBlocks);
  if (k == ncclSymkKernelId_AllGather_RailRing_LsaSTMC) {
    nMaxBlocks = std::min<int>(nMaxBlocks, divUp((comm->cudaArch < 1000 ? 16 : 32), comm->nvlsResources->nHeads));
  }
  int nMinBlocks = input->minCTAs;
  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;
  if (nMinBlocks > nMaxBlocks) nMinBlocks = nMaxBlocks;

  *timeUs = FLT_MAX;
  *nBlocks = 0;
  switch (k) {
  case ncclSymkKernelId_AllGather_RailRing_LsaSTMC:
    {
      constexpr int railChunkSize = ncclSymkAllGather_RailRing_ChunkSize;
      int requiredBlocks = DIVUP(nBytes, railChunkSize);
      float intraBw = lsaBw;
      float interBw = ginBw;
      float intraTime = (float)(nBytes * comm->nRanks) / intraBw;
      float interTime = (float)(nBytes * (rail.nRanks - 1)) / interBw;
      uint32_t steps = DIVUP(nBytes, railChunkSize) * (rail.nRanks - 1);
      *timeUs = steps * ginLat + std::max(intraTime, interTime);
      *nBlocks = std::max(nMinBlocks, std::min(nMaxBlocks, requiredBlocks));
    }
    break;
  case ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD:
  case ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC:
    {
      bool ldmc = k == ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;
      nMaxBlocks = std::min(nMaxBlocks, symk->maxGinInboxBlocks);
      nMaxBlocks = std::min(nMaxBlocks, ncclTuningCalcSatBlocksReduceScatterRailA2A(comm, ldmc));
      size_t chunkSize = ncclSymkRsGinChunkBytes();
      double smBw = getSmBw_ReduceScatter_RailA2A(comm, ldmc);
      double smMul, lsaMul, ginMul;
      ncclTuningGetBusMulReduceScatterRailA2A(comm, ldmc, &smMul, &lsaMul, &ginMul);
      *nBlocks = (int)divUp(nBytes, chunkSize);
      // max against nMinBlocks last since we may have nMaxBlocks < nMinBlocks
      *nBlocks = std::max(nMinBlocks, std::min(nMaxBlocks, *nBlocks));
      double effBw = (*nBlocks) * (smBw / smMul);
      effBw = std::min(effBw, lsaBw / lsaMul);
      effBw = std::min(effBw, ginBw / ginMul);
      double time = nBytes / effBw;
      // Delayed by LSA processing of first chunk.
      time += std::min(nBytes, chunkSize * (size_t)(*nBlocks)) * (lsaMul / lsaBw + ginMul / ginBw);
      // Delay by GIN latency of first chunk.
      time += ginLat;
      *timeUs = (/*usec/sec=*/1.e6) * time;
    }
    break;
  default:
    break;
  }
}

static void queryModel_lsa(struct ncclTuningInput_t* input, ncclSymkKernelId k, size_t nBytes, float* timeUs,
                           int* nBlocks) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling

  struct ncclComm* comm = input->comm;
  int nRanks = comm->nRanks;
  int nMaxBlocks = ncclSymkMaxBlocks;
  int nMaxBlocksNvls = divUp((comm->cudaArch < 1000 ? 16 : 32), nRanks);
  size_t busBytes; // max(bytes sent, bytes received)
  double busMultiplier = 1;

  switch (k) {
  default:
    busBytes = size_t(1) << 50;
    break;

  case ncclSymkKernelId_AllReduce_AGxLL_R:
    busBytes = nRanks * nBytes * LL_BusFactor;
    break;
  case ncclSymkKernelId_AllReduce_AGxLLMC_R:
    busBytes = nRanks * nBytes * LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST:
  case ncclSymkKernelId_AllReduce_RSxLD_AGxST:
    busBytes = 2 * nBytes * (nRanks - 1) / nRanks;
    break;
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
    busBytes = nBytes / nRanks + nBytes;
    busMultiplier = nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;

  case ncclSymkKernelId_AllGather_LL:
    busBytes = nRanks * nBytes * LL_BusFactor;
    break;
  case ncclSymkKernelId_AllGather_LLMC:
    busBytes = nRanks * nBytes * LL_BusFactor;
    busMultiplier = 1.1; // To beat non-MC LL
    break;
  case ncclSymkKernelId_AllGather_TmaST:
  case ncclSymkKernelId_AllGather_ST:
    busBytes = (nRanks - 1) * nBytes;
    break;
  case ncclSymkKernelId_AllGather_TmaSTMC:
    busMultiplier = 0.99;
    // fall through
  case ncclSymkKernelId_AllGather_STMC:
    busBytes = (nRanks - 1) * nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier *= 0.55 * nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;

  case ncclSymkKernelId_ReduceScatter_LL:
    busBytes = nRanks * nBytes * LL_BusFactor;
    break;
  case ncclSymkKernelId_ReduceScatter_TmaLD:
  case ncclSymkKernelId_ReduceScatter_LD:
    busBytes = (nRanks - 1) * nBytes;
    break;
  case ncclSymkKernelId_ReduceScatter_LDMC:
    busBytes = (nRanks - 1) * nBytes; // Wrong. Should be nRanks*nBytes but we want to beat non-MC.
    busMultiplier = 0.55 * nRanks;
    nMaxBlocks = nMaxBlocksNvls;
    break;
  }

  bool isTma = ncclSymkTmaKernelMask() >> k & 1;
  bool isLL = ncclSymkLLKernelMask() >> k & 1;
  bool isAG = ncclSymkAGKernelMask() >> k & 1;
  bool isAR = ncclSymkARKernelMask() >> k & 1;
  bool isRS = ncclSymkRSKernelMask() >> k & 1;
  constexpr double GBps = (1 << 30) / 1.e6;
  double baseLat, smBw, peakBw;
  if (comm->cudaArch < 1000) {
    baseLat = isLL ? 4.5 : 7.8;
    smBw = isAR ? 65 * GBps : 44 * GBps;
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480 * GBps : 320 * GBps;
  } else {
    baseLat = isLL ? (isAG ? 8.5 : (isRS ? 10.5 : 11.0)) : (isAR ? 19.5 : 13.0);
    smBw = 55 * GBps;
    peakBw = k == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000 * GBps : 600 * GBps;
    if (isRS) peakBw = 650 * GBps;
    if (isTma) {
      baseLat += 4.0;
      if (k == ncclSymkKernelId_AllGather_TmaST) {
        peakBw = 700 * GBps;
        baseLat += 19.0; // This is for optimal CTA and kernel selection.
      }
      if (k != ncclSymkKernelId_AllGather_TmaSTMC) peakBw *= 1.07;
    }
  }

  // maxCTAs/minCTAs are resolved (env > per-call > comm) at task-append time.
  nMaxBlocks = std::min<int>(nMaxBlocks, input->maxCTAs);
  int nMinBlocks = input->minCTAs;
  int nUserCTAs = std::min<int>(ncclSymkMaxBlocks, ncclParamSymCTAs());
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;
  if (nMinBlocks > nMaxBlocks) nMinBlocks = nMaxBlocks;

  // Even CTA counts are preferred for optimal performance, except for when CTAs==1
  if (nMinBlocks != nMaxBlocks) {
    if (nMinBlocks != 1) nMinBlocks = roundUp(nMinBlocks, 2);
    if (nMaxBlocks != 1) nMaxBlocks = roundDown(nMaxBlocks, 2);
  }

  if (isTma) {
    size_t maxWorkBytes = input->countMax * ncclTypeSize(input->datatype);
    // Only use TMA if the selected CTA count is able to take advantage of the TMA fast paths.
    // Except for the case when user sets NCCL_SYM_KERNEL.
    // If TMA is not eligible for the minimum block at the largest grouped work size
    // it will not be eligible for higher block count.
    if (!ncclSymkTmaDeepEligible(comm, k, maxWorkBytes, nMinBlocks)) {
      const char* symKernelIdEnv = ncclGetEnv("NCCL_SYM_KERNEL");
      if (symKernelIdEnv) {
        INFO(NCCL_TUNING,
             "NCCL_SYM_KERNEL set to %s. At largest grouped work size %zu Bytes, kernel will not exercise TMA paths.",
             symKernelIdEnv, maxWorkBytes);
      } else {
        *nBlocks = -1;
        return;
      }
    } else {
      // Decrease max block count until it is eligible for the current msg size.
      while (nMaxBlocks > nMinBlocks && !ncclSymkTmaDeepEligible(comm, k, maxWorkBytes, nMaxBlocks)) {
        nMaxBlocks -= (nMaxBlocks == 2 ? 1 : 2);
      }
    }
  }

  *nBlocks = nMaxBlocks;
  *timeUs = model(busBytes, baseLat, nMaxBlocks, smBw, busMultiplier, peakBw);
  for (int bn = nMinBlocks; bn < nMaxBlocks; bn += (bn == 1) ? 1 : 2) {
    double time = model(busBytes, baseLat, bn, smBw, busMultiplier, peakBw);
    if (time <= 1.025 * (*timeUs)) {
      *nBlocks = bn;
      *timeUs = time;
      break;
    }
  }
}

ncclResult_t ncclTuningSymkModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;

  if (tuning->symKernelId == ncclSymkKernelId_Count) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if (!ncclSymkAvailable(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype, inputs->count)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  uint32_t tuning_kmask = (1 << tuning->symKernelId);
  uint32_t valid_kmask = ncclSymkMask(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype, inputs->countMax,
                                      inputs->symAligned16B);
  if ((tuning_kmask & valid_kmask) == 0) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if ((inputs->nWorks > 1 &&
       ((tuning_kmask & ncclSymkLLKernelMask()) != 0)) // We currently don't support grouping for LL kernels.
      || (inputs->func == ncclFuncAllReduce && inputs->winRegType != ncclSymSendRegRecvReg &&
          (tuning_kmask & ncclSymkLLKernelMask()) == 0) ||
      (inputs->func == ncclFuncAllGather && inputs->winRegType != ncclSymSendRegRecvReg &&
       inputs->winRegType != ncclSymSendNonregRecvReg && (tuning_kmask & ncclSymkLLKernelMask()) == 0) ||
      (inputs->func == ncclFuncReduceScatter && inputs->winRegType != ncclSymSendRegRecvReg &&
       inputs->winRegType != ncclSymSendRegRecvNonreg && (tuning_kmask & ncclSymkLLKernelMask()) == 0) ||
      (inputs->func == ncclFuncAllGather && inputs->winRegType != ncclSymSendRegRecvReg && inputs->comm->nNodes > 1 &&
       (tuning_kmask & ncclSymkGinKernelMask()) != 0)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  float kTime = 0.0f;
  int kBlocks = 0;
  constexpr float smPenalty = .025f; // 2.5% percent increase in time per SM
  queryModel(inputs, (ncclSymkKernelId)tuning->symKernelId, inputs->nBytes, &kTime, &kBlocks);
  if (kBlocks <= 0) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  tuning->timeUs = kTime * (1.0f + smPenalty * kBlocks);
  tuning->nChannels = kBlocks;
  tuning->nWarps = 16;
  return ret;
}
