/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"

#include "comm.h"
#include "core.h"
#include "transport.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#define NCCL_NVLINK_BW_IDX_HOPPER 0
#define NCCL_NVLINK_BW_IDX_BLACKWELL 1
#define NCCL_NVLINK_BW_IDX_NUM 2

// NVLS max bws NCCL can achieve.
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

// Calculate saturation block count.
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

void ncclSymkGinModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                      int* nBlocks) {
  struct ncclComm* comm = input->comm;
  struct ncclSymkState* symk = &comm->symkState;
  // ncclTeam world = ncclTeamWorld(comm);
  // ncclTeam lsa = ncclTeamLsa(comm);
  ncclTeam rail = ncclTeamRail(comm);
  double lsaBw = ncclTuningGetLsaBw(comm);
  double ginLat = ncclTuningGetGinLat(comm);
  double ginBw = ncclTuningGetGinBw(comm);
  // minCTAs/maxCTAs are resolved (env > per-call > comm) at task-append time.
  int nMaxBlocks = std::min<int>(input->maxCTAs, ncclSymkMaxBlocks);
  if (kernelId == ncclSymkKernelId_AllGather_RailRing_LsaSTMC) {
    nMaxBlocks = std::min<int>(nMaxBlocks, divUp((comm->cudaArch < 1000 ? 16 : 32), comm->nvlsResources->nHeads));
  }
  int nMinBlocks = input->minCTAs;
  nMinBlocks = std::min(nMinBlocks, nMaxBlocks);
  // NCCL_SYM_CTAS is an explicit override of the resolved bounds.
  int nUserCTAs = ncclSymkModelCtasEnvOverride();
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  *timeUs = FLT_MAX;
  *nBlocks = 0;
  switch (kernelId) {
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
      bool ldmc = kernelId == ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;
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

  if (*nBlocks > 0 && std::isfinite(*timeUs)) {
    constexpr float smPenalty = .025f; // 2.5% increase in time per SM.
    *timeUs *= 1.0f + smPenalty * (*nBlocks);
  }
}
