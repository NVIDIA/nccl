/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"
#include "gin_ag.h"

#include "comm.h"
#include "core.h"
#include "transport.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

static_assert(ncclSymkGinAgMaxCtas == ncclSymkMaxBlocks, "GIN model spans must cover every symmetric CTA");

#define NCCL_NVLINK_BW_IDX_HOPPER 0
#define NCCL_NVLINK_BW_IDX_BLACKWELL 1
#define NCCL_NVLINK_BW_IDX_NUM 2
#define NCCL_SM_PENALTY_IDX_NUM 2

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
  int lsaRanks = comm->devrState.lsaTeamSize;
  int railRanks = comm->nRanks / lsaRanks;
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

// Time penalty per SM/CTA for resource use. Keep the standard 2.5% penalty
// except for Blackwell unicast RS+GIN, where the max penalty makes MC win.
static const float smPenalty[NCCL_SM_PENALTY_IDX_NUM][2] = {
  {.025f, .025f}, // Hopper
  {.025f, 1.0f}, // Blackwell
};

static double ncclTuningGetSmPenalty(struct ncclComm* comm, enum ncclSymkKernelId kernelId) {
  int compCapIndex = comm->minCompCap >= 100 ? NCCL_NVLINK_BW_IDX_BLACKWELL : NCCL_NVLINK_BW_IDX_HOPPER;
  int slot = kernelId == ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD ? 1 : 0;
  return smPenalty[compCapIndex][slot];
}

// Initial SM100 fit from the GB200-CX8-NVL4 AlltoAll_LL sweep. Bandwidths
// are physical LL-line GB/s; every 8-byte payload line occupies 16 wire bytes.
static constexpr double alltoAllLLLsaBaseLatUs = 7.47960;
static constexpr double alltoAllLLGinBaseLatUs = 11.37416;
static constexpr double alltoAllLLGinPeerLatUs = 0.12808;
static constexpr double alltoAllLLGinWaveLatUs = 5.02774;
static constexpr double alltoAllLLLsaCtaBwGBps = 9.50640;
static constexpr double alltoAllLLGinCtaBwGBps = 7.09306;
static constexpr double alltoAllLLSelfCtaBwGBps = 213.60945;
// Normalize the fitted 438.33/46.05 GB/s ceilings by this topology's
// 720 GB/s LSA and 96 GB/s GIN rates.
static constexpr double alltoAllLLLsaPeakEfficiency = 0.60880;
static constexpr double alltoAllLLGinPeakEfficiency = 0.47968;

static bool ncclSymkAlltoAllLLModel(struct ncclTuningInput_t* input, size_t payloadBytes, int activeBlocks,
                                    int warpsPerBlock, float* timeUs) {
  struct ncclComm* comm = input->comm;
  if (activeBlocks < 1 || warpsPerBlock < 1 || payloadBytes == 0) return false;

  int lsaRanks = comm->devrState.lsaTeamSize;
  int lsaPeerCount = std::max(0, lsaRanks - 1);
  int ginPeerCount = comm->nRanks - lsaRanks;
  int ginLinesPerWarp = ncclSymkAlltoAllLLGinLinesPerWarp(payloadBytes);
  size_t ginChunkCount = ncclSymkAlltoAllLLChunkCount(ginPeerCount, payloadBytes, ginLinesPerWarp);
  int activeWarps = activeBlocks * warpsPerBlock;
  size_t ginWaves = ginChunkCount == 0 ? 0 : divUp(ginChunkCount, (size_t)activeWarps);
  size_t wireBytesPerPeer = ncclSymkAlltoAllLLWireBytes(payloadBytes);

  // The CTA term models LL publication and consumption. LSA and GIN have
  // separate per-CTA rates because GIN also packs staging and posts puts.
  constexpr double bytesPerUsPerGBps = 1000.0;
  double ctaTimeUs = lsaPeerCount * wireBytesPerPeer / (activeBlocks * alltoAllLLLsaCtaBwGBps * bytesPerUsPerGBps) +
                     ginPeerCount * wireBytesPerPeer / (activeBlocks * alltoAllLLGinCtaBwGBps * bytesPerUsPerGBps) +
                     payloadBytes / (activeBlocks * alltoAllLLSelfCtaBwGBps * bytesPerUsPerGBps);

  // LSA stores and GIN puts run concurrently, so the slower physical fabric
  // ceiling bounds the transfer rather than their sum.
  double lsaPeakBwGBps = ncclTuningGetLsaBw(comm) * alltoAllLLLsaPeakEfficiency / 1.e9;
  double ginPeakBwGBps = ncclTuningGetGinBw(comm) * alltoAllLLGinPeakEfficiency / 1.e9;
  if ((lsaPeerCount != 0 && lsaPeakBwGBps <= 0.0) || (ginPeerCount != 0 && ginPeakBwGBps <= 0.0)) return false;
  double lsaTimeUs = lsaPeerCount == 0 ? 0.0 : lsaPeerCount * wireBytesPerPeer / (lsaPeakBwGBps * bytesPerUsPerGBps);
  double ginTimeUs = ginPeerCount == 0 ? 0.0 : ginPeerCount * wireBytesPerPeer / (ginPeakBwGBps * bytesPerUsPerGBps);

  double latencyUs = alltoAllLLLsaBaseLatUs;
  if (ginPeerCount != 0) {
    latencyUs += alltoAllLLGinBaseLatUs + ginPeerCount * alltoAllLLGinPeerLatUs;
    latencyUs += (ginWaves > 0 ? ginWaves - 1 : 0) * alltoAllLLGinWaveLatUs;
  }
  double estimateUs = latencyUs + std::max(ctaTimeUs, std::max(lsaTimeUs, ginTimeUs));
  if (!std::isfinite(estimateUs) || estimateUs <= 0.0) return false;
  *timeUs = (float)estimateUs;
  return true;
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

static int roundDownGinBlocks(int blocks, bool forced) {
  return forced || blocks <= 1 ? blocks : blocks & ~1;
}

ncclResult_t ncclSymkGinModel(struct ncclTuningInput_t* input, struct ncclTuningResult_t* tuning) {
  ncclSymkKernelId kernelId = (ncclSymkKernelId)tuning->symKernelId;
  size_t nBytes = input->nBytes;
  float* timeUs = &tuning->timeUs;
  int* nBlocks = &tuning->nChannels;
  tuning->chunkSizeBytes = 0;
  struct ncclComm* comm = input->comm;
  struct ncclSymkState* symk = &comm->symkState;
  int lsaRanks = comm->devrState.lsaTeamSize;
  int railRanks = comm->nRanks / lsaRanks;
  double lsaBw = ncclTuningGetLsaBw(comm);
  double ginLat = ncclTuningGetGinLat(comm);
  double ginBw = ncclTuningGetGinBw(comm);
  // minCTAs/maxCTAs are resolved (env > per-call > comm) at task-append time.
  int nMaxBlocks = std::min<int>(input->maxCTAs, ncclSymkMaxBlocks);
  int defaultMaxBlocks = nMaxBlocks;
  enum ncclSymkGinAgArch agArch = ncclSymkGinAgArch_Count;
  int agContexts = 0;
  bool useAgPipeline = false;
  if (kernelId == ncclSymkKernelId_AllGather_RailRing_LsaSTMC) {
    const ncclDevCommRequirements requirements = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    agContexts = symk->initialized ? symk->kcomm.devComm.ginContextCount : requirements.ginContextCount;
    agArch = ncclSymkGinAgArchForCapability(comm->minCompCap);
    useAgPipeline = agArch != ncclSymkGinAgArch_Count && input->nWorks == 1;
    defaultMaxBlocks =
      std::min<int>(nMaxBlocks, divUp((comm->minCompCap >= 100 ? 32 : 16), comm->nvlsResources->nHeads));
    // Grouped work keeps the legacy estimate and its CTA cap.
    if (!useAgPipeline) nMaxBlocks = defaultMaxBlocks;
  }
  int nMinBlocks = input->minCTAs;
  nMinBlocks = std::min(nMinBlocks, nMaxBlocks);
  // NCCL_SYM_CTAS is an explicit override of the resolved bounds.
  int nUserCTAs = ncclSymkModelCtasEnvOverride();
  if (nUserCTAs > 0) {
    if (kernelId == ncclSymkKernelId_AlltoAll_LL) {
      // AlltoAll_LL defines NCCL_SYM_CTAS as a cap on useful CTAs.
      nMaxBlocks = std::min(nMaxBlocks, nUserCTAs);
      nMinBlocks = std::min(nMinBlocks, nMaxBlocks);
    } else {
      nMinBlocks = nMaxBlocks = nUserCTAs;
    }
  }

  // nLimitBlocks is a loose limit for functional correctness: the largest CTA count the kernel's
  // resources allow, reported as maxChannels so grouped launches stay within them. nMaxBlocks may be
  // tightened further for performance (e.g. bandwidth saturation), which only shapes the tuned count.
  int nLimitBlocks = nMaxBlocks;

  bool applySmPenalty = true;
  *timeUs = FLT_MAX;
  tuning->selectionTimeUs = FLT_MAX;
  *nBlocks = 0;
  tuning->maxChannels = 0;
  // Automatic AG choices use one CTA or an even count; explicit overrides stay exact.
  if (kernelId == ncclSymkKernelId_AllGather_RailRing_LsaSTMC && useAgPipeline) {
    nMaxBlocks = roundDownGinBlocks(nMaxBlocks, nUserCTAs > 0);
    if (nUserCTAs <= 0 && nMinBlocks > 1 && (nMinBlocks & 1)) nMinBlocks++;
    if (nMinBlocks > nMaxBlocks) return ncclSuccess;
  }
  switch (kernelId) {
  case ncclSymkKernelId_AlltoAll_LL:
    {
      // AlltoAll_LL is opt-in. Report it unavailable unless forced.
      if (!ncclSymkAlltoAllLLForced(comm)) break;
      // nBytes is the payload sent to each peer, as the kernel's chunk and
      // scratch formulas consume it.
      size_t payloadBytes = nBytes;
      int lsaPeerCount = std::max(0, lsaRanks - 1);
      int ginPeerCount = comm->nRanks - lsaRanks;
      int warpsPerBlock = ncclSymkKernelMaxWarps(kernelId, input->devRedOp, input->datatype);
      int ginLinesPerWarp = ncclSymkAlltoAllLLGinLinesPerWarp(payloadBytes);
      *nBlocks =
        ncclSymkAlltoAllLLSelectedBlockCount(lsaPeerCount, ginPeerCount, payloadBytes,
                                             ncclSymkAlltoAllLLLsaLinesPerWarp, ginLinesPerWarp, warpsPerBlock,
                                             symk->alltoAllLLMaxResidentCtas, nMaxBlocks, /*userBlockLimit=*/0);
      if (!ncclSymkAlltoAllLLModel(input, payloadBytes, *nBlocks, warpsPerBlock, timeUs)) *nBlocks = 0;
      // Report predicted latency directly. The generic GIN SM penalty is a
      // selection heuristic and was not part of this forced-CTA fit.
      applySmPenalty = false;
    }
    break;
  case ncclSymkKernelId_AllGather_RailRing_LsaSTMC:
    {
      bool forcedChunk = false;
      size_t railChunkSize = ncclSymkGinChunkBytes(ncclFuncAllGather, &forcedChunk);
      size_t requiredBlocks = DIVUP(nBytes, railChunkSize);
      int defaultBlocks = std::max(
        nMinBlocks, roundDownGinBlocks((int)std::min((size_t)std::min(nMaxBlocks, defaultMaxBlocks), requiredBlocks),
                                       nUserCTAs > 0 || !useAgPipeline));
      // The model consumes the input bytes per rank, not the total gathered extent.
      // The selector owns numeric input validation; retain the legacy estimate
      // outside a supported single-work schedule or when no pair can be modeled.
      if (useAgPipeline) {
        size_t rankBytes = input->count * ncclTypeSize(input->datatype);
        const struct ncclSymkGinAgChoice best =
          ncclSymkGinAgSelect(agArch, rankBytes, lsaRanks, railRanks, agContexts, nMinBlocks, nMaxBlocks,
                              forcedChunk ? railChunkSize : 0, defaultBlocks, railChunkSize);
        if (best.ctas != 0) {
          *nBlocks = best.ctas;
          tuning->maxChannels = std::max(nLimitBlocks, best.ctas);
          *timeUs = (float)best.timeUs;
          tuning->selectionTimeUs = (float)best.selectionUs;
          tuning->chunkSizeBytes = (int)best.chunkBytes;
          return ncclSuccess;
        }
      }
      tuning->chunkSizeBytes = (int)railChunkSize;
      float intraBw = lsaBw;
      float interBw = ginBw;
      float intraTime = (float)(nBytes * comm->nRanks) / intraBw;
      float interTime = (float)(nBytes * (railRanks - 1)) / interBw;
      size_t steps = DIVUP(nBytes, railChunkSize) * (railRanks - 1);
      *timeUs = steps * ginLat + std::max(intraTime, interTime);
      *nBlocks = defaultBlocks;
    }
    break;
  case ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD:
  case ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC:
    {
      bool ldmc = kernelId == ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC;
      // Per-block GIN inbox, outbox and accumulation slots exist for maxGinInboxBlocks CTAs only.
      nMaxBlocks = std::min(nMaxBlocks, symk->maxGinInboxBlocks);
      nLimitBlocks = nMaxBlocks;
      nMaxBlocks = std::min(nMaxBlocks, ncclTuningCalcSatBlocksReduceScatterRailA2A(comm, ldmc));
      size_t chunkSize = ncclSymkGinChunkBytes(ncclFuncReduceScatter, nullptr);
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

  if (applySmPenalty && *nBlocks > 0 && std::isfinite(*timeUs)) {
    *timeUs *= 1.0f + ncclTuningGetSmPenalty(comm, kernelId) * (*nBlocks);
  }
  tuning->selectionTimeUs = *timeUs;
  if (*nBlocks > 0) tuning->maxChannels = std::max(nLimitBlocks, *nBlocks);
  return ncclSuccess;
}
