/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"

#include "comm.h"
#include "core.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

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

static int maxBlocksLsa(struct ncclComm* comm, enum ncclSymkKernelId kernelId) {
  switch (kernelId) {
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
  case ncclSymkKernelId_AllGather_TmaSTMC:
  case ncclSymkKernelId_AllGather_STMC:
  case ncclSymkKernelId_ReduceScatter_LDMC:
    return divUp((comm->cudaArch < 1000 ? 16 : 32), comm->nRanks);
  default:
    return ncclSymkMaxBlocks;
  }
}

static void queryBaseLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              int nMinBlocks, int nMaxBlocks, float* timeUs, int* nBlocks) {
  constexpr double LL_BusFactor = 9; // 2X the bytes, plus some processing, plus no unrolling

  struct ncclComm* comm = input->comm;
  int nRanks = comm->nRanks;
  size_t busBytes; // max(bytes sent, bytes received)
  double busMultiplier = 1;

  switch (kernelId) {
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
    break;
  }

  bool isTma = ncclSymkTmaKernelMask() >> kernelId & 1;
  bool isLL = ncclSymkLLKernelMask() >> kernelId & 1;
  bool isAG = ncclSymkAGKernelMask() >> kernelId & 1;
  bool isAR = ncclSymkARKernelMask() >> kernelId & 1;
  bool isRS = ncclSymkRSKernelMask() >> kernelId & 1;
  constexpr double GBps = (1 << 30) / 1.e6;
  double baseLat, smBw, peakBw;
  if (comm->cudaArch < 1000) {
    baseLat = isLL ? 4.5 : 7.8;
    smBw = isAR ? 65 * GBps : 44 * GBps;
    peakBw = kernelId == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 480 * GBps : 320 * GBps;
  } else {
    baseLat = isLL ? (isAG ? 8.5 : (isRS ? 10.5 : 11.0)) : (isAR ? 19.5 : 13.0);
    smBw = 55 * GBps;
    peakBw = kernelId == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC ? 1000 * GBps : 600 * GBps;
    if (isRS) peakBw = 650 * GBps;
    if (isTma) {
      baseLat += 4.0;
      if (kernelId == ncclSymkKernelId_AllGather_TmaST) {
        peakBw = 700 * GBps;
        baseLat += 19.0; // This is for optimal CTA and kernel selection.
      }
      if (kernelId != ncclSymkKernelId_AllGather_TmaSTMC) peakBw *= 1.07;
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

static bool queryLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, int* nBlocks) {
  struct ncclComm* comm = input->comm;
  int nMaxBlocks = maxBlocksLsa(comm, kernelId);
  bool isTma = ncclSymkTmaKernelMask() >> kernelId & 1;

  *timeUs = FLT_MAX;
  *nBlocks = 0;

  // minCTAs/maxCTAs are resolved (env > per-call > comm) at task-append time.
  nMaxBlocks = std::min<int>(nMaxBlocks, input->maxCTAs);
  int nMinBlocks = input->minCTAs;
  nMinBlocks = std::min(nMinBlocks, nMaxBlocks);
  // NCCL_SYM_CTAS is an explicit override of the resolved bounds.
  int nUserCTAs = ncclSymkModelCtasEnvOverride();
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  // Even CTA counts are preferred for optimal performance, except for when CTAs==1
  if (nMinBlocks != nMaxBlocks) {
    if (nMinBlocks != 1) nMinBlocks = roundUp(nMinBlocks, 2);
    if (nMaxBlocks != 1) nMaxBlocks = roundDown(nMaxBlocks, 2);
  }

  if (isTma) {
    size_t maxWorkBytes = input->countMax * ncclTypeSize(input->datatype);
    if (!ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMinBlocks)) {
      const char* symKernelIdEnv = ncclGetEnv("NCCL_SYM_KERNEL");
      if (symKernelIdEnv) {
        INFO(NCCL_TUNING,
             "NCCL_SYM_KERNEL set to %s. At largest grouped work size %zu Bytes, kernel will not exercise TMA paths.",
             symKernelIdEnv, maxWorkBytes);
      } else {
        return false;
      }
    } else {
      while (nMaxBlocks > nMinBlocks && !ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMaxBlocks)) {
        nMaxBlocks -= (nMaxBlocks == 2 ? 1 : 2);
      }
    }
  }

  queryBaseLsaModel(input, kernelId, nBytes, nMinBlocks, nMaxBlocks, timeUs, nBlocks);
  return *nBlocks > 0 && std::isfinite(*timeUs);
}

bool ncclSymkLsaBaseCtas(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, int* nBlocks) {
  float timeUs;
  return queryLsaBaseModel(input, kernelId, nBytes, &timeUs, nBlocks);
}

void ncclSymkLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                          int* nBlocks) {
  if (queryLsaBaseModel(input, kernelId, nBytes, timeUs, nBlocks)) {
    constexpr float smPenalty = .025f; // 2.5% increase in time per SM.
    *timeUs *= 1.0f + smPenalty * (*nBlocks);
  }
}
