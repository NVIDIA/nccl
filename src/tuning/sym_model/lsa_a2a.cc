/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "lsa_a2a.h"

#include "comm.h"
#include "core.h"

#include <algorithm>
#include <cmath>

static constexpr double bwToBytesPerUs = 1.e9 / 1.e6;

// Multimem AllGather CTA bandwidth is the measured store injection rate before
// multicast fanout; peak bandwidth is aggregate traffic.
const struct ncclSymkLsaA2AParameters
  ncclSymkLsaA2AParameterTable[NCCL_SYMK_LSA_A2A_ARCH_COUNT][ncclSymkKernelId_Count] = {
    // SM100
    {
      {}, // AllReduce_AGxLL_R
      {}, // AllReduce_AGxLLMC_R
      {}, // AllReduce_RSxTmaLD_AGxTmaST
      {}, // AllReduce_RSxLD_AGxST
      {}, // AllReduce_RSxLDMC_AGxSTMC
      {true, 7.1084, 0.1065, 2.32, 15.87, 250.0, {{0.99, 0.69}}, 0.0, false}, // AllGather_LL
      {true, 7.4229, 0.1030, 2.1932, 27.46, 316.0699, {{1.0, 1.0}}, 3.3805, true}, // AllGather_LLMC
      {true, 10.0618, 0.0679, 0.0, 64.1577, 671.6052, {{1.0, 1.0}}, 0.0, false}, // AllGather_TmaST
      {true, 10.5902, 0.0563, 0.0, 64.4810, 650.4378, {{1.0, 1.0}}, 0.0, false}, // AllGather_ST
      {true, 8.2723, 0.0623, 0.0, 51.55, 715.1451, {{1.0, 1.0}}, 0.0, true}, // AllGather_TmaSTMC
      {true, 8.3313, 0.0561, 0.0, 50.83, 715.1451, {{1.0, 1.0}}, 0.0, true}, // AllGather_STMC
      {}, // AllGather_RailRing_LsaSTMC
      {}, // ReduceScatter_LL
      {}, // ReduceScatter_TmaLD
      {}, // ReduceScatter_LD
      {}, // ReduceScatter_LDMC
      {}, // ReduceScatter_RailA2A_LsaLD
      {}, // ReduceScatter_RailA2A_LsaLDMC
    },
    // SM103
    {
      {}, // AllReduce_AGxLL_R
      {}, // AllReduce_AGxLLMC_R
      {}, // AllReduce_RSxTmaLD_AGxTmaST
      {}, // AllReduce_RSxLD_AGxST
      {}, // AllReduce_RSxLDMC_AGxSTMC
      {true, 7.1084, 0.1065, 2.32, 15.87, 250.0, {{0.99, 0.69}}, 0.0, false}, // AllGather_LL
      {true, 7.4229, 0.1030, 2.1932, 27.46, 316.0699, {{1.0, 1.0}}, 3.3805, true}, // AllGather_LLMC
      {
        true, 10.0618, 0.0679, 0.0, 64.1577, 671.6052, {{1.0, 1.0}}, 0.0, false, 12.1475, 639.6826, 585.7896
      }, // AllGather_TmaST
      {true, 10.5902, 0.0563, 0.0, 64.4810, 650.4378, {{1.0, 1.0}}, 0.0, false}, // AllGather_ST
      {true, 8.2723, 0.0623, 0.0, 51.55, 715.1451, {{1.0, 1.0}}, 0.0, true}, // AllGather_TmaSTMC
      {true, 8.3313, 0.0561, 0.0, 50.83, 715.1451, {{1.0, 1.0}}, 0.0, true}, // AllGather_STMC
      {}, // AllGather_RailRing_LsaSTMC
      {}, // ReduceScatter_LL
      {}, // ReduceScatter_TmaLD
      {}, // ReduceScatter_LD
      {}, // ReduceScatter_LDMC
      {}, // ReduceScatter_RailA2A_LsaLD
      {}, // ReduceScatter_RailA2A_LsaLDMC
    },
};

static inline int ncclSymkLsaA2AArchBucket(struct ncclComm* comm) {
  if (comm->minCompCap == 100) return NCCL_SYMK_LSA_A2A_ARCH_SM100;
  if (comm->minCompCap == 103) return NCCL_SYMK_LSA_A2A_ARCH_SM103;
  return -1;
}

// AllGather ST and STMC use forEachWork<char>. For the current single-work
// model, the scheduler assigns ceil(cells / requestedCtas) cells per CTA and
// leaves the tail of the requested grid idle.
static int activeCtasForEachWork(size_t logicalBytes, int requestedCtas) {
  size_t cells = divUp(logicalBytes, size_t(NCCL_SYM_KERNEL_CELL_SIZE));
  size_t cellsPerCta = divUp(cells, size_t(requestedCtas));
  return (int)divUp(cells, cellsPerCta);
}

// Apply measured effective-CTA scaling for AllGather kernels.
static double allGatherCtaScale(const struct ncclSymkLsaA2AParameters* parameters, enum ncclSymkKernelId kernelId,
                                int nRanks, int activeCtas) {
  if (kernelId == ncclSymkKernelId_AllGather_TmaSTMC && activeCtas == 1) return 1.1445;
  if (kernelId != ncclSymkKernelId_AllGather_LL || nRanks <= 4 || activeCtas <= 4 || activeCtas >= 64) return 1.0;
  static constexpr double log2CtaPositions[] = {2.0, 3.0, 5.0, 6.0};
  static constexpr int positionCount = sizeof(log2CtaPositions) / sizeof(log2CtaPositions[0]);
  double log2ActiveCtas = std::log2(static_cast<double>(activeCtas));
  for (int upper = 1; upper < positionCount; upper++) {
    if (log2ActiveCtas <= log2CtaPositions[upper]) {
      double lowerScale = upper == 1 ? 1.0 : parameters->ctaScalingCurve.ctaScale[upper - 2];
      double upperScale = upper == positionCount - 1 ? 1.0 : parameters->ctaScalingCurve.ctaScale[upper - 1];
      double fraction =
        (log2ActiveCtas - log2CtaPositions[upper - 1]) / (log2CtaPositions[upper] - log2CtaPositions[upper - 1]);
      return lowerScale + fraction * (upperScale - lowerScale);
    }
  }
  return 1.0;
}

static bool modelBlackwellTmaSTBehavior(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId,
                                        const struct ncclSymkLsaA2AParameters* parameters, int activeCtas,
                                        double* peakBw, double* extraLatencyUs) {
  *peakBw = parameters->peakBw;
  *extraLatencyUs = 0.0;
  if (kernelId != ncclSymkKernelId_AllGather_TmaST || input->comm->minCompCap < 100 ||
      RUBIN_AND_LATER(input->comm->minCompCap)) {
    return true;
  }

  if (parameters->ctaTroughPeakBw > 0.0 && activeCtas >= 14 && activeCtas % 4 == 2) {
    if (parameters->ctaTroughLatUs < 0.0) return false;
    *peakBw = std::min(*peakBw, parameters->ctaTroughPeakBw);
    *extraLatencyUs = parameters->ctaTroughLatUs;
  }
  if (parameters->rankLimitedPeakBw > 0.0 && activeCtas < input->comm->nRanks) {
    *peakBw = std::min(*peakBw, parameters->rankLimitedPeakBw);
  }
  return true;
}

static const struct ncclSymkLsaA2AParameters* parametersFor(const struct ncclTuningInput_t* input,
                                                            enum ncclSymkKernelId kernelId) {
  int kernel = static_cast<int>(kernelId);
  int arch = input == nullptr || input->comm == nullptr ? -1 : ncclSymkLsaA2AArchBucket(input->comm);
  if (input == nullptr || input->comm == nullptr || kernel < 0 || kernel >= ncclSymkKernelId_Count || arch < 0 ||
      !ncclSymkLsaA2AParameterTable[arch][kernel].valid) {
    return nullptr;
  }

  return input->comm->nRanks >= 2 && input->nWorks == 1 ? &ncclSymkLsaA2AParameterTable[arch][kernel] : nullptr;
}

bool ncclSymkLsaA2AModel(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, int requestedCtas,
                         float* timeUs, int* activeCtas, const struct ncclSymkLsaA2AParameters* parameters) {
  const struct ncclSymkLsaA2AParameters* tableParameters = parametersFor(input, kernelId);
  if (tableParameters == nullptr || requestedCtas < 1) return false;
  if (parameters == nullptr) parameters = tableParameters;
  if (!parameters->valid) return false;

  size_t logicalBytes = input->count * ncclTypeSize(input->datatype);
  bool isLowLatencyMulticast = kernelId == ncclSymkKernelId_AllGather_LLMC;
  bool isStoreMulticast = kernelId == ncclSymkKernelId_AllGather_TmaSTMC || kernelId == ncclSymkKernelId_AllGather_STMC;
  int nRanks = input->comm->nRanks;

  int modeledCtas = activeCtasForEachWork(logicalBytes, requestedCtas);
  double effectiveCtas = modeledCtas * allGatherCtaScale(parameters, kernelId, nRanks, modeledCtas);
  double peakBw;
  double extraLatencyUs;
  if (!modelBlackwellTmaSTBehavior(input, kernelId, parameters, modeledCtas, &peakBw, &extraLatencyUs)) return false;
  double rankLatencyUs = (nRanks - 1) * parameters->rankLatUs;
  double rankFactor = parameters->peakRankEfficiency ? (double)(nRanks - 1) / nRanks : 1.0;
  double ctaCopies = isLowLatencyMulticast ? nRanks : isStoreMulticast ? 1.0 : nRanks - (input->inPlace != 0);
  double ctaTransferTimeUs = ctaCopies * logicalBytes / (effectiveCtas * parameters->ctaBw * bwToBytesPerUs);
  double computeTimeUs = 0.0;
  if (parameters->computeCtaBw > 0.0) {
    computeTimeUs = logicalBytes / (effectiveCtas * parameters->computeCtaBw * bwToBytesPerUs);
  }
  double ctaTimeUs = computeTimeUs + ctaTransferTimeUs;
  double peakTransferTimeUs = (nRanks - 1) * logicalBytes / (peakBw * rankFactor * bwToBytesPerUs);
  double bandwidthBoundTime = std::max(ctaTimeUs, peakTransferTimeUs);
  double estimateUs;
  if (isLowLatencyMulticast) {
    double overlapFraction = std::min(1.0, parameters->fullOverlapCtas / modeledCtas);
    double exposedShorterTimeUs = (1.0 - overlapFraction) * std::min(rankLatencyUs, bandwidthBoundTime);
    estimateUs = parameters->baseLatUs + std::max(rankLatencyUs, bandwidthBoundTime) + exposedShorterTimeUs;
  } else {
    estimateUs = parameters->baseLatUs + rankLatencyUs + extraLatencyUs + bandwidthBoundTime;
  }
  if (!std::isfinite(estimateUs) || !(estimateUs > 0.0)) return false;

  *timeUs = static_cast<float>(estimateUs);
  *activeCtas = modeledCtas;
  return true;
}
