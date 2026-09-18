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
#include <cmath>

enum ncclSymkLsaA2AAllGatherKernel {
  ncclSymkLsaA2AAllGatherKernel_LL,
  ncclSymkLsaA2AAllGatherKernel_LLMC,
  ncclSymkLsaA2AAllGatherKernel_TmaST,
  ncclSymkLsaA2AAllGatherKernel_ST,
  ncclSymkLsaA2AAllGatherKernel_TmaSTMC,
  ncclSymkLsaA2AAllGatherKernel_STMC,
  ncclSymkLsaA2AAllGatherKernel_Count,
};

struct ncclSymkLsaA2ACtaScalingCurve {
  double ctaScale[2];
};

struct ncclSymkLsaA2AKernelTuningParameters {
  double baseLatencyUs;
  double rankLatencyUs;
  double computeCtaBandwidthGbps;
  double transferCtaBandwidthGbps;
  double peakBandwidthGbps;
  struct ncclSymkLsaA2ACtaScalingCurve ctaScalingCurve;
  double fullOverlapCtas;
  bool peakRankEfficiency;
  double ctaTroughLatUs;
  double ctaTroughPeakBw;
  double rankLimitedPeakBw;
};

struct ncclSymkLsaA2AArchTuningParameters {
  int computeCapability;
  int rankSharedMulticastCtaBudget;
  int allGatherMulticastCtaLimit;
  struct ncclSymkLsaA2AKernelTuningParameters allGather[ncclSymkLsaA2AAllGatherKernel_Count];
};

// Each row owns CTA limits and fitted AllGather timing terms for one compute capability.
static constexpr struct ncclSymkLsaA2AArchTuningParameters lsaA2AArchTuningParameters[] = {
  {100,
   32,
   0,
   {
     {7.1084, 0.1065, 2.32, 15.87, 250.0, {{0.99, 0.69}}, 0.0, false},
     {7.4229, 0.1030, 2.1932, 27.46, 316.0699, {{1.0, 1.0}}, 3.3805, true},
     {10.0618, 0.0679, 0.0, 64.1577, 671.6052, {{1.0, 1.0}}, 0.0, false},
     {10.5902, 0.0563, 0.0, 64.4810, 650.4378, {{1.0, 1.0}}, 0.0, false},
     {8.2723, 0.0623, 0.0, 51.55, 715.1451, {{1.0, 1.0}}, 0.0, true},
     {8.3313, 0.0561, 0.0, 50.83, 715.1451, {{1.0, 1.0}}, 0.0, true},
   }},
  {103,
   32,
   0,
   {
     {7.1084, 0.1065, 2.32, 15.87, 250.0, {{0.99, 0.69}}, 0.0, false},
     {7.4229, 0.1030, 2.1932, 27.46, 316.0699, {{1.0, 1.0}}, 3.3805, true},
     {10.0618, 0.0679, 0.0, 64.1577, 671.6052, {{1.0, 1.0}}, 0.0, false, 12.1475, 639.6826, 585.7896},
     {10.5902, 0.0563, 0.0, 64.4810, 650.4378, {{1.0, 1.0}}, 0.0, false},
     {8.2723, 0.0623, 0.0, 51.55, 715.1451, {{1.0, 1.0}}, 0.0, true},
     {8.3313, 0.0561, 0.0, 50.83, 715.1451, {{1.0, 1.0}}, 0.0, true},
   }},
  {107,
   32,
   ncclSymkMaxBlocks,
   {
     {7.90, 0.29, 1.50, 18.05, 380.00, {{0.88, 0.60}}, 0.00, false},
     {8.02, 0.24, 1.55, 26.63, 497.24, {{1.00, 1.00}}, 1.52, true},
     {22.60, 0.30, 0.00, 73.98, 1170.00, {{1.00, 1.00}}, 0.00, false},
     {16.59, 0.31, 0.00, 74.10, 1000.00, {{1.00, 1.00}}, 0.00, false},
     {19.51, 0.19, 0.00, 61.09, 1237.49, {{1.00, 1.00}}, 0.00, true},
     {16.14, 0.22, 0.00, 61.21, 1100.15, {{1.00, 1.00}}, 0.00, true},
   }},
};

static const struct ncclSymkLsaA2AArchTuningParameters* lsaA2AArchTuningForComm(const struct ncclComm* comm) {
  for (const struct ncclSymkLsaA2AArchTuningParameters& tuning : lsaA2AArchTuningParameters) {
    if (tuning.computeCapability == comm->minCompCap) return &tuning;
  }
  return nullptr;
}

static int lsaA2AAllGatherKernelIndex(enum ncclSymkKernelId kernelId) {
  switch (kernelId) {
  case ncclSymkKernelId_AllGather_LL:
    return ncclSymkLsaA2AAllGatherKernel_LL;
  case ncclSymkKernelId_AllGather_LLMC:
    return ncclSymkLsaA2AAllGatherKernel_LLMC;
  case ncclSymkKernelId_AllGather_TmaST:
    return ncclSymkLsaA2AAllGatherKernel_TmaST;
  case ncclSymkKernelId_AllGather_ST:
    return ncclSymkLsaA2AAllGatherKernel_ST;
  case ncclSymkKernelId_AllGather_TmaSTMC:
    return ncclSymkLsaA2AAllGatherKernel_TmaSTMC;
  case ncclSymkKernelId_AllGather_STMC:
    return ncclSymkLsaA2AAllGatherKernel_STMC;
  default:
    return -1;
  }
}

int ncclSymkLsaMaxCtas(const struct ncclComm* comm, enum ncclSymkKernelId kernelId) {
  const struct ncclSymkLsaA2AArchTuningParameters* archTuning = lsaA2AArchTuningForComm(comm);
  int rankSharedMulticastCtaBudget =
    archTuning == nullptr ? (comm->minCompCap < 100 ? 16 : 32) : archTuning->rankSharedMulticastCtaBudget;
  switch (kernelId) {
  case ncclSymkKernelId_AllGather_TmaSTMC:
  case ncclSymkKernelId_AllGather_STMC:
    if (archTuning != nullptr && archTuning->allGatherMulticastCtaLimit > 0) {
      return archTuning->allGatherMulticastCtaLimit;
    }
    // fall through
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
  case ncclSymkKernelId_ReduceScatter_LDMC:
    return divUp(rankSharedMulticastCtaBudget, comm->nRanks);
  default:
    return ncclSymkMaxBlocks;
  }
}

static int lsaA2AActiveCtas(size_t logicalBytes, int requestedCtas) {
  size_t cells = divUp(logicalBytes, size_t(NCCL_SYM_KERNEL_CELL_SIZE));
  size_t cellsPerCta = divUp(cells, size_t(requestedCtas));
  return (int)divUp(cells, cellsPerCta);
}

// Apply measured effective-CTA scaling for AllGather kernels.
static double lsaA2AAllGatherCtaScale(const struct ncclSymkLsaA2AKernelTuningParameters* tuning,
                                      enum ncclSymkKernelId kernelId, int nRanks, int activeCtas) {
  if (kernelId == ncclSymkKernelId_AllGather_TmaSTMC && activeCtas == 1) return 1.1445;
  if (kernelId != ncclSymkKernelId_AllGather_LL || nRanks <= 4 || activeCtas <= 4 || activeCtas >= 64) {
    return 1.0;
  }

  static constexpr double log2CtaPositions[] = {2.0, 3.0, 5.0, 6.0};
  static constexpr int positionCount = sizeof(log2CtaPositions) / sizeof(log2CtaPositions[0]);
  static_assert(positionCount - 2 == sizeof(tuning->ctaScalingCurve.ctaScale) / sizeof(double),
                "CTA scale positions and fitted values differ");
  double log2ActiveCtas = std::log2(static_cast<double>(activeCtas));
  for (int upper = 1; upper < positionCount; upper++) {
    if (log2ActiveCtas <= log2CtaPositions[upper]) {
      double lowerScale = upper == 1 ? 1.0 : tuning->ctaScalingCurve.ctaScale[upper - 2];
      double upperScale = upper == positionCount - 1 ? 1.0 : tuning->ctaScalingCurve.ctaScale[upper - 1];
      double fraction =
        (log2ActiveCtas - log2CtaPositions[upper - 1]) / (log2CtaPositions[upper] - log2CtaPositions[upper - 1]);
      return lowerScale + fraction * (upperScale - lowerScale);
    }
  }
  return 1.0;
}

static const struct ncclSymkLsaA2AKernelTuningParameters* lsaA2AParametersForKernel(
  const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId) {
  if (input == nullptr || input->comm == nullptr || input->comm->nRanks < 2 || input->nWorks != 1) return nullptr;
  const struct ncclSymkLsaA2AArchTuningParameters* archTuning = lsaA2AArchTuningForComm(input->comm);
  int kernelIndex = lsaA2AAllGatherKernelIndex(kernelId);
  return archTuning == nullptr || kernelIndex < 0 ? nullptr : &archTuning->allGather[kernelIndex];
}

ncclResult_t ncclSymkLsaA2AModel(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId,
                                 int requestedCtas, float* timeUs, bool* modeled) {
  *modeled = false;
  *timeUs = 0.0f;
  if (requestedCtas < 1) return ncclSuccess;
  const struct ncclSymkLsaA2AKernelTuningParameters* tuning = lsaA2AParametersForKernel(input, kernelId);
  if (tuning == nullptr) return ncclSuccess;

  constexpr double bytesPerUsPerGbps = 1.e9 / 1.e6;
  size_t logicalBytes = input->count * ncclTypeSize(input->datatype);
  if (logicalBytes == 0) return ncclSuccess;
  bool isLowLatencyMulticast = kernelId == ncclSymkKernelId_AllGather_LLMC;
  bool exposeCtaComputeTime = kernelId == ncclSymkKernelId_AllGather_LL && input->comm->minCompCap == 107;
  bool isStoreMulticast = kernelId == ncclSymkKernelId_AllGather_TmaSTMC || kernelId == ncclSymkKernelId_AllGather_STMC;
  int nRanks = input->comm->nRanks;
  double rankLatencyUs = (nRanks - 1) * tuning->rankLatencyUs;
  double rankEfficiency = tuning->peakRankEfficiency ? (double)(nRanks - 1) / nRanks : 1.0;
  double ctaCopies = isLowLatencyMulticast ? nRanks : isStoreMulticast ? 1.0 : nRanks - (input->inPlace != 0);
  int activeCtas = lsaA2AActiveCtas(logicalBytes, requestedCtas);
  double scaledCtas = activeCtas * lsaA2AAllGatherCtaScale(tuning, kernelId, nRanks, activeCtas);
  double peakBandwidthGbps = tuning->peakBandwidthGbps;
  double extraLatencyUs = 0.0;
  if (kernelId == ncclSymkKernelId_AllGather_TmaST && input->comm->minCompCap >= 100 &&
      !RUBIN_AND_LATER(input->comm->minCompCap)) {
    if (tuning->ctaTroughPeakBw > 0.0 && activeCtas >= 14 && activeCtas % 4 == 2) {
      if (tuning->ctaTroughLatUs < 0.0) return ncclSuccess;
      peakBandwidthGbps = std::min(peakBandwidthGbps, tuning->ctaTroughPeakBw);
      extraLatencyUs = tuning->ctaTroughLatUs;
    }
    if (tuning->rankLimitedPeakBw > 0.0 && activeCtas < nRanks) {
      peakBandwidthGbps = std::min(peakBandwidthGbps, tuning->rankLimitedPeakBw);
    }
  }
  double ctaTransferTimeUs =
    ctaCopies * logicalBytes / (scaledCtas * tuning->transferCtaBandwidthGbps * bytesPerUsPerGbps);
  double ctaComputeTimeUs = 0.0;
  if (tuning->computeCtaBandwidthGbps > 0.0) {
    ctaComputeTimeUs = logicalBytes / (scaledCtas * tuning->computeCtaBandwidthGbps * bytesPerUsPerGbps);
  }
  double peakTransferTimeUs = (nRanks - 1) * logicalBytes / (peakBandwidthGbps * rankEfficiency * bytesPerUsPerGbps);
  double bandwidthBoundTimeUs = exposeCtaComputeTime ?
                                  ctaComputeTimeUs + std::max(ctaTransferTimeUs, peakTransferTimeUs) :
                                  std::max(ctaComputeTimeUs + ctaTransferTimeUs, peakTransferTimeUs);
  double estimateUs;
  if (isLowLatencyMulticast) {
    double overlapFraction = std::min(1.0, tuning->fullOverlapCtas / activeCtas);
    double exposedShorterTimeUs = (1.0 - overlapFraction) * std::min(rankLatencyUs, bandwidthBoundTimeUs);
    estimateUs = tuning->baseLatencyUs + std::max(rankLatencyUs, bandwidthBoundTimeUs) + exposedShorterTimeUs;
  } else {
    estimateUs = tuning->baseLatencyUs + rankLatencyUs + extraLatencyUs + bandwidthBoundTimeUs;
  }
  if (!std::isfinite(estimateUs) || !(estimateUs > 0.0)) return ncclSuccess;
  *timeUs = static_cast<float>(estimateUs);
  *modeled = true;
  return ncclSuccess;
}
