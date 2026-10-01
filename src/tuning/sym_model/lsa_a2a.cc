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

enum ncclSymkLsaA2AReduceScatterKernel {
  ncclSymkLsaA2AReduceScatterKernel_LL,
  ncclSymkLsaA2AReduceScatterKernel_TmaLD,
  ncclSymkLsaA2AReduceScatterKernel_LD,
  ncclSymkLsaA2AReduceScatterKernel_LDMC,
  ncclSymkLsaA2AReduceScatterKernel_Count,
};

struct ncclSymkLsaA2ACtaScalingCurve {
  double ctaScale[2];
};

struct ncclSymkLsaA2AKernelTuningParameters {
  double baseLatencyUs;
  double rankLatencyUs;
  double transferCtaBandwidthGbps;
  double smallChunkCtaBw; // GB/s per CTA for 2 KiB chunks; zero uses the bulk rate.
  double reduceScalarCtaBw; // GB/s per CTA for the scalar tail; zero uses the bulk rate.
  double llCtaBw; // GB/s per CTA for rank-independent LL compute work.
  double peakBandwidthGbps;
  struct ncclSymkLsaA2ACtaScalingCurve ctaScalingCurve;
  double fullOverlapCtas;
  bool peakRankEfficiency;
  double ctaTroughLatUs;
  double ctaTroughPeakBw;
  double rankLimitedPeakBw;
};

struct ncclSymkLsaA2AReduceScatterTuningParameters {
  struct ncclSymkLsaA2AKernelTuningParameters defaultParameters;
  // Complete Sum rows; an empty row uses defaultParameters.
  struct ncclSymkLsaA2AKernelTuningParameters fp16Bf16SumParameters;
  struct ncclSymkLsaA2AKernelTuningParameters fp8SumParameters;
};

struct ncclSymkLsaA2AArchTuningParameters {
  int computeCapability;
  int rankSharedMulticastCtaBudget;
  int allGatherMulticastCtaLimit;
  struct ncclSymkLsaA2AReduceScatterTuningParameters reduceScatter[ncclSymkLsaA2AReduceScatterKernel_Count];
  struct ncclSymkLsaA2AKernelTuningParameters allGather[ncclSymkLsaA2AAllGatherKernel_Count];
};

static constexpr struct ncclSymkLsaA2AKernelTuningParameters reduceScatterParameters(
  double baseLatUs, double rankLatUs, double ctaBw, double smallChunkCtaBw, double reduceScalarCtaBw, double llCtaBw,
  double peakBw, bool peakRankEfficiency = false) {
  return {baseLatUs,
          rankLatUs,
          ctaBw,
          smallChunkCtaBw,
          reduceScalarCtaBw,
          llCtaBw,
          peakBw,
          {{1.0, 1.0}},
          0.0,
          peakRankEfficiency,
          0.0,
          0.0,
          0.0};
}

// Each row owns CTA limits and fitted timing terms for one compute capability.
static constexpr struct ncclSymkLsaA2AArchTuningParameters lsaA2AArchTuningParameters[] = {
  {100,
   32,
   0,
   {},
   {
     {7.1084, 0.1065, 15.87, 0.0, 0.0, 2.32, 250.0, {{0.99, 0.69}}, 0.0, false},
     {7.4229, 0.1030, 27.46, 0.0, 0.0, 2.1932, 316.0699, {{1.0, 1.0}}, 3.3805, true},
     {10.0618, 0.0679, 64.1577, 0.0, 0.0, 0.0, 671.6052, {{1.0, 1.0}}, 0.0, false},
     {10.5902, 0.0563, 64.4810, 0.0, 0.0, 0.0, 650.4378, {{1.0, 1.0}}, 0.0, false},
     {8.2723, 0.0623, 51.55, 0.0, 0.0, 0.0, 715.1451, {{1.0, 1.0}}, 0.0, true},
     {8.3313, 0.0561, 50.83, 0.0, 0.0, 0.0, 715.1451, {{1.0, 1.0}}, 0.0, true},
   }},
  {103,
   32,
   0,
   {
     {
       // LL
       reduceScatterParameters(10.5737, 0.0129, 8.6531, 0, 0, 2.1255, 248.9148),
       {},
       {},
     },
     {
       // TmaLD
       reduceScatterParameters(5.9025, 0.2089, 46.7794, 9.5412, 7.3964, 0, 640.7738), // Default
       reduceScatterParameters(5.9025, 0.2089, 41.1659, 7.3839, 3.6620, 0, 640.7738), // FP16/BF16 Sum
       reduceScatterParameters(5.9025, 0.2089, 24.6948, 6.5968, 0.4053, 0, 640.7738), // FP8 Sum
     },
     {
       // LD
       reduceScatterParameters(9.6736, 0.2801, 26.0000, 12.3392, 7.3964, 0, 660), // Default
       reduceScatterParameters(8.4739, 0.2096, 16.9749, 9.5826, 4.0022, 0, 660.0000), // FP16/BF16 Sum
       reduceScatterParameters(8.4739, 0.2096, 15.8819, 9.3889, 1.9578, 0, 660.0000), // FP8 Sum
     },
     {
       // LDMC
       reduceScatterParameters(10.9124, 0.0736, 24.5386, 0, 0, 0, 750.1525, true),
       {},
       {},
     },
   },
   {
     {7.1084, 0.1065, 15.87, 0.0, 0.0, 2.32, 250.0, {{0.99, 0.69}}, 0.0, false},
     {7.4229, 0.1030, 27.46, 0.0, 0.0, 2.1932, 316.0699, {{1.0, 1.0}}, 3.3805, true},
     {10.0618, 0.0679, 64.1577, 0.0, 0.0, 0.0, 671.6052, {{1.0, 1.0}}, 0.0, false, 12.1475, 639.6826, 585.7896},
     {10.5902, 0.0563, 64.4810, 0.0, 0.0, 0.0, 650.4378, {{1.0, 1.0}}, 0.0, false},
     {8.2723, 0.0623, 51.55, 0.0, 0.0, 0.0, 715.1451, {{1.0, 1.0}}, 0.0, true},
     {8.3313, 0.0561, 50.83, 0.0, 0.0, 0.0, 715.1451, {{1.0, 1.0}}, 0.0, true},
   }},
  {107,
   32,
   ncclSymkMaxBlocks,
   {
     {
       // LL
       reduceScatterParameters(10.1682, 0.1318, 8.6659, 0, 0, 1.5849, 378.2311),
       {},
       {},
     },
     {
       // TmaLD
       reduceScatterParameters(13.2601, 0.8953, 33.9867, 7.7706, 7.4807, 0, 1004.8552), // Default
       reduceScatterParameters(13.2601, 0.8953, 37.2708, 5.9251, 3.0701, 0, 1004.8552), // FP16/BF16 Sum
       reduceScatterParameters(13.2601, 0.8953, 23.4600, 5.6772, 0.3082, 0, 1004.8552), // FP8 Sum
     },
     {
       // LD
       reduceScatterParameters(13.2750, 1.0829, 17.9128, 9.4630, 5.6894, 0, 692.0016), // Default
       reduceScatterParameters(13.2750, 1.0829, 10.6581, 7.3404, 2.9289, 0, 692.0016), // FP16/BF16 Sum
       reduceScatterParameters(13.2750, 1.0829, 10.1315, 7.1995, 1.3376, 0, 692.0016), // FP8 Sum
     },
     {
       // LDMC
       reduceScatterParameters(11.7527, 0.2263, 15.1821, 0, 0, 0, 1134.4294, true),
       {},
       {},
     },
   },
   {
     {7.90, 0.29, 18.05, 0.0, 0.0, 1.50, 380.00, {{0.88, 0.60}}, 0.00, false},
     {8.02, 0.24, 26.63, 0.0, 0.0, 1.55, 497.24, {{1.00, 1.00}}, 1.52, true},
     {22.60, 0.30, 73.98, 0.0, 0.0, 0.00, 1170.00, {{1.00, 1.00}}, 0.00, false},
     {16.59, 0.31, 74.10, 0.0, 0.0, 0.00, 1000.00, {{1.00, 1.00}}, 0.00, false},
     {19.51, 0.19, 61.09, 0.0, 0.0, 0.00, 1237.49, {{1.00, 1.00}}, 0.00, true},
     {16.14, 0.22, 61.21, 0.0, 0.0, 0.00, 1100.15, {{1.00, 1.00}}, 0.00, true},
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

static int lsaA2AReduceScatterKernelIndex(enum ncclSymkKernelId kernelId) {
  switch (kernelId) {
  case ncclSymkKernelId_ReduceScatter_LL:
    return ncclSymkLsaA2AReduceScatterKernel_LL;
  case ncclSymkKernelId_ReduceScatter_TmaLD:
    return ncclSymkLsaA2AReduceScatterKernel_TmaLD;
  case ncclSymkKernelId_ReduceScatter_LD:
    return ncclSymkLsaA2AReduceScatterKernel_LD;
  case ncclSymkKernelId_ReduceScatter_LDMC:
    return ncclSymkLsaA2AReduceScatterKernel_LDMC;
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
    return divUp(rankSharedMulticastCtaBudget, comm->nRanks);
  case ncclSymkKernelId_ReduceScatter_LDMC:
    // Let the fitted RS model choose saturation instead of imposing a rank-scaled cap.
    return comm->minCompCap == 103 || comm->minCompCap == 107 ? ncclSymkMaxBlocks :
                                                                divUp(rankSharedMulticastCtaBudget, comm->nRanks);
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
  if (archTuning == nullptr) return nullptr;
  int rsIndex = lsaA2AReduceScatterKernelIndex(kernelId);
  // The LD, TmaLD, and LDMC fits require 16-byte-aligned buffers and output size.
  if (rsIndex >= 0) {
    if (input->func != ncclFuncReduceScatter || input->count == 0) return nullptr;
    size_t logicalBytes = input->count * ncclTypeSize(input->datatype);
    if (kernelId != ncclSymkKernelId_ReduceScatter_LL &&
        (!input->symAligned16B || !input->symInputAligned16B || logicalBytes % ncclSymkBytePerPack != 0))
      return nullptr;
    const struct ncclSymkLsaA2AReduceScatterTuningParameters& parameters = archTuning->reduceScatter[rsIndex];
    if (parameters.defaultParameters.transferCtaBandwidthGbps <= 0.0) return nullptr;
    if (input->redOp == ncclSum && input->devRedOp == ncclDevSum) {
      switch (input->datatype) {
      case ncclFloat16:
      case ncclBfloat16:
        if (parameters.fp16Bf16SumParameters.transferCtaBandwidthGbps > 0.0) return &parameters.fp16Bf16SumParameters;
        break;
      case ncclFloat8e4m3:
      case ncclFloat8e5m2:
        if (parameters.fp8SumParameters.transferCtaBandwidthGbps > 0.0) return &parameters.fp8SumParameters;
        break;
      default:
        break;
      }
    }
    return &parameters.defaultParameters;
  }
  int kernelIndex = lsaA2AAllGatherKernelIndex(kernelId);
  return kernelIndex < 0 ? nullptr : &archTuning->allGather[kernelIndex];
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
  bool isReduceScatter = lsaA2AReduceScatterKernelIndex(kernelId) >= 0;
  bool isLoadStoreMulticast = kernelId == ncclSymkKernelId_AllGather_TmaSTMC ||
                              kernelId == ncclSymkKernelId_AllGather_STMC ||
                              kernelId == ncclSymkKernelId_ReduceScatter_LDMC;
  int nRanks = input->comm->nRanks;
  double rankLatencyUs = (nRanks - 1) * tuning->rankLatencyUs;
  double rankEfficiency = tuning->peakRankEfficiency ? (double)(nRanks - 1) / nRanks : 1.0;
  double ctaCopies = isLoadStoreMulticast                     ? 1.0 :
                     isReduceScatter || isLowLatencyMulticast ? nRanks :
                                                                nRanks - (input->inPlace != 0);
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
  if (tuning->llCtaBw > 0.0) {
    ctaComputeTimeUs = logicalBytes / (scaledCtas * tuning->llCtaBw * bytesPerUsPerGbps);
  }
  double peakTransferTimeUs = (nRanks - 1) * logicalBytes / (peakBandwidthGbps * rankEfficiency * bytesPerUsPerGbps);
  double bandwidthBoundTimeUs = exposeCtaComputeTime ?
                                  ctaComputeTimeUs + std::max(ctaTransferTimeUs, peakTransferTimeUs) :
                                  std::max(ctaComputeTimeUs + ctaTransferTimeUs, peakTransferTimeUs);
  if (kernelId == ncclSymkKernelId_ReduceScatter_LD || kernelId == ncclSymkKernelId_ReduceScatter_TmaLD) {
    // Follow reduce_scatter.cuh: consume bulk chunks, then 2 KiB small chunks,
    // then the scalar tail. Each chunk loop rounds down to complete rank/CTA
    // groups; aligned input/output leaves no scalar prefix.
    size_t fullChunk =
      kernelId == ncclSymkKernelId_ReduceScatter_TmaLD ? ncclSymkDeepBytePerChunk : ncclSymkBytePerChunk;
    constexpr size_t smallChunk = ncclSymkSmallBytePerChunk;
    size_t group = size_t(nRanks) * activeCtas;
    size_t fullBytes = (logicalBytes / (group * fullChunk)) * (group * fullChunk);
    size_t smallBytes = ((logicalBytes - fullBytes) / (group * smallChunk)) * (group * smallChunk);
    size_t endsBytes = logicalBytes - fullBytes - smallBytes;

    // Each remainder rate independently defaults to the normal CTA bandwidth.
    double endsCtaBw = tuning->reduceScalarCtaBw > 0.0 ? tuning->reduceScalarCtaBw : tuning->transferCtaBandwidthGbps;
    double smallChunkCtaBw = tuning->smallChunkCtaBw > 0.0 ? tuning->smallChunkCtaBw : tuning->transferCtaBandwidthGbps;
    // Reuse the common compute and peak times in proportion to each portion's bytes.
    double endsFraction = double(endsBytes) / logicalBytes;
    double smallFraction = double(smallBytes) / logicalBytes;
    double fullFraction = double(fullBytes) / logicalBytes;
    // Compare CTA and peak limits separately for each portion before adding them.
    double endsTimeUs =
      std::max(ctaComputeTimeUs * endsFraction + ctaCopies * endsBytes / (scaledCtas * endsCtaBw * bytesPerUsPerGbps),
               peakTransferTimeUs * endsFraction);
    double smallTimeUs = std::max(ctaComputeTimeUs * smallFraction +
                                    ctaCopies * smallBytes / (scaledCtas * smallChunkCtaBw * bytesPerUsPerGbps),
                                  peakTransferTimeUs * smallFraction);
    double fullTimeUs = bandwidthBoundTimeUs * fullFraction;
    bandwidthBoundTimeUs = endsTimeUs + smallTimeUs + fullTimeUs;
  }
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
