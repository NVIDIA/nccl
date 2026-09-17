/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_LSA_A2A_H_
#define NCCL_INT_SYM_MODEL_LSA_A2A_H_

#include "sym_kernels.h"
#include "tuning.h"

struct ncclSymkLsaA2ACtaScalingCurve {
  double ctaScale[2];
};

struct ncclSymkLsaA2AParameters {
  bool valid;
  double baseLatUs;
  double rankLatUs;
  double computeCtaBw; // GB/s; rank-independent LL CTA compute work; zero for other kernels.
  double ctaBw; // GB/s.
  double peakBw; // GB/s.
  struct ncclSymkLsaA2ACtaScalingCurve ctaScalingCurve;
  double fullOverlapCtas; // LLMC active CTA threshold for full overlap; zero otherwise.
  bool peakRankEfficiency;
  double ctaTroughLatUs; // Additional latency for the measured TmaST 4k+2 CTA series.
  double ctaTroughPeakBw; // GB/s; measured TmaST 4k+2 CTA ceiling.
  double rankLimitedPeakBw; // GB/s; TmaST ceiling when active CTAs are fewer than ranks.
};

enum {
  NCCL_SYMK_LSA_A2A_ARCH_SM100 = 0,
  NCCL_SYMK_LSA_A2A_ARCH_SM103 = 1,
  NCCL_SYMK_LSA_A2A_ARCH_COUNT
};

extern const struct ncclSymkLsaA2AParameters ncclSymkLsaA2AParameterTable[NCCL_SYMK_LSA_A2A_ARCH_COUNT]
                                                                         [ncclSymkKernelId_Count];

// Evaluate a fixed requested CTA geometry. An explicit row is used by the offline fitter.
bool ncclSymkLsaA2AModel(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, int requestedCtas,
                         float* timeUs, int* activeCtas, const struct ncclSymkLsaA2AParameters* parameters = nullptr);

#endif // NCCL_INT_SYM_MODEL_LSA_A2A_H_
