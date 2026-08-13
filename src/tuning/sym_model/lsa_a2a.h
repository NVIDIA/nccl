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

#include <cstddef>

struct ncclSymkLsaA2AParameters {
  bool valid;
  double baseLatUs;
  double rankLatUs;
  double ctaBwGBps;
  double peakBwGBps;
  bool peakRankEfficiency;
};

enum {
  NCCL_SYMK_LSA_A2A_ARCH_SM100 = 0,
  NCCL_SYMK_LSA_A2A_ARCH_SM103 = 1,
  NCCL_SYMK_LSA_A2A_ARCH_COUNT
};

extern const struct ncclSymkLsaA2AParameters ncclSymkLsaA2AParameterTable[NCCL_SYMK_LSA_A2A_ARCH_COUNT]
                                                                         [ncclSymkKernelId_Count];

// Evaluate one parameter row after generic tuning has validated the kernel and
// the base LSA model has selected its requested CTA geometry.
bool ncclSymkLsaA2AEvaluate(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t logicalBytes,
                            int requestedCtas, const struct ncclSymkLsaA2AParameters& parameters, float* timeUs,
                            int* activeCtas);

#endif // NCCL_INT_SYM_MODEL_LSA_A2A_H_
