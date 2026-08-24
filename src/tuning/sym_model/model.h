/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_MODEL_H_
#define NCCL_INT_SYM_MODEL_MODEL_H_

#include "sym_kernels.h"
#include "tuning.h"

#include <cstddef>

int ncclSymkModelCtasEnvOverride();

struct ncclSymkLsaEstimate {
  double selectionTimeUs;
  float timeUs;
};

void ncclSymkGinModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                      int* nBlocks);
void ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                      int* nBlocks);
bool ncclSymkLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, int nBlocks,
                          struct ncclSymkLsaEstimate* estimate);

#endif // NCCL_INT_SYM_MODEL_MODEL_H_
