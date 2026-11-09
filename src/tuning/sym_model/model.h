/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_MODEL_H_
#define NCCL_INT_SYM_MODEL_MODEL_H_

#include "../model.h"
#include "sym_kernels.h"
#include "tuning.h"

#include <cstddef>

int ncclSymkModelCtasEnvOverride();

struct ncclSymkLsaEstimate {
  double ctaSelectionTimeUs;
  float timeUs;
  float selectionTimeUs;
};

ncclResult_t ncclSymkGinModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, int* nBlocks);
ncclResult_t ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, float* selectionTimeUs, int* nBlocks);
ncclResult_t ncclSymkLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                                  int nBlocks, struct ncclSymkLsaEstimate* estimate, bool* modeled);

ncclResult_t ncclSymkLsaA2AModel(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId,
                                 int requestedCtas, float* timeUs, bool* modeled);

#endif // NCCL_INT_SYM_MODEL_MODEL_H_
