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

void ncclSymkGinModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                      int* nBlocks);
bool ncclSymkLsaBaseCtas(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, int* nBlocks);
void ncclSymkLsaBaseModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                          int* nBlocks);
bool ncclSymkLsaA2AModelEnabled(const struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId);
void ncclSymkLsaA2AModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                         int* nBlocks);

#endif // NCCL_INT_SYM_MODEL_MODEL_H_
