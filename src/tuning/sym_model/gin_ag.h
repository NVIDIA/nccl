/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_GIN_AG_H_
#define NCCL_INT_SYM_MODEL_GIN_AG_H_

#include <cfloat>
#include <cstdint>

enum ncclSymkGinAgArch {
  ncclSymkGinAgArch_Rubin,
  ncclSymkGinAgArch_Gb200,
  ncclSymkGinAgArch_Gb300,
  ncclSymkGinAgArch_Count,
};

enum ncclSymkGinAgArch ncclSymkGinAgArchForCapability(int capability);

struct ncclSymkGinAgChoice {
  int ctas = 0;
  uint64_t chunkBytes = 0;
  double timeUs = 0, selectionUs = DBL_MAX;
};

constexpr int ncclSymkGinAgMaxCtas = 64; // Symmetric kernels support at most 64 CTAs.

struct ncclSymkGinAgChoice ncclSymkGinAgSelect(enum ncclSymkGinAgArch arch, uint64_t rankBytes, int localRanks,
                                               int remoteRanks, int contexts, int minimum, int maximum,
                                               uint64_t forcedChunk, int preferredCtas, uint64_t preferredChunk);

#endif // NCCL_INT_SYM_MODEL_GIN_AG_H_
