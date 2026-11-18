/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_SYM_MODEL_GIN_AG_H_
#define NCCL_INT_SYM_MODEL_GIN_AG_H_

#include <cstddef>
#include <cstdint>

// Bandwidths are GB/s and latencies are microseconds.
struct ncclSymkGinAgParameters {
  double ctaBw;
  double peakBw;
  double gpuNetBw;
  double baseUs;
  double hopUs;
  double shareUs;
  double ctaNetBw;
  double postUs;
  double localChunkUs; // Completion cost per nonempty local chunk.
  double firstContextUs; // Delay before the last occupied context first posts.
  double localPower; // Smooth transition between CTA and shared local bandwidth.
};

enum ncclSymkGinAgArch {
  ncclSymkGinAgArch_Rubin,
  ncclSymkGinAgArch_Gb200,
  ncclSymkGinAgArch_Gb300,
  ncclSymkGinAgArch_Count,
};

enum ncclSymkGinAgArch ncclSymkGinAgArchForCapability(int capability);

constexpr int ncclSymkGinAgMaxCtas = 64; // Symmetric kernels support at most 64 CTAs.
constexpr int ncclSymkGinAgMaxLocalRanks = 72; // Support 72-rank LSA teams with the existing coefficients.

// One prepared partition and its rates serve every chunk candidate for a CTA count.
struct ncclSymkGinAgPipeline {
  // Sorted unique per-CTA payload lengths; multiplicity and CTA order are omitted.
  // The final entry is the maximum span used by bandwidth and bound calculations.
  uint64_t distinctSpanBytes[ncclSymkGinAgMaxCtas];
  int distinctSpanCount;
  int lsaRanks;
  int railRanks;
  double networkUsPerByte;
  double localUsPerByte;
  double postingUs;
  double firstIssueUs;

  double baseUs;
  double hopUs;
  double localChunkUs;
};

struct ncclSymkGinAgParameters ncclSymkGinAgDefaults(enum ncclSymkGinAgArch arch);

struct ncclSymkGinAgPipeline ncclSymkGinAgPrepare(const struct ncclSymkGinAgParameters& parameters, size_t rankBytes,
                                                  int requestedCtas, int localRanks, int remoteRanks, int contexts);

double ncclSymkGinAgPredict(const struct ncclSymkGinAgPipeline& pipeline, size_t chunkBytes);

#endif // NCCL_INT_SYM_MODEL_GIN_AG_H_
