/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"

#include "comm.h"
#include "core.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

static ncclResult_t evaluateLsaEstimate(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                                        int nBlocks, struct ncclSymkLsaEstimate* estimate, bool* modeled) {
  float selectionCostPercent = 0.0250f;
  NCCLCHECK(ncclSymkLsaA2AModel(input, kernelId, nBlocks, &estimate->timeUs, modeled));
  if (*modeled) {
    estimate->ctaSelectionTimeUs = estimate->timeUs;
    bool vrReduction =
      input->comm->minCompCap == 107 && (input->func == ncclFuncReduceScatter || input->func == ncclFuncAllReduce);
    selectionCostPercent = vrReduction ? 0.0100f : 0.0200f;
  } else {
    NCCLCHECK(ncclSymkLsaBaseModel(input, kernelId, nBytes, nBlocks, estimate, modeled));
    if (!*modeled) return ncclSuccess;
  }
  estimate->selectionTimeUs = estimate->timeUs * (1.0f + selectionCostPercent * nBlocks);
  return ncclSuccess;
}

// Select the CTA count with the shared policy using either the A2A or base model.
ncclResult_t ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, float* selectionTimeUs, int* nBlocks) {
  struct ncclComm* comm = input->comm;
  int nMaxBlocks = std::min<int>(ncclSymkMaxBlocks, input->maxCTAs);
  int nMinBlocks = std::min(input->minCTAs, nMaxBlocks);

  *timeUs = FLT_MAX;
  *nBlocks = 0;

  // minCTAs/maxCTAs are resolved (env > per-call > comm) at task-append time.
  // NCCL_SYM_CTAS is an explicit override of the resolved bounds.
  int nUserCTAs = ncclSymkModelCtasEnvOverride();
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;

  // Even CTA counts are preferred for optimal performance, except for when CTAs==1.
  if (nMinBlocks != nMaxBlocks) {
    if (nMinBlocks != 1) nMinBlocks = roundUp(nMinBlocks, 2);
    if (nMaxBlocks != 1) nMaxBlocks = roundDown(nMaxBlocks, 2);
  }

  if (ncclSymkTmaKernelMask() >> kernelId & 1) {
    size_t maxWorkBytes = input->countMax * ncclTypeSize(input->datatype);
    if (!ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMinBlocks)) {
      const char* symKernelIdEnv = ncclGetEnv("NCCL_SYM_KERNEL");
      if (symKernelIdEnv) {
        INFO(NCCL_TUNING,
             "NCCL_SYM_KERNEL set to %s. At largest grouped work size %zu Bytes, kernel will not exercise TMA paths.",
             symKernelIdEnv, maxWorkBytes);
      } else {
        return ncclSuccess;
      }
    } else {
      while (nMaxBlocks > nMinBlocks && !ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMaxBlocks)) {
        nMaxBlocks -= (nMaxBlocks == 2 ? 1 : 2);
      }
    }
  }

  struct ncclSymkLsaEstimate selectedEstimate;
  bool modeled = false;
  NCCLCHECK(evaluateLsaEstimate(input, kernelId, nBytes, nMaxBlocks, &selectedEstimate, &modeled));
  if (!modeled) return ncclSuccess;
  *nBlocks = nMaxBlocks;
  float maxCtaSelectionTimeUs = static_cast<float>(selectedEstimate.ctaSelectionTimeUs);
  for (int candidate = nMinBlocks; candidate < nMaxBlocks; candidate += candidate == 1 ? 1 : 2) {
    struct ncclSymkLsaEstimate candidateEstimate;
    NCCLCHECK(evaluateLsaEstimate(input, kernelId, nBytes, candidate, &candidateEstimate, &modeled));
    if (modeled && candidateEstimate.ctaSelectionTimeUs <= 1.025 * maxCtaSelectionTimeUs) {
      selectedEstimate = candidateEstimate;
      *nBlocks = candidate;
      break;
    }
  }
  *timeUs = selectedEstimate.timeUs;
  *selectionTimeUs = selectedEstimate.selectionTimeUs;
  return ncclSuccess;
}
