/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"
#include "lsa_a2a.h"

#include "comm.h"
#include "core.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

static int maxBlocksLsa(struct ncclComm* comm, enum ncclSymkKernelId kernelId) {
  switch (kernelId) {
  case ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC:
  case ncclSymkKernelId_AllGather_TmaSTMC:
  case ncclSymkKernelId_AllGather_STMC:
  case ncclSymkKernelId_ReduceScatter_LDMC:
    return divUp((comm->minCompCap < 100 ? 16 : 32), comm->nRanks);
  default:
    return ncclSymkMaxBlocks;
  }
}

static bool evaluateLsaEstimate(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                                int nBlocks, struct ncclSymkLsaEstimate* estimate) {
  int activeCtas;
  if (ncclSymkLsaA2AModel(input, kernelId, nBlocks, &estimate->timeUs, &activeCtas)) {
    estimate->selectionTimeUs = estimate->timeUs;
    return true;
  }
  return ncclSymkLsaBaseModel(input, kernelId, nBytes, nBlocks, estimate);
}

// Select the CTA count with the shared policy using either the A2A or base model.
void ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes, float* timeUs,
                      int* nBlocks) {
  struct ncclComm* comm = input->comm;
  int nMaxBlocks = std::min<int>(maxBlocksLsa(comm, kernelId), input->maxCTAs);
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
        return;
      }
    } else {
      while (nMaxBlocks > nMinBlocks && !ncclSymkTmaDeepEligible(comm, kernelId, maxWorkBytes, nMaxBlocks)) {
        nMaxBlocks -= (nMaxBlocks == 2 ? 1 : 2);
      }
    }
  }

  struct ncclSymkLsaEstimate selectedEstimate;
  if (!evaluateLsaEstimate(input, kernelId, nBytes, nMaxBlocks, &selectedEstimate)) return;
  *nBlocks = nMaxBlocks;
  float maxSelectionTimeUs = static_cast<float>(selectedEstimate.selectionTimeUs);
  for (int candidate = nMinBlocks; candidate < nMaxBlocks; candidate += candidate == 1 ? 1 : 2) {
    struct ncclSymkLsaEstimate candidateEstimate;
    if (evaluateLsaEstimate(input, kernelId, nBytes, candidate, &candidateEstimate) &&
        candidateEstimate.selectionTimeUs <= 1.025 * maxSelectionTimeUs) {
      selectedEstimate = candidateEstimate;
      *nBlocks = candidate;
      break;
    }
  }
  *timeUs = selectedEstimate.timeUs;
}
