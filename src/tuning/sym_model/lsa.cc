/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "model.h"
#include "selection_cache.h"

#include "comm.h"
#include "core.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>

namespace {
struct ncclSymkLsaCtaChoice {
  int nBlocks;
  float timeUs;
  float selectionTimeUs;
};
} // namespace

static ncclResult_t evaluateLsaEstimate(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                                        int nBlocks, struct ncclSymkLsaEstimate* estimate, bool* modeled) {
  float selectionCostPercent = 0.0250f;
  NCCLCHECK(ncclSymkLsaA2AModel(input, kernelId, nBlocks, &estimate->timeUs, modeled));
  if (*modeled) {
    estimate->ctaSelectionTimeUs = estimate->timeUs;
    bool lowerCtaPenalty = input->func == ncclFuncAllReduce ||
                           (RUBIN_AND_LATER(input->comm->minCompCap) && input->func == ncclFuncReduceScatter);
    selectionCostPercent = lowerCtaPenalty ? 0.0100f : 0.0200f;
  } else {
    NCCLCHECK(ncclSymkLsaBaseModel(input, kernelId, nBytes, nBlocks, estimate, modeled));
    if (!*modeled) return ncclSuccess;
  }
  estimate->selectionTimeUs = estimate->timeUs * (1.0f + selectionCostPercent * nBlocks);
  return ncclSuccess;
}

// Select the CTA count with the shared policy using either the A2A or base model.
ncclResult_t ncclSymkLsaModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                              float* timeUs, float* selectionTimeUs, int* nBlocks, int* maxBlocks) {
  struct ncclComm* comm = input->comm;
  int nMaxBlocks = std::min<int>(ncclSymkMaxBlocks, input->maxCTAs);
  int nMinBlocks = std::min(input->minCTAs, nMaxBlocks);

  *timeUs = FLT_MAX;
  *nBlocks = 0;
  if (maxBlocks) *maxBlocks = 0;

  // minCTAs/maxCTAs are resolved (env > per-call > comm) at task-append time.
  // NCCL_SYM_CTAS is an explicit override of the resolved bounds.
  int nUserCTAs = ncclSymkModelCtasEnvOverride();
  if (nUserCTAs > 0) nMinBlocks = nMaxBlocks = nUserCTAs;
  // nLimitBlocks is a loose limit for functional correctness: the largest CTA count the kernel may
  // run with, reported as maxBlocks so grouped launches stay within it. The even-count rounding and
  // TMA eligibility below tighten nMaxBlocks for performance and only shape the tuned count.
  int nLimitBlocks = nMaxBlocks;

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

  // Prefer power-of-two CTAs near modeled saturation for GB200 four-rank LDMC.
  // pow2Up(nMinBlocks) <= nMaxBlocks guarantees pow2Up(nMinBlocks) <= pow2Down(nMaxBlocks).
  bool preferPowerOfTwoCtas = comm->minCompCap == 100 && comm->nRanks == 4 && input->nWorks == 1 &&
                              input->func == ncclFuncAllReduce &&
                              kernelId == ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC && nUserCTAs <= 0 &&
                              nMinBlocks != nMaxBlocks && pow2Up(nMinBlocks) <= nMaxBlocks;
  if (preferPowerOfTwoCtas) {
    nMinBlocks = pow2Up(nMinBlocks);
    nMaxBlocks = pow2Down(nMaxBlocks);
  }
  double ctaTolerance = preferPowerOfTwoCtas ? 1.05 : 1.025;

  // Tuning runs on every call and the search below evaluates every CTA count, so reuse the choice of an
  // earlier call with the same values. The key holds every value the search and the models read.
  using Key = std::array<uint64_t, 15>;
  static thread_local ncclSymkSelectionCache<Key, ncclSymkLsaCtaChoice> cache;
  const Key key{{nBytes, uint64_t(kernelId), uint64_t(input->func), uint64_t(input->datatype), uint64_t(input->redOp),
                 uint64_t(input->devRedOp), input->count, uint64_t(input->nWorks), uint64_t(input->inPlace),
                 uint64_t(nMinBlocks), uint64_t(nMaxBlocks), uint64_t(preferPowerOfTwoCtas), uint64_t(comm->nRanks),
                 uint64_t(comm->minCompCap), uint64_t(comm->cudaArch)}};
  const ncclSymkLsaCtaChoice* cached = ncclSymkSelectionCacheFind(cache, key);
  if (cached) {
    *nBlocks = cached->nBlocks;
    *timeUs = cached->timeUs;
    *selectionTimeUs = cached->selectionTimeUs;
    if (maxBlocks) *maxBlocks = std::max(nLimitBlocks, *nBlocks);
    return ncclSuccess;
  }

  struct ncclSymkLsaEstimate candidates[ncclSymkMaxBlocks + 1];
  bool modeled = false;
  NCCLCHECK(evaluateLsaEstimate(input, kernelId, nBytes, nMaxBlocks, &candidates[nMaxBlocks], &modeled));
  if (!modeled) return ncclSuccess;
  *nBlocks = nMaxBlocks;
  // Evaluate every candidate before choosing the fewest CTAs within tolerance of the best estimate.
  double bestCtaSelectionTimeUs = candidates[nMaxBlocks].ctaSelectionTimeUs;
  for (int candidate = nMinBlocks; candidate < nMaxBlocks;
       candidate = preferPowerOfTwoCtas ? 2 * candidate : candidate + (candidate == 1 ? 1 : 2)) {
    struct ncclSymkLsaEstimate& candidateEstimate = candidates[candidate];
    NCCLCHECK(evaluateLsaEstimate(input, kernelId, nBytes, candidate, &candidateEstimate, &modeled));
    if (!modeled) {
      candidateEstimate.ctaSelectionTimeUs = INFINITY;
      continue;
    }
    bestCtaSelectionTimeUs = std::min(bestCtaSelectionTimeUs, candidateEstimate.ctaSelectionTimeUs);
  }
  for (int candidate = nMinBlocks; candidate < nMaxBlocks;
       candidate = preferPowerOfTwoCtas ? 2 * candidate : candidate + (candidate == 1 ? 1 : 2)) {
    if (candidates[candidate].ctaSelectionTimeUs <= ctaTolerance * bestCtaSelectionTimeUs) {
      *nBlocks = candidate;
      break;
    }
  }
  *timeUs = candidates[*nBlocks].timeUs;
  *selectionTimeUs = candidates[*nBlocks].selectionTimeUs;
  ncclSymkSelectionCacheInsert(cache, key, ncclSymkLsaCtaChoice{*nBlocks, *timeUs, *selectionTimeUs});
  if (maxBlocks) *maxBlocks = std::max(nLimitBlocks, *nBlocks);
  return ncclSuccess;
}
