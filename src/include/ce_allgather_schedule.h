/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CE_ALLGATHER_SCHEDULE_H_
#define NCCL_CE_ALLGATHER_SCHEDULE_H_

#include "comm.h"

// Equal-sized UC copies use one batch, with additional rounds above the byte budget.
struct ncclCeBatchCopyPlan {
  static constexpr size_t syncFreq = 8;
  static constexpr size_t syncBytes = size_t(512) << 20;
  bool intraBatchSync;
  size_t chunkBytes;
  size_t rounds;

  ncclCeBatchCopyPlan(size_t copies, size_t bytes) {
    intraBatchSync = copies > syncFreq && bytes >= 1 + (syncBytes - 1) / copies;
    chunkBytes = intraBatchSync ? syncBytes / copies : bytes;
    rounds = intraBatchSync ? 1 + (bytes - 1) / chunkBytes : 1;
  }
};

struct ncclCeMcCopyPlan {
  size_t payloadBytes;
  size_t startOffset;
  size_t numCopies;

  size_t offset(size_t step) const {
    return step == 0 ? startOffset : 0;
  }
  size_t size(size_t step) const {
    return step == 0 ? payloadBytes - startOffset : startOffset;
  }

  // Rank 1 has phase 1. If its offset is zero, every rank's offset is zero;
  // otherwise two copies are the collective's maximum, including when self is rank 0.
  static size_t maxCopies(int ranks, size_t bytes) {
    return 1 + (ranks > 1 && ncclSymkMcStartOffset(1, bytes) != 0);
  }
};

inline struct ncclCeMcCopyPlan ncclCeAllGatherMcCopyPlan(const struct ncclComm* comm, size_t perRankBytes) {
  // STMC/TmaSTMC rotate the aligned main region, then round to their instruction tile.
  // CE supports byte-granular copies: rotate the whole payload and wrap once, without
  // imposing SM tile sizes or splitting contiguous spans into extra dispatches.
  size_t start = ncclSymkMcStartOffset(comm->devrState.lsaTeamSelf, perRankBytes);
  return {perRankBytes, start, 1 + size_t(start != 0)};
}

#endif // NCCL_CE_ALLGATHER_SCHEDULE_H_
