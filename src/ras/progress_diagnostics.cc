/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more information
 *************************************************************************/

// GPU progress diagnostics of a communicator: monitor state, classification and its reasons.

#include <cstdint>
#include <cstring>

#include "ras_internal.h"

uint8_t rasProgressMonitorStateForQuery(ncclCounterMonitorState state, uint64_t refreshed, uint64_t started,
                                        uint64_t now, uint64_t staleNs) {
  if (state != ncclCounterRunning) return (uint8_t)state;
  uint64_t base = refreshed ? refreshed : started;
  if (base != 0 && now >= base && now - base >= staleNs)
    return refreshed ? ncclCounterSnapshotStale : ncclCounterFirstSampleStale;
  if (refreshed != 0 && now >= refreshed) return ncclCounterRunning;
  return ncclCounterWaitingFirstSample;
}

static bool rasProgressSlotPending(const struct rasCollComms::comm::rank* rank, int slot) {
  if (rasProgressSlotActive(rank, slot)) return true;
  return rasProgressHostDeviceCountsComparable(rank, slot) &&
         rank->enqueuedCollOpCounts[slot] > rank->completedOpCounts[slot];
}

void rasClassifyProgress(const struct rasCollComms::comm::rank* ranks, int nRanks, bool completeRankSet,
                         uint64_t thresholdSec, struct rasProgressClassification* out) {
  memset(out, 0, sizeof(*out));
  if (nRanks <= 0) return;

  const struct rasCollComms::comm::rank* ref = nullptr; // First rank with a usable snapshot.
  int nUsable = 0;
  uint64_t usableUninstrumentedMask = 0; // Coverage of the fresh subset that GPU counts compare.
  for (int rankIdx = 0; rankIdx < nRanks; rankIdx++) {
    const struct rasCollComms::comm::rank* rank = ranks + rankIdx;
    out->anyCounterEnabled |= rank->devCounterEnabled;
    out->deviceCounterUnavailable |= rasProgressCounterUnavailable(rank);
    out->counterMirrorStale |= rasProgressCounterStale(rank);
    out->uninstrumentedSlotMask |= rank->uninstrumentedSlotMask;
    for (int slot = 0; slot < NCCL_NUM_FUNCTIONS; slot++)
      if (rank->enqueuedCollOpCounts[slot] != ranks[0].enqueuedCollOpCounts[slot])
        out->hostMismatchMask |= 1ULL << slot;
    if (rasProgressSnapshotUsable(rank)) {
      if (ref == nullptr) ref = rank;
      nUsable++;
      usableUninstrumentedMask |= rank->uninstrumentedSlotMask;
    }
  }

  out->nUsableRanks = nUsable;
  // Host counts need no snapshot; their mismatch is age-gated only when every rank is fresh and covered.
  out->ungatedHostMismatchMask =
    nUsable == nRanks ? out->hostMismatchMask & out->uninstrumentedSlotMask : out->hostMismatchMask;
  if (nUsable == 0) return;
  bool completeEvidence = completeRankSet && nUsable == nRanks;

  // The synthetic P2P slot is skipped: asymmetric send/receive counts are valid.
  for (int slot = 0; slot < ncclNumFuncs; slot++) {
    uint64_t bit = 1ULL << slot;
    if (usableUninstrumentedMask & bit) continue;
    bool hostMismatch = false;
    bool deviceMismatch = false;
    for (int rankIdx = 0; rankIdx < nRanks; rankIdx++) {
      const struct rasCollComms::comm::rank* rank = ranks + rankIdx;
      if (!rasProgressSnapshotUsable(rank)) continue;
      if (slot < NCCL_NUM_FUNCTIONS) // enqueuedCollOpCounts[] only covers the legacy five.
        hostMismatch |= rank->enqueuedCollOpCounts[slot] != ref->enqueuedCollOpCounts[slot];
      deviceMismatch |= rank->completedOpCounts[slot] != ref->completedOpCounts[slot];
    }
    if (hostMismatch || deviceMismatch) out->observedCounterMismatchMask |= bit;
    if (deviceMismatch) out->observedDeviceMismatchMask |= bit;
  }

  for (int slot = 0; slot < NCCL_NUM_PROGRESS_COUNTERS; slot++) {
    uint64_t bit = 1ULL << slot;
    uint64_t minLaunched = UINT64_MAX;
    if (slot < NCCL_NUM_FUNCTIONS) {
      for (int rankIdx = 0; rankIdx < nRanks; rankIdx++) {
        const struct rasCollComms::comm::rank* rank = ranks + rankIdx;
        if (rasProgressSnapshotUsable(rank) && rank->enqueuedCollOpCounts[slot] < minLaunched)
          minLaunched = rank->enqueuedCollOpCounts[slot];
      }
    }

    int activeRanks = 0;
    bool allActivityKnown = true;
    bool anyPending = false;
    bool pendingAgeHit = false;
    bool leaderAgeHit = false;
    bool activeAgeHit = false;
    bool allActiveAndAged = true;
    for (int rankIdx = 0; rankIdx < nRanks; rankIdx++) {
      const struct rasCollComms::comm::rank* rank = ranks + rankIdx;
      if (!rasProgressSnapshotUsable(rank)) continue;
      bool active = rasProgressSlotActive(rank, slot);
      allActivityKnown &= rasProgressSlotActivityUsable(rank, slot);
      bool aged = rank->noProgressSecAgo[slot] > thresholdSec;
      bool pending = rasProgressSlotPending(rank, slot);
      if (active) activeRanks++;
      anyPending |= pending;
      pendingAgeHit |= pending && aged;
      // A leader finished its lead before its completed count was first observed, so that age bounds the lead's.
      leaderAgeHit |= slot < NCCL_NUM_FUNCTIONS && rank->enqueuedCollOpCounts[slot] > minLaunched &&
                      rasProgressHostDeviceCountsComparable(rank, slot) && rank->completedAgeSec[slot] > thresholdSec;
      activeAgeHit |= active && aged;
      allActiveAndAged &= active && aged;
    }

    if (completeEvidence && allActiveAndAged) out->stuckSlotMask |= bit;
    if (allActivityKnown && activeRanks > 0 && activeRanks < nUsable && activeAgeHit) out->blockedSlotMask |= bit;
    // With nothing pending, as when a Reduce root lags, the leaders' completed-count age qualifies instead.
    if ((out->observedCounterMismatchMask & bit) && (pendingAgeHit || (!anyPending && leaderAgeHit)))
      out->qualifiedCounterMismatchMask |= bit;
    if ((out->observedDeviceMismatchMask & bit) && pendingAgeHit) out->qualifiedDeviceMismatchMask |= bit;
  }
  out->qualifiedHostMismatchMask = out->qualifiedCounterMismatchMask & out->hostMismatchMask;
}
