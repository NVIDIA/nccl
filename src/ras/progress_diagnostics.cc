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
