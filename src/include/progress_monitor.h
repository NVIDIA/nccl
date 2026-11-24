/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROGRESS_COUNTER_MONITOR_H_
#define NCCL_PROGRESS_COUNTER_MONITOR_H_

#include "nccl_common.h"

struct ncclComm;
struct ncclProgressCountersBlock;

// RAS represents local CUDA devices with a 64-bit mask.
static constexpr int kRasMaxCudaDevices = 64;

enum ncclCounterMonitorState : uint32_t {
  // Stored by communicator initialization and the sampler.
  ncclCounterInitializing = 0,
  // The last read, if any, was within budget; after the RAS gather it means a fresh snapshot.
  ncclCounterRunning,
  ncclCounterInitFailed,
  // Derived by the RAS gather from Running and the snapshot age; never stored.
  ncclCounterWaitingFirstSample,
  ncclCounterSnapshotStale,
  ncclCounterFirstSampleStale,
  ncclCounterMonitorStateCount
};
static_assert(ncclCounterMonitorStateCount - 1 <= UINT8_MAX, "RAS monitor state exceeds its uint8_t wire field");

ncclResult_t ncclProgressCounterMonitorInit(struct ncclComm* comm);
ncclResult_t ncclProgressCounterMonitorDestroy(struct ncclComm* comm);
int ncclProgressCounterMonitorStaleThresholdMs();
// Publish a within-budget sample; the sampler holds ncclCommsMutex. Exposed for unit tests.
void ncclProgressCounterMonitorPublish(struct ncclComm* comm, const struct ncclProgressCountersBlock* sample,
                                       uint64_t beginNs, uint64_t nowNs, uint64_t staleNs);

// Mark the function's counts as missing uninstrumented work; sticky, since count gaps persist.
void ncclProgressCounterMarkUninstrumented(struct ncclComm* comm, int func);
// Mark the function's host count as missing graph replays, unlike GPU completions; sticky.
void ncclProgressCounterMarkGraphUnsafe(struct ncclComm* comm, int func);

#endif
