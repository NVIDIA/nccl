/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PROFILER_EVENT_H_
#define NCCL_PROFILER_EVENT_H_

// Per-plugin event handles, copied by value the way a single void* was.
// Kept separate from profiler.h so proxy and CE structs can store handles
// without pulling the profiler wrapper API.
//
// Slots are not free, and their cost is paid whether or not a profiler plugin
// is loaded. Each slot adds 8 bytes per handle, and handles are replicated
// across NCCL's two largest host allocations:
//   * ncclProxyOp holds one handle and lives in the proxy shm pool, which has
//     MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS entries: ~2.4 MB per slot.
//   * ncclProxySubArgs holds three handles plus one per pHandles[] entry, so
//     3 + NCCL_STEPS = 11 of them, and ncclProxyArgs embeds
//     NCCL_PROXY_MAX_SUBS subs across NCCL_MAX_OPS pool entries: ~11.5 MB per
//     slot, resident because ncclCalloc() memsets the pool.
// That is roughly 14 MB of host memory per rank per slot. Two slots cover the
// combinations the feature exists to serve (one telemetry plugin alongside one
// tracer); raising this is a deliberate memory decision, not a free knob.
#define NCCL_PROFILER_MAX_PLUGINS 2

struct ncclProfilerEventRef {
  void* h[NCCL_PROFILER_MAX_PLUGINS];
};

#endif
