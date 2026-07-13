/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/*************************************************************************
 * Device-side profiler hook ABI (mirror of NCCL's src/include/profiler_dev.h).
 *
 * A profiler plugin provides a __device__ callback (via ncclProfiler_v7's
 * getDeviceHook) that ncclKernelMain's profiler() invokes at each kernel
 * work-item boundary. The plugin owns its telemetry (format, buffer, atomicity,
 * depth, drain). When a hook is registered NCCL does not arm the profiler proxy
 * op, so no CUDA host callback is added to captured graphs.
 *************************************************************************/
#ifndef NCCL_PROFILER_DEV_H_
#define NCCL_PROFILER_DEV_H_

#include <stdint.h>

#define NCCL_PROFILER_DEV_START 0
#define NCCL_PROFILER_DEV_STOP 1

// One kernel work-item boundary event. POD + fixed layout for ABI stability.
typedef struct {
  uint64_t timestamp;   // globaltimer() ns (device clock)
  uint64_t workCounter; // monotonic per-channel work counter
  uint32_t funcId;      // collective identity (index into ncclDevFuncTable)
  uint8_t channelId;
  uint8_t phase;       // NCCL_PROFILER_DEV_START / _STOP
  uint16_t rsvd;
} ncclProfilerDevEvent_t;

// Plugin device hook. Called on lane 0, once per profiler-enabled work item,
// when registered. Must be cheap and must not touch the collective's data/sync.
typedef void (*ncclProfilerDevHook_t)(const ncclProfilerDevEvent_t* ev, void* devCtx);

#ifdef __cplusplus
extern "C" {
#endif
// Exported by libnccl: resolve an event's funcId to a human-readable name (e.g.
// "AllReduce_Sum_f32_RING_LL"), or "unknown" if out of range.
const char* ncclProfilerDevFuncName(int funcId);
#ifdef __cplusplus
}
#endif

#endif // NCCL_PROFILER_DEV_H_
