/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_KERNEL_PROFILER_H_
#define NCCL_DEVICE_KERNEL_PROFILER_H_

#include "sym_kernels.h"

// Symmetric and general kernel profiler support: write GPU timestamps to
// host-pinned profiler counter arrays so the CPU proxy thread can fire
// KernelCh events.
__device__ __forceinline__ unsigned long long int ncclSymkGlobaltimer() {
  unsigned long long int timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

template <typename Args>
__device__ __forceinline__ void ncclSymkProfilerStart(Args const* args) {
  if (threadIdx.x == 0 && args->profilerMode) {
    int ch = blockIdx.x;
    uint64_t wc = ncclSymkGetProfilerCounters(args)[ch];
    int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
    uint64_t ts = ncclSymkGlobaltimer();
    // The BEGIN stamp lands on another line and is ordered by ncclSymkProfilerStop's fence.
    if (args->profilerMode & ncclDevProfilerModeKernelPhase)
      args->kcomm.workPhases[ch].data[slot].timestamps[NCCL_KERNEL_PHASE_BEGIN] = ts;
    // workStarted timestamp+counter share one 16B slot, so publishing them needs no fence.
    args->kcomm.workStarted[ch].data[slot].timestamp = ts;
    args->kcomm.workStarted[ch].data[slot].counter = wc;
  }
}

template <typename Args>
__device__ __forceinline__ void ncclSymkProfilerStop(Args const* args) {
  if (threadIdx.x == 0 && args->profilerMode) {
    int ch = blockIdx.x;
    uint64_t wc = ncclSymkGetProfilerCounters(args)[ch];
    int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
    uint64_t ts = ncclSymkGlobaltimer();
    args->kcomm.workCompleted[ch].data[slot].timestamp = ts;
    // Only the phase stamps span the kernel and straddle lines, so the fence they need
    // is confined to the mode that writes them.
    if (args->profilerMode & ncclDevProfilerModeKernelPhase) {
      args->kcomm.workPhases[ch].data[slot].timestamps[NCCL_KERNEL_PHASE_END] = ts;
      __threadfence_system();
      args->kcomm.workPhases[ch].data[slot].counter = wc;
    }
    // workCompleted timestamp+counter share one 16B slot, so this needs no fence.
    args->kcomm.workCompleted[ch].data[slot].counter = wc;
  }
}

template <typename Args>
__device__ __forceinline__ void ncclSymkProfilerPhase(Args const* args, int phaseId) {
  if (threadIdx.x == 0 && (args->profilerMode & ncclDevProfilerModeKernelPhase)) {
    int ch = blockIdx.x;
    uint64_t wc = ncclSymkGetProfilerCounters(args)[ch];
    int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
    args->kcomm.workPhases[ch].data[slot].timestamps[phaseId] = ncclSymkGlobaltimer();
  }
}

#endif // NCCL_DEVICE_KERNEL_PROFILER_H_
