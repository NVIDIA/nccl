/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVICE_SYMMETRIC_KERNEL_H_
#define NCCL_DEVICE_SYMMETRIC_KERNEL_H_

#include "sym_kernels.h"

// Symmetric kernel profiler support: write GPU timestamps to host-pinned
// profiler counter arrays so the CPU proxy thread can fire KernelCh events.
__device__ __forceinline__ unsigned long long int ncclSymkGlobaltimer() {
  unsigned long long int timer;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(timer));
  return timer;
}

__device__ __forceinline__ void ncclSymkProfilerStart(struct ncclSymkDevWorkArgs const* args) {
  if (threadIdx.x == 0 && args->profilerMode) {
    int ch = blockIdx.x;
    uint64_t wc = args->getProfilerCounters()[ch];
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

__device__ __forceinline__ void ncclSymkProfilerStop(struct ncclSymkDevWorkArgs const* args) {
  if (threadIdx.x == 0 && args->profilerMode) {
    int ch = blockIdx.x;
    uint64_t wc = args->getProfilerCounters()[ch];
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

__device__ __forceinline__ void ncclSymkProfilerPhase(struct ncclSymkDevWorkArgs const* args, int phaseId) {
  if (threadIdx.x == 0 && (args->profilerMode & ncclDevProfilerModeKernelPhase)) {
    int ch = blockIdx.x;
    uint64_t wc = args->getProfilerCounters()[ch];
    int slot = wc % MAX_PROFILER_EVENTS_PER_CHANNEL;
    args->kcomm.workPhases[ch].data[slot].timestamps[phaseId] = ncclSymkGlobaltimer();
  }
}

// ncclSymkRun_* entrypoints take a leading compile-time `bool EnableProfiler`: false
// emits no instrumentation (byte-identical to the base kernels), true emits the phase
// stamp call sites, which then check profilerMode at runtime. The host picks the variant
// (ncclSymkKernelList vs ncclSymkKernelListProfile).
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLL_R(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_AGxLLMC_R(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLD_AGxST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxLDMC_AGxSTMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_AllReduce_RSxTmaLD_AGxTmaST(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LL(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_LLMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_ST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_STMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaST(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_TmaSTMC(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LL(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LD(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_LDMC(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_TmaLD(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLD(struct ncclSymkDevWorkArgs const* args);
template <bool EnableProfiler, template <typename> typename Red, typename T>
__device__ __forceinline__ void ncclSymkRun_ReduceScatter_RailA2A_LsaLDMC(struct ncclSymkDevWorkArgs const* args);

template <bool EnableProfiler>
__device__ __forceinline__ void ncclSymkRun_AllGather_RailRing_LsaSTMC(struct ncclSymkDevWorkArgs const* args);
#endif
