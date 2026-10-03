/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#define _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
#include "../gin_barrier.h"
#include "core__types.h"
#include "gin__types.h"

struct ncclGinBarrierHandle {
  ncclGinSignal_t signal0;
  ncclDevResourceHandle_t unused;
};

#if __cplusplus
// Internal only, not part of the public device API.
// Signal slots a barrier instance owns, by barrier preference and team size.  The default per-peer barrier
// owns one slot per sender rank; the signal-efficient barrier owns 2 regardless of team size.
NCCL_HOST_DEVICE_INLINE constexpr int ncclGinBarrierSlots(ncclGinBarrierOptions_t barrierOptions, int nRanks) {
  return barrierOptions == NCCL_GIN_BARRIER_SIGNAL_EFFICIENT ? 2 : (nRanks < 2 ? 2 : nRanks);
}
#endif

#ifdef __CUDACC__
// Phase-bit bookkeeping for the two-signal barrier, packed into the shadow words of its two signals:
//   shadow[signal+0]: bit 63    phase of the barrier about to run (0 => slot 0)
//                     bits 31:0 cumulative arrivals expected on slot 0
//   shadow[signal+1]: bits 31:0 cumulative arrivals expected on slot 1
// Arrival counts are 32-bit rolling, matching the width every readSignal/rollingLessEq in the barrier uses.
#define ncclGinBarrierPhaseBit 63

struct ncclGinBarrierState {
  int phase;
  uint32_t waitVal;
};

template <typename Coop>
struct ncclGinBarrierSession_internal {
  Coop coop;
  ncclGin net;
  ncclTeam team;
  ncclGinBarrierHandle handle;
  int index;
  ncclGinSignal_t signal;
  // True when the fence covers every GIN context on the comm.
  bool fenceAllContexts;

  template <bool EnableTimeout>
  NCCL_DEVICE_INLINE ncclResult_t syncInternal(Coop, cuda::memory_order ord, ncclGinFenceLevel fence,
                                               uint64_t timeoutCycles);
  // The two-signal phase-bit barrier, used when the backend asks for NCCL_GIN_BARRIER_SIGNAL_EFFICIENT.
  template <bool EnableTimeout>
  NCCL_DEVICE_INLINE ncclResult_t syncPhase(cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles);
};
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__TYPES_H_
