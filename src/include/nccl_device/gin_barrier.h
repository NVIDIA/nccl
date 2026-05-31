/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER_H_
#define _NCCL_DEVICE_GIN_BARRIER_H_
#include "core.h"
#if defined(NCCL_OS_WINDOWS)
#include "gin_win_stub.h"
#else
#include "gin.h"
#endif

struct ncclGinBarrierHandle;

NCCL_EXTERN_C __host__ ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                                                    ncclGinBarrierHandle_t* outHandle,
                                                                    ncclDevResourceRequirements_t* outReq);

#ifdef __CUDACC__
// Bit-flag enum: Put and Get (and any future flags) are independent bits that compose via
// bitwise OR.
enum ncclGinFenceLevel : uint32_t {
  None = 0,        // Pure synchronization. No drain.
  Put = 1u << 0,  // After the barrier returns, puts issued by other team members
                      // targeting the calling rank prior to the barrier are visible in
                      // the calling rank's memory.
  Get = 1u << 1,  // After the barrier returns, gets issued by the calling rank prior
                      // to the barrier have landed in the calling rank's local memory.
  Relaxed = None,     // Deprecated alias for None; kept for source-level backward compatibility.
};

// Composition operators so callers can write `ncclGinFenceLevel::Put | ncclGinFenceLevel::Get`.
NCCL_HOST_DEVICE_INLINE constexpr ncclGinFenceLevel operator|(ncclGinFenceLevel a, ncclGinFenceLevel b) {
  return static_cast<ncclGinFenceLevel>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
NCCL_HOST_DEVICE_INLINE constexpr ncclGinFenceLevel operator&(ncclGinFenceLevel a, ncclGinFenceLevel b) {
  return static_cast<ncclGinFenceLevel>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

// Pass `ncclGinAllContexts(comm)` to a barrier in place of an `ncclGin` to expand the fence
// across every GIN context on the comm.
struct ncclGinAllContexts {
  ncclDevComm const& comm;
  NCCL_HOST_DEVICE_INLINE constexpr ncclGinAllContexts(ncclDevComm const& comm_) : comm(comm_) {}
};

template <typename Coop>
struct ncclGinBarrierSession_internal;

template <typename Coop>
struct ncclGinBarrierSession : ncclGinBarrierSession_internal<Coop> {
  // Bind the barrier's fence to a single GIN context.
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGin, ncclTeamTagWorld, uint32_t index);

  // Bind the barrier's fence to every GIN context on the comm.
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeam, ncclGinBarrierHandle, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeamTagRail, uint32_t index);
  NCCL_DEVICE_INLINE ncclGinBarrierSession(Coop, ncclGinAllContexts, ncclTeamTagWorld, uint32_t index);

  NCCL_DEVICE_INLINE ~ncclGinBarrierSession();

  ncclGinBarrierSession(ncclGinBarrierSession const&) = delete; // Sessions are not copyable

  NCCL_DEVICE_INLINE void sync(Coop, cuda::memory_order,
                               ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
  NCCL_DEVICE_INLINE ncclResult_t sync(Coop, cuda::memory_order, ncclGinFenceLevel, uint64_t timeoutCycles);
};

// Free-function GIN barrier. Wraps session construct + sync + destruct so callers don't need
// to manage a session object for one-shot barriers.
//
// `gin_or_allCtx` is either an `ncclGin` (single context for both signal and fence) or
// `ncclGinAllContexts(comm)` (signal on context 0; fence iterates every context on the comm).

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeam, ncclGinBarrierHandle, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeamTagRail, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGin, ncclTeamTagWorld, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeam, ncclGinBarrierHandle, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeamTagRail, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
template <typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrier(Coop, ncclGinAllContexts, ncclTeamTagWorld, uint32_t index,
                                       cuda::memory_order = cuda::memory_order_acq_rel,
                                       ncclGinFenceLevel = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER_H_
