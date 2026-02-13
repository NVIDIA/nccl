/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  this->signal = handle.signal0 + barrierIndex * team.nRanks;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagRail, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamRail(net.comm), net.comm.railGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeamTagWorld, uint32_t barrierIndex
  ):
  ncclGinBarrierSession(coop, net, ncclTeamWorld(net.comm), net.comm.worldGinBarrier, barrierIndex) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
template<typename Fn>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession_internal<Coop>::syncInternal(Coop, cuda::memory_order ord,
                                                                      ncclGinFenceLevel fence, Fn const& fn) {
  ncclResult_t ret = ncclSuccess;
  this->coop.sync();
  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
    // Use a rotating pattern to avoid hot spots
    int peer = 1 + this->team.rank + i;
    if (this->team.nRanks <= peer) peer -= this->team.nRanks;

    // Initiate signal
    this->net.signal(
      this->team, peer, ncclGin_SignalInc{this->signal + this->team.rank}, ncclCoopThread(), ncclGin_None(),
      nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed
        ? cuda::thread_scope_thread
        : cuda::thread_scope_system
    );

    // Load and update barrier state in memory. The load/store should be covered by the GIN signal latency.
    uint32_t* shadowPtr = (uint32_t*)this->net.getSignalShadowPtr(this->signal + peer);
    int waitVal = ++*shadowPtr;

    ret = fn(this->signal + peer, waitVal);
    if (ret != ncclSuccess) {
      break;
    }
  }
  this->coop.sync();
  return ret;
}
#endif


#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  (void)syncInternal(coop, ord, fence, [&]__device__(ncclGinSignal_t signal, int waitVal) {
        this->net.waitSignal(ncclCoopThread(), signal, waitVal, 32, nccl::utility::acquireOrderOf(ord));
        return ncclSuccess;
      }
    );
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclResult_t ncclGinBarrierSession<Coop>::sync(
    Coop coop, cuda::memory_order ord, ncclGinFenceLevel fence, uint64_t timeoutCycles) {
  return syncInternal(coop, ord, fence, [&]__device__(ncclGinSignal_t signal, int waitVal) {
        // Wait for GIN signal with timeout
        uint64_t startCycle = clock64();
        while (true) {
          uint64_t got = this->net.readSignal(signal, 32, nccl::utility::acquireOrderOf(ord));
          if (nccl::utility::rollingLessEq(static_cast<uint64_t>(waitVal), got, 32)) {
            return ncclSuccess;
          }
          if (clock64() - startCycle >= timeoutCycles) {
            return ncclTimeout;
          }
        }
      }
    );
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
