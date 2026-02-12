/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"
#include "nccl_device/gin_barrier.h"

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
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::~ncclGinBarrierSession() {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  this->coop.sync();

  int nSignals = this->team.nRanks - 1;
  if (fence == ncclGinFenceLevel::Release) {
    nSignals = this->team.nRanks;
  }

  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < nSignals; i += this->coop.size()) {
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

    // Wait for GIN signal
    this->net.waitSignal(ncclCoopThread(), this->signal + peer, waitVal, 32, nccl::utility::acquireOrderOf(ord));
  }
  this->coop.sync();
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
