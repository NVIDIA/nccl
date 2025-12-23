/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
#include "gin_barrier__types.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclGinBarrierSession<Coop>::ncclGinBarrierSession(
    Coop coop, ncclGin net, ncclTeam team, ncclGinBarrierHandle handle, uint32_t barrierIndex
  ):
  ncclGinBarrierSession_internal<Coop>{coop, net, team, handle, (int)barrierIndex} {
  uint32_t* epochs = (uint32_t*)ncclGetResourceBufferLocalPointer(net.comm, handle.bufHandle);
  this->epoch = epochs[barrierIndex*NCCL_GIN_MAX_CONTEXTS + net.contextId];
  this->signal = handle.signal0 + barrierIndex;
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
  if (this->coop.thread_rank() == 0) {
    uint32_t* epochs = (uint32_t*)ncclGetResourceBufferLocalPointer(this->net.comm, this->handle.bufHandle);
    epochs[this->index*NCCL_GIN_MAX_CONTEXTS + this->net.contextId] = this->epoch;
  }
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGinBarrierSession<Coop>::sync(Coop, cuda::memory_order ord, ncclGinFenceLevel fence) {
  this->coop.sync();
  #pragma unroll 1
  for (int i=this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
    int peer = 1 + this->team.rank + i;
    if (this->team.nRanks <= peer) peer -= this->team.nRanks;
    this->net.signal(
      this->team, peer, ncclGin_SignalInc{this->signal}, ncclCoopThread(), ncclGin_None(),
      nccl::utility::releaseOrderOf(ord) != cuda::memory_order_relaxed
        ? cuda::thread_scope_thread
        : cuda::thread_scope_system
    );
  }
  this->epoch += this->team.nRanks-1;
  if (this->coop.thread_rank() == 0) {
    this->net.waitSignal(ncclCoopThread(), this->signal, this->epoch, 32, nccl::utility::acquireOrderOf(ord));
  }
  this->coop.sync();
}
#endif

#endif // _NCCL_DEVICE_GIN_BARRIER__FUNCS_H_
