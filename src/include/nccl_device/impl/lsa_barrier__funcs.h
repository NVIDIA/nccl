/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#define _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
#include "lsa_barrier__types.h"
#include "comm__types.h"

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeam team,
    ncclLsaBarrierHandle handle, uint32_t index,
    bool multimem, ncclMultimemHandle mmHandle
  ):
  ncclLsaBarrierSession_internal<Coop>{
    coop, comm, team, handle, (int)index,
#if CUDART_VERSION >= 12060
    multimem,
#else // WAR for an issue with ptxas in CTK < 12.6
    /*multimem=*/false,
#endif
    mmHandle, /*epoch=*/0
  } {
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(comm, handle.bufHandle);
  this->epoch = state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index];
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::ncclLsaBarrierSession(
    Coop coop, ncclDevComm const& comm, ncclTeamTagLsa, uint32_t index, bool multimem
  ): ncclLsaBarrierSession(
    coop, comm, ncclTeamLsa(comm), comm.lsaBarrier, index, multimem, comm.lsaMultimem
  ) {
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE ncclLsaBarrierSession<Coop>::~ncclLsaBarrierSession() {
  uint32_t* state = (uint32_t*)ncclGetResourceBufferLocalPointer(this->comm, this->handle.bufHandle);
  if (this->coop.thread_rank() == 0) {
#if __CUDA_ARCH__ == 1200 && CUDART_VERSION < 13000
    // WAR for a compiler issue with CTK < 13.0
    if (this->index == 0)
      state[(this->multimem ? 0 : 1)*this->handle.nBarriers] = this->epoch;
    else
#endif
    state[(this->multimem ? 0 : 1)*this->handle.nBarriers + this->index] = this->epoch;
  }
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::arrive(Coop, cuda::memory_order order) {
  this->coop.sync();
  if (this->multimem) {
  #if __CUDA_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      uint32_t* inbox = this->mcInbox(/*multimem=*/true);
      if (nccl::utility::releaseOrderOf(order) != cuda::memory_order_relaxed) {
        asm volatile("multimem.red.release.sys.add.u32 [%0],1;" :: "l"(inbox));
      } else {
        asm volatile("multimem.red.relaxed.sys.add.u32 [%0],1;" :: "l"(inbox));
      }
    }
  #endif
  } else {
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(peer, this->team.rank));
      inbox.store(this->epoch+1, nccl::utility::releaseOrderOf(order));
    }
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::wait(Coop, cuda::memory_order order) {
  if (this->multimem) {
  #if __CUDA_ARCH__ >= 900
    if (this->coop.thread_rank() == 0) {
      cuda::atomic_ref<uint32_t> inbox(*this->mcInbox(/*multimem=*/false));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        if (got - (this->epoch + this->team.nRanks) <= uint32_t(-1)>>1) break;
      }
      this->epoch += this->team.nRanks;
    }
  #endif
  } else {
    #pragma unroll 1
    for (int i = this->coop.thread_rank(); i < this->team.nRanks-1; i += this->coop.size()) {
      int peer = i + (this->team.rank <= i ? 1 : 0);
      cuda::atomic_ref<uint32_t> inbox(*this->ucInbox(this->team.rank, peer));
      #pragma unroll 1
      while (true) {
        uint32_t got = inbox.load(nccl::utility::acquireOrderOf(order));
        if (got - (this->epoch + 1) <= uint32_t(-1)>>1) break;
      }
    }
    this->epoch += 1;
  }
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop>
NCCL_DEVICE_INLINE void ncclLsaBarrierSession<Coop>::sync(Coop coop, cuda::memory_order order) {
  this->arrive(coop, order);
  this->wait(coop, order);
}
#endif

#endif // _NCCL_DEVICE_MEM_BARRIER__FUNCS_H_
