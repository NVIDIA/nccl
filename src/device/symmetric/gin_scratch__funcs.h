/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SCRATCH__FUNCS_H_
#define _NCCL_DEVICE_GIN_SCRATCH__FUNCS_H_
#include "gin_scratch__types.h"

////////////////////////////////////////////////////////////////////////////////
// ncclGinOutboxSession:

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclGinOutboxSession<Coop, ginBackendMask>::ncclGinOutboxSession(
    Coop coop, ncclGin_BackendMask<ginBackendMask> const& gin, ncclGinOutboxHandle handle, uint32_t index
  ):
  ncclGinOutboxSession_internal<Coop, ginBackendMask>{coop, gin.comm, gin, handle, (int)index} {
  this->state = this->getStatePtr()->unpadded;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclGinOutboxSession<Coop, ginBackendMask>::~ncclGinOutboxSession() {
  if (this->coop.thread_rank() == 0) {
    this->getStatePtr()->unpadded = this->state;
  }
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
template<typename SubCoop>
NCCL_DEVICE_INLINE void ncclGinOutboxSession<Coop, ginBackendMask>::apportion(
    Coop, SubCoop subcoop, bool subcoopIsNonTrivial, int nBufs_log2_next, bool deferSync
  ) {
  int nBufs_log2_cur = this->state.nBufs_log2;
  if (nBufs_log2_cur != nBufs_log2_next) {
    if (subcoopIsNonTrivial) {
      #pragma unroll 1
      for (int i = subcoop.thread_rank(); i < (1<<nBufs_log2_cur); i += subcoop.size()) {
        uint32_t id = this->state.cursor + i;
        ncclGinCounter_t ctr = this->handle.counter0 + (this->block << ncclGinScratchMaxBufs_log2);
        ctr += id & (1<<nBufs_log2_cur)-1;
        uint32_t val = id >> nBufs_log2_cur;
        this->gin.waitCounter(ncclCoopThread(), ctr, val, ncclGinOutboxState::CursorBits);
        this->gin.resetCounter(ctr);
      }
    }
    if (!deferSync) this->coop.sync();
    this->state.nBufs_log2 = nBufs_log2_next;
    this->state.cursor = 0;
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
template<typename SubCoop>
NCCL_DEVICE_INLINE void ncclGinOutboxSession<Coop, ginBackendMask>::waitBufs(SubCoop subcoop, int i0, int n) {
  #pragma unroll 1
  for (int i=subcoop.thread_rank(); i < n; i += subcoop.size()) {
    uint32_t id = this->state.cursor + i0 + i;
    ncclGinCounter_t ctr = this->handle.counter0 + (this->block << ncclGinScratchMaxBufs_log2);
    ctr += id & (1<<this->state.nBufs_log2)-1;
    uint32_t val = id >> this->state.nBufs_log2;
    this->gin.waitCounter(ncclCoopThread(), ctr, val, ncclGinOutboxState::CursorBits);
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGinOutboxSession<Coop, ginBackendMask>::getBuf(int i) const {
  uint32_t id = this->state.cursor + i;
  int nBufs_log2 = this->state.nBufs_log2;
  ncclSymPtr<char> bufs = (ncclSymPtr<char>)(this->getStateSymPtr() + 1);
  return bufs + ((id & (1<<nBufs_log2)-1) << (this->handle.size_log2 - nBufs_log2));
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE  ncclGinScratch_GetBufPtr ncclGinOutboxSession<Coop, ginBackendMask>::make_getBufPtr(int i0) const {
  ncclGinScratch_GetBufPtr ret;
  ret.bufs = (char*)(this->getStatePtr() + 1);
  ret.nBufs_minus_1 = (1<<this->state.nBufs_log2)-1;
  ret.bufSize_log2 = this->handle.size_log2 - this->state.nBufs_log2;
  ret.cursor = this->state.cursor + i0;
  return ret;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclGinCounter_t ncclGinOutboxSession<Coop, ginBackendMask>::getCounter(int i) const {
  uint32_t id = this->state.cursor + i;
  ncclGinCounter_t ctr = this->handle.counter0 + (this->block << ncclGinScratchMaxBufs_log2);
  return ctr + (id & (1<<this->state.nBufs_log2)-1);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE void ncclGinOutboxSession<Coop, ginBackendMask>::advance(Coop, int n) {
  this->state.cursor += n;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// ncclGinInboxA2ASession:

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclGinInboxA2ASession<Coop, ginBackendMask>::ncclGinInboxA2ASession(
    Coop coop, ncclGin_BackendMask<ginBackendMask> const& gin, ncclTeam team, ncclGinInboxA2AHandle handle, uint32_t index
  ):
  ncclGinInboxA2ASession_internal<Coop, ginBackendMask>
    {coop, gin.comm, gin, team, handle, (int)index} {
  this->nPeers = team.nRanks - 1;
  this->state = this->getStatePtr()->unpadded;
  // The `index` to `context` relationship must be 1:1.
  assert(this->state.ginContextId_plus_1 == 0 || this->state.ginContextId_plus_1 == gin.contextId + 1);
  this->state.ginContextId_plus_1 = gin.contextId + 1;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclGinInboxA2ASession<Coop, ginBackendMask>::~ncclGinInboxA2ASession() {
  if (this->coop.thread_rank() == 0) {
    this->getStatePtr()->unpadded = this->state;
  }
  this->coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
template<typename SubCoop>
NCCL_DEVICE_INLINE void ncclGinInboxA2ASession<Coop, ginBackendMask>::apportion(
    Coop coop, SubCoop subcoop, bool subcoopIsNonTrivial, int nBufs_log2_next
  ) {
  int nBufs_log2_cur = this->state.nBufs_log2_plus_1 - 1;
  if (nBufs_log2_cur != nBufs_log2_next) {
    if (subcoopIsNonTrivial) {
      // Send initial C2S's for all bufs for next (+1) phase.
      int nPeers = this->nPeers;
      int nBufs = 1<<nBufs_log2_next;
      uint32_t nBufs_div_nPeers, nBufs_mod_nPeers;
      idivmodFast32(&nBufs_div_nPeers, &nBufs_mod_nPeers, nBufs, nPeers, this->handle.nPeers_rcp32);
      #pragma unroll 1
      for (int step = subcoop.thread_rank(); step < min(nPeers, nBufs); step += subcoop.size()) {
        int credits = nBufs_div_nPeers + (step < (int)nBufs_mod_nPeers ? 1 : 0);
        this->sendC2S(/*phaseDelta=*/+1, step, /*step_lt_nPeers=*/true, credits);
      }

      // Reset all signals of previous (-1) phase. The current phase can have
      // inbound C2S still in flight but the previous cannot because of signal's
      // release semantics combined with fact that we've communicated with all peers.
      this->resetSignals(subcoop, /*phaseDelta=*/-1);
    }
    // Move to next phase
    this->state.nBufs_log2_plus_1 = nBufs_log2_next + 1;
    this->state.phase += 1; // implicitly modulo 4
    this->state.monoRound = 0; // round resets with phase change.
    this->state.monoStep = 0;
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE int ncclGinInboxA2ASession<Coop, ginBackendMask>::getSendPeer(int step, bool step_lt_nPeers) const {
  return this->getPeer(/*sendNotRecv=*/true, step, step_lt_nPeers);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE int ncclGinInboxA2ASession<Coop, ginBackendMask>::getRecvPeer(int step, bool step_lt_nPeers) const {
  return this->getPeer(/*sendNotRecv=*/false, step, step_lt_nPeers);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclSymPtr<char> ncclGinInboxA2ASession<Coop, ginBackendMask>::getBuf(int step) const {
  return this->getBufSymPtr(this->state.monoStep + step);
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE ncclGinScratch_GetBufPtr ncclGinInboxA2ASession<Coop, ginBackendMask>::make_getBufPtr(int step0) const {
  int nBufs_log2 = this->state.nBufs_log2_plus_1 - 1;
  ncclGinScratch_GetBufPtr ret;
  ret.bufs = this->getBufsPtr();
  ret.nBufs_minus_1 = (1<<nBufs_log2)-1;
  ret.bufSize_log2 = this->handle.size_log2 - nBufs_log2;
  ret.cursor = this->state.monoStep + step0;
  return ret;
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
template<typename SubCoop, typename GetPtr, typename GetEltCount, typename GetCompletion>
NCCL_DEVICE_INLINE void ncclGinInboxA2ASession<Coop, ginBackendMask>::postSends(
    SubCoop subcoop, int step0, int nSteps, GetPtr getPtr, GetEltCount getEltCount, GetCompletion getCompletion
  ) {
  #pragma unroll 1
  for (int i=subcoop.thread_rank(); i < nSteps; i += subcoop.size()) {
    int step = step0 + i;
    uint32_t monoStep = this->state.monoStep + step;
    this->waitC2S(step);

    int peer = this->getSendPeer(step, /*step_lt_nPeers=*/true);
    auto srcPtr = getPtr(i, peer);
    int nElts = getEltCount(i, peer);
    this->gin.put(this->team, peer,
      /*dst=*/this->getBufSymPtr(monoStep), /*src=*/(ncclSymPtr<char>)srcPtr,
      /*size=*/nElts*(int)sizeof(decltype(srcPtr)::ElementType),
      ncclGin_SignalInc{this->getR2RSignal(monoStep)},
      getCompletion(i, peer));
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
template<typename SubCoop>
NCCL_DEVICE_INLINE void ncclGinInboxA2ASession<Coop, ginBackendMask>::waitRecvs(
    SubCoop subcoop, int step0, int nSteps
  ) {
  #pragma unroll 1
  for (int i=subcoop.thread_rank(); i < nSteps; i += subcoop.size()) {
    this->waitR2R(this->state.monoStep + step0 + i);
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
template<typename SubCoop>
NCCL_DEVICE_INLINE void ncclGinInboxA2ASession<Coop, ginBackendMask>::finishRecvs(
    SubCoop subcoop, int step0, int nSteps
  ) {
  int nBufs_log2 = this->state.nBufs_log2_plus_1 - 1;
  int nBufs_mod_nPeers = imodFast32(1<<nBufs_log2, this->nPeers, this->handle.nPeers_rcp32);
  #pragma unroll 1
  for (int i=subcoop.thread_rank(); i < nSteps; i += subcoop.size()) {
    // Determine next step that will alias the buffer of this step.
    int step = step0 + i; // guaranteed: step < nPeers
    int nextStep = step + nBufs_mod_nPeers;
    if (this->nPeers <= nextStep) nextStep -= this->nPeers; // modulo for + nBufs_mod_nPeers
    this->sendC2S(/*phaseDelta=*/0, nextStep, /*step_lt_nPeers=*/true, /*credits=*/1);
  }
}
#endif

#if NCCL_CHECK_CUDACC
template<typename Coop, unsigned ginBackendMask>
NCCL_DEVICE_INLINE void ncclGinInboxA2ASession<Coop, ginBackendMask>::endRound(Coop coop) {
  this->state.monoRound += 1;
  this->state.monoStep += this->nPeers;
}
#endif

#endif // _NCCL_DEVICE_GIN_SCRATCH__FUNCS_H_
