/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
#define _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
#include "gin__types.h"
#include "ptr__types.h"
#if __CUDACC__
#include "nccl_device/gin/gin_device_api.h"
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE ncclGin_BackendMask<beMask>::ncclGin_BackendMask(ncclDevComm const& comm, int contextIndex):
  comm(comm) {
  this->nContexts = comm.ginContextCount;

  static_assert(NCCL_GIN_MAX_CONTEXTS == 4, "Required for following modulo hack to work.");
  // this->contextId = contextIndex % comm.ginContextCount;
  this->contextId = comm.ginContextCount == 3
    ? uint32_t(contextIndex)%3 // 3 is only non power of 2
    : contextIndex & (comm.ginContextCount-1); // powers of 2

  this->_ginBackend = comm.ginTypes[this->contextId];
  this->_ginHandle = comm.ginHandles[this->contextId];
  this->_signalShadows = comm.ginSignalShadows + this->contextId*comm.ginSignalCount;
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE ncclGinCtx_M<beMask> ncclGin_BackendMask<beMask>::_makeCtx() const {
  ncclGinCtx_M<beMask> ans;
  ans.backend = (ncclNetDeviceType)_ginBackend;
  ans.rank = comm.rank;
  ans.nRanks = comm.nRanks;
  ans.handle = _ginHandle;
  return ans;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclGin descriptor helpers:

#if __CUDACC__
template<typename Descriptor>
NCCL_DEVICE_INLINE constexpr bool ncclGin_isDescriptor(Descriptor) { return false; }
template<typename Descriptor>
NCCL_DEVICE_INLINE constexpr ncclGinDescriptorSmem* ncclGin_getDescriptor(Descriptor) { return nullptr; }

NCCL_DEVICE_INLINE constexpr bool ncclGin_isDescriptor(ncclGin_DescriptorSmem) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinDescriptorSmem* ncclGin_getDescriptor(ncclGin_DescriptorSmem arg) { return arg.descriptor; }
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclGin signal helpers:

#if __CUDACC__
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr bool ncclGin_isSignal(RemoteAction) { return false; }
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getSignalId(ncclGin const&, RemoteAction) { return -1u; }
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(RemoteAction) { return (ncclGinSignalOp_t)0; }
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(RemoteAction) { return 0; }
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE constexpr bool ncclGin_isSignal(ncclGin_SignalInc) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getSignalId(
    ncclGin const& net, ncclGin_SignalInc arg
  ) {
  return net.comm.ginSignalBase + arg.signal;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_SignalInc arg) {
  return ncclGinSignalInc;
}
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_SignalInc) { return 1; }
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE constexpr bool ncclGin_isSignal(ncclGin_SignalAdd) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getSignalId(
    ncclGin const& net, ncclGin_SignalAdd arg
  ) {
  return net.comm.ginSignalBase + arg.signal;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_SignalAdd arg) {
  return ncclGinSignalAdd;
}
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_SignalAdd arg) { return arg.value; }
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclGin counter helpers:

#if __CUDACC__
template<typename LocalAction>
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(LocalAction) { return false; }
template<typename LocalAction>
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getCounterId(ncclGin const&, LocalAction) { return -1u; }
#endif

#if __CUDACC__
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(ncclGin_CounterInc) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getCounterId(ncclGin const& net, ncclGin_CounterInc arg) { return net.comm.ginCounterBase + arg.counter; }
#endif

////////////////////////////////////////////////////////////////////////////////

#if __CUDACC__
template<unsigned beMask>
template<
  typename RemoteAction, // one of ncclGin_{None|SignalInc}
  typename LocalAction, // one of ncclGin_{None|CounterInc}
  typename Coop,
  typename DescriptorSmem
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope requiredRelease,  cuda::thread_scope givenRelease
  ) const {
  using nccl::utility::loadConst;
  ncclGinCtx_M<beMask> ctx = this->_makeCtx();
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Put>(ctx,
      ncclCoopThread(), ncclTeamRankToWorld(this->comm, team, peer), /*hasWins=*/true,
      loadConst(&dstWin->ginWins[this->contextId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      loadConst(&srcWin->ginWins[this->contextId]),
      4096*size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes,
      ncclGin_isSignal(remoteAction),
      ncclGin_getSignalId(*this, remoteAction),
      ncclGin_getSignalOp(remoteAction),
      ncclGin_getSignalOpArg(remoteAction),
      ncclGin_isCounter(localAction),
      ncclGin_getCounterId(*this, localAction),
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      requiredRelease,
      givenRelease
    );
  }
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<
  typename T,
  typename RemoteAction, // one of ncclGin_{None|SignalInc}
  typename LocalAction, // one of ncclGin_{None|CounterInc}
  typename Coop,
  typename DescriptorSmem
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(
    ncclTeam team, int peer,
    ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope requiredRelease,
    cuda::thread_scope givenRelease
  ) const {
  this->put(
    team, peer, dstElts.window, dstElts.offset, srcElts.window, srcElts.offset, nElts*sizeof(T),
    remoteAction, localAction, coop, descriptor, requiredRelease, givenRelease
  );
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<
  typename T,
  typename RemoteAction, // one of ncclGin_{None|SignalInc}
  typename Coop,
  typename DescriptorSmem
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::putValue(
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset, T value,
    RemoteAction remoteAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope requiredRelease,
    cuda::thread_scope givenRelease
  ) const {
  static_assert(sizeof(T) <= 8, "Required: sizeof(T) <= 8");
  using nccl::utility::loadConst;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_PutValue>(this->_makeCtx(),
      ncclCoopThread(), ncclTeamRankToWorld(this->comm, team, peer),
      loadConst(&dstWin->ginWins[this->contextId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      value,
      ncclGin_isSignal(remoteAction),
      ncclGin_getSignalId(*this, remoteAction),
      ncclGin_getSignalOp(remoteAction),
      ncclGin_getSignalOpArg(remoteAction),
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      requiredRelease, givenRelease
    );
  }
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<
  typename T,
  typename RemoteAction, // one of ncclGin_{None|SignalInc}
  typename Coop,
  typename DescriptorSmem
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::putValue(
    ncclTeam team, int peer,
    ncclSymPtr<T> dst, T value,
    RemoteAction remoteAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope requiredRelease,
    cuda::thread_scope givenRelease
  ) const {
  this->putValue(
    team, peer, dst.window, dst.offset, value, remoteAction, coop, descriptor, requiredRelease, givenRelease
  );
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<typename RemoteAction, typename Coop, typename DescriptorSmem>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::signal(
    ncclTeam team, int peer, RemoteAction action, Coop coop, DescriptorSmem descriptor,
    cuda::thread_scope requiredRelease,
    cuda::thread_scope givenRelease
  ) const {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Put>(this->_makeCtx(),
      ncclCoopThread(), ncclTeamRankToWorld(this->comm, team, peer),
      /*hasWins=*/false, nullptr, 0, nullptr, 0, 0,
      ncclGin_isSignal(action),
      ncclGin_getSignalId(*this, action),
      ncclGin_getSignalOp(action),
      ncclGin_getSignalOpArg(action),
      /*hasCounter=*/false, 0,
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      requiredRelease, givenRelease
    );
  }
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::flush(Coop coop, cuda::memory_order ord) const {
  coop.sync();
  ncclGinCall<ncclGinApi_Flush>(this->_makeCtx(), coop, ord);
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitCounter(
    Coop coop, ncclGinCounter_t counter, uint64_t least, int bits, cuda::memory_order ord
  ) const {
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetCounterPtr>(this->_makeCtx(), this->comm.ginCounterBase + counter);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got));
  }
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t ncclGin_BackendMask<beMask>::readCounter(ncclGinCounter_t counter, int bits, cuda::memory_order ord) const {
  uint64_t* ptr = ncclGinCall<ncclGinApi_GetCounterPtr>(this->_makeCtx(), this->comm.ginCounterBase + counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t* ncclGin_BackendMask<beMask>::getSignalShadowPtr(ncclGinSignal_t signal) const {
  return &this->_signalShadows[signal];
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const {
  asm volatile("red.relaxed.cta.add.u64 [%0],%1;" :: "l"(this->_signalShadows + signal), "l"(delta) : "memory");
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t ncclGin_BackendMask<beMask>::readSignal(ncclGinSignal_t signal, int bits, cuda::memory_order ord) const {
  uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits, cuda::memory_order ord) const {
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits));
  }
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignalMeetShadow(Coop coop, ncclGinSignal_t signal, int bits, cuda::memory_order ord) const {
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    uint64_t least = this->_signalShadows[signal];
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits));
  }
  coop.sync();
}
#endif

#if __CUDACC__
template<unsigned beMask>
template<typename Coop, typename Uint>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignalFollowShadow(Coop coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits, cuda::memory_order ord) const {
  coop.sync();
  uint64_t before64 = this->_signalShadows[signal];
  uint64_t after64;
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    #pragma unroll 1
    do after64 = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(before64 + leastDelta, after64, bits));
  }
  if (ncclCoopWithinWarp(coop) && bits <= 32) { // do a single __shfl_sync instead of 2
    uint32_t mask = uint32_t(-1)>>(32-bits);
    after64 = ncclCoopBcast(coop, (uint32_t)after64, 0, /*entrySync=*/false);
    *before = (Uint)(mask & before64);
    *delta = (Uint)(mask & (after64 - before64));
  } else {
    uint64_t mask = uint64_t(-1)>>(64-bits);
    after64 = ncclCoopBcast(coop, after64, 0, /*entrySync=*/false);
    *before = (Uint)(mask & before64);
    *delta = (Uint)(mask & (after64 - before64));
  }
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::resetCounter(ncclGinCounter_t counter) const {
  ncclGinCall<ncclGinApi_ResetCounter>(this->_makeCtx(), this->comm.ginCounterBase + counter);
}
#endif

#if __CUDACC__
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::resetSignal(ncclGinSignal_t signal) const {
  ncclGinCall<ncclGinApi_ResetSignal>(this->_makeCtx(), this->comm.ginSignalBase + signal);
  this->_signalShadows[signal] = 0;
}
#endif

#endif // _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
