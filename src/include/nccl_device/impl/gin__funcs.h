/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
#define _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
#include "gin__types.h"
#include "ptr__types.h"
#if NCCL_CHECK_CUDACC
#include "nccl_device/gin/gin_device_api.h"
#endif
#include <new>

namespace nccl {
namespace gin {
namespace internal {
#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE size_t windowOffsetToGinOffset(ncclWindow_t window, size_t offset) {
  using nccl::utility::loadConst;
  return 4096*size_t(loadConst(&window->ginOffset4K)) + offset;
}

NCCL_DEVICE_INLINE ncclGinWindow_t getGinWindow(ncclWindow_t window, int connectionId) {
  using nccl::utility::loadConst;
  return loadConst(&window->ginWins[connectionId]);
}

NCCL_DEVICE_INLINE int teamRankToGinRank(ncclDevComm const& comm, ncclTeam team, int teamRank) {
  int worldRank = ncclTeamRankToWorld(comm, team, teamRank);
  if (comm.ginIsRailed) {
    return utility::idivFast32(worldRank, comm.lsaSize, comm.lsaSize_rcp32);
  } else {
    return worldRank;
  }
}

#endif // NCCL_CHECK_CUDACC
} // namespace internal
} // namespace gin
} // namespace nccl


#if NCCL_CHECK_CUDACC
// Common initialization helper for GIN backend
template<typename GinType>
NCCL_DEVICE_INLINE void ncclGinInitCommon(GinType* gin, ncclDevComm const& comm, int contextIndex) {
  gin->nConnections = comm.ginConnectionCount;
  contextIndex += comm.ginContextBase;

  static_assert(NCCL_GIN_MAX_CONNECTIONS == 4, "Required for following modulo hack to work.");
  // this->connectionId = contextIndex % comm.ginConnectionCount;
  gin->connectionId = comm.ginConnectionCount == 3
    ? uint32_t(contextIndex)%3 // 3 is only non power of 2
    : contextIndex & (comm.ginConnectionCount-1); // powers of 2
  // gin->contextId = contextIndex / comm.ginConnectionCount;
  gin->contextId = comm.ginConnectionCount == 3
    ? uint32_t(contextIndex)/3 // 3 is only non power of 2
    : contextIndex >> (comm.ginConnectionCount==4 ? 2 : comm.ginConnectionCount-1); // powers of 2

  gin->_ginBackend = comm.ginNetDeviceTypes[gin->connectionId];
  gin->_ginHandle = comm.ginHandles[gin->connectionId];
  gin->_signalShadows = comm.ginSignalShadows + contextIndex * comm.ginSignalCount;
}

template<unsigned beMask>
NCCL_DEVICE_INLINE ncclGin_BackendMask<beMask>::ncclGin_BackendMask(ncclDevComm const& comm, int contextIndex):
  comm(comm) {
  ncclGinInitCommon(this, comm, contextIndex);
}

NCCL_DEVICE_INLINE ncclGin_C::ncclGin_C(
    ncclDevComm const& comm, unsigned backendMask, int contextIndex)
  : comm(comm), backendMask(backendMask) {
  ncclGinInitCommon(this, comm, contextIndex);
}

NCCL_DEVICE_INLINE void ncclGin_C_init(
    ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex) {
  ::new (net) ncclGin_C(comm, backendMask, contextIndex);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE ncclGinCtx_M<beMask> ncclGin_BackendMask<beMask>::_makeCtx() const {
  ncclGinCtx_M<beMask> ans;
  ans.backend = (ncclNetDeviceType)_ginBackend;
  if (comm.ginIsRailed) {
    ncclTeam teamRail = ncclTeamRail(comm);
    ans.rank = teamRail.rank;
    ans.nRanks = teamRail.nRanks;
  } else {
    ans.rank = comm.rank;
    ans.nRanks = comm.nRanks;
  }
  ans.handle = _ginHandle;
  ans.contextId = contextId;
  return ans;
}

NCCL_DEVICE_INLINE ncclGinCtx ncclGin_C_makeCtx(ncclGin_C* net) {
  ncclGinCtx ans;
  ans.backendMask = net->backendMask;
  ans.backend = (ncclNetDeviceType)net->_ginBackend;
  if (net->comm.ginIsRailed) {
    ncclTeam teamRail = ncclTeamRail(net->comm);
    ans.rank = teamRail.rank;
    ans.nRanks = teamRail.nRanks;
  } else {
    ans.rank = net->comm.rank;
    ans.nRanks = net->comm.nRanks;
  }
  ans.handle = net->_ginHandle;
  ans.contextId = net->contextId;
  return ans;
}
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclGin descriptor helpers:

#if NCCL_CHECK_CUDACC
template<typename Descriptor>
NCCL_DEVICE_INLINE constexpr bool ncclGin_isDescriptor(Descriptor) { return false; }
template<typename Descriptor>
NCCL_DEVICE_INLINE constexpr ncclGinDescriptorSmem* ncclGin_getDescriptor(Descriptor) { return nullptr; }

NCCL_DEVICE_INLINE constexpr bool ncclGin_isDescriptor(ncclGin_DescriptorSmem) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinDescriptorSmem* ncclGin_getDescriptor(ncclGin_DescriptorSmem arg) { return arg.descriptor; }
#endif

////////////////////////////////////////////////////////////////////////////////
// ncclGin signal helpers:

#if NCCL_CHECK_CUDACC
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(RemoteAction) { return (ncclGinSignalOp_t)0; }
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(RemoteAction) { return 0; }
template<typename RemoteAction>
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const&, RemoteAction) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  return desc;
}
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_SignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = net.comm.ginSignalBase + arg.signal;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_SignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_SignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_SignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = net.comm.ginSignalBase + arg.signal;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_SignalAdd) {
  return ncclGinSignalAdd;
}
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_SignalAdd arg) { return arg.value; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_VASignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_VASignalInc arg) {
  return ncclGinSignalInc;
}
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_VASignalInc arg) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_VASignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_VASignalAdd arg) {
  return ncclGinSignalAdd;
}
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_VASignalAdd arg) { return arg.value; }
#endif


////////////////////////////////////////////////////////////////////////////////
// ncclGin counter helpers:

#if NCCL_CHECK_CUDACC
template<typename LocalAction>
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(LocalAction) { return false; }
template<typename LocalAction>
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getCounterId(ncclGin const&, LocalAction) { return -1u; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(ncclGin_CounterInc) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinSignal_t ncclGin_getCounterId(ncclGin const& net, ncclGin_CounterInc arg) { return net.comm.ginCounterBase + arg.counter; }
#endif

////////////////////////////////////////////////////////////////////////////////

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void ncclGinPutEx(
    ncclGin_C* net,
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
    bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
    bool isCounter, ncclGinCounter_t counterId,
    ncclCoopAny coop,
    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  coop.sync();
  ncclGinSignalDescriptor signal{};
  signal.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
    signal.indexedSignal.signalId = net->comm.ginSignalBase + signalId;
  }
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Put>(ctx,
      ncclCoopThread(), teamRankToGinRank(net->comm, team, peer), /*hasWins=*/true,
      loadConst(&dstWin->ginWins[net->connectionId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      loadConst(&srcWin->ginWins[net->connectionId]),
      4096*size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes,
      signal,
      signalOp,
      signalOpArg,
      isCounter,
      net->comm.ginCounterBase + counterId,
      isDescriptor,
      descriptor,
      requiredRelease,
      givenRelease,
      optFlags
    );
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinPut(
    ncclGin_C* net,
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
    bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
    bool isCounter, ncclGinCounter_t counterId,
    ncclCoopAny coop,
    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease
  ) {
  ncclGinPutEx(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, isSignal, signalId,
               signalOp, signalOpArg, isCounter, counterId, coop, isDescriptor, descriptor,
               givenRelease, requiredRelease, ncclGinOptFlagsDefault);
}

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
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  ncclGinCtx_M<beMask> ctx = this->_makeCtx();
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Put>(ctx,
      ncclCoopThread(), teamRankToGinRank(this->comm, team, peer), /*hasWins=*/true,
      loadConst(&dstWin->ginWins[this->connectionId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      loadConst(&srcWin->ginWins[this->connectionId]),
      4096*size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes,
      ncclGin_getSignalDescriptor(*this, remoteAction),
      ncclGin_getSignalOp(remoteAction),
      ncclGin_getSignalOpArg(remoteAction),
      ncclGin_isCounter(localAction),
      ncclGin_getCounterId(*this, localAction),
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      requiredRelease,
      givenRelease,
      optFlags
    );
  }
  coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
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
    cuda::thread_scope givenRelease,
    cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  this->put(
    team, peer, dstElts.window, dstElts.offset, srcElts.window, srcElts.offset, nElts*sizeof(T),
    remoteAction, localAction, coop, descriptor, givenRelease, requiredRelease, optFlags
  );
}
#endif

#if NCCL_CHECK_CUDACC
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
    cuda::thread_scope givenRelease,
    cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  static_assert(sizeof(T) <= 8, "Required: sizeof(T) <= 8");
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_PutValue>(this->_makeCtx(),
      ncclCoopThread(), teamRankToGinRank(this->comm, team, peer),
      loadConst(&dstWin->ginWins[this->connectionId]),
      4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
      value,
      ncclGin_getSignalDescriptor(*this, remoteAction),
      ncclGin_getSignalOp(remoteAction),
      ncclGin_getSignalOpArg(remoteAction),
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      requiredRelease, givenRelease, optFlags
    );
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinPutValueEx(
    ncclGin_C* net,
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    uint64_t value, size_t size,
    bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
    ncclCoopAny coop,
    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinSignalDescriptor signal{};
    signal.type = NCCL_GIN_SIGNAL_TYPE_NONE;
    if (isSignal) {
      signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
      signal.indexedSignal.signalId = net->comm.ginSignalBase + signalId;
    }
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    // Dispatch based on size to call with appropriate type
    if (size == 1) {
      ncclGinCall<ncclGinApi_PutValue>(ctx,
        ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        (uint8_t)value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    } else if (size == 2) {
      ncclGinCall<ncclGinApi_PutValue>(ctx,
        ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        (uint16_t)value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    } else if (size == 4) {
      ncclGinCall<ncclGinApi_PutValue>(ctx,
        ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        (uint32_t)value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    } else {
      ncclGinCall<ncclGinApi_PutValue>(ctx,
        ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        loadConst(&dstWin->ginWins[net->connectionId]),
        4096*size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        value,
        signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
    }
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinPutValue(
    ncclGin_C* net,
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    uint64_t value, size_t size,
    bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
    ncclCoopAny coop,
    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease
  ) {
  ncclGinPutValueEx(net, team, peer, dstWin, dstOffset, value, size, isSignal, signalId, signalOp,
                    signalOpArg, coop, isDescriptor, descriptor, givenRelease, requiredRelease,
                    ncclGinOptFlagsDefault);
}
#endif

#if NCCL_CHECK_CUDACC
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
    cuda::thread_scope givenRelease,
    cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  this->putValue(
    team, peer, dst.window, dst.offset, value, remoteAction, coop, descriptor, givenRelease, requiredRelease, optFlags
  );
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename RemoteAction, typename Coop, typename DescriptorSmem>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::signal(
    ncclTeam team, int peer, RemoteAction action, Coop coop, DescriptorSmem descriptor,
    cuda::thread_scope givenRelease,
    cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) const {
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Put>(this->_makeCtx(),
      ncclCoopThread(), teamRankToGinRank(this->comm, team, peer),
      /*hasWins=*/false, nullptr, 0, nullptr, 0, 0,
      ncclGin_getSignalDescriptor(*this, action),
      ncclGin_getSignalOp(action),
      ncclGin_getSignalOpArg(action),
      /*hasCounter=*/false, 0,
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      requiredRelease, givenRelease, optFlags
    );
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinSignalEx(
    ncclGin_C* net,
    ncclTeam team, int peer,
    bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
    ncclCoopAny coop,
    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
    uint32_t optFlags
  ) {
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  ncclGinSignalDescriptor signal{};
  signal.type = NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
    signal.indexedSignal.signalId = net->comm.ginSignalBase + signalId;
  }
  if (coop.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    ncclGinCall<ncclGinApi_Put>(ctx,
      ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
      /*hasWins=*/false, nullptr, 0, nullptr, 0, 0,
      signal,
      signalOp,
      signalOpArg,
      /*hasCounter=*/false, 0,
      isDescriptor,
      descriptor,
      requiredRelease, givenRelease,
      optFlags
    );
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinSignal(
    ncclGin_C* net,
    ncclTeam team, int peer,
    bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
    ncclCoopAny coop,
    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease
  ) {
  ncclGinSignalEx(net, team, peer, isSignal, signalId, signalOp, signalOpArg, coop, isDescriptor,
                  descriptor, givenRelease, requiredRelease, ncclGinOptFlagsDefault);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::flush(Coop coop, cuda::memory_order ord) const {
  coop.sync();
  ncclGinCall<ncclGinApi_Flush>(this->_makeCtx(), coop, ord, this->comm.abortFlag);
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinFlush(
    ncclGin_C* net,
    ncclCoopAny coop,
    cuda::memory_order ord
  ) {
  coop.sync();
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_Flush>(ctx, coop, ord, net->comm.abortFlag);
  coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitCounter(
    Coop coop, ncclGinCounter_t counter, uint64_t least, int bits, cuda::memory_order ord
  ) const {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetCounterPtr>(this->_makeCtx(), this->comm.ginCounterBase + counter);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinWaitCounter(
    ncclGin_C* net,
    ncclCoopAny coop,
    ncclGinCounter_t counter,
    uint64_t least,
    int bits,
    cuda::memory_order ord
  ) {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, net->comm.ginCounterBase + counter);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(net->comm.abortFlag, steps));
  }
  coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t ncclGin_BackendMask<beMask>::readCounter(ncclGinCounter_t counter, int bits, cuda::memory_order ord) const {
  uint64_t* ptr = ncclGinCall<ncclGinApi_GetCounterPtr>(this->_makeCtx(), this->comm.ginCounterBase + counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}

NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(
    ncclGin_C* net,
    ncclGinCounter_t counter,
    int bits,
    cuda::memory_order ord
  ) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  uint64_t* ptr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, net->comm.ginCounterBase + counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t* ncclGin_BackendMask<beMask>::getSignalShadowPtr(ncclGinSignal_t signal) const {
  return &this->_signalShadows[signal];
}

NCCL_DEVICE_INLINE uint64_t* ncclGinGetSignalShadowPtr(
    ncclGin_C* net,
    ncclGinSignal_t signal
  ) {
  return &net->_signalShadows[signal];
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const {
  asm volatile("red.relaxed.cta.add.u64 [%0],%1;" :: "l"(this->_signalShadows + signal), "l"(delta) : "memory");
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t ncclGin_BackendMask<beMask>::readSignal(ncclGinSignal_t signal, int bits, cuda::memory_order ord) const {
  uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}

template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t ncclGin_BackendMask<beMask>::readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits, cuda::memory_order ord) const {
  uint64_t* ptr = (uint64_t*)ncclGetLocalPointer(signalWindow, signalOffset);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}

NCCL_DEVICE_INLINE uint64_t ncclGinReadSignal(
    ncclGin_C* net,
    ncclGinSignal_t signal,
    int bits,
    cuda::memory_order ord
  ) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, net->comm.ginSignalBase + signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignal(Coop coop, ncclGinSignal_t signal, uint64_t least, int bits, cuda::memory_order ord) const {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignal(Coop coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits, cuda::memory_order ord) const {
  using nccl::utility::loadConst;
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = (uint64_t*)ncclGetLocalPointer(signalWindow, signalOffset);
    uint64_t got;
    #pragma unroll 1
    do {
      got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    } while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinWaitSignal(
    ncclGin_C* net,
    ncclCoopAny coop,
    ncclGinSignal_t signal,
    uint64_t least,
    int bits,
    cuda::memory_order ord
  ) {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, net->comm.ginSignalBase + signal);
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(net->comm.abortFlag, steps));
  }
  coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename Coop>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignalMeetShadow(Coop coop, ncclGinSignal_t signal, int bits, cuda::memory_order ord) const {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    uint64_t least = this->_signalShadows[signal];
    uint64_t got;
    #pragma unroll 1
    do got = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(this->comm.abortFlag, steps));
  }
  coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename Coop, typename Uint>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::waitSignalFollowShadow(Coop coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits, cuda::memory_order ord) const {
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coop.sync();
  uint64_t before64 = this->_signalShadows[signal];
  uint64_t after64;
  if (coop.thread_rank() == 0) {
    uint64_t* ptr = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), this->comm.ginSignalBase + signal);
    #pragma unroll 1
    do after64 = cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(before64 + leastDelta, after64, bits) && !testAbort(this->comm.abortFlag, steps));
    this->_signalShadows[signal] = after64;
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

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::resetCounter(ncclGinCounter_t counter) const {
  ncclGinCall<ncclGinApi_ResetCounter>(this->_makeCtx(), this->comm.ginCounterBase + counter);
}

NCCL_DEVICE_INLINE void ncclGinResetCounter(
    ncclGin_C* net,
    ncclGinCounter_t counter
  ) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_ResetCounter>(ctx, net->comm.ginCounterBase + counter);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::resetSignal(ncclGinSignal_t signal) const {
  ncclGinSignalDescriptor signalDesc;
  signalDesc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  signalDesc.indexedSignal.signalId = this->comm.ginSignalBase + signal;
  ncclGinCall<ncclGinApi_ResetSignal>(this->_makeCtx(), signalDesc);
  this->_signalShadows[signal] = 0;
}

template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::resetSignal(ncclWindow_t signalWindow, size_t signalOffset) const {
  ncclGinSignalDescriptor signal;
  signal.type = NCCL_GIN_SIGNAL_TYPE_VA;
  signal.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(signalWindow, this->connectionId);
  signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWindow, signalOffset);
  signal.vaSignal.ncclWindow = signalWindow;
  ncclGinCall<ncclGinApi_ResetSignal>(this->_makeCtx(), signal);
}

NCCL_DEVICE_INLINE void ncclGinResetSignal(
    ncclGin_C* net,
    ncclGinSignal_t signal
  ) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinSignalDescriptor signalDesc{};
  signalDesc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  signalDesc.indexedSignal.signalId = net->comm.ginSignalBase + signal;
  ncclGinCall<ncclGinApi_ResetSignal>(ctx, signalDesc);
  net->_signalShadows[signal] = 0;
}
#endif

#endif // _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
