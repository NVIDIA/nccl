/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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
  if (comm.ginConnectionsRailed) {
    return utility::idivFast32(worldRank, comm.lsaSize, comm.lsaSize_rcp32);
  } else {
    return worldRank;
  }
}

// Multi-segment put/get helpers
NCCL_DEVICE_INLINE void findSegmentFromWindow(ncclWindow_t win, size_t offset, int* outSeg, size_t* outSegOffset) {
  int seg = 0;
  size_t segOffset = offset;
  size_t cumulative = 0;
  for (int i = 0; i < win->numSegments; i++) {
    struct ncclSegmentWindow const& w = win->ginMultiSegmentWins[i];
    if (cumulative + w.segmentSize > offset) {
      seg = i;
      segOffset = offset - cumulative;
      break;
    }
    cumulative += w.segmentSize;
  }
  *outSeg = seg;
  *outSegOffset = segOffset;
}

NCCL_DEVICE_INLINE size_t getSegmentChunkSize(size_t srcRemaining, size_t dstRemaining, size_t remaining) {
  size_t chunk = srcRemaining;
  if (dstRemaining < chunk) chunk = dstRemaining;
  if (remaining < chunk) chunk = remaining;
  return chunk;
}

NCCL_DEVICE_INLINE void advanceSegmentCursor(
    int* seg, size_t* segOffset, size_t chunkSize, size_t segmentSize) {
  *segOffset += chunkSize;
  if (*segOffset >= segmentSize) {
    (*seg)++;
    *segOffset = 0;
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
NCCL_DEVICE_INLINE ncclGin_BackendMask<beMask>::ncclGin_BackendMask(
    ncclDevComm const& comm, int contextIndex, ncclGinResourceSharingMode resourceSharingMode_):
  comm(comm), resourceSharingMode(resourceSharingMode_) {
  ncclGinInitCommon(this, comm, contextIndex);
}

NCCL_DEVICE_INLINE ncclGin_C::ncclGin_C(
    ncclDevComm const& comm, unsigned backendMask, int contextIndex,
    ncclGinResourceSharingMode resourceSharingMode)
  : comm(comm), resourceSharingMode(resourceSharingMode), backendMask(backendMask) {
  ncclGinInitCommon(this, comm, contextIndex);
}

NCCL_DEVICE_INLINE void ncclGin_C_init(
    ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex) {
  ::new (net) ncclGin_C(comm, backendMask, contextIndex, NCCL_GIN_RESOURCE_SHARING_GPU);
}

NCCL_DEVICE_INLINE void ncclGin_C_initWithResourceSharingMode(
    ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex,
    ncclGinResourceSharingMode resourceSharingMode) {
  ::new (net) ncclGin_C(comm, backendMask, contextIndex, resourceSharingMode);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE ncclGinCtx_M<beMask> ncclGin_BackendMask<beMask>::_makeCtx() const {
  ncclGinCtx_M<beMask> ans;
  ans.backend = (ncclNetDeviceType)_ginBackend;
  if (comm.ginConnectionsRailed) {
    ncclTeam teamRail = ncclTeamRail(comm);
    ans.rank = teamRail.rank;
    ans.nRanks = teamRail.nRanks;
  } else {
    ans.rank = comm.rank;
    ans.nRanks = comm.nRanks;
  }
  ans.handle = _ginHandle;
  ans.contextId = contextId;
  ans.resourceSharingMode = (uint8_t)this->resourceSharingMode;
  return ans;
}

NCCL_DEVICE_INLINE ncclGinCtx ncclGin_C_makeCtx(ncclGin_C* net) {
  ncclGinCtx ans;
  ans.backendMask = net->backendMask;
  ans.backend = (ncclNetDeviceType)net->_ginBackend;
  if (net->comm.ginConnectionsRailed) {
    ncclTeam teamRail = ncclTeamRail(net->comm);
    ans.rank = teamRail.rank;
    ans.nRanks = teamRail.nRanks;
  } else {
    ans.rank = net->comm.rank;
    ans.nRanks = net->comm.nRanks;
  }
  ans.handle = net->_ginHandle;
  ans.contextId = net->contextId;
  ans.resourceSharingMode = (uint8_t)net->resourceSharingMode;
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
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_SignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  desc.isStrong = net.comm.ginStrongLegacySignals;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_SignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_SignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_StrongSignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  desc.isStrong = true;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_StrongSignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_StrongSignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_WeakSignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  desc.isStrong = false;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_WeakSignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_WeakSignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_SignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  desc.isStrong = net.comm.ginStrongLegacySignals;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_SignalAdd) { return ncclGinSignalAdd; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_SignalAdd arg) { return arg.value; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_StrongSignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  desc.isStrong = true;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_StrongSignalAdd) { return ncclGinSignalAdd; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_StrongSignalAdd arg) { return arg.value; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_WeakSignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  desc.indexedSignal.signalId = arg.signal;
  desc.isStrong = false;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_WeakSignalAdd) { return ncclGinSignalAdd; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_WeakSignalAdd arg) { return arg.value; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_VASignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  desc.isStrong = net.comm.ginStrongLegacySignals;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_VASignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_VASignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_StrongVASignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  desc.isStrong = true;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_StrongVASignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_StrongVASignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_WeakVASignalInc arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  desc.isStrong = false;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_WeakVASignalInc) { return ncclGinSignalInc; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_WeakVASignalInc) { return 1; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_VASignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  desc.isStrong = net.comm.ginStrongLegacySignals;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_VASignalAdd) { return ncclGinSignalAdd; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_VASignalAdd arg) { return arg.value; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_StrongVASignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  desc.isStrong = true;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_StrongVASignalAdd) { return ncclGinSignalAdd; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_StrongVASignalAdd arg) { return arg.value; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const& net, ncclGin_WeakVASignalAdd arg) {
  ncclGinSignalDescriptor desc{};
  desc.type = NCCL_GIN_SIGNAL_TYPE_VA;
  desc.vaSignal.signalWindow = nccl::gin::internal::getGinWindow(arg.signalWindow, net.connectionId);
  desc.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(arg.signalWindow, arg.signalOffset);
  desc.vaSignal.ncclWindow = arg.signalWindow;
  desc.isStrong = false;
  return desc;
}
NCCL_DEVICE_INLINE constexpr ncclGinSignalOp_t ncclGin_getSignalOp(ncclGin_WeakVASignalAdd) { return ncclGinSignalAdd; }
NCCL_DEVICE_INLINE constexpr uint64_t ncclGin_getSignalOpArg(ncclGin_WeakVASignalAdd arg) { return arg.value; }
#endif


////////////////////////////////////////////////////////////////////////////////
// ncclGin counter helpers:

#if NCCL_CHECK_CUDACC
template<typename LocalAction>
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(LocalAction) { return false; }
template<typename LocalAction>
NCCL_DEVICE_INLINE constexpr ncclGinCounter_t ncclGin_getCounterId(ncclGin const&, LocalAction) { return -1u; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(ncclGin_CounterInc) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinCounter_t ncclGin_getCounterId(ncclGin const& net, ncclGin_CounterInc arg) { return arg.counter; }
#endif

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr bool ncclGin_isCounter(ncclGin_WeakCounterInc) { return true; }
NCCL_DEVICE_INLINE constexpr ncclGinCounter_t ncclGin_getCounterId(ncclGin const& net, ncclGin_WeakCounterInc arg) { return arg.counter; }
#endif

////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
// ncclGin bufType helpers:

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE constexpr bool ncclGin_isDeviceOnly(ncclGin_SegmentDevice) { return true; }
NCCL_DEVICE_INLINE constexpr bool ncclGin_isDeviceOnly(ncclGin_SegmentMixed) { return false; }
NCCL_DEVICE_INLINE constexpr bool ncclGin_isDeviceOnly(ncclGin_SegmentHostNuma) { return false; }
#endif
////////////////////////////////////////////////////////////////////////////////

#if NCCL_CHECK_CUDACC
NCCL_DEVICE_INLINE void ncclGinPut_v2(
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
    signal.indexedSignal.signalId = signalId;
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
      counterId,
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
  ncclGinPut_v2(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, isSignal, signalId,
               signalOp, signalOpArg, isCounter, counterId, coop, isDescriptor, descriptor,
               givenRelease, requiredRelease, ncclGinOptFlagsDefault);
}

template<unsigned beMask>
template<
  typename RemoteAction, // one of ncclGin_{None|SignalInc}
  typename LocalAction, // one of ncclGin_{None|CounterInc}
  typename Coop,
  typename DescriptorSmem,
  typename SegmentType // one of ncclGin_{SegmentDevice|SegmentMixed|SegmentHostNuma}
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(
    ncclTeam team, int peer,
    ncclWindow_t dstWin, size_t dstOffset,
    ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
    uint32_t optFlags, SegmentType bufType
  ) const {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  ncclGinCtx_M<beMask> ctx = this->_makeCtx();
  coop.sync();
  if (coop.thread_rank() == 0) {
    if (ncclGin_isDeviceOnly(bufType)) {
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
    } else {
      if (srcWin->numSegments == 1 && dstWin->numSegments == 1) {
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
            cuda::thread_scope_system, // for safety, escalate to system regardless of what the user requested
            givenRelease,
            optFlags
            );
      } else {
        // Multi-segment case. The puts are chunked to handle multiple registration entries and src/dst windows that potentially have a different number of segments
        int srcSeg;
        size_t srcSegOffset;
        nccl::gin::internal::findSegmentFromWindow(srcWin, srcOffset, &srcSeg, &srcSegOffset);
        int dstSeg;
        size_t dstSegOffset;
        nccl::gin::internal::findSegmentFromWindow(dstWin, dstOffset, &dstSeg, &dstSegOffset);
        bool doneSysmemFence = false;
        size_t remaining = bytes;
        cuda::thread_scope localRequiredRelease = requiredRelease;

        while (remaining > 0) {
          struct ncclSegmentWindow const& dstSegmentWindow = dstWin->ginMultiSegmentWins[dstSeg];
          struct ncclSegmentWindow const& srcSegmentWindow = srcWin->ginMultiSegmentWins[srcSeg];
          if (!doneSysmemFence && srcSegmentWindow.memType == CU_MEM_LOCATION_TYPE_HOST_NUMA) {
            // for safety, escalate to system regardless of what the user requested
            localRequiredRelease = cuda::thread_scope_system;
            doneSysmemFence = true;
          }
          const size_t srcRemaining = srcSegmentWindow.segmentSize - srcSegOffset;
          const size_t dstRemaining = dstSegmentWindow.segmentSize - dstSegOffset;
          const size_t putSize = nccl::gin::internal::getSegmentChunkSize(srcRemaining, dstRemaining, remaining);
          const bool isLastPut = (remaining == putSize);
          ncclGinCall<ncclGinApi_Put>(ctx,
              ncclCoopThread(), teamRankToGinRank(this->comm, team, peer), /*hasWins=*/true,
              loadConst(&dstWin->ginMultiSegmentWins[dstSeg].ginWins[this->connectionId]),
              dstSegOffset,
              loadConst(&srcWin->ginMultiSegmentWins[srcSeg].ginWins[this->connectionId]),
              srcSegOffset, putSize,
              isLastPut ? ncclGin_getSignalDescriptor(*this, remoteAction) : ncclGin_getSignalDescriptor(*this, ncclGin_None{}),
              ncclGin_getSignalOp(remoteAction),
              ncclGin_getSignalOpArg(remoteAction),
              isLastPut ? ncclGin_isCounter(localAction) : false,
              ncclGin_getCounterId(*this, localAction),
              ncclGin_isDescriptor(descriptor),
              ncclGin_getDescriptor(descriptor),
              localRequiredRelease,
              givenRelease,
              optFlags
              );
          remaining -= putSize;
          nccl::gin::internal::advanceSegmentCursor(&srcSeg, &srcSegOffset, putSize, srcSegmentWindow.segmentSize);
          nccl::gin::internal::advanceSegmentCursor(&dstSeg, &dstSegOffset, putSize, dstSegmentWindow.segmentSize);
          localRequiredRelease = cuda::thread_scope_thread;
        }
      }
    }
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
  typename DescriptorSmem,
  typename SegmentType
>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::put(
    ncclTeam team, int peer,
    ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction, LocalAction localAction,
    Coop coop,
    DescriptorSmem descriptor,
    cuda::thread_scope givenRelease,
    cuda::thread_scope requiredRelease,
    uint32_t optFlags, SegmentType bufType
  ) const {
  this->put(
    team, peer, dstElts.window, dstElts.offset, srcElts.window, srcElts.offset, nElts*sizeof(T),
    remoteAction, localAction, coop, descriptor, givenRelease, requiredRelease, optFlags, bufType
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

NCCL_DEVICE_INLINE void ncclGinPutValue_v2(
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
      signal.indexedSignal.signalId = signalId;
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
  ncclGinPutValue_v2(net, team, peer, dstWin, dstOffset, value, size, isSignal, signalId, signalOp,
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
  static_assert(sizeof(T) <= sizeof(uint64_t), "Required: T must fit into 64 bits");
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

NCCL_DEVICE_INLINE void ncclGinSignal_v2(
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
    signal.indexedSignal.signalId = signalId;
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
  ncclGinSignal_v2(net, team, peer, isSignal, signalId, signalOp, signalOpArg, coop, isDescriptor,
                  descriptor, givenRelease, requiredRelease, ncclGinOptFlagsDefault);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
template<typename Coop, typename DescriptorSmem>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::flush(Coop coop, cuda::memory_order ord,
                                                           DescriptorSmem descriptor) const {
  coop.sync();
  ncclGinCall<ncclGinApi_Flush>(this->_makeCtx(), coop,
                                ncclGin_isDescriptor(descriptor), ncclGin_getDescriptor(descriptor),
                                ord, this->comm.abortFlag);
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinFlush(
    ncclGin_C* net,
    ncclCoopAny coop,
    cuda::memory_order ord
  ) {
  coop.sync();
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_Flush>(ctx, coop, /*hasDescriptor=*/false, /*descriptor=*/nullptr,
                                ord, net->comm.abortFlag);
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
    auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(this->_makeCtx(), counter);
    least += ctr.offset;
    uint64_t got;
    NVCC_PRAGMA_UNROLL_DISABLED
    do got = cuda::atomic_ref<uint64_t>{*ctr.ptr}.load(ord);
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
    auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, counter);
    least += ctr.offset;
    uint64_t got;
    NVCC_PRAGMA_UNROLL_DISABLED
    do got = cuda::atomic_ref<uint64_t>{*ctr.ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, got, bits) && !testAbort(net->comm.abortFlag, steps));
  }
  coop.sync();
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE uint64_t ncclGin_BackendMask<beMask>::readCounter(ncclGinCounter_t counter, int bits, cuda::memory_order ord) const {
  auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(this->_makeCtx(), counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  uint64_t raw = cuda::atomic_ref<uint64_t>{*ctr.ptr}.load(ord);
  return (raw - ctr.offset) & mask;
}

NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(
    ncclGin_C* net,
    ncclGinCounter_t counter,
    int bits,
    cuda::memory_order ord
  ) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, counter);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  uint64_t raw = cuda::atomic_ref<uint64_t>{*ctr.ptr}.load(ord);
  return (raw - ctr.offset) & mask;
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
  auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  uint64_t raw = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
  return (raw - sig.offset) & mask;
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
  auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
  uint64_t mask = uint64_t(-1)>>(64-bits);
  uint64_t raw = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
  return (raw - sig.offset) & mask;
}
#endif

#if NCCL_CHECK_CUDACC

template<unsigned beMask>
template<typename Coop, typename DescriptorSmem>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::wait(ncclGinRequest_t& request, Coop coop,
                                                          DescriptorSmem descriptor, cuda::memory_order ord) const {
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_Wait>(this->_makeCtx(), request,
        ncclGin_isDescriptor(descriptor), ncclGin_getDescriptor(descriptor), ord, this->comm.abortFlag);
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop, typename DescriptorSmem>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::flushAsync(ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                                                Coop coop, uint32_t optFlags,
                                                                DescriptorSmem descriptor) const {
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCall<ncclGinApi_FlushAsync>(this->_makeCtx(), teamRankToGinRank(this->comm, team, peer), outRequest,
                                       ncclGin_isDescriptor(descriptor), ncclGin_getDescriptor(descriptor),
                                       optFlags);
  }
  coop.sync();
}

template<unsigned beMask>
template<typename Coop, typename DescriptorSmem, typename SegmentType>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::get(
    ncclTeam team, int peer,
    ncclWindow_t remoteWnd, size_t remoteOffset,
    ncclWindow_t localWnd, size_t localOffset,
    size_t bytes, Coop coop,
    DescriptorSmem descriptor,
    uint32_t optFlags, SegmentType bufType) const {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  ncclGinCtx_M<beMask> ctx = this->_makeCtx();
  coop.sync();
  if (ncclGin_isDeviceOnly(bufType)) {
    ncclGinCall<ncclGinApi_Get>(ctx, coop,
        teamRankToGinRank(this->comm, team, peer),
        loadConst(&remoteWnd->ginWins[this->connectionId]),
        4096*size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
        loadConst(&localWnd->ginWins[this->connectionId]),
        4096*size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes,
        ncclGin_isDescriptor(descriptor),
        ncclGin_getDescriptor(descriptor),
        optFlags);
  } else {
    if (remoteWnd->numSegments == 1 && localWnd->numSegments == 1) {
      ncclGinCall<ncclGinApi_Get>(ctx, coop,
          teamRankToGinRank(this->comm, team, peer),
          loadConst(&remoteWnd->ginWins[this->connectionId]),
          4096*size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
          loadConst(&localWnd->ginWins[this->connectionId]),
          4096*size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes,
          ncclGin_isDescriptor(descriptor),
          ncclGin_getDescriptor(descriptor),
          optFlags);
    } else {
      // Multi-segment case: chunk the get across separate per-segment registrations
      int remoteSeg, localSeg;
      size_t remoteSegOffset, localSegOffset;
      nccl::gin::internal::findSegmentFromWindow(remoteWnd, remoteOffset, &remoteSeg, &remoteSegOffset);
      nccl::gin::internal::findSegmentFromWindow(localWnd, localOffset, &localSeg, &localSegOffset);
      size_t remaining = bytes;
      while (remaining > 0) {
        struct ncclSegmentWindow const& remoteSegmentWindow = remoteWnd->ginMultiSegmentWins[remoteSeg];
        struct ncclSegmentWindow const& localSegmentWindow = localWnd->ginMultiSegmentWins[localSeg];
        const size_t remoteRemaining = remoteSegmentWindow.segmentSize - remoteSegOffset;
        const size_t localRemaining = localSegmentWindow.segmentSize - localSegOffset;
        const size_t getSize = nccl::gin::internal::getSegmentChunkSize(remoteRemaining, localRemaining, remaining);
        ncclGinCall<ncclGinApi_Get>(ctx, coop,
            teamRankToGinRank(this->comm, team, peer),
            loadConst(&remoteWnd->ginMultiSegmentWins[remoteSeg].ginWins[this->connectionId]),
            remoteSegOffset,
            loadConst(&localWnd->ginMultiSegmentWins[localSeg].ginWins[this->connectionId]),
            localSegOffset, getSize,
            ncclGin_isDescriptor(descriptor),
            ncclGin_getDescriptor(descriptor),
            optFlags);
        remaining -= getSize;
        nccl::gin::internal::advanceSegmentCursor(&remoteSeg, &remoteSegOffset, getSize, remoteSegmentWindow.segmentSize);
        nccl::gin::internal::advanceSegmentCursor(&localSeg, &localSegOffset, getSize, localSegmentWindow.segmentSize);
      }
    }
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinGet(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t remoteWnd, size_t remoteOffset,
  ncclWindow_t localWnd, size_t localOffset, size_t bytes,
  ncclCoopAny coop,
  bool isDescriptor,
  ncclGinDescriptorSmem* descriptor,
  uint32_t optFlags
) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_Get>(ctx, coop,
      teamRankToGinRank(net->comm, team, peer),
      loadConst(&remoteWnd->ginWins[net->connectionId]),
      4096*size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
      loadConst(&localWnd->ginWins[net->connectionId]),
      4096*size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes,
      ncclGin_isDescriptor(descriptor),
      ncclGin_getDescriptor(descriptor),
      optFlags);
  coop.sync();
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
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
    least = least + sig.offset;
    uint64_t got;
    NVCC_PRAGMA_UNROLL_DISABLED
    do got = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
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
    NVCC_PRAGMA_UNROLL_DISABLED
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
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
    least = least + sig.offset;
    uint64_t got;
    NVCC_PRAGMA_UNROLL_DISABLED
    do got = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
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
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
    uint64_t least = this->_signalShadows[signal] + sig.offset;
    uint64_t got;
    NVCC_PRAGMA_UNROLL_DISABLED
    do got = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
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
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(this->_makeCtx(), signal);
    uint64_t offset = sig.offset;
    uint64_t least = before64 + leastDelta + offset;
    NVCC_PRAGMA_UNROLL_DISABLED
    do after64 = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, after64, bits) && !testAbort(this->comm.abortFlag, steps));
    // Convert NIC value back to logical space for shadow
    after64 = after64 - offset;
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
  ncclGinCall<ncclGinApi_ResetCounter>(this->_makeCtx(), counter);
}

NCCL_DEVICE_INLINE void ncclGinResetCounter(
    ncclGin_C* net,
    ncclGinCounter_t counter
  ) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_ResetCounter>(ctx, counter);
}
#endif

#if NCCL_CHECK_CUDACC
template<unsigned beMask>
NCCL_DEVICE_INLINE void ncclGin_BackendMask<beMask>::resetSignal(ncclGinSignal_t signal) const {
  ncclGinSignalDescriptor signalDesc;
  signalDesc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  signalDesc.indexedSignal.signalId = signal;
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
  signalDesc.indexedSignal.signalId = signal;
  ncclGinCall<ncclGinApi_ResetSignal>(ctx, signalDesc);
  net->_signalShadows[signal] = 0;
}
#endif

#endif // _NCCL_DEVICE_GIN_SESSION__FUNCS_H_
