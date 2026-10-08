/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_IMPL_GIN_H_
#define _NCCL_DEVICE_WRAPPER_IMPL_GIN_H_

// Internal implementation fragment; include only from nccl_device_wrapper.cu
// after its prerequisite headers.

/* GIN C API definitions for LLVM IR generation. */
NCCL_DEVICE_INLINE ncclGin_C::ncclGin_C(ncclDevComm const& comm, unsigned backendMask, int contextIndex,
                                        ncclGinResourceSharingMode resourceSharingMode)
  : comm(comm), resourceSharingMode(resourceSharingMode), backendMask(backendMask) {
  ncclGinInitCommon(this, comm, contextIndex);
}

NCCL_DEVICE_INLINE void ncclGin_C_init(ncclGin_C* net, unsigned backendMask, ncclDevComm const* comm,
                                       int contextIndex) {
  ::new (net) ncclGin_C(*comm, backendMask, contextIndex, NCCL_GIN_RESOURCE_SHARING_GPU);
}

NCCL_DEVICE_INLINE void ncclGin_C_initWithResourceSharingMode(ncclGin_C* net, unsigned backendMask,
                                                              ncclDevComm const* comm, int contextIndex,
                                                              ncclGinResourceSharingMode resourceSharingMode) {
  ::new (net) ncclGin_C(*comm, backendMask, contextIndex, resourceSharingMode);
}

NCCL_DEVICE_INLINE ncclGinCtx ncclGin_C_makeCtx(ncclGin_C* net) {
  using nccl::utility::idivFast32;
  ncclGinCtx ans;
  ans.backendMask = net->backendMask;
  ans.backend = (ncclNetDeviceType)net->_ginBackend;
  if (net->comm.ginConnectionStride == 1) {
    ans.rank = net->comm.rank;
    ans.nRanks = net->comm.nRanks;
  } else {
    ans.rank = idivFast32(net->comm.rank, net->comm.ginConnectionStride, net->comm.ginConnectionStride_rcp32);
    ans.nRanks = idivFast32(net->comm.nRanks, net->comm.ginConnectionStride, net->comm.ginConnectionStride_rcp32);
  }
  ans.handle = net->_ginHandle;
  ans.contextId = net->contextId;
  ans.resourceSharingMode = (uint8_t)net->resourceSharingMode;
  return ans;
}

NCCL_DEVICE_INLINE void ncclGinPut_C_impl(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset,
  size_t bytes, ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool isCounter,
  ncclGinCounter_t counterId, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease, uint32_t optFlags, bool isDeviceOnly) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    ncclDevComm const& comm = net->comm;
    uint8_t connectionId = net->connectionId;
    if (isDeviceOnly) {
      ncclGinCall<ncclGinApi_Put>(ctx, ncclCoopThread(), teamRankToGinRank(comm, team, peer), /*hasWins=*/true,
                                  nccl::gin::internal::getGinWindow(dstWin, comm.backendIndex, connectionId),
                                  4096 * size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
                                  nccl::gin::internal::getGinWindow(srcWin, comm.backendIndex, connectionId),
                                  4096 * size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes, signal, signalOp,
                                  signalOpArg, isCounter, counterId, isDescriptor, descriptor, requiredRelease,
                                  givenRelease, optFlags);
    } else if (srcWin->numSegments == 1 && dstWin->numSegments == 1) {
      ncclGinCall<ncclGinApi_Put>(
        ctx, ncclCoopThread(), teamRankToGinRank(comm, team, peer), /*hasWins=*/true,
        nccl::gin::internal::getGinWindow(dstWin, comm.backendIndex, connectionId),
        4096 * size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset,
        nccl::gin::internal::getGinWindow(srcWin, comm.backendIndex, connectionId),
        4096 * size_t(loadConst(&srcWin->ginOffset4K)) + srcOffset, bytes, signal, signalOp, signalOpArg, isCounter,
        counterId, isDescriptor, descriptor,
        cuda::thread_scope_system, // for safety, escalate to system regardless of what the user requested
        givenRelease, optFlags);
    } else {
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
          localRequiredRelease = cuda::thread_scope_system;
          doneSysmemFence = true;
        }
        const size_t srcRemaining = srcSegmentWindow.segmentSize - srcSegOffset;
        const size_t dstRemaining = dstSegmentWindow.segmentSize - dstSegOffset;
        const size_t putSize = nccl::gin::internal::getSegmentChunkSize(srcRemaining, dstRemaining, remaining);
        const bool isLastPut = (remaining == putSize);
        // Value initialization sets type to NCCL_GIN_SIGNAL_TYPE_NONE, which is zero.
        ncclGinSignalDescriptor putSignal = isLastPut ? signal : ncclGinSignalDescriptor{};
        ncclGinCall<ncclGinApi_Put>(
          ctx, ncclCoopThread(), teamRankToGinRank(comm, team, peer), /*hasWins=*/true,
          nccl::gin::internal::getSegmentGinWindow(dstWin, dstSeg, dstSegmentWindow, comm.backendIndex, connectionId),
          dstSegOffset,
          nccl::gin::internal::getSegmentGinWindow(srcWin, srcSeg, srcSegmentWindow, comm.backendIndex, connectionId),
          srcSegOffset, putSize, putSignal, signalOp, signalOpArg, isLastPut ? isCounter : false, counterId,
          isDescriptor, descriptor, localRequiredRelease, givenRelease, optFlags);
        remaining -= putSize;
        nccl::gin::internal::advanceSegmentCursor(&srcSeg, &srcSegOffset, putSize, srcSegmentWindow.segmentSize);
        nccl::gin::internal::advanceSegmentCursor(&dstSeg, &dstSegOffset, putSize, dstSegmentWindow.segmentSize);
        localRequiredRelease = cuda::thread_scope_thread;
      }
    }
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinPut(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                   ncclWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal,
                                   ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                   bool isCounter, ncclGinCounter_t counterId, ncclIrCoop const* coop,
                                   bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                   cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = isSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = net->comm.ginStrongLegacySignals;
  }
  ncclGinPut_C_impl(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, signal, signalOp, signalOpArg,
                    isCounter, counterId, coopImpl, isDescriptor, descriptor, givenRelease, requiredRelease,
                    ncclGinOptFlagsDefault, /*isDeviceOnly=*/true);
}

NCCL_DEVICE_INLINE void ncclGinPut_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                      ncclWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal,
                                      ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                      bool isCounter, ncclGinCounter_t counterId, ncclIrCoop const* coop,
                                      bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                      uint32_t optFlags) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = isSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = net->comm.ginStrongLegacySignals;
  }
  ncclGinPut_C_impl(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, signal, signalOp, signalOpArg,
                    isCounter, counterId, coopImpl, isDescriptor, descriptor, givenRelease, requiredRelease, optFlags,
                    /*isDeviceOnly=*/true);
}

NCCL_DEVICE_INLINE void ncclGinPut_v3(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset,
  size_t bytes, ncclGinSignalType signalType, ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId,
  bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool isCounter, ncclGinCounter_t counterId,
  ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
  cuda::thread_scope requiredRelease, uint32_t optFlags, ncclGinSegmentType_t segmentType) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = signalType;
  if (signalType == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = isStrong;
  } else if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    signal.vaSignal.signalWindow =
      nccl::gin::internal::getGinWindow(signalWin, net->comm.backendIndex, net->connectionId);
    signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWin, signalOffset);
    signal.vaSignal.ncclWindow = signalWin;
    signal.isStrong = isStrong;
  }
  ncclGinPut_C_impl(net, team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes, signal, signalOp, signalOpArg,
                    isCounter, counterId, coopImpl, isDescriptor, descriptor, givenRelease, requiredRelease, optFlags,
                    segmentType == ncclGinSegmentTypeDevice);
}

NCCL_DEVICE_INLINE void ncclGinPutValue_C_impl(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin,
                                               size_t dstOffset, uint64_t value, size_t size,
                                               ncclGinSignalDescriptor signal, ncclGinSignalOp_t signalOp,
                                               uint64_t signalOpArg, ncclCoopAny coop, bool isDescriptor,
                                               ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                               cuda::thread_scope requiredRelease, uint32_t optFlags) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    // Convert the runtime C API size into the compile-time type used by the backend.
    switch (size) {
    case 1:
      ncclGinCall<ncclGinApi_PutValue>(
        ctx, ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        nccl::gin::internal::getGinWindow(dstWin, net->comm.backendIndex, net->connectionId),
        4096 * size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset, (uint8_t)value, signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
      break;
    case 2:
      ncclGinCall<ncclGinApi_PutValue>(
        ctx, ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        nccl::gin::internal::getGinWindow(dstWin, net->comm.backendIndex, net->connectionId),
        4096 * size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset, (uint16_t)value, signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
      break;
    case 4:
      ncclGinCall<ncclGinApi_PutValue>(
        ctx, ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        nccl::gin::internal::getGinWindow(dstWin, net->comm.backendIndex, net->connectionId),
        4096 * size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset, (uint32_t)value, signal, signalOp, signalOpArg,
        isDescriptor, descriptor, requiredRelease, givenRelease, optFlags);
      break;
    case 8:
    default:
      ncclGinCall<ncclGinApi_PutValue>(
        ctx, ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
        nccl::gin::internal::getGinWindow(dstWin, net->comm.backendIndex, net->connectionId),
        4096 * size_t(loadConst(&dstWin->ginOffset4K)) + dstOffset, value, signal, signalOp, signalOpArg, isDescriptor,
        descriptor, requiredRelease, givenRelease, optFlags);
      break;
    }
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinPutValue(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                        uint64_t value, size_t size, bool isSignal, ncclGinSignal_t signalId,
                                        ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                        bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                        cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = isSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = net->comm.ginStrongLegacySignals;
  }
  ncclGinPutValue_C_impl(net, team, peer, dstWin, dstOffset, value, size, signal, signalOp, signalOpArg, coopImpl,
                         isDescriptor, descriptor, givenRelease, requiredRelease, ncclGinOptFlagsDefault);
}

NCCL_DEVICE_INLINE void ncclGinPutValue_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin,
                                           size_t dstOffset, uint64_t value, size_t size, bool isSignal,
                                           ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                           ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                           cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                           uint32_t optFlags) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = isSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = net->comm.ginStrongLegacySignals;
  }
  ncclGinPutValue_C_impl(net, team, peer, dstWin, dstOffset, value, size, signal, signalOp, signalOpArg, coopImpl,
                         isDescriptor, descriptor, givenRelease, requiredRelease, optFlags);
}

NCCL_DEVICE_INLINE void ncclGinPutValue_v3(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin,
                                           size_t dstOffset, uint64_t value, size_t size, ncclGinSignalType signalType,
                                           ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId,
                                           bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                           ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                           cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                           uint32_t optFlags) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = signalType;
  if (signalType == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = isStrong;
  } else if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    signal.vaSignal.signalWindow =
      nccl::gin::internal::getGinWindow(signalWin, net->comm.backendIndex, net->connectionId);
    signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWin, signalOffset);
    signal.vaSignal.ncclWindow = signalWin;
    signal.isStrong = isStrong;
  }
  ncclGinPutValue_C_impl(net, team, peer, dstWin, dstOffset, value, size, signal, signalOp, signalOpArg, coopImpl,
                         isDescriptor, descriptor, givenRelease, requiredRelease, optFlags);
}

NCCL_DEVICE_INLINE void ncclGinGet_C_impl(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd,
                                          size_t remoteOffset, ncclWindow_t localWnd, size_t localOffset, size_t bytes,
                                          ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                          uint32_t optFlags, bool isDeviceOnly) {
  using nccl::utility::loadConst;
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclDevComm const& comm = net->comm;
  uint8_t connectionId = net->connectionId;
  if (isDeviceOnly) {
    ncclGinCall<ncclGinApi_Get>(ctx, coop, teamRankToGinRank(comm, team, peer),
                                nccl::gin::internal::getGinWindow(remoteWnd, comm.backendIndex, connectionId),
                                4096 * size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
                                nccl::gin::internal::getGinWindow(localWnd, comm.backendIndex, connectionId),
                                4096 * size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes, isDescriptor,
                                descriptor, optFlags);
  } else if (remoteWnd->numSegments == 1 && localWnd->numSegments == 1) {
    ncclGinCall<ncclGinApi_Get>(ctx, coop, teamRankToGinRank(comm, team, peer),
                                nccl::gin::internal::getGinWindow(remoteWnd, comm.backendIndex, connectionId),
                                4096 * size_t(loadConst(&remoteWnd->ginOffset4K)) + remoteOffset,
                                nccl::gin::internal::getGinWindow(localWnd, comm.backendIndex, connectionId),
                                4096 * size_t(loadConst(&localWnd->ginOffset4K)) + localOffset, bytes, isDescriptor,
                                descriptor, optFlags);
  } else {
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
      ncclGinCall<ncclGinApi_Get>(ctx, coop, teamRankToGinRank(comm, team, peer),
                                  nccl::gin::internal::getSegmentGinWindow(remoteWnd, remoteSeg, remoteSegmentWindow,
                                                                           comm.backendIndex, connectionId),
                                  remoteSegOffset,
                                  nccl::gin::internal::getSegmentGinWindow(localWnd, localSeg, localSegmentWindow,
                                                                           comm.backendIndex, connectionId),
                                  localSegOffset, getSize, isDescriptor, descriptor, optFlags);
      remaining -= getSize;
      nccl::gin::internal::advanceSegmentCursor(&remoteSeg, &remoteSegOffset, getSize, remoteSegmentWindow.segmentSize);
      nccl::gin::internal::advanceSegmentCursor(&localSeg, &localSegOffset, getSize, localSegmentWindow.segmentSize);
    }
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinGet(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd, size_t remoteOffset,
                                   ncclWindow_t localWnd, size_t localOffset, size_t bytes, ncclIrCoop const* coop,
                                   bool isDescriptor, ncclGinDescriptorSmem* descriptor, uint32_t optFlags) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinGet_C_impl(net, team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coopImpl, isDescriptor,
                    descriptor, optFlags, /*isDeviceOnly=*/true);
}

NCCL_DEVICE_INLINE void ncclGinGet_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd,
                                      size_t remoteOffset, ncclWindow_t localWnd, size_t localOffset, size_t bytes,
                                      ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                      uint32_t optFlags, ncclGinSegmentType_t segmentType) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinGet_C_impl(net, team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coopImpl, isDescriptor,
                    descriptor, optFlags, segmentType == ncclGinSegmentTypeDevice);
}

NCCL_DEVICE_INLINE void ncclGinSignal_C_impl(ncclGin_C* net, ncclTeam team, int peer, ncclGinSignalDescriptor signal,
                                             ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
                                             bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                             cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                             uint32_t optFlags) {
  using nccl::gin::internal::teamRankToGinRank;
  coop.sync();
  if (coop.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    ncclGinCall<ncclGinApi_Put>(ctx, ncclCoopThread(), teamRankToGinRank(net->comm, team, peer),
                                /*hasWins=*/false, nullptr, 0, nullptr, 0, 0, signal, signalOp, signalOpArg,
                                /*hasCounter=*/false, 0, isDescriptor, descriptor, requiredRelease, givenRelease,
                                optFlags);
  }
  coop.sync();
}

NCCL_DEVICE_INLINE void ncclGinSignal(ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId,
                                      ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                      bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                      cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = isSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = net->comm.ginStrongLegacySignals;
  }
  ncclGinSignal_C_impl(net, team, peer, signal, signalOp, signalOpArg, coopImpl, isDescriptor, descriptor, givenRelease,
                       requiredRelease, ncclGinOptFlagsDefault);
}

NCCL_DEVICE_INLINE void ncclGinSignal_v2(ncclGin_C* net, ncclTeam team, int peer, bool isSignal,
                                         ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                         ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                         cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                         uint32_t optFlags) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = isSignal ? NCCL_GIN_SIGNAL_TYPE_INDEXED : NCCL_GIN_SIGNAL_TYPE_NONE;
  if (isSignal) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = net->comm.ginStrongLegacySignals;
  }
  ncclGinSignal_C_impl(net, team, peer, signal, signalOp, signalOpArg, coopImpl, isDescriptor, descriptor, givenRelease,
                       requiredRelease, optFlags);
}

NCCL_DEVICE_INLINE void ncclGinSignal_v3(ncclGin_C* net, ncclTeam team, int peer, ncclGinSignalType signalType,
                                         ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId,
                                         bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                         ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                         cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                         uint32_t optFlags) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = signalType;
  if (signalType == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = isStrong;
  } else if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    signal.vaSignal.signalWindow =
      nccl::gin::internal::getGinWindow(signalWin, net->comm.backendIndex, net->connectionId);
    signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWin, signalOffset);
    signal.vaSignal.ncclWindow = signalWin;
    signal.isStrong = isStrong;
  }
  ncclGinSignal_C_impl(net, team, peer, signal, signalOp, signalOpArg, coopImpl, isDescriptor, descriptor, givenRelease,
                       requiredRelease, optFlags);
}

NCCL_DEVICE_INLINE uint64_t ncclGinReadSignal(ncclGin_C* net, ncclGinSignal_t signal, int bits,
                                              cuda::memory_order ord) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
  uint64_t mask = uint64_t(-1) >> (64 - bits);
  uint64_t raw = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
  return (raw - sig.offset) & mask;
}

NCCL_DEVICE_INLINE uint64_t ncclGinReadSignalVA(ncclGin_C* net, ncclWindow_t signalWindow, size_t signalOffset,
                                                int bits, cuda::memory_order ord) {
  (void)net;
  // VA signals live in the window as-is, so unlike ncclGinReadSignal there is no NIC offset to subtract.
  uint64_t* ptr = (uint64_t*)ncclGetLocalPointer(signalWindow, signalOffset);
  uint64_t mask = uint64_t(-1) >> (64 - bits);
  return mask & cuda::atomic_ref<uint64_t>{*ptr}.load(ord);
}

NCCL_DEVICE_INLINE void ncclGinWaitSignal(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                          uint64_t least, int bits, cuda::memory_order ord) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
    (void)nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/false>(sig, least, bits, ord, net->comm.abortFlag,
                                                                          0ULL);
  }
  coopImpl.sync();
}

NCCL_DEVICE_INLINE ncclResult_t ncclGinWaitSignalTimeout(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                                         uint64_t least, int bits, cuda::memory_order ord,
                                                         uint64_t timeoutCycles) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclResult_t ret = ncclSuccess;
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
    ret = nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/true>(sig, least, bits, ord, net->comm.abortFlag,
                                                                         timeoutCycles);
  }
  coopImpl.sync();
  return ret;
}

NCCL_DEVICE_INLINE void ncclGinWaitSignalVA(ncclGin_C* net, ncclIrCoop const* coop, ncclWindow_t signalWindow,
                                            size_t signalOffset, uint64_t least, int bits, cuda::memory_order ord) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    uint64_t* ptr = (uint64_t*)ncclGetLocalPointer(signalWindow, signalOffset);
    (void)nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/false>({ptr, 0}, least, bits, ord,
                                                                          net->comm.abortFlag, 0ULL);
  }
  coopImpl.sync();
}

NCCL_DEVICE_INLINE ncclResult_t ncclGinWaitSignalTimeoutVA(ncclGin_C* net, ncclIrCoop const* coop,
                                                           ncclWindow_t signalWindow, size_t signalOffset,
                                                           uint64_t least, int bits, cuda::memory_order ord,
                                                           uint64_t timeoutCycles) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclResult_t ret = ncclSuccess;
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    uint64_t* ptr = (uint64_t*)ncclGetLocalPointer(signalWindow, signalOffset);
    ret = nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/true>({ptr, 0}, least, bits, ord,
                                                                         net->comm.abortFlag, timeoutCycles);
  }
  coopImpl.sync();
  return ret;
}

NCCL_DEVICE_INLINE void ncclGinIncreaseSignalShadow(ncclGin_C* net, ncclGinSignal_t signal, uint64_t delta) {
  asm volatile("red.relaxed.cta.add.u64 [%0],%1;" ::"l"(net->_signalShadows + signal), "l"(delta) : "memory");
}

NCCL_DEVICE_INLINE void ncclGinWaitSignalMeetShadow(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                                    int bits, cuda::memory_order ord) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
    (void)nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/false>(sig, net->_signalShadows[signal], bits, ord,
                                                                          net->comm.abortFlag, 0ULL);
  }
  coopImpl.sync();
}

NCCL_DEVICE_INLINE void ncclGinWaitSignalFollowShadow(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                                      uint64_t leastDelta, uint64_t* before, uint64_t* delta, int bits,
                                                      cuda::memory_order ord) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  using nccl::utility::testAbort;
  uint32_t steps = 0;
  coopImpl.sync();
  uint64_t before64 = net->_signalShadows[signal];
  uint64_t after64;
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    auto sig = ncclGinCall<ncclGinApi_GetSignalPtr>(ctx, signal);
    uint64_t offset = sig.offset;
    uint64_t least = before64 + leastDelta + offset;
    NVCC_PRAGMA_UNROLL_DISABLED
    do after64 = cuda::atomic_ref<uint64_t>{*sig.ptr}.load(ord);
    while (!nccl::utility::rollingLessEq(least, after64, bits) && !testAbort(net->comm.abortFlag, steps));
    // Convert NIC value back to logical space for shadow.
    after64 = after64 - offset;
    net->_signalShadows[signal] = after64;
  }
  // TODO: use ncclCoopBcast once ncclCoopAny supports it; its vtable exposes only thread_rank/size/sync. Until then
  // the updated shadow doubles as broadcast storage, costing an extra sync and the bits<=32 shuffle path.
  coopImpl.sync();
  after64 = net->_signalShadows[signal];
  uint64_t mask = uint64_t(-1) >> (64 - bits);
  *before = mask & before64;
  *delta = mask & (after64 - before64);
}

NCCL_DEVICE_INLINE uint64_t* ncclGinGetSignalShadowPtr(ncclGin_C* net, ncclGinSignal_t signal) {
  return &net->_signalShadows[signal];
}

NCCL_DEVICE_INLINE void ncclGinResetSignal(ncclGin_C* net, ncclGinSignal_t signal) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinSignalDescriptor signalDesc{};
  signalDesc.type = NCCL_GIN_SIGNAL_TYPE_INDEXED;
  signalDesc.indexedSignal.signalId = signal;
  ncclGinCall<ncclGinApi_ResetSignal>(ctx, signalDesc);
  net->_signalShadows[signal] = 0;
}

NCCL_DEVICE_INLINE void ncclGinResetSignalVA(ncclGin_C* net, ncclWindow_t signalWindow, size_t signalOffset) {
  ncclGinSignalDescriptor signal{};
  signal.type = NCCL_GIN_SIGNAL_TYPE_VA;
  signal.vaSignal.signalWindow =
    nccl::gin::internal::getGinWindow(signalWindow, net->comm.backendIndex, net->connectionId);
  signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWindow, signalOffset);
  signal.vaSignal.ncclWindow = signalWindow;
  ncclGinCall<ncclGinApi_ResetSignal>(ncclGin_C_makeCtx(net), signal);
}

NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(ncclGin_C* net, ncclGinCounter_t counter, int bits,
                                               cuda::memory_order ord) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, counter);
  uint64_t mask = uint64_t(-1) >> (64 - bits);
  uint64_t raw = cuda::atomic_ref<uint64_t>{*ctr.ptr}.load(ord);
  return (raw - ctr.offset) & mask;
}

NCCL_DEVICE_INLINE void ncclGinWaitCounter(ncclGin_C* net, ncclIrCoop const* coop, ncclGinCounter_t counter,
                                           uint64_t least, int bits, cuda::memory_order ord) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, counter);
    (void)nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/false>(ctr, least, bits, ord, net->comm.abortFlag,
                                                                          0ULL);
  }
  coopImpl.sync();
}

NCCL_DEVICE_INLINE ncclResult_t ncclGinWaitCounterTimeout(ncclGin_C* net, ncclIrCoop const* coop,
                                                          ncclGinCounter_t counter, uint64_t least, int bits,
                                                          cuda::memory_order ord, uint64_t timeoutCycles) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclResult_t ret = ncclSuccess;
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    auto ctr = ncclGinCall<ncclGinApi_GetCounterPtr>(ctx, counter);
    ret = nccl::gin::internal::waitRollingLessEq</*EnableTimeout=*/true>(ctr, least, bits, ord, net->comm.abortFlag,
                                                                         timeoutCycles);
  }
  coopImpl.sync();
  return ret;
}

NCCL_DEVICE_INLINE void ncclGinResetCounter(ncclGin_C* net, ncclGinCounter_t counter) {
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_ResetCounter>(ctx, counter);
}

NCCL_DEVICE_INLINE void ncclGinFlush(ncclGin_C* net, ncclIrCoop const* coop, cuda::memory_order ord) {
  ncclGinFlush_v2(net, coop, ord, /*isDescriptor=*/false, /*descriptor=*/nullptr);
}

NCCL_DEVICE_INLINE void ncclGinFlush_v2(ncclGin_C* net, ncclIrCoop const* coop, cuda::memory_order ord,
                                        bool isDescriptor, ncclGinDescriptorSmem* descriptor) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclGinCall<ncclGinApi_Flush>(ctx, coopImpl, isDescriptor, descriptor, ord, net->comm.abortFlag);
  coopImpl.sync();
}

NCCL_DEVICE_INLINE ncclResult_t ncclGinFlushTimeout(ncclGin_C* net, ncclIrCoop const* coop, cuda::memory_order ord,
                                                    bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                                    uint64_t timeoutCycles) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  ncclGinCtx ctx = ncclGin_C_makeCtx(net);
  ncclResult_t ret =
    ncclGinCall<ncclGinApi_Flush>(ctx, coopImpl, isDescriptor, descriptor, ord, net->comm.abortFlag, timeoutCycles);
  coopImpl.sync();
  return ret;
}

NCCL_DEVICE_INLINE void ncclGinFlushAsync(ncclGin_C* net, ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                          ncclIrCoop const* coop, uint32_t optFlags, bool isDescriptor,
                                          ncclGinDescriptorSmem* descriptor) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  using nccl::gin::internal::teamRankToGinRank;
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    ncclGinCall<ncclGinApi_FlushAsync>(ctx, teamRankToGinRank(net->comm, team, peer), outRequest, isDescriptor,
                                       descriptor, optFlags);
  }
  coopImpl.sync();
}

NCCL_DEVICE_INLINE void ncclGinWait(ncclGin_C* net, ncclGinRequest_t* request, ncclIrCoop const* coop,
                                    bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::memory_order ord) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    ncclGinCall<ncclGinApi_Wait>(ctx, *request, isDescriptor, descriptor, ord, net->comm.abortFlag);
  }
  coopImpl.sync();
}

NCCL_DEVICE_INLINE ncclResult_t ncclGinWaitTimeout(ncclGin_C* net, ncclGinRequest_t* request, ncclIrCoop const* coop,
                                                   bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                                   cuda::memory_order ord, uint64_t timeoutCycles) {
  ncclCoopAny coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclResult_t ret = ncclSuccess;
  coopImpl.sync();
  if (coopImpl.thread_rank() == 0) {
    ncclGinCtx ctx = ncclGin_C_makeCtx(net);
    ret =
      ncclGinCall<ncclGinApi_Wait>(ctx, *request, isDescriptor, descriptor, ord, net->comm.abortFlag, timeoutCycles);
  }
  coopImpl.sync();
  return ret;
}

#endif // _NCCL_DEVICE_WRAPPER_IMPL_GIN_H_
