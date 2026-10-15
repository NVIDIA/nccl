/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
// Internal implementation fragment; include only from nccl_device_ir.cu
// after its prerequisite headers.

static_assert(sizeof(ncclIrGin) == sizeof(ncclGin), "GIN storage size mismatch");
static_assert(alignof(ncclIrGin) == alignof(ncclGin), "GIN storage alignment mismatch");

NCCL_IR_DEVICE_INLINE void ncclIrGinInit(ncclIrGin* net, ncclDevComm const* comm, int contextIndex,
                                         ncclGinResourceSharingMode resourceSharingMode) {
  ::new (net) ncclGin(*comm, contextIndex, resourceSharingMode);
}

// Runtime C ABI arguments adapt to the native method's template parameters.
// Native GIN templates are defined before these overloads. Their hook calls must
// remain unqualified so ADL can find the overloads through the adapter types at
// instantiation. Keep hook names and parameter types compatible with those calls;
// hooks taking a GIN object must accept the const ALL-mask ncclGin used here.
// If an overload stops matching, the native catch-all templates can silently
// select no signal, counter or descriptor instead of reporting a compile error.
struct ncclIrGinRemoteAction {
  ncclGinSignalDescriptor signal;
  ncclGinSignalOp_t op;
  uint64_t arg;
};
static __device__ __forceinline__ ncclGinSignalDescriptor ncclGin_getSignalDescriptor(ncclGin const&,
                                                                                      ncclIrGinRemoteAction action) {
  return action.signal;
}
static __device__ __forceinline__ ncclGinSignalOp_t ncclGin_getSignalOp(ncclIrGinRemoteAction action) {
  return action.op;
}
static __device__ __forceinline__ uint64_t ncclGin_getSignalOpArg(ncclIrGinRemoteAction action) {
  return action.arg;
}

struct ncclIrGinLocalAction {
  bool isCounter;
  ncclGinCounter_t counter;
};
static __device__ __forceinline__ bool ncclGin_isCounter(ncclIrGinLocalAction action) {
  return action.isCounter;
}
static __device__ __forceinline__ ncclGinCounter_t ncclGin_getCounterId(ncclGin const&, ncclIrGinLocalAction action) {
  return action.counter;
}

struct ncclIrGinDescriptor {
  bool isDescriptor;
  ncclGinDescriptorSmem* descriptor;
};
static __device__ __forceinline__ bool ncclGin_isDescriptor(ncclIrGinDescriptor descriptor) {
  return descriptor.isDescriptor;
}
static __device__ __forceinline__ ncclGinDescriptorSmem* ncclGin_getDescriptor(ncclIrGinDescriptor descriptor) {
  return descriptor.descriptor;
}

struct ncclIrGinSegment {
  bool isDeviceOnly;
};
static __device__ __forceinline__ bool ncclGin_isDeviceOnly(ncclIrGinSegment segment) {
  return segment.isDeviceOnly;
}

NCCL_IR_DEVICE_INLINE void ncclIrGinPut(ncclIrGin const* net, ncclTeam team, int peer, ncclWindow_t dstWin,
                                        size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
                                        ncclGinSignalType signalType, ncclWindow_t signalWin, size_t signalOffset,
                                        ncclGinSignal_t signalId, bool isStrong, ncclGinSignalOp_t signalOp,
                                        uint64_t signalOpArg, bool isCounter, ncclGinCounter_t counterId,
                                        ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                        cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                        uint32_t optFlags, ncclGinSegmentType_t segmentType) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = signalType;
  if (signalType == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = isStrong;
  } else if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    signal.vaSignal.signalWindow =
      nccl::gin::internal::getGinWindow(signalWin, netImpl.comm.backendIndex, netImpl.connectionId);
    signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWin, signalOffset);
    signal.vaSignal.ncclWindow = signalWin;
    signal.isStrong = isStrong;
  }
  netImpl.put(team, peer, dstWin, dstOffset, srcWin, srcOffset, bytes,
              ncclIrGinRemoteAction{signal, signalOp, signalOpArg}, ncclIrGinLocalAction{isCounter, counterId},
              coopImpl, ncclIrGinDescriptor{isDescriptor, descriptor}, givenRelease, requiredRelease, optFlags,
              ncclIrGinSegment{segmentType == ncclGinSegmentTypeDevice});
}

NCCL_IR_DEVICE_INLINE void ncclIrGinPutValue(ncclIrGin const* net, ncclTeam team, int peer, ncclWindow_t dstWin,
                                             size_t dstOffset, uint64_t value, size_t size,
                                             ncclGinSignalType signalType, ncclWindow_t signalWin, size_t signalOffset,
                                             ncclGinSignal_t signalId, bool isStrong, ncclGinSignalOp_t signalOp,
                                             uint64_t signalOpArg, ncclIrCoop const* coop, bool isDescriptor,
                                             ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                             cuda::thread_scope requiredRelease, uint32_t optFlags) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = signalType;
  if (signalType == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = isStrong;
  } else if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    signal.vaSignal.signalWindow =
      nccl::gin::internal::getGinWindow(signalWin, netImpl.comm.backendIndex, netImpl.connectionId);
    signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWin, signalOffset);
    signal.vaSignal.ncclWindow = signalWin;
    signal.isStrong = isStrong;
  }
  switch (size) {
  case 1:
    netImpl.putValue(team, peer, dstWin, dstOffset, (uint8_t)value,
                     ncclIrGinRemoteAction{signal, signalOp, signalOpArg}, coopImpl,
                     ncclIrGinDescriptor{isDescriptor, descriptor}, givenRelease, requiredRelease, optFlags);
    break;
  case 2:
    netImpl.putValue(team, peer, dstWin, dstOffset, (uint16_t)value,
                     ncclIrGinRemoteAction{signal, signalOp, signalOpArg}, coopImpl,
                     ncclIrGinDescriptor{isDescriptor, descriptor}, givenRelease, requiredRelease, optFlags);
    break;
  case 4:
    netImpl.putValue(team, peer, dstWin, dstOffset, (uint32_t)value,
                     ncclIrGinRemoteAction{signal, signalOp, signalOpArg}, coopImpl,
                     ncclIrGinDescriptor{isDescriptor, descriptor}, givenRelease, requiredRelease, optFlags);
    break;
  case 8:
  default:
    netImpl.putValue(team, peer, dstWin, dstOffset, (uint64_t)value,
                     ncclIrGinRemoteAction{signal, signalOp, signalOpArg}, coopImpl,
                     ncclIrGinDescriptor{isDescriptor, descriptor}, givenRelease, requiredRelease, optFlags);
    break;
  }
}

NCCL_IR_DEVICE_INLINE void ncclIrGinGet(ncclIrGin const* net, ncclTeam team, int peer, ncclWindow_t remoteWnd,
                                        size_t remoteOffset, ncclWindow_t localWnd, size_t localOffset, size_t bytes,
                                        ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                        uint32_t optFlags, ncclGinSegmentType_t segmentType) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.get(team, peer, remoteWnd, remoteOffset, localWnd, localOffset, bytes, coopImpl,
              ncclIrGinDescriptor{isDescriptor, descriptor}, optFlags,
              ncclIrGinSegment{segmentType == ncclGinSegmentTypeDevice});
}

NCCL_IR_DEVICE_INLINE void ncclIrGinSignal(ncclIrGin const* net, ncclTeam team, int peer, ncclGinSignalType signalType,
                                           ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId,
                                           bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                           ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                           cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                           uint32_t optFlags) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  ncclGinSignalDescriptor signal{};
  signal.type = signalType;
  if (signalType == NCCL_GIN_SIGNAL_TYPE_INDEXED) {
    signal.indexedSignal.signalId = signalId;
    signal.isStrong = isStrong;
  } else if (signalType == NCCL_GIN_SIGNAL_TYPE_VA) {
    signal.vaSignal.signalWindow =
      nccl::gin::internal::getGinWindow(signalWin, netImpl.comm.backendIndex, netImpl.connectionId);
    signal.vaSignal.signalOffset = nccl::gin::internal::windowOffsetToGinOffset(signalWin, signalOffset);
    signal.vaSignal.ncclWindow = signalWin;
    signal.isStrong = isStrong;
  }
  netImpl.signal(team, peer, ncclIrGinRemoteAction{signal, signalOp, signalOpArg}, coopImpl,
                 ncclIrGinDescriptor{isDescriptor, descriptor}, givenRelease, requiredRelease, optFlags);
}

NCCL_IR_DEVICE_INLINE uint64_t ncclIrGinReadSignal(ncclIrGin const* net, ncclGinSignal_t signal, int bits,
                                                   cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  return netImpl.readSignal(signal, bits, ord);
}

NCCL_IR_DEVICE_INLINE uint64_t ncclIrGinReadSignalVA(ncclIrGin const* net, ncclWindow_t signalWindow,
                                                     size_t signalOffset, int bits, cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  return netImpl.readSignal(signalWindow, signalOffset, bits, ord);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinWaitSignal(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                               uint64_t least, int bits, cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.waitSignal(coopImpl, signal, least, bits, ord);
}

NCCL_IR_DEVICE_INLINE ncclResult_t ncclIrGinWaitSignalTimeout(ncclIrGin const* net, ncclIrCoop const* coop,
                                                              ncclGinSignal_t signal, uint64_t least, int bits,
                                                              cuda::memory_order ord, uint64_t timeoutCycles) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  return netImpl.waitSignal(coopImpl, signal, least, bits, ord, timeoutCycles);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinWaitSignalVA(ncclIrGin const* net, ncclIrCoop const* coop,
                                                 ncclWindow_t signalWindow, size_t signalOffset, uint64_t least,
                                                 int bits, cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.waitSignal(coopImpl, signalWindow, signalOffset, least, bits, ord);
}

NCCL_IR_DEVICE_INLINE ncclResult_t ncclIrGinWaitSignalTimeoutVA(ncclIrGin const* net, ncclIrCoop const* coop,
                                                                ncclWindow_t signalWindow, size_t signalOffset,
                                                                uint64_t least, int bits, cuda::memory_order ord,
                                                                uint64_t timeoutCycles) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  return netImpl.waitSignal(coopImpl, signalWindow, signalOffset, least, bits, ord, timeoutCycles);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinIncreaseSignalShadow(ncclIrGin const* net, ncclGinSignal_t signal, uint64_t delta) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  netImpl.increaseSignalShadow(signal, delta);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinWaitSignalMeetShadow(ncclIrGin const* net, ncclIrCoop const* coop,
                                                         ncclGinSignal_t signal, int bits, cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.waitSignalMeetShadow(coopImpl, signal, bits, ord);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinWaitSignalFollowShadow(ncclIrGin const* net, ncclIrCoop const* coop,
                                                           ncclGinSignal_t signal, uint64_t leastDelta,
                                                           uint64_t* before, uint64_t* delta, int bits,
                                                           cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  // Native broadcast requires the concrete coop type initialized in this storage.
  if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopThread>()) {
    netImpl.waitSignalFollowShadow(*reinterpret_cast<ncclCoopThread const*>(&coopImpl.storage), signal, leastDelta,
                                   before, delta, bits, ord);
    return;
  }
  if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopWarp>()) {
    netImpl.waitSignalFollowShadow(*reinterpret_cast<ncclCoopWarp const*>(&coopImpl.storage), signal, leastDelta,
                                   before, delta, bits, ord);
    return;
  }
  if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopLanes>()) {
    netImpl.waitSignalFollowShadow(*reinterpret_cast<ncclCoopLanes const*>(&coopImpl.storage), signal, leastDelta,
                                   before, delta, bits, ord);
    return;
  }
  if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopWarpSpan>()) {
    netImpl.waitSignalFollowShadow(*reinterpret_cast<ncclCoopWarpSpan const*>(&coopImpl.storage), signal, leastDelta,
                                   before, delta, bits, ord);
    return;
  }
  if (coopImpl.vtable == ncclCoopAny::get_vtable<ncclCoopCta>()) {
    netImpl.waitSignalFollowShadow(*reinterpret_cast<ncclCoopCta const*>(&coopImpl.storage), signal, leastDelta, before,
                                   delta, bits, ord);
    return;
  }
  __builtin_unreachable();
}

NCCL_IR_DEVICE_INLINE uint64_t* ncclIrGinGetSignalShadowPtr(ncclIrGin const* net, ncclGinSignal_t signal) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  return netImpl.getSignalShadowPtr(signal);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinResetSignal(ncclIrGin const* net, ncclGinSignal_t signal) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  netImpl.resetSignal(signal);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinResetSignalVA(ncclIrGin const* net, ncclWindow_t signalWindow,
                                                  size_t signalOffset) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  netImpl.resetSignal(signalWindow, signalOffset);
}

NCCL_IR_DEVICE_INLINE uint64_t ncclIrGinReadCounter(ncclIrGin const* net, ncclGinCounter_t counter, int bits,
                                                    cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  return netImpl.readCounter(counter, bits, ord);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinWaitCounter(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinCounter_t counter,
                                                uint64_t least, int bits, cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.waitCounter(coopImpl, counter, least, bits, ord);
}

NCCL_IR_DEVICE_INLINE ncclResult_t ncclIrGinWaitCounterTimeout(ncclIrGin const* net, ncclIrCoop const* coop,
                                                               ncclGinCounter_t counter, uint64_t least, int bits,
                                                               cuda::memory_order ord, uint64_t timeoutCycles) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  return netImpl.waitCounter(coopImpl, counter, least, bits, ord, timeoutCycles);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinResetCounter(ncclIrGin const* net, ncclGinCounter_t counter) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  netImpl.resetCounter(counter);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinFlush(ncclIrGin const* net, ncclIrCoop const* coop, cuda::memory_order ord,
                                          bool isDescriptor, ncclGinDescriptorSmem* descriptor) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.flush(coopImpl, ord, ncclIrGinDescriptor{isDescriptor, descriptor});
}

NCCL_IR_DEVICE_INLINE ncclResult_t ncclIrGinFlushTimeout(ncclIrGin const* net, ncclIrCoop const* coop,
                                                         cuda::memory_order ord, bool isDescriptor,
                                                         ncclGinDescriptorSmem* descriptor, uint64_t timeoutCycles) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  return netImpl.flush(coopImpl, ord, ncclIrGinDescriptor{isDescriptor, descriptor}, timeoutCycles);
}

NCCL_IR_DEVICE_INLINE void ncclIrGinFlushAsync(ncclIrGin const* net, ncclTeam team, uint32_t peer,
                                               ncclGinRequest_t* outRequest, ncclIrCoop const* coop, uint32_t optFlags,
                                               bool isDescriptor, ncclGinDescriptorSmem* descriptor) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.flushAsync(team, peer, outRequest, coopImpl, optFlags, ncclIrGinDescriptor{isDescriptor, descriptor});
}

NCCL_IR_DEVICE_INLINE void ncclIrGinWait(ncclIrGin const* net, ncclGinRequest_t* request, ncclIrCoop const* coop,
                                         bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::memory_order ord) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  netImpl.wait(*request, coopImpl, ncclIrGinDescriptor{isDescriptor, descriptor}, ord);
}

NCCL_IR_DEVICE_INLINE ncclResult_t ncclIrGinWaitTimeout(ncclIrGin const* net, ncclGinRequest_t* request,
                                                        ncclIrCoop const* coop, bool isDescriptor,
                                                        ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                                        uint64_t timeoutCycles) {
  auto const& netImpl = *reinterpret_cast<ncclGin const*>(net);
  auto const& coopImpl = *reinterpret_cast<ncclCoopAny const*>(coop);
  return netImpl.wait(*request, coopImpl, ncclIrGinDescriptor{isDescriptor, descriptor}, ord, timeoutCycles);
}
