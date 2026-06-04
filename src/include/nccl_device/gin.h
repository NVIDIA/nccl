/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SESSION_H_
#define _NCCL_DEVICE_GIN_SESSION_H_
#include "core.h"
#include "gin/gin_device_common.h"

#if NCCL_CHECK_CUDACC
struct ncclGinCtx; // Definition in nccl_device/gin/gin_device_host_common.h
template <unsigned>
struct ncclGinCtx_M; // ...

struct ncclGinDescriptorSmem; // A type user allocates in __shared__ memory

// Used as completion actions for ncclGinSession::put
struct ncclGin_None {};

// Strong VA signal: visibility implies all preceding puts are settled.
struct ncclGin_StrongVASignalInc {
  ncclWindow_t signalWindow;
  size_t signalOffset;
};
// Weak VA signal: guarantees only the bundled put is settled.
struct ncclGin_WeakVASignalInc {
  ncclWindow_t signalWindow;
  size_t signalOffset;
};
// Deprecated: use ncclGin_StrongVASignalInc or ncclGin_WeakVASignalInc.
struct ncclGin_VASignalInc {
  ncclWindow_t signalWindow;
  size_t signalOffset;
};

// Strong VA add signal: visibility implies all preceding puts are settled.
struct ncclGin_StrongVASignalAdd {
  ncclWindow_t signalWindow;
  size_t signalOffset;
  uint64_t value;
};
// Weak VA add signal: guarantees only the bundled put is settled.
struct ncclGin_WeakVASignalAdd {
  ncclWindow_t signalWindow;
  size_t signalOffset;
  uint64_t value;
};
// Deprecated: use ncclGin_StrongVASignalAdd or ncclGin_WeakVASignalAdd.
struct ncclGin_VASignalAdd {
  ncclWindow_t signalWindow;
  size_t signalOffset;
  uint64_t value;
};

// Strong add signal: visibility implies all preceding puts are settled.
struct ncclGin_StrongSignalAdd {
  ncclGinSignal_t signal;
  uint64_t value;
};
// Weak add signal: guarantees only the bundled put is settled.
struct ncclGin_WeakSignalAdd {
  ncclGinSignal_t signal;
  uint64_t value;
};
// Deprecated: use ncclGin_StrongSignalAdd or ncclGin_WeakSignalAdd.
struct ncclGin_SignalAdd {
  ncclGinSignal_t signal;
  uint64_t value;
};

// Strong signal: visibility implies all preceding puts are settled.
// Inc may not be mixed with other signal operators without an intervening reset().
struct ncclGin_StrongSignalInc {
  ncclGinSignal_t signal;
};

// Weak signal: guarantees only the bundled put is settled.
// Inc may not be mixed with other signal operators without an
// intervening reset().
struct ncclGin_WeakSignalInc {
  ncclGinSignal_t signal;
};

// Deprecated: use ncclGin_StrongSignalInc or ncclGin_WeakSignalInc explicitly.
struct ncclGin_SignalInc {
  ncclGinSignal_t signal;
};

// Support deferred:
// struct ncclGin_SignalSet { ncclGinSignal_t signal; uint64_t value; };

// Deprecated: use ncclGin_WeakCounterInc.
struct ncclGin_CounterInc {
  ncclGinCounter_t counter;
};

// Weak counter increment: only guarantees that the bundled put is locally complete.
struct ncclGin_WeakCounterInc {
  ncclGinCounter_t counter;
};

struct ncclGin_DescriptorSmem {
  ncclGinDescriptorSmem* descriptor;
};

// Segment type tags describe the composition of a buffer's physical cuMem segments.
struct ncclGin_SegmentDevice {};       // all segments are device-backed
struct ncclGin_SegmentMixed {}; // mix of HOST_NUMA and device-backed segments
struct ncclGin_SegmentHostNuma {};     // all segments are HOST_NUMA (CPU-backed)

template <unsigned backendMask>
struct ncclGin_BackendMask;

template <ncclNetDeviceType backend>
using ncclGin_BackendOne = ncclGin_BackendMask<(1u << (int)backend)>;

using ncclGin = ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>;

#endif

#if NCCL_CHECK_CUDACC
struct ncclGin_C {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  ncclGinResourceSharingMode resourceSharingMode;

  //////////////////////////////////////////////////////////////////////////////
  // internal:
  void* _ginHandle;
  uint64_t* _signalShadows;
  unsigned backendMask;

  NCCL_DEVICE_INLINE ncclGin_C(ncclDevComm const& comm_, unsigned backendMask_, int contextIndex,
                               ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);
};

// Helper init function that wraps placement new
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_init(ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm,
                                                        int contextIndex);

// Helper init function with explicit resource sharing mode.
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_initWithResourceSharingMode(
  ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex,
  ncclGinResourceSharingMode resourceSharingMode);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset,
  size_t bytes, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal(
  ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
  uint64_t signalOpArg, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut_v2(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset,
  size_t bytes, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease, uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal_v2(
  ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp,
  uint64_t signalOpArg, ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease, uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinFlush(ncclGin_C* net, ncclCoopAny coop, cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(ncclGin_C* net, ncclGinCounter_t counter, int bits,
                                                                cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitCounter(ncclGin_C* net, ncclCoopAny coop, ncclGinCounter_t counter,
                                                            uint64_t least, int bits, cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadSignal(ncclGin_C* net, ncclGinSignal_t signal, int bits,
                                                               cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitSignal(ncclGin_C* net, ncclCoopAny coop, ncclGinSignal_t signal,
                                                           uint64_t least, int bits, cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetCounter(ncclGin_C* net, ncclGinCounter_t counter);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetSignal(ncclGin_C* net, ncclGinSignal_t signal);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
  cuda::thread_scope requiredRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue_v2(
  ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset, uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
  cuda::thread_scope requiredRelease, uint32_t optFlags);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t* ncclGinGetSignalShadowPtr(ncclGin_C* net, ncclGinSignal_t signal);

template <unsigned backendMask>
struct ncclGin_BackendMask {
  ncclDevComm const& comm;
  uint32_t nConnections:8, connectionId:8, _ginBackend:8;
  uint32_t contextId;
  // Runtime-selected resource sharing mode for this context.
  ncclGinResourceSharingMode resourceSharingMode;

  // Loads GIN context into registers. Each context has one QP per peer.
  NCCL_DEVICE_INLINE ncclGin_BackendMask(
    ncclDevComm const&, int contextIndex,
    ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void flushAsync(ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                     Coop coop = ncclCoopThread{}, uint32_t optFlags = ncclGinOptFlagsDefault,
                                     DescriptorSmem descriptor = ncclGin_None{}) const;

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void wait(ncclGinRequest_t& outRequest, Coop coop = ncclCoopThread{},
                               DescriptorSmem descriptor = ncclGin_None{},
                               cuda::memory_order ord = cuda::memory_order_acquire) const;

  template <typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None,
            typename SegmentType = ncclGin_SegmentDevice>
  NCCL_DEVICE_INLINE void get(ncclTeam, int peer, ncclWindow_t remoteWnd, size_t remoteOffset, ncclWindow_t localWnd,
                              size_t localOffset, size_t bytes, Coop coop = ncclCoopThread{},
                              DescriptorSmem descriptor = ncclGin_None{}, uint32_t optFlags = ncclGinOptFlagsDefault,
                              SegmentType bufType = ncclGin_SegmentDevice{}) const;

  template <
    // Action to take on peer when put completes.
    // For strong signals: guarantees this put AND all
    // preceding puts on this context to the same peer are settled.
    // For weak signals: only guarantees the bundled put is settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|StrongVASignal[Inc|Add]|WeakVASignal[Inc|Add],
                                          // StrongSignal[Inc|Add]|WeakSignal[Inc|Add]}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|WeakCounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None,
    // Use ncclGin_SegmentMixed or ncclGin_SegmentHostNuma when the VA contains
    // CPU-backed (HOST_NUMA) segments
    typename SegmentType = ncclGin_SegmentDevice>
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction = ncclGin_None{}, LocalAction localAction = ncclGin_None{}, Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{}, cuda::thread_scope givenRelease = cuda::thread_scope_thread,
    cuda::thread_scope requiredRelease = cuda::thread_scope_device, uint32_t optFlags = ncclGinOptFlagsDefault,
    SegmentType bufType = ncclGin_SegmentDevice{}) const;

  template <
    typename T,
    // Action to take on peer when put completes.
    // For strong signals: guarantees this put AND all preceding puts on this context to the same peer are settled.
    // For weak signals: only guarantees the bundled put is settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|StrongVASignal[Inc|Add]|WeakVASignal[Inc|Add],
                                          // StrongSignal[Inc|Add]|WeakSignal[Inc|Add]}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|ncclGin_WeakCounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None,
    // One of ncclGin_{SegmentDevice|SegmentMixed|SegmentHostNuma}; use a non-Device tag when the VA contains
    // CPU-backed (HOST_NUMA) segments
    typename SegmentType = ncclGin_SegmentDevice>
  NCCL_DEVICE_INLINE void put(ncclTeam, int peer, ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
                              RemoteAction remoteAction = ncclGin_None{}, LocalAction localAction = ncclGin_None{},
                              Coop coop = ncclCoopThread{}, DescriptorSmem descriptor = ncclGin_None{},
                              cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                              cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                              uint32_t optFlags = ncclGinOptFlagsDefault,
                              SegmentType bufType = ncclGin_SegmentDevice{}) const;

  template <typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
            typename RemoteAction = ncclGin_None, typename Coop = ncclCoopThread,
            typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void putValue(ncclTeam, int peer, ncclWindow_t dstWnd, size_t dstOffset, T value,
                                   RemoteAction remoteAction = ncclGin_None{}, Coop coop = ncclCoopThread{},
                                   DescriptorSmem descriptor = ncclGin_None{},
                                   cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                                   cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                                   uint32_t optFlags = ncclGinOptFlagsDefault) const;

  template <typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
            typename RemoteAction = ncclGin_None, typename Coop = ncclCoopThread,
            typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void putValue(ncclTeam, int peer, ncclSymPtr<T> dst, T value,
                                   RemoteAction remoteAction = ncclGin_None{}, Coop coop = ncclCoopThread{},
                                   DescriptorSmem descriptor = ncclGin_None{},
                                   cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                                   cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                                   uint32_t optFlags = ncclGinOptFlagsDefault) const;

  template <typename RemoteAction, typename Coop = ncclCoopThread, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void signal(ncclTeam, int peer, RemoteAction remoteAction, Coop coop = ncclCoopThread(),
                                 DescriptorSmem descriptor = ncclGin_None{},
                                 cuda::thread_scope givenRelease = cuda::thread_scope_thread,
                                 cuda::thread_scope requiredRelease = cuda::thread_scope_device,
                                 uint32_t optFlags = ncclGinOptFlagsDefault) const;

  // All source buffers from put's from any thread in this coop will be safe to reuse.
  // Flush does not guarantee that data has settled in remote memory.
  template <typename Coop, typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire,
                                DescriptorSmem descriptor = ncclGin_None{}) const;

  // Counter and signal wait use "rolling" comparison logic of a given bit-width
  // such that unsigned overflow does not disturb the property that: x < x+1.
  //
  // bool rolling_less_equal(uint64_t a, uint64_t b, int bits) {
  //   uint64_t m = uint64_t(-1)>>(64-bits);
  //   return ((b-a) & m) <= (m>>1);
  // }
  //
  // The condition waited for is that the supplied value is rolling_less_equal
  // to the internal value.
  //
  // Counters are restricted to using a maximum of 56 bits despite that being fewer
  // than a uint64_t can carry.

  NCCL_DEVICE_INLINE uint64_t readCounter(ncclGinCounter_t counter, int bits = 56,
                                          cuda::memory_order ord = cuda::memory_order_acquire) const;

  template <typename Coop>
  NCCL_DEVICE_INLINE void waitCounter(Coop, ncclGinCounter_t counter, uint64_t least, int bits = 56,
                                      cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Each signal has a dedicated "shadow" which the user is free to manipulate for
  // any reason. The only calls which manipulate the shadow are `increaseSignalShadow`
  // and `resetSignal`.
  NCCL_DEVICE_INLINE uint64_t* getSignalShadowPtr(ncclGinSignal_t signal) const;
  NCCL_DEVICE_INLINE void increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const;

  // Returns current value of signal with all but bottom bits set to zero.
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclGinSignal_t signal, int bits = 64,
                                         cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Returns current value of VA signal at given window and offset with all but bottom bits set to zero.
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits = 64,
                                         cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for signal to meet or exceed value.
  template <typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclGinSignal_t signal, uint64_t least, int bits = 64,
                                     cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for VA signal at given window and offset to meet or exceed value.
  template <typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least,
                                     int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for signal to meet or exceed shadow value.
  template <typename Coop>
  NCCL_DEVICE_INLINE void waitSignalMeetShadow(Coop, ncclGinSignal_t signal, int bits = 64,
                                               cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait until signal exceeds shadow by `leastDelta` (typically 1), updates shadow
  // with latest value, and returns with `before` equal to previous shadow value
  // and `delta` equal to difference.
  template <typename Coop, typename Uint>
  NCCL_DEVICE_INLINE void waitSignalFollowShadow(Coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before,
                                                 Uint* delta, int bits = 64,
                                                 cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Sets to zero. May not race with concurrent modifications to counter.
  NCCL_DEVICE_INLINE void resetCounter(ncclGinCounter_t counter) const;
  // Sets signal and shadow to zero. May not race with concurrent modifcations to signal.
  NCCL_DEVICE_INLINE void resetSignal(ncclGinSignal_t signal) const;
  // Resets a VA signal at the given window and offset.
  NCCL_DEVICE_INLINE void resetSignal(ncclWindow_t signalWindow, size_t signalOffset) const;

  //////////////////////////////////////////////////////////////////////////////
  // internal:

  void* _ginHandle;
  uint64_t* _signalShadows;

  NCCL_DEVICE_INLINE ncclGinCtx_M<backendMask> _makeCtx() const;
};
#endif

#endif // _NCCL_DEVICE_GIN_SESSION_H_
