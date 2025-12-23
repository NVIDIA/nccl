/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_SESSION_H_
#define _NCCL_DEVICE_GIN_SESSION_H_
#include "core.h"
#include "gin/gin_device_common.h"

#if NCCL_CHECK_CUDACC
struct ncclGinCtx; // Definition in nccl_device/gin/gin_device_host_common.h
template<unsigned> struct ncclGinCtx_M; // ...

struct ncclGinDescriptorSmem; // A type user allocates in __shared__ memory

// Used as completion actions for ncclGinSession::put
struct ncclGin_None {};

struct ncclGin_SignalAdd { ncclGinSignal_t signal; uint64_t value; };
// SignalInc: equivalent to SignalAdd{+1} except it may not be mixed with any
// other signal operator without intervening signal reset(). Formally: for a
// given signal, all operations between successive reset()'s of that signal must
// either all be SignalInc or all not SignalInc.
struct ncclGin_SignalInc { ncclGinSignal_t signal; };
// Support deferred:
// struct ncclGin_SignalSet { ncclGinSignal_t signal; uint64_t value; };
struct ncclGin_CounterInc { ncclGinCounter_t counter; };

struct ncclGin_DescriptorSmem { ncclGinDescriptorSmem* descriptor; };

template<unsigned backendMask>
struct ncclGin_BackendMask;

template<ncclNetDeviceType backend>
using ncclGin_BackendOne = ncclGin_BackendMask<(1u<<(int)backend)>;

using ncclGin = ncclGin_BackendMask<NCCL_GIN_BACKEND_MASK_ALL>;

#endif

#if NCCL_CHECK_CUDACC
struct ncclGin_C {
  ncclDevComm const& comm;
  uint32_t nContexts:8, contextId:8, _ginBackend:8;

  //////////////////////////////////////////////////////////////////////////////
  // internal:
  void* _ginHandle;
  uint64_t* _signalShadows;
  unsigned backendMask;

  NCCL_DEVICE_INLINE ncclGin_C(ncclDevComm const& comm_, unsigned backendMask_, int contextIndex);
};

// Helper init function that wraps placement new
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGin_C_init(
  ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPut(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  bool isCounter, ncclGinCounter_t counterId,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope requiredRelease,  cuda::thread_scope givenRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinSignal(
  ncclGin_C* net,
  ncclTeam team, int peer,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope requiredRelease, cuda::thread_scope givenRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinFlush(
  ncclGin_C* net,
  ncclCoopAny coop,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadCounter(
  ncclGin_C* net,
  ncclGinCounter_t counter,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitCounter(
  ncclGin_C* net,
  ncclCoopAny coop,
  ncclGinCounter_t counter,
  uint64_t least,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t ncclGinReadSignal(
  ncclGin_C* net,
  ncclGinSignal_t signal,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinWaitSignal(
  ncclGin_C* net,
  ncclCoopAny coop,
  ncclGinSignal_t signal,
  uint64_t least,
  int bits,
  cuda::memory_order ord);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetCounter(
  ncclGin_C* net,
  ncclGinCounter_t counter);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinResetSignal(
  ncclGin_C* net,
  ncclGinSignal_t signal);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinPutValue(
  ncclGin_C* net,
  ncclTeam team, int peer,
  ncclWindow_t dstWin, size_t dstOffset,
  uint64_t value, size_t size,
  bool isSignal, ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
  ncclCoopAny coop,
  bool isDescriptor, ncclGinDescriptorSmem* descriptor,
  cuda::thread_scope requiredRelease, cuda::thread_scope givenRelease);

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE uint64_t* ncclGinGetSignalShadowPtr(
  ncclGin_C* net,
  ncclGinSignal_t signal);

template<unsigned backendMask>
struct ncclGin_BackendMask {
  ncclDevComm const& comm;
  uint32_t nContexts:8, contextId:8, _ginBackend:8;

  // Loads GIN context into registers. Each context has one QP per peer.
  NCCL_DEVICE_INLINE ncclGin_BackendMask(ncclDevComm const&, int contextIndex);

  template<
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this netContext to the same peer are settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer,
    ncclWindow_t dstWnd, size_t dstOffset,
    ncclWindow_t srcWnd, size_t srcOffset, size_t bytes,
    RemoteAction remoteAction = ncclGin_None{},
    LocalAction localAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
    cuda::thread_scope expected_scope = cuda::thread_scope_device
  ) const;

  template<
    typename T,
    // Action to take on peer when put completes. If a signalling action is used
    // then that signal will be visible only after the payload of this put as well as
    // the payloads of preceding puts on this context to the same peer are settled.
    typename RemoteAction = ncclGin_None, // one of ncclGin_{None|SignalInc|SignalAdd|SignalSet}
    // Action to take locally when source has been consumed.
    typename LocalAction = ncclGin_None, // one of ncclGin_{None|CounterInc}
    // Set of threads participating in this put. Must be a subset of Coop.
    typename Coop = ncclCoopThread,
    // Optional smem descriptor space to use. Either ncclGin_{None|DescriptorSmem}
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void put(
    ncclTeam, int peer,
    ncclSymPtr<T> dstElts, ncclSymPtr<T> srcElts, size_t nElts,
    RemoteAction remoteAction = ncclGin_None{},
    LocalAction localAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
    cuda::thread_scope expected_scope = cuda::thread_scope_device
  ) const;

  template<
    typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
    typename RemoteAction = ncclGin_None,
    typename Coop = ncclCoopThread,
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void putValue(
    ncclTeam, int peer,
    ncclWindow_t dstWnd, size_t dstOffset, T value,
    RemoteAction remoteAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
    cuda::thread_scope expected_scope = cuda::thread_scope_device
  ) const;

  template<
    typename T, // requires sizeof(T) <= 8
    // See put() for all template arguments.
    typename RemoteAction = ncclGin_None,
    typename Coop = ncclCoopThread,
    typename DescriptorSmem = ncclGin_None
  >
  NCCL_DEVICE_INLINE void putValue(
    ncclTeam, int peer,
    ncclSymPtr<T> dst, T value,
    RemoteAction remoteAction = ncclGin_None{},
    Coop coop = ncclCoopThread{},
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
    cuda::thread_scope expected_scope = cuda::thread_scope_device
  ) const;

  template<typename RemoteAction,
           typename Coop = ncclCoopThread,
           typename DescriptorSmem = ncclGin_None>
  NCCL_DEVICE_INLINE void signal(
    ncclTeam, int peer, RemoteAction remoteAction,
    Coop coop = ncclCoopThread(),
    DescriptorSmem descriptor = ncclGin_None{},
    cuda::thread_scope alreadyReleased = cuda::thread_scope_thread,
    cuda::thread_scope expected_scope = cuda::thread_scope_device
  ) const;

  // All source buffers from put's from any thread in this coop will be safe to reuse.
  // Flush does not guarantee that data has settled in remote memory.
  template<typename Coop>
  NCCL_DEVICE_INLINE void flush(Coop, cuda::memory_order ord = cuda::memory_order_acquire) const;

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

  NCCL_DEVICE_INLINE uint64_t readCounter(ncclGinCounter_t counter, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire) const;

  template<typename Coop>
  NCCL_DEVICE_INLINE void waitCounter(Coop, ncclGinCounter_t counter, uint64_t least, int bits=56, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Each signal has a dedicated "shadow" which the user is free to manipulate for
  // any reason. The only calls which manipulate the shadow are `increaseSignalShadow`
  // and `resetSignal`.
  NCCL_DEVICE_INLINE uint64_t* getSignalShadowPtr(ncclGinSignal_t signal) const;
  NCCL_DEVICE_INLINE void increaseSignalShadow(ncclGinSignal_t signal, uint64_t delta) const;

  // Returns current value of signal with all but bottom bits set to zero.
  NCCL_DEVICE_INLINE uint64_t readSignal(ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for signal to meet or exceed value.
  template<typename Coop>
  NCCL_DEVICE_INLINE void waitSignal(Coop, ncclGinSignal_t signal, uint64_t least, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait for signal to meet or exceed shadow value.
  template<typename Coop>
  NCCL_DEVICE_INLINE void waitSignalMeetShadow(Coop, ncclGinSignal_t signal, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Wait until signal exceeds shadow by `leastDelta` (typically 1), updates shadow
  // with latest value, and returns with `before` equal to previous shadow value
  // and `delta` equal to difference.
  template<typename Coop, typename Uint>
  NCCL_DEVICE_INLINE void waitSignalFollowShadow(Coop, ncclGinSignal_t signal, Uint leastDelta, Uint* before, Uint* delta, int bits=64, cuda::memory_order ord = cuda::memory_order_acquire) const;

  // Sets to zero. May not race with concurrent modifications to counter.
  NCCL_DEVICE_INLINE void resetCounter(ncclGinCounter_t counter) const;
  // Sets signal and shadow to zero. May not race with concurrent modifcations to signal.
  NCCL_DEVICE_INLINE void resetSignal(ncclGinSignal_t signal) const;

  //////////////////////////////////////////////////////////////////////////////
  // internal:

  void* _ginHandle;
  uint64_t* _signalShadows;

  NCCL_DEVICE_INLINE ncclGinCtx_M<backendMask> _makeCtx() const;
};
#endif

#endif // _NCCL_DEVICE_GIN_SESSION_H_
