/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_H_
#define _NCCL_DEVICE_WRAPPER_H_

/*
 * NCCL Device API C-style wrapper functions
 */

/*
 * Declaration/type-only view of the NCCL Device API for LLVM IR users.
 *
 * This header intentionally excludes nccl_device/impl/xxx__funcs.h so user IR
 * bitcode can resolve NCCL Device API implementations from libnccl_device.bc.
 */

/*
 * Emit __activemask() as inline asm, like CUDA's sm_30_intrinsics.hpp does.
 * Clang's version returns __nvvm_activemask(), which libNVVM 13.3.27 fails to
 * lower: the PTX calls an .extern llvm.nvvm.activemask that ptxas rejects.
 *
 * Must precede the NCCL and cooperative_groups includes below so all call sites
 * expand. Device pass only: CUDA declares __activemask in the host pass.
 */
 #if defined(__clang__) && defined(__CUDA_ARCH__)
 __attribute__((device)) __attribute__((always_inline)) static unsigned ncclIrActivemask() {
   unsigned _nccl_activemask_ret;
   asm volatile("activemask.b32 %0;" : "=r"(_nccl_activemask_ret));
   return _nccl_activemask_ret;
 }
 #undef __activemask
 #define __activemask ncclIrActivemask
 #endif 

/*
 * Production's __forceinline__ (__inline__ __attribute__((always_inline))) is
 * linkonce_odr and emits no symbol. Drop __inline__ so the device API has
 * external linkage: the bitcode lib emits symbols that consumers resolve from
 * libnccl_device.bc. Must precede the API includes below.
 */
#include "nccl_device/utility.h"
#undef NCCL_DEVICE_INLINE
#undef NCCL_HOST_DEVICE_INLINE

#if defined(__NCCL_DEVICE_LTOIR_LIB__)
#define NCCL_DEVICE_INLINE __device__ __inline_hint__
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __inline_hint__
#elif defined(__clang_llvm_bitcode_lib__)
#define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline))
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __attribute__((always_inline))
#else
#error "nccl_device_wrapper.h requires __NCCL_DEVICE_LTOIR_LIB__ or __clang_llvm_bitcode_lib__"
#endif

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "nccl_device/core.h"
#include "nccl_device/gin_barrier.h"

#include "nccl_device/impl/core__types.h"
#include "nccl_device/impl/lsa_barrier__types.h"
#include "nccl_device/impl/gin_barrier__types.h"

struct ncclDevComm;

////////////////////////////////////////////////////////////////////////////////
// IR wrapper types
////////////////////////////////////////////////////////////////////////////////

/****************************** Coop types ***********************************/

struct ncclIrCoop {
  uint64_t storage[3];
};

/****************************** GIN types ************************************/

// C API counterpart of the ncclGin_Segment* tags in nccl_device/gin.h.
typedef enum ncclGinSegmentType {
  ncclGinSegmentTypeDevice = 0,
  ncclGinSegmentTypeMixed = 1,
  ncclGinSegmentTypeHostNuma = 2,
} ncclGinSegmentType_t;

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

  __device__ ncclGin_C(ncclDevComm const& comm_, unsigned backendMask_, int contextIndex,
                       ncclGinResourceSharingMode resourceSharingMode_ = NCCL_GIN_RESOURCE_SHARING_GPU);
};

/************************ Barrier session storage ****************************/

struct ncclIrLsaBarrierSession {
  uint64_t storage[10];
};

struct ncclIrGinBarrierSession {
  uint64_t storage[12];
};

struct ncclIrBarrierSession {
  uint64_t storage[46];
};

extern "C" {

////////////////////////////////////////////////////////////////////////////////
// Core API
////////////////////////////////////////////////////////////////////////////////

/******************************* Team APIs ************************************/

__device__ ncclTeam ncclIrTeamWorld(ncclDevComm const* comm);
__device__ ncclTeam ncclIrTeamLsa(ncclDevComm const* comm);
__device__ ncclTeam ncclIrTeamCft(ncclDevComm const* comm, ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
__device__ ncclTeam ncclIrTeamCftMultimem(ncclDevComm const* comm);
__device__ int ncclIrTeamRankToWorld(ncclDevComm const* comm, ncclTeam team, int rank);
__device__ int ncclIrTeamRankToLsa(ncclDevComm const* comm, ncclTeam team, int rank);
__device__ ncclTeam ncclIrTeamRail(ncclDevComm const* comm);

/****************************** Window APIs ***********************************/

__device__ void* ncclIrGetLocalPointer(ncclWindow_t w, size_t offset);
__device__ void* ncclIrGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
__device__ void* ncclIrGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
__device__ void* ncclIrGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle mmHandle);
__device__ void* ncclIrGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const* comm);
__device__ void ncclIrGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam cftTeam,
                                   ncclDevComm const* comm, ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetPeerLeInfo(ncclWindow_t w, size_t offset, int peerWorld, ncclDevComm const* comm,
                                    ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclDevComm const* comm, ncclCftLeId* leId,
                                        size_t* leOffset);

/************************** Resource buffer APIs ******************************/

__device__ void* ncclIrGetResourceBufferLocalPointer(ncclDevComm const* comm, ncclDevResourceHandle h);
__device__ void* ncclIrGetResourceBufferLsaPointer(ncclDevComm const* comm, ncclDevResourceHandle h, int peer);
__device__ void* ncclIrGetResourceBufferPeerPointer(ncclDevComm const* comm, ncclDevResourceHandle h, ncclTeam team,
                                                    int peer);
__device__ void* ncclIrGetResourceBufferMultimemPointer(ncclDevComm const* comm, ncclDevResourceHandle h,
                                                        ncclMultimemHandle mmHandle);
__device__ void* ncclIrGetResourceBufferLsaMultimemPointer(ncclDevComm const* comm, ncclDevResourceHandle h);
__device__ void ncclIrGetResourceBufferCftLeInfo(ncclDevComm const* comm, ncclDevResourceHandle h, int peerCft,
                                                 ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetResourceBufferPeerLeInfo(ncclDevComm const* comm, ncclDevResourceHandle h, int peerWorld,
                                                  ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetResourceBufferMultimemLeInfo(ncclDevComm const* comm, ncclDevResourceHandle h,
                                                      ncclCftLeId* leId, size_t* leOffset);

/************************* ncclDevComm field accessors ***********************/

/*
 * ncclDevComm is a public C struct, the following accessors are deprecated and will be removed.
 */
__device__ int ncclIrDevCommRank(ncclDevComm const* comm);
__device__ int ncclIrDevCommNRanks(ncclDevComm const* comm);
__device__ int ncclIrDevCommLsaRank(ncclDevComm const* comm);
__device__ int ncclIrDevCommLsaSize(ncclDevComm const* comm);
__device__ ncclLsaBarrierHandle ncclIrDevCommLsaBarrier(ncclDevComm const* comm);
__device__ ncclGinBarrierHandle ncclIrDevCommRailGinBarrier(ncclDevComm const* comm);
__device__ ncclLsaBarrierHandle ncclIrDevCommHybridLsaBarrier(ncclDevComm const* comm);
__device__ ncclGinBarrierHandle ncclIrDevCommHybridRailGinBarrier(ncclDevComm const* comm);
__device__ ncclGinBarrierHandle ncclIrDevCommWorldGinBarrier(ncclDevComm const* comm);
__device__ ncclMultimemHandle ncclIrDevCommLsaMultimem(ncclDevComm const* comm);

/****************************** Peer pointer API ******************************/
__device__ void* ncclIrGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);

////////////////////////////////////////////////////////////////////////////////
// Cooperative group API
////////////////////////////////////////////////////////////////////////////////

__device__ void ncclIrCoopInitThread(ncclIrCoop* coop);
__device__ void ncclIrCoopInitWarp(ncclIrCoop* coop);
__device__ void ncclIrCoopInitLanes(ncclIrCoop* coop, uint32_t lane_mask);
__device__ void ncclIrCoopInitWarpSpan(ncclIrCoop* coop, int warp0, int nWarps, int id);
__device__ void ncclIrCoopInitCta(ncclIrCoop* coop);

__device__ int ncclIrCoopThreadRank(ncclIrCoop const* coop);
__device__ int ncclIrCoopSize(ncclIrCoop const* coop);
__device__ int ncclIrCoopNumThreads(ncclIrCoop const* coop);
__device__ void ncclIrCoopSync(ncclIrCoop* coop);

////////////////////////////////////////////////////////////////////////////////
// GIN API
////////////////////////////////////////////////////////////////////////////////

/************************** GIN initialization APIs ***************************/

// Helper init function that wraps placement new
__device__ void ncclGin_C_init(ncclGin_C* net, unsigned backendMask, ncclDevComm const* comm, int contextIndex);

// Helper init function with explicit resource sharing mode.
__device__ void ncclGin_C_initWithResourceSharingMode(ncclGin_C* net, unsigned backendMask, ncclDevComm const* comm,
                                                      int contextIndex, ncclGinResourceSharingMode resourceSharingMode);

/************************** GIN data movement APIs ****************************/

__device__ void ncclGinPut(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                           ncclWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal, ncclGinSignal_t signalId,
                           ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool isCounter, ncclGinCounter_t counterId,
                           ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                           cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);

__device__ void ncclGinPut_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                              ncclWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal,
                              ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                              bool isCounter, ncclGinCounter_t counterId, ncclIrCoop const* coop, bool isDescriptor,
                              ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                              cuda::thread_scope requiredRelease, uint32_t optFlags);

// Full C API counterpart of ncclGin::put. Unlike ncclGinPut_v2, this entry point can express VA and strong signals and
// handles windows containing HOST_NUMA segments.
__device__ void ncclGinPut_v3(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                              ncclWindow_t srcWin, size_t srcOffset, size_t bytes, ncclGinSignalType signalType,
                              ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId, bool isStrong,
                              ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool isCounter,
                              ncclGinCounter_t counterId, ncclIrCoop const* coop, bool isDescriptor,
                              ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                              cuda::thread_scope requiredRelease, uint32_t optFlags, ncclGinSegmentType_t segmentType);

__device__ void ncclGinPutValue(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                uint64_t value, size_t size, bool isSignal, ncclGinSignal_t signalId,
                                ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                cuda::thread_scope requiredRelease);

__device__ void ncclGinPutValue_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                   uint64_t value, size_t size, bool isSignal, ncclGinSignal_t signalId,
                                   ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                   bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                   cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                   uint32_t optFlags);

// Full C API counterpart of ncclGin::putValue with explicit indexed/VA and strong/weak signal selection.
__device__ void ncclGinPutValue_v3(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                   uint64_t value, size_t size, ncclGinSignalType signalType, ncclWindow_t signalWin,
                                   size_t signalOffset, ncclGinSignal_t signalId, bool isStrong,
                                   ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                   bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                   cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                   uint32_t optFlags);

__device__ void ncclGinGet(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd, size_t remoteOffset,
                           ncclWindow_t localWnd, size_t localOffset, size_t bytes, ncclIrCoop const* coop,
                           bool isDescriptor, ncclGinDescriptorSmem* descriptor, uint32_t optFlags);

// Full C API counterpart of ncclGin::get with explicit segment type selection.
__device__ void ncclGinGet_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd, size_t remoteOffset,
                              ncclWindow_t localWnd, size_t localOffset, size_t bytes, ncclIrCoop const* coop,
                              bool isDescriptor, ncclGinDescriptorSmem* descriptor, uint32_t optFlags,
                              ncclGinSegmentType_t segmentType);

/***************************** GIN signaling APIs *****************************/

__device__ void ncclGinSignal(ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId,
                              ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                              bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                              cuda::thread_scope requiredRelease);

__device__ void ncclGinSignal_v2(ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId,
                                 ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                 bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                 cuda::thread_scope requiredRelease, uint32_t optFlags);

// Full C API counterpart of ncclGin::signal with explicit indexed/VA and strong/weak signal selection.
__device__ void ncclGinSignal_v3(ncclGin_C* net, ncclTeam team, int peer, ncclGinSignalType signalType,
                                 ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId, bool isStrong,
                                 ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                 bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                 cuda::thread_scope requiredRelease, uint32_t optFlags);

__device__ uint64_t ncclGinReadSignal(ncclGin_C* net, ncclGinSignal_t signal, int bits, cuda::memory_order ord);

__device__ uint64_t ncclGinReadSignalVA(ncclGin_C* net, ncclWindow_t signalWindow, size_t signalOffset, int bits,
                                        cuda::memory_order ord);

__device__ void ncclGinWaitSignal(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal, uint64_t least,
                                  int bits, cuda::memory_order ord);

__device__ ncclResult_t ncclGinWaitSignalTimeout(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                                 uint64_t least, int bits, cuda::memory_order ord,
                                                 uint64_t timeoutCycles);

__device__ void ncclGinWaitSignalVA(ncclGin_C* net, ncclIrCoop const* coop, ncclWindow_t signalWindow,
                                    size_t signalOffset, uint64_t least, int bits, cuda::memory_order ord);

__device__ ncclResult_t ncclGinWaitSignalTimeoutVA(ncclGin_C* net, ncclIrCoop const* coop, ncclWindow_t signalWindow,
                                                   size_t signalOffset, uint64_t least, int bits,
                                                   cuda::memory_order ord, uint64_t timeoutCycles);

__device__ void ncclGinIncreaseSignalShadow(ncclGin_C* net, ncclGinSignal_t signal, uint64_t delta);

__device__ void ncclGinWaitSignalMeetShadow(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal, int bits,
                                            cuda::memory_order ord);

__device__ void ncclGinWaitSignalFollowShadow(ncclGin_C* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                              uint64_t leastDelta, uint64_t* before, uint64_t* delta, int bits,
                                              cuda::memory_order ord);

__device__ uint64_t* ncclGinGetSignalShadowPtr(ncclGin_C* net, ncclGinSignal_t signal);

__device__ void ncclGinResetSignal(ncclGin_C* net, ncclGinSignal_t signal);

__device__ void ncclGinResetSignalVA(ncclGin_C* net, ncclWindow_t signalWindow, size_t signalOffset);

/****************************** GIN counter APIs ******************************/

__device__ uint64_t ncclGinReadCounter(ncclGin_C* net, ncclGinCounter_t counter, int bits, cuda::memory_order ord);

__device__ void ncclGinWaitCounter(ncclGin_C* net, ncclIrCoop const* coop, ncclGinCounter_t counter, uint64_t least,
                                   int bits, cuda::memory_order ord);

__device__ ncclResult_t ncclGinWaitCounterTimeout(ncclGin_C* net, ncclIrCoop const* coop, ncclGinCounter_t counter,
                                                  uint64_t least, int bits, cuda::memory_order ord,
                                                  uint64_t timeoutCycles);

__device__ void ncclGinResetCounter(ncclGin_C* net, ncclGinCounter_t counter);

/************************ GIN flush and request APIs *************************/

__device__ void ncclGinFlush(ncclGin_C* net, ncclIrCoop const* coop, cuda::memory_order ord);

__device__ void ncclGinFlush_v2(ncclGin_C* net, ncclIrCoop const* coop, cuda::memory_order ord, bool isDescriptor,
                                ncclGinDescriptorSmem* descriptor);

__device__ ncclResult_t ncclGinFlushTimeout(ncclGin_C* net, ncclIrCoop const* coop, cuda::memory_order ord,
                                            bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                            uint64_t timeoutCycles);

__device__ void ncclGinFlushAsync(ncclGin_C* net, ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                  ncclIrCoop const* coop, uint32_t optFlags, bool isDescriptor,
                                  ncclGinDescriptorSmem* descriptor);

__device__ void ncclGinWait(ncclGin_C* net, ncclGinRequest_t* request, ncclIrCoop const* coop, bool isDescriptor,
                            ncclGinDescriptorSmem* descriptor, cuda::memory_order ord);

__device__ ncclResult_t ncclGinWaitTimeout(ncclGin_C* net, ncclGinRequest_t* request, ncclIrCoop const* coop,
                                           bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::memory_order ord,
                                           uint64_t timeoutCycles);

////////////////////////////////////////////////////////////////////////////////
// Barrier APIs
////////////////////////////////////////////////////////////////////////////////

/************************ Session struct size getters ***********************/

/*
 * Used by the Python device API to allocate session storage with the correct
 * size via llvm.alloca, without duplicating the C++ struct layout in Python.
 */
__device__ size_t ncclIrLsaBarrierSessionSize();
__device__ size_t ncclIrGinBarrierSessionSize();
__device__ size_t ncclIrBarrierSessionSize();

/************************* LSA Barrier Session APIs **************************/
__device__ void ncclIrLsaBarrierSessionInit(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                            ncclDevComm const* comm, ncclTeam team, ncclLsaBarrierHandle handle,
                                            uint32_t index, bool multimem = false, ncclMultimemHandle mmHandle = {});
__device__ void ncclIrLsaBarrierSessionArrive(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                              cuda::memory_order order);
__device__ void ncclIrLsaBarrierSessionWait(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                            cuda::memory_order order);
__device__ void ncclIrLsaBarrierSessionSync(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                            cuda::memory_order order);
// Collectively finalize the placement-new object and persist its epoch. Does not free storage.
__device__ void ncclIrLsaBarrierSessionDestroy(ncclIrLsaBarrierSession* session);

/************************* GIN Barrier Session APIs **************************/
__device__ void ncclIrGinBarrierSessionInit(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                            ncclGin_C const* net, ncclTeam team, ncclGinBarrierHandle handle,
                                            uint32_t index);

// All-contexts variant of session-init: rail/world/etc. signal/wait happens on context 0,
// fence iterates every GIN context on the comm.
__device__ void ncclIrGinBarrierSessionInitAllContexts(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                       ncclDevComm const* comm, ncclTeam team,
                                                       ncclGinBarrierHandle handle, uint32_t index);

__device__ void ncclIrGinBarrierSessionSync(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                            cuda::memory_order order,
                                            ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Finalize the placement-new object. Currently a no-op -- the underlying destructor is
// empty -- but required for lifetime symmetry with Init. Does not free storage.
__device__ void ncclIrGinBarrierSessionDestroy(ncclIrGinBarrierSession* session);

/*************************** Barrier Session APIs ****************************/
__device__ void ncclIrBarrierSessionInit(ncclIrBarrierSession* session, ncclIrCoop const* coop, ncclTeam innerTeam,
                                         ncclTeam outerTeam, ncclGin_C const* net,
                                         ncclLsaBarrierHandle const innerBarHandle,
                                         ncclGinBarrierHandle const outerBarHandle, uint32_t index,
                                         bool multimem = false, ncclMultimemHandle const innerMmHandle = {});

__device__ void ncclIrBarrierSessionSync(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                         cuda::memory_order order,
                                         ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Collectively finalize all nested sessions and persist the inner LSA epoch. Does not free storage.
__device__ void ncclIrBarrierSessionDestroy(ncclIrBarrierSession* session);

////////////////////////////////////////////////////////////////////////////////
// Reduce and copy APIs
////////////////////////////////////////////////////////////////////////////////

/******************************* LSA reduce-sum *******************************/
__device__ void ncclIrLsaReduceSum_I8(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, int8_t* dst,
                                      size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_U8(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, uint8_t* dst,
                                      size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_I32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, int32_t* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_U32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, uint32_t* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_I64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, int64_t* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_U64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, uint64_t* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, half* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, float* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, double* dst,
                                       size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_BF16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                        __nv_bfloat16* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F8E4M3(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          __nv_fp8_e4m3* dst, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSum_F8E5M2(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          __nv_fp8_e5m2* dst, size_t count, ncclTeam team);

/*************************** Multimem reduce-sum *****************************/
__device__ void ncclIrMultimemReduceSum_I32(ncclIrCoop const* coop, int32_t* mcSrc, int32_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_U32(ncclIrCoop const* coop, uint32_t* mcSrc, uint32_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_I64(ncclIrCoop const* coop, int64_t* mcSrc, int64_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_U64(ncclIrCoop const* coop, uint64_t* mcSrc, uint64_t* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F16(ncclIrCoop const* coop, half* mcSrc, half* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F32(ncclIrCoop const* coop, float* mcSrc, float* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_F64(ncclIrCoop const* coop, double* mcSrc, double* dst, size_t count);
__device__ void ncclIrMultimemReduceSum_BF16(ncclIrCoop const* coop, __nv_bfloat16* mcSrc, __nv_bfloat16* dst,
                                             size_t count);
__device__ void ncclIrMultimemReduceSum_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* dst,
                                               size_t count);
__device__ void ncclIrMultimemReduceSum_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* dst,
                                               size_t count);

/********************************** LSA copy *********************************/
__device__ void ncclIrLsaCopy_I8(ncclIrCoop const* coop, int8_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                 size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_U8(ncclIrCoop const* coop, uint8_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                 size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_I32(ncclIrCoop const* coop, int32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_U32(ncclIrCoop const* coop, uint32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_I64(ncclIrCoop const* coop, int64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_U64(ncclIrCoop const* coop, uint64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F16(ncclIrCoop const* coop, half* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F32(ncclIrCoop const* coop, float* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F64(ncclIrCoop const* coop, double* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_BF16(ncclIrCoop const* coop, __nv_bfloat16* src, ncclWindow_t dstWindow, size_t dstOffset,
                                   size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
                                     size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaCopy_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
                                     size_t dstOffset, size_t count, ncclTeam team);

/****************************** Multimem copy ********************************/
__device__ void ncclIrMultimemCopy_I32(ncclIrCoop const* coop, int32_t* src, int32_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_U32(ncclIrCoop const* coop, uint32_t* src, uint32_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_I64(ncclIrCoop const* coop, int64_t* src, int64_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_U64(ncclIrCoop const* coop, uint64_t* src, uint64_t* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F16(ncclIrCoop const* coop, half* src, half* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F32(ncclIrCoop const* coop, float* src, float* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F64(ncclIrCoop const* coop, double* src, double* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_BF16(ncclIrCoop const* coop, __nv_bfloat16* src, __nv_bfloat16* mcDst, size_t count);
__device__ void ncclIrMultimemCopy_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* src, __nv_fp8_e4m3* mcDst,
                                          size_t count);
__device__ void ncclIrMultimemCopy_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* src, __nv_fp8_e5m2* mcDst,
                                          size_t count);

/**************************** LSA reduce-sum-copy ****************************/
__device__ void ncclIrLsaReduceSumCopy_I8(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_U8(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_I32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_U32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_I64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_U64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_BF16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                            ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F8E4M3(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                              ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
__device__ void ncclIrLsaReduceSumCopy_F8E5M2(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                              ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);

/************************ Multimem reduce-sum-copy **************************/
__device__ void ncclIrMultimemReduceSumCopy_I32(ncclIrCoop const* coop, int32_t* mcSrc, int32_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_U32(ncclIrCoop const* coop, uint32_t* mcSrc, uint32_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_I64(ncclIrCoop const* coop, int64_t* mcSrc, int64_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_U64(ncclIrCoop const* coop, uint64_t* mcSrc, uint64_t* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F16(ncclIrCoop const* coop, half* mcSrc, half* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F32(ncclIrCoop const* coop, float* mcSrc, float* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F64(ncclIrCoop const* coop, double* mcSrc, double* mcDst, size_t count);
__device__ void ncclIrMultimemReduceSumCopy_BF16(ncclIrCoop const* coop, __nv_bfloat16* mcSrc, __nv_bfloat16* mcDst,
                                                 size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* mcDst,
                                                   size_t count);
__device__ void ncclIrMultimemReduceSumCopy_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* mcDst,
                                                   size_t count);

/*************************** Local reduce-sum-copy ***************************/
__device__ void ncclIrLocalReduceSumCopy_I8(ncclIrCoop const* coop, int nSrc, int8_t* srcBase, size_t srcDispl,
                                            int nDst, int8_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_U8(ncclIrCoop const* coop, int nSrc, uint8_t* srcBase, size_t srcDispl,
                                            int nDst, uint8_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_I32(ncclIrCoop const* coop, int nSrc, int32_t* srcBase, size_t srcDispl,
                                             int nDst, int32_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_U32(ncclIrCoop const* coop, int nSrc, uint32_t* srcBase, size_t srcDispl,
                                             int nDst, uint32_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_I64(ncclIrCoop const* coop, int nSrc, int64_t* srcBase, size_t srcDispl,
                                             int nDst, int64_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_U64(ncclIrCoop const* coop, int nSrc, uint64_t* srcBase, size_t srcDispl,
                                             int nDst, uint64_t* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F16(ncclIrCoop const* coop, int nSrc, half* srcBase, size_t srcDispl, int nDst,
                                             half* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F32(ncclIrCoop const* coop, int nSrc, float* srcBase, size_t srcDispl,
                                             int nDst, float* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F64(ncclIrCoop const* coop, int nSrc, double* srcBase, size_t srcDispl,
                                             int nDst, double* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_BF16(ncclIrCoop const* coop, int nSrc, __nv_bfloat16* srcBase, size_t srcDispl,
                                              int nDst, __nv_bfloat16* dstBase, size_t dstDispl, size_t count);
__device__ void ncclIrLocalReduceSumCopy_F8E4M3(ncclIrCoop const* coop, int nSrc, __nv_fp8_e4m3* srcBase,
                                                size_t srcDispl, int nDst, __nv_fp8_e4m3* dstBase, size_t dstDispl,
                                                size_t count);
__device__ void ncclIrLocalReduceSumCopy_F8E5M2(ncclIrCoop const* coop, int nSrc, __nv_fp8_e5m2* srcBase,
                                                size_t srcDispl, int nDst, __nv_fp8_e5m2* dstBase, size_t dstDispl,
                                                size_t count);

/*************************** TMA copy ***************************/
// Requires a coop initialized as Thread, Warp, or CTA.
__device__ void ncclIrLsaCopyTma_I8(ncclIrCoop const* coop, int8_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                    size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_U8(ncclIrCoop const* coop, uint8_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                    size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_I32(ncclIrCoop const* coop, int32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_U32(ncclIrCoop const* coop, uint32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_I64(ncclIrCoop const* coop, int64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_U64(ncclIrCoop const* coop, uint64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F16(ncclIrCoop const* coop, half* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F32(ncclIrCoop const* coop, float* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F64(ncclIrCoop const* coop, double* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_BF16(ncclIrCoop const* coop, __nv_bfloat16* src, ncclWindow_t dstWindow,
                                      size_t dstOffset, size_t count, ncclTeam team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
                                        size_t dstOffset, size_t count, ncclTeam team, char* smemPtr,
                                        int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
                                        size_t dstOffset, size_t count, ncclTeam team, char* smemPtr,
                                        int smemBytesTotal);

} // extern "C"

#endif // _NCCL_DEVICE_WRAPPER_H_
