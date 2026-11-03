/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_IR_H_
#define _NCCL_DEVICE_IR_H_

/*
 * NCCL Device API C ABI for LLVM bitcode and NVCC LTO IR
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "nccl_device/core.h"
#include "nccl_device/gin_barrier.h"

#include "nccl_device/impl/core__types.h"
#include "nccl_device/impl/lsa_barrier__types.h"
#include "nccl_device/impl/gin_barrier__types.h"

////////////////////////////////////////////////////////////////////////////////
// IR wrapper types
////////////////////////////////////////////////////////////////////////////////

/**************************** DevComm storage ********************************/

// Same-build opaque storage for a host-initialized ncclDevComm.
// Must outlive GIN and barrier sessions that reference it.
struct ncclIrDevComm {
  uint64_t storage[31];
};

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

struct ncclIrGin {
  // Opaque storage for a kernel-local ALL-mask ncclGin, initialized through the IR API.
  uint64_t storage[5];
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

__device__ void ncclIrTeamWorld(ncclIrDevComm const* comm, ncclTeam* outTeam);
__device__ void ncclIrTeamLsa(ncclIrDevComm const* comm, ncclTeam* outTeam);
__device__ void ncclIrTeamCft(ncclIrDevComm const* comm, ncclTeam* outTeam,
                              ncclCftTeamMode_t mode = NCCL_CFT_TEAM_FLAT);
__device__ void ncclIrTeamCftMultimem(ncclIrDevComm const* comm, ncclTeam* outTeam);
__device__ bool ncclIrTeamRankIsMember(ncclTeam const* a, ncclTeam const* b, int bPeer);
__device__ int ncclIrTeamRankToTeam(ncclTeam const* a, ncclTeam const* b, int bPeer);
__device__ int ncclIrTeamRankToWorld(ncclIrDevComm const* comm, ncclTeam const* team, int rank);
__device__ int ncclIrTeamRankToLsa(ncclIrDevComm const* comm, ncclTeam const* team, int rank);
__device__ void ncclIrTeamInnerFactor(ncclTeam const* parent, int innerSize, ncclTeam* outTeam);
__device__ void ncclIrTeamOuterFactor(ncclTeam const* parent, int innerSize, ncclTeam* outTeam);
__device__ int ncclIrTeamRankInDifference(ncclTeam const* parent, ncclTeam const* subset, int index);
__device__ void ncclIrTeamRail(ncclIrDevComm const* comm, ncclTeam* outTeam);

/****************************** Window APIs ***********************************/

__device__ void* ncclIrGetLocalPointer(ncclWindow_t w, size_t offset);
__device__ void* ncclIrGetLsaPointer(ncclWindow_t w, size_t offset, int peer);
__device__ void* ncclIrGetPeerPointer(ncclWindow_t w, size_t offset, int peer);
__device__ void* ncclIrGetMultimemPointer(ncclWindow_t w, size_t offset, ncclMultimemHandle const* mmHandle);
__device__ void* ncclIrGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclIrDevComm const* comm);
__device__ void ncclIrGetCftLeInfo(ncclWindow_t w, size_t offset, int peerCft, ncclTeam const* cftTeam,
                                   ncclIrDevComm const* comm, ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetPeerLeInfo(ncclWindow_t w, size_t offset, int peerWorld, ncclIrDevComm const* comm,
                                    ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetMultimemLeInfo(ncclWindow_t w, size_t offset, ncclIrDevComm const* comm, ncclCftLeId* leId,
                                        size_t* leOffset);

/************************** Resource buffer APIs ******************************/

__device__ size_t ncclIrGetResourceBufferOffset(ncclDevResourceHandle h);
__device__ void ncclIrGetResourceBuffer(ncclIrDevComm const* comm, ncclDevResourceHandle h, ncclWindow_t* window,
                                        size_t* offset);
__device__ void* ncclIrGetResourceBufferLocalPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h);
__device__ void* ncclIrGetResourceBufferLsaPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h, int peer);
__device__ void* ncclIrGetResourceBufferPeerPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                    ncclTeam const* team, int peer);
__device__ void* ncclIrGetResourceBufferMultimemPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                        ncclMultimemHandle const* mmHandle);
__device__ void* ncclIrGetResourceBufferLsaMultimemPointer(ncclIrDevComm const* comm, ncclDevResourceHandle h);
__device__ void ncclIrGetResourceBufferCftLeInfo(ncclIrDevComm const* comm, ncclDevResourceHandle h, int peerCft,
                                                 ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetResourceBufferPeerLeInfo(ncclIrDevComm const* comm, ncclDevResourceHandle h, int peerWorld,
                                                  ncclCftLeId* leId, size_t* leOffset);
__device__ void ncclIrGetResourceBufferMultimemLeInfo(ncclIrDevComm const* comm, ncclDevResourceHandle h,
                                                      ncclCftLeId* leId, size_t* leOffset);

/************************* ncclDevComm field accessors ***********************/

__device__ int ncclIrDevCommRank(ncclIrDevComm const* comm);
__device__ int ncclIrDevCommNRanks(ncclIrDevComm const* comm);
__device__ int ncclIrDevCommLsaRank(ncclIrDevComm const* comm);
__device__ int ncclIrDevCommLsaSize(ncclIrDevComm const* comm);
__device__ uint32_t ncclIrDevCommGinContextCount(ncclIrDevComm const* comm);

/****************************** Peer pointer API ******************************/
__device__ void* ncclIrGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam const* tm, int peer);

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

// Construct a kernel-local GIN object in opaque storage; comm must outlive its use.
__device__ void ncclIrGinInit(ncclIrGin* net, ncclIrDevComm const* comm, int contextIndex,
                              ncclGinResourceSharingMode resourceSharingMode = NCCL_GIN_RESOURCE_SHARING_GPU);

/************************** GIN data movement APIs ****************************/

__device__ void ncclIrGinPut(ncclIrGin const* net, ncclTeam const* team, int peer, ncclWindow_t dstWin,
                             size_t dstOffset, ncclWindow_t srcWin, size_t srcOffset, size_t bytes,
                             ncclGinSignalType signalType, ncclWindow_t signalWin, size_t signalOffset,
                             ncclGinSignal_t signalId, bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                             bool isCounter, ncclGinCounter_t counterId, ncclIrCoop const* coop, bool isDescriptor,
                             ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                             cuda::thread_scope requiredRelease, uint32_t optFlags, ncclGinSegmentType_t segmentType);

__device__ void ncclIrGinPutValue(ncclIrGin const* net, ncclTeam const* team, int peer, ncclWindow_t dstWin,
                                  size_t dstOffset, uint64_t value, size_t size, ncclGinSignalType signalType,
                                  ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId, bool isStrong,
                                  ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                  bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                  cuda::thread_scope requiredRelease, uint32_t optFlags);

__device__ void ncclIrGinGet(ncclIrGin const* net, ncclTeam const* team, int peer, ncclWindow_t remoteWnd,
                             size_t remoteOffset, ncclWindow_t localWnd, size_t localOffset, size_t bytes,
                             ncclIrCoop const* coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                             uint32_t optFlags, ncclGinSegmentType_t segmentType);

/***************************** GIN signaling APIs *****************************/

__device__ void ncclIrGinSignal(ncclIrGin const* net, ncclTeam const* team, int peer, ncclGinSignalType signalType,
                                ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId, bool isStrong,
                                ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclIrCoop const* coop,
                                bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::thread_scope givenRelease,
                                cuda::thread_scope requiredRelease, uint32_t optFlags);

__device__ uint64_t ncclIrGinReadSignal(ncclIrGin const* net, ncclGinSignal_t signal, int bits, cuda::memory_order ord);

__device__ uint64_t ncclIrGinReadSignalVA(ncclIrGin const* net, ncclWindow_t signalWindow, size_t signalOffset,
                                          int bits, cuda::memory_order ord);

__device__ void ncclIrGinWaitSignal(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                    uint64_t least, int bits, cuda::memory_order ord);

__device__ ncclResult_t ncclIrGinWaitSignalTimeout(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                                   uint64_t least, int bits, cuda::memory_order ord,
                                                   uint64_t timeoutCycles);

__device__ void ncclIrGinWaitSignalVA(ncclIrGin const* net, ncclIrCoop const* coop, ncclWindow_t signalWindow,
                                      size_t signalOffset, uint64_t least, int bits, cuda::memory_order ord);

__device__ ncclResult_t ncclIrGinWaitSignalTimeoutVA(ncclIrGin const* net, ncclIrCoop const* coop,
                                                     ncclWindow_t signalWindow, size_t signalOffset, uint64_t least,
                                                     int bits, cuda::memory_order ord, uint64_t timeoutCycles);

__device__ void ncclIrGinIncreaseSignalShadow(ncclIrGin const* net, ncclGinSignal_t signal, uint64_t delta);

__device__ void ncclIrGinWaitSignalMeetShadow(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                              int bits, cuda::memory_order ord);

__device__ void ncclIrGinWaitSignalFollowShadow(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinSignal_t signal,
                                                uint64_t leastDelta, uint64_t* before, uint64_t* delta, int bits,
                                                cuda::memory_order ord);

__device__ uint64_t* ncclIrGinGetSignalShadowPtr(ncclIrGin const* net, ncclGinSignal_t signal);

__device__ void ncclIrGinResetSignal(ncclIrGin const* net, ncclGinSignal_t signal);

__device__ void ncclIrGinResetSignalVA(ncclIrGin const* net, ncclWindow_t signalWindow, size_t signalOffset);

/****************************** GIN counter APIs ******************************/

__device__ uint64_t ncclIrGinReadCounter(ncclIrGin const* net, ncclGinCounter_t counter, int bits,
                                         cuda::memory_order ord);

__device__ void ncclIrGinWaitCounter(ncclIrGin const* net, ncclIrCoop const* coop, ncclGinCounter_t counter,
                                     uint64_t least, int bits, cuda::memory_order ord);

__device__ ncclResult_t ncclIrGinWaitCounterTimeout(ncclIrGin const* net, ncclIrCoop const* coop,
                                                    ncclGinCounter_t counter, uint64_t least, int bits,
                                                    cuda::memory_order ord, uint64_t timeoutCycles);

__device__ void ncclIrGinResetCounter(ncclIrGin const* net, ncclGinCounter_t counter);

/************************ GIN flush and request APIs *************************/

__device__ void ncclIrGinFlush(ncclIrGin const* net, ncclIrCoop const* coop, cuda::memory_order ord, bool isDescriptor,
                               ncclGinDescriptorSmem* descriptor);

__device__ ncclResult_t ncclIrGinFlushTimeout(ncclIrGin const* net, ncclIrCoop const* coop, cuda::memory_order ord,
                                              bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                              uint64_t timeoutCycles);

__device__ void ncclIrGinFlushAsync(ncclIrGin const* net, ncclTeam const* team, uint32_t peer,
                                    ncclGinRequest_t* outRequest, ncclIrCoop const* coop, uint32_t optFlags,
                                    bool isDescriptor, ncclGinDescriptorSmem* descriptor);

__device__ void ncclIrGinWait(ncclIrGin const* net, ncclGinRequest_t* request, ncclIrCoop const* coop,
                              bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::memory_order ord);

__device__ ncclResult_t ncclIrGinWaitTimeout(ncclIrGin const* net, ncclGinRequest_t* request, ncclIrCoop const* coop,
                                             bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                             cuda::memory_order ord, uint64_t timeoutCycles);

////////////////////////////////////////////////////////////////////////////////
// Barrier APIs
////////////////////////////////////////////////////////////////////////////////

/************************* LSA Barrier Session APIs **************************/
__device__ void ncclIrLsaBarrierSessionInit(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                            ncclIrDevComm const* comm, ncclTeam const* team,
                                            ncclLsaBarrierHandle const* handle, uint32_t index, bool multimem = false,
                                            ncclMultimemHandle const* mmHandle = nullptr);
__device__ void ncclIrLsaBarrierSessionInitLsa(ncclIrLsaBarrierSession* session, ncclIrCoop const* coop,
                                               ncclIrDevComm const* comm, uint32_t index, bool multimem = false);
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
                                            ncclIrGin const* net, ncclTeam const* team,
                                            ncclGinBarrierHandle const* handle, uint32_t index);

__device__ void ncclIrGinBarrierSessionInitRail(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                ncclIrGin const* net, uint32_t index);
__device__ void ncclIrGinBarrierSessionInitWorld(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                 ncclIrGin const* net, uint32_t index);

// All-contexts variant of session-init: rail/world/etc. signal/wait happens on context 0,
// fence iterates every GIN context on the comm.
__device__ void ncclIrGinBarrierSessionInitAllContexts(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                       ncclIrDevComm const* comm, ncclTeam const* team,
                                                       ncclGinBarrierHandle const* handle, uint32_t index);

__device__ void ncclIrGinBarrierSessionInitAllContextsRail(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                           ncclIrDevComm const* comm, uint32_t index);
__device__ void ncclIrGinBarrierSessionInitAllContextsWorld(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                                            ncclIrDevComm const* comm, uint32_t index);
__device__ void ncclIrGinBarrierSessionSync(ncclIrGinBarrierSession* session, ncclIrCoop const* coop,
                                            cuda::memory_order order,
                                            ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
// Finalize the placement-new object without freeing storage.
__device__ void ncclIrGinBarrierSessionDestroy(ncclIrGinBarrierSession* session);

/*************************** Barrier Session APIs ****************************/
__device__ void ncclIrBarrierSessionInit(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                         ncclTeam const* innerTeam, ncclTeam const* outerTeam, ncclIrGin const* net,
                                         ncclLsaBarrierHandle const* innerBarHandle,
                                         ncclGinBarrierHandle const* outerBarHandle, uint32_t index,
                                         bool multimem = false, ncclMultimemHandle const* innerMmHandle = nullptr);

__device__ void ncclIrBarrierSessionInitWorld(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                              ncclIrGin const* net, uint32_t index, bool multimem = false);
__device__ void ncclIrBarrierSessionInitLsa(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                            ncclIrDevComm const* comm, uint32_t index, bool multimem = false);
__device__ void ncclIrBarrierSessionInitRail(ncclIrBarrierSession* session, ncclIrCoop const* coop,
                                             ncclIrGin const* net, uint32_t index);
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
                                      size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_U8(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, uint8_t* dst,
                                      size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_I32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, int32_t* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_U32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, uint32_t* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_I64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, int64_t* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_U64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, uint64_t* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_F16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, half* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_F32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, float* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_F64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset, double* dst,
                                       size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_BF16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                        __nv_bfloat16* dst, size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_F8E4M3(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          __nv_fp8_e4m3* dst, size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSum_F8E5M2(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          __nv_fp8_e5m2* dst, size_t count, ncclTeam const* team);

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
                                 size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_U8(ncclIrCoop const* coop, uint8_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                 size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_I32(ncclIrCoop const* coop, int32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_U32(ncclIrCoop const* coop, uint32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_I64(ncclIrCoop const* coop, int64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_U64(ncclIrCoop const* coop, uint64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_F16(ncclIrCoop const* coop, half* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_F32(ncclIrCoop const* coop, float* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_F64(ncclIrCoop const* coop, double* src, ncclWindow_t dstWindow, size_t dstOffset,
                                  size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_BF16(ncclIrCoop const* coop, __nv_bfloat16* src, ncclWindow_t dstWindow, size_t dstOffset,
                                   size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
                                     size_t dstOffset, size_t count, ncclTeam const* team);
__device__ void ncclIrLsaCopy_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
                                     size_t dstOffset, size_t count, ncclTeam const* team);

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
                                          ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_U8(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                          ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_I32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_U32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_I64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_U64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_F16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_F32(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_F64(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                           ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                           ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_BF16(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                            ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                            ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_F8E4M3(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                              ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                              ncclTeam const* team);
__device__ void ncclIrLsaReduceSumCopy_F8E5M2(ncclIrCoop const* coop, ncclWindow_t srcWindow, size_t srcOffset,
                                              ncclWindow_t dstWindow, size_t dstOffset, size_t count,
                                              ncclTeam const* team);

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
                                    size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_U8(ncclIrCoop const* coop, uint8_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                    size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_I32(ncclIrCoop const* coop, int32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_U32(ncclIrCoop const* coop, uint32_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_I64(ncclIrCoop const* coop, int64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_U64(ncclIrCoop const* coop, uint64_t* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F16(ncclIrCoop const* coop, half* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F32(ncclIrCoop const* coop, float* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F64(ncclIrCoop const* coop, double* src, ncclWindow_t dstWindow, size_t dstOffset,
                                     size_t count, ncclTeam const* team, char* smemPtr, int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_BF16(ncclIrCoop const* coop, __nv_bfloat16* src, ncclWindow_t dstWindow,
                                      size_t dstOffset, size_t count, ncclTeam const* team, char* smemPtr,
                                      int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F8E4M3(ncclIrCoop const* coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
                                        size_t dstOffset, size_t count, ncclTeam const* team, char* smemPtr,
                                        int smemBytesTotal);
__device__ void ncclIrLsaCopyTma_F8E5M2(ncclIrCoop const* coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
                                        size_t dstOffset, size_t count, ncclTeam const* team, char* smemPtr,
                                        int smemBytesTotal);

} // extern "C"

#endif // _NCCL_DEVICE_IR_H_
