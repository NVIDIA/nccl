/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER__IMPL_H_
#define _NCCL_DEVICE_WRAPPER__IMPL_H_

/*
 * NCCL Device API force instantiation and C style APIs for LLVM IR generation
 */

#include "nccl_device_wrapper.h"
#include <new>

#if NCCL_CHECK_CUDACC
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer) {
    return ncclGetPeerPointer(w, offset, tm, peer);
}

/* coop */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitThread(ncclCoopAny* coop) {
    ::new (coop) ncclCoopAny(ncclCoopThread());
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitWarp(ncclCoopAny* coop) {
    ::new (coop) ncclCoopAny(ncclCoopWarp());
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitLanes(ncclCoopAny* coop, uint32_t lane_mask) {
    ::new (coop) ncclCoopAny(ncclCoopLanes(lane_mask));
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id) {
    ::new (coop) ncclCoopAny(ncclCoopWarpSpan(warp0, nWarps, id));
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopAnyInitCta(ncclCoopAny* coop) {
    ::new (coop) ncclCoopAny(ncclCoopCta());
}

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclCoopThreadRank(const ncclCoopAny* coop) {
    return coop->thread_rank();
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclCoopSize(const ncclCoopAny* coop) {
    return coop->size();
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE int ncclCoopNumThreads(const ncclCoopAny* coop) {
    return coop->num_threads();
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclCoopSync(const ncclCoopAny* coop) {
    const_cast<ncclCoopAny*>(coop)->sync();
}

/* lsa barrier session */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem,
    ncclMultimemHandle mmHandle) {
    ::new (&(session->bar)) ncclLsaBarrierSession<ncclCoopAny>(coop, comm, team, handle, index, multimem, mmHandle);
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order) {
    session->bar.arrive(coop, order);
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order) {
    session->bar.wait(coop, order);
}
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order) {
    session->bar.sync(coop, order);
}

/* GIN barrier session */
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index) {
    ::new (&(session->bar)) ncclGinBarrierSession<ncclCoopAny>(coop, reinterpret_cast<ncclGin const&>(net), team, handle, index);
}

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence) {
    session->bar.sync(coop, order, fence);
}

/* Barrier Session*/
NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclBarrierSessionInit(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem, ncclMultimemHandle const innerMmHandle) {
    ::new (&(session->bar)) ncclBarrierSession<ncclCoopAny>(coop, innerTeam, outerTeam, reinterpret_cast<ncclGin const&>(net),
           innerBarHandle, outerBarHandle, index, multimem, innerMmHandle);
}

NCCL_IR_EXTERN_C NCCL_DEVICE_INLINE void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence) {
    session->bar.sync(coop, order, fence);
}
#endif

#endif // _NCCL_DEVICE_WRAPPER__IMPL_H_
