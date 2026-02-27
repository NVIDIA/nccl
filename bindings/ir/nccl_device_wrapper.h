/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef _NCCL_DEVICE_WRAPPER_H_
#define _NCCL_DEVICE_WRAPPER_H_

/*
 * NCCL Device API C-style wrapper functions
 */

#include "nccl_device.h"

/* Struct definitions */
struct ncclLsaBarrierSession_C {
    ncclLsaBarrierSession<ncclCoopAny> bar;
};

struct ncclGinBarrierSession_C {
    ncclGinBarrierSession<ncclCoopAny> bar;
};

struct ncclBarrierSession_C {
    ncclBarrierSession<ncclCoopAny> bar;
};

/* Peer pointer API */
NCCL_IR_EXTERN_C __device__ void* ncclGetPeerPointerTeam(ncclWindow_t w, size_t offset, ncclTeam tm, int peer);

/* Coop initialization and utility functions */
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitThread(ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitWarp(ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitLanes(ncclCoopAny* coop, uint32_t lane_mask);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitWarpSpan(ncclCoopAny* coop, int warp0, int nWarps, int id);
NCCL_IR_EXTERN_C __device__ void ncclCoopAnyInitCta(ncclCoopAny* coop);

NCCL_IR_EXTERN_C __device__ int ncclCoopThreadRank(const ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ int ncclCoopSize(const ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ int ncclCoopNumThreads(const ncclCoopAny* coop);
NCCL_IR_EXTERN_C __device__ void ncclCoopSync(const ncclCoopAny* coop);

/* LSA Barrier Session APIs */
NCCL_IR_EXTERN_C __device__ void ncclLsaBarrierSessionInit(
    ncclLsaBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclLsaBarrierHandle handle,
    uint32_t index,
    bool multimem = false,
    ncclMultimemHandle mmHandle = {});
NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionArrive(ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order);
NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionWait(ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order);
NCCL_IR_EXTERN_C __device__
void ncclLsaBarrierSessionSync(ncclLsaBarrierSession_C* session, ncclCoopAny coop, cuda::memory_order order);

/* GIN Barrier Session APIs */
NCCL_IR_EXTERN_C __device__ void ncclGinBarrierSessionInit(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclGin_C net,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

NCCL_IR_EXTERN_C __device__ void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence);

/* Barrier Session APIs */
NCCL_IR_EXTERN_C __device__ void ncclBarrierSessionInit(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    ncclTeam innerTeam,
    ncclTeam outerTeam,
    ncclGin_C net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem=false, ncclMultimemHandle const innerMmHandle={});

NCCL_IR_EXTERN_C __device__ void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence);

#endif // _NCCL_DEVICE_WRAPPER_H_

