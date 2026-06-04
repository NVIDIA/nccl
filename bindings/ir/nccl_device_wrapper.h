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
#include "nccl_device/coop.h"
#include "nccl_device/core.h"
#include "nccl_device/ll_a2a.h"
#include "nccl_device/lsa_barrier.h"
#include "nccl_device/gin_barrier.h"
#include "nccl_device/barrier.h"
#include "nccl_device/ptr.h"
#include "nccl_device/reduce_copy.h"

#include "nccl_device/impl/core__types.h"
#include "nccl_device/impl/comm__types.h"
#include "nccl_device/impl/ll_a2a__types.h"
#include "nccl_device/impl/lsa_barrier__types.h"
#include "nccl_device/impl/gin__types.h"
#include "nccl_device/impl/gin_barrier__types.h"
#include "nccl_device/impl/barrier__types.h"
#include "nccl_device/impl/ptr__types.h"
#include "nccl_device/impl/reduce_copy__types.h"

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

/* Session struct size getters
 *
 * Used by the Python device API to allocate session storage with the correct
 * size via llvm.alloca, without duplicating the C++ struct layout in Python.
 */
NCCL_IR_EXTERN_C __device__ size_t ncclLsaBarrierSession_C_size();
NCCL_IR_EXTERN_C __device__ size_t ncclGinBarrierSession_C_size();
NCCL_IR_EXTERN_C __device__ size_t ncclBarrierSession_C_size();

/* ncclDevComm field accessors
 *
 * ncclDevComm is a public C struct, but its full layout (~200 bytes with
 * embedded arrays and structs) is not mirrored in Python. The Python device
 * layer reads its public fields through these accessor functions.
 */
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_Rank(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_NRanks(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_LsaRank(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ int                  ncclDevComm_LsaSize(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclLsaBarrierHandle ncclDevComm_LsaBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclGinBarrierHandle ncclDevComm_RailGinBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclLsaBarrierHandle ncclDevComm_HybridLsaBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclGinBarrierHandle ncclDevComm_HybridRailGinBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclGinBarrierHandle ncclDevComm_WorldGinBarrier(ncclDevComm const* comm);
NCCL_IR_EXTERN_C __device__ ncclMultimemHandle   ncclDevComm_LsaMultimem(ncclDevComm const* comm);

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

// All-contexts variant of session-init: rail/world/etc. signal/wait happens on context 0,
// fence iterates every GIN context on the comm.
NCCL_IR_EXTERN_C __device__ void ncclGinBarrierSessionInitAllContexts(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    ncclDevComm const& comm,
    ncclTeam team,
    ncclGinBarrierHandle handle,
    uint32_t index);

NCCL_IR_EXTERN_C __device__ void ncclGinBarrierSessionSync(
    ncclGinBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

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
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

#endif // _NCCL_DEVICE_WRAPPER_H_
