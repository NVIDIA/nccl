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
 * Production's __forceinline__ (__inline__ __attribute__((always_inline))) is
 * linkonce_odr and emits no symbol. Drop __inline__ so the device API has
 * external linkage: the bitcode lib emits symbols that consumers resolve from
 * libnccl_device.bc. Must precede the API includes below.
 */
#include "nccl_device/utility.h"
#undef NCCL_DEVICE_INLINE
#undef NCCL_HOST_DEVICE_INLINE
#ifdef __CUDACC__
#if defined(__NCCL_DEVICE_LTOIR_LIB__)
#define NCCL_DEVICE_INLINE __device__ __inline_hint__
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __inline_hint__
#elif defined(__clang_llvm_bitcode_lib__)
#define NCCL_DEVICE_INLINE __device__ __attribute__((always_inline))
#define NCCL_HOST_DEVICE_INLINE __host__ __device__ __attribute__((always_inline))
#else
#define NCCL_DEVICE_INLINE
#define NCCL_HOST_DEVICE_INLINE inline __attribute__((always_inline))
#endif
#endif

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
 * ncclDevComm is a public C struct, the following accessors are deprecated and will be removed.
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
    ncclGin_C const* net,
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
    ncclGin_C const* net,
    ncclLsaBarrierHandle const innerBarHandle,
    ncclGinBarrierHandle const outerBarHandle,
    uint32_t index,
    bool multimem=false, ncclMultimemHandle const innerMmHandle={});

NCCL_IR_EXTERN_C __device__ void ncclBarrierSessionSync(
    ncclBarrierSession_C* session,
    ncclCoopAny coop,
    cuda::memory_order order,
    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);

/* ReduceCopy APIs */
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_I8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int8_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_U8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint8_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_I32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int32_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_U32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint32_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_I64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    int64_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_U64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    uint64_t* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    half* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    float* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    double* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_BF16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_bfloat16* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F8E4M3(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_fp8_e4m3* dst, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSum_F8E5M2(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    __nv_fp8_e5m2* dst, size_t count, ncclTeam team);

NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_I32(
    ncclCoopAny coop, int32_t* mcSrc, int32_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_U32(
    ncclCoopAny coop, uint32_t* mcSrc, uint32_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_I64(
    ncclCoopAny coop, int64_t* mcSrc, int64_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_U64(
    ncclCoopAny coop, uint64_t* mcSrc, uint64_t* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F16(
    ncclCoopAny coop, half* mcSrc, half* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F32(
    ncclCoopAny coop, float* mcSrc, float* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F64(
    ncclCoopAny coop, double* mcSrc, double* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_BF16(
    ncclCoopAny coop, __nv_bfloat16* mcSrc, __nv_bfloat16* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* dst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSum_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* dst, size_t count);

NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_I8(
    ncclCoopAny coop, int8_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_U8(
    ncclCoopAny coop, uint8_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_I32(
    ncclCoopAny coop, int32_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_U32(
    ncclCoopAny coop, uint32_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_I64(
    ncclCoopAny coop, int64_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_U64(
    ncclCoopAny coop, uint64_t* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F16(
    ncclCoopAny coop, half* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F32(
    ncclCoopAny coop, float* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F64(
    ncclCoopAny coop, double* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_BF16(
    ncclCoopAny coop, __nv_bfloat16* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* src, ncclWindow_t dstWindow,
    size_t dstOffset, size_t count, ncclTeam team);

NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_I32(
    ncclCoopAny coop, int32_t* src, int32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_U32(
    ncclCoopAny coop, uint32_t* src, uint32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_I64(
    ncclCoopAny coop, int64_t* src, int64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_U64(
    ncclCoopAny coop, uint64_t* src, uint64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F16(
    ncclCoopAny coop, half* src, half* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F32(
    ncclCoopAny coop, float* src, float* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F64(
    ncclCoopAny coop, double* src, double* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_BF16(
    ncclCoopAny coop, __nv_bfloat16* src, __nv_bfloat16* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* src, __nv_fp8_e4m3* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* src, __nv_fp8_e5m2* mcDst, size_t count);

NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_I8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_U8(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_I32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_U32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_I64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_U64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F32(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F64(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_BF16(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F8E4M3(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);
NCCL_IR_EXTERN_C __device__ void ncclLsaReduceSumCopy_F8E5M2(
    ncclCoopAny coop, ncclWindow_t srcWindow, size_t srcOffset,
    ncclWindow_t dstWindow, size_t dstOffset, size_t count, ncclTeam team);

NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_I32(
    ncclCoopAny coop, int32_t* mcSrc, int32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_U32(
    ncclCoopAny coop, uint32_t* mcSrc, uint32_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_I64(
    ncclCoopAny coop, int64_t* mcSrc, int64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_U64(
    ncclCoopAny coop, uint64_t* mcSrc, uint64_t* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F16(
    ncclCoopAny coop, half* mcSrc, half* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F32(
    ncclCoopAny coop, float* mcSrc, float* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F64(
    ncclCoopAny coop, double* mcSrc, double* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_BF16(
    ncclCoopAny coop, __nv_bfloat16* mcSrc, __nv_bfloat16* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F8E4M3(
    ncclCoopAny coop, __nv_fp8_e4m3* mcSrc, __nv_fp8_e4m3* mcDst, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclMultimemReduceSumCopy_F8E5M2(
    ncclCoopAny coop, __nv_fp8_e5m2* mcSrc, __nv_fp8_e5m2* mcDst, size_t count);

NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_I8(
    ncclCoopAny coop, int nSrc, int8_t* srcBase, size_t srcDispl,
    int nDst, int8_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_U8(
    ncclCoopAny coop, int nSrc, uint8_t* srcBase, size_t srcDispl,
    int nDst, uint8_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_I32(
    ncclCoopAny coop, int nSrc, int32_t* srcBase, size_t srcDispl,
    int nDst, int32_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_U32(
    ncclCoopAny coop, int nSrc, uint32_t* srcBase, size_t srcDispl,
    int nDst, uint32_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_I64(
    ncclCoopAny coop, int nSrc, int64_t* srcBase, size_t srcDispl,
    int nDst, int64_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_U64(
    ncclCoopAny coop, int nSrc, uint64_t* srcBase, size_t srcDispl,
    int nDst, uint64_t* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F16(
    ncclCoopAny coop, int nSrc, half* srcBase, size_t srcDispl,
    int nDst, half* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F32(
    ncclCoopAny coop, int nSrc, float* srcBase, size_t srcDispl,
    int nDst, float* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F64(
    ncclCoopAny coop, int nSrc, double* srcBase, size_t srcDispl,
    int nDst, double* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_BF16(
    ncclCoopAny coop, int nSrc, __nv_bfloat16* srcBase, size_t srcDispl,
    int nDst, __nv_bfloat16* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F8E4M3(
    ncclCoopAny coop, int nSrc, __nv_fp8_e4m3* srcBase, size_t srcDispl,
    int nDst, __nv_fp8_e4m3* dstBase, size_t dstDispl, size_t count);
NCCL_IR_EXTERN_C __device__ void ncclLocalReduceSumCopy_F8E5M2(
    ncclCoopAny coop, int nSrc, __nv_fp8_e5m2* srcBase, size_t srcDispl,
    int nDst, __nv_fp8_e5m2* dstBase, size_t dstDispl, size_t count);

#endif // _NCCL_DEVICE_WRAPPER_H_
