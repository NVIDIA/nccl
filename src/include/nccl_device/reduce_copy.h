/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_REDUCE_COPY_H_
#define _NCCL_DEVICE_REDUCE_COPY_H_
#include "core.h"
#include "impl/reduce_copy__types.h"

// Forward declarations for public API functions
// Implementations are in impl/reduce_copy__funcs.h

#if NCCL_CHECK_CUDACC
// SERIES 1.x - Generic ReduceCopy with RedOp (LSA sources only)
template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename RedOp, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceLsaCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename RedOp, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceMultimemCopy(Coop, SrcLambda, int, DstLambda, int, RedOp const&, IntCount);

// SERIES 2.x - Sum-Specific ReduceCopy (lambda-based foundation)
template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

template<typename T, typename Coop, typename SrcLambda, typename DstLambda,
         typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumMultimemCopy(Coop, SrcLambda, int, DstLambda, int, IntCount);

// SERIES 3.x - ReduceSum (N->1)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, SrcLambda, int, T*, IntCount);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclTeam);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclDevComm_t);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclTeam);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclDevComm_t);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, ncclSymPtr<T>, T*, IntCount, ncclMultimemHandle);

// 3.3b] Multimem ReduceSum (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, T*, T*, IntCount);

template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSum(Coop, ncclWindow_t, size_t, T*, IntCount, ncclMultimemHandle);

// 3.4] Local ReduceSum (lambda-based)
template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop, SrcLambda, int, T*, IntCount);

// 3.5] Local ReduceSum (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSum(Coop, int, T*, size_t, T*, IntCount);

// SERIES 4.x - Copy/Broadcast (1->N)

// 4.1] LSA Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, DstLambda, int, IntCount);

// 4.2a] LSA Copy (with ncclSymPtr + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclTeam);

// 4.2b] LSA Copy (with ncclSymPtr + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclDevComm_t);

// 4.2c] LSA Copy (with window + offset + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclTeam);

// 4.2d] LSA Copy (with window + offset + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclDevComm_t);

// 4.3a] Multimem Copy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, ncclSymPtr<T>, IntCount, ncclMultimemHandle);

// 4.3b] Multimem Copy (with raw pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, T*, IntCount);

// 4.3c] Multimem Copy (with window + offset)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemCopy(Coop, T*, ncclWindow_t, size_t, IntCount, ncclMultimemHandle);

// 4.4] Local Copy (lambda-based)
template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop, T*, DstLambda, int, IntCount);

// 4.5] Local Copy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalCopy(Coop, T*, int, T*, size_t, IntCount);

// SERIES 5.x - ReduceSumCopy (N->M)

// 5.1a] LSA ReduceSumCopy (same team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclTeam);

// 5.1b] LSA ReduceSumCopy (with devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclSymPtr<T>, IntCount, ncclDevComm_t);

// 5.1c] LSA ReduceSumCopy (with windows + team)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclTeam);

// 5.1d] LSA ReduceSumCopy (with windows + devComm)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclWindow_t, size_t, ncclWindow_t, size_t, IntCount, ncclDevComm_t);

// 5.1e] LSA ReduceSumCopy (different teams)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumCopy(Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclTeam, IntCount);

// 5.2a] Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclMultimemHandle, IntCount);

// 5.2b] Multimem ReduceSumCopy (with raw pointers)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, T*, T*, IntCount);

// 5.2c] Multimem ReduceSumCopy (with windows)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumCopy(Coop, ncclWindow_t, size_t, ncclMultimemHandle, ncclWindow_t, size_t, ncclMultimemHandle, IntCount);

// 5.3a] LSA -> Multimem ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, ncclSymPtr<T>, ncclTeam, ncclSymPtr<T>, ncclMultimemHandle, IntCount);

// 5.3b] LSA -> Multimem ReduceSumCopy (with raw dst pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLsaReduceSumMultimemCopy(Coop, ncclSymPtr<T>, ncclTeam, T*, IntCount);

// 5.3c] Multimem -> LSA ReduceSumCopy (with ncclSymPtr)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, ncclSymPtr<T>, ncclMultimemHandle, ncclSymPtr<T>, ncclTeam, IntCount);

// 5.3d] Multimem -> LSA ReduceSumCopy (with raw src pointer)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclMultimemReduceSumLsaCopy(Coop, T*, ncclSymPtr<T>, ncclTeam, IntCount);

// 5.4] Local ReduceSumCopy (strided)
template<typename T, typename Coop, typename IntCount, int UNROLL=4*16/sizeof(T)>
NCCL_DEVICE_INLINE void ncclLocalReduceSumCopy(Coop, int, T*, size_t, int, T*, size_t, IntCount);

#endif // NCCL_CHECK_CUDACC

#endif // _NCCL_DEVICE_REDUCE_COPY_H_
