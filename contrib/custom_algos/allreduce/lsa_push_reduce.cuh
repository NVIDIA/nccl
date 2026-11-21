/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// lsa_push_reduce.cuh
//
// Element-wise sum of two 16-byte lines for the LSA push kernels.  The push
// kernels move data as 16-byte lines (float4) whatever the element type, so
// only this add depends on it:
//   float          4 elements per line
//   __half         8 elements per line (__hadd2)
//   __nv_bfloat16  8 elements per line (__hadd2; emulated below sm_80)
//
// The kernels sum in rank order on every rank, so all ranks get bit-identical
// results for every type.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

template <typename T> struct LsaPushSum;

template <> struct LsaPushSum<float> {
    static __device__ __forceinline__ float4 add(float4 a, float4 b) {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
};

template <> struct LsaPushSum<__half> {
    static __device__ __forceinline__ float4 add(float4 a, float4 b) {
        __half2*       pa = reinterpret_cast<__half2*>(&a);
        const __half2* pb = reinterpret_cast<const __half2*>(&b);
#pragma unroll
        for (int i = 0; i < 4; i++) pa[i] = __hadd2(pa[i], pb[i]);
        return a;
    }
};

template <> struct LsaPushSum<__nv_bfloat16> {
    static __device__ __forceinline__ float4 add(float4 a, float4 b) {
        __nv_bfloat162*       pa = reinterpret_cast<__nv_bfloat162*>(&a);
        const __nv_bfloat162* pb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
        for (int i = 0; i < 4; i++) pa[i] = __hadd2(pa[i], pb[i]);
        return a;
    }
};
