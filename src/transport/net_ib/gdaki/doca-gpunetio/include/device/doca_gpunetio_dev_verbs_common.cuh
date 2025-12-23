/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file doca_gpunetio_dev_verbs_common.cuh
 * @brief GDAKI common device structs and functions
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_DEV_VERBS_COMMON_H
#define DOCA_GPUNETIO_DEV_VERBS_COMMON_H

#include <stdio.h>
#include <stdint.h>
#include <cuda/atomic>
#include <math.h>

#include "../common/doca_gpunetio_verbs_dev.h"

#if __CUDA_ARCH__ >= 1000
#define DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE 1
#endif

#if __CUDA_ARCH__ >= 900
#define DOCA_GPUNETIO_VERBS_HAS_TMA_COPY 1
#endif

#if CUDA_VERSION >= 12020
#define DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO 1
#else
#warning "warning: doca_gpunetio should be used with a CUDA version >= 12020."
#endif

#if CUDA_VERSION >= 12080 && __CUDA_ARCH__ >= 900
#define DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX 1
#endif

/**
 * @brief Queries the global timer
 *
 * @return The value of the global timer
 */
__device__ static __forceinline__ uint64_t doca_gpu_dev_verbs_query_globaltimer() {
    uint64_t ret;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(ret)::"memory");
    return ret;
}

__device__ static __forceinline__ unsigned int doca_gpu_dev_verbs_get_lane_id() {
    unsigned int ret;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(ret));
    return ret;
}

__device__ static __forceinline__ uint64_t doca_gpu_dev_verbs_bswap64(uint64_t x) {
    uint64_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        ".reg .b32 lo;\n\t"
        ".reg .b32 hi;\n\t"
        ".reg .b32 new_lo;\n\t"
        ".reg .b32 new_hi;\n\t"
        "mov.b32 mask, 0x0123;\n\t"
        "mov.b64 {lo,hi}, %1;\n\t"
        "prmt.b32 new_hi, lo, ign, mask;\n\t"
        "prmt.b32 new_lo, hi, ign, mask;\n\t"
        "mov.b64 %0, {new_lo,new_hi};\n\t"
        "}"
        : "=l"(ret)
        : "l"(x));
    return ret;
}

__device__ static __forceinline__ uint32_t doca_gpu_dev_verbs_bswap32(uint32_t x) {
    uint32_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b32 mask;\n\t"
        ".reg .b32 ign;\n\t"
        "mov.b32 mask, 0x0123;\n\t"
        "prmt.b32 %0, %1, ign, mask;\n\t"
        "}"
        : "=r"(ret)
        : "r"(x));
    return ret;
}

__device__ static __forceinline__ uint16_t doca_gpu_dev_verbs_bswap16(uint16_t x) {
    uint16_t ret;
    asm volatile(
        "{\n\t"
        ".reg .b8 hi;\n\t"
        ".reg .b8 lo;\n\t"
        "mov.b16 {hi, lo}, %1;\n\t"
        "mov.b16 %0, {lo, hi};\n\t"
        "}"
        : "=h"(ret)
        : "h"(x));
    return ret;
}

#ifdef DOCA_GPUNETIO_VERBS_HAS_STORE_RELAXED_MMIO
__device__ static __forceinline__ void doca_gpu_dev_verbs_store_relaxed_mmio(uint64_t *ptr,
                                                                             uint64_t val) {
    asm volatile("st.mmio.relaxed.sys.global.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}
#endif

template <enum doca_gpu_dev_verbs_sync_scope sync_scope>
__device__ static __forceinline__ void doca_gpu_dev_verbs_fence_acquire() {
#ifdef DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("fence.acquire.cta;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("fence.acquire.gpu;");
    else
        asm volatile("fence.acquire.sys;");
#else
    // fence.acquire is not available in PTX. Emulate that with st.release.
    uint32_t dummy;
    const uint32_t val = 0;
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("ld.acquire.cta.b32 %0, [%1];" : : "r"(val), "l"(&dummy));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("ld.acquire.gpu.b32 %0, [%1];" : : "r"(val), "l"(&dummy));
    else
        asm volatile("ld.acquire.sys.b32 %0, [%1];" : : "r"(val), "l"(&dummy));
#endif
}

template <enum doca_gpu_dev_verbs_sync_scope sync_scope>
__device__ static __forceinline__ void doca_gpu_dev_verbs_fence_release() {
#ifdef DOCA_GPUNETIO_VERBS_HAS_FENCE_ACQUIRE_RELEASE_PTX
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("fence.release.cta;");
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("fence.release.gpu;");
    else
        asm volatile("fence.release.sys;");
#else
    // fence.release is not available in PTX. Emulate that with st.release.
    uint32_t dummy;
    const uint32_t val = 0;
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA)
        asm volatile("st.release.cta.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
    else if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
    else
        asm volatile("st.release.sys.u32 [%0], %1;" : : "l"(&dummy), "r"(val));
#endif
}

#ifdef DOCA_GPUNETIO_VERBS_HAS_ASYNC_STORE_RELEASE
template <enum doca_gpu_dev_verbs_sync_scope sync_scope>
__device__ static __forceinline__ void doca_gpu_dev_verbs_async_store_release(uint32_t *ptr,
                                                                              uint32_t val) {
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("st.async.mmio.release.gpu.b32 [%0], %1;" : : "l"(ptr), "r"(val));
    else
        asm volatile("st.async.mmio.release.sys.global.b32 [%0], %1;" : : "l"(ptr), "r"(val));
}

template <enum doca_gpu_dev_verbs_sync_scope sync_scope>
__device__ static __forceinline__ void doca_gpu_dev_verbs_async_store_release(uint64_t *ptr,
                                                                              uint64_t val) {
    if (sync_scope == DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU)
        asm volatile("st.async.mmio.release.gpu.global.b64 [%0], %1;" : : "l"(ptr), "l"(val));
    else
        asm volatile("st.async.mmio.release.sys.global.b64 [%0], %1;" : : "l"(ptr), "l"(val));
}
#endif

__device__ static __forceinline__ bool doca_gpu_dev_verbs_isaligned(void *ptr, size_t alignment) {
    bool status;
    status = (((uintptr_t)ptr & (alignment - 1)) == 0);
    return status;
}

/**
 * @brief Copy data from src to dst. The data must have natural alignment with it's size.
 *
 * @param dst - Destination pointer
 * @param src - Source pointer
 * @param bytes - Number of bytes to copy
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_memcpy_aligned_data(void *dst, void *src,
                                                                              size_t bytes) {
    size_t remaining_bytes = bytes;
    size_t copied_size;
    while (remaining_bytes > 0) {
        if (remaining_bytes >= sizeof(uint32_t)) {
            *(uint32_t *)dst = *(uint32_t *)src;
            copied_size = sizeof(uint32_t);
        } else if (remaining_bytes >= sizeof(uint16_t)) {
            *(uint16_t *)dst = *(uint16_t *)src;
            copied_size = sizeof(uint16_t);
        } else {
            *(uint8_t *)dst = *(uint8_t *)src;
            copied_size = sizeof(uint8_t);
        }
        remaining_bytes -= copied_size;
        dst = (void *)((uintptr_t)dst + copied_size);
        src = (void *)((uintptr_t)src + copied_size);
    }
}

/**
 * @brief Copy data from src to dst. The data may or may not have natural alignment with it's size.
 *
 * @param dst - Destination pointer
 * @param src - Source pointer
 * @param bytes - Number of bytes to copy
 */
__device__ static __forceinline__ void doca_gpu_dev_verbs_memcpy_data(void *dst, void *src,
                                                                      size_t bytes) {
    size_t remaining_bytes = bytes;
    size_t copied_size;
    while (remaining_bytes > 0) {
        if (doca_gpu_dev_verbs_isaligned(dst, sizeof(uint64_t)) &&
            doca_gpu_dev_verbs_isaligned(src, sizeof(uint64_t)) &&
            remaining_bytes >= sizeof(uint64_t)) {
            *(uint64_t *)dst = *(uint64_t *)src;
            copied_size = sizeof(uint64_t);
        } else if (doca_gpu_dev_verbs_isaligned(dst, sizeof(uint32_t)) &&
                   doca_gpu_dev_verbs_isaligned(src, sizeof(uint32_t)) &&
                   remaining_bytes >= sizeof(uint32_t)) {
            *(uint32_t *)dst = *(uint32_t *)src;
            copied_size = sizeof(uint32_t);
        } else if (doca_gpu_dev_verbs_isaligned(dst, sizeof(uint16_t)) &&
                   doca_gpu_dev_verbs_isaligned(src, sizeof(uint16_t)) &&
                   remaining_bytes >= sizeof(uint16_t)) {
            *(uint16_t *)dst = *(uint16_t *)src;
            copied_size = sizeof(uint16_t);
        } else {
            *(uint8_t *)dst = *(uint8_t *)src;
            copied_size = sizeof(uint8_t);
        }
        remaining_bytes -= copied_size;
        dst = (void *)((uintptr_t)dst + copied_size);
        src = (void *)((uintptr_t)src + copied_size);
    }
}

template <typename T>
__device__ static __forceinline__ void doca_gpu_dev_verbs_memcpy_inl_aligned_data(T *dst, T *src,
                                                                                  size_t bytes) {
    size_t remaining_bytes = bytes;
    const size_t copied_size = sizeof(T);
    while (remaining_bytes > 0) {
        remaining_bytes -= copied_size;
        dst = (void *)((uintptr_t)dst + copied_size);
        src = (void *)((uintptr_t)src + copied_size);
    }
}

template <typename T, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode,
          bool need_fence_acquire = false>
__device__ static __forceinline__ T doca_gpu_dev_verbs_atomic_max(T *ptr, T val) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        T old_val = *ptr;
        *ptr = max(old_val, val);
        return old_val;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
        return ptr_aref.fetch_max(
            val, need_fence_acquire ? cuda::memory_order_acquire : cuda::memory_order_relaxed);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
        return ptr_aref.fetch_max(
            val, need_fence_acquire ? cuda::memory_order_acquire : cuda::memory_order_relaxed);
    }
    return 0;
}

template <typename T, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ T doca_gpu_dev_verbs_atomic_add(T *ptr, T val) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        T old_val = *ptr;
        *ptr = old_val + val;
        return old_val;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<T, cuda::thread_scope_block> ptr_aref(*ptr);
        return ptr_aref.fetch_add(val, cuda::memory_order_relaxed);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<T, cuda::thread_scope_device> ptr_aref(*ptr);
        return ptr_aref.fetch_add(val, cuda::memory_order_relaxed);
    }
    return 0;
}

template <typename T, enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ T doca_gpu_dev_verbs_atomic_read(T *ptr) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
        return *ptr;
    else
        return READ_ONCE(*ptr);
}

/**
 * @brief Lock a resource
 *
 * @param lock - Pointer to the lock
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ void doca_gpu_dev_verbs_lock(int *lock) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        *lock = 1;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        while (atomicCAS_block(lock, 0, 1) != 0) continue;
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA>();
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        while (atomicCAS(lock, 0, 1) != 0) continue;
        doca_gpu_dev_verbs_fence_acquire<DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU>();
    }
}

/**
 * @brief Unlock a resource
 *
 * @param lock - Pointer to the lock
 */
template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ void doca_gpu_dev_verbs_unlock(int *lock) {
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE) {
        *lock = 0;
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA) {
        cuda::atomic_ref<int, cuda::thread_scope_block> lock_aref(*lock);
        lock_aref.store(0, cuda::memory_order_release);
    } else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU) {
        cuda::atomic_ref<int, cuda::thread_scope_device> lock_aref(*lock);
        lock_aref.store(0, cuda::memory_order_release);
    }
}

__device__ static __forceinline__ uint8_t doca_gpu_dev_verbs_load_relaxed_sys_global(uint8_t *ptr) {
    uint16_t ret;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b8 %0, [%1];" : "=h"(ret) : "l"(ptr));
    return (uint8_t)ret;
}

__device__ static __forceinline__ uint32_t
doca_gpu_dev_verbs_load_relaxed_sys_global(uint32_t *ptr) {
    uint32_t ret;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b32 %0, [%1];" : "=r"(ret) : "l"(ptr));
    return ret;
}

__device__ static __forceinline__ uint64_t
doca_gpu_dev_verbs_load_relaxed_sys_global(uint64_t *ptr) {
    uint64_t ret;
    asm volatile("ld.relaxed.sys.global.L1::no_allocate.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

template <enum doca_gpu_dev_verbs_resource_sharing_mode resource_sharing_mode>
__device__ static __forceinline__ uint64_t doca_gpu_dev_verbs_load_relaxed(uint64_t *ptr) {
    uint64_t ret = 0;
    if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE)
        ret = *ptr;
    else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA)
        asm volatile("ld.relaxed.cta.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    else if (resource_sharing_mode == DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU)
        asm volatile("ld.relaxed.gpu.b64 %0, [%1];" : "=l"(ret) : "l"(ptr));
    return ret;
}

/**
 * @brief Calculate the ceiling of x / y, where y is (2^denominator_shift)
 *
 * @param x - Numerator
 * @param denominator_shift - Denominator shift (y = 2^denominator_shift)
 * @return The ceiling of x / y
 */
__device__ static __forceinline__ uint64_t
doca_gpu_dev_verbs_div_ceil_aligned_pow2(uint64_t x, unsigned int denominator_shift) {
    uint64_t y = 1ULL << denominator_shift;
    return ((x & ~(y - 1)) >> denominator_shift) + (!!(x & (y - 1)));
}

/**
 * @brief Calculate the ceiling of x / y, where y is (2^denominator_shift).
 * The result must fit in 32 bits. This is a faster implementation than gdaki_div_ceil_aligned_pow2.
 *
 * @param x - Numerator
 * @param denominator_shift - Denominator shift (y = 2^denominator_shift)
 * @return The ceiling of x / y
 */
__device__ static __forceinline__ uint32_t
doca_gpu_dev_verbs_div_ceil_aligned_pow2_32bits(uint64_t x, int denominator_shift) {
    return uint32_t(x >> denominator_shift) + !!__funnelshift_r(0, uint32_t(x), denominator_shift);
}

#endif /* DOCA_GPUNETIO_DEV_VERBS_COMMON_H */
