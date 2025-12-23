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

#ifndef DOCA_VERBS_CUDA_WRAPPER_H
#define DOCA_VERBS_CUDA_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef DOCA_VERBS_USE_CUDA_WRAPPER

/* CUDA type declarations for builds without cuda.h */
typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_INITIALIZED = 3,
} CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef enum CUmemRangeHandleType_enum {
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 0x1,
    CU_MEM_RANGE_HANDLE_TYPE_MAX = 0x7FFFFFFF
} CUmemRangeHandleType;

typedef enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS =
        6, /**< Synchronize every synchronous memory operation initiated on this region */
} CUpointer_attribute;

typedef enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED =
        124, /**< Device supports buffer sharing with dma_buf mechanism. */
} CUdevice_attribute;

typedef void *CUcontext;

/* Wrapper function declarations */
CUresult doca_verbs_wrapper_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
CUresult doca_verbs_wrapper_cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                                                  CUdeviceptr ptr);
CUresult doca_verbs_wrapper_cuMemGetHandleForAddressRange(int *pHandle, CUdeviceptr dptr,
                                                          size_t size,
                                                          CUmemRangeHandleType handleType,
                                                          unsigned long long flags);
CUresult doca_verbs_wrapper_cuCtxGetCurrent(CUcontext *pctx);

/* Initialization function */
int doca_cuda_wrapper_init(void);

#else

#include <cuda.h>

/* Direct API calls when wrapper is not enabled */
#define doca_verbs_wrapper_cuDeviceGetAttribute cuDeviceGetAttribute
#define doca_verbs_wrapper_cuPointerSetAttribute cuPointerSetAttribute
#define doca_verbs_wrapper_cuMemGetHandleForAddressRange cuMemGetHandleForAddressRange
#define doca_verbs_wrapper_cuCtxGetCurrent cuCtxGetCurrent

/* No initialization needed when wrapper is not enabled */
#define doca_cuda_wrapper_init() 0

#endif

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_CUDA_WRAPPER_H */
