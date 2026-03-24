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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syslog.h>
#include <mutex>

#include "doca_verbs_cuda_wrapper.h"
#include "doca_gpunetio_log.hpp"

/* Function pointer types for CUDA device APIs */
typedef CUresult (*cuDeviceGetAttribute_t)(int *pi, CUdevice_attribute attrib, CUdevice dev);
typedef CUresult (*cuPointerSetAttribute_t)(const void *value, CUpointer_attribute attribute,
                                            CUdeviceptr ptr);
typedef CUresult (*cuMemGetHandleForAddressRange_t)(int *pHandle, CUdeviceptr dptr, size_t size,
                                                    CUmemRangeHandleType handleType,
                                                    unsigned long long flags);
typedef CUresult (*cuCtxGetCurrent_t)(CUcontext *pctx);

/* Global function pointers */
cuDeviceGetAttribute_t p_cuDeviceGetAttribute = nullptr;
cuPointerSetAttribute_t p_cuPointerSetAttribute = nullptr;
cuMemGetHandleForAddressRange_t p_cuMemGetHandleForAddressRange = nullptr;
cuCtxGetCurrent_t p_cuCtxGetCurrent = nullptr;

static void *cuda_handle = nullptr;

/* Helper function to get function pointer from libcuda */
static void *get_cuda_symbol(const char *symbol_name) {
    void *symbol = dlsym(cuda_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void doca_verbs_wrapper_init_once(int *ret) {
    /* Open libcuda.so */
    cuda_handle = dlopen("libcuda.so.1", RTLD_NOW);
    if (!cuda_handle) {
        cuda_handle = dlopen("libcuda.so", RTLD_NOW);
    }
    if (!cuda_handle) {
        DOCA_LOG(LOG_ERR, "Failed to open libcuda: %s\n", dlerror());
        *ret = -1;
        return;
    }

    /* Get function pointers */
    p_cuDeviceGetAttribute = (cuDeviceGetAttribute_t)get_cuda_symbol("cuDeviceGetAttribute");
    p_cuPointerSetAttribute = (cuPointerSetAttribute_t)get_cuda_symbol("cuPointerSetAttribute");
    p_cuMemGetHandleForAddressRange =
        (cuMemGetHandleForAddressRange_t)get_cuda_symbol("cuMemGetHandleForAddressRange");
    p_cuCtxGetCurrent = (cuCtxGetCurrent_t)get_cuda_symbol("cuCtxGetCurrent");

    /* Check if all symbols were found */
    if (!p_cuDeviceGetAttribute || !p_cuPointerSetAttribute || !p_cuMemGetHandleForAddressRange ||
        !p_cuCtxGetCurrent) {
        DOCA_LOG(LOG_ERR, "Failed to get all required CUDA symbols\n");
        dlclose(cuda_handle);
        cuda_handle = nullptr;
        *ret = -1;
        return;
    }

    *ret = 0;
}

static int init_cuda_wrapper(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_verbs_wrapper_init_once, &ret);
    return ret;
}

/* Wrapper function implementations */
CUresult doca_verbs_wrapper_cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    if (init_cuda_wrapper() != 0) return CUDA_ERROR_NOT_INITIALIZED;
    return p_cuDeviceGetAttribute(pi, attrib, dev);
}

CUresult doca_verbs_wrapper_cuPointerSetAttribute(const void *value, CUpointer_attribute attribute,
                                                  CUdeviceptr ptr) {
    if (init_cuda_wrapper() != 0) return CUDA_ERROR_NOT_INITIALIZED;
    return p_cuPointerSetAttribute(value, attribute, ptr);
}

CUresult doca_verbs_wrapper_cuMemGetHandleForAddressRange(int *pHandle, CUdeviceptr dptr,
                                                          size_t size,
                                                          CUmemRangeHandleType handleType,
                                                          unsigned long long flags) {
    if (init_cuda_wrapper() != 0) return CUDA_ERROR_NOT_INITIALIZED;
    return p_cuMemGetHandleForAddressRange(pHandle, dptr, size, handleType, flags);
}

CUresult doca_verbs_wrapper_cuCtxGetCurrent(CUcontext *pctx) {
    if (init_cuda_wrapper() != 0) return CUDA_ERROR_NOT_INITIALIZED;
    return p_cuCtxGetCurrent(pctx);
}
