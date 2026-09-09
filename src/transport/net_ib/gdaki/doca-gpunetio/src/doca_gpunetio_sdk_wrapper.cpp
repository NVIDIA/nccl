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
#include <string.h>

#include "doca_gpunetio_log.hpp"
#include "doca_gpunetio_sdk_wrapper.h"

/*
 * Enable DOCA SDK log errors on stderr.
 * Disabled by default to avoid extra-prints on screen.
 */
#define DOCA_GPUNETIO_SDK_WRAPPER_ENABLE_DEBUG 0

#ifdef __cplusplus
extern "C" {
#endif

/* Function pointer types for DOCA GPUNetIO SDK APIs */
typedef doca_error_t (*doca_gpu_create_t)(const char *gpu_bus_id, void **gpu_dev);
typedef doca_error_t (*doca_gpu_destroy_t)(void *gpu_dev);
typedef doca_error_t (*doca_gpu_mem_alloc_t)(void *gpu_dev, size_t size, size_t alignment,
                                             enum doca_gpu_mem_type mtype, void **memptr_gpu,
                                             void **memptr_cpu);
typedef doca_error_t (*doca_gpu_mem_free_t)(void *gpu, void *memptr_gpu);
typedef doca_error_t (*doca_gpu_dmabuf_fd_t)(void *gpu_dev, void *memptr_gpu, size_t size,
                                             int *dmabuf_fd);
typedef doca_error_t (*doca_gpu_verbs_req_notify_cq_t)(void *gpu_dev, void *verbs_cq);

#if DOCA_GPUNETIO_SDK_WRAPPER_ENABLE_DEBUG == 1
typedef doca_error_t (*doca_log_backend_create_with_file_sdk_t)(FILE *fptr, void **backend);
static doca_log_backend_create_with_file_sdk_t p_doca_log_backend_create_with_file_sdk = nullptr;
#endif

/* Global function pointers */
static doca_gpu_create_t p_doca_gpu_create = nullptr;
static doca_gpu_destroy_t p_doca_gpu_destroy = nullptr;
static doca_gpu_mem_alloc_t p_doca_gpu_mem_alloc = nullptr;
static doca_gpu_mem_free_t p_doca_gpu_mem_free = nullptr;
static doca_gpu_dmabuf_fd_t p_doca_gpu_dmabuf_fd = nullptr;
static doca_gpu_verbs_req_notify_cq_t p_doca_gpu_verbs_req_notify_cq = nullptr;

static void *common_handle = nullptr;
static void *verbs_handle = nullptr;
static void *gpunetio_handle = nullptr;

/* Helper function to get function pointer from libcuda */
static void *get_gpunetio_sdk_symbol(const char *symbol_name) {
    void *symbol = dlsym(gpunetio_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void doca_gpunetio_sdk_wrapper_init(int *ret) {
    char libcommon_path[2048];
    char libverbs_path[2048];
    char libgpunetio_path[2048];

    memset(libcommon_path, '\0', 2048);
    memset(libverbs_path, '\0', 2048);
    memset(libgpunetio_path, '\0', 2048);

    /*
     * libdoca_gpunetio.so depends on libdoca_verbs.so which depends on libdoca_common.so
     * Assuming LD_LIBRARY_PATH doesn't include the DOCA SDK libs directories, here we need to
     * explicitely dlopen all dependencies with RTLD_GLOBAL.
     */
    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libcommon_path, 2048, "%s", "libdoca_common.so");
    else
        snprintf(libcommon_path, 2048, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_common.so");

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libverbs_path, 2048, "%s", "libdoca_verbs.so");
    else
        snprintf(libverbs_path, 2048, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_verbs.so");

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libgpunetio_path, 2048, "%s", "libdoca_gpunetio.so");
    else
        snprintf(libgpunetio_path, 2048, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_gpunetio.so");

    common_handle = dlopen(libcommon_path, RTLD_NOW | RTLD_GLOBAL);
    if (!common_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_common.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libcommon_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    verbs_handle = dlopen(libverbs_path, RTLD_NOW | RTLD_GLOBAL);
    if (!verbs_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_verbs.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libverbs_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    gpunetio_handle = dlopen(libgpunetio_path, RTLD_NOW | RTLD_LOCAL);
    if (!gpunetio_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_gpunetio.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libgpunetio_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    /* Get function pointers */
    p_doca_gpu_create = (doca_gpu_create_t)get_gpunetio_sdk_symbol("doca_gpu_create");
    p_doca_gpu_destroy = (doca_gpu_destroy_t)get_gpunetio_sdk_symbol("doca_gpu_destroy");
    p_doca_gpu_mem_alloc = (doca_gpu_mem_alloc_t)get_gpunetio_sdk_symbol("doca_gpu_mem_alloc");
    p_doca_gpu_mem_free = (doca_gpu_mem_free_t)get_gpunetio_sdk_symbol("doca_gpu_mem_free");
    p_doca_gpu_dmabuf_fd = (doca_gpu_dmabuf_fd_t)get_gpunetio_sdk_symbol("doca_gpu_dmabuf_fd");
    p_doca_gpu_verbs_req_notify_cq =
        (doca_gpu_verbs_req_notify_cq_t)get_gpunetio_sdk_symbol("doca_gpu_verbs_req_notify_cq");

#if DOCA_GPUNETIO_SDK_WRAPPER_ENABLE_DEBUG == 1
    p_doca_log_backend_create_with_file_sdk =
        (doca_log_backend_create_with_file_sdk_t)get_gpunetio_sdk_symbol(
            "doca_log_backend_create_with_file_sdk");
#endif

    /*
     * Check if all symbols were found.
     * Symbol p_doca_gpu_verbs_req_notify_cq is optional as not present in DOCA < 3.5
     */
    if (!p_doca_gpu_create || !p_doca_gpu_destroy || !p_doca_gpu_mem_alloc ||
        !p_doca_gpu_mem_free || !p_doca_gpu_dmabuf_fd) {
        DOCA_LOG(LOG_ERR, "Failed to get all required DOCA GPUNetIO SDK symbols\n");
        dlclose(gpunetio_handle);
        dlclose(common_handle);
        dlclose(verbs_handle);
        gpunetio_handle = nullptr;
        common_handle = nullptr;
        verbs_handle = nullptr;
        *ret = -1;
        goto exit_error;
    }

#if DOCA_GPUNETIO_SDK_WRAPPER_ENABLE_DEBUG == 1
    if (!p_doca_log_backend_create_with_file_sdk) {
        DOCA_LOG(LOG_ERR,
                 "Failed to get doca_log_backend_create_with_file_sdk DOCA Verbs Dev SDK symbol\n");
        dlclose(verbs_handle);
        verbs_handle = nullptr;
        *ret = -1;
        goto exit_error;
    }
#endif

    *ret = 0;
    return;

exit_error:
    if (gpunetio_handle) dlclose(gpunetio_handle);
    if (common_handle) dlclose(common_handle);
    if (verbs_handle) dlclose(verbs_handle);
    gpunetio_handle = nullptr;
    common_handle = nullptr;
    verbs_handle = nullptr;
    return;
}

static int init_gpunetio_sdk_wrapper(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_gpunetio_sdk_wrapper_init, &ret);
    return ret;
}

static int get_sdk_wrapper_env_var(void) {
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (!val) return -1;

    if (strcmp(val, "") == 0) return -1;

    return 1;
}

/* Wrapper function implementations */
doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_create(const char *gpu_bus_id, void **gpu_dev) {
    doca_error_t doca_err;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

#if DOCA_GPUNETIO_SDK_WRAPPER_ENABLE_DEBUG == 1
    void *sdk_log;
#endif

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_gpunetio_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

#if DOCA_GPUNETIO_SDK_WRAPPER_ENABLE_DEBUG == 1
        doca_err = p_doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
        if (doca_err != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
#endif

        doca_err = p_doca_gpu_create(gpu_bus_id, gpu_dev);
        if (doca_err == DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH set to %s. DOCA SDK is in use", val);
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else {
        DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH not set %s. DOCA SDK is not in use", val);
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
    }
}

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_destroy(void *gpu_dev) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_gpunetio_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_gpu_destroy(gpu_dev);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_mem_alloc(void *gpu_dev, size_t size,
                                                        size_t alignment,
                                                        enum doca_gpu_mem_type mtype,
                                                        void **memptr_gpu, void **memptr_cpu) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_gpunetio_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_gpu_mem_alloc(gpu_dev, size, alignment, mtype, memptr_gpu, memptr_cpu);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_mem_free(void *gpu_dev, void *memptr_gpu) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_gpunetio_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_gpu_mem_free(gpu_dev, memptr_gpu);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_dmabuf_fd(void *gpu_dev, void *memptr_gpu,
                                                        size_t size, int *dmabuf_fd) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_gpunetio_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_gpu_dmabuf_fd(gpu_dev, memptr_gpu, size, dmabuf_fd);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_gpu_sdk_wrapper_verbs_req_notify_cq(void *gpu_dev, void *verbs_cq) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_gpunetio_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (p_doca_gpu_verbs_req_notify_cq == nullptr) {
            DOCA_LOG(LOG_ERR, "DOCA SDK symbol doca_gpu_verbs_req_notify_cq not found at %s",
                     __func__);
            return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
        }

        doca_err = p_doca_gpu_verbs_req_notify_cq(gpu_dev, verbs_cq);
        if (doca_err == DOCA_SUCCESS) {
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif
