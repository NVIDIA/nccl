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
#include "doca_verbs_uar_sdk_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function pointer types for DOCA Verbs UAR SDK APIs */
typedef doca_error_t (*doca_verbs_uar_create_t)(void *dev,
                                                enum doca_verbs_uar_allocation_type allocation_type,
                                                void **uar);
typedef doca_error_t (*doca_verbs_uar_destroy_t)(void *uar);
typedef doca_error_t (*doca_verbs_uar_get_id_t)(void *uar, uint32_t *id);
typedef doca_error_t (*doca_verbs_uar_reg_addr_get_t)(void *uar, void **reg);

/* Global function pointers */
static doca_verbs_uar_create_t p_doca_verbs_uar_create = nullptr;
static doca_verbs_uar_destroy_t p_doca_verbs_uar_destroy = nullptr;
static doca_verbs_uar_get_id_t p_doca_verbs_uar_get_id = nullptr;
static doca_verbs_uar_reg_addr_get_t p_doca_verbs_uar_reg_addr_get = nullptr;

static void *common_handle = nullptr;

/* Helper function to get function pointer from libcuda */
static void *get_verbs_sdk_symbol(const char *symbol_name) {
    void *symbol = dlsym(common_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void doca_verbs_sdk_wrapper_init(int *ret) {
    char libcommon_path[doca_sdk_path_length];

    memset(libcommon_path, '\0', doca_sdk_path_length);

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libcommon_path, doca_sdk_path_length, "%s", "libdoca_common.so");
    else
        snprintf(libcommon_path, doca_sdk_path_length, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_common.so");

    common_handle = dlopen(libcommon_path, RTLD_NOW | RTLD_GLOBAL);
    if (!common_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_common.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libcommon_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    /* Get function pointers */
    p_doca_verbs_uar_create = (doca_verbs_uar_create_t)get_verbs_sdk_symbol("doca_uar_create");
    p_doca_verbs_uar_destroy = (doca_verbs_uar_destroy_t)get_verbs_sdk_symbol("doca_uar_destroy");
    p_doca_verbs_uar_get_id = (doca_verbs_uar_get_id_t)get_verbs_sdk_symbol("doca_uar_id_get");
    p_doca_verbs_uar_reg_addr_get =
        (doca_verbs_uar_reg_addr_get_t)get_verbs_sdk_symbol("doca_uar_reg_addr_get");

    /* Check if all symbols were found */
    if (!p_doca_verbs_uar_create || !p_doca_verbs_uar_destroy || !p_doca_verbs_uar_get_id ||
        !p_doca_verbs_uar_reg_addr_get) {
        DOCA_LOG(LOG_ERR, "Failed to get all required DOCA Verbs UMEM SDK symbols\n");
        *ret = -1;
        goto exit_error;
    }

    *ret = 0;
    return;

exit_error:
    if (common_handle) dlclose(common_handle);
    common_handle = nullptr;
    return;
}

static int init_verbs_sdk_wrapper(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_verbs_sdk_wrapper_init, &ret);
    return ret;
}

static int get_sdk_wrapper_env_var(void) {
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (!val) return -1;

    if (strcmp(val, "") == 0) return -1;

    return 1;
}

/* Wrapper function implementations */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_uar_create(
    doca_dev_t *net_dev, enum doca_verbs_uar_allocation_type allocation_type, void **uar) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Can create umem, net_dev is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        /* According to library logic, if net_dev is SDK, then also GPU is SDK */
        doca_err = p_doca_verbs_uar_create(net_dev->sdk, allocation_type, uar);
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

    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_uar_destroy(void *uar) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_uar_destroy(uar);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_uar_get_id(void *uar, uint32_t *uar_id) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_uar_get_id(uar, uar_id);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_uar_reg_addr_get(void *uar, void **reg) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_uar_reg_addr_get(uar, reg);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif