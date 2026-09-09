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
#include "doca_verbs_net_wrapper.h"
#include "doca_verbs_dev_sdk_wrapper.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Function pointer types for DOCA Verbs Dev SDK APIs */
typedef doca_error_t (*doca_verbs_sdk_wrapper_dev_open_from_pd_t)(struct ibv_pd *pd, void **dev);
typedef doca_error_t (*doca_verbs_sdk_wrapper_dev_close_t)(void *dev);
typedef doca_error_t (*doca_verbs_bridge_verbs_pd_import_t)(struct ibv_pd *pd, void **verbs_pd);
typedef doca_error_t (*doca_verbs_bridge_verbs_context_import_t)(struct ibv_context *ibv_ctx,
                                                                 uint32_t flags,
                                                                 void **verbs_context);
typedef doca_error_t (*doca_verbs_pd_destroy_t)(void *verbs_pd);
typedef doca_error_t (*doca_verbs_context_destroy_t)(void *verbs_context);

/* Global function pointers */
doca_verbs_sdk_wrapper_dev_open_from_pd_t p_doca_verbs_sdk_wrapper_dev_open_from_pd = nullptr;
doca_verbs_sdk_wrapper_dev_close_t p_doca_verbs_dev_close = nullptr;

static doca_verbs_bridge_verbs_pd_import_t p_doca_verbs_bridge_verbs_pd_import = nullptr;
static doca_verbs_bridge_verbs_context_import_t p_doca_verbs_bridge_verbs_context_import = nullptr;
static doca_verbs_pd_destroy_t p_doca_verbs_pd_destroy = nullptr;
static doca_verbs_context_destroy_t p_doca_verbs_context_destroy = nullptr;

static void *common_handle = nullptr;
static void *verbs_handle = nullptr;

/* Helper function to get function pointer from libcuda */
static void *get_verbs_sdk_symbol(const char *symbol_name) {
    void *symbol = dlsym(verbs_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void doca_verbs_sdk_wrapper_init(int *ret) {
    char libcommon_path[doca_sdk_path_length];
    char libverbs_path[doca_sdk_path_length];

    memset(libcommon_path, '\0', doca_sdk_path_length);
    memset(libverbs_path, '\0', doca_sdk_path_length);

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libcommon_path, doca_sdk_path_length, "%s", "libdoca_common.so");
    else
        snprintf(libcommon_path, doca_sdk_path_length, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_common.so");

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libverbs_path, doca_sdk_path_length, "%s", "libdoca_verbs.so");
    else
        snprintf(libverbs_path, doca_sdk_path_length, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_verbs.so");

    common_handle = dlopen(libcommon_path, RTLD_NOW | RTLD_GLOBAL);
    if (!common_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_common.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libcommon_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    verbs_handle = dlopen(libverbs_path, RTLD_NOW | RTLD_LOCAL);
    if (!verbs_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_verbs.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libverbs_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    /* Get function pointers */
    p_doca_verbs_sdk_wrapper_dev_open_from_pd =
        (doca_verbs_sdk_wrapper_dev_open_from_pd_t)get_verbs_sdk_symbol(
            "doca_rdma_bridge_open_dev_from_pd");
    p_doca_verbs_dev_close =
        (doca_verbs_sdk_wrapper_dev_close_t)get_verbs_sdk_symbol("doca_dev_close");

    p_doca_verbs_bridge_verbs_pd_import = (doca_verbs_bridge_verbs_pd_import_t)get_verbs_sdk_symbol(
        "doca_verbs_bridge_verbs_pd_import");
    p_doca_verbs_bridge_verbs_context_import =
        (doca_verbs_bridge_verbs_context_import_t)get_verbs_sdk_symbol(
            "doca_verbs_bridge_verbs_context_import");

    p_doca_verbs_pd_destroy =
        (doca_verbs_pd_destroy_t)get_verbs_sdk_symbol("doca_verbs_pd_destroy");
    p_doca_verbs_context_destroy =
        (doca_verbs_context_destroy_t)get_verbs_sdk_symbol("doca_verbs_context_destroy");

    /* Check if all symbols were found */
    if (!p_doca_verbs_sdk_wrapper_dev_open_from_pd || !p_doca_verbs_dev_close ||
        !p_doca_verbs_bridge_verbs_pd_import || !p_doca_verbs_bridge_verbs_context_import ||
        !p_doca_verbs_pd_destroy || !p_doca_verbs_context_destroy) {
        DOCA_LOG(LOG_ERR, "Failed to get all required DOCA Verbs Dev SDK symbols\n");
        dlclose(verbs_handle);
        verbs_handle = nullptr;
        *ret = -1;
        goto exit_error;
    }

    *ret = 0;
    return;

exit_error:
    if (verbs_handle) dlclose(verbs_handle);
    if (common_handle) dlclose(common_handle);
    verbs_handle = nullptr;
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
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_dev_open_from_pd(struct ibv_pd *pd,
                                                                 doca_dev_t *net_dev) {
    doca_error_t doca_err;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_sdk_wrapper_dev_open_from_pd(pd, &(net_dev->sdk));
        if (doca_err == DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH set to %s. DOCA SDK is in use", val);

            doca_err = p_doca_verbs_bridge_verbs_pd_import(pd, &net_dev->sdk_pd);
            if (doca_err != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
                p_doca_verbs_dev_close(net_dev->sdk);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }

            doca_err =
                p_doca_verbs_bridge_verbs_context_import(pd->context, 0, &net_dev->sdk_context);
            if (doca_err != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
                p_doca_verbs_pd_destroy(net_dev->sdk_pd);
                p_doca_verbs_dev_close(net_dev->sdk);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }
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

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_dev_close(doca_dev_t *net_dev) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (net_dev->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->sdk_pd == nullptr || net_dev->sdk_context == nullptr) {
            DOCA_LOG(LOG_ERR, "doca_dev_t has no SDK elements.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_pd_destroy(net_dev->sdk_pd);
        if (doca_err != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);

        doca_err = p_doca_verbs_context_destroy(net_dev->sdk_context);
        if (doca_err != DOCA_SUCCESS)
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);

        doca_err = p_doca_verbs_dev_close(net_dev->sdk);
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
