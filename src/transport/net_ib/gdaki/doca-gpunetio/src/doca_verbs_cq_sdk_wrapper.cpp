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
#include <string.h>
#include <mutex>
#include <fcntl.h>

#include "doca_gpunetio_log.hpp"
#include "doca_verbs_uar_sdk_wrapper.h"

/*
 * Enable DOCA SDK log errors on stderr.
 * Disabled by default to avoid extra-prints on screen.
 */
#define DOCA_VERBS_CQ_SDK_WRAPPER_ENABLE_DEBUG 0

#ifdef __cplusplus
extern "C" {
#endif

/* Function pointer types for DOCA Verbs CQ SDK APIs */
typedef doca_error_t (*doca_verbs_cq_attr_create_t)(void **verbs_cq_attr);
typedef doca_error_t (*doca_verbs_cq_attr_destroy_t)(void *cq_attr);
typedef doca_error_t (*doca_verbs_cq_attr_set_cq_size_t)(void *cq_attr, uint32_t cq_size);
typedef doca_error_t (*doca_verbs_cq_attr_set_cq_context_t)(void *cq_attr, void *cq_context);
typedef doca_error_t (*doca_verbs_cq_attr_set_external_datapath_en_t)(void *cq_attr,
                                                                      uint8_t external_datapath_en);
typedef doca_error_t (*doca_verbs_cq_attr_set_external_umem_t)(void *cq_attr, void *external_umem,
                                                               uint64_t external_umem_offset);
typedef doca_error_t (*doca_verbs_cq_attr_set_external_dbr_umem_t)(
    void *cq_attr, void *external_dbr_umem, uint64_t external_dbr_umem_offset);
typedef doca_error_t (*doca_verbs_cq_attr_set_external_uar_t)(void *cq_attr, void *external_uar);
typedef doca_error_t (*doca_verbs_cq_attr_set_cq_overrun_t)(void *cq_attr,
                                                            enum doca_verbs_cq_overrun overrun);
typedef doca_error_t (*doca_verbs_cq_attr_set_cq_collapsed_t)(void *cq_attr, uint8_t cc);
typedef doca_error_t (*doca_verbs_cq_attr_set_comp_channel_t)(void *cq_attr, void *comp_channel);
typedef doca_error_t (*doca_verbs_cq_attr_set_st_t)(void *cq_attr,
                                                    enum doca_verbs_cq_state cq_state);
typedef doca_error_t (*doca_verbs_cq_create_t)(void *verbs_ctx, void *cq_attr, void **verbs_cq);
typedef doca_error_t (*doca_verbs_cq_destroy_t)(void *verbs_cq);
typedef void (*doca_verbs_cq_get_wq_t)(void *verbs_cq, void **cq_buf, uint32_t *cq_num_entries,
                                       uint8_t *cq_entry_size);
typedef void (*doca_verbs_cq_get_dbr_addr_t)(void *verbs_cq, uint64_t **uar_db_reg,
                                             uint32_t **ci_dbr, uint32_t **arm_dbr);
typedef uint32_t (*doca_verbs_cq_get_cqn_t)(const void *verbs_cq);
typedef doca_error_t (*doca_verbs_cq_set_cq_context_t)(void *verbs_cq, void *cq_context);
typedef doca_error_t (*doca_verbs_comp_channel_create_t)(void *verbs_context,
                                                         void **verbs_comp_channel);
typedef doca_error_t (*doca_verbs_comp_channel_destroy_t)(void *verbs_comp_channel);
typedef int (*doca_verbs_comp_channel_get_handle_t)(void *verbs_comp_channel);
typedef void (*doca_verbs_ack_cq_events_t)(void *verbs_cq, unsigned int nevents);

typedef doca_error_t (*doca_verbs_get_cq_event_t)(void *verbs_comp_channel, void **verbs_cq,
                                                  void **cq_context);

typedef doca_error_t (*doca_rdma_bridge_get_dev_pd_t)(void *dev, struct ibv_pd **pd);

#if DOCA_VERBS_CQ_SDK_WRAPPER_ENABLE_DEBUG == 1
typedef doca_error_t (*doca_log_backend_create_with_file_sdk_t)(FILE *fptr, void **backend);
static doca_log_backend_create_with_file_sdk_t p_doca_log_backend_create_with_file_sdk = nullptr;
#endif

/* Global function pointers */
static doca_verbs_cq_attr_create_t p_doca_verbs_cq_attr_create = nullptr;
static doca_verbs_cq_attr_destroy_t p_doca_verbs_cq_attr_destroy = nullptr;
static doca_verbs_cq_attr_set_cq_size_t p_doca_verbs_cq_attr_set_cq_size = nullptr;
static doca_verbs_cq_attr_set_cq_context_t p_doca_verbs_cq_attr_set_cq_context = nullptr;
static doca_verbs_cq_attr_set_external_datapath_en_t p_doca_verbs_cq_attr_set_external_datapath_en =
    nullptr;
static doca_verbs_cq_attr_set_external_umem_t p_doca_verbs_cq_attr_set_external_umem = nullptr;
static doca_verbs_cq_attr_set_external_dbr_umem_t p_doca_verbs_cq_attr_set_external_dbr_umem =
    nullptr;
static doca_verbs_cq_attr_set_external_uar_t p_doca_verbs_cq_attr_set_external_uar = nullptr;
static doca_verbs_cq_attr_set_comp_channel_t p_doca_verbs_cq_attr_set_comp_channel = nullptr;
static doca_verbs_cq_attr_set_st_t p_doca_verbs_cq_attr_set_st = nullptr;
static doca_verbs_cq_attr_set_cq_overrun_t p_doca_verbs_cq_attr_set_cq_overrun = nullptr;
static doca_verbs_cq_attr_set_cq_collapsed_t p_doca_verbs_cq_attr_set_cq_collapsed = nullptr;

static doca_verbs_cq_create_t p_doca_verbs_cq_create = nullptr;
static doca_verbs_cq_destroy_t p_doca_verbs_cq_destroy = nullptr;
static doca_verbs_cq_get_wq_t p_doca_verbs_cq_get_wq = nullptr;
static doca_verbs_cq_get_dbr_addr_t p_doca_verbs_cq_get_dbr_addr = nullptr;
static doca_verbs_cq_get_cqn_t p_doca_verbs_cq_get_cqn = nullptr;
static doca_verbs_cq_set_cq_context_t p_doca_verbs_cq_set_cq_context = nullptr;
static doca_verbs_comp_channel_create_t p_doca_verbs_comp_channel_create = nullptr;
static doca_verbs_comp_channel_destroy_t p_doca_verbs_comp_channel_destroy = nullptr;
static doca_verbs_comp_channel_get_handle_t p_doca_verbs_comp_channel_get_handle = nullptr;
static doca_verbs_get_cq_event_t p_doca_verbs_get_cq_event = nullptr;
static doca_verbs_ack_cq_events_t p_doca_verbs_ack_cq_events = nullptr;

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
    p_doca_verbs_cq_attr_destroy =
        (doca_verbs_cq_attr_destroy_t)get_verbs_sdk_symbol("doca_verbs_cq_attr_destroy");
    p_doca_verbs_cq_attr_create =
        (doca_verbs_cq_attr_create_t)get_verbs_sdk_symbol("doca_verbs_cq_attr_create");
    p_doca_verbs_cq_attr_set_cq_size =
        (doca_verbs_cq_attr_set_cq_size_t)get_verbs_sdk_symbol("doca_verbs_cq_attr_set_cq_size");
    p_doca_verbs_cq_attr_set_cq_context = (doca_verbs_cq_attr_set_cq_context_t)get_verbs_sdk_symbol(
        "doca_verbs_cq_attr_set_cq_context");
    p_doca_verbs_cq_attr_set_external_datapath_en =
        (doca_verbs_cq_attr_set_external_datapath_en_t)get_verbs_sdk_symbol(
            "doca_verbs_cq_attr_set_external_datapath_en");
    p_doca_verbs_cq_attr_set_external_umem =
        (doca_verbs_cq_attr_set_external_umem_t)get_verbs_sdk_symbol(
            "doca_verbs_cq_attr_set_external_umem");
    p_doca_verbs_cq_attr_set_external_dbr_umem =
        (doca_verbs_cq_attr_set_external_dbr_umem_t)get_verbs_sdk_symbol(
            "doca_verbs_cq_attr_set_external_dbr_umem");
    p_doca_verbs_cq_attr_set_external_uar =
        (doca_verbs_cq_attr_set_external_uar_t)get_verbs_sdk_symbol(
            "doca_verbs_cq_attr_set_external_uar");
    p_doca_verbs_cq_attr_set_cq_overrun = (doca_verbs_cq_attr_set_cq_overrun_t)get_verbs_sdk_symbol(
        "doca_verbs_cq_attr_set_cq_overrun");
    p_doca_verbs_cq_attr_set_cq_collapsed =
        (doca_verbs_cq_attr_set_cq_collapsed_t)get_verbs_sdk_symbol(
            "doca_verbs_cq_attr_set_cq_collapsed");
    p_doca_verbs_cq_attr_set_comp_channel =
        (doca_verbs_cq_attr_set_comp_channel_t)get_verbs_sdk_symbol(
            "doca_verbs_cq_attr_set_comp_channel");
    p_doca_verbs_cq_attr_set_st =
        (doca_verbs_cq_attr_set_st_t)get_verbs_sdk_symbol("doca_verbs_cq_attr_set_st");
    p_doca_verbs_cq_create = (doca_verbs_cq_create_t)get_verbs_sdk_symbol("doca_verbs_cq_create");
    p_doca_verbs_cq_destroy =
        (doca_verbs_cq_destroy_t)get_verbs_sdk_symbol("doca_verbs_cq_destroy");
    p_doca_verbs_cq_get_wq = (doca_verbs_cq_get_wq_t)get_verbs_sdk_symbol("doca_verbs_cq_get_wq");
    p_doca_verbs_cq_get_dbr_addr =
        (doca_verbs_cq_get_dbr_addr_t)get_verbs_sdk_symbol("doca_verbs_cq_get_dbr_addr");
    p_doca_verbs_cq_get_cqn =
        (doca_verbs_cq_get_cqn_t)get_verbs_sdk_symbol("doca_verbs_cq_get_cqn");
    p_doca_verbs_cq_set_cq_context =
        (doca_verbs_cq_set_cq_context_t)get_verbs_sdk_symbol("doca_verbs_cq_set_cq_context");
    p_doca_verbs_comp_channel_create =
        (doca_verbs_comp_channel_create_t)get_verbs_sdk_symbol("doca_verbs_comp_channel_create");
    p_doca_verbs_comp_channel_destroy =
        (doca_verbs_comp_channel_destroy_t)get_verbs_sdk_symbol("doca_verbs_comp_channel_destroy");
    p_doca_verbs_comp_channel_get_handle =
        (doca_verbs_comp_channel_get_handle_t)get_verbs_sdk_symbol(
            "doca_verbs_comp_channel_get_handle");
    p_doca_verbs_get_cq_event =
        (doca_verbs_get_cq_event_t)get_verbs_sdk_symbol("doca_verbs_get_cq_event");
    p_doca_verbs_ack_cq_events =
        (doca_verbs_ack_cq_events_t)get_verbs_sdk_symbol("doca_verbs_ack_cq_events");

#if DOCA_VERBS_CQ_SDK_WRAPPER_ENABLE_DEBUG == 1
    p_doca_log_backend_create_with_file_sdk =
        (doca_log_backend_create_with_file_sdk_t)get_verbs_sdk_symbol(
            "doca_log_backend_create_with_file_sdk");
#endif

    /*
     * Check if all symbols were found.
     * Symbol p_doca_verbs_cq_attr_set_cq_collapsed is optional as not present in DOCA 3.2 LTS
     * Symbol p_doca_verbs_cq_attr_set_external_dbr_umem is optional as not present in DOCA < 3.5
     * Symbol p_doca_verbs_cq_set_cq_context is optional as not present in DOCA < 3.5
     * Symbol p_doca_verbs_cq_attr_set_st is optional as not present in DOCA < 3.5
     */
    if (!p_doca_verbs_cq_attr_destroy || !p_doca_verbs_cq_attr_create ||
        !p_doca_verbs_cq_attr_set_cq_size || !p_doca_verbs_cq_attr_set_cq_context ||
        !p_doca_verbs_cq_attr_set_external_datapath_en || !p_doca_verbs_cq_attr_set_external_umem ||
        !p_doca_verbs_cq_attr_set_external_uar || !p_doca_verbs_cq_attr_set_cq_overrun ||
        !p_doca_verbs_cq_attr_set_comp_channel || !p_doca_verbs_cq_create ||
        !p_doca_verbs_cq_destroy || !p_doca_verbs_cq_get_wq || !p_doca_verbs_cq_get_dbr_addr ||
        !p_doca_verbs_cq_get_cqn || !p_doca_verbs_comp_channel_create ||
        !p_doca_verbs_comp_channel_destroy || !p_doca_verbs_comp_channel_get_handle ||
        !p_doca_verbs_get_cq_event || !p_doca_verbs_ack_cq_events) {
        DOCA_LOG(LOG_ERR, "Failed to get all required DOCA Verbs Dev SDK symbols\n");
        dlclose(verbs_handle);
        verbs_handle = nullptr;
        *ret = -1;
        goto exit_error;
    }

#if DOCA_VERBS_CQ_SDK_WRAPPER_ENABLE_DEBUG == 1
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
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_create(void **verbs_cq_attr) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

#if DOCA_VERBS_CQ_SDK_WRAPPER_ENABLE_DEBUG == 1
    void *sdk_log;
#endif

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

#if DOCA_VERBS_CQ_SDK_WRAPPER_ENABLE_DEBUG == 1
        doca_err = p_doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
        if (doca_err != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
#endif

        doca_err = p_doca_verbs_cq_attr_create(verbs_cq_attr);
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

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_destroy(void *cq_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_cq_attr_destroy(cq_attr);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_size(void *cq_attr,
                                                                    uint32_t cq_size) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_cq_attr_set_cq_size(cq_attr, cq_size);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_context(void *cq_attr,
                                                                       void *cq_context) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_cq_attr_set_cq_context(cq_attr, cq_context);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_external_umem(
    void *cq_attr, doca_verbs_umem_t *external_umem, uint64_t external_umem_offset) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (external_umem->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_umem_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_cq_attr_set_external_umem(cq_attr, external_umem->sdk,
                                                          external_umem_offset);
        if (doca_err == DOCA_SUCCESS) {
            doca_err = p_doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1);
            if (doca_err == DOCA_SUCCESS)
                return DOCA_SDK_WRAPPER_SUCCESS;
            else {
                DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_external_dbr_umem(
    void *cq_attr, doca_verbs_umem_t *external_dbr_umem, uint64_t external_dbr_umem_offset) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (external_dbr_umem->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_umem_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (p_doca_verbs_cq_attr_set_external_dbr_umem == nullptr) {
            DOCA_LOG(LOG_ERR,
                     "DOCA SDK symbol doca_verbs_cq_attr_set_external_dbr_umem not found at %s",
                     __func__);
            return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
        }

        doca_err = p_doca_verbs_cq_attr_set_external_dbr_umem(cq_attr, external_dbr_umem->sdk,
                                                              external_dbr_umem_offset);
        if (doca_err == DOCA_SUCCESS) {
            doca_err = p_doca_verbs_cq_attr_set_external_datapath_en(cq_attr, 1);
            if (doca_err == DOCA_SUCCESS)
                return DOCA_SDK_WRAPPER_SUCCESS;
            else {
                DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_external_uar(
    void *cq_attr, doca_verbs_uar_t *external_uar) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (external_uar->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_uar_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_cq_attr_set_external_uar(cq_attr, external_uar->sdk);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_overrun(
    void *cq_attr, enum doca_verbs_cq_overrun overrun) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_cq_attr_set_cq_overrun(cq_attr, overrun);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_collapsed(void *cq_attr,
                                                                         uint8_t cc) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_cq_attr_set_cq_collapsed) return DOCA_SDK_WRAPPER_NOT_SUPPORTED;

        doca_err = p_doca_verbs_cq_attr_set_cq_collapsed(cq_attr, cc);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_comp_channel(void *cq_attr,
                                                                         void *compch) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_cq_attr_set_comp_channel) return DOCA_SDK_WRAPPER_NOT_SUPPORTED;

        doca_err = p_doca_verbs_cq_attr_set_comp_channel(cq_attr, compch);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_st(void *cq_attr,
                                                               enum doca_verbs_cq_state cq_state) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_cq_attr_set_st) return DOCA_SDK_WRAPPER_NOT_SUPPORTED;

        doca_err = p_doca_verbs_cq_attr_set_st(cq_attr, cq_state);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_create(doca_dev_t *net_dev,
                                                          doca_verbs_cq_attr_t *cq_attr,
                                                          void **verbs_cq) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create CQ, net_dev is null", __func__);
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

        if (cq_attr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_cq_attr_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        /* According to library logic, if net_dev is SDK, then also GPU is SDK */
        doca_err = p_doca_verbs_cq_create(net_dev->sdk_context, cq_attr->sdk, verbs_cq);
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

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_destroy(void *verbs_cq) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_cq_destroy(verbs_cq);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_get_wq(void *verbs_cq, void **cq_buf,
                                                          uint32_t *cq_num_entries,
                                                          uint8_t *cq_entry_size) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        // DOCA SDK function it's a simple getter, no error code returned
        p_doca_verbs_cq_get_wq(verbs_cq, cq_buf, cq_num_entries, cq_entry_size);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_get_dbr_addr(void *verbs_cq,
                                                                uint64_t **uar_db_reg,
                                                                uint32_t **ci_dbr,
                                                                uint32_t **arm_dbr) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        // DOCA SDK function it's a simple getter, no error code returned
        p_doca_verbs_cq_get_dbr_addr(verbs_cq, uar_db_reg, ci_dbr, arm_dbr);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_get_cqn(const void *verbs_cq, uint32_t *cqn) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        // DOCA SDK function it's a simple getter, no error code returned
        *cqn = p_doca_verbs_cq_get_cqn(verbs_cq);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_set_cq_context(void *cq, void *cq_context) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (p_doca_verbs_cq_set_cq_context == nullptr) {
            DOCA_LOG(LOG_ERR,
                     "DOCA SDK symbol doca_verbs_cq_attr_set_external_dbr_umem not found at %s",
                     __func__);
            return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
        }

        doca_err = p_doca_verbs_cq_set_cq_context(cq, cq_context);
        if (doca_err == DOCA_SUCCESS) {
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_comp_channel_create(void *sdk_context,
                                                                    void **compch) {
    doca_error_t doca_err;
    int fd;
    int fd_flags = 0;
    int fd_fcntl_status = 0;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        /*
         * If symbol p_doca_verbs_cq_set_cq_context is not present, it means the DOCA SDK version is
         * < 3.5. This implies there is the lack of GPUNetIO open source required symbol so the SDK
         * version of this feature can't be used.
         */
        if (p_doca_verbs_cq_set_cq_context == nullptr) {
            DOCA_LOG(LOG_ERR,
                     "DOCA SDK symbol p_doca_verbs_cq_set_cq_context not found at %s. Can't use "
                     "SDK version for doca_verbs_comp_channel_create.",
                     __func__);
            return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
        }

        doca_err = p_doca_verbs_comp_channel_create(sdk_context, compch);
        if (doca_err == DOCA_SUCCESS) {
            fd = static_cast<int>(p_doca_verbs_comp_channel_get_handle(*compch));

            fd_flags = fcntl(fd, F_GETFL);
            if (fd_flags < 0) {
                DOCA_LOG(LOG_ERR, "Failed to set at %s fcntl fd_flags", __func__);
                p_doca_verbs_comp_channel_destroy(*compch);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }

            fd_fcntl_status = fcntl(fd, F_SETFL, fd_flags | O_NONBLOCK);
            if (fd_fcntl_status < 0) {
                DOCA_LOG(LOG_ERR, "Failed to set at %s fcntl fd_fcntl_status", __func__);
                p_doca_verbs_comp_channel_destroy(*compch);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }

            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_comp_channel_destroy(void *compch) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_comp_channel_destroy(compch);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_get_cq_event(void *compch, void **cq_context) {
    doca_error_t doca_err;
    void *verbs_cq;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_get_cq_event(compch, &verbs_cq, cq_context);
        if (doca_err == DOCA_SUCCESS) return DOCA_SDK_WRAPPER_SUCCESS;
        if (doca_err == DOCA_ERROR_AGAIN)
            return DOCA_SDK_WRAPPER_ERROR_AGAIN;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ack_cq_events(void *verbs_cq,
                                                              unsigned int nevents) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        p_doca_verbs_ack_cq_events(verbs_cq, nevents);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif