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
 * @file doca_verbs_ibv_wrapper.cpp
 * @brief Implementation of IB Verbs API wrapper using dlopen
 *
 * This file implements the IB Verbs API wrapper using dynamic loading.
 * It is only compiled when DOCA_VERBS_USE_IBV_WRAPPER is defined.
 */

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>

#include "doca_verbs_net_wrapper.h"
#include "doca_gpunetio_log.hpp"
#include "host/doca_error.h"

/* *********** Function Pointer Types *********** */

typedef struct ibv_device **(*ibv_get_device_list_func_t)(int *num_devices);
typedef void (*ibv_free_device_list_func_t)(struct ibv_device **list);
typedef const char *(*ibv_get_device_name_func_t)(struct ibv_device *device);
typedef struct ibv_context *(*ibv_open_device_func_t)(struct ibv_device *device);
typedef int (*ibv_close_device_func_t)(struct ibv_context *context);
typedef struct ibv_pd *(*ibv_alloc_pd_func_t)(struct ibv_context *context);
typedef int (*ibv_dealloc_pd_func_t)(struct ibv_pd *pd);
typedef struct ibv_mr *(*ibv_reg_mr_func_t)(struct ibv_pd *pd, void *addr, size_t length,
                                            int access);
typedef int (*ibv_dereg_mr_func_t)(struct ibv_mr *mr);
typedef int (*ibv_query_device_func_t)(struct ibv_context *context,
                                       struct ibv_device_attr *device_attr);
typedef int (*ibv_query_port_func_t)(struct ibv_context *context, uint8_t port_num,
                                     struct ibv_port_attr *port_attr);
typedef int (*ibv_query_gid_func_t)(struct ibv_context *context, uint8_t port_num, int index,
                                    union ibv_gid *gid);
typedef struct ibv_ah *(*ibv_create_ah_func_t)(struct ibv_pd *pd, struct ibv_ah_attr *attr);
typedef int (*ibv_destroy_ah_func_t)(struct ibv_ah *ah);
typedef struct ibv_cq *(*ibv_create_cq_func_t)(struct ibv_context *context, int cqe,
                                               void *cq_context, struct ibv_comp_channel *channel,
                                               int comp_vector);
typedef int (*ibv_destroy_cq_func_t)(struct ibv_cq *cq);
typedef struct ibv_srq *(*ibv_create_srq_func_t)(struct ibv_pd *pd,
                                                 struct ibv_srq_init_attr *srq_init_attr);
typedef int (*ibv_destroy_srq_func_t)(struct ibv_srq *srq);
typedef struct ibv_qp *(*ibv_create_qp_func_t)(struct ibv_pd *pd,
                                               struct ibv_qp_init_attr *qp_init_attr);
typedef int (*ibv_destroy_qp_func_t)(struct ibv_qp *qp);
typedef int (*ibv_modify_qp_func_t)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask);
typedef int (*ibv_query_qp_func_t)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attr_mask,
                                   struct ibv_qp_init_attr *init_attr);

/* *********** Global Function Pointers *********** */

static ibv_get_device_list_func_t real_ibv_get_device_list = NULL;
static ibv_free_device_list_func_t real_ibv_free_device_list = NULL;
static ibv_get_device_name_func_t real_ibv_get_device_name = NULL;
static ibv_open_device_func_t real_ibv_open_device = NULL;
static ibv_close_device_func_t real_ibv_close_device = NULL;
static ibv_alloc_pd_func_t real_ibv_alloc_pd = NULL;
static ibv_dealloc_pd_func_t real_ibv_dealloc_pd = NULL;
static ibv_reg_mr_func_t real_ibv_reg_mr = NULL;
static ibv_dereg_mr_func_t real_ibv_dereg_mr = NULL;
static ibv_query_device_func_t real_ibv_query_device = NULL;
static ibv_query_port_func_t real_ibv_query_port = NULL;
static ibv_query_gid_func_t real_ibv_query_gid = NULL;
static ibv_create_ah_func_t real_ibv_create_ah = NULL;
static ibv_destroy_ah_func_t real_ibv_destroy_ah = NULL;
static ibv_create_cq_func_t real_ibv_create_cq = NULL;
static ibv_destroy_cq_func_t real_ibv_destroy_cq = NULL;
static ibv_create_srq_func_t real_ibv_create_srq = NULL;
static ibv_destroy_srq_func_t real_ibv_destroy_srq = NULL;
static ibv_create_qp_func_t real_ibv_create_qp = NULL;
static ibv_destroy_qp_func_t real_ibv_destroy_qp = NULL;
static ibv_modify_qp_func_t real_ibv_modify_qp = NULL;
static ibv_query_qp_func_t real_ibv_query_qp = NULL;

/* *********** Library Handle *********** */

static void *ibverbs_handle = NULL;

/* *********** Helper Functions *********** */

/**
 * @brief Initialize the IB Verbs library using dlopen
 *
 * @return 0 on success, -1 on failure
 */
static void doca_verbs_wrapper_init_once(int *ret) {
    /* Try to open the IB Verbs library */
    ibverbs_handle = dlopen("libibverbs.so.1", RTLD_NOW);
    if (!ibverbs_handle) {
        ibverbs_handle = dlopen("libibverbs.so", RTLD_NOW);
    }
    if (!ibverbs_handle) {
        DOCA_LOG(LOG_ERR, "Failed to load libibverbs: %s\n", dlerror());
        *ret = -1;
        return;
    }

    /* Load all function pointers */
    real_ibv_get_device_list =
        (ibv_get_device_list_func_t)dlsym(ibverbs_handle, "ibv_get_device_list");
    real_ibv_free_device_list =
        (ibv_free_device_list_func_t)dlsym(ibverbs_handle, "ibv_free_device_list");
    real_ibv_get_device_name =
        (ibv_get_device_name_func_t)dlsym(ibverbs_handle, "ibv_get_device_name");
    real_ibv_open_device = (ibv_open_device_func_t)dlsym(ibverbs_handle, "ibv_open_device");
    real_ibv_close_device = (ibv_close_device_func_t)dlsym(ibverbs_handle, "ibv_close_device");
    real_ibv_alloc_pd = (ibv_alloc_pd_func_t)dlsym(ibverbs_handle, "ibv_alloc_pd");
    real_ibv_dealloc_pd = (ibv_dealloc_pd_func_t)dlsym(ibverbs_handle, "ibv_dealloc_pd");
    real_ibv_reg_mr = (ibv_reg_mr_func_t)dlsym(ibverbs_handle, "ibv_reg_mr");
    real_ibv_dereg_mr = (ibv_dereg_mr_func_t)dlsym(ibverbs_handle, "ibv_dereg_mr");
    real_ibv_query_device = (ibv_query_device_func_t)dlsym(ibverbs_handle, "ibv_query_device");
    real_ibv_query_port = (ibv_query_port_func_t)dlsym(ibverbs_handle, "ibv_query_port");
    real_ibv_query_gid = (ibv_query_gid_func_t)dlsym(ibverbs_handle, "ibv_query_gid");
    real_ibv_create_ah = (ibv_create_ah_func_t)dlsym(ibverbs_handle, "ibv_create_ah");
    real_ibv_destroy_ah = (ibv_destroy_ah_func_t)dlsym(ibverbs_handle, "ibv_destroy_ah");
    real_ibv_create_cq = (ibv_create_cq_func_t)dlsym(ibverbs_handle, "ibv_create_cq");
    real_ibv_destroy_cq = (ibv_destroy_cq_func_t)dlsym(ibverbs_handle, "ibv_destroy_cq");
    real_ibv_create_srq = (ibv_create_srq_func_t)dlsym(ibverbs_handle, "ibv_create_srq");
    real_ibv_destroy_srq = (ibv_destroy_srq_func_t)dlsym(ibverbs_handle, "ibv_destroy_srq");
    real_ibv_create_qp = (ibv_create_qp_func_t)dlsym(ibverbs_handle, "ibv_create_qp");
    real_ibv_destroy_qp = (ibv_destroy_qp_func_t)dlsym(ibverbs_handle, "ibv_destroy_qp");
    real_ibv_modify_qp = (ibv_modify_qp_func_t)dlsym(ibverbs_handle, "ibv_modify_qp");
    real_ibv_query_qp = (ibv_query_qp_func_t)dlsym(ibverbs_handle, "ibv_query_qp");

    /* Check if all functions were loaded successfully */
    if (!real_ibv_get_device_list || !real_ibv_free_device_list || !real_ibv_get_device_name ||
        !real_ibv_open_device || !real_ibv_close_device || !real_ibv_alloc_pd ||
        !real_ibv_dealloc_pd || !real_ibv_reg_mr || !real_ibv_dereg_mr || !real_ibv_query_device ||
        !real_ibv_query_port || !real_ibv_query_gid || !real_ibv_create_ah ||
        !real_ibv_destroy_ah || !real_ibv_create_cq || !real_ibv_destroy_cq ||
        !real_ibv_create_srq || !real_ibv_destroy_srq || !real_ibv_create_qp ||
        !real_ibv_destroy_qp || !real_ibv_modify_qp || !real_ibv_query_qp) {
        fprintf(stderr, "Failed to load IB Verbs functions: %s\n", dlerror());
        dlclose(ibverbs_handle);
        ibverbs_handle = NULL;
        *ret = -1;
        return;
    }

    *ret = 0;
}

static int init_ibverbs_library(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_verbs_wrapper_init_once, &ret);
    return ret;
}

/* *********** Wrapper Implementations *********** */

doca_error_t doca_verbs_wrapper_ibv_get_device_list(int *num_devices,
                                                    struct ibv_device ***device_list) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *device_list = real_ibv_get_device_list(num_devices);
    return (*device_list != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_free_device_list(struct ibv_device **list) {
    if (real_ibv_free_device_list) {
        real_ibv_free_device_list(list);
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_get_device_name(struct ibv_device *device,
                                                    const char **device_name) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *device_name = real_ibv_get_device_name(device);
    return (*device_name != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_open_device(struct ibv_device *device,
                                                struct ibv_context **context) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *context = real_ibv_open_device(device);
    return (*context != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_close_device(struct ibv_context *context) {
    if (real_ibv_close_device) {
        int ret = real_ibv_close_device(context);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_alloc_pd(struct ibv_context *context, struct ibv_pd **pd) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *pd = real_ibv_alloc_pd(context);
    return (*pd != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_dealloc_pd(struct ibv_pd *pd) {
    if (real_ibv_dealloc_pd) {
        int ret = real_ibv_dealloc_pd(pd);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access,
                                           struct ibv_mr **mr) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *mr = real_ibv_reg_mr(pd, addr, length, access);
    return (*mr != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_dereg_mr(struct ibv_mr *mr) {
    if (real_ibv_dereg_mr) {
        int ret = real_ibv_dereg_mr(mr);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_query_device(struct ibv_context *context,
                                                 struct ibv_device_attr *device_attr) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = real_ibv_query_device(context, device_attr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_query_port(struct ibv_context *context, uint8_t port_num,
                                               struct ibv_port_attr *port_attr) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = real_ibv_query_port(context, port_num, port_attr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_query_gid(struct ibv_context *context, uint8_t port_num,
                                              int index, union ibv_gid *gid) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = real_ibv_query_gid(context, port_num, index, gid);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr,
                                              struct ibv_ah **ah) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *ah = real_ibv_create_ah(pd, attr);
    return (*ah != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_destroy_ah(struct ibv_ah *ah) {
    if (real_ibv_destroy_ah) {
        int ret = real_ibv_destroy_ah(ah);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_create_cq(struct ibv_context *context, int cqe,
                                              void *cq_context, struct ibv_comp_channel *channel,
                                              int comp_vector, struct ibv_cq **cq) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *cq = real_ibv_create_cq(context, cqe, cq_context, channel, comp_vector);
    return (*cq != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_destroy_cq(struct ibv_cq *cq) {
    if (real_ibv_destroy_cq) {
        int ret = real_ibv_destroy_cq(cq);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_create_srq(struct ibv_pd *pd,
                                               struct ibv_srq_init_attr *srq_init_attr,
                                               struct ibv_srq **srq) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *srq = real_ibv_create_srq(pd, srq_init_attr);
    return (*srq != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_destroy_srq(struct ibv_srq *srq) {
    if (real_ibv_destroy_srq) {
        int ret = real_ibv_destroy_srq(srq);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_create_qp(struct ibv_pd *pd,
                                              struct ibv_qp_init_attr *qp_init_attr,
                                              struct ibv_qp **qp) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    *qp = real_ibv_create_qp(pd, qp_init_attr);
    return (*qp != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_destroy_qp(struct ibv_qp *qp) {
    if (real_ibv_destroy_qp) {
        int ret = real_ibv_destroy_qp(qp);
        return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                              int attr_mask) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = real_ibv_modify_qp(qp, attr, attr_mask);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                             int attr_mask, struct ibv_qp_init_attr *init_attr) {
    if (init_ibverbs_library() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = real_ibv_query_qp(qp, attr, attr_mask, init_attr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}
