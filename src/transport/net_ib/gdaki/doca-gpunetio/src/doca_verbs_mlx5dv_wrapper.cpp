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
 * @file doca_verbs_mlx5dv_wrapper.cpp
 * @brief Implementation of mlx5dv API wrapper using dlopen
 *
 * This file contains the implementation of the mlx5dv API wrapper
 * using dynamic loading with dlopen when DOCA_VERBS_USE_MLX5DV_WRAPPER is defined.
 */

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <endian.h>
#include <mutex>

#include "doca_verbs_net_wrapper.h"
#include "doca_gpunetio_log.hpp"
#include "host/doca_error.h"

/* *********** dlopen Function Pointers *********** */

static void *mlx5dv_handle = NULL;

/* Function pointer types */
typedef int (*mlx5dv_init_obj_func_t)(struct mlx5dv_obj *obj, enum mlx5dv_obj_type obj_type);
typedef struct mlx5dv_devx_obj *(*mlx5dv_devx_obj_create_func_t)(struct ibv_context *context,
                                                                 const void *in, size_t inlen,
                                                                 void *out, size_t outlen);
typedef int (*mlx5dv_devx_obj_destroy_func_t)(struct mlx5dv_devx_obj *obj);
typedef int (*mlx5dv_devx_obj_query_func_t)(struct mlx5dv_devx_obj *obj, const void *in,
                                            size_t inlen, void *out, size_t outlen);
typedef int (*mlx5dv_devx_obj_modify_func_t)(struct mlx5dv_devx_obj *obj, const void *in,
                                             size_t inlen, void *out, size_t outlen);
typedef int (*mlx5dv_devx_general_cmd_func_t)(struct ibv_context *context, const void *in,
                                              size_t inlen, void *out, size_t outlen);
typedef int (*mlx5dv_devx_query_eqn_func_t)(struct ibv_context *context, uint32_t cpus,
                                            uint32_t *eqn);
typedef struct mlx5dv_devx_umem *(*mlx5dv_devx_umem_reg_func_t)(struct ibv_context *context,
                                                                void *addr, size_t size,
                                                                uint32_t access);
typedef struct mlx5dv_devx_umem *(*mlx5dv_devx_umem_reg_ex_func_t)(
    struct ibv_context *context, struct mlx5dv_devx_umem_in *umem_in);
typedef int (*mlx5dv_devx_umem_dereg_func_t)(struct mlx5dv_devx_umem *umem);
typedef struct mlx5dv_devx_uar *(*mlx5dv_devx_alloc_uar_func_t)(struct ibv_context *context,
                                                                uint32_t uar_type);
typedef void (*mlx5dv_devx_free_uar_func_t)(struct mlx5dv_devx_uar *uar);
typedef int (*mlx5dv_query_device_func_t)(struct ibv_context *context,
                                          struct mlx5dv_context *attrs_out);

/* Function pointers */
static mlx5dv_init_obj_func_t mlx5dv_init_obj_func = NULL;
static mlx5dv_devx_obj_create_func_t mlx5dv_devx_obj_create_func = NULL;
static mlx5dv_devx_obj_destroy_func_t mlx5dv_devx_obj_destroy_func = NULL;
static mlx5dv_devx_obj_query_func_t mlx5dv_devx_obj_query_func = NULL;
static mlx5dv_devx_obj_modify_func_t mlx5dv_devx_obj_modify_func = NULL;
static mlx5dv_devx_general_cmd_func_t mlx5dv_devx_general_cmd_func = NULL;
static mlx5dv_devx_query_eqn_func_t mlx5dv_devx_query_eqn_func = NULL;
static mlx5dv_devx_umem_reg_func_t mlx5dv_devx_umem_reg_func = NULL;
static mlx5dv_devx_umem_reg_ex_func_t mlx5dv_devx_umem_reg_ex_func = NULL;
static mlx5dv_devx_umem_dereg_func_t mlx5dv_devx_umem_dereg_func = NULL;
static mlx5dv_devx_alloc_uar_func_t mlx5dv_devx_alloc_uar_func = NULL;
static mlx5dv_devx_free_uar_func_t mlx5dv_devx_free_uar_func = NULL;
static mlx5dv_query_device_func_t mlx5dv_query_device_func = NULL;

/* *********** dlopen Initialization *********** */

static void doca_verbs_wrapper_init_once(int *ret) {
    mlx5dv_handle = dlopen("libmlx5.so.1", RTLD_NOW);
    if (!mlx5dv_handle) {
        mlx5dv_handle = dlopen("libmlx5.so", RTLD_NOW);
    }
    if (!mlx5dv_handle) {
        DOCA_LOG(LOG_ERR, "Failed to load libmlx5: %s\n", dlerror());
        *ret = -1;
        return;
    }

    /* Load function pointers */
    mlx5dv_init_obj_func = (mlx5dv_init_obj_func_t)dlsym(mlx5dv_handle, "mlx5dv_init_obj");
    mlx5dv_devx_obj_create_func =
        (mlx5dv_devx_obj_create_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_obj_create");
    mlx5dv_devx_obj_destroy_func =
        (mlx5dv_devx_obj_destroy_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_obj_destroy");
    mlx5dv_devx_obj_query_func =
        (mlx5dv_devx_obj_query_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_obj_query");
    mlx5dv_devx_obj_modify_func =
        (mlx5dv_devx_obj_modify_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_obj_modify");
    mlx5dv_devx_general_cmd_func =
        (mlx5dv_devx_general_cmd_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_general_cmd");
    mlx5dv_devx_query_eqn_func =
        (mlx5dv_devx_query_eqn_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_query_eqn");
    mlx5dv_devx_umem_reg_func =
        (mlx5dv_devx_umem_reg_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_umem_reg");
    mlx5dv_devx_umem_reg_ex_func =
        (mlx5dv_devx_umem_reg_ex_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_umem_reg_ex");
    mlx5dv_devx_umem_dereg_func =
        (mlx5dv_devx_umem_dereg_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_umem_dereg");
    mlx5dv_devx_alloc_uar_func =
        (mlx5dv_devx_alloc_uar_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_alloc_uar");
    mlx5dv_devx_free_uar_func =
        (mlx5dv_devx_free_uar_func_t)dlsym(mlx5dv_handle, "mlx5dv_devx_free_uar");
    mlx5dv_query_device_func =
        (mlx5dv_query_device_func_t)dlsym(mlx5dv_handle, "mlx5dv_query_device");

    /* Check if all functions were loaded successfully */
    if (!mlx5dv_init_obj_func || !mlx5dv_devx_obj_create_func || !mlx5dv_devx_obj_destroy_func ||
        !mlx5dv_devx_obj_query_func || !mlx5dv_devx_obj_modify_func ||
        !mlx5dv_devx_general_cmd_func || !mlx5dv_devx_query_eqn_func ||
        !mlx5dv_devx_umem_reg_func || !mlx5dv_devx_umem_reg_ex_func ||
        !mlx5dv_devx_umem_dereg_func || !mlx5dv_devx_alloc_uar_func || !mlx5dv_devx_free_uar_func ||
        !mlx5dv_query_device_func) {
        dlclose(mlx5dv_handle);
        mlx5dv_handle = NULL;
        *ret = -1; /* Failed to load some functions */
        return;
    }

    *ret = 0;
}

static int doca_verbs_wrapper_init_dlopen(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_verbs_wrapper_init_once, &ret);
    return ret;
}

/* *********** Wrapper Implementation with dlopen *********** */

doca_error_t doca_verbs_wrapper_mlx5dv_init_obj(struct mlx5dv_obj *obj,
                                                enum mlx5dv_obj_type obj_type) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_init_obj_func(obj, obj_type);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_create(struct ibv_context *context, const void *in,
                                                       size_t inlen, void *out, size_t outlen,
                                                       struct mlx5dv_devx_obj **obj_out) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    struct mlx5dv_devx_obj *obj = mlx5dv_devx_obj_create_func(context, in, inlen, out, outlen);
    if (obj) {
        *obj_out = obj;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_destroy(struct mlx5dv_devx_obj *obj) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_devx_obj_destroy_func(obj);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_query(struct mlx5dv_devx_obj *obj, const void *in,
                                                      size_t inlen, void *out, size_t outlen) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_devx_obj_query_func(obj, in, inlen, out, outlen);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_modify(struct mlx5dv_devx_obj *obj, const void *in,
                                                       size_t inlen, void *out, size_t outlen) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_devx_obj_modify_func(obj, in, inlen, out, outlen);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_general_cmd(struct ibv_context *context, const void *in,
                                                        size_t inlen, void *out, size_t outlen) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_devx_general_cmd_func(context, in, inlen, out, outlen);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_query_eqn(struct ibv_context *context, uint32_t cpus,
                                                      uint32_t *eqn) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_devx_query_eqn_func(context, cpus, eqn);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_reg(struct ibv_context *context, void *addr,
                                                     size_t size, uint32_t access,
                                                     struct mlx5dv_devx_umem **umem_out) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    struct mlx5dv_devx_umem *umem = mlx5dv_devx_umem_reg_func(context, addr, size, access);
    if (umem) {
        *umem_out = umem;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_reg_ex(struct ibv_context *context,
                                                        struct mlx5dv_devx_umem_in *umem_in,
                                                        struct mlx5dv_devx_umem **umem_out) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    struct mlx5dv_devx_umem *umem = mlx5dv_devx_umem_reg_ex_func(context, umem_in);
    if (umem) {
        *umem_out = umem;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_dereg(struct mlx5dv_devx_umem *umem) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_devx_umem_dereg_func(umem);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_alloc_uar(struct ibv_context *context,
                                                      uint32_t uar_type,
                                                      struct mlx5dv_devx_uar **uar_out) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    struct mlx5dv_devx_uar *uar = mlx5dv_devx_alloc_uar_func(context, uar_type);
    if (uar) {
        *uar_out = uar;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

doca_error_t doca_verbs_wrapper_mlx5dv_devx_free_uar(struct mlx5dv_devx_uar *uar) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    mlx5dv_devx_free_uar_func(uar);
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_wrapper_mlx5dv_query_device(struct ibv_context *context,
                                                    struct mlx5dv_context *attrs_out) {
    if (doca_verbs_wrapper_init_dlopen() != 0) {
        return DOCA_ERROR_NOT_FOUND;
    }
    int ret = mlx5dv_query_device_func(context, attrs_out);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}
