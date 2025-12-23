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
 * @file doca_verbs_mlx5dv_wrapper.h
 * @brief Wrapper for mlx5dv API calls and structs
 *
 * This wrapper provides an abstraction layer over mlx5dv APIs.
 * It can be enabled by defining DOCA_VERBS_USE_MLX5DV_WRAPPER.
 *
 * When DOCA_VERBS_USE_MLX5DV_WRAPPER is defined:
 * - All mlx5dv API calls are wrapped using dlopen
 * - All mlx5dv structs are wrapped
 * - The wrapper provides a clean abstraction layer with dynamic loading
 *
 * When DOCA_VERBS_USE_MLX5DV_WRAPPER is not defined:
 * - Direct mlx5dv APIs are used
 * - No overhead is introduced
 *
 * @{
 */
#ifndef DOCA_VERBS_MLX5DV_WRAPPER_H
#define DOCA_VERBS_MLX5DV_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"

#ifdef DOCA_VERBS_USE_MLX5DV_WRAPPER

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>
#include <sys/types.h>
#include <endian.h>

#include "doca_verbs_ibv_wrapper.h"

#define ETHERNET_LL_SIZE 6

enum mlx5_ib_uapi_uar_alloc_type {
    MLX5_IB_UAPI_UAR_ALLOC_TYPE_BF = 0x0,
    MLX5_IB_UAPI_UAR_ALLOC_TYPE_NC = 0x1,
};

#define MLX5DV_UAR_ALLOC_TYPE_BF MLX5_IB_UAPI_UAR_ALLOC_TYPE_BF
#define MLX5DV_UAR_ALLOC_TYPE_NC MLX5_IB_UAPI_UAR_ALLOC_TYPE_NC

enum mlx5dv_devx_umem_in_mask {
    MLX5DV_UMEM_MASK_DMABUF = 1 << 0,
};

struct mlx5dv_devx_umem_in {
    void *addr;
    size_t size;
    uint32_t access;
    uint64_t pgsz_bitmap;
    uint64_t comp_mask;
    int dmabuf_fd;
};

enum mlx5dv_obj_type {
    MLX5DV_OBJ_QP = 1 << 0,
    MLX5DV_OBJ_CQ = 1 << 1,
    MLX5DV_OBJ_SRQ = 1 << 2,
    MLX5DV_OBJ_RWQ = 1 << 3,
    MLX5DV_OBJ_DM = 1 << 4,
    MLX5DV_OBJ_AH = 1 << 5,
    MLX5DV_OBJ_PD = 1 << 6,
    MLX5DV_OBJ_DEVX = 1 << 7,
};

struct mlx5dv_devx_umem {
    uint32_t umem_id;
};

struct mlx5dv_devx_obj {
    /* Opaque structure - implementation details hidden */
    void *obj;
};

struct doca_gpunetio_ib_mlx5_wqe_av {
    union {
        struct {
            __be32 qkey;
            __be32 reserved;
        } qkey;
        __be64 dc_key;
    } key;
    __be32 dqp_dct;
    uint8_t stat_rate_sl;
    uint8_t fl_mlid;
    __be16 rlid;
    uint8_t reserved0[4];
    uint8_t rmac[ETHERNET_LL_SIZE];
    uint8_t tclass;
    uint8_t hop_limit;
    __be32 grh_gid_fl;
    uint8_t rgid[16];
};

struct mlx5dv_ah {
    struct doca_gpunetio_ib_mlx5_wqe_av *av;
    uint64_t comp_mask;
};

struct mlx5dv_pd {
    uint32_t pdn;
    uint64_t comp_mask;
};

struct mlx5dv_obj {
    struct {
        struct ibv_qp *in;
        struct mlx5dv_qp *out;
    } qp;
    struct {
        struct ibv_cq *in;
        struct mlx5dv_cq *out;
    } cq;
    struct {
        struct ibv_srq *in;
        struct mlx5dv_srq *out;
    } srq;
    struct {
        struct ibv_wq *in;
        struct mlx5dv_rwq *out;
    } rwq;
    struct {
        struct ibv_dm *in;
        struct mlx5dv_dm *out;
    } dm;
    struct {
        struct ibv_ah *in;
        struct mlx5dv_ah *out;
    } ah;
    struct {
        struct ibv_pd *in;
        struct mlx5dv_pd *out;
    } pd;
    struct {
        struct mlx5dv_devx_obj *in;
        struct mlx5dv_devx *out;
    } devx;
};

struct mlx5dv_devx_uar {
    void *reg_addr;
    void *base_addr;
    uint32_t page_id;
    off_t mmap_off;
    uint64_t comp_mask;
};

#define __devx_nullp(typ) ((struct mlx5_ifc_##typ##_bits *)NULL)
#define __devx_st_sz_bits(typ) sizeof(struct mlx5_ifc_##typ##_bits)
#define __devx_bit_sz(typ, fld) sizeof(__devx_nullp(typ)->fld)
#define __devx_bit_off(typ, fld) offsetof(struct mlx5_ifc_##typ##_bits, fld)
#define __devx_dw_off(bit_off) ((bit_off) / 32)
#define __devx_64_off(bit_off) ((bit_off) / 64)
#define __devx_dw_bit_off(bit_sz, bit_off) (32 - (bit_sz) - ((bit_off) & 0x1f))
#define __devx_mask(bit_sz) ((uint32_t)((1ull << (bit_sz)) - 1))
#define __devx_dw_mask(bit_sz, bit_off) (__devx_mask(bit_sz) << __devx_dw_bit_off(bit_sz, bit_off))

#define DEVX_FLD_SZ_BYTES(typ, fld) (__devx_bit_sz(typ, fld) / 8)
#define DEVX_ST_SZ_BYTES(typ) (sizeof(struct mlx5_ifc_##typ##_bits) / 8)
#define DEVX_ST_SZ_DW(typ) (sizeof(struct mlx5_ifc_##typ##_bits) / 32)
#define DEVX_ST_SZ_QW(typ) (sizeof(struct mlx5_ifc_##typ##_bits) / 64)
#define DEVX_UN_SZ_BYTES(typ) (sizeof(union mlx5_ifc_##typ##_bits) / 8)
#define DEVX_UN_SZ_DW(typ) (sizeof(union mlx5_ifc_##typ##_bits) / 32)
#define DEVX_BYTE_OFF(typ, fld) (__devx_bit_off(typ, fld) / 8)
#define DEVX_ADDR_OF(typ, p, fld) ((unsigned char *)(p) + DEVX_BYTE_OFF(typ, fld))

static inline void _devx_set(void *p, uint32_t value, size_t bit_off, size_t bit_sz) {
    __be32 *fld = (__be32 *)(p) + __devx_dw_off(bit_off);
    uint32_t dw_mask = __devx_dw_mask(bit_sz, bit_off);
    uint32_t mask = __devx_mask(bit_sz);

    *fld = htobe32((be32toh(*fld) & (~dw_mask)) |
                   ((value & mask) << __devx_dw_bit_off(bit_sz, bit_off)));
}

#define DEVX_SET(typ, p, fld, v) _devx_set(p, v, __devx_bit_off(typ, fld), __devx_bit_sz(typ, fld))

static inline uint32_t _devx_get(const void *p, size_t bit_off, size_t bit_sz) {
    return ((be32toh(*((const __be32 *)(p) + __devx_dw_off(bit_off))) >>
             __devx_dw_bit_off(bit_sz, bit_off)) &
            __devx_mask(bit_sz));
}

#define DEVX_GET(typ, p, fld) _devx_get(p, __devx_bit_off(typ, fld), __devx_bit_sz(typ, fld))

static inline void _devx_set64(void *p, uint64_t v, size_t bit_off) {
    *((__be64 *)(p) + __devx_64_off(bit_off)) = htobe64(v);
}

#define DEVX_SET64(typ, p, fld, v) _devx_set64(p, v, __devx_bit_off(typ, fld))

static inline uint64_t _devx_get64(const void *p, size_t bit_off) {
    return be64toh(*((const __be64 *)(p) + __devx_64_off(bit_off)));
}

#define DEVX_GET64(typ, p, fld) _devx_get64(p, __devx_bit_off(typ, fld))

struct mlx5dv_context;
struct mlx5dv_port;

/* *********** mlx5dv API Wrappers *********** */

/**
 * @brief Wrapper for mlx5dv_init_obj
 */
doca_error_t doca_verbs_wrapper_mlx5dv_init_obj(struct mlx5dv_obj *obj,
                                                enum mlx5dv_obj_type obj_type);

/**
 * @brief Wrapper for mlx5dv_devx_obj_create
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_create(struct ibv_context *context, const void *in,
                                                       size_t inlen, void *out, size_t outlen,
                                                       struct mlx5dv_devx_obj **obj_out);

/**
 * @brief Wrapper for mlx5dv_devx_obj_destroy
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_destroy(struct mlx5dv_devx_obj *obj);

/**
 * @brief Wrapper for mlx5dv_devx_obj_query
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_query(struct mlx5dv_devx_obj *obj, const void *in,
                                                      size_t inlen, void *out, size_t outlen);

/**
 * @brief Wrapper for mlx5dv_devx_obj_modify
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_modify(struct mlx5dv_devx_obj *obj, const void *in,
                                                       size_t inlen, void *out, size_t outlen);

/**
 * @brief Wrapper for mlx5dv_devx_general_cmd
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_general_cmd(struct ibv_context *context, const void *in,
                                                        size_t inlen, void *out, size_t outlen);

/**
 * @brief Wrapper for mlx5dv_devx_query_eqn
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_query_eqn(struct ibv_context *context, uint32_t cpus,
                                                      uint32_t *eqn);

/**
 * @brief Wrapper for mlx5dv_devx_umem_reg
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_reg(struct ibv_context *context, void *addr,
                                                     size_t size, uint32_t access,
                                                     struct mlx5dv_devx_umem **umem_out);

/**
 * @brief Wrapper for mlx5dv_devx_umem_reg_ex
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_reg_ex(struct ibv_context *context,
                                                        struct mlx5dv_devx_umem_in *umem_in,
                                                        struct mlx5dv_devx_umem **umem_out);

/**
 * @brief Wrapper for mlx5dv_devx_umem_dereg
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_dereg(struct mlx5dv_devx_umem *umem);

/**
 * @brief Wrapper for mlx5dv_devx_alloc_uar
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_alloc_uar(struct ibv_context *context,
                                                      uint32_t uar_type,
                                                      struct mlx5dv_devx_uar **uar_out);

/**
 * @brief Wrapper for mlx5dv_devx_free_uar
 */
doca_error_t doca_verbs_wrapper_mlx5dv_devx_free_uar(struct mlx5dv_devx_uar *uar);

/**
 * @brief Wrapper for mlx5dv_query_device
 */
doca_error_t doca_verbs_wrapper_mlx5dv_query_device(struct ibv_context *context,
                                                    struct mlx5dv_context *attrs_out);

#else /* !DOCA_VERBS_USE_MLX5DV_WRAPPER */

#include <infiniband/mlx5dv.h>

/* *********** Direct API Implementation (inline) *********** */

static inline doca_error_t doca_verbs_wrapper_mlx5dv_init_obj(struct mlx5dv_obj *obj,
                                                              enum mlx5dv_obj_type obj_type) {
    int ret = mlx5dv_init_obj(obj, obj_type);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_create(
    struct ibv_context *context, const void *in, size_t inlen, void *out, size_t outlen,
    struct mlx5dv_devx_obj **obj_out) {
    struct mlx5dv_devx_obj *obj = mlx5dv_devx_obj_create(context, in, inlen, out, outlen);
    if (obj) {
        *obj_out = obj;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_destroy(struct mlx5dv_devx_obj *obj) {
    int ret = mlx5dv_devx_obj_destroy(obj);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_query(struct mlx5dv_devx_obj *obj,
                                                                    const void *in, size_t inlen,
                                                                    void *out, size_t outlen) {
    int ret = mlx5dv_devx_obj_query(obj, in, inlen, out, outlen);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_obj_modify(struct mlx5dv_devx_obj *obj,
                                                                     const void *in, size_t inlen,
                                                                     void *out, size_t outlen) {
    int ret = mlx5dv_devx_obj_modify(obj, in, inlen, out, outlen);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_general_cmd(struct ibv_context *context,
                                                                      const void *in, size_t inlen,
                                                                      void *out, size_t outlen) {
    int ret = mlx5dv_devx_general_cmd(context, in, inlen, out, outlen);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_query_eqn(struct ibv_context *context,
                                                                    uint32_t cpus, uint32_t *eqn) {
    int ret = mlx5dv_devx_query_eqn(context, cpus, eqn);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_reg(
    struct ibv_context *context, void *addr, size_t size, uint32_t access,
    struct mlx5dv_devx_umem **umem_out) {
    struct mlx5dv_devx_umem *umem = mlx5dv_devx_umem_reg(context, addr, size, access);
    if (umem) {
        *umem_out = umem;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_reg_ex(
    struct ibv_context *context, struct mlx5dv_devx_umem_in *umem_in,
    struct mlx5dv_devx_umem **umem_out) {
    struct mlx5dv_devx_umem *umem = mlx5dv_devx_umem_reg_ex(context, umem_in);
    if (umem) {
        *umem_out = umem;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_umem_dereg(
    struct mlx5dv_devx_umem *umem) {
    int ret = mlx5dv_devx_umem_dereg(umem);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_alloc_uar(
    struct ibv_context *context, uint32_t uar_type, struct mlx5dv_devx_uar **uar_out) {
    struct mlx5dv_devx_uar *uar = mlx5dv_devx_alloc_uar(context, uar_type);
    if (uar) {
        *uar_out = uar;
        return DOCA_SUCCESS;
    }
    return DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_devx_free_uar(struct mlx5dv_devx_uar *uar) {
    mlx5dv_devx_free_uar(uar);
    return DOCA_SUCCESS;
}

static inline doca_error_t doca_verbs_wrapper_mlx5dv_query_device(
    struct ibv_context *context, struct mlx5dv_context *attrs_out) {
    int ret = mlx5dv_query_device(context, attrs_out);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

#endif /* !DOCA_VERBS_USE_MLX5DV_WRAPPER */

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_MLX5DV_WRAPPER_H */

/** @} */
