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
 * @file doca_verbs_ibv_wrapper.h
 * @brief Wrapper for IB Verbs API calls and structs
 *
 * This wrapper provides an abstraction layer over IB Verbs APIs.
 * It can be enabled by defining DOCA_VERBS_USE_IBV_WRAPPER.
 *
 * When DOCA_VERBS_USE_IBV_WRAPPER is defined:
 * - All IB Verbs API calls are wrapped using dlopen
 * - All IB Verbs structs are wrapped
 * - The wrapper provides a clean abstraction layer
 *
 * When DOCA_VERBS_USE_IBV_WRAPPER is not defined:
 * - Direct IB Verbs APIs are used
 * - No overhead is introduced
 *
 * @{
 */
#ifndef DOCA_VERBS_IBV_WRAPPER_H
#define DOCA_VERBS_IBV_WRAPPER_H

#ifdef DOCA_VERBS_USE_IBV_WRAPPER
#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <linux/types.h>

union ibv_gid {
    uint8_t raw[16];
    struct {
        __be64 subnet_prefix;
        __be64 interface_id;
    } global;
};

struct ibv_global_route {
    union ibv_gid dgid;
    uint32_t flow_label;
    uint8_t sgid_index;
    uint8_t hop_limit;
    uint8_t traffic_class;
};

struct ibv_ah_attr {
    struct ibv_global_route grh;
    uint16_t dlid;
    uint8_t sl;
    uint8_t src_path_bits;
    uint8_t static_rate;
    uint8_t is_global;
    uint8_t port_num;
};

enum ibv_atomic_cap { IBV_ATOMIC_NONE, IBV_ATOMIC_HCA, IBV_ATOMIC_GLOB };

struct ibv_device_attr {
    char fw_ver[64];
    __be64 node_guid;
    __be64 sys_image_guid;
    uint64_t max_mr_size;
    uint64_t page_size_cap;
    uint32_t vendor_id;
    uint32_t vendor_part_id;
    uint32_t hw_ver;
    int max_qp;
    int max_qp_wr;
    unsigned int device_cap_flags;
    int max_sge;
    int max_sge_rd;
    int max_cq;
    int max_cqe;
    int max_mr;
    int max_pd;
    int max_qp_rd_atom;
    int max_ee_rd_atom;
    int max_res_rd_atom;
    int max_qp_init_rd_atom;
    int max_ee_init_rd_atom;
    enum ibv_atomic_cap atomic_cap;
    int max_ee;
    int max_rdd;
    int max_mw;
    int max_raw_ipv6_qp;
    int max_raw_ethy_qp;
    int max_mcast_grp;
    int max_mcast_qp_attach;
    int max_total_mcast_qp_attach;
    int max_ah;
    int max_fmr;
    int max_map_per_fmr;
    int max_srq;
    int max_srq_wr;
    int max_srq_sge;
    uint16_t max_pkeys;
    uint8_t local_ca_ack_delay;
    uint8_t phys_port_cnt;
};

struct ibv_pd {
    struct ibv_context *context;
    uint32_t handle;
};

enum ibv_access_flags {
    IBV_ACCESS_LOCAL_WRITE = 1,
    IBV_ACCESS_REMOTE_WRITE = (1 << 1),
    IBV_ACCESS_REMOTE_READ = (1 << 2),
    IBV_ACCESS_REMOTE_ATOMIC = (1 << 3),
    IBV_ACCESS_MW_BIND = (1 << 4),
    IBV_ACCESS_ZERO_BASED = (1 << 5),
    IBV_ACCESS_ON_DEMAND = (1 << 6),
    IBV_ACCESS_HUGETLB = (1 << 7),
    IBV_ACCESS_FLUSH_GLOBAL = (1 << 8),
    IBV_ACCESS_FLUSH_PERSISTENT = (1 << 9),
    IBV_ACCESS_RELAXED_ORDERING = (1 << 20),
};

struct ibv_device;
struct ibv_context;
struct ibv_mr;
struct ibv_ah;
struct ibv_cq;
struct ibv_comp_channel;
struct ibv_srq;
struct ibv_srq_init_attr;
struct ibv_qp;
struct ibv_qp_init_attr;
struct ibv_qp_attr;
struct ibv_port_attr;

/* *********** IB Verbs API Wrappers *********** */

/**
 * @brief Wrapper for ibv_get_device_list
 */
doca_error_t doca_verbs_wrapper_ibv_get_device_list(int *num_devices,
                                                    struct ibv_device ***device_list);

/**
 * @brief Wrapper for ibv_free_device_list
 */
doca_error_t doca_verbs_wrapper_ibv_free_device_list(struct ibv_device **list);

/**
 * @brief Wrapper for ibv_get_device_name
 */
doca_error_t doca_verbs_wrapper_ibv_get_device_name(struct ibv_device *device,
                                                    const char **device_name);

/**
 * @brief Wrapper for ibv_open_device
 */
doca_error_t doca_verbs_wrapper_ibv_open_device(struct ibv_device *device,
                                                struct ibv_context **context);

/**
 * @brief Wrapper for ibv_close_device
 */
doca_error_t doca_verbs_wrapper_ibv_close_device(struct ibv_context *context);

/**
 * @brief Wrapper for ibv_alloc_pd
 */
doca_error_t doca_verbs_wrapper_ibv_alloc_pd(struct ibv_context *context, struct ibv_pd **pd);

/**
 * @brief Wrapper for ibv_dealloc_pd
 */
doca_error_t doca_verbs_wrapper_ibv_dealloc_pd(struct ibv_pd *pd);

/**
 * @brief Wrapper for ibv_reg_mr
 */
doca_error_t doca_verbs_wrapper_ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length, int access,
                                           struct ibv_mr **mr);

/**
 * @brief Wrapper for ibv_dereg_mr
 */
doca_error_t doca_verbs_wrapper_ibv_dereg_mr(struct ibv_mr *mr);

/**
 * @brief Wrapper for ibv_query_device
 */
doca_error_t doca_verbs_wrapper_ibv_query_device(struct ibv_context *context,
                                                 struct ibv_device_attr *device_attr);

/**
 * @brief Wrapper for ibv_query_port
 */
doca_error_t doca_verbs_wrapper_ibv_query_port(struct ibv_context *context, uint8_t port_num,
                                               struct ibv_port_attr *port_attr);

/**
 * @brief Wrapper for ibv_query_gid
 */
doca_error_t doca_verbs_wrapper_ibv_query_gid(struct ibv_context *context, uint8_t port_num,
                                              int index, union ibv_gid *gid);

/**
 * @brief Wrapper for ibv_create_ah
 */
doca_error_t doca_verbs_wrapper_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr,
                                              struct ibv_ah **ah);

/**
 * @brief Wrapper for ibv_destroy_ah
 */
doca_error_t doca_verbs_wrapper_ibv_destroy_ah(struct ibv_ah *ah);

/**
 * @brief Wrapper for ibv_create_cq
 */
doca_error_t doca_verbs_wrapper_ibv_create_cq(struct ibv_context *context, int cqe,
                                              void *cq_context, struct ibv_comp_channel *channel,
                                              int comp_vector, struct ibv_cq **cq);

/**
 * @brief Wrapper for ibv_destroy_cq
 */
doca_error_t doca_verbs_wrapper_ibv_destroy_cq(struct ibv_cq *cq);

/**
 * @brief Wrapper for ibv_create_srq
 */
doca_error_t doca_verbs_wrapper_ibv_create_srq(struct ibv_pd *pd,
                                               struct ibv_srq_init_attr *srq_init_attr,
                                               struct ibv_srq **srq);

/**
 * @brief Wrapper for ibv_destroy_srq
 */
doca_error_t doca_verbs_wrapper_ibv_destroy_srq(struct ibv_srq *srq);

/**
 * @brief Wrapper for ibv_create_qp
 */
doca_error_t doca_verbs_wrapper_ibv_create_qp(struct ibv_pd *pd,
                                              struct ibv_qp_init_attr *qp_init_attr,
                                              struct ibv_qp **qp);

/**
 * @brief Wrapper for ibv_destroy_qp
 */
doca_error_t doca_verbs_wrapper_ibv_destroy_qp(struct ibv_qp *qp);

/**
 * @brief Wrapper for ibv_modify_qp
 */
doca_error_t doca_verbs_wrapper_ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                              int attr_mask);

/**
 * @brief Wrapper for ibv_query_qp
 */
doca_error_t doca_verbs_wrapper_ibv_query_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
                                             int attr_mask, struct ibv_qp_init_attr *init_attr);

#ifdef __cplusplus
}
#endif

#else /* !DOCA_VERBS_USE_IBV_WRAPPER */

#include <infiniband/verbs.h>

#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"

/* *********** Direct Implementation (when wrapper not used) *********** */

static inline doca_error_t doca_verbs_wrapper_ibv_get_device_list(
    int *num_devices, struct ibv_device ***device_list) {
    *device_list = ibv_get_device_list(num_devices);
    return (*device_list != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_free_device_list(struct ibv_device **list) {
    ibv_free_device_list(list);
    return DOCA_SUCCESS;
}

static inline doca_error_t doca_verbs_wrapper_ibv_get_device_name(struct ibv_device *device,
                                                                  const char **device_name) {
    *device_name = ibv_get_device_name(device);
    return (*device_name != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_open_device(struct ibv_device *device,
                                                              struct ibv_context **context) {
    *context = ibv_open_device(device);
    return (*context != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_close_device(struct ibv_context *context) {
    int ret = ibv_close_device(context);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_alloc_pd(struct ibv_context *context,
                                                           struct ibv_pd **pd) {
    *pd = ibv_alloc_pd(context);
    return (*pd != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_dealloc_pd(struct ibv_pd *pd) {
    int ret = ibv_dealloc_pd(pd);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_reg_mr(struct ibv_pd *pd, void *addr,
                                                         size_t length, int access,
                                                         struct ibv_mr **mr) {
    *mr = ibv_reg_mr(pd, addr, length, access);
    return (*mr != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_dereg_mr(struct ibv_mr *mr) {
    int ret = ibv_dereg_mr(mr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_query_device(
    struct ibv_context *context, struct ibv_device_attr *device_attr) {
    int ret = ibv_query_device(context, device_attr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_query_port(struct ibv_context *context,
                                                             uint8_t port_num,
                                                             struct ibv_port_attr *port_attr) {
    int ret = ibv_query_port(context, port_num, port_attr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_query_gid(struct ibv_context *context,
                                                            uint8_t port_num, int index,
                                                            union ibv_gid *gid) {
    int ret = ibv_query_gid(context, port_num, index, gid);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_create_ah(struct ibv_pd *pd,
                                                            struct ibv_ah_attr *attr,
                                                            struct ibv_ah **ah) {
    *ah = ibv_create_ah(pd, attr);
    return (*ah != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_destroy_ah(struct ibv_ah *ah) {
    int ret = ibv_destroy_ah(ah);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_create_cq(struct ibv_context *context, int cqe,
                                                            void *cq_context,
                                                            struct ibv_comp_channel *channel,
                                                            int comp_vector, struct ibv_cq **cq) {
    *cq = ibv_create_cq(context, cqe, cq_context, channel, comp_vector);
    return (*cq != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_destroy_cq(struct ibv_cq *cq) {
    int ret = ibv_destroy_cq(cq);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_create_srq(
    struct ibv_pd *pd, struct ibv_srq_init_attr *srq_init_attr, struct ibv_srq **srq) {
    *srq = ibv_create_srq(pd, srq_init_attr);
    return (*srq != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_destroy_srq(struct ibv_srq *srq) {
    int ret = ibv_destroy_srq(srq);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_create_qp(struct ibv_pd *pd,
                                                            struct ibv_qp_init_attr *qp_init_attr,
                                                            struct ibv_qp **qp) {
    *qp = ibv_create_qp(pd, qp_init_attr);
    return (*qp != NULL) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_destroy_qp(struct ibv_qp *qp) {
    int ret = ibv_destroy_qp(qp);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_modify_qp(struct ibv_qp *qp,
                                                            struct ibv_qp_attr *attr,
                                                            int attr_mask) {
    int ret = ibv_modify_qp(qp, attr, attr_mask);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

static inline doca_error_t doca_verbs_wrapper_ibv_query_qp(struct ibv_qp *qp,
                                                           struct ibv_qp_attr *attr, int attr_mask,
                                                           struct ibv_qp_init_attr *init_attr) {
    int ret = ibv_query_qp(qp, attr, attr_mask, init_attr);
    return (ret == 0) ? DOCA_SUCCESS : DOCA_ERROR_DRIVER;
}

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_USE_IBV_WRAPPER */

/** @} */

#endif /* DOCA_VERBS_IBV_WRAPPER_H */
