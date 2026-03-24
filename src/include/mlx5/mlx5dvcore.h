/*
 * Copyright (c) 2017 Mellanox Technologies, Inc.  All rights reserved.
 *
 * This software is available to you under the OpenIB.org BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MLX5DV_CORE_H_
#define NCCL_MLX5DV_CORE_H_

/* Basic MLX5 direct verbs structs. Needed to dynamically load MLX5 direct verbs functions without
 * explicit including of MLX5 direct verbs header.
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include "ibvwrap.h"

enum mlx5dv_context_comp_mask {
	MLX5DV_CONTEXT_MASK_CQE_COMPRESION	= 1 << 0,
	MLX5DV_CONTEXT_MASK_SWP			= 1 << 1,
	MLX5DV_CONTEXT_MASK_STRIDING_RQ		= 1 << 2,
	MLX5DV_CONTEXT_MASK_TUNNEL_OFFLOADS	= 1 << 3,
	MLX5DV_CONTEXT_MASK_DYN_BFREGS		= 1 << 4,
	MLX5DV_CONTEXT_MASK_CLOCK_INFO_UPDATE	= 1 << 5,
	MLX5DV_CONTEXT_MASK_FLOW_ACTION_FLAGS	= 1 << 6,
	MLX5DV_CONTEXT_MASK_DC_ODP_CAPS		= 1 << 7,
	MLX5DV_CONTEXT_MASK_HCA_CORE_CLOCK	= 1 << 8,
	MLX5DV_CONTEXT_MASK_NUM_LAG_PORTS	= 1 << 9,
	MLX5DV_CONTEXT_MASK_SIGNATURE_OFFLOAD	= 1 << 10,
	MLX5DV_CONTEXT_MASK_DCI_STREAMS		= 1 << 11,
	MLX5DV_CONTEXT_MASK_WR_MEMCPY_LENGTH	= 1 << 12,
	MLX5DV_CONTEXT_MASK_CRYPTO_OFFLOAD	= 1 << 13,
	MLX5DV_CONTEXT_MASK_MAX_DC_RD_ATOM	= 1 << 14,
	MLX5DV_CONTEXT_MASK_REG_C0		= 1 << 15,
	MLX5DV_CONTEXT_MASK_OOO_RECV_WRS	= 1 << 16,
};

enum mlx5dv_reg_dmabuf_access  {
	MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT		= (1<<0),
};

struct mlx5dv_cqe_comp_caps {
    uint32_t max_num;
    uint32_t supported_format; /* enum mlx5dv_cqe_comp_res_format */
};

struct mlx5dv_sw_parsing_caps {
    uint32_t sw_parsing_offloads; /* Use enum mlx5dv_sw_parsing_offloads */
    uint32_t supported_qpts;
};

struct mlx5dv_striding_rq_caps {
    uint32_t min_single_stride_log_num_of_bytes;
    uint32_t max_single_stride_log_num_of_bytes;
    uint32_t min_single_wqe_log_num_of_strides;
    uint32_t max_single_wqe_log_num_of_strides;
    uint32_t supported_qpts;
};

struct mlx5dv_dci_streams_caps {
    uint8_t max_log_num_concurent;
    uint8_t max_log_num_errored;
};

struct mlx5dv_sig_caps {
    uint64_t block_size; /* use enum mlx5dv_block_size_caps */
    uint32_t block_prot; /* use enum mlx5dv_sig_prot_caps */
    uint16_t t10dif_bg; /* use enum mlx5dv_sig_t10dif_bg_caps */
    uint16_t crc_type; /* use enum mlx5dv_sig_crc_type_caps */
};

struct mlx5dv_crypto_caps {
    /*
     * if failed_selftests != 0 it means there are some self tests errors
     * that may render specific crypto engines unusable. Exact code meaning
     * should be consulted with NVIDIA.
     */
    uint16_t failed_selftests;
    uint8_t crypto_engines; /* use enum mlx5dv_crypto_engines_caps */
    uint8_t wrapped_import_method; /* use enum mlx5dv_crypto_wrapped_import_method_caps */
    uint8_t log_max_num_deks;
    uint32_t flags; /* use enum mlx5dv_crypto_caps_flags */
};

struct mlx5dv_ooo_recv_wrs_caps {
    uint32_t max_rc;
    uint32_t max_xrc;
    uint32_t max_dct;
    uint32_t max_ud;
    uint32_t max_uc;
};

struct mlx5dv_reg {
    uint32_t value;
    uint32_t mask;
};

/*
 * Direct verbs device-specific attributes
 */
struct mlx5dv_context {
    uint8_t     version;
    uint64_t    flags;
    uint64_t    comp_mask;
    struct mlx5dv_cqe_comp_caps cqe_comp_caps;
    struct mlx5dv_sw_parsing_caps sw_parsing_caps;
    struct mlx5dv_striding_rq_caps striding_rq_caps;
    uint32_t    tunnel_offloads_caps;
    uint32_t    max_dynamic_bfregs;
    uint64_t    max_clock_info_update_nsec;
    uint32_t        flow_action_flags; /* use enum mlx5dv_flow_action_cap_flags */
    uint32_t    dc_odp_caps; /* use enum ibv_odp_transport_cap_bits */
    void        *hca_core_clock;
    uint8_t     num_lag_ports;
    struct mlx5dv_sig_caps sig_caps;
    struct mlx5dv_dci_streams_caps dci_streams_caps;
    size_t max_wr_memcpy_length;
    struct mlx5dv_crypto_caps crypto_caps;
    uint64_t max_dc_rd_atom;
    uint64_t max_dc_init_rd_atom;
    struct mlx5dv_reg reg_c0;
    struct mlx5dv_ooo_recv_wrs_caps ooo_recv_wrs_caps;
};

enum mlx5dv_qp_create_flags {
    MLX5DV_QP_CREATE_TUNNEL_OFFLOADS = 1 << 0,
    MLX5DV_QP_CREATE_TIR_ALLOW_SELF_LOOPBACK_UC = 1 << 1,
    MLX5DV_QP_CREATE_TIR_ALLOW_SELF_LOOPBACK_MC = 1 << 2,
    MLX5DV_QP_CREATE_DISABLE_SCATTER_TO_CQE = 1 << 3,
    MLX5DV_QP_CREATE_ALLOW_SCATTER_TO_CQE = 1 << 4,
    MLX5DV_QP_CREATE_PACKET_BASED_CREDIT_MODE = 1 << 5,
    MLX5DV_QP_CREATE_SIG_PIPELINING = 1 << 6,
    MLX5DV_QP_CREATE_OOO_DP = 1 << 7,
};

enum mlx5dv_qp_init_attr_mask {
    MLX5DV_QP_INIT_ATTR_MASK_QP_CREATE_FLAGS    = 1 << 0,
    MLX5DV_QP_INIT_ATTR_MASK_DC         = 1 << 1,
    MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS     = 1 << 2,
    MLX5DV_QP_INIT_ATTR_MASK_DCI_STREAMS            = 1 << 3,
};

enum mlx5dv_dc_type {
    MLX5DV_DCTYPE_DCT     = 1,
    MLX5DV_DCTYPE_DCI,
};

struct mlx5dv_dci_streams {
    uint8_t log_num_concurent;
    uint8_t log_num_errored;
};
struct mlx5dv_dc_init_attr {
    enum mlx5dv_dc_type dc_type;
    union {
        uint64_t dct_access_key;
        struct mlx5dv_dci_streams dci_streams;
    };
};

struct mlx5dv_qp_init_attr {
    uint64_t comp_mask; /* Use enum mlx5dv_qp_init_attr_mask */
    uint32_t create_flags;  /* Use enum mlx5dv_qp_create_flags */
    struct mlx5dv_dc_init_attr  dc_init_attr;
    uint64_t send_ops_flags; /* Use enum mlx5dv_qp_create_send_ops_flags */
};

#endif  // NCCL_MLX5DV_CORE_H_
