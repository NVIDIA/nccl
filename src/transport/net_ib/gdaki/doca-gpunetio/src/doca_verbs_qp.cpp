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

#include <malloc.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <mutex>
#include <time.h>
#include <string.h>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_internal.hpp"
#include "doca_verbs_device_attr.hpp"
#include "doca_verbs_srq.hpp"
#include "doca_verbs_cq.hpp"
#include "doca_verbs_qp.hpp"
#include "doca_verbs_net_wrapper.h"
#include "common/doca_gpunetio_verbs_def.h"

#define USER_INDEX_MSB_8BITS_MASK 0xFF000000
#define DOCA_VERBS_LOG_OCTOWORD_SIZE 4
#define DOCA_VERBS_OCTOWORD_SIZE (1U << DOCA_VERBS_LOG_OCTOWORD_SIZE)
#define DOCA_VERBS_DATA_SEG_SIZE_IN_BYTES sizeof(struct doca_internal_mlx5_wqe_data_seg)
#define DOCA_VERBS_LOG_WQEBB_SIZE 6
#define DOCA_VERBS_WQEBB_SIZE (1U << DOCA_VERBS_LOG_WQEBB_SIZE)
#define MAX(a, b) std::max(a, b)
#define QP_ATTR(_mask) (DOCA_VERBS_QP_ATTR_##_mask)
#define PRIV_DOCA_MAC_BYTE_LENGTH 6
#define PRIV_DOCA_VERBS_PORT_NUM 1
#define PRIV_DOCA_GID_BYTE_LENGTH 16

enum {
    PRIV_DOCA_MLX5_QP_OPT_PARAM_RRE = (1 << 1),
    PRIV_DOCA_MLX5_QP_OPT_PARAM_RWE = (1 << 3),
    PRIV_DOCA_MLX5_QP_OPT_PARAM_PKEY_INDEX = (1 << 4),
    PRIV_DOCA_MLX5_QP_OPT_PARAM_MIN_RNR_NAK = (1 << 6),
    PRIV_DOCA_MLX5_QP_OPT_PARAM_PORT_NUM = (1 << 16),
    PRIV_DOCA_MLX5_QP_OPT_DSCP = (1 << 17),
    PRIV_DOCA_MLX5_QP_OPT_SGID_INDEX = (1 << 23),
};

enum doca_verbs_qp_state_mod {
    DOCA_VERBS_QP_RST2INIT,
    DOCA_VERBS_QP_INIT2INIT,
    DOCA_VERBS_QP_INIT2RTR,
    DOCA_VERBS_QP_RTR2RTS,
    DOCA_VERBS_QP_RTS2RTS,
};

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {

constexpr uint32_t sc_verbs_qp_doorbell_size = 64;
constexpr uint8_t sc_verbs_qp_log_rq_stride_shift = 4;
constexpr uint32_t sc_verbs_mac_addr_len = 6;
constexpr uint32_t sc_verbs_mac_addr_2msbytes_len = 2;
constexpr uint32_t sc_verbs_log_msg_max = 30;

using create_qp_in = uint32_t[MLX5_ST_SZ_DW(create_qp_in)];
using create_qp_out = uint32_t[MLX5_ST_SZ_DW(create_qp_out)];

using rst2init_qp_in = uint32_t[MLX5_ST_SZ_DW(rst2init_qp_in)];
using rst2init_qp_out = uint32_t[MLX5_ST_SZ_DW(rst2init_qp_out)];

using init2init_qp_in = uint32_t[MLX5_ST_SZ_DW(init2init_qp_in)];
using init2init_qp_out = uint32_t[MLX5_ST_SZ_DW(init2init_qp_out)];

using init2rtr_qp_in = uint32_t[MLX5_ST_SZ_DW(init2rtr_qp_in)];
using init2rtr_qp_out = uint32_t[MLX5_ST_SZ_DW(init2rtr_qp_out)];

using rtr2rts_qp_in = uint32_t[MLX5_ST_SZ_DW(rtr2rts_qp_in)];
using rtr2rts_qp_out = uint32_t[MLX5_ST_SZ_DW(rtr2rts_qp_out)];

using rts2rts_qp_in = uint32_t[MLX5_ST_SZ_DW(rts2rts_qp_in)];
using rts2rts_qp_out = uint32_t[MLX5_ST_SZ_DW(rts2rts_qp_out)];

using qp_2err_in = uint32_t[MLX5_ST_SZ_DW(qp_2err_in)];
using qp_2err_out = uint32_t[MLX5_ST_SZ_DW(qp_2err_out)];

using qp_2rst_in = uint32_t[MLX5_ST_SZ_DW(qp_2rst_in)];
using qp_2rst_out = uint32_t[MLX5_ST_SZ_DW(qp_2rst_out)];

using query_qp_in = uint32_t[MLX5_ST_SZ_DW(query_qp_in)];
using query_qp_out = uint32_t[MLX5_ST_SZ_DW(query_qp_out)];

int rst2init_requested_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(PKEY_INDEX) | QP_ATTR(PORT_NUM) | QP_ATTR(ALLOW_REMOTE_WRITE) |
        QP_ATTR(ALLOW_REMOTE_READ),
};

int init2rtr_requested_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(RQ_PSN) | QP_ATTR(DEST_QP_NUM) | QP_ATTR(PATH_MTU) | QP_ATTR(AH_ATTR) |
        QP_ATTR(MIN_RNR_TIMER),
};

int rtr2rts_requested_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(SQ_PSN) | QP_ATTR(ACK_TIMEOUT) | QP_ATTR(RETRY_CNT) | QP_ATTR(RNR_RETRY),
};

int init2init_optional_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(CURRENT_STATE) | QP_ATTR(NEXT_STATE) | QP_ATTR(PKEY_INDEX) | QP_ATTR(PORT_NUM) |
        QP_ATTR(ALLOW_REMOTE_WRITE) | QP_ATTR(ALLOW_REMOTE_READ),
};

int init2rtr_optional_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(CURRENT_STATE) | QP_ATTR(NEXT_STATE) | QP_ATTR(PKEY_INDEX) |
        QP_ATTR(ALLOW_REMOTE_WRITE) | QP_ATTR(ALLOW_REMOTE_READ),
};

int rtr2rts_optional_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(CURRENT_STATE) | QP_ATTR(NEXT_STATE) | QP_ATTR(MIN_RNR_TIMER) |
        QP_ATTR(ALLOW_REMOTE_WRITE),
};

int rts2rts_optional_attr[DOCA_VERBS_QP_TYPE_RC + 1] = {
    /* [DOCA_VERBS_QP_TYPE_RC] */
    QP_ATTR(CURRENT_STATE) | QP_ATTR(NEXT_STATE) | QP_ATTR(ALLOW_REMOTE_WRITE) |
        QP_ATTR(ALLOW_REMOTE_READ) | QP_ATTR(MIN_RNR_TIMER) | QP_ATTR(AH_ATTR),
};

const char *qp_attr_to_string(int attr) {
    switch (attr) {
        case DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE:
            return "ALLOW_REMOTE_WRITE";
        case DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ:
            return "ALLOW_REMOTE_READ";
        case DOCA_VERBS_QP_ATTR_PKEY_INDEX:
            return "PKEY_INDEX";
        case DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER:
            return "MIN_RNR_TIMER";
        case DOCA_VERBS_QP_ATTR_PORT_NUM:
            return "PORT_NUM";
        case DOCA_VERBS_QP_ATTR_NEXT_STATE:
            return "NEXT_STATE";
        case DOCA_VERBS_QP_ATTR_CURRENT_STATE:
            return "CURRENT_STATE";
        case DOCA_VERBS_QP_ATTR_PATH_MTU:
            return "PATH_MTU";
        case DOCA_VERBS_QP_ATTR_RQ_PSN:
            return "RQ_PSN";
        case DOCA_VERBS_QP_ATTR_SQ_PSN:
            return "SQ_PSN";
        case DOCA_VERBS_QP_ATTR_DEST_QP_NUM:
            return "DEST_QP_NUM";
        case DOCA_VERBS_QP_ATTR_ACK_TIMEOUT:
            return "ACK_TIMEOUT";
        case DOCA_VERBS_QP_ATTR_RETRY_CNT:
            return "RETRY_CNT";
        case DOCA_VERBS_QP_ATTR_RNR_RETRY:
            return "RNR_RETRY";
        case DOCA_VERBS_QP_ATTR_AH_ATTR:
            return "AH_ATTR";
        default:
            break;
    }

    return "UNKNOWN";
}

void print_if_missing_attr(int required_attr_mask, int attr_mask, int attr_to_check) {
    if ((required_attr_mask & attr_to_check) != 0 && (attr_mask & attr_to_check) == 0)
        DOCA_LOG(LOG_ERR, "%s is required but diabled in attr_mask (%d)",
                 qp_attr_to_string(attr_to_check), attr_mask);
}

void print_missing_attrs(int required_attr_mask, int attr_mask) {
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_PKEY_INDEX);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_PORT_NUM);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_NEXT_STATE);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_CURRENT_STATE);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_PATH_MTU);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_RQ_PSN);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_SQ_PSN);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_DEST_QP_NUM);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_ACK_TIMEOUT);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_RETRY_CNT);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_RNR_RETRY);
    print_if_missing_attr(required_attr_mask, attr_mask, DOCA_VERBS_QP_ATTR_AH_ATTR);
}

bool is_X2rst_attrs_valid(int attr_mask) {
    int valid_attr = (DOCA_VERBS_QP_ATTR_CURRENT_STATE | DOCA_VERBS_QP_ATTR_NEXT_STATE);

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    return true;
}

bool is_X2err_attrs_valid(int attr_mask) {
    int valid_attr = (DOCA_VERBS_QP_ATTR_CURRENT_STATE | DOCA_VERBS_QP_ATTR_NEXT_STATE);

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    return true;
}

bool is_rst2init_attrs_valid(int attr_mask, uint32_t qp_type) {
    int required_attr = rst2init_requested_attr[qp_type];
    int valid_attr =
        required_attr | DOCA_VERBS_QP_ATTR_CURRENT_STATE | DOCA_VERBS_QP_ATTR_NEXT_STATE;

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    if ((required_attr & attr_mask) != required_attr) {
        print_missing_attrs(required_attr, attr_mask);
        return false;
    }

    return true;
}

bool is_init2init_attrs_valid(int attr_mask, uint32_t qp_type) {
    int valid_attr = init2init_optional_attr[qp_type];

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    return true;
}

bool is_init2rtr_attrs_valid(int attr_mask, uint32_t qp_type) {
    int required_attr = init2rtr_requested_attr[qp_type];
    int valid_attr = required_attr | init2rtr_optional_attr[qp_type];

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    if ((required_attr & attr_mask) != required_attr) {
        print_missing_attrs(required_attr, attr_mask);
        return false;
    }

    return true;
}

bool is_rtr2rts_attrs_valid(int attr_mask, uint32_t qp_type) {
    int required_attr = rtr2rts_requested_attr[qp_type];
    int valid_attr = required_attr | rtr2rts_optional_attr[qp_type];

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    if ((required_attr & attr_mask) != required_attr) {
        print_missing_attrs(required_attr, attr_mask);
        return false;
    }

    return true;
}

bool is_rts2rts_attrs_valid(int attr_mask, uint32_t qp_type) {
    int valid_attr = rts2rts_optional_attr[qp_type];

    if (attr_mask & ~(valid_attr)) {
        DOCA_LOG(LOG_ERR, "attr_mask contains invalid bit attr_masks (attr_mask=%d)", attr_mask);
        return false;
    }

    return true;
}

void convert_doca_verbs_qp_attr_mask_to_legal_mlx5_qp_opt_param_mask(
    int attr_mask, int &mlx5_opt_mask, doca_verbs_qp_state_mod state_mod) {
    mlx5_opt_mask = 0;

    static const int valid_opt_mask[] = {
        // RST2INIT
        0,
        // INIT2INIT
        DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
            DOCA_VERBS_QP_ATTR_PKEY_INDEX | DOCA_VERBS_QP_ATTR_PORT_NUM,
        // INIT2RTR
        DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
            DOCA_VERBS_QP_ATTR_PKEY_INDEX,
        // RTR2RTS
        DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER,
        // RTS2RTS
        DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE | DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ |
            DOCA_VERBS_QP_ATTR_AH_ATTR | DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER,
    };

    attr_mask &= valid_opt_mask[state_mod];

    if (attr_mask & DOCA_VERBS_QP_ATTR_PKEY_INDEX)
        mlx5_opt_mask |= PRIV_DOCA_MLX5_QP_OPT_PARAM_PKEY_INDEX;

    if (attr_mask & DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER)
        mlx5_opt_mask |= PRIV_DOCA_MLX5_QP_OPT_PARAM_MIN_RNR_NAK;

    if (attr_mask & DOCA_VERBS_QP_ATTR_PORT_NUM)
        mlx5_opt_mask |= PRIV_DOCA_MLX5_QP_OPT_PARAM_PORT_NUM;

    if (attr_mask & DOCA_VERBS_QP_ATTR_AH_ATTR)
        mlx5_opt_mask |= (PRIV_DOCA_MLX5_QP_OPT_SGID_INDEX | PRIV_DOCA_MLX5_QP_OPT_DSCP);

    if (attr_mask & DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE) {
        mlx5_opt_mask |= PRIV_DOCA_MLX5_QP_OPT_PARAM_RWE;
    }

    if (attr_mask & DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ) {
        mlx5_opt_mask |= PRIV_DOCA_MLX5_QP_OPT_PARAM_RRE;
    }
}

doca_error_t query_roce_version(struct ibv_context *ctx, uint8_t sgid_index,
                                uint8_t &roce_version) noexcept {
    uint32_t in[MLX5_ST_SZ_DW(query_roce_address_in)] = {0};
    constexpr auto out_size =
        MLX5_ST_SZ_DW(query_roce_address_out) + MLX5_ST_SZ_DW(roce_addr_layout);
    uint32_t out[out_size] = {0};

    DEVX_SET(query_roce_address_in, &in, opcode, MLX5_CMD_OP_QUERY_ROCE_ADDRESS);
    DEVX_SET(query_roce_address_in, &in, roce_address_index, sgid_index);

    auto ret = doca_verbs_wrapper_mlx5dv_devx_general_cmd(ctx, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query roce version");
        return DOCA_ERROR_DRIVER;
    }

    roce_version = DEVX_GET(query_roce_address_out, out, roce_address[0].roce_version);

    DOCA_LOG(LOG_INFO, "roce_version = %d", roce_version);

    return DOCA_SUCCESS;
}

doca_error_t convert_doca_mtu_size_to_prm_mtu_size(doca_verbs_mtu_size mtu_size,
                                                   uint32_t &prm_mtu_size) noexcept {
    switch (mtu_size) {
        case DOCA_VERBS_MTU_SIZE_256_BYTES:
            prm_mtu_size = MLX5_QPC_MTU_256_BYTES;
            break;
        case DOCA_VERBS_MTU_SIZE_512_BYTES:
            prm_mtu_size = MLX5_QPC_MTU_512_BYTES;
            break;
        case DOCA_VERBS_MTU_SIZE_1K_BYTES:
            prm_mtu_size = MLX5_QPC_MTU_1K_BYTES;
            break;
        case DOCA_VERBS_MTU_SIZE_2K_BYTES:
            prm_mtu_size = MLX5_QPC_MTU_2K_BYTES;
            break;
        case DOCA_VERBS_MTU_SIZE_4K_BYTES:
            prm_mtu_size = MLX5_QPC_MTU_4K_BYTES;
            break;
        case DOCA_VERBS_MTU_SIZE_RAW_ETHERNET:
            prm_mtu_size = MLX5_QPC_MTU_RAW_ETHERNET_QP;
            break;
        default:
            DOCA_LOG(LOG_ERR, "Can't convert invalid DOCA mtu size=%d", mtu_size);
            return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

doca_error_t convert_prm_mtu_size_to_doca_verbs_mtu_size(uint32_t prm_mtu_size,
                                                         doca_verbs_mtu_size &mtu_size) noexcept {
    switch (prm_mtu_size) {
        case MLX5_QPC_MTU_256_BYTES:
            mtu_size = DOCA_VERBS_MTU_SIZE_256_BYTES;
            break;
        case MLX5_QPC_MTU_512_BYTES:
            mtu_size = DOCA_VERBS_MTU_SIZE_512_BYTES;
            break;
        case MLX5_QPC_MTU_1K_BYTES:
            mtu_size = DOCA_VERBS_MTU_SIZE_1K_BYTES;
            break;
        case MLX5_QPC_MTU_2K_BYTES:
            mtu_size = DOCA_VERBS_MTU_SIZE_2K_BYTES;
            break;
        case MLX5_QPC_MTU_4K_BYTES:
            mtu_size = DOCA_VERBS_MTU_SIZE_4K_BYTES;
            break;
        case MLX5_QPC_MTU_RAW_ETHERNET_QP:
            mtu_size = DOCA_VERBS_MTU_SIZE_RAW_ETHERNET;
            break;
        default:
            DOCA_LOG(LOG_ERR, "Can't convert invalid prm mtu size=%d", mtu_size);
            return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

int random_in_range(int min, int max) { return min + rand() % (max - min + 1); }

doca_error_t resolve_remote_mac(ibv_pd *pd_handle, uint8_t local_port_num, uint32_t local_gid_index,
                                uint8_t remote_gid[PRIV_DOCA_GID_BYTE_LENGTH], uint8_t hop_limit,
                                uint8_t is_global,
                                uint8_t mac[PRIV_DOCA_MAC_BYTE_LENGTH]) noexcept {
    struct ibv_ah_attr attr = {};

    attr.port_num = local_port_num;
    attr.grh.sgid_index = local_gid_index;
    memcpy(attr.grh.dgid.raw, remote_gid, PRIV_DOCA_GID_BYTE_LENGTH);
    attr.grh.hop_limit = hop_limit;
    attr.is_global = is_global;

    struct ibv_ah *ah;
    auto ah_ret = doca_verbs_wrapper_ibv_create_ah(pd_handle, &attr, &ah);
    if (ah_ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create ibv_ah. ret=%d", ah_ret);
        return ah_ret;
    }

    struct mlx5dv_obj dv_obj {};
    struct mlx5dv_ah dv_ah {};

    dv_obj.ah.in = ah;
    dv_obj.ah.out = &dv_ah;

    auto ret = doca_verbs_wrapper_mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_AH);
    if (ret != DOCA_SUCCESS) {
        auto destroy_ret = doca_verbs_wrapper_ibv_destroy_ah(ah);
        if (destroy_ret != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy ibv_ah. ret=%d", destroy_ret);
        }
        DOCA_LOG(LOG_ERR, "Failed to initialize mlx5dv_ah from ibv_ah. ret=%d", ret);
        return DOCA_ERROR_DRIVER;
    }

    // Check needed for coverity
    if (dv_ah.av == nullptr) {
        auto destroy_ret = doca_verbs_wrapper_ibv_destroy_ah(ah);
        if (destroy_ret != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy ibv_ah. ret=%d", destroy_ret);
        }
        DOCA_LOG(LOG_ERR, "Failed to initialize mlx5dv_ah from ibv_ah mlx5dv_ah::av is NULL");
        return DOCA_ERROR_DRIVER;
    }

    memcpy(mac, dv_ah.av->rmac, PRIV_DOCA_MAC_BYTE_LENGTH);

    auto destroy_ah_status = doca_verbs_wrapper_ibv_destroy_ah(ah);
    if (destroy_ah_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy ibv_ah. ret=%d", destroy_ah_status);
        return destroy_ah_status;
    }

    return DOCA_SUCCESS;
}

doca_error_t convert_prm_qp_state_to_doca_verbs_qp_state(uint32_t qp_state,
                                                         doca_verbs_qp_state &state) {
    switch (qp_state) {
        case MLX5_QPC_STATE_RST:
            state = DOCA_VERBS_QP_STATE_RST;
            break;
        case MLX5_QPC_STATE_INIT:
            state = DOCA_VERBS_QP_STATE_INIT;
            break;
        case MLX5_QPC_STATE_RTR:
            state = DOCA_VERBS_QP_STATE_RTR;
            break;
        case MLX5_QPC_STATE_RTS:
            state = DOCA_VERBS_QP_STATE_RTS;
            break;
        case MLX5_QPC_STATE_ERR:
            state = DOCA_VERBS_QP_STATE_ERR;
            break;
        default:
            DOCA_LOG(LOG_ERR, "Can't convert invalid prm qp state=%d", qp_state);
            return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

} /* namespace */

/**********************************************************************************************************************
 * doca_verbs Member Functions
 *********************************************************************************************************************/

bool doca_verbs_qp::is_qp_attr_state_valid(enum doca_verbs_qp_state state) noexcept {
    switch (state) {
        case DOCA_VERBS_QP_STATE_RST:
        case DOCA_VERBS_QP_STATE_INIT:
        case DOCA_VERBS_QP_STATE_RTR:
        case DOCA_VERBS_QP_STATE_RTS:
        case DOCA_VERBS_QP_STATE_ERR:
            return true;
        default:
            DOCA_LOG(LOG_ERR, "state is invalid (value is %u)", state);
            return false;
    }

    // Shouldn't reach this
    return true;
}

bool doca_verbs_qp::is_qp_attr_path_mtu_valid(enum doca_verbs_mtu_size path_mtu) noexcept {
    switch (path_mtu) {
        case DOCA_VERBS_MTU_SIZE_256_BYTES:
        case DOCA_VERBS_MTU_SIZE_512_BYTES:
        case DOCA_VERBS_MTU_SIZE_1K_BYTES:
        case DOCA_VERBS_MTU_SIZE_2K_BYTES:
        case DOCA_VERBS_MTU_SIZE_4K_BYTES:
            return true;
        default:
            DOCA_LOG(LOG_ERR, "path_mtu is invalid (value is %u)", path_mtu);
            return false;
    }

    // Shouldn't reach this
    return true;
}

// No value of PSN causes a value (we print a warning and mask it in case of overflow)
uint32_t doca_verbs_qp::is_qp_attr_queue_psn_valid(uint32_t psn) noexcept {
    if (psn & ~0xffffff) {
        DOCA_LOG(LOG_ERR, "PSN value overflow (max is %x). Masking to 24 bits", 0xffffff);
        psn &= 0xffffff;
    }

    return psn;
}

bool doca_verbs_qp::is_qp_attr_ah_add_type_valid(enum doca_verbs_addr_type addr_type) noexcept {
    switch (addr_type) {
        case DOCA_VERBS_ADDR_TYPE_IPv4:
        case DOCA_VERBS_ADDR_TYPE_IPv6:
        case DOCA_VERBS_ADDR_TYPE_IB_GRH:
        case DOCA_VERBS_ADDR_TYPE_IB_NO_GRH:
            return true;
        default:
            DOCA_LOG(LOG_ERR, "addr_type is invalid (value is %u)", addr_type);
            return false;
    }

    // Shouldn't reach this
    return true;
}

bool doca_verbs_qp::is_qp_attr_ah_sgid_index_valid(uint8_t sgid_index) noexcept {
    if (sgid_index >= m_verbs_device_attr->m_gid_table_size) {
        DOCA_LOG(LOG_ERR, "sgid_index should be less than %u (value is %u)",
                 m_verbs_device_attr->m_gid_table_size - 1, sgid_index);
        return false;
    }

    return true;
}

bool doca_verbs_qp::is_qp_attr_pkey_index_valid(uint16_t pkey_index) noexcept {
    if (pkey_index > m_verbs_device_attr->m_max_pkeys) {
        DOCA_LOG(LOG_ERR, "pkey_index should be less than %u (value is %u)",
                 m_verbs_device_attr->m_max_pkeys, pkey_index);
        return false;
    }

    return true;
}

bool doca_verbs_qp::is_qp_attr_port_num_valid(uint16_t port_num) noexcept {
    if (port_num > m_verbs_device_attr->m_phys_port_cnt || port_num < 1) {
        DOCA_LOG(LOG_ERR, "port_num should be from %u to %u (value is %u)", 1,
                 m_verbs_device_attr->m_phys_port_cnt, port_num);
        return false;
    }

    return true;
}

bool doca_verbs_qp::is_qp_attr_valid(struct doca_verbs_qp_attr *verbs_qp_attr,
                                     int attr_mask) noexcept {
    if ((attr_mask & DOCA_VERBS_QP_ATTR_CURRENT_STATE) &&
        !is_qp_attr_state_valid(verbs_qp_attr->current_state))
        return false;
    if ((attr_mask & DOCA_VERBS_QP_ATTR_NEXT_STATE) &&
        !is_qp_attr_state_valid(verbs_qp_attr->next_state))
        return false;
    if ((attr_mask & DOCA_VERBS_QP_ATTR_PATH_MTU) &&
        !is_qp_attr_path_mtu_valid(verbs_qp_attr->path_mtu))
        return false;
    if ((attr_mask & DOCA_VERBS_QP_ATTR_RQ_PSN))
        verbs_qp_attr->rq_psn = is_qp_attr_queue_psn_valid(verbs_qp_attr->rq_psn);
    if ((attr_mask & DOCA_VERBS_QP_ATTR_SQ_PSN))
        verbs_qp_attr->sq_psn = is_qp_attr_queue_psn_valid(verbs_qp_attr->sq_psn);
    if ((attr_mask & DOCA_VERBS_QP_ATTR_AH_ATTR) &&
        !is_qp_attr_ah_add_type_valid(verbs_qp_attr->ah_attr->addr_type))
        return false;
    if ((attr_mask & DOCA_VERBS_QP_ATTR_AH_ATTR) &&
        !is_qp_attr_ah_sgid_index_valid(verbs_qp_attr->ah_attr->sgid_index))
        return false;
    if ((attr_mask & DOCA_VERBS_QP_ATTR_PKEY_INDEX) &&
        !is_qp_attr_pkey_index_valid(verbs_qp_attr->pkey_index))
        return false;
    if ((attr_mask & DOCA_VERBS_QP_ATTR_PORT_NUM) &&
        !is_qp_attr_port_num_valid(verbs_qp_attr->port_num))
        return false;

    return true;
}

doca_verbs_qp_state doca_verbs_qp::get_current_state() const noexcept { return m_current_state; }

doca_error_t doca_verbs_qp::create_qp_obj(
    uint32_t uar_id, uint32_t log_rq_size, uint32_t log_sq_size_wqebb, uint32_t log_stride,
    uint64_t dbr_umem_offset, uint32_t dbr_umem_id, uint32_t wq_umem_id,
    struct doca_verbs_qp_init_attr &verbs_qp_init_attr) noexcept {
    create_qp_in create_in{0};
    create_qp_out create_out{0};

    void *qpc = MLX5_ADDR_OF(create_qp_in, create_in, qpc);

    DEVX_SET(create_qp_in, create_in, opcode, MLX5_CMD_OP_CREATE_QP);
    DEVX_SET(qpc, qpc, st, MLX5_QPC_ST_RC);

    struct mlx5dv_pd dvpd;
    struct mlx5dv_obj dv_obj;
    // Query pdn
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.pd.in = m_pd;
    dv_obj.pd.out = &dvpd;

    auto ret = doca_verbs_wrapper_mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Error in mlx5dv PD initialization");
        return DOCA_ERROR_DRIVER;
    }

    DEVX_SET(qpc, qpc, pd, dvpd.pdn);

    DEVX_SET(qpc, qpc, user_index, verbs_qp_init_attr.user_index);
    DEVX_SET(qpc, qpc, uar_page, uar_id);

    if (m_sq_size_wqebb > 0) {
        if (verbs_qp_init_attr.send_cq == nullptr) {
            DOCA_LOG(LOG_ERR, "Failed to create QP. Send CQ is null");
            return DOCA_ERROR_INVALID_VALUE;
        }
        DEVX_SET(qpc, qpc, cqn_snd, verbs_qp_init_attr.send_cq->get_cqn());
        DEVX_SET(qpc, qpc, log_sq_size, log_sq_size_wqebb);
    } else {
        DEVX_SET(qpc, qpc, no_sq, 1);
    }

    if ((m_rq_size > 0) || (verbs_qp_init_attr.srq != nullptr)) {
        if (verbs_qp_init_attr.receive_cq == nullptr) {
            DOCA_LOG(LOG_ERR, "Failed to create QP. Receive CQ is null");
            return DOCA_ERROR_INVALID_VALUE;
        }

        DEVX_SET(qpc, qpc, cqn_rcv, verbs_qp_init_attr.receive_cq->get_cqn());

        if (verbs_qp_init_attr.srq != nullptr) {
            /* Case of SRQ */
            DEVX_SET(qpc, qpc, srqn_rmpn_xrqn, verbs_qp_init_attr.srq->get_srqn());
            DEVX_SET(qpc, qpc, rq_type, MLX5_QPC_RQ_TYPE_SRQ_RMP_XRC_SRQ_XRQ);
            m_srq = verbs_qp_init_attr.srq;
        } else if (m_rq_size > 0) {
            /* Case of regular RQ */
            DEVX_SET(qpc, qpc, log_rq_stride, log_stride);
            DEVX_SET(qpc, qpc, log_rq_size, log_rq_size);
            DEVX_SET(qpc, qpc, rq_type, MLX5_QPC_RQ_TYPE_REGULAR);
        }
    } else {
        /* Case of no RQ */
        DEVX_SET(qpc, qpc, rq_type, MLX5_QPC_RQ_TYPE_ZERO_SIZE_RQ);
    }

    // DEVX_SET(qpc, qpc, cs_req, 0);            // Disable CS Request
    // DEVX_SET(qpc, qpc, cs_res, 0);            // Disable CS Response

    DEVX_SET(qpc, qpc, dbr_umem_valid, 1);
    DEVX_SET(qpc, qpc, dbr_umem_id, dbr_umem_id);
    DEVX_SET64(qpc, qpc, dbr_addr, dbr_umem_offset);
    DEVX_SET64(qpc, qpc, cd_master, verbs_qp_init_attr.core_direct_master);
    DEVX_SET(create_qp_in, create_in, wq_umem_id, wq_umem_id);
    DEVX_SET(create_qp_in, create_in, wq_umem_valid, 1);

    /* Since wq_umem_valid == 1, FW deduces page size from umem and this field is reserved */
    DEVX_SET(qpc, qpc, log_page_size, 0);

    /* Create DevX object */
    auto status = doca_verbs_wrapper_mlx5dv_devx_obj_create(
        m_ibv_ctx, create_in, sizeof(create_in), create_out, sizeof(create_out), &m_qp_obj);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create QP. DevX error, syndrome=0x%x",
                 DEVX_GET(nop_out, create_out, syndrome));
        return status;
    }

    m_qp_num = DEVX_GET(create_qp_out, create_out, qpn);
    m_current_state = DOCA_VERBS_QP_STATE_RST;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::rst2init(struct doca_verbs_qp_attr &verbs_qp_attr,
                                     int attr_mask) noexcept {
    rst2init_qp_in in{0};
    rst2init_qp_out out{0};

    if (!is_rst2init_attrs_valid(attr_mask, m_qp_type)) {
        DOCA_LOG(LOG_ERR, "rst2init attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    void *qpc = MLX5_ADDR_OF(rst2init_qp_in, &in, qpc);
    DEVX_SET(rst2init_qp_in, &in, opcode, MLX5_CMD_OP_RST2INIT_QP);
    DEVX_SET(rst2init_qp_in, &in, qpn, m_qp_num);
    DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, verbs_qp_attr.port_num);
    DEVX_SET(qpc, qpc, pm_state, MLX5_QPC_PM_STATE_MIGRATED);
    // DEVX_SET(qpc, qpc, counter_set_id, 0x0);  // Not connected to a counter set
    DEVX_SET(qpc, qpc, primary_address_path.pkey_index, verbs_qp_attr.pkey_index);

    if (verbs_qp_attr.allow_remote_write == 1) {
        DEVX_SET(qpc, qpc, rwe, 1);
    }

    if ((attr_mask & DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ) &&
        verbs_qp_attr.allow_remote_read == 1) {
        DEVX_SET(qpc, qpc, rre, 1);
    }

    if (verbs_qp_attr.allow_remote_write == 1) {
        DEVX_SET(qpc, qpc, rwe, 1);
    }

    if (verbs_qp_attr.allow_remote_atomic > DOCA_VERBS_QP_ATOMIC_MODE_NONE) {
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode, verbs_qp_attr.allow_remote_atomic);
    }

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP rst2init");
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_INIT;

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully moved to Init state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::init2init(struct doca_verbs_qp_attr &verbs_qp_attr,
                                      int attr_mask) noexcept {
    init2init_qp_in in{0};
    init2init_qp_out out{0};

    if (!is_init2init_attrs_valid(attr_mask, m_qp_type)) {
        DOCA_LOG(LOG_ERR, "init2init attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    void *qpc = MLX5_ADDR_OF(init2init_qp_in, &in, qpc);
    DEVX_SET(init2init_qp_in, &in, opcode, MLX5_CMD_OP_INIT2INIT_QP);
    DEVX_SET(init2init_qp_in, &in, qpn, m_qp_num);
    DEVX_SET(qpc, qpc, primary_address_path.vhca_port_num, verbs_qp_attr.port_num);
    DEVX_SET(qpc, qpc, primary_address_path.pkey_index, verbs_qp_attr.pkey_index);

    if (verbs_qp_attr.allow_remote_write == 1) {
        DEVX_SET(qpc, qpc, rwe, 1);
    }

    if ((attr_mask & DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ) &&
        verbs_qp_attr.allow_remote_read == 1) {
        DEVX_SET(qpc, qpc, rre, 1);
    }

    if (verbs_qp_attr.allow_remote_atomic > DOCA_VERBS_QP_ATOMIC_MODE_NONE) {
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode, verbs_qp_attr.allow_remote_atomic);
    }

    int mlx5_opt_param_mask{0};
    convert_doca_verbs_qp_attr_mask_to_legal_mlx5_qp_opt_param_mask(attr_mask, mlx5_opt_param_mask,
                                                                    DOCA_VERBS_QP_INIT2INIT);
    DEVX_SET(init2init_qp_in, &in, opt_param_mask, mlx5_opt_param_mask);

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP init2init");
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_INIT;

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully moved to Init state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::init2rtr(struct doca_verbs_qp_attr &verbs_qp_attr,
                                     int attr_mask) noexcept {
    if (!is_init2rtr_attrs_valid(attr_mask, m_qp_type)) {
        DOCA_LOG(LOG_ERR, "init2rtr attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if ((attr_mask & DOCA_VERBS_QP_ATTR_AH_ATTR) && !verbs_qp_attr.ah_attr) {
        DOCA_LOG(LOG_ERR, "AH_ATTR mask is enabled but ah_attr=nullptr");
        return DOCA_ERROR_INVALID_VALUE;
    }

    init2rtr_qp_in in{0};
    init2rtr_qp_out out{0};

    void *qpc = MLX5_ADDR_OF(init2rtr_qp_in, in, qpc);
    DEVX_SET(init2rtr_qp_in, in, opcode, MLX5_CMD_OP_INIT2RTR_QP);
    DEVX_SET(init2rtr_qp_in, in, qpn, m_qp_num);
    DEVX_SET(qpc, qpc, next_rcv_psn, verbs_qp_attr.rq_psn);
    DEVX_SET(qpc, qpc, remote_qpn, verbs_qp_attr.dest_qp_num);
    DEVX_SET(qpc, qpc, log_msg_max, sc_verbs_log_msg_max);

    uint32_t prm_mtu{};
    auto status = convert_doca_mtu_size_to_prm_mtu_size(verbs_qp_attr.path_mtu, prm_mtu);
    if (status != DOCA_SUCCESS) return status;
    DEVX_SET(qpc, qpc, mtu, prm_mtu);

    if (attr_mask & DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER)
        DEVX_SET(qpc, qpc, min_rnr_nak, verbs_qp_attr.min_rnr_timer);
    if ((verbs_qp_attr.ah_attr->addr_type == DOCA_VERBS_ADDR_TYPE_IB_GRH) ||
        (verbs_qp_attr.ah_attr->addr_type == DOCA_VERBS_ADDR_TYPE_IB_NO_GRH)) { /* IB */
        DEVX_SET(qpc, qpc, primary_address_path.tclass, verbs_qp_attr.ah_attr->traffic_class);
        DEVX_SET(qpc, qpc, primary_address_path.rlid, verbs_qp_attr.ah_attr->dlid);
        DEVX_SET(qpc, qpc, primary_address_path.sl, verbs_qp_attr.ah_attr->sl);
    }
    DEVX_SET(qpc, qpc, primary_address_path.stat_rate, verbs_qp_attr.ah_attr->static_rate);

    if (verbs_qp_attr.ah_attr->addr_type != DOCA_VERBS_ADDR_TYPE_IB_NO_GRH) {
        memcpy(MLX5_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
               verbs_qp_attr.ah_attr->gid.raw, sizeof(struct doca_verbs_gid));
        DEVX_SET(qpc, qpc, primary_address_path.hop_limit, verbs_qp_attr.ah_attr->hop_limit);
        DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, verbs_qp_attr.ah_attr->sgid_index);
    }

    if ((verbs_qp_attr.ah_attr->addr_type == DOCA_VERBS_ADDR_TYPE_IPv4) ||
        (verbs_qp_attr.ah_attr->addr_type == DOCA_VERBS_ADDR_TYPE_IPv6)) { /* ROCE */
        uint8_t dest_mac[PRIV_DOCA_MAC_BYTE_LENGTH];
        status =
            resolve_remote_mac(m_pd, PRIV_DOCA_VERBS_PORT_NUM, verbs_qp_attr.ah_attr->sgid_index,
                               verbs_qp_attr.ah_attr->gid.raw, verbs_qp_attr.ah_attr->hop_limit,
                               verbs_qp_attr.ah_attr->is_global, dest_mac);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get remote MAC");
            return status;
        }

        memcpy(MLX5_ADDR_OF(qpc, qpc, primary_address_path.rmac_47_32), dest_mac,
               sc_verbs_mac_addr_2msbytes_len);
        memcpy(MLX5_ADDR_OF(qpc, qpc, primary_address_path.rmac_31_0),
               dest_mac + sc_verbs_mac_addr_2msbytes_len,
               sc_verbs_mac_addr_len - sc_verbs_mac_addr_2msbytes_len);
    }

    if (verbs_qp_attr.ah_attr->addr_type == DOCA_VERBS_ADDR_TYPE_IB_GRH) {
        DEVX_SET(qpc, qpc, primary_address_path.grh, 1);
    }

    if (m_verbs_device_attr->m_port_type == MLX5_CAP_PORT_TYPE_ETH) {
        uint8_t roce_version{};
        status = query_roce_version(m_ibv_ctx, verbs_qp_attr.ah_attr->sgid_index, roce_version);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to query roce version");
            return status;
        }

        if (roce_version >= MLX5_ROCE_ADDR_LAYOUT_ROCE_VERSION_VERSION_2_0) {
            // generate a random udp_sport
            srand(time(NULL));
            uint16_t udp_sport = (uint16_t)random_in_range(m_verbs_device_attr->m_min_udp_sport,
                                                           m_verbs_device_attr->m_max_udp_sport);
            DOCA_LOG(LOG_INFO, "Generated udp_sport = %d", udp_sport);

            DEVX_SET(qpc, qpc, primary_address_path.udp_sport, udp_sport);
            DEVX_SET(qpc, qpc, primary_address_path.dscp,
                     verbs_qp_attr.ah_attr->traffic_class >> 2);
        }
    }

    DEVX_SET(qpc, qpc, primary_address_path.pkey_index, verbs_qp_attr.pkey_index);
    if (verbs_qp_attr.allow_remote_write == 1) {
        DEVX_SET(qpc, qpc, rwe, 1);
    }

    if ((attr_mask & DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ) &&
        verbs_qp_attr.allow_remote_read == 1) {
        DEVX_SET(qpc, qpc, rre, 1);
    }

    if (verbs_qp_attr.allow_remote_atomic > DOCA_VERBS_QP_ATOMIC_MODE_NONE) {
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode, verbs_qp_attr.allow_remote_atomic);
    }

    int mlx5_opt_param_mask{0};
    convert_doca_verbs_qp_attr_mask_to_legal_mlx5_qp_opt_param_mask(attr_mask, mlx5_opt_param_mask,
                                                                    DOCA_VERBS_QP_INIT2RTR);
    DEVX_SET(init2rtr_qp_in, in, opt_param_mask, mlx5_opt_param_mask);

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP init2rtr, syndrome=0x%x",
                 DEVX_GET(nop_out, out, syndrome));
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_RTR;
    m_addr_type = verbs_qp_attr.ah_attr->addr_type;

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully moved to RTR state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::rtr2rts(struct doca_verbs_qp_attr &verbs_qp_attr,
                                    int attr_mask) noexcept {
    rtr2rts_qp_in in{0};
    rtr2rts_qp_out out{0};

    if (!is_rtr2rts_attrs_valid(attr_mask, m_qp_type)) {
        DOCA_LOG(LOG_ERR, "rtr2rts attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    void *qpc = MLX5_ADDR_OF(rtr2rts_qp_in, &in, qpc);
    DEVX_SET(rtr2rts_qp_in, &in, opcode, MLX5_CMD_OP_RTR2RTS_QP);
    DEVX_SET(rtr2rts_qp_in, &in, qpn, m_qp_num);
    DEVX_SET(qpc, qpc, next_send_psn, verbs_qp_attr.sq_psn);
    if (attr_mask & DOCA_VERBS_QP_ATTR_ACK_TIMEOUT)
        DEVX_SET(qpc, qpc, primary_address_path.ack_timeout, verbs_qp_attr.ack_timeout);
    if (attr_mask & DOCA_VERBS_QP_ATTR_RETRY_CNT)
        DEVX_SET(qpc, qpc, retry_count, verbs_qp_attr.retry_cnt);
    if (attr_mask & DOCA_VERBS_QP_ATTR_RNR_RETRY)
        DEVX_SET(qpc, qpc, rnr_retry, verbs_qp_attr.rnr_retry);
    if (attr_mask & DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER)
        DEVX_SET(qpc, qpc, min_rnr_nak, verbs_qp_attr.min_rnr_timer);
    if (verbs_qp_attr.allow_remote_write == 1) {
        DEVX_SET(qpc, qpc, rwe, 1);
    }
    if (verbs_qp_attr.allow_remote_atomic > DOCA_VERBS_QP_ATOMIC_MODE_NONE) {
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode, verbs_qp_attr.allow_remote_atomic);
    }

    DEVX_SET(qpc, qpc, log_ack_req_freq, 0x0);  // 8

    int mlx5_opt_param_mask{0};
    convert_doca_verbs_qp_attr_mask_to_legal_mlx5_qp_opt_param_mask(attr_mask, mlx5_opt_param_mask,
                                                                    DOCA_VERBS_QP_RTR2RTS);

    DEVX_SET(rtr2rts_qp_in, &in, opt_param_mask, mlx5_opt_param_mask);

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP rtr2rts");
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_RTS;

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully moved to RTS state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::rts2rts(struct doca_verbs_qp_attr &verbs_qp_attr,
                                    int attr_mask) noexcept {
    if (!is_rts2rts_attrs_valid(attr_mask, m_qp_type)) {
        DOCA_LOG(LOG_ERR, "rts2rts attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if ((attr_mask & DOCA_VERBS_QP_ATTR_AH_ATTR) && !verbs_qp_attr.ah_attr) {
        DOCA_LOG(LOG_ERR, "AH_ATTR mask is enabled but ah_attr=nullptr");
        return DOCA_ERROR_INVALID_VALUE;
    }

    rts2rts_qp_in in{0};
    rts2rts_qp_out out{0};

    void *qpc = MLX5_ADDR_OF(rts2rts_qp_in, in, qpc);
    DEVX_SET(rts2rts_qp_in, in, opcode, MLX5_CMD_OP_RTS2RTS_QP);
    DEVX_SET(rts2rts_qp_in, in, qpn, m_qp_num);

    if (attr_mask & DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER)
        DEVX_SET(qpc, qpc, min_rnr_nak, verbs_qp_attr.min_rnr_timer);
    if (verbs_qp_attr.allow_remote_write == 1) {
        DEVX_SET(qpc, qpc, rwe, 1);
    }

    if ((attr_mask & DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ) &&
        verbs_qp_attr.allow_remote_read == 1) {
        DEVX_SET(qpc, qpc, rre, 1);
    }
    if (verbs_qp_attr.allow_remote_atomic > DOCA_VERBS_QP_ATOMIC_MODE_NONE) {
        DEVX_SET(qpc, qpc, rae, 1);
        DEVX_SET(qpc, qpc, atomic_mode, verbs_qp_attr.allow_remote_atomic);
    }

    if (attr_mask & DOCA_VERBS_QP_ATTR_AH_ATTR) {
        DEVX_SET(qpc, qpc, primary_address_path.src_addr_index, verbs_qp_attr.ah_attr->sgid_index);

        if (m_verbs_device_attr->m_is_rts2rts_qp_dscp_supported &&
            m_verbs_device_attr->m_port_type == MLX5_CAP_PORT_TYPE_ETH) {
            uint8_t roce_version{};
            auto status =
                query_roce_version(m_ibv_ctx, verbs_qp_attr.ah_attr->sgid_index, roce_version);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to query roce version");
                return status;
            }

            if (roce_version >= MLX5_ROCE_ADDR_LAYOUT_ROCE_VERSION_VERSION_2_0)
                DEVX_SET(qpc, qpc, primary_address_path.dscp,
                         verbs_qp_attr.ah_attr->traffic_class >> 2);
        }
    }

    int mlx5_opt_param_mask{0};
    convert_doca_verbs_qp_attr_mask_to_legal_mlx5_qp_opt_param_mask(attr_mask, mlx5_opt_param_mask,
                                                                    DOCA_VERBS_QP_RTS2RTS);

    DEVX_SET(rts2rts_qp_in, in, opt_param_mask, mlx5_opt_param_mask);

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP rts2rts");
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_RTS;

    DOCA_LOG(LOG_INFO, "IB Verbs QP %p: has been successfully moved to RTS state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::qp2err(struct doca_verbs_qp_attr &verbs_qp_attr,
                                   int attr_mask) noexcept {
    qp_2err_in in{0};
    qp_2err_out out{0};

    if (!is_X2err_attrs_valid(attr_mask)) {
        DOCA_LOG(LOG_ERR, "X2err attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    DEVX_SET(qp_2err_in, in, opcode, MLX5_CMD_OP_QP_2ERR);
    DEVX_SET(qp_2err_in, in, qpn, m_qp_num);

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP 2err");
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_ERR;

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully moved to Error state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::qp2rst(struct doca_verbs_qp_attr &verbs_qp_attr,
                                   int attr_mask) noexcept {
    qp_2rst_in in{0};
    qp_2rst_out out{0};

    if (!is_X2rst_attrs_valid(attr_mask)) {
        DOCA_LOG(LOG_ERR, "X2rst attrs are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    DEVX_SET(qp_2rst_in, in, opcode, MLX5_CMD_OP_QP_2RST);
    DEVX_SET(qp_2rst_in, in, qpn, m_qp_num);

    auto ret =
        doca_verbs_wrapper_mlx5dv_devx_obj_modify(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to modify QP 2rst");
        return ret;
    }

    m_current_state = DOCA_VERBS_QP_STATE_RST;

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully moved to Reset state", this);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp::query_qp(struct doca_verbs_qp_attr &verbs_qp_attr,
                                     struct doca_verbs_qp_init_attr &verbs_qp_init_attr) noexcept {
    query_qp_in in{0};
    query_qp_out out{0};

    DEVX_SET(query_qp_in, in, opcode, MLX5_CMD_OP_QUERY_QP);
    DEVX_SET(query_qp_in, in, qpn, m_qp_num);

    auto ret = doca_verbs_wrapper_mlx5dv_devx_obj_query(m_qp_obj, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query QP");
        return DOCA_ERROR_DRIVER;
    }

    /* Set verbs_qp_attr with the QP information */
    const void *qpc = MLX5_ADDR_OF(query_qp_out, out, qpc);
    auto prm_qp_state = DEVX_GET(qpc, qpc, state);

    auto status =
        convert_prm_qp_state_to_doca_verbs_qp_state(prm_qp_state, verbs_qp_attr.current_state);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to get state, invalid qp state");
        return DOCA_ERROR_UNEXPECTED;
    }

    verbs_qp_attr.next_state = verbs_qp_attr.current_state;

    auto prm_mtu_size = DEVX_GET(qpc, qpc, mtu);
    status = convert_prm_mtu_size_to_doca_verbs_mtu_size(prm_mtu_size, verbs_qp_attr.path_mtu);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to get state, invalid MTU size");
        return DOCA_ERROR_UNEXPECTED;
    }

    verbs_qp_attr.rq_psn = DEVX_GET(qpc, qpc, next_rcv_psn);
    verbs_qp_attr.sq_psn = DEVX_GET(qpc, qpc, next_send_psn);
    verbs_qp_attr.dest_qp_num = DEVX_GET(qpc, qpc, remote_qpn);
    verbs_qp_attr.pkey_index = DEVX_GET(qpc, qpc, primary_address_path.pkey_index);
    verbs_qp_attr.port_num = DEVX_GET(qpc, qpc, primary_address_path.vhca_port_num);
    verbs_qp_attr.ack_timeout = DEVX_GET(qpc, qpc, primary_address_path.ack_timeout);
    verbs_qp_attr.retry_cnt = DEVX_GET(qpc, qpc, retry_count);
    verbs_qp_attr.rnr_retry = DEVX_GET(qpc, qpc, rnr_retry);
    verbs_qp_attr.min_rnr_timer = DEVX_GET(qpc, qpc, min_rnr_nak);
    verbs_qp_attr.allow_remote_write = DEVX_GET(qpc, qpc, rwe);
    verbs_qp_attr.allow_remote_read = DEVX_GET(qpc, qpc, rre);
    // verbs_qp_attr.allow_remote_atomic = DEVX_GET(qpc, qpc, rae);

    if (verbs_qp_attr.ah_attr != nullptr) {
        verbs_qp_attr.ah_attr->addr_type = m_addr_type;
        verbs_qp_attr.ah_attr->dlid = DEVX_GET(qpc, qpc, primary_address_path.rlid);
        verbs_qp_attr.ah_attr->sl = DEVX_GET(qpc, qpc, primary_address_path.sl);
        verbs_qp_attr.ah_attr->sgid_index = DEVX_GET(qpc, qpc, primary_address_path.src_addr_index);
        verbs_qp_attr.ah_attr->static_rate = DEVX_GET(qpc, qpc, primary_address_path.stat_rate);
        verbs_qp_attr.ah_attr->hop_limit = DEVX_GET(qpc, qpc, primary_address_path.hop_limit);
        verbs_qp_attr.ah_attr->traffic_class = DEVX_GET(qpc, qpc, primary_address_path.tclass);

        memcpy(verbs_qp_attr.ah_attr->gid.raw,
               MLX5_ADDR_OF(qpc, qpc, primary_address_path.rgid_rip),
               sizeof(struct doca_verbs_gid));
    }

    /* Set verbs_qp_init_attr with the QP information */
    verbs_qp_init_attr.send_cq = m_init_attr.send_cq;
    verbs_qp_init_attr.receive_cq = m_init_attr.receive_cq;
    verbs_qp_init_attr.sq_sig_all = m_init_attr.sq_sig_all;
    verbs_qp_init_attr.qp_context = m_init_attr.qp_context;
    verbs_qp_init_attr.pd = m_pd;
    verbs_qp_init_attr.sq_wr = m_sq_size_wr;
    verbs_qp_init_attr.rq_wr = m_rq_size;
    verbs_qp_init_attr.receive_max_sges = m_rcv_max_sges;
    verbs_qp_init_attr.user_index = DEVX_GET(qpc, qpc, user_index);
    verbs_qp_init_attr.qp_type = m_qp_type;
    verbs_qp_init_attr.send_max_sges = m_send_max_sges;
    verbs_qp_init_attr.max_inline_data = m_init_attr.max_inline_data;
    verbs_qp_init_attr.external_umem = m_init_attr.external_umem;
    verbs_qp_init_attr.external_umem_offset = m_init_attr.external_umem_offset;
    verbs_qp_init_attr.external_uar = m_init_attr.external_uar;

    return DOCA_SUCCESS;
}

void doca_verbs_qp::create(struct ibv_context *ibv_ctx) {
    auto status{DOCA_SUCCESS};
    m_ibv_ctx = ibv_ctx;
    m_pd = m_init_attr.pd;

    if ((m_init_attr.external_umem != nullptr && m_init_attr.external_umem_dbr == nullptr) ||
        (m_init_attr.external_umem == nullptr && m_init_attr.external_umem_dbr != nullptr)) {
        DOCA_LOG(LOG_ERR, "Both UMEM should be either external or internal");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    /* Query device attr */
    status = doca_verbs_query_device(ibv_ctx, &m_verbs_device_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query device attr");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    if (m_init_attr.qp_type != DOCA_VERBS_QP_TYPE_RC) {
        DOCA_LOG(LOG_ERR, "QP type is not valid");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    uint32_t log_rq_size{0};
    uint32_t log_stride{0};
    uint32_t log_sq_size_wqebb{0};

    /* Calculate Work Queue sizes */
    if (m_init_attr.rq_wr > 0 && m_init_attr.srq == nullptr) {
        if (m_init_attr.rq_wr > m_verbs_device_attr->m_max_qp_wr) {
            DOCA_LOG(LOG_ERR, "Failed to create IB Verbs QP: rq_wr is too big");
            throw DOCA_ERROR_INVALID_VALUE;
        }
        if (m_init_attr.receive_max_sges == 0) {
            DOCA_LOG(
                LOG_ERR,
                "Failed to create IB Verbs QP: rq_wr is greater than 0 but receive_max_sges is 0");
            throw DOCA_ERROR_INVALID_VALUE;
        }
        m_rcv_max_sges = doca_internal_utils_next_power_of_two(m_init_attr.receive_max_sges);
        /* Calculate receive_wqe size */
        m_rcv_wqe_size = m_rcv_max_sges * DOCA_VERBS_DATA_SEG_SIZE_IN_BYTES;
        if (m_rcv_wqe_size > m_verbs_device_attr->m_max_rq_desc_size) {
            DOCA_LOG(LOG_ERR, "Failed to create IB Verbs QP: rcv_max_sges is too big");
            throw DOCA_ERROR_INVALID_VALUE;
        }
        m_log_rcv_wqe_size = static_cast<uint8_t>(doca_internal_utils_log2(m_rcv_wqe_size));
        log_stride = m_log_rcv_wqe_size - sc_verbs_qp_log_rq_stride_shift;

        /* Calculate RQ size in bytes */
        auto rq_size_bytes = static_cast<uint32_t>(
            doca_internal_utils_next_power_of_two(m_init_attr.rq_wr * m_rcv_wqe_size));
        /* Minimum size of RQ is 64 bytes */
        rq_size_bytes = MAX(rq_size_bytes, DOCA_VERBS_WQEBB_SIZE);
        /* Calculate RQ size in receive_wqe units */
        m_rq_size = rq_size_bytes / m_rcv_wqe_size;
        log_rq_size = doca_internal_utils_log2(m_rq_size);
    }

    if (m_init_attr.sq_wr > 0) {
        // This check is done in rdma-core
        if (m_init_attr.sq_wr > (0x7fffffff / m_verbs_device_attr->m_max_sq_desc_size)) {
            DOCA_LOG(LOG_ERR, "Failed to create IB Verbs QP: sq_wr is too big");
            throw DOCA_ERROR_INVALID_VALUE;
        }
        if (m_init_attr.send_max_sges == 0) {
            DOCA_LOG(
                LOG_ERR,
                "Failed to create IB Verbs QP: sq_wr is greater than 0 but send_max_sges is 0");
            throw DOCA_ERROR_INVALID_VALUE;
        }
        m_send_max_sges = m_init_attr.send_max_sges;
        /* Calculate Send WQE size, which is the size of one control segment, size of one rdma
         * segment and a single data segment multiplied by the maximum number of send SGEs */
        uint32_t send_wqe_size =
            sizeof(struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg) +
            sizeof(struct doca_gpunetio_ib_mlx5_wqe_raddr_seg) +
            (m_init_attr.send_max_sges * sizeof(struct doca_gpunetio_ib_mlx5_wqe_data_seg));
        if (send_wqe_size > m_verbs_device_attr->m_max_sq_desc_size) {
            DOCA_LOG(LOG_ERR, "Failed to create IB Verbs QP: send_max_sges is too big");
            throw DOCA_ERROR_INVALID_VALUE;
        }

        uint32_t send_wqe_inline_size{};
        if (m_init_attr.max_inline_data > 0) {
            /* Calculate inline data segment size, which is composed of:
             * - size of mlx5_wqe_inl_data_seg (4 Bytes for byte_count and is_inline
             * attributes)
             * - max_inline_data size */
            uint32_t inline_data_seg_size =
                sizeof(struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg) + m_init_attr.max_inline_data;
            /* Align the size to OCTOWORD_SIZE (16 bytes) */
            inline_data_seg_size =
                doca_internal_utils_align_up_uint64(inline_data_seg_size, DOCA_VERBS_OCTOWORD_SIZE);
            /* Calculate Send WQE with inline data size, which is the size of one control
             * segment, size of one rdma segment and the total inline data segment size */
            send_wqe_inline_size = sizeof(struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg) +
                                   sizeof(struct doca_gpunetio_ib_mlx5_wqe_raddr_seg) +
                                   inline_data_seg_size;
            if (send_wqe_inline_size > m_verbs_device_attr->m_max_sq_desc_size) {
                DOCA_LOG(LOG_ERR, "Failed to create IB Verbs QP: max_inline_data is too big");
                throw DOCA_ERROR_INVALID_VALUE;
            }
        }

        /* Set m_send_wqe_size to the maximum value between the sizes of send_wqe_size and
         * send_wqe_inline_size
         */
        m_send_wqe_size = MAX(send_wqe_size, send_wqe_inline_size);

        /* Align size of send_wqe_size to WQEBB size */
        m_send_wqe_size =
            doca_internal_utils_align_up_uint32(m_send_wqe_size, DOCA_VERBS_WQEBB_SIZE);
        /* Calculate sq_size in bytes */
        auto sq_size_bytes = static_cast<uint32_t>(
            doca_internal_utils_next_power_of_two(m_send_wqe_size * m_init_attr.sq_wr));
        /* Calculate sq_size in wqebb units */
        m_sq_size_wqebb = sq_size_bytes / DOCA_VERBS_WQEBB_SIZE;
        if (m_sq_size_wqebb > m_verbs_device_attr->m_max_send_wqebb) {
            DOCA_LOG(LOG_ERR, "Failed to create IB Verbs QP: sq_wr is too big");
            throw DOCA_ERROR_INVALID_VALUE;
        }
        log_sq_size_wqebb = doca_internal_utils_log2(m_sq_size_wqebb);
        /* Calculate sq_size in Work Request units */
        m_sq_size_wr = sq_size_bytes / m_send_wqe_size;

        /* Due to alignments we may have more space for inline data */
        if (m_init_attr.max_inline_data > 0) {
            m_init_attr.max_inline_data =
                m_send_wqe_size - (sizeof(struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg) +
                                   sizeof(struct doca_gpunetio_ib_mlx5_wqe_raddr_seg) +
                                   sizeof(struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg));
            m_max_inline_data_length = m_init_attr.max_inline_data;
        }
    }

    uint32_t uar_id{};
    if (m_init_attr.external_uar == nullptr) {
        /* Case of internal UAR */
        auto uar_status = doca_verbs_wrapper_mlx5dv_devx_alloc_uar(
            m_ibv_ctx, MLX5DV_UAR_ALLOC_TYPE_BF, &m_uar_obj);
        if (uar_status != DOCA_SUCCESS) {
            uar_status = doca_verbs_wrapper_mlx5dv_devx_alloc_uar(
                m_ibv_ctx, MLX5DV_UAR_ALLOC_TYPE_NC, &m_uar_obj);
            if (uar_status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to create UAR");
                throw DOCA_ERROR_DRIVER;
            }
        }

        m_uar_db_reg = reinterpret_cast<uint64_t *>(m_uar_obj->reg_addr);
        uar_id = m_uar_obj->page_id;
    } else {
        /* Case of external UAR */
        status = doca_verbs_uar_id_get(m_init_attr.external_uar, &uar_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external UAR ID");
            throw status;
        }

        void *reg_addr{};
        status = doca_verbs_uar_reg_addr_get(m_init_attr.external_uar, &reg_addr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external UAR reg_addr");
            throw status;
        }
        m_uar_db_reg = reinterpret_cast<uint64_t *>(reg_addr);
    }

    uint32_t dbr_umem_id{0};
    uint64_t dbr_umem_offset{0};
    uint32_t wq_umem_id{0};

    if (m_init_attr.external_umem == nullptr) {
        auto db_umem_offset =
            (m_rq_size * m_rcv_wqe_size) + (m_sq_size_wqebb * DOCA_VERBS_WQEBB_SIZE);
        /* Align the Work Queue size to cacheline size for better performance */
        db_umem_offset =
            doca_internal_utils_align_up_uint32(db_umem_offset, DOCA_VERBS_CACHELINE_SIZE);

        /* Case of internal umem */
        auto total_umem_size = doca_internal_utils_align_up_uint32(
            db_umem_offset + sc_verbs_qp_doorbell_size, DOCA_VERBS_PAGE_SIZE);

        m_umem_buf = (uint8_t *)memalign(DOCA_VERBS_PAGE_SIZE, total_umem_size);

        memset(m_umem_buf, 0, total_umem_size);

        m_wq_buf = m_umem_buf;
        m_rq_buf = m_wq_buf;
        m_sq_buf = m_wq_buf + ((uintptr_t)m_rq_size << m_log_rcv_wqe_size);

        auto umem_status = doca_verbs_wrapper_mlx5dv_devx_umem_reg(m_ibv_ctx, m_wq_buf,
                                                                   total_umem_size, 0, &m_umem_obj);
        if (umem_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create QP UMEM");
            throw DOCA_ERROR_DRIVER;
        }

        wq_umem_id = m_umem_obj->umem_id;
        dbr_umem_offset = db_umem_offset;
        dbr_umem_id = wq_umem_id;

        m_db_buffer = reinterpret_cast<uint32_t *>(m_wq_buf + db_umem_offset);
    } else {
        uint8_t *tmp_db_buffer;

        /* Case of external umem for wq and dbr */
        status = doca_verbs_umem_get_address(m_init_attr.external_umem,
                                             reinterpret_cast<void **>(&m_wq_buf));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem address");
            throw status;
        }

        m_wq_buf += m_init_attr.external_umem_offset;
        m_rq_buf = m_wq_buf;
        m_sq_buf = m_wq_buf + ((uintptr_t)m_rq_size << m_log_rcv_wqe_size);

        status = doca_verbs_umem_get_id(m_init_attr.external_umem, &wq_umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem id");
            throw status;
        }

        /* Case of external umem */
        status = doca_verbs_umem_get_address(m_init_attr.external_umem_dbr,
                                             reinterpret_cast<void **>(&tmp_db_buffer));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem address");
            throw status;
        }

        status = doca_verbs_umem_get_id(m_init_attr.external_umem_dbr, &dbr_umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem id");
            throw status;
        }

        dbr_umem_offset = m_init_attr.external_umem_dbr_offset;
        m_db_buffer = reinterpret_cast<uint32_t *>(tmp_db_buffer + dbr_umem_offset);
    }

    /* Create QP object */
    status = create_qp_obj(uar_id, log_rq_size, log_sq_size_wqebb, log_stride, dbr_umem_offset,
                           dbr_umem_id, wq_umem_id, m_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create QP object");
        throw DOCA_ERROR_DRIVER;
    }

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs QP %p: has been successfully created", this);
}

doca_error_t doca_verbs_qp::destroy() noexcept {
    doca_error_t ret = DOCA_SUCCESS;

    if (m_verbs_device_attr) {
        auto status = doca_verbs_device_attr_free(m_verbs_device_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to free device attr");
            return DOCA_ERROR_INVALID_VALUE;
        }
        m_verbs_device_attr = nullptr;
    }

    if (m_qp_obj) {
        ret = doca_verbs_wrapper_mlx5dv_devx_obj_destroy(m_qp_obj);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy QP object");
            return DOCA_ERROR_DRIVER;
        }
        m_qp_obj = nullptr;
    }

    if (m_uar_obj) {
        doca_verbs_wrapper_mlx5dv_devx_free_uar(m_uar_obj);
        m_uar_obj = nullptr;
    }

    if (m_umem_obj) {
        ret = doca_verbs_wrapper_mlx5dv_devx_umem_dereg(m_umem_obj);
        if (ret != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy UMEM object");
            return DOCA_ERROR_DRIVER;
        }
        m_umem_obj = nullptr;
    }

    if (m_umem_buf) {
        free(m_umem_buf);
        m_umem_buf = nullptr;
    }

    return DOCA_SUCCESS;
}

doca_verbs_qp::doca_verbs_qp(struct ibv_context *ibv_ctx,
                             struct doca_verbs_qp_init_attr &verbs_qp_init_attr)
    : m_ibv_ctx(ibv_ctx), m_init_attr(verbs_qp_init_attr) {
    try {
        create(ibv_ctx);
    } catch (...) {
        (void)destroy();
        DOCA_LOG(LOG_ERR, "Failed to create QP");
        throw;
    }
}

doca_verbs_qp::~doca_verbs_qp() { static_cast<void>(destroy()); }

uint32_t doca_verbs_qp::get_qpn() const noexcept { return m_qp_num; }

void *doca_verbs_qp::get_dbr_addr() const noexcept { return (void *)m_db_buffer; }

void *doca_verbs_qp::get_uar_addr() const noexcept { return (void *)m_uar_db_reg; }

enum doca_verbs_uar_allocation_type doca_verbs_qp::get_uar_mtype() const noexcept {
    return m_init_attr.external_uar->get_uar_mtype();
}

void *doca_verbs_qp::get_sq_buf() const noexcept { return m_sq_buf; }

void *doca_verbs_qp::get_rq_buf() const noexcept { return (void *)m_wq_buf; }

uint32_t doca_verbs_qp::get_sq_size_wqebb() const noexcept { return m_sq_size_wqebb; }

uint32_t doca_verbs_qp::get_rq_size() const noexcept { return m_rq_size; }

uint32_t doca_verbs_qp::get_rcv_wqe_size() const noexcept { return m_rcv_wqe_size; }

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_qp_init_attr_create(struct doca_verbs_qp_init_attr **verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create qp_init_attr: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *verbs_qp_init_attr =
        (struct doca_verbs_qp_init_attr *)calloc(1, sizeof(struct doca_verbs_qp_init_attr));
    if (*verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create qp_init_attr: failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_destroy(struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy qp_init_attr: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    free(verbs_qp_init_attr);
    verbs_qp_init_attr = nullptr;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_set_pd(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                            struct ibv_pd *pd) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set pd: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (pd == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set pd: parameter pd is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->pd = pd;

    return DOCA_SUCCESS;
}

struct ibv_pd *doca_verbs_qp_init_attr_get_pd(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get pd: parameter verbs_qp_init_attr is NULL");
        return nullptr;
    }

    return verbs_qp_init_attr->pd;
}

doca_error_t doca_verbs_qp_init_attr_set_send_cq(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                                 struct doca_verbs_cq *send_cq) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set send_cq: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (send_cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set send_cq: parameter send_cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->send_cq = send_cq;

    return DOCA_SUCCESS;
}

struct doca_verbs_cq *doca_verbs_qp_init_attr_get_send_cq(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get send_cq: parameter verbs_qp_init_attr is NULL");
        return nullptr;
    }

    return verbs_qp_init_attr->send_cq;
}

doca_error_t doca_verbs_qp_init_attr_set_receive_cq(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_cq *receive_cq) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_cq: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (receive_cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_cq: parameter receive_cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->receive_cq = receive_cq;

    return DOCA_SUCCESS;
}

struct doca_verbs_cq *doca_verbs_qp_init_attr_get_receive_cq(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get receive_cq: parameter verbs_qp_init_attr is NULL");
        return nullptr;
    }

    return verbs_qp_init_attr->receive_cq;
}

doca_error_t doca_verbs_qp_init_attr_set_sq_sig_all(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, int sq_sig_all) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set sq_sig_all: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->sq_sig_all = sq_sig_all;

    return DOCA_SUCCESS;
}

int doca_verbs_qp_init_attr_get_sq_sig_all(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get sq_sig_all: parameter verbs_qp_init_attr is NULL");
        return -1;
    }

    return verbs_qp_init_attr->sq_sig_all;
}

doca_error_t doca_verbs_qp_init_attr_set_sq_wr(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                               uint32_t sq_wr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set sq_wr: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->sq_wr = sq_wr;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_sq_wr(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get sq_wr: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->sq_wr;
}

doca_error_t doca_verbs_qp_init_attr_set_rq_wr(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                               uint32_t rq_wr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_cq: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->rq_wr = rq_wr;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_rq_wr(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get rq_wr: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->rq_wr;
}

doca_error_t doca_verbs_qp_init_attr_set_send_max_sges(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t send_max_sges) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set send_max_sges: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->send_max_sges = send_max_sges;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_send_max_sges(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get send_max_sges: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->send_max_sges;
}

doca_error_t doca_verbs_qp_init_attr_set_receive_max_sges(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t receive_max_sges) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->receive_max_sges = receive_max_sges;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_receive_max_sges(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get receive_max_sges: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->receive_max_sges;
}

doca_error_t doca_verbs_qp_init_attr_set_max_inline_data(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t max_inline_data) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set max_inline_data: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->max_inline_data = max_inline_data;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_max_inline_data(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get max_inline_data: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->max_inline_data;
}

doca_error_t doca_verbs_qp_init_attr_set_user_index(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t user_index) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set user_index: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if ((user_index & USER_INDEX_MSB_8BITS_MASK) != 0) {
        DOCA_LOG(LOG_ERR, "Failed to set user_index: input parameter user_index=%u exceeds 24 bits",
                 user_index);
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->user_index = user_index;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_user_index(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get user_index: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->user_index;
}

doca_error_t doca_verbs_qp_init_attr_set_qp_type(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                                 uint32_t qp_type) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set qp_type: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->qp_type = qp_type;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_init_attr_get_qp_type(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get qp_type: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->qp_type;
}

doca_error_t doca_verbs_qp_init_attr_set_external_umem(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_umem *external_umem,
    uint64_t external_umem_offset) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter external_umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->external_umem = external_umem;
    verbs_qp_init_attr->external_umem_offset = external_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_set_external_dbr_umem(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_umem *external_umem,
    uint64_t external_umem_offset) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter external_umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->external_umem_dbr = external_umem;
    verbs_qp_init_attr->external_umem_dbr_offset = external_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_get_external_umem(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
    struct doca_verbs_umem **external_umem, uint64_t *external_umem_offset) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get external_umem: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get external_umem: parameter external_umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem_offset == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get external_umem: parameter external_umem_offset is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *external_umem = verbs_qp_init_attr->external_umem;
    *external_umem_offset = verbs_qp_init_attr->external_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_set_external_uar(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_uar *external_uar) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_uar == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter external_uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->external_uar = external_uar;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_get_external_uar(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
    struct doca_verbs_uar **external_uar) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get external_uar: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_uar == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get external_uar: parameter external_uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *external_uar = verbs_qp_init_attr->external_uar;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_init_attr_set_qp_context(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, void *qp_context) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set qp_context: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (qp_context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set qp_context: parameter qp_context is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->qp_context = qp_context;

    return DOCA_SUCCESS;
}

void *doca_verbs_qp_init_attr_get_qp_context(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get qp_context: parameter verbs_qp_init_attr is NULL");
        return nullptr;
    }

    return verbs_qp_init_attr->qp_context;
}

doca_error_t doca_verbs_qp_init_attr_set_core_direct_master(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint8_t core_direct_master) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set core_direct_master: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (core_direct_master != 0x0 && core_direct_master != 0x1) {
        DOCA_LOG(LOG_ERR, "Failed to set core_direct_master: invalid input value %d",
                 core_direct_master);
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->core_direct_master = core_direct_master;

    return DOCA_SUCCESS;
}

uint8_t doca_verbs_qp_init_attr_get_core_direct_master(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get core_direct_master: parameter verbs_qp_init_attr is NULL");
        return 0;
    }

    return verbs_qp_init_attr->core_direct_master;
}

doca_error_t doca_verbs_qp_attr_create(struct doca_verbs_qp_attr **verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create qp_attr: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *verbs_qp_attr = (struct doca_verbs_qp_attr *)calloc(1, sizeof(struct doca_verbs_qp_attr));
    if (*verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create qp_attr: failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_attr_destroy(struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy qp_attr: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    free(verbs_qp_attr);
    verbs_qp_attr = nullptr;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_attr_set_next_state(struct doca_verbs_qp_attr *verbs_qp_attr,
                                               enum doca_verbs_qp_state next_state) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set next_state: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->next_state = next_state;

    return DOCA_SUCCESS;
}

enum doca_verbs_qp_state doca_verbs_qp_attr_get_next_state(
    const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get next_state: parameter verbs_qp_attr is NULL");
        return static_cast<enum doca_verbs_qp_state>(0);
    }

    return verbs_qp_attr->next_state;
}

doca_error_t doca_verbs_qp_attr_set_current_state(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                  enum doca_verbs_qp_state current_state) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set current_state: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->current_state = current_state;

    return DOCA_SUCCESS;
}

enum doca_verbs_qp_state doca_verbs_qp_attr_get_current_state(
    const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get current_state: parameter verbs_qp_attr is NULL");
        return static_cast<enum doca_verbs_qp_state>(0);
    }

    return verbs_qp_attr->current_state;
}

doca_error_t doca_verbs_qp_attr_set_path_mtu(struct doca_verbs_qp_attr *verbs_qp_attr,
                                             enum doca_verbs_mtu_size path_mtu) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set path_mtu: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->path_mtu = path_mtu;

    return DOCA_SUCCESS;
}

enum doca_verbs_mtu_size doca_verbs_qp_attr_get_path_mtu(
    const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get path_mtu: parameter verbs_qp_attr is NULL");
        return static_cast<enum doca_verbs_mtu_size>(0);
    }

    return verbs_qp_attr->path_mtu;
}

doca_error_t doca_verbs_qp_attr_set_rq_psn(struct doca_verbs_qp_attr *verbs_qp_attr,
                                           uint32_t rq_psn) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set rq_psn: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->rq_psn = rq_psn;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_attr_get_rq_psn(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get rq_psn: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->rq_psn;
}

doca_error_t doca_verbs_qp_attr_set_sq_psn(struct doca_verbs_qp_attr *verbs_qp_attr,
                                           uint32_t sq_psn) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set sq_psn: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->sq_psn = sq_psn;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_attr_get_sq_psn(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get sq_psn: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->sq_psn;
}

doca_error_t doca_verbs_qp_attr_set_dest_qp_num(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                uint32_t dest_qp_num) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set dest_qp_num: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->dest_qp_num = dest_qp_num;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_qp_attr_get_dest_qp_num(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get dest_qp_num: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->dest_qp_num;
}

doca_error_t doca_verbs_qp_attr_set_allow_remote_write(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                       int allow_remote_write) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set allow_remote_write: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->allow_remote_write = allow_remote_write;

    return DOCA_SUCCESS;
}

int doca_verbs_qp_attr_get_allow_remote_write(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get allow_remote_write: parameter verbs_qp_attr is NULL");
        return -1;
    }

    return verbs_qp_attr->allow_remote_write;
}

doca_error_t doca_verbs_qp_attr_set_allow_remote_read(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                      int allow_remote_read) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set allow_remote_read: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->allow_remote_read = allow_remote_read;

    return DOCA_SUCCESS;
}

int doca_verbs_qp_attr_get_allow_remote_read(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get allow_remote_read: parameter verbs_qp_attr is NULL");
        return -1;
    }

    return verbs_qp_attr->allow_remote_read;
}

doca_error_t doca_verbs_qp_attr_set_allow_remote_atomic(
    struct doca_verbs_qp_attr *verbs_qp_attr, enum doca_verbs_qp_atomic_type atomic_type) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set allow_remote_atomic: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->allow_remote_atomic = atomic_type;

    return DOCA_SUCCESS;
}

enum doca_verbs_qp_atomic_type doca_verbs_qp_attr_get_allow_remote_atomic(
    const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get allow_remote_atomic: parameter verbs_qp_attr is NULL");
        return DOCA_VERBS_QP_ATOMIC_MODE_NONE;
    }

    return verbs_qp_attr->allow_remote_atomic;
}

doca_error_t doca_verbs_qp_attr_set_ah_attr(struct doca_verbs_qp_attr *verbs_qp_attr,
                                            doca_verbs_ah_attr *ah_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set ah_attr: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (ah_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set ah_attr: parameter ah_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->ah_attr = ah_attr;

    return DOCA_SUCCESS;
}

struct doca_verbs_ah_attr *doca_verbs_qp_attr_get_ah_attr(
    const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get ah_attr: parameter verbs_qp_attr is NULL");
        return nullptr;
    }
    if (verbs_qp_attr->ah_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get ah_attr: ah_attr object was not set previously");
        return nullptr;
    }

    return verbs_qp_attr->ah_attr;
}

doca_error_t doca_verbs_qp_attr_set_pkey_index(struct doca_verbs_qp_attr *verbs_qp_attr,
                                               uint16_t pkey_index) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set pkey_index: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->pkey_index = pkey_index;

    return DOCA_SUCCESS;
}

uint16_t doca_verbs_qp_attr_get_pkey_index(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get pkey_index: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->pkey_index;
}

doca_error_t doca_verbs_qp_attr_set_port_num(struct doca_verbs_qp_attr *verbs_qp_attr,
                                             uint16_t port_num) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set port_num: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->port_num = port_num;

    return DOCA_SUCCESS;
}

uint16_t doca_verbs_qp_attr_get_port_num(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get port_num: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->port_num;
}

doca_error_t doca_verbs_qp_attr_set_ack_timeout(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                uint16_t ack_timeout) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set ack_timeout: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->ack_timeout = ack_timeout;

    return DOCA_SUCCESS;
}

uint16_t doca_verbs_qp_attr_get_ack_timeout(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get ack_timeout: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->ack_timeout;
}

doca_error_t doca_verbs_qp_attr_set_retry_cnt(struct doca_verbs_qp_attr *verbs_qp_attr,
                                              uint16_t retry_cnt) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set retry_cnt: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->retry_cnt = retry_cnt;

    return DOCA_SUCCESS;
}

uint16_t doca_verbs_qp_attr_get_retry_cnt(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get retry_cnt: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->retry_cnt;
}

doca_error_t doca_verbs_qp_attr_set_rnr_retry(struct doca_verbs_qp_attr *verbs_qp_attr,
                                              uint16_t rnr_retry) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set rnr_retry: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->rnr_retry = rnr_retry;

    return DOCA_SUCCESS;
}

uint16_t doca_verbs_qp_attr_get_rnr_retry(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get rnr_retry: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->rnr_retry;
}

doca_error_t doca_verbs_qp_attr_set_min_rnr_timer(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                  uint16_t min_rnr_timer) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set min_rnr_timer: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_attr->min_rnr_timer = min_rnr_timer;

    return DOCA_SUCCESS;
}

uint16_t doca_verbs_qp_attr_get_min_rnr_timer(const struct doca_verbs_qp_attr *verbs_qp_attr) {
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get min_rnr_timer: parameter verbs_qp_attr is NULL");
        return 0;
    }

    return verbs_qp_attr->min_rnr_timer;
}

doca_error_t doca_verbs_ah_attr_create(struct ibv_context *context,
                                       struct doca_verbs_ah_attr **verbs_ah) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_ah: parameter context is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_ah: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *verbs_ah = (struct doca_verbs_ah_attr *)calloc(1, sizeof(struct doca_verbs_ah_attr));
    if (*verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_ah: failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    (*verbs_ah)->is_global = 1;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_ah_attr_destroy(struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy verbs_ah: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    free(verbs_ah);
    verbs_ah = nullptr;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_ah_attr_set_gid(struct doca_verbs_ah_attr *verbs_ah,
                                        struct doca_verbs_gid gid) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set gid: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->gid = gid;

    return DOCA_SUCCESS;
}

struct doca_verbs_gid doca_verbs_ah_get_gid(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get gid: parameter verbs_ah is NULL");
        struct doca_verbs_gid zero_gid {};
        memset(&zero_gid, 0, sizeof(zero_gid));
        return zero_gid;
    }

    return verbs_ah->gid;
}

doca_error_t doca_verbs_ah_attr_set_addr_type(struct doca_verbs_ah_attr *verbs_ah,
                                              enum doca_verbs_addr_type addr_type) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set addr_type: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->addr_type = addr_type;

    return DOCA_SUCCESS;
}

enum doca_verbs_addr_type doca_verbs_ah_get_addr_type(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get addr_type: parameter verbs_ah is NULL");
        return static_cast<enum doca_verbs_addr_type>(0);
    }

    return verbs_ah->addr_type;
}

doca_error_t doca_verbs_ah_attr_set_dlid(struct doca_verbs_ah_attr *verbs_ah, uint32_t dlid) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set dlid: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->dlid = dlid;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_ah_get_dlid(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get dlid: parameter verbs_ah is NULL");
        return 0;
    }

    return verbs_ah->dlid;
}

doca_error_t doca_verbs_ah_attr_set_sl(struct doca_verbs_ah_attr *verbs_ah, uint8_t sl) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set sl: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->sl = sl;

    return DOCA_SUCCESS;
}

uint8_t doca_verbs_ah_get_sl(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get sl: parameter verbs_ah is NULL");
        return 0;
    }

    return verbs_ah->sl;
}

doca_error_t doca_verbs_ah_attr_set_sgid_index(struct doca_verbs_ah_attr *verbs_ah,
                                               uint8_t sgid_index) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set sgid_index: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->sgid_index = sgid_index;

    return DOCA_SUCCESS;
}

uint8_t doca_verbs_ah_get_sgid_index(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get sgid_index: parameter verbs_ah is NULL");
        return 0;
    }

    return verbs_ah->sgid_index;
}

doca_error_t doca_verbs_ah_attr_set_static_rate(struct doca_verbs_ah_attr *verbs_ah,
                                                uint8_t static_rate) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set static_rate: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->static_rate = static_rate;

    return DOCA_SUCCESS;
}

uint8_t doca_verbs_ah_get_static_rate(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get static_rate: parameter verbs_ah is NULL");
        return 0;
    }

    return verbs_ah->static_rate;
}

doca_error_t doca_verbs_ah_attr_set_hop_limit(struct doca_verbs_ah_attr *verbs_ah,
                                              uint8_t hop_limit) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set hop_limit: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->hop_limit = hop_limit;

    return DOCA_SUCCESS;
}

uint8_t doca_verbs_ah_get_hop_limit(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get hop_limit: parameter verbs_ah is NULL");
        return 0;
    }

    return verbs_ah->hop_limit;
}

doca_error_t doca_verbs_ah_attr_set_traffic_class(struct doca_verbs_ah_attr *verbs_ah,
                                                  uint8_t traffic_class) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set traffic_class: parameter verbs_ah is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_ah->traffic_class = traffic_class;

    return DOCA_SUCCESS;
}

uint8_t doca_verbs_ah_get_traffic_class(const struct doca_verbs_ah_attr *verbs_ah) {
    if (verbs_ah == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get traffic_class: parameter verbs_ah is NULL");
        return 0;
    }

    return verbs_ah->traffic_class;
}

doca_error_t doca_verbs_qp_create(struct ibv_context *context,
                                  struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                  struct doca_verbs_qp **verbs_qp) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_qp: parameter context is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_qp: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_qp == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_qp: parameter verbs_qp is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    try {
        *verbs_qp = new doca_verbs_qp(context, *verbs_qp_init_attr);
        DOCA_LOG(LOG_INFO, "IB Verbs Context %p: verbs_qp=%p was created", context, *verbs_qp);
        return DOCA_SUCCESS;
    } catch (doca_error_t err) {
        return err;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_destroy(struct doca_verbs_qp *verbs_qp) {
    if (verbs_qp == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to destroy verbs_qp: parameter verbs_qp is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    auto status = verbs_qp->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Failed to destroy verbs_qp.");
        return status;
    }

    delete (verbs_qp);
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_qp_modify(struct doca_verbs_qp *verbs_qp,
                                  struct doca_verbs_qp_attr *verbs_qp_attr, int attr_mask) {
    if (verbs_qp == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to modify verbs_qp: parameter verbs_qp is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to modify verbs_qp: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (!verbs_qp->is_qp_attr_valid(verbs_qp_attr, attr_mask)) {
        DOCA_LOG(LOG_ERR, "Failed to modify verbs_qp: some QP attributes values are invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    doca_verbs_qp_state current_state;
    doca_verbs_qp_state next_state;
    if (!(attr_mask & DOCA_VERBS_QP_ATTR_CURRENT_STATE))
        current_state = verbs_qp->get_current_state();
    else
        current_state = verbs_qp_attr->current_state;
    if (!(attr_mask & DOCA_VERBS_QP_ATTR_NEXT_STATE))
        next_state = current_state;
    else
        next_state = verbs_qp_attr->next_state;

    switch (next_state) {
        case DOCA_VERBS_QP_STATE_RST:
            return verbs_qp->qp2rst(*verbs_qp_attr, attr_mask);
        case DOCA_VERBS_QP_STATE_INIT:
            if (current_state == DOCA_VERBS_QP_STATE_RST)
                return verbs_qp->rst2init(*verbs_qp_attr, attr_mask);
            else if (current_state == DOCA_VERBS_QP_STATE_INIT)
                return verbs_qp->init2init(*verbs_qp_attr, attr_mask);
            else
                goto invalid_input;
        case DOCA_VERBS_QP_STATE_RTR:
            if (current_state == DOCA_VERBS_QP_STATE_INIT)
                return verbs_qp->init2rtr(*verbs_qp_attr, attr_mask);
            else
                goto invalid_input;
        case DOCA_VERBS_QP_STATE_RTS:
            if (current_state == DOCA_VERBS_QP_STATE_RTR)
                return verbs_qp->rtr2rts(*verbs_qp_attr, attr_mask);
            else if (current_state == DOCA_VERBS_QP_STATE_RTS)
                return verbs_qp->rts2rts(*verbs_qp_attr, attr_mask);
            else
                goto invalid_input;
        case DOCA_VERBS_QP_STATE_ERR:
            return verbs_qp->qp2err(*verbs_qp_attr, attr_mask);
        default:
            DOCA_LOG(LOG_ERR, "Failed to modify verbs_qp: invalid next_state");
            return DOCA_ERROR_INVALID_VALUE;
    }

invalid_input:
    DOCA_LOG(LOG_ERR,
             "Failed to modify verbs_qp: invalid combination of current_state and next_state");
    return DOCA_ERROR_INVALID_VALUE;
}

doca_error_t doca_verbs_qp_query(struct doca_verbs_qp *verbs_qp,
                                 struct doca_verbs_qp_attr *verbs_qp_attr,
                                 struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to query verbs_qp: parameter verbs_qp is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to query verbs_qp: parameter verbs_qp_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to query verbs_qp: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    return verbs_qp->query_qp(*verbs_qp_attr, *verbs_qp_init_attr);
}

uint32_t doca_verbs_qp_get_qpn(const struct doca_verbs_qp *verbs_qp) { return verbs_qp->get_qpn(); }

void *doca_verbs_qp_get_dbr_addr(const struct doca_verbs_qp *verbs_qp) {
    return verbs_qp->get_dbr_addr();
}

void *doca_verbs_qp_get_uar_addr(const struct doca_verbs_qp *verbs_qp) {
    return verbs_qp->get_uar_addr();
}

void doca_verbs_qp_get_wq(const struct doca_verbs_qp *verbs_qp, void **sq_buf,
                          uint32_t *sq_num_entries, void **rq_buf, uint32_t *rq_num_entries,
                          uint32_t *rwqe_size_bytes) {
    *sq_buf = verbs_qp->get_sq_buf();
    *rq_buf = verbs_qp->get_rq_buf();
    *sq_num_entries = verbs_qp->get_sq_size_wqebb();
    *rq_num_entries = verbs_qp->get_rq_size();
    *rwqe_size_bytes = verbs_qp->get_rcv_wqe_size();
}

doca_error_t doca_verbs_qp_init_attr_set_srq(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                             struct doca_verbs_srq *srq) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set srq: parameter verbs_qp_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (srq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set srq: parameter srq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_qp_init_attr->srq = srq;

    return DOCA_SUCCESS;
}

struct doca_verbs_srq *doca_verbs_qp_init_attr_get_srq(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr) {
    if (verbs_qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get srq: parameter verbs_qp_init_attr is NULL");
        return nullptr;
    }

    return verbs_qp_init_attr->srq;
}
