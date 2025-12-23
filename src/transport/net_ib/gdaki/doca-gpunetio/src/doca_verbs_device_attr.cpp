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

#include <unistd.h>
#include <stdlib.h>
#include <mutex>
#include <time.h>
#include <string.h>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_verbs_net_wrapper.h"
#include "host/doca_verbs.h"
#include "doca_verbs_device_attr.hpp"

#define PRIV_DOCA_MLX5_GID_TABLE_8_ENTRIES 0x0
#define PRIV_DOCA_MLX5_GID_TABLE_16_ENTRIES 0x1
#define PRIV_DOCA_MLX5_GID_TABLE_32_ENTRIES 0x2
#define PRIV_DOCA_MLX5_GID_TABLE_64_ENTRIES 0x3
#define PRIV_DOCA_MLX5_GID_TABLE_128_ENTRIES 0x4

#define PRIV_DOCA_MLX5_HCA_CAP_OPMOD_GET_MAX 0
#define PRIV_DOCA_MLX5_HCA_CAP_OPMOD_GET_CUR 1

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {

uint16_t translate_gid_table_size(uint16_t gid_table_size_prm) {
    switch (gid_table_size_prm) {
        case PRIV_DOCA_MLX5_GID_TABLE_8_ENTRIES:
            return 8;
        case PRIV_DOCA_MLX5_GID_TABLE_16_ENTRIES:
            return 16;
        case PRIV_DOCA_MLX5_GID_TABLE_32_ENTRIES:
            return 32;
        case PRIV_DOCA_MLX5_GID_TABLE_64_ENTRIES:
            return 64;
        case PRIV_DOCA_MLX5_GID_TABLE_128_ENTRIES:
            return 128;
        default:
            // Shouldn't reach this
            return 0;
    }

    // Shouldn't reach this
    return 0;
}

} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_device_attr Member Functions
 *********************************************************************************************************************/

doca_verbs_device_attr::doca_verbs_device_attr(struct ibv_context *ibv_ctx) {
    try {
        query_caps(ibv_ctx);
    } catch (...) {
        DOCA_LOG(LOG_ERR, "Failed to create device_attr");
        throw;
    }
}

void doca_verbs_device_attr::query_caps(struct ibv_context *ibv_ctx) {
    struct ibv_device_attr device_attr {};
    auto ret = doca_verbs_wrapper_ibv_query_device(ibv_ctx, &device_attr);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query device attr");
        throw ret;
    }

    m_max_qp = device_attr.max_qp;
    m_max_qp_wr = device_attr.max_qp_wr;
    m_max_sge = device_attr.max_sge;
    m_max_cq = device_attr.max_cq;
    m_max_cqe = device_attr.max_cqe;
    m_max_mr = device_attr.max_mr;
    m_max_pd = device_attr.max_pd;
    m_max_ah = device_attr.max_ah;
    m_max_srq = device_attr.max_srq;
    m_max_srq_wr = device_attr.max_srq_wr;
    m_max_srq_sge = device_attr.max_srq_sge;
    m_max_pkeys = device_attr.max_pkeys;
    m_phys_port_cnt = device_attr.phys_port_cnt;

    uint32_t in[MLX5_ST_SZ_DW(query_hca_cap_in)] = {0};
    uint32_t out[MLX5_ST_SZ_DW(query_hca_cap_out)] = {0};

    DEVX_SET(query_hca_cap_in, in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(query_hca_cap_in, in, op_mod,
             PRIV_DOCA_MLX5_HCA_CAP_OPMOD_GET_CUR | MLX5_SET_HCA_CAP_OP_MOD_GENERAL_DEVICE);

    ret = doca_verbs_wrapper_mlx5dv_devx_general_cmd(ibv_ctx, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query device capabilities");
        throw ret;
    }

    m_port_type = DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.port_type);
    if (m_port_type == MLX5_CAP_PORT_TYPE_IB)
        m_gid_table_size = translate_gid_table_size(
            DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.gid_table_size));
    m_is_qp_rc_supported = DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.rc);
    m_is_rts2rts_qp_dscp_supported =
        DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.rts2rts_qp_dscp);
    m_max_sq_desc_size = DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.max_wqe_sz_sq);
    m_max_rq_desc_size = DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.max_wqe_sz_rq);
    m_max_send_wqebb = 1 << DEVX_GET(query_hca_cap_out, out, capability.cmd_hca_cap.log_max_qp_sz);

    memset(in, 0, sizeof(in));
    memset(out, 0, sizeof(out));

    DEVX_SET(query_hca_cap_in, in, opcode, MLX5_CMD_OP_QUERY_HCA_CAP);
    DEVX_SET(query_hca_cap_in, in, op_mod,
             PRIV_DOCA_MLX5_HCA_CAP_OPMOD_GET_CUR | MLX5_SET_HCA_CAP_OP_MOD_ROCE);

    ret = doca_verbs_wrapper_mlx5dv_devx_general_cmd(ibv_ctx, in, sizeof(in), out, sizeof(out));
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query ROCE capabilities");
        throw ret;
    }

    if (m_port_type == MLX5_CAP_PORT_TYPE_ETH)
        m_gid_table_size =
            DEVX_GET(query_hca_cap_out, out, capability.roce_caps.roce_address_table_size);
    m_min_udp_sport =
        DEVX_GET(query_hca_cap_out, out, capability.roce_caps.r_roce_min_src_udp_port);
    m_max_udp_sport =
        DEVX_GET(query_hca_cap_out, out, capability.roce_caps.r_roce_max_src_udp_port);
}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_query_device(struct ibv_context *context,
                                     struct doca_verbs_device_attr **verbs_device_attr) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to query doca_verbs_device_attr. param context=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (verbs_device_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to query doca_verbs_device_attr. param verbs_device_attr=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    try {
        *verbs_device_attr = new doca_verbs_device_attr(context);
        return DOCA_SUCCESS;
    } catch (doca_error_t err) {
        return err;
    }
}

doca_error_t doca_verbs_device_attr_free(struct doca_verbs_device_attr *verbs_device_attr) {
    if (verbs_device_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to free doca_verbs_device_attr. param verbs_device_attr=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    delete (verbs_device_attr);
    return DOCA_SUCCESS;
}

uint32_t doca_verbs_device_attr_get_max_qp(const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_qp;
}

uint32_t doca_verbs_device_attr_get_max_qp_wr(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_qp_wr;
}

uint32_t doca_verbs_device_attr_get_max_sge(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_sge;
}

uint32_t doca_verbs_device_attr_get_max_cq(const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_cq;
}

uint32_t doca_verbs_device_attr_get_max_cqe(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_cqe;
}

uint32_t doca_verbs_device_attr_get_max_mr(const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_mr;
}

uint32_t doca_verbs_device_attr_get_max_pd(const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_pd;
}

uint32_t doca_verbs_device_attr_get_max_ah(const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_ah;
}

uint32_t doca_verbs_device_attr_get_max_srq(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_srq;
}

uint32_t doca_verbs_device_attr_get_max_srq_wr(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_srq_wr;
}

uint32_t doca_verbs_device_attr_get_max_srq_sge(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_srq_sge;
}

uint16_t doca_verbs_device_attr_get_max_pkeys(
    const struct doca_verbs_device_attr *verbs_device_attr) {
    return verbs_device_attr->m_max_pkeys;
}

doca_error_t doca_verbs_device_attr_get_is_qp_type_supported(
    const struct doca_verbs_device_attr *verbs_device_attr, uint32_t qp_type) {
    switch (qp_type) {
        case DOCA_VERBS_QP_TYPE_RC:
            return verbs_device_attr->m_is_qp_rc_supported ? DOCA_SUCCESS
                                                           : DOCA_ERROR_NOT_SUPPORTED;
            break;
        default:
            DOCA_LOG(LOG_ERR, "Failed to check if QP type is supported. param QP type is invalid");
            return DOCA_ERROR_INVALID_VALUE;
    }

    // Shouldn't reach this
    return DOCA_ERROR_UNEXPECTED;
}
