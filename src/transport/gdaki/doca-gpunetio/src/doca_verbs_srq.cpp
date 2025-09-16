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
#include "doca_verbs_net_wrapper.h"

#define DOCA_VERBS_SRQ_DB_SIZE 64
#define DOCA_VERBS_LOG_WQEBB_SIZE 6
#define DOCA_VERBS_MIN_SRQ_SIZE 32
#define DOCA_VERBS_DATA_SEG_SIZE_IN_BYTES sizeof(struct doca_internal_mlx5_wqe_data_seg)
#define DOCA_VERBS_CONTROL_SEG_SIZE_IN_BYTES sizeof(struct doca_internal_mlx5_wqe_mprq_next_seg)
#define MAX(a, b) std::max(a, b)

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {

using create_rmp_in = uint32_t[MLX5_ST_SZ_DW(create_rmp_in)];
using create_rmp_out = uint32_t[MLX5_ST_SZ_DW(create_rmp_out)];

} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_srq Member Functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_srq::create_srq_obj(
    uint32_t log_srq_size, uint32_t log_stride, uint64_t dbr_umem_offset, uint32_t dbr_umem_id,
    uint64_t wq_umem_offset, uint32_t wq_umem_id,
    struct doca_verbs_srq_init_attr &verbs_srq_init_attr) noexcept {
    create_rmp_in create_in{0};
    create_rmp_out create_out{0};

    DEVX_SET(create_rmp_in, create_in, opcode, MLX5_CMD_OP_CREATE_RMP);

    void *rmp_context = MLX5_ADDR_OF(create_rmp_in, create_in, ctx);
    DEVX_SET(rmpc, rmp_context, state, MLX5_SQC_STATE_RDY);

    void *wq_context = MLX5_ADDR_OF(rmpc, rmp_context, wq);

    struct mlx5dv_pd dvpd;
    struct mlx5dv_obj dv_obj;
    // Query pdn
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.pd.in = m_pd;
    dv_obj.pd.out = &dvpd;

    auto ret = doca_verbs_wrapper_mlx5dv_init_obj(&dv_obj, MLX5DV_OBJ_PD);
    if (ret) {
        DOCA_LOG(LOG_ERR, "Error in mlx5dv PD initialization");
        return DOCA_ERROR_DRIVER;
    }

    DEVX_SET(wq, wq_context, pd, dvpd.pdn);

    if (verbs_srq_init_attr.srq_type == DOCA_VERBS_SRQ_TYPE_LINKED_LIST) {
        DEVX_SET(wq, wq_context, wq_type, 0);
        DEVX_SET(rmpc, rmp_context, basic_cyclic_rcv_wqe, 0);
        m_srq_type = DOCA_VERBS_SRQ_TYPE_LINKED_LIST;
    } else {
        DEVX_SET(wq, wq_context, wq_type, 1);
        DEVX_SET(rmpc, rmp_context, basic_cyclic_rcv_wqe, 1);
        m_srq_type = DOCA_VERBS_SRQ_TYPE_CONTIGUOUS;
    }

    DEVX_SET(wq, wq_context, log_wq_sz, log_srq_size);
    DEVX_SET(wq, wq_context, log_wq_stride, log_stride);
    DEVX_SET(wq, wq_context, end_padding_mode, 1);

    DEVX_SET(wq, wq_context, wq_umem_id, wq_umem_id);
    DEVX_SET64(wq, wq_context, wq_umem_offset, wq_umem_offset);
    DEVX_SET(wq, wq_context, wq_umem_valid, 1);

    DEVX_SET(wq, wq_context, dbr_umem_id, dbr_umem_id);
    DEVX_SET64(wq, wq_context, dbr_addr, dbr_umem_offset);
    DEVX_SET(wq, wq_context, dbr_umem_valid, 1);

    /* Create DevX object */
    auto status = doca_verbs_wrapper_mlx5dv_devx_obj_create(
        m_ctx, create_in, sizeof(create_in), create_out, sizeof(create_out), &m_srq_obj);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create SRQ. DevX error, syndrome=0x%x",
                 DEVX_GET(nop_out, create_out, syndrome));
        return status;
    }

    m_srq_num = DEVX_GET(create_rmp_out, create_out, rmpn);

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_srq::destroy() noexcept {
    if (m_verbs_device_attr) {
        auto status = doca_verbs_device_attr_free(m_verbs_device_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to free device attr");
            return DOCA_ERROR_INVALID_VALUE;
        }
        m_verbs_device_attr = nullptr;
    }

    if (m_srq_obj) {
        auto destroy_status = doca_verbs_wrapper_mlx5dv_devx_obj_destroy(m_srq_obj);
        if (destroy_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy SRQ object");
            return destroy_status;
        }
        m_srq_obj = nullptr;
    }

    if (m_umem_obj) {
        auto dereg_status = doca_verbs_wrapper_mlx5dv_devx_umem_dereg(m_umem_obj);
        if (dereg_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy UMEM object");
            return dereg_status;
        }
        m_umem_obj = nullptr;
    }

    if (m_umem_buf) {
        free(m_umem_buf);
        m_umem_buf = nullptr;
    }

    return DOCA_SUCCESS;
}

void doca_verbs_srq::create(struct ibv_context *ctx) {
    auto status{DOCA_SUCCESS};
    m_pd = m_init_attr.pd;
    m_ctx = ctx;

    /* Query device attr */
    status = doca_verbs_query_device(m_ctx, &m_verbs_device_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query device attr");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    if ((m_init_attr.srq_type != DOCA_VERBS_SRQ_TYPE_LINKED_LIST) &&
        (m_init_attr.srq_type != DOCA_VERBS_SRQ_TYPE_CONTIGUOUS)) {
        DOCA_LOG(LOG_ERR, "SRQ type is invalid");
        throw DOCA_ERROR_INVALID_VALUE;
    }
    if (m_init_attr.pd == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create DOCA IB Verbs SRQ: pd is NUL");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    m_pd = m_init_attr.pd;

    if (m_init_attr.srq_wr == 0) {
        DOCA_LOG(LOG_ERR, "Failed to create DOCA IB Verbs SRQ: srq_wr is 0");
        throw DOCA_ERROR_INVALID_VALUE;
    }
    if (m_init_attr.srq_wr > m_verbs_device_attr->m_max_srq_wr) {
        DOCA_LOG(LOG_ERR,
                 "Failed to create DOCA IB Verbs SRQ: The requested srq_wr is larger than the "
                 "maximum supported srq_wr by the device");
        throw DOCA_ERROR_NOT_SUPPORTED;
    }
    if (m_init_attr.receive_max_sges == 0) {
        DOCA_LOG(LOG_ERR, "Failed to create DOCA IB Verbs SRQ: receive_max_sges is 0");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    if (m_init_attr.receive_max_sges > m_verbs_device_attr->m_max_srq_sge) {
        DOCA_LOG(LOG_ERR,
                 "Failed to create DOCA IB Verbs SRQ: The requested sge size is larger than the "
                 "maximum supported sge size by the device");
        throw DOCA_ERROR_NOT_SUPPORTED;
    }

    m_rcv_max_sges = m_init_attr.receive_max_sges;

    /* Calculate receive_wqe size */
    m_rcv_wqe_size = m_rcv_max_sges * DOCA_VERBS_DATA_SEG_SIZE_IN_BYTES;
    if (m_init_attr.srq_type == DOCA_VERBS_SRQ_TYPE_LINKED_LIST) {
        m_rcv_wqe_size += DOCA_VERBS_CONTROL_SEG_SIZE_IN_BYTES;
        /* For LL SRQ: Minimum receive WQE size for SRQ is 32 bytes */
        m_rcv_wqe_size = MAX(m_rcv_wqe_size, static_cast<uint32_t>(DOCA_VERBS_MIN_SRQ_SIZE));
    }

    m_rcv_wqe_size = doca_internal_utils_next_power_of_two(m_rcv_wqe_size);

    /* Calculate the actual max_sges size according to the actual wqe size */
    if (m_init_attr.srq_type == DOCA_VERBS_SRQ_TYPE_LINKED_LIST) {
        m_rcv_max_sges = (m_rcv_wqe_size - DOCA_VERBS_CONTROL_SEG_SIZE_IN_BYTES) /
                         DOCA_VERBS_DATA_SEG_SIZE_IN_BYTES;
    } else {  // m_init_attr.srq_type = DOCA_VERBS_SRQ_TYPE_CONTIGUOUS
        m_rcv_max_sges = m_rcv_wqe_size / DOCA_VERBS_DATA_SEG_SIZE_IN_BYTES;
    }

    m_log_rcv_wqe_size = static_cast<uint8_t>(doca_internal_utils_log2(m_rcv_wqe_size));

    /* Calculate SRQ size in bytes */
    auto srq_size_bytes = static_cast<uint32_t>(
        doca_internal_utils_next_power_of_two(m_init_attr.srq_wr * m_rcv_wqe_size));

    /* Calculate SRQ size in receive_wqe units */
    m_srq_size = srq_size_bytes / m_rcv_wqe_size;
    auto log_srq_size = doca_internal_utils_log2(m_srq_size);

    uint32_t dbr_umem_id{0};
    uint64_t dbr_umem_offset{0};
    uint32_t wq_umem_id{0};
    uint64_t wq_umem_offset{0};

    /* Calculate DBR offset */
    auto db_umem_offset = m_srq_size * m_rcv_wqe_size;

    /* Align the Work Queue size to cacheline size for better performance */
    db_umem_offset = doca_internal_utils_align_up_uint32(db_umem_offset, DOCA_VERBS_CACHELINE_SIZE);

    if (m_init_attr.external_umem == nullptr) {
        /* Case of internal umem */

        auto total_umem_size = doca_internal_utils_align_up_uint32(
            db_umem_offset + DOCA_VERBS_SRQ_DB_SIZE, DOCA_VERBS_PAGE_SIZE);
        m_umem_buf = (uint8_t *)memalign(DOCA_VERBS_PAGE_SIZE, total_umem_size);
        memset(m_umem_buf, 0, total_umem_size);

        m_srq_buf = m_umem_buf;

        /* Create UMEM object */
        auto umem_status = doca_verbs_wrapper_mlx5dv_devx_umem_reg(m_ctx, m_srq_buf,
                                                                   total_umem_size, 0, &m_umem_obj);
        if (umem_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create SRQ UMEM");
            throw umem_status;
        }

        dbr_umem_id = m_umem_obj->umem_id;
        dbr_umem_offset = db_umem_offset;

        wq_umem_id = m_umem_obj->umem_id;
        wq_umem_offset = 0;
    } else {
        status = doca_verbs_umem_get_address(m_init_attr.external_umem,
                                             reinterpret_cast<void **>(&m_srq_buf));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem address");
            throw status;
        }

        m_srq_buf += m_init_attr.external_umem_offset;

        status = doca_verbs_umem_get_id(m_init_attr.external_umem, &wq_umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get umem id");
            throw status;
        }

        wq_umem_offset = m_init_attr.external_umem_offset;

        m_db_buffer = reinterpret_cast<uint32_t *>(m_srq_buf + db_umem_offset);
        m_db_buffer = reinterpret_cast<uint32_t *>((reinterpret_cast<uint8_t *>(m_db_buffer)));

        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem base offset");
            throw status;
        }

        dbr_umem_offset = m_init_attr.external_umem_offset + db_umem_offset;
        dbr_umem_id = wq_umem_id;
    }

    m_db_buffer = reinterpret_cast<uint32_t *>(m_srq_buf + db_umem_offset);

    /* Create SRQ object */
    status = create_srq_obj(log_srq_size, m_log_rcv_wqe_size, dbr_umem_offset, dbr_umem_id,
                            wq_umem_offset, wq_umem_id, m_init_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create SRQ object");
        throw status;
    }
}

doca_verbs_srq::doca_verbs_srq(struct ibv_context *ctx,
                               struct doca_verbs_srq_init_attr &verbs_srq_init_attr)
    : m_ctx(ctx), m_init_attr(verbs_srq_init_attr) {
    try {
        create(ctx);
    } catch (...) {
        (void)destroy();
        DOCA_LOG(LOG_ERR, "Failed to create SRQ");
        throw;
    }
}

doca_verbs_srq::~doca_verbs_srq() { static_cast<void>(destroy()); }

void *doca_verbs_srq::get_srq_buf() const noexcept { return (void *)m_srq_buf; }

uint32_t doca_verbs_srq::get_srq_size() const noexcept { return m_srq_size; }

uint32_t doca_verbs_srq::get_rcv_wqe_size() const noexcept { return m_rcv_wqe_size; }

void *doca_verbs_srq::get_dbr_addr() const noexcept { return (void *)m_db_buffer; }

uint32_t doca_verbs_srq::get_srqn() const noexcept { return m_srq_num; }

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_srq_init_attr_create(
    struct doca_verbs_srq_init_attr **verbs_srq_init_attr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create srq_init_attr: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *verbs_srq_init_attr =
        (struct doca_verbs_srq_init_attr *)calloc(1, sizeof(struct doca_verbs_srq_init_attr));
    if (*verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create srq_init_attr: failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_srq_init_attr_destroy(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy srq_init_attr: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    free(verbs_srq_init_attr);
    verbs_srq_init_attr = nullptr;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_srq_create(struct ibv_context *context,
                                   struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
                                   struct doca_verbs_srq **verbs_srq) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_srq: parameter context is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_srq: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (verbs_srq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create verbs_srq: parameter verbs_srq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    try {
        *verbs_srq = new doca_verbs_srq(context, *verbs_srq_init_attr);
        DOCA_LOG(LOG_INFO, "doca_verbs_srq=%p was created", verbs_srq);
        return DOCA_SUCCESS;
    } catch (doca_error_t err) {
        return err;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_srq_destroy(struct doca_verbs_srq *verbs_srq) {
    if (verbs_srq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy verbs_srq: parameter verbs_srq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    auto status = verbs_srq->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy verbs_srq");
        return status;
    }

    delete (verbs_srq);
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_srq_init_attr_set_srq_wr(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, uint32_t srq_wr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set srq_wr: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (srq_wr == 0) {
        DOCA_LOG(LOG_ERR, "Failed to set srq_wr: parameter srq_wr is 0");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_srq_init_attr->srq_wr = srq_wr;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_srq_init_attr_get_srq_wr(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get srq_wr: parameter verbs_srq_init_attr is NULL");
        return 0;
    }

    return verbs_srq_init_attr->srq_wr;
}

doca_error_t doca_verbs_srq_init_attr_set_receive_max_sges(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, uint32_t receive_max_sges) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (receive_max_sges == 0) {
        DOCA_LOG(LOG_ERR, "Failed to set receive_max_sges: parameter receive_max_sges is 0");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_srq_init_attr->receive_max_sges = receive_max_sges;

    return DOCA_SUCCESS;
}

uint32_t doca_verbs_srq_init_attr_get_receive_max_sges(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get receive_max_sges: parameter verbs_srq_init_attr is NULL");
        return 0;
    }

    return verbs_srq_init_attr->receive_max_sges;
}

doca_error_t doca_verbs_srq_init_attr_set_type(struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
                                               enum doca_verbs_srq_type srq_type) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set srq_type: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (srq_type != DOCA_VERBS_SRQ_TYPE_LINKED_LIST && srq_type != DOCA_VERBS_SRQ_TYPE_CONTIGUOUS) {
        DOCA_LOG(LOG_ERR, "Failed to set srq_type: parameter srq_type is invalid");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_srq_init_attr->srq_type = srq_type;

    return DOCA_SUCCESS;
}

enum doca_verbs_srq_type doca_verbs_srq_init_attr_get_type(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get srq_type: parameter verbs_srq_init_attr is NULL");
        return DOCA_VERBS_SRQ_TYPE_LINKED_LIST;
    }

    return verbs_srq_init_attr->srq_type;
}

doca_error_t doca_verbs_srq_init_attr_set_pd(struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
                                             struct ibv_pd *pd) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set pd: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (pd == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set pd: parameter pd is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_srq_init_attr->pd = pd;

    return DOCA_SUCCESS;
}

struct ibv_pd *doca_verbs_srq_init_attr_get_pd(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get pd: parameter verbs_srq_init_attr is NULL");
        return nullptr;
    }

    return verbs_srq_init_attr->pd;
}

doca_error_t doca_verbs_srq_init_attr_set_external_umem(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, struct doca_verbs_umem *external_umem,
    uint64_t external_umem_offset) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter verbs_srq_init_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter external_umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    verbs_srq_init_attr->external_umem = external_umem;
    verbs_srq_init_attr->external_umem_offset = external_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_srq_init_attr_get_external_umem(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
    struct doca_verbs_umem **external_umem, uint64_t *external_umem_offset) {
    if (verbs_srq_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get external_umem: parameter verbs_srq_init_attr is NULL");
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

    *external_umem = verbs_srq_init_attr->external_umem;
    *external_umem_offset = verbs_srq_init_attr->external_umem_offset;

    return DOCA_SUCCESS;
}

void doca_verbs_srq_get_wq(const struct doca_verbs_srq *verbs_srq, void **srq_buf,
                           uint32_t *srq_num_entries, uint32_t *rwqe_size_bytes) {
    *srq_buf = verbs_srq->get_srq_buf();
    *srq_num_entries = verbs_srq->get_srq_size();
    *rwqe_size_bytes = verbs_srq->get_rcv_wqe_size();
}

void *doca_verbs_srq_get_dbr_addr(const struct doca_verbs_srq *verbs_srq) {
    return verbs_srq->get_dbr_addr();
}

uint32_t doca_verbs_srq_get_srqn(const struct doca_verbs_srq *verbs_srq) {
    return verbs_srq->get_srqn();
}
