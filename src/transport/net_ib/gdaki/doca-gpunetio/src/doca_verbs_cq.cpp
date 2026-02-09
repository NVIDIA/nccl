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
#include "doca_verbs_cq.hpp"
#include "doca_verbs_net_wrapper.h"

#define DOCA_VERBS_CQE_SIZE 64

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {
static constexpr uint32_t sc_cq_doorbell_size = 64;

using create_cq_in = uint32_t[MLX5_ST_SZ_DW(create_cq_in)];
using create_cq_out = uint32_t[MLX5_ST_SZ_DW(create_cq_out)];

} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_cq Member Functions
 *********************************************************************************************************************/

doca_verbs_cq::doca_verbs_cq(struct ibv_context *ibv_ctx, struct doca_verbs_cq_attr &cq_attr)
    : m_ibv_ctx(ibv_ctx), m_cq_attr(cq_attr) {
    try {
        create(cq_attr);
    } catch (...) {
        (void)destroy();
        DOCA_LOG(LOG_ERR, "Failed to create CQ");
        throw;
    }
}

doca_verbs_cq::~doca_verbs_cq() { static_cast<void>(destroy()); }

doca_error_t doca_verbs_cq::create_cq_obj(uint32_t uar_id, uint32_t log_nb_cqes,
                                          uint64_t db_umem_offset, uint32_t db_umem_id,
                                          uint32_t wq_umem_id, bool cq_overrun) noexcept {
    create_cq_in create_in{0};
    create_cq_out create_out{0};

    DEVX_SET(create_cq_in, create_in, opcode, MLX5_CMD_OP_CREATE_CQ);
    DEVX_SET(create_cq_in, create_in, cq_context.cqe_sz, MLX5_CQC_CQE_SZ_BYTES_64);
    DEVX_SET(create_cq_in, create_in, cq_context.cc, 0x0);  // Disable collapsed CQ
    DEVX_SET(create_cq_in, create_in, cq_context.oi,
             static_cast<uint8_t>(cq_overrun));                              // Enable overrun
    DEVX_SET(create_cq_in, create_in, cq_context.log_cq_size, log_nb_cqes);  //<--
    DEVX_SET(create_cq_in, create_in, cq_context.uar_page, uar_id);
    DEVX_SET(create_cq_in, create_in, cq_umem_id, wq_umem_id);
    DEVX_SET(create_cq_in, create_in, cq_umem_valid, 1);
    DEVX_SET64(create_cq_in, create_in, cq_umem_offset, 0x0);
    DEVX_SET(create_cq_in, create_in, cq_context.dbr_umem_id, db_umem_id);
    DEVX_SET(create_cq_in, create_in, cq_context.dbr_umem_valid, 1);
    DEVX_SET64(create_cq_in, create_in, cq_context.dbr_addr, db_umem_offset);

    uint32_t element_id;
    auto ret = doca_verbs_wrapper_mlx5dv_devx_query_eqn(m_ibv_ctx, 0, &element_id);
    if (ret != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query devx eqn");
        return DOCA_ERROR_OPERATING_SYSTEM;
    }

    DEVX_SET(create_cq_in, create_in, cq_context.c_eqn, element_id);

    /* Since cq_umem_valid == 1, FW deduces page size from umem and this field is reserved */
    DEVX_SET(create_cq_in, create_in, cq_context.log_page_size,
             0);  // GPU_PAGE_SHIFT - MLX5_ADAPTER_PAGE_SHIFT

    /* Create DevX object */
    auto status = doca_verbs_wrapper_mlx5dv_devx_obj_create(
        m_ibv_ctx, create_in, sizeof(create_in), create_out, sizeof(create_out), &m_cq_obj);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create CQ. DevX error, syndrome=0x%x",
                 DEVX_GET(nop_out, create_out, syndrome));
        return status;
    }

    m_cqn = DEVX_GET(create_cq_out, create_out, cqn);

    return DOCA_SUCCESS;
}

void doca_verbs_cq::create(struct doca_verbs_cq_attr &cq_attr) {
    auto status{DOCA_SUCCESS};

    if ((cq_attr.external_umem != nullptr && cq_attr.external_umem_dbr == nullptr) ||
        (cq_attr.external_umem == nullptr && cq_attr.external_umem_dbr != nullptr)) {
        DOCA_LOG(LOG_ERR, "Both UMEM should be either external or internal");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    /* Query device attr */
    status = doca_verbs_query_device(m_ibv_ctx, &m_verbs_device_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query device attr");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    if (doca_internal_utils_is_power_of_two(cq_attr.cq_size) == false) {
        DOCA_LOG(LOG_ERR, "Number of CQE is not a power of 2");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    m_num_cqes = static_cast<uint32_t>(cq_attr.cq_size);
    uint32_t log_nb_cqes = doca_internal_utils_log2(m_num_cqes);

    if (m_num_cqes > m_verbs_device_attr->m_max_cqe) {
        DOCA_LOG(LOG_ERR, "CQ cq_size is invalid");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    uint32_t umem_id{0};
    uint32_t dbr_umem_id{0};
    uint64_t dbr_umem_offset{0};

    dbr_umem_offset = m_num_cqes * DOCA_VERBS_CQE_SIZE;
    dbr_umem_offset =
        doca_internal_utils_align_up_uint32(dbr_umem_offset, DOCA_VERBS_CACHELINE_SIZE);

    if (cq_attr.external_umem == nullptr) {
        /* Case of internal umem */
        uint32_t total_umem_size = doca_internal_utils_align_up_uint32(
            dbr_umem_offset + sc_cq_doorbell_size, DOCA_VERBS_PAGE_SIZE);

        m_umem_buf = (uint8_t *)memalign(DOCA_VERBS_PAGE_SIZE, total_umem_size);
        memset(m_umem_buf, 0, total_umem_size);

        auto umem_status = doca_verbs_wrapper_mlx5dv_devx_umem_reg(m_ibv_ctx, m_umem_buf,
                                                                   total_umem_size, 0, &m_umem_obj);
        if (umem_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create CQ UMEM");
            throw umem_status;
        }

        m_cq_buf = m_umem_buf;
        umem_id = m_umem_obj->umem_id;
        m_db_buffer = reinterpret_cast<uint32_t *>(m_cq_buf + dbr_umem_offset);
    } else {
        uint8_t *tmp_db_buffer;

        /* Case of external umem */
        status = doca_verbs_umem_get_address(cq_attr.external_umem,
                                             reinterpret_cast<void **>(&m_cq_buf));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem address");
            throw status;
        }

        status = doca_verbs_umem_get_id(cq_attr.external_umem, &umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem id");
            throw status;
        }

        /* Case of external umem */
        status = doca_verbs_umem_get_address(cq_attr.external_umem_dbr,
                                             reinterpret_cast<void **>(&tmp_db_buffer));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem address");
            throw status;
        }

        status = doca_verbs_umem_get_id(cq_attr.external_umem_dbr, &dbr_umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem id");
            throw status;
        }

        dbr_umem_offset = cq_attr.external_umem_dbr_offset;
        m_db_buffer = reinterpret_cast<uint32_t *>(tmp_db_buffer + dbr_umem_offset);
    }

    m_ci_dbr = &m_db_buffer[MLX5_CQ_SET_CI];
    m_arm_dbr = &m_db_buffer[MLX5_CQ_ARM_DB];

    uint32_t uar_id{};
    if (cq_attr.external_uar == nullptr) {
        auto uar_status = doca_verbs_wrapper_mlx5dv_devx_alloc_uar(
            m_ibv_ctx, MLX5DV_UAR_ALLOC_TYPE_NC, &m_uar_obj);
        if (uar_status != DOCA_SUCCESS) {
            uar_status = doca_verbs_wrapper_mlx5dv_devx_alloc_uar(
                m_ibv_ctx, MLX5DV_UAR_ALLOC_TYPE_BF, &m_uar_obj);
            if (uar_status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to create UAR");
                throw uar_status;
            }
        }

        m_uar_db_reg = reinterpret_cast<uint64_t *>(m_uar_obj->reg_addr);
        uar_id = m_uar_obj->page_id;
    } else {
        /* Case of external UAR */
        status = doca_verbs_uar_id_get(cq_attr.external_uar, &uar_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external UAR ID");
            throw status;
        }

        void *reg_addr{};
        status = doca_verbs_uar_reg_addr_get(cq_attr.external_uar, &reg_addr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external UAR reg_addr");
            throw status;
        }
        m_uar_db_reg = reinterpret_cast<uint64_t *>(reg_addr);
    }

    /* Create CQ object */
    status = create_cq_obj(uar_id, log_nb_cqes, dbr_umem_offset, dbr_umem_id, umem_id,
                           cq_attr.cq_overrun);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create CQ object");
        throw DOCA_ERROR_DRIVER;
    }

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs CQ %p: has been successfully created", this);
}

doca_error_t doca_verbs_cq::destroy() noexcept {
    if (m_verbs_device_attr) {
        auto status = doca_verbs_device_attr_free(m_verbs_device_attr);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to free device attr");
            return DOCA_ERROR_INVALID_VALUE;
        }
        m_verbs_device_attr = nullptr;
    }

    if (m_cq_obj) {
        auto destroy_status = doca_verbs_wrapper_mlx5dv_devx_obj_destroy(m_cq_obj);
        if (destroy_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy CQ object");
            return destroy_status;
        }
        m_cq_obj = nullptr;
    }

    if (m_uar_obj) {
        auto free_uar_status = doca_verbs_wrapper_mlx5dv_devx_free_uar(m_uar_obj);
        if (free_uar_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to free UAR");
            return free_uar_status;
        }
        m_uar_obj = nullptr;
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

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_cq_attr_create(struct doca_verbs_cq_attr **verbs_cq_attr) {
    if (verbs_cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create cq_attr: parameter verbs_cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *verbs_cq_attr = (struct doca_verbs_cq_attr *)calloc(1, sizeof(struct doca_verbs_cq_attr));
    if (*verbs_cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create cq_attr: failed to allocate memory");
        return DOCA_ERROR_NO_MEMORY;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_destroy(struct doca_verbs_cq_attr *cq_attr) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy doca_verbs_cq_attr. parameter cq_attr=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    free(cq_attr);
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_cq_size(struct doca_verbs_cq_attr *cq_attr, uint32_t cq_size) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_size: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->cq_size = cq_size;
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_cq_context(struct doca_verbs_cq_attr *cq_attr,
                                               void *cq_context) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_context: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->cq_context = cq_context;
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_external_umem(struct doca_verbs_cq_attr *cq_attr,
                                                  struct doca_verbs_umem *external_umem,
                                                  uint64_t external_umem_offset) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter cq_attr is NULL.");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter external_umem is NULL.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->external_umem = external_umem;
    cq_attr->external_umem_offset = external_umem_offset;
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_external_dbr_umem(struct doca_verbs_cq_attr *cq_attr,
                                                      struct doca_verbs_umem *external_umem,
                                                      uint64_t external_umem_offset) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter external_umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->external_umem_dbr = external_umem;
    cq_attr->external_umem_dbr_offset = external_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_external_uar(struct doca_verbs_cq_attr *cq_attr,
                                                 struct doca_verbs_uar *external_uar) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_uar == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter external_uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->external_uar = external_uar;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_cq_overrun(struct doca_verbs_cq_attr *cq_attr,
                                               enum doca_verbs_cq_overrun overrun) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->cq_overrun = overrun;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_create(struct ibv_context *context,
                                  struct doca_verbs_cq_attr *verbs_cq_attr,
                                  struct doca_verbs_cq **verbs_cq) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create doca_verbs_cq. param context=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (verbs_cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create doca_verbs_cq. param verbs_cq_attr=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    try {
        *verbs_cq = new doca_verbs_cq(context, *verbs_cq_attr);
        DOCA_LOG(LOG_INFO, "IB Verbs Context %p: verbs_cq=%p was created", context, *verbs_cq);
        return DOCA_SUCCESS;
    } catch (doca_error_t err) {
        return err;
    }
}

doca_error_t doca_verbs_cq_destroy(struct doca_verbs_cq *verbs_cq) {
    if (verbs_cq == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to destroy verbs_cq: parameter verbs_cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    auto status = verbs_cq->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Failed to destroy verbs_cq.");
        return status;
    }

    delete (verbs_cq);
    return DOCA_SUCCESS;
}

void doca_verbs_cq_get_wq(struct doca_verbs_cq *verbs_cq, void **cq_buf, uint32_t *cq_num_entries,
                          uint8_t *cq_entry_size) {
    *cq_buf = verbs_cq->get_cq_buf();
    *cq_num_entries = verbs_cq->get_cq_num_entries();
    *cq_entry_size = DOCA_VERBS_CQE_SIZE;
}

void doca_verbs_cq_get_dbr_addr(struct doca_verbs_cq *verbs_cq, uint64_t **uar_db_reg,
                                uint32_t **ci_dbr, uint32_t **arm_dbr) {
    *uar_db_reg = verbs_cq->get_cq_uar_db_reg();
    *ci_dbr = verbs_cq->get_cq_ci_dbr();
    *arm_dbr = verbs_cq->get_cq_arm_dbr();
}

uint32_t doca_verbs_cq_get_cqn(const struct doca_verbs_cq *verbs_cq) { return verbs_cq->get_cqn(); }
