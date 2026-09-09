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
#include <stdexcept>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_internal.hpp"
#include "doca_verbs_device_attr.hpp"
#include "doca_verbs_cq.hpp"
#include "doca_verbs_net_wrapper.h"
#include "doca_verbs_dev.hpp"
#include "doca_verbs_cq_sdk_wrapper.h"
#include "doca_verbs_uar.hpp"

#define DOCA_VERBS_CQE_SIZE 64

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {
static constexpr uint32_t sc_cq_dbr_size = 8;
using create_cq_in = uint32_t[MLX5_ST_SZ_DW(create_cq_in)];
using create_cq_out = uint32_t[MLX5_ST_SZ_DW(create_cq_out)];

} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_cq_attr_open Member Functions
 *********************************************************************************************************************/

doca_verbs_cq_attr_open::doca_verbs_cq_attr_open()
    : cq_size(0),
      cq_context(nullptr),
      external_umem(nullptr),
      external_umem_offset(0),
      external_dbr_umem(nullptr),
      external_dbr_umem_offset(0),
      external_uar(nullptr),
      cq_overrun(DOCA_VERBS_CQ_DISABLE_OVERRUN),
      cq_collapsed(0),
      m_comp_channel(nullptr) {}

doca_verbs_cq_attr_open::~doca_verbs_cq_attr_open() {}

/**********************************************************************************************************************
 * doca_verbs_cq_open Member Functions
 *********************************************************************************************************************/

doca_verbs_cq_open::doca_verbs_cq_open(struct ibv_context *ibv_ctx,
                                       struct doca_verbs_cq_attr_open *cq_attr)
    : m_ibv_ctx(ibv_ctx) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create CQ, cq_attr is NULL");
        throw std::invalid_argument("cq_attr is NULL");
    }

    m_cq_attr.cq_size = cq_attr->cq_size;
    m_cq_attr.cq_context = cq_attr->cq_context;
    m_cq_attr.external_umem = cq_attr->external_umem;
    m_cq_attr.external_umem_offset = cq_attr->external_umem_offset;
    m_cq_attr.external_uar = cq_attr->external_uar;
    m_cq_attr.cq_overrun = cq_attr->cq_overrun;
    m_cq_attr.cq_collapsed = cq_attr->cq_collapsed;
    m_cq_attr.external_dbr_umem = cq_attr->external_dbr_umem;
    m_cq_attr.external_dbr_umem_offset = cq_attr->external_dbr_umem_offset;
    m_cq_attr.m_comp_channel = cq_attr->m_comp_channel;

    try {
        create();
    } catch (...) {
        (void)destroy();
        DOCA_LOG(LOG_ERR, "Failed to create CQ");
        throw;
    }
}

doca_verbs_cq_open::~doca_verbs_cq_open() { static_cast<void>(destroy()); }

doca_error_t doca_verbs_cq_open::create_cq_obj(uint32_t uar_id, uint32_t log_nb_cqes,
                                               uint64_t db_umem_offset, uint32_t db_umem_id,
                                               uint32_t wq_umem_id, uint64_t cq_umem_offset,
                                               bool cq_overrun, uint8_t cq_collapsed) noexcept {
    create_cq_in create_in{0};
    create_cq_out create_out{0};

    DEVX_SET(create_cq_in, create_in, opcode, MLX5_CMD_OP_CREATE_CQ);
    DEVX_SET(create_cq_in, create_in, cq_context.cqe_sz, MLX5_CQC_CQE_SZ_BYTES_64);
    DEVX_SET(create_cq_in, create_in, cq_context.cc, cq_collapsed);  // Enable/Disable collapsed CQ
    DEVX_SET(create_cq_in, create_in, cq_context.oi,
             static_cast<uint8_t>(cq_overrun));                              // Enable overrun
    DEVX_SET(create_cq_in, create_in, cq_context.log_cq_size, log_nb_cqes);  //<--
    DEVX_SET(create_cq_in, create_in, cq_context.uar_page, uar_id);
    DEVX_SET(create_cq_in, create_in, cq_umem_id, wq_umem_id);
    DEVX_SET(create_cq_in, create_in, cq_umem_valid, 1);
    DEVX_SET64(create_cq_in, create_in, cq_umem_offset, cq_umem_offset);
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

void doca_verbs_cq_open::create() {
    auto status{DOCA_SUCCESS};

    /* Query device attr */
    status = doca_verbs_query_device(m_ibv_ctx, &m_verbs_device_attr);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to query device attr");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    if (doca_internal_utils_is_power_of_two(m_cq_attr.cq_size) == false) {
        DOCA_LOG(LOG_ERR, "Number of CQE is not a power of 2");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    m_num_cqes = static_cast<uint32_t>(m_cq_attr.cq_size);
    uint32_t log_nb_cqes = doca_internal_utils_log2(m_num_cqes);

    if (m_num_cqes > m_verbs_device_attr->m_max_cqe) {
        DOCA_LOG(LOG_ERR, "CQ cq_size is invalid");
        throw DOCA_ERROR_INVALID_VALUE;
    }

    uint32_t umem_id{0};
    uint32_t dbr_umem_id{0};
    uint64_t dbr_umem_offset{0};

    if (m_cq_attr.external_umem == nullptr) {
        /* Case of internal umem */
        uint32_t total_umem_size = doca_internal_utils_align_up_uint32(
            m_num_cqes * DOCA_VERBS_CQE_SIZE, DOCA_VERBS_PAGE_SIZE);

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
    } else {
        /* Case of external umem */
        uint8_t *umem_base = nullptr;
        status = doca_verbs_umem_get_address(m_cq_attr.external_umem,
                                             reinterpret_cast<void **>(&umem_base));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem address");
            throw status;
        }

        /* Apply ring umem offset so m_cq_buf points to the start of CQEs */
        m_cq_buf = umem_base + m_cq_attr.external_umem_offset;

        status = doca_verbs_umem_get_id(m_cq_attr.external_umem, &umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external umem id");
            throw status;
        }
    }

    if (m_cq_attr.external_dbr_umem == nullptr) {
        uint32_t total_dbr_umem_size =
            doca_internal_utils_align_up_uint32(sc_cq_dbr_size, DOCA_VERBS_PAGE_SIZE);
        m_dbr_umem_buf = (uint32_t *)memalign(DOCA_VERBS_PAGE_SIZE, total_dbr_umem_size);
        memset(m_dbr_umem_buf, 0, total_dbr_umem_size);

        auto dbr_umem_status = doca_verbs_wrapper_mlx5dv_devx_umem_reg(
            m_ibv_ctx, m_dbr_umem_buf, total_dbr_umem_size, 0, &m_dbr_umem_obj);
        if (dbr_umem_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to create CQ DBR UMEM");
            throw dbr_umem_status;
        }

        dbr_umem_offset = 0;
        dbr_umem_id = m_dbr_umem_obj->umem_id;
        m_db_buffer = m_dbr_umem_buf;
    } else {
        /* Separate DBR umem path */
        status = doca_verbs_umem_get_id(m_cq_attr.external_dbr_umem, &dbr_umem_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external DBR umem id");
            throw status;
        }

        uint8_t *dbr_base = nullptr;
        status = doca_verbs_umem_get_address(m_cq_attr.external_dbr_umem,
                                             reinterpret_cast<void **>(&dbr_base));
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external DBR umem address");
            throw status;
        }

        dbr_umem_offset = m_cq_attr.external_dbr_umem_offset;
        m_db_buffer = reinterpret_cast<uint32_t *>(dbr_base + dbr_umem_offset);
    }

    m_ci_dbr = &m_db_buffer[MLX5_CQ_SET_CI];
    m_arm_dbr = &m_db_buffer[MLX5_CQ_ARM_DB];

    uint32_t uar_id{};
    if (m_cq_attr.external_uar == nullptr) {
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

        m_uar_db_reg = reinterpret_cast<uint64_t *>(
            reinterpret_cast<uintptr_t>(m_uar_obj->base_addr) + MLX5_CQ_DOORBELL);
        uar_id = m_uar_obj->page_id;
    } else {
        /* Case of external UAR */
        status = doca_verbs_uar_id_get(m_cq_attr.external_uar, &uar_id);
        if (status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to get external UAR ID");
            throw status;
        }

        void *base_addr{};
        if (m_cq_attr.external_uar->type != DOCA_VERBS_SDK_LIB_TYPE_OPEN) {
            DOCA_LOG(LOG_ERR, "External UAR for CQ from SDK is not supported");
            status = DOCA_ERROR_NOT_SUPPORTED;
            throw status;
        }
        base_addr = m_cq_attr.external_uar->open->get_base_addr();
        m_uar_db_reg =
            reinterpret_cast<uint64_t *>(reinterpret_cast<uintptr_t>(base_addr) + MLX5_CQ_DOORBELL);
    }

    /* Create CQ object */
    uint64_t ring_offset =
        (m_cq_attr.external_umem != nullptr) ? m_cq_attr.external_umem_offset : 0;
    status = create_cq_obj(uar_id, log_nb_cqes, dbr_umem_offset, dbr_umem_id, umem_id, ring_offset,
                           m_cq_attr.cq_overrun, m_cq_attr.cq_collapsed);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create CQ object");
        throw DOCA_ERROR_DRIVER;
    }

    if (m_cq_attr.m_comp_channel &&
        m_cq_attr.m_comp_channel->type != DOCA_VERBS_SDK_LIB_TYPE_OPEN) {
        DOCA_LOG(LOG_ERR, "Completion channel for CQ from SDK is not supported");
        status = DOCA_ERROR_NOT_SUPPORTED;
        throw status;
    }

    if (m_cq_attr.m_comp_channel) {
        m_comp_channel = m_cq_attr.m_comp_channel->open;
        if (m_comp_channel != nullptr) {
            DOCA_LOG(LOG_INFO, "Verbs cq %p: Registering to comp channel %p", this, m_comp_channel);
            status = m_comp_channel->register_cq(this);
            if (status != DOCA_SUCCESS) {
                DOCA_LOG(LOG_ERR, "Failed to register CQ to completion events channel");
                throw DOCA_ERROR_DRIVER;
            }
        }
    }

    DOCA_LOG(LOG_INFO, "DOCA IB Verbs CQ %p: has been successfully created", this);
}

doca_error_t doca_verbs_cq_open::destroy() noexcept {
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

    if (m_dbr_umem_obj) {
        auto dereg_status = doca_verbs_wrapper_mlx5dv_devx_umem_dereg(m_dbr_umem_obj);
        if (dereg_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy DBR UMEM object");
            return dereg_status;
        }
        m_dbr_umem_obj = nullptr;
    }

    if (m_dbr_umem_buf) {
        free(m_dbr_umem_buf);
        m_dbr_umem_buf = nullptr;
    }

    return DOCA_SUCCESS;
}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_cq_attr_create(doca_verbs_cq_attr_t **cq_attr) {
    doca_verbs_cq_attr_t *attr_ = nullptr;

    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create cq_attr: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    attr_ = (doca_verbs_cq_attr_t *)calloc(1, sizeof(doca_verbs_cq_attr_t));
    if (attr_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_verbs_cq_attr_t",
                 __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Try with DOCA SDK first */
    auto err = doca_verbs_sdk_wrapper_cq_attr_create(&(attr_->sdk));
    if (err == DOCA_SDK_WRAPPER_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Use DOCA Verbs CQ Attr SDK", __func__);
        attr_->type = DOCA_VERBS_SDK_LIB_TYPE_SDK;
        (*cq_attr) = attr_;
        return DOCA_SUCCESS;
    } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
        DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA Verbs CQ Attr open", __func__);

    attr_->type = DOCA_VERBS_SDK_LIB_TYPE_OPEN;

    try {
        attr_->open = new doca_verbs_cq_attr_open();
        DOCA_LOG(LOG_INFO, "doca_verbs_cq_attr_open=%p was created", attr_);
        (*cq_attr) = attr_;
        return DOCA_SUCCESS;
    } catch (...) {
        DOCA_LOG(LOG_ERR, "doca_verbs_cq_attr_open allocation failed");
        goto exit_error;
    }

exit_error:
    if (attr_ != nullptr) {
        if (attr_->open) delete attr_->open;
        free(attr_);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_verbs_cq_attr_destroy(doca_verbs_cq_attr_t *cq_attr) {
    doca_error_t status = DOCA_SUCCESS;

    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy cq_attr: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_destroy(cq_attr->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            cq_attr->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

exit:
    if (cq_attr->open) delete cq_attr->open;
    memset(cq_attr, 0, sizeof(doca_verbs_cq_attr_t));
    free(cq_attr);

    return status;
}

doca_error_t doca_verbs_cq_attr_set_cq_size(doca_verbs_cq_attr_t *cq_attr, uint32_t cq_size) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set_cq_size: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_cq_size(cq_attr->sdk, cq_size);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->cq_size = cq_size;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_comp_channel(doca_verbs_cq_attr_t *cq_attr,
                                                 doca_verbs_comp_channel_t *comp_channel) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set_comp_channel: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (comp_channel == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set_comp_channel: parameter comp_channel is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        if (comp_channel->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "Failed to set_comp_channel: parameter comp_channel is not SDK");
            return DOCA_ERROR_INVALID_VALUE;
        }

        auto err = doca_verbs_sdk_wrapper_cq_attr_set_comp_channel(cq_attr->sdk, comp_channel->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->m_comp_channel = comp_channel;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_st(doca_verbs_cq_attr_t *cq_attr,
                                       enum doca_verbs_cq_state cq_state) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set_comp_channel: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_st(cq_attr->sdk, cq_state);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        } else if (err == DOCA_SDK_WRAPPER_NOT_SUPPORTED) {
            return DOCA_ERROR_NOT_SUPPORTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    // No action for open source implementation

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_external_umem(doca_verbs_cq_attr_t *cq_attr,
                                                  doca_verbs_umem_t *external_umem,
                                                  uint64_t external_umem_offset) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter cq_attr is NULL.");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_umem: parameter external_umem is NULL.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_external_umem(cq_attr->sdk, external_umem,
                                                                    external_umem_offset);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->external_umem = external_umem;
    cq_attr->open->external_umem_offset = external_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_external_dbr_umem(doca_verbs_cq_attr_t *cq_attr,
                                                      doca_verbs_umem_t *external_dbr_umem,
                                                      uint64_t external_dbr_umem_offset) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_dbr_umem: parameter cq_attr is NULL.");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_dbr_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_dbr_umem: parameter external_dbr_umem is NULL.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_external_dbr_umem(
            cq_attr->sdk, external_dbr_umem, external_dbr_umem_offset);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        } else if (err == DOCA_SDK_WRAPPER_NOT_SUPPORTED) {
            return DOCA_ERROR_NOT_SUPPORTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->external_dbr_umem = external_dbr_umem;
    cq_attr->open->external_dbr_umem_offset = external_dbr_umem_offset;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_external_uar(doca_verbs_cq_attr_t *cq_attr,
                                                 doca_verbs_uar_t *external_uar) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (external_uar == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set external_uar: parameter external_uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_external_uar(cq_attr->sdk, external_uar);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->external_uar = external_uar;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_cq_overrun(doca_verbs_cq_attr_t *cq_attr,
                                               enum doca_verbs_cq_overrun cq_overrun) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_overrun: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_cq_overrun(cq_attr->sdk, cq_overrun);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->cq_overrun = cq_overrun;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_attr_set_cq_collapsed(doca_verbs_cq_attr_t *cq_attr,
                                                 uint8_t cq_collapsed) {
    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_collapsed: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_attr->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_attr_set_cq_collapsed(cq_attr->sdk, cq_collapsed);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq_attr->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ attr open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_collapsed != 0 && cq_collapsed != 1) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_collapsed: value cq_collapsed can only be 0 or 1");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_attr->open->cq_collapsed = cq_collapsed;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_create(doca_dev_t *net_dev, doca_verbs_cq_attr_t *cq_attr,
                                  doca_verbs_cq_t **cq) {
    doca_verbs_cq_t *cq_ = nullptr;

    if (cq_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create cq: parameter cq_attr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    cq_ = (doca_verbs_cq_t *)calloc(1, sizeof(doca_verbs_cq_t));
    if (cq_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_verbs_cq_t", __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Try with DOCA SDK first */
    auto err = doca_verbs_sdk_wrapper_cq_create(net_dev, cq_attr, &(cq_->sdk));
    if (err == DOCA_SDK_WRAPPER_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Use DOCA Verbs CQ SDK", __func__);
        cq_->type = DOCA_VERBS_SDK_LIB_TYPE_SDK;
        (*cq) = cq_;
        return DOCA_SUCCESS;
    } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
        DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA Verbs CQ open", __func__);

    cq_->type = DOCA_VERBS_SDK_LIB_TYPE_OPEN;

    try {
        cq_->open = new doca_verbs_cq_open(net_dev->open->get_ctx(), cq_attr->open);
        DOCA_LOG(LOG_INFO, "doca_verbs_cq_open=%p was created", cq_);
        (*cq) = cq_;
        return DOCA_SUCCESS;
    } catch (...) {
        DOCA_LOG(LOG_ERR, "doca_verbs_cq_open allocation failed");
        goto exit_error;
    }

exit_error:
    if (cq_ != nullptr) {
        if (cq_->open) delete cq_->open;
        free(cq_);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_verbs_cq_destroy(doca_verbs_cq_t *cq) {
    doca_error_t status = DOCA_SUCCESS;

    if (cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy cq: parameter cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_destroy(cq->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            cq->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (cq->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

    status = cq->open->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy cq.");
        goto exit;
    }

exit:
    if (cq->open) delete cq->open;
    memset(cq, 0, sizeof(doca_verbs_cq_t));
    free(cq);

    return status;
}

doca_error_t doca_verbs_cq_get_wq(doca_verbs_cq_t *cq, void **cq_buf, uint32_t *cq_num_entries,
                                  uint8_t *cq_entry_size) {
    if (cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get cq wq: parameter cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_get_wq(cq->sdk, cq_buf, cq_num_entries, cq_entry_size);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        return DOCA_ERROR_INVALID_VALUE;
    }

    *cq_buf = cq->open->get_cq_buf();
    *cq_num_entries = cq->open->get_cq_num_entries();
    *cq_entry_size = DOCA_VERBS_CQE_SIZE;

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_get_dbr_db_addr(doca_verbs_cq_t *cq, uint64_t **uar_db_reg,
                                           uint32_t **ci_dbr, uint32_t **arm_dbr) {
    if (cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get cq dbr_addr: parameter cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_get_dbr_addr(cq->sdk, uar_db_reg, ci_dbr, arm_dbr);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (cq->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        return DOCA_ERROR_INVALID_VALUE;
    }

    *uar_db_reg = cq->open->get_cq_uar_db_reg();
    *ci_dbr = cq->open->get_cq_ci_dbr();
    *arm_dbr = cq->open->get_cq_arm_dbr();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_get_cq_num(const doca_verbs_cq_t *cq, uint32_t *cqn) {
    if (cq == nullptr || cqn == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get cq cqn: invalid NULL parameter");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_get_cqn(cq->sdk, cqn);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) return DOCA_SUCCESS;
        if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    if (cq->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        return DOCA_ERROR_INVALID_VALUE;
    }

    *cqn = cq->open->get_cqn();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_cq_set_cq_context(doca_verbs_cq_t *cq, void *cq_context) {
    if (cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_context: parameter cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_cq_set_cq_context(cq->sdk, cq_context);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        } else if (err == DOCA_SDK_WRAPPER_NOT_SUPPORTED) {
            return DOCA_ERROR_NOT_SUPPORTED;
        }
    }

    if (cq->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        return DOCA_ERROR_INVALID_VALUE;
    }

    DOCA_LOG(LOG_INFO, "DOCA open, setting cq_context %p.", cq_context);
    cq->open->set_cq_context(cq_context);

    return DOCA_SUCCESS;
}

doca_verbs_comp_channel_open::doca_verbs_comp_channel_open(struct ibv_context *ibv_ctx_)
    : ibv_ctx(ibv_ctx_) {
    doca_error_t status = DOCA_SUCCESS;
    int fd_flags = 0;
    int fd_fcntl_status = 0;

    status = doca_verbs_wrapper_mlx5dv_devx_create_event_channel(
        ibv_ctx, MLX5DV_DEVX_CREATE_EVENT_CHANNEL_FLAGS_OMIT_EV_DATA, &channel);
    if (status != DOCA_SUCCESS)
        throw std::runtime_error("Failed to create doca_verbs_comp_channel_open");

    fd_flags = fcntl(channel->fd, F_GETFL);
    if (fd_flags < 0) {
        doca_verbs_wrapper_mlx5dv_devx_destroy_event_channel(channel);
        throw std::runtime_error("Failed to set doca_verbs_comp_channel_open F_GETFL");
    }

    fd_fcntl_status = fcntl(channel->fd, F_SETFL, fd_flags | O_NONBLOCK);
    if (fd_fcntl_status < 0) {
        doca_verbs_wrapper_mlx5dv_devx_destroy_event_channel(channel);
        throw std::runtime_error("Failed to set doca_verbs_comp_channel_open O_NONBLOCK");
    }
}

doca_verbs_comp_channel_open::~doca_verbs_comp_channel_open() {
    doca_verbs_wrapper_mlx5dv_devx_destroy_event_channel(channel);
    channel = nullptr;
}

doca_error_t doca_verbs_comp_channel_open::register_cq(struct doca_verbs_cq_open *cq) {
    doca_error_t status;
    uint16_t event_num = MLX5_EVENT_TYPE_CODING_COMPLETION_EVENTS;

    status = doca_verbs_wrapper_mlx5dv_devx_subscribe_devx_event(
        channel, cq->get_cq_obj(), sizeof(event_num), &event_num, reinterpret_cast<uint64_t>(cq));
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to create doca_verbs_comp_channel_open");
        return status;
    }

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_comp_channel_open::get_cq_event(void **cq_context) {
    doca_error_t status = DOCA_SUCCESS;
    ssize_t bytes_read = 0;
    struct mlx5dv_devx_async_event_hdr event_hdr {
        0
    };
    struct doca_verbs_cq_open *cq;

    status = doca_verbs_wrapper_mlx5dv_devx_get_event(channel, &event_hdr, sizeof(event_hdr),
                                                      &bytes_read);
    if (status) goto out;

    if (bytes_read < 0) {
        if (bytes_read == -1) {
            status = DOCA_ERROR_AGAIN;
            *cq_context = nullptr;
        } else {
            status = DOCA_ERROR_DRIVER;
            *cq_context = nullptr;
        }
        goto out;
    } else if ((size_t)bytes_read < sizeof(event_hdr)) {
        status = DOCA_ERROR_UNEXPECTED;
        *cq_context = nullptr;
        goto out;
    }

    DOCA_LOG(LOG_INFO, "bytes_read %ld event_hdr.cookie %lx", bytes_read, event_hdr.cookie);
    cq = reinterpret_cast<struct doca_verbs_cq_open *>(event_hdr.cookie);
    *cq_context = cq->get_cq_context();

out:
    return status;
}

doca_error_t doca_verbs_comp_channel_create(const doca_dev_t *net_dev,
                                            doca_verbs_comp_channel_t **verbs_comp_channel) {
    doca_verbs_comp_channel_t *compch = nullptr;

    if (net_dev == nullptr || verbs_comp_channel == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to set cq_context: parameter cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    compch = (doca_verbs_comp_channel_t *)calloc(1, sizeof(doca_verbs_comp_channel_t));
    if (compch == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_verbs_comp_channel_t",
                 __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    if (net_dev->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_comp_channel_create(net_dev->sdk_context, &(compch->sdk));
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            compch->type = DOCA_VERBS_SDK_LIB_TYPE_SDK;
            (*verbs_comp_channel) = compch;
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            free(compch);
            return DOCA_ERROR_UNEXPECTED;
        } else if (err == DOCA_SDK_WRAPPER_NOT_SUPPORTED) {
            free(compch);
            return DOCA_ERROR_NOT_SUPPORTED;
        }
    }

    if (net_dev->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA Verbs CQ Attr open", __func__);

    compch->type = DOCA_VERBS_SDK_LIB_TYPE_OPEN;

    try {
        compch->open = new doca_verbs_comp_channel_open(net_dev->open->get_ctx());
        DOCA_LOG(LOG_INFO, "doca_verbs_comp_channel_open=%p was created", compch);
        (*verbs_comp_channel) = compch;
        return DOCA_SUCCESS;
    } catch (...) {
        DOCA_LOG(LOG_ERR, "doca_verbs_comp_channel_open allocation failed");
        goto exit_error;
    }

exit_error:
    if (compch != nullptr) {
        if (compch->open) delete compch->open;
        free(compch);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_verbs_comp_channel_destroy(doca_verbs_comp_channel_t *compch) {
    doca_error_t status = DOCA_SUCCESS;

    if (compch == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy compch: parameter compch is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (compch->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_comp_channel_destroy(compch->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            compch->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (compch->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

exit:
    if (compch->open) delete compch->open;
    memset(compch, 0, sizeof(doca_verbs_comp_channel_t));
    free(compch);

    return status;
}

doca_error_t doca_verbs_get_cq_comp_channel_event(doca_verbs_comp_channel_t *compch,
                                                  void **cq_context) {
    if (compch == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get cq comp channel: parameter compch is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (cq_context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get cq cq_context: parameter cq_context is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (compch->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_get_cq_event(compch->sdk, cq_context);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) return DOCA_SUCCESS;
        if (err == DOCA_SDK_WRAPPER_ERROR_AGAIN)
            return DOCA_ERROR_AGAIN;
        else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (compch->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs CQ open instance provided at %s line %d.", __func__,
                 __LINE__);
        return DOCA_ERROR_INVALID_VALUE;
    }

    return compch->open->get_cq_event(cq_context);
}

doca_error_t doca_verbs_ack_cq_events(doca_verbs_cq_t *verbs_cq, unsigned int nevents) {
    if (verbs_cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to ack cq event: parameter verbs_cq is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (nevents == 0) {
        DOCA_LOG(LOG_ERR, "Failed to ack cq event: parameter nevents is 0");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (verbs_cq->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_ack_cq_events(verbs_cq->sdk, nevents);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) return DOCA_SUCCESS;
        if (err == DOCA_SDK_WRAPPER_ERROR_AGAIN)
            return DOCA_ERROR_AGAIN;
        else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    // No need for open source version
    return DOCA_SUCCESS;
}
