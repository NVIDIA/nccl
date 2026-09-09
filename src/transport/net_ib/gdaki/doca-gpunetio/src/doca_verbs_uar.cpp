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

#include "doca_gpunetio_config.h"
#include "doca_verbs_net_wrapper.h"
#include "doca_internal.hpp"
#include "doca_verbs_device_attr.hpp"
#include "doca_verbs_uar.hpp"
#include "doca_verbs_uar_sdk_wrapper.h"

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {

doca_error_t convert_doca_verbs_uar_type_to_mlx5_uar_type(doca_verbs_uar_allocation_type uar_type,
                                                          uint32_t &mlx5_uar_type) noexcept {
    switch (uar_type) {
        case DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME:
            mlx5_uar_type = MLX5DV_UAR_ALLOC_TYPE_BF;
            break;
        case DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE:
            mlx5_uar_type = MLX5DV_UAR_ALLOC_TYPE_NC;
            break;
        case DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED:
#if DOCA_GPUNETIO_HAVE_DEDICATED_NC_UAR == 1
            mlx5_uar_type = MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED;
            break;
#else
            DOCA_LOG(LOG_ERR, "DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED is not supported");
            return DOCA_ERROR_NOT_SUPPORTED;
#endif
        default:
            DOCA_LOG(LOG_ERR, "Can't convert invalid UAR type=%d", mlx5_uar_type);
            return DOCA_ERROR_INVALID_VALUE;
    }

    return DOCA_SUCCESS;
}

} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_uar Member Functions
 *********************************************************************************************************************/

doca_verbs_uar_open::doca_verbs_uar_open(struct ibv_context *context,
                                         enum doca_verbs_uar_allocation_type allocation_type)
    : m_ibv_ctx(context), m_allocation_type(allocation_type) {
    try {
        create();
    } catch (...) {
        (void)destroy();
        DOCA_LOG(LOG_ERR, "Failed to create UAR");
        throw;
    }
}

doca_verbs_uar_open::~doca_verbs_uar_open() { static_cast<void>(destroy()); }

static const off64_t DOCA_VERBS_UAR_DBR_LESS_DB_OFFSET = 0x600LLU;

void doca_verbs_uar_open::create() {
    uint32_t mlx5_uar_type{};
    auto status = convert_doca_verbs_uar_type_to_mlx5_uar_type(m_allocation_type, mlx5_uar_type);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to convert UAR");
        throw DOCA_ERROR_DRIVER;
    }

    auto uar_status = doca_verbs_wrapper_mlx5dv_devx_alloc_uar(m_ibv_ctx, mlx5_uar_type, &m_uar);
    if (uar_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to alloc UAR");
        throw uar_status;
    }

    m_uar_id = m_uar->page_id;
    m_reg_addr = m_uar->reg_addr;
    m_base_addr = m_uar->base_addr;
    m_dbr_less_addr = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(m_uar->base_addr) +
                                               DOCA_VERBS_UAR_DBR_LESS_DB_OFFSET);
}

doca_error_t doca_verbs_uar_open::destroy() noexcept {
    if (m_uar) {
        auto free_uar_status = doca_verbs_wrapper_mlx5dv_devx_free_uar(m_uar);
        if (free_uar_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to free UAR");
            return free_uar_status;
        }
        m_uar = nullptr;
    }

    return DOCA_SUCCESS;
}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_uar_create(doca_dev_t *net_dev,
                                   enum doca_verbs_uar_allocation_type allocation_type,
                                   doca_verbs_uar_t **uar) {
    doca_verbs_uar_t *uar_ = nullptr;

    if (uar == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create uar: parameter uar=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create uar: parameter net_dev=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    uar_ = (doca_verbs_uar_t *)calloc(1, sizeof(doca_verbs_uar_t));
    if (uar_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_verbs_uar_t", __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Valid for both SDK and open */
    uar_->allocation_type = allocation_type;

    /* Try with DOCA SDK first */
    auto err = doca_verbs_sdk_wrapper_uar_create(net_dev, allocation_type, &(uar_->sdk));
    if (err == DOCA_SDK_WRAPPER_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Use DOCA Verbs UAR SDK", __func__);
        uar_->type = DOCA_VERBS_SDK_LIB_TYPE_SDK;
        (*uar) = uar_;
        return DOCA_SUCCESS;
    } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
        DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA Verbs UAR open", __func__);

    uar_->type = DOCA_VERBS_SDK_LIB_TYPE_OPEN;

    try {
        uar_->open = new doca_verbs_uar_open(net_dev->open->get_ctx(), allocation_type);
        DOCA_LOG(LOG_INFO, "doca_verbs_uar_open=%p was created", uar_);
        (*uar) = uar_;
        return DOCA_SUCCESS;
    } catch (...) {
        DOCA_LOG(LOG_ERR, "doca_verbs_uar_open allocation failed");
        goto exit_error;
    }

exit_error:
    if (uar_ != nullptr) {
        if (uar_->open) delete uar_->open;
        free(uar_);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_verbs_uar_destroy(doca_verbs_uar_t *uar) {
    doca_error_t status = DOCA_SUCCESS;

    if (uar == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to destroy uar: parameter uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (uar->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_uar_destroy(uar->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            uar->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (uar->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

    status = uar->open->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Failed to destroy uar.");
        goto exit;
    }

exit:
    if (uar->open) delete uar->open;
    memset(uar, 0, sizeof(doca_verbs_uar_t));
    free(uar);

    return status;
}

doca_error_t doca_verbs_uar_id_get(const doca_verbs_uar_t *uar, uint32_t *uar_id) {
    if (uar == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get uar id: parameter uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (uar_id == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get uar id: parameter uar_id is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (uar->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_uar_get_id(uar->sdk, uar_id);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (uar->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs UAR open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *uar_id = uar->open->get_uar_id();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_uar_reg_addr_get(const doca_verbs_uar_t *uar, void **reg_addr) {
    if (uar == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get uar address: parameter uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (reg_addr == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get uar address: parameter reg_addr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (uar->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_uar_reg_addr_get(uar->sdk, reg_addr);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (uar->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs UAR open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *reg_addr = uar->open->get_reg_addr();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_uar_dbr_less_addr_get(const doca_verbs_uar_t *uar, void **reg_addr) {
    if (uar == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get uar address: parameter uar is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (reg_addr == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get uar address: parameter reg_addr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (uar->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_uar_reg_addr_get(uar->sdk, reg_addr);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (uar->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs UAR open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *reg_addr = uar->open->get_dbr_less_addr();

    return DOCA_SUCCESS;
}
