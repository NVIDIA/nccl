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

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_gpunetio_config.h"
#include "doca_verbs_net_wrapper.h"
#include "doca_internal.hpp"
#include "doca_verbs_device_attr.hpp"
#include "doca_verbs_uar.hpp"

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

doca_verbs_uar::doca_verbs_uar(struct ibv_context *context,
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

doca_verbs_uar::~doca_verbs_uar() { static_cast<void>(destroy()); }

void doca_verbs_uar::create() {
    uint32_t mlx5_uar_type{};
    auto status = convert_doca_verbs_uar_type_to_mlx5_uar_type(m_allocation_type, mlx5_uar_type);
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to convert UAR");
        throw DOCA_ERROR_DRIVER;
    }

    auto uar_status =
        doca_verbs_wrapper_mlx5dv_devx_alloc_uar(m_ibv_ctx, mlx5_uar_type, &m_uar_obj);
    if (uar_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to alloc UAR");
        throw uar_status;
    }

    m_uar_id = m_uar_obj->page_id;
    m_reg_addr = m_uar_obj->reg_addr;
}

doca_error_t doca_verbs_uar::destroy() noexcept {
    if (m_uar_obj) {
        auto free_uar_status = doca_verbs_wrapper_mlx5dv_devx_free_uar(m_uar_obj);
        if (free_uar_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to free UAR");
            return free_uar_status;
        }
        m_uar_obj = nullptr;
    }

    return DOCA_SUCCESS;
}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_uar_create(struct ibv_context *context,
                                   enum doca_verbs_uar_allocation_type allocation_type,
                                   struct doca_verbs_uar **uar_obj) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create uar: parameter context=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (uar_obj == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create uar: parameter uar_obj=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    try {
        *uar_obj = new doca_verbs_uar(context, allocation_type);
        DOCA_LOG(LOG_INFO, "doca_verbs_uar=%p was created", *uar_obj);
        return DOCA_SUCCESS;
    } catch (doca_error_t err) {
        return err;
    }
}

doca_error_t doca_verbs_uar_destroy(struct doca_verbs_uar *uar_obj) {
    if (uar_obj == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to destroy uar: parameter uar_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    auto status = uar_obj->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Failed to destroy uar.");
        return status;
    }

    delete (uar_obj);
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_uar_id_get(const struct doca_verbs_uar *uar_obj, uint32_t *uar_id) {
    if (uar_obj == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get uar id: parameter uar_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (uar_id == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get uar id: parameter uar_id is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *uar_id = uar_obj->get_uar_id();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_uar_reg_addr_get(const struct doca_verbs_uar *uar_obj, void **reg_addr) {
    if (uar_obj == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get uar reg_addr: parameter uar_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (reg_addr == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to get uar reg_addr: parameter reg_addr is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *reg_addr = uar_obj->get_reg_addr();

    return DOCA_SUCCESS;
}
