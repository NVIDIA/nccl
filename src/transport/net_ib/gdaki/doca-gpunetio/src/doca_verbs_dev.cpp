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
#include <string.h>
#include <mutex>
#include <time.h>
#include <stdexcept>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_gpunetio_config.h"
#include "doca_verbs_net_wrapper.h"
#include "doca_internal.hpp"
#include "doca_verbs_dev.hpp"
#include "doca_verbs_dev_sdk_wrapper.h"

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_dev Member Functions
 *********************************************************************************************************************/

doca_dev_open::doca_dev_open(struct ibv_context *verbs_ctx, struct ibv_pd *verbs_pd)
    : m_ibv_ctx(verbs_ctx), m_ibv_pd(verbs_pd) {
    if (m_ibv_ctx == nullptr || m_ibv_pd == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create doca_dev");
        throw std::invalid_argument("bad arg");
    }
}

doca_dev_open::~doca_dev_open() {}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_dev_open(struct ibv_pd *verbs_pd, doca_dev_t **net_dev) {
    doca_dev_t *net_dev_ = nullptr;

    net_dev_ = (doca_dev_t *)calloc(1, sizeof(doca_dev_t));
    if (net_dev_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_dev_t", __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Try with DOCA SDK first */
    auto err = doca_verbs_sdk_wrapper_dev_open_from_pd(verbs_pd, net_dev_);
    if (err == DOCA_SDK_WRAPPER_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Use DOCA Verbs Dev SDK", __func__);
        net_dev_->type = DOCA_VERBS_SDK_LIB_TYPE_SDK;
        (*net_dev) = net_dev_;
        return DOCA_SUCCESS;
    } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
        DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA Verbs Dev open", __func__);

    if (verbs_pd == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create dev: parameter verbs_pd=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    net_dev_->type = DOCA_VERBS_SDK_LIB_TYPE_OPEN;

    try {
        net_dev_->open = new doca_dev_open(verbs_pd->context, verbs_pd);
        (*net_dev) = net_dev_;
        DOCA_LOG(LOG_INFO, "doca_verbs_dev_open=%p was created", net_dev_);
        return DOCA_SUCCESS;
    } catch (...) {
        DOCA_LOG(LOG_ERR, "doca_verbs_dev_open allocation failed");
        goto exit_error;
    }

exit_error:
    if (net_dev_ != nullptr) {
        if (net_dev_->open) free(net_dev_->open);
        free(net_dev_);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_verbs_dev_close(doca_dev_t *net_dev) {
    doca_error_t status = DOCA_SUCCESS;

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to destroy net_dev: parameter net_dev is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (net_dev->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_dev_close(net_dev);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            net_dev->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (net_dev->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

exit:
    if (net_dev->open) delete net_dev->open;
    memset(net_dev, 0, sizeof(doca_dev_t));
    free(net_dev);

    return status;
}
