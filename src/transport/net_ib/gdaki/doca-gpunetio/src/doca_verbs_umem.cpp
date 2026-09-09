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
#include "doca_verbs_umem.hpp"
#include "doca_verbs_umem_sdk_wrapper.h"

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_umem Member Functions
 *********************************************************************************************************************/

doca_verbs_umem_open::doca_verbs_umem_open(struct ibv_context *ibv_ctx, void *address, size_t size,
                                           uint32_t access_flags, int dmabuf_fd,
                                           size_t dmabuf_offset)
    : m_ibv_ctx(ibv_ctx),
      m_address(address),
      m_size(size),
      m_access_flags(access_flags),
      m_dmabuf_fd(dmabuf_fd),
      m_dmabuf_offset(dmabuf_offset) {
    try {
        create();
    } catch (...) {
        (void)destroy();
        DOCA_LOG(LOG_ERR, "Failed to create UMEM");
        throw;
    }
}

doca_verbs_umem_open::~doca_verbs_umem_open() { static_cast<void>(destroy()); }

void doca_verbs_umem_open::create() {
    doca_error_t umem_status;

    struct mlx5dv_devx_umem_in umem_in {
        0,
    };

    umem_in.access = m_access_flags;
    umem_in.pgsz_bitmap = sysconf(_SC_PAGESIZE);

#if DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF != 1
    if (m_dmabuf_fd != (int)DOCA_VERBS_DMABUF_INVALID_FD) {
        DOCA_LOG(LOG_ERR, "Creating UMEM with DMABUF is not supported");
        throw DOCA_ERROR_NOT_SUPPORTED;
    }
#endif

#if DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF == 1
    if (m_dmabuf_fd != (int)DOCA_VERBS_DMABUF_INVALID_FD) {
        umem_in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
        umem_in.dmabuf_fd = m_dmabuf_fd;
        /* umem_in.addr is interpreted as the starting offset of the dmabuf */
        umem_in.addr = reinterpret_cast<void *>(m_dmabuf_offset);
    } else
#endif
    {
        umem_in.addr = m_address;
    }
    umem_in.size = m_size;

    umem_status = doca_verbs_wrapper_mlx5dv_devx_umem_reg_ex(m_ibv_ctx, &umem_in, &m_umem);
    if (umem_status != DOCA_SUCCESS) {
        if (m_dmabuf_fd != (int)DOCA_VERBS_DMABUF_INVALID_FD) {
            DOCA_LOG(LOG_ERR,
                     "Failed to create UMEM with DMABUF, m_address %p m_size %zd m_access_flags %x "
                     "m_dmabuf_fd %d "
                     "m_dmabuf_offset %zd err %d",
                     m_address, m_size, m_access_flags, m_dmabuf_fd, m_dmabuf_offset, errno);
        } else {
            DOCA_LOG(LOG_ERR,
                     "Failed to create UMEM, m_address %p m_size %zd m_access_flags %x err %d",
                     m_address, m_size, m_access_flags, errno);
        }
        throw umem_status;
    }

    m_umem_id = m_umem->umem_id;
}

doca_error_t doca_verbs_umem_open::destroy() noexcept {
    if (m_umem) {
        auto dereg_status = doca_verbs_wrapper_mlx5dv_devx_umem_dereg(m_umem);
        if (dereg_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy UMEM object");
            return dereg_status;
        }
        m_umem = nullptr;
    }

    return DOCA_SUCCESS;
}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_umem_create(doca_dev_t *net_dev, doca_gpu_t *gpu, void *address,
                                    size_t size, uint32_t access_flags, int dmabuf_fd,
                                    size_t dmabuf_offset, doca_verbs_umem_t **umem) {
    doca_verbs_umem_t *umem_ = nullptr;

    if (umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter umem=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter net_dev=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    umem_ = (doca_verbs_umem_t *)calloc(1, sizeof(doca_verbs_umem_t));
    if (umem_ == nullptr) {
        DOCA_LOG(LOG_ERR, "error in %s: failed to allocate memory for doca_verbs_umem_t", __func__);
        return DOCA_ERROR_NO_MEMORY;
    }

    /* Try with DOCA SDK first */
    auto err = doca_verbs_sdk_wrapper_umem_create(net_dev, gpu, address, size, access_flags,
                                                  dmabuf_fd, dmabuf_offset, &(umem_->sdk));
    if (err == DOCA_SDK_WRAPPER_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Use DOCA Verbs UMEM SDK", __func__);
        umem_->type = DOCA_VERBS_SDK_LIB_TYPE_SDK;
        (*umem) = umem_;
        return DOCA_SUCCESS;
    } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
        DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
        goto exit_error;
    }

    /* In case of DOCA_SDK_WRAPPER_NOT_FOUND or DOCA_SDK_WRAPPER_NOT_SUPPORTED, just rely on open
     * version */
    DOCA_LOG(LOG_INFO, "Use DOCA Verbs UMEM open", __func__);

    if (address == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter address=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (size == 0) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter size=0");
        return DOCA_ERROR_INVALID_VALUE;
    }

    umem_->type = DOCA_VERBS_SDK_LIB_TYPE_OPEN;

    try {
        umem_->open = new doca_verbs_umem_open(net_dev->open->get_ctx(), address, size,
                                               access_flags, dmabuf_fd, dmabuf_offset);
        DOCA_LOG(LOG_INFO, "doca_verbs_umem_open=%p was created", umem_);
        (*umem) = umem_;
        return DOCA_SUCCESS;
    } catch (...) {
        DOCA_LOG(LOG_ERR, "doca_verbs_umem_open allocation failed");
        goto exit_error;
    }

exit_error:
    if (umem_ != nullptr) {
        if (umem_->open) delete umem_->open;
        free(umem_);
    }

    return DOCA_ERROR_INITIALIZATION;
}

doca_error_t doca_verbs_umem_destroy(doca_verbs_umem_t *umem) {
    doca_error_t status = DOCA_SUCCESS;

    if (umem == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to destroy umem: parameter umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (umem->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_umem_destroy(umem->sdk);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            umem->sdk = nullptr;
            goto exit;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            status = DOCA_ERROR_UNEXPECTED;
            goto exit;
        }
    }

    if (umem->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid input parameters.");
        status = DOCA_ERROR_INVALID_VALUE;
        goto exit;
    }

    status = umem->open->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Failed to destroy umem.");
        goto exit;
    }

exit:
    if (umem->open) delete umem->open;
    memset(umem, 0, sizeof(doca_verbs_umem_t));
    free(umem);

    return status;
}

doca_error_t doca_verbs_umem_get_id(const doca_verbs_umem_t *umem, uint32_t *umem_id) {
    if (umem == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem id: parameter umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_id == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem id: parameter umem_id is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (umem->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_umem_get_id(umem->sdk, umem_id);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (umem->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs UMEM open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *umem_id = umem->open->get_umem_id();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_umem_get_size(const doca_verbs_umem_t *umem, size_t *umem_size) {
    if (umem == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem size: parameter umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_size == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem size: parameter umem_size is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (umem->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_umem_get_size(umem->sdk, umem_size);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (umem->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs UMEM open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *umem_size = umem->open->get_umem_size();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_umem_get_address(const doca_verbs_umem_t *umem, void **umem_address) {
    if (umem == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem address: parameter umem is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_address == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem address: parameter umem_address is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    if (umem->type == DOCA_VERBS_SDK_LIB_TYPE_SDK) {
        auto err = doca_verbs_sdk_wrapper_umem_get_address(umem->sdk, umem_address);
        if (err == DOCA_SDK_WRAPPER_SUCCESS) {
            return DOCA_SUCCESS;
        } else if (err == DOCA_SDK_WRAPPER_API_ERROR) {
            DOCA_LOG(LOG_INFO, "DOCA SDK function returned an error", __func__);
            return DOCA_ERROR_UNEXPECTED;
        }
    }

    if (umem->open == nullptr) {
        DOCA_LOG(LOG_ERR, "Invalid DOCA Verbs UMEM open instance provided.");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *umem_address = umem->open->get_umem_address();

    return DOCA_SUCCESS;
}
