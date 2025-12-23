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
#include "doca_verbs_umem.hpp"

/*********************************************************************************************************************
 * Helper functions
 *********************************************************************************************************************/

namespace {} /* namespace */

/**********************************************************************************************************************
 * doca_verbs_umem Member Functions
 *********************************************************************************************************************/

doca_verbs_umem::doca_verbs_umem(struct ibv_context *ibv_ctx, void *address, size_t size,
                                 uint32_t access_flags, int dmabuf_fd, size_t dmabuf_offset)
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

doca_verbs_umem::~doca_verbs_umem() { static_cast<void>(destroy()); }

void doca_verbs_umem::create() {
    struct mlx5dv_devx_umem_in umem_in {};

    umem_in.addr = m_address;
    umem_in.size = m_size;
    umem_in.access = m_access_flags;
    umem_in.pgsz_bitmap = sysconf(_SC_PAGESIZE);
    umem_in.comp_mask = 0;

#if DOCA_GPUNETIO_HAVE_MLX5DV_UMEM_DMABUF == 1
    /* check if dmabuf file descriptor was set to determine mask */
    if (m_dmabuf_fd != (int)DOCA_VERBS_DMABUF_INVALID_FD) {
        umem_in.comp_mask = MLX5DV_UMEM_MASK_DMABUF;
        umem_in.dmabuf_fd = m_dmabuf_fd;
        /* umem_in.addr is interpreted as the starting offset of the dmabuf */
        umem_in.addr = reinterpret_cast<void *>(m_dmabuf_offset);
    }
#endif

    auto umem_status = doca_verbs_wrapper_mlx5dv_devx_umem_reg_ex(m_ibv_ctx, &umem_in, &m_umem_obj);
    if (umem_status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR,
                 "Failed to create UMEM, m_address %p m_size %zd m_access_flags %x m_dmabuf_fd %d "
                 "m_dmabuf_offset %zd err %d",
                 m_address, m_size, m_access_flags, m_dmabuf_fd, m_dmabuf_offset, errno);
        throw umem_status;
    }

    m_umem_id = m_umem_obj->umem_id;
}

doca_error_t doca_verbs_umem::destroy() noexcept {
    if (m_umem_obj) {
        auto dereg_status = doca_verbs_wrapper_mlx5dv_devx_umem_dereg(m_umem_obj);
        if (dereg_status != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "Failed to destroy UMEM object");
            return dereg_status;
        }
        m_umem_obj = nullptr;
    }

    return DOCA_SUCCESS;
}

/**********************************************************************************************************************
 * Public API functions
 *********************************************************************************************************************/

doca_error_t doca_verbs_umem_create(struct ibv_context *context, void *address, size_t size,
                                    uint32_t access_flags, int dmabuf_id, size_t dmabuf_offset,
                                    struct doca_verbs_umem **umem_obj) {
    if (context == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter context=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (address == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter address=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (size == 0) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter size=0");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_obj == nullptr) {
        DOCA_LOG(LOG_ERR, "Failed to create umem: parameter umem_obj=NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    try {
        *umem_obj =
            new doca_verbs_umem(context, address, size, access_flags, dmabuf_id, dmabuf_offset);
        DOCA_LOG(LOG_INFO, "doca_verbs_umem=%p was created", *umem_obj);
        return DOCA_SUCCESS;
    } catch (doca_error_t err) {
        return err;
    }
}

doca_error_t doca_verbs_umem_destroy(struct doca_verbs_umem *umem_obj) {
    if (umem_obj == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to destroy umem: parameter umem_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    auto status = umem_obj->destroy();
    if (status != DOCA_SUCCESS) {
        DOCA_LOG(LOG_INFO, "Failed to destroy umem.");
        return status;
    }

    delete (umem_obj);
    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_umem_get_id(const struct doca_verbs_umem *umem_obj, uint32_t *umem_id) {
    if (umem_obj == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem id: parameter umem_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_id == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem id: parameter umem_id is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *umem_id = umem_obj->get_umem_id();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_umem_get_size(const struct doca_verbs_umem *umem_obj, size_t *umem_size) {
    if (umem_obj == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem size: parameter umem_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_size == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem size: parameter umem_size is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *umem_size = umem_obj->get_umem_size();

    return DOCA_SUCCESS;
}

doca_error_t doca_verbs_umem_get_address(const struct doca_verbs_umem *umem_obj,
                                         void **umem_address) {
    if (umem_obj == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem address: parameter umem_obj is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }
    if (umem_address == nullptr) {
        DOCA_LOG(LOG_INFO, "Failed to get umem address: parameter umem_address is NULL");
        return DOCA_ERROR_INVALID_VALUE;
    }

    *umem_address = umem_obj->get_umem_address();

    return DOCA_SUCCESS;
}
