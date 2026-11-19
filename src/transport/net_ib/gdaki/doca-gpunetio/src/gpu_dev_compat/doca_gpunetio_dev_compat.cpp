/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "doca_gpunetio_dev_compat.hpp"
#include "../doca_gpunetio_log.hpp"
#include "host/doca_gpunetio.h"

/* First device code version that reads cq_type and drives free flow. */
static constexpr uint32_t sc_gpu_verbs_qp_dev_cq_type_min_version =
    (4u << DOCA_GPUNETIO_VERSION_MAJOR_SHIFT);

static constexpr uint32_t sc_gpu_verbs_qp_dev_pre_cq_type_nic_handler_flags =
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY | DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_GPU_SM |
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_BF | DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_NO_DBR;

static constexpr uint32_t sc_gpu_verbs_nic_handler_flags =
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY | DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_GPU_SM |
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_BF | DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_NO_DBR |
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_FREE_FLOW;

doca_error_t doca_internal_gpu_verbs_qp_dev_select(uint32_t device_code_version, uint32_t *layout,
                                                   size_t *qp_dev_size) {
    if (layout == nullptr || qp_dev_size == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (doca_gpu_verbs_check_device_code_compatibility(device_code_version) != DOCA_SUCCESS) {
        DOCA_LOG(LOG_ERR, "Device code version %u.%u.%u is not supported by this library",
                 (device_code_version >> DOCA_GPUNETIO_VERSION_MAJOR_SHIFT) & 0xff,
                 (device_code_version >> DOCA_GPUNETIO_VERSION_MINOR_SHIFT) & 0xff,
                 (device_code_version >> DOCA_GPUNETIO_VERSION_PATCH_SHIFT) & 0xff);
        return DOCA_ERROR_NOT_SUPPORTED;
    }

    *layout = sc_gpu_verbs_qp_dev_layout_v1;
    *qp_dev_size = doca_internal_gpu_verbs_qp_dev_size(*layout);
    return DOCA_SUCCESS;
}

/* Accepts only what device code of that version can drive, so features added later are refused
 * without editing this function. */
doca_error_t doca_internal_gpu_verbs_qp_dev_check_version(uint32_t device_code_version,
                                                          const struct doca_gpu_dev_verbs_qp *src) {
    if (src == nullptr) return DOCA_ERROR_INVALID_VALUE;
    if (device_code_version >= sc_gpu_verbs_qp_dev_cq_type_min_version) return DOCA_SUCCESS;

    /* This device code ignores cq_type and polls every CQ as plain 64B CQEs. */
    if (src->cq_sq.cq_type != DOCA_GPUNETIO_VERBS_CQ_64B) {
        DOCA_LOG(LOG_ERR, "CQ type %d is not supported by device code before 4.0.0",
                 (int)src->cq_sq.cq_type);
        return DOCA_ERROR_NOT_SUPPORTED;
    }
    if ((src->nic_handler & ~sc_gpu_verbs_qp_dev_pre_cq_type_nic_handler_flags) != 0) {
        DOCA_LOG(LOG_ERR, "NIC handler 0x%x is not supported by device code before 4.0.0",
                 (unsigned int)src->nic_handler);
        return DOCA_ERROR_NOT_SUPPORTED;
    }
    return DOCA_SUCCESS;
}

size_t doca_internal_gpu_verbs_qp_dev_size(uint32_t layout) {
    switch (layout) {
        case sc_gpu_verbs_qp_dev_layout_v1:
            return sizeof(struct doca_gpu_dev_verbs_qp_v1);
        default:
            return 0;
    }
}

doca_error_t doca_internal_gpu_verbs_qp_dev_check(uint32_t layout,
                                                  const struct doca_gpu_dev_verbs_qp *src) {
    enum doca_gpu_dev_verbs_cq_type cq_type;

    if (src == nullptr) return DOCA_ERROR_INVALID_VALUE;

    if (doca_internal_gpu_verbs_qp_dev_size(layout) == 0) {
        DOCA_LOG(LOG_ERR, "Invalid QP device layout %u", layout);
        return DOCA_ERROR_INVALID_VALUE;
    }

    /* Device code polls only resolved CQ types; CQ_UNKNOWN must be resolved at QP creation. */
    cq_type = src->cq_sq.cq_type;
    if ((src->nic_handler & ~sc_gpu_verbs_nic_handler_flags) != 0 ||
        (cq_type != DOCA_GPUNETIO_VERBS_CQ_64B && cq_type != DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED &&
         cq_type != DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) ||
        (uint32_t)src->mem_type > DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU ||
        (uint32_t)src->cq_sq.mem_type > DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU) {
        DOCA_LOG(LOG_ERR, "Unknown QP device state: nic_handler 0x%x, cq_type 0x%x, mem_type %u/%u",
                 (unsigned int)src->nic_handler, (unsigned int)cq_type, (unsigned int)src->mem_type,
                 (unsigned int)src->cq_sq.mem_type);
        return DOCA_ERROR_INVALID_VALUE;
    }

    switch (layout) {
        case sc_gpu_verbs_qp_dev_layout_v1:
            return doca_internal_gpu_verbs_qp_dev_v1_check(src);
        default:
            return DOCA_ERROR_INVALID_VALUE;
    }
}

doca_error_t doca_internal_gpu_verbs_qp_dev_fill(uint32_t layout, void *qp_array, unsigned int idx,
                                                 const struct doca_gpu_dev_verbs_qp *src) {
    doca_error_t status = doca_internal_gpu_verbs_qp_dev_check(layout, src);

    if (status != DOCA_SUCCESS) return status;
    if (qp_array == nullptr) return DOCA_ERROR_INVALID_VALUE;

    switch (layout) {
        case sc_gpu_verbs_qp_dev_layout_v1:
            doca_internal_gpu_verbs_qp_dev_v1_fill(qp_array, idx, src);
            break;
    }
    return DOCA_SUCCESS;
}

doca_error_t doca_internal_gpu_verbs_qp_dev_reset(uint32_t layout, void *qp_dev,
                                                  struct doca_gpu_dev_verbs_qp *qp_cpu) {
    doca_error_t status = doca_internal_gpu_verbs_qp_dev_check(layout, qp_cpu);
    if (status != DOCA_SUCCESS) return status;
    if (qp_dev == nullptr) return DOCA_ERROR_INVALID_VALUE;
    switch (layout) {
        case sc_gpu_verbs_qp_dev_layout_v1:
            doca_internal_gpu_verbs_qp_dev_v1_reset(qp_dev, qp_cpu);
            break;
    }
    return DOCA_SUCCESS;
}

uint64_t *doca_internal_gpu_verbs_qp_dev_sq_wqe_pi(uint32_t layout, void *qp_dev) {
    switch (layout) {
        case sc_gpu_verbs_qp_dev_layout_v1:
            return doca_internal_gpu_verbs_qp_dev_v1_sq_wqe_pi(qp_dev);
        default:
            return nullptr;
    }
}

uint64_t *doca_internal_gpu_verbs_qp_dev_cqe_ci(uint32_t layout, void *qp_dev) {
    switch (layout) {
        case sc_gpu_verbs_qp_dev_layout_v1:
            return doca_internal_gpu_verbs_qp_dev_v1_cqe_ci(qp_dev);
        default:
            return nullptr;
    }
}
