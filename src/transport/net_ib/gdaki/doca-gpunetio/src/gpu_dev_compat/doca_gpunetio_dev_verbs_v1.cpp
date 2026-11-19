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

#include <string.h>

#include "doca_gpunetio_dev_verbs_v1.h"
#include "../doca_gpunetio_log.hpp"

static constexpr uint32_t sc_gpu_verbs_qp_dev_v1_nic_handler_flags =
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_CPU_PROXY | DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_GPU_SM |
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_BF | DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_NO_DBR |
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_FLAG_FREE_FLOW;

/* Accepts only what v1 device code can express, so values added later are refused without
 * editing this file. */
doca_error_t doca_internal_gpu_verbs_qp_dev_v1_check(const struct doca_gpu_dev_verbs_qp *src) {
    enum doca_gpu_dev_verbs_cq_type cq_type = src->cq_sq.cq_type;

    if (cq_type != DOCA_GPUNETIO_VERBS_CQ_64B && cq_type != DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED &&
        cq_type != DOCA_GPUNETIO_VERBS_CQ_64B_COLLAPSED_HOST) {
        DOCA_LOG(LOG_ERR, "CQ type %d requires newer device code", (int)cq_type);
        return DOCA_ERROR_NOT_SUPPORTED;
    }
    if ((src->nic_handler & ~sc_gpu_verbs_qp_dev_v1_nic_handler_flags) != 0) {
        DOCA_LOG(LOG_ERR, "NIC handler 0x%x requires newer device code",
                 (unsigned int)src->nic_handler);
        return DOCA_ERROR_NOT_SUPPORTED;
    }
    if ((uint32_t)src->mem_type > DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU ||
        (uint32_t)src->cq_sq.mem_type > DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU) {
        DOCA_LOG(LOG_ERR, "Memory type %u/%u requires newer device code",
                 (unsigned int)src->mem_type, (unsigned int)src->cq_sq.mem_type);
        return DOCA_ERROR_NOT_SUPPORTED;
    }
    return DOCA_SUCCESS;
}

void doca_internal_gpu_verbs_qp_dev_v1_fill(void *qp_array, unsigned int idx,
                                            const struct doca_gpu_dev_verbs_qp *src) {
    struct doca_gpu_dev_verbs_qp_v1 *qp = (struct doca_gpu_dev_verbs_qp_v1 *)qp_array + idx;

    memset(qp, 0, sizeof(*qp));
    qp->sq_rsvd_index = src->sq_rsvd_index;
    qp->sq_ready_index = src->sq_ready_index;
    qp->sq_wqe_pi = src->sq_wqe_pi;
    qp->sq_num = src->sq_num;
    qp->sq_num_shift8 = src->sq_num_shift8;
    qp->sq_num_shift8_be = src->sq_num_shift8_be;
    qp->sq_num_shift8_be_1ds = src->sq_num_shift8_be_1ds;
    qp->sq_num_shift8_be_2ds = src->sq_num_shift8_be_2ds;
    qp->sq_num_shift8_be_3ds = src->sq_num_shift8_be_3ds;
    qp->sq_num_shift8_be_4ds = src->sq_num_shift8_be_4ds;
    qp->sq_num_shift8_be_5ds = src->sq_num_shift8_be_5ds;
    qp->sq_lock = src->sq_lock;
    qp->sq_wqe_num = src->sq_wqe_num;
    qp->sq_wqe_mask = src->sq_wqe_mask;
    qp->sq_wqe_daddr = src->sq_wqe_daddr;
    qp->sq_dbrec = src->sq_dbrec;
    qp->sq_db = src->sq_db;

    qp->cq_sq.cqe_daddr = src->cq_sq.cqe_daddr;
    qp->cq_sq.cq_num = src->cq_sq.cq_num;
    qp->cq_sq.cqe_num = src->cq_sq.cqe_num;
    qp->cq_sq.dbrec = src->cq_sq.dbrec;
    qp->cq_sq.cqe_ci = src->cq_sq.cqe_ci;
    qp->cq_sq.cqe_mask = src->cq_sq.cqe_mask;
    qp->cq_sq.cqe_size = src->cq_sq.cqe_size;
    qp->cq_sq.cq_type = src->cq_sq.cq_type;
    qp->cq_sq.cqe_rsvd = src->cq_sq.cqe_rsvd;
    qp->cq_sq.mem_type = src->cq_sq.mem_type;

    qp->nic_handler = src->nic_handler;
    qp->mem_type = src->mem_type;
}

void doca_internal_gpu_verbs_qp_dev_v1_reset(void *qp_dev, struct doca_gpu_dev_verbs_qp *qp_cpu) {
    qp_cpu->cq_sq.cqe_rsvd = ((struct doca_gpu_dev_verbs_qp_v1 *)qp_dev)->sq_rsvd_index;
    doca_internal_gpu_verbs_qp_dev_v1_fill(qp_dev, 0, qp_cpu);
}

uint64_t *doca_internal_gpu_verbs_qp_dev_v1_sq_wqe_pi(void *qp_dev) {
    return &((struct doca_gpu_dev_verbs_qp_v1 *)qp_dev)->sq_wqe_pi;
}

uint64_t *doca_internal_gpu_verbs_qp_dev_v1_cqe_ci(void *qp_dev) {
    return &((struct doca_gpu_dev_verbs_qp_v1 *)qp_dev)->cq_sq.cqe_ci;
}
