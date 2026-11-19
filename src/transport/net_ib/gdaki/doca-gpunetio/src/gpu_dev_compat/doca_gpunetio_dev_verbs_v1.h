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

#pragma once

#include <stdint.h>
#include <linux/types.h>

#include "common/doca_gpunetio_verbs_dev.h"
#include "host/doca_error.h"

/* Device code from 2.0.0. Frozen: never edit, add a new version instead.
 * Byte 37 is cq_type from 4.0.0; older device code reserves it and ignores the value. */
struct doca_gpu_dev_verbs_cq_v1 {
    uint8_t *cqe_daddr;
    uint32_t cq_num;
    uint32_t cqe_num;
    __be32 *dbrec;
    uint64_t cqe_ci;
    uint32_t cqe_mask;
    uint8_t cqe_size;
    uint8_t cq_type;
    uint8_t reserved1[2];
    uint64_t cqe_rsvd;
    uint32_t mem_type;
    uint8_t reserved2[12];
} __attribute__((__aligned__(8))) __attribute__((__packed__));

struct doca_gpu_dev_verbs_qp_v1 {
    uint64_t sq_rsvd_index;
    uint64_t sq_ready_index;
    uint64_t sq_wqe_pi;
    uint32_t sq_num;
    uint32_t sq_num_shift8;
    uint32_t sq_num_shift8_be;
    uint32_t sq_num_shift8_be_1ds;
    uint32_t sq_num_shift8_be_2ds;
    uint32_t sq_num_shift8_be_3ds;
    uint32_t sq_num_shift8_be_4ds;
    uint32_t sq_num_shift8_be_5ds;
    int sq_lock;
    uint16_t sq_wqe_num;
    uint16_t sq_wqe_mask;
    uint8_t *sq_wqe_daddr;
    __be32 *sq_dbrec;
    uint64_t *sq_db;
    uint8_t reserved1[8];
    uint8_t reserved2[64];
    struct doca_gpu_dev_verbs_cq_v1 cq_sq;
    uint8_t reserved3[64];
    uint32_t nic_handler;
    uint32_t mem_type;
} __attribute__((__aligned__(8))) __attribute__((__packed__));

doca_error_t doca_internal_gpu_verbs_qp_dev_v1_check(const struct doca_gpu_dev_verbs_qp *src);
void doca_internal_gpu_verbs_qp_dev_v1_fill(void *qp_array, unsigned int idx,
                                            const struct doca_gpu_dev_verbs_qp *src);
void doca_internal_gpu_verbs_qp_dev_v1_reset(void *qp_dev, struct doca_gpu_dev_verbs_qp *qp_cpu);
uint64_t *doca_internal_gpu_verbs_qp_dev_v1_sq_wqe_pi(void *qp_dev);
uint64_t *doca_internal_gpu_verbs_qp_dev_v1_cqe_ci(void *qp_dev);
