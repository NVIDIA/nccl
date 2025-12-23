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

/**
 * @file doca_gpunetio_high_level.h
 * @brief A header file for the doca_gpunetio High-level APIs
 */

#ifndef DOCA_GPUNETIO_HIGH_LEVEL_H
#define DOCA_GPUNETIO_HIGH_LEVEL_H

#include "doca_gpunetio.h"

#ifdef __cplusplus
extern "C" {
#endif

enum doca_gpu_verbs_mem_reg_type {
    DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_DEFAULT =
        0,  ///< Automatically select the most appropriate method
    DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_DMABUF = 1,   ///< Use CUDA DMABUF to register memory
    DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_CUDA_PEERMEM = 2,  ///< Use CUDA PeerMem to register memory
    DOCA_GPUNETIO_VERBS_MEM_REG_TYPE_MAX,               ///< Sentinel value
};

struct doca_gpu_verbs_qp_init_attr_hl {
    struct doca_gpu *gpu_dev;
    struct ibv_pd *ibpd;
    uint16_t sq_nwqe;
    enum doca_gpu_dev_verbs_nic_handler nic_handler;
    enum doca_gpu_verbs_mem_reg_type mreg_type;
};

struct doca_gpu_verbs_qp_hl {
    struct doca_gpu *gpu_dev; /* DOCA GPU device to use */

    // CQ
    struct doca_verbs_cq *cq_sq;
    void *cq_sq_umem_gpu_ptr;
    struct doca_verbs_umem *cq_sq_umem;
    void *cq_sq_umem_dbr_gpu_ptr;
    struct doca_verbs_umem *cq_sq_umem_dbr;

    // QP
    struct doca_verbs_qp *qp;
    void *qp_umem_gpu_ptr;
    struct doca_verbs_umem *qp_umem;
    void *qp_umem_dbr_gpu_ptr;
    struct doca_verbs_umem *qp_umem_dbr;
    struct doca_verbs_uar *external_uar;

    enum doca_gpu_dev_verbs_nic_handler nic_handler;

    // QP GPUNetIO Object
    struct doca_gpu_verbs_qp *qp_gverbs;
};

struct doca_gpu_verbs_qp_group_hl {
    struct doca_gpu_verbs_qp_hl qp_main;
    struct doca_gpu_verbs_qp_hl qp_companion;
};

/**
 * Create an high-level GPUNetIO QP.
 * This function encapsulate all required steps using doca verbs and doca gpunetio to
 * create a GDAKI QP.
 *
 * @param [in] qp_init_attr
 * High-level QP init attributes.
 * @param [out] qp
 * GPUNetIO QP device handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_create_qp_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                         struct doca_gpu_verbs_qp_hl **qp);

/**
 * Destroy an high-level GPUNetIO QP.
 *
 * @param [in] qp
 * GPUNetIO high-level QP to destroy
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_destroy_qp_hl(struct doca_gpu_verbs_qp_hl *qp);

/**
 * Create an high-level GPUNetIO QP group (main and companion).
 * This function encapsulate all required steps using doca verbs and doca gpunetio to
 * create two GDAKI QPs, main one and the one used for core direct operations.
 * The two QPs share the same UAR.
 *
 * @param [in] qp_init_attr
 * High-level QP init attributes.
 * @param [out] qpg
 * GPUNetIO QP Group device handler.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_create_qp_group_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                               struct doca_gpu_verbs_qp_group_hl **qpg);

/**
 * Destroy an high-level GPUNetIO QP group.
 *
 * @param [in] qp
 * GPUNetIO high-level QP group to destroy
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_destroy_qp_group_hl(struct doca_gpu_verbs_qp_group_hl *qpg);

/**
 * Creates a flat list of GPU QP.
 * Copies each struct doca_gpu_dev_verbs_qp inside the struct doca_gpu_verbs_qp_hl into
 * a GPU array to avoid pointers dereferencing.
 *
 * @param [in] qp_list
 * GPUNetIO high-level QP array
 * @param [in] num_elems
 * Number of QP in the qp_list
 * @param [out] qp_gpu
 * Array of GPU QP structures.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_qp_flat_list_create_hl(struct doca_gpu_verbs_qp_hl **qp_list,
                                                   uint32_t num_elems,
                                                   struct doca_gpu_dev_verbs_qp **qp_gpu);

/**
 * Destry a flat list of GPU QP.
 *
 * @param [in] qp_gpu
 * Array of GPU QP structures.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_gpu_verbs_qp_flat_list_destroy_hl(struct doca_gpu_dev_verbs_qp *qp_gpu);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_HIGH_LEVEL_H */
