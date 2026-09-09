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
#include "doca_verbs.h"

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

/**
 * @enum doca_gpu_verbs_qp_init_attr_flags_hl
 * @brief High-level QP creation flags.
 *
 * If DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT is set with
 * host-CQ mode, cqe_ci mapping must use GDRCopy v2 forced-PCIe
 * (GDR_PIN_FLAG_FORCE_PCIE). If the runtime does not advertise
 * GDR_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE, QP creation fails and does not silently
 * fall back to the default mapping.
 */
enum doca_gpu_verbs_qp_init_attr_flags_hl {
    DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_NONE = 0,
    DOCA_GPUNETIO_VERBS_QP_INIT_ATTR_FLAGS_SUPPORT_DATA_DIRECT = (1u << 0),
};

struct doca_gpu_verbs_qp_init_attr_hl {
    doca_gpu_t *gpu_dev;
    doca_dev_t *net_dev;
    struct ibv_pd *ibpd;
    uint16_t sq_nwqe;
    /**
     * CQ type. UNKNOWN preserves legacy behavior: 64B when cq_collapsed is false,
     * 64B_COLLAPSED when cq_collapsed is true. 64B uses non-collapsed CQEs in GPU
     * memory. 64B_COLLAPSED uses collapsed CQEs in GPU memory. 64B_COLLAPSED_HOST
     * stores CQEs in host memory; callers must set cq_collapsed=true and drive progress
     * with doca_gpu_verbs_cpu_proxy_progress() directly or through doca_gpu_verbs_service.
     * 64B_COLLAPSED_HOST does not create an internal standalone polling thread.
     */
    enum doca_gpu_dev_verbs_cq_type cq_type;
    uint8_t reserved1[1];
    enum doca_gpu_dev_verbs_nic_handler nic_handler;
    enum doca_gpu_verbs_mem_reg_type mreg_type;
    enum doca_gpu_verbs_send_dbr_mode_ext send_dbr_mode_ext;
    bool cq_collapsed;
    bool enable_umem_cpu;
    uint8_t reserved2[2];
    enum doca_verbs_qp_ordering_semantic ordering_semantic;
    uint32_t flags;
    doca_verbs_comp_channel_t *comp_channel;
    uint8_t reserved3[4];
} __attribute__((__aligned__(8))) __attribute__((__packed__));

struct doca_gpu_verbs_qp_hl {
    doca_gpu_t *gpu_dev; /* DOCA GPU device to use */

    // CQ
    doca_verbs_cq_t *cq_sq;
    void *cq_sq_umem_gpu_ptr;
    doca_verbs_umem_t *cq_sq_umem;

    // QP
    doca_verbs_qp_t *qp;
    void *qp_umem_gpu_ptr;
    doca_verbs_umem_t *qp_umem;
    void *qp_umem_dbr_gpu_ptr;
    doca_verbs_umem_t *qp_umem_dbr;
    doca_verbs_uar_t *external_uar;

    enum doca_gpu_dev_verbs_nic_handler nic_handler;
    enum doca_gpu_verbs_send_dbr_mode_ext send_dbr_mode_ext;

    // QP GPUNetIO Object
    struct doca_gpu_verbs_qp *qp_gverbs;
};

struct doca_gpu_verbs_qp_group_hl {
    struct doca_gpu_verbs_qp_hl qp_main;
    struct doca_gpu_verbs_qp_hl qp_companion;
};

struct doca_gpu_verbs_umem_hl {
    void *base_cpu_ptr;
    void *base_gpu_ptr;
    size_t size;
    size_t next_offset;
    uint32_t refcount;
    doca_verbs_umem_t *umem;
    enum doca_gpu_verbs_mem_reg_type mreg_type;
};

struct doca_gpu_verbs_qp_list_hl {
    uint32_t num_qps;
    struct doca_gpu_verbs_qp_hl *qps;
    struct doca_gpu_verbs_umem_hl *cq_umem;
    struct doca_gpu_verbs_umem_hl *cq_dbr_umem;
    struct doca_gpu_verbs_umem_hl *sq_umem;
    struct doca_gpu_verbs_umem_hl *sq_dbr_umem;
};

struct doca_gpu_verbs_qp_group_list_hl {
    uint32_t num_qp_groups;
    struct doca_gpu_verbs_qp_group_hl *qpgs;
    struct doca_gpu_verbs_umem_hl *cq_umem;
    struct doca_gpu_verbs_umem_hl *cq_dbr_umem;
    struct doca_gpu_verbs_umem_hl *sq_umem;
    struct doca_gpu_verbs_umem_hl *sq_dbr_umem;
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

/**
 * Create a list of high-level GPUNetIO QPs backed by shared control-buffer slabs.
 * All QPs share four registered memory slabs (CQ ring, CQ DBR, SQ WQ, SQ DBR) to reduce
 * the number of umem registrations to O(1) instead of O(N).
 *
 * @param [in] qp_init_attr
 * High-level QP init attributes (gpu_dev, net_dev, ibpd, sq_nwqe, mreg_type must be set).
 * @param [in] num_qps
 * Number of QPs to create.
 * @param [out] qp_list
 * Allocated QP list handle.
 *
 * @return DOCA_SUCCESS or a doca_error code.
 */
doca_error_t doca_gpu_verbs_create_qp_list_hl(struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr,
                                              uint32_t num_qps,
                                              struct doca_gpu_verbs_qp_list_hl **qp_list);

/**
 * Destroy a QP list created by doca_gpu_verbs_create_qp_list_hl.
 *
 * @param [in] qp_list
 * QP list to destroy.
 *
 * @return DOCA_SUCCESS or a doca_error code.
 */
doca_error_t doca_gpu_verbs_destroy_qp_list_hl(struct doca_gpu_verbs_qp_list_hl *qp_list);

/**
 * Create a list of high-level GPUNetIO QP groups (main+companion pairs) backed by shared slabs.
 * All QP groups share four registered memory slabs (CQ ring, CQ DBR, SQ WQ, SQ DBR) to reduce
 * the number of umem registrations to O(1) instead of O(N).
 *
 * @param [in] qp_init_attr
 * High-level QP init attributes.
 * @param [in] num_qp_groups
 * Number of QP groups to create.
 * @param [out] qpg_list
 * Allocated QP group list handle.
 *
 * @return DOCA_SUCCESS or a doca_error code.
 */
doca_error_t doca_gpu_verbs_create_qp_group_list_hl(
    struct doca_gpu_verbs_qp_init_attr_hl *qp_init_attr, uint32_t num_qp_groups,
    struct doca_gpu_verbs_qp_group_list_hl **qpg_list);

/**
 * Destroy a QP group list created by doca_gpu_verbs_create_qp_group_list_hl.
 *
 * @param [in] qpg_list
 * QP group list to destroy.
 *
 * @return DOCA_SUCCESS or a doca_error code.
 */
doca_error_t doca_gpu_verbs_destroy_qp_group_list_hl(
    struct doca_gpu_verbs_qp_group_list_hl *qpg_list);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_HIGH_LEVEL_H */
