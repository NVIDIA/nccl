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
 * @file doca_gpunetio_dev_verbs_dev.h
 * @brief GDAKI common definitions
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_VERBS_DEV_H
#define DOCA_GPUNETIO_VERBS_DEV_H

#include "doca_gpunetio_verbs_def.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @typedef doca_gpu_dev_verbs_ticket_t
 * @brief Ticket type used in one-sided APIs.
 */
typedef uint64_t doca_gpu_dev_verbs_ticket_t;

/**
 * Describes GPUNetIO dev WQE crtl segment.
 */
struct doca_gpu_dev_verbs_wqe_ctrl_seg {
    __be32 opmod_idx_opcode; /**< opcode + wqe idx */
    __be32 qpn_ds;           /**< qp number */
    union {
        struct {
            uint8_t signature; /**< signature */
            uint8_t rsvd[2];   /**< reserved */
            uint8_t fm_ce_se;  /**< fm_ce_se */
        };
        struct {
            __be32 signature_fm_ce_se; /**< all flags in or */
        };
    };

    __be32 imm; /**< immediate */
} __attribute__((__aligned__(8)));

/**
 * Describes GPUNetIO dev WQE crtl segment.
 */
struct doca_gpu_dev_verbs_wqe_wait_seg {
    uint32_t resv[2];
    __be32 max_index;
    __be32 qpn_cqn;
} __attribute__((__packed__)) __attribute__((__aligned__(8)));

/**
 * @struct doca_gpu_dev_verbs_addr
 * @brief This structure holds the address and key of a memory region.
 */
struct doca_gpu_dev_verbs_addr {
    uint64_t addr;
    __be32 key;
};

/**
 * Describes GPUNetIO dev general WQE.
 */
struct doca_gpu_dev_verbs_wqe {
    union {
        /* Generic inline Data */
        struct {
            uint8_t inl_data[64];
        };

        /* Generic Data */
        struct {
            struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg0;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg1;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg2;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg dseg3;
        };

        /* Read/Write */
        struct {
            struct doca_gpu_dev_verbs_wqe_ctrl_seg rw_cseg;
            struct doca_gpunetio_ib_mlx5_wqe_raddr_seg rw_rseg;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg rw_dseg0;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg rw_dseg1;
        };

        /* Atomic */
        struct {
            struct doca_gpu_dev_verbs_wqe_ctrl_seg at_cseg;
            struct doca_gpunetio_ib_mlx5_wqe_raddr_seg at_rseg;
            struct doca_gpunetio_ib_mlx5_wqe_atomic_seg at_seg;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg at_dseg;
        };

        /* Send */
        struct {
            struct doca_gpu_dev_verbs_wqe_ctrl_seg snd_cseg;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg snd_dseg0;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg snd_dseg1;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg snd_dseg2;
        };

        /* Wait */
        struct {
            struct doca_gpu_dev_verbs_wqe_ctrl_seg wait_cseg;
            struct doca_gpu_dev_verbs_wqe_wait_seg wait_dseg;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg padding0;
            struct doca_gpunetio_ib_mlx5_wqe_data_seg padding1;
        };
    };
} __attribute__((__aligned__(8)));

/**
 * Describes GPUNetIO dev CQ
 */
struct doca_gpu_dev_verbs_cq {
    uint8_t *cqe_daddr;                         /**< CQE address */
    uint32_t cq_num;                            /**< CQ number */
    uint32_t cqe_num;                           /**< Total number of CQEs in CQ */
    __be32 *dbrec;                              /**< CQE Doorbell Record */
    uint64_t cqe_ci;                            /**< CQE Consumer Index */
    uint32_t cqe_mask;                          /**< Mask of total number of CQEs in CQ */
    uint8_t cqe_size;                           /**< Single CQE size (64B default) */
    uint64_t cqe_rsvd;                          /**< All previous CQEs are polled */
    enum doca_gpu_dev_verbs_mem_type mem_type;  ///< Memory type of the completion queue
};

/**
 * Describes GPUNetIO dev QP
 */
struct doca_gpu_dev_verbs_qp {
    uint64_t sq_rsvd_index;        ///< All WQE slots prior to this index are reserved
    uint64_t sq_ready_index;       ///< All WQE slots prior to this index are ready
    uint64_t sq_wqe_pi;            /**< tbd */
    uint32_t sq_num;               /**< SQ num */
    uint32_t sq_num_shift8;        /**< SQ num << 8 */
    uint32_t sq_num_shift8_be;     /**< SQ num << 8 big endian */
    uint32_t sq_num_shift8_be_1ds; /**< SQ num << 8 big endian */
    uint32_t sq_num_shift8_be_2ds; /**< SQ num << 8 big endian */
    uint32_t sq_num_shift8_be_3ds; /**< SQ num << 8 big endian */
    uint32_t sq_num_shift8_be_4ds; /**< SQ num << 8 big endian */
    int sq_lock;                   /**< SQ lock */
    uint16_t sq_wqe_num;           /**< tbd */
    uint16_t sq_wqe_mask;          /**< tbd */
    uint8_t *sq_wqe_daddr;         /**< tbd */
    __be32 *sq_dbrec;              /**< tbd */
    uint64_t *sq_db;               /**< tbd */

    /* Compatibility with DOCA GPUNetIO full, not really used */
    uint32_t rq_num;         /**< tbd */
    uint64_t rq_wqe_pi;      /**< tbd */
    uint32_t rq_wqe_num;     /**< tbd */
    uint32_t rq_wqe_mask;    /**< tbd */
    uint8_t *rq_wqe_daddr;   /**< tbd */
    __be32 *rq_dbrec;        /**< tbd */
    uint32_t rcv_wqe_size;   /**< tbd */
    uint64_t rq_rsvd_index;  /**< All previous WQEs are reserved */
    uint64_t rq_ready_index; /**< All previous WQEs are ready */
    int rq_lock;             /**< RQ lock */

    struct doca_gpu_dev_verbs_cq cq_sq; /**< SQ CQ connected to QP */
    struct doca_gpu_dev_verbs_cq cq_rq; /**< RQ CQ connected to QP */

    enum doca_gpu_dev_verbs_nic_handler nic_handler;  ///< NIC handler
    enum doca_gpu_dev_verbs_mem_type mem_type;        ///< Memory type of the completion
} __attribute__((__aligned__(8)));

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_VERBS_DEV_H */

/** @} */
