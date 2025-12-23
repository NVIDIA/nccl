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
 * @file doca_gpunetio_dev_verbs_def.h
 * @brief GDAKI common definitions
 *
 * @{
 */
#ifndef DOCA_GPUNETIO_VERBS_DEF_H
#define DOCA_GPUNETIO_VERBS_DEF_H

#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <linux/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro to temporarily cast a variable to volatile.
 */
#define DOCA_GPUNETIO_VOLATILE(x) (*(volatile typeof(x) *)&(x))

/**
 * Default warp size value of 32 threads
 */
#define DOCA_GPUNETIO_VERBS_WARP_SIZE 32

/**
 * Default warp full mask value
 */
#define DOCA_GPUNETIO_VERBS_WARP_FULL_MASK 0xffffffff

/**
 * Default page size alignment on GPU
 */
#define DOCA_GPUNETIO_VERBS_PAGE_SIZE 65536

/**
 * CQE Consumer Index Mask - 24bits counter
 */
#define DOCA_GPUNETIO_VERBS_CQE_CI_MASK 0xFFFFFF

/**
 * WQE Producer Index Mask - 16bits counter
 */
#define DOCA_GPUNETIO_VERBS_WQE_PI_MASK 0xFFFF

#define DOCA_GPUNETIO_IB_MLX5_WQE_SQ_SHIFT 6

/**
 * Set to 1 if mkeys passed to the wqe functions
 * are already swapped by application.
 * Otherwise set it to 0.
 */
#define DOCA_GPUNETIO_VERBS_MKEY_SWAPPED 1

/**
 * Enable debug prints in this headerfile.
 * Bad for performance, should be used only for debugging
 */
#ifndef DOCA_GPUNETIO_VERBS_ENABLE_DEBUG
#define DOCA_GPUNETIO_VERBS_ENABLE_DEBUG 0
#endif

#if DOCA_GPUNETIO_VERBS_ENABLE_DEBUG == 1
#include <assert.h>
#define DOCA_GPUNETIO_VERBS_ASSERT(x) assert(x)
#else
#define DOCA_GPUNETIO_VERBS_ASSERT(x) \
    do {                              \
    } while (0)
#endif

/**
 * WQE data segment inline data with byte count
 */
#define DOCA_GPUNETIO_VERBS_MAX_INLINE_SIZE 28

/**
 * CQE Opcode Shift Bytes
 */
#define DOCA_GPUNETIO_VERBS_MLX5_CQE_OPCODE_SHIFT 4

#define DOCA_GPUNETIO_VERBS_CQE_SIZE 64

#define DOCA_GPUNETIO_VERBS_WQE_IDX_SHIFT 8
/**
 * Max RDMA transfer size
 */
#define DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT 30
#define DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE \
    (1ULL << DOCA_GPUNETIO_VERBS_MAX_TRANSFER_SIZE_SHIFT)  // 1GiB

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

#ifndef WRITE_ONCE
#define WRITE_ONCE(x, v) (ACCESS_ONCE(x) = (v))
#endif

enum {
    DOCA_GPUNETIO_IB_MLX5_OPCODE_NOP = 0x00,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_SEND_INVAL = 0x01,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE = 0x08,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_WRITE_IMM = 0x09,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_SEND = 0x0a,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_SEND_IMM = 0x0b,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_TSO = 0x0e,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_RDMA_READ = 0x10,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_CS = 0x11,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_FA = 0x12,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_CS = 0x14,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_ATOMIC_MASKED_FA = 0x15,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_FMR = 0x19,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_LOCAL_INVAL = 0x1b,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_WAIT = 0x0f,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_CONFIG_CMD = 0x1f,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_SET_PSV = 0x20,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_DUMP = 0x23,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_UMR = 0x25,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_TAG_MATCHING = 0x28,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_FLOW_TBL_ACCESS = 0x2c,
    DOCA_GPUNETIO_IB_MLX5_OPCODE_MMO = 0x2F,
};

enum {
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_ON_CQE_ERROR = 0x0,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_ON_FIRST_CQE_ERROR = 0x1,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_ALWAYS = 0x2,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_AND_EQE = 0x3,
};

enum {
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_NO_FENCE = 0x0,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_INITIATOR_SMALL_FENCE = 0x1,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_FENCE = 0x2,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_STRONG_ORDERING = 0x3,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_FENCE_AND_INITIATOR_SMALL_FENCE = 0x4,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_CUSTOM = 0x100, /* None of the previous, use custom value */
};

/**
 * GPUNetIO Verbs flags for WQE control segment
 */
enum doca_gpu_dev_verbs_wqe_ctrl_flags {
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_UPDATE = DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_ALWAYS << 2,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_ERROR_UPDATE =
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_ON_CQE_ERROR << 2,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CQ_FIRST_CQE_ERROR =
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_CE_CQE_ON_FIRST_CQE_ERROR << 2,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_SOLICITED = 1 << 1,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FENCE =
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_FENCE_AND_INITIATOR_SMALL_FENCE << 5,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_INITIATOR_SMALL_FENCE =
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_INITIATOR_SMALL_FENCE << 5,
    DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_STRONG_ORDERING =
        DOCA_GPUNETIO_IB_MLX5_WQE_CTRL_FM_STRONG_ORDERING << 5
};

enum {
    DOCA_GPUNETIO_IB_MLX5_RCV_DBR = 0,
    DOCA_GPUNETIO_IB_MLX5_SND_DBR = 1,
};

/**
 * @enum doca_gpu_dev_verbs_mem_type
 * @brief Memory type of the buffer.
 */
enum doca_gpu_dev_verbs_mem_type {
    DOCA_GPUNETIO_VERBS_MEM_TYPE_AUTO =
        0,  ///< Automatically select the most performant memory type
    DOCA_GPUNETIO_VERBS_MEM_TYPE_HOST = 1,      ///< Allocate resource on host memory
    DOCA_GPUNETIO_VERBS_MEM_TYPE_GPU = 2,       ///< Allocate resource on GPU memory
    DOCA_GPUNETIO_VERBS_MEM_TYPE_MAX = INT_MAX  ///< Sentinel value
};

/**
 * @enum doca_gpu_dev_verbs_mem_type
 * @brief Memory type of the buffer.
 */
enum doca_gpu_dev_verbs_qp_type {
    DOCA_GPUNETIO_VERBS_QP_SQ = 0,  ///< Use QP SQ
};

enum doca_gpu_dev_verbs_exec_scope {
    DOCA_GPUNETIO_VERBS_EXEC_SCOPE_THREAD = 0,
    DOCA_GPUNETIO_VERBS_EXEC_SCOPE_WARP
};

/**
 * @enum doca_gpu_dev_verbs_sync_scope
 * @brief Synchronization scope.
 */
enum doca_gpu_dev_verbs_sync_scope {
    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_SYS = 0,       ///< System synchronization scope
    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_GPU = 1,       ///< GPU synchronization scope
    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_CTA = 2,       ///< CTA synchronization scope
    DOCA_GPUNETIO_VERBS_SYNC_SCOPE_MAX = INT_MAX  ///< Sentinel value
};

/**
 * @enum doca_gpu_dev_verbs_resource_sharing_mode
 * @brief Resource sharing mode.
 */
enum doca_gpu_dev_verbs_resource_sharing_mode {
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_EXCLUSIVE =
        0,  ///< The resource is exclusive to one CUDA thread
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_CTA = 1,       ///< The resource is shared among CUDA
                                                             ///< threads in the same CTA
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_GPU = 2,       ///< The resource is shared among CUDA
                                                             ///< threads in the same GPU
    DOCA_GPUNETIO_VERBS_RESOURCE_SHARING_MODE_MAX = INT_MAX  ///< Sentinel value
};

/**
 * @enum doca_gpu_dev_verbs_nic_handler
 * @brief The processor that handles the NIC.
 */
enum doca_gpu_dev_verbs_nic_handler {
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_AUTO = 0,  ///< Automatically select the most performant handler
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_CPU_PROXY = 1,  ///< CPU Proxy
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_DB = 2,  ///< GPU SM, regular DB
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_GPU_SM_BF = 3,  ///< GPU SM, BlueFlame DB
    DOCA_GPUNETIO_VERBS_NIC_HANDLER_TYPE_MAX,       ///< Sentinel value
};

/**
 * @enum doca_gpu_dev_verbs_gpu_code_opt
 * @brief GPU code optimization for GDA-KI. They can be combined using bitwise or.
 */
enum doca_gpu_dev_verbs_gpu_code_opt {
    DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_DEFAULT = 0,  ///< Use default code optimization
    DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_ASYNC_STORE_RELEASE = (1 << 0),  ///< Use store.async.release
                                                                      ///< code optimization
    DOCA_GPUNETIO_VERBS_GPU_CODE_OPT_MAX = INT_MAX                    ///< Sentinel value
};

enum doca_gpu_dev_verbs_signal_op {
    DOCA_GPUNETIO_VERBS_SIGNAL_OP_ADD = 0,  ///< Signal operation - Add
};

enum {
    DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MIN = 3,
    DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_RDMA_WRITE_INL_MAX = 4,
    DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_ATOMIC_FA_CAS = 4,
    DOCA_GPUNETIO_VERBS_WQE_SEG_CNT_WAIT = 2
};

enum {
    DOCA_GPUNETIO_IB_MLX5_INLINE_SEG = 0x80000000,
};

enum {
    DOCA_GPUNETIO_IB_MLX5_CQE_OWNER_MASK = 1,
    DOCA_GPUNETIO_IB_MLX5_CQE_REQ = 0,
    DOCA_GPUNETIO_IB_MLX5_CQE_RESP_WR_IMM = 1,
    DOCA_GPUNETIO_IB_MLX5_CQE_RESP_SEND = 2,
    DOCA_GPUNETIO_IB_MLX5_CQE_RESP_SEND_IMM = 3,
    DOCA_GPUNETIO_IB_MLX5_CQE_RESP_SEND_INV = 4,
    DOCA_GPUNETIO_IB_MLX5_CQE_RESIZE_CQ = 5,
    DOCA_GPUNETIO_IB_MLX5_CQE_NO_PACKET = 6,
    DOCA_GPUNETIO_IB_MLX5_CQE_SIG_ERR = 12,
    DOCA_GPUNETIO_IB_MLX5_CQE_REQ_ERR = 13,
    DOCA_GPUNETIO_IB_MLX5_CQE_RESP_ERR = 14,
    DOCA_GPUNETIO_IB_MLX5_CQE_INVALID = 15,
};

struct doca_gpunetio_ib_mlx5_wqe_data_seg {
    __be32 byte_count;
    __be32 lkey;
    __be64 addr;
};

struct doca_gpunetio_ib_mlx5_wqe_ctrl_seg {
    __be32 opmod_idx_opcode;
    __be32 qpn_ds;
    uint8_t signature;
    __be16 dci_stream_channel_id;
    uint8_t fm_ce_se;
    __be32 imm;
} __attribute__((__packed__)) __attribute__((__aligned__(4)));

struct doca_gpunetio_ib_mlx5_wqe_raddr_seg {
    __be64 raddr;
    __be32 rkey;
    __be32 reserved;
};

struct doca_gpunetio_ib_mlx5_wqe_atomic_seg {
    __be64 swap_add;
    __be64 compare;
};

struct doca_gpunetio_ib_mlx5_wqe_inl_data_seg {
    uint32_t byte_count;
};

struct doca_gpunetio_ib_mlx5_tm_cqe {
    __be32 success;
    __be16 hw_phase_cnt;
    uint8_t rsvd0[12];
};

struct doca_gpunetio_ib_ibv_tmh {
    uint8_t opcode;      /* from enum ibv_tmh_op */
    uint8_t reserved[3]; /* must be zero */
    __be32 app_ctx;      /* opaque user data */
    __be64 tag;
};

struct doca_gpunetio_ib_mlx5_cqe64 {
    union {
        struct {
            uint8_t rsvd0[2];
            __be16 wqe_id;
            uint8_t rsvd4[13];
            uint8_t ml_path;
            uint8_t rsvd20[4];
            __be16 slid;
            __be32 flags_rqpn;
            uint8_t hds_ip_ext;
            uint8_t l4_hdr_type_etc;
            __be16 vlan_info;
        };
        struct doca_gpunetio_ib_mlx5_tm_cqe tm_cqe;
        /* TMH is scattered to CQE upon match */
        struct doca_gpunetio_ib_ibv_tmh tmh;
    };
    __be32 srqn_uidx;
    __be32 imm_inval_pkey;
    uint8_t app;
    uint8_t app_op;
    __be16 app_info;
    __be32 byte_cnt;
    __be64 timestamp;
    __be32 sop_drop_qpn;
    __be16 wqe_counter;
    uint8_t signature;
    uint8_t op_own;
};

struct doca_gpunetio_ib_mlx5_err_cqe_ex {
    uint8_t rsvd0[32];
    __be32 srqn;
    uint8_t rsvd1[16];
    uint8_t hw_err_synd;
    uint8_t hw_synd_type;
    uint8_t vendor_err_synd;
    uint8_t syndrome;
    __be32 s_wqe_opcode_qpn;
    __be16 wqe_counter;
    uint8_t signature;
    uint8_t op_own;
};

#ifdef __cplusplus
}
#endif

#endif /* DOCA_GPUNETIO_VERBS_DEF_H */

/** @} */
