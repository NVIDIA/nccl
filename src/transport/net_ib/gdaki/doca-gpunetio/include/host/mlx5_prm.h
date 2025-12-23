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

#ifndef RTE_PMD_MLX5_PRM_H_
#define RTE_PMD_MLX5_PRM_H_

#include <unistd.h>
#include <linux/types.h>

#define MLX5_ADAPTER_PAGE_SHIFT 12

enum {
    MLX5_CQE_SIZE_64B = 0x0,
    MLX5_CQE_SIZE_128B = 0x1,
};

enum {
    MLX5_QPC_RQ_TYPE_REGULAR = 0x0,
    MLX5_QPC_RQ_TYPE_SRQ_RMP_XRC_SRQ_XRQ = 0x1,
    MLX5_QPC_RQ_TYPE_ZERO_SIZE_RQ = 0x3,
};

enum {
    MLX5_ROCE_ADDR_LAYOUT_ROCE_VERSION_VERSION_1_0 = 0x0,
    MLX5_ROCE_ADDR_LAYOUT_ROCE_VERSION_VERSION_1_5 = 0x1,
    MLX5_ROCE_ADDR_LAYOUT_ROCE_VERSION_VERSION_2_0 = 0x2,
};

enum {
    MLX5_QPC_MTU_256_BYTES = 0x1,
    MLX5_QPC_MTU_512_BYTES = 0x2,
    MLX5_QPC_MTU_1K_BYTES = 0x3,
    MLX5_QPC_MTU_2K_BYTES = 0x4,
    MLX5_QPC_MTU_4K_BYTES = 0x5,
    MLX5_QPC_MTU_8K_BYTES = 0x6,
    MLX5_QPC_MTU_RAW_ETHERNET_QP = 0x7,
};

enum {
    MLX5_QPC_STATE_RST = 0x0,
    MLX5_QPC_STATE_INIT = 0x1,
    MLX5_QPC_STATE_RTR = 0x2,
    MLX5_QPC_STATE_RTS = 0x3,
    MLX5_QPC_STATE_SQER = 0x4,
    MLX5_QPC_STATE_SQDRAINED = 0x5,
    MLX5_QPC_STATE_ERR = 0x6,
};

enum {
    MLX5_CQC_CQE_SZ_BYTES_64 = 0x0,
};

enum {
    MLX5_CQ_SET_CI = 0,
    MLX5_CQ_ARM_DB = 1,
};

struct mlx5_ifc_cqc_bits {
    uint8_t status[0x4];
    uint8_t as_notify[0x1];
    uint8_t initiator_src_dct[0x1];
    uint8_t dbr_umem_valid[0x1];
    uint8_t reserved_at_7[0x1];
    uint8_t cqe_sz[0x3];
    uint8_t cc[0x1];
    uint8_t reserved_at_c[0x1];
    uint8_t scqe_break_moderation_en[0x1];
    uint8_t oi[0x1];
    uint8_t cq_period_mode[0x2];
    uint8_t cqe_comp_en[0x1];
    uint8_t mini_cqe_res_format[0x2];
    uint8_t st[0x4];
    uint8_t reserved_at_18[0x1];
    uint8_t cqe_comp_layout[0x7];
    uint8_t dbr_umem_id[0x20];
    uint8_t reserved_at_40[0x14];
    uint8_t page_offset[0x6];
    uint8_t reserved_at_5a[0x2];
    uint8_t mini_cqe_res_format_ext[0x2];
    uint8_t cq_timestamp_format[0x2];
    uint8_t reserved_at_60[0x3];
    uint8_t log_cq_size[0x5];
    uint8_t uar_page[0x18];
    uint8_t reserved_at_80[0x4];
    uint8_t cq_period[0xc];
    uint8_t cq_max_count[0x10];
    uint8_t reserved_at_a0[0x18];
    uint8_t c_eqn[0x8];
    uint8_t reserved_at_c0[0x3];
    uint8_t log_page_size[0x5];
    uint8_t reserved_at_c8[0x18];
    uint8_t reserved_at_e0[0x20];
    uint8_t reserved_at_100[0x8];
    uint8_t last_notified_index[0x18];
    uint8_t reserved_at_120[0x8];
    uint8_t last_solicit_index[0x18];
    uint8_t reserved_at_140[0x8];
    uint8_t consumer_counter[0x18];
    uint8_t reserved_at_160[0x8];
    uint8_t producer_counter[0x18];
    uint8_t local_partition_id[0xc];
    uint8_t process_id[0x14];
    uint8_t reserved_at_1A0[0x20];
    uint8_t dbr_addr[0x40];
};

struct mlx5_ifc_create_cq_in_bits {
    uint8_t opcode[0x10];
    uint8_t uid[0x10];
    uint8_t reserved_at_20[0x10];
    uint8_t op_mod[0x10];
    uint8_t reserved_at_40[0x40];
    struct mlx5_ifc_cqc_bits cq_context;
    uint8_t cq_umem_offset[0x40];
    uint8_t cq_umem_id[0x20];
    uint8_t cq_umem_valid[0x1];
    uint8_t reserved_at_2e1[0x1f];
    uint8_t reserved_at_300[0x580];
    uint8_t pas[];
};

struct mlx5_err_cqe_ex {
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

/* If not present, it will compile but it will not work.
 * Fallback UAR mechanism is in place.
 */
#ifndef MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
#define MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED (1U << 31)
#endif

#endif /* RTE_PMD_MLX5_PRM_H_ */
