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

#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "host/doca_verbs.h"
#include "doca_verbs_uar.hpp"

struct doca_verbs_ah_attr {
    struct doca_verbs_gid gid {};
    enum doca_verbs_addr_type addr_type { DOCA_VERBS_ADDR_TYPE_IPv4 };
    uint32_t dlid{};
    uint8_t sl{};
    uint8_t sgid_index{};
    uint8_t static_rate{};
    uint8_t hop_limit{};
    uint8_t traffic_class{};
    uint8_t is_global{};
};

struct doca_verbs_qp_init_attr {
    struct ibv_pd *pd{};
    struct doca_verbs_cq *send_cq{};
    struct doca_verbs_cq *receive_cq{};
    struct doca_verbs_srq *srq{};
    struct doca_verbs_umem *external_umem{};
    struct doca_verbs_umem *external_umem_dbr{};
    struct doca_verbs_uar *external_uar{};
    uint64_t external_umem_offset{};
    uint64_t external_umem_dbr_offset{};
    int sq_sig_all{};
    uint32_t sq_wr{};
    uint32_t rq_wr{};
    uint32_t send_max_sges{};
    uint32_t receive_max_sges{};
    uint32_t max_inline_data{};
    uint32_t user_index{};
    uint32_t qp_type{};
    void *qp_context{};
    uint32_t send_cqn{};
    uint32_t receive_cqn{};
    uint8_t core_direct_master{};
};

struct doca_verbs_qp_attr {
    enum doca_verbs_qp_state next_state { DOCA_VERBS_QP_STATE_RST };
    enum doca_verbs_qp_state current_state { DOCA_VERBS_QP_STATE_RST };
    enum doca_verbs_mtu_size path_mtu { DOCA_VERBS_MTU_SIZE_256_BYTES };
    uint32_t rq_psn{};
    uint32_t sq_psn{};
    uint32_t dest_qp_num{};
    int allow_remote_write{};
    int allow_remote_read{};
    enum doca_verbs_qp_atomic_type allow_remote_atomic {};
    doca_verbs_ah_attr *ah_attr{};
    uint16_t pkey_index{};
    uint16_t port_num{};
    uint8_t ack_timeout{};
    uint8_t retry_cnt{};
    uint8_t rnr_retry{};
    uint8_t min_rnr_timer{};
    uint8_t core_direct_master{};
};

/**
 *  @brief This struct implements the doca_verbs_qp
 */
struct doca_verbs_qp {
   public:
    /**
     * @brief constructor
     *
     * @param [in] ibv_ctx
     * The ibv context
     * @param [in] verbs_qp_init_attr
     * The DOCA IB Verbs QP attributes
     *
     */
    doca_verbs_qp(struct ibv_context *ibv_ctx, struct doca_verbs_qp_init_attr &verbs_qp_init_attr);

    /**
     * @brief destructor
     */
    ~doca_verbs_qp();

    void create(struct ibv_context *ibv_ctx);

    doca_error_t destroy() noexcept;

    uint32_t get_qpn() const noexcept;

    void *get_sq_buf() const noexcept;

    void *get_rq_buf() const noexcept;

    uint32_t get_sq_size_wqebb() const noexcept;

    uint32_t get_rq_size() const noexcept;

    uint32_t get_rcv_wqe_size() const noexcept;

    void *get_dbr_addr() const noexcept;

    void *get_uar_addr() const noexcept;

    enum doca_verbs_uar_allocation_type get_uar_mtype() const noexcept;

    doca_error_t create_qp_obj(uint32_t uar_id, uint32_t log_rq_size, uint32_t log_sq_size,
                               uint32_t log_stride, uint64_t dbr_umem_offset, uint32_t dbr_umem_id,
                               uint32_t wq_umem_id,
                               struct doca_verbs_qp_init_attr &verbs_qp_init_attr) noexcept;

    doca_error_t rst2init(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t init2init(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t init2rtr(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t rtr2rts(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t rts2rts(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t qp2err(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t qp2rst(struct doca_verbs_qp_attr &verbs_qp_attr, int param_mask) noexcept;

    doca_error_t query_qp(struct doca_verbs_qp_attr &verbs_qp_attr,
                          struct doca_verbs_qp_init_attr &verbs_qp_init_attr) noexcept;

    bool is_qp_attr_state_valid(enum doca_verbs_qp_state state) noexcept;

    bool is_qp_attr_path_mtu_valid(enum doca_verbs_mtu_size path_mtu) noexcept;

    uint32_t is_qp_attr_queue_psn_valid(uint32_t psn) noexcept;

    bool is_qp_attr_ah_add_type_valid(enum doca_verbs_addr_type addr_type) noexcept;

    bool is_qp_attr_ah_sgid_index_valid(uint8_t sgid_index) noexcept;

    bool is_qp_attr_pkey_index_valid(uint16_t pkey_index) noexcept;

    bool is_qp_attr_port_num_valid(uint16_t port_num) noexcept;

    bool is_qp_attr_valid(struct doca_verbs_qp_attr *verbs_qp_attr, int attr_mask) noexcept;

    doca_verbs_qp_state get_current_state() const noexcept;

   private:
    struct mlx5dv_devx_obj *m_qp_obj{};
    struct mlx5dv_devx_umem *m_umem_obj{};
    struct mlx5dv_devx_uar *m_uar_obj{};
    uint8_t *m_umem_buf{};
    uint8_t *m_wq_buf{};
    struct ibv_context *m_ibv_ctx{};
    struct ibv_pd *m_pd{};
    uint32_t m_qp_type{DOCA_VERBS_QP_TYPE_RC};
    doca_verbs_addr_type m_addr_type{DOCA_VERBS_ADDR_TYPE_IPv4};
    doca_verbs_qp_state m_current_state{DOCA_VERBS_QP_STATE_RST};
    uint32_t m_rcv_wqe_size{};
    uint32_t m_send_wqe_size{};
    doca_verbs_qp_init_attr m_init_attr{};
    struct doca_verbs_device_attr *m_verbs_device_attr{};
    uint32_t m_rcv_max_sges{};
    uint8_t m_log_rcv_wqe_size{};
    uint32_t m_rq_size{};
    uint32_t m_send_max_sges{};
    uint32_t m_sq_size_wqebb{};
    uint32_t m_sq_size_wr{};
    uint32_t m_max_inline_data_length{};
    uint64_t *m_uar_db_reg{};
    uint8_t *m_sq_buf{};
    uint8_t *m_rq_buf{};
    uint32_t m_qp_num{};
    uint32_t *m_db_buffer{};
    struct doca_verbs_srq *m_srq{};

    doca_verbs_qp(doca_verbs_qp const &) = delete;
    doca_verbs_qp &operator=(doca_verbs_qp const &) = delete;
};
