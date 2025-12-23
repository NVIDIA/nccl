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

struct doca_verbs_srq_init_attr {
    struct ibv_pd *pd{};
    enum doca_verbs_srq_type srq_type { DOCA_VERBS_SRQ_TYPE_LINKED_LIST };
    uint32_t srq_wr{};
    uint32_t receive_max_sges{};
    struct doca_verbs_umem *external_umem{};
    uint64_t external_umem_offset{};
};

/**
 *  @brief This struct implements the doca verbs srq
 */
struct doca_verbs_srq {
   public:
    /**
     * @brief constructor
     *
     * @param [in] verbs_ctx
     * The DOCA IB Verbs Context
     * @param [in] verbs_srq_init_attr
     * The DOCA IB Verbs SRQ attributes
     *
     */
    doca_verbs_srq(struct ibv_context *ctx, struct doca_verbs_srq_init_attr &verbs_srq_init_attr);

    /**
     * @brief destructor
     */
    ~doca_verbs_srq();

    doca_error_t create_srq_obj(uint32_t log_srq_size, uint32_t log_stride,
                                uint64_t dbr_umem_offset, uint32_t dbr_umem_id,
                                uint64_t wq_umem_offset, uint32_t wq_umem_id,
                                struct doca_verbs_srq_init_attr &verbs_srq_init_attr) noexcept;

    void create(struct ibv_context *ctx);

    doca_error_t destroy() noexcept;

    void *get_srq_buf() const noexcept;

    uint32_t get_srq_size() const noexcept;

    uint32_t get_rcv_wqe_size() const noexcept;

    void *get_dbr_addr() const noexcept;

    uint32_t get_srqn() const noexcept;

   private:
    struct mlx5dv_devx_obj *m_srq_obj{};
    struct mlx5dv_devx_umem *m_umem_obj{};
    uint8_t *m_umem_buf{};
    uint8_t *m_srq_buf{};
    struct ibv_context *m_ctx{};
    struct ibv_pd *m_pd{};
    uint32_t m_srq_num{};
    uint32_t *m_db_buffer{};
    uint32_t m_rcv_wqe_size{};
    uint32_t m_rcv_max_sges{};
    uint32_t m_srq_size{};
    uint8_t m_log_rcv_wqe_size{};
    enum doca_verbs_srq_type m_srq_type { DOCA_VERBS_SRQ_TYPE_LINKED_LIST };
    doca_verbs_srq_init_attr m_init_attr{};
    doca_verbs_device_attr *m_verbs_device_attr{};

    doca_verbs_srq(doca_verbs_srq const &) = delete;
    doca_verbs_srq &operator=(doca_verbs_srq const &) = delete;
};
