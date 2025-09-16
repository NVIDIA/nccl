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

struct doca_verbs_cq_attr {
    uint32_t cq_size{};
    void *cq_context{};
    struct doca_verbs_umem *external_umem{};
    struct doca_verbs_umem *external_umem_dbr{};
    uint32_t external_umem_offset{};
    uint64_t external_umem_dbr_offset{};
    struct doca_verbs_uar *external_uar{};
    enum doca_verbs_cq_overrun cq_overrun;
};

/**
 *  @brief This struct implements the doca verbs cq
 */
struct doca_verbs_cq {
   public:
    /**
     * @brief constructor
     *
     * @param [in] verbs_ctx
     * ibv_context
     * @param [in] cq_attr
     * The DOCA IB Verbs CQ attributes
     *
     */
    doca_verbs_cq(struct ibv_context *ibv_ctx, struct doca_verbs_cq_attr &cq_attr);

    /**
     * @brief destructor
     */
    ~doca_verbs_cq();

    /**
     * @brief destroy the cq
     *
     * @return
     * DOCA_SUCCESS on successful destroy.
     * DOCA_ERROR_DRIVER on failure to destroy the cq.
     *
     */
    doca_error_t destroy() noexcept;

    /**
     * @brief create the cq
     *
     */
    void create(struct doca_verbs_cq_attr &cq_attr);

    doca_error_t create_cq_obj(uint32_t uar_id, uint32_t log_nb_cqes, uint64_t db_umem_offset,
                               uint32_t db_umem_id, uint32_t wq_umem_id, bool cq_overrun) noexcept;

    /**
     * @brief Get CQ number
     *
     * @return CQ number
     */
    uint32_t get_cqn() const noexcept { return m_cqn; }

    /**
     * @brief Get CQ buff
     *
     * @return CQ buff
     */
    void *get_cq_buf() const noexcept { return m_cq_buf; }

    /**
     * @brief Get CQ num entries
     *
     * @return CQ num entries
     */
    uint32_t get_cq_num_entries() const noexcept { return m_num_cqes; }

    /**
     * @brief Get CQ UAR reg
     *
     * @return CQ UAR reg
     */
    uint64_t *get_cq_uar_db_reg() const noexcept { return m_uar_db_reg; }

    /**
     * @brief Get CQ ci dbr
     *
     * @return CQ ci dbr
     */
    uint32_t *get_cq_ci_dbr() const noexcept { return m_ci_dbr; }

    /**
     * @brief Get CQ arm dbr
     *
     * @return CQ arm dbr
     */
    uint32_t *get_cq_arm_dbr() const noexcept { return m_arm_dbr; }

   private:
    struct mlx5dv_devx_obj *m_cq_obj{};
    struct mlx5dv_devx_umem *m_umem_obj{};
    struct mlx5dv_devx_uar *m_uar_obj{};
    struct ibv_context *m_ibv_ctx{};
    uint8_t *m_umem_buf{};
    uint8_t *m_cq_buf{};
    uint32_t *m_db_buffer;
    uint64_t *m_uar_db_reg{};
    uint32_t m_num_cqes{};
    uint32_t m_cqn{};
    uint32_t *m_ci_dbr{};
    uint32_t *m_arm_dbr{};
    struct doca_verbs_cq_attr m_cq_attr {};
    struct doca_verbs_device_attr *m_verbs_device_attr{};

    doca_verbs_cq(doca_verbs_cq const &) = delete;
    doca_verbs_cq &operator=(doca_verbs_cq const &) = delete;
};
