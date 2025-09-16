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

/**
 *  @brief This struct implements the doca verbs uar
 */
struct doca_verbs_uar {
   public:
    /**
     * @brief constructor
     *
     * @param [in] context
     * ibv_context
     * @param [in] allocation_type
     * The uar allocation type.
     */
    doca_verbs_uar(struct ibv_context *context,
                   enum doca_verbs_uar_allocation_type allocation_type);

    /**
     * @brief destructor
     */
    ~doca_verbs_uar();

    /**
     * @brief destroy the uar
     *
     * @return
     * DOCA_SUCCESS on successful destroy.
     * DOCA_ERROR_DRIVER on failure to destroy the uar.
     *
     */
    doca_error_t destroy() noexcept;

    /**
     * @brief create the uar
     *
     */
    void create();

    /**
     * @brief Get uar ID
     *
     * @return uar ID
     */
    uint32_t get_uar_id() const noexcept { return m_uar_id; }

    /**
     * @brief Get UAR reg address
     *
     * @return UAR reg address
     */
    void *get_reg_addr() const noexcept { return m_reg_addr; }

    /**
     * @brief Get UAR memory allocation type
     *
     * @return UAR memory allocation type
     */
    enum doca_verbs_uar_allocation_type get_uar_mtype() const noexcept { return m_allocation_type; }

   private:
    struct mlx5dv_devx_uar *m_uar_obj{};
    struct ibv_context *m_ibv_ctx{};
    enum doca_verbs_uar_allocation_type m_allocation_type {
        DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME
    };
    uint32_t m_uar_id{};
    void *m_reg_addr{};

    doca_verbs_uar(doca_verbs_uar const &) = delete;
    doca_verbs_uar &operator=(doca_verbs_uar const &) = delete;
};
