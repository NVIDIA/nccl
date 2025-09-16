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

#include <unistd.h>
#include <stdlib.h>
#include <mutex>
#include <time.h>

#include "host/mlx5_prm.h"
#include "host/mlx5_ifc.h"

#include "doca_internal.hpp"

/**
 *  @brief This struct implements the doca rdma_verbs device attributes
 */
struct doca_verbs_device_attr {
   public:
    /**
     * @brief constructor
     *
     * @param [in] ibv_ctx
     * IBV context to query device attributes from
     *
     */
    doca_verbs_device_attr(struct ibv_context *ibv_ctx);

    /**
     * @brief destructor
     */
    ~doca_verbs_device_attr() = default;

    /**
     * @brief Query device capabilities
     *
     * @param [in] ibv_ctx
     * IBV context to query device attributes from
     *
     */
    void query_caps(struct ibv_context *ibv_ctx);

    uint32_t m_max_qp{};
    uint32_t m_max_qp_wr{};
    uint32_t m_max_sge{};
    uint32_t m_max_cq{};
    uint32_t m_max_cqe{};
    uint32_t m_max_mr{};
    uint32_t m_max_pd{};
    uint32_t m_max_ah{};
    uint32_t m_max_srq{};
    uint32_t m_max_srq_wr{};
    uint32_t m_max_srq_sge{};
    uint32_t m_max_pkeys{};
    uint32_t m_max_sq_desc_size{};
    uint32_t m_max_rq_desc_size{};
    uint32_t m_max_send_wqebb{};
    uint16_t m_min_udp_sport{};
    uint16_t m_max_udp_sport{};
    uint16_t m_gid_table_size{};
    uint8_t m_is_qp_rc_supported{};
    uint8_t m_port_type{};
    uint8_t m_is_rts2rts_qp_dscp_supported{};
    uint8_t m_phys_port_cnt{};

   private:
    doca_verbs_device_attr &operator=(doca_verbs_device_attr const &) = delete;
};
