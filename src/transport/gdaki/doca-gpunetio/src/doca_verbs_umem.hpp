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
 *  @brief This struct implements the doca verbs umem
 */
struct doca_verbs_umem {
   public:
    /**
     * @brief constructor
     *
     * @param [in] ibv_ctx
     * ibv_context
     * @param [in] address
     * The umem address.
     * @param [in] size
     * The umem size.
     * @param [in] access_flags
     * The umem access flags.
     * @param [in] dmabuf_fd
     * The umem dmabuf file descriptor id.
     * @param [in] dmabuf_offset
     * The umem dmabuf offset.
     */
    doca_verbs_umem(struct ibv_context *ibv_ctx, void *address, size_t size, uint32_t access_flags,
                    int dmabuf_fd, size_t dmabuf_offset);

    /**
     * @brief destructor
     */
    ~doca_verbs_umem();

    /**
     * @brief destroy the umem
     *
     * @return
     * DOCA_SUCCESS on successful destroy.
     * DOCA_ERROR_DRIVER on failure to destroy the umem.
     *
     */
    doca_error_t destroy() noexcept;

    /**
     * @brief create the umem
     *
     */
    void create();

    /**
     * @brief Get umem ID
     *
     * @return umem ID
     */
    uint32_t get_umem_id() const noexcept { return m_umem_id; }

    /**
     * @brief Get umem size
     *
     * @return umem size
     */
    size_t get_umem_size() const noexcept { return m_size; }

    /**
     * @brief Get umem address
     *
     * @return umem address
     */
    void *get_umem_address() const noexcept { return m_address; }

   private:
    struct mlx5dv_devx_umem *m_umem_obj{};
    struct ibv_context *m_ibv_ctx{};
    void *m_address{};
    size_t m_size{};
    uint32_t m_access_flags{};
    uint32_t m_umem_id{};
    int m_dmabuf_fd;
    size_t m_dmabuf_offset;

    doca_verbs_umem(doca_verbs_umem const &) = delete;
    doca_verbs_umem &operator=(doca_verbs_umem const &) = delete;
};
