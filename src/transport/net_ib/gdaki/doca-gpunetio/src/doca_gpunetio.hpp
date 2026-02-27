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

#include <unordered_map>

#include "doca_verbs_cuda_wrapper.h"

struct doca_gpu {
    CUdevice cuda_dev; /* CUDA device handler */
    std::unordered_map<uintptr_t, struct doca_gpu_mtable *>
        *mtable;                       /* Table of GPU/CPU memory allocated addresses */
    bool support_gdrcopy;              ///< Boolean value that indicates if gdrcopy is
                                       ///< supported
    bool support_dmabuf;               ///< Boolean value that indicates if dmabuf is
                                       ///< supported by the gpu
    bool support_wq_gpumem;            ///< Boolean value that indicates if gpumem is
                                       ///< available and nic-gpu mapping is supported
    bool support_cq_gpumem;            ///< Boolean value that indicates if gpumem is
                                       ///< available and nic-gpu mapping is supported
    bool support_uar_gpumem;           ///< Boolean value that indicates if gpumem is
                                       ///< available and gpu-nic mapping is supported
    bool support_async_store_release;  ///< Boolean value that indicates if
                                       ///< async store release is supported
    bool support_bf_uar;               ///< Boolean value that indicates if BlueFlame
                                       ///< is supported
};
