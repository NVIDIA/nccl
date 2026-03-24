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
 * @file doca_verbs_net_wrapper.h
 * @brief Main wrapper header for IB Verbs and mlx5dv API calls and structs
 *
 * This header includes the separate IB Verbs and mlx5dv wrappers.
 * It provides backward compatibility with the original unified wrapper.
 *
 * For IB Verbs wrapper, define DOCA_VERBS_USE_IBV_WRAPPER
 * For mlx5dv wrapper, define DOCA_VERBS_USE_MLX5DV_WRAPPER
 * For backward compatibility, define DOCA_VERBS_USE_WRAPPER (enables both)
 *
 * @{
 */
#ifndef DOCA_VERBS_NET_WRAPPER_H
#define DOCA_VERBS_NET_WRAPPER_H

#ifdef DOCA_VERBS_USE_NET_WRAPPER
#ifndef DOCA_VERBS_USE_IBV_WRAPPER
#define DOCA_VERBS_USE_IBV_WRAPPER
#endif
#ifndef DOCA_VERBS_USE_MLX5DV_WRAPPER
#define DOCA_VERBS_USE_MLX5DV_WRAPPER
#endif
#endif

/* Include the separate wrappers */
#include "doca_verbs_ibv_wrapper.h"
#include "doca_verbs_mlx5dv_wrapper.h"

#endif /* DOCA_VERBS_NET_WRAPPER_H */

/** @} */
