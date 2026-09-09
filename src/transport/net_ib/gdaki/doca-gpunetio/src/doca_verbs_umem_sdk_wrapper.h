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
 * @file doca_verbs_umem_sdk_wrapper.h
 * @brief Wrapper for DOCA SDK API calls and structs
 *
 * This wrapper provides an abstraction layer over DOCA SDK APIs.
 * It's enabled by default. At runtime, if the DOCA_SDK_LIB_PATH env var
 * is set, DOCA open will look for DOCA SDK to execute functions.
 * When DOCA_SDK_LIB_PATH env var is defined:
 * - All DOCA SDK API calls are wrapped using dlopen
 * - All DOCA SDK API structs are wrapped
 * - The wrapper provides a clean abstraction layer with dynamic loading
 *
 * If the env var DOCA_SDK_LIB_PATH is not set, the standalone open source implementation
 * is used. This means, DOCA SDK restricted features are not available.
 *
 * @{
 */
#ifndef DOCA_VERBS_SDK_WRAPPER_UMEM_H
#define DOCA_VERBS_SDK_WRAPPER_UMEM_H

#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"
#include "host/doca_verbs.h"
#include "doca_sdk_wrapper.h"

/* Wrapper function declarations */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_umem_create(doca_dev_t *net_dev, doca_gpu_t *gpu,
                                                            void *address, size_t size,
                                                            uint32_t access_flags, int dmabuf_id,
                                                            size_t dmabuf_offset, void **umem);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_umem_destroy(void *umem);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_umem_get_id(void *umem, uint32_t *umem_id);

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_umem_get_size(void *umem, size_t *umem_size);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_umem_get_address(void *umem, void **umem_address);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_SDK_WRAPPER_UMEM_H */

/** @} */
