/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <stddef.h>
#include <stdint.h>

#include "common/doca_gpunetio_verbs_def.h"
#include "host/doca_error.h"
#include "doca_gpunetio_dev_verbs_v1.h"

/* Callers of the unversioned symbols predate _ver and were built against 5.0. Fixed rather than
 * tied to MIN_COMPAT_HOST, so raising that cannot change the layout they receive. */
static constexpr uint32_t sc_gpu_verbs_unversioned_device_code_version =
    5u << DOCA_GPUNETIO_VERSION_MAJOR_SHIFT;

/* One per frozen device layout. Start a new one when the bytes device code reads change; gate
 * device-visible features on the version in check_version instead. Stored on the QP so the host
 * keeps addressing the block in the layout it was exported in. */
static constexpr uint32_t sc_gpu_verbs_qp_dev_layout_v1 = 1;

doca_error_t doca_internal_gpu_verbs_qp_dev_select(uint32_t device_code_version, uint32_t *layout,
                                                   size_t *qp_dev_size);
doca_error_t doca_internal_gpu_verbs_qp_dev_check_version(uint32_t device_code_version,
                                                          const struct doca_gpu_dev_verbs_qp *src);
size_t doca_internal_gpu_verbs_qp_dev_size(uint32_t layout);
doca_error_t doca_internal_gpu_verbs_qp_dev_check(uint32_t layout,
                                                  const struct doca_gpu_dev_verbs_qp *src);
doca_error_t doca_internal_gpu_verbs_qp_dev_fill(uint32_t layout, void *qp_array, unsigned int idx,
                                                 const struct doca_gpu_dev_verbs_qp *src);

/* qp_dev is a host copy of the exported block, taken while the QP is not in use. It is rebuilt in
 * place from qp_cpu, which first takes the state the layout carries over. */
doca_error_t doca_internal_gpu_verbs_qp_dev_reset(uint32_t layout, void *qp_dev,
                                                  struct doca_gpu_dev_verbs_qp *qp_cpu);

/* Fields of the live block the host keeps addressing while device code runs. The host must not
 * reach them through struct doca_gpu_dev_verbs_qp, whose layout may differ from the block's. */
uint64_t *doca_internal_gpu_verbs_qp_dev_sq_wqe_pi(uint32_t layout, void *qp_dev);
uint64_t *doca_internal_gpu_verbs_qp_dev_cqe_ci(uint32_t layout, void *qp_dev);
