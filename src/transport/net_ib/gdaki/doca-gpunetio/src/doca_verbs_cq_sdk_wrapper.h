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
 * @file doca_verbs_cq_sdk_wrapper.h
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
#ifndef DOCA_VERBS_SDK_WRAPPER_CQ_H
#define DOCA_VERBS_SDK_WRAPPER_CQ_H

#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"
#include "host/doca_verbs.h"
#include "doca_verbs_uar_sdk_wrapper.h"
#include "doca_verbs_umem_sdk_wrapper.h"

/* Wrapper function declarations */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_create(void **cq_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_destroy(void *cq_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_size(void *cq_attr,
                                                                    uint32_t cq_size);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_context(void *cq_attr,
                                                                       void *cq_context);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_external_umem(
    void *cq_attr, doca_verbs_umem_t *external_umem, uint64_t external_umem_offset);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_external_dbr_umem(
    void *cq_attr, doca_verbs_umem_t *external_dbr_umem, uint64_t external_dbr_umem_offset);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_external_uar(
    void *cq_attr, doca_verbs_uar_t *external_uar);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_overrun(
    void *cq_attr, enum doca_verbs_cq_overrun overrun);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_cq_collapsed(void *cq_attr, uint8_t cc);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_comp_channel(void *cq_attr,
                                                                         void *compch);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_attr_set_st(void *cq_attr,
                                                               enum doca_verbs_cq_state cq_state);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_create(doca_dev_t *net_dev,
                                                          doca_verbs_cq_attr_t *cq_attr,
                                                          void **verbs_cq);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_destroy(void *verbs_cq);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_get_wq(void *verbs_cq, void **cq_buf,
                                                          uint32_t *cq_num_entries,
                                                          uint8_t *cq_entry_size);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_get_dbr_addr(void *verbs_cq,
                                                                uint64_t **uar_db_reg,
                                                                uint32_t **ci_dbr,
                                                                uint32_t **arm_dbr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_get_cqn(const void *verbs_cq, uint32_t *cqn);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cq_set_cq_context(void *cq, void *cq_context);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_comp_channel_create(void *sdk_context,
                                                                    void **compch);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_comp_channel_destroy(void *compch);

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_get_cq_event(void *compch, void **cq_context);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ack_cq_events(void *verbs_cq, unsigned int nevents);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_SDK_WRAPPER_CQ_H */

/** @} */
