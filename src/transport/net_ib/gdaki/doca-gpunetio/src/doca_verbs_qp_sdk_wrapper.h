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
 * @file doca_verbs_qp_sdk_wrapper.h
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
#ifndef DOCA_VERBS_SDK_WRAPPER_QP_H
#define DOCA_VERBS_SDK_WRAPPER_QP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "host/doca_error.h"
#include "host/doca_verbs.h"
#include "doca_verbs_uar_sdk_wrapper.h"
#include "doca_verbs_umem_sdk_wrapper.h"
#include "doca_verbs_cq_sdk_wrapper.h"

/* Wrapper function declarations */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_create(void **qp_init_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_destroy(void *qp_init_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_pd(void *qp_init_attr,
                                                                    doca_dev_t *net_dev);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_send_cq(void *qp_init_attr,
                                                                         doca_verbs_cq_t *send_cq);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_receive_cq(
    void *qp_init_attr, doca_verbs_cq_t *receive_cq);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_sq_sig_all(void *qp_init_attr,
                                                                            int sq_sig_all);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_sq_wr(void *qp_init_attr,
                                                                       uint32_t sq_wr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_rq_wr(void *qp_init_attr,
                                                                       uint32_t rq_wr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_send_max_sges(
    void *qp_init_attr, uint32_t send_max_sges);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_receive_max_sges(
    void *qp_init_attr, uint32_t receive_max_sges);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_max_inline_data(
    void *qp_init_attr, uint32_t max_inline_data);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_user_index(void *qp_init_attr,
                                                                            uint32_t user_index);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_qp_type(void *qp_init_attr,
                                                                         uint32_t qp_type);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_external_umem(
    void *qp_init_attr, doca_verbs_umem_t *external_umem, uint64_t external_umem_offset);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_external_umem_dbr(
    void *qp_init_attr, doca_verbs_umem_t *external_umem_dbr, uint64_t external_umem_dbr_offset);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_external_uar(
    void *qp_init_attr, doca_verbs_uar_t *external_uar);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_qp_context(void *qp_init_attr,
                                                                            void *qp_context);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_core_direct_master(
    void *qp_init_attr, uint8_t core_direct_master);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_send_dbr_mode(
    void *qp_init_attr, enum doca_verbs_qp_send_dbr_mode send_dbr_mode);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_get_send_dbr_mode(
    void *qp_init_attr, enum doca_verbs_qp_send_dbr_mode *send_dbr_mode);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_ordering_semantic(
    void *qp_init_attr, enum doca_verbs_qp_ordering_semantic ordering_semantic);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_get_ordering_semantic(
    void *qp_init_attr, enum doca_verbs_qp_ordering_semantic *ordering_semantic);

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_create(void **qp_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_destroy(void *qp_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_current_state(
    void *qp_attr, enum doca_verbs_qp_state current_state);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_get_current_state(
    void *qp_attr, enum doca_verbs_qp_state *current_state);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_next_state(
    void *qp_attr, enum doca_verbs_qp_state next_state);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_path_mtu(
    void *qp_attr, enum doca_verbs_mtu_size path_mtu);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_rq_psn(void *qp_attr, uint32_t rq_psn);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_sq_psn(void *qp_attr, uint32_t sq_psn);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_dest_qp_num(void *qp_attr,
                                                                        uint32_t dest_qp_num);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_allow_remote_write(
    void *qp_attr, uint32_t allow_remote_write);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_allow_remote_read(
    void *qp_attr, uint32_t allow_remote_read);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_atomic_mode(
    void *qp_attr, enum doca_verbs_qp_atomic_mode atomic_mode);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_pkey_index(void *qp_attr,
                                                                       uint16_t pkey_index);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_port_num(void *qp_attr,
                                                                     uint16_t port_num);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_ack_timeout(void *qp_attr,
                                                                        uint16_t ack_timeout);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_retry_cnt(void *qp_attr,
                                                                      uint16_t retry_cnt);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_rnr_retry(void *qp_attr,
                                                                      uint16_t rnr_retry);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_min_rnr_timer(void *qp_attr,
                                                                          uint16_t min_rnr_timer);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_max_rd_atomic(void *qp_attr,
                                                                          uint8_t max_rd_atomic);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_max_dest_rd_atomic(
    void *qp_attr, uint8_t max_dest_rd_atomic);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_ah_attr(void *qp_attr,
                                                                    doca_verbs_ah_attr_t *ah_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_cc_group(
    void *qp_attr, doca_verbs_cc_group_t *cc_group);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_get_cc_group(void *qp_attr,
                                                                     void **cc_group);

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_query_cc_group_caps(doca_dev_t *net_dev,
                                                                    void **cc_group_caps);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_caps_get_data(void *cc_group_caps,
                                                                       const void **data,
                                                                       size_t *size);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_caps_free(void *cc_group_caps);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_attr_create(void **verbs_cc_group_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_attr_destroy(void *verbs_cc_group_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_attr_set_hint(void *verbs_cc_group_attr,
                                                                       const void *hint_data,
                                                                       size_t hint_data_size);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_create(doca_dev_t *net_dev,
                                                                void *verbs_cc_group_attr,
                                                                void **verbs_cc_group);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_destroy(void *verbs_cc_group);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_modify(void *verbs_cc_group,
                                                                void *verbs_cc_group_attr);

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_create(doca_dev_t *net_dev, void **ah_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_destroy(void *ah_attr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_gid(void *qp_attr,
                                                                struct doca_verbs_gid gid);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_addr_type(
    void *qp_attr, enum doca_verbs_addr_type addr_type);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_dlid(void *qp_attr, uint32_t dlid);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_sl(void *qp_attr, uint8_t sl);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_sgid_index(void *qp_attr,
                                                                       uint8_t sgid_index);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_static_rate(void *qp_attr,
                                                                        uint8_t static_rate);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_hop_limit(void *qp_attr,
                                                                      uint8_t hop_limit);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_traffic_class(void *qp_attr,
                                                                          uint8_t traffic_class);

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_create(doca_dev_t *net_dev,
                                                          doca_verbs_qp_init_attr_t *qp_init_attr,
                                                          void **verbs_qp);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_destroy(void *verbs_qp);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_modify(void *verbs_qp,
                                                          doca_verbs_qp_attr_t *qp_attr,
                                                          int attr_mask);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_query(void *verbs_qp,
                                                         doca_verbs_qp_attr_t *qp_attr,
                                                         doca_verbs_qp_init_attr_t *qp_init_attr,
                                                         int *attr_mask);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_qpn(void *verbs_qp, uint32_t *qpn);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_dbr_addr(void *verbs_qp, void **dbr_addr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_uar_addr(void *verbs_qp, void **uar_addr);
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_wq(void *verbs_qp, void **sq_buf,
                                                          uint32_t *sq_num_entries, void **rq_buf,
                                                          uint32_t *rq_num_entries,
                                                          uint32_t *rwqe_size_bytes);
#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_SDK_WRAPPER_QP_H */

/** @} */
