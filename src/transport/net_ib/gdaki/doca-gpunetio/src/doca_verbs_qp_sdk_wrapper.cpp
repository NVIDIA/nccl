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

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syslog.h>
#include <string.h>
#include <mutex>

#include "doca_gpunetio_log.hpp"
#include "doca_verbs_uar_sdk_wrapper.h"

/*
 * Enable DOCA SDK log errors on stderr.
 * Disabled by default to avoid extra-prints on screen.
 */
#define DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG 0

#ifdef __cplusplus
extern "C" {
#endif

/* Function pointer types for DOCA Verbs QP SDK APIs */
typedef doca_error_t (*doca_verbs_qp_init_attr_create_t)(void **verbs_qp_init_attr);
typedef doca_error_t (*doca_verbs_qp_init_attr_destroy_t)(void *verbs_qp_init_attr);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_pd_t)(void *verbs_qp_init_attr, void *verbs_pd);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_send_cq_t)(void *verbs_qp_init_attr,
                                                              void *send_cq);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_receive_cq_t)(void *verbs_qp_init_attr,
                                                                 void *receive_cq);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_sq_sig_all_t)(void *verbs_qp_init_attr,
                                                                 int sq_sig_all);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_sq_wr_t)(void *verbs_qp_init_attr,
                                                            uint32_t sq_wr);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_rq_wr_t)(void *verbs_qp_init_attr,
                                                            uint32_t rq_wr);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_send_max_sges_t)(void *verbs_qp_init_attr,
                                                                    uint32_t send_max_sges);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_receive_max_sges_t)(void *verbs_qp_init_attr,
                                                                       uint32_t receive_max_sges);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_max_inline_data_t)(void *verbs_qp_init_attr,
                                                                      uint32_t max_inline_data);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_user_index_t)(void *verbs_qp_init_attr,
                                                                 uint32_t user_index);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_qp_type_t)(void *verbs_qp_init_attr,
                                                              uint32_t qp_type);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_external_umem_t)(void *verbs_qp_init_attr,
                                                                    void *external_umem,
                                                                    uint64_t external_umem_offset);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_external_dbr_umem_t)(
    void *verbs_qp_init_attr, void *external_umem, uint64_t external_umem_offset);
typedef doca_error_t (*doca_verbs_qp_init_attr_get_external_umem_t)(const void *verbs_qp_init_attr,
                                                                    void **external_umem,
                                                                    uint64_t *external_umem_offset);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_external_datapath_en_t)(
    void *verbs_qp_init_attr, uint8_t external_datapath_en);

typedef doca_error_t (*doca_verbs_qp_init_attr_set_external_uar_t)(void *verbs_qp_init_attr,
                                                                   void *external_uar);
typedef doca_error_t (*doca_verbs_qp_init_attr_get_external_uar_t)(const void *verbs_qp_init_attr,
                                                                   void **external_uar);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_qp_context_t)(void *verbs_qp_init_attr,
                                                                 void *qp_context);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_srq_t)(void *verbs_qp_init_attr,
                                                          struct doca_verbs_srq *srq);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_core_direct_master_t)(
    void *verbs_qp_init_attr, uint8_t core_direct_master);
typedef doca_error_t (*doca_verbs_qp_init_attr_set_send_dbr_mode_t)(
    void *verbs_qp_init_attr, enum doca_verbs_qp_send_dbr_mode send_dbr_mode);
typedef enum doca_verbs_qp_send_dbr_mode (*doca_verbs_qp_init_attr_get_send_dbr_mode_t)(
    void *verbs_qp_init_attr);

typedef doca_error_t (*doca_verbs_qp_init_attr_set_ordering_semantic_t)(
    void *verbs_qp_init_attr, enum doca_verbs_qp_ordering_semantic ordering_semantic);
typedef enum doca_verbs_qp_ordering_semantic (*doca_verbs_qp_init_attr_get_ordering_semantic_t)(
    void *verbs_qp_init_attr);

typedef doca_error_t (*doca_verbs_qp_attr_create_t)(void **verbs_qp_attr);
typedef doca_error_t (*doca_verbs_qp_attr_destroy_t)(void *verbs_qp_attr);
typedef doca_error_t (*doca_verbs_qp_attr_set_next_state_t)(void *verbs_qp_attr,
                                                            enum doca_verbs_qp_state next_state);
typedef doca_error_t (*doca_verbs_qp_attr_set_current_state_t)(
    void *verbs_qp_attr, enum doca_verbs_qp_state current_state);
typedef enum doca_verbs_qp_state (*doca_verbs_qp_attr_get_current_state_t)(void *verbs_qp_attr);
typedef doca_error_t (*doca_verbs_qp_attr_set_path_mtu_t)(void *verbs_qp_attr,
                                                          enum doca_verbs_mtu_size path_mtu);
typedef doca_error_t (*doca_verbs_qp_attr_set_rq_psn_t)(void *verbs_qp_attr, uint32_t rq_psn);
typedef doca_error_t (*doca_verbs_qp_attr_set_sq_psn_t)(void *verbs_qp_attr, uint32_t sq_psn);
typedef doca_error_t (*doca_verbs_qp_attr_set_dest_qp_num_t)(void *verbs_qp_attr,
                                                             uint32_t dest_qp_num);
typedef doca_error_t (*doca_verbs_qp_attr_set_allow_remote_write_t)(void *verbs_qp_attr,
                                                                    int allow_remote_write);
typedef doca_error_t (*doca_verbs_qp_attr_set_allow_remote_read_t)(void *verbs_qp_attr,
                                                                   int allow_remote_read);
typedef doca_error_t (*doca_verbs_qp_attr_set_atomic_mode_t)(
    void *verbs_qp_attr, enum doca_verbs_qp_atomic_mode allow_atomic_type);
typedef doca_error_t (*doca_verbs_qp_attr_set_ah_attr_t)(void *verbs_qp_attr, void *ah_attr);
typedef doca_error_t (*doca_verbs_qp_attr_set_pkey_index_t)(void *verbs_qp_attr,
                                                            uint16_t pkey_index);
typedef doca_error_t (*doca_verbs_qp_attr_set_port_num_t)(void *verbs_qp_attr, uint16_t port_num);
typedef doca_error_t (*doca_verbs_qp_attr_set_ack_timeout_t)(void *verbs_qp_attr,
                                                             uint16_t ack_timeout);
typedef doca_error_t (*doca_verbs_qp_attr_set_retry_cnt_t)(void *verbs_qp_attr, uint16_t retry_cnt);
typedef doca_error_t (*doca_verbs_qp_attr_set_rnr_retry_t)(void *verbs_qp_attr, uint16_t rnr_retry);
typedef doca_error_t (*doca_verbs_qp_attr_set_min_rnr_timer_t)(void *verbs_qp_attr,
                                                               uint16_t min_rnr_timer);
typedef doca_error_t (*doca_verbs_qp_attr_set_max_rd_atomic_t)(void *verbs_qp_attr,
                                                               uint8_t max_rd_atomic);
typedef doca_error_t (*doca_verbs_qp_attr_set_max_dest_rd_atomic_t)(void *verbs_qp_attr,
                                                                    uint8_t max_dest_rd_atomic);
typedef doca_error_t (*doca_verbs_qp_attr_set_cc_group_t)(void *verbs_qp_attr, void *cc_group);
typedef void *(*doca_verbs_qp_attr_get_cc_group_t)(const void *verbs_qp_attr);

typedef doca_error_t (*doca_verbs_query_cc_group_caps_t)(void *verbs_context, void **cc_group_caps);
typedef doca_error_t (*doca_verbs_cc_group_caps_get_data_t)(void *cc_group_caps, const void **data,
                                                            size_t *size);
typedef doca_error_t (*doca_verbs_cc_group_caps_free_t)(void *cc_group_caps);
typedef doca_error_t (*doca_verbs_cc_group_attr_create_t)(void **verbs_cc_group_attr);
typedef doca_error_t (*doca_verbs_cc_group_attr_destroy_t)(void *verbs_cc_group_attr);
typedef doca_error_t (*doca_verbs_cc_group_attr_set_hint_t)(void *verbs_cc_group_attr,
                                                            const void *hint_data,
                                                            size_t hint_data_size);
typedef doca_error_t (*doca_verbs_cc_group_create_t)(void *verbs_context, void *verbs_cc_group_attr,
                                                     void **verbs_cc_group);
typedef doca_error_t (*doca_verbs_cc_group_destroy_t)(void *verbs_cc_group);
typedef doca_error_t (*doca_verbs_cc_group_modify_t)(void *verbs_cc_group,
                                                     void *verbs_cc_group_attr);

typedef doca_error_t (*doca_verbs_ah_attr_create_t)(void *verbs_context, void **verbs_ah_attr);
typedef doca_error_t (*doca_verbs_ah_attr_destroy_t)(void *verbs_ah_attr);
typedef doca_error_t (*doca_verbs_ah_attr_set_gid_t)(void *verbs_ah_attr,
                                                     struct doca_verbs_gid gid);
typedef doca_error_t (*doca_verbs_ah_attr_set_addr_type_t)(void *verbs_ah_attr,
                                                           enum doca_verbs_addr_type addr_type);
typedef doca_error_t (*doca_verbs_ah_attr_set_dlid_t)(void *verbs_ah_attr, uint32_t dlid);
typedef doca_error_t (*doca_verbs_ah_attr_set_sl_t)(void *verbs_ah_attr, uint8_t sl);
typedef doca_error_t (*doca_verbs_ah_attr_set_sgid_index_t)(void *verbs_ah_attr,
                                                            uint8_t sgid_index);
typedef doca_error_t (*doca_verbs_ah_attr_set_static_rate_t)(void *verbs_ah_attr,
                                                             uint8_t static_rate);
typedef doca_error_t (*doca_verbs_ah_attr_set_hop_limit_t)(void *verbs_ah_attr, uint8_t hop_limit);
typedef doca_error_t (*doca_verbs_ah_attr_set_traffic_class_t)(void *verbs_ah_attr,
                                                               uint8_t traffic_class);

typedef doca_error_t (*doca_verbs_qp_create_t)(void *verbs_context, void *verbs_qp_init_attr,
                                               void **verbs_qp);
typedef doca_error_t (*doca_verbs_qp_destroy_t)(void *verbs_qp);
typedef doca_error_t (*doca_verbs_qp_modify_t)(void *verbs_qp, void *qp_attr, int attr_mask);
typedef doca_error_t (*doca_verbs_qp_query_t)(void *verbs_qp, void *qp_attr,
                                              void *verbs_qp_init_attr, int *attr_mask);
typedef void (*doca_verbs_qp_get_wq_t)(const void *verbs_qp, void **sq_buf,
                                       uint32_t *sq_num_entries, void **rq_buf,
                                       uint32_t *rq_num_entries, uint32_t *rwqe_size_bytes);
typedef void *(*doca_verbs_qp_init_attr_get_qp_context_t)(const void *verbs_qp_init_attr);
typedef void *(*doca_verbs_qp_get_dbr_addr_t)(const void *verbs_qp);
typedef void *(*doca_verbs_qp_get_uar_addr_t)(const void *verbs_qp);
typedef uint32_t (*doca_verbs_qp_get_qpn_t)(const void *verbs_qp);

typedef doca_error_t (*doca_rdma_bridge_get_dev_pd_t)(void *dev, struct ibv_pd **pd);
typedef doca_error_t (*doca_verbs_bridge_verbs_pd_import_t)(struct ibv_pd *pd, void **verbs_pd);
typedef doca_error_t (*doca_verbs_bridge_verbs_context_import_t)(struct ibv_context *ibv_ctx,
                                                                 uint32_t flags,
                                                                 void **verbs_context);

#if DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG == 1
typedef doca_error_t (*doca_log_backend_create_with_file_sdk_t)(FILE *fptr, void **backend);
#endif

/* Global function pointers */
static doca_verbs_qp_init_attr_create_t p_doca_verbs_qp_init_attr_create = nullptr;
static doca_verbs_qp_init_attr_destroy_t p_doca_verbs_qp_init_attr_destroy = nullptr;
static doca_verbs_qp_init_attr_set_pd_t p_doca_verbs_qp_init_attr_set_pd = nullptr;
static doca_verbs_qp_init_attr_set_send_cq_t p_doca_verbs_qp_init_attr_set_send_cq = nullptr;
static doca_verbs_qp_init_attr_set_receive_cq_t p_doca_verbs_qp_init_attr_set_receive_cq = nullptr;
static doca_verbs_qp_init_attr_set_sq_sig_all_t p_doca_verbs_qp_init_attr_set_sq_sig_all = nullptr;
static doca_verbs_qp_init_attr_set_sq_wr_t p_doca_verbs_qp_init_attr_set_sq_wr = nullptr;
static doca_verbs_qp_init_attr_set_rq_wr_t p_doca_verbs_qp_init_attr_set_rq_wr = nullptr;
static doca_verbs_qp_init_attr_set_send_max_sges_t p_doca_verbs_qp_init_attr_set_send_max_sges =
    nullptr;
static doca_verbs_qp_init_attr_set_receive_max_sges_t
    p_doca_verbs_qp_init_attr_set_receive_max_sges = nullptr;
static doca_verbs_qp_init_attr_set_max_inline_data_t p_doca_verbs_qp_init_attr_set_max_inline_data =
    nullptr;
static doca_verbs_qp_init_attr_set_user_index_t p_doca_verbs_qp_init_attr_set_user_index = nullptr;
static doca_verbs_qp_init_attr_set_qp_type_t p_doca_verbs_qp_init_attr_set_qp_type = nullptr;
static doca_verbs_qp_init_attr_set_external_umem_t p_doca_verbs_qp_init_attr_set_external_umem =
    nullptr;
static doca_verbs_qp_init_attr_set_external_dbr_umem_t
    p_doca_verbs_qp_init_attr_set_external_dbr_umem = nullptr;
static doca_verbs_qp_init_attr_get_external_umem_t p_doca_verbs_qp_init_attr_get_external_umem =
    nullptr;
static doca_verbs_qp_init_attr_set_external_datapath_en_t
    p_doca_verbs_qp_init_attr_set_external_datapath_en = nullptr;
static doca_verbs_qp_init_attr_set_external_uar_t p_doca_verbs_qp_init_attr_set_external_uar =
    nullptr;
static doca_verbs_qp_init_attr_get_external_uar_t p_doca_verbs_qp_init_attr_get_external_uar =
    nullptr;
static doca_verbs_qp_init_attr_set_qp_context_t p_doca_verbs_qp_init_attr_set_qp_context = nullptr;
static doca_verbs_qp_init_attr_set_srq_t p_doca_verbs_qp_init_attr_set_srq = nullptr;
static doca_verbs_qp_init_attr_set_core_direct_master_t
    p_doca_verbs_qp_init_attr_set_core_direct_master = nullptr;
static doca_verbs_qp_init_attr_set_send_dbr_mode_t p_doca_verbs_qp_init_attr_set_send_dbr_mode =
    nullptr;
static doca_verbs_qp_init_attr_get_send_dbr_mode_t p_doca_verbs_qp_init_attr_get_send_dbr_mode =
    nullptr;

static doca_verbs_qp_init_attr_set_ordering_semantic_t
    p_doca_verbs_qp_init_attr_set_ordering_semantic = nullptr;
static doca_verbs_qp_init_attr_get_ordering_semantic_t
    p_doca_verbs_qp_init_attr_get_ordering_semantic = nullptr;

static doca_verbs_qp_attr_create_t p_doca_verbs_qp_attr_create = nullptr;
static doca_verbs_qp_attr_destroy_t p_doca_verbs_qp_attr_destroy = nullptr;
static doca_verbs_qp_attr_set_next_state_t p_doca_verbs_qp_attr_set_next_state = nullptr;
static doca_verbs_qp_attr_set_current_state_t p_doca_verbs_qp_attr_set_current_state = nullptr;
static doca_verbs_qp_attr_get_current_state_t p_doca_verbs_qp_attr_get_current_state = nullptr;
static doca_verbs_qp_attr_set_path_mtu_t p_doca_verbs_qp_attr_set_path_mtu = nullptr;
static doca_verbs_qp_attr_set_rq_psn_t p_doca_verbs_qp_attr_set_rq_psn = nullptr;
static doca_verbs_qp_attr_set_sq_psn_t p_doca_verbs_qp_attr_set_sq_psn = nullptr;
static doca_verbs_qp_attr_set_dest_qp_num_t p_doca_verbs_qp_attr_set_dest_qp_num = nullptr;
static doca_verbs_qp_attr_set_allow_remote_write_t p_doca_verbs_qp_attr_set_allow_remote_write =
    nullptr;
static doca_verbs_qp_attr_set_allow_remote_read_t p_doca_verbs_qp_attr_set_allow_remote_read =
    nullptr;
static doca_verbs_qp_attr_set_atomic_mode_t p_doca_verbs_qp_attr_set_atomic_mode = nullptr;
static doca_verbs_qp_attr_set_ah_attr_t p_doca_verbs_qp_attr_set_ah_attr = nullptr;
static doca_verbs_qp_attr_set_pkey_index_t p_doca_verbs_qp_attr_set_pkey_index = nullptr;
static doca_verbs_qp_attr_set_port_num_t p_doca_verbs_qp_attr_set_port_num = nullptr;
static doca_verbs_qp_attr_set_ack_timeout_t p_doca_verbs_qp_attr_set_ack_timeout = nullptr;
static doca_verbs_qp_attr_set_retry_cnt_t p_doca_verbs_qp_attr_set_retry_cnt = nullptr;
static doca_verbs_qp_attr_set_rnr_retry_t p_doca_verbs_qp_attr_set_rnr_retry = nullptr;
static doca_verbs_qp_attr_set_min_rnr_timer_t p_doca_verbs_qp_attr_set_min_rnr_timer = nullptr;
static doca_verbs_qp_attr_set_max_rd_atomic_t p_doca_verbs_qp_attr_set_max_rd_atomic = nullptr;
static doca_verbs_qp_attr_set_max_dest_rd_atomic_t p_doca_verbs_qp_attr_set_max_dest_rd_atomic =
    nullptr;

static doca_verbs_ah_attr_create_t p_doca_verbs_ah_attr_create = nullptr;
static doca_verbs_ah_attr_destroy_t p_doca_verbs_ah_attr_destroy = nullptr;
static doca_verbs_ah_attr_set_gid_t p_doca_verbs_ah_attr_set_gid = nullptr;
static doca_verbs_ah_attr_set_addr_type_t p_doca_verbs_ah_attr_set_addr_type = nullptr;
static doca_verbs_ah_attr_set_dlid_t p_doca_verbs_ah_attr_set_dlid = nullptr;
static doca_verbs_ah_attr_set_sl_t p_doca_verbs_ah_attr_set_sl = nullptr;
static doca_verbs_ah_attr_set_sgid_index_t p_doca_verbs_ah_attr_set_sgid_index = nullptr;
static doca_verbs_ah_attr_set_static_rate_t p_doca_verbs_ah_attr_set_static_rate = nullptr;
static doca_verbs_ah_attr_set_hop_limit_t p_doca_verbs_ah_attr_set_hop_limit = nullptr;
static doca_verbs_ah_attr_set_traffic_class_t p_doca_verbs_ah_attr_set_traffic_class = nullptr;

static doca_verbs_qp_create_t p_doca_verbs_qp_create = nullptr;
static doca_verbs_qp_destroy_t p_doca_verbs_qp_destroy = nullptr;
static doca_verbs_qp_modify_t p_doca_verbs_qp_modify = nullptr;
static doca_verbs_qp_query_t p_doca_verbs_qp_query = nullptr;
static doca_verbs_qp_get_wq_t p_doca_verbs_qp_get_wq = nullptr;
static doca_verbs_qp_init_attr_get_qp_context_t p_doca_verbs_qp_init_attr_get_qp_context = nullptr;
static doca_verbs_qp_get_dbr_addr_t p_doca_verbs_qp_get_dbr_addr = nullptr;
static doca_verbs_qp_get_uar_addr_t p_doca_verbs_qp_get_uar_addr = nullptr;
static doca_verbs_qp_get_qpn_t p_doca_verbs_qp_get_qpn = nullptr;

static doca_rdma_bridge_get_dev_pd_t p_doca_rdma_bridge_get_dev_pd = nullptr;
static doca_verbs_bridge_verbs_pd_import_t p_doca_verbs_bridge_verbs_pd_import = nullptr;
static doca_verbs_bridge_verbs_context_import_t p_doca_verbs_bridge_verbs_context_import = nullptr;

#if DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG == 1
static doca_log_backend_create_with_file_sdk_t p_doca_log_backend_create_with_file_sdk = nullptr;
#endif

static doca_verbs_qp_attr_set_cc_group_t p_doca_verbs_qp_attr_set_cc_group = nullptr;
static doca_verbs_qp_attr_get_cc_group_t p_doca_verbs_qp_attr_get_cc_group = nullptr;
static doca_verbs_query_cc_group_caps_t p_doca_verbs_query_cc_group_caps = nullptr;
static doca_verbs_cc_group_caps_get_data_t p_doca_verbs_cc_group_caps_get_data = nullptr;
static doca_verbs_cc_group_caps_free_t p_doca_verbs_cc_group_caps_free = nullptr;
static doca_verbs_cc_group_attr_create_t p_doca_verbs_cc_group_attr_create = nullptr;
static doca_verbs_cc_group_attr_destroy_t p_doca_verbs_cc_group_attr_destroy = nullptr;
static doca_verbs_cc_group_attr_set_hint_t p_doca_verbs_cc_group_attr_set_hint = nullptr;
static doca_verbs_cc_group_create_t p_doca_verbs_cc_group_create = nullptr;
static doca_verbs_cc_group_destroy_t p_doca_verbs_cc_group_destroy = nullptr;
static doca_verbs_cc_group_modify_t p_doca_verbs_cc_group_modify = nullptr;

static void *common_handle = nullptr;
static void *verbs_handle = nullptr;

/* Helper function to get function pointer from libcuda */
static void *get_verbs_sdk_symbol(const char *symbol_name) {
    void *symbol = dlsym(verbs_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void *get_common_sdk_symbol(const char *symbol_name) {
    void *symbol = dlsym(common_handle, symbol_name);
    if (!symbol) {
        DOCA_LOG(LOG_ERR, "Failed to get symbol %s: %s\n", symbol_name, dlerror());
        return nullptr;
    }
    return symbol;
}

static void doca_verbs_sdk_wrapper_init(int *ret) {
    char libcommon_path[doca_sdk_path_length];
    char libverbs_path[doca_sdk_path_length];

    memset(libcommon_path, '\0', doca_sdk_path_length);
    memset(libverbs_path, '\0', doca_sdk_path_length);

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libcommon_path, doca_sdk_path_length, "%s", "libdoca_common.so");
    else
        snprintf(libcommon_path, doca_sdk_path_length, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_common.so");

    if (getenv(DOCA_SDK_LIB_PATH_ENV_VAR) == NULL)
        snprintf(libverbs_path, doca_sdk_path_length, "%s", "libdoca_verbs.so");
    else
        snprintf(libverbs_path, doca_sdk_path_length, "%s/%s", getenv(DOCA_SDK_LIB_PATH_ENV_VAR),
                 "libdoca_verbs.so");

    common_handle = dlopen(libcommon_path, RTLD_NOW | RTLD_GLOBAL);
    if (!common_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_common.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libcommon_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    verbs_handle = dlopen(libverbs_path, RTLD_NOW | RTLD_LOCAL);
    if (!verbs_handle) {
        DOCA_LOG(LOG_ERR, "Failed to find libdoca_verbs.so library %s (DOCA_SDK_LIB_PATH=%s)",
                 libverbs_path, getenv(DOCA_SDK_LIB_PATH_ENV_VAR));

        *ret = -1;
        goto exit_error;
    }

    /* Get function pointers */
    p_doca_verbs_qp_init_attr_create =
        (doca_verbs_qp_init_attr_create_t)get_verbs_sdk_symbol("doca_verbs_qp_init_attr_create");
    p_doca_verbs_qp_init_attr_destroy =
        (doca_verbs_qp_init_attr_destroy_t)get_verbs_sdk_symbol("doca_verbs_qp_init_attr_destroy");
    p_doca_verbs_qp_init_attr_set_pd =
        (doca_verbs_qp_init_attr_set_pd_t)get_verbs_sdk_symbol("doca_verbs_qp_init_attr_set_pd");
    p_doca_verbs_qp_init_attr_set_send_cq =
        (doca_verbs_qp_init_attr_set_send_cq_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_send_cq");
    p_doca_verbs_qp_init_attr_set_receive_cq =
        (doca_verbs_qp_init_attr_set_receive_cq_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_receive_cq");
    p_doca_verbs_qp_init_attr_set_sq_sig_all =
        (doca_verbs_qp_init_attr_set_sq_sig_all_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_sq_sig_all");
    p_doca_verbs_qp_init_attr_set_sq_wr = (doca_verbs_qp_init_attr_set_sq_wr_t)get_verbs_sdk_symbol(
        "doca_verbs_qp_init_attr_set_sq_wr");
    p_doca_verbs_qp_init_attr_set_rq_wr = (doca_verbs_qp_init_attr_set_rq_wr_t)get_verbs_sdk_symbol(
        "doca_verbs_qp_init_attr_set_rq_wr");
    p_doca_verbs_qp_init_attr_set_send_max_sges =
        (doca_verbs_qp_init_attr_set_send_max_sges_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_send_max_sges");
    p_doca_verbs_qp_init_attr_set_receive_max_sges =
        (doca_verbs_qp_init_attr_set_receive_max_sges_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_receive_max_sges");
    p_doca_verbs_qp_init_attr_set_max_inline_data =
        (doca_verbs_qp_init_attr_set_max_inline_data_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_max_inline_data");
    p_doca_verbs_qp_init_attr_set_user_index =
        (doca_verbs_qp_init_attr_set_user_index_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_user_index");
    p_doca_verbs_qp_init_attr_set_qp_type =
        (doca_verbs_qp_init_attr_set_qp_type_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_qp_type");
    p_doca_verbs_qp_init_attr_set_external_umem =
        (doca_verbs_qp_init_attr_set_external_umem_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_external_umem");
    p_doca_verbs_qp_init_attr_set_external_dbr_umem =
        (doca_verbs_qp_init_attr_set_external_dbr_umem_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_external_dbr_umem");
    p_doca_verbs_qp_init_attr_get_external_umem =
        (doca_verbs_qp_init_attr_get_external_umem_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_get_external_umem");
    p_doca_verbs_qp_init_attr_set_external_datapath_en =
        (doca_verbs_qp_init_attr_set_external_datapath_en_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_external_datapath_en");
    p_doca_verbs_qp_init_attr_set_external_uar =
        (doca_verbs_qp_init_attr_set_external_uar_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_external_uar");
    p_doca_verbs_qp_init_attr_get_external_uar =
        (doca_verbs_qp_init_attr_get_external_uar_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_get_external_uar");
    p_doca_verbs_qp_init_attr_set_qp_context =
        (doca_verbs_qp_init_attr_set_qp_context_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_qp_context");
    p_doca_verbs_qp_init_attr_set_srq =
        (doca_verbs_qp_init_attr_set_srq_t)get_verbs_sdk_symbol("doca_verbs_qp_init_attr_set_srq");
    p_doca_verbs_qp_init_attr_set_core_direct_master =
        (doca_verbs_qp_init_attr_set_core_direct_master_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_core_direct_master");
    p_doca_verbs_qp_init_attr_set_send_dbr_mode =
        (doca_verbs_qp_init_attr_set_send_dbr_mode_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_send_dbr_mode");
    p_doca_verbs_qp_init_attr_get_send_dbr_mode =
        (doca_verbs_qp_init_attr_get_send_dbr_mode_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_get_send_dbr_mode");

    // DDP setter
    p_doca_verbs_qp_init_attr_set_ordering_semantic =
        (doca_verbs_qp_init_attr_set_ordering_semantic_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_set_ordering_semantic");
    p_doca_verbs_qp_init_attr_get_ordering_semantic =
        (doca_verbs_qp_init_attr_get_ordering_semantic_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_get_ordering_semantic");

    p_doca_verbs_qp_attr_create =
        (doca_verbs_qp_attr_create_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_create");
    p_doca_verbs_qp_attr_destroy =
        (doca_verbs_qp_attr_destroy_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_destroy");
    p_doca_verbs_qp_attr_set_next_state = (doca_verbs_qp_attr_set_next_state_t)get_verbs_sdk_symbol(
        "doca_verbs_qp_attr_set_next_state");
    p_doca_verbs_qp_attr_set_current_state =
        (doca_verbs_qp_attr_set_current_state_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_current_state");
    p_doca_verbs_qp_attr_get_current_state =
        (doca_verbs_qp_attr_get_current_state_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_get_current_state");
    p_doca_verbs_qp_attr_set_path_mtu =
        (doca_verbs_qp_attr_set_path_mtu_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_set_path_mtu");
    p_doca_verbs_qp_attr_set_rq_psn =
        (doca_verbs_qp_attr_set_rq_psn_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_set_rq_psn");
    p_doca_verbs_qp_attr_set_sq_psn =
        (doca_verbs_qp_attr_set_sq_psn_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_set_sq_psn");
    p_doca_verbs_qp_attr_set_dest_qp_num =
        (doca_verbs_qp_attr_set_dest_qp_num_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_dest_qp_num");
    p_doca_verbs_qp_attr_set_allow_remote_write =
        (doca_verbs_qp_attr_set_allow_remote_write_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_allow_remote_write");
    p_doca_verbs_qp_attr_set_allow_remote_read =
        (doca_verbs_qp_attr_set_allow_remote_read_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_allow_remote_read");
    p_doca_verbs_qp_attr_set_atomic_mode =
        (doca_verbs_qp_attr_set_atomic_mode_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_atomic_mode");
    p_doca_verbs_qp_attr_set_ah_attr =
        (doca_verbs_qp_attr_set_ah_attr_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_set_ah_attr");
    p_doca_verbs_qp_attr_set_pkey_index = (doca_verbs_qp_attr_set_pkey_index_t)get_verbs_sdk_symbol(
        "doca_verbs_qp_attr_set_pkey_index");
    p_doca_verbs_qp_attr_set_port_num =
        (doca_verbs_qp_attr_set_port_num_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_set_port_num");
    p_doca_verbs_qp_attr_set_ack_timeout =
        (doca_verbs_qp_attr_set_ack_timeout_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_ack_timeout");
    p_doca_verbs_qp_attr_set_retry_cnt = (doca_verbs_qp_attr_set_retry_cnt_t)get_verbs_sdk_symbol(
        "doca_verbs_qp_attr_set_retry_cnt");
    p_doca_verbs_qp_attr_set_rnr_retry = (doca_verbs_qp_attr_set_rnr_retry_t)get_verbs_sdk_symbol(
        "doca_verbs_qp_attr_set_rnr_retry");
    p_doca_verbs_qp_attr_set_min_rnr_timer =
        (doca_verbs_qp_attr_set_min_rnr_timer_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_min_rnr_timer");
    p_doca_verbs_qp_attr_set_max_rd_atomic =
        (doca_verbs_qp_attr_set_max_rd_atomic_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_max_rd_atomic");
    p_doca_verbs_qp_attr_set_max_dest_rd_atomic =
        (doca_verbs_qp_attr_set_max_dest_rd_atomic_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_attr_set_max_dest_rd_atomic");

    p_doca_verbs_qp_attr_set_cc_group =
        (doca_verbs_qp_attr_set_cc_group_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_set_cc_group");
    p_doca_verbs_qp_attr_get_cc_group =
        (doca_verbs_qp_attr_get_cc_group_t)get_verbs_sdk_symbol("doca_verbs_qp_attr_get_cc_group");
    p_doca_verbs_query_cc_group_caps =
        (doca_verbs_query_cc_group_caps_t)get_verbs_sdk_symbol("doca_verbs_query_cc_group_caps");
    p_doca_verbs_cc_group_caps_get_data = (doca_verbs_cc_group_caps_get_data_t)get_verbs_sdk_symbol(
        "doca_verbs_cc_group_caps_get_data");
    p_doca_verbs_cc_group_caps_free =
        (doca_verbs_cc_group_caps_free_t)get_verbs_sdk_symbol("doca_verbs_cc_group_caps_free");
    p_doca_verbs_cc_group_attr_create =
        (doca_verbs_cc_group_attr_create_t)get_verbs_sdk_symbol("doca_verbs_cc_group_attr_create");
    p_doca_verbs_cc_group_attr_destroy = (doca_verbs_cc_group_attr_destroy_t)get_verbs_sdk_symbol(
        "doca_verbs_cc_group_attr_destroy");
    p_doca_verbs_cc_group_attr_set_hint = (doca_verbs_cc_group_attr_set_hint_t)get_verbs_sdk_symbol(
        "doca_verbs_cc_group_attr_set_hint");
    p_doca_verbs_cc_group_create =
        (doca_verbs_cc_group_create_t)get_verbs_sdk_symbol("doca_verbs_cc_group_create");
    p_doca_verbs_cc_group_destroy =
        (doca_verbs_cc_group_destroy_t)get_verbs_sdk_symbol("doca_verbs_cc_group_destroy");
    p_doca_verbs_cc_group_modify =
        (doca_verbs_cc_group_modify_t)get_verbs_sdk_symbol("doca_verbs_cc_group_modify");

    p_doca_verbs_ah_attr_create =
        (doca_verbs_ah_attr_create_t)get_verbs_sdk_symbol("doca_verbs_ah_attr_create");
    p_doca_verbs_ah_attr_destroy =
        (doca_verbs_ah_attr_destroy_t)get_verbs_sdk_symbol("doca_verbs_ah_attr_destroy");
    p_doca_verbs_ah_attr_set_gid =
        (doca_verbs_ah_attr_set_gid_t)get_verbs_sdk_symbol("doca_verbs_ah_attr_set_gid");
    p_doca_verbs_ah_attr_set_addr_type = (doca_verbs_ah_attr_set_addr_type_t)get_verbs_sdk_symbol(
        "doca_verbs_ah_attr_set_addr_type");
    p_doca_verbs_ah_attr_set_dlid =
        (doca_verbs_ah_attr_set_dlid_t)get_verbs_sdk_symbol("doca_verbs_ah_attr_set_dlid");
    p_doca_verbs_ah_attr_set_sl =
        (doca_verbs_ah_attr_set_sl_t)get_verbs_sdk_symbol("doca_verbs_ah_attr_set_sl");
    p_doca_verbs_ah_attr_set_sgid_index = (doca_verbs_ah_attr_set_sgid_index_t)get_verbs_sdk_symbol(
        "doca_verbs_ah_attr_set_sgid_index");
    p_doca_verbs_ah_attr_set_static_rate =
        (doca_verbs_ah_attr_set_static_rate_t)get_verbs_sdk_symbol(
            "doca_verbs_ah_attr_set_static_rate");
    p_doca_verbs_ah_attr_set_hop_limit = (doca_verbs_ah_attr_set_hop_limit_t)get_verbs_sdk_symbol(
        "doca_verbs_ah_attr_set_hop_limit");
    p_doca_verbs_ah_attr_set_traffic_class =
        (doca_verbs_ah_attr_set_traffic_class_t)get_verbs_sdk_symbol(
            "doca_verbs_ah_attr_set_traffic_class");

    p_doca_verbs_qp_create = (doca_verbs_qp_create_t)get_verbs_sdk_symbol("doca_verbs_qp_create");
    p_doca_verbs_qp_destroy =
        (doca_verbs_qp_destroy_t)get_verbs_sdk_symbol("doca_verbs_qp_destroy");
    p_doca_verbs_qp_modify = (doca_verbs_qp_modify_t)get_verbs_sdk_symbol("doca_verbs_qp_modify");
    p_doca_verbs_qp_query = (doca_verbs_qp_query_t)get_verbs_sdk_symbol("doca_verbs_qp_query");
    p_doca_verbs_qp_get_wq = (doca_verbs_qp_get_wq_t)get_verbs_sdk_symbol("doca_verbs_qp_get_wq");
    p_doca_verbs_qp_init_attr_get_qp_context =
        (doca_verbs_qp_init_attr_get_qp_context_t)get_verbs_sdk_symbol(
            "doca_verbs_qp_init_attr_get_qp_context");
    p_doca_verbs_qp_get_dbr_addr =
        (doca_verbs_qp_get_dbr_addr_t)get_verbs_sdk_symbol("doca_verbs_qp_get_dbr_addr");
    p_doca_verbs_qp_get_uar_addr =
        (doca_verbs_qp_get_uar_addr_t)get_verbs_sdk_symbol("doca_verbs_qp_get_uar_addr");
    p_doca_verbs_qp_get_qpn =
        (doca_verbs_qp_get_qpn_t)get_verbs_sdk_symbol("doca_verbs_qp_get_qpn");

    p_doca_rdma_bridge_get_dev_pd =
        (doca_rdma_bridge_get_dev_pd_t)get_common_sdk_symbol("doca_rdma_bridge_get_dev_pd");
    p_doca_verbs_bridge_verbs_pd_import = (doca_verbs_bridge_verbs_pd_import_t)get_verbs_sdk_symbol(
        "doca_verbs_bridge_verbs_pd_import");
    p_doca_verbs_bridge_verbs_context_import =
        (doca_verbs_bridge_verbs_context_import_t)get_verbs_sdk_symbol(
            "doca_verbs_bridge_verbs_context_import");

#if DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG == 1
    p_doca_log_backend_create_with_file_sdk =
        (doca_log_backend_create_with_file_sdk_t)get_verbs_sdk_symbol(
            "doca_log_backend_create_with_file_sdk");
#endif

    /* Check if all symbols were found */
    /*
     * Symbols p_doca_verbs_qp_init_attr_set_send_dbr_mode and
     * p_doca_verbs_qp_init_attr_get_send_dbr_mode are optional as not present in DOCA 3.2 LTS
     * version.
     */
    if (!p_doca_verbs_qp_init_attr_create || !p_doca_verbs_qp_init_attr_destroy ||
        !p_doca_verbs_qp_init_attr_set_pd || !p_doca_verbs_qp_init_attr_set_send_cq ||
        !p_doca_verbs_qp_init_attr_set_receive_cq || !p_doca_verbs_qp_init_attr_set_sq_sig_all ||
        !p_doca_verbs_qp_init_attr_set_sq_wr || !p_doca_verbs_qp_init_attr_set_rq_wr ||
        !p_doca_verbs_qp_init_attr_set_send_max_sges ||
        !p_doca_verbs_qp_init_attr_set_receive_max_sges ||
        !p_doca_verbs_qp_init_attr_set_max_inline_data ||
        !p_doca_verbs_qp_init_attr_set_user_index || !p_doca_verbs_qp_init_attr_set_qp_type ||
        !p_doca_verbs_qp_init_attr_set_external_umem ||
        !p_doca_verbs_qp_init_attr_set_external_dbr_umem ||
        !p_doca_verbs_qp_init_attr_get_external_umem ||
        !p_doca_verbs_qp_init_attr_set_external_datapath_en ||
        !p_doca_verbs_qp_init_attr_set_external_uar ||
        !p_doca_verbs_qp_init_attr_get_external_uar || !p_doca_verbs_qp_init_attr_set_qp_context ||
        !p_doca_verbs_qp_init_attr_set_srq || !p_doca_verbs_qp_init_attr_set_core_direct_master ||
        !p_doca_verbs_qp_attr_create || !p_doca_verbs_qp_attr_destroy ||
        !p_doca_verbs_qp_attr_set_next_state || !p_doca_verbs_qp_attr_set_current_state ||
        !p_doca_verbs_qp_attr_get_current_state || !p_doca_verbs_qp_attr_set_path_mtu ||
        !p_doca_verbs_qp_attr_set_rq_psn || !p_doca_verbs_qp_attr_set_sq_psn ||
        !p_doca_verbs_qp_attr_set_dest_qp_num || !p_doca_verbs_qp_attr_set_allow_remote_write ||
        !p_doca_verbs_qp_attr_set_allow_remote_read || !p_doca_verbs_qp_attr_set_atomic_mode ||
        !p_doca_verbs_qp_attr_set_ah_attr || !p_doca_verbs_qp_attr_set_pkey_index ||
        !p_doca_verbs_qp_attr_set_port_num || !p_doca_verbs_qp_attr_set_ack_timeout ||
        !p_doca_verbs_qp_attr_set_retry_cnt || !p_doca_verbs_qp_attr_set_rnr_retry ||
        !p_doca_verbs_qp_attr_set_min_rnr_timer || !p_doca_verbs_qp_attr_set_max_rd_atomic ||
        !p_doca_verbs_qp_attr_set_max_dest_rd_atomic ||

        !p_doca_verbs_ah_attr_create || !p_doca_verbs_ah_attr_destroy ||
        !p_doca_verbs_ah_attr_set_gid || !p_doca_verbs_ah_attr_set_addr_type ||
        !p_doca_verbs_ah_attr_set_dlid || !p_doca_verbs_ah_attr_set_sl ||
        !p_doca_verbs_ah_attr_set_sgid_index || !p_doca_verbs_ah_attr_set_static_rate ||
        !p_doca_verbs_ah_attr_set_hop_limit || !p_doca_verbs_ah_attr_set_traffic_class ||

        !p_doca_verbs_qp_create || !p_doca_verbs_qp_destroy || !p_doca_verbs_qp_modify ||
        !p_doca_verbs_qp_query || !p_doca_verbs_qp_get_wq ||
        !p_doca_verbs_qp_init_attr_get_qp_context || !p_doca_verbs_qp_get_dbr_addr ||
        !p_doca_verbs_qp_get_uar_addr || !p_doca_verbs_qp_get_qpn) {
        DOCA_LOG(LOG_ERR, "Failed to get all required DOCA Verbs Dev SDK symbols\n");
        dlclose(verbs_handle);
        verbs_handle = nullptr;
        *ret = -1;
        goto exit_error;
    }

    if (!p_doca_verbs_qp_init_attr_set_ordering_semantic ||
        !p_doca_verbs_qp_init_attr_get_ordering_semantic)
        DOCA_LOG(LOG_WARNING,
                 "The DOCA SDK installed on the system doesn't provide symbols for the set/get "
                 "ordering semantic\n");

    if (!p_doca_verbs_qp_attr_set_cc_group || !p_doca_verbs_qp_attr_get_cc_group)
        DOCA_LOG(LOG_WARNING,
                 "The DOCA SDK installed on the system doesn't provide symbols for QP CC group "
                 "set/get\n");

    if (!p_doca_verbs_query_cc_group_caps || !p_doca_verbs_cc_group_caps_get_data ||
        !p_doca_verbs_cc_group_caps_free || !p_doca_verbs_cc_group_attr_create ||
        !p_doca_verbs_cc_group_attr_destroy || !p_doca_verbs_cc_group_attr_set_hint ||
        !p_doca_verbs_cc_group_create || !p_doca_verbs_cc_group_destroy ||
        !p_doca_verbs_cc_group_modify)
        DOCA_LOG(LOG_WARNING,
                 "The DOCA SDK installed on the system doesn't provide CC group symbols "
                 "(doca_verbs_cc_group_* / query_cc_group_caps)\n");

#if DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG == 1
    if (!p_doca_log_backend_create_with_file_sdk) {
        DOCA_LOG(LOG_ERR,
                 "Failed to get doca_log_backend_create_with_file_sdk DOCA Verbs Dev SDK symbol\n");
        dlclose(verbs_handle);
        verbs_handle = nullptr;
        *ret = -1;
        goto exit_error;
    }
#endif

    *ret = 0;
    return;

exit_error:
    if (verbs_handle) dlclose(verbs_handle);
    if (common_handle) dlclose(common_handle);
    verbs_handle = nullptr;
    common_handle = nullptr;
    return;
}

static int init_verbs_sdk_wrapper(void) {
    static int ret = 0;
    static std::once_flag once;
    std::call_once(once, doca_verbs_sdk_wrapper_init, &ret);
    return ret;
}

static int get_sdk_wrapper_env_var(void) {
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (!val) return -1;

    if (strcmp(val, "") == 0) return -1;

    return 1;
}

/* Wrapper function implementations -- QP Init Attr */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_create(void **qp_init_attr) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);
#if DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG == 1
    void *sdk_log;
#endif

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

#if DOCA_VERBS_QP_SDK_WRAPPER_ENABLE_DEBUG == 1
        doca_err = p_doca_log_backend_create_with_file_sdk(stderr, &sdk_log);
        if (doca_err != DOCA_SUCCESS) {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
#endif

        doca_err = p_doca_verbs_qp_init_attr_create(qp_init_attr);
        if (doca_err == DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH set to %s. DOCA SDK is in use", val);
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else {
        DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH not set %s. DOCA SDK is not in use", val);
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
    }

    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_destroy(void *qp_init_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_destroy(qp_init_attr);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_pd(void *qp_init_attr,
                                                                    doca_dev_t *net_dev) {
    doca_error_t doca_err;

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't set QP pd, net_dev is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (net_dev->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->sdk_pd == nullptr || net_dev->sdk_context == nullptr) {
            DOCA_LOG(LOG_ERR, "doca_dev_t has no SDK elements.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_pd(qp_init_attr, net_dev->sdk_pd);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_send_cq(void *qp_init_attr,
                                                                         doca_verbs_cq_t *send_cq) {
    doca_error_t doca_err;

    if (send_cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't set QP pd, send_cq is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (send_cq->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_send_cq(qp_init_attr, send_cq->sdk);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_receive_cq(
    void *qp_init_attr, doca_verbs_cq_t *receive_cq) {
    doca_error_t doca_err;

    if (receive_cq == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't set QP pd, receive_cq is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (receive_cq->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_receive_cq(qp_init_attr, receive_cq->sdk);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_sq_sig_all(void *qp_init_attr,
                                                                            int sq_sig_all) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_sq_sig_all(qp_init_attr, sq_sig_all);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_sq_wr(void *qp_init_attr,
                                                                       uint32_t sq_wr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_sq_wr(qp_init_attr, sq_wr);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_rq_wr(void *qp_init_attr,
                                                                       uint32_t rq_wr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_rq_wr(qp_init_attr, rq_wr);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_send_max_sges(
    void *qp_init_attr, uint32_t send_max_sges) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_send_max_sges(qp_init_attr, send_max_sges);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_receive_max_sges(
    void *qp_init_attr, uint32_t receive_max_sges) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_receive_max_sges(qp_init_attr, receive_max_sges);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_max_inline_data(
    void *qp_init_attr, uint32_t max_inline_data) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_max_inline_data(qp_init_attr, max_inline_data);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_user_index(void *qp_init_attr,
                                                                            uint32_t user_index) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_user_index(qp_init_attr, user_index);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_qp_type(void *qp_init_attr,
                                                                         uint32_t qp_type) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_qp_type(qp_init_attr, qp_type);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_external_umem(
    void *qp_init_attr, doca_verbs_umem_t *external_umem, uint64_t external_umem_offset) {
    doca_error_t doca_err;

    if (external_umem == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't set QP external umem, external_umem is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (external_umem->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_umem_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_external_umem(qp_init_attr, external_umem->sdk,
                                                               external_umem_offset);
        if (doca_err == DOCA_SUCCESS) {
            doca_err = p_doca_verbs_qp_init_attr_set_external_datapath_en(qp_init_attr, 1);
            if (doca_err == DOCA_SUCCESS)
                return DOCA_SDK_WRAPPER_SUCCESS;
            else {
                DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
                return DOCA_SDK_WRAPPER_API_ERROR;
            }
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_external_umem_dbr(
    void *qp_init_attr, doca_verbs_umem_t *external_umem_dbr, uint64_t external_umem_dbr_offset) {
    doca_error_t doca_err;

    if (external_umem_dbr == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't set QP external umem, external_umem_dbr is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (external_umem_dbr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_umem_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_external_dbr_umem(
            qp_init_attr, external_umem_dbr->sdk, external_umem_dbr_offset);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_external_uar(
    void *qp_init_attr, doca_verbs_uar_t *external_uar) {
    doca_error_t doca_err;

    if (external_uar == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't set QP external uar, external_uar is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (external_uar->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_uar_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_external_uar(qp_init_attr, external_uar->sdk);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_qp_context(void *qp_init_attr,
                                                                            void *qp_context) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_init_attr_set_qp_context(qp_init_attr, qp_context);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_core_direct_master(
    void *qp_init_attr, uint8_t core_direct_master) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err =
            p_doca_verbs_qp_init_attr_set_core_direct_master(qp_init_attr, core_direct_master);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_send_dbr_mode(
    void *qp_init_attr, enum doca_verbs_qp_send_dbr_mode send_dbr_mode) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_qp_init_attr_set_send_dbr_mode) return DOCA_SDK_WRAPPER_NOT_SUPPORTED;

        doca_err = p_doca_verbs_qp_init_attr_set_send_dbr_mode(qp_init_attr, send_dbr_mode);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_get_send_dbr_mode(
    void *qp_init_attr, enum doca_verbs_qp_send_dbr_mode *send_dbr_mode) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_qp_init_attr_get_send_dbr_mode) return DOCA_SDK_WRAPPER_NOT_SUPPORTED;

        *send_dbr_mode = p_doca_verbs_qp_init_attr_get_send_dbr_mode(qp_init_attr);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_set_ordering_semantic(
    void *qp_init_attr, enum doca_verbs_qp_ordering_semantic ordering_semantic) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_qp_init_attr_set_ordering_semantic) {
            DOCA_LOG(LOG_WARNING,
                     "The DOCA SDK installed on the system doesn't provide symbols for the "
                     "set_ordering_semantic\n");
            return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
        }

        doca_err = p_doca_verbs_qp_init_attr_set_ordering_semantic(qp_init_attr, ordering_semantic);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_init_attr_get_ordering_semantic(
    void *qp_init_attr, enum doca_verbs_qp_ordering_semantic *ordering_semantic) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (!p_doca_verbs_qp_init_attr_get_ordering_semantic) {
            DOCA_LOG(LOG_WARNING,
                     "The DOCA SDK installed on the system doesn't provide symbols for the "
                     "get_ordering_semantic\n");
            return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
        }

        *ordering_semantic = p_doca_verbs_qp_init_attr_get_ordering_semantic(qp_init_attr);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

/* Wrapper function implementations -- QP Attr */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_create(void **qp_attr) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_attr_create(qp_attr);
        if (doca_err == DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH set to %s. DOCA SDK is in use", val);
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else {
        DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH not set %s. DOCA SDK is not in use", val);
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
    }

    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_destroy(void *qp_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_destroy(qp_attr);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_current_state(
    void *qp_attr, enum doca_verbs_qp_state current_state) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_current_state(qp_attr, current_state);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_get_current_state(
    void *qp_attr, enum doca_verbs_qp_state *current_state) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        *current_state = p_doca_verbs_qp_attr_get_current_state(qp_attr);
        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_next_state(
    void *qp_attr, enum doca_verbs_qp_state next_state) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_next_state(qp_attr, next_state);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_path_mtu(
    void *qp_attr, enum doca_verbs_mtu_size path_mtu) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_path_mtu(qp_attr, path_mtu);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_rq_psn(void *qp_attr, uint32_t rq_psn) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_rq_psn(qp_attr, rq_psn);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_sq_psn(void *qp_attr, uint32_t sq_psn) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_sq_psn(qp_attr, sq_psn);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_dest_qp_num(void *qp_attr,
                                                                        uint32_t dest_qp_num) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_dest_qp_num(qp_attr, dest_qp_num);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_allow_remote_write(
    void *qp_attr, uint32_t allow_remote_write) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_allow_remote_write(qp_attr, allow_remote_write);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_allow_remote_read(
    void *qp_attr, uint32_t allow_remote_read) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_allow_remote_read(qp_attr, allow_remote_read);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_atomic_mode(
    void *qp_attr, enum doca_verbs_qp_atomic_mode atomic_mode) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_atomic_mode(qp_attr, atomic_mode);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_pkey_index(void *qp_attr,
                                                                       uint16_t pkey_index) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_pkey_index(qp_attr, pkey_index);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_port_num(void *qp_attr,
                                                                     uint16_t port_num) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_port_num(qp_attr, port_num);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_ack_timeout(void *qp_attr,
                                                                        uint16_t ack_timeout) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_ack_timeout(qp_attr, ack_timeout);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_retry_cnt(void *qp_attr,
                                                                      uint16_t retry_cnt) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_retry_cnt(qp_attr, retry_cnt);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_rnr_retry(void *qp_attr,
                                                                      uint16_t rnr_retry) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_rnr_retry(qp_attr, rnr_retry);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_min_rnr_timer(void *qp_attr,
                                                                          uint16_t min_rnr_timer) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_min_rnr_timer(qp_attr, min_rnr_timer);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_max_rd_atomic(void *qp_attr,
                                                                          uint8_t max_rd_atomic) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_max_rd_atomic(qp_attr, max_rd_atomic);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_max_dest_rd_atomic(
    void *qp_attr, uint8_t max_dest_rd_atomic) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_attr_set_max_dest_rd_atomic(qp_attr, max_dest_rd_atomic);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_ah_attr(void *qp_attr,
                                                                    doca_verbs_ah_attr_t *ah_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (ah_attr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_ah_attr_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_attr_set_ah_attr(qp_attr, ah_attr->sdk);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

/* Wrapper function implementations -- QP AH attr */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_create(doca_dev_t *net_dev,
                                                               void **ah_attr) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create CQ, net_dev is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->sdk_context == nullptr) {
            DOCA_LOG(LOG_ERR, "doca_dev_t has no SDK elements.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_ah_attr_create(net_dev->sdk_context, ah_attr);
        if (doca_err == DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH set to %s. DOCA SDK is in use", val);
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else {
        DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH not set %s. DOCA SDK is not in use", val);
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
    }

    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_destroy(void *ah_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_destroy(ah_attr);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_gid(void *ah_attr,
                                                                struct doca_verbs_gid gid) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_gid(ah_attr, gid);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_addr_type(
    void *ah_attr, enum doca_verbs_addr_type addr_type) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_addr_type(ah_attr, addr_type);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_dlid(void *ah_attr, uint32_t dlid) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_dlid(ah_attr, dlid);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_sl(void *ah_attr, uint8_t sl) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_sl(ah_attr, sl);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_sgid_index(void *ah_attr,
                                                                       uint8_t sgid_index) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_sgid_index(ah_attr, sgid_index);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_static_rate(void *ah_attr,
                                                                        uint8_t static_rate) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_static_rate(ah_attr, static_rate);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_hop_limit(void *ah_attr,
                                                                      uint8_t hop_limit) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_hop_limit(ah_attr, hop_limit);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_ah_attr_set_traffic_class(void *ah_attr,
                                                                          uint8_t traffic_class) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_ah_attr_set_traffic_class(ah_attr, traffic_class);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

/* Wrapper function implementations -- QP */
doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_create(doca_dev_t *net_dev,
                                                          doca_verbs_qp_init_attr_t *qp_init_attr,
                                                          void **verbs_qp) {
    doca_error_t doca_err = DOCA_SUCCESS;
    const char *val = getenv(DOCA_SDK_LIB_PATH_ENV_VAR);

    if (net_dev == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create qp, net_dev is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create qp, qp_init_attr is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            DOCA_LOG(LOG_WARNING,
                     "Env var DOCA_SDK_LIB_PATH set to %s, but DOCA SDK libraries not found. DOCA "
                     "SDK is not in use",
                     val);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->sdk_pd == nullptr || net_dev->sdk_context == nullptr) {
            DOCA_LOG(LOG_ERR, "doca_dev_t has no SDK elements.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (qp_init_attr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_qp_init_attr_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        /* According to library logic, if net_dev is SDK, then also GPU is SDK */
        doca_err = p_doca_verbs_qp_create(net_dev->sdk_context, qp_init_attr->sdk, verbs_qp);
        if (doca_err == DOCA_SUCCESS) {
            DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH set to %s. DOCA SDK is in use", val);
            return DOCA_SDK_WRAPPER_SUCCESS;
        } else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else {
        DOCA_LOG(LOG_WARNING, "Env var DOCA_SDK_LIB_PATH not set %s. DOCA SDK is not in use", val);
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
    }

    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_destroy(void *verbs_qp) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        doca_err = p_doca_verbs_qp_destroy(verbs_qp);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_modify(void *verbs_qp,
                                                          doca_verbs_qp_attr_t *qp_attr,
                                                          int attr_mask) {
    doca_error_t doca_err;

    if (qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create qp, doca_verbs_qp_attr_t is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (qp_attr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_qp_attr_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_modify(verbs_qp, qp_attr->sdk, attr_mask);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_query(void *verbs_qp,
                                                         doca_verbs_qp_attr_t *qp_attr,
                                                         doca_verbs_qp_init_attr_t *qp_init_attr,
                                                         int *attr_mask) {
    doca_error_t doca_err;

    if (qp_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create qp, doca_verbs_qp_attr_t is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (qp_init_attr == nullptr) {
        DOCA_LOG(LOG_ERR, "Can't create qp, doca_verbs_qp_init_attr_t is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        if (qp_init_attr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_qp_init_attr_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (qp_attr->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_verbs_qp_attr_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        doca_err = p_doca_verbs_qp_query(verbs_qp, qp_attr->sdk, qp_init_attr->sdk, attr_mask);
        if (doca_err == DOCA_SUCCESS)
            return DOCA_SDK_WRAPPER_SUCCESS;
        else {
            DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
            return DOCA_SDK_WRAPPER_API_ERROR;
        }
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_qpn(void *verbs_qp, uint32_t *qpn) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        *qpn = p_doca_verbs_qp_get_qpn(verbs_qp);

        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_dbr_addr(void *verbs_qp, void **dbr_addr) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        *dbr_addr = p_doca_verbs_qp_get_dbr_addr(verbs_qp);

        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_uar_addr(void *verbs_qp, void **uar_addr) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        *uar_addr = p_doca_verbs_qp_get_uar_addr(verbs_qp);

        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_get_wq(void *verbs_qp, void **sq_buf,
                                                          uint32_t *sq_num_entries, void **rq_buf,
                                                          uint32_t *rq_num_entries,
                                                          uint32_t *rwqe_size_bytes) {
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) return DOCA_SDK_WRAPPER_NOT_FOUND;

        p_doca_verbs_qp_get_wq(verbs_qp, sq_buf, sq_num_entries, rq_buf, rq_num_entries,
                               rwqe_size_bytes);

        return DOCA_SDK_WRAPPER_SUCCESS;
    } else
        return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

static doca_sdk_wrapper_error_t map_doca_cc_group(doca_error_t e) {
    if (e == DOCA_SUCCESS) {
        return DOCA_SDK_WRAPPER_SUCCESS;
    }
    return DOCA_SDK_WRAPPER_API_ERROR;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_query_cc_group_caps(doca_dev_t *net_dev,
                                                                    void **cc_group_caps) {
    doca_error_t doca_err;

    if (!net_dev || !cc_group_caps) {
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }
    if (net_dev->sdk_context == nullptr) {
        DOCA_LOG(LOG_ERR, "%s: net_dev->sdk_context is NULL (SDK verbs context required)",
                 __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_query_cc_group_caps) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_query_cc_group_caps(net_dev->sdk_context, cc_group_caps);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_caps_get_data(void *cc_group_caps,
                                                                       const void **data,
                                                                       size_t *size) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_caps_get_data) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_cc_group_caps_get_data(cc_group_caps, data, size);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_caps_free(void *cc_group_caps) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_caps_free) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_cc_group_caps_free(cc_group_caps);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_attr_create(void **verbs_cc_group_attr) {
    doca_error_t doca_err;

    if (!verbs_cc_group_attr) {
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }
    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_attr_create) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_cc_group_attr_create(verbs_cc_group_attr);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_attr_destroy(void *verbs_cc_group_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_attr_destroy) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_cc_group_attr_destroy(verbs_cc_group_attr);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_attr_set_hint(void *verbs_cc_group_attr,
                                                                       const void *hint_data,
                                                                       size_t hint_data_size) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_attr_set_hint) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err =
            p_doca_verbs_cc_group_attr_set_hint(verbs_cc_group_attr, hint_data, hint_data_size);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_create(doca_dev_t *net_dev,
                                                                void *verbs_cc_group_attr,
                                                                void **verbs_cc_group) {
    doca_error_t doca_err;

    if (!net_dev || !verbs_cc_group) {
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
            DOCA_LOG(LOG_ERR, "doca_dev_t is not a SDK instance.");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (net_dev->sdk_context == nullptr) {
            DOCA_LOG(LOG_ERR, "%s: net_dev->sdk_context is NULL (SDK verbs context required)",
                     __func__);
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }

        if (!p_doca_verbs_cc_group_create) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err =
            p_doca_verbs_cc_group_create(net_dev->sdk_context, verbs_cc_group_attr, verbs_cc_group);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_destroy(void *verbs_cc_group) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_destroy) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_cc_group_destroy(verbs_cc_group);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_cc_group_modify(void *verbs_cc_group,
                                                                void *verbs_cc_group_attr) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_cc_group_modify) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        doca_err = p_doca_verbs_cc_group_modify(verbs_cc_group, verbs_cc_group_attr);
        return map_doca_cc_group(doca_err);
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_set_cc_group(
    void *qp_attr, doca_verbs_cc_group_t *cc_group) {
    doca_error_t doca_err;

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_qp_attr_set_cc_group) {
            DOCA_LOG(LOG_WARNING,
                     "The DOCA SDK doesn't provide doca_verbs_qp_attr_set_cc_group "
                     "(older libdoca_verbs?)\n");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        void *sdk_cc_group = nullptr;
        if (cc_group != nullptr) {
            if (cc_group->type != DOCA_VERBS_SDK_LIB_TYPE_SDK) {
                DOCA_LOG(LOG_ERR, "doca_verbs_cc_group_t is not a SDK instance.");
                return DOCA_SDK_WRAPPER_NOT_FOUND;
            }
            sdk_cc_group = cc_group->sdk;
        }
        doca_err = p_doca_verbs_qp_attr_set_cc_group(qp_attr, sdk_cc_group);
        if (doca_err == DOCA_SUCCESS) {
            return DOCA_SDK_WRAPPER_SUCCESS;
        }
        DOCA_LOG(LOG_ERR, "DOCA SDK function in %s returned error %d", __func__, doca_err);
        return DOCA_SDK_WRAPPER_API_ERROR;
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

doca_sdk_wrapper_error_t doca_verbs_sdk_wrapper_qp_attr_get_cc_group(void *qp_attr,
                                                                     void **cc_group) {
    if (cc_group == nullptr) {
        DOCA_LOG(LOG_ERR, "cc_group output pointer is null", __func__);
        return DOCA_SDK_WRAPPER_API_INVALID_VALUE;
    }

    if (get_sdk_wrapper_env_var() > 0) {
        if (init_verbs_sdk_wrapper() != 0) {
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        if (!p_doca_verbs_qp_attr_get_cc_group) {
            DOCA_LOG(LOG_WARNING,
                     "The DOCA SDK doesn't provide doca_verbs_qp_attr_get_cc_group "
                     "(older libdoca_verbs?)\n");
            return DOCA_SDK_WRAPPER_NOT_FOUND;
        }
        *cc_group = p_doca_verbs_qp_attr_get_cc_group(qp_attr);
        return DOCA_SDK_WRAPPER_SUCCESS;
    }
    return DOCA_SDK_WRAPPER_NOT_SUPPORTED;
}

#ifdef __cplusplus
}
#endif