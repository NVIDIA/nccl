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
 * @file doca_verbs.h
 * @brief A header file for the doca_verbs APIs
 */

#ifndef DOCA_VERBS_H
#define DOCA_VERBS_H

#include <errno.h>

#include "doca_error.h"

#ifdef __cplusplus
extern "C" {
#endif

/**********************************************************************************************************************
 * DOCA Verbs opaque types
 *********************************************************************************************************************/
/**
 * Opaque structure representing a DOCA Verbs QP Init Attributes instance.
 */
struct doca_verbs_qp_init_attr;
/**
 * Opaque structure representing a DOCA Verbs QP Attributes instance.
 */
struct doca_verbs_qp_attr;
/**
 * Opaque structure representing a DOCA Verbs Queue Pair instance.
 */
struct doca_verbs_qp;
/**
 * Opaque structure representing a DOCA Verbs CQ Attributes instance.
 */
struct doca_verbs_cq_attr;
/**
 * Opaque structure representing a DOCA Verbs Completion Queue instance.
 */
struct doca_verbs_cq;
/**
 * Opaque structure representing a DOCA Verbs Shared Receive Queue instance.
 */
struct doca_verbs_srq;
/**
 * Opaque structure representing a DOCA Verbs SRQ Init Attributes
 */
struct doca_verbs_srq_init_attr;
/**
 * Opaque structure representing a DOCA Verbs AH instance.
 */
struct doca_verbs_ah_attr;
/**
 * Opaque structure representing a DOCA UMEM instance.
 */
struct doca_verbs_umem;
/**
 * Opaque structure representing a DOCA UAR instance.
 */
struct doca_verbs_uar;
/**
 * Opaque structure representing a DOCA Device Attributes instance.
 */
struct doca_verbs_device_attr;

/**
 * @brief Verbs RC QP type define.
 */
#define DOCA_VERBS_QP_TYPE_RC 0x0

/**
 * @brief Verbs QP state.
 */
enum doca_verbs_qp_state {
    DOCA_VERBS_QP_STATE_RST = 0x0,
    DOCA_VERBS_QP_STATE_INIT = 0x1,
    DOCA_VERBS_QP_STATE_RTR = 0x2,
    DOCA_VERBS_QP_STATE_RTS = 0x3,
    DOCA_VERBS_QP_STATE_ERR = 0x4,
};

/**
 * @brief Verbs address type.
 */
enum doca_verbs_addr_type {
    DOCA_VERBS_ADDR_TYPE_IPv4,      /**< IPv4 type */
    DOCA_VERBS_ADDR_TYPE_IPv6,      /**< IPv6 type */
    DOCA_VERBS_ADDR_TYPE_IB_GRH,    /**< IB with GRH type */
    DOCA_VERBS_ADDR_TYPE_IB_NO_GRH, /**< IB without GRH type */
};

/**
 * @brief MTU size in bytes.
 */
enum doca_verbs_mtu_size {
    DOCA_VERBS_MTU_SIZE_256_BYTES = 0x0,
    DOCA_VERBS_MTU_SIZE_512_BYTES = 0x1,
    DOCA_VERBS_MTU_SIZE_1K_BYTES = 0x2,
    DOCA_VERBS_MTU_SIZE_2K_BYTES = 0x3,
    DOCA_VERBS_MTU_SIZE_4K_BYTES = 0x4,
    DOCA_VERBS_MTU_SIZE_RAW_ETHERNET = 0x5, /* Reserved */
};

/**
 * @brief DOCA Verbs UAR allocation type.
 */
enum doca_verbs_uar_allocation_type {
    DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME = 0,
    DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE = 1,
    DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED = 2,
};

/**
 * @brief CQ overrun
 */
enum doca_verbs_cq_overrun {
    DOCA_VERBS_CQ_DISABLE_OVERRUN = 0, /**< Disable overrun by default. */
    DOCA_VERBS_CQ_ENABLE_OVERRUN = 1,  /**< Enable overrun. */
};

/**
 * @brief DOCA Verbs SRQ type.
 */
enum doca_verbs_srq_type {
    DOCA_VERBS_SRQ_TYPE_LINKED_LIST,
    DOCA_VERBS_SRQ_TYPE_CONTIGUOUS,
};

/**
 * @brief DOCA Verbs Atomic Type.
 */
enum doca_verbs_qp_atomic_type {
    DOCA_VERBS_QP_ATOMIC_MODE_NONE = 0x0,
    DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC = 0x1,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_8BYTES = 0x3
};

/**
 * @brief Verbs QP attributes
 *
 * @details These defines can be used with doca_verbs_qp_modify() to set QP attributes.
 * These attributes are used in several QP state transition commands.
 *
 * For each command bellow there are optional and required attributes depending on QP type:
 * - *->rst:
 *		QP type RC:
 *			required: next_state
 *			optional: NONE
 *		QP type UC:
 *			required: next_state
 *			optional: NONE
 * - *->err:
 *		QP type RC:
 *			required: next_state
 *			optional: NONE
 *		QP type UC:
 *			required: next_state
 *			optional: NONE
 * - rst->init:
 * 		QP type RC:
 *			required: next_state, allow_remote_write, allow_remote_read, allow_atomic,
 *pkey_index, port_num optional: NONE QP type UC: required: next_state, allow_remote_write,
 *pkey_index, port_num optional: NONE
 * - init->init:
 *		QP type RC:
 *			required: NONE
 *			optional: allow_remote_write, allow_remote_read, allow_atomic, pkey_index,
 *port_num QP type UC: required: NONE optional: allow_remote_write, pkey_index, port_num
 * - init->rtr:
 *		QP type RC:
 *			required: next_state, rq_psn, dest_qp_num, path_mtu, ah_attr, min_rnr_timer
 *			optional: allow_remote_write, allow_remote_read, allow_atomic, pkey_index
 *		QP type UC:
 *			required: next_state, rq_psn, dest_qp_num, path_mtu, ah_attr
 *			optional: allow_remote_write, pkey_index
 * - rtr->rts:
 *		QP type RC:
 *			required: next_state, sq_psn, ack_timeout, retry_cnt, rnr_retry
 *			optional: allow_remote_write, min_rnr_timer
 *		QP type UC:
 *			required: next_state, sq_psn,
 *			optional: allow_remote_write
 * - rts->rts:
 *		QP type RC:
 *			required: NONE
 *			optional: allow_remote_write, allow_remote_read, allow_atomic,
 *min_rnr_timer, ah_attr QP type UC: required: NONE optional: allow_remote_write, ah_attr
 *
 */
/**
 * @brief Allow Remote Write attribute.
 */
#define DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_WRITE (1 << 0)
/**
 * @brief Allow Remote Read attribute.
 */
#define DOCA_VERBS_QP_ATTR_ALLOW_REMOTE_READ (1 << 1)
/**
 * @brief PKEY Index attribute.
 */
#define DOCA_VERBS_QP_ATTR_PKEY_INDEX (1 << 2)
/**
 * @brief Minimum RNR Timer attribute.
 */
#define DOCA_VERBS_QP_ATTR_MIN_RNR_TIMER (1 << 3)
/**
 * @brief Port Number attribute.
 */
#define DOCA_VERBS_QP_ATTR_PORT_NUM (1 << 4)
/**
 * @brief Next State attribute.
 */
#define DOCA_VERBS_QP_ATTR_NEXT_STATE (1 << 5)
/**
 * @brief Current State attribute.
 */
#define DOCA_VERBS_QP_ATTR_CURRENT_STATE (1 << 6)
/**
 * @brief Path MTU attribute.
 */
#define DOCA_VERBS_QP_ATTR_PATH_MTU (1 << 7)
/**
 * @brief RQ PSN attribute.
 */
#define DOCA_VERBS_QP_ATTR_RQ_PSN (1 << 8)
/**
 * @brief SQ PSN attribute.
 */
#define DOCA_VERBS_QP_ATTR_SQ_PSN (1 << 9)
/**
 * @brief Destination QP attribute.
 */
#define DOCA_VERBS_QP_ATTR_DEST_QP_NUM (1 << 10)
/**
 * @brief ACK Timeout attribute.
 */
#define DOCA_VERBS_QP_ATTR_ACK_TIMEOUT (1 << 11)
/**
 * @brief Retry Counter attribute.
 */
#define DOCA_VERBS_QP_ATTR_RETRY_CNT (1 << 12)
/**
 * @brief RNR Retry attribute.
 */
#define DOCA_VERBS_QP_ATTR_RNR_RETRY (1 << 13)
/**
 * @brief AH attribute.
 */
#define DOCA_VERBS_QP_ATTR_AH_ATTR (1 << 14)

/**
 * @brief Specifies the length of a GID (Global ID) in bytes.
 */
#define DOCA_VERBS_GID_BYTE_LENGTH 16

/**
 * @brief Invalid dmabuf_fd value. Used to notify the umem must be registered without dmabuf.
 */
#define DOCA_VERBS_DMABUF_INVALID_FD 0xFFFFFFFF
/**
 * @brief GID struct.
 */
struct doca_verbs_gid {
    uint8_t raw[DOCA_VERBS_GID_BYTE_LENGTH]; /**< The raw value of the GID */
};

/**********************************************************************************************************************
 * DOCA Verbs functions
 *********************************************************************************************************************/

/**
 * @brief Create a DOCA Verbs QP Init Attributes instance.
 *
 * @param [out] verbs_qp_init_attr
 * Pointer to pointer to be set to point to the created verbs_qp_init_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_qp_init_attr_create(struct doca_verbs_qp_init_attr **verbs_qp_init_attr);

/**
 * @brief Destroy a DOCA Verbs QP Init Attributes instance.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_destroy(struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set pd attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] pd
 * pd attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_pd(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                            struct ibv_pd *pd);

/**
 * @brief Get pd attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * pd attribute.
 */
struct ibv_pd *doca_verbs_qp_init_attr_get_pd(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set send_cq attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] send_cq
 * send_cq attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_send_cq(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                                 struct doca_verbs_cq *send_cq);

/**
 * @brief Get send_cq attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * send_cq attribute.
 */
struct doca_verbs_cq *doca_verbs_qp_init_attr_get_send_cq(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set receive_cq attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] receive_cq
 * receive_cq attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_receive_cq(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_cq *receive_cq);

/**
 * @brief Get receive_cq attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * receive_cq attribute.
 */
struct doca_verbs_cq *doca_verbs_qp_init_attr_get_receive_cq(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set sq_sig_all attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] sq_sig_all
 * sq_sig_all attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_sq_sig_all(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, int sq_sig_all);

/**
 * @brief Get sq_sig_all attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * sq_sig_all attribute.
 */
int doca_verbs_qp_init_attr_get_sq_sig_all(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set sq_wr attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] sq_wr
 * sq_wr attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_sq_wr(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                               uint32_t sq_wr);

/**
 * @brief Get sq_wr attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * sq_wr attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_sq_wr(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set rq_wr attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] rq_wr
 * rq_wr attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_rq_wr(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                               uint32_t rq_wr);

/**
 * @brief Get rq_wr attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * rq_wr attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_rq_wr(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set send_max_sges attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] send_max_sges
 * send_max_sges attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_send_max_sges(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t send_max_sges);

/**
 * @brief Get send_max_sges attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * send_max_sges attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_send_max_sges(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set receive_max_sges attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] receive_max_sges
 * receive_max_sges attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_receive_max_sges(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t receive_max_sges);

/**
 * @brief Get receive_max_sges attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * receive_max_sges attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_receive_max_sges(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set max_inline_data attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] max_inline_data
 * max_inline_data attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_max_inline_data(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t max_inline_data);

/**
 * @brief Get max_inline_data attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * max_inline_data attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_max_inline_data(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set user_index attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] user_index
 * user_index attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_user_index(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint32_t user_index);

/**
 * @brief Get user_index attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * user_index attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_user_index(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set qp_type attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] qp_type
 * qp_type attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_qp_type(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                                 uint32_t qp_type);

/**
 * @brief Get qp_type attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * qp_type attribute.
 */
uint32_t doca_verbs_qp_init_attr_get_qp_type(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set external umem attributes for verbs_qp_init_attr.
 *
 * Setting these attributes means that the user wants to create and provide the umem by himself,
 * in compare with the default mode where the umem is created internally.
 * In that case it is the user responsibility to allocate enough memory for the umem and to free it.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] external_umem
 * External umem instance.
 * @param [in] external_umem_offset
 * The offset in the external umem buffer to set the Work Queue
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_external_umem(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_umem *external_umem,
    uint64_t external_umem_offset);

/**
 * @brief Set external DBR umem attributes for verbs_qp_init_attr.
 *
 * Setting these attributes means that the user wants to create and provide the dbr umem by himself,
 * in compare with the default mode where the dbr umem is created internally.
 * In that case it is the user responsibility to allocate enough memory for the umem and to free it.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] external_dbr_umem
 * External dbr umem instance.
 * @param [in] external_dbr_umem_offset
 * The offset in the external dbr umem buffer to set the DBR
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_external_dbr_umem(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_umem *external_umem,
    uint64_t external_umem_offset);

/**
 * @brief Get external umem attributes from verbs_qp_init_attr.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [out] external_umem
 * External umem instance.
 * @param [out] external_umem_offset
 * The offset in the external umem buffer to set the Work Queue
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_get_external_umem(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
    struct doca_verbs_umem **external_umem, uint64_t *external_umem_offset);

/**
 * @brief Set external uar attribute for verbs_qp_init_attr.
 *
 * Setting these attribute means that the user wants to create and provide the uar by himself,
 * in compare with the default mode where the uar is created internally.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] external_uar
 * External uar instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_external_uar(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_uar *external_uar);

/**
 * @brief Get external uar attribute from verbs_qp_init_attr.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [out] external_uar
 * External uar instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_get_external_uar(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr, struct doca_verbs_uar **external_uar);

/**
 * @brief Set qp_context attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] qp_context
 * qp_context attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_qp_context(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, void *qp_context);

/**
 * @brief Get qp_context attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * qp_context attribute.
 */
void *doca_verbs_qp_init_attr_get_qp_context(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set srq attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] srq
 * srq attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_srq(struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                             struct doca_verbs_srq *srq);

/**
 * @brief Get srq attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * srq attribute.
 */
struct doca_verbs_srq *doca_verbs_qp_init_attr_get_srq(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Set CORE direct for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] core_direct_master
 * Set core direct attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_core_direct_master(
    struct doca_verbs_qp_init_attr *verbs_qp_init_attr, uint8_t core_direct_master);

/**
 * @brief Get CORE Direct attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * min_rnr_timer attribute.
 */
uint8_t doca_verbs_qp_init_attr_get_core_direct_master(
    const struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Create a DOCA Verbs QP Attributes instance.
 *
 * @param [out] verbs_qp_attr
 * Pointer to pointer to be set to point to the created verbs_qp_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_qp_attr_create(struct doca_verbs_qp_attr **verbs_qp_attr);

/**
 * @brief Destroy a DOCA Verbs QP Attributes instance.
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_destroy(struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set next_state attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] next_state
 * next_state attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_next_state(struct doca_verbs_qp_attr *verbs_qp_attr,
                                               enum doca_verbs_qp_state next_state);

/**
 * @brief Get next_state attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * next_state attribute.
 */
enum doca_verbs_qp_state doca_verbs_qp_attr_get_next_state(
    const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set current_state attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] current_state
 * current_state attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_current_state(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                  enum doca_verbs_qp_state current_state);

/**
 * @brief Get current_state attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * current_state attribute.
 */
enum doca_verbs_qp_state doca_verbs_qp_attr_get_current_state(
    const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set path_mtu attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] path_mtu
 * path_mtu attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_path_mtu(struct doca_verbs_qp_attr *verbs_qp_attr,
                                             enum doca_verbs_mtu_size path_mtu);

/**
 * @brief Get path_mtu attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * path_mtu attribute.
 */
enum doca_verbs_mtu_size doca_verbs_qp_attr_get_path_mtu(
    const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set rq_psn attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] rq_psn
 * rq_psn attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_rq_psn(struct doca_verbs_qp_attr *verbs_qp_attr,
                                           uint32_t rq_psn);

/**
 * @brief Get rq_psn attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * rq_psn attribute.
 */
uint32_t doca_verbs_qp_attr_get_rq_psn(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set sq_psn attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] sq_psn
 * sq_psn attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_sq_psn(struct doca_verbs_qp_attr *verbs_qp_attr,
                                           uint32_t sq_psn);

/**
 * @brief Get sq_psn attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * sq_psn attribute.
 */
uint32_t doca_verbs_qp_attr_get_sq_psn(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set dest_qp_num attribute for verbs_qp_attr
 * @note The destination QP number used to establish a connection with the destination QP during the
 * QP state modification.
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] dest_qp_num
 * dest_qp_num attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_dest_qp_num(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                uint32_t dest_qp_num);

/**
 * @brief Get dest_qp_num attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * dest_qp_num attribute.
 */
uint32_t doca_verbs_qp_attr_get_dest_qp_num(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set allow_remote_write attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] allow_remote_write
 * allow_remote_write attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_allow_remote_write(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                       int allow_remote_write);

/**
 * @brief Get allow_remote_write attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * allow_remote_write attribute.
 */
int doca_verbs_qp_attr_get_allow_remote_write(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set allow_remote_read attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] allow_remote_read
 * allow_remote_read attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_allow_remote_read(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                      int allow_remote_read);

/**
 * @brief Get allow_remote_read attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * allow_remote_read attribute.
 */
int doca_verbs_qp_attr_get_allow_remote_read(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set allow_atomic attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] allow_atomic
 * allow_atomic attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_allow_remote_atomic(
    struct doca_verbs_qp_attr *verbs_qp_attr, enum doca_verbs_qp_atomic_type allow_atomic_type);

/**
 * @brief Get allow_atomic attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * allow_atomic attribute.
 */
enum doca_verbs_qp_atomic_type doca_verbs_qp_attr_get_allow_remote_atomic(
    const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set ah_attr attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] ah_attr
 * ah_attr attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_ah_attr(struct doca_verbs_qp_attr *verbs_qp_attr,
                                            struct doca_verbs_ah_attr *ah_attr);

/**
 * @brief Get ah_attr attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * ah_attr attribute.
 */
struct doca_verbs_ah_attr *doca_verbs_qp_attr_get_ah_attr(
    const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set pkey_index attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] pkey_index
 * pkey_index attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_pkey_index(struct doca_verbs_qp_attr *verbs_qp_attr,
                                               uint16_t pkey_index);

/**
 * @brief Get pkey_index attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * pkey_index attribute.
 */
uint16_t doca_verbs_qp_attr_get_pkey_index(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set port_num attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] port_num
 * port_num attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_port_num(struct doca_verbs_qp_attr *verbs_qp_attr,
                                             uint16_t port_num);

/**
 * @brief Get port_num attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * port_num attribute.
 */
uint16_t doca_verbs_qp_attr_get_port_num(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set ack_timeout attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] ack_timeout
 * ack_timeout attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_ack_timeout(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                uint16_t ack_timeout);

/**
 * @brief Get ack_timeout attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * ack_timeout attribute.
 */
uint16_t doca_verbs_qp_attr_get_ack_timeout(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set retry_cnt attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] retry_cnt
 * retry_cnt attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_retry_cnt(struct doca_verbs_qp_attr *verbs_qp_attr,
                                              uint16_t retry_cnt);

/**
 * @brief Get retry_cnt attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * retry_cnt attribute.
 */
uint16_t doca_verbs_qp_attr_get_retry_cnt(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set rnr_retry attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] rnr_retry
 * rnr_retry attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_rnr_retry(struct doca_verbs_qp_attr *verbs_qp_attr,
                                              uint16_t rnr_retry);

/**
 * @brief Get rnr_retry attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * rnr_retry attribute.
 */
uint16_t doca_verbs_qp_attr_get_rnr_retry(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Set min_rnr_timer attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] min_rnr_timer
 * min_rnr_timer attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_min_rnr_timer(struct doca_verbs_qp_attr *verbs_qp_attr,
                                                  uint16_t min_rnr_timer);

/**
 * @brief Get min_rnr_timer attribute from verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 *
 * @return
 * min_rnr_timer attribute.
 */
uint16_t doca_verbs_qp_attr_get_min_rnr_timer(const struct doca_verbs_qp_attr *verbs_qp_attr);

/**
 * @brief Create a DOCA Verbs AH instance.
 *
 * @param [in] context
 * Pointer to context instance.
 * @param [out] verbs_ah
 * Pointer to pointer to be set to point to the created verbs_ah instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_ah_attr_create(struct ibv_context *context,
                                       struct doca_verbs_ah_attr **verbs_ah);

/**
 * @brief Destroy a DOCA Verbs AH instance.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_destroy(struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set gid attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] gid
 * gid attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_gid(struct doca_verbs_ah_attr *verbs_ah,
                                        struct doca_verbs_gid gid);

/**
 * @brief Get gid attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * gid attribute.
 */
struct doca_verbs_gid doca_verbs_ah_get_gid(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set addr_type attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] addr_type
 * addr_type attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_addr_type(struct doca_verbs_ah_attr *verbs_ah,
                                              enum doca_verbs_addr_type addr_type);

/**
 * @brief Get addr_type attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * addr_type attribute.
 */
enum doca_verbs_addr_type doca_verbs_ah_get_addr_type(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set dlid attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] dlid
 * dlid attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_dlid(struct doca_verbs_ah_attr *verbs_ah, uint32_t dlid);

/**
 * @brief Get dlid attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * dlid attribute.
 */
uint32_t doca_verbs_ah_get_dlid(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set sl attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] sl
 * sl attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_sl(struct doca_verbs_ah_attr *verbs_ah, uint8_t sl);

/**
 * @brief Get sl attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * sl attribute.
 */
uint8_t doca_verbs_ah_get_sl(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set sgid_index attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] sgid_index
 * sgid_index attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_sgid_index(struct doca_verbs_ah_attr *verbs_ah,
                                               uint8_t sgid_index);

/**
 * @brief Get sgid_index attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * sgid_index attribute.
 */
uint8_t doca_verbs_ah_get_sgid_index(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set static_rate attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] static_rate
 * static_rate attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_static_rate(struct doca_verbs_ah_attr *verbs_ah,
                                                uint8_t static_rate);

/**
 * @brief Get static_rate attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * static_rate attribute.
 */
uint8_t doca_verbs_ah_get_static_rate(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set hop_limit attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] hop_limit
 * hop_limit attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_hop_limit(struct doca_verbs_ah_attr *verbs_ah,
                                              uint8_t hop_limit);

/**
 * @brief Get hop_limit attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * hop_limit attribute.
 */
uint8_t doca_verbs_ah_get_hop_limit(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Set traffic_class attribute for verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 * @param [in] traffic_class
 * traffic_class attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_ah_attr_set_traffic_class(struct doca_verbs_ah_attr *verbs_ah,
                                                  uint8_t traffic_class);

/**
 * @brief Get traffic_class attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * traffic_class attribute.
 */
uint8_t doca_verbs_ah_get_traffic_class(const struct doca_verbs_ah_attr *verbs_ah);

/**
 * @brief Create a DOCA Verbs Queue Pair instance.
 *
 * @param [in] context
 * Pointer to ibv_context instance.
 * @param [in] verbs_qp_init_attr
 * Pointer to qp_init_attr instance.
 * @param [out] verbs_qp
 * Pointer to pointer to be set to point to the created verbs_qp instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_qp_create(struct ibv_context *context,
                                  struct doca_verbs_qp_init_attr *verbs_qp_init_attr,
                                  struct doca_verbs_qp **verbs_qp);

/**
 * @brief Destroy a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_destroy(struct doca_verbs_qp *verbs_qp);

/**
 * @brief Modify a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] attr_mask
 * Mask for QP attributes. see define for DOCA_VERBS_QP_ATTR_*
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_qp_modify(struct doca_verbs_qp *verbs_qp,
                                  struct doca_verbs_qp_attr *verbs_qp_attr, int attr_mask);

/**
 * @brief Query the attributes of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [out] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [out] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_qp_query(struct doca_verbs_qp *verbs_qp,
                                 struct doca_verbs_qp_attr *verbs_qp_attr,
                                 struct doca_verbs_qp_init_attr *verbs_qp_init_attr);

/**
 * @brief Get the Work Queue attributes of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [out] sq_buf
 * Pointer to Send Queue buffer.
 * @param [out] sq_num_entries
 * The number of entries in Send Queue buffer.
 * @param [out] rq_buf
 * Pointer to Receive Queue buffer.
 * @param [out] rq_num_entries
 * The number of entries in Receive Queue buffer.
 * @param [out] rwqe_size_bytes
 * Receive WQE size in bytes.
 *
 */
void doca_verbs_qp_get_wq(const struct doca_verbs_qp *verbs_qp, void **sq_buf,
                          uint32_t *sq_num_entries, void **rq_buf, uint32_t *rq_num_entries,
                          uint32_t *rwqe_size_bytes);

/**
 * @brief Get the DBR address of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 *
 * @return
 * The DBR address.
 */
void *doca_verbs_qp_get_dbr_addr(const struct doca_verbs_qp *verbs_qp);

/**
 * @brief Get the UAR address of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 *
 * @return
 * The UAR register address.
 */
void *doca_verbs_qp_get_uar_addr(const struct doca_verbs_qp *verbs_qp);

/**
 * @brief Get the QP number of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 *
 * @return
 * The QP number.
 */
uint32_t doca_verbs_qp_get_qpn(const struct doca_verbs_qp *verbs_qp);

/**
 * @brief Create a DOCA Verbs CQ Attributes instance.
 *
 * @param [out] verbs_cq_attr
 * Pointer to pointer to be set to point to the created verbs_cq_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_cq_attr_create(struct doca_verbs_cq_attr **verbs_cq_attr);

/**
 * @brief Destroy a DOCA Verbs CQ Attributes instance.
 *
 * @param [in] verbs_cq_attr
 * Pointer to verbs_cq_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_destroy(struct doca_verbs_cq_attr *verbs_cq_attr);

/**
 * @brief Set cq_size attribute for doca_verbs_cq_attr.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] cq_size
 * cq size (num entries).
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_cq_size(struct doca_verbs_cq_attr *cq_attr, uint32_t cq_size);

/**
 * @brief Set cq_context attribute for doca_verbs_cq_attr.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] cq_context
 * User data. cq_context may be null in case the application regrets setting a user data.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_cq_context(struct doca_verbs_cq_attr *cq_attr,
                                               void *cq_context);

/**
 * @brief Set external umem attribute for doca_verbs_cq_attr.
 *
 * Setting this attribute means that the user wants to create and provide the umem by himself,
 * in compare with the default mode where the umem is created internally.
 * In that case it is the user responsibility to allocate enough memory for the umem and to free it.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] external_umem
 * External umem instance.
 * @param [in] external_umem_offset
 * The offset in the external umem buffer to set the Completion Queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_external_umem(struct doca_verbs_cq_attr *cq_attr,
                                                  struct doca_verbs_umem *external_umem,
                                                  uint64_t external_umem_offset);

/**
 * @brief Set external dbr umem attribute for doca_verbs_cq_attr.
 *
 * Setting this attribute means that the user wants to create and provide the dbr umem by himself,
 * in compare with the default mode where the umem is created internally.
 * In that case it is the user responsibility to allocate enough memory for the umem and to free it.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] external_umem
 * External umem instance.
 * @param [in] external_umem_offset
 * The offset in the external umem buffer to set the Completion Queue.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_external_dbr_umem(struct doca_verbs_cq_attr *cq_attr,
                                                      struct doca_verbs_umem *external_umem,
                                                      uint64_t external_umem_offset);

/**
 * @brief Set external uar attribute for doca_verbs_cq_attr.
 *
 * Setting this attribute means that the user wants to provide an external uar by himself,
 * in compare with the default mode where uar is created internally.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] external_uar
 * External uar.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_external_uar(struct doca_verbs_cq_attr *cq_attr,
                                                 struct doca_verbs_uar *external_uar);

/**
 * @brief Enable cq_overrun attribute for doca_verbs_cq_attr.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] overrun
 * enable or disable overrun (@see doca_verbs_cq_overrun).
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_cq_overrun(struct doca_verbs_cq_attr *cq_attr,
                                               enum doca_verbs_cq_overrun overrun);
/**
 * @brief Create a DOCA Verbs Completion Queue instance.
 *
 * @param [in] context
 * Pointer to ibv_context instance.
 * @param [in] verbs_cq_attr
 * Pointer to verbs_cq_attr instance.
 * @param [out] verbs_cq
 * Pointer to pointer to be set to point to the created doca_verbs_cq instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_cq_create(struct ibv_context *context,
                                  struct doca_verbs_cq_attr *verbs_cq_attr,
                                  struct doca_verbs_cq **verbs_cq);

/**
 * @brief Destroy a DOCA Verbs Completion Queue instance.
 *
 * @param [in] verbs_cq
 * Pointer to verbs_cq instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_destroy(struct doca_verbs_cq *verbs_cq);

/**
 * @brief Get the Completion Queue attributes of a DOCA Verbs Completion Queue instance.
 *
 * @param [in] verbs_cq
 * Pointer to verbs_cq instance.
 * @param [out] cq_buf
 * Pointer to Completion Queue buffer.
 * @param [out] cq_num_entries
 * The number of entries in Completion Queue buffer.
 * @param [out] cq_entry_size
 * The size of each entry in Completion Queue buffer.
 *
 */
void doca_verbs_cq_get_wq(struct doca_verbs_cq *verbs_cq, void **cq_buf, uint32_t *cq_num_entries,
                          uint8_t *cq_entry_size);

/**
 * @brief Get the DBR address of a DOCA Verbs Completion Queue instance.
 *
 * @param [in] verbs_cq
 * Pointer to verbs_cq instance.
 * @param [out] uar_db_reg
 * Pointer to the UAR doorbell record
 * @param [out] ci_dbr
 * Pointer to the CI doorbell record
 * @param [out] arm_dbr
 * Pointer to the arm doorbell record
 */
void doca_verbs_cq_get_dbr_addr(struct doca_verbs_cq *verbs_cq, uint64_t **uar_db_reg,
                                uint32_t **ci_dbr, uint32_t **arm_dbr);

/**
 * @brief Get the CQ number of a DOCA Verbs CQ instance.
 *
 * @param [in] verbs_cq
 * Pointer to verbs_cq instance.
 *
 * @return
 * The CQ number.
 */
uint32_t doca_verbs_cq_get_cqn(const struct doca_verbs_cq *verbs_cq);

/**
 * @brief Create a DOCA Verbs SRQ Init Attributes instance.
 *
 * @param [out] verbs_srq_init_attr
 * Pointer to pointer to be set to point to the created verbs_srq_init_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_srq_init_attr_create(struct doca_verbs_srq_init_attr **verbs_srq_init_attr);

/**
 * @brief Destroy a DOCA Verbs SRQ Init Attributes instance.
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_destroy(struct doca_verbs_srq_init_attr *verbs_srq_init_attr);

/**
 * @brief Set srq_wr attribute for verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 * @param [in] srq_wr
 * srq_wr attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_set_srq_wr(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, uint32_t srq_wr);

/**
 * @brief Get srq_wr attribute from verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 *
 * @return
 * srq_wr attribute.
 */
uint32_t doca_verbs_srq_init_attr_get_srq_wr(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr);
/**
 * @brief Set receive_max_sges attribute for verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 * @param [in] receive_max_sges
 * receive_max_sges attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_set_receive_max_sges(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, uint32_t receive_max_sges);

/**
 * @brief Get receive_max_sges attribute from verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 *
 * @return
 * receive_max_sges attribute.
 */
uint32_t doca_verbs_srq_init_attr_get_receive_max_sges(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr);

/**
 * @brief Set srq_type attribute for verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 * @param [in] srq_type
 * srq_type attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_set_type(struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
                                               enum doca_verbs_srq_type srq_type);

/**
 * @brief Get srq_type attribute from verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 *
 * @return
 * srq_type attribute.
 */
enum doca_verbs_srq_type doca_verbs_srq_init_attr_get_type(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr);

/**
 * @brief Set pd attribute for verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 * @param [in] pd
 * pd attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_set_pd(struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
                                             struct ibv_pd *pd);

/**
 * @brief Get pd attribute from verbs_srq_init_attr
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 *
 * @return
 * pd attribute.
 */
struct ibv_pd *doca_verbs_srq_init_attr_get_pd(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr);

/**
 * @brief Set external umem attributes for verbs_srq_init_attr.
 *
 * Setting these attributes means that the user wants to create and provide the umem by himself,
 * in compare with the default mode where the umem is created internally.
 * In that case it is the user responsibility to allocate enough memory for the umem and to free it.
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 * @param [in] external_umem
 * External umem instance.
 * @param [in] external_umem_offset
 * The offset in the external umem buffer to set the Work Queue
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_set_external_umem(
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, struct doca_verbs_umem *external_umem,
    uint64_t external_umem_offset);

/**
 * @brief Get external umem attributes from verbs_srq_init_attr.
 *
 * @param [in] verbs_srq_init_attr
 * Pointer to verbs_srq_init_attr instance.
 * @param [out] external_umem
 * External umem instance.
 * @param [out] external_umem_offset
 * The offset in the external umem buffer to set the Work Queue
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_init_attr_get_external_umem(
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
    struct doca_verbs_umem **external_umem, uint64_t *external_umem_offset);

/**
 * @brief Create a DOCA Verbs Shared Receive Queue instance.
 *
 * @param [in] verbs_context
 * Pointer to verbs_context instance.
 * @param [in] verbs_srq_init_attr
 * Pointer to srq_init_attr instance.
 * @param [out] verbs_srq
 * Pointer to pointer to be set to point to the created verbs_srq instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_srq_create(struct ibv_context *verbs_context,
                                   struct doca_verbs_srq_init_attr *verbs_srq_init_attr,
                                   struct doca_verbs_srq **verbs_srq);

/**
 * @brief Destroy a DOCA IB Shared Receive Queue instance.
 *
 * @param [in] verbs_srq
 * Pointer to verbs_srq instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_srq_destroy(struct doca_verbs_srq *verbs_srq);

/**
 * @brief Get the SRQ number of a DOCA Verbs Shared Receive Queue instance.
 *
 * @param [in] verbs_srq
 * Pointer to verbs_srq instance.
 *
 * @return
 * The SRQ number.
 */
uint32_t doca_verbs_srq_get_srqn(const struct doca_verbs_srq *verbs_srq);

/**
 * @brief Get the Work Queue attributes of a DOCA Verbs Shared Receive Queue instance.
 *
 * @param [in] verbs_srq
 * Pointer to verbs_srq instance.
 * @param [out] srq_buf
 * Pointer to Shared Receive Queue buffer.
 * @param [out] srq_num_entries
 * The number of entries in Shared Receive Queue buffer.
 * @param [out] rwqe_size_bytes
 * Receive WQE size in bytes.
 *
 */
void doca_verbs_srq_get_wq(const struct doca_verbs_srq *verbs_srq, void **srq_buf,
                           uint32_t *srq_num_entries, uint32_t *rwqe_size_bytes);

/**
 * @brief Get the DBR address of a DOCA Verbs Shared Receive Queue instance.
 *
 * @param [in] verbs_srq
 * Pointer to verbs_srq instance.
 *
 * @return
 * The DBR address.
 */
void *doca_verbs_srq_get_dbr_addr(const struct doca_verbs_srq *verbs_srq);

/**********************************************************************************************************************
 * Capabilities functions
 *********************************************************************************************************************/

/**
 * @brief Query DOCA Verbs device attributes.
 *
 * @param [in] context
 * Pointer to ibv_context instance.
 * @param [out] verbs_device_attr
 * Pointer to pointer to be set to point to the created verbs_device_attr instance.
 * User is expected to free this object with "doca_verbs_device_attr_free()".
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 * - DOCA_ERROR_NOT_DRIVER - low level layer failure.
 */
doca_error_t doca_verbs_query_device(struct ibv_context *context,
                                     struct doca_verbs_device_attr **verbs_device_attr);

/**
 * @brief Free a DOCA Verbs Device Attributes instance.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_device_attr_free(struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of QPs supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of QPs supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_qp(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of work requests on send/receive queue supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of work requests on send/receive queue supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_qp_wr(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of scatter/gather entries per send/receive work request in a QP
 * other than RD supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of scatter/gather entries per send/receive work request in a QP other than RD
 * supported by the device.
 *
 */
uint32_t doca_verbs_device_attr_get_max_sge(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of CQs supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of CQs supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_cq(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of entries on CQ supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of entries on CQ supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_cqe(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of MRs supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of MRs supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_mr(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of PDs supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of MRs supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_pd(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of AHs supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of AHs supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_ah(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of SRQs supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of SRQs supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_srq(const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of work requests on SRQ supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of work requests on SRQ supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_srq_wr(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of scatter entries per receive work request in a SRQ supported by
 * the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of scatter entries per receive work request in a SRQ supported by the device.
 */
uint32_t doca_verbs_device_attr_get_max_srq_sge(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of partitions supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The max number of partitions supported by the device.
 */
uint16_t doca_verbs_device_attr_get_max_pkeys(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Check if a given QP type is supported on this device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 * @param [in] qp_type
 * The QP type to check its support.
 *
 * @return
 * DOCA_SUCCESS - in case QP type is supported.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid parameter was given.
 * - DOCA_ERROR_NOT_SUPPORTED - if QP type is not supported.
 */
doca_error_t doca_verbs_device_attr_get_is_qp_type_supported(
    const struct doca_verbs_device_attr *verbs_device_attr, uint32_t qp_type);

/**
 * @brief Create an instance of DOCA Verbs UMEM.
 *
 * @param [in] context
 * Pointer to ibv_context instance.
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
 * @param [out] umem_obj
 * The umem object
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_DRIVER - in case of error in a DOCA driver call.
 */
doca_error_t doca_verbs_umem_create(struct ibv_context *context, void *address, size_t size,
                                    uint32_t access_flags, int dmabuf_id, size_t dmabuf_offset,
                                    struct doca_verbs_umem **umem_obj);

/**
 * @brief Destroy an instance of DOCA Verbs UMEM.
 *
 * @param [in] umem_obj
 * Pointer to the umem instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 * - DOCA_ERROR_DRIVER - in case of error in a DOCA driver call.
 */
doca_error_t doca_verbs_umem_destroy(struct doca_verbs_umem *umem_obj);

/**
 * @brief This method retrieves the umem id
 *
 * @param [in] umem_obj
 * Pointer to the umem instance.
 * @param [out] umem_id
 * the umem id.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_verbs_umem_get_id(const struct doca_verbs_umem *umem_obj, uint32_t *umem_id);

/**
 * @brief This method retrieves the umem size
 *
 * @param [in] umem_obj
 * Pointer to the umem instance.
 * @param [out] umem_size
 * the umem size.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_verbs_umem_get_size(const struct doca_verbs_umem *umem_obj, size_t *umem_size);

/**
 * @brief This method retrieves the umem address
 *
 * @param [in] umem_obj
 * Pointer to the umem instance.
 * @param [out] umem_address
 * the umem address.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - if an invalid input had been received.
 */
doca_error_t doca_verbs_umem_get_address(const struct doca_verbs_umem *umem_obj,
                                         void **umem_address);

/**
 * @brief Create a UAR object
 *
 * @param [in] context
 * Pointer to ibv_context
 * @param [in] allocation_type
 * doca_uar_allocation_type
 * @param [out] uar
 * UAR object
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_DRIVER - in case of error in a DOCA driver call.
 */
doca_error_t doca_verbs_uar_create(struct ibv_context *context,
                                   enum doca_verbs_uar_allocation_type allocation_type,
                                   struct doca_verbs_uar **uar_obj);

/**
 * @brief Destroy a UAR object
 *
 * @param [in] uar
 * UAR object
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_uar_destroy(struct doca_verbs_uar *uar_obj);

/**
 * @brief This method retrieves the UAR ID
 *
 * @param [in] uar
 * UAR object
 * @param [out] id
 * The UAR ID
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_uar_id_get(const struct doca_verbs_uar *uar, uint32_t *id);

/**
 * @brief This method retrieves the uar register address
 *
 * @param [in] uar
 * UAR object
 * @param [out] reg_addr
 * UAR register address
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_uar_reg_addr_get(const struct doca_verbs_uar *uar_obj, void **reg_addr);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_H */
