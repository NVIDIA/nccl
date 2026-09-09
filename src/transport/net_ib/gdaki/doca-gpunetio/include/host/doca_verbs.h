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
#include "doca_gpunetio.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief DOCA Verbs library type: SDK or open
 *
 */
enum doca_verbs_lib_type {
    /* Use DOCA Verbs open source. */
    DOCA_VERBS_SDK_LIB_TYPE_OPEN = 0,
    /* Use DOCA Verbs SDK. */
    DOCA_VERBS_SDK_LIB_TYPE_SDK = 1
};

/**
 * @brief DOCA Verbs UAR allocation type.
 * Same for SDK and open.
 */
enum doca_verbs_uar_allocation_type {
    DOCA_VERBS_UAR_ALLOCATION_TYPE_BLUEFLAME = 0,
    DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE = 1,
    DOCA_VERBS_UAR_ALLOCATION_TYPE_NONCACHE_DEDICATED = 2,
};

/**
 * @brief Verbs QP ordering semantic.
 */
enum doca_verbs_qp_ordering_semantic {
    DOCA_VERBS_QP_ORDERING_SEMANTIC_IBTA = 0x0,
    DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_RW = 0x1,
    DOCA_VERBS_QP_ORDERING_SEMANTIC_OOO_ALL = 0x2,
};

/**********************************************************************************************************************
 * DOCA Verbs opaque types
 *********************************************************************************************************************/
/**
 * Opaque structure representing a DOCA DEV open instance.
 */
struct doca_dev_open;
/**
 * Opaque structure representing a DOCA Verbs DEV handler open and SDK.
 */
typedef struct doca_dev {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_dev_open *open;
    };
    void *sdk_pd;
    void *sdk_context;
} doca_dev_t;

/**
 * Opaque structure representing a DOCA Verbs QP Init Attributes open instance.
 */
struct doca_verbs_qp_init_attr_open;
/**
 * Opaque structure representing a DOCA Verbs QP Init Attributes handler open or SDK.
 */
typedef struct {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_qp_init_attr_open *open;
    };
} doca_verbs_qp_init_attr_t;
/**
 * Opaque structure representing a DOCA Verbs QP Attributes open instance.
 */
struct doca_verbs_qp_attr_open;
/**
 * Opaque structure representing a DOCA Verbs QP Attributes handler open or SDK.
 */
typedef struct {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_qp_attr_open *open;
    };
} doca_verbs_qp_attr_t;
/**
 * Opaque structure representing a DOCA Completion Queue open instance.
 */
struct doca_verbs_qp_open;
/**
 * Opaque structure representing a DOCA Verbs Queue Pair handler open and SDK.
 */
struct doca_verbs_qp_t {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_qp_open *open;
    };
};

/**
 * Opaque structure representing a DOCA Verbs CQ Attributes open instance.
 */
struct doca_verbs_cq_attr_open;
/**
 * Opaque structure representing a DOCA Verbs CQ Attributes handler open or SDK.
 */
typedef struct {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_cq_attr_open *open;
    };
} doca_verbs_cq_attr_t;
/**
 * Opaque structure representing a DOCA Completion Queue open instance.
 */
struct doca_verbs_cq_open;
struct mlx5dv_devx_obj;
/**
 * Opaque structure representing a DOCA Verbs Completion Queue handler open and SDK.
 */
struct doca_verbs_cq_t {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_cq_open *open;
    };
};

/**
 * Opaque structure representing a DOCA Comp Channel open instance.
 */
struct doca_verbs_comp_channel_open;
/**
 * Opaque structure representing a DOCA Verbs Comp Channel handler open and SDK.
 */
typedef struct doca_verbs_comp_channel {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_comp_channel_open *open;
    };
} doca_verbs_comp_channel_t;

/**
 * Opaque structure representing a DOCA Verbs Shared Receive Queue instance.
 */
struct doca_verbs_srq;
/**
 * Opaque structure representing a DOCA Verbs SRQ Init Attributes
 */
struct doca_verbs_srq_init_attr;
/**
 * Opaque structure representing a DOCA AH attr open instance.
 */
struct doca_verbs_ah_attr_open;
/**
 * Opaque structure representing a DOCA Verbs AH Attr handler open and SDK.
 */
struct doca_verbs_ah_attr_t {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_ah_attr_open *open;
    };
};

/**
 * Opaque structure representing a DOCA Verbs CC group handle open instance.
 */
struct doca_verbs_cc_group_open;
/**
 * Opaque structure representing a DOCA Verbs CC group handler open and SDK.
 */
struct doca_verbs_cc_group_t {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_cc_group_open *open;
    };
};

/**
 * Opaque structure representing a DOCA UMEM open instance.
 */
struct doca_verbs_umem_open;
/**
 * Opaque structure representing a DOCA Verbs UMEM handler open and SDK.
 */
typedef struct {
    enum doca_verbs_lib_type type;
    union {
        void *sdk;
        struct doca_verbs_umem_open *open;
    };
} doca_verbs_umem_t;

/**
 * Opaque structure representing a DOCA Verbs UAR handler open and SDK.
 */
typedef struct {
    enum doca_verbs_lib_type type;
    enum doca_verbs_uar_allocation_type allocation_type;
    union {
        void *sdk;
        struct doca_verbs_uar_open *open;
    };
} doca_verbs_uar_t;

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
 * @brief DOCA Verbs Atomic Mode.
 */
enum doca_verbs_qp_atomic_mode {
    DOCA_VERBS_QP_ATOMIC_MODE_NONE = 0x0,
    DOCA_VERBS_QP_ATOMIC_MODE_IB_SPEC = 0x1,
    DOCA_VERBS_QP_ATOMIC_MODE_ONLY_8BYTES = 0x2,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_8BYTES = 0x3,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_16BYTES = 0x4,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_32BYTES = 0x5,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_64BYTES = 0x6,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_128BYTES = 0x7,
    DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_256BYTES = 0x8
};

/**
 * @brief DOCA Verbs QP Send DBR Mode.
 */
enum doca_verbs_qp_send_dbr_mode {
    DOCA_VERBS_QP_SEND_DBR_MODE_DBR_VALID = 0x0,
    DOCA_VERBS_QP_SEND_DBR_MODE_NO_DBR_EXT = 0x1,
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
 * @brief Atomic Mode attribute.
 */
#define DOCA_VERBS_QP_ATTR_ATOMIC_MODE (1 << 15)

/**
 * @brief The maximum number of outstanding RDMA Read/Atomic requests that a single QP is allowed to
 * initiate concurrently.
 */
#define DOCA_VERBS_QP_ATTR_MAX_QP_RD_ATOMIC (1 << 16)
/**
 * @brief The maximum number of incoming RDMA Read/Atomic requests that a single QP can handle
 * concurrently as a responder.
 */
#define DOCA_VERBS_QP_ATTR_MAX_DEST_RD_ATOMIC (1 << 17)

/**
 * @brief CC group (congestion control) attribute (experimental SDK).
 */
#define DOCA_VERBS_QP_ATTR_CC_GROUP (1 << 18)

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
 * @brief Create a DOCA Verbs Device instance.
 *
 * @param [in] verbs_context
 * ibv_context
 * @param [in] verbs_pd
 * ibv_pd
 * @param [out] net_dev
 * network device handler
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_dev_open(struct ibv_pd *verbs_pd, doca_dev_t **net_dev);

/**
 * @brief Close a DOCA Verbs Device instance.
 *
 * @param [out] net_dev
 * network device handler
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_dev_close(doca_dev_t *net_dev);

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
doca_error_t doca_verbs_qp_init_attr_create(doca_verbs_qp_init_attr_t **verbs_qp_init_attr);

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
doca_error_t doca_verbs_qp_init_attr_destroy(doca_verbs_qp_init_attr_t *qp_init_attr);

/**
 * @brief Set pd attribute for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] net_dev
 * net_dev handler with pd attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_pd(doca_verbs_qp_init_attr_t *qp_init_attr,
                                            doca_dev_t *net_dev);

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
doca_error_t doca_verbs_qp_init_attr_set_send_cq(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                 doca_verbs_cq_t *send_cq);

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
doca_error_t doca_verbs_qp_init_attr_set_receive_cq(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                    doca_verbs_cq_t *receive_cq);

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
doca_error_t doca_verbs_qp_init_attr_set_sq_sig_all(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                    int sq_sig_all);

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
doca_error_t doca_verbs_qp_init_attr_set_sq_wr(doca_verbs_qp_init_attr_t *qp_init_attr,
                                               uint32_t sq_wr);

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
doca_error_t doca_verbs_qp_init_attr_set_rq_wr(doca_verbs_qp_init_attr_t *qp_init_attr,
                                               uint32_t rq_wr);

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
doca_error_t doca_verbs_qp_init_attr_set_send_max_sges(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                       uint32_t send_max_sges);

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
doca_error_t doca_verbs_qp_init_attr_set_receive_max_sges(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                          uint32_t receive_max_sges);

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
doca_error_t doca_verbs_qp_init_attr_set_max_inline_data(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                         uint32_t max_inline_data);

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
doca_error_t doca_verbs_qp_init_attr_set_user_index(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                    uint32_t user_index);

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
doca_error_t doca_verbs_qp_init_attr_set_qp_type(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                 uint32_t qp_type);

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
doca_error_t doca_verbs_qp_init_attr_set_external_umem(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                       doca_verbs_umem_t *external_umem,
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
 * @param [in] external_umem_dbr
 * External dbr umem instance.
 * @param [in] external_umem_dbr_offset
 * The offset in the external dbr umem buffer to set the DBR
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_external_umem_dbr(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                           doca_verbs_umem_t *external_umem,
                                                           uint64_t external_umem_offset);

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
doca_error_t doca_verbs_qp_init_attr_set_external_uar(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                      doca_verbs_uar_t *external_uar);

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
doca_error_t doca_verbs_qp_init_attr_set_qp_context(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                    void *qp_context);

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
doca_error_t doca_verbs_qp_init_attr_set_srq(doca_verbs_qp_init_attr_t *qp_init_attr,
                                             struct doca_verbs_srq *srq);

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
doca_error_t doca_verbs_qp_init_attr_set_core_direct_master(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                            uint8_t core_direct_master);

/**
 * @brief Set Send DBR Mode for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] send_dbr_mode
 * Send DBR Mode attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_send_dbr_mode(
    doca_verbs_qp_init_attr_t *qp_init_attr, enum doca_verbs_qp_send_dbr_mode send_dbr_mode);

/**
 * @brief Get Send DBR Mode attribute from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [out] send_dbr_mode
 * Send DBR Mode attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_get_send_dbr_mode(
    const doca_verbs_qp_init_attr_t *qp_init_attr, enum doca_verbs_qp_send_dbr_mode *send_dbr_mode);

/**
 * @brief Set the emulate no dbr ext flag for verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] emulate_no_dbr_ext
 * The emulate no dbr ext flag.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_set_emulate_no_dbr_ext(doca_verbs_qp_init_attr_t *qp_init_attr,
                                                            bool emulate_no_dbr_ext);

/**
 * @brief Get the emulate no dbr ext flag from verbs_qp_init_attr
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [out] emulate_no_dbr_ext
 * The emulate no dbr ext value.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_init_attr_get_emulate_no_dbr_ext(
    const doca_verbs_qp_init_attr_t *qp_init_attr, bool *emulate_no_dbr_ext);

/**
 * @brief Set ordering semantic attribute for verbs_qp_init_attr
 * @note Not setting ordering semantic doesn't guarantee ordering semantic didn't change (setting
 * ECE may change it) Supported only by SDK mode.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [in] ordering_semantic
 * ordering semantic attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - if qp_init_attr is called in open mode.
 */
doca_error_t doca_verbs_qp_init_attr_set_ordering_semantic(
    doca_verbs_qp_init_attr_t *verbs_qp_init_attr,
    enum doca_verbs_qp_ordering_semantic ordering_semantic);

/**
 * @brief Get ordering semantic attribute from verbs_qp_init_attr
 * Supported only by SDK mode.
 *
 * @param [in] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [out] ordering_semantic
 * ordering semantic attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - if qp_init_attr is called in open mode.
 */
doca_error_t doca_verbs_qp_init_attr_get_ordering_semantic(
    doca_verbs_qp_init_attr_t *verbs_qp_init_attr,
    enum doca_verbs_qp_ordering_semantic *ordering_semantic);

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
doca_error_t doca_verbs_qp_attr_create(doca_verbs_qp_attr_t **verbs_qp_attr);

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
doca_error_t doca_verbs_qp_attr_destroy(doca_verbs_qp_attr_t *qp_attr);

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
doca_error_t doca_verbs_qp_attr_set_next_state(doca_verbs_qp_attr_t *qp_attr,
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
enum doca_verbs_qp_state doca_verbs_qp_attr_get_next_state(const doca_verbs_qp_attr_t *qp_attr);

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
doca_error_t doca_verbs_qp_attr_set_current_state(doca_verbs_qp_attr_t *qp_attr,
                                                  enum doca_verbs_qp_state current_state);

/**
 * @brief Get current_state attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [out] current_state
 * current_state attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_get_current_state(doca_verbs_qp_attr_t *qp_attr,
                                                  enum doca_verbs_qp_state *current_state);

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
doca_error_t doca_verbs_qp_attr_set_path_mtu(doca_verbs_qp_attr_t *qp_attr,
                                             enum doca_verbs_mtu_size path_mtu);

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
doca_error_t doca_verbs_qp_attr_set_rq_psn(doca_verbs_qp_attr_t *qp_attr, uint32_t rq_psn);

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
doca_error_t doca_verbs_qp_attr_set_sq_psn(doca_verbs_qp_attr_t *qp_attr, uint32_t sq_psn);

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
doca_error_t doca_verbs_qp_attr_set_dest_qp_num(doca_verbs_qp_attr_t *qp_attr,
                                                uint32_t dest_qp_num);

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
doca_error_t doca_verbs_qp_attr_set_allow_remote_write(doca_verbs_qp_attr_t *qp_attr,
                                                       int allow_remote_write);

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
doca_error_t doca_verbs_qp_attr_set_allow_remote_read(doca_verbs_qp_attr_t *qp_attr,
                                                      int allow_remote_read);

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
doca_error_t doca_verbs_qp_attr_set_atomic_mode(doca_verbs_qp_attr_t *qp_attr,
                                                enum doca_verbs_qp_atomic_mode allow_atomic_type);

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
doca_error_t doca_verbs_qp_attr_set_ah_attr(doca_verbs_qp_attr_t *qp_attr,
                                            doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_qp_attr_set_pkey_index(doca_verbs_qp_attr_t *qp_attr, uint16_t pkey_index);

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
doca_error_t doca_verbs_qp_attr_set_port_num(doca_verbs_qp_attr_t *qp_attr, uint16_t port_num);

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
doca_error_t doca_verbs_qp_attr_set_ack_timeout(doca_verbs_qp_attr_t *qp_attr,
                                                uint16_t ack_timeout);

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
doca_error_t doca_verbs_qp_attr_set_retry_cnt(doca_verbs_qp_attr_t *qp_attr, uint16_t retry_cnt);

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
doca_error_t doca_verbs_qp_attr_set_rnr_retry(doca_verbs_qp_attr_t *qp_attr, uint16_t rnr_retry);

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
doca_error_t doca_verbs_qp_attr_set_min_rnr_timer(doca_verbs_qp_attr_t *qp_attr,
                                                  uint16_t min_rnr_timer);

/**
 * @brief Set max_rd_atomic attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] max_rd_atomic
 * max_rd_atomic attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_max_rd_atomic(doca_verbs_qp_attr_t *qp_attr,
                                                  uint8_t max_rd_atomic);

/**
 * @brief Set max_dest_rd_atomic attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] max_dest_rd_atomic
 * max_dest_rd_atomic attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_attr_set_max_dest_rd_atomic(doca_verbs_qp_attr_t *qp_attr,
                                                       uint8_t max_dest_rd_atomic);

/**
 * @brief Associate an experimental CC group with QP attribute state (DOCA SDK / runtime dlopen
 * path).
 *
 * @param[in] verbs_qp_attr QP attributes.
 * @param[in] cc_group CC group handle from doca_verbs_cc_group_* (or NULL to clear when supported).
 *
 * @return DOCA_SUCCESS on success, doca_error code on failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - called in open-source mode without SDK.
 * - DOCA_SDK_WRAPPER_NOT_SUPPORTED - SDK wrapper path is enabled but CC-group symbols are
 * unavailable.
 */
doca_error_t doca_verbs_qp_attr_set_cc_group(doca_verbs_qp_attr_t *verbs_qp_attr,
                                             doca_verbs_cc_group_t *cc_group);

/**
 * @brief Set counter_set_id attribute for verbs_qp_attr
 *
 * @param [in] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [in] counter_set_id
 * counter_set_id attribute.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NOT_SUPPORTED - if called in SDK mode.
 */
doca_error_t doca_verbs_qp_attr_set_counter_set_id(doca_verbs_qp_attr_t *qp_attr,
                                                   uint32_t counter_set_id);

/**
 * @brief Create a DOCA Verbs AH instance.
 *
 * @param [in] net_dev
 * Pointer to net_dev with the ibv_context instance.
 * @param [out] verbs_ah
 * Pointer to pointer to be set to point to the created verbs_ah instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_ah_attr_create(doca_dev_t *net_dev, doca_verbs_ah_attr_t **ah_attr);

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
doca_error_t doca_verbs_ah_attr_destroy(doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_gid(doca_verbs_ah_attr_t *ah_attr, struct doca_verbs_gid gid);

/**
 * @brief Get gid attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * gid attribute.
 */
struct doca_verbs_gid doca_verbs_ah_get_gid(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_addr_type(doca_verbs_ah_attr_t *ah_attr,
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
enum doca_verbs_addr_type doca_verbs_ah_get_addr_type(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_dlid(doca_verbs_ah_attr_t *ah_attr, uint32_t dlid);

/**
 * @brief Get dlid attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * dlid attribute.
 */
uint32_t doca_verbs_ah_get_dlid(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_sl(doca_verbs_ah_attr_t *ah_attr, uint8_t sl);

/**
 * @brief Get sl attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * sl attribute.
 */
uint8_t doca_verbs_ah_get_sl(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_sgid_index(doca_verbs_ah_attr_t *ah_attr, uint8_t sgid_index);

/**
 * @brief Get sgid_index attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * sgid_index attribute.
 */
uint8_t doca_verbs_ah_get_sgid_index(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_static_rate(doca_verbs_ah_attr_t *ah_attr, uint8_t static_rate);

/**
 * @brief Get static_rate attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * static_rate attribute.
 */
uint8_t doca_verbs_ah_get_static_rate(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_hop_limit(doca_verbs_ah_attr_t *ah_attr, uint8_t hop_limit);

/**
 * @brief Get hop_limit attribute from verbs_ah.
 *
 * @param [in] verbs_ah
 * Pointer to verbs_ah instance.
 *
 * @return
 * hop_limit attribute.
 */
uint8_t doca_verbs_ah_get_hop_limit(const doca_verbs_ah_attr_t *ah_attr);

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
doca_error_t doca_verbs_ah_attr_set_traffic_class(doca_verbs_ah_attr_t *ah_attr,
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
uint8_t doca_verbs_ah_get_traffic_class(const doca_verbs_ah_attr_t *ah_attr);

/**
 * @brief Create a DOCA Verbs Queue Pair instance.
 *
 * @param [in] net_dev
 * Pointer to net_dev with the ibv_context instance.
 * @param [in] qp_init_attr
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
doca_error_t doca_verbs_qp_create(doca_dev_t *net_dev, doca_verbs_qp_init_attr_t *qp_init_attr,
                                  doca_verbs_qp_t **verbs_qp);

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
doca_error_t doca_verbs_qp_destroy(doca_verbs_qp_t *verbs_qp);

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
doca_error_t doca_verbs_qp_modify(doca_verbs_qp_t *verbs_qp, doca_verbs_qp_attr_t *qp_attr,
                                  int attr_mask);

/**
 * @brief Query the attributes of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [out] verbs_qp_attr
 * Pointer to verbs_qp_attr instance.
 * @param [out] verbs_qp_init_attr
 * Pointer to verbs_qp_init_attr instance.
 * @param [out] attr_mask
 * Pointer to attr_mask. Used only in case of DOCA SDK.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_qp_query(doca_verbs_qp_t *verbs_qp, doca_verbs_qp_attr_t *qp_attr,
                                 doca_verbs_qp_init_attr_t *qp_init_attr, int *attr_mask = nullptr);

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
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_get_wq(const doca_verbs_qp_t *verbs_qp, void **sq_buf,
                                  uint32_t *sq_num_entries, void **rq_buf, uint32_t *rq_num_entries,
                                  uint32_t *rwqe_size_bytes);

/**
 * @brief Get the DBR address of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [out] dbr_addr
 * Pointer to the DBR address.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_get_dbr_addr(const doca_verbs_qp_t *verbs_qp, void **dbr_addr);

/**
 * @brief Get the UAR address of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [out] uar_addr
 * Pointer to the UAR address.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_get_uar_addr(const doca_verbs_qp_t *verbs_qp, void **uar_addr);

/**
 * @brief Get the QP number of a DOCA Verbs Queue Pair instance.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 * @param [out] qpn
 * Pointer to the QP number.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_qp_get_qpn(const doca_verbs_qp_t *verbs_qp, uint32_t *qpn);

/**
 * @brief Get the emulate no dbr ext flag of a DOCA Verbs Queue Pair instance.
 * Emulated no dbr ext is not supported by DOCA SDK.
 *
 * @param [in] verbs_qp
 * Pointer to verbs_qp instance.
 *
 * @return
 * The emulate no dbr ext flag.
 */
bool doca_verbs_qp_get_emulate_no_dbr_ext(const doca_verbs_qp_t *verbs_qp);

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
doca_error_t doca_verbs_cq_attr_create(doca_verbs_cq_attr_t **verbs_cq_attr);

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
doca_error_t doca_verbs_cq_attr_destroy(doca_verbs_cq_attr_t *verbs_cq_attr);

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
doca_error_t doca_verbs_cq_attr_set_cq_size(doca_verbs_cq_attr_t *cq_attr, uint32_t cq_size);

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
doca_error_t doca_verbs_cq_attr_set_cq_context(doca_verbs_cq_attr_t *cq_attr, void *cq_context);

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
doca_error_t doca_verbs_cq_attr_set_external_umem(doca_verbs_cq_attr_t *cq_attr,
                                                  doca_verbs_umem_t *external_umem,
                                                  uint64_t external_umem_offset);

/**
 * @brief Set external DBR umem attribute for doca_verbs_cq_attr.
 *
 * Setting this attribute provides a separate umem for the CQ doorbell record,
 * instead of packing the DBR inside the CQ ring umem.
 * This is used when CQ ring and CQ DBR live in different umem slabs
 * (e.g., for control buffer suballocation).
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] external_dbr_umem
 * External umem instance for the CQ doorbell record.
 * @param [in] external_dbr_umem_offset
 * The offset in the external DBR umem buffer for the CQ doorbell record.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_external_dbr_umem(doca_verbs_cq_attr_t *cq_attr,
                                                      doca_verbs_umem_t *external_dbr_umem,
                                                      uint64_t external_dbr_umem_offset);

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
doca_error_t doca_verbs_cq_attr_set_external_uar(doca_verbs_cq_attr_t *cq_attr,
                                                 doca_verbs_uar_t *external_uar);

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
doca_error_t doca_verbs_cq_attr_set_cq_overrun(doca_verbs_cq_attr_t *cq_attr,
                                               enum doca_verbs_cq_overrun overrun);

/**
 * @brief Enable cq_overrun attribute for doca_verbs_cq_attr.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] cc
 * enable or disable collapsed cq
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_cq_collapsed(doca_verbs_cq_attr_t *cq_attr, uint8_t cc);

/**
 * @brief Set comp_channel attribute for doca_verbs_cq_attr.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] comp_channel
 * Pointer to completion channel to bind the CQ to. comp_channel may be null in case the application
 * regrets setting a completion channel.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_comp_channel(doca_verbs_cq_attr_t *cq_attr,
                                                 doca_verbs_comp_channel_t *comp_channel);

/**
 * @brief CQ doorbell create state
 */
enum doca_verbs_cq_state {
    DOCA_VERBS_CQ_ST_NO_ACTION,
    DOCA_VERBS_CQ_ST_SOLICITED_NOTIFICATION_REQUEST_ARMED,
    DOCA_VERBS_CQ_ST_NOTIFICATION_REQUEST_ARMED,
    DOCA_VERBS_CQ_ST_FIRED,
};

/**
 * @brief Set CQ doorbell create state attribute for doca_verbs_cq_attr.
 *
 * @param [in] cq_attr
 * Pointer to doca_verbs_cq_attr instance.
 * @param [in] cq_state
 * Create state (@see doca_verbs_cq_state).
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_attr_set_st(doca_verbs_cq_attr_t *cq_attr,
                                       enum doca_verbs_cq_state cq_state);

/**
 * @brief Create a DOCA Verbs Completion Queue instance.
 *
 * @param [in] net_dev
 * DOCA network device
 * @param [in] cq_attr
 * Pointer to cq_attr instance.
 * @param [out] cq
 * Pointer to pointer to be set to point to the created doca_verbs_cq_t instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_cq_create(doca_dev_t *net_dev, doca_verbs_cq_attr_t *cq_attr,
                                  doca_verbs_cq_t **cq);

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
doca_error_t doca_verbs_cq_destroy(doca_verbs_cq_t *verbs_cq);

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
doca_error_t doca_verbs_cq_get_wq(doca_verbs_cq_t *verbs_cq, void **cq_buf,
                                  uint32_t *cq_num_entries, uint8_t *cq_entry_size);

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
doca_error_t doca_verbs_cq_get_dbr_db_addr(doca_verbs_cq_t *verbs_cq, uint64_t **uar_db_reg,
                                           uint32_t **ci_dbr, uint32_t **arm_dbr);

/**
 * @brief Get the CQ number of a DOCA Verbs CQ instance.
 *
 * @param [in] verbs_cq
 * Pointer to verbs_cq instance.
 * @param [out] cqn
 * Pointer to the CQ number
 *
 * @return
 * The CQ number.
 */
doca_error_t doca_verbs_cq_get_cq_num(const doca_verbs_cq_t *verbs_cq, uint32_t *cqn);

/**
 * @brief Set cq_context attribute for verbs_cq. This function allows to set cq_context if the
 * desired user data is not available at CQ creation time.
 *
 * @param [in] verbs_cq
 * Pointer to verbs_cq instance.
 * @param [in] cq_context
 * User data. cq_context may be null in case the application regrets setting a user data.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_cq_set_cq_context(doca_verbs_cq_t *verbs_cq, void *cq_context);

/**
 * @brief Create a DOCA Verbs Completion Channel instance.
 *
 * @param [in] net_dev
 * Pointer to net_dev instance.
 * @param [out] verbs_comp_channel
 * Pointer to pointer to be set to point to the created verbs_comp_channel instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 * - DOCA_ERROR_NO_MEMORY - failed to allocate resources.
 */
doca_error_t doca_verbs_comp_channel_create(const doca_dev_t *net_dev,
                                            doca_verbs_comp_channel_t **verbs_comp_channel);

/**
 * @brief Destroy a DOCA Verbs Completion Channel instance.
 *
 * @param [in] verbs_comp_channel
 * Pointer to verbs_comp_channel instance.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_comp_channel_destroy(doca_verbs_comp_channel_t *verbs_comp_channel);

/**
 * @brief Get the next event from a DOCA Verbs Completion Channel.
 *
 * @param [in] verbs_comp_channel
 * Pointer to verbs_comp_channel instance to get the next event from.
 * @param [out] cq_context
 * Pointer to the user-data associated with the event.
 *
 * @return
 * DOCA_SUCCESS - in case of success.
 * doca_error code - in case of failure:
 * - DOCA_ERROR_INVALID_VALUE - received invalid input.
 */
doca_error_t doca_verbs_get_cq_comp_channel_event(doca_verbs_comp_channel_t *verbs_comp_channel,
                                                  void **cq_context);

/**
 * Acknowledge completion events
 *
 * Every event received from doca_verbs_get_cq_event() must be acknowledged using this API.
 * To prevent races, the CQ destroy will wait until all events are acknowledged.
 *
 * @param [in] verbs_cq
 * Pointer to the verbs_cq instance.
 * @param [in] nevents
 * The number of events to acknowledge.
 */
doca_error_t doca_verbs_ack_cq_events(doca_verbs_cq_t *verbs_cq, unsigned int nevents);

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
    struct doca_verbs_srq_init_attr *verbs_srq_init_attr, doca_verbs_umem_t *external_umem,
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
    const struct doca_verbs_srq_init_attr *verbs_srq_init_attr, doca_verbs_umem_t **external_umem,
    uint64_t *external_umem_offset);

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
 * @brief Get the no DBR-ext support flag of the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The no DBR-ext support flag.
 */
uint8_t doca_verbs_device_attr_get_send_dbr_mode_no_dbr_ext(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of outstanding RDMA Read or Atomic requests that a single QP is
 * allowed to initiate concurrently, as supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The maximum number of outstanding RDMA Read or Atomic requests supported by the device.
 */
uint8_t doca_verbs_device_attr_get_max_qp_rd_atom(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Get the maximum number of incoming RDMA Read or Atomic requests that a single QP can
 * handle concurrently as a responder, as supported by the device.
 *
 * @param [in] verbs_device_attr
 * Pointer to doca_verbs_device_attr instance.
 *
 * @return
 * The maximum number of incoming RDMA Read or Atomic requests supported by the device.
 */
uint8_t doca_verbs_device_attr_get_max_qp_init_rd_atom(
    const struct doca_verbs_device_attr *verbs_device_attr);

/**
 * @brief Create an instance of DOCA Verbs UMEM.
 *
 * @param [in] net_dev
 * Network device handler.
 * @param [in] gpu
 * GPU device handler.
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
doca_error_t doca_verbs_umem_create(doca_dev_t *net_dev, doca_gpu_t *gpu, void *address,
                                    size_t size, uint32_t access_flags, int dmabuf_id,
                                    size_t dmabuf_offset, doca_verbs_umem_t **umem_obj);

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
doca_error_t doca_verbs_umem_destroy(doca_verbs_umem_t *umem_obj);

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
doca_error_t doca_verbs_umem_get_id(const doca_verbs_umem_t *umem_obj, uint32_t *umem_id);

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
doca_error_t doca_verbs_umem_get_size(const doca_verbs_umem_t *umem_obj, size_t *umem_size);

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
doca_error_t doca_verbs_umem_get_address(const doca_verbs_umem_t *umem_obj, void **umem_address);

/**
 * @brief Create a UAR object
 *
 * @param [in] net_dev
 * Pointer to doca_dev_t
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
doca_error_t doca_verbs_uar_create(doca_dev_t *net_dev,
                                   enum doca_verbs_uar_allocation_type allocation_type,
                                   doca_verbs_uar_t **uar_obj);

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
doca_error_t doca_verbs_uar_destroy(doca_verbs_uar_t *uar_obj);

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
doca_error_t doca_verbs_uar_id_get(const doca_verbs_uar_t *uar, uint32_t *id);

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
doca_error_t doca_verbs_uar_reg_addr_get(const doca_verbs_uar_t *uar_obj, void **reg_addr);

/**
 * @brief This method retrieves the dbr less address
 *
 * @param [in] uar
 * UAR object
 * @param [out] dbr_less_addr
 * The dbr less address
 */
doca_error_t doca_verbs_uar_dbr_less_addr_get(const doca_verbs_uar_t *uar_obj,
                                              void **dbr_less_addr);

#ifdef __cplusplus
}
#endif

#endif /* DOCA_VERBS_H */
