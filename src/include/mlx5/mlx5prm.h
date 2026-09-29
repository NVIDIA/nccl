/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MLX5PRM_H_
#define NCCL_MLX5PRM_H_

/* mlx5 PRM layouts of the firmware commands that NCCL issues itself through DevX (wrap_mlx5dv_devx_obj_*).
 * Command mailboxes are arrays of big-endian 32-bit dwords (DW): *_SIZE_DW are mailbox sizes and the other *_DW
 * values are field offsets, both counted in dwords. Names are NCCL_ prefixed because the DOCA GPUNetIO mlx5_ifc.h,
 * which may be included alongside this header, defines some of the same PRM names.
 */

/* Command header, common to all commands */
#define NCCL_MLX5_CMD_IN_OPCODE_DW 0  /* bits 31:16 */
#define NCCL_MLX5_CMD_OUT_STATUS_DW 0 /* bits 31:24 */
#define NCCL_MLX5_CMD_OUT_SYNDROME_DW 1

/* ALLOC_Q_COUNTER */
#define NCCL_MLX5_CMD_OP_ALLOC_Q_COUNTER 0x771
#define NCCL_MLX5_ALLOC_Q_COUNTER_IN_SIZE_DW 4
#define NCCL_MLX5_ALLOC_Q_COUNTER_OUT_SIZE_DW 4
#define NCCL_MLX5_ALLOC_Q_COUNTER_OUT_COUNTER_SET_ID_DW 2 /* bits 7:0 */

/* QUERY_Q_COUNTER */
#define NCCL_MLX5_CMD_OP_QUERY_Q_COUNTER 0x773
#define NCCL_MLX5_QUERY_Q_COUNTER_IN_SIZE_DW 8
#define NCCL_MLX5_QUERY_Q_COUNTER_IN_COUNTER_SET_ID_DW 7 /* bits 7:0 */
#define NCCL_MLX5_QUERY_Q_COUNTER_OUT_SIZE_DW 64

/* 32-bit counters in the QUERY_Q_COUNTER output */
#define NCCL_MLX5_Q_COUNTER_RX_WRITE_REQUESTS_DW 4
#define NCCL_MLX5_Q_COUNTER_RX_READ_REQUESTS_DW 6
#define NCCL_MLX5_Q_COUNTER_RX_ATOMIC_REQUESTS_DW 8
#define NCCL_MLX5_Q_COUNTER_OUT_OF_BUFFER_DW 12
#define NCCL_MLX5_Q_COUNTER_OUT_OF_SEQUENCE_DW 14
#define NCCL_MLX5_Q_COUNTER_DUPLICATE_REQUEST_DW 16
#define NCCL_MLX5_Q_COUNTER_RNR_NAK_RETRY_ERR_DW 18
#define NCCL_MLX5_Q_COUNTER_PACKET_SEQ_ERR_DW 20
#define NCCL_MLX5_Q_COUNTER_IMPLIED_NAK_SEQ_ERR_DW 22
#define NCCL_MLX5_Q_COUNTER_LOCAL_ACK_TIMEOUT_ERR_DW 24
#define NCCL_MLX5_Q_COUNTER_RESP_CQE_ERROR_DW 36
#define NCCL_MLX5_Q_COUNTER_REQ_CQE_ERROR_DW 37
#define NCCL_MLX5_Q_COUNTER_REQ_TRANSPORT_RETRIES_EXCEEDED_DW 45
#define NCCL_MLX5_Q_COUNTER_RESP_CQE_FLUSH_ERROR_DW 47
#define NCCL_MLX5_Q_COUNTER_REQ_CQE_FLUSH_ERROR_DW 48

#endif  // NCCL_MLX5PRM_H_
