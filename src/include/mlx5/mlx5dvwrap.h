/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MLX5DVWRAP_H_
#define NCCL_MLX5DVWRAP_H_

#include <arpa/inet.h>
#include <netinet/in.h>
#ifdef NCCL_BUILD_MLX5DV
#include <infiniband/mlx5dv.h>
#else
#include "mlx5/mlx5dvcore.h"
#endif

#include "core.h"
#include "ibvwrap.h"
#include <sys/types.h>
#include <unistd.h>

typedef enum mlx5dv_return_enum
{
    MLX5DV_SUCCESS = 0,                   //!< The operation was successful
} mlx5dv_return_t;

ncclResult_t wrap_mlx5dv_symbols(void);
/* NCCL wrappers of MLX5 direct verbs functions */
bool wrap_mlx5dv_is_supported(struct ibv_device *device);
ncclResult_t wrap_mlx5dv_get_data_direct_sysfs_path(struct ibv_context *context, char *buf, size_t buf_len);
/* DMA-BUF support */
ncclResult_t wrap_mlx5dv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);
struct ibv_mr * wrap_direct_mlx5dv_reg_dmabuf_mr(struct ibv_pd *pd, uint64_t offset, size_t length, uint64_t iova, int fd, int access, int mlx5_access);
ncclResult_t wrap_mlx5dv_query_device(struct ibv_context *ctx_in, struct mlx5dv_context *attrs_out);
struct ibv_qp *wrap_mlx5dv_create_qp(struct ibv_context *context, struct ibv_qp_init_attr_ex *qp_attr, struct mlx5dv_qp_init_attr *mlx5_qp_attr);

#endif // NCCL_MLX5DVWRAP_H_
