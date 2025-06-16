/*************************************************************************
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2004, 2011-2012 Intel Corporation.  All rights reserved.
 * Copyright (c) 2005, 2006, 2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2005 PathScale, Inc.  All rights reserved.
 *
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

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

#endif // NCCL_MLX5DVWRAP_H_
