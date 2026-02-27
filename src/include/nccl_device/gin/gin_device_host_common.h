/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_GIN_DEVICE_HOST_COMMON_H_
#define _NCCL_GIN_DEVICE_HOST_COMMON_H_

#include <cuda.h>
#include "../net_device.h"
#include "../core.h"  // for ncclGin{Signal|Counter}_t

#define NCCL_GIN_MAX_CONNECTIONS 4

typedef struct ncclGinGpuCtx *ncclGinGpuCtx_t;
typedef void *ncclGinWindow_t;

typedef enum ncclGinSignalOp_t {
  ncclGinSignalInc = 0,
  ncclGinSignalAdd,
} ncclGinSignalOp_t;

#endif
