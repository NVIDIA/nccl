/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RMA_H_
#define NCCL_RMA_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include <stdint.h>
#include "nccl_gin.h"

// Max number of ncclNet objects which can live in the same process
#ifndef NCCL_RMA_MAX_PLUGINS
#define NCCL_RMA_MAX_PLUGINS 16
#endif

#include "rma/rma_v14.h"
#include "rma/rma_v13.h"

typedef ncclRma_v14_t ncclRma_t;
typedef ncclRmaConfig_v14_t ncclRmaConfig_t;

#endif // end include guard
