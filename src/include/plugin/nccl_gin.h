/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_GIN_H_
#define NCCL_GIN_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include <stdint.h>

#define NCCL_GIN_HANDLE_MAXSIZE 128
#define MAX_GIN_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// Max number of ncclNet objects which can live in the same process
#ifndef NCCL_GIN_MAX_PLUGINS
#define NCCL_GIN_MAX_PLUGINS 16
#endif

#define NCCL_GIN_SIGNAL_OP_INC 0x1
#define NCCL_GIN_SIGNAL_OP_ADD 0x2

#include "gin/gin_v12.h"
#include "gin/gin_v11.h"

typedef ncclGin_v12_t ncclGin_t;

#define NCCL_GIN_PLUGIN_SYMBOL ncclGinPlugin_v12

#endif // end include guard
