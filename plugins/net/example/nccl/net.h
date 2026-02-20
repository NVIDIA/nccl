/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_H_
#define NET_H_

#include <stdint.h>
#include <stdlib.h>

#include "err.h"
#include "net_device.h"
#include "common.h"

#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) //1TB
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 32
#include "net_v12.h"
#include "net_v11.h"
#include "net_v10.h"
#include "net_v9.h"
#include "net_v8.h"
#include "net_v7.h"
#include "net_v6.h"
#include "net_v5.h"
#include "net_v4.h"
#include "net_v3.h"
#include "net_v2.h"

#define NCCL_NET_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC_V12

typedef ncclNet_v12_t ncclNet_t;
typedef ncclNetProperties_v12_t ncclNetProperties_t;
typedef ncclNetVDeviceProps_v12_t ncclNetVDeviceProps_t;
typedef ncclNetCommConfig_v12_t ncclNetCommConfig_t;

#define NCCL_GIN_HANDLE_MAXSIZE 128
#define MAX_GIN_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// Max number of ncclNet objects which can live in the same process
#ifndef NCCL_GIN_MAX_PLUGINS
#define NCCL_GIN_MAX_PLUGINS 16
#endif

#define NCCL_GIN_SIGNAL_OP_INC 0x1
#define NCCL_GIN_SIGNAL_OP_ADD 0x2

#include "gin_v11.h"

typedef ncclGin_v11_t ncclGin_t;

#endif // end include guard
