/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"
#include "nccl_common.h"
#include "nccl_device/net_device.h"
#include <stdint.h>

#define NCCL_NET_HANDLE_MAXSIZE 128
//Maximum value NCCL can accept for maxP2pBytes and maxCollBytes net properties
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L)
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1
#define NCCL_NET_MULTI_REQUEST 0x2

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

#define NCCL_NET_MR_FLAG_FORCE_SO (1 << 0)
#define NCCL_NET_SIGNAL_OP_INC 0x1
#define NCCL_NET_SIGNAL_OP_ADD 0x2

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 32

// Max number of ncclNet objects which can live in the same process
#ifndef NCCL_NET_MAX_PLUGINS
#define NCCL_NET_MAX_PLUGINS 16
#endif

#define NCCL_NET_MAX_DEVS_PER_NIC 4

#include "net/net_v11.h"
#include "net/net_v10.h"
#include "net/net_v9.h"
#include "net/net_v8.h"
#include "net/net_v7.h"
#include "net/net_v6.h"

typedef ncclNet_v11_t ncclNet_t;
typedef ncclCollNet_v11_t ncclCollNet_t;
typedef ncclGin_v11_t ncclGin_t;
typedef ncclNetSGE_v11_t ncclNetSGE_t;
typedef ncclNetProperties_v11_t ncclNetProperties_t;
typedef ncclNetAttr_v11_t ncclNetAttr_t;
typedef ncclNetVDeviceProps_v11_t ncclNetVDeviceProps_t;
typedef ncclNetCommConfig_v11_t ncclNetCommConfig_t;

#define NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v11
#define NCCL_COLLNET_PLUGIN_SYMBOL ncclCollNetPlugin_v11

#endif // end include guard
