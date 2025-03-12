/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"
#include "nccl_common.h"
#include "net_device.h"
#include <stdint.h>

#define NCCL_NET_HANDLE_MAXSIZE 128
//Maximum value NCCL can accept for maxP2pBytes and maxCollBytes net properties
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L)
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 32

// Max number of ncclNet objects which can live in the same process
#define NCCL_NET_MAX_PLUGINS 3

// NCCL core profiler callback for network defined events instrumentation
typedef ncclResult_t (*ncclProfilerCallback_t)(void** eHandle, int type, void* pHandle, int64_t pluginId, void* extData);

#include "net/net_v10.h"
#include "net/net_v9.h"
#include "net/net_v8.h"
#include "net/net_v7.h"
#include "net/net_v6.h"

typedef ncclNet_v10_t ncclNet_t;
typedef ncclCollNet_v10_t ncclCollNet_t;
typedef ncclNetSGE_v10_t ncclNetSGE_t;
typedef ncclNetProperties_v10_t ncclNetProperties_t;
typedef ncclNetVDeviceProps_v10_t ncclNetVDeviceProps_t;
typedef ncclNetCommConfig_v10_t ncclNetCommConfig_t;

#define NCCL_NET_MAX_DEVS_PER_NIC NCCL_NET_MAX_DEVS_PER_NIC_V10

#define NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v10
#define NCCL_COLLNET_PLUGIN_SYMBOL ncclCollNetPlugin_v10

#endif // end include guard
