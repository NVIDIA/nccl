/*
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 */

#ifndef NET_H_
#define NET_H_

#include <stdint.h>
#include <stdlib.h>

#include "common.h"
#include "err.h"
#include "net_device.h"

#define NCCL_NET_HANDLE_MAXSIZE 128
#define NCCL_MAX_NET_SIZE_BYTES (1*1024*1024*1024*1024L) //1TB
#define NCCL_NET_OPTIONAL_RECV_COMPLETION 0x1

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 32

typedef ncclResult_t (*ncclProfilerCallback_t)(void** eHandle, int type, void* phandle, int64_t pluginId, void* extData);

#include "net_v10.h"
#include "net_v9.h"
#include "net_v8.h"
#include "net_v7.h"
#include "net_v6.h"
#include "net_v5.h"
#include "net_v4.h"
#include "net_v3.h"
#include "net_v2.h"

typedef ncclNet_v10_t ncclNet_t;
typedef ncclNetProperties_v10_t ncclNetProperties_t;
typedef ncclNetVDeviceProps_v10_t ncclNetVDeviceProps_t;
typedef ncclNetCommConfig_v10_t ncclNetCommConfig_t;

#endif // end include guard
