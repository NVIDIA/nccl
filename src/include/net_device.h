/*************************************************************************
 * Copyright (c) 2023-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_DEVICE_H_
#define NET_DEVICE_H_

#include "devcomm.h"

#define NCCL_NET_DEVICE_INVALID_VERSION      0x0
#define NCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7  

typedef enum {NCCL_NET_DEVICE_HOST=0, NCCL_NET_DEVICE_UNPACK=1} ncclNetDeviceType;

struct ncclNetDeviceHandle {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
};

#endif // NET_DEVICE_H_
