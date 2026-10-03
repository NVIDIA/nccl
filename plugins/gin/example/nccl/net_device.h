/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_DEVICE_H_
#define NET_DEVICE_H_

#define NCCL_NET_DEVICE_INVALID_VERSION      0x0
#define NCCL_NET_MTU_SIZE                    4096

// Arbitrary version number - A given NCCL build will only be compatible with a single device networking plugin
// version. NCCL will check the supplied version number from net->getProperties() and compare to its internal version.
#define NCCL_NET_DEVICE_UNPACK_VERSION 0x7

typedef enum {
  NCCL_NET_DEVICE_HOST=0,
  NCCL_NET_DEVICE_UNPACK=1,
  NCCL_NET_DEVICE_GIN_PROXY=2,
  NCCL_NET_DEVICE_GIN_GDAKI=3,
} ncclNetDeviceType;

// Barrier preference a GIN backend reports in ncclGinProperties_t::barrierOptions (plugin API v15+). NCCL
// picks the barrier algorithm that satisfies it and may change that choice between releases. The device code
// NCCL compiles in for the backend type must make the same choice (ncclGinApi_BarrierOptions<backend>).
typedef enum {
  NCCL_GIN_BARRIER_DEFAULT = 0,          // NCCL's default barrier; may use one signal per rank per barrier.
  NCCL_GIN_BARRIER_SIGNAL_EFFICIENT = 1, // When possible, choose a barrier algo that uses fewer signals
} ncclGinBarrierOptions_t;

typedef struct {
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  void* handle;
  size_t size;
  int needsProxyProgress;
} ncclNetDeviceHandle_v7_t;

typedef ncclNetDeviceHandle_v7_t ncclNetDeviceHandle_v8_t;
typedef ncclNetDeviceHandle_v8_t ncclNetDeviceHandle_v9_t;
typedef ncclNetDeviceHandle_v9_t ncclNetDeviceHandle_v10_t;
typedef ncclNetDeviceHandle_v10_t ncclNetDeviceHandle_v11_t;
typedef ncclNetDeviceHandle_v11_t ncclNetDeviceHandle_v12_t;
typedef ncclNetDeviceHandle_v12_t ncclNetDeviceHandle_t;

#endif
