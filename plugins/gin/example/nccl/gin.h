/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_H_
#define GIN_H_

#include <stdint.h>
#include <stdlib.h>

#include "err.h"
#include "net_device.h"
#include "common.h"

#define NCCL_GIN_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

#define NCCL_GIN_SIGNAL_OP_INC 0x1
#define NCCL_GIN_SIGNAL_OP_ADD 0x2

#define NCCL_NET_MR_FLAG_FORCE_SO (1 << 0)

/* Net properties needed by GIN plugin interfaces */

#define NCCL_NET_MAX_DEVS_PER_NIC_V11 4

typedef struct {
  int ndevs;
  int devs[NCCL_NET_MAX_DEVS_PER_NIC_V11];
} ncclNetVDeviceProps_v11_t;

typedef struct {
  char* name;
  char* pciPath;
  uint64_t guid;
  int ptrSupport;
  int regIsGlobal;
  int forceFlush;
  int speed;
  int port;
  float latency;
  int maxComms;
  int maxRecvs;
  ncclNetDeviceType netDeviceType;
  int netDeviceVersion;
  ncclNetVDeviceProps_v11_t vProps;
  size_t maxP2pBytes;
  size_t maxCollBytes;
  int maxMultiRequestSize;
} ncclNetProperties_v11_t;

#define NCCL_NET_MAX_DEVS_PER_NIC_V12 8

typedef struct {
  int ndevs;
  int devs[NCCL_NET_MAX_DEVS_PER_NIC_V12];
} ncclNetVDeviceProps_v12_t;

typedef struct {
  char* name;
  char* pciPath;
  uint64_t guid;
  int ptrSupport;
  int regIsGlobal;
  int forceFlush;
  int speed;
  int port;
  float latency;
  int maxComms;
  int maxRecvs;
  ncclNetDeviceType netDeviceType;
  int netDeviceVersion;
  ncclNetVDeviceProps_v12_t vProps;
  size_t maxP2pBytes;
  size_t maxCollBytes;
  int maxMultiRequestSize;
  int16_t railId;
  int16_t planeId;
} ncclNetProperties_v12_t;

/* GIN plugin versioned interfaces */
#include "gin_v13.h"
#include "gin_v12.h"
#include "gin_v11.h"

typedef ncclGin_v13_t ncclGin_t;
typedef ncclGinConfig_v13_t ncclGinConfig_t;

#endif // end include guard
