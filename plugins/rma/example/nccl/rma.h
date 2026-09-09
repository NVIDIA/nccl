/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMA_H_
#define RMA_H_

#include <stdint.h>
#include <stdlib.h>

#include "err.h"
#include "common.h"

#define NCCL_RMA_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

#define NCCL_RMA_SIGNAL_OP_INC 0x1
#define NCCL_RMA_SIGNAL_OP_ADD 0x2

#define NCCL_NET_MR_FLAG_FORCE_SO (1 << 0)
#define NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET (1 << 1)

/* Net properties needed by RMA plugin interfaces */

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
  int netDeviceType;
  int netDeviceVersion;
  ncclNetVDeviceProps_v12_t vProps;
  size_t maxP2pBytes;
  size_t maxCollBytes;
  int maxMultiRequestSize;
  int16_t railId;
  int16_t planeId;
} ncclNetProperties_v12_t;

/* RMA plugin versioned interfaces */
#include "rma_v14.h"

typedef ncclRma_v14_t ncclRma_t;
typedef ncclRmaConfig_v14_t ncclRmaConfig_t;

#endif // end include guard
