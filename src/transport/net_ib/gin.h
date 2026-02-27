/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_NET_IB_GIN_H_
#define _NCCL_NET_IB_GIN_H_

#include <stddef.h>
#include <stdint.h>
#include "nccl.h"

struct ncclGinIbCollComm {
  int           rank;
  int           nranks;
  int           connectionId;
  int           nConnections;
  int           queueDepth;
  void*         recvComm;
  void*         sendComm;
  void**        fullRecvComm;
  void**        fullSendComm;
  int           dev;
  void*         ginCtx;
  void*         ibvCtx;
  ncclResult_t (*getProperties)(int dev, void *props);
  ncclResult_t (*allGather)(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len);
  ncclResult_t (*allToAll)(struct ncclGinIbCollComm *cComm, void *srcBuf, void *recvBuf, size_t len);
  ncclResult_t (*getGidIndex)(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr* portAttr, int *gidIndex);
};

#endif
