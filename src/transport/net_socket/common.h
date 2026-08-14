/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_SOCKET_COMMON_H_
#define NCCL_NET_SOCKET_COMMON_H_

#include "core.h"
#include "socket.h"

/* Device discovery shared by NET/Socket and RMA/Socket. */
struct ncclNetSocketDev {
  union ncclSocketAddress addr;
  char devName[MAX_IF_NAME_SIZE];
  char* pciPath;
};

extern int ncclNetIfs;
extern struct ncclNetSocketDev ncclNetSocketDevs[MAX_IFS];

ncclResult_t ncclNetSocketInitDevices(const char* logPrefix);
ncclResult_t ncclNetSocketGetSpeed(char* devName, int* speed);
/* Initialize a caller-owned listener and publish its bootstrap address and magic. */
ncclResult_t ncclNetSocketCreateListener(const char* logPrefix, int dev, struct ncclSocket* sock,
                                         union ncclSocketAddress* connectAddr, uint64_t* magic);

#endif
