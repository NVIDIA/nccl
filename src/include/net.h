/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"
#include "comm.h"
#include "checks.h"

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetPluginInit();
ncclResult_t ncclNetInit(struct ncclComm* comm);
int ncclNetVersion(struct ncclComm* comm);

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
