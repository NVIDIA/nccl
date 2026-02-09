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

#define NCCL_UNDEF_DEV_COUNT -1

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetInit(struct ncclComm* comm);
ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent);
ncclResult_t ncclNetFinalize(struct ncclComm* comm);
ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDev);
ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDev, int* nVirtDev);
ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDev);

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;
extern ncclGin_t ncclGinIbGdaki;
extern ncclGin_t ncclGinIbProxy;

#endif
