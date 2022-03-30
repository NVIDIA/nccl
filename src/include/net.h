/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"
#include "checks.h"

extern ncclNet_t* ncclNet;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

ncclResult_t ncclNetInit();
int ncclNetVersion();

// Translation to external API
static const char* ncclNetName() { return ncclNet->name; }
static ncclResult_t ncclNetDevices(int* ndev) { NCCLCHECK(ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetGetProperties(int dev, ncclNetProperties_t* props) { NCCLCHECK(ncclNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t ncclNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm) { NCCLCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(void* listenComm, void** recvComm) { NCCLCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetRegMr(void* comm, void* data, int size, int type, void** mhandle) { NCCLCHECK(ncclNet->regMr(comm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetDeregMr(void* comm, void* mhandle) { NCCLCHECK(ncclNet->deregMr(comm, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { NCCLCHECK(ncclNet->isend(sendComm, data, size, tag, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { NCCLCHECK(ncclNet->irecv(recvComm, n, data, sizes, tags, mhandles, request)); return ncclSuccess; }
static ncclResult_t ncclNetIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { NCCLCHECK(ncclNet->iflush(recvComm, n, data, sizes, mhandles, request)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* sizes) { NCCLCHECK(ncclNet->test(request, done, sizes)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(void* sendComm) { NCCLCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(void* recvComm) { NCCLCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(void* listenComm) { NCCLCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
