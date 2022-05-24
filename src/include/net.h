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

// Translation to external API
static const char* ncclNetName(struct ncclComm* comm) { return comm->ncclNet->name; }
static ncclResult_t ncclNetDevices(struct ncclComm* comm, int* ndev) { NCCLCHECK(comm->ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetGetProperties(struct ncclComm* comm, int dev, ncclNetProperties_t* props) { NCCLCHECK(comm->ncclNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t ncclNetListen(struct ncclComm* comm, int dev, void* handle, void** listenComm) { NCCLCHECK(comm->ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(struct ncclComm* comm, int dev, void* handle, void** sendComm) { NCCLCHECK(comm->ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(struct ncclComm* comm, void* listenComm, void** recvComm) { NCCLCHECK(comm->ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetRegMr(struct ncclComm* comm, void* netComm, void* data, int size, int type, void** mhandle) { NCCLCHECK(comm->ncclNet->regMr(netComm, data, size, type, mhandle)); return ncclSuccess; }
/* DMA-BUF support */
static ncclResult_t ncclNetRegMrDmaBuf(struct ncclComm* comm, void* netComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { NCCLCHECK(comm->ncclNet->regMrDmaBuf(netComm, data, size, type, offset, fd, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetDeregMr(struct ncclComm* comm, void* netComm, void* mhandle) { NCCLCHECK(comm->ncclNet->deregMr(netComm, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(struct ncclComm* comm, void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { NCCLCHECK(comm->ncclNet->isend(sendComm, data, size, tag, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(struct ncclComm* comm, void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { NCCLCHECK(comm->ncclNet->irecv(recvComm, n, data, sizes, tags, mhandles, request)); return ncclSuccess; }
static ncclResult_t ncclNetIflush(struct ncclComm* comm, void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { NCCLCHECK(comm->ncclNet->iflush(recvComm, n, data, sizes, mhandles, request)); return ncclSuccess; }
static ncclResult_t ncclNetTest(struct ncclComm* comm, void* request, int* done, int* sizes) { NCCLCHECK(comm->ncclNet->test(request, done, sizes)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(struct ncclComm* comm, void* sendComm) { NCCLCHECK(comm->ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(struct ncclComm* comm, void* recvComm) { NCCLCHECK(comm->ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(struct ncclComm* comm, void* listenComm) { NCCLCHECK(comm->ncclNet->closeListen(listenComm)); return ncclSuccess; }

// Test whether the current GPU support GPU Direct RDMA.
ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport);

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
