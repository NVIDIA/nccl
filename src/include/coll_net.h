/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef COLL_NET_H_
#define COLL_NET_H_

#include "nccl.h"
#include "nccl_net.h"

typedef char collNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* collNetName(struct ncclComm* comm) { return comm->ncclCollNet->name; }
static ncclResult_t collNetDevices(struct ncclComm* comm, int* ndev) { NCCLCHECK(comm->ncclCollNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t collNetGetProperties(struct ncclComm* comm, int dev, ncclNetProperties_t* props) { NCCLCHECK(comm->ncclCollNet->getProperties(dev, props)); return ncclSuccess; }
static ncclResult_t collNetListen(struct ncclComm* comm, int dev, void* handle, void** listenComm) { NCCLCHECK(comm->ncclCollNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t collNetConnect(struct ncclComm* comm, void* handles[], int nranks, int rank, void* listenComm, void** collComm) { NCCLCHECK(comm->ncclCollNet->connect(handles, nranks, rank, listenComm, collComm)); return ncclSuccess; }
static ncclResult_t collNetReduceSupport(struct ncclComm* comm, ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) { NCCLCHECK(comm->ncclCollNet->reduceSupport(dataType, redOp, supported)); return ncclSuccess; }
static ncclResult_t collNetRegMr(struct ncclComm* comm, void* collComm, void* data, size_t size, int type, void** mhandle) { NCCLCHECK(comm->ncclCollNet->regMr(collComm, data, size, type, mhandle)); return ncclSuccess; }
/* DMA-BUF support */
static ncclResult_t collNetRegMrDmaBuf(struct ncclComm* comm, void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { NCCLCHECK(comm->ncclCollNet->regMrDmaBuf(collComm, data, size, type, offset, fd, mhandle)); return ncclSuccess; }
static ncclResult_t collNetDeregMr(struct ncclComm* comm, void* collComm, void* mhandle) { NCCLCHECK(comm->ncclCollNet->deregMr(collComm, mhandle)); return ncclSuccess; }
static ncclResult_t collNetIallreduce(struct ncclComm* comm, void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle,  void** request) {
  NCCLCHECK(comm->ncclCollNet->iallreduce(collComm, sendData, recvData, count, dataType, redOp, sendMhandle, recvMhandle, request)); return ncclSuccess; }
static ncclResult_t collNetIflush(struct ncclComm* comm, void* collComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(comm->ncclCollNet->iflush(collComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t collNetTest(struct ncclComm* comm, void* request, int* done, int* size) { NCCLCHECK(comm->ncclCollNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t collNetCloseColl(struct ncclComm* comm, void* collComm) { NCCLCHECK(comm->ncclCollNet->closeColl(collComm)); return ncclSuccess; }
static ncclResult_t collNetCloseListen(struct ncclComm* comm, void* listenComm) { NCCLCHECK(comm->ncclCollNet->closeListen(listenComm)); return ncclSuccess; }

static int collNetSupport(struct ncclComm* comm) { return comm->ncclCollNet != nullptr ? 1 : 0; }

#endif
