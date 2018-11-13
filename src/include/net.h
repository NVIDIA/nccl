/*************************************************************************
 * Copyright (c) 2016-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"
#include "nccl_net.h"

extern ncclNet_t* ncclNet;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

/* Socket Interface Selection type */
typedef enum { findSubnetIf   = -1,
    dontCareIf     = -2
} ncclSocketIfSl_t;

// Translation to external API
static const char* ncclNetName() { return ncclNet->name; }
static ncclResult_t ncclNetDevices(int* ndev) { NCCLCHECK(ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetPciPath(int dev, char** path) { NCCLCHECK(ncclNet->pciPath(dev, path)); return ncclSuccess; }
static ncclResult_t ncclNetPtrSupport(int dev, int* supportedTypes) { NCCLCHECK(ncclNet->ptrSupport(dev, supportedTypes)); return ncclSuccess; }
static ncclResult_t ncclNetListen(int dev, void* handle, void** listenComm) { NCCLCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(int dev, void* handle, void** sendComm) { NCCLCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(void* listenComm, void** recvComm) { NCCLCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(void* sendComm, void* data, int size, int type, void** request) { NCCLCHECK(ncclNet->isend(sendComm, data, size, type, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(void* recvComm, void* data, int size, int type, void** request) { NCCLCHECK(ncclNet->irecv(recvComm, data, size, type, request)); return ncclSuccess; }
static ncclResult_t ncclNetFlush(void* recvComm, void* data, int size) { NCCLCHECK(ncclNet->flush(recvComm, data, size)); return ncclSuccess; }
static ncclResult_t ncclNetTest(void* request, int* done, int* size) { NCCLCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(void* sendComm) { NCCLCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(void* recvComm) { NCCLCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(void* listenComm) { NCCLCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

extern ncclResult_t ncclSocketCreateHandle(void* opaqueHandle, const char* str);
extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
