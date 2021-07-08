/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <nccl.h>
#include <nccl_net.h>

#define __hidden __attribute__ ((visibility("hidden")))

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
__hidden ncclResult_t pluginPciPath(int dev, char** path) { return ncclInternalError; }
__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) { return ncclInternalError; }
__hidden ncclResult_t pluginListen(int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(int dev, void* handle, void** sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, int size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, int size, void* mhandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, void* data, int size, void* mhandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginFlush(void* recvComm, void* data, int size, void* mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }

ncclNet_t NCCL_PLUGIN_SYMBOL = {
  "Dummy",
  pluginInit,
  pluginDevices,
  pluginPciPath,
  pluginPtrSupport,
  pluginListen,
  pluginConnect,
  pluginAccept,
  pluginRegMr,
  pluginDeregMr,
  pluginIsend,
  pluginIrecv,
  pluginFlush,
  pluginTest,
  pluginCloseSend,
  pluginCloseRecv,
  pluginCloseListen
};

__hidden ncclResult_t pluginCollNetInit(ncclDebugLogger_t logFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginCollNetDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
__hidden ncclResult_t pluginCollNetPciPath(int dev, char** path) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetPtrSupport(int dev, int* supportedTypes) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetListen(int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetConnect(void* handles[], int nranks, int rank, void* listenComm, void** collComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetReduceSupport(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetRegMr(void* collComm, void* data, int size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginCollNetIallreduce(void* collComm, void* sendData, void* recvData, int count, ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetFlush(void* collComm, void* data, int size, void* mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetCloseColl(void* collComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCollNetCloseListen(void* listenComm) { return ncclInternalError; }

ncclCollNet_t NCCL_COLLNET_PLUGIN_SYMBOL = {
  "Dummy",
  pluginCollNetInit,
  pluginCollNetDevices,
  pluginCollNetPciPath,
  pluginCollNetPtrSupport,
  pluginCollNetListen,
  pluginCollNetConnect,
  pluginCollNetReduceSupport,
  pluginCollNetRegMr,
  pluginCollNetDeregMr,
  pluginCollNetIallreduce,
  pluginCollNetFlush,
  pluginCollNetTest,
  pluginCollNetCloseColl,
  pluginCollNetCloseListen
};
