/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <nccl/net.h>

#define __hidden __attribute__ ((visibility("hidden")))

int max_requests = NCCL_NET_MAX_REQUESTS;

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 0; return ncclSuccess; }

__hidden ncclResult_t pluginPciPath(int dev, char** path) { return ncclInternalError; }
__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) { return ncclInternalError; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v6_t* props) {
  //pluginPciPath(dev, &props.pciPath);
  //pluginPtrSupport(dev, &props.ptrSupport);
  return ncclInternalError;
}
__hidden ncclResult_t pluginListen(int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(int dev, void* handle, void** sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, int size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }

#define PLUGIN_NAME "Plugin"

const ncclNet_v6_t ncclNetPlugin_v6 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v5 Compat */
const ncclNet_v5_t ncclNetPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v4 Compat */
static ncclResult_t pluginGetProperties_v4(int dev, ncclNetProperties_v4_t* props) {
  ncclNetProperties_v6_t props_v6;
  ncclResult_t ret = pluginGetProperties(dev, &props_v6);
  if (ret != ncclSuccess) return ret;
  props->name = props_v6.name;
  props->pciPath = props_v6.pciPath;
  props->guid = props_v6.guid;
  props->ptrSupport = props_v6.ptrSupport;
  props->speed = props_v6.speed;
  props->port = props_v6.port;
  props->maxComms = props_v6.maxComms;
  return ncclSuccess;
}
static ncclResult_t pluginIsend_v4(void *sendComm, void* data, int size, void *mhandle, void** request) {
  return pluginIsend(sendComm, data, size, 0, mhandle, request);
}
static ncclResult_t pluginIrecv_v4(void* recvComm, void* data, int size, void* mhandle, void** request) {
  int tag = 0;
  return pluginIrecv(recvComm, 1, &data, &size, &tag, &mhandle, request);
}
static ncclResult_t pluginIflush_v4(void* recvComm, void* data, int size, void* mhandle, void** request) {
  return pluginIflush(recvComm, 1, &data, &size, &mhandle, request);
}
static ncclResult_t pluginConnect_v4(int dev, void* handle, void** sendComm) {
  ncclResult_t ret;
  do {
    ret = pluginConnect(dev, handle, sendComm);
  } while (ret == ncclSuccess && *sendComm == NULL);
  return ret;
}
static ncclResult_t pluginAccept_v4(void* listenComm, void** recvComm) {
  ncclResult_t ret;
  do {
    ret = pluginAccept(listenComm, recvComm);
  } while (ret == ncclSuccess && *recvComm == NULL);
  return ret;
}
const ncclNet_v4_t ncclNetPlugin_v4 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v4,
  .listen = pluginListen,
  .connect = pluginConnect_v4,
  .accept = pluginAccept_v4,
  .regMr = pluginRegMr,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .iflush = pluginIflush_v4,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v3 Compat */
static ncclResult_t pluginFlush(void* recvComm, void* data, int size, void* mhandle) {
  void* req;
  ncclResult_t ret = pluginIflush_v4(recvComm, data, size, mhandle, &req);
  int done = 0;
  while (ret == ncclSuccess && done == 0) {
    ret = pluginTest(req, &done, NULL);
  }
  return ret;
}
static ncclResult_t pluginInit_v3(ncclDebugLogger_t logFunction) {
  max_requests = NCCL_NET_MAX_REQUESTS_V3;
  return pluginInit(logFunction);
}
#include <string.h>
static ncclResult_t pluginListen_v3(int dev, void* handle, void** listenComm) {
  char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];
  ncclResult_t ret = pluginListen(dev, &pluginHandle, listenComm);
  memcpy(handle, &pluginHandle, NCCL_NET_HANDLE_MAXSIZE_V3);
  return ret;
}
static ncclResult_t pluginConnect_v3(int dev, void* handle, void** sendComm) {
  char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];
  memcpy(&pluginHandle, handle, NCCL_NET_HANDLE_MAXSIZE_V3);
  return pluginConnect_v4(dev, &pluginHandle, sendComm);
}
const ncclNet_v3_t ncclNetPlugin_v3 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v3,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v4,
  .listen = pluginListen_v3,
  .connect = pluginConnect_v3,
  .accept = pluginAccept_v4,
  .regMr = pluginRegMr,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .flush = pluginFlush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};

/* v2 Compat */
const ncclNet_v2_t ncclNetPlugin_v2 = {
  .name = PLUGIN_NAME,
  .init = pluginInit_v3,
  .devices = pluginDevices,
  .pciPath = pluginPciPath,
  .ptrSupport = pluginPtrSupport,
  .listen = pluginListen,
  .connect = pluginConnect_v4,
  .accept = pluginAccept_v4,
  .regMr = pluginRegMr,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .flush = pluginFlush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};
