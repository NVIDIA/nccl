/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"

#define __hidden __attribute__ ((visibility("hidden")))

int max_requests = NCCL_NET_MAX_REQUESTS;

__hidden ncclResult_t pluginInit(ncclDebugLogger_t logFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 0; return ncclSuccess; }

__hidden ncclResult_t pluginPciPath(int dev, char** path) { return ncclInternalError; }
__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) { return ncclInternalError; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_v8_t* props) {
  // Below are default values, if unsure don't change.

  props->name = "Example";
  // Fill for proper topology detection, e.g. /sys/devices/pci0000:00/0000:00:10.0/0000:0b:00.0
  props->pciPath = NULL;
  // Only used to detect NICs with multiple PCI attachments.
  props->guid = 0;
  // Add NCCL_PTR_CUDA if GPU Direct RDMA is supported and regMr can take CUDA pointers.
  props->ptrSupport = NCCL_PTR_HOST;
  // If you regMr has a fast registration cache, set to 1. If set to 0, user buffer registration may be disabled.
  props->regIsGlobal = 0;
  // Speed in *Mbps*. 100000 means 100G
  props->speed = 100000;
  // Port number, used in conjunction with guid
  props->port = 0;
  // Custom latency (used to help tuning if latency is high. If set to 0, use default NCCL values.
  props->latency = 0;
  // Maximum number of comm objects we can create.
  props->maxComms = 1024*1024;
  // Maximum number of receive operations taken by irecv().
  props->maxRecvs = 1;
  // Coupling with NCCL network device-side code.
  props->netDeviceType = 0;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclInternalError;
}
__hidden ncclResult_t pluginListen(int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v8_t** sendDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_v8_t** recvDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) { return ncclInternalError; }
__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) { return ncclInternalError; }

#define PLUGIN_NAME "Plugin"

const ncclNet_v8_t ncclNetPlugin_v8 = {
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
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
};

__hidden ncclResult_t pluginGetProperties_v7(int dev, ncclNetProperties_v7_t* props_v7) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v7->name = props.name;
  props_v7->pciPath = props.pciPath;
  props_v7->guid = props.guid;
  props_v7->ptrSupport = props.ptrSupport;
  props_v7->speed = props.speed;
  props_v7->port = props.port;
  props_v7->maxComms = props.maxComms;
  props_v7->maxRecvs = props.maxRecvs;
  props_v7->netDeviceType = props.netDeviceType;
  props_v7->netDeviceVersion = props.netDeviceVersion;
  return ncclSuccess;
}

__hidden ncclResult_t pluginRegMr_v7(void* collComm, void* data, int size, int type, void** mhandle) {
  return pluginRegMr(collComm, data, size, type, mhandle);
}

const ncclNet_v7_t ncclNetPlugin_v7 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v7,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr_v7,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
};

__hidden ncclResult_t pluginGetProperties_v6(int dev, ncclNetProperties_v6_t* props_v6) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v6->name = props.name;
  props_v6->pciPath = props.pciPath;
  props_v6->guid = props.guid;
  props_v6->ptrSupport = props.ptrSupport;
  props_v6->speed = props.speed;
  props_v6->port = props.port;
  props_v6->maxComms = props.maxComms;
  props_v6->maxRecvs = props.maxRecvs;
  return ncclSuccess;
}

__hidden ncclResult_t pluginConnect_v6(int dev, void* handle, void** sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept_v6(void* listenComm, void** recvComm) { return ncclInternalError; }

const ncclNet_v6_t ncclNetPlugin_v6 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v6,
  .listen = pluginListen,
  .connect = pluginConnect_v6,
  .accept = pluginAccept_v6,
  .regMr = pluginRegMr_v7,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen
};

/* v5 Compat */
const ncclNet_v5_t ncclNetPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties_v6,
  .listen = pluginListen,
  .connect = pluginConnect_v6,
  .accept = pluginAccept_v6,
  .regMr = pluginRegMr_v7,
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
static ncclResult_t pluginGetProperties_v4(int dev, ncclNetProperties_v4_t* props_v4) {
  ncclNetProperties_t props;
  ncclResult_t ret = pluginGetProperties(dev, &props);
  if (ret != ncclSuccess) return ret;
  props_v4->name = props.name;
  props_v4->pciPath = props.pciPath;
  props_v4->guid = props.guid;
  props_v4->ptrSupport = props.ptrSupport;
  props_v4->speed = props.speed;
  props_v4->port = props.port;
  props_v4->maxComms = props.maxComms;
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
    ncclNetDeviceHandle_v7_t* handle = NULL;
    ret = pluginConnect(dev, handle, sendComm, &handle);
  } while (ret == ncclSuccess && *sendComm == NULL);
  return ret;
}
static ncclResult_t pluginAccept_v4(void* listenComm, void** recvComm) {
  ncclResult_t ret;
  do {
    ncclNetDeviceHandle_v7_t* handle = NULL;
    ret = pluginAccept(listenComm, recvComm, &handle);
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
  .regMr = pluginRegMr_v7,
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
  memcpy(handle, &pluginHandle, NCCL_NET_HANDLE_MAXSIZE_V4);
  return ret;
}
static ncclResult_t pluginConnect_v3(int dev, void* handle, void** sendComm) {
  char pluginHandle[NCCL_NET_HANDLE_MAXSIZE];
  memcpy(&pluginHandle, handle, NCCL_NET_HANDLE_MAXSIZE_V4);
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
  .regMr = pluginRegMr_v7,
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
  .regMr = pluginRegMr_v7,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend_v4,
  .irecv = pluginIrecv_v4,
  .flush = pluginFlush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
};
