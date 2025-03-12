/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_device.h"
#include "proxy.h"
#include "checks.h"

static ncclNet_t ncclNet;
static ncclCollNet_t ncclCollNet;
static ncclNet_v6_t* ncclNet_v6;
static ncclCollNet_v6_t* ncclCollNet_v6;

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_connect(int dev, ncclNetCommConfig_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v6->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v6->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* pHandle, void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v6->isend(sendComm, data, sizeInt, tag, mhandle, request);
  return ans;
}

static ncclResult_t ncclNet_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** pHandles, void** request) {
  int sizesInt[NCCL_PROXY_MAX_SUBS];
  //reset to nullptr if optional receive completion is set
  if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = nullptr;
  for (int i=0; i<n; i++) {
    if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
    sizesInt[i] = (int) sizes[i];
  }
  ncclResult_t ans = ncclNet_v6->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
  return ans;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclCollNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
     ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v6->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

static ncclResult_t ncclNet_init(ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  NCCLCHECK(ncclNet_v6->init(logfn));
  ncclNet.devices = ncclNet_v6->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_v6->listen;
  ncclNet.connect = ncclNet_connect;
  ncclNet.accept =  ncclNet_accept;
  ncclNet.regMr = ncclNet_regMr;
  ncclNet.regMrDmaBuf = ncclNet_v6->regMrDmaBuf;
  ncclNet.deregMr = ncclNet_v6->deregMr;
  ncclNet.isend = ncclNet_isend;
  ncclNet.irecv = ncclNet_irecv;
  ncclNet.iflush = ncclNet_v6->iflush;
  ncclNet.test = ncclNet_v6->test;
  ncclNet.closeSend = ncclNet_v6->closeSend;
  ncclNet.closeRecv = ncclNet_v6->closeRecv;
  ncclNet.closeListen = ncclNet_v6->closeListen;
  ncclNet.getDeviceMr = NULL;
  ncclNet.irecvConsumed = NULL;
  ncclNet.makeVDevice  = NULL;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v6(void* lib) {
  ncclNet_v6 = (ncclNet_v6_t*)dlsym(lib, "ncclNetPlugin_v6");
  if (ncclNet_v6) {
    ncclNet.name = ncclNet_v6->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v6)", ncclNet_v6->name);
    return &ncclNet;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v6 symbol.");
  return nullptr;
}

static ncclResult_t ncclCollNet_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v6->init(logfn));
  ncclCollNet.devices = ncclCollNet_v6->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_v6->listen;
  ncclCollNet.connect = ncclCollNet_v6->connect;
  ncclCollNet.reduceSupport = ncclCollNet_v6->reduceSupport;
  ncclCollNet.regMr = ncclCollNet_regMr;
  ncclCollNet.regMrDmaBuf = ncclCollNet_v6->regMrDmaBuf;
  ncclCollNet.deregMr = ncclCollNet_v6->deregMr;
  ncclCollNet.iallreduce = ncclCollNet_iallreduce;
  ncclCollNet.iallgather = nullptr;
  ncclCollNet.ireducescatter = nullptr;
  ncclCollNet.iflush = ncclCollNet_v6->iflush;
  ncclCollNet.test = ncclCollNet_v6->test;
  ncclCollNet.closeColl = ncclCollNet_v6->closeColl;
  ncclCollNet.closeListen = ncclCollNet_v6->closeListen;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v6(void* lib) {
  ncclCollNet_v6 = (ncclCollNet_v6_t*)dlsym(lib, "ncclCollNetPlugin_v6");
  if (ncclCollNet_v6) {
    ncclCollNet.name = ncclCollNet_v6->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v6)", ncclCollNet_v6->name);
    return &ncclCollNet;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.");
  return nullptr;
}
