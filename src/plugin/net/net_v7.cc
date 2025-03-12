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
static ncclNet_v7_t* ncclNet_v7;
static ncclCollNet_v7_t* ncclCollNet_v7;

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v7_t p7;
  ncclResult_t ans = ncclNet_v7->getProperties(dev, &p7);
  if (ans != ncclSuccess) return ans;
  props->name = p7.name;
  props->pciPath = p7.pciPath;
  props->guid = p7.guid;
  props->ptrSupport = p7.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p7.speed;
  props->port = p7.port;
  props->maxComms = p7.maxComms;
  props->maxRecvs = p7.maxRecvs;
  props->latency = p7.latency;
  props->netDeviceType = p7.netDeviceType;
  props->netDeviceVersion = p7.netDeviceVersion;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_connect(int dev, ncclNetCommConfig_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclNet_v7->connect(dev, handle, sendComm, sendDevComm);
}

static ncclResult_t ncclNet_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v7->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* pHandle, void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v7->isend(sendComm, data, sizeInt, tag, mhandle, request);
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
  ncclResult_t ans = ncclNet_v7->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
  return ans;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v7_t p7;
  ncclResult_t ans = ncclCollNet_v7->getProperties(dev, &p7);
  if (ans != ncclSuccess) return ans;
  props->name = p7.name;
  props->pciPath = p7.pciPath;
  props->guid = p7.guid;
  props->ptrSupport = p7.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p7.speed;
  props->port = p7.port;
  props->maxComms = p7.maxComms;
  props->maxRecvs = p7.maxRecvs;
  props->latency = p7.latency;
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
  return ncclCollNet_v7->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v7->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

static ncclResult_t ncclNet_init(ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  NCCLCHECK(ncclNet_v7->init(logfn));
  ncclNet.devices = ncclNet_v7->devices;
  ncclNet.getProperties = ncclNet_getProperties; // ncclNet_v5->getProperties;
  ncclNet.listen = ncclNet_v7->listen;
  ncclNet.connect = ncclNet_connect;
  ncclNet.accept =  ncclNet_v7->accept;
  ncclNet.regMr = ncclNet_regMr;
  ncclNet.regMrDmaBuf = ncclNet_v7->regMrDmaBuf;
  ncclNet.deregMr = ncclNet_v7->deregMr;
  ncclNet.isend = ncclNet_isend;
  ncclNet.irecv = ncclNet_irecv;
  ncclNet.iflush = ncclNet_v7->iflush;
  ncclNet.test = ncclNet_v7->test;
  ncclNet.closeSend = ncclNet_v7->closeSend;
  ncclNet.closeRecv = ncclNet_v7->closeRecv;
  ncclNet.closeListen = ncclNet_v7->closeListen;
  ncclNet.getDeviceMr = ncclNet_v7->getDeviceMr;
  ncclNet.irecvConsumed = ncclNet_v7->irecvConsumed;
  ncclNet.makeVDevice  = NULL;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v7(void* lib) {
  ncclNet_v7 = (ncclNet_v7_t*)dlsym(lib, "ncclNetPlugin_v7");
  if (ncclNet_v7) {
    ncclNet.name = ncclNet_v7->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v7)", ncclNet_v7->name);
    return &ncclNet;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v7 symbol.");
  return nullptr;
}

static ncclResult_t ncclCollNet_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v7->init(logfn));
  ncclCollNet.devices = ncclCollNet_v7->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_v7->listen;
  ncclCollNet.connect = ncclCollNet_v7->connect;
  ncclCollNet.reduceSupport = ncclCollNet_v7->reduceSupport;
  ncclCollNet.regMr = ncclCollNet_regMr;
  ncclCollNet.regMrDmaBuf = ncclCollNet_v7->regMrDmaBuf;
  ncclCollNet.deregMr = ncclCollNet_v7->deregMr;
  ncclCollNet.iallreduce = ncclCollNet_iallreduce;
  ncclCollNet.iallgather = nullptr;
  ncclCollNet.ireducescatter = nullptr;
  ncclCollNet.iflush = ncclCollNet_v7->iflush;
  ncclCollNet.test = ncclCollNet_v7->test;
  ncclCollNet.closeColl = ncclCollNet_v7->closeColl;
  ncclCollNet.closeListen = ncclCollNet_v7->closeListen;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v7(void* lib) {
  ncclCollNet_v7 = (ncclCollNet_v7_t*)dlsym(lib, "ncclCollNetPlugin_v7");
  if (ncclCollNet_v7) {
    ncclCollNet.name = ncclCollNet_v7->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v7)", ncclCollNet_v7->name);
    return &ncclCollNet;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v7 symbol.");
  return nullptr;
}
