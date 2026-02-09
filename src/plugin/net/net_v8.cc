/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "proxy.h"
#include "checks.h"
#include <dlfcn.h>

static ncclNet_t ncclNet;
static ncclCollNet_t ncclCollNet;
static ncclNet_v8_t* ncclNet_v8;
static ncclCollNet_v8_t* ncclCollNet_v8;

#define NET_INDEX 0
#define COLLNET_INDEX 1
#define INDEX_NUMS 2
static int refCount[INDEX_NUMS];

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v8_t p8;
  ncclResult_t ans = ncclNet_v8->getProperties(dev, &p8);
  if (ans != ncclSuccess) return ans;
  props->name = p8.name;
  props->pciPath = p8.pciPath;
  props->guid = p8.guid;
  props->ptrSupport = p8.ptrSupport;
  props->regIsGlobal = p8.regIsGlobal;
  props->forceFlush = 0;
  props->speed = p8.speed;
  props->port = p8.port;
  props->maxComms = p8.maxComms;
  props->maxRecvs = p8.maxRecvs;
  props->latency = p8.latency;
  props->netDeviceType = p8.netDeviceType;
  props->netDeviceVersion = p8.netDeviceVersion;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t ncclNet_listen(void* ctx __attribute__((unused)),
    int dev, void* handle, void** listenComm) {
  return ncclNet_v8->listen(dev, handle, listenComm);
}

static ncclResult_t ncclNet_connect(void* ctx __attribute__((unused)),
    int dev,
    void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclNet_v8->connect(dev, handle, sendComm, sendDevComm);
}

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle,
    void* pHandle __attribute__((unused)),
    void** request) {
  int sizeInt;
  if (size > MAX_NET_SIZE) return ncclInternalError;
  sizeInt = (int)size;
  ncclResult_t ans = ncclNet_v8->isend(sendComm, data, sizeInt, tag, mhandle, request);
  return ans;
}

static ncclResult_t ncclNet_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
    void** pHandles __attribute__((unused)),
    void** request) {
  int sizesInt[NCCL_PROXY_MAX_SUBS];
  //reset to nullptr if optional receive completion is set
  if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = nullptr;
  for (int i=0; i<n; i++) {
    if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
    sizesInt[i] = (int) sizes[i];
  }
  ncclResult_t ans = ncclNet_v8->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
  return ans;
}

static ncclResult_t ncclNet_finalize(void* ctx __attribute__((unused))) {
  refCount[NET_INDEX]--;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v8_t p8;
  ncclResult_t ans = ncclCollNet_v8->getProperties(dev, &p8);
  if (ans != ncclSuccess) return ans;
  props->name = p8.name;
  props->pciPath = p8.pciPath;
  props->guid = p8.guid;
  props->ptrSupport = p8.ptrSupport;
  props->regIsGlobal = p8.regIsGlobal;
  props->forceFlush = 0;
  props->speed = p8.speed;
  props->port = p8.port;
  props->maxComms = p8.maxComms;
  props->maxRecvs = p8.maxRecvs;
  props->latency = p8.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_listen(void* ctx __attribute__((unused)),
    int dev, void* handle, void** listenComm) {
  return ncclCollNet_v8->listen(dev, handle, listenComm);
}

static ncclResult_t ncclCollNet_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
  int countInt;
  if (count > MAX_NET_SIZE) return ncclInternalError;
  countInt = (int)count;
  ncclResult_t ans = ncclCollNet_v8->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                 sendMhandle, recvMhandle, request);
  return ans;
}

static ncclResult_t ncclCollNet_iallgather (void* collComm, void* sendData, int nRecvParts, ncclNetSGE_t* recvParts,
                           size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                           void* sendMhandle, void** request) {
  ncclNetSGE_v8_t recvPartsInt;
  if (nRecvParts > 1) return ncclInternalError;
  if (recvParts->size > MAX_COLLNET_SIZE) return ncclInternalError;
  recvPartsInt.mhandle = recvParts->mhandle;
  recvPartsInt.address = recvParts->address;
  recvPartsInt.size = (int)recvParts->size;
  ncclResult_t ans = ncclCollNet_v8->iallgather(collComm, sendData, nRecvParts, &recvPartsInt,
                  bytesPerRank, windowOffset, windowBytes,
                  sendMhandle, request);
  return ans;
}

static ncclResult_t ncclCollNet_ireducescatter(void* collComm, int nSendParts, ncclNetSGE_t* sendParts, void* recvData,
                               size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                               ncclDataType_t dataType, ncclRedOp_t redOp,
                               void* recvMhandle, void** request) {
  ncclNetSGE_v8_t sendPartsInt;
  if (nSendParts > 1) return ncclInternalError;
  if (sendParts->size > MAX_COLLNET_SIZE) return ncclInternalError;
  sendPartsInt.mhandle = sendParts->mhandle;
  sendPartsInt.address = sendParts->address;
  sendPartsInt.size = (int)sendParts->size;
  ncclResult_t ans = ncclCollNet_v8->ireducescatter(collComm, nSendParts, &sendPartsInt,
                  recvData, bytesPerRank, windowOffset, windowBytes,
                  dataType, redOp,
                  recvMhandle, request);
  return ans;
}

static ncclResult_t ncclCollNet_finalize(void* ctx __attribute__((unused))) {
  refCount[COLLNET_INDEX]--;
  return ncclSuccess;
}

static ncclResult_t ncclNet_init(void** ctx __attribute__((unused)),
    uint64_t commId __attribute__((unused)),
    ncclNetCommConfig_t* config __attribute__((unused)),
    ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  // before ncclNet_v11 the net plugin was initialized only once. With ncclNet_v11 this is no longer the case.
  // The compat layer preserves the ncclNet_v8 behavior using a refCount to track the number of times the plugin
  // is initialized, and avoid initializing it multiple times.
  if (refCount[NET_INDEX]++) return ncclSuccess;
  NCCLCHECK(ncclNet_v8->init(logfn));
  ncclNet.devices = ncclNet_v8->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_listen;
  ncclNet.connect = ncclNet_connect;
  ncclNet.accept =  ncclNet_v8->accept;
  ncclNet.regMr = ncclNet_v8->regMr;
  ncclNet.regMrDmaBuf = ncclNet_v8->regMrDmaBuf;
  ncclNet.deregMr = ncclNet_v8->deregMr;
  ncclNet.isend = ncclNet_isend;
  ncclNet.irecv = ncclNet_irecv;
  ncclNet.iflush = ncclNet_v8->iflush;
  ncclNet.test = ncclNet_v8->test;
  ncclNet.closeSend = ncclNet_v8->closeSend;
  ncclNet.closeRecv = ncclNet_v8->closeRecv;
  ncclNet.closeListen = ncclNet_v8->closeListen;
  ncclNet.getDeviceMr = ncclNet_v8->getDeviceMr;
  ncclNet.irecvConsumed = ncclNet_v8->irecvConsumed;
  ncclNet.makeVDevice   = NULL;
  ncclNet.finalize = ncclNet_finalize;
  ncclNet.setNetAttr = nullptr;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v8(void* lib) {
  ncclNet_v8 = (ncclNet_v8_t*)dlsym(lib, "ncclNetPlugin_v8");
  if (ncclNet_v8) {
    ncclNet.name = ncclNet_v8->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v8)", ncclNet_v8->name);
    return &ncclNet;
  }
  return nullptr;
}

static ncclResult_t ncclCollNet_init(void** ctx __attribute__((unused)),
    uint64_t commId __attribute__((unused)),
    ncclDebugLogger_t logfn) {
  // before ncclCollNet_v11 the collnet plugin was initialized only once. With ncclCollNet_v11 this is no longer the case.
  // The compat layer preserves the ncclCollNet_v8 behavior using a refCount to track the number of times the plugin
  // is initialized, and avoid initializing it multiple times.
  if (refCount[COLLNET_INDEX]++) return ncclSuccess;
  NCCLCHECK(ncclCollNet_v8->init(logfn));
  ncclCollNet.devices = ncclCollNet_v8->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_listen;
  ncclCollNet.connect = ncclCollNet_v8->connect;
  ncclCollNet.reduceSupport = ncclCollNet_v8->reduceSupport;
  ncclCollNet.regMr = ncclCollNet_v8->regMr;
  ncclCollNet.regMrDmaBuf = ncclCollNet_v8->regMrDmaBuf;
  ncclCollNet.deregMr = ncclCollNet_v8->deregMr;
  ncclCollNet.iallreduce = ncclCollNet_iallreduce;
  ncclCollNet.iallgather = ncclCollNet_iallgather;
  ncclCollNet.ireducescatter = ncclCollNet_ireducescatter;
  ncclCollNet.iflush = ncclCollNet_v8->iflush;
  ncclCollNet.test = ncclCollNet_v8->test;
  ncclCollNet.closeColl = ncclCollNet_v8->closeColl;
  ncclCollNet.closeListen = ncclCollNet_v8->closeListen;
  ncclCollNet.makeVDevice = nullptr;
  ncclCollNet.finalize = ncclCollNet_finalize;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v8(void* lib) {
  ncclCollNet_v8 = (ncclCollNet_v8_t*)dlsym(lib, "ncclCollNetPlugin_v8");
  if (ncclCollNet_v8) {
    ncclCollNet.name = ncclCollNet_v8->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v8)", ncclCollNet_v8->name);
    return &ncclCollNet;
  }
  return nullptr;
}
