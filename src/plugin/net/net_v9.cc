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
static ncclNet_v9_t* ncclNet_v9;
static ncclCollNet_v9_t* ncclCollNet_v9;

#define NET_INDEX 0
#define COLLNET_INDEX 1
#define INDEX_NUMS 2
static int refCount[INDEX_NUMS];

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v9_t props_v9;
  NCCLCHECK(ncclNet_v9->getProperties(dev, &props_v9));
  props->name = props_v9.name;
  props->pciPath = props_v9.pciPath;
  props->guid = props_v9.guid;
  props->ptrSupport = props_v9.ptrSupport;
  props->regIsGlobal = props_v9.regIsGlobal;
  props->forceFlush = props_v9.forceFlush;
  props->speed = props_v9.speed;
  props->port = props_v9.port;
  props->latency = props_v9.latency;
  props->maxComms = props_v9.maxComms;
  props->maxRecvs = props_v9.maxRecvs;
  props->netDeviceType = props_v9.netDeviceType;
  props->netDeviceVersion = props_v9.netDeviceVersion;
  props->vProps.ndevs = props_v9.vProps.ndevs;
  for (int i = 0; i < props->vProps.ndevs; i++) {
    props->vProps.devs[i] = props_v9.vProps.devs[i];
  }
  props->maxP2pBytes = props_v9.maxP2pBytes;
  props->maxCollBytes = props_v9.maxCollBytes;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle,
    void* pHandle __attribute__((unused)),
    void** request) {
  return ncclNet_v9->isend(sendComm, data, size, tag, mhandle, request);
}

static ncclResult_t ncclNet_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles,
    void** pHandles __attribute__((unused)),
    void** request) {
  return ncclNet_v9->irecv(recvComm, n, data, sizes, tags, mhandles, request);
}

static ncclResult_t ncclNet_listen(void* ctx __attribute__((unused)),
    int dev, void* handle, void** listenComm) {
  return ncclNet_v9->listen(dev, handle, listenComm);
}

static ncclResult_t ncclNet_connect(void* ctx __attribute__((unused)),
    int dev,
    void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclNet_v9->connect(dev, handle, sendComm, sendDevComm);
}

static ncclResult_t ncclNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclNet_v9->makeVDevice(d, (ncclNetVDeviceProps_v9_t*)props);
}

static ncclResult_t ncclNet_finalize(void* ctx __attribute__((unused))) {
  refCount[NET_INDEX]--;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v9_t props_v9;
  NCCLCHECK(ncclCollNet_v9->getProperties(dev, &props_v9));
  props->name = props_v9.name;
  props->pciPath = props_v9.pciPath;
  props->guid = props_v9.guid;
  props->ptrSupport = props_v9.ptrSupport;
  props->regIsGlobal = props_v9.regIsGlobal;
  props->forceFlush = props_v9.forceFlush;
  props->speed = props_v9.speed;
  props->port = props_v9.port;
  props->latency = props_v9.latency;
  props->maxComms = props_v9.maxComms;
  props->maxRecvs = props_v9.maxRecvs;
  props->netDeviceType = props_v9.netDeviceType;
  props->netDeviceVersion = props_v9.netDeviceVersion;
  props->vProps.ndevs = props_v9.vProps.ndevs;
  for (int i = 0; i < props->vProps.ndevs; i++) {
    props->vProps.devs[i] = props_v9.vProps.devs[i];
  }
  props->maxP2pBytes = props_v9.maxP2pBytes;
  props->maxCollBytes = props_v9.maxCollBytes;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_listen(void* ctx __attribute__((unused)),
    int d, void* handle, void** listenComm) {
  return ncclCollNet_v9->listen(d, handle, listenComm);
}

static ncclResult_t ncclCollNet_iallgather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request) {
  return ncclCollNet_v9->iallgather(collComm, sendData, nRecvParts, (ncclNetSGE_v9_t*)recvParts, bytesPerRank,
                             windowOffset, windowBytes, sendMhandle, request);
}

static ncclResult_t ncclCollNet_ireducescatter(void* collComm, int nSendParts, ncclNetSGE_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request) {
  return ncclCollNet_v9->ireducescatter(collComm, nSendParts, (ncclNetSGE_v9_t*)sendParts, recvData, bytesPerRank,
                                 windowOffset, windowBytes, dataType, redOp, recvMhandle, request);
}

static ncclResult_t ncclCollNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclCollNet_v9->makeVDevice(d, (ncclNetVDeviceProps_v9_t *)props);
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
  // The compat layer preserves the ncclNet_v9 behavior using a refCount to track the number of times the plugin
  // is initialized, and avoid initializing it multiple times.
  if (refCount[NET_INDEX]++) return ncclSuccess;
  NCCLCHECK(ncclNet_v9->init(logfn));
  ncclNet.devices = ncclNet_v9->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_listen;
  ncclNet.connect = ncclNet_connect;
  ncclNet.accept = ncclNet_v9->accept;
  ncclNet.regMr = ncclNet_v9->regMr;
  ncclNet.regMrDmaBuf = ncclNet_v9->regMrDmaBuf;
  ncclNet.deregMr = ncclNet_v9->deregMr;
  ncclNet.isend = ncclNet_isend;
  ncclNet.irecv = ncclNet_irecv;
  ncclNet.iflush = ncclNet_v9->iflush;
  ncclNet.test = ncclNet_v9->test;
  ncclNet.closeSend = ncclNet_v9->closeSend;
  ncclNet.closeRecv = ncclNet_v9->closeRecv;
  ncclNet.closeListen = ncclNet_v9->closeListen;
  ncclNet.getDeviceMr = ncclNet_v9->getDeviceMr;
  ncclNet.irecvConsumed = ncclNet_v9->irecvConsumed;
  ncclNet.makeVDevice = (ncclNet_v9->makeVDevice) ? ncclNet_makeVDevice : nullptr;
  ncclNet.finalize = ncclNet_finalize;
  ncclNet.setNetAttr = nullptr;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v9(void* lib) {
  ncclNet_v9 = (ncclNet_v9_t*)dlsym(lib, "ncclNetPlugin_v9");
  if (ncclNet_v9) {
    ncclNet.name = ncclNet_v9->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v9)", ncclNet_v9->name);
    return &ncclNet;
  }
  return nullptr;
}

static ncclResult_t ncclCollNet_init(void** ctx __attribute__((unused)),
    uint64_t commId __attribute__((unused)),
    ncclDebugLogger_t logfn) {
  // before ncclCollNet_v11 the collnet plugin was initialized only once. With ncclCollNet_v11 this is no longer the case.
  // The compat layer preserves the ncclCollNet_v9 behavior using a refCount to track the number of times the plugin
  // is initialized, and avoid initializing it multiple times.
  if (refCount[COLLNET_INDEX]++) return ncclSuccess;
  NCCLCHECK(ncclCollNet_v9->init(logfn));
  ncclCollNet.devices = ncclCollNet_v9->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_listen;
  ncclCollNet.connect = ncclCollNet_v9->connect;
  ncclCollNet.reduceSupport = ncclCollNet_v9->reduceSupport;
  ncclCollNet.regMr = ncclCollNet_v9->regMr;
  ncclCollNet.regMrDmaBuf = ncclCollNet_v9->regMrDmaBuf;
  ncclCollNet.deregMr = ncclCollNet_v9->deregMr;
  ncclCollNet.iallreduce = ncclCollNet_v9->iallreduce;
  ncclCollNet.iallgather = ncclCollNet_iallgather;
  ncclCollNet.ireducescatter = ncclCollNet_ireducescatter;
  ncclCollNet.iflush = ncclCollNet_v9->iflush;
  ncclCollNet.test = ncclCollNet_v9->test;
  ncclCollNet.closeColl = ncclCollNet_v9->closeColl;
  ncclCollNet.closeListen = ncclCollNet_v9->closeListen;
  ncclCollNet.makeVDevice = (ncclCollNet_v9->makeVDevice) ? ncclCollNet_makeVDevice : nullptr;
  ncclCollNet.finalize = ncclCollNet_finalize;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v9(void* lib) {
  ncclCollNet_v9 = (ncclCollNet_v9_t*)dlsym(lib, "ncclCollNetPlugin_v9");
  if (ncclCollNet_v9) {
    ncclCollNet.name = ncclCollNet_v9->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v9)", ncclCollNet_v9->name);
    return &ncclCollNet;
  }
  return nullptr;
}
