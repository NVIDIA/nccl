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
static ncclNet_v10_t* ncclNet_v10;
static ncclCollNet_v10_t* ncclCollNet_v10;

#define NET_INDEX 0
#define COLLNET_INDEX 1
#define INDEX_NUMS 2
static int refCount[INDEX_NUMS];

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v10_t props_v10;
  NCCLCHECK(ncclNet_v10->getProperties(dev, &props_v10));
  props->name = props_v10.name;
  props->pciPath = props_v10.pciPath;
  props->guid = props_v10.guid;
  props->ptrSupport = props_v10.ptrSupport;
  props->regIsGlobal = props_v10.regIsGlobal;
  props->forceFlush = props_v10.forceFlush;
  props->speed = props_v10.speed;
  props->port = props_v10.port;
  props->latency = props_v10.latency;
  props->maxComms = props_v10.maxComms;
  props->maxRecvs = props_v10.maxRecvs;
  props->netDeviceType = props_v10.netDeviceType;
  props->netDeviceVersion = props_v10.netDeviceVersion;
  props->vProps.ndevs = props_v10.vProps.ndevs;
  for (int i = 0; i < props->vProps.ndevs; i++) {
    props->vProps.devs[i] = props_v10.vProps.devs[i];
  }
  props->maxP2pBytes = props_v10.maxP2pBytes;
  props->maxCollBytes = props_v10.maxCollBytes;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t ncclNet_listen(void* ctx __attribute__((unused)),
    int dev, void* handle, void** listenComm) {
  return ncclNet_v10->listen(dev, handle, listenComm);
}

static ncclResult_t ncclNet_connect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclNet_v10->connect(dev, (ncclNetCommConfig_v10_t *)ctx, handle, sendComm, sendDevComm);
}

static ncclResult_t ncclNet_makeVDevice(int* d, ncclNetVDeviceProps_v11_t* props) {
  return ncclNet_v10->makeVDevice(d, (ncclNetVDeviceProps_v10_t *)props);
}

static ncclResult_t ncclNet_finalize(void* ctx) {
  refCount[NET_INDEX]--;
  free(ctx);
  return ncclSuccess;
}

static ncclResult_t ncclNet_init(void** ctx, uint64_t commId __attribute__((unused)),
    ncclNetCommConfig_t* config, ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  // since ncclNet_v11, the ncclNetCommConfig_t has been moved from connect to init. Since the config is per comm,
  // this allows the config to be passed only once, instead of multiple times (once per connect). To preserve the
  // ncclNet_v10 behavior, in the compat layer, we store the config in the context pointer and pass it to the connect
  // function.
  ncclNetCommConfig_v10_t* config_v10 = nullptr;
  NCCLCHECK(ncclCalloc(&config_v10, 1));
  config_v10->trafficClass = config->trafficClass;
  *ctx = config_v10;
  // before ncclNet_v11 the net plugin was initialized only once. With ncclNet_v11 this is no longer the case.
  // The compat layer preserves the ncclNet_v10 behavior using a refCount to track the number of times the plugin
  // is initialized, and avoid initializing it multiple times.
  if (refCount[NET_INDEX]++) return ncclSuccess;
  NCCLCHECK(ncclNet_v10->init(logfn, proffn));
  ncclNet.devices = ncclNet_v10->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_listen;
  ncclNet.connect = ncclNet_connect;
  ncclNet.accept = ncclNet_v10->accept;
  ncclNet.regMr = ncclNet_v10->regMr;
  ncclNet.regMrDmaBuf = ncclNet_v10->regMrDmaBuf;
  ncclNet.deregMr = ncclNet_v10->deregMr;
  ncclNet.isend = ncclNet_v10->isend;
  ncclNet.irecv = ncclNet_v10->irecv;
  ncclNet.iflush = ncclNet_v10->iflush;
  ncclNet.test = ncclNet_v10->test;
  ncclNet.closeSend = ncclNet_v10->closeSend;
  ncclNet.closeRecv = ncclNet_v10->closeRecv;
  ncclNet.closeListen = ncclNet_v10->closeListen;
  ncclNet.getDeviceMr = ncclNet_v10->getDeviceMr;
  ncclNet.irecvConsumed = ncclNet_v10->irecvConsumed;
  ncclNet.makeVDevice = (ncclNet_v10->makeVDevice) ? ncclNet_makeVDevice : nullptr;
  ncclNet.finalize = ncclNet_finalize;
  ncclNet.setNetAttr = nullptr;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v10(void* lib) {
  ncclNet_v10 = (ncclNet_v10_t*)dlsym(lib, "ncclNetPlugin_v10");
  if (ncclNet_v10) {
    ncclNet.name = ncclNet_v10->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v10)", ncclNet_v10->name);
    return &ncclNet;
  }
  return nullptr;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v10_t props_v10;
  NCCLCHECK(ncclCollNet_v10->getProperties(dev, &props_v10));
  props->name = props_v10.name;
  props->pciPath = props_v10.pciPath;
  props->guid = props_v10.guid;
  props->ptrSupport = props_v10.ptrSupport;
  props->regIsGlobal = props_v10.regIsGlobal;
  props->forceFlush = props_v10.forceFlush;
  props->speed = props_v10.speed;
  props->port = props_v10.port;
  props->latency = props_v10.latency;
  props->maxComms = props_v10.maxComms;
  props->maxRecvs = props_v10.maxRecvs;
  props->netDeviceType = props_v10.netDeviceType;
  props->netDeviceVersion = props_v10.netDeviceVersion;
  props->vProps.ndevs = props_v10.vProps.ndevs;
  for (int i = 0; i < props->vProps.ndevs; i++) {
    props->vProps.devs[i] = props_v10.vProps.devs[i];
  }
  props->maxP2pBytes = props_v10.maxP2pBytes;
  props->maxCollBytes = props_v10.maxCollBytes;
  props->maxMultiRequestSize = 1;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_listen(void* ctx __attribute__((unused)),
    int dev, void* handle , void** listenComm) {
  return ncclCollNet_v10->listen(dev, handle, listenComm);
}

static ncclResult_t ncclCollNet_iallgather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request) {
  return ncclCollNet_v10->iallgather(collComm, sendData, nRecvParts, (ncclNetSGE_v10_t*)recvParts, bytesPerRank,
                             windowOffset, windowBytes, sendMhandle, request);
}

static ncclResult_t ncclCollNet_ireducescatter(void* collComm, int nSendParts, ncclNetSGE_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request) {
  return ncclCollNet_v10->ireducescatter(collComm, nSendParts, (ncclNetSGE_v10_t*)sendParts, recvData, bytesPerRank,
                                 windowOffset, windowBytes, dataType, redOp, recvMhandle, request);
}

static ncclResult_t ncclCollNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclCollNet_v10->makeVDevice(d, (ncclNetVDeviceProps_v10_t *)props);
}

static ncclResult_t ncclCollNet_finalize(void* ctx __attribute__((unused))) {
  refCount[COLLNET_INDEX]--;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_init(void** ctx __attribute__((unused)),
    uint64_t commId __attribute__((unused)),
    ncclDebugLogger_t logfn) {
  // before ncclCollNet_v11 the collnet plugin was initialized only once. With ncclCollNet_v11 this is no longer the case.
  // The compat layer preserves the ncclCollNet_v10 behavior using a refCount to track the number of times the plugin
  // is initialized, and avoid initializing it multiple times.
  if (refCount[COLLNET_INDEX]++) return ncclSuccess;
  NCCLCHECK(ncclCollNet_v10->init(logfn));
  ncclCollNet.devices = ncclCollNet_v10->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_listen;
  ncclCollNet.connect = ncclCollNet_v10->connect;
  ncclCollNet.reduceSupport = ncclCollNet_v10->reduceSupport;
  ncclCollNet.regMr = ncclCollNet_v10->regMr;
  ncclCollNet.regMrDmaBuf = ncclCollNet_v10->regMrDmaBuf;
  ncclCollNet.deregMr = ncclCollNet_v10->deregMr;
  ncclCollNet.iallreduce = ncclCollNet_v10->iallreduce;
  ncclCollNet.iallgather = ncclCollNet_iallgather;
  ncclCollNet.ireducescatter = ncclCollNet_ireducescatter;
  ncclCollNet.iflush = ncclCollNet_v10->iflush;
  ncclCollNet.test = ncclCollNet_v10->test;
  ncclCollNet.closeColl = ncclCollNet_v10->closeColl;
  ncclCollNet.closeListen = ncclCollNet_v10->closeListen;
  ncclCollNet.makeVDevice = (ncclCollNet_v10->makeVDevice) ? ncclCollNet_makeVDevice : nullptr;
  ncclCollNet.finalize = ncclCollNet_finalize;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v10(void* lib) {
  ncclCollNet_v10 = (ncclCollNet_v10_t*)dlsym(lib, "ncclCollNetPlugin_v10");
  if (ncclCollNet_v10) {
    ncclCollNet.name = ncclCollNet_v10->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v10)", ncclCollNet_v10->name);
    return &ncclCollNet;
  }
  return nullptr;
}
