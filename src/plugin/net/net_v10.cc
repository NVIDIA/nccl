/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_device.h"
#include "proxy.h"

static ncclNet_t ncclNet;
static ncclCollNet_t ncclCollNet;
static ncclNet_v10_t* ncclNet_v10;
static ncclCollNet_v10_t* ncclCollNet_v10;

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v10_t p10;
  ncclResult_t ans = ncclNet_v10->getProperties(dev, &p10);
  if (ans != ncclSuccess) return ans;
  props->name = p10.name;
  props->pciPath = p10.pciPath;
  props->guid = p10.guid;
  props->ptrSupport = p10.ptrSupport;
  props->regIsGlobal = p10.regIsGlobal;
  props->forceFlush = p10.forceFlush;
  props->speed = p10.speed;
  props->port = p10.port;
  props->maxComms = p10.maxComms;
  props->maxRecvs = p10.maxRecvs;
  props->latency = p10.latency;
  props->netDeviceType = p10.netDeviceType;
  props->netDeviceVersion = p10.netDeviceVersion;
  props->vProps.ndevs = p10.vProps.ndevs;
  memcpy(props->vProps.devs, p10.vProps.devs, sizeof(p10.vProps.devs));
  props->maxP2pBytes = p10.maxP2pBytes;
  props->maxCollBytes = p10.maxCollBytes;
  props->fabricId = 0; // all devs are on the same rail if v10
  return ncclSuccess;
}

static ncclResult_t ncclNet_getNetPath(uint64_t fabricId0, uint64_t fabricId1, ncclNetPath_t* path) {
  if (!path) return ncclInvalidArgument;
  path->loc = (fabricId0 == fabricId1) ? NET_LOC_DCL0 : NET_LOC_DISC;
  return ncclSuccess;
}


static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v10_t p10;
  ncclResult_t ans = ncclCollNet_v10->getProperties(dev, &p10);
  if (ans != ncclSuccess) return ans;
  props->name = p10.name;
  props->pciPath = p10.pciPath;
  props->guid = p10.guid;
  props->ptrSupport = p10.ptrSupport;
  props->regIsGlobal = p10.regIsGlobal;
  props->forceFlush = p10.forceFlush;
  props->speed = p10.speed;
  props->port = p10.port;
  props->maxComms = p10.maxComms;
  props->maxRecvs = p10.maxRecvs;
  props->latency = p10.latency;
  props->netDeviceType = p10.netDeviceType;
  props->netDeviceVersion = p10.netDeviceVersion;
  props->vProps.ndevs = p10.vProps.ndevs;
  memcpy(props->vProps.devs, p10.vProps.devs, sizeof(p10.vProps.devs));
  props->maxP2pBytes = p10.maxP2pBytes;
  props->maxCollBytes = p10.maxCollBytes;
  props->fabricId = 0; // all devs are on the same rail if v10
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_getNetPath(uint64_t fabricId0, uint64_t fabricId1, ncclNetPath_t* path) {
  if (!path) return ncclInvalidArgument;
  path->loc = (fabricId0 == fabricId1) ? NET_LOC_DCL0 : NET_LOC_DISC;
  return ncclSuccess;
}

static ncclResult_t ncclNet_connect(int dev, ncclNetCommConfig_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclNet_v10->connect(dev, (ncclNetCommConfig_v10_t*)config, handle, sendComm, sendDevComm);
}

static ncclResult_t ncclNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclNet_v10->makeVDevice(d, (ncclNetVDeviceProps_v10_t*)props);
}

static ncclResult_t ncclNet_init(ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  NCCLCHECK(ncclNet_v10->init(logfn, proffn));
  ncclNet.devices = ncclNet_v10->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_v10->listen;
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
  ncclNet.makeVDevice = ncclNet_v10->makeVDevice ? ncclNet_makeVDevice : nullptr;
  ncclNet.getNetPath = ncclNet_getNetPath;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v10(void* lib) {
  ncclNet_v10 = (ncclNet_v10_t*)dlsym(lib, "ncclNetPlugin_v9");
  if (ncclNet_v10) {
    ncclNet.name = ncclNet_v10->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT | NCCL_NET, "NET/Plugin: Loaded net plugin %s (v10)", ncclNet_v10->name);
    return &ncclNet;
  }
  INFO(NCCL_INIT | NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v9 symbol.");
  return nullptr;
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

static ncclResult_t ncclCollNet_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v10->init(logfn));
  ncclCollNet.devices = ncclCollNet_v10->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_v10->listen;
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
  ncclCollNet.getNetPath = ncclCollNet_getNetPath;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v10(void* lib) {
  ncclCollNet_v10 = (ncclCollNet_v10_t*)dlsym(lib, "ncclCollNetPlugin_v10");
  if (ncclCollNet_v10) {
    ncclCollNet.name = ncclCollNet_v10->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT | NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v10)", ncclCollNet_v10->name);
    return &ncclCollNet;
  }
  INFO(NCCL_INIT | NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.");
  return nullptr;
}
