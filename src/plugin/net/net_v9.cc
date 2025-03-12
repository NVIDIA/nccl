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
static ncclNet_v9_t* ncclNet_v9;
static ncclCollNet_v9_t* ncclCollNet_v9;

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  return ncclNet_v9->getProperties(dev, (ncclNetProperties_v9_t *)props);
}

static ncclResult_t ncclNet_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* pHandle, void** request) {
  return ncclNet_v9->isend(sendComm, data, size, tag, mhandle, request);
}

static ncclResult_t ncclNet_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** pHandles, void** request) {
  return ncclNet_v9->irecv(recvComm, n, data, sizes, tags, mhandles, request);
}

static ncclResult_t ncclNet_connect(int dev, ncclNetCommConfig_t* config, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclNet_v9->connect(dev, handle, sendComm, sendDevComm);
}

static ncclResult_t ncclNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclNet_v9->makeVDevice(d, (ncclNetVDeviceProps_v9_t*)props);
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  return ncclCollNet_v9->getProperties(dev, (ncclNetProperties_v9_t *)props);
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

static ncclResult_t ncclNet_init(ncclDebugLogger_t logfn, ncclProfilerCallback_t proffn) {
  NCCLCHECK(ncclNet_v9->init(logfn));
  ncclNet.devices = ncclNet_v9->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_v9->listen;
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
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v9 symbol.");
  return nullptr;
}

static ncclResult_t ncclCollNet_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v9->init(logfn));
  ncclCollNet.devices = ncclCollNet_v9->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_v9->listen;
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
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.");
  return nullptr;
}
