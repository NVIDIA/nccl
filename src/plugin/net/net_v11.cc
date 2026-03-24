/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_net.h"
#include "proxy.h"
#include "os.h"

static ncclNet_t ncclNet;
static ncclCollNet_t ncclCollNet;
static ncclNet_v11_t* ncclNet_v11;
static ncclCollNet_v11_t* ncclCollNet_v11;

static ncclResult_t ncclNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v11_t props_v11;
  NCCLCHECK(ncclNet_v11->getProperties(dev, &props_v11));
  props->name = props_v11.name;
  props->pciPath = props_v11.pciPath;
  props->guid = props_v11.guid;
  props->ptrSupport = props_v11.ptrSupport;
  props->regIsGlobal = props_v11.regIsGlobal;
  props->forceFlush = props_v11.forceFlush;
  props->speed = props_v11.speed;
  props->port = props_v11.port;
  props->latency = props_v11.latency;
  props->maxComms = props_v11.maxComms;
  props->maxRecvs = props_v11.maxRecvs;
  props->netDeviceType = props_v11.netDeviceType;
  props->netDeviceVersion = props_v11.netDeviceVersion;
  props->vProps.ndevs = props_v11.vProps.ndevs;
  for (int i = 0; i < props_v11.vProps.ndevs; i++)
    props->vProps.devs[i] = props_v11.vProps.devs[i];
  props->maxP2pBytes = props_v11.maxP2pBytes;
  props->maxCollBytes = props_v11.maxCollBytes;
  props->maxMultiRequestSize = props_v11.maxMultiRequestSize;
  // Undefined to be ignore in NCCL core
  props->railId = NCCL_NET_ID_UNDEF;
  props->planeId = NCCL_NET_ID_UNDEF;
  return ncclSuccess;
}

static ncclResult_t ncclNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  // Safe cast: devs[] is at the end of the struct and NCCL limits ndevs to NCCL_NET_MAX_DEVS_PER_NIC_V11 for v11 plugins.
  return ncclNet_v11->makeVDevice(d, (ncclNetVDeviceProps_v11_t*)props);
}

static ncclResult_t ncclNet_setNetAttr(void* ctx, ncclNetAttr_t* netAttr) {
  return ncclNet_v11->setNetAttr(ctx, (ncclNetAttr_v11_t*)netAttr);
}

static ncclResult_t ncclNet_init(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  // Safe cast: ncclNetCommConfig_v11_t and ncclNetCommConfig_v12_t are binary identical.
  NCCLCHECK(ncclNet_v11->init(ctx, commId, (ncclNetCommConfig_v11_t*)config, logFunction, profFunction));
  ncclNet.devices = ncclNet_v11->devices;
  ncclNet.getProperties = ncclNet_getProperties;
  ncclNet.listen = ncclNet_v11->listen;
  ncclNet.connect = ncclNet_v11->connect;
  ncclNet.accept = ncclNet_v11->accept;
  ncclNet.regMr = ncclNet_v11->regMr;
  ncclNet.regMrDmaBuf = ncclNet_v11->regMrDmaBuf;
  ncclNet.deregMr = ncclNet_v11->deregMr;
  ncclNet.isend = ncclNet_v11->isend;
  ncclNet.irecv = ncclNet_v11->irecv;
  ncclNet.iflush = ncclNet_v11->iflush;
  ncclNet.test = ncclNet_v11->test;
  ncclNet.closeSend = ncclNet_v11->closeSend;
  ncclNet.closeRecv = ncclNet_v11->closeRecv;
  ncclNet.closeListen = ncclNet_v11->closeListen;
  ncclNet.getDeviceMr = ncclNet_v11->getDeviceMr;
  ncclNet.irecvConsumed = ncclNet_v11->irecvConsumed;
  ncclNet.makeVDevice = (ncclNet_v11->makeVDevice) ? ncclNet_makeVDevice : nullptr;
  ncclNet.finalize = ncclNet_v11->finalize;
  ncclNet.setNetAttr = (ncclNet_v11->setNetAttr) ? ncclNet_setNetAttr : nullptr;
  return ncclSuccess;
}

ncclNet_t* getNcclNet_v11(void* lib) {
  ncclNet_v11 = (ncclNet_v11_t*)ncclOsDlsym(lib, "ncclNetPlugin_v11");
  if (ncclNet_v11) {
    ncclNet.name = ncclNet_v11->name;
    ncclNet.init = ncclNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v11)", ncclNet_v11->name);
    return &ncclNet;
  }
  return nullptr;
}

static ncclResult_t ncclCollNet_getProperties(int dev, ncclNetProperties_t* props) {
  ncclNetProperties_v11_t props_v11;
  NCCLCHECK(ncclCollNet_v11->getProperties(dev, &props_v11));
  props->name = props_v11.name;
  props->pciPath = props_v11.pciPath;
  props->guid = props_v11.guid;
  props->ptrSupport = props_v11.ptrSupport;
  props->regIsGlobal = props_v11.regIsGlobal;
  props->forceFlush = props_v11.forceFlush;
  props->speed = props_v11.speed;
  props->port = props_v11.port;
  props->latency = props_v11.latency;
  props->maxComms = props_v11.maxComms;
  props->maxRecvs = props_v11.maxRecvs;
  props->netDeviceType = props_v11.netDeviceType;
  props->netDeviceVersion = props_v11.netDeviceVersion;
  props->vProps.ndevs = props_v11.vProps.ndevs;
  for (int i = 0; i < props_v11.vProps.ndevs; i++)
    props->vProps.devs[i] = props_v11.vProps.devs[i];
  props->maxP2pBytes = props_v11.maxP2pBytes;
  props->maxCollBytes = props_v11.maxCollBytes;
  props->maxMultiRequestSize = props_v11.maxMultiRequestSize;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_iallgather(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_t* recvParts,
                                           size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                           void* sendMhandle, void** request) {
  // Safe cast: ncclNetSGE_v11_t and ncclNetSGE_v12_t are binary identical.
  return ncclCollNet_v11->iallgather(collComm, sendData, nRecvParts, (ncclNetSGE_v11_t*)recvParts,
                                     bytesPerRank, windowOffset, windowBytes, sendMhandle, request);
}

static ncclResult_t ncclCollNet_ireducescatter(void* collComm, int nSendParts, ncclNetSGE_t* sendParts, void* recvData,
                                               size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                               ncclDataType_t dataType, ncclRedOp_t redOp,
                                               void* recvMhandle, void** request) {
  // Safe cast: ncclNetSGE_v11_t and ncclNetSGE_v12_t are binary identical.
  return ncclCollNet_v11->ireducescatter(collComm, nSendParts, (ncclNetSGE_v11_t*)sendParts, recvData,
                                         bytesPerRank, windowOffset, windowBytes, dataType, redOp,
                                         recvMhandle, request);
}

static ncclResult_t ncclCollNet_makeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  // Safe cast: devs[] is at the end of the struct and NCCL limits ndevs to NCCL_NET_MAX_DEVS_PER_NIC_V11 for v11 plugins.
  return ncclCollNet_v11->makeVDevice(d, (ncclNetVDeviceProps_v11_t*)props);
}

static ncclResult_t ncclCollNet_init(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  NCCLCHECK(ncclCollNet_v11->init(ctx, commId, logFunction));
  ncclCollNet.devices = ncclCollNet_v11->devices;
  ncclCollNet.getProperties = ncclCollNet_getProperties;
  ncclCollNet.listen = ncclCollNet_v11->listen;
  ncclCollNet.connect = ncclCollNet_v11->connect;
  ncclCollNet.reduceSupport = ncclCollNet_v11->reduceSupport;
  ncclCollNet.regMr = ncclCollNet_v11->regMr;
  ncclCollNet.regMrDmaBuf = ncclCollNet_v11->regMrDmaBuf;
  ncclCollNet.deregMr = ncclCollNet_v11->deregMr;
  ncclCollNet.iallreduce = ncclCollNet_v11->iallreduce;
  ncclCollNet.iallgather = ncclCollNet_iallgather;
  ncclCollNet.ireducescatter = ncclCollNet_ireducescatter;
  ncclCollNet.iflush = ncclCollNet_v11->iflush;
  ncclCollNet.test = ncclCollNet_v11->test;
  ncclCollNet.closeColl = ncclCollNet_v11->closeColl;
  ncclCollNet.closeListen = ncclCollNet_v11->closeListen;
  ncclCollNet.makeVDevice = (ncclCollNet_v11->makeVDevice) ? ncclCollNet_makeVDevice : nullptr;
  ncclCollNet.finalize = ncclCollNet_v11->finalize;
  return ncclSuccess;
}

ncclCollNet_t* getNcclCollNet_v11(void* lib) {
  ncclCollNet_v11 = (ncclCollNet_v11_t*)ncclOsDlsym(lib, "ncclCollNetPlugin_v11");
  if (ncclCollNet_v11) {
    ncclCollNet.name = ncclCollNet_v11->name;
    ncclCollNet.init = ncclCollNet_init;
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v11)", ncclCollNet_v11->name);
    return &ncclCollNet;
  }
  return nullptr;
}
