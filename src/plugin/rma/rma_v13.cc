/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "rma/rma_v13.h"
#include "nccl_rma.h"
#include "checks.h"
#include "os.h"
#include <string.h>

static ncclRma_v13_t* ncclRma_v13;
static ncclRma_t ncclRma;

static ncclResult_t ncclRma_init(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction) {
  NCCLCHECK(ncclRma_v13->init(ctx, commId, logFunction));

  // RMA plugin must report GIN proxy type
  ncclNetProperties_t props;
  NCCLCHECK(ncclRma_v13->getProperties(0, &props));
  if (props.netDeviceType != NCCL_NET_DEVICE_GIN_PROXY) {
    WARN("RMA v13 (%s) requires GIN PROXY type, got netDeviceType %d", ncclRma_v13->name, props.netDeviceType);
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t ncclRma_createContext(void* collComm, ncclRmaConfig_v15_t* config, void** rmaCtx) {
  ncclNetDeviceHandle_v11_t* devHandle;
  ncclGinConfig_v13_t config_v13;
  memset(&config_v13, 0, sizeof(config_v13));
  config_v13.nContexts = config->nContexts;
  config_v13.trafficClass = config->trafficClass;
  // ignore config.rankStride
  NCCLCHECK(ncclRma_v13->createContext(collComm, &config_v13, rmaCtx, &devHandle));
  return ncclSuccess;
}

static ncclResult_t ncclRma_regMrSym(void* collComm, void* data, size_t size, int type, uint64_t mrFlags,
                                     void** mhandle) {
  void* unusedGinHandle;
  NCCLCHECK(ncclRma_v13->regMrSym(collComm, data, size, type, mrFlags, mhandle, &unusedGinHandle));
  return ncclSuccess;
}

static ncclResult_t ncclRma_regMrSymDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                           uint64_t mrFlags, void** mhandle) {
  void* unusedGinHandle;
  NCCLCHECK(ncclRma_v13->regMrSymDmaBuf(collComm, data, size, type, offset, fd, mrFlags, mhandle, &unusedGinHandle));
  return ncclSuccess;
}

// Drop isStrongSignal. All signals in v13 are strong signals.
static ncclResult_t ncclRma_iputSignal(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                       uint64_t dstOff, void* dstMhandle, uint32_t rank, uint64_t signalOff,
                                       void* signalMhandle, uint64_t signalValue, uint32_t signalOp,
                                       bool isStrongSignal, uint32_t optFlags, void** request) {
  (void)isStrongSignal;
  (void)optFlags;
  return ncclRma_v13->iputSignal(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, signalOff,
                                 signalMhandle, signalValue, signalOp, request);
}

static ncclResult_t ncclRma_iput(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size,
                                 uint64_t dstOff, void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request) {
  (void)optFlags;
  return ncclRma_v13->iput(rmaCtx, context, srcOff, srcMhandle, size, dstOff, dstMhandle, rank, request);
}

static ncclResult_t ncclRma_iget(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                                 uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags,
                                 void** request) {
  (void)optFlags;
  return ncclRma_v13->iget(rmaCtx, context, remoteOff, remoteMhandle, size, localOff, localMhandle, rank, request);
}

ncclRma_t* getNcclRma_v13(void* lib) {
  ncclRma_v13 = (ncclRma_v13_t*)ncclOsDlsym(lib, "ncclRmaPlugin_v13");
  // Also try the GIN symbol, as the two should have an equal signature.
  if (!ncclRma_v13) {
    ncclRma_v13 = (ncclRma_v13_t*)ncclOsDlsym(lib, "ncclGinPlugin_v13");
  }
  if (ncclRma_v13) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v13)", ncclRma_v13->name);
    ncclRma.name = ncclRma_v13->name;
    ncclRma.init = ncclRma_init;
    ncclRma.devices = ncclRma_v13->devices;
    ncclRma.getProperties = ncclRma_v13->getProperties;
    ncclRma.listen = ncclRma_v13->listen;
    ncclRma.connect = ncclRma_v13->connect;
    ncclRma.createContext = ncclRma_createContext;
    ncclRma.regMrSym = ncclRma_regMrSym;
    ncclRma.regMrSymDmaBuf = ncclRma_regMrSymDmaBuf;
    ncclRma.deregMrSym = ncclRma_v13->deregMrSym;
    ncclRma.destroyContext = ncclRma_v13->destroyContext;
    ncclRma.closeColl = ncclRma_v13->closeColl;
    ncclRma.closeListen = ncclRma_v13->closeListen;
    ncclRma.iput = ncclRma_iput;
    ncclRma.iputSignal = ncclRma_iputSignal;
    ncclRma.iget = ncclRma_iget;
    ncclRma.iflush = ncclRma_v13->iflush;
    ncclRma.test = ncclRma_v13->test;
    ncclRma.rmaProgress = ncclRma_v13->ginProgress;
    ncclRma.queryLastError = ncclRma_v13->queryLastError;
    ncclRma.finalize = ncclRma_v13->finalize;
    return &ncclRma;
  }
  return nullptr;
}
