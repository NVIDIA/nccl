/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "proxy.h"
#include <dlfcn.h>
#include <string.h>

static ncclRma_v16_t* ncclRma_v16;
static ncclRma_t ncclRma;

static ncclResult_t ncclRma_v16_connect(void* ctx, void* handles[], int nranks, int rank, void* listenComm,
                                        void** collComm, volatile uint32_t* abortFlag) {
  (void)abortFlag;
  return ncclRma_v16->connect(ctx, handles, nranks, rank, listenComm, collComm);
}

static ncclResult_t ncclRma_v16_createContext(void* collComm, ncclRmaConfig_v17_t* config, void** rmaCtx) {
  ncclRmaConfig_v16_t config_v16;
  memset(&config_v16, 0, sizeof(config_v16));
  config_v16.nContexts = config->nContexts;
  config_v16.trafficClass = config->trafficClass;
  config_v16.rankStride = 1;
  return ncclRma_v16->createContext(collComm, &config_v16, rmaCtx);
}

ncclRma_t* getNcclRma_v16(void* lib) {
  ncclRma_v16 = (ncclRma_v16_t*)dlsym(lib, "ncclRmaPlugin_v16");
  if (ncclRma_v16) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v16)", ncclRma_v16->name);
    ncclRma.name = ncclRma_v16->name;
    ncclRma.init = ncclRma_v16->init;
    ncclRma.devices = ncclRma_v16->devices;
    ncclRma.getRmaProperties = ncclRma_v16->getRmaProperties;
    ncclRma.getProperties = ncclRma_v16->getProperties;
    ncclRma.listen = ncclRma_v16->listen;
    ncclRma.connect = ncclRma_v16_connect;
    ncclRma.createContext = ncclRma_v16_createContext;
    ncclRma.regMrSym = ncclRma_v16->regMrSym;
    ncclRma.regMrSymDmaBuf = ncclRma_v16->regMrSymDmaBuf;
    ncclRma.deregMrSym = ncclRma_v16->deregMrSym;
    ncclRma.destroyContext = ncclRma_v16->destroyContext;
    ncclRma.closeColl = ncclRma_v16->closeColl;
    ncclRma.closeListen = ncclRma_v16->closeListen;
    ncclRma.iput = ncclRma_v16->iput;
    ncclRma.iputSignal = ncclRma_v16->iputSignal;
    ncclRma.iget = ncclRma_v16->iget;
    ncclRma.iflush = ncclRma_v16->iflush;
    ncclRma.test = ncclRma_v16->test;
    ncclRma.rmaProgress = ncclRma_v16->rmaProgress;
    ncclRma.queryLastError = ncclRma_v16->queryLastError;
    ncclRma.finalize = ncclRma_v16->finalize;
    return &ncclRma;
  }
  return nullptr;
}
