/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclRma_v15_t* ncclRma_v15;
static ncclRma_t ncclRma;

static ncclResult_t ncclRma_v15_getRmaProperties(void* collComm, ncclRmaProperties_t* rmaProps) {
  (void)collComm;
  rmaProps->flushesAllPutsOnAnySignal = false;
  return ncclSuccess;
}

ncclRma_t* getNcclRma_v15(void* lib) {
  ncclRma_v15 = (ncclRma_v15_t*)dlsym(lib, "ncclRmaPlugin_v15");
  if (ncclRma_v15) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v15)", ncclRma_v15->name);
    ncclRma.name = ncclRma_v15->name;
    ncclRma.init = ncclRma_v15->init;
    ncclRma.devices = ncclRma_v15->devices;
    ncclRma.getRmaProperties = ncclRma_v15_getRmaProperties;
    ncclRma.getProperties = ncclRma_v15->getProperties;
    ncclRma.listen = ncclRma_v15->listen;
    ncclRma.connect = ncclRma_v15->connect;
    ncclRma.createContext = ncclRma_v15->createContext;
    ncclRma.regMrSym = ncclRma_v15->regMrSym;
    ncclRma.regMrSymDmaBuf = ncclRma_v15->regMrSymDmaBuf;
    ncclRma.deregMrSym = ncclRma_v15->deregMrSym;
    ncclRma.destroyContext = ncclRma_v15->destroyContext;
    ncclRma.closeColl = ncclRma_v15->closeColl;
    ncclRma.closeListen = ncclRma_v15->closeListen;
    ncclRma.iput = ncclRma_v15->iput;
    ncclRma.iputSignal = ncclRma_v15->iputSignal;
    ncclRma.iget = ncclRma_v15->iget;
    ncclRma.iflush = ncclRma_v15->iflush;
    ncclRma.test = ncclRma_v15->test;
    ncclRma.rmaProgress = ncclRma_v15->rmaProgress;
    ncclRma.queryLastError = ncclRma_v15->queryLastError;
    ncclRma.finalize = ncclRma_v15->finalize;
    return &ncclRma;
  }
  return nullptr;
}
