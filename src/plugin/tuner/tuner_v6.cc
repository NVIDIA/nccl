/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"
#include "checks.h"

static ncclTuner_v6_t* ncclTuner_v6;
static ncclTuner_t ncclTuner;

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                                   ncclDebugLogger_t logfn, ncclNvlDomainInfo_v5_t* nvlDomainInfo,
                                   ncclTunerConstants_t* constants) {
  ncclTunerConstants_v6_t constants_v6;
  ncclTunerConstants_v7_to_v5(&constants_v6, constants);
  NCCLCHECK(ncclTuner_v6->init(context, commId, nRanks, nNodes, logfn, nvlDomainInfo, &constants_v6));
  ncclTunerConstants_v5_to_v7(constants, &constants_v6);
  ncclTuner.getCollInfo = ncclTuner_v6->getCollInfo;
  ncclTuner.getChunkSize = ncclTuner_v6->getChunkSize;
  ncclTuner.finalize = ncclTuner_v6->finalize;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v6(void* lib) {
  ncclTuner_v6 = (ncclTuner_v6_t*)dlsym(lib, "ncclTunerPlugin_v6");
  if (ncclTuner_v6) {
    ncclTuner.init = ncclTuner_init;
    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v6)", ncclTuner_v6->name);
    return &ncclTuner;
  }
  return NULL;
}
