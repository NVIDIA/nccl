/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "checks.h"
#include "nccl_tuner.h"

static ncclTuner_v3_t* ncclTuner_v3;
static ncclTuner_t ncclTuner;

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps, float** collCostTable, int numAlgo, int numProto, int regBuff __attribute__((unused)), int* nChannels) {
  NCCLCHECK(ncclTuner_v3->getCollInfo(context, collType, nBytes, numPipeOps, collCostTable, numAlgo, numProto,  nChannels));
  return ncclSuccess;
}

static ncclResult_t ncclTuner_finalize(void* ctx) {
  return ncclTuner_v3->destroy(ctx);
}

static ncclResult_t ncclTuner_init(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logfn,
                                   ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_t* /*constants*/) {
  NCCLCHECK(ncclTuner_v3->init(nRanks, nNodes, logfn, context));
  ncclTuner.getCollInfo = ncclTuner_getCollInfo;
  ncclTuner.finalize = ncclTuner_finalize;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v3(void* lib) {
  ncclTuner_v3 = (ncclTuner_v3_t*)dlsym(lib, "ncclTunerPlugin_v3");
  if (ncclTuner_v3) {
    ncclTuner.name = ncclTuner_v3->name;
    ncclTuner.init = ncclTuner_init;
    INFO(NCCL_INIT|NCCL_TUNING, "TUNER/Plugin: Using %s (v3)", ncclTuner_v3->name);
    return &ncclTuner;
  }
  return NULL;
}
