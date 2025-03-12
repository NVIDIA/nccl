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

static ncclTuner_v2_t* ncclTuner_v2;
static ncclTuner_t ncclTuner;

static int hasNvlsSupport(float** collCostTable) {
  // Requirements for support of different algorithms:
  //
  // - NVLS intra-node: nvlsSupport
  // - NVLS intra+inter-node: collNetSupport
  // - NVLSTree intra-node: always disabled
  // - NVLSTree inter-node: nvlsSupport
  // - Collnet* inter-node: collNetSupport
  //
  // nvlsSupport = 1 if either NVLS or NVLS_TREE entries in the cost table are not -1
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  return (table[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] != NCCL_ALGO_PROTO_IGNORE || table[NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] != NCCL_ALGO_PROTO_IGNORE) ? 1 : 0;
}

static int hasCollNetSupport(float** collCostTable) {
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
  return (table[NCCL_ALGO_COLLNET_CHAIN][NCCL_PROTO_SIMPLE] == NCCL_ALGO_PROTO_IGNORE) ? 0 : 1;
}

static ncclResult_t ncclTuner_getCollInfo(void* context, ncclFunc_t collType, size_t nBytes, int numPipeOps, float** collCostTable, int numAlgo __attribute__((unused)), int numProto __attribute__((unused)), int regBuff __attribute__((unused)), int* nChannels) {
  int algorithm = NCCL_ALGO_UNDEF;
  int protocol = NCCL_PROTO_UNDEF;
  int nvlsSupport = hasNvlsSupport(collCostTable);
  int collNetSupport = hasCollNetSupport(collCostTable);
  NCCLCHECK(ncclTuner_v2->getCollInfo(context, collType, nBytes, collNetSupport, nvlsSupport, numPipeOps, &algorithm, &protocol, nChannels));
  // set time to 0 below to make sure this algorithm/protocol is selected later on
  if (algorithm >= 0 && algorithm < NCCL_NUM_ALGORITHMS && protocol >= 0 && protocol < NCCL_NUM_PROTOCOLS) {
    float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS])collCostTable;
    if (table[algorithm][protocol] != NCCL_ALGO_PROTO_IGNORE) table[algorithm][protocol] = 0.0;
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuner_init(size_t nRanks, size_t nNodes, ncclDebugLogger_t logfn, void** context) {
  NCCLCHECK(ncclTuner_v2->init(nRanks, nNodes, logfn, context));
  ncclTuner.getCollInfo = ncclTuner_getCollInfo;
  ncclTuner.destroy = ncclTuner_v2->destroy;
  return ncclSuccess;
}

ncclTuner_t* getNcclTuner_v2(void* lib) {
  ncclTuner_v2 = (ncclTuner_v2_t*)dlsym(lib, "ncclTunerPlugin_v2");
  if (ncclTuner_v2) {
    ncclTuner.name = ncclTuner_v2->name;
    ncclTuner.init = ncclTuner_init;
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using tuner plugin %s", ncclTuner_v2->name);
    return &ncclTuner;
  }
  INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Failed to find ncclTunerPlugin_v2 symbol, using internal tuner instead.");
  return NULL;
}
