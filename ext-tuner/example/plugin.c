/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tuner.h"

#define __hidden __attribute__ ((visibility("hidden")))

__hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) { return ncclSuccess; }

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int* nChannels) {
  // Update NCCL core generated cost table. Updated table will be evaluated by NCCL to pick the best algo/proto combo
  if (collCostTable[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] != NCCL_ALGO_PROTO_IGNORE) {
    collCostTable[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
  }
  *nChannels = 1;
  return ncclSuccess;
}

__hidden ncclResult_t pluginDestroy(void* context) { return ncclSuccess; }

#define PLUGIN_NAME "Example"

const ncclTuner_v3_t ncclTunerPlugin_v3 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};
