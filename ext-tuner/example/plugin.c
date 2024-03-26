/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "tuner.h"

#define __hidden __attribute__ ((visibility("hidden")))

__hidden ncclResult_t pluginInit(size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, void **context) { return ncclSuccess; }

__hidden ncclResult_t pluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int collNetSupport, int nvlsSupport, int numPipeOps,
                              int *algorithm, int *protocol, int* nChannels) { *algorithm = NCCL_ALGO_RING; *protocol = NCCL_PROTO_SIMPLE; return ncclSuccess; }

__hidden ncclResult_t pluginDestroy(void* context) { return ncclSuccess; }

#define PLUGIN_NAME "Example"

const ncclTuner_v2_t ncclTunerPlugin_v2 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .getCollInfo = pluginGetCollInfo,
  .destroy = pluginDestroy
};
