/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"

static ncclTuner_v4_t* ncclTuner_v4;

ncclTuner_t* getNcclTuner_v4(void* lib) {
  ncclTuner_v4 = (ncclTuner_v4_t*)dlsym(lib, "ncclTunerPlugin_v4");
  if (ncclTuner_v4) {
    INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Using tuner plugin %s", ncclTuner_v4->name);
    return ncclTuner_v4;
  }
  INFO(NCCL_ENV|NCCL_TUNING, "TUNER/Plugin: Failed to find ncclTunerPlugin_v4 symbol.");
  return NULL;
}
