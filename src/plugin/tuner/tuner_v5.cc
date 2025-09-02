/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"

static ncclTuner_v5_t* ncclTuner_v5;

ncclTuner_t* getNcclTuner_v5(void* lib) {
  ncclTuner_v5 = (ncclTuner_v5_t*)dlsym(lib, "ncclTunerPlugin_v5");
  if (ncclTuner_v5) {
    INFO(NCCL_INIT|NCCL_TUNING, "TUNER/Plugin: Using %s (v5)", ncclTuner_v5->name);
    return ncclTuner_v5;
  }
  return NULL;
}
