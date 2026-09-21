/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"

static ncclTuner_v7_t* ncclTuner_v7;

ncclTuner_t* getNcclTuner_v7(void* lib) {
  ncclTuner_v7 = (ncclTuner_v7_t*)dlsym(lib, "ncclTunerPlugin_v7");
  if (ncclTuner_v7) {
    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v7)", ncclTuner_v7->name);
    return ncclTuner_v7;
  }
  return NULL;
}
