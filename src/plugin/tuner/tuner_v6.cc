/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <dlfcn.h>
#include "debug.h"
#include "nccl_tuner.h"

static ncclTuner_v6_t* ncclTuner_v6;

ncclTuner_t* getNcclTuner_v6(void* lib) {
  ncclTuner_v6 = (ncclTuner_v6_t*)dlsym(lib, "ncclTunerPlugin_v6");
  if (ncclTuner_v6) {
    INFO(NCCL_INIT | NCCL_TUNING, "TUNER/Plugin: Using %s (v6)", ncclTuner_v6->name);
    return ncclTuner_v6;
  }
  return NULL;
}
