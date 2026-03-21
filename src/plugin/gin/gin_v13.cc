/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v13_t* ncclGin_v13;

ncclGin_t* getNcclGin_v13(void* lib) {
  ncclGin_v13 = (ncclGin_v13_t*)dlsym(lib, "ncclGinPlugin_v13");
  if (ncclGin_v13) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v13)", ncclGin_v13->name);
    return ncclGin_v13;
  }
  return nullptr;
}
