/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl_net.h"
#include "net_device.h"
#include "proxy.h"

static ncclNet_v10_t* ncclNet_v10;
static ncclCollNet_v10_t* ncclCollNet_v10;

ncclNet_t* getNcclNet_v10(void* lib) {
  ncclNet_v10 = (ncclNet_v10_t*)dlsym(lib, "ncclNetPlugin_v10");
  if (ncclNet_v10) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v10)", ncclNet_v10->name);
    return ncclNet_v10;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v10 symbol.");
  return nullptr;
}

ncclCollNet_t* getNcclCollNet_v10(void* lib) {
  ncclCollNet_v10 = (ncclCollNet_v10_t*)dlsym(lib, "ncclCollNetPlugin_v10");
  if (ncclCollNet_v10) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v10)", ncclNet_v10->name);
    return ncclCollNet_v10;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v10 symbol.");
  return nullptr;
}
