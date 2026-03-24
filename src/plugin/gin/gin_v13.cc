/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
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

static ncclGin_v13_t* ncclRma_v13;

ncclGin_t* getNcclRma_v13(void* lib) {
  ncclRma_v13 = (ncclGin_v13_t*)dlsym(lib, "ncclRmaPlugin_v13");
  if (ncclRma_v13) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded rma plugin %s (v13)", ncclRma_v13->name);
    return ncclRma_v13;
  }
  return nullptr;
}
