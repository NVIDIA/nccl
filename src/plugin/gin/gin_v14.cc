/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v14_t* ncclGin_v14;

ncclGin_t* getNcclGin_v14(void* lib) {
  ncclGin_v14 = (ncclGin_v14_t*)dlsym(lib, "ncclGinPlugin_v14");
  if (ncclGin_v14) {
    INFO(NCCL_INIT|NCCL_NET, "GIN/Plugin: Loaded gin plugin %s (v14)", ncclGin_v14->name);
    return ncclGin_v14;
  }
  return nullptr;
}
