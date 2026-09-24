/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v15_t* ncclGin_v15;

ncclGin_t* getNcclGin_v15(void* lib) {
  ncclGin_v15 = (ncclGin_v15_t*)dlsym(lib, "ncclGinPlugin_v15");
  if (ncclGin_v15) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Loaded gin plugin %s (v15)", ncclGin_v15->name);
    return ncclGin_v15;
  }
  return nullptr;
}
