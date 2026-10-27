/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_gin.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v16_t* ncclGin_v16;

ncclGin_t* getNcclGin_v16(void* lib) {
  ncclGin_v16 = (ncclGin_v16_t*)dlsym(lib, "ncclGinPlugin_v16");
  if (ncclGin_v16) {
    INFO(NCCL_INIT | NCCL_NET, "GIN/Plugin: Loaded gin plugin %s (v16)", ncclGin_v16->name);
    return ncclGin_v16;
  }
  return nullptr;
}
