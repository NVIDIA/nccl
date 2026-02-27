/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_net.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclGin_v12_t* ncclGin_v12;

ncclGin_t* getNcclGin_v12(void* lib) {
  ncclGin_v12 = (ncclGin_v12_t*)dlsym(lib, "ncclGinPlugin_v12");
  if (ncclGin_v12) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded gin plugin %s (v12)", ncclGin_v12->name);
    return ncclGin_v12;
  }
  return nullptr;
}
