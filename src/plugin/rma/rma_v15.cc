/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclRma_v15_t* ncclRma_v15;

ncclRma_t* getNcclRma_v15(void* lib) {
  ncclRma_v15 = (ncclRma_v15_t*)dlsym(lib, "ncclRmaPlugin_v15");
  if (ncclRma_v15) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v15)", ncclRma_v15->name);
    return ncclRma_v15;
  }
  return nullptr;
}
