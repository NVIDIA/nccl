/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclRma_v14_t* ncclRma_v14;

ncclRma_t* getNcclRma_v14(void* lib) {
  ncclRma_v14 = (ncclRma_v14_t*)dlsym(lib, "ncclRmaPlugin_v14");
  if (ncclRma_v14) {
    INFO(NCCL_INIT|NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v14)", ncclRma_v14->name);
    return ncclRma_v14;
  }
  return nullptr;
}

