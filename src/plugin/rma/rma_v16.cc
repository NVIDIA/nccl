/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclRma_v16_t* ncclRma_v16;

ncclRma_t* getNcclRma_v16(void* lib) {
  ncclRma_v16 = (ncclRma_v16_t*)dlsym(lib, "ncclRmaPlugin_v16");
  if (ncclRma_v16) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v16)", ncclRma_v16->name);
    return ncclRma_v16;
  }
  return nullptr;
}
