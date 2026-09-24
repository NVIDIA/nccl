/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "nccl_rma.h"
#include "proxy.h"
#include <dlfcn.h>

static ncclRma_v17_t* ncclRma_v17;

ncclRma_t* getNcclRma_v17(void* lib) {
  ncclRma_v17 = (ncclRma_v17_t*)dlsym(lib, "ncclRmaPlugin_v17");
  if (ncclRma_v17) {
    INFO(NCCL_INIT | NCCL_NET, "RMA/Plugin: Loaded rma plugin %s (v17)", ncclRma_v17->name);
    return ncclRma_v17;
  }
  return nullptr;
}
