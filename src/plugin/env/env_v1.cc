/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>
#include "os.h"
#include "debug.h"
#include "nccl_env.h"
#include "checks.h"

static ncclEnv_t ncclEnv;
static ncclEnv_v1_t* ncclEnv_v1;

static ncclResult_t ncclEnv_init(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix,
                                 ncclDebugLogger_t logFunction) {
  NCCLCHECK(ncclEnv_v1->init(ncclMajor, ncclMinor, ncclPatch, suffix));
  return ncclSuccess;
}

ncclEnv_t* getNcclEnv_v1(void* lib) {
  ncclEnv_v1 = (ncclEnv_v1_t*)ncclOsDlsym(lib, "ncclEnvPlugin_v1");
  if (ncclEnv_v1) {
    ncclEnv.init = ncclEnv_init;
    ncclEnv.finalize = ncclEnv_v1->finalize;
    ncclEnv.getEnv = ncclEnv_v1->getEnv;
    INFO(NCCL_INIT | NCCL_ENV, "ENV/Plugin: Using %s (v1)", ncclEnv_v1->name);
    return &ncclEnv;
  }
  return nullptr;
}
