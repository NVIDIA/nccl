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

static ncclEnv_v2_t* ncclEnv_v2;

ncclEnv_t* getNcclEnv_v2(void* lib) {
  ncclEnv_v2 = (ncclEnv_v2_t*)ncclOsDlsym(lib, "ncclEnvPlugin_v2");
  if (ncclEnv_v2) {
    INFO(NCCL_INIT | NCCL_ENV, "ENV/Plugin: Using %s (v2)", ncclEnv_v2->name);
    return (ncclEnv_t*)ncclEnv_v2;
  }
  return nullptr;
}
