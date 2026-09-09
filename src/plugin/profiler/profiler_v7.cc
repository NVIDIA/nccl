/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "plugin/profiler/profiler_v7.h"
#include "checks.h"
#include "os.h"

static ncclProfiler_v7_t* ncclProfiler_v7;

ncclProfiler_t* getNcclProfiler_v7(void* lib) {
  ncclProfiler_v7 = (ncclProfiler_v7_t*)ncclOsDlsym(lib, "ncclProfiler_v7");
  if (ncclProfiler_v7) {
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v7)", ncclProfiler_v7->name);
    return ncclProfiler_v7;
  }
  return NULL;
}
