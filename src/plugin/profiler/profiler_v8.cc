/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "nccl_profiler.h"
#include "plugin/profiler/profiler_v8.h"
#include "checks.h"
#include "os.h"

ncclProfiler_t* getNcclProfiler_v8(void* lib, void** target) {
  ncclProfiler_v8_t* profiler = (ncclProfiler_v8_t*)ncclOsDlsym(lib, "ncclProfiler_v8");
  *target = profiler;
  if (profiler) {
    INFO(NCCL_INIT, "PROFILER/Plugin: Loaded %s (v8)", profiler->name);
    return profiler;
  }
  return NULL;
}
