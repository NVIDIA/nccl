/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_LOG_H_
#define NCCL_LOG_H_

#include "common.h"
#include "err.h"

// Mirrors ncclDebugLogRecord_v1_t / ncclLogSink_v1_t in nccl.h, so an out-of-tree plugin can be built
// without the full NCCL headers. Keep in sync with nccl.h.
typedef struct {
  int level;
  unsigned long subSys;
  const char* file;
  const char* func;
  int line;
  ncclResult_t code;
  const char* format;
  const char* message;
  const char* hostname;
  int pid;
  int tid;
  int cudaDev;
} ncclDebugLogRecord_v1_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** context);
  ncclResult_t (*onRecord)(void* context, const ncclDebugLogRecord_v1_t* record);
  ncclResult_t (*finalize)(void* context);
} ncclLogSink_v1_t;

// Plugin symbol name
#define NCCL_LOG_PLUGIN_SYMBOL ncclLogPlugin_v1

#endif
