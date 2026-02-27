/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PLUGIN_H_
#define NCCL_PLUGIN_H_

#include "nccl.h"

enum ncclPluginType {
  ncclPluginTypeNet,
  ncclPluginTypeGin,
  ncclPluginTypeTuner,
  ncclPluginTypeProfiler,
  ncclPluginTypeEnv,
};

void* ncclOpenNetPluginLib(const char* name);
void* ncclOpenGinPluginLib(const char* name);
void* ncclOpenTunerPluginLib(const char* name);
void* ncclOpenProfilerPluginLib(const char* name);
void* ncclOpenEnvPluginLib(const char* name);
void* ncclGetNetPluginLib(enum ncclPluginType type);
ncclResult_t ncclClosePluginLib(void* handle, enum ncclPluginType type);

extern char* ncclPluginLibPaths[];

#endif
