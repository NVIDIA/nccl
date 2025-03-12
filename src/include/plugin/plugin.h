/*************************************************************************
 * Copyright (c) 2024-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PLUGIN_H_
#define NCCL_PLUGIN_H_

#include "nccl.h"

void* ncclOpenNetPluginLib(const char* name);
void* ncclOpenTunerPluginLib(const char* name);
void* ncclOpenProfilerPluginLib(const char* name);
void* ncclGetNetPluginLib(void);
ncclResult_t ncclClosePluginLib(void* handle);

#endif
