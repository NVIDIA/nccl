/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_ENV_H_
#define NCCL_INT_ENV_H_

#include "nccl_env.h"

// Initialize Env Plugin
ncclResult_t ncclEnvPluginInit(void);
// Finalize Env Plugin
void ncclEnvPluginFinalize(void);
// Env plugin get function for NCCL params, called in ncclGetEnv()
const char* ncclEnvPluginGetEnv(const char* name);

bool ncclEnvPluginInitialized(void);

ncclResult_t ncclInitEnv(void);

#endif
