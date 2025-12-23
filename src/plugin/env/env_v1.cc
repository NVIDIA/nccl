/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>
#include <dlfcn.h>
#include "debug.h"
#include "nccl_env.h"

static ncclEnv_v1_t* ncclEnv_v1;

ncclEnv_t* getNcclEnv_v1(void* lib) {
  ncclEnv_v1 = (ncclEnv_v1_t*)dlsym(lib, "ncclEnvPlugin_v1");
  if (ncclEnv_v1) {
    INFO(NCCL_INIT|NCCL_ENV, "ENV/Plugin: Using %s (v1)", ncclEnv_v1->name);
    return ncclEnv_v1;
  }
  return nullptr;
}

static ncclResult_t ncclEnvInit(uint8_t ncclMajor, uint8_t ncclMinor, uint8_t ncclPatch, const char* suffix) {
  return ncclSuccess;
}

static ncclResult_t ncclEnvFinalize(void) {
  return ncclSuccess;
}

static const char* ncclEnvGetEnv(const char* name) {
  return std::getenv(name);
}

ncclEnv_v1_t ncclIntEnv_v1 = {
  "ncclEnvDefault",
  ncclEnvInit,
  ncclEnvFinalize,
  ncclEnvGetEnv,
};
