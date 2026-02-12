/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include <stdint.h>
#include "compiler.h"

const char* userHomeDir();
void setEnvFile(const char* fileName);
void initEnv();
const char *ncclGetEnv(const char *name);

int64_t ncclLoadParam(char const* env, int64_t deftVal, int64_t uninitialized, int64_t* cache, int8_t* noCache);

#define NCCL_PARAM(name, env, deftVal) \
  int64_t ncclParam##name() { \
    constexpr int64_t uninitialized = INT64_MIN; \
    static int8_t noCache = /*uninitialized*/ -1; \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized; \
    if (COMPILER_EXPECT(COMPILER_ATOMIC_LOAD(&cache, std::memory_order_relaxed) == uninitialized, false)) { \
      return ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache, &noCache); \
    } \
    return cache; \
  }

#endif
