/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PARAM_H_
#define NCCL_PARAM_H_

#include "platform.h"
#include <stdint.h>

const char *userHomeDir();
void setEnvFile(const char *fileName);
void initEnv();
const char *ncclGetEnv(const char *name);

void ncclLoadParam(char const *env, int64_t deftVal, int64_t uninitialized, int64_t *cache);

/*
 * NCCL_PARAM macro - platform-portable parameter loading
 * Uses platform-specific atomics for thread-safe caching
 */
#if NCCL_PLATFORM_WINDOWS
/* On Windows, use Interlocked functions for atomic operations */
#define NCCL_PARAM(name, env, deftVal)                                                                             \
  int64_t ncclParam##name()                                                                                        \
  {                                                                                                                \
    constexpr int64_t uninitialized = INT64_MIN;                                                                   \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value.");                   \
    static int64_t cache = uninitialized;                                                                          \
    int64_t cachedVal = _InterlockedCompareExchange64((volatile long long *)&cache, uninitialized, uninitialized); \
    if (cachedVal == uninitialized)                                                                                \
    {                                                                                                              \
      ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache);                                                  \
    }                                                                                                              \
    return cache;                                                                                                  \
  }
#else
/* On Linux/POSIX, use GCC atomics */
#define NCCL_PARAM(name, env, deftVal)                                                           \
  int64_t ncclParam##name()                                                                      \
  {                                                                                              \
    constexpr int64_t uninitialized = INT64_MIN;                                                 \
    static_assert(deftVal != uninitialized, "default value cannot be the uninitialized value."); \
    static int64_t cache = uninitialized;                                                        \
    if (__builtin_expect(__atomic_load_n(&cache, __ATOMIC_RELAXED) == uninitialized, false))     \
    {                                                                                            \
      ncclLoadParam("NCCL_" env, deftVal, uninitialized, &cache);                                \
    }                                                                                            \
    return cache;                                                                                \
  }
#endif

#endif
