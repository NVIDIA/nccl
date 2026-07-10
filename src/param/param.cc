/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Parameter definitions for the ncclParam system itself.

#include "param/param.h"
#include "param/parsers.h"
#include "debug.h"

#include <unordered_set>

DEFINE_NCCL_PARAM(ncclParamDumpAllFlag, bool, NCCL_PARAM_DUMP_ALL, false, NCCL_PARAM_FLAG_NONE, NCCL_PARAM_DEFAULT,
                  "Print all parameters including private ones");

using ncclStringSet = std::unordered_set<std::string>;
DEFINE_NCCL_PARAM(ncclParamNoCacheStr, const char*, NCCL_NO_CACHE, nullptr, NCCL_PARAM_FLAG_CACHED, NCCL_PARAM_DEFAULT,
                  "Comma-separated list of param keys to disable caching (or ALL)");

extern "C" bool ncclParamIsCacheDisabled(const char* key) {
  // Short-circuit for NCCL_NO_CACHE itself to prevent circular dependency
  if (std::strcmp(key, "NCCL_NO_CACHE") == 0) return false;

  static std::once_flag initFlag;
  static ncclStringSet set;
  static bool noCacheAll = false;

  std::call_once(initFlag, []() {
    auto parser = ncclParamListOf<ncclStringSet>(',');
    if (parser.resolve(ncclParamNoCacheStr(), set) == ncclSuccess) {
      noCacheAll = set.count("ALL") > 0;
    }
  });

  bool ret = noCacheAll || set.count(key) > 0;
  if (ret) INFO(NCCL_ENV, "PARAM: Disabling caching for environment variable %s.", key);
  return ret;
}

// Exported helper for ncclParam<T>::loadValue() so plugins can resolve
// a single symbol instead of requiring ncclInitEnv + ncclEnvPluginGetEnv
// to be exported.
#include "env.h"
extern "C" const char* ncclParamEnvPluginGet(const char* key, bool env_init) {
  if (env_init) {
    // regular parameters will init env plugins before reading env
    ncclInitEnv();
    return ncclEnvPluginGetEnv(key);
  } else {
    // special parameters that do not attempt to initialize env plugins
    return ncclEnvPluginInitialized() ? ncclEnvPluginGetEnv(key) : std::getenv(key);
  }
}
