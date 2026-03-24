/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Parameter definitions for the NcclParam system itself.

#include "param/param.h"
#include "debug.h"

#include <unordered_set>
#include <cstring>
#include "debug.h"

DEFINE_NCCL_PARAM(ncclParamDumpAllFlag, bool, NCCL_PARAM_DUMP_ALL, false,
                  NCCL_PARAM_FLAG_NONE, NcclParamParserDefault,
                  "Print all parameters including private ones");

using NcclStringSet = std::unordered_set<std::string>;
DEFINE_NCCL_PARAM(ncclParamNoCacheSet, NcclStringSet, NCCL_NO_CACHE, {},
                  NCCL_PARAM_FLAG_CACHED, NcclParamListOf<NcclStringSet>(','),
                  "Comma-separated list of param keys to disable caching (or ALL)");

extern "C" NCCL_PARAM_COMPILER_EXPORT_SYMBOL bool ncclParamIsCacheDisabled(const char* key) {
  // Short-circuit for NCCL_NO_CACHE itself to prevent circular dependency
  if (std::strcmp(key, "NCCL_NO_CACHE") == 0) return false;

  NcclStringSet set = ncclParamNoCacheSet();

  static bool noCacheAll = set.count("ALL") > 0;
  bool ret = noCacheAll || set.count(key) > 0;
  if (ret) INFO(NCCL_ENV, "PARAM: Disabling caching for environment variable %s.", key);

  return ret;
}

// Exported helper for NcclParam<T>::load_value() so plugins can resolve
// a single symbol instead of requiring ncclInitEnv + ncclEnvPluginGetEnv
// to be exported.
#include "env.h"
extern "C" NCCL_PARAM_COMPILER_EXPORT_SYMBOL const char* ncclParamEnvPluginGet(const char* key) {
  ncclInitEnv();
  return ncclEnvPluginGetEnv(key);
}

extern "C" NCCL_PARAM_COMPILER_EXPORT_SYMBOL void* ncclParamRegistryInstance() {
  static NcclParamRegistry::RegistryState registry_state;
  return &registry_state;
}
