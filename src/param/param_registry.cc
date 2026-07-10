/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "param/param_registry.h"
#include "debug.h"

extern "C" void* ncclParamRegistryInstance() {
  static ncclParamRegistry::registryState state;
  return &state;
}

ncclResult_t ncclParamRegistry::add(std::string key, ncclParamInfo_t info, ncclParamInterface* param) {
  auto& reg = state();
  std::lock_guard<std::mutex> lock(reg.mtx);
  if (reg.map.find(key) != reg.map.end()) {
    WARN("PARAM: Duplicate registration for key \"%s\", ignoring.", key.c_str());
    return ncclInternalError;
  }
  reg.map[key] = {info, param};
  return ncclSuccess;
}

ncclParamRegistry::mapEntry* ncclParamRegistry::find(std::string key) {
  auto& reg = state();
  std::lock_guard<std::mutex> lock(reg.mtx);
  auto it = reg.map.find(key);
  return (it != reg.map.end()) ? &it->second : nullptr;
}

ncclResult_t ncclParamRegistry::remove(std::string key) {
  auto& reg = state();
  std::lock_guard<std::mutex> lock(reg.mtx);
  reg.map.erase(key);
  return ncclSuccess;
}
