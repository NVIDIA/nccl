/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_H_INCLUDED
#define PARAM_H_INCLUDED

#include "nccl.h"
#include "param/common.h"
#include "param/utils.h"
#include "param/parsers.h"
#include "param/param_registry.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <mutex>
// using C++ atomic because we don't include compiler.h here
#include <atomic>
#include <memory>

// ============================================================================
// Main Macros
// ============================================================================

// Defining and Using (Including) a ncclParam
// Usage: DEFINE_NCCL_PARAM(name, type, key, default, flags, parser, desc)
// Generate global symbols for key to make compiler check for the uniqueness of the key.
// The check will happen at both compile and link time.
// Define parameter in .cc files
// Note: NCCL_DEFINE_PARAM is not designed to be put inside of a namespace. This can be
// changed if there is a need.
#define DEFINE_NCCL_PARAM(name, type, key, default, flags, parser, desc) \
  namespace key_guards { \
  struct guard_##key {}; \
  }; \
  extern constexpr char name##Key[] = #key; \
  ncclParam<type> name{name##Key, default, parser, #type, flags, desc};

// Usage: USE_NCCL_PARAM(name, type)
// name and type must match the DEFINE_NCCL_PARAM.
#define USE_NCCL_PARAM(name, type) extern ncclParam<type> name;

// ============================================================================
// ncclParam Template
// ============================================================================

template <typename T>
struct ncclParam : public ncclParamInterface {
  const ncclParamInfo_t info;
  const T defaultValue;

  T value;
  // cstrData adds 24B overhead to ncclParam for non-const-char* parameters.
  // This is a trade-off for not using complex template stuff.
  std::string cstrData{};

  const char* srcStr = nullptr;

  std::mutex mtx;
  std::atomic<bool> loaded{false};

  ncclParamParser<T> parser;

  ~ncclParam() override = default;

  ncclParam(const char* key, T defVal, ncclParamParser<T> parser = {}, const char* typeStr = "",
            uint64_t flags = NCCL_PARAM_FLAG_NONE, const char* desc = "")
    : info({ncclParamTypeIdOf<T>(), flags, typeStr, key, desc}), defaultValue(defVal), value(defVal),
      parser(std::move(parser)) {
    if (!this->parser) {
      this->parser = ncclParamDefault<T>();
    }
    ncclParamRegistry::add(info.key, info, this);
  }

  // Prevent copy/move assignment
  ncclParam& operator=(const ncclParam&) = delete;
  ncclParam& operator=(ncclParam&&) = delete;

  // Main access function for parameter value through function call-like interface
  // This handles non-const-char* types, for const char *, see specializations below
  T operator()() {
    auto lock = ensureLoaded();
    return value;
  }

  // C API accessor of raw parameter value
  ncclResult_t getRawData(void* out, int maxLen, int* len) override {
    if (!out || !len || maxLen <= 0) return ncclInvalidArgument;
    auto lock = ensureLoaded();
    if (static_cast<int>(sizeof(T)) > maxLen) {
      *len = 0;
      return ncclInvalidArgument;
    }
    std::memcpy(out, &value, sizeof(T));
    *len = static_cast<int>(sizeof(T));
    return ncclSuccess;
  }

  std::string toString() override {
    auto lock = ensureLoaded();
    return parser.toString(value);
  }

  std::string dump() override {
    std::string currentStr = this->toString();
    std::string defaultStr = parser.toString(defaultValue);
    std::string flagStr = nccl::param::utils::flagsStr(info.flags);

    // Line 1: Key (type) [flags] desc
    // Line 2: Current value, set_by=srcStr and default value
    // Line 3+: Accepted values
    using nccl::param::utils::stringFormat;
    return stringFormat("%s (%s) [%s] %s\n"
                        "    Current value=%s set_by=%s default=%s\n"
                        "    Accepted value: %s\n",
                        info.key, info.typeStr, flagStr.c_str(), info.desc,
                        (currentStr.empty() ? "<unset>" : currentStr.c_str()), srcStr,
                        (defaultStr.empty() ? "<unset>" : defaultStr.c_str()), parser.desc.c_str());
  }

private:
  // Core function to make sure value is loaded, check against all conditions including
  // NCCL_NO_CACHE
  std::unique_lock<std::mutex> ensureLoaded() {
    if (NCCL_PARAM_COMPILER_EXPECT(loaded.load(std::memory_order_relaxed), true)) {
      if ((info.flags & NCCL_PARAM_FLAG_CACHED) &&
          NCCL_PARAM_COMPILER_EXPECT(!ncclParamIsCacheDisabled(info.key), true)) {
        // fast path for cached parameters
        return {};
      } else {
        std::unique_lock<std::mutex> lock(mtx);
        loadValue();
        return lock;
      }
    } else {
      std::unique_lock<std::mutex> lock(mtx);
      if (NCCL_PARAM_COMPILER_EXPECT(!loaded.load(std::memory_order_acquire), true)) {
        loadValue();
        loaded.store(true, std::memory_order_release);
      }
      return lock;
    }
  }

  // Load value from environment variable via EnvPlugin chain
  void loadValue() {
    // Special params with NO_ENVPLUGIN_INIT flag do not try init EnvPlugin
    bool tryEnvPluginInit = !(info.flags & NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT);
    const char* envPluginValue = ncclParamEnvPluginGet(info.key, tryEnvPluginInit);
    if (envPluginValue != nullptr) {
      T resolvedValue;
      ncclResult_t resolved = parser.resolve(envPluginValue, resolvedValue);
      if (resolved == ncclSuccess && parser.validate(resolvedValue)) {
        value = resolvedValue;
        srcStr = nccl::param::utils::srcEnvPlugin();
        return;
      }
    }

    // env is empty or parsing is failed
    srcStr = nccl::param::utils::srcDefault();
    value = defaultValue;
  }
};

// ============================================================================
// const char* specializations, it need special version for some functions
// ============================================================================

template <>
inline const char* ncclParam<const char*>::operator()() {
  auto lock = ensureLoaded();
  if (value == nullptr) return nullptr;
  static thread_local std::string tlsCstrCopy;
  tlsCstrCopy = cstrData;
  return tlsCstrCopy.c_str();
}

template <>
inline void ncclParam<const char*>::loadValue() {
  // Special params with NO_ENVPLUGIN_INIT flag do not try init EnvPlugin
  bool tryEnvPluginInit = !(info.flags & NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT);
  const char* envPluginValue = ncclParamEnvPluginGet(info.key, tryEnvPluginInit);
  if (envPluginValue != nullptr) {
    cstrData = envPluginValue;
    value = cstrData.c_str();
    srcStr = nccl::param::utils::srcEnvPlugin();
  } else {
    if (defaultValue) {
      cstrData = defaultValue;
      value = cstrData.c_str();
    } else {
      cstrData.clear();
      value = nullptr;
    }
    srcStr = nccl::param::utils::srcDefault();
  }
}

template <>
inline ncclResult_t ncclParam<const char*>::getRawData(void* out, int maxLen, int* len) {
  if (!out || !len || maxLen <= 0) return ncclInvalidArgument;
  auto lock = ensureLoaded();
  int sz = static_cast<int>(cstrData.size()) + 1;
  if (sz > maxLen) {
    *len = 0;
    return ncclInvalidArgument;
  }
  std::memcpy(out, cstrData.c_str(), static_cast<size_t>(sz));
  *len = sz;
  return ncclSuccess;
}

#endif /* PARAM_H_INCLUDED */
