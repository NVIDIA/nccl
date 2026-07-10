/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_UTILS_H_INCLUDED
#define PARAM_UTILS_H_INCLUDED

#include "param/common.h"
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <type_traits>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstring>

// TODO: This should be consolidate with NCCL_IF_CONSTEXPR in device headers
#if defined(__cpp_if_constexpr) && __cpp_if_constexpr >= 201606
#ifndef NCCL_PARAM_IF_CONSTEXPR
#define NCCL_PARAM_IF_CONSTEXPR constexpr
#endif
#else
#ifndef NCCL_PARAM_IF_CONSTEXPR
#define NCCL_PARAM_IF_CONSTEXPR
#endif
#endif

// Compiler detection macros
#if defined(__GNUC__) || defined(__clang__)
#define NCCL_PARAM_COMPILER_EXPECT(x, v) __builtin_expect((x), (v))
#elif defined(_MSC_VER)
#define NCCL_PARAM_COMPILER_EXPECT(x, v) (x)
#else
#error "Unsupported compiler"
#endif

template <typename T>
constexpr ncclParamTypeId_t ncclParamTypeIdOf() noexcept {
  if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, int8_t>::value) return NCCL_PARAM_TYPE_I8;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, int16_t>::value) return NCCL_PARAM_TYPE_I16;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, int32_t>::value) return NCCL_PARAM_TYPE_I32;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, int64_t>::value) return NCCL_PARAM_TYPE_I64;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, uint8_t>::value) return NCCL_PARAM_TYPE_U8;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, uint16_t>::value) return NCCL_PARAM_TYPE_U16;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, uint32_t>::value) return NCCL_PARAM_TYPE_U32;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, uint64_t>::value) return NCCL_PARAM_TYPE_U64;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, bool>::value) return NCCL_PARAM_TYPE_BOOL;
  else if NCCL_PARAM_IF_CONSTEXPR (std::is_same<T, const char*>::value) return NCCL_PARAM_TYPE_CSTR;
  else return NCCL_PARAM_TYPE_RAW;
}

extern "C" const char* ncclParamEnvPluginGet(const char* key, bool env_init);
extern "C" bool ncclParamIsCacheDisabled(const char* key);

namespace nccl {
namespace param {
namespace utils {

inline std::string flagsStr(uint64_t flags) {
  static const struct {
    uint64_t flag;
    const char* name;
  } entries[] = {
    {NCCL_PARAM_FLAG_DEPRECATED, "Deprecated"},
    {NCCL_PARAM_FLAG_CACHED, "Cached"},
    {NCCL_PARAM_FLAG_UNUSED, "Unused"},
    {NCCL_PARAM_FLAG_NO_ENVPLUGIN_INIT, "NoEnvPluginInit"},
  };
  std::string result = (flags & NCCL_PARAM_FLAG_PUBLISHED) ? "" : "Private";
  for (const auto& e : entries) {
    if (flags & e.flag) {
      if (!result.empty()) result += ", ";
      result += e.name;
    }
  }
  return result;
}

inline constexpr const char* srcDefault() {
  return "Default";
}
inline constexpr const char* srcEnvPlugin() {
  return "EnvPlugin";
}

// simple formatting helper that does
// std::string s = string_format("Name: %s, type: %s", name, type);
template <typename... Args>
std::string stringFormat(const char* fmt, Args... args) {
  int n = std::snprintf(nullptr, 0, fmt, args...);
  if (n < 0) return {};

  std::vector<char> buf(static_cast<size_t>(n) + 1);
  std::snprintf(buf.data(), buf.size(), fmt, args...);
  return std::string(buf.data());
}

// Case-insensitive ASCII compare without allocating.
inline bool iequals(std::string a, std::string b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(a[i])) != std::tolower(static_cast<unsigned char>(b[i]))) {
      return false;
    }
  }
  return true;
}

inline std::string trim(std::string s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) --end;
  return s.substr(start, end - start);
}

inline std::vector<std::string> split(std::string sv, char delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = sv.find(delimiter);
  while (end != std::string::npos) {
    result.push_back(sv.substr(start, end - start));
    start = end + 1;
    end = sv.find(delimiter, start);
  }
  result.push_back(sv.substr(start));
  return result;
}

} // namespace utils
} // namespace param
} // namespace nccl

#endif /* PARAM_UTILS_H_INCLUDED */
