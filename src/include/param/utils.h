/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_UTILS_H_INCLUDED
#define PARAM_UTILS_H_INCLUDED

#include <string>
#include <algorithm>
#include <cctype>
#include <vector>
#include <type_traits>
#include <limits>

// ============================================================================
// Utility Functions for NcclParam
// ============================================================================

namespace nccl {
namespace param {
namespace utils {

  // Case-insensitive ASCII compare without allocating.
  inline bool iequals(std::string a, std::string b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(a[i])) !=
          std::tolower(static_cast<unsigned char>(b[i]))) {
        return false;
      }
    }
    return true;
  }

  inline std::string toUpper(std::string s) {
    std::string result(s);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::toupper(c); });
    return result;
  }

  inline std::string trim(std::string s) {
    size_t start = 0;
    while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) ++start;
    size_t end = s.size();
    while (end > start && std::isspace(static_cast<unsigned char>(s[end-1]))) --end;
    return s.substr(start, end - start);
  }

  // Trim overload for const char* - returns pointer to first non-whitespace character
  // Note: Only trims leading whitespace; use trim_copy for full trimming
  inline const char* trim(const char* str) {
    if (str == nullptr) return nullptr;
    while (*str != '\0' && std::isspace(static_cast<unsigned char>(*str))) {
      ++str;
    }
    return str;
  }

  // Trim that returns a copy with both leading and trailing whitespace removed
  inline std::string trimCopy(const char* str) {
    if (str == nullptr) return std::string();
    return trim(std::string(str));
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

  // Helper to parse signed integers using strtoll
  template <typename T>
  typename std::enable_if<std::is_signed<T>::value, bool>::type
  parseIntegral(const char* input, T& out) {
    const char* begin = input;
    char* endPtr = nullptr;

    errno = 0;
    long long val = std::strtoll(begin, &endPtr, 10);

    // Check for errors: no conversion, partial conversion, overflow
    if (endPtr == begin || *endPtr != '\0' || errno == ERANGE) return false;

    // Check if value fits in T
    if (val < static_cast<long long>(std::numeric_limits<T>::min()) ||
        val > static_cast<long long>(std::numeric_limits<T>::max())) {
      return false;
    }

    out = static_cast<T>(val);
    return true;
  }

  // Helper to parse unsigned integers using strtoull
  template <typename T>
  typename std::enable_if<std::is_unsigned<T>::value, bool>::type
  parseIntegral(const char* input, T& out) {
    const char* begin = input;
    char* endPtr = nullptr;

    errno = 0;
    unsigned long long val = std::strtoull(begin, &endPtr, 10);

    // Check for errors: no conversion, partial conversion, overflow
    if (endPtr == begin || *endPtr != '\0' || errno == ERANGE) return false;

    // Check if value fits in T
    if (val > static_cast<unsigned long long>(std::numeric_limits<T>::max())) return false;

    out = static_cast<T>(val);
    return true;
  }

} // namespace utils
} // namespace param
} // namespace nccl

#endif /* PARAM_UTILS_H_INCLUDED */
