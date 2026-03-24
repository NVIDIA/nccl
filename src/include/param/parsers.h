/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef PARAM_PARSERS_H_INCLUDED
#define PARAM_PARSERS_H_INCLUDED

#include "nccl.h"
#include "param/utils.h"

#include <string>
#include <type_traits>
#include <functional>
#include <array>
#include <cstdlib>
#include <cerrno>

// ============================================================================
// Parser Infrastructure
// ============================================================================

// NcclParserFuncs: Runtime parser (std::function with captured state)
// Used for oneOf(), makeBitsetOf(), and custom lambda parsers
template <typename T>
struct NcclParserFuncs {
  std::function<ncclResult_t(const char*, T&)> resolve;
  std::function<ncclResult_t(const T&)> validate;  // Post-parse validation
  std::function<std::string(const T&)> toString;
  std::function<std::string()> desc;  // Returns description of accepted values
};

// NcclParamParserDefault: Use as the parser argument in DEFINE_NCCL_PARAM when no
// custom parser is needed.  Implicitly converts to an empty NcclParserFuncs<T>
// (all functions null), causing NcclParam to fall back to NcclDefaultParser<T>.
struct NcclParamParserDefaultTag {
  template <typename T>
    operator NcclParserFuncs<T>() const { return {}; }
};
constexpr NcclParamParserDefaultTag NcclParamParserDefault{};

// ============================================================================
// Default Parsers (Static - Zero Overhead)
// ============================================================================

// Primary template - catch all for unsupported types
template <typename T, typename Enable = void>
struct NcclDefaultParser {
  static ncclResult_t resolve(const char*, T&) { return ncclInvalidArgument; }

  static ncclResult_t validate(const T&) { return ncclSuccess; }

  static std::string toString(const T&) {
    return "<unsupported>";
  }

  static std::string desc() {
    return {};
  }
};

// Specialization for integral types (excluding bool which has its own specialization)
template <typename T>
struct NcclDefaultParser<T, std::enable_if_t<
std::is_integral<T>::value && !std::is_same<T, bool>::value>> {

  static ncclResult_t resolve(const char* input, T& out) {
    if (input == nullptr) return ncclInvalidArgument;

    std::string trimmed = nccl::param::utils::trimCopy(input);
    if (trimmed.empty()) return ncclInvalidArgument;

    // Reject negative values for unsigned types
    if (std::is_unsigned<T>::value && trimmed[0] == '-') return ncclInvalidArgument;

    // Do not accept a leading '+'
    if (trimmed[0] == '+') return ncclInvalidArgument;

    return nccl::param::utils::parseIntegral<T>(trimmed.c_str(), out)
           ? ncclSuccess : ncclInvalidArgument;
  }

  static ncclResult_t validate(const T&) { return ncclSuccess; }

  static std::string toString(const T& value) {
    return std::to_string(value);
  }

  static std::string desc() {
    return "Integer";
  }
};

// Specialization for bool
template <>
struct NcclDefaultParser<bool> {
  static ncclResult_t resolve(const char* input, bool& out) {
    if (input == nullptr) return ncclInvalidArgument;
    std::string upper = nccl::param::utils::toUpper(input);
    if (upper == "1" || upper == "T" || upper == "TRUE")  { out = true;  return ncclSuccess; }
    if (upper == "0" || upper == "F" || upper == "FALSE") { out = false; return ncclSuccess; }
    return ncclInvalidArgument;
  }

  static ncclResult_t validate(const bool&) { return ncclSuccess; }

  static std::string toString(const bool& value) {
    return value ? "TRUE" : "FALSE";
  }

  static std::string desc() {
    return "Boolean: 1/T/TRUE or 0/F/FALSE";
  }
};

// Specialization for const char*
// Note: This parser returns a pointer into the provided input; NcclParam<const char*>
// owns/copies the string into internal storage in load_value()/assignment.
template <>
struct NcclDefaultParser<const char*> {
  static ncclResult_t resolve(const char* input, const char*& out) {
    out = input;
    return ncclSuccess;
  }

  static ncclResult_t validate(const char* const&) { return ncclSuccess; }

  static std::string toString(const char* const& value) {
    return value ? std::string(value) : std::string();
  }

  static std::string desc() {
    return "String";
  }
};

// ============================================================================
// Enum and Bitmask Parser Helpers
// ============================================================================

template <typename T>
struct NcclOption {
  std::string name;
  T value;
  std::string desc;  // Per-option description (default empty)
};

// NcclOptionSet: Fixed-size option set (compile-time N, zero heap allocation)
template <typename T, size_t N>
struct NcclOptionSet {
  std::array<NcclOption<T>, N> options;

  constexpr const NcclOption<T>* begin() const { return options.data(); }
  constexpr const NcclOption<T>* end() const { return options.data() + N; }
  constexpr size_t size() const { return N; }
};

// makeOption: Create an option (2-arg: no description)
template <typename T>
NcclOption<T> makeOption(std::string name, T value) {
  return NcclOption<T>{name, value, {}};
}

// makeOption: Create an option with description (3-arg)
template <typename T>
NcclOption<T> makeOption(std::string name, T value, std::string desc) {
  return NcclOption<T>{name, value, desc};
}

// makeOptions: Create a fixed-size option set from variadic arguments
template <typename T, typename... Args>
auto makeOptions(NcclOption<T> first, Args... rest)
  -> NcclOptionSet<T, 1 + sizeof...(Args)> {
    return NcclOptionSet<T, 1 + sizeof...(Args)>{{{first, rest...}}};
  }

// NcclParamOneOf: Create a parser for enum types
template <typename T, size_t N>
NcclParserFuncs<T> NcclParamOneOf(NcclOptionSet<T, N> options) {
  // Capture options by value (fixed-size, no heap allocation)
  NcclParserFuncs<T> result;
  result.resolve = [options](const char* input, T& out) -> ncclResult_t {
    if (input == nullptr) return ncclInvalidArgument;
    std::string token = nccl::param::utils::trimCopy(input);
    if (token.empty()) return ncclInvalidArgument;
    for (const auto& opt : options) {
      if (nccl::param::utils::iequals(opt.name, token)) { out = opt.value; return ncclSuccess; }
    }
    return ncclInvalidArgument;
  };
  result.toString = [options](const T& value) -> std::string {
    for (const auto& opt : options) {
      if (opt.value == value) {
        return opt.name;
      }
    }
    return "<unknown>";
  };
  std::string d = "One of:";
  for (const auto& opt : options) {
    d += "\n        ";
    d += opt.name;
    if (!opt.desc.empty()) {
      d += " - ";
      d += opt.desc;
    }
  }
  result.desc = [d = std::move(d)]() -> std::string { return d; };
  return result;
}

// NcclParamBitsetOf: Create a parser for bitmask types
// EnumT is the enum type used in options, ResultT is derived from its underlying type
// Usage: NcclParamBitsetOf<MyEnum>(makeOptions(...))
template <typename EnumT, size_t N>
NcclParserFuncs<std::underlying_type_t<EnumT>> NcclParamBitsetOf(NcclOptionSet<EnumT, N> options,
                                                                 char delimiter = ',') {
  using ResultT = std::underlying_type_t<EnumT>;

  // Capture options by value (fixed-size, no heap allocation)
  NcclParserFuncs<ResultT> result;
  result.resolve = [options, delimiter](const char* input, ResultT& out) -> ncclResult_t {
    if (input == nullptr) return ncclInvalidArgument;
    ResultT res = 0;
    std::string str(input);
    size_t start = 0;
    while (start <= str.size()) {
      size_t end = str.find(delimiter, start);
      if (end == std::string::npos) end = str.size();
      std::string token = nccl::param::utils::trimCopy(str.substr(start, end - start).c_str());
      if (!token.empty()) {
        bool found = false;
        for (const auto& opt : options) {
          if (nccl::param::utils::iequals(opt.name, token)) {
            res |= static_cast<ResultT>(opt.value); found = true; break;
          }
        }
        if (!found) return ncclInvalidArgument;
      }
      if (end == str.size()) break;
      start = end + 1;
    }
    out = res;
    return ncclSuccess;
  };
  result.toString = [options](const ResultT& value) -> std::string {
    std::string strResult;
    ResultT remaining = value;

    // First check for exact matches (like "ALL")
    for (const auto& opt : options) {
      if (static_cast<ResultT>(opt.value) == value) {
        return opt.name;
      }
    }

    // Otherwise, decompose into individual bits
    auto isSingleBit = [](ResultT v) -> bool {
      using U = std::make_unsigned_t<ResultT>;
      U uv = static_cast<U>(v);
      return uv != 0 && (uv & (uv - 1)) == 0;
    };
    for (const auto& opt : options) {
      ResultT optVal = static_cast<ResultT>(opt.value);
      if (!isSingleBit(optVal)) continue;  // skip composite aliases like MOST
      if (optVal != 0 && (remaining & optVal) == optVal) {
        if (!strResult.empty()) strResult += ",";
        strResult += opt.name;
        remaining &= ~optVal;
      }
    }
    return strResult.empty() ? "NONE" : strResult;
  };
  std::string d = "Comma-separated list of:";
  for (const auto& opt : options) {
    d += "\n        ";
    d += opt.name;
    if (!opt.desc.empty()) {
      d += " - ";
      d += opt.desc;
    }
  }
  result.desc = [d = std::move(d)]() -> std::string { return d; };
  return result;
}

// NcclParamBounded: Create a parser with range bounds
// Both lower and upper bounds
template <typename T>
NcclParserFuncs<T> NcclParamBounded(T lower, T upper) {
  NcclParserFuncs<T> result;
  result.validate = [lower, upper](const T& val) -> ncclResult_t {
    return (val >= lower && val <= upper) ? ncclSuccess : ncclInvalidArgument;
  };
  std::string d = "Integer in range [" + std::to_string(lower) + ", " + std::to_string(upper) + "]";
  result.desc = [d = std::move(d)]() -> std::string { return d; };
  return result;
}

// Lower bound only (no upper bound)
template <typename T>
NcclParserFuncs<T> NcclParamBounded(T lower) {
  NcclParserFuncs<T> result;
  result.validate = [lower](const T& val) -> ncclResult_t {
    return (val >= lower) ? ncclSuccess : ncclInvalidArgument;
  };
  std::string d = "Integer >= " + std::to_string(lower);
  result.desc = [d = std::move(d)]() -> std::string { return d; };
  return result;
}

// NcclParamListOf: Parser for delimiter-separated strings into a container.
// ContainerT must support insert(end(), std::string) and iteration.
// Usage: NcclParamListOf<std::unordered_set<std::string>>(',')
template <typename ContainerT>
NcclParserFuncs<ContainerT> NcclParamListOf(char delimiter = ',') {
  NcclParserFuncs<ContainerT> result;

  result.resolve = [delimiter](const char* input, ContainerT& out) -> ncclResult_t {
    if (input == nullptr || *input == '\0') return ncclInvalidArgument;
    auto tokens = nccl::param::utils::split(input, delimiter);
    for (const auto& tok : tokens) {
      auto trimmed = nccl::param::utils::trim(tok);
      if (!trimmed.empty()) {
        out.insert(out.end(), std::string(trimmed.data(), trimmed.size()));
      }
    }
    return ncclSuccess;
  };

  result.toString = [delimiter](const ContainerT& value) -> std::string {
    std::string s;
    for (const auto& elem : value) {
      if (!s.empty()) { s += delimiter; }
      s += elem;
    }
    return s;
  };

  std::string d = std::string("Delimiter-separated list (delimiter='") + delimiter + "')";
  result.desc = [d = std::move(d)]() -> std::string { return d; };

  return result;
}

#endif /* PARAM_PARSERS_H_INCLUDED */
