// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef TRACE_UTILES_H
#define TRACE_UTILES_H

#include <chrono>
#include <deque>
#include <iomanip>
#include <list>
#include <set>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

// TODO: should this be util function?
static inline std::string timePointToStr(
    std::chrono::time_point<std::chrono::high_resolution_clock> ts) {
  std::time_t ts_c = std::chrono::system_clock::to_time_t(ts);
  auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
                   ts.time_since_epoch()) %
      1000000;
  std::stringstream ts_ss;
  struct tm nowTm;
  localtime_r(&ts_c, &nowTm);
  ts_ss << std::put_time(&nowTm, "%T.") << std::setfill('0') << std::setw(6)
        << ts_us.count();
  return ts_ss.str();
}

// Check if a type is in a variant type. Got the code from:
// https://stackoverflow.com/questions/45892170/how-do-i-check-if-an-stdvariant-can-hold-a-certain-type
template <typename T, typename VARIANT_T>
struct isVariantMember;

template <typename T, typename... ALL_T>
struct isVariantMember<T, std::variant<ALL_T...>>
    : public std::disjunction<std::is_same<T, ALL_T>...> {};

template <typename VariantType, typename T, std::size_t index = 0>
constexpr std::size_t variant_index() {
  static_assert(
      std::variant_size_v<VariantType> > index, "Type not found in variant");
  if constexpr (index == std::variant_size_v<VariantType>) {
    return index;
  } else if constexpr (std::is_same_v<
                           std::variant_alternative_t<index, VariantType>,
                           T>) {
    return index;
  } else {
    return variant_index<VariantType, T, index + 1>();
  }
}

#endif
