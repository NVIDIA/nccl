// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <cstdint>
#include <optional>

class RankUtils {
 public:
  static std::optional<int64_t> getInt64FromEnv(const char* envVar);
  static std::optional<int64_t> getWorldSize();
  static std::optional<int64_t> getGlobalRank();
};
