// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "RankUtils.h"

#include <cstdlib>

#include <folly/Conv.h>

/* static */
std::optional<int64_t> RankUtils::getInt64FromEnv(const char* envVar) {
  char* envVarValue = getenv(envVar);
  if (envVarValue && strlen(envVarValue)) {
    if (auto result = folly::tryTo<int64_t>(envVarValue); result.hasValue()) {
      return result.value();
    }
  }
  return std::nullopt;
}

/* static */
std::optional<int64_t> RankUtils::getWorldSize() {
  auto worldSize = getInt64FromEnv("WORLD_SIZE");
  if (worldSize.has_value()) {
    return worldSize;
  }
  worldSize = getInt64FromEnv("OMPI_COMM_WORLD_SIZE");
  if (worldSize.has_value()) {
    return worldSize;
  }
  return getInt64FromEnv("SLURM_NTASKS");  
}

/* static */
std::optional<int64_t> RankUtils::getGlobalRank() {
  auto rank = getInt64FromEnv("RANK");
  if (rank.has_value()) {
    return rank;
  }
  rank = getInt64FromEnv("OMPI_COMM_WORLD_RANK");
  if (rank.has_value()) {
    return rank;
  }
  return getInt64FromEnv("SLURM_PROCID");
}
