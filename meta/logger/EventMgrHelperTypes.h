// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <sstream>
#include <string>
#include <unordered_map>

#include "meta/logger/StrUtils.h"

class CollSignature {
 public:
  uint64_t commHash{};
  uint64_t pgid{};
  int rank{};
  std::string opName;
  int nThreads{};
  std::string dataType;
  std::string algoName;
  int count{};
  int root{};
  std::string redOp;

  CollSignature() {}
  CollSignature(
      uint64_t commHash,
      uint64_t pgid,
      int rank,
      const std::string& opName,
      int nThreads,
      const std::string& dataType,
      int count = -1,
      int nChannels = -1,
      int root = -1,
      const std::string& redOp = "Unknown op",
      const std::string& protocol = "N/A",
      const std::string& algorithm = "N/A",
      const std::string& pattern = "N/A")
      : commHash(commHash),
        pgid(pgid),
        rank(rank),
        opName(opName),
        nThreads(nThreads),
        dataType(dataType),
        count(count),
        root(root),
        redOp(redOp) {
    algoName = protocol + "_" + algorithm + "_" + pattern + "_" +
        std::to_string(nChannels);
  }

  CollSignature(
      uint64_t commHash,
      uint64_t pgid,
      int rank,
      const std::string& opName,
      int nThreads,
      const std::string& dataType,
      const std::string& algoName,
      int count = -1,
      int root = -1,
      const std::string& redOp = "Unknown op")
      : commHash(commHash),
        pgid(pgid),
        rank(rank),
        opName(opName),
        nThreads(nThreads),
        dataType(dataType),
        algoName(algoName),
        count(count),
        root(root),
        redOp(redOp) {}

  bool operator==(const CollSignature& other) const {
    return commHash == other.commHash && pgid == other.pgid &&
        rank == other.rank && opName == other.opName &&
        nThreads == other.nThreads && dataType == other.dataType &&
        count == other.count && root == other.root && redOp == other.redOp &&
        algoName == other.algoName;
  }

  std::string toString() const {
    std::stringstream ss;
    ss << "commHash: 0x" << std::hex << commHash << ", pgid: " << std::dec
       << pgid << ", rank: " << std::dec << rank << ", opName: " << opName
       << ", nThreads: " << std::dec << nThreads << ", dataType: " << dataType
       << ", count: " << count << ", root: " << std::to_string(root)
       << ", redOp: " << redOp << ", algoName: " << algoName;

    return ss.str();
  }
};

template <>
struct std::hash<CollSignature> {
  size_t operator()(const CollSignature& sig) const noexcept {
    std::size_t seed = 0xfaceb00c;
    hash_combine(seed, sig.commHash);
    hash_combine(seed, sig.pgid);
    hash_combine(seed, sig.rank);
    hash_combine(seed, sig.opName);
    hash_combine(seed, sig.nThreads);
    hash_combine(seed, sig.dataType);
    hash_combine(seed, sig.count);
    hash_combine(seed, sig.root);
    hash_combine(seed, sig.redOp);
    hash_combine(seed, sig.algoName);

    return seed;
  }
};

class ScubaEntry {
 public:
  using int_type = int64_t;
  using double_type = double;

  ScubaEntry() = default;
  void addCommonFields();
  void addNormalValue(const std::string& key, const std::string& value) {
    normalMap_[key] = value;
  }

  void addIntValue(const std::string& key, const int_type value) {
    intMap_[key] = value;
  }
  void addDoubleValue(const std::string& key, const double_type value) {
    doubleMap_[key] = value;
  }

  std::unordered_map<std::string, std::string>& getNormalMap() {
    return normalMap_;
  }
  std::unordered_map<std::string, int_type>& getIntMap() {
    return intMap_;
  }
  std::unordered_map<std::string, double_type>& getDoubleMap() {
    return doubleMap_;
  }

  template <typename T>
  std::vector<std::string> getKeys(
      const std::unordered_map<std::string, T>& map) const {
    std::vector<std::string> allKeys;
    for (auto& [key, _] : map) {
      allKeys.push_back(key);
    }
    return allKeys;
  }

 private:
  std::unordered_map<std::string, std::string> normalMap_;
  std::unordered_map<std::string, int_type> intMap_;
  std::unordered_map<std::string, double_type> doubleMap_;
};

// Event helper class for filtering events by global rank.
// To use it, an event class is required to call initialize() once to set up the
// filter.
class EventGlobalRankFilter {
 public:
  EventGlobalRankFilter() = default;
  ~EventGlobalRankFilter() = default;

  // Initialize the filter with the rank list from the cvar
  void initialize(
      const std::vector<std::string>& rankListCvar,
      const std::string& filterName);

  // Return whether the global rank is allowed to log.
  bool isAllowed() const {
    return isAllowed_;
  }

 private:
  int globalRank_{-1};
  bool isAllowed_{true};
  std::string filterName_{"undefined"};
};
