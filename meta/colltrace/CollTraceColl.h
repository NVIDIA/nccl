// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <optional>

#include "info.h"
#include "meta/logger/EventMgrHelperTypes.h"
#include "nccl.h"
#include "nccl_common.h"
#include "proxy.h"

struct CollBaselineAttr {
  ncclFunc_t coll;
  int algorithm;
  int protocol;
  ncclRedOp_t op;
  int root; // peer for p2p operations
  ncclPattern_t pattern;
  int nChannels;
  int channelId;
};

// Result data structure
struct CollTraceColl {
  uint64_t opCount;
  ncclComm_t comm;
  CommLogData logMetaData; // Ensure CollTraceColl could be serialized after
                               // comm is destroyed

  int64_t iteration;
  cudaStream_t stream;

  std::string opName;
  std::string algoName;
  std::optional<const void*> sendbuff;
  std::optional<const void*> recvbuff;
  std::optional<std::vector<int>> ranksInGroupedP2P;
  // alltoallv doesn't have a single count so this must be optional
  std::optional<size_t> count;
  ncclDataType_t dataType{ncclNumTypes};
  int nThreads;

  enum class Codepath {
    UNDEFINED,
    BASELINE,
    CTRAN,
    CTRAN_CPU,
  } codepath{Codepath::UNDEFINED};

  // Baseline only attributes
  std::optional<CollBaselineAttr> baselineAttr;

  float latency{-1};
  // This is achieved by waiting for the start event. We can only guarantee
  // before this time point kernel has already started, but we cannot guarantee
  // kernel started exactly at this time point.
  std::chrono::time_point<std::chrono::high_resolution_clock> startTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> enqueueTs{};
  std::chrono::microseconds interCollTime;

  ScubaEntry toScubaEntry() const;
  CollSignature toCollSignature() const;

  // serialize the entry to a json format string
  std::string serialize(bool quoted = false) const;
  // flatten the entry to a plain string
  std::string toString() const;

 private:
  // internal helper function to retrieve the struct to a string map
  std::unordered_map<std::string, std::string> retrieveMap(bool quoted) const;
};
