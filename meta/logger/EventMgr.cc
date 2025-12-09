// (c) Meta Platforms, Inc. and affiliates.

#include "meta/logger/EventMgr.h"

#include <memory>
#include <mutex>

#include "meta/cvars/nccl_cvars.h"
#include "meta/logger/EventMgrHelperTypes.h"

NcclScubaSample CommEvent::toSample() {
  NcclScubaSample sample("CommEvent");
  sample.addInt("commId", commId);
  sample.addInt("commHash", commHash);
  sample.addNormal("commDesc", commDesc);
  sample.addInt("rank", rank);
  sample.addInt("nranks", nRanks);
  sample.addInt("localRank", localRank);
  sample.addInt("localRanks", localRanks);

  sample.addNormal("stage", stage);
  sample.addNormal("split", split);

  sample.addDouble("timerDeltaMs", timerDeltaMs);
  sample.addNormal("timestamp", timestamp);

  return sample;
}

// Define as unique ptr to reset the flag for testing
static std::unique_ptr<std::once_flag> memoryEventFilterFlag =
    std::make_unique<std::once_flag>();
static EventGlobalRankFilter memoryEventFilter;

static std::unique_ptr<std::once_flag> memoryRegEventFilterFlag =
    std::make_unique<std::once_flag>();
static EventGlobalRankFilter memoryRegEventFilter;

void MemoryEvent::resetFilter() {
  memoryEventFilterFlag = std::make_unique<std::once_flag>();
  memoryRegEventFilterFlag = std::make_unique<std::once_flag>();
}

bool MemoryEvent::shouldLog() {
  std::call_once(*memoryEventFilterFlag, []() {
    memoryEventFilter.initialize(
        NCCL_FILTER_MEM_LOGGING_BY_RANKS, "NCCL_FILTER_MEM_LOGGING_BY_RANKS");
    memoryRegEventFilter.initialize(
        NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS,
        "NCCL_FILTER_MEM_REG_LOGGING_BY_RANKS");
  });

  // Apply different filter for reg and non-reg events
  if (isRegMemEvent) {
    return memoryRegEventFilter.isAllowed();
  } else {
    return memoryEventFilter.isAllowed();
  }
}

NcclScubaSample MemoryEvent::toSample() {
  NcclScubaSample sample("MemoryEvent");
  sample.addInt("commHash", commHash);
  sample.addNormal("commDesc", commDesc);
  sample.addInt("rank", rank);
  sample.addInt("nranks", nRanks);
  sample.addInt("memoryAddr", memoryAddr);
  if (bytes.has_value()) {
    sample.addInt("bytes", bytes.value());
  }
  if (numSegments.has_value()) {
    sample.addInt("numSegments", numSegments.value());
  }
  if (durationUs.has_value()) {
    sample.addInt("durationUs", durationUs.value());
  }
  sample.addNormal("callsite", callsite);
  sample.addNormal("use", use);
  sample.addInt("iteration", iteration);
  return sample;
}
