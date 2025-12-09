// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/logger/alloc.h"

#include <memory>

#include "meta/logger/ScubaLogger.h"

// This file is needed to avoid cicular dependency between
// comms/ncclx/.../src/include/alloc.h and meta/logger/EventMgr.h.
// For more context see D60405975 implement logMemoryEvent
void logMemoryEvent(
    const CommLogData& logMetaData,
    const std::string& callsite,
    const std::string& use,
    uintptr_t memoryAddr,
    std::optional<int64_t> bytes,
    std::optional<int> numSegments,
    std::optional<int64_t> durationUs,
    bool isRegMemEvent) {
  auto memoryEvent = std::make_unique<MemoryEvent>(
      logMetaData,
      callsite,
      use,
      memoryAddr,
      bytes,
      numSegments,
      durationUs,
      isRegMemEvent);
  if (memoryEvent->shouldLog()) {
    NcclScubaEvent event(std::move(memoryEvent));
    event.record();
  }
}
