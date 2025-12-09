// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/logger/ScubaLogger.h"

#include <chrono>

#include <folly/logging/xlog.h>

DataTableWrapper* getTablePtrFromEvent(LoggerEventType event) {
  switch (event) {
    case LoggerEventType::ErrorEventType:
      return SCUBA_nccl_error_logging_ptr.get();
    case LoggerEventType::MemoryEventType:
      return SCUBA_nccl_memory_logging_ptr.get();
    case LoggerEventType::CommEventType:
      return SCUBA_nccl_structured_logging_ptr.get();
    default:
      XLOG(ERR) << "Invalid event type";
      break;
  }
  throw std::runtime_error("Invalid event type");
}

void ncclLogToScuba(LoggerEventType event, NcclScubaSample& sample) {
  const auto tablePtr = getTablePtrFromEvent(event);
  if (tablePtr != nullptr) {
    tablePtr->addSample(std::move(sample));
  }
}

void NcclScubaEvent::startAndRecord() {
  timer_.reset();
  if (!stage_.empty()) {
    sample_.addNormal("stage", stage_ + " START");
  }
  record();
}

void NcclScubaEvent::stopAndRecord() {
  auto latency = std::chrono::duration<double, std::milli>(timer_.lap());
  sample_.addDouble("timerDeltaMs", latency.count());
  if (!stage_.empty()) {
    sample_.addNormal("stage", stage_ + " COMPLETE");
  }
  record();
}

void NcclScubaEvent::lapAndRecord(const std::string& stage) {
  auto latency = std::chrono::duration<double, std::milli>(timer_.lap());
  sample_.addDouble("timerDeltaMs", latency.count());
  if (!stage.empty()) {
    sample_.addNormal("stage", stage);
  }
  record();
}

void NcclScubaEvent::record() {
  auto copySample = sample_.makeCopy();
  ncclLogToScuba(type_, copySample);
}

void NcclScubaEvent::setLogMetatData(const CommLogData* logMetaData) {
  sample_.setCommunicatorMetadata(logMetaData);
}

NcclScubaEvent::NcclScubaEvent(const std::string& stage)
    : sample_(""), type_(LoggerEventType::CommEventType) {
  sample_.addNormal("stage", stage);
}

NcclScubaEvent::NcclScubaEvent(const CommLogData* logMetaData)
    : sample_(""), type_(LoggerEventType::CommEventType) {
  sample_.setCommunicatorMetadata(logMetaData);
}

NcclScubaEvent::NcclScubaEvent(const std::unique_ptr<LoggerEvent> loggerEvent)
    : sample_(loggerEvent->toSample()) {
  stage_ = loggerEvent->getStage();
  type_ = loggerEvent->getEventType();
}

NcclScubaEvent::NcclScubaEvent(
    const std::string& stage,
    const CommLogData* logMetaData)
    : sample_(""), stage_(stage), type_(LoggerEventType::CommEventType) {
  sample_.addNormal("stage", stage);
  sample_.setCommunicatorMetadata(logMetaData);
}
