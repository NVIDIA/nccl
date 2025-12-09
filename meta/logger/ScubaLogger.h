// (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <string>

#include <folly/stop_watch.h>

#include "meta/commSpecs.h"
#include "meta/logger/DataTableWrapper.h"
#include "meta/logger/EventMgr.h"
#include "meta/logger/NcclScubaSample.h"

class NcclScubaEvent {
 public:
  void startAndRecord();
  void stopAndRecord();
  void lapAndRecord(const std::string& stage = "");
  void record();
  void setLogMetatData(const CommLogData* logMetaData);

  explicit NcclScubaEvent(const std::string& stage);
  explicit NcclScubaEvent(const CommLogData* logMetaData);
  explicit NcclScubaEvent(const std::unique_ptr<LoggerEvent> loggerEvent);

  NcclScubaEvent(const std::string& stage, const CommLogData* logMetaData);

 private:
  NcclScubaSample sample_;
  folly::stop_watch<std::chrono::microseconds> timer_;
  std::string stage_{};
  LoggerEventType type_;
};

void ncclLogToScuba(LoggerEventType event, NcclScubaSample& sample);
DataTableWrapper* getTablePtrFromEvent(LoggerEventType event);
