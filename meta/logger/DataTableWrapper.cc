// (c) Meta Platforms, Inc. and affiliates.

#include "meta/logger/DataTableWrapper.h"

#include <chrono>

#include <folly/Singleton.h>
#include <folly/String.h>
#include <folly/logging/xlog.h>
#include <iostream>

#include "meta/cvars/nccl_cvars.h"

DEFINE_scuba_table(nccl_error_logging);
DEFINE_scuba_table(nccl_memory_logging);
DEFINE_scuba_table(nccl_structured_logging);
DEFINE_scuba_table(nccl_coll_trace);

DataTableWrapper::DataTableWrapper(const std::string& tableName)
    : tableName_(tableName) {
  table_ = std::make_shared<DataTable>(tableName);
}

void DataTableWrapper::addSample(NcclScubaSample sample) {
  if (table_ != nullptr) {
    // Add timestamp when sample is published
    const auto timestamp = std::chrono::system_clock::now().time_since_epoch();
    sample.addInt(
        "time",
        std::chrono::duration_cast<std::chrono::seconds>(timestamp).count());
    sample.addNormal(
        "timestamp",
        std::to_string(
            std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp)
                .count()));
    table_->addSample(std::move(sample));
  }
}

std::shared_ptr<DataTable> DataTableWrapper::getTable() {
  return table_;
}

void DataTableWrapper::init() {
  INIT_scuba_table(nccl_error_logging);
  INIT_scuba_table(nccl_memory_logging);
  INIT_scuba_table(nccl_structured_logging);
  INIT_scuba_table(nccl_coll_trace);
}

void initLogger() {
  initLoggerProvider();
  DataTableWrapper::init();
}
