// (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "meta/logger/DataTable.h"
#include "meta/logger/NcclScubaSample.h"

class DataTableWrapper {
 public:
  explicit DataTableWrapper(const std::string& tableName);

  void addSample(NcclScubaSample sample);

  static void init();

  std::shared_ptr<DataTable> getTable();

 private:
  std::string tableName_;
  std::shared_ptr<DataTable> table_;
};

// Convenience macros to make it easier to log. Example usage:
//
// header file:
// DECLARE_scuba_table(nccl_structured_logging);
//
// cc file:
// DEFINE_scuba_table(nccl_structured_logging);
//
// call sites:
// NcclScubaSample sample;
// ... // add fields to sample
// SCUBA_nccl_structured_logging.addSample(sample);
#define DECLARE_scuba_table(tablename) \
  extern std::unique_ptr<DataTableWrapper> SCUBA_##tablename##_ptr

#define DEFINE_scuba_table(tablename) \
  std::unique_ptr<DataTableWrapper> SCUBA_##tablename##_ptr = nullptr;

#define INIT_scuba_table(tablename) \
  SCUBA_##tablename##_ptr = std::make_unique<DataTableWrapper>(#tablename)

DECLARE_scuba_table(nccl_error_logging);
DECLARE_scuba_table(nccl_memory_logging);
DECLARE_scuba_table(nccl_structured_logging);
DECLARE_scuba_table(nccl_coll_trace);

#define SCUBA_nccl_structured_logging (*SCUBA_nccl_structured_logging_ptr)
#define SCUBA_nccl_memory_logging (*SCUBA_nccl_memory_logging_ptr)
#define SCUBA_nccl_error_logging (*SCUBA_nccl_error_logging_ptr)
#define SCUBA_nccl_coll_trace (*SCUBA_nccl_coll_trace_ptr)
