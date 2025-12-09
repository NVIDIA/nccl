// (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include <folly/File.h>
#include <folly/Synchronized.h>

#include "meta/logger/DataSink.h"
#include "meta/logger/NcclScubaSample.h"

class DataTable {
 public:
  DataTable(const std::string& tableName);
  virtual ~DataTable();
  // std::mutex is neither copyable nor movable. So neither is this class.
  DataTable(DataTable const&) = delete;
  DataTable& operator=(DataTable const&) = delete;
  DataTable(DataTable&&) = delete;
  DataTable& operator=(DataTable&&) = delete;

  void addSample(NcclScubaSample sample);
  void shutdown();

 protected:
  virtual void writeMessage(const std::string& message);

 private:
  struct State {
    std::vector<NcclScubaSample> samples;
    bool stopTriggered{false};
  };

  // Wait until there are messages, or until shutdown is triggered
  // Returns State
  State waitAndGetAllMessages();
  void loggingFunc();

  folly::Synchronized<State, std::mutex> state_;
  std::condition_variable cv_;
  std::unique_ptr<DataSink> sink_;
  const std::string tableName_;
  std::atomic_bool done_;
  std::thread thread_;
};
