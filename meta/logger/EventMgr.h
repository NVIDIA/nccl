// (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <sys/types.h>
#include <chrono>
#include <cstdint>
#include <optional>
#include <string>

#include "meta/commSpecs.h"
#include "meta/logger/NcclScubaSample.h"

#define LOGGER_PG_ID_DEFAULT 0x80000000UL

enum class LoggerEventType {
  DebugEventType,
  CommEventType,
  MemoryEventType,
  CollEventType,
  ErrorEventType,
  TerminateEventType,
};

class LoggerEvent {
 public:
  virtual void setTimerDelta(double delta) = 0;
  virtual void setTimestamp() = 0;
  virtual bool shouldLog() = 0;
  virtual LoggerEventType getEventType() = 0;
  virtual ~LoggerEvent() = default;
  virtual NcclScubaSample toSample() = 0;
  virtual std::string getStage() = 0;
};

class CommEvent : public LoggerEvent {
 public:
  CommEvent() = default;
  CommEvent(
      const CommLogData* logMetaData,
      const std::string& stage,
      const std::string& split,
      double delta = 0.0)
      : commId(logMetaData ? logMetaData->commId : 0),
        commHash(
            logMetaData
                ? logMetaData->commHash
                : 0xfaceb00c12345678 /*Dummy placeholder value for commHash*/),
        commDesc(logMetaData ? std::string(logMetaData->commDesc) : ""),
        rank(logMetaData ? logMetaData->rank : 0),
        nRanks(logMetaData ? logMetaData->nRanks : 0),
        stage(stage),
        split(split),
        timerDeltaMs(delta) {}

  CommEvent(
      const CommLogData* logMetaData,
      int localRank,
      int localRanks,
      const std::string& stage)
      : commId(logMetaData ? logMetaData->commId : 0),
        commHash(logMetaData ? logMetaData->commHash : 0xfaceb00c12345678),
        commDesc(logMetaData ? std::string(logMetaData->commDesc) : ""),
        rank(logMetaData ? logMetaData->rank : 0),
        nRanks(logMetaData ? logMetaData->nRanks : 0),
        localRank(localRank),
        localRanks(localRanks),
        stage(stage) {}

  ~CommEvent() override = default;

  void setTimerDelta(double delta) override {
    timerDeltaMs = delta;
  }

  void setTimestamp() override {
    auto now = std::chrono::steady_clock::now().time_since_epoch();
    timestamp = std::to_string(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  }

  bool shouldLog() override {
    return true;
  }

  LoggerEventType getEventType() override {
    return LoggerEventType::CommEventType;
  }

  std::string getStage() override {
    return stage;
  }
  NcclScubaSample toSample() override;

 private:
  const unsigned long long commId = 0;
  const uint64_t commHash = 0xfaceb00c12345678;
  const std::string commDesc;
  const int rank = 0;
  const int nRanks = 0;
  int localRank = -1;
  int localRanks = -1;
  const std::string stage;
  const std::string split;
  double timerDeltaMs = 0.0;
  std::string timestamp;
};

class MemoryEvent : public LoggerEvent {
 public:
  MemoryEvent() = default;
  MemoryEvent(
      const CommLogData& logMetaData,
      const std::string& callsite,
      const std::string& use,
      uintptr_t memoryAddr,
      std::optional<int64_t> bytes = std::nullopt,
      std::optional<int> numSegments = std::nullopt,
      std::optional<int64_t> durationUs = std::nullopt,
      bool isRegMemEvent = false)
      : commHash(logMetaData.commHash),
        commDesc(logMetaData.commDesc),
        rank(logMetaData.rank),
        nRanks(logMetaData.nRanks),
        callsite(callsite),
        use(use),
        memoryAddr(memoryAddr),
        bytes(bytes),
        numSegments(numSegments),
        durationUs(durationUs),
        isRegMemEvent(isRegMemEvent) {
  }

  ~MemoryEvent() override = default;

  void setTimerDelta(double delta) override {}
  void setTimestamp() override {}
  bool shouldLog() override;
  std::string getStage() override {
    return std::string("");
  }

  LoggerEventType getEventType() override {
    return LoggerEventType::MemoryEventType;
  }

  // For testing only
  static void resetFilter();
  NcclScubaSample toSample() override;

 private:
  const struct CommLogData* logMetaData{};
  const uint64_t commHash = 0xfaceb00c12345678;
  const std::string commDesc = "undefined";
  const int rank = -1;
  const int nRanks = -1;
  const std::string callsite;
  const std::string use;
  int64_t iteration = -1;
  uintptr_t memoryAddr{};
  std::optional<int64_t> bytes;
  std::optional<int> numSegments;
  std::optional<int64_t> durationUs;
  bool isRegMemEvent = false;
};
