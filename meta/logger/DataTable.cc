// (c) Meta Platforms, Inc. and affiliates.

#include "meta/logger/DataTable.h"

#include <atomic>
#include <filesystem>
#include <optional>
#include <iostream>
#include <fmt/format.h>
#include <folly/File.h>
#include <folly/FileUtil.h>
#include <folly/system/ThreadName.h>

#include "meta/RankUtils.h"
#include "meta/logger/StrUtils.h"
#include "meta/cvars/nccl_cvars.h"

namespace {

struct JobFields {
  std::string jobName;
  int64_t jobVersion{0};
  int64_t jobAttempt{0};
  int64_t jobQuorumRestartId{0};
  std::string jobIdStr;
};

JobFields getJobFields() {
  JobFields jobFields;
  jobFields.jobName = getenv("SLURM_JOB_NAME") == nullptr? "" : getenv("SLURM_JOB_NAME");
  jobFields.jobVersion = 0;
  jobFields.jobAttempt = RankUtils::getInt64FromEnv("SLURM_RESTART_COUNT").value_or(0);
  jobFields.jobQuorumRestartId = -1;
  jobFields.jobIdStr = getenv("SLURM_JOB_ID") == nullptr? "" : getenv("SLURM_JOB_ID");
  if (auto arrayJobId = getenv("SLURM_ARRAY_JOB_ID");
      arrayJobId != nullptr) {
    const char* arrayTaskId = getenv("SLURM_ARRAY_TASK_ID");
    jobFields.jobIdStr = fmt::format("{}_{}", arrayJobId, arrayTaskId);
  }
  return jobFields;
}

// Fields that never change upon initialization, and should be included
// with every sample.
struct CommonFields {
  std::string hostname;
  int64_t globalRank{-1};
  int64_t worldSize{-1};
  JobFields jobFields;
  std::string fastInitMode;
  std::string cluster;
};

static CommonFields kCommonFields;
std::once_flag kCommonFieldsOnceFlag;

std::string getHostname() {
  char hostname[64];
  bzero(hostname, sizeof(hostname));
  // To make sure string is null terminated when hostname exceeds
  // the buffer size pass buffer size - 1
  if (gethostname(hostname, sizeof(hostname) - 1) == 0) {
    return hostname;
  }
  // return empty string on error
  return "";
}

std::string getClusterName() {
  const char* cluster = std::getenv("SLURM_CLUSTER_NAME");
  if (cluster == nullptr) {
    cluster = std::getenv("SCENV");
    if (cluster == nullptr) {
      return "unknown";
    }
  }
  return cluster;
}

void setCommonFields() {
  kCommonFields.hostname = getHostname();
  kCommonFields.globalRank = RankUtils::getGlobalRank().value_or(-1);
  kCommonFields.worldSize = RankUtils::getWorldSize().value_or(-1);
  kCommonFields.jobFields = getJobFields();
  kCommonFields.cluster = getClusterName();
  kCommonFields.fastInitMode = "0";
}

// Every log gets a monotonically increasing sequence number.
// This allows us to identify the last sample from this rank
// (scuba query: take the MAX of this column) to see what the rank
// was doing last.
std::atomic<int64_t> kSampleSequenceNumber{0};

void addCommonFieldsToSample(NcclScubaSample& sample) {
  std::call_once(kCommonFieldsOnceFlag, setCommonFields);
  // Start of Lite Scuba Sample Fields
  sample.addInt("sequenceNumber", kSampleSequenceNumber++);
  sample.addNormal("hostname", kCommonFields.hostname);
  sample.addInt("globalRank", kCommonFields.globalRank);
  sample.addNormal("jobName", kCommonFields.jobFields.jobName);
  sample.addInt("jobVersion", kCommonFields.jobFields.jobVersion);
  sample.addInt("jobAttempt", kCommonFields.jobFields.jobAttempt);
  sample.addInt(
      "jobQuorumRestartId", kCommonFields.jobFields.jobQuorumRestartId);

  if (sample.getLogType() == NcclScubaSample::ScubaLogType::LITE) {
    return;
  }

  // Start of Regular Scuba Sample Fields
  sample.addInt("worldSize", kCommonFields.worldSize);
  sample.addNormal("jobIdStr", kCommonFields.jobFields.jobIdStr);
  sample.addNormal("fastinit_mode", kCommonFields.fastInitMode);
}
} // namespace

// We cannot log to scuba directly from conda. Instead, we log to a file
// and then a separate process scans the logs and uploads to scuba.
DataTable::DataTable(const std::string& tableName)
    : tableName_(tableName) {
  sink_ = std::make_unique<DataSink>(tableName);
  thread_ = std::thread([this, tableName] {
    folly::setThreadName(tableName);
    loggingFunc();
  });
}

void DataTable::shutdown() {
  // use thread joinable check as proxy for whether scuba table is active.
  if (thread_.joinable()) {
    state_.lock()->stopTriggered = true;
    cv_.notify_one();
    thread_.join();
  }
}

DataTable::~DataTable() {
  shutdown();
}

void DataTable::addSample(NcclScubaSample sample) {
  state_.lock()->samples.emplace_back(std::move(sample));
  cv_.notify_one();
}

// Wait until there are messages, or until shutdown is triggered.
// Returns State
DataTable::State DataTable::waitAndGetAllMessages() {
  auto locked = state_.lock();
  cv_.wait(locked.as_lock(), [&locked] {
    return !locked->samples.empty() || locked->stopTriggered;
  });
  State state;
  std::swap(state.samples, locked->samples);
  state.stopTriggered = locked->stopTriggered;
  return state;
}

void DataTable::loggingFunc() {
  if (!sink_) {
    return;
  }
  while (!state_.lock()->stopTriggered) {
    auto state = waitAndGetAllMessages();
    if (state.stopTriggered) {
      return;
    }
    // Log all scuba-samples to the file. We do populate common fields and
    // perform serialization here to keep the work to bare minimum when
    // sample is being submitted
    for (auto& sample : state.samples) {
      addCommonFieldsToSample(sample);
      auto message = sample.toJson();
      writeMessage(message);
    }
  }
}

void DataTable::writeMessage(const std::string& message) {
  if (sink_ && !state_.lock()->stopTriggered) {
    sink_->addRawData(tableName_, message, folly::none);
  }
}
