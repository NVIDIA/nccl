// (c) Meta Platforms, Inc. and affiliates.

#include "meta/logger/NcclScubaSample.h"

#include <sstream>

#include <folly/debugging/symbolizer/Symbolizer.h>
#include <folly/json/json.h>

#include "meta/cvars/nccl_cvars.h"

// Format uses these keys:
// fbcode/rfe/scubadata/ScubaDataSample.cpp
NcclScubaSample::NcclScubaSample(std::string type, ScubaLogType logType)
    : logType_(logType), sample_(folly::dynamic::object()) {
  sample_["int"] = folly::dynamic::object;
  sample_["normal"] = folly::dynamic::object;
  sample_["normvector"] = folly::dynamic::object;
  sample_["tags"] = folly::dynamic::object;
  sample_["double"] = folly::dynamic::object;
  sample_["normal"]["type"] = std::move(type);
}

NcclScubaSample::ScubaLogType NcclScubaSample::getLogType() {
  return logType_;
}

void NcclScubaSample::addNormal(const std::string& key, std::string value) {
  sample_["normal"][key] = std::move(value);
}

void NcclScubaSample::addInt(const std::string& key, int64_t value) {
  sample_["int"][key] = value;
}

void NcclScubaSample::addDouble(const std::string& key, double value) {
  sample_["double"][key] = value;
}

void NcclScubaSample::addNormVector(
    const std::string& key,
    std::vector<std::string> value) {
  sample_["normvector"][key] =
      folly::dynamic::array(value.begin(), value.end());
}

void NcclScubaSample::addTagSet(
    const std::string& key,
    const std::set<std::string>& value) {
  sample_["tags"][key] = folly::dynamic::array(value.begin(), value.end());
}

std::string NcclScubaSample::toJson() const {
  return folly::toJson(sample_);
}

void NcclScubaSample::setExceptionInfo(const std::exception& ex) {
  setError(folly::exceptionStr(ex).toStdString());
}

void NcclScubaSample::setError(const std::string& error) {
  // Get stack trace
  if (NCCL_SCUBA_STACK_TRACE_ON_ERROR_ENABLED) {
    std::stringstream ss;
    ss << folly::symbolizer::getStackTraceStr();
    std::vector<std::string> stackTraceMangled;
    // @lint-ignore CLANGTIDY
    folly::split('\n', ss.str(), stackTraceMangled);
    for (auto& line : stackTraceMangled) {
      auto demangledLine = folly::demangle(line.c_str()).toStdString();
      line.swap(demangledLine);
    }
    this->stackTrace = stackTraceMangled;
    addNormVector("stack_trace", std::move(stackTraceMangled));
  }

  // Set attributes locally
  this->hasException = true;
  this->exceptionMessage = error;

  // Add attributes to underlying sample
  addInt("exception_set", 1);
  addNormal("exception_message", error);
}

void NcclScubaSample::setData(std::string data) {
  addNormal("event_data", std::move(data));
}

void NcclScubaSample::setExecResult(std::string result) {
  // TODO: We should change the field name to "exec_result" to be more generic
  // but this will break existing dashboards, so we might need more testing
  // before we can do that.
  addNormal("nccl_result", std::move(result));
}

void NcclScubaSample::setCommunicatorMetadata(const CommLogData* commMetadata) {
  if (commMetadata == nullptr) {
    return;
  }

  addInt("rank", commMetadata->rank);
  addInt("nRanks", commMetadata->nRanks);
  addInt("commId", commMetadata->commId);
  addInt("commHash", commMetadata->commHash);
  addNormal("commDesc", commMetadata->commDesc);
}
