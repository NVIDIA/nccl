// (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <exception>
#include <set>
#include <string>
#include <vector>

#include <folly/json/dynamic.h>

#include "meta/commSpecs.h"

// See rfe/scubadata/ScubaDataSample.h
// We can't use it here to avoid all nccl headers including this type.
// Keys are scuba column names
// Each sample must explicitly define its own type so that different types of
// events within the system can be observed in different way.
class NcclScubaSample {
 public:
  enum ScubaLogType {
    REGULAR,
    LITE // Only reports bare minimum common fields. Used for heavy duty tables
  };

  explicit NcclScubaSample(std::string type, ScubaLogType logType = REGULAR);

  // Only allow moves not copies
  NcclScubaSample(NcclScubaSample&& other) = default;
  NcclScubaSample& operator=(NcclScubaSample&& other) = default;
  ~NcclScubaSample() = default;

  ScubaLogType getLogType();

  void addNormal(const std::string& key, std::string value);
  void addInt(const std::string& key, int64_t value);
  void addDouble(const std::string& key, double value);
  void addNormVector(const std::string& key, std::vector<std::string> value);
  void addTagSet(const std::string& key, const std::set<std::string>& value);
  std::string toJson() const;

  // Helper to set exception info and collect stack traces
  void setExceptionInfo(const std::exception& ex);

  // Helper to include a custom error and collect stack trace
  void setError(const std::string& error);

  // Set custom data attribute
  void setData(std::string data);

  // Add communicator metadata details to the sample
  void setCommunicatorMetadata(const CommLogData* commMetadata);
  void setExecResult(std::string result);

  // Extra attributes for subsequent retrieval. We do so to avoid retrieval
  // from dynamic object which may pose undesired exceptions e.g. type
  // conversion if not set properly
  bool hasException{false};
  std::string exceptionMessage;
  std::vector<std::string> stackTrace;

  // explicit copy function to avoid implicit copy constructor
  NcclScubaSample makeCopy() const {
    return NcclScubaSample(*this);
  }

 private:
  NcclScubaSample(const NcclScubaSample& other) = default;
  NcclScubaSample& operator=(const NcclScubaSample& other) = default;

  ScubaLogType logType_;
  folly::dynamic sample_;
};
