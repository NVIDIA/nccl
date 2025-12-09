// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ErrorStackTraceUtil.h"

#include <folly/ExceptionString.h>
#include <folly/debugging/exception_tracer/SmartExceptionTracer.h>
#include <folly/logging/xlog.h>

#include "EventsScubaUtil.h"
#include "ProcessGlobalErrorsUtil.h"
#include "checks.h"
#include "debug.h"

namespace {
void addToProcessGlobalErrors(EventsScubaUtil::SampleGuard& sampleGuard) {
  const auto& sample = sampleGuard.sample();
  ProcessGlobalErrorsUtil::addErrorAndStackTrace(
      sample.exceptionMessage, sample.stackTrace);
}
} // namespace

/* static */
ncclResult_t ErrorStackTraceUtil::log(ncclResult_t result) {
  logErrorMessage(ncclCodeToString(result));
  return result;
}

/* static */
void ErrorStackTraceUtil::logErrorMessage(std::string errorMessage) {
  auto sampleGuard = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("ERROR");
  sampleGuard.sample().setError(errorMessage);
  addToProcessGlobalErrors(sampleGuard);
}
