// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <functional>
#include <string>

#include "nccl.h"

class ErrorStackTraceUtil {
 public:
  static ncclResult_t log(ncclResult_t result);

  // Useful if we detect an error in a function that does not return
  // ncclResult_t.
  static void logErrorMessage(std::string errorMessage);
};
