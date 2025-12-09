// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// This allows us to keep track of errors that affect the entire process,
// not necessarily a specific communicator
class ProcessGlobalErrorsUtil {
 public:
  struct ErrorAndStackTrace {
    // timestamp this error was reported
    std::chrono::milliseconds timestampMs{};
    std::string errorMessage;
    std::vector<std::string> stackTrace;
  };

  struct NicError {
    // timestamp this error was reported
    std::chrono::milliseconds timestampMs{};
    std::string errorMessage;
  };

  struct State {
    // Map of device name -> port -> error message
    std::unordered_map<std::string, std::unordered_map<int, NicError>> badNics;
    std::deque<ErrorAndStackTrace> errorAndStackTraces;
  };

  // Report an error on a NIC. If errorMessage is std::nullopt, then
  // the error is cleared.
  static void setNic(
      const std::string& devName,
      int port,
      std::optional<std::string> errorMessage);

  // Report an internal error and stack trace
  static void addErrorAndStackTrace(
      std::string errorMessage,
      std::vector<std::string> stackTrace);

  static State getAllState();
};
