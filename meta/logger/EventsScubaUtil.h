// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include <folly/stop_watch.h>

#include "meta/logger/ScubaLogger.h"

// Keys to use for StickyContextGuard
class ScubaContextKeys {
 public:
  // Keep in alphabetical order
  static constexpr std::string_view color{"color"};
  static constexpr std::string_view comm_id{"comm_id"};
  static constexpr std::string_view cuda_dev{"cuda_dev"};
  static constexpr std::string_view num_ranks{"num_ranks"};
  static constexpr std::string_view rank{"rank"};
};

// Static utility class for logging debugging events and profiling information
// to scuba
//
// Example usage:
//
// void foo(...) {
//   auto cg1 = EventsScubaUtil::StickyContextGuard("run_id", "ABCDEF");
//   auto cg2 = EventsScubaUtil::StickyContextGuard("pool_name", "twshared");
//   auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("a_event");
//   bar(...);
//   ... // do more work
// }
//
// void bar(...) {
//   auto cg = EventsScubaUtil::StickyContextGuard("major_types", "100-600");
//   auto sg = EVENTS_SCUBA_UTIL_SAMPLE_GUARD("b_event");
//   try {
//     ...
//   } catch (const std::exception& ex) {
//     // Include optional exception debugging info
//     sg.sample().setExceptionInfo(ex);
//   }
//
//   // Or collect the stack trace from the current point
//   sg.sample().includeStackTrace();
// }
//
// This will generate the following two samples,
// one when foo returns and one when bar returns:
//
// When bar returns:
// {
//   "context_run_id": "ABCDEF",
//   "context_pool_name": "twshared",
//   "context_major_types": "100-600",
//   "event_name_stack": ["a_event", "b_event"]
//   "event_name_stack_with_file_line": [
//     "a_event@foo@Foo.cpp:10",
//     "b_event@bar@Bar.cpp:20"
//   ],
//   "event_latency_ms": 3500,
//   "event_self_latency_ms": 3500,
//   ... // possibly more fields
// }
//
// When foo returns:
// {
//   "context_run_id": "ABCDEF",
//   "context_pool_name": "twshared",
//   "event_name_stack": ["a_event"],
//   "event_name_stack_with_file_line": ["a_event@bar@Foo.cpp:10"],
//   "event_latency_ms": 5500,
//   "event_self_latency_ms": 2000,
//   ... // possibly more fields
// }
//
// To visualize the data:
// - You can use scuba "icicle" views
// - group by stack == event_name_stack
// - order by event_self_latency_ms
//
// Note that event_latency_ms gives event e2e time including children execution
// time, and event_self_latency_ms is the event e2e time excluding children
// execution time.
class EventsScubaUtil {
 public:
  struct EventNameInfo {
    std::string eventName;
    const char* fileName{nullptr};
    int line{0};
    const char* functionName{nullptr};
    // Total time taken by children
    // Needs to be shared pointer, because every child will have
    // a copy of their parent's struct
    std::shared_ptr<std::chrono::milliseconds> childrenLatency{nullptr};
  };

  // Sticky context added by the guards
  struct Context {
    // All key value pairs for sticky context from StickyContextGuard
    std::unordered_map<std::string, std::string> keyValuePairs;
    // All events that are currently in progress from SampleGuard
    std::vector<EventNameInfo> eventNameStack;
    // stack of section guards
    std::vector<folly::stop_watch<std::chrono::microseconds>> sectionTimers;
  };

  // Get all the context that will be logged with the next sample
  static Context getAllContext();

  // TODO: use folly::ShallowCopyRequestContextScopeGuard instead
  // We can't use it for now because there is a segfault in folly thread locals
  // https://fb.workplace.com/groups/560979627394613/posts/2984688408357044
  // This will ensure that the previous context is reset on destruction
  class ShallowCopyRequestContextScopeGuard {
   public:
    ShallowCopyRequestContextScopeGuard();
    ~ShallowCopyRequestContextScopeGuard();

   private:
    Context prevContext_;
  };

  // Adds additional context. As long as this object is in scope, any samples
  // logged will include this additional context.
  // `key` will be logged as a column name to scuba
  // If `key` already exists, this will override the value with the new value,
  // and the old value will be restored on destruction.
  class StickyContextGuard {
   public:
    StickyContextGuard(std::string_view key, const std::string& value);

   private:
    ShallowCopyRequestContextScopeGuard requestContextScopeGuard_;
  };

  // RAII wrapper that will log the sample to scuba on destruction.
  // Latency is measured from when the guard is created to when it is
  // destroyed.
  //
  // If a SampleGuard is created while another SampleGuard is active, the
  // eventNames will form a stack. You can use scuba icicle views to visualize
  // the duration spent in each event and sub event.
  class SampleGuard {
   public:
    SampleGuard(
        const std::string& eventName,
        const char* fileName,
        int line,
        const char* functionName);
    ~SampleGuard();
    SampleGuard(const SampleGuard&) = delete;
    SampleGuard& operator=(const SampleGuard&) = delete;
    SampleGuard(SampleGuard&&) = delete;
    SampleGuard& operator=(SampleGuard&&) = delete;

    NcclScubaSample& sample();
    // If called, the sample will not be logged to scuba
    void dismiss();

   private:
    ShallowCopyRequestContextScopeGuard requestContextScopeGuard_;
    bool dismiss_{false};
    folly::stop_watch<std::chrono::milliseconds> timer_;
    NcclScubaSample sample_;
  };
};

// Helper macro to also include file name and line number
#define EVENTS_SCUBA_UTIL_SAMPLE_GUARD(eventName) \
  ::EventsScubaUtil::SampleGuard((eventName), __FILE__, __LINE__, __FUNCTION__)

void setNewStickyContext(EventsScubaUtil::Context stickyContext);
