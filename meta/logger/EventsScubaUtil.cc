// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/logger/EventsScubaUtil.h"

#include <fmt/core.h>
#include <folly/MapUtil.h>

thread_local EventsScubaUtil::Context kEventContext;

void setNewStickyContext(EventsScubaUtil::Context stickyContext) {
  kEventContext = std::move(stickyContext);
}

namespace {
/*
// TODO: use folly::RequestContext instead
// We can't use it for now because there is a segfault in folly thread locals
// https://fb.workplace.com/groups/560979627394613/posts/2984688408357044

const folly::RequestToken& getRequestToken() noexcept {
  static folly::RequestToken token("ncclx_events_scuba_util_token");
  return token;
}

using StickyContextRequestData =
    folly::ImmutableRequestData<EventsScubaUtil::Context>;

EventsScubaUtil::Context getCurrentStickyContext() {
  auto requestContext = folly::RequestContext::get();
  if (requestContext == nullptr) {
    // If there is no request context, then sticky context is empty
    return EventsScubaUtil::Context{};
  }
  // Try to get the sticky context, if it exists
  if (auto requestData = dynamic_cast<StickyContextRequestData*>(
          requestContext->getContextData(getRequestToken()));
      requestData != nullptr) {
    return requestData->value();
  }
  // If it does not exist, then the old sticky context is empty
  return EventsScubaUtil::Context{};
}

void setNewStickyContext(EventsScubaUtil::Context stickyContext) {
  auto requestData =
      std::make_unique<StickyContextRequestData>(std::move(stickyContext));
  folly::RequestContext::get()->overwriteContextData(
      getRequestToken(), std::move(requestData));
}
*/

EventsScubaUtil::Context getCurrentStickyContext() {
  return kEventContext;
}

} // namespace

EventsScubaUtil::ShallowCopyRequestContextScopeGuard::
    ShallowCopyRequestContextScopeGuard() {
  prevContext_ = getCurrentStickyContext();
}

EventsScubaUtil::ShallowCopyRequestContextScopeGuard::
    ~ShallowCopyRequestContextScopeGuard() {
  setNewStickyContext(std::move(prevContext_));
}

EventsScubaUtil::StickyContextGuard::StickyContextGuard(
    std::string_view key,
    const std::string& value) {
  auto context = getCurrentStickyContext();
  // Add the new key value pair to the sticky context
  context.keyValuePairs[std::string(key)] = value;
  setNewStickyContext(std::move(context));
}

EventsScubaUtil::SampleGuard::SampleGuard(
    const std::string& eventName,
    const char* fileName,
    int line,
    const char* functionName)
    : sample_(eventName) {
  auto context = getCurrentStickyContext();
  // Add the new event name to the stack
  context.eventNameStack.push_back(EventNameInfo{
      .eventName = eventName,
      .fileName = fileName,
      .line = line,
      .functionName = functionName,
      .childrenLatency = std::make_shared<std::chrono::milliseconds>(0),
  });
  setNewStickyContext(std::move(context));
}

EventsScubaUtil::SampleGuard::~SampleGuard() {
  if (dismiss_) {
    return;
  }

  auto context = getCurrentStickyContext();

  // Total time, including children
  auto latency = timer_.lap();

  // Duration for this event
  sample_.addInt("event_latency_ms", latency.count());

  // The current event should always be the last item on the stack
  auto childrenLatency = context.eventNameStack.empty()
      ? std::chrono::milliseconds(0)
      : *context.eventNameStack.back().childrenLatency;

  // Time for this event, excluding children
  auto selfLatency = latency - childrenLatency;

  sample_.addInt("event_self_latency_ms", selfLatency.count());

  // Add the total time latency to the parent event
  if (context.eventNameStack.size() >= 2) {
    auto it = context.eventNameStack.end() - 2;
    *it->childrenLatency += latency;
  }

  // Prefix all context keys with "context_"
  for (auto&& [key, value] : context.keyValuePairs) {
    sample_.addNormal(fmt::format("context_{}", key), std::move(value));
  }

  // Event name stack
  std::vector<std::string> eventNames;
  std::vector<std::string> eventNamesWithFileLine;
  for (const auto& eventNameInfo : context.eventNameStack) {
    eventNames.push_back(eventNameInfo.eventName);
    eventNamesWithFileLine.push_back(fmt::format(
        "{}@{}@{}:{}",
        eventNameInfo.eventName,
        eventNameInfo.functionName,
        eventNameInfo.fileName,
        eventNameInfo.line));
  }
  sample_.addNormVector("event_name_stack", std::move(eventNames));
  sample_.addNormVector(
      "event_name_stack_with_file_line", std::move(eventNamesWithFileLine));

  const auto scubaTablePtr = SCUBA_nccl_structured_logging_ptr.get();
  // In the rare event of failure during initialization, scubaTablePtr can be
  // null.
  if (scubaTablePtr != nullptr) {
    scubaTablePtr->addSample(std::move(sample_));
  }
}

EventsScubaUtil::Context EventsScubaUtil::getAllContext() {
  return getCurrentStickyContext();
}

NcclScubaSample& EventsScubaUtil::SampleGuard::sample() {
  return sample_;
}

void EventsScubaUtil::SampleGuard::dismiss() {
  dismiss_ = true;
}
