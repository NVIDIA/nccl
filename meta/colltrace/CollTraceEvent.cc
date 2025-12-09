// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "CollTraceEvent.h"

#include <chrono>
#include <ratio>

ncclResult_t CpuWaitEvent::waitEventFinish() {
  bool finished;
  finishedSync_.withLock(
      [&finished](const bool& curStat) { finished = curStat; });
  if (finished) {
    return ncclSuccess;
  }
  auto lockedFinishStat = finishedSync_.lock();
  cv_.wait(lockedFinishStat.as_lock(), [&lockedFinishStat]() {
    return *lockedFinishStat;
  });
  return ncclSuccess;
}

std::optional<float> CpuWaitEvent::getElapsedTimeSinceEvent(
    CollWaitEvent* start) {
  static bool warned = false;
  if (typeid(*start) != typeid(CpuWaitEvent)) {
    if (!warned) {
      WARN(
          "CollTrace: Error trying to compare %s with CpuWaitEvent. getElapsedTimeSinceEvent only supports comparing events of the same type",
          typeid(*start).name());
      warned = true;
    }
    return std::nullopt;
  }
  auto* startCpuEvent = static_cast<CpuWaitEvent*>(start);
  auto timeDuration = this->finishTime_ - startCpuEvent->finishTime_;
  return std::chrono::duration_cast<std::chrono::duration<float, std::milli>>(
             timeDuration)
      .count();
}

void CpuWaitEvent::setFinished() {
  finishTime_ = std::chrono::high_resolution_clock::now();
  finishedSync_.withLock([](bool& curStat) { curStat = true; });
  cv_.notify_all();
}

std::chrono::time_point<std::chrono::high_resolution_clock>
CpuWaitEvent::getFinishTime() {
  bool finished;
  finishedSync_.withLock(
      [&finished](const bool& curStat) { finished = curStat; });
  if (!finished) {
    WARN("CpuWaitEvent::getFinishTime() called before event finished!");
    // Return current time as a placeholder
    return std::chrono::high_resolution_clock::now();
  }
  return finishTime_;
}

std::optional<float> CudaWaitEvent::getElapsedTime(cudaEvent_t start) {
  float elapsedTime;
  auto res = cudaEventElapsedTime(&elapsedTime, start, event_.get());
  if (res != cudaSuccess) {
    return std::nullopt;
  }
  return elapsedTime;
}

ncclResult_t CudaWaitEvent::waitEventFinish() {
  if (NCCL_COLLTRACE_EVENT_BLOCKING_SYNC) {
    CUDACHECK(cudaEventSynchronize(event_.get()));
    return ncclSuccess;
  }
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS milliseconds
  auto res = cudaEventQuery(event_.get());
  while (res != cudaSuccess) {
    if (res != cudaErrorNotReady) {
      CUDACHECK(res);
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS));
    res = cudaEventQuery(event_.get());
  }
  return ncclSuccess;
}

std::optional<float> CudaWaitEvent::getElapsedTimeSinceEvent(
    CollWaitEvent* start) {
  static bool warned = false;
  if (typeid(*start) != typeid(CudaWaitEvent)) {
    if (!warned) {
      WARN(
          "CollTrace: Error trying to compare %s with CudaWaitEvent. getElapsedTimeSinceEvent only supports comparing events of the same type",
          typeid(*start).name());
      warned = true;
    }
    return std::nullopt;
  }
  auto* startCudaEvent = static_cast<CudaWaitEvent*>(start);
  return getElapsedTime(startCudaEvent->event_.get());
}
