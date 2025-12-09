// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#ifndef COLL_TRACE_H
#define COLL_TRACE_H

#include <folly/Synchronized.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <ratio>
#include <string>
#include <thread>
#include <unordered_map>

#include "checks.h"
#include "debug.h"
#include "info.h"
#include "meta/colltrace/CollTraceColl.h"
#include "meta/colltrace/CollTraceEvent.h"
#include "meta/logger/EventMgr.h"
#include "nccl.h"
#include "nccl_common.h"

struct CudaStreamDeleter {
  void operator()(cudaStream_t e) {
    // Ignore error at destroy
    cudaStreamDestroy(e);
  }
};
using CudaStreamPtr = std::unique_ptr<
    std::pointer_traits<cudaStream_t>::element_type,
    CudaStreamDeleter>;

class SlowCollReporter {
 public:
  using DurationOpt = std::optional<std::chrono::milliseconds>;

  SlowCollReporter(CommLogData logMetaData);
  bool shouldReportColl(const CollTraceColl& coll);

  static DurationOpt getSlowThreshold(const std::string& pgName);
  // Made public for testing purpose. DO NOT directly call this function.
  static void initThresholdMap();

 private:
  static std::unordered_map<std::string, DurationOpt> pgPrefixToSlowThreshold_;
  static std::once_flag slowThresholdMapInitFlag_;

  DurationOpt slowThreshold_{std::nullopt};
};

struct ncclComm;

// Class for colltrace
class CollTrace {
 public:
  CollTrace(ncclComm* comm);
  ~CollTrace();

  enum class CurrentCollState {
    PENDING,
    WAIT_START,
    IN_PROGRESS,
    DONE,
  };

  struct Dump {
    std::deque<CollTraceColl> pastColls;
    std::deque<CollTraceColl> pendingColls;
    std::unique_ptr<CollTraceColl> currentColl;
  };

 private:
  // Work queue data structure
  class EventQueue {
   private:
    std::deque<std::unique_ptr<CollTraceEvent>> queue_;
    std::condition_variable cv_;
    mutable std::mutex mutex_;

   public:
    std::deque<CollTraceColl> dumpQueue() const {
      std::deque<CollTraceColl> tmp{};
      {
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& item : queue_) {
          // copy content of coll within each event
          tmp.emplace_back(item->coll);
        }
      }
      return tmp;
    }

    void push(std::unique_ptr<CollTraceEvent> item) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push_back(std::move(item));
      }
      cv_.notify_one();
    }

    bool isEmpty() {
      std::lock_guard<std::mutex> lock(mutex_);
      return queue_.empty();
    }

    std::unique_ptr<CollTraceEvent> waitPop() {
      std::unique_lock<std::mutex> lock(mutex_);
      if (queue_.empty()) {
        cv_.wait(lock, [this] { return !queue_.empty(); });
      }
      std::unique_ptr<CollTraceEvent> item = std::move(queue_.front());
      queue_.pop_front();

      return item;
    }
  };

 private:
  static ncclResult_t recordReferenceEvent();

  // Used to build a correlation between CPU and GPU timestamps, so we can use
  // it to calculate the latency in the future.
  static CudaStreamPtr referenceStream_;
  static CudaEventPtr referenceEvent_;
  static std::once_flag referenceInitFlag_;
  static std::chrono::system_clock::time_point referenceTime_;

  bool loggedTimeError_{false};
  // cudaEvent pool to avoid cudaEvent destory during run and enable reuse.
  SharedPool cudaEventPool_;
  EventQueue eventQueue_;

  std::unique_ptr<CollTraceEvent> curEvent_;
  std::atomic<CurrentCollState> curCollState_{CurrentCollState::PENDING};
  std::deque<std::unique_ptr<CollTraceColl>> pastColls_;
  // Lock changes from worker thread to curEvent_, eventQueue_ and pastColls_
  mutable std::mutex workerMutex_;
  std::chrono::time_point<std::chrono::high_resolution_clock> lastStopTime_;

  // To aggregate P2P info for baseline. This is the ranks for current
  // SUBMITTING collective, not the running collective.
  folly::Synchronized<std::unordered_set<int>> p2pCurRanks;

  // For testing purpose
  std::atomic<bool> waitingForQueueEmpty_;
  std::mutex waitQueueEmptyMutex_;
  std::condition_variable waitQueueEmptyCv_;

  // Comm should only be used to identify the communicator we are in.
  // E.g. INFO("comm %p", comm_)
  // For logging the metadata of the communicator, always use logMetaData_
  ncclComm* comm_{nullptr};
  // In some rare occasion, profilingWorkerThread_ might outlive comm. So we
  // cache CommLogData to ensure all the logging/scuba works properly
  const CommLogData logMetaData_;

  // For reporting slow colls
  SlowCollReporter slowCollReporter_;

  std::thread profilingWorkerThread_;

 public:
  enum Features {
    VERBOSE = 1 << 0,
    TRACE_MODE = 1 << 1, // Need mode prefix to prevent shadowing TRACE logging
  };
  int features{0}; // bitwise OR of Features

  CollTrace::Dump dump() const;

  void resetPastColls();

  void* collTraceThreadFn(int cudaDev);

  // Create a CollTraceEvent object and assign cuda events from pool if it is a
  // COMM event.
  std::unique_ptr<CollTraceEvent> createEvent(
      CollTraceEvent::EventType type = CollTraceEvent::EventType::COMM);

  void enqueueEvent(std::unique_ptr<CollTraceEvent> event);

  void waitForWorkerFinishQueue();

  void addPeerForP2P(int peer);

  void recordCurCollResult(float latency);

  void conditionalReportColl(const CollTraceColl& coll);

  cudaError_t waitEventFinish(cudaEvent_t event);

  std::chrono::time_point<std::chrono::system_clock> getEventTime(
      CollWaitEvent* event);

  std::vector<int> getRanksForCurGroup();
};

#endif
