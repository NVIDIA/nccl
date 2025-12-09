// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <unistd.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include <folly/String.h>

#include "CollTrace.h"
#include "CollTraceUtils.h"
#include "bootstrap.h"
#include "comm.h"
#include "meta/colltrace/TraceUtils.h"
#include "meta/logger/ScubaLogger.h"
#include "nccl.h"

#include "meta/cvars/nccl_cvars.h"

using namespace ncclx::colltrace;

std::unordered_map<std::string, std::optional<std::chrono::milliseconds>>
    SlowCollReporter::pgPrefixToSlowThreshold_{};
std::once_flag SlowCollReporter::slowThresholdMapInitFlag_{};

// Initialize the threshold map using the value of
// NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG
void SlowCollReporter::initThresholdMap() {
  pgPrefixToSlowThreshold_.clear();

  for (const auto& pgThresholdPairStr :
       NCCL_COLLTRACE_SLOW_COLL_THRESHOLD_BY_PG) {
    std::vector<std::string> pgThresholdPair;
    folly::split(":", pgThresholdPairStr, pgThresholdPair);
    if (pgThresholdPair.size() != 2) {
      WARN(
          "SlowCollReporter Init: Invalid PG threshold pair: %s",
          pgThresholdPairStr.c_str());
      continue;
    }
    std::string pgPrefix = pgThresholdPair[0];
    std::string thresholdStr = pgThresholdPair[1];
    auto thresholdInt = folly::tryTo<int>(thresholdStr);
    if (!thresholdInt.hasValue()) {
      WARN(
          "SlowCollReporter Init: Invalid threshold: %s", thresholdStr.c_str());
      continue;
    }
    if (thresholdInt.value() < 0) {
      pgPrefixToSlowThreshold_[pgPrefix] = std::nullopt;
      INFO(
          NCCL_INIT,
          "SlowCollReporter Init: Not reporting collective for PG Prefix %s",
          pgPrefix.c_str());
    } else {
      pgPrefixToSlowThreshold_[pgPrefix] =
          std::chrono::milliseconds(thresholdInt.value());
      INFO(
          NCCL_INIT,
          "SlowCollReporter Init: Set threshold for PG Prefix %s to be %dms",
          pgPrefix.c_str(),
          thresholdInt.value());
    }
  }
}

SlowCollReporter::DurationOpt SlowCollReporter::getSlowThreshold(
    const std::string& pgName) {
  for (int i = pgName.size(); i > 0; i--) { // find the longest prefix match
    const auto pgPrefix = pgName.substr(0, i);
    if (pgPrefixToSlowThreshold_.contains(pgPrefix)) {
      return pgPrefixToSlowThreshold_[pgPrefix];
    }
  }

  if (pgPrefixToSlowThreshold_.contains("ANY")) {
    return pgPrefixToSlowThreshold_["ANY"];
  }
  return std::nullopt;
}

SlowCollReporter::SlowCollReporter(CommLogData logMetaData) {
  std::call_once(slowThresholdMapInitFlag_, initThresholdMap);

  slowThreshold_ = getSlowThreshold(logMetaData.commDesc);
  if (slowThreshold_.has_value()) {
    INFO(
        NCCL_COLL,
        "SlowCollReporter: Found PG %s, setting its threshold to %ldms",
        logMetaData.commDesc.c_str(),
        slowThreshold_.value().count());
  }
}

bool SlowCollReporter::shouldReportColl(const CollTraceColl& coll) {
  if (slowThreshold_ == std::nullopt) {
    return false;
  }
  if (coll.latency < 0) {
    return false;
  }
  auto latencyMs = std::chrono::duration<float, std::milli>(coll.latency);
  return latencyMs >= slowThreshold_.value();
}

CudaStreamPtr CollTrace::referenceStream_{nullptr};
CudaEventPtr CollTrace::referenceEvent_{nullptr};
std::chrono::system_clock::time_point CollTrace::referenceTime_{};
std::once_flag CollTrace::referenceInitFlag_{};

CollTrace::CollTrace(ncclComm* comm)
    : comm_(comm),
      logMetaData_(comm->logMetaData),
      slowCollReporter_(SlowCollReporter(comm->logMetaData)) {
  std::vector<std::string> enabledFeatures;
  if (!NCCL_COLLTRACE.empty()) {
    for (auto& f : NCCL_COLLTRACE) {
      if (f == "verbose") {
        features |= CollTrace::Features::VERBOSE;
        enabledFeatures.push_back(f);
      } else if (f == "trace") {
        features |= CollTrace::Features::TRACE_MODE;
        enabledFeatures.push_back(f);
      }
    }
  }

  lastStopTime_ = std::chrono::high_resolution_clock::now();

  // Create reference event and stream once for all communicators
  std::call_once(referenceInitFlag_, CollTrace::recordReferenceEvent);

  // create worker thread
  profilingWorkerThread_ =
      std::thread{[this]() { return collTraceThreadFn(comm_->cudaDev); }};

  std::string enabledFeaturesStr = vecToStr(enabledFeatures);
  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx commDesc %s rank %d enabled features: %s - Init COMPLETE",
      comm_,
      logMetaData_.commHash,
      logMetaData_.commDesc.c_str(),
      logMetaData_.rank,
      enabledFeaturesStr.c_str());
}

CollTrace::~CollTrace() {
  try {
    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy START",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank);

    eventQueue_.push(std::unique_ptr<CollTraceEvent>(
        new CollTraceEvent(CollTraceEvent::EventType::TERMINATE)));
    if (profilingWorkerThread_.joinable()) {
      profilingWorkerThread_.join();
    }

    INFO(
        NCCL_INIT,
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy COMPLETE",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank);
  } catch (const std::exception& e) {
    WARN(
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy FAILED: %s",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank,
        e.what());
  } catch (...) {
    WARN(
        "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - Destroy FAILED: Unknown exception",
        comm_,
        logMetaData_.commHash,
        logMetaData_.commDesc.c_str(),
        logMetaData_.rank);
  }
}

CollTrace::Dump CollTrace::dump() const {
  std::lock_guard<std::mutex> lock(workerMutex_);
  CollTrace::Dump dump{};

  if (curCollState_ == CurrentCollState::IN_PROGRESS ||
      curCollState_ == CurrentCollState::WAIT_START) {
    // copy contents
    dump.currentColl =
        std::unique_ptr<CollTraceColl>(new CollTraceColl(curEvent_->coll));
  }

  dump.pendingColls = eventQueue_.dumpQueue();

  for (auto& result : pastColls_) {
    // copy contents
    dump.pastColls.emplace_back(*result);
  }
  return dump;
}

void CollTrace::resetPastColls() {
  std::lock_guard<std::mutex> lock(workerMutex_);
  pastColls_.clear();
}

void CollTrace::conditionalReportColl(const CollTraceColl& coll) {
  if (slowCollReporter_.shouldReportColl(coll)) {
    //WARN("COLLTRACE: %s taking too long to finish", coll.toString().c_str());
    reportCollToScuba("SlowColl", coll, logMetaData_);
  }

  if (coll.opCount < NCCL_COLLTRACE_REPORT_FIRST_N_COLL &&
      coll.opName != "PutNotify" && coll.opName != "WaitNotify") {
    reportCollToScuba(
        fmt::format("First{}", NCCL_COLLTRACE_REPORT_FIRST_N_COLL),
        coll,
        logMetaData_);
  }
}

void CollTrace::recordCurCollResult(float latency) {
  auto result = std::make_unique<CollTraceColl>(curEvent_->coll);
  result->latency = latency;

  if (features & CollTrace::Features::VERBOSE) {
    INFO(NCCL_COLL, "COLLTRACE: %s", result->toString().c_str());
  }

  conditionalReportColl(*result);

  std::lock_guard<std::mutex> lock(workerMutex_);
  pastColls_.push_back(std::move(result));
  while (!pastColls_.empty() &&
         pastColls_.front()->iteration <= pastColls_.back()->iteration -
                 NCCL_COLLTRACE_RECORD_MAX_ITERATIONS) {
    pastColls_.pop_front();
  }
  if (pastColls_.size() > NCCL_COLLTRACE_RECORD_MAX) {
    pastColls_.pop_front();
  }
}

void* CollTrace::collTraceThreadFn(int cudaDev) {

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - worker thread STARTED",
      comm_,
      logMetaData_.commHash,
      logMetaData_.commDesc.c_str(),
      logMetaData_.rank);

  while (true) {
    curCollState_ = CurrentCollState::PENDING;

    // For testing purpose only. During testing, we want to ensure the worker
    // thread reached a steady state before dumping so that the trace dump
    // result is predictable. Otherwise the test can be flaky.
    if (waitingForQueueEmpty_ && eventQueue_.isEmpty()) {
      {
        std::unique_lock<std::mutex> lock(waitQueueEmptyMutex_);
        waitingForQueueEmpty_ = false;
      }
      waitQueueEmptyCv_.notify_all();
    }

    // We intentionally didn't hold the event queue lock till curEvent is
    // updated. That will potentially create deadlock.
    // Downside of current approach is we might miss one pending event in the
    // dump in very rare occasion. But since the worker thread haven't started
    // to wait for the event, it should be fine.
    {
      auto tmp_event = eventQueue_.waitPop();
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_ = std::move(tmp_event);
    }

    if (curEvent_->eventType == CollTraceEvent::EventType::TERMINATE) {
      break;
    } else if (curEvent_->eventType == CollTraceEvent::EventType::WAKE_UP) {
      continue;
    }
    curCollState_ = CurrentCollState::WAIT_START;

    auto ncclRes = curEvent_->start->waitEventFinish();

    auto startTs = getEventTime(curEvent_->start.get());
    {
      std::lock_guard<std::mutex> lock(workerMutex_);
      curEvent_->coll.startTs = startTs;
      curEvent_->coll.interCollTime =
          std::chrono::duration_cast<std::chrono::microseconds>(
              curEvent_->coll.startTs - lastStopTime_);
    }
    curCollState_ = CurrentCollState::IN_PROGRESS;
    ncclRes = curEvent_->stop->waitEventFinish();
    lastStopTime_ = getEventTime(curEvent_->stop.get());
    curCollState_ = CurrentCollState::DONE;
    float latency = -1;

    if (ncclRes == ncclSuccess) {
      auto latencyMaybe =
          curEvent_->stop->getElapsedTimeSinceEvent(curEvent_->start.get());
      if (latencyMaybe.has_value()) {
        latency = *latencyMaybe;
      }
    }

    recordCurCollResult(latency);
    curEvent_.reset();
  }

  INFO(
      NCCL_INIT,
      "COLLTRACE: comm %p commHash %lx commDesc %s rank %d - worker thread TERMINATE",
      comm_,
      logMetaData_.commHash,
      logMetaData_.commDesc.c_str(),
      logMetaData_.rank);
  return nullptr;
}

std::unique_ptr<CollTraceEvent> CollTrace::createEvent(
    CollTraceEvent::EventType type) {
  std::unique_ptr<CollTraceEvent> eventInfo(new CollTraceEvent);
  if (type == CollTraceEvent::EventType::COMM) {
    eventInfo->start = std::make_unique<CudaWaitEvent>(
        cudaEventPool_.takeOne(), cudaEventPool_);
    eventInfo->stop = std::make_unique<CudaWaitEvent>(
        cudaEventPool_.takeOne(), cudaEventPool_);
  } else if (type == CollTraceEvent::EventType::COMM_CPU) {
    eventInfo->start = std::make_unique<CpuWaitEvent>();
    eventInfo->stop = std::make_unique<CpuWaitEvent>();
  }
  if (!eventInfo->start || !eventInfo->stop) {
    std::unique_ptr<CollTraceEvent> nullCollTraceEvent(nullptr);
    return nullCollTraceEvent;
  }
  return eventInfo;
}

// Functions for recording P2P
void CollTrace::addPeerForP2P(int peer) {
  p2pCurRanks.withWLock([&peer](auto& rankSet) { rankSet.insert(peer); });
}

std::vector<int> CollTrace::getRanksForCurGroup() {
  return p2pCurRanks.withWLock([](auto& rankSet) {
    std::vector<int> ranks(rankSet.begin(), rankSet.end());
    rankSet.clear();
    std::ranges::sort(ranks);
    return ranks;
  });
}

void CollTrace::enqueueEvent(std::unique_ptr<CollTraceEvent> event) {
  eventQueue_.push(std::move(event));
}

void CollTrace::waitForWorkerFinishQueue() {
  std::unique_lock<std::mutex> waitLock(waitQueueEmptyMutex_);
  waitingForQueueEmpty_ = true;
  eventQueue_.push(std::unique_ptr<CollTraceEvent>(
      new CollTraceEvent(CollTraceEvent::EventType::WAKE_UP)));
  waitQueueEmptyCv_.wait(waitLock, [this] { return !waitingForQueueEmpty_; });
}

ncclResult_t CollTrace::recordReferenceEvent() {
  cudaStream_t stream{};
  CUDACHECK(cudaStreamCreate(&stream));
  referenceStream_ = CudaStreamPtr(stream);

  cudaEvent_t newEvent{};
  CUDACHECK(cudaEventCreate(&newEvent));
  referenceEvent_ = CudaEventPtr(newEvent);

  CUDACHECK(cudaEventRecord(newEvent, referenceStream_.get()));
  referenceTime_ = std::chrono::system_clock::now();

  return ncclSuccess;
}

cudaError_t CollTrace::waitEventFinish(cudaEvent_t event) {
  if (NCCL_COLLTRACE_EVENT_BLOCKING_SYNC) {
    return cudaEventSynchronize(event);
  }
  // async polling case, query cuda whether event is ready every
  // NCCL_COLLTRACE_CHECK_INTERVAL_MS milliseconds
  auto res = cudaEventQuery(event);
  while (res != cudaSuccess) {
    if (res != cudaErrorNotReady) {
      return res;
    }
    std::this_thread::sleep_for(
        std::chrono::milliseconds(NCCL_COLLTRACE_CHECK_INTERVAL_MS));
    res = cudaEventQuery(event);
  }
  return cudaSuccess;
}

std::chrono::time_point<std::chrono::system_clock> CollTrace::getEventTime(
    CollWaitEvent* event) {
  if (typeid(*event) == typeid(CpuWaitEvent)) {
    auto* cpuWaitEvent = dynamic_cast<CpuWaitEvent*>(event);
    return cpuWaitEvent->getFinishTime();
  } else if (typeid(*event) != typeid(CudaWaitEvent)) {
    if (!loggedTimeError_) {
      WARN("COLLTRACE: Unsupported event type %s", typeid(*event).name());
      loggedTimeError_ = true;
    }
    // Return a dummy value so that at least we are not crashing the program
    return std::chrono::high_resolution_clock::now();
  }
  auto* cudaWaitEvent = dynamic_cast<CudaWaitEvent*>(event);
  auto timeMsMaybe = cudaWaitEvent->getElapsedTime(referenceEvent_.get());

  if (timeMsMaybe == std::nullopt) {
    if (!loggedTimeError_) {
      WARN("COLLTRACE: cudaEventElapsedTime failed, all the time measurement will be meaningless.");
      loggedTimeError_ = true;
    }
    // Return a dummy value so that at least we are not crashing the program
    return std::chrono::high_resolution_clock::now();
  }

  return referenceTime_ + std::chrono::microseconds(long(*timeMsMaybe * 1000));
}
