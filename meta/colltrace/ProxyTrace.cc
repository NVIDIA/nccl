// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "ProxyTrace.h"

#include <chrono>
#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>

#include "comm.h"
#include "debug.h"
#include "proxy.h"

#include "meta/cvars/nccl_cvars.h"

static std::map<ProxyOpStepStatus, std::string> proxySendStepStatusStrMap = {
    {ProxyOpStepStatus::POSTED, "POSTED"},
    {ProxyOpStepStatus::REM_FIFO_WAIT, "REM_FIFO_WAIT"},
    {ProxyOpStepStatus::TRANSMITTED, "TRANSMITTED"},
    {ProxyOpStepStatus::DONE, "DONE"},
};
static std::map<ProxyOpStepStatus, std::string> proxyRecvStepStatusStrMap = {
    {ProxyOpStepStatus::POSTED, "POSTED"},
    {ProxyOpStepStatus::RECEIVED, "RECEIVED"},
    {ProxyOpStepStatus::TRANSMITTED, "TRANSMITTED"},
    {ProxyOpStepStatus::DONE, "DONE"},
};

static std::unordered_map<ProxyTraceOp::OpType, std::string> proxyOpTypetrMap =
    {
        {ProxyTraceOp::OpType::SEND, "SEND"},
        {ProxyTraceOp::OpType::RECV, "RECV"},
};

// Do not use ncclFuncStr because it doesn't include P2P
static std::unordered_map<ncclFunc_t, std::string> proxyCollStr = {
    {ncclFuncBroadcast, "Broadcast"},
    {ncclFuncReduce, "Reduce"},
    {ncclFuncAllGather, "AllGather"},
    {ncclFuncReduceScatter, "ReduceScatter"},
    {ncclFuncAllReduce, "AllReduce"},
    {ncclFuncSendRecv, "SendRecv"},
    {ncclFuncSend, "Send"},
    {ncclFuncRecv, "Recv"},
};

ProxyTrace::ProxyTrace() {
  std::vector<std::string> enabledFeatures;
  for (auto& f : NCCL_PROXYTRACE) {
    if (f == "verbose") {
      features_ |= ProxyTrace::Features::VERBOSE;
      enabledFeatures.push_back(f);
    } else if (f == "trace") {
      features_ |= ProxyTrace::Features::TRACE;
      enabledFeatures.push_back(f);
    }
  }

  std::string enabledFeaturesStr = vecToStr(enabledFeatures);
  INFO(
      NCCL_INIT,
      "PROXYTRACE: initialized with features: %s",
      enabledFeaturesStr.c_str());
}

// Check if a given commHash:opCount:proxyOpId exists in activeMap.
// Return true if it exists, false otherwise.
inline bool ProxyTrace::checkActiveOpExist(
    uint64_t commHash,
    uint64_t opCount,
    int proxyOpId) {
  return (
      activeOps_.find(commHash) != activeOps_.end() &&
      activeOps_[commHash].find(opCount) != activeOps_[commHash].end() &&
      activeOps_[commHash][opCount].find(proxyOpId) !=
          activeOps_[commHash][opCount].end());
}

inline bool ProxyTrace::checkActiveCollExist(
    uint64_t commHash,
    uint64_t opCount) {
  return (
      activeColls_.find(commHash) != activeColls_.end() &&
      activeColls_[commHash].find(opCount) != activeColls_[commHash].end());
}

inline ncclResult_t ProxyTrace::createActiveEntries(
    struct ncclProxyArgs* args,
    ProxyTraceOp::OpType opType) {
  for (int subIdx = 0; subIdx < args->nsubs; subIdx++) {
    struct ncclProxySubArgs* sub = &args->subs[subIdx];
    auto commHash = sub->traceArgs.collInfo.commHash;
    auto opCount = sub->traceArgs.collInfo.opCount;
    int proxyOpId = 0;

    // Create a collective entry for a given commHash:opCount at first proxyOp
    // and aggregate info
    // - num of belonging proxyOps
    // - totalSendSize and totalRecvSize
    // - unique channelIds
    if (!checkActiveCollExist(commHash, opCount)) {
      auto coll = std::unique_ptr<ProxyTraceColl>(new ProxyTraceColl());
      coll->collInfo = sub->traceArgs.collInfo;
      activeColls_[commHash][opCount] = std::move(coll);
    }

    // Assign proxyOpId to the sub for trace to track all ops belonging to
    // single commHash:opCount Note that channelId is always 0 in p2p case, thus
    // cannot use it to distinguish ops.
    proxyOpId = activeColls_[commHash][opCount]->nProxyOps;
    sub->traceArgs.proxyOpId = proxyOpId;

    auto entry = std::unique_ptr<ProxyTraceOp>(new ProxyTraceOp());
    entry->collInfo = sub->traceArgs.collInfo;
    entry->nSteps = sub->nsteps;
    entry->channelId = sub->channelId;
    entry->proxyOpId = proxyOpId;
    entry->rank = sub->traceArgs.rank;
    entry->remoteRank = sub->traceArgs.remoteRank;
    entry->stepSize = sub->nbytes;
    entry->startTs = std::chrono::high_resolution_clock::now();
    entry->opType = opType;

    if (features_ & ProxyTrace::Features::VERBOSE) {
      std::string entryStr = entry->serialize(false);
      INFO(
          NCCL_COLL,
          "PROXYTRACE: sub %p created entry %s",
          sub,
          entryStr.c_str());
    }
    // Append new proxyOp to a given commHash:opCount
    activeOps_[commHash][opCount][proxyOpId] = std::move(entry);

    // Update collective info
    activeColls_[commHash][opCount]->channelIds.insert(sub->channelId);
    activeColls_[commHash][opCount]->nProxyOps++;
  }
  return ncclSuccess;
}

inline ncclResult_t ProxyTrace::completeTraceEntries(
    struct ncclProxyArgs* args,
    ProxyTraceOp::OpType opType) {
  // For each completed channel, move to completed queue
  for (int subIdx = 0; subIdx < args->nsubs; subIdx++) {
    struct ncclProxySubArgs* sub = &args->subs[subIdx];
    auto proxyOpId = sub->traceArgs.proxyOpId;
    auto commHash = sub->traceArgs.collInfo.commHash;
    auto opCount = sub->traceArgs.collInfo.opCount;

    if (!checkActiveOpExist(commHash, opCount, proxyOpId)) {
      WARN(
          "PROXYTRACE: failed to complete %s entry of commHash %s opCount %lx proxyOpId %d, because no active entry exists",
          proxyOpTypetrMap[opType].c_str(),
          hashToHexStr(commHash).c_str(),
          opCount,
          proxyOpId);
      return ncclInternalError;
    }

    auto& entry = activeOps_[commHash][opCount][proxyOpId];
    entry->doneTs = std::chrono::high_resolution_clock::now();
    entry->done = true;

    if (features_ & ProxyTrace::Features::VERBOSE) {
      std::string entryStr = entry->serialize(false);
      INFO(
          NCCL_COLL,
          "PROXYTRACE: sub %p completed entry %s",
          sub,
          entryStr.c_str());
    }

    auto& coll = activeColls_[commHash][opCount];

    // Update total send/recv size once an op is completed
    if (opType == ProxyTraceOp::OpType::SEND) {
      coll->totalSendSize += entry->transSize;
    } else {
      coll->totalRecvSize += entry->transSize;
    }

    // Temporiarly move to past queue when completing a commHash:opCount
    pastOps_[commHash][opCount].push_back(std::move(entry));
    activeOps_[commHash][opCount].erase(proxyOpId);

    // Erase opCount from activeOps_ if all proxyOps have finished
    if (activeOps_[commHash][opCount].empty()) {
      activeOps_[commHash].erase(opCount);

      // Finished a full collective, move the activeColl to past queue and
      // aggregate info
      // - update coll to sendrecv if see both send and recv transmitted bytes
      auto& coll = activeColls_[commHash][opCount];
      if ((coll->collInfo.coll == ncclFuncSend ||
           coll->collInfo.coll == ncclFuncRecv) &&
          coll->totalSendSize > 0 && coll->totalRecvSize > 0) {
        coll->collInfo.coll = ncclFuncSendRecv;
      }
      // Free temporary storage for given commHash:opCount
      pastOps_[commHash][opCount].clear();
      pastOps_[commHash].erase(opCount);

      if (features_ & ProxyTrace::Features::VERBOSE) {
        INFO(
            NCCL_COLL,
            "PROXYTRACE: completed collective %s",
            coll->serialize(false).c_str());
      }

      pastColls_[commHash].push_back(std::move(coll));
      if (NCCL_PROXYTRACE_RECORD_MAX >= 0 &&
          pastColls_[commHash].size() > NCCL_PROXYTRACE_RECORD_MAX) {
        pastColls_[commHash].pop_front();
      }
      activeColls_[commHash].erase(opCount);
    }
  }
  return ncclSuccess;
}

inline ncclResult_t ProxyTrace::updateTraceEntryStep(
    struct ncclProxyArgs* args,
    int subIdx,
    int step,
    ProxyOpStepStatus status,
    ProxyTraceOp::OpType opType) {
  struct ncclProxySubArgs* sub = &args->subs[subIdx];
  auto proxyOpId = sub->traceArgs.proxyOpId;
  auto commHash = sub->traceArgs.collInfo.commHash;
  auto opCount = sub->traceArgs.collInfo.opCount;

  if (!checkActiveOpExist(commHash, opCount, proxyOpId)) {
    WARN(
        "PROXYTRACE: failed to update %s entry of commHash %s opCount %lx proxyOpId %d, because no active entry exists",
        proxyOpTypetrMap[opType].c_str(),
        hashToHexStr(commHash).c_str(),
        opCount,
        proxyOpId);
    return ncclInternalError;
  }

  auto& entry = activeOps_[commHash][opCount][proxyOpId];
  if (status == ProxyOpStepStatus::REM_FIFO_WAIT &&
      entry->stepRecords[status].step == step) {
    // Skip if a step is already in REM_FIFO_WAIT status, since we want to
    // record the first time when the step is updated to REM_FIFO_WAIT
    return ncclSuccess;
  }

  entry->stepRecords[status].step = step;
  entry->stepRecords[status].ts = std::chrono::high_resolution_clock::now();
  entry->transSize = sub->traceArgs.transSize;
  return ncclSuccess;
}

ncclResult_t ProxyTrace::startSend(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return createActiveEntries(args, ProxyTraceOp::OpType::SEND);
};

ncclResult_t ProxyTrace::completeSend(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return completeTraceEntries(args, ProxyTraceOp::OpType::SEND);
}

ncclResult_t ProxyTrace::recordSendProgress(
    struct ncclProxyArgs* args,
    int sub,
    int step,
    ProxyOpStepStatus status) {
  std::lock_guard<std::mutex> lock(mutex_);
  return updateTraceEntryStep(
      args, sub, step, status, ProxyTraceOp::OpType::SEND);
};

ncclResult_t ProxyTrace::startRecv(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return createActiveEntries(args, ProxyTraceOp::OpType::RECV);
};

ncclResult_t ProxyTrace::completeRecv(struct ncclProxyArgs* args) {
  std::lock_guard<std::mutex> lock(mutex_);
  return completeTraceEntries(args, ProxyTraceOp::OpType::RECV);
}

ncclResult_t ProxyTrace::recordRecvProgress(
    struct ncclProxyArgs* args,
    int sub,
    int step,
    ProxyOpStepStatus status) {
  std::lock_guard<std::mutex> lock(mutex_);
  return updateTraceEntryStep(
      args, sub, step, status, ProxyTraceOp::OpType::RECV);
};

static std::vector<std::string> infoKeys = {
    "commHash",
    "opCount",
    "coll",
    "channelIds",
    "totalSendSize",
    "totalRecvSize",
    "nProxyOps"};

std::string ProxyTraceColl::serialize(bool quoted) {
  std::unordered_map<std::string, std::string> map;
  map["commHash"] = quoted ? toQuotedString(hashToHexStr(collInfo.commHash))
                           : hashToHexStr(collInfo.commHash);
  map["opCount"] = std::to_string(collInfo.opCount);
  map["coll"] = quoted ? toQuotedString(proxyCollStr[collInfo.coll])
                       : proxyCollStr[collInfo.coll];
  map["channelIds"] = serializeSet(channelIds);
  map["totalSendSize"] = std::to_string(totalSendSize);
  map["totalRecvSize"] = std::to_string(totalRecvSize);
  map["nProxyOps"] = std::to_string(nProxyOps);

  return serializeMap(infoKeys, map, quoted);
}

static std::vector<std::string> entryKeys = {
    "commHash",
    "opCount",
    "coll",
    "proxyOpId",
    "channelId",
    "rank",
    "remoteRank",
    "stepSize",
    "nSteps",
    "opType",
    "status",
    "transSize",
    "startTs",
    "doneTs",
    "POSTED",
    "REM_FIFO_WAIT",
    "RECEIVED",
    "TRANSMITTED",
    "DONE"};

std::string ProxyTraceOp::serialize(bool quoted) {
  std::unordered_map<std::string, std::string> map;
  map["commHash"] = quoted ? toQuotedString(hashToHexStr(collInfo.commHash))
                           : hashToHexStr(collInfo.commHash);
  map["opCount"] = std::to_string(collInfo.opCount);
  map["coll"] = quoted ? toQuotedString(proxyCollStr[collInfo.coll])
                       : proxyCollStr[collInfo.coll];
  map["proxyOpId"] = std::to_string(proxyOpId);
  map["channelId"] = std::to_string(channelId);
  map["rank"] = std::to_string(rank);
  map["remoteRank"] = std::to_string(remoteRank);
  map["stepSize"] = std::to_string(stepSize);
  map["nSteps"] = std::to_string(nSteps);
  if (quoted) {
    map["opType"] = opType == ProxyTraceOp::OpType::SEND
        ? toQuotedString("SEND")
        : toQuotedString("RECV");
    map["status"] =
        done ? toQuotedString("DONE") : toQuotedString("IN_PROGRESS");
  } else {
    map["opType"] = opType == ProxyTraceOp::OpType::SEND ? "SEND" : "RECV";
    map["status"] = done ? "DONE" : "IN_PROGRESS";
  }
  map["transSize"] = std::to_string(transSize);
  map["startTs"] =
      std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                         startTs.time_since_epoch())
                         .count());
  if (done) {
    map["doneTs"] =
        std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                           doneTs.time_since_epoch())
                           .count());
  } else {
    map["doneTs"] = "null";
  }

  // pick status used in each OP and flatten stepRecords as STATUS:{step, ts} in
  // map
  auto& statusStrMap = opType == ProxyTraceOp::OpType::SEND
      ? proxySendStepStatusStrMap
      : proxyRecvStepStatusStrMap;
  for (auto it : statusStrMap) {
    auto status = it.first;
    auto& statusStr = it.second;
    map[statusStr] = stepRecords[status].serialize(quoted);
  }

  return serializeMap(entryKeys, map, quoted);
}

static std::vector<std::string> stepRecordKeys = {"step", "ts"};

std::string ProxyTraceOp::StepRecord::serialize(bool quoted) {
  std::unordered_map<std::string, std::string> map;
  map["step"] = std::to_string(step);
  map["ts"] = step > 0
      ? std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(
                           ts.time_since_epoch())
                           .count())
      : "null";
  return serializeMap(stepRecordKeys, map, quoted);
}

static inline void dumpActiveOps(
    uint64_t commHash,
    const ProxyActiveOpMap& activeMap,
    std::deque<ProxyTraceOp>& deq) {
  const auto it = activeMap.find(commHash);
  if (it == activeMap.end()) {
    return;
  }

  for (const auto& opCountEntry : it->second) {
    for (const auto& channelEntry : opCountEntry.second) {
      deq.emplace_back(*(channelEntry.second));
    }
  }
}

static inline void dumpActiveColls(
    uint64_t commHash,
    const ProxyActiveCollMap& activeCollsMap,
    std::deque<ProxyTraceColl>& deq) {
  const auto it = activeCollsMap.find(commHash);
  if (it == activeCollsMap.end()) {
    return;
  }

  for (const auto& collEntry : it->second) {
    deq.emplace_back(*(collEntry.second));
  }
}

static inline void dumpPastOps(
    uint64_t commHash,
    const ProxyPastOpMap& pastOpMap,
    std::deque<ProxyTraceOp>& deq) {
  const auto it = pastOpMap.find(commHash);
  if (it == pastOpMap.end()) {
    return;
  }

  for (const auto& opCountEntry : it->second) {
    for (const auto& entry : opCountEntry.second) {
      deq.emplace_back(*entry);
    }
  }
}

static inline void dumpPastColls(
    uint64_t commHash,
    const ProxyPastCollMap& pastCollsMap,
    std::deque<ProxyTraceColl>& deq) {
  const auto it = pastCollsMap.find(commHash);
  if (it == pastCollsMap.end()) {
    return;
  }

  for (const auto& pastEntry : it->second) {
    deq.emplace_back(*pastEntry);
  }
}

ProxyTrace::Dump ProxyTrace::dump(uint64_t commHash) const {
  std::lock_guard<std::mutex> lock(mutex_);
  ProxyTrace::Dump dump;

  dumpActiveOps(commHash, activeOps_, dump.activeOps);
  dumpActiveColls(commHash, activeColls_, dump.activeColls);
  dumpPastOps(commHash, pastOps_, dump.pastOps);
  dumpPastColls(commHash, pastColls_, dump.pastColls);

  return dump;
}
