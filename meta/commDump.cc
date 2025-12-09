// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <fmt/core.h>
#include <iterator>
#include <string>
#include <unordered_map>

#include <folly/json/dynamic.h>
#include <folly/json/json.h>

//#include "ExtUtils.h"
#include "comm.h"
#include "meta/colltrace/CollTrace.h"
#include "meta/colltrace/ProxyTrace.h"
#include "meta/colltrace/TraceUtils.h"
#include "meta/logger/ProcessGlobalErrorsUtil.h"
#include "nccl.h"

static void dumpCommInfo(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  map["commHash"] = toQuotedString(hashToHexStr(comm->commHash));
  map["rank"] = std::to_string(comm->rank);
  map["localRank"] = std::to_string(comm->localRank);
  map["node"] = std::to_string(comm->node);

  map["nRanks"] = std::to_string(comm->nRanks);
  map["localRanks"] = std::to_string(comm->localRanks);
  map["nNodes"] = std::to_string(comm->nNodes);
  map["commDesc"] = toQuotedString(comm->config.commDesc);
}

static void dumpCollTrace(
    const CollTrace* collTrace,
    std::unordered_map<std::string, std::string>& map) {
  if (collTrace != nullptr) {
    auto dump = collTrace->dump();

    INFO(
        NCCL_ALL,
        "CommDump: COLLTRACE dump: %zu past, %zu pending, %d current collective records",
        dump.pastColls.size(),
        dump.pendingColls.size(),
        dump.currentColl == nullptr ? 0 : 1);

    map["CT_pastColls"] = serializeObjects(dump.pastColls);
    map["CT_pendingColls"] = serializeObjects(dump.pendingColls);

    if (dump.currentColl != nullptr) {
      map["CT_currentColl"] = dump.currentColl->serialize(true);
    } else {
      map["CT_currentColl"] = "null";
    }
  } else {
    INFO(NCCL_ALL, "CommDump: COLLTRACE is disabled. No trace to dump");
  }
}

static void dumpProxyTrace(
    const ProxyTrace* ProxyTrace,
    uint64_t commHash,
    std::unordered_map<std::string, std::string>& map) {
  if (ProxyTrace) {
    auto dump = ProxyTrace->dump(commHash);

    INFO(
        NCCL_ALL,
        "CommDump: PROXYTRACE dump: %zu past collectives, %zu active network operations",
        dump.pastColls.size(),
        dump.activeOps.size());

    map["PT_pastColls"] = serializeObjects(dump.pastColls);
    map["PT_activeOps"] = serializeObjects(dump.activeOps);
    map["PT_activeColls"] = serializeObjects(dump.activeColls);
  } else {
    INFO(NCCL_ALL, "CommDump: PROXYTRACE is disabled. No trace to dump");
  }
}

__attribute__((visibility("default"))) ncclResult_t ncclCommDump(
    const ncclComm_t comm,
    std::unordered_map<std::string, std::string>& map) {
  if (comm != nullptr) {
    INFO(
        NCCL_ALL,
        "ncclCommDump by comm: rank %d comm %p commHash %lx commDesc %s",
        comm->rank,
        comm,
        comm->commHash,
        comm->config.commDesc);

    dumpCommInfo(comm, map);
    dumpCollTrace(comm->collTrace.get(), map);
    if (comm->proxyState != nullptr) {
      dumpProxyTrace(comm->proxyState->trace.get(), comm->commHash, map);
    }
  }

  return ncclSuccess;
}

__attribute__((visibility("default"))) ncclResult_t
ncclCommDumpAll(std::unordered_map<
                std::string,
                std::unordered_map<std::string, std::string>>& map) {
  return ncclSuccess;
}
