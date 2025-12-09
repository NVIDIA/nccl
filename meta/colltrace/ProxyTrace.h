// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "TraceUtils.h"
#include "debug.h"
#include "info.h"
#include "nccl_common.h"

struct ProxyTraceCollInfo {
  uint64_t commHash{0};
  uint64_t opCount{UINT64_MAX};
  int nChannels{0};
  ncclFunc_t coll{ncclFuncBroadcast};
};

enum ProxyOpStepStatus {
  POSTED,
  REM_FIFO_WAIT,
  RECEIVED,
  TRANSMITTED,
  DONE,
  NUM_STATUS,
};

struct ProxyTraceColl {
  ProxyTraceCollInfo collInfo;
  int nProxyOps{0};
  size_t totalSendSize{0};
  // NOTE: totalRecvSize can be inaccurate when multiple requests are used at a
  // receive op; use it only to distingush send/recv v.s. sendrecv
  size_t totalRecvSize{0};
  std::unordered_set<int> channelIds;

  // serialize the entry to a json format string
  std::string serialize(bool quoted = false);
};

// record progress state per comm per collective per proxyOp
struct ProxyTraceOp {
  ProxyTraceCollInfo collInfo;

  // Op level fields
  int channelId{-1};
  int proxyOpId{0};
  int nSteps{0};
  // ranks in current communicator;
  // rank may be from another one on the node
  int rank{-1};
  int remoteRank{-1};
  size_t stepSize;
  size_t transSize{0};

  enum OpType { SEND, RECV };
  OpType opType{SEND};

  std::chrono::time_point<std::chrono::high_resolution_clock> startTs{};
  std::chrono::time_point<std::chrono::high_resolution_clock> doneTs{};
  bool done{false};

  // Step level fields
  struct StepRecord {
    int step{0};
    std::chrono::time_point<std::chrono::high_resolution_clock> ts{};
    // serialize the stepRecord to a json format string
    std::string serialize(bool quoted = false);
  };
  StepRecord stepRecords[NUM_STATUS];

  // serialize the entry to a json format string
  std::string serialize(bool quoted = false);
};

struct ncclProxyArgs;
struct ncclComm;

using ProxyActiveOpMap = std::unordered_map<
    uint64_t /* commHash*/,
    std::unordered_map<
        uint64_t /* opCount*/,
        /* proxyOpId : op */
        std::unordered_map<int, std::unique_ptr<ProxyTraceOp>>>>;

// Hacky way to figure out whether it is a send|recv|sendrecv in grouped P2P
using ProxyActiveCollMap = std::unordered_map<
    uint64_t /* commHash*/,
    std::unordered_map<uint64_t /* opCount*/, std::unique_ptr<ProxyTraceColl>>>;

using ProxyPastOpMap = std::unordered_map<
    uint64_t /* commHash*/,
    std::unordered_map<
        uint64_t /* opCount*/,
        /* list of past ops of current active collective in completion order */
        std::deque<std::unique_ptr<ProxyTraceOp>>>>;

using ProxyPastCollMap = std::unordered_map<
    uint64_t /* commHash*/,
    /* list of past collectives in completion order */
    std::deque<std::unique_ptr<ProxyTraceColl>>>;

class ProxyTrace {
 public:
  ProxyTrace();
  ~ProxyTrace(){};

  // Record when starts a send operation on proxy thread (see sendProxyProgress)
  ncclResult_t startSend(struct ncclProxyArgs* args);

  // Record when completes a send operation on proxy thread (see
  // sendProxyProgress)
  ncclResult_t completeSend(struct ncclProxyArgs* args);

  // Record internal step and timestamp for an ongoing send operation (see
  // sendProxyProgress)
  ncclResult_t recordSendProgress(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      ProxyOpStepStatus status);

  // Record when starts a recv operation on proxy thread (see recvProxyProgress)
  ncclResult_t startRecv(struct ncclProxyArgs* args);

  // Record when completes a recv operation on proxy thread (see
  // recvProxyProgress)
  ncclResult_t completeRecv(struct ncclProxyArgs* args);

  // Record internal step and timestamp for an ongoing recv operation (see
  // recvProxyProgress)
  ncclResult_t recordRecvProgress(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      ProxyOpStepStatus status);

  // print details of internal structures for both active and completed
  // send/recvs. For debugging.
  void print();

  struct Dump {
    // active ops in start time order
    std::deque<ProxyTraceOp> activeOps;
    // finished ops in current active opCount in completion time order
    std::deque<ProxyTraceOp> pastOps;
    // pastColls in completion time order
    std::deque<ProxyTraceColl> pastColls;
    // activeColls in start time order
    std::deque<ProxyTraceColl> activeColls;
  };

  // Dump all trace for a given communicator
  ProxyTrace::Dump dump(uint64_t commHash) const;

 private:
  inline ncclResult_t createActiveEntries(
      struct ncclProxyArgs* args,
      ProxyTraceOp::OpType opType);
  inline ncclResult_t completeTraceEntries(
      struct ncclProxyArgs* args,
      ProxyTraceOp::OpType opType);
  inline ncclResult_t updateTraceEntryStep(
      struct ncclProxyArgs* args,
      int sub,
      int step,
      ProxyOpStepStatus status,
      ProxyTraceOp::OpType opType);
  inline bool checkActiveCollExist(uint64_t commHash, uint64_t opCount);
  inline bool
  checkActiveOpExist(uint64_t commHash, uint64_t opCount, int proxyOpId);

  enum Features {
    TRACE = 1,
    VERBOSE = 2,
  };
  int features_{0}; // bitwise OR of Features

  mutable std::mutex mutex_;

  // Current active send/recv operations.
  // Use map to quickly find the record with commHash:opCount:proxyOpId during
  // active progress. Note that each op may not complete in order, e.g.,
  // proxyOpId 1 may finish before proxyOpId 0 if they are to different peers.
  // Thus, the inner-most layer has to still be a map for searching by
  // proxyOpId, no matter other ops are completed or not.
  ProxyActiveOpMap activeOps_;

  // Current active collective.
  // Initialized when first proxy op is created and moved to pastColls_ when all
  // ops are completed.
  ProxyActiveCollMap activeColls_;

  // Completed send/recv operations in current commHash:opCount.
  // Moved from activeOps_ when the op is complete.
  // Discarded once all ops are completed.
  ProxyPastOpMap pastOps_;

  // Completed collectives from each communicator.
  // Updated when both send and recv are completed. Allow to search by commHash
  ProxyPastCollMap pastColls_;

  friend class CollTrace;
};

// ProxyTraceArgs is used to pass arguments to proxy thread (see ncclProxyOp and
// ncclProxySubArgs in proxy.h)
struct ProxyTraceArgs {
  struct ProxyTraceCollInfo collInfo;
  int proxyOpId{-1}; // id of a proxy op in an given communicator and grouped
                     // collective/p2p (identified as commHash:opCount),
                     // assigned when creating ProxyTraceOp entry
  int rank{-1}; // op owner's rank in communicator; may submit to proxy thread
                // belonging to other local rank
  int remoteRank{-1}; // peer's rank in the communicator
  size_t transSize{0}; // size of data has been transferred
};

#define PROXY_TRACE_CALL(state, cmd) \
  do {                               \
    if (state->trace) {              \
      NCCLCHECK((cmd));              \
    }                                \
  } while (0)

#define PROXY_TRACE_OP_TO_SUBARGS(subArgs, op) \
  do {                                         \
    (subArgs)->traceArgs = (op)->traceArgs;    \
  } while (0)

// Simply execute the statement. This is just to ensure we don't forget porting
// these statements in the future.
#define PROXY_TRACE_EXECUTE(statement) \
  do {                                 \
    statement;                         \
  } while (0)
