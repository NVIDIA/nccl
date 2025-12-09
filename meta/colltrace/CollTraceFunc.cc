// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include "meta/colltrace/CollTraceFunc.h"

#include <chrono>
#include <vector>

#include "collectives.h"

#include "meta/colltrace/CollTraceColl.h"
#include "meta/cvars/nccl_cvars.h"

namespace ncclx::colltrace {

namespace {
CpuWaitEvent* getCpuWaitEventOrNull(CollWaitEvent* event) {
  if (event == nullptr) {
    return nullptr;
  }
  if (typeid(*event) == typeid(CpuWaitEvent)) {
    return static_cast<CpuWaitEvent*>(event);
  }
  return nullptr;
}

std::string getAlgoNameFromCollTask(const ncclTaskColl& collTask) {
  return fmt::format(
      "Baseline_{}_{}_{}",
      ncclProtoToString(collTask.protocol),
      ncclAlgoToString(collTask.algorithm),
      collTask.nMaxChannels);
}

std::string
getAlgoNameFromP2PGroup(std::string_view opName, int sendCount, int recvCount) {
  return fmt::format("Baseline_{}_S{}_R{}", opName, sendCount, recvCount);
}

CollTraceColl parseCollInfoFromP2PTasks(
    const ncclTaskP2p& p2pTaskHead,
    int myRank) {
  // Missing: opCount, comm, logMetaData, stream
  // Will add later in this func: opName, algoName, ranksInGroupedP2P
  // Missing inside BaselineAttr: everything
  // Will set in BaselineAttr: coll
  auto coll = CollTraceColl{
      .iteration = -1,
      // Currently do not add the buffer information, as it is not meaningful
      // for grouped send/recv
      .sendbuff = std::nullopt,
      .recvbuff = std::nullopt,
      .count = std::nullopt,
      // Effectively unknown type
      .dataType = ncclInt8, // we are counting bytes
      .codepath = CollTraceColl::Codepath::BASELINE,
      .baselineAttr = CollBaselineAttr{},
  };

  std::set<int> ranksInGroupedP2PSet = {};
  auto sendTaskCount = 0;
  auto recvTaskCount = 0;
  int64_t byteCount = p2pTaskHead.bytes;
  // Root stores the peer rank
  ranksInGroupedP2PSet.insert(myRank);
  ranksInGroupedP2PSet.insert(p2pTaskHead.root);
  if (p2pTaskHead.func == ncclFuncSend) {
    sendTaskCount += 1;
  } else {
    recvTaskCount += 1;
  }

  auto curP2PTask = p2pTaskHead.next;
  while (curP2PTask != nullptr) {
    if (curP2PTask->func == ncclFuncSend) {
      sendTaskCount += 1;
    } else {
      recvTaskCount += 1;
    }
    byteCount += curP2PTask->bytes;
    ranksInGroupedP2PSet.insert(curP2PTask->root);
    curP2PTask = curP2PTask->next;
  }

  if (sendTaskCount > 0 && recvTaskCount > 0) {
    coll.baselineAttr->coll = ncclFuncSendRecv;
  } else if (sendTaskCount > 0) {
    coll.baselineAttr->coll = ncclFuncSend;
  } else {
    coll.baselineAttr->coll = ncclFuncRecv;
  }
  coll.opName = std::string{ncclFuncToString(coll.baselineAttr->coll)};
  coll.algoName =
      getAlgoNameFromP2PGroup(coll.opName, sendTaskCount, recvTaskCount);

  coll.ranksInGroupedP2P = std::vector<int>{
      ranksInGroupedP2PSet.begin(), ranksInGroupedP2PSet.end()};

  coll.count = byteCount;

  return coll;
}

CollTraceColl parseCollInfoFromCollTask(const ncclTaskColl& collTask) {
  // Missing: opCount, comm, logMetaData, stream
  // Missing inside BaselineAttr: pattern, nChannels, channelId
  return CollTraceColl{
      .iteration = -1,
      .opName = std::string{ncclFuncToString(collTask.func)},
      .algoName = getAlgoNameFromCollTask(collTask),
      .sendbuff = collTask.sendbuff,
      .recvbuff = collTask.recvbuff,
      .count = collTask.count,
      .dataType = collTask.datatype,
      .codepath = CollTraceColl::Codepath::BASELINE,
      .baselineAttr =
          CollBaselineAttr{
              .coll = collTask.func,
              .algorithm = collTask.algorithm,
              .protocol = collTask.protocol,
              .op = collTask.opHost,
              .root = collTask.root,
          },
  };
}

std::optional<CollTraceColl> parseCollInfoFromNcclKernelPlan(
    ncclKernelPlan& plan,
    cudaStream_t stream) {
  auto collTaskHead = ncclIntruQueueHead(&plan.collTaskQueue);
  auto p2pTaskHead = ncclIntruQueueHead(&plan.p2pTaskQueue);
  // TODO: Limit the frequency of the logging
  if (collTaskHead == nullptr && p2pTaskHead == nullptr) {
    WARN("CollTrace: no coll or p2p task in this plan, this plan is empty");
    return std::nullopt;
  } else if (collTaskHead != nullptr && collTaskHead->next != nullptr) {
    WARN(
        "CollTrace: more than one coll task in this plan, this is currently not supported");
    return std::nullopt;
  } else if (collTaskHead != nullptr && p2pTaskHead != nullptr) {
    WARN(
        "CollTrace: both coll and p2p task in this plan, this is currently not supported");
    return std::nullopt;
  }
  CollTraceColl collInfo = collTaskHead != nullptr
      ? parseCollInfoFromCollTask(*collTaskHead)
      : parseCollInfoFromP2PTasks(*p2pTaskHead, plan.comm->rank);

  // Need to add: opCount, comm, logMetaData, stream
  collInfo.opCount = plan.comm->opCount;
  collInfo.comm = plan.comm;
  collInfo.logMetaData = plan.comm->logMetaData;
  collInfo.stream = stream;

  return collInfo;
}

} // namespace

ncclResult_t collTraceInit(ncclComm* comm) {
  if (NCCL_COLLTRACE.empty()) {
    return ncclSuccess;
  }
  comm->collTrace = std::make_unique<CollTrace>(comm);
  return ncclSuccess;
}

ncclResult_t collTraceDestroy(ncclComm* comm) {
  if (comm->collTrace == nullptr) {
    return ncclSuccess;
  }
  comm->collTrace.reset();
  return ncclSuccess;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventCommon(
    ncclComm* comm,
    CollTraceEvent::EventType type) {
  if (!comm->collTrace) {
    return nullptr;
  }
  auto event = comm->collTrace->createEvent(type);
  if (!event) {
    throw CollTraceError("Event init failed");
    return nullptr; /*Event init failed*/
  }
  return event;
}

std::unique_ptr<CollTraceEvent> collTraceAquireEventBaseline(
    ncclKernelPlan* plan,
    cudaStream_t stream) {
  auto collOpt = parseCollInfoFromNcclKernelPlan(*plan, stream);
  if (!collOpt.has_value()) {
    return nullptr;
  }
  auto comm = plan->comm;
  if (!comm->collTrace) {
    return nullptr;
  }

  auto event =
      collTraceAquireEventCommon(comm, CollTraceEvent::EventType::COMM);
  if (event == nullptr) {
    return nullptr;
  }
  event->coll = collOpt.value();
  return event;
}

ncclResult_t collTraceRecordStartEvent(
    cudaStream_t launchStream,
    CollTraceEvent* event) {
  if (event) {
    if (typeid(*event->stop) == typeid(CudaWaitEvent)) {
      CUDACHECK(cudaEventRecord(
          static_cast<CudaWaitEvent*>(event->start.get())->getCudaEvent(),
          launchStream));
    }
  }
  return ncclSuccess;
}

ncclResult_t collTraceRecordEndEvent(
    ncclComm* comm,
    ncclKernelPlan* plan,
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (comm->collTrace && event) {
    if (typeid(*event->stop) == typeid(CudaWaitEvent)) {
      CUDACHECK(cudaEventRecord(
          static_cast<CudaWaitEvent*>(event->stop.get())->getCudaEvent(),
          launchStream));
    }
    event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
    comm->collTrace->enqueueEvent(std::move(event));
  }
  return ncclSuccess;
}

ncclResult_t collTraceCtranRecordEndEvent(
    cudaStream_t launchStream,
    std::unique_ptr<CollTraceEvent> event) {
  if (!event) {
    return ncclSuccess;
  }

  if (typeid(*event->stop) == typeid(CudaWaitEvent)) {
    CUDACHECK(cudaEventRecord(
        static_cast<CudaWaitEvent*>(event->stop.get())->getCudaEvent(),
        launchStream));
  }
  event->coll.stream = launchStream;
  event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
  auto collTrace = event->coll.comm->collTrace.get();
  collTrace->enqueueEvent(std::move(event));

  return ncclSuccess;
}

ncclResult_t collTraceCtranSubmitEvent(std::unique_ptr<CollTraceEvent> event) {
  if (!event) {
    return ncclSuccess;
  }

  event->coll.enqueueTs = std::chrono::high_resolution_clock::now();
  auto collTrace = event->coll.comm->collTrace.get();
  collTrace->enqueueEvent(std::move(event));

  return ncclSuccess;
}

CpuWaitEvent* collTraceGetCpuStartWaitEvent(CollTraceEvent& event) {
  return getCpuWaitEventOrNull(event.start.get());
}

CpuWaitEvent* collTraceGetCpuEndWaitEvent(CollTraceEvent& event) {
  return getCpuWaitEventOrNull(event.stop.get());
}

} // namespace ncclx::colltrace
