/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#include <atomic>
#include <map>
#include <mutex>
#include <set>

#include <dirent.h>
#include <dlfcn.h>
#include <error.h>
#include <link.h>

#include "alloc.h"
#include "checks.h"
#include "graph/topo.h"

#include "msccl/msccl_lifecycle.h"
#include "msccl/msccl_parser.h"
#include "msccl/msccl_setup.h"
#include "msccl/msccl_status.h"

NCCL_PARAM(MscclEnabled, "MSCCL_ENABLE", 1);
static std::atomic<bool> mscclInitialized;
static bool mscclSchedulerTriedLoadAlgo = false;
static std::mutex mscclLifecycleMutex;

int getEnvInt(const char* env, int64_t deftVal) {
  char* str = getenv(env);
  int64_t value = deftVal;
  if (str && strlen(str) > 0) {
    errno = 0;
    value = strtoll(str, nullptr, 0);
  }
  return value;
}

bool mscclEnabled() {
  return ncclParamMscclEnabled();
}

void mscclSetIsCallerFlag() {
  mscclGetThreadLocalStatus().mscclIsCallerFlag = true;
}

void mscclClearIsCallerFlag() {
  mscclGetThreadLocalStatus().mscclIsCallerFlag = false;
}

bool mscclIsCaller() {
  return mscclGetThreadLocalStatus().mscclIsCallerFlag;
}

bool mscclAvailable() {
  return mscclEnabled() && mscclInitialized.load(std::memory_order_acquire);
}

static bool mscclCommCompatible(ncclComm_t comm) {
  std::map<uint64_t, std::set<uint64_t>> hostHashToPidHashes;
  for (int i = 0; i < comm->nRanks; i++) {
    uint64_t hostHash = comm->peerInfo[i].hostHash;
    uint64_t pidHash = comm->peerInfo[i].pidHash;
    if (hostHashToPidHashes.find(hostHash) != hostHashToPidHashes.end()) {
      auto& pidHashSet = hostHashToPidHashes[hostHash];
      if (pidHashSet.find(pidHash) != pidHashSet.end()) {
        return false;
      }
    }
    hostHashToPidHashes[hostHash].insert(pidHash);
  }
  return true;
}

static const char* mscclSchedulerPathEnv = "MSCCL_SCHEDULER";
static const char* mscclSchedulerDefaultPath = "libmsccl-scheduler.so";
static const char* mscclAlgoDirEnv = "MSCCL_ALGO_DIR";
static const char* mscclAlgoDefaultDir = "msccl-algorithms";
extern "C" bool mscclUnitTestMode() __attribute__((__weak__));
static const char* mscclUnitTestAlgoDefaultDir = "msccl-unit-test-algorithms";

static ncclResult_t mscclInternalSchedulerInit() {
  mscclStatus& status = mscclGetStatus();
  const char* mscclAlgoDir = getenv(mscclAlgoDirEnv);
  std::string mscclAlgoDirStr;
  if (mscclAlgoDir == nullptr) {
    // Try to find default algorithm directory based on librccl.so path
    Dl_info dl_info;
    struct link_map *link_map_ptr = nullptr;
    if (!dladdr1((void *)mscclInternalSchedulerInit, &dl_info, (void **)&link_map_ptr, RTLD_DL_LINKMAP)) {
      WARN("MSCCL Internal Scheduler: dladdr1 failed");
      return ncclInvalidUsage;
    }
    std::string selfLibPath = link_map_ptr->l_name;
    mscclAlgoDirStr = selfLibPath.substr(0, selfLibPath.find_last_of("/\\") + 1);
    mscclAlgoDirStr += (mscclUnitTestMode && mscclUnitTestMode()) ? mscclUnitTestAlgoDefaultDir : mscclAlgoDefaultDir;
    mscclAlgoDir = mscclAlgoDirStr.c_str();
  }
  struct dirent *entry = nullptr;
  DIR *dp = nullptr;
  dp = opendir(mscclAlgoDir);
  if (dp == nullptr) {
    WARN("MSCCL Internal Scheduler: open algorithm directory %s failed", mscclAlgoDir);
    return ncclInvalidUsage;
  }
  while ((entry = readdir(dp))) {
    if (entry->d_type != DT_LNK && entry->d_type != DT_REG) {
      continue;
    }
    status.algoMetas.emplace_back();
    std::string fullPath = mscclAlgoDir;
    fullPath += "/";
    fullPath += entry->d_name;
    NCCLCHECK(mscclGetAlgoMetaFromXmlFile(fullPath.c_str(), &(status.algoMetas.back())));
  }
  if (closedir(dp)) {
    WARN("MSCCL Internal Scheduler: closedir failed, error %d", errno);
    return ncclInvalidUsage;
  }
  status.rankToAlgoHandles.resize(status.algoMetas.size());
  return ncclSuccess;
}

static ncclResult_t mscclSchedulerInit() {
  mscclStatus& status = mscclGetStatus();
  bool useInternalScheduler = false;

  const char* mscclSchedulerPath = getenv(mscclSchedulerPathEnv);
  if (mscclSchedulerPath) {
    status.mscclSchedulerLib = dlopen(mscclSchedulerPath, RTLD_NOW | RTLD_LOCAL);
  } else {
    status.mscclSchedulerLib = dlopen(mscclSchedulerDefaultPath, RTLD_NOW | RTLD_LOCAL);
  }
  if (status.mscclSchedulerLib == nullptr) {
    INFO(NCCL_INIT, "MSCCL: No external scheduler found, using internal implementation");
    useInternalScheduler = true;
  } else {
    status.mscclSchedulerPtr = (mscclSchedulerInterface *)dlsym(status.mscclSchedulerLib, "mscclScheduler");
    if (status.mscclSchedulerPtr == nullptr) {
      INFO(NCCL_INIT, "MSCCL: Failed to find mscclScheduler symbol, using internal implementation");
      useInternalScheduler = true;
    }
  }
  if (useInternalScheduler) {
    NCCLCHECK(mscclInternalSchedulerInit());
  } else {
    NCCLCHECK(status.mscclSchedulerPtr->init());
  }
  return ncclSuccess;
}

ncclResult_t mscclInit(ncclComm_t comm) {
  if (comm->intraRanks > 1) {
    mscclInitialized.store(false, std::memory_order_release);
    INFO(NCCL_INIT, "MSCCL doesn't support multiple GPUs in one process and is not available");
    return ncclSuccess;
  }
  // Always initialize thread local status
  mscclThreadLocalStatus threadLocalStatus = mscclGetThreadLocalStatus();
  threadLocalStatus.groupStatus = mscclNoGroup;
  threadLocalStatus.groupDepth = 0;
  threadLocalStatus.captureId = ULLONG_MAX;
  threadLocalStatus.captureStatus = mscclNoCapture;
  comm->mscclCompatible = mscclCommCompatible(comm);

  {
    std::lock_guard<std::mutex> lock(mscclLifecycleMutex);

    if (mscclInitialized.load(std::memory_order_acquire)) {
      return ncclSuccess;
    }

    mscclStatus& status = mscclGetStatus();
    status.scratchBuffer = nullptr;
    status.scratchBufferSize = 0;
    status.workIndex = 1;
    status.freeAlgoHandles.resize(MSCCL_MAX_NUM_ALGOS);
    for (int i = 0; i < MSCCL_MAX_NUM_ALGOS; i++) {
      status.freeAlgoHandles[i] = MSCCL_MAX_NUM_ALGOS - i - 1;
    }
    NCCLCHECK(ncclCudaCalloc(&status.syncFlags, MSCCL_MAX_NUM_THREAD_BLOCKS));
    status.lastStream = nullptr;
    mscclSchedulerTriedLoadAlgo = false;

    NCCLCHECK(mscclSchedulerInit());

    mscclInitialized.store(true, std::memory_order_release);
  }


  size_t maxLocalSizeBytes = 0, mscclMaxLocalSizeBytes = 0;
  cudaDeviceGetLimit(&maxLocalSizeBytes, cudaLimitStackSize);
  NCCLCHECK(mscclInitKernelsForDevice(comm->cudaArch, &mscclMaxLocalSizeBytes));
  if (mscclMaxLocalSizeBytes > maxLocalSizeBytes && getEnvInt("NCCL_SET_STACK_SIZE", 0) == 1) {
    // Reset the maximum kernel stack size of all msccl kernels to avoid
    // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
    TRACE(NCCL_INIT, "Msccl Resetting cudaLimitStackSize to %zi", mscclMaxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, mscclMaxLocalSizeBytes));
  }

  INFO(NCCL_INIT, "MSCCL: Initialization finished");
  return ncclSuccess;
}

ncclResult_t mscclGroupStart() {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  threadLocalStatus.groupDepth++;
  if (threadLocalStatus.groupStatus == mscclNoGroup) {
    threadLocalStatus.groupStatus = mscclGroupSupportedOp;
  }
  return ncclSuccess;
}

static ncclResult_t mscclInternalSchedulerSelectAlgo(struct mscclSchedulerParam* param) {
  mscclStatus& status = mscclGetStatus();
  param->scheduled = false;

  // Current MSCCL doesn't support pre/post op
  if (param->op >= ncclAvg) {
    return ncclSuccess;
  }
  //nccl imp: ncclRedOp_t netOp = info->op == ncclAvg || info->op >= ncclNumOps ? ncclSum : info->op;

  // Whether the algorithm is in-place
  bool isInPlace = false;
  if (param->func == mscclFuncReduce ||
      param->func == mscclFuncBroadcast ||
      param->func == mscclFuncAllReduce ||
      param->func == mscclFuncAllToAll) {
    isInPlace = param->sendBuff == param->recvBuff;
  } else if (param->func == mscclFuncAllGather ||
             param->func == mscclFuncGather) {
    isInPlace = (char*)param->sendBuff == (char*)param->recvBuff + param->rank * param->count * ncclTypeSize(param->dataType);
  } else if (param->func == mscclFuncReduceScatter ||
             param->func == mscclFuncScatter) {
    isInPlace = (char*)param->recvBuff == (char*)param->sendBuff + param->rank * param->count * ncclTypeSize(param->dataType);
  }

  // Search suitable algorithms
  for (size_t i = 0; i < status.algoMetas.size(); i++) {
    auto &m = status.algoMetas[i];
    size_t nBytes = param->count * ncclTypeSize(param->dataType) * m.sizeMultiplier;
    bool msgSizeIsValid =
      param->count > 0 && (param->count % m.nChunksPerLoop) == 0 &&
      nBytes >= m.minBytes && (m.maxBytes == 0 || nBytes <= m.maxBytes);
    if (msgSizeIsValid &&
        m.nRanks == param->nRanks &&
        m.func == param->func &&
        (isInPlace ? m.inPlace : m.outOfPlace)) {
      // If not loaded for current rank, load it
      if (status.rankToAlgoHandles[i].find(param->rank) == status.rankToAlgoHandles[i].end()) {
        mscclAlgoHandle_t algoHandle;
        NCCLCHECK(mscclLoadAlgo(m.filePath.c_str(), &algoHandle, param->rank));
        status.rankToAlgoHandles[i][param->rank] = algoHandle;
      }
      param->handle = status.rankToAlgoHandles[i][param->rank];
      param->scheduled = true;
      TRACE("MSCCL: SchedulerSelectAlgo: Algo %s is selected", m.filePath.c_str());
      return ncclSuccess;
    }
  }

  return ncclSuccess;
}

static ncclResult_t mscclSchedulerSelectAlgo(struct mscclSavedSchedulerParam* param) {
  mscclStatus& status = mscclGetStatus();
  if (status.mscclSchedulerPtr) {
    NCCLCHECK(status.mscclSchedulerPtr->selectAlgo(&(param->p)));
  } else {
    NCCLCHECK(mscclInternalSchedulerSelectAlgo(&(param->p)));
  }
  return ncclSuccess;
}

static ncclResult_t mscclSetSavedSchedulerParam(
  const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
  void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
  size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
  mscclFunc_t func, ncclComm_t comm, cudaStream_t stream,
  struct mscclSavedSchedulerParam* param) {
  param->p.sendBuff = sendBuff;
  param->p.sendCounts = sendCounts;
  param->p.sDisPls = sDisPls;
  param->p.recvBuff = recvBuff;
  param->p.recvCounts = recvCounts;
  param->p.rDisPls = rDisPls;
  param->p.count = count;
  param->p.dataType = dataType;
  param->p.root = root;
  param->p.peer = peer;
  param->p.op = op;
  param->p.func = func;
  param->p.rank = comm->rank;
  param->p.nRanks = comm->nRanks;
  param->comm = comm;
  param->stream = stream;
  return ncclSuccess;
}

static ncclResult_t mscclSaveCountsAndDispls(struct mscclSavedSchedulerParam* param) {
  if (param->p.sendCounts) {
    param->savedSendCounts.assign(param->p.sendCounts, param->p.sendCounts + param->p.nRanks);
    param->p.sendCounts = param->savedSendCounts.data();
    param->savedSDisPls.assign(param->p.sDisPls, param->p.sDisPls + param->p.nRanks);
    param->p.sDisPls = param->savedSDisPls.data();
    param->savedRecvCounts.assign(param->p.recvCounts, param->p.recvCounts + param->p.nRanks);
    param->p.recvCounts = param->savedRecvCounts.data();
    param->savedRDisPls.assign(param->p.rDisPls, param->p.rDisPls + param->p.nRanks);
    param->p.rDisPls = param->savedRDisPls.data();
  }
  return ncclSuccess;
}

static ncclResult_t mscclRunSavedParams() {
  mscclStatus& status = mscclGetStatus();
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  struct ncclCudaGraph graph;
  for (size_t i = 0; i < threadLocalStatus.savedSchedulerParams.size(); i++) {
    auto& param = threadLocalStatus.savedSchedulerParams[i];
    // Get graph status before batch call
    if (i == 0) {
      NCCLCHECK(ncclCudaGetCapturingGraph(&graph, param.stream));
      status.graphEnabled = (graph.graph != nullptr);
      status.graphFirstKernel = status.graphEnabled;
    }
    NCCLCHECK(mscclRunAlgo(
      param.p.sendBuff, param.p.sendCounts, param.p.sDisPls,
      param.p.recvBuff, param.p.recvCounts, param.p.rDisPls,
      param.p.count, param.p.dataType, param.p.root, param.p.peer, param.p.op, param.p.handle, param.comm, param.stream));
  }
  threadLocalStatus.savedSchedulerParams.clear();
  return ncclSuccess;
}

static ncclResult_t mscclFallBackSavedParams() {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  mscclSetIsCallerFlag();
  for (auto& param : threadLocalStatus.savedSchedulerParams) {
    switch (param.p.func) {
      case mscclFuncReduce:
        NCCLCHECK(ncclReduce(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.op, param.p.root, param.comm, param.stream));
        break;
      case mscclFuncBroadcast:
        NCCLCHECK(ncclBroadcast(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.root, param.comm, param.stream));
        break;
      case mscclFuncAllReduce:
        NCCLCHECK(ncclAllReduce(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.op, param.comm, param.stream));
        break;
      case mscclFuncReduceScatter:
        NCCLCHECK(ncclReduceScatter(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.op, param.comm, param.stream));
        break;
      case mscclFuncAllGather:
        NCCLCHECK(ncclAllGather(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.comm, param.stream));
        break;
      case mscclFuncSend:
        NCCLCHECK(ncclSend(param.p.sendBuff, param.p.count, param.p.dataType,
          param.p.peer, param.comm, param.stream));
        break;
      case mscclFuncRecv:
        NCCLCHECK(ncclRecv(param.p.recvBuff, param.p.count, param.p.dataType,
          param.p.peer, param.comm, param.stream));
        break;
      case mscclFuncAllToAll:
        NCCLCHECK(ncclAllToAll(param.p.sendBuff, param.p.recvBuff, param.p.count, param.p.dataType,
          param.comm, param.stream));
        break;
      default:
        WARN("Invalid MSCCL function type in saved parameter");
        return ncclInvalidUsage;
    }
  }
  mscclClearIsCallerFlag();
  threadLocalStatus.savedSchedulerParams.clear();
  return ncclSuccess;
}

ncclResult_t mscclEnqueueCheck(
    const void* sendBuff, const size_t sendCounts[], const size_t sDisPls[],
    void* recvBuff, const size_t recvCounts[], const size_t rDisPls[],
    size_t count, ncclDataType_t dataType, int root, int peer, ncclRedOp_t op,
    mscclFunc_t func, ncclComm_t comm, cudaStream_t stream) {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();

  threadLocalStatus.savedSchedulerParams.push_back({});
  NCCLCHECK(mscclSetSavedSchedulerParam(
    sendBuff, sendCounts, sDisPls, recvBuff, recvCounts, rDisPls,
    count, dataType, root, peer, op, func, comm, stream,
    &threadLocalStatus.savedSchedulerParams.back()));

  switch (threadLocalStatus.groupStatus) {
    case mscclNoGroup:
      if (comm->mscclCompatible) {
            NCCLCHECK(mscclSchedulerSelectAlgo(&threadLocalStatus.savedSchedulerParams.back()));
            if (threadLocalStatus.savedSchedulerParams.back().p.scheduled) {
              NCCLCHECK(mscclRunSavedParams());
              break;
            }
        }
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    case mscclGroupSupportedOp:
      if (comm->mscclCompatible) {
          NCCLCHECK(mscclSchedulerSelectAlgo(&threadLocalStatus.savedSchedulerParams.back()));
          if (threadLocalStatus.savedSchedulerParams.back().p.scheduled) {
            // Only save counts and displs when there is suitable MSCCL algorithm for this
            NCCLCHECK(mscclSaveCountsAndDispls(&threadLocalStatus.savedSchedulerParams.back()));
            break;
          }
        }
      threadLocalStatus.groupStatus = mscclGroupUnsupportedOp;
      NCCLCHECK(mscclFallBackSavedParams());
    case mscclGroupUnsupportedOp:
      NCCLCHECK(mscclFallBackSavedParams());
      break;
    default:
      return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t mscclGroupEnd() {
  mscclThreadLocalStatus& threadLocalStatus = mscclGetThreadLocalStatus();
  threadLocalStatus.groupDepth--;
  if (threadLocalStatus.groupDepth == 0) {
    if (threadLocalStatus.groupStatus == mscclGroupSupportedOp) {
      NCCLCHECK(mscclRunSavedParams());
    }
    threadLocalStatus.groupStatus = mscclNoGroup;
  }
  return ncclSuccess;
}

static ncclResult_t mscclInternalSchedulerTeardown() {
  ncclResult_t ret = ncclSuccess, tmpRet = ncclSuccess;
  mscclStatus& status = mscclGetStatus();
  for (auto &m : status.rankToAlgoHandles) {
    for (auto &p : m) {
      tmpRet = mscclUnloadAlgo(p.second);
      if (ret == ncclSuccess) {
        ret = tmpRet;
      }
    }
  }
  status.algoMetas.clear();
  status.rankToAlgoHandles.clear();
  return ret;
}

ncclResult_t mscclTeardown() {
  // Always teardown thread local status
  mscclThreadLocalStatus threadLocalStatus = mscclGetThreadLocalStatus();
  threadLocalStatus.savedSchedulerParams.clear();

  {
    std::lock_guard<std::mutex> lock(mscclLifecycleMutex);

    if (!mscclInitialized.load(std::memory_order_acquire)) {
      return ncclSuccess;
    }
    mscclStatus& status = mscclGetStatus();
    for (auto &p : status.hostAlgos) {
      free(p.second);
      status.freeAlgoHandles.push_back(p.first);
    }
    for (auto &p : status.devAlgos) {
      CUDACHECK(cudaFree(p.second));
    }
    CUDACHECK(cudaFree(status.scratchBuffer));
    CUDACHECK(cudaFree(status.syncFlags));
    status.hostAlgos.clear();
    status.devAlgos.clear();
    status.freeAlgoHandles.clear();
    status.scratchBuffer = nullptr;
    status.scratchBufferSize = 0;
    status.connectedAlgos.clear();
    if (status.mscclSchedulerPtr) {
      NCCLCHECK(status.mscclSchedulerPtr->teardown());
      status.mscclSchedulerPtr = nullptr;
      dlclose(status.mscclSchedulerLib);
      status.mscclSchedulerLib = nullptr;
    } else {
      NCCLCHECK(mscclInternalSchedulerTeardown());
    }
    mscclInitialized.store(false, std::memory_order_release);
  }

  INFO(NCCL_INIT, "MSCCL: Teardown finished");
  return ncclSuccess;
}
