/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "channel.h"
#include "nvmlwrap.h"
#include "gdrwrap.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "net.h"
#include "coll_net.h"
#include "enqueue.h"
#include "graph.h"
#include "argcheck.h"
#include "tuner.h"
#include "ras.h"
#include "compiler.h"
#include "profiler.h"
#include "mnnvl.h"
#include <sys/stat.h>
#include "param.h"
#include "nvtx_payload_schemas.h"
#include "utils.h"
#include <mutex>
#include "ce_coll.h"
#include "nvtx.h"
#include "os.h"
#include "env.h"
#include "rma/rma.h"

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree", "PAT" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(RuntimeConnect, "RUNTIME_CONNECT", 1);
NCCL_PARAM(WinEnable, "WIN_ENABLE", 1);
NCCL_PARAM(CollnetEnable, "COLLNET_ENABLE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlsChannels, "NVLS_NCHANNELS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NumRmaCtx, "NUM_RMA_CTX", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(SetCpuStackSize, "SET_CPU_STACK_SIZE", 1);

extern int64_t ncclParamSingleProcMemRegEnable();


static bool ctaPolicyIsValid(int ctaPolicy) {
  int availCtaPolicies[3] = {NCCL_CTA_POLICY_DEFAULT, NCCL_CTA_POLICY_EFFICIENCY, NCCL_CTA_POLICY_ZERO};
  int maxPolicy = 0;
  for (int i = 0; i < 3; ++i) {
    maxPolicy |= availCtaPolicies[i];
  }
  return ctaPolicy >= 0 && ctaPolicy <= maxPolicy;
}

static int ctaPolicyEnv = NCCL_CONFIG_UNDEF_INT;
static void getEnvCtaPolicyOnce(){
  const char* env = ncclGetEnv("NCCL_CTA_POLICY");
  if (env == NULL) return;

  // support the old CTA Policy assignment (only valid for 0-9)
  if (isdigit(env[0])) {
    switch (env[0]) {
    case '0':
      ctaPolicyEnv = NCCL_CTA_POLICY_DEFAULT;
      break;
    case '1':
      ctaPolicyEnv = NCCL_CTA_POLICY_EFFICIENCY;
      break;
    case '2':
      ctaPolicyEnv = NCCL_CTA_POLICY_ZERO;
      break;
    default:
      INFO(NCCL_ENV, "Unknown CTA policy; the legacy usage of NCCL_CTA_POLICY only supports the value of 0 (DEFAULT), 1 (EFFICIENCY), or 2 (ZERO). Using DEFAULT instead.");
    };
  } else {
    // newer way allows the user to combine the modes
    char* str = strdup(env);
    char* token = strtok(str, "|");
    while (token) {
      int tokenPolicy = NCCL_CONFIG_UNDEF_INT;
      if (strcasecmp(token, "DEFAULT")==0) tokenPolicy = NCCL_CTA_POLICY_DEFAULT;
      else if (strcasecmp(token, "EFFICIENCY")==0) tokenPolicy = NCCL_CTA_POLICY_EFFICIENCY;
      else if (strcasecmp(token, "ZERO")==0) tokenPolicy = NCCL_CTA_POLICY_ZERO;
      else INFO(NCCL_ENV, "Unknown CTA policy %s passed as environment variable. Ignoring.", token);
      if (tokenPolicy != NCCL_CONFIG_UNDEF_INT) {
        if (ctaPolicyEnv == NCCL_CONFIG_UNDEF_INT) ctaPolicyEnv = tokenPolicy;
        else ctaPolicyEnv |= tokenPolicy;
      }
      token = strtok(NULL, "|");
    }
    if (ctaPolicyEnv == NCCL_CONFIG_UNDEF_INT) {
      INFO(NCCL_ENV, "No valid CTA policies found in NCCL_CTA_POLICY=%s.", env);
    } else {
      INFO(NCCL_ENV, "Parsed environment variable NCCL_CTA_POLICY=%s to %d", env, ctaPolicyEnv);
    }
    free(str);
  }
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_);

// GDRCOPY support: Off by default
NCCL_PARAM(GdrCopyEnable, "GDRCOPY_ENABLE", 0);

// GDRCOPY support
gdr_t ncclGdrCopy = NULL;

ncclResult_t initGdrCopy() {
  if (ncclParamGdrCopyEnable() == 1) {
    ncclGdrCopy = ncclGdrInit();
  }
  return ncclSuccess;
}


static ncclResult_t setCpuStackSize() {
  if (ncclParamSetCpuStackSize() != 0) {
    return ncclOsSetCpuStackSize();
  }

  return ncclSuccess;
}

static ncclResult_t initResult = ncclSuccess;
static std::once_flag initOnceFlag;

static void initOnceFunc() {
  setCpuStackSize();
  initGdrCopy();
  // Always initialize bootstrap network
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);

  initNvtxRegisteredEnums();
exit:;
}

static ncclResult_t ncclInit() {
  std::call_once(initOnceFlag, initOnceFunc);
  return initResult;
}

static ncclResult_t envInitResult = ncclSuccess;
static std::once_flag envInitOnceFlag;

static void envInitOnceFunc() {
  NCCLCHECKGOTO(ncclEnvPluginInit(), envInitResult, exit);
exit:;
}

ncclResult_t ncclInitEnv() {
  std::call_once(envInitOnceFlag, envInitOnceFunc);
  return envInitResult;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInitEnv());
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  struct ncclBootstrapHandle handle;
  NCCLCHECK(bootstrapGetUniqueId(&handle, NULL));
  // ncclUniqueId and bootstrapHandle don't have the same size and alignment
  // reset to 0 to avoid undefined data
  memset(out, 0, sizeof(*out));
  // copy to avoid alignment mismatch
  memcpy(out, &handle, sizeof(handle));
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)getHash(out->internal, NCCL_UNIQUE_ID_BYTES));
  return ncclSuccess;
}

// Prevent compiler from optimizing out these operations
#ifdef __clang__
#define NCCL_NO_OPTIMIZE __attribute__((optnone))
#else
#define NCCL_NO_OPTIMIZE __attribute__((optimize("O0")))
#endif

void NCCL_NO_OPTIMIZE commPoison(ncclComm_t comm) {
  // Important that this does not trash intraComm0.
  comm->rank = comm->cudaDev = comm->busId = comm->nRanks = -1;
  comm->startMagic = comm->endMagic = 0;
}

#undef NCCL_NO_OPTIMIZE


static ncclResult_t ncclDestructorFnFree(struct ncclDestructor* dtor) {
  free(dtor->obj);
  return ncclSuccess;
}
void ncclCommPushFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaHostFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclCudaHostFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaHostFree(struct ncclComm* comm, void* obj) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaHostFree;
  dtor->obj = obj;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t ncclDestructorFnCudaGdrFree(struct ncclDestructor* dtor) {
  NCCLCHECK(ncclGdrCudaFree(dtor->obj));
  return ncclSuccess;
}
void ncclCommPushCudaGdrFree(struct ncclComm* comm, void* handle) {
  struct ncclDestructor* dtor = ncclMemoryStackAlloc<struct ncclDestructor>(&comm->memPermanent);
  dtor->fn = ncclDestructorFnCudaGdrFree;
  dtor->obj = handle;
  dtor->next = comm->destructorHead;
  comm->destructorHead = dtor;
}

static ncclResult_t commFree(ncclComm_t comm) {
  int abort = 0;
  /* commFree() should not involve any sync among ranks. */
  if (comm == NULL)
    return ncclSuccess;

  NCCLCHECK(ncclCeFinalize(comm));
  NCCLCHECK(ncclRmaCeFinalize(comm));

  if (comm->symmetricSupport) {
    NCCLCHECK(ncclSymkFinalize(comm));
    NCCLCHECK(ncclDevrFinalize(comm));
  }
  NCCLCHECK(ncclRasCommFini(comm));

  /* in commReclaim, we have guaranteed only last rank which calls ncclCommDestroy() will
   * free all intra-process communicators; therefore, we only need to focus on local
   * resource cleanup in commFree(). */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread.joinable()) {
    comm->proxyState->thread.join();
    if (comm->proxyState->threadUDS.joinable()) {
      // UDS support
      comm->proxyState->threadUDS.join();
    }
  }

  if (comm->memPool) CUDACHECK(cudaMemPoolDestroy(comm->memPool));

  delete[] comm->userRedOps;

  free(comm->connectSend);
  free(comm->connectRecv);

  free(comm->peerInfo);
  if (comm->topo)
    ncclTopoFree(comm->topo);
  if (comm->nodeRanks) {
    for (int n=0; n<comm->nNodes; n++) free(comm->nodeRanks[n].localRankToRank);
    free(comm->nodeRanks);
  }
  free(comm->rankToNode);
  free(comm->rankToLocalRank);
  free(comm->collNetHeads);
  free(comm->clique.ranks);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, 1, comm->localRanks));

  // GIN may use proxy. We need to finalize it before destroying the proxy.
  NCCLCHECK(ncclGinFinalize(comm));
  NCCLCHECK(ncclRmaProxyFinalize(comm));

  int sharedResRefCount = 0;
  if (comm->sharedRes) {
    sharedResRefCount = ncclAtomicRefCountDecrement(&comm->sharedRes->refCount);
    if (sharedResRefCount == 0) {
      for (int c=0; c<MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c]);
      }
      free(comm->sharedRes->tpRankToLocalRank);
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->launchEvent));
      CUDACHECK(cudaEventDestroy(comm->sharedRes->scratchEvent));
      NCCLCHECK(ncclProxyDestroy(comm));
      delete comm->sharedRes;
    }
  }

  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  abort = *comm->abortFlag;
  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    free(comm->abortFlag);
    NCCLCHECK(ncclCudaHostFree((void*)comm->abortFlagDev));
    free(comm->abortFlagRefCount);
  }
  free((void*)comm->config.netName);

  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);
  free(comm->gproxyConn);

  NCCLCHECK(ncclRegCleanup(comm));

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - %s COMPLETE", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId, abort ? "Abort" : "Destroy");

  commPoison(comm); // poison comm before free to avoid comm reuse.
  NCCLCHECK(ncclProfilerPluginFinalize(comm));
  if (sharedResRefCount == 0) NCCLCHECK(ncclNetFinalize(comm));
  ncclCudaContextDrop(comm->context);
  free(comm);

  return ncclSuccess;
}

NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
#define NCCL_WORK_FIFO_BYTES_DEFAULT (1<<20)
NCCL_PARAM(WorkFifoBytes, "WORK_FIFO_BYTES", NCCL_WORK_FIFO_BYTES_DEFAULT);
NCCL_PARAM(WorkArgsBytes, "WORK_ARGS_BYTES", INT64_MAX);
enum ncclLaunchMode ncclParamLaunchMode;

NCCL_PARAM(DmaBufEnable, "DMABUF_ENABLE", 1);

// Detect DMA-BUF support
static ncclResult_t dmaBufSupported(struct ncclComm* comm) {
  if (ncclParamDmaBufEnable() == 0 || comm->ncclNet->regMrDmaBuf == NULL || ncclCudaLibraryInit() != ncclSuccess) return ncclInternalError;
#if CUDA_VERSION >= 11070
  int flag = 0;
  CUdevice dev;
  int cudaDriverVersion;
  CUDACHECK(cudaDriverGetVersion(&cudaDriverVersion));
  if (CUPFN(cuDeviceGet) == NULL || cudaDriverVersion < 11070) return ncclInternalError;
  CUCHECK(cuDeviceGet(&dev, comm->cudaDev));
  // Query device to see if DMA-BUF support is available
  (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, dev));
  if (flag == 0) return ncclInternalError;
  INFO(NCCL_INIT, "DMA-BUF is available on GPU device %d", comm->cudaDev);
  return ncclSuccess;
#endif
  return ncclInternalError;
}

ncclResult_t ncclCommEnsureReady(ncclComm_t comm) {
  /* comm must be ready, or error will be reported */
  ncclResult_t ret = ncclSuccess;
  if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret == ncclInProgress) {
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      ret = ncclInvalidArgument;
      goto exit;
    }
    /* if ret is not ncclInProgress, we just keep it. */
  }

exit:
  return ret;
}

static ncclResult_t commAlloc(struct ncclComm* comm, struct ncclComm* parent, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  ncclMemoryStackConstruct(&comm->memPermanent);
  ncclMemoryStackConstruct(&comm->memScoped);
  comm->destructorHead = nullptr;
  comm->rank = rank;
  comm->nRanks = ndev;

  if (parent == NULL || !parent->shareResources) {
    struct ncclSharedResources* sharedRes;
    NEW_NOTHROW(sharedRes, ncclSharedResources);
    /* most of attributes are assigned later in initTransportsRank(). */
    sharedRes->owner = comm;
    sharedRes->tpNRanks = comm->nRanks;
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->launchEvent, cudaEventDisableTiming));
    CUDACHECK(cudaEventCreateWithFlags(&sharedRes->scratchEvent, cudaEventDisableTiming));
    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;
    NCCLCHECK(ncclNetInit(comm));
  } else {
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
    NCCLCHECK(ncclNetInitFromParent(comm, parent));
  }

  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  if (parent && parent->shareResources) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }
  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  CUDACHECK(cudaGetDevice(&comm->cudaDev));

  NCCLCHECK(ncclCudaContextTrack(&comm->context));

  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  nvmlDevice_t nvmlDev;
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  NCCLCHECK(int64ToBusId(comm->busId, busId));
  NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
  NCCLCHECK(ncclNvmlDeviceGetIndex(nvmlDev, (unsigned int*)&comm->nvmlDev));

  comm->compCap = ncclCudaCompCap();
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);

  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);
  ncclMemoryPoolConstruct(&comm->memPool_ncclRmaProxyDesc);

  for (int i = 0; i < ncclGroupTaskTypeNum; i++) {
    comm->groupNext[i] = reinterpret_cast<struct ncclComm*>(0x1);
  }
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;
  // for bcast: init the ringTasks and min/max bcast peer
  comm->ringTasks = ncclMemoryStackAlloc<void*>(&comm->memPermanent, comm->nRanks);
  comm->planner.bcast_info.minBcastPeer = INT_MAX;
  comm->planner.bcast_info.maxBcastPeer = INT_MIN;

  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);
  ncclIntruQueueConstruct(&comm->legacyRegCleanupQueue);
  ncclIntruQueueConstruct(&comm->ceInitTaskQueue);

  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);

  do {
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypeNone;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = comm->cudaDev;
    CUDACHECK(cudaMemPoolCreate(&comm->memPool, &props));
    uint64_t releaseThreshold = ~uint64_t(0);
    CUDACHECK(cudaMemPoolSetAttribute(comm->memPool, cudaMemPoolAttrReleaseThreshold, &releaseThreshold));
  } while (0);

  ncclIntruQueueConstruct(&comm->eventCallbackQueue);

  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclKernelCommAndChannels tmpCommAndChans;
  struct ncclKernelCommAndChannels *devCommAndChans = NULL;
  struct ncclNvmlCCStatus ccStatus;
  bool ccEnable;
  cudaStream_t deviceStream;

  memset(&tmpCommAndChans, '\0', sizeof(tmpCommAndChans));
  NCCLCHECKGOTO(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false, &deviceStream), ret, fail);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  ncclCommPushCudaFree(comm, tmpCommAndChans.comm.rankToLocalRank);
  NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.rankToLocalRank, comm->rankToLocalRank, comm->nRanks, deviceStream), ret, fail);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlagDev;
  tmpCommAndChans.comm.isAllNvlink = comm->isAllNvlink;
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workArgsBytes = std::min<size_t>(ncclParamWorkArgsBytes(), ncclMaxKernelArgsSize(comm->cudaArch));

  memset(&ccStatus, 0, sizeof(ccStatus));
  ccEnable = (ncclSuccess == ncclNvmlGetCCStatus(&ccStatus)) && (ccStatus.CCEnabled || ccStatus.multiGpuProtectedPCIE || ccStatus.multiGpuNVLE);
  comm->ccEnable = ccEnable;
  if (ccEnable) {
    comm->workFifoBytes = 0;
  } else {
    comm->workFifoBytes = ncclParamWorkFifoBytes();
    if (0 != (comm->workFifoBytes & (comm->workFifoBytes-1))) {
      WARN("NCCL_WORK_FIFO_BYTES=%d is being ignored because it is not a power of 2.", comm->workFifoBytes);
      comm->workFifoBytes = NCCL_WORK_FIFO_BYTES_DEFAULT;
    }
    comm->workFifoBytes = std::min(comm->workFifoBytes, 1u<<30);
  }

  if (comm->rank == 0) {
    INFO(NCCL_INIT, "CC %s, workFifoBytes %d", ccEnable ? "On" : "Off", comm->workFifoBytes);
  }

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoBuf lives in GDR mapped CUDA memory.
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoBuf, &comm->workFifoBufDev, comm->workFifoBytes, &comm->workFifoBufGdrHandle), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoBufGdrHandle);
  } else {
    // The workFifoBuf lives in cudaHost memory.
    comm->workFifoBufGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoBuf, comm->workFifoBytes), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoBuf);
    comm->workFifoBufDev = comm->workFifoBuf;
  }

  comm->workFifoProduced = 0;
  comm->workFifoProducedLastRecorded = 0;
  comm->workFifoConsumed = 0;

  // Alloc profiler counters for the kernel
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workStarted, MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->profiler.workCompleted, MAXCHANNELS), ret, fail);
  tmpCommAndChans.comm.workStarted = comm->profiler.workStarted;
  tmpCommAndChans.comm.workCompleted = comm->profiler.workCompleted;
  ncclCommPushCudaHostFree(comm, comm->profiler.workStarted);
  ncclCommPushCudaHostFree(comm, comm->profiler.workCompleted);

  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, deviceStream), ret, fail);
  }

  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, deviceStream), ret, fail);
    }
  }

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, deviceStream), ret, fail);
exit:
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &comm->sharedRes->deviceStream, /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
extern const char *ncclGetGitVersion(void);
static void showVersion() {
  if (ncclDebugLevel == NCCL_LOG_VERSION || ncclDebugLevel == NCCL_LOG_WARN) {
    VERSION("%s", VERSION_STRING);
  } else {
    INFO(NCCL_ALL,"%s", VERSION_STRING);
  }
  INFO(NCCL_ALL,"%s", ncclGetGitVersion());
}

NCCL_PARAM(MNNVLUUID, "MNNVL_UUID", -1);
NCCL_PARAM(MNNVLCliqueId, "MNNVL_CLIQUE_ID", -1);

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  cudaDeviceProp prop;
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->nvmlDev = comm->nvmlDev;
  NCCLCHECK(ncclGetVersion(&info->version));
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;
  info->cuMemSupport = ncclCuMemEnable();
  CUDACHECK(cudaGetDeviceProperties(&prop, comm->cudaDev));
  info->totalGlobalMem = ROUNDUP(prop.totalGlobalMem, (1L << 32));

  // Get the device MAJOR:MINOR of /dev/shm so we can use that
  // information to decide whether we can use SHM for inter-process
  // communication in a container environment
  struct stat statbuf;
  SYSCHECK(stat("/dev/shm", &statbuf), "stat");
  info->shmDev = statbuf.st_dev;

  info->busId = comm->busId;

  NCCLCHECK(ncclGpuGdrSupport(comm, &info->gdrSupport));
  info->comm = comm;
  info->cudaCompCap = comm->minCompCap = comm->maxCompCap = comm->compCap;

  // MNNVL support
  {
    // MNNVL: Request the fabric UUID and partition info
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    nvmlDevice_t nvmlDev;
    NCCLCHECK(int64ToBusId(info->busId, busId));
    NCCLCHECK(ncclNvmlDeviceGetHandleByPciBusId(busId, &nvmlDev));
    info->fabricInfo.state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED;
    (void) ncclNvmlDeviceGetGpuFabricInfoV(nvmlDev, &info->fabricInfo);
    if (info->fabricInfo.state != NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
      unsigned long uuid0 = 0;
      unsigned long uuid1 = 0;
      if (ncclParamMNNVLUUID() != -1) {
        unsigned long temp_uuid0 = (unsigned long)ncclParamMNNVLUUID();
        unsigned long temp_uuid1 = (unsigned long)ncclParamMNNVLUUID();
        memcpy(info->fabricInfo.clusterUuid, &temp_uuid0, sizeof(temp_uuid0));
        memcpy(info->fabricInfo.clusterUuid + sizeof(temp_uuid0), &temp_uuid1, sizeof(temp_uuid1));
      }
      memcpy(&uuid0, info->fabricInfo.clusterUuid, sizeof(uuid0));
      memcpy(&uuid1, info->fabricInfo.clusterUuid + sizeof(uuid0), sizeof(uuid1));
      if (ncclParamMNNVLCliqueId() == -2) {
        nvmlPlatformInfo_t platformInfo = { 0 };
        NCCLCHECK(ncclNvmlDeviceGetPlatformInfo(nvmlDev, &platformInfo));
        INFO(NCCL_INIT, "MNNVL rack serial %s slot %d tray %d hostId %d peerType %d moduleId %d",
             platformInfo.chassisSerialNumber, platformInfo.slotNumber, platformInfo.trayIndex,
             platformInfo.hostId, platformInfo.peerType, platformInfo.moduleId);
        // Use a hash of the Rack serial number to partition the NVLD clique
        info->fabricInfo.cliqueId = getHash(platformInfo.chassisSerialNumber, sizeof(platformInfo.chassisSerialNumber));
      } else if (ncclParamMNNVLCliqueId() != -1) info->fabricInfo.cliqueId = ncclParamMNNVLCliqueId();
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId,
           uuid0, uuid1,
           info->fabricInfo.cliqueId, info->fabricInfo.state, info->fabricInfo.healthMask);
    }
  }

  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclRing* ring = &comm->channels[channelId].ring;
  // Find our ring-distance from rank zero and reorganize ranks to start with rank.
  int ixZero=0, ixRank=0;
  for (int i=0; i < nranks; i++) {
    if (ringRanks[i] == 0) ixZero = i;
    if (ringRanks[i] == rank) ixRank = i;
  }
  ring->index = (ixRank-ixZero + nranks)%nranks;
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+ixRank)%nranks];
    ring->rankToIndex[ring->userRanks[i]] = i;
  }
  return ncclSuccess;
}

#define DEFAULT_LL_BUFFSIZE (NCCL_LL_LINES_PER_THREAD*NCCL_LL_MAX_NTHREADS*NCCL_STEPS*sizeof(union ncclLLFifoLine))
#define DEFAULT_LL128_BUFFSIZE (NCCL_LL128_ELEMS_PER_THREAD*NCCL_LL128_MAX_NTHREADS*NCCL_STEPS*sizeof(uint64_t))
#define DEFAULT_BUFFSIZE (1 << 22) /* 4MiB */
NCCL_PARAM(BuffSize, "BUFFSIZE", -2);
NCCL_PARAM(LlBuffSize, "LL_BUFFSIZE", -2);
NCCL_PARAM(Ll128BuffSize, "LL128_BUFFSIZE", -2);

NCCL_PARAM(P2pNetChunkSize, "P2P_NET_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pPciChunkSize, "P2P_PCI_CHUNKSIZE", (1 << 17)); /* 128 kB */
NCCL_PARAM(P2pNvlChunkSize, "P2P_NVL_CHUNKSIZE", (1 << 19)); /* 512 kB */

static ncclResult_t computeBuffSizes(struct ncclComm* comm) {
  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  if (comm->nNodes > 1) {
    // GBx platforms only have 4 GPUs per host, which reduces the aggregation factor.
    // Increase the ratio by a factor of 2 to compensate.
    int ratio = (comm->cpuArch == NCCL_TOPO_CPU_ARCH_ARM && comm->minCompCap >= 100) ? 2 : 1;
    comm->p2pChunkSize = ratio * ncclParamP2pNetChunkSize();
  } else if (comm->isAllNvlink) {
    comm->p2pChunkSize = ncclParamP2pNvlChunkSize();
  } else {
    comm->p2pChunkSize = ncclParamP2pPciChunkSize();
  }

  // Make sure P2P chunksize is not larger than coll chunksize.
  if (comm->p2pChunkSize * NCCL_STEPS > comm->buffSizes[NCCL_PROTO_SIMPLE]) comm->p2pChunkSize = comm->buffSizes[NCCL_PROTO_SIMPLE]/NCCL_STEPS;

  if (comm->sharedRes->owner != comm) {
    /* make sure split comm p2pChunkSize won't exceed shared p2pChunkSize. */
    comm->p2pChunkSize = std::min(comm->p2pChunkSize, comm->sharedRes->tpP2pChunkSize);
  } else {
    comm->sharedRes->tpP2pChunkSize = comm->p2pChunkSize;
  }

  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVLEnable, "MNNVL_ENABLE", 2);

#define TIMER_INIT_TOTAL 0
#define TIMER_INIT_KERNELS 1
#define TIMER_INIT_BOOTSTRAP 2
#define TIMER_INIT_ALLGATHER 3
#define TIMER_INIT_TOPO 4
#define TIMER_INIT_GRAPHS 5
#define TIMER_INIT_CONNECT 6
#define TIMER_INIT_ALLOC 7
#define TIMERS_INIT_COUNT 8

extern int64_t ncclParamWinStride();

static ncclResult_t initNvlDomainInfo(struct ncclComm* comm) {
  // Initialize NVLink domain info
  comm->nvlDomainInfo.nNvlDomains = comm->nNodes;
  comm->nvlDomainInfo.minRanksPerNvlDomain = comm->minLocalRanks;
  comm->nvlDomainInfo.maxRanksPerNvlDomain = comm->maxLocalRanks;

  TRACE(NCCL_INIT, "NVLink domains: %d domains, min ranks per domain: %d, max ranks per domain: %d",
        comm->nNodes, comm->nvlDomainInfo.minRanksPerNvlDomain, comm->nvlDomainInfo.maxRanksPerNvlDomain);

  return ncclSuccess;
}

NCCL_PARAM(GroupSize, "P2P_SCHEDULE_GROUP_SIZE", NCCL_MAX_DEV_WORK_P2P_PER_BATCH);

static ncclResult_t ncclP2pSchedule(struct ncclComm* comm) {
  struct ncclNodeRanks* nodeRanks = comm->nodeRanks;
  // For MNNVL systems, we need to split the nodes into different groups to guarantee the proper PXN aggregation factor.
  int groupSize = (comm->nNodes > 1) ? ncclParamGroupSize() : comm->maxLocalRanks;
  for (int node = 0; node < comm->nNodes; node++) {
    int localRanks = nodeRanks[node].localRanks;
    if (localRanks % groupSize != 0 || localRanks < groupSize) groupSize = gcd(groupSize, nodeRanks[node].localRanks);
  }
  comm->p2pSchedGroupSize = groupSize;

  int local = comm->localRank % groupSize; // local id inside my group
  int group = comm->localRank / groupSize; // id of my group, incremented when going over the previous nodes
  int nGroups = comm->nRanks / groupSize;
  int nGroupsPow2 = pow2Up(nGroups);

  int *groupToNode, *groupToLocal;
  NCCLCHECK(ncclCalloc(&groupToNode, nGroups));  // node hosting the group
  NCCLCHECK(ncclCalloc(&groupToLocal, nGroups)); // local offset of the group
  int groupCount = 0;
  for (int n = 0; n < comm->nNodes; ++n) {
    if (0 != comm->nodeRanks[n].localRanks % groupSize) {
      WARN("nLocals = %d should be a diviser of the number of ranks in node %d = %d", groupSize, n, comm->nodeRanks[n].localRanks);
      return ncclInternalError;
    }
    int nGroupsInNode = comm->nodeRanks[n].localRanks / groupSize;
    for (int g = 0; g < nGroupsInNode; ++g) {
      groupToLocal[groupCount] = g * groupSize;
      groupToNode[groupCount] = n;
      groupCount++;
    }
    if (n < comm->node) group += nGroupsInNode;
  }
  if (groupCount != nGroups) {
    WARN("Group creation failed: %d vs %d", groupCount, nGroups);
    return ncclInternalError;
  }
  INFO(NCCL_GRAPH,"%s: group size used is %d",__func__,groupSize);

  uint32_t groupRound = 0, groupDelta = 0;
  int round = 0;
  // When enumerating peer deltas we use the quadratic formula (x*x+x)/2 mod N.
  // Since that formula only produces valid permutations when N is a pow of 2,
  // we let N = pow2Up(n) and filter out results greater-eq to n.
  // Example sequence for 16 ranks: 0, 1, 3, 6, 10, 15, 5, 12, 4, 13, 7, 2, 14, 11, 9, 8
  do {
    if (groupDelta < nGroups) { // Filter nonsensical group deltas
      int sendGroup = (group + groupDelta) % nGroups;
      int recvGroup = (group - groupDelta + nGroups) % nGroups;
      int sendNode = groupToNode[sendGroup];
      int recvNode = groupToNode[recvGroup];
      for (int delta = 0; delta < groupSize; delta++) {
        int sendLocal = groupToLocal[sendGroup] + (local + delta) % groupSize;
        int recvLocal = groupToLocal[recvGroup] + (local - delta + groupSize) % groupSize;
        comm->p2pSchedule[round].sendRank = nodeRanks[sendNode].localRankToRank[sendLocal];
        comm->p2pSchedule[round].recvRank = nodeRanks[recvNode].localRankToRank[recvLocal];
        round += 1;
      }
    }
    groupRound += 1;
    groupDelta = (groupDelta + groupRound) & (nGroupsPow2 - 1); // Quadratic update
  } while (groupRound != nGroupsPow2);

  free(groupToNode);
  free(groupToLocal);

  if (round != comm->nRanks) {
    WARN("P2p schedule creation has bugs.");
    return ncclInternalError;
  }
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent, uint64_t timers[TIMERS_INIT_COUNT]) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;
  ncclAffinity affinitySave;
  struct ncclTopoGraph* ringGraph = &comm->graphs[NCCL_ALGO_RING];
  struct ncclTopoGraph* treeGraph = &comm->graphs[NCCL_ALGO_TREE];
  struct ncclTopoGraph* collNetChainGraph = &comm->graphs[NCCL_ALGO_COLLNET_CHAIN];
  struct ncclTopoGraph* collNetDirectGraph = &comm->graphs[NCCL_ALGO_COLLNET_DIRECT];
  struct ncclTopoGraph* nvlsGraph = &comm->graphs[NCCL_ALGO_NVLS];
  struct ncclTopoGraph* graphs[NCCL_NUM_ALGORITHMS] = { treeGraph, ringGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph, nvlsGraph, treeGraph };

  struct graphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
    int crossNic;
  };

  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
    struct ncclTopoRanks topoRanks;
    int cpuArch;
    int cpuVendor;
    int localRanks;
    int p2pnChannelsPerPeer;
    bool nicFused;
  };

  int nChannelsOrig;
  struct allGatherInfo *allGather3Data = NULL;
  struct ncclTopoRanks** allTopoRanks = NULL;
  int *nodesFirstRank = NULL, *nodesTreePatterns = NULL;
  int *rings = NULL;
  int* nvbPeers = NULL;
  struct ncclProxyConnector proxyConn;
  int* pxnPeers = NULL;
  int *topParentLocalRanks = NULL;
  int p2pLevel = -1;
  bool globalNicFused = false;

  timers[TIMER_INIT_ALLGATHER] = clockNano();
  // AllGather1 - begin
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail); // Extra rank to represent CollNet root
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, comm->commHash), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);
  COMPILER_ATOMIC_STORE(&comm->peerInfoValid, true, std::memory_order_release);

  comm->cuMemSupport = 1;
  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].version != comm->peerInfo[rank].version) {
      WARN("Mismatched NCCL version detected : rank %d version %d rank %d version %d",
           i, comm->peerInfo[i].version, rank, comm->peerInfo[rank].version);
      ret = ncclInvalidUsage;
      goto fail;
    }
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    if (!comm->peerInfo[i].cuMemSupport) comm->cuMemSupport = 0;
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }
  // AllGather1 - end
  timers[TIMER_INIT_ALLGATHER] = clockNano() - timers[TIMER_INIT_ALLGATHER];

  // Check for MNNVL support
  NCCLCHECKGOTO(ncclGetUserP2pLevel(&p2pLevel), ret, fail);
  if ((nNodes > 1 && ncclParamMNNVLEnable() != 0 && p2pLevel != 0) || ncclParamMNNVLEnable() == 1) {
    NCCLCHECKGOTO(ncclMnnvlCheck(comm), ret, fail);
  }

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;

    comm->nvlsRegSupport = 1;
    for (int i = 0; i < nranks; i++) {
      comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[i].cudaCompCap);
      comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[i].cudaCompCap);
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) &&
          (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
        // Rank is in same process
        if (intraProcRanks == 0) intraProcRank0 = i;
        if (i == rank) intraProcRank = intraProcRanks;
        intraProcRanks++;
        if (intraProcRank0 == rank && rank != i) {
          comm->peerInfo[i].comm->intraNext = comm->intraNext;
          comm->intraNext = comm->peerInfo[i].comm;
        }
      }

      if (comm->nvlsRegSupport) {
        for (int j = i + 1; j < nranks; j++) {
          if (comm->peerInfo[i].hostHash == comm->peerInfo[j].hostHash &&
            comm->peerInfo[i].pidHash == comm->peerInfo[j].pidHash) {
            comm->nvlsRegSupport = 0;
            break;
          }
        }
      }
    }

    // Buffer Registration is not supported with MNNVL
    if (comm->MNNVL) comm->nvlsRegSupport = 0;
    else if (ncclParamSingleProcMemRegEnable()) comm->nvlsRegSupport = 1;

    TRACE(NCCL_INIT,"pidHash[%d] %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
        rank, comm->peerInfo[rank].pidHash, intraProcRank, intraProcRanks, intraProcRank0);
    if (intraProcRank == -1 || intraProcRank0 == -1 || comm->peerInfo[intraProcRank0].comm == NULL) {
      WARN("Failed to determine intra proc ranks rank %d hostHash %lx pidHash %lx intraProcRank %d intraProcRanks %d intraProcRank0 %d",
          rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
          intraProcRank, intraProcRanks, intraProcRank0);
      ret = ncclInternalError;
      goto fail;
    }
    struct ncclComm* comm0 = comm->peerInfo[intraProcRank0].comm;
    assert(intraProcRank==0 ? comm==comm0 : true);
    comm->intraComm0 = comm0;
    comm->intraRank = intraProcRank;
    comm->intraRanks = intraProcRanks;
    comm->intraBarrierPhase = 0;
    comm->intraBarrierCounter = 0;
    comm->intraBarrierGate = 0;
  } while(0);

  timers[TIMER_INIT_TOPO] = clockNano();

  // Dump XML if requested by user
  const char* dumpXmlFile;
  dumpXmlFile = ncclGetEnv("NCCL_TOPO_DUMP_FILE");
  if (dumpXmlFile) {
    NCCLCHECKGOTO(ncclTopoGetSystem(comm, NULL, dumpXmlFile), ret, fail);
  }

  // Topo detection / System graph creation
  NCCLCHECKGOTO(ncclTopoGetSystem(comm, &comm->topo), ret, fail);
  // Compute paths between GPUs and NICs
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Remove inaccessible GPUs and unused NICs
  NCCLCHECKGOTO(ncclTopoTrimSystem(comm->topo, comm), ret, fail);
  // Recompute paths after trimming
  NCCLCHECKGOTO(ncclTopoComputePaths(comm->topo, comm), ret, fail);
  // Init search
  NCCLCHECKGOTO(ncclTopoSearchInit(comm->topo), ret, fail);
  // Decide on comm's CPU architecture.
  NCCLCHECKGOTO(ncclTopoComputeCommCPU(comm), ret, fail);
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);
  timers[TIMER_INIT_TOPO] = clockNano() - timers[TIMER_INIT_TOPO];

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  if (ncclOsCpuCount(comm->cpuAffinity)) {
    NCCLCHECKGOTO(ncclOsGetAffinity(&affinitySave), ret, fail);
    NCCLCHECKGOTO(ncclOsSetAffinity(comm->cpuAffinity), ret, fail);
  }

  // Determine local CollNet support
  if (!collNetSupport(comm)) {
    comm->config.collnetEnable = 0;
  }

  // Determine local Nvls support
  NCCLCHECK(ncclNvlsInit(comm));

  timers[TIMER_INIT_GRAPHS] = clockNano();
  // Get rings and trees
  memset(ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph->id = 0;
  ringGraph->pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph->minChannels = 1;
  ringGraph->maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, ringGraph), ret, fail);

  memset(treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph->id = 1;
  treeGraph->pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph->minChannels = ringGraph->nChannels;
  treeGraph->maxChannels = ringGraph->nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, treeGraph), ret, fail);

  memset(collNetChainGraph, 0, sizeof(struct ncclTopoGraph));
  collNetChainGraph->id = 2;
  collNetChainGraph->pattern = NCCL_TOPO_PATTERN_TREE;
  collNetChainGraph->collNet = 1;
  collNetChainGraph->minChannels = ringGraph->nChannels;
  collNetChainGraph->maxChannels = ringGraph->nChannels;

  memset(collNetDirectGraph, 0, sizeof(struct ncclTopoGraph));
  collNetDirectGraph->id = 4;
  collNetDirectGraph->pattern = NCCL_TOPO_PATTERN_COLLNET_DIRECT;
  collNetDirectGraph->collNet = 1;
  collNetDirectGraph->minChannels = 1;
  collNetDirectGraph->maxChannels = MAXCHANNELS;
  if (comm->config.collnetEnable) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetChainGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, collNetDirectGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, collNetDirectGraph), ret, fail);
  }

  memset(nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph->id = 3;
  nvlsGraph->pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph->minChannels = 1;
  nvlsGraph->maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, nvlsGraph), ret, fail);
  }
  timers[TIMER_INIT_GRAPHS] = clockNano() - timers[TIMER_INIT_GRAPHS];

  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[5] = { ringGraph, treeGraph, collNetDirectGraph, collNetChainGraph, nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 5, dumpGraphs), ret, fail);
  }

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannelsPerPeer(comm), ret, fail);

  // Because timers[[TIMER_INIT_ALLGATHER] already contains the timing of the first allgather,
  // we temporarily store the start time of the subsequent one in an as-of-yet unused CONNECT timer.
  timers[TIMER_INIT_CONNECT] = clockNano();
  // AllGather3 - begin
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);

  for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
    allGather3Data[rank].graphInfo[a].pattern = graphs[a]->pattern;
    allGather3Data[rank].graphInfo[a].nChannels = graphs[a]->nChannels;
    allGather3Data[rank].graphInfo[a].sameChannels = graphs[a]->sameChannels;
    allGather3Data[rank].graphInfo[a].bwIntra = graphs[a]->bwIntra;
    allGather3Data[rank].graphInfo[a].bwInter = graphs[a]->bwInter;
    allGather3Data[rank].graphInfo[a].typeIntra = graphs[a]->typeIntra;
    allGather3Data[rank].graphInfo[a].typeInter = graphs[a]->typeInter;
    allGather3Data[rank].graphInfo[a].crossNic = graphs[a]->crossNic;
  }

  allGather3Data[rank].cpuArch = comm->cpuArch;
  allGather3Data[rank].cpuVendor = comm->cpuVendor;
  allGather3Data[rank].p2pnChannelsPerPeer = comm->p2pnChannelsPerPeer;
  NCCLCHECKGOTO(ncclTopoCheckNicFused(comm, &allGather3Data[rank].nicFused), ret, fail);

  comm->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  NCCLCHECKGOTO(ncclTopoPreset(comm, graphs, &allGather3Data[rank].topoRanks), ret, fail);

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)), ret, fail);

  // Determine nNodes, firstRanks, ...
  NCCLCHECKGOTO(ncclCalloc(&nodesFirstRank, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nodesTreePatterns, nranks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToNode, comm->nRanks), ret, fail);
  for (int r=0; r<nranks; r++) {
    int node;
    int firstRank = allGather3Data[r].topoRanks.ringRecv[0];
    for (node=0; node<comm->nNodes && nodesFirstRank[node] != firstRank; node++);
    if (node == comm->nNodes) {
      comm->nNodes++;
      nodesFirstRank[node] = firstRank;
      // Record tree pattern of each node as they can be different depending on sm arch
      nodesTreePatterns[node] = allGather3Data[r].graphInfo[NCCL_ALGO_TREE].pattern;
    }
    comm->rankToNode[r] = node;

    if (comm->cpuArch != allGather3Data[r].cpuArch &&
        comm->cpuArch != NCCL_TOPO_CPU_ARCH_MIXED) {
      comm->cpuArch = NCCL_TOPO_CPU_ARCH_MIXED;
    }
    if (comm->cpuVendor != allGather3Data[r].cpuVendor &&
        comm->cpuVendor != NCCL_TOPO_CPU_VENDOR_MIXED) {
      comm->cpuVendor = NCCL_TOPO_CPU_VENDOR_MIXED;
    }
    if (allGather3Data[r].nicFused) {
      globalNicFused = true;
    }
  }

  // Alert the user to the presence of mixed CPUs. In the past this has caused
  // locks in some collective routines. This may help debug issues in the future.
  if (rank==0) {
    if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed architecture were detected.");
    }
    if (comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_MIXED) {
      INFO(NCCL_GRAPH, "CPUs with mixed vendors were detected.");
    }
  }

  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  comm->minLocalRanks = INT_MAX;
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
    comm->minLocalRanks = std::min(comm->minLocalRanks, comm->nodeRanks[n].localRanks);
    comm->nodeRanks[n].localRanks = 0;
  }
  // And fill the ranks arrays
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->nodeRanks[node].localRankToRank[comm->nodeRanks[node].localRanks++] = r;
  }
  comm->node = comm->rankToNode[rank];
  comm->localRankToRank = comm->nodeRanks[comm->node].localRankToRank;
  comm->localRank = comm->rankToLocalRank[rank];
  comm->localRanks = comm->nodeRanks[comm->node].localRanks;

  NCCLCHECKGOTO(initNvlDomainInfo(comm), ret, fail);

  TRACE(NCCL_INIT,"hostHash[%d] %lx localRank %d localRanks %d localRank0 %d",
        rank, comm->peerInfo[rank].hostHash, comm->localRank, comm->localRanks, comm->localRankToRank[0]);
  if (comm->localRank == -1 || comm->localRankToRank[0] == -1 || comm->localRanks == 0) {
    WARN("Failed to determine local ranks rank %d hostHash %lx pidHash %lx localRank %d localRanks %d localRank0 %d",
         rank, comm->peerInfo[rank].hostHash, comm->peerInfo[rank].pidHash,
         comm->localRank, comm->localRanks, comm->localRankToRank[0]);
    ret = ncclInternalError;
    goto fail;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d nNodes %d localRanks %d localRank %d MNNVL %d",
       comm, rank, comm->nRanks, comm->nNodes, comm->localRanks, comm->localRank, comm->MNNVL);

  nChannelsOrig = comm->nChannels;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  for (int i=0; i<nranks; i++) {
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    for (int a=0; a<NCCL_NUM_ALGORITHMS; a++) {
      graphs[a]->nChannels = std::min(allGather3Data[i].graphInfo[a].nChannels, graphs[a]->nChannels);
      graphs[a]->sameChannels = std::min(allGather3Data[i].graphInfo[a].sameChannels, graphs[a]->sameChannels);
      graphs[a]->bwIntra = std::min(allGather3Data[i].graphInfo[a].bwIntra, graphs[a]->bwIntra);
      graphs[a]->bwInter = std::min(allGather3Data[i].graphInfo[a].bwInter, graphs[a]->bwInter);
      graphs[a]->typeIntra = std::max(allGather3Data[i].graphInfo[a].typeIntra, graphs[a]->typeIntra);
      graphs[a]->typeInter = std::max(allGather3Data[i].graphInfo[a].typeInter, graphs[a]->typeInter);
      graphs[a]->crossNic = std::max(allGather3Data[i].graphInfo[a].crossNic, graphs[a]->crossNic);
    }
    comm->maxTreePattern = std::max(comm->maxTreePattern, allGather3Data[i].graphInfo[NCCL_ALGO_TREE].pattern);
    comm->p2pnChannelsPerPeer = std::min(comm->p2pnChannelsPerPeer, allGather3Data[i].p2pnChannelsPerPeer);
  }
  if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->config.collnetEnable = 0;
  if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = comm->nvlsChannels = 0;

  comm->nChannels = treeGraph->nChannels = ringGraph->nChannels = std::min(treeGraph->nChannels, ringGraph->nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->config.collnetEnable == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->config.collnetEnable = 0;
    }
  }
  NCCLCHECK(ncclTopoPathAllNVLink(comm->topo, &comm->isAllNvlink));
  comm->isOneRPN = (comm->maxLocalRanks == 1);

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs, parent), ret, fail);
  // AllGather3 - end
  timers[TIMER_INIT_ALLGATHER] += clockNano() - timers[TIMER_INIT_CONNECT];

  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d TREES/RINGS", rank, nranks, comm->nChannels);

  char line[1024];
  line[0]='\0';
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclTree* tree = &comm->channels[c].tree;
    snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d/%d/%d->%d->%d",
        c, tree->down[0], tree->down[1], tree->down[2], rank, tree->up);
    INFO(NCCL_GRAPH, "Ring %02d : %d -> %d -> %d", c, comm->channels[c].ring.prev, comm->rank, comm->channels[c].ring.next);
  }
  line[1023] = '\0';
  INFO(NCCL_INIT, "Trees%s", line);

  NCCLCHECKGOTO(computeBuffSizes(comm), ret, fail);
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  /* until now, all info of comm should be known. We can initialize shared resources and
   * map localRanks to top parent local ranks. NOTE: this shareRes init must be put before
   * all proxy operations. */
  if (comm->sharedRes->owner == comm) {
    comm->sharedRes->tpNLocalRanks = comm->localRanks;
    comm->sharedRes->magic = comm->magic;
    comm->sharedRes->tpNChannels = comm->nChannels;
    comm->sharedRes->tpP2pNChannels = comm->p2pnChannels;
    memcpy(comm->sharedRes->tpRankToLocalRank, comm->rankToLocalRank, sizeof(int) * comm->nRanks);
  }
  NCCLCHECKGOTO(ncclCalloc(&topParentLocalRanks, comm->localRanks), ret, fail);
  for (int i = 0; i < comm->localRanks; ++i) {
    int tpRank = comm->topParentRanks[comm->localRankToRank[i]];
    topParentLocalRanks[i] = comm->sharedRes->tpRankToLocalRank[tpRank];
  }
  comm->topParentLocalRanks = topParentLocalRanks;

  // Profiler plugin context has to be initialized before proxy thread
  NCCLCHECK(ncclProfilerPluginInit(comm));

  NCCLCHECKGOTO(ncclTransportCheckP2pType(comm, &comm->isAllDirectP2p, &comm->directMode, &comm->isAllCudaP2p), ret, fail);
  // Launch proxy service thread, after this, the proxy calls can be used.
  if (parent && parent->shareResources) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }
  NCCLCHECKGOTO(ncclCalloc(&comm->gproxyConn, comm->nRanks), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano();
  // Build p2p schedule
  comm->p2pSchedule = ncclMemoryStackAlloc<ncclComm::P2pSchedulePair>(&comm->memPermanent, comm->nRanks);
  comm->planner.peers = ncclMemoryStackAlloc<ncclKernelPlanner::Peer>(&comm->memPermanent, comm->nRanks);
  NCCLCHECK(ncclP2pSchedule(comm));
  // initialize non-zero fields.
  comm->planner.bcast_info.minBcastPeer = INT_MAX;
  comm->planner.bcast_info.maxBcastPeer = INT_MIN;

  if (comm->config.numRmaCtx > 0) {
    comm->planner.rmaTaskQueues = ncclMemoryStackAlloc<ncclIntruQueue<ncclTaskRma, &ncclTaskRma::next>>(&comm->memPermanent, comm->config.numRmaCtx);
    for (int i = 0; i < comm->config.numRmaCtx; i++) {
      ncclIntruQueueConstruct(&comm->planner.rmaTaskQueues[i]);
    }
  } else {
    comm->planner.rmaTaskQueues = NULL;
  }

  comm->runtimeConn = comm->cuMemSupport && ncclParamRuntimeConnect();
  if (comm->runtimeConn) {
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    // Attempt to setup NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    // Check if we can setup CollNet
    if (comm->config.collnetEnable) ncclCollNetSetup(comm, parent, graphs);
  } else {
    for (int c=0; c<comm->nChannels; c++) {
      NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportRingConnect(comm), ret, fail);

    // Connect Trees
    NCCLCHECKGOTO(ncclTransportTreeConnect(comm), ret, fail);

    // Connect PAT only for communicators with 1 GPU per node
    if (comm->maxLocalRanks == 1) NCCLCHECKGOTO(ncclTransportPatConnect(comm), ret, fail);

    // Attempt to setup NVLS
    NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
    NCCLCHECKGOTO(ncclNvlsBufferSetup(comm), ret, fail);

    // And NVLS trees if needed
    NCCLCHECKGOTO(ncclNvlsTreeConnect(comm), ret, fail);

    // Check if we can setup CollNet
    if (comm->config.collnetEnable) {
      ncclCollNetSetup(comm, parent, graphs);
      NCCLCHECKGOTO(ncclCollNetChainBufferSetup(comm), ret, fail);
      if (comm->maxLocalRanks <= NCCL_MAX_DIRECT_ARITY+1) {
        NCCLCHECKGOTO(ncclCollNetDirectBufferSetup(comm), ret, fail);
      }
    }

    // Connect to local net proxy
    NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
    NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

    // Then to remote ones when using PXN
    if (ncclPxnDisable(comm) == 0) {
      int nranks;
      NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
      for (int r=0; r<nranks; r++) {
        NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
        NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
      }
    }

    if (ncclParamNvbPreconnect()) {
      // Connect p2p when using NVB path
      int nvbNpeers;
      NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
      for (int r=0; r<nvbNpeers; r++) {
        int peer = nvbPeers[r];
        int sendRound=0, recvRound=0;
        while (comm->p2pSchedule[sendRound].sendRank != peer) sendRound++;
        while (comm->p2pSchedule[recvRound].recvRank != peer) recvRound++;
        uint8_t sendBase = ncclP2pChannelBaseForRound(comm, sendRound);
        uint8_t recvBase = ncclP2pChannelBaseForRound(comm, recvRound);
        for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
          int channelId;
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, sendBase, c);
          if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
            comm->connectSend[peer] |= (1UL<<channelId);
          }
          channelId = ncclP2pChannelForPart(comm->p2pnChannels, recvBase, c);
          if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
            comm->connectRecv[peer] |= (1UL<<channelId);
          }
        }
      }

      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
    }
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoInitTunerConstants(comm), ret, fail);
  NCCLCHECKGOTO(ncclTunerPluginLoad(comm), ret, fail);
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(&comm->tunerContext, comm->commHash, comm->nRanks, comm->nNodes, ncclDebugLog, &comm->nvlDomainInfo, &comm->tunerConstants));
  }
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    const char* str = ncclGetEnv("NCCL_LAUNCH_MODE");
    enum ncclLaunchMode mode, modeOld;
    if (str && strcasecmp(str, "GROUP") == 0) {
      mode = ncclLaunchModeGroup;
    } else {
      mode = ncclLaunchModeParallel;
    }
    // In theory we could be racing with other communicators not associated with
    // this one if the user is connecting to multiple ncclUniqueId's concurrently.
    modeOld = COMPILER_ATOMIC_EXCHANGE(&ncclParamLaunchMode, mode, std::memory_order_relaxed);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  bool crossNicSupported;
  NCCLCHECKGOTO(ncclTopoPathAllDirectNVLink(comm->topo, &comm->isAllDirectNvlink), ret, fail);
  NCCLCHECKGOTO(ncclTopoCheckCrossNicSupport(&crossNicSupported), ret, fail);
  comm->ginSupport = comm->sharedRes->ginState.ncclGin != nullptr && crossNicSupported && !globalNicFused;
  comm->rmaProxySupport = comm->rmaState.rmaProxyState.ncclGin != nullptr && crossNicSupported && !globalNicFused;
  comm->symmetricSupport = comm->isAllCudaP2p && ncclParamWinEnable() && ncclCuMemEnable() && (comm->ginSupport || comm->nNodes == 1);
  if (!comm->symmetricSupport) {
    INFO(NCCL_INIT, "Symmetric memory is not supported. cuMemEnable %d, "
      "ginSupport %d, globalNicFused %d", ncclCuMemEnable(), comm->ginSupport, globalNicFused);
  }
  comm->devrState.bigSize = 0;

  comm->ceColl.baseUCSymReadyPtr = NULL;
  comm->ceColl.baseUCSymComplPtr = NULL;

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  timers[TIMER_INIT_CONNECT] = clockNano() -  timers[TIMER_INIT_CONNECT];
  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapIntraNodeBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  if (ncclOsCpuCount(comm->cpuAffinity)) ncclOsSetAffinity(comm->cpuAffinity);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  if (comm->sharedRes->owner == comm && !comm->shareResources && ret == ncclSuccess && !ncclCuMemEnable()) ncclProxyShmUnlink(comm);
  free(allTopoRanks);
  free(nodesTreePatterns);
  free(nodesFirstRank);
  free(allGather3Data);
  free(rings);
  free(nvbPeers);
  free(pxnPeers);
  return ret;
fail:
  goto exit;
}

NCCL_PARAM(SetStackSize, "SET_STACK_SIZE", 0);
NCCL_PARAM(CGAClusterSize, "CGA_CLUSTER_SIZE", NCCL_CONFIG_UNDEF_INT);
// Match config max/minCTAs
NCCL_PARAM(MaxCTAs, "MAX_CTAS", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(MinCTAs, "MIN_CTAS", NCCL_CONFIG_UNDEF_INT);
#define NCCL_MAX_CGA_CLUSTER_SIZE 8

NCCL_PARAM(NChannelsPerNetPeer, "NCHANNELS_PER_NET_PEER", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(NvlinkUtilCentricSchedEnable, "NVLINK_UTIL_CENTRIC_SCHED_ENABLE", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", NCCL_CONFIG_UNDEF_INT)

#define NCCL_COMMINIT_FUNCNAME_LEN 128
struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // For ncclCommInitRank
  int nranks, myrank, nId;
  ncclUniqueId* commId;
  // for ncclCommSplit
  struct ncclComm* parent;
  int color, key;
  int childCount;
  // For Shrink
  int* excludeRanksList;
  int excludeRanksCount;
  // name of the function calling
  char funcName[NCCL_COMMINIT_FUNCNAME_LEN];
  // For grow operations
  bool isGrow;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static void ncclCommFinalizeAsyncJobFree(void* _job) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*)_job;
  delete job;
}

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);
NCCL_PARAM(CommShrinkShareResources, "COMM_SHRINK_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

typedef struct{
  int key;
  int color;
} commSplitInfo;
static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  commSplitInfo* info = NULL;
  NCCLCHECKGOTO(ncclCalloc(&info, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  info[parent->rank].color = color;
  info[parent->rank].key = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, info, sizeof(commSplitInfo)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (info[i].color != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && info[parentRanksRet[insert]].key <= info[i].key) insert++;
    // Shift ranks by one after insert
    for (int r = nRanks; r > insert; r--) parentRanksRet[r] = parentRanksRet[r - 1];
    // Insert our rank
    parentRanksRet[insert] = i;
    nRanks++;
  }

  for (int i = 0; i < nRanks; i++) {
    if (parentRanksRet[i] == parent->rank) myRank = i;
  }

  *nRanksRet = nRanks;
  *myRankRet = myRank;

exit:
  free(info);
  return ret;
fail:
  goto exit;
}

static ncclResult_t getParentRanks(int parentRanks, int parentRank, int* excludeRanksList, int excludeRanksCount, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int count = 0, j = 0;
  for (int i = 0; i < parentRanks; i++) {
    // we assume excludeRanksList is sorted
    if (j < excludeRanksCount && excludeRanksList[j] == i) {
      j++;
      continue;
    }
    if (i == parentRank) *myRankRet = count;
    parentRanksRet[count++] = i;
  }
  *nRanksRet = parentRanks - excludeRanksCount;
  return ncclSuccess;
}

static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  int cudaDev = job->cudaDev;
  int* parentRanks = NULL;
  int cudaArch;
  int maxSharedMem = 0;
  double sum_timers = 0;
  uint64_t timers[TIMERS_INIT_COUNT] = {0};
  unsigned long long commIdHash;

  timers[TIMER_INIT_TOTAL] = clockNano();
  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&maxSharedMem, cudaDevAttrMaxSharedMemoryPerBlockOptin, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  cudaArch = 100*archMajor + 10*archMinor;

  timers[TIMER_INIT_KERNELS] = clockNano();
  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, maxSharedMem, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zu", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  timers[TIMER_INIT_KERNELS] = clockNano() - timers[TIMER_INIT_KERNELS];

  if (job->parent && !job->isGrow) {
    // SPLIT/SHRINK: use bootstrapSplit
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    if (job->excludeRanksCount) {
      NCCLCHECKGOTO(getParentRanks(job->parent->nRanks, job->parent->rank, job->excludeRanksList, job->excludeRanksCount, &job->nranks, &job->myrank, parentRanks), res, fail);
    } else {
      NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
      // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
      if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    }
    // child hash obtained from (parent hash, split count, color)
    uint64_t hacc[2] = {1, 1};
    eatHash(hacc, &job->parent->commHash);
    eatHash(hacc, &job->childCount);
    eatHash(hacc, &job->color);
    comm->commHash = digestHash(hacc);
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];
    comm->isGrow = false;
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p childCount %d color %d key %d- Init START", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->childCount, job->color, job->key);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapSplit(comm->commHash, comm, job->parent, job->color, job->key, parentRanks), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
    // debug info, no commId was used
    commIdHash = 0;
  } else {
    // GROW or NORMAL INIT: use bootstrapInit
    if (job->isGrow) {
      struct ncclBootstrapHandle* growHandle = (struct ncclBootstrapHandle*)job->commId;
      uint64_t baseMagic = growHandle ? growHandle->magic : hashCombine(job->parent->magic, job->parent->childCount);
      comm->commHash = commIdHash = hashCombine(baseMagic, job->nranks);
      INFO(NCCL_INIT, "Rank %d: Generated commHash 0x%lx from baseMagic 0x%lx and newNRanks %d",
           job->myrank, comm->commHash, baseMagic, job->nranks);
    } else {
      // obtain a unique hash using the first commId
      comm->commHash = commIdHash = getHash(job->commId->internal, NCCL_UNIQUE_ID_BYTES);
    }
    timers[TIMER_INIT_ALLOC] = clockNano();
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    timers[TIMER_INIT_ALLOC] = clockNano() - timers[TIMER_INIT_ALLOC];

    comm->isGrow = job->isGrow;
    INFO(NCCL_INIT, "[Rank %d] %s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START",
         job->myrank, job->funcName, comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano();
    NCCLCHECKGOTO(bootstrapInit(job->nId, (struct ncclBootstrapHandle*)job->commId, comm, job->parent), res, fail);
    timers[TIMER_INIT_BOOTSTRAP] = clockNano() - timers[TIMER_INIT_BOOTSTRAP];
  }
  comm->cudaArch = cudaArch;

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent, timers), res, fail);

  // update communicator state
  comm->initState = ncclSuccess;
  timers[TIMER_INIT_TOTAL] = clockNano() - timers[TIMER_INIT_TOTAL];

  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    COMPILER_ATOMIC_STORE(&job->parent->childAbortFlag, NULL, std::memory_order_release);
    if (job->isGrow) {
      TRACE_CALL("ncclCommGrow(%p, %d, %p)", job->parent, comm->nRanks, comm);
    } else {
      TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)", job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
    }
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx parent %p childCount %d color %d key %d - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, job->parent, job->childCount, job->color, job->key);
  } else {
    // the name for the replay tool is ncclCommInitRank for all the variations
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)", comm, comm->nRanks, commIdHash, comm->rank, comm->cudaDev);
    INFO(NCCL_INIT, "%s comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", job->funcName,
         comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, commIdHash);
  }
  sum_timers = 0.0;
  for (int it = 1; it < TIMERS_INIT_COUNT; ++it)
    sum_timers += (timers[it] / 1e9);
  INFO(NCCL_INIT | NCCL_PROFILE,
       "Init timings - %s: rank %d nranks %d total %.2f (kernels %.2f, alloc %.2f, bootstrap %.2f, allgathers %.2f, topo %.2f, graphs %.2f, "
       "connections %.2f, rest %.2f)",
       job->funcName, comm->rank, comm->nRanks,
       timers[TIMER_INIT_TOTAL] / 1e9, timers[TIMER_INIT_KERNELS] / 1e9, timers[TIMER_INIT_ALLOC] / 1e9,
       timers[TIMER_INIT_BOOTSTRAP] / 1e9, timers[TIMER_INIT_ALLGATHER] / 1e9, timers[TIMER_INIT_TOPO] / 1e9,
       timers[TIMER_INIT_GRAPHS] / 1e9, timers[TIMER_INIT_CONNECT] / 1e9, timers[TIMER_INIT_TOTAL] / 1e9 - sum_timers);
exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    COMPILER_ATOMIC_STORE(job->newcomm, comm, std::memory_order_release);
  }
  if (parentRanks) free(parentRanks);
  return res;
fail:
  comm->initState = res;
  goto exit;
}

#define NCCL_CONFIG_DEFAULT(config, field, undef, defvalue, fieldStr, format) \
  if (config->field == undef) { \
    config->field = defvalue; \
  } else { \
    INFO(NCCL_ENV, "Comm config " fieldStr " set to " format, config->field); \
  }

static ncclResult_t envConfigOverride(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  const char* tmpNetName = comm->config.netName;
  const char* envNetName;
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  int splitShareEnv;
  const char* collnetEnableEnv;
  int shrinkShareEnv;
  int nvlsCTAsEnv;
  int nChannelsPerNetPeerEnv;
  int nvlinkUtilCentricSchedEnableEnv;
  int graphMixingSupportEnv;
  int numRmaCtxEnv;

  /* override configuration with env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    if (comm->config.cgaClusterSize != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config cgaClusterSize reset to NCCL_MAX_CGA_CLUSTER_SIZE=%d", cgaClusterSizeEnv);
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    INFO(NCCL_ENV, "NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (minCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MIN_CTAS %d is too low, leaving it set at %d", minCTAsEnv, comm->config.minCTAs);
    else {
      if (comm->config.minCTAs != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config minCTAs reset to NCCL_MIN_CTAS=%d", minCTAsEnv);
      comm->config.minCTAs = minCTAsEnv;
    }
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (maxCTAsEnv <= 0)
      INFO(NCCL_ENV, "NCCL_MAX_CTAS %d is too low, leaving it set at %d", maxCTAsEnv, comm->config.maxCTAs);
    else {
      if (comm->config.maxCTAs != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config maxCTAs reset to NCCL_MAX_CTAS=%d", maxCTAsEnv);
      comm->config.maxCTAs = maxCTAsEnv;
    }
  }

  /* override configuration with env variable. */
  nChannelsPerNetPeerEnv = ncclParamNChannelsPerNetPeer();
  if (nChannelsPerNetPeerEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nChannelsPerNetPeerEnv <= 0)
      INFO(NCCL_ENV, "NCCL_NCHANNELS_PER_NET_PEER %d is too low, leaving it set at %d", nChannelsPerNetPeerEnv, comm->config.nChannelsPerNetPeer);
    else {
      if (comm->config.nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config nChannelsPerNetPeer reset to NCCL_NCHANNELS_PER_NET_PEER=%d", nChannelsPerNetPeerEnv);
      comm->config.nChannelsPerNetPeer = nChannelsPerNetPeerEnv;
    }
  }

  nvlinkUtilCentricSchedEnableEnv = ncclParamNvlinkUtilCentricSchedEnable();
  if (nvlinkUtilCentricSchedEnableEnv != NCCL_CONFIG_UNDEF_INT) {
    if (nvlinkUtilCentricSchedEnableEnv != 0 && nvlinkUtilCentricSchedEnableEnv != 1)
      INFO(NCCL_ENV, "NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE %d is not valid, leaving it set at %d", nvlinkUtilCentricSchedEnableEnv, comm->config.nvlinkCentricSched);
    else {
      if (comm->config.nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config nvlinkCentricSched reset to NCCL_NVLINK_UTIL_CENTRIC_SCHED_ENABLE=%d", nvlinkUtilCentricSchedEnableEnv);
      comm->config.nvlinkCentricSched = nvlinkUtilCentricSchedEnableEnv;
    }
  }

  graphMixingSupportEnv = ncclParamGraphMixingSupport();
  if (graphMixingSupportEnv != NCCL_CONFIG_UNDEF_INT) {
    if (graphMixingSupportEnv != 0 && graphMixingSupportEnv != 1) {
      INFO(NCCL_ENV, "NCCL_GRAPH_MIXING_SUPPORT %d is not valid, leaving it set at %d", graphMixingSupportEnv, comm->config.graphUsageMode);
    } else {
      if (comm->config.graphUsageMode != NCCL_CONFIG_UNDEF_INT) {
        INFO(NCCL_ENV, "Comm config graphUsageMode reset to %d by NCCL_GRAPH_MIXING_SUPPORT=%d", graphMixingSupportEnv == 1 ? 2 : 0, graphMixingSupportEnv);
      }
      comm->config.graphUsageMode = graphMixingSupportEnv == 1 ? 2 : 0;
    }
  }

  numRmaCtxEnv = ncclParamNumRmaCtx();
  if (numRmaCtxEnv != NCCL_CONFIG_UNDEF_INT) {
    if (numRmaCtxEnv <= 0)
      INFO(NCCL_ENV, "NCCL_NUM_RMA_CTX %d is too low, leaving it set at %d", numRmaCtxEnv, comm->config.numRmaCtx);
    else
      comm->config.numRmaCtx = numRmaCtxEnv;
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR)
      INFO(NCCL_ENV, "Comm config netName reset to NCCL_NET=%s", tmpNetName);
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.splitShare != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config splitShare reset to NCCL_COMM_SPLIT_SHARE_RESOURCES=%d", splitShareEnv);
    comm->config.splitShare = splitShareEnv;
  }
  shrinkShareEnv = ncclParamCommShrinkShareResources();
  if (shrinkShareEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.shrinkShare != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config shrinkShare reset to NCCL_COMM_SHRINK_SHARE_RESOURCES=%d", shrinkShareEnv);
    comm->config.shrinkShare = shrinkShareEnv;
  }

  // NCCL_COLLNET_ENABLE needs to be reloaded each time for comm init
  // since users might change the env on the fly to enable/disable collnet
  collnetEnableEnv = ncclGetEnv("NCCL_COLLNET_ENABLE");
  if (collnetEnableEnv != NULL) {
    int collnetEnableInt = (int)strtol(collnetEnableEnv, NULL, 0);
    if (collnetEnableInt != NCCL_CONFIG_UNDEF_INT) {
      if (comm->config.collnetEnable != NCCL_CONFIG_UNDEF_INT)
        INFO(NCCL_ENV, "Comm config collnetEnable reset to NCCL_COLLNET_ENABLE=%d", collnetEnableInt);
      comm->config.collnetEnable = collnetEnableInt;
      INFO(NCCL_ENV, "NCCL_COLLNET_ENABLE set by environment to %d.", collnetEnableInt);
    }
  }

  static pthread_once_t onceEnvCtaPolicy = PTHREAD_ONCE_INIT;
  pthread_once(&onceEnvCtaPolicy, getEnvCtaPolicyOnce);
  if (ctaPolicyEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.CTAPolicy != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config CTAPolicy reset to NCCL_CTA_POLICY=%d", ctaPolicyEnv);
    comm->config.CTAPolicy = ctaPolicyEnv;
  }

  nvlsCTAsEnv = ncclParamNvlsChannels();
  if (nvlsCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT)
      INFO(NCCL_ENV, "Comm config nvlsCTAs reset to NCCL_NVLS_NCHANNELS=%d", nvlsCTAsEnv);
    comm->config.nvlsCTAs = nvlsCTAsEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    INFO(NCCL_ENV, "maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    INFO(NCCL_ENV, "minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    INFO(NCCL_ENV, "splitShare %d is not a valid value 0/1, set it to 0", comm->config.splitShare);
    comm->config.splitShare = 0;
  }

  if (comm->config.collnetEnable != 1 && comm->config.collnetEnable != 0) {
    INFO(NCCL_ENV, "collnetEnable %d is not a valid value 0/1, set it to 0", comm->config.collnetEnable);
    comm->config.collnetEnable = 0;
  }

  if (!ctaPolicyIsValid(comm->config.CTAPolicy)) {
    INFO(NCCL_ENV, "CTAPolicy %d is not a valid value, set it to %d", comm->config.CTAPolicy, NCCL_CTA_POLICY_DEFAULT);
    comm->config.CTAPolicy = NCCL_CTA_POLICY_DEFAULT;
  }

  if (comm->config.nvlsCTAs != NCCL_CONFIG_UNDEF_INT && comm->config.nvlsCTAs <= 0) {
    INFO(NCCL_ENV, "nvlsCTAs %d is not a valid value, NCCL will decide the default value automatically", comm->config.nvlsCTAs);
    comm->config.nvlsCTAs = NCCL_CONFIG_UNDEF_INT;
  }

  // If POLICY_ZERO and POLICY_EFFICIENCY are set in CTAPolicy, unset POLICY_EFFICIENCY.
  if ((comm->config.CTAPolicy & NCCL_CTA_POLICY_ZERO) &&
      (comm->config.CTAPolicy & NCCL_CTA_POLICY_EFFICIENCY)) {
    WARN("Both NCCL_CTA_POLICY_ZERO and NCCL_CTA_POLICY_EFFICIENCY are set in CTAPolicy (%d). Unsetting POLICY_EFFICIENCY.", comm->config.CTAPolicy);
    comm->config.CTAPolicy &= ~NCCL_CTA_POLICY_EFFICIENCY;
  }

  return ret;
}

static ncclResult_t copyCommConfig(ncclComm_t childComm, ncclComm_t parnet) {
  memcpy(&childComm->config, &parnet->config, sizeof(ncclConfig_t));
  NCCLCHECK(envConfigOverride(childComm));
  return ncclSuccess;
}

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  /* config must not be NULL in this function */
  ncclConfig_t defaultConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr;
  size_t realSize;

  internalConfig.magic = 0;
  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != NCCL_API_MAGIC) {
      WARN("ncclConfig_t argument not initialized via NCCL_CONFIG_INITIALIZER");
      ret = ncclInvalidArgument;
      goto fail;
    }

    /* check version. */
    if (internalConfigPtr->version < NCCL_VERSION(2, 14, 0)) {
      internalConfigPtr->blocking = defaultConfig.blocking;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 17, 0)) {
      internalConfigPtr->cgaClusterSize = defaultConfig.cgaClusterSize;
      internalConfigPtr->minCTAs = defaultConfig.minCTAs;
      internalConfigPtr->maxCTAs = defaultConfig.maxCTAs;
      internalConfigPtr->netName = defaultConfig.netName;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 25, 0)) {
      internalConfigPtr->trafficClass = defaultConfig.trafficClass;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 27, 0)) {
      internalConfigPtr->collnetEnable = defaultConfig.collnetEnable;
      internalConfigPtr->CTAPolicy = defaultConfig.CTAPolicy;
      internalConfigPtr->shrinkShare = defaultConfig.shrinkShare;
      internalConfigPtr->nvlsCTAs = defaultConfig.nvlsCTAs;
    }
    if (internalConfigPtr->version < NCCL_VERSION(2, 28, 0)) {
      internalConfigPtr->nChannelsPerNetPeer = defaultConfig.nChannelsPerNetPeer;
      internalConfigPtr->nvlinkCentricSched = defaultConfig.nvlinkCentricSched;
    }

    if (internalConfigPtr->version < NCCL_VERSION(2, 29, 0)) {
      internalConfigPtr->graphUsageMode = defaultConfig.graphUsageMode;
      internalConfigPtr->numRmaCtx = defaultConfig.numRmaCtx;
    }
  }

  /* check input config attributes, -1 means user-undefined and we should use default value from NCCL. */
  if (internalConfigPtr->blocking != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->blocking != 0 && internalConfigPtr->blocking != 1) {
    WARN("Invalid config blocking attribute value %d", internalConfigPtr->blocking);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->cgaClusterSize != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->cgaClusterSize < 0) {
    WARN("Invalid config cgaClusterSize attribute value %d", internalConfigPtr->cgaClusterSize);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if ((internalConfigPtr->minCTAs != NCCL_CONFIG_UNDEF_INT &&
    internalConfigPtr->minCTAs <= 0) ||
    (internalConfigPtr->maxCTAs != NCCL_CONFIG_UNDEF_INT &&
      internalConfigPtr->maxCTAs <= 0) ||
    (internalConfigPtr->minCTAs > internalConfigPtr->maxCTAs)) {
    WARN("Invalid config min/max channels attribute value %d/%d", internalConfigPtr->minCTAs, internalConfigPtr->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->splitShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->splitShare != 0 && internalConfigPtr->splitShare != 1) {
    WARN("Invalid config splitShare attribute value %d", internalConfigPtr->splitShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->collnetEnable != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->collnetEnable < 0 || internalConfigPtr->collnetEnable > 1)) {
    WARN("Invalid config collnetEnable attribute value %d", internalConfigPtr->collnetEnable);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->CTAPolicy != NCCL_CONFIG_UNDEF_INT) {
    if (!ctaPolicyIsValid(internalConfigPtr->CTAPolicy)) {
      WARN("Invalid config policy attribute value %d", internalConfigPtr->CTAPolicy);
      ret = ncclInvalidArgument;
      goto fail;
    }
  }

  if (internalConfigPtr->shrinkShare != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->shrinkShare != 0 && internalConfigPtr->shrinkShare != 1) {
    WARN("Invalid config shrinkShare attribute value %d", internalConfigPtr->shrinkShare);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlsCTAs != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlsCTAs <= 0) {
    WARN("Invalid config nvlsCTAs attribute value %d", internalConfigPtr->nvlsCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nChannelsPerNetPeer != NCCL_CONFIG_UNDEF_INT && (internalConfigPtr->nChannelsPerNetPeer <= 0 || internalConfigPtr->nChannelsPerNetPeer > MAXCHANNELS)) {
    WARN("Invalid config nChannelsPerNetPeer attribute value %d", internalConfigPtr->nChannelsPerNetPeer);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->nvlinkCentricSched != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->nvlinkCentricSched != 0 && internalConfigPtr->nvlinkCentricSched != 1) {
    WARN("Invalid config nvlinkCentricSched attribute value %d", internalConfigPtr->nvlinkCentricSched);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->graphUsageMode != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->graphUsageMode != 0 && internalConfigPtr->graphUsageMode != 1 && internalConfigPtr->graphUsageMode != 2) {
    WARN("Invalig config graphUsageMode attribute value %d", internalConfigPtr->graphUsageMode);
    ret = ncclInvalidArgument;
    goto fail;
  }

  if (internalConfigPtr->numRmaCtx != NCCL_CONFIG_UNDEF_INT && internalConfigPtr->numRmaCtx <= 0) {
    WARN("Invalid config numRmaCtx attribute value %d", internalConfigPtr->numRmaCtx);
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, trafficClass, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "Traffic class", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, commName, NCCL_CONFIG_UNDEF_PTR, NULL, "Comm name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, collnetEnable, NCCL_CONFIG_UNDEF_INT, 0, "Collnet enable", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, CTAPolicy, NCCL_CONFIG_UNDEF_INT, NCCL_CTA_POLICY_DEFAULT, "CTA policy flags", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, shrinkShare, NCCL_CONFIG_UNDEF_INT, 0, "shrinkShare", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlsCTAs, NCCL_CONFIG_UNDEF_INT, NCCL_CONFIG_UNDEF_INT, "nvlsCTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nChannelsPerNetPeer, NCCL_CONFIG_UNDEF_INT,
                      NCCL_CONFIG_UNDEF_INT, "nChannelsPerNetPeer", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, nvlinkCentricSched, NCCL_CONFIG_UNDEF_INT, 0, "nvlinkCentricSched", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, graphUsageMode, NCCL_CONFIG_UNDEF_INT, 2, "graphUsageMode", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, numRmaCtx, NCCL_CONFIG_UNDEF_INT, 1, "numRmaCtx", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;
  comm->config.trafficClass = internalConfigPtr->trafficClass;
  comm->config.commName = internalConfigPtr->commName;
  comm->config.collnetEnable = internalConfigPtr->collnetEnable;
  comm->config.CTAPolicy = internalConfigPtr->CTAPolicy;
  comm->config.shrinkShare = internalConfigPtr->shrinkShare;
  comm->config.nvlsCTAs = internalConfigPtr->nvlsCTAs;
  comm->config.nChannelsPerNetPeer = internalConfigPtr->nChannelsPerNetPeer;
  comm->config.nvlinkCentricSched = internalConfigPtr->nvlinkCentricSched;
  comm->config.graphUsageMode = internalConfigPtr->graphUsageMode;
  comm->config.numRmaCtx = internalConfigPtr->numRmaCtx;
  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static void ncclCommInitJobFree(void* _job) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)_job;
  free(job->commId);
  delete job;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, int nId, ncclUniqueId* commId, int myrank, int cudaDev, ncclConfig_t *config, const char funcName[]) {
  if (nId <= 0 || nId > nranks) {
    WARN("improper usage of ncclCommInitRank: nId = %d, nranks=%d", nId, nranks);
    return ncclInvalidArgument;
  }
  ncclResult_t res = ncclSuccess;
  const char* commIdEnv = NULL;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob* job = NULL;
  bool launchedJob = false;
  // first call ncclInit, this will setup the environment
  NCCLCHECKGOTO(ncclInit(), res, fail);

  if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && myrank == 0)) {
    static std::once_flag once;
    std::call_once(once, showVersion);
  }
  // Make sure the CUDA runtime is initialized.
  CUDACHECKGOTO(cudaFree(NULL), res, fail);

  NCCLCHECKGOTO(PtrCheck(newcomm, "CommInitRank", "newcomm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(config, "CommInitRank", "config"), res, fail);
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    res = ncclInvalidArgument;
    goto fail;
  }

  NCCLCHECKGOTO(ncclCalloc(&comm, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->abortFlagRefCount, 1), res, fail);
  comm->startMagic = comm->endMagic = NCCL_MAGIC; // Used to detect comm corruption.
  *comm->abortFlagRefCount = 1;
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInProgress and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInProgress;
  *newcomm = comm;

  NEW_NOTHROW_GOTO(job, ncclCommInitRankAsyncJob, res, fail);
  job->nId = nId;
  job->comm = comm;
  job->nranks = nranks;
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", funcName);
  // need to copy the commIds to allow async commInit and to avoid alignement issues when casting from ncclUNiqueId and ncclBootstrapHandle
  // ncclUniqueIds and ncclBootstrapHandle don't have the same alignment requirements.
  // Therefore the array of Ids coming from the user might not be properly aligned to be cast into a ncclBootstrapHandle
  // copying into allocated memory guarantees that the memory is properly aligned for any objects, removing that issue
  NCCLCHECKGOTO(ncclCalloc(&job->commId, nId), res, fail);
  memcpy(job->commId, commId, nId * NCCL_UNIQUE_ID_BYTES);

  commIdEnv = ncclGetEnv("NCCL_COMM_ID");
  if (commIdEnv && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", commIdEnv);
    if (nId > 1) {
      INFO(NCCL_INIT | NCCL_ENV, "NCCL_COMM_ID cannot be used with more than one ncclUniqueId");
      job->nId = 1;
    }
    // start the bootstrap root before bootstrapping, use only the first handle
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&job->commId[0], true), res, fail);
  }
  launchedJob = true;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, ncclCommInitJobFree, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  if (job && !launchedJob) ncclCommInitJobFree(job);
  if (comm) {
    free(comm->abortFlag);
    if (comm->abortFlagDev) (void)ncclCudaHostFree((void*)comm->abortFlagDev);
    free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRank)
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECK(cudaGetDevice(&cudaDev));

  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, &config, __func__));

  NVTX3_RANGE_ADD_PAYLOAD(CommInitRank, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  int oldDev = 0;

  NVTX3_RANGE(NcclNvtxParamsCommInitAll);

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(PtrCheck(comms, "CommInitAll", "comms"), ret, fail);
  if (ndev < 0) {
    WARN("Invalid device count requested : %d", ndev);
    ret = ncclInvalidArgument;
    goto fail;
  }

  CUDACHECKGOTO(cudaGetDeviceCount(&totalnDev), ret, fail);
  if (devlist) {
    NCCLCHECKGOTO(ncclCalloc(&gpuFlags, totalnDev), ret, fail);
    for (int i = 0; i < ndev; ++i) {
      /* invalid device check. */
      if (devlist[i] < 0 || devlist[i] >= totalnDev) {
        WARN("Invalid device %d (totalnDev=%d)", devlist[i], totalnDev);
        ret = ncclInvalidArgument;
        goto fail;
      }

      /* duplicate device check. */
      if (gpuFlags[devlist[i]] != 0) {
        ret = ncclInvalidUsage;
        goto fail;
      }

      gpuFlags[devlist[i]] = 1;
    }
    free(gpuFlags);
    gpuFlags = nullptr;
  }

  ncclUniqueId uniqueId;
  NCCLCHECKGOTO(ncclGetUniqueId(&uniqueId), ret, fail);
  NCCLCHECKGOTO(ncclGroupStartInternal(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    int dev = devlist ? devlist[i] : i;
    CUDACHECKGOTO(cudaSetDevice(dev), ret, fail);
    ncclCommInitRankDev(comms+i, ndev,1, &uniqueId, i, dev, &config, __func__);
  }
  NCCLCHECKGOTO(ncclGroupEndInternal(), ret, fail);

  NVTX3_RANGE_ADD_PAYLOAD(CommInitAll, NcclNvtxParamsCommInitAllSchema,
    NVTX3_PAYLOAD(comms[0]->commHash, ndev));

exit:
  (void)cudaSetDevice(oldDev);
  free(gpuFlags);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  COMPILER_ATOMIC_STORE(&comm->asyncResult, nextState, std::memory_order_release);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;

  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankConfig);

  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, 1, &commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankConfig, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommInitRankScalable, ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config);
ncclResult_t ncclCommInitRankScalable(ncclComm_t* newcomm, int nranks, int myrank, int nId, ncclUniqueId* commId, ncclConfig_t* config) {
  NCCLCHECK(ncclInitEnv());
  NVTX3_RANGE(NcclNvtxParamsCommInitRankScalable);

  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECK(cudaGetDevice(&cudaDev));

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, nId, commId, myrank, cudaDev, internalConfigPtr, __func__), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm) {
    if (!(*newcomm)->config.blocking) {
      (void) ncclCommGetAsyncError(*newcomm, &ret);
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommInitRankScalable, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, nranks, myrank, cudaDev));
  }
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), ret, fail);

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->hostStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync hostStream error %d\n", comm, comm->rank, ret);
    }
    if ((ret = ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream)) != ncclSuccess) {
      WARN("commDestroySync: comm %p rank %d sync deviceStream error %d\n", comm, comm->rank, ret);
    }

    NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, true), ret, fail);
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
    // And keep polling until all graphs referencing us die.
    while (comm->localPersistentRefs != 0) {
      NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
    }
    while (!ncclIntruQueueEmpty(&comm->legacyRegCleanupQueue)) {
      struct ncclCommCallback* cb = ncclIntruQueueDequeue(&comm->legacyRegCleanupQueue);
      if (cb->fn(comm, cb) != ncclSuccess) {
        WARN("Legacy IPC cleanup callback failed comm %p (rank = %d) cb %p", comm, comm->rank, cb);
      }
    }
  }

  if ((ret = ncclProxyStop(comm)) != ncclSuccess) {
    WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, ret);
  }

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  CUDACHECK(cudaSetDevice(comm->cudaDev));
  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->finalize(comm->tunerContext));
    NCCLCHECK(ncclTunerPluginUnload(comm));
  }
  NCCLCHECK(commFree(comm));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommFinalize);

  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  comm->finalizeCalled = true;
  /* launch async thread to finalize comm. */
  NEW_NOTHROW_GOTO(job, ncclCommFinalizeAsyncJob, ret, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commDestroySync, nullptr, ncclCommFinalizeAsyncJobFree, comm), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommFinalize, NcclNvtxParamsCommFinalizeSchema,
      NVTX3_PAYLOAD(comm->commHash));
  }
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  ncclResult_t ret = ncclSuccess;

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int curRank; /* Debug info */
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = COMPILER_ATOMIC_ADD_FETCH(finalizeRankCnt, 1, std::memory_order_acq_rel);
    if (curRankCnt == intraRanks) {
      ncclComm_t curIntraComm;
      ncclComm_t nextIntraComm = intracomm0;

      /* this is  the last call to ncclCommDestroy/Abort, we need to make sure all comms
       * in the process have been finalized before we free local resources. */
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if (curIntraComm->finalizeCalled == false) {
          struct ncclCommFinalizeAsyncJob job;
          job.comm = curIntraComm;
          /* every comm aborts, commDestroySync should not be blocked. */
          if ((ret = commDestroySync((struct ncclAsyncJob*) &job)) != ncclSuccess)
            WARN("commReclaim: comm %p (rank = %d) in commDestroySync, error %d", curIntraComm, curRank, ret);
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          // We pass a freed pointer, but we don't dereference; we merely print its value, so it's OK.
          // coverity[pass_freed_arg]
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) {
    NCCL_NVTX3_FUNC_RANGE;
    return ncclSuccess;
  }

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_FUNC_WITH_PARAMS(CommDestroy, NcclNvtxParamsCommInitRank,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);
  NCCLCHECK(ncclGroupStartInternal());
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));
  NEW_NOTHROW_GOTO(job, ncclCommFinalizeAsyncJob, res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, nullptr, ncclCommFinalizeAsyncJobFree, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static ncclResult_t setCommAbortFlags(ncclComm_t comm, int value) {
  // Set abort flags
  if (comm->childAbortFlag != nullptr) {
    COMPILER_ATOMIC_STORE(comm->childAbortFlag, value, std::memory_order_release);
    COMPILER_ATOMIC_STORE(comm->childAbortFlagDev, value, std::memory_order_release);
  }
  COMPILER_ATOMIC_STORE(comm->abortFlag, value, std::memory_order_release);
  COMPILER_ATOMIC_STORE(comm->abortFlagDev, value, std::memory_order_release);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommRevoke, ncclComm_t comm, int revokeFlags);
struct ncclCommRevokeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t commRevokeAsync(struct ncclAsyncJob* job_) {
  struct ncclCommRevokeAsyncJob* job = (struct ncclCommRevokeAsyncJob*)job_;
  ncclComm_t comm = job->comm;
  ncclResult_t res = ncclSuccess;
  NCCLCHECKGOTO(PtrCheck(comm, "CommRevokeAsync", "comm"), res, exit);
  INFO(NCCL_INIT, "CommRevokeAsync START comm %p rank %d nRanks %d nNodes %d localRank %d cudaDev %d",
      comm, comm->rank, comm->nRanks, comm->nNodes, comm->localRank, comm->cudaDev);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream), res, exit);
  NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
  NCCLCHECKGOTO(ncclCommPollEventCallbacks(comm, /*waitSome=*/true), res, exit);
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/false), res, exit);
  {
    ncclResult_t _tmpret = ncclSuccess;
    if ((_tmpret = ncclProxyStop(comm)) != ncclSuccess) {
      WARN("ncclProxyStop: comm %p (rank = %d) destroys proxy resource error %d", comm, comm->rank, _tmpret);
    }
    if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread.joinable()) {
      comm->proxyState->thread.join();
      if (comm->proxyState->threadUDS.joinable()) {
        // UDS support
        comm->proxyState->threadUDS.join();
      }
    }
  }
  NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
exit:
  (void)ncclCommSetAsyncError(comm, res);
  INFO(NCCL_INIT, "CommRevokeAsync END comm %p result %d", comm, res);
  return res;
}

ncclResult_t ncclCommRevoke(ncclComm_t comm, int revokeFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommRevoke);

  if (comm == NULL) {
    return ncclSuccess;
  }
  // For now only NCCL_REVOKE_DEFAULT (0) is supported
  if (revokeFlags != NCCL_REVOKE_DEFAULT) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if destroy/finalize in progress
  if (comm->destroyFlag || comm->finalizeCalled) {
    return ncclInvalidArgument;
  }
  // Disallow revoke if revoke in progress
  if (comm->revokedFlag) {
    return ncclInvalidArgument;
  }
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke START",
      comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  (void)setCommAbortFlags(comm,1);
  comm->revokedFlag = 1;
  (void)ncclCommEnsureReady(comm);
  comm->finalizeCalled = true;

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommRevokeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NEW_NOTHROW_GOTO(job, ncclCommRevokeAsyncJob, res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commRevokeAsync, nullptr, ncclCommFinalizeAsyncJobFree, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm) {
    if (!comm->config.blocking) {
      NCCLCHECK(ncclCommGetAsyncError(comm, &res));
    }
    NVTX3_RANGE_ADD_PAYLOAD(CommRevoke, NcclNvtxParamsCommInitRankSchema,
      NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));
  }
  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Revoke COMPLETE, result %d", comm, rank, nranks, cudaDev, comm->busId, res);
  return res;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, res);
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  NVTX3_RANGE(NcclNvtxParamsCommAbort);

  if (comm == NULL) {
    return ncclSuccess;
  }

  INFO(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx - Abort START",
      comm, comm->rank, comm->nRanks, comm->cudaDev, comm->busId);

  NCCLCHECK(ncclGroupStartInternal());
  // Ask anything that might still be running on the device to quit
  NCCLCHECK(setCommAbortFlags(comm,1));
  comm->destroyFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  (void)ncclCommEnsureReady(comm);

  // once the comm is ready, we can access ranks etc
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;
  struct ncclCommFinalizeAsyncJob *job = NULL;
  ncclResult_t res = ncclSuccess;

  NVTX3_RANGE_ADD_PAYLOAD(CommAbort, NcclNvtxParamsCommInitRankSchema,
    NVTX3_PAYLOAD(comm->commHash, nranks, rank, cudaDev));

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, comm->busId);

  NEW_NOTHROW_GOTO(job, ncclCommFinalizeAsyncJob, res, fail);
  job->comm = comm;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, commReclaim, nullptr, ncclCommFinalizeAsyncJobFree, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  goto exit;
}

static void childCommCleanupJob(void* job) {
  struct ncclCommInitRankAsyncJob* initJob = (struct ncclCommInitRankAsyncJob*)job;
  if (initJob->excludeRanksList) free(initJob->excludeRanksList);
  delete initJob;
}

// initializing a child communicator (for both split and shrink)
static ncclResult_t ncclCommInitChildComm(ncclComm_t comm, ncclComm_t* newcomm, bool isShrink, int flags, int color, int key, int* excludeRanksList, int excludeRanksCount,
                                          ncclConfig_t* config, const char* caller) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  int oldDev;
  CUDACHECK(cudaGetDevice(&oldDev));
  NCCLCHECKGOTO(CommCheck(comm, caller, "comm"), res, exit);
  NCCLCHECKGOTO(PtrCheck(newcomm, caller, "newcomm"), res, exit);
  if (isShrink) {
    NCCLCHECKGOTO(PtrCheck(excludeRanksList, caller, "excludeRanksList"), res, exit);
    NCCLCHECKGOTO(excludeRanksCount > 0 ? ncclSuccess : ncclInvalidArgument, res, exit);
    // excludeRanksList may not be sorted, need to sort it
    qsort(excludeRanksList, excludeRanksCount, sizeof(int), compareInts);
    // ranks in excludeRanksList should not call into this function
    NCCLCHECKGOTO(bsearch(&comm->rank, excludeRanksList, excludeRanksCount, sizeof(int), compareInts) ? ncclInvalidArgument : ncclSuccess, res, exit);
  }
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
  CUDACHECKGOTO(cudaSetDevice(comm->cudaDev), res, exit);

  /* *newcomm should be NCCL_COMM_NULL until comm split fully complete. */
  *newcomm = NCCL_COMM_NULL;
  if (!isShrink && color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    childComm->startMagic = childComm->endMagic = NCCL_MAGIC;

    // Set the shareResource field, this is used throughout the init and must be reset every time.
    // Never share resources if the parent communicator has been revoked.
    // If we shrink, we only reuse resources in default mode.
    comm->shareResources = !comm->revokedFlag && (isShrink ? (!(flags & NCCL_SHRINK_ABORT) && comm->config.shrinkShare) : comm->config.splitShare);
    if (comm->shareResources) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagDev = comm->abortFlagDev;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCudaHostCalloc(&childComm->abortFlagDev, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc(&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
      comm->childAbortFlagDev = childComm->abortFlagDev;
      *childComm->abortFlagRefCount = 1;
    }
    if (config == NULL) {
      NCCLCHECKGOTO(copyCommConfig(childComm, comm), res, fail);
    } else {
      NCCLCHECKGOTO(parseCommConfig(childComm, config), res, fail);
    }

    /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
    childComm->initState = ncclInternalError;
  }

  NEW_NOTHROW_GOTO(job, ncclCommInitRankAsyncJob, res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  if (excludeRanksList) {
    // need to copy the list of ranks to exclude because the job is async
    job->excludeRanksCount = excludeRanksCount;
    NCCLCHECKGOTO(ncclCalloc(&job->excludeRanksList, excludeRanksCount), res, fail);
    memcpy(job->excludeRanksList, excludeRanksList, excludeRanksCount * sizeof(int));
  } else {
    // each split has to lead to a unique comm, so increment the childCount
    job->childCount = ++comm->childCount;
    job->excludeRanksList = NULL;
  }
  job->cudaDev = comm->cudaDev;
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "%s", caller);
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, /*undo=*/nullptr, /*destructor=*/childCommCleanupJob, comm), res, fail);

exit:
  (void)cudaSetDevice(oldDev);
  return res;
fail:
  if (childComm) {
    if (!comm->shareResources) {
      if (childComm->abortFlag) free(childComm->abortFlag);
      if (childComm->abortFlagDev) ncclCudaHostFree(childComm->abortFlagDev);
      if (childComm->abortFlagRefCount) free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommShrink, ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t* newcomm, ncclConfig_t* config, int shrinkFlags);
ncclResult_t  ncclCommShrink(ncclComm_t comm, int* excludeRanksList, int excludeRanksCount, ncclComm_t *newcomm, ncclConfig_t* config, int shrinkFlags) {
  NVTX3_RANGE(NcclNvtxParamsCommShrink)
  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  // Handle error mode by setting abort flags and waiting for kernels to complete and unset the flags to avoid bootstrap issues
  if (shrinkFlags & NCCL_SHRINK_ABORT) {
    NCCLCHECKGOTO(setCommAbortFlags(comm, 1), res, exit);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), res, exit);
    NCCLCHECKGOTO(setCommAbortFlags(comm, 0), res, exit);
  }
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/true, shrinkFlags, /*color=*/0, /*key=*/comm->rank, excludeRanksList, excludeRanksCount, config, __func__), res, exit);

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());

  if (newcomm && *newcomm) NVTX3_RANGE_ADD_PAYLOAD(CommShrink, NcclNvtxParamsCommShrinkSchema, NVTX3_PAYLOAD(comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, excludeRanksCount));

  return res;
}

NCCL_API(ncclResult_t, ncclCommGetUniqueId, ncclComm_t comm, ncclUniqueId* uniqueId);
ncclResult_t ncclCommGetUniqueId(ncclComm_t comm, ncclUniqueId* uniqueId) {
  NCCLCHECK(CommCheck(comm, __func__, "comm"));
  NCCLCHECK(ncclCommEnsureReady(comm));
  NCCLCHECK(PtrCheck(uniqueId, "CommGetUniqueId", "uniqueId"));

  struct ncclBootstrapHandle growHandle;
  NCCLCHECK(bootstrapGetUniqueId(&growHandle, comm));

  // Broadcast the grow handle to boundary ranks (rank 0 and N-1)
  NCCLCHECK(bcastGrowHandle(&growHandle, comm, /*isRoot=*/true));

  memset(uniqueId, 0, sizeof(ncclUniqueId));
  memcpy(uniqueId, &growHandle, sizeof(growHandle));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommGrow, ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config);
ncclResult_t ncclCommGrow(ncclComm_t comm, int nRanks, const ncclUniqueId* uniqueId, int rank, ncclComm_t* newcomm, ncclConfig_t* config) {
  NVTX3_RANGE(NcclNvtxParamsCommGrow)

  if (newcomm == NULL) return ncclInvalidArgument;
  if (nRanks <= 0) {
    WARN("ncclCommGrow: total ranks must be positive, got %d", nRanks);
    return ncclInvalidArgument;
  }
  if (comm && nRanks <= comm->nRanks) {
    WARN("ncclCommGrow: total ranks %d is less than current ranks %d", nRanks, comm->nRanks);
    return ncclInvalidArgument;
  }

  ncclResult_t res = ncclSuccess;
  bool isExistingRank = (comm != NULL);
  struct ncclCommInitRankAsyncJob *job = NULL;
  ncclComm_t newComm = NULL;
  struct ncclBootstrapHandle recvHandle;

  *newcomm = NULL;  // Initialize output parameter early in case of early errors

  NCCLCHECK(ncclGroupStartInternal());

  if (isExistingRank) {
    NCCLCHECKGOTO(CommCheck(comm, __func__, "comm"), res, exit);
    NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, exit);
    if (rank != -1) {
      WARN("ncclCommGrow: existing ranks must pass rank=-1, got %d", rank);
      res = ncclInvalidArgument;
      goto exit;
    }

    // each grow/shrink/split has to lead to a unique comm->magic value, increment before using to be consistent with ncclCommInitChildComm
    ++comm->childCount;

    // Only boundary ranks (0 and N-1) receive grow handle from coordinator
    if ((comm->nRanks > 1) && (comm->rank == 0 || comm->rank == comm->nRanks - 1)) {
      NCCLCHECKGOTO(bcastGrowHandle(&recvHandle, comm, /*isRoot=*/false), res, exit);
      // verify the magic is the same as the one computed by the root
      if (recvHandle.magic != hashCombine(comm->magic, comm->childCount)) {
        WARN("ncclCommGrow: magic mismatch computed by the root, got %lx expected %lx",
          recvHandle.magic, hashCombine(comm->magic, comm->childCount));
        res = ncclInvalidArgument;
        goto exit;
      }
      uniqueId = (const ncclUniqueId*)&recvHandle;
    }
  } else {
    // New ranks: validate parameters
    if (rank < 0) {
      WARN("ncclCommGrow: new ranks must pass valid rank >= 0, got %d", rank);
      res = ncclInvalidArgument;
      goto exit;
    }
    if (uniqueId == NULL) {
      WARN("ncclCommGrow: new ranks must pass non-NULL uniqueId");
      res = ncclInvalidArgument;
      goto exit;
    }
    if (rank >= nRanks) {
      WARN("ncclCommGrow: new rank %d exceeds total ranks %d", rank, nRanks);
      res = ncclInvalidArgument;
      goto exit;
    }

    // Initialize NCCL and CUDA for new ranks (same sequence as ncclCommInitRank)
    NCCLCHECKGOTO(ncclInitEnv(), res, exit);  // Environment plugins
    (void)ncclCudaLibraryInit();               // CUDA driver and dlsym hooks
    NCCLCHECKGOTO(ncclInit(), res, exit);      // Bootstrap network, CPU stack, GDR
    if (ncclDebugLevel > NCCL_LOG_WARN || (ncclDebugLevel != NCCL_LOG_NONE && rank == 0)) {
      static std::once_flag once;
      std::call_once(once, showVersion);       // Version display
    }
    CUDACHECKGOTO(cudaFree(NULL), res, exit);  // CUDA runtime initialization
  }

  INFO(NCCL_INIT, "ncclCommGrow: %s rank creating new communicator with %d total ranks",
       isExistingRank ? "existing" : "new", nRanks);

  // All ranks allocate a NEW comm structure for the grown communicator
  NCCLCHECKGOTO(ncclCalloc(&newComm, 1), res, fail);
  newComm->startMagic = newComm->endMagic = NCCL_MAGIC;

  // All ranks allocate fresh resources for grown communicator
  NCCLCHECKGOTO(ncclCalloc(&newComm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCudaHostCalloc(&newComm->abortFlagDev, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc(&newComm->abortFlagRefCount, 1), res, fail);
  *newComm->abortFlagRefCount = 1;

  // Configure the new communicator
  if (isExistingRank && config == NULL) {
    NCCLCHECKGOTO(copyCommConfig(newComm, comm), res, fail);
  } else {
    NCCLCHECKGOTO(parseCommConfig(newComm, config), res, fail);
  }

  newComm->initState = ncclInProgress;
  *newcomm = newComm;

  NEW_NOTHROW_GOTO(job, ncclCommInitRankAsyncJob, res, fail);
  job->nId = (uniqueId != NULL) ? 1 : 0;
  if (job->nId == 1) {
    NCCLCHECKGOTO(ncclCalloc(&job->commId, 1), res, fail);
  }
  snprintf(job->funcName, NCCL_COMMINIT_FUNCNAME_LEN, "ncclCommGrow");

  if (isExistingRank) {
    // Existing rank: new comm, parent is old comm
    job->parent = comm;
    job->cudaDev = comm->cudaDev;
    job->myrank = comm->rank;  // Keep same rank in expanded comm
  } else {
    // New rank: new comm, no parent
    int device;
    CUDACHECK(cudaGetDevice(&device));
    job->parent = NULL;
    job->cudaDev = device;
    job->myrank = rank;
  }

  // All ranks use the newly allocated comm
  job->comm = newComm;

  if (job->nId == 1) memcpy(job->commId, uniqueId, sizeof(ncclUniqueId));
  job->nranks = nRanks;
  job->newcomm = newcomm;
  job->isGrow = 1;
  job->color = 0;
  job->key = job->myrank;
  NCCLCHECKGOTO(ncclAsyncLaunch((struct ncclAsyncJob*)job, ncclCommInitRankFunc, NULL, childCommCleanupJob, newComm), res, fail);

exit:
  if (*newcomm) {
    uint64_t parentHash = isExistingRank ? comm->commHash : 0;
    NVTX3_RANGE_ADD_PAYLOAD(CommGrow, NcclNvtxParamsCommGrowSchema,
      NVTX3_PAYLOAD((*newcomm)->commHash, parentHash, nRanks, (*newcomm)->rank, (*newcomm)->cudaDev));
  }
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;

fail:
  if (job) {
    if (job->commId) {
      free(job->commId);
      job->commId = NULL;
    }
    delete job;
  }
  // Clean up newly allocated comm on failure
  if (newComm) {
    // Only free abort resources if we allocated them (not shared)
    if (!isExistingRank || !comm->shareResources) {
      free(newComm->abortFlag);
      if (newComm->abortFlagDev) (void)ncclCudaHostFree((void*)newComm->abortFlagDev);
      free(newComm->abortFlagRefCount);
    }
    free(newComm);
    if (newcomm) *newcomm = NULL;
  }
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  NVTX3_RANGE(NcclNvtxParamsCommSplit)

  ncclResult_t res = ncclSuccess;
  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(ncclCommInitChildComm(comm, newcomm, /*isShrink=*/false, /*shrink mode=*/NCCL_SHRINK_DEFAULT, color, key, NULL, 0, config, __func__), res, exit);

exit:
  (void)ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());

  if (newcomm && *newcomm)
    NVTX3_RANGE_ADD_PAYLOAD(CommSplit, NcclNvtxParamsCommSplitSchema, NVTX3_PAYLOAD((*newcomm)->commHash, comm->commHash, comm->nRanks, comm->rank, comm->cudaDev, color, key));

  return res;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error (run with NCCL_DEBUG=INFO for details)";
    case ncclSystemError            : return "unhandled system error (run with NCCL_DEBUG=INFO for details)";
    case ncclInternalError          : return "internal error - please report this issue to the NCCL developers";
    case ncclInvalidArgument        : return "invalid argument (run with NCCL_DEBUG=WARN for details)";
    case ncclInvalidUsage           : return "invalid usage (run with NCCL_DEBUG=WARN for details)";
    case ncclRemoteError            : return "remote process exited or there was a network error";
    case ncclInProgress             : return "NCCL operation in progress";
    default                         : return "unknown result code";
  }
}

/* Returns a human-readable message of the last error that occurred.
 * comm is currently unused and can be set to NULL
 */
NCCL_API(const char*, ncclGetLastError, const ncclComm_t comm);
const char* ncclGetLastError(ncclComm_t comm) {
  return ncclLastError;
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = COMPILER_ATOMIC_LOAD(&comm->asyncResult, std::memory_order_acquire);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = COMPILER_ATOMIC_LOAD(&comm->proxyState->asyncResult, std::memory_order_acquire);

  /* Check gin status */
  if (*asyncError == ncclSuccess && comm->sharedRes && comm->sharedRes->ginState.ncclGin) {
    struct ncclGinState* ginState = &comm->sharedRes->ginState;
    // Gin progress thread status
    if (ginState->needsProxyProgress) *asyncError = COMPILER_ATOMIC_LOAD(&comm->sharedRes->ginState.asyncResult, std::memory_order_acquire);
    // Gin side errors, also works when we have no GIN progress thread.
    if (*asyncError == ncclSuccess) {
      bool ginError;
      for (int c=0; c<comm->sharedRes->ginState.ginCommCount; c++) {
        NCCLCHECK(ncclGinQueryLastError(&comm->sharedRes->ginState, &ginError));
        if (ginError) {
          WARN("GIN Error on gin context %d\n", c);
          *asyncError = ncclRemoteError;
          break;
        }
      }
    }
  }

  /* if there is linked group job, we should complete it. */
  if (*asyncError == ncclSuccess && comm->groupJob) {
    NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
    comm->groupJob = NULL;
  }
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCL_NVTX3_FUNC_RANGE;

  NCCLCHECK(CommCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}
