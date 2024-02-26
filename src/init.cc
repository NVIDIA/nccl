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
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "param.h"

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS", "NVLSTree" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);

static ncclResult_t commReclaim(ncclComm_t comm);

static uint64_t hashUniqueId(ncclUniqueId const &id) {
  char const *bytes = (char const*)&id;
  uint64_t h = 0xdeadbeef;
  for(int i=0; i < (int)sizeof(ncclUniqueId); i++) {
    h ^= h >> 32;
    h *= 0x8db3db47fa2994ad;
    h += bytes[i];
  }
  return h;
}

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

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;

static ncclResult_t ncclInit() {
  if (__atomic_load_n(&initialized, __ATOMIC_ACQUIRE)) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initGdrCopy();
    // Always initialize bootstrap network
    NCCLCHECK(bootstrapNetInit());
    NCCLCHECK(ncclNetPluginInit());

    initNvtxRegisteredEnums();
    __atomic_store_n(&initialized, true, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&initLock);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetVersion, int* version);
ncclResult_t ncclGetVersion(int* version) {
  if (version == NULL) return ncclInvalidArgument;
  *version = NCCL_VERSION_CODE;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  ncclResult_t res = bootstrapGetUniqueId((struct ncclBootstrapHandle*)out);
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
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
  CUDACHECK(cudaFreeHost(dtor->obj));
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
  /* commFree() should not involve any sync among ranks. */
  if (comm == NULL)
    return ncclSuccess;

  /* in commReclaim, we have guaranteed only last rank which calls ncclCommDestroy() will
   * free all intra-process communicators; therefore, we only need to focus on local
   * resource cleanup in commFree(). */
  if (comm->proxyState && comm->proxyRefCountOld == 0 && comm->proxyState->thread) {
    pthread_join(comm->proxyState->thread, nullptr);
    if (comm->proxyState->threadUDS) {
      // UDS support
      pthread_join(comm->proxyState->threadUDS, nullptr);;
    }
  }

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

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks, 1, comm->localRanks));

  if (comm->sharedRes) {
    if (ncclAtomicRefCountDecrement(&comm->sharedRes->refCount) == 0) {
      for (int c=0; c<MAXCHANNELS; c++) {
        if (comm->sharedRes->peers[c]) free(comm->sharedRes->peers[c]);
        if (comm->sharedRes->devPeers[c]) ncclCudaFree(comm->sharedRes->devPeers[c]);
      }
      free(comm->sharedRes->tpRankToLocalRank);
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->hostStream));
      NCCLCHECK(ncclStrongStreamDestruct(&comm->sharedRes->deviceStream));
      NCCLCHECK(ncclProxyDestroy(comm));
      free(comm->sharedRes);
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

  if (ncclAtomicRefCountDecrement(comm->abortFlagRefCount) == 0) {
    NCCLCHECK(ncclCudaHostFree((void *)comm->abortFlag));
    free(comm->abortFlagRefCount);
  }
  free((void*)comm->config.netName);

  free(comm->topParentRanks);
  free(comm->topParentLocalRanks);

  NCCLCHECK(ncclRegCleanup(comm));

  commPoison(comm); // poison comm before free to avoid comm reuse.
  free(comm);

  return ncclSuccess;
}

NCCL_PARAM(DisableGraphHelper, "GRAPH_HELPER_DISABLE", 0);
// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);
NCCL_PARAM(WorkFifoDepth, "WORK_FIFO_DEPTH", 64<<10);
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

  if (__atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED)) {
    ncclGroupJobAbort(comm->groupJob);
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret != ncclSuccess) {
      /* if ret is not ncclInProgress, we just keep it. */
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      if (ret == ncclInProgress) ret = ncclInvalidArgument;
      goto exit;
    }
    /* if there is linked group job, we should complete it. */
    if (comm->groupJob) {
      NCCLCHECK(ncclGroupJobComplete(comm->groupJob));
      comm->groupJob = NULL;
    }
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

  NCCLCHECK(ncclNetInit(comm));
  INFO(NCCL_INIT, "Using network %s", comm->ncclNet->name);

  if (parent && parent->config.splitShare) {
    if (parent->ncclNet != comm->ncclNet) {
      WARN("Split shares resources, but parent comm netName %s is different from child comm netName %s", parent->ncclNet->name, comm->ncclNet->name);
      return ncclInvalidUsage;
    }
  }
  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  CUDACHECK(cudaGetDevice(&comm->cudaDev));

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

  comm->collNetSupport = 0;
  memset(comm->collNetSupportMatrix, 0, sizeof(comm->collNetSupportMatrix));

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);
  ncclMemoryPoolConstruct(&comm->memPool_ncclPointerList);
  ncclMemoryPoolConstruct(&comm->memPool_ncclNvlsHandleList);

  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  if (parent == NULL || !parent->config.splitShare) {
    struct ncclSharedResources* sharedRes = NULL;
    NCCLCHECK(ncclCalloc(&sharedRes, 1));
    /* most of attributes are assigned later in initTransportsRank(). */
    sharedRes->owner = comm;
    sharedRes->tpNRanks = comm->nRanks;
    NCCLCHECK(ncclCalloc(&sharedRes->tpRankToLocalRank, comm->nRanks));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->deviceStream));
    NCCLCHECK(ncclStrongStreamConstruct(&sharedRes->hostStream));
    comm->sharedRes = sharedRes;
    sharedRes->refCount = 1;
  } else {
    comm->sharedRes = parent->sharedRes;
    ncclAtomicRefCountIncrement(&parent->sharedRes->refCount);
  }

  if (comm->topParentRanks == NULL) {
    NCCLCHECK(ncclCalloc(&comm->topParentRanks, comm->nRanks));
    for (int i = 0; i < comm->nRanks; ++i)
      comm->topParentRanks[i] = i;
  }

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);

  comm->regCache.pageSize = sysconf(_SC_PAGESIZE);
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclDevCommAndChannels tmpCommAndChans;
  struct ncclDevCommAndChannels *devCommAndChans = NULL;

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->sharedRes->deviceStream), ret, fail);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, comm->sharedRes->deviceStream.cudaStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.node = comm->node;
  tmpCommAndChans.comm.nNodes = comm->nNodes;
  tmpCommAndChans.comm.abortFlag = comm->abortFlag;
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
  tmpCommAndChans.comm.p2pChunkSize = comm->p2pChunkSize;
  tmpCommAndChans.comm.channels = &devCommAndChans->channels[0];

  comm->workFifoDepth = ncclParamWorkFifoDepth();
  if (0 != (comm->workFifoDepth & (comm->workFifoDepth-1))) {
    WARN("NCCL_WORK_FIFO_DEPTH=%d is being ignored because it is not a power of 2.", comm->workFifoDepth);
    comm->workFifoDepth = 64<<10;
  }
  tmpCommAndChans.comm.workFifoDepth = comm->workFifoDepth;

  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // The workFifoHeap lives in GDR mapped CUDA memory.
    NCCLCHECKGOTO(ncclGdrCudaCalloc(&comm->workFifoHeap, &comm->devWorkFifoHeap, comm->workFifoDepth, &comm->workFifoHeapGdrHandle), ret, fail);
    ncclCommPushCudaGdrFree(comm, comm->workFifoHeapGdrHandle);
  } else {
    // The workFifoHeap lives in cudaHost memory.
    comm->workFifoHeapGdrHandle = nullptr;
    NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoHeap, comm->workFifoDepth), ret, fail);
    ncclCommPushCudaHostFree(comm, comm->workFifoHeap);
    comm->devWorkFifoHeap = comm->workFifoHeap;
  }
  tmpCommAndChans.comm.workFifoHeap = comm->devWorkFifoHeap;

  NCCLCHECKGOTO(ncclCudaHostCalloc(&comm->workFifoDone, MAXCHANNELS), ret, fail);
  ncclCommPushCudaHostFree(comm, comm->workFifoDone);
  comm->workFifoSent = 0;
  comm->workFifoAckdMin = 0;

  if (comm->collNetDenseToUserRank != nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&tmpCommAndChans.comm.collNetDenseToUserRank, nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
    ncclCommPushCudaFree(comm, tmpCommAndChans.comm.collNetDenseToUserRank);
    NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.comm.collNetDenseToUserRank, comm->collNetDenseToUserRank, nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
  }

  for (int c=0; c < MAXCHANNELS; c++) {
    tmpCommAndChans.channels[c].peers = comm->channels[c].devPeers;
    tmpCommAndChans.channels[c].ring = comm->channels[c].ring;
    tmpCommAndChans.channels[c].ring.userRanks = comm->channels[c].devRingUserRanks;
    tmpCommAndChans.channels[c].tree = comm->channels[c].tree;
    tmpCommAndChans.channels[c].collnetChain = comm->channels[c].collnetChain;
    tmpCommAndChans.channels[c].collnetDirect = comm->channels[c].collnetDirect;
    tmpCommAndChans.channels[c].nvls = comm->channels[c].nvls;
    tmpCommAndChans.channels[c].workFifoDone = &comm->workFifoDone[c];

    if (comm->channels[c].ring.userRanks != nullptr) {
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, comm->sharedRes->deviceStream.cudaStream), ret, fail);
    }
  }

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, comm->sharedRes->deviceStream.cudaStream), ret, fail);
exit:
  NCCLCHECK(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->sharedRes->deviceStream));
  return ret;
fail:
  goto exit;
}

// Pre-process the string so that running "strings" on the lib can quickly reveal the version.
#define VERSION_STRING "NCCL version " STR(NCCL_MAJOR) "." STR(NCCL_MINOR) "." STR(NCCL_PATCH) NCCL_SUFFIX "+cuda" STR(CUDA_MAJOR) "." STR(CUDA_MINOR)
static void showVersion() {
  static int shown = 0;
  if (shown == 0 && ncclDebugLevel >= NCCL_LOG_VERSION) {
    printf("%s\n", VERSION_STRING);
    fflush(stdout);
    if (ncclDebugFile != stdout)
      INFO(NCCL_ALL,"%s", VERSION_STRING); // Also log NCCL version in one of the files
    shown = 1;
  }
}

static ncclResult_t fillInfo(struct ncclComm* comm, struct ncclPeerInfo* info, uint64_t commHash) {
  info->rank = comm->rank;
  info->cudaDev = comm->cudaDev;
  info->nvmlDev = comm->nvmlDev;
  info->hostHash=getHostHash()+commHash;
  info->pidHash=getPidHash()+commHash;

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
      INFO(NCCL_INIT, "MNNVL busId 0x%lx fabric UUID %lx.%lx cliqueId 0x%x state %d healthMask 0x%x",
           info->busId,
           ((long *)&info->fabricInfo.clusterUuid)[0], ((long *)&info->fabricInfo.clusterUuid)[1],
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
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));

  int64_t envs[NCCL_NUM_PROTOCOLS] = { ncclParamLlBuffSize(), ncclParamLl128BuffSize(), ncclParamBuffSize() };
  int defaults[NCCL_NUM_PROTOCOLS] = { DEFAULT_LL_BUFFSIZE, DEFAULT_LL128_BUFFSIZE, DEFAULT_BUFFSIZE };

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  // MNNVL support
  if (!comm->MNNVL && comm->nNodes > 1) comm->p2pChunkSize = ncclParamP2pNetChunkSize();
  else if (comm->MNNVL || ncclTopoPathAllNVLink(comm->topo)) comm->p2pChunkSize = ncclParamP2pNvlChunkSize();
  else comm->p2pChunkSize = ncclParamP2pPciChunkSize();

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

static ncclResult_t collNetTrySetup(ncclComm_t comm, ncclComm_t parent, struct ncclTopoGraph* collNetGraph) {
  ncclResult_t ret = ncclSuccess;
  int* heads = NULL;
  int rank = comm->rank;
  int collNetSetupFail = 0;
  int highestTypes[NCCL_MAX_LOCAL_RANKS] = { TRANSPORT_P2P };
  // Find all head ranks
  int nHeads = collNetGraph->nChannels;
  int nHeadsUnique = 0;
  int headsUnique[NCCL_MAX_LOCAL_RANKS];
  int highestTransportType0, highestTransportType1;
  char line[1024];
  bool share;

  struct collnetShareInfo {
    int headPosition;
    int isMaster;
  };
  struct collnetShareInfo* infos = NULL;

  NCCLCHECKGOTO(ncclCalloc(&heads, nHeads), ret, fail);
  { uint64_t mask = 0;
    // Head GPU index is always 0
    for (int c = 0; c < nHeads; c++) {
      heads[c] = collNetGraph->intra[c * comm->localRanks + 0];
      assert(comm->rankToNode[heads[c]] == comm->node);
      uint64_t mask0 = mask;
      mask |= 1ull<<comm->rankToLocalRank[heads[c]];
      if (mask != mask0) headsUnique[nHeadsUnique++] = heads[c];
    }
  }

  comm->collNetHeads = heads;
  comm->collNetHeadsNum = nHeads;
  comm->collNetHeadsUniqueNum = nHeadsUnique;
  if (parent && parent->collNetSupport && parent->config.splitShare && parent->nNodes == comm->nNodes) {
    NCCLCHECKGOTO(ncclCalloc(&infos, comm->nRanks), ret, fail);
    /* check whether child can share collnet resources of parent. Since parent builds each collnet communicator
     * based on heads with the same head position in each node, as long as the collnet heads of child comm
     * can match parent's heads, we can let child communicator share parent's collnet resources. */
    for (int h = 0; h < nHeads; ++h) {
      int prev = INT_MIN;
      struct collnetShareInfo* myinfo;

      share = true;
      myinfo = infos + comm->rank;
      memset(myinfo, 0, sizeof(struct collnetShareInfo));
      /* find the child head position in parent collnet heads. */
      if (heads[h] == comm->rank) {
        myinfo->headPosition = -1;
        myinfo->isMaster = 1;
        for (int th = 0; th < parent->collNetHeadsNum; ++th)
          if (parent->topParentRanks[parent->collNetHeads[th]] == comm->topParentRanks[comm->rank]) {
            myinfo->headPosition = th;
            break;
          }
      }

      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, infos, sizeof(struct collnetShareInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (infos[i].isMaster) {
          if (prev == INT_MIN)
            prev = infos[i].headPosition;

          if (infos[i].headPosition == -1 || prev != infos[i].headPosition) {
            share = false;
            break;
          }
        }
      }

      if (share) {
        if (myinfo->isMaster) {
          comm->collNetSharedRes = parent->collNetSharedRes;
          comm->collNetChannels = std::min(comm->nChannels, parent->collNetSharedRes->nChannels);
          for (int c = 0; c < comm->collNetChannels; ++c)
            NCCLCHECKGOTO(initCollnetChannel(comm, c, parent, true), ret, fail);
        }
      } else {
        /* TODO: CX-6 and CX-7 both do not support multiple sharp resources per process, if child comm cannot
         * share the sharp resource from parent, we cannot use sharp in this case. This restriction might be
         * lifted by sharp plugin/IB hardware in the future. */
        collNetSetupFail = 1;
        if (comm->rank == 0) {
          WARN("Child comms (nRanks %d) fails to share parent comms (nRanks %d) sharp resources", comm->nRanks, parent->nRanks);
        }
        goto fail;
      }
    }
    share = true;
  } else {
    /* this allocated buffer will be freed on proxy side */
    NCCLCHECK(ncclCalloc(&comm->collNetSharedRes, 1));
    comm->collNetChannels = comm->collNetSharedRes->nChannels = comm->nChannels;
    comm->collNetSharedRes->buffSize = comm->buffSizes[NCCL_PROTO_SIMPLE];

    comm->collNetDenseToUserRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
    comm->collNetUserToDenseRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
    { // initialize collNetUserToDenseRank[rank]
      uint64_t nonHeadMask = (1ull<<comm->localRanks)-1;
      comm->collNetUserToDenseRank[rank] = -1;
      for (int h=0; h < nHeadsUnique; h++) {
        nonHeadMask ^= 1ull<<comm->rankToLocalRank[headsUnique[h]];
        if (headsUnique[h] == rank) { comm->collNetUserToDenseRank[rank] = h; break; }
      }
      if (comm->collNetUserToDenseRank[rank] == -1) {
        comm->collNetUserToDenseRank[rank] = __builtin_popcountll(nonHeadMask & ((1ull<<comm->localRank)-1));
      }
      comm->collNetUserToDenseRank[rank] += comm->node*comm->localRanks;
    }
    NCCLCHECK(bootstrapAllGather(comm->bootstrap, comm->collNetUserToDenseRank, sizeof(int)));
    for (int r=0; r < comm->nRanks; r++) {
      comm->collNetDenseToUserRank[comm->collNetUserToDenseRank[r]] = r;
    }

    for (int c = 0; c < comm->collNetChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(initCollnetChannel(comm, c, parent, false), ret, fail);
      for (int h = 0; h < nHeads; h++) {
        const int head = heads[h];
        collNetSetupFail |= ncclTransportCollNetSetup(comm, collNetGraph, channel, head, head, h, collNetRecv);
        if (!collNetSetupFail) collNetSetupFail |= ncclTransportCollNetSetup(comm, collNetGraph, channel, head, head, h, collNetSend);
      }
      // Verify CollNet setup across ranks after trying the first channel
      if (c == 0) {
        NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, fail);
      }
    }
    share = false;
  }

  if (share) {
    memcpy(comm->collNetSupportMatrix, parent->collNetSupportMatrix, sizeof(comm->collNetSupportMatrix));
  } else {
    do {
      /* Initialize all entries in collNetSupportMatrix[redop][type]. Since some
      ranks don't connect to sharp we enable a (redop,type) if any rank claims
      support. */
      const ncclRedOp_t redops[] = {ncclSum, ncclProd, ncclMin, ncclMax};
      uint8_t(*matrix)[4][ncclNumTypes];
      bool isHead = false;
      matrix = nullptr;
      NCCLCHECKGOTO(ncclCalloc(&matrix, comm->nRanks), ret, matrix_end);
      for (int h = 0; h < nHeads; h++) isHead |= (heads[h] == comm->rank);
      if (isHead) {
        for (int ty=0; ty < ncclNumTypes; ty++) {
          for (int i=0; i < 4; i++) {
            int support = 0;
            NCCLCHECKGOTO(collNetReduceSupport(comm, (ncclDataType_t)ty, redops[i], &support), ret, matrix_end);
            // bit 0 = not supported, bit 1 = supported
            matrix[rank][redops[i]][ty] = 1<<(support ? 1 : 0);
          }
        }
      }
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, matrix, sizeof(*matrix)), ret, matrix_end);
      for (int ty=0; ty < ncclNumTypes; ty++) {
        for (int i=0; i < 4; i++) {
          int op = redops[i];
          uint8_t accum = 0;
          for (int r=0; r < comm->nRanks; r++) accum |= matrix[r][op][ty];
          // We support (redop, type) if some rank supports it and no rank doesn't support it
          comm->collNetSupportMatrix[op][ty] = (accum == (1<<1));
        }
      }
    matrix_end:
      free(matrix);
      if (ret != ncclSuccess) goto fail;
    } while (0);
  }

  // Verify CollNet setup across ranks after trying all channels
  NCCLCHECKGOTO(ncclTransportCollNetCheck(comm, collNetSetupFail), ret, fail);
  TRACE(NCCL_INIT, "rank %d Connected inter-node CollNet", rank);

  line[0] = '\0';
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclTree* chain = &comm->channels[c].collnetChain;
    snprintf(line + strlen(line), 1023 - strlen(line), " [%d] %d->%d->%d",
      c, chain->down[0], rank, chain->up);
  }
  line[1023] = '\0';

  INFO(NCCL_INIT, "Collnet Chains %s", line);
  // Connect Collnet + chain
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->collnetChain.up, 1, channel->collnetChain.down, 0), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 0), ret, fail);
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, channel->collnetChain.down, 1, &channel->collnetChain.up, 1), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 1), ret, fail);
  INFO(NCCL_INIT, "Connected collnet + chain");

  // Connect intra-node CollNet + Direct
  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channelRecv = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.up, NCCL_MAX_DIRECT_ARITY, channelRecv->collnetDirect.down, 0), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 0, &highestTransportType0), ret, fail);

  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channelSend = comm->channels + c;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.down, NCCL_MAX_DIRECT_ARITY, channelSend->collnetDirect.up, 1), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, collNetGraph, 1, &highestTransportType1), ret, fail);

  // Exchange highest intra-node transport type among ranks
  // because we need to know whether all ranks can p2p each other to determine whether we can directly read/write registered user buffer
  comm->intraHighestTransportType = highestTypes[comm->localRank] = highestTransportType0 > highestTransportType1 ? highestTransportType0 : highestTransportType1;
  if (share) {
    comm->intraHighestTransportType = std::max(comm->intraHighestTransportType, parent->intraHighestTransportType);
  }
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, highestTypes, sizeof(int)), ret, fail);
  for (int i = 0; i < comm->localRanks; i++) {
    if (highestTypes[i] > comm->intraHighestTransportType)
      comm->intraHighestTransportType = highestTypes[i];
  }

  INFO(NCCL_INIT, "rank %d Connected CollNet", rank);

exit:
  free(infos);
  return ret;
fail:
  ncclTransportCollNetFree(comm);
  comm->collNetSupport = 0;
  goto exit;
}

// MNNVL: Flag to indicate whether to enable Multi-Node NVLink
NCCL_PARAM(MNNVL, "MNNVL", -2);

static ncclResult_t initTransportsRank(struct ncclComm* comm, struct ncclComm* parent = NULL) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  int nNodes = 1;
  cpu_set_t affinitySave;
  struct ncclTopoGraph ringGraph;
  struct ncclTopoGraph treeGraph;
  struct ncclTopoGraph collNetGraph;
  struct ncclTopoGraph nvlsGraph;
  struct ncclTopoGraph* graphs[] = { &treeGraph, &ringGraph, &collNetGraph, &collNetGraph, &nvlsGraph, &nvlsGraph };

  struct graphInfo {
    int pattern;
    int nChannels;
    int sameChannels;
    float bwIntra;
    float bwInter;
    int typeIntra;
    int typeInter;
  };

  struct allGatherInfo {
    struct graphInfo graphInfo[NCCL_NUM_ALGORITHMS];
    struct ncclTopoRanks topoRanks;
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
  int tpProxyRank;

  // AllGather1 - begin
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail); // Extra rank to represent CollNet root
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, comm->commHash), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);

  for (int i = 0; i < nranks; i++) {
    if (comm->peerInfo[i].hostHash != comm->peerInfo[rank].hostHash) nNodes++;
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }
  // AllGather1 - end

#if CUDART_VERSION >= 11030

#include <cuda.h>
#include "cudawrap.h"

  // MNNVL support
  if (nNodes > 1) {
    int cliqueSize = 0;
    comm->MNNVL = 0;
    // Determine the size of the MNNVL domain/clique
    for (int i = 0; i < nranks; i++) {
      nvmlGpuFabricInfoV_t *fabricInfo1 = &comm->peerInfo[rank].fabricInfo;
      nvmlGpuFabricInfoV_t *fabricInfo2 = &comm->peerInfo[i].fabricInfo;
      // Check that the Fabric state is fully initialized
      if (fabricInfo2->state != NVML_GPU_FABRIC_STATE_COMPLETED) continue;
      // Check that the cluster UUID and cliqueId match in each rank
      // A zero UUID means we don't have MNNVL fabric info - disable MNNVL
      if ((((long *)&fabricInfo2->clusterUuid)[0]|((long *)fabricInfo2->clusterUuid)[1]) == 0) continue;
      if ((memcmp(fabricInfo1->clusterUuid, fabricInfo2->clusterUuid, NVML_GPU_FABRIC_UUID_LEN) == 0) &&
          (fabricInfo1->cliqueId == fabricInfo2->cliqueId)) {
        cliqueSize++;
      }
    }
    // Determine whether this is a MNNVL system
    comm->MNNVL = ncclParamMNNVL() < 0 ? cliqueSize == comm->nRanks : ncclParamMNNVL();
    // MNNVL requires cuMem to be enabled
    if (!ncclCuMemEnable()) comm->MNNVL = 0;
    if (comm->MNNVL) {
      // MNNVL also requires FABRIC handle support
      int cudaDev;
      int flag = 0;
      CUdevice currentDev;
      CUDACHECK(cudaGetDevice(&cudaDev));
      CUCHECK(cuDeviceGet(&currentDev, cudaDev));
      // Ignore error if CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED is not supported
      (void) CUPFN(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, currentDev));;
      if (!flag)
        comm->MNNVL = 0;
      else
        // Force the handle type to be FABRIC for MNNVL
        ncclCuMemHandleType = CU_MEM_HANDLE_TYPE_FABRIC;
    }
    if (ncclParamMNNVL() == 1 && !comm->MNNVL) {
      WARN("MNNVL is not supported on this system");
      ret = ncclSystemError;
      goto fail;
    }
  }
#endif

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
    for (int i = 0; i < nranks; i++) comm->minCompCap = std::min(comm->minCompCap, comm->peerInfo[rank].cudaCompCap);
    for (int i = 0; i < nranks; i++) comm->maxCompCap = std::max(comm->maxCompCap, comm->peerInfo[rank].cudaCompCap);

    comm->nvlsRegSupport = 1;
    for (int i = 0; i < nranks; i++) {
      if ((comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash)
          && (comm->peerInfo[i].pidHash == comm->peerInfo[rank].pidHash)) {
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
  // Print final topology
  NCCLCHECKGOTO(ncclTopoPrint(comm->topo), ret, fail);

  // Set Affinity to a CPU local the our GPU, so that all memory we allocate
  // on the host is local.
  NCCLCHECKGOTO(ncclTopoGetCpuAffinity(comm->topo, comm->rank, &comm->cpuAffinity), ret, fail);
  if (CPU_COUNT(&comm->cpuAffinity)) {
    sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);
    sched_setaffinity(0, sizeof(cpu_set_t), &comm->cpuAffinity);
  }

  // Determine local CollNet support
  if (collNetSupport(comm)) {
    const char *collNetEnable = ncclGetEnv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->collNetSupport = 1;
      }
    }
  }

  // Determine local Nvls support
  NCCLCHECK(ncclNvlsInit(comm));

  // Get rings and trees
  memset(&ringGraph, 0, sizeof(struct ncclTopoGraph));
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &ringGraph), ret, fail);

  memset(&treeGraph, 0, sizeof(struct ncclTopoGraph));
  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.minChannels = ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &treeGraph), ret, fail);

  memset(&collNetGraph, 0, sizeof(struct ncclTopoGraph));
  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  if (comm->collNetSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &collNetGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &collNetGraph), ret, fail);
  }

  memset(&nvlsGraph, 0, sizeof(struct ncclTopoGraph));
  nvlsGraph.id = 3;
  nvlsGraph.pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph.minChannels = 1;
  nvlsGraph.maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {
    NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &nvlsGraph), ret, fail);
    NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &nvlsGraph), ret, fail);
  }

  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* dumpGraphs[4] = { &ringGraph, &treeGraph, &collNetGraph, &nvlsGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 4, dumpGraphs), ret, fail);
  }

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
  }

  comm->nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
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
  }
  // Now that we know nNodes, alloc nodeRanks and compute localRanks for each node
  NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks, comm->nNodes), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&comm->rankToLocalRank, comm->nRanks), ret, fail);
  for (int r=0; r<comm->nRanks; r++) {
    int node = comm->rankToNode[r];
    comm->rankToLocalRank[r] = comm->nodeRanks[node].localRanks;
    comm->nodeRanks[node].localRanks++;
  }
  // Allocate ranks arrays for each node
  for (int n=0; n<comm->nNodes; n++) {
    NCCLCHECKGOTO(ncclCalloc(&comm->nodeRanks[n].localRankToRank, comm->nodeRanks[n].localRanks), ret, fail);
    comm->maxLocalRanks = std::max(comm->maxLocalRanks, comm->nodeRanks[n].localRanks);
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
    }
    if (graphs[NCCL_ALGO_COLLNET_CHAIN]->nChannels == 0) comm->collNetSupport = 0;
    if (graphs[NCCL_ALGO_NVLS]->nChannels == 0) comm->nvlsSupport = 0;
  }

  comm->nChannels = treeGraph.nChannels = ringGraph.nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  if (comm->nChannels < nChannelsOrig) {
    // We started duplicating channels during Preset(), so we need to move the
    // duplicated channels since we have removed some.
    for (int i=0; i<comm->nChannels; i++) memcpy(comm->channels+comm->nChannels+i, comm->channels+nChannelsOrig+i, sizeof(struct ncclChannel));
  }

  // Determine CollNet support after all-gather now that we know nNodes and each node localRanks
  if (comm->collNetSupport == 1) {
    int collNetNodeThreshold = ncclParamCollNetNodeThreshold();
    if (comm->nNodes < collNetNodeThreshold) {
      INFO(NCCL_INIT, "Communicator has %d nodes which is less than CollNet node threshold %d, disabling CollNet", comm->nNodes, collNetNodeThreshold);
      comm->collNetSupport = 0;
    }
    for (int n=0; n<comm->nNodes; n++) {
      if (comm->nodeRanks[n].localRanks > NCCL_MAX_DIRECT_ARITY+1) {
        WARN("CollNet currently only supports up to %d GPUs per node, disabling CollNet", NCCL_MAX_DIRECT_ARITY+1);
        comm->collNetSupport = 0;
        break;
      }
    }
  }

  NCCLCHECKGOTO(ncclCalloc(&rings, nranks*MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, graphs), ret, fail);
  // AllGather3 - end

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

  // Compute nChannels per peer for p2p
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

  // Launch proxy service thread, after this, the proxy calls can be used.
  if (parent && parent->config.splitShare) {
    comm->proxyState = parent->sharedRes->proxyState;
    ncclAtomicRefCountIncrement(&parent->sharedRes->proxyState->refCount);
  } else {
    NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);
  }

  // Connect with prev/next for each ring
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    NCCLCHECKGOTO(setupChannel(comm, c, rank, nranks, rings+c*nranks), ret, fail);
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &ringGraph, 0), ret, fail);
  INFO(NCCL_INIT, "Connected all rings");

  // Connect Trees
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    if (comm->nRanks == 1) continue;
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0), ret, fail);
    NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0), ret, fail);
  }
  NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &treeGraph, 0), ret, fail);
  INFO(NCCL_INIT, "Connected all trees");

  // Setup NVLS
  NCCLCHECKGOTO(ncclNvlsSetup(comm, parent), ret, fail);
  // And NVLS trees if needed
  if (comm->nvlsSupport && comm->nNodes > 1) {
    for (int c=0; c<comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels+c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 1, &channel->nvls.treeUp, 0), ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->nvls.treeUp, NCCL_MAX_NVLS_TREE_ARITY, channel->nvls.treeDown, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &nvlsGraph, 0), ret, fail);
    INFO(NCCL_INIT, "Connected NVLS tree");
  }

  // Check if we can setup CollNet
  if (comm->collNetSupport > 0) collNetTrySetup(comm, parent, &collNetGraph);

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  NCCLCHECKGOTO(ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d collnet channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->collNetChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  do { // Setup p2p structures in comm->tasks
    struct ncclTasks* tasks = &comm->tasks;
    int node = comm->node;
    int nNodes = comm->nNodes;
    struct ncclNodeRanks *nodeRanks = comm->nodeRanks;
    int localRank = comm->localRank;
    // We want to fuse along node boundaries. Make sure nsteps is a multiple or divides 8.
    int steps = ALIGN_POWER(comm->maxLocalRanks, NCCL_MAX_WORK_ELEMENTS_P2P/2);
    tasks->p2pOrderSteps = comm->nNodes * steps;
    tasks->peers = ncclMemoryStackAlloc<ncclTasks::Peer>(&comm->memPermanent, tasks->p2pOrderSteps);
    tasks->p2pSendOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, tasks->p2pOrderSteps);
    tasks->p2pRecvOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, tasks->p2pOrderSteps);
    int i=0;
    // schedule delta 0, +1, -1, +2, -2, ...
    // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
    for (int d=0; d <= nNodes/4; d++) {
      int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
      int index = 0;
      int delta = deltas[index];
    sched_delta:
      int recvNode = (node+nNodes-delta)%nNodes;
      int sendNode = (node+delta)%nNodes;
      for (int step=0; step < steps; step++) {
        int recvIndex = (localRank-step+steps)%steps;
        int recvRank = recvIndex < nodeRanks[recvNode].localRanks ? nodeRanks[recvNode].localRankToRank[recvIndex] : -1;
        tasks->p2pRecvOrder[i] = recvRank;
        int sendIndex = (localRank+step)%steps;
        int sendRank = sendIndex < nodeRanks[sendNode].localRanks ? nodeRanks[sendNode].localRankToRank[sendIndex] : -1;
        tasks->p2pSendOrder[i] = sendRank;
        i++;
      }
      index++;
      if (index == 1 && deltas[1] == deltas[0]) index++;
      if (index == 2 && deltas[2] == deltas[0]) index++;
      if (index == 3 && deltas[3] == deltas[2]) index++;
      if (index == 3 && deltas[3] == deltas[1]) index++;
      if (index < 4) {
        delta = deltas[index];
        goto sched_delta;
      }
    }
    assert(i == tasks->p2pOrderSteps);
  } while (0);

  if (ncclParamNvbPreconnect()) {
    // Connect p2p when using NVB path
    int nvbNpeers;
    NCCLCHECKGOTO(ncclTopoGetNvbGpus(comm->topo, comm->rank, &nvbNpeers, &nvbPeers), ret, fail);
    for (int r=0; r<nvbNpeers; r++) {
      int peer = nvbPeers[r];
      int channelId;
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECKGOTO(ncclChannelCompute(comm, peer, c, ncclFuncSend, &channelId), ret, fail);
        if (comm->channels[channelId].peers[peer]->send[1].connected == 0) {
          comm->connectSend[peer] |= (1UL<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECKGOTO(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId), ret, fail);
        if (comm->channels[channelId].peers[peer]->recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1UL<<channelId);
        }
      }
    }

    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
  }

  // Connect to local net proxy
  tpProxyRank = comm->topParentRanks[comm->rank];
  NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, tpProxyRank, &proxyConn), ret, fail);
  NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

  // Then to remote ones when using PXN
  if (ncclPxnDisable(comm) == 0) {
    int nranks;
    NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
    for (int r=0; r<nranks; r++) {
      tpProxyRank = comm->topParentRanks[pxnPeers[r]];
      NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, tpProxyRank, &proxyConn), ret, fail);
      NCCLCHECKGOTO(ncclProxyCallBlocking(comm, &proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
    }
  }

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
    modeOld = __atomic_exchange_n(&ncclParamLaunchMode, mode, __ATOMIC_RELAXED);
    if (modeOld == ncclLaunchModeInvalid && str && str[0]!='\0') {
      INFO(NCCL_ENV, "NCCL_LAUNCH_MODE set by environment to %s", mode == ncclLaunchModeParallel ? "PARALLEL" : "GROUP");
    }
  }

  // Call devCommSetup before the last barrier, making sure we don't have a thread running in front and starting to
  // launch NCCL kernels before all cuda mem allocation is complete. That could cause a deadlock.
  NCCLCHECKGOTO(devCommSetup(comm), ret, fail);

  /* Local intra-node barrier */
  NCCLCHECKGOTO(bootstrapBarrier(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, comm->localRankToRank[0]), ret, fail);

  // We should have allocated all buffers, collective fifos, ... we can
  // restore the affinity.
  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);

exit:
  if (CPU_COUNT(&comm->cpuAffinity)) sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  /* If split resource is shared, we are not able to unlink the proxy ops pool here since the child comm can
   * attach the proxy ops pool of parent at any time; otherwise, unlink it here to make sure the pool will be
   * properly cleaned up. */
  if (comm->sharedRes->owner == comm && !comm->config.splitShare && ret == ncclSuccess) ncclProxyShmUnlink(comm);
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

struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  struct ncclComm* comm;
  struct ncclComm** newcomm;
  int cudaDev;
  // For ncclCommInitRank
  int nranks, myrank;
  ncclUniqueId commId;
  // for ncclCommSplit
  struct ncclComm* parent;
  int color, key;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

NCCL_PARAM(CommSplitShareResources, "COMM_SPLIT_SHARE_RESOURCES", NCCL_CONFIG_UNDEF_INT);

static ncclResult_t commGetSplitInfo(struct ncclComm* comm, struct ncclComm* parent, int color, int key, int* nRanksRet, int* myRankRet, int* parentRanksRet) {
  int* colors = NULL;
  int* keys = NULL;
  int nRanks = 0, myRank = 0;
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKGOTO(ncclCalloc(&colors, parent->nRanks), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&keys, parent->nRanks), ret, fail);

  // Compute nRanks, my rank and the ranks (of the original comm) before and after me
  colors[parent->rank] = color;
  keys[parent->rank] = key;
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, colors, sizeof(int)), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(parent->bootstrap, keys, sizeof(int)), ret, fail);

  // Negative color does not create a new comm. Return now.
  if (color == NCCL_SPLIT_NOCOLOR) goto exit;

  memset(parentRanksRet, 0xff, sizeof(int) * parent->nRanks);
  for (int i = 0; i < parent->nRanks; i++) {
    if (colors[i] != color) continue;
    // Find where to insert this rank
    int insert = 0;
    while (insert < nRanks && keys[parentRanksRet[insert]] <= keys[i]) insert++;
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
  free(colors);
  free(keys);
  return ret;
fail:
  goto exit;
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

  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev), res, fail);
  CUDACHECKGOTO(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev), res, fail);
  cudaArch = 100*archMajor + 10*archMinor;

  NCCLCHECK(ncclInitKernelsForDevice(cudaArch, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zi", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }

  if (job->parent) {
    NCCLCHECKGOTO(ncclCalloc(&parentRanks, job->parent->nRanks), res, fail);
    NCCLCHECKGOTO(commGetSplitInfo(comm, job->parent, job->color, job->key, &job->nranks, &job->myrank, parentRanks), res, fail);
    // Negative color does not create a new comm object. We needed to take part in the allgather, but we're done now.
    if (job->color == NCCL_SPLIT_NOCOLOR) goto exit;
    snprintf((char*)&job->commId, sizeof(job->commId), "%016lx-%d", job->parent->commHash, job->color);
    NCCLCHECKGOTO(commAlloc(comm, job->parent, job->nranks, job->myrank), res, fail);
    NCCLCHECKGOTO(bootstrapSplit((struct ncclBootstrapHandle*)&job->commId, comm, job->parent, job->color, job->key, parentRanks), res, fail);
  } else {
    NCCLCHECKGOTO(commAlloc(comm, NULL, job->nranks, job->myrank), res, fail);
    NCCLCHECKGOTO(bootstrapInit((struct ncclBootstrapHandle*)&job->commId, comm), res, fail);
  }

  comm->cudaArch = cudaArch;
  comm->commHash = getHash(job->commId.internal, NCCL_UNIQUE_ID_BYTES);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init START", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, (unsigned long long)hashUniqueId(job->commId));

  NCCLCHECKGOTO(initTransportsRank(comm, job->parent), res, fail);

  NCCLCHECKGOTO(ncclLoadTunerPlugin(&comm->tuner), res, fail);
  if (comm->tuner) {
    NCCLCHECK(comm->tuner->init(comm->nRanks, comm->nNodes, ncclDebugLog));
  }

  // update communicator state
  comm->initState = ncclSuccess;

  // Trace this call for replay tool
  if (job->parent) {
    /* unlink child abort flag. */
    __atomic_store_n(&job->parent->childAbortFlag, NULL, __ATOMIC_RELEASE);
    TRACE_CALL("ncclCommSplit(%p, %d, %d, %p, %d, %d)",
                job->parent, job->color, job->key, comm, comm->rank, comm->nRanks);
  } else {
    TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)",
                comm, comm->nRanks, (unsigned long long)hashUniqueId(job->commId), comm->rank, comm->cudaDev);
  }


  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d nvmlDev %d busId %lx commId 0x%llx - Init COMPLETE", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev, comm->busId, (unsigned long long)hashUniqueId(job->commId));
exit:
  if (job->newcomm) {
    /* assign it to user pointer. */
    __atomic_store_n(job->newcomm, comm, __ATOMIC_RELEASE);
  }
  free(parentRanks);
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

  /* override configuration from env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->config.blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    comm->config.cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    WARN("NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->config.cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.minCTAs = minCTAsEnv;
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.maxCTAs = maxCTAsEnv;
  }

  envNetName = ncclGetEnv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    int netNameLen = strlen(tmpNetName) + 1;
    comm->config.netName = (char*)malloc(netNameLen);
    memcpy((void*)comm->config.netName, tmpNetName, netNameLen);
  } else {
    comm->config.netName = NULL;
  }

  splitShareEnv = ncclParamCommSplitShareResources();
  if (splitShareEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->config.splitShare = splitShareEnv;
  }

  /* cap channels if needed */
  if (comm->config.minCTAs > MAXCHANNELS) {
    WARN("minCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.minCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.minCTAs = MAXCHANNELS;
  }

  if (comm->config.maxCTAs > MAXCHANNELS) {
    WARN("maxCTAs %d is larger than #channels upper limit %d, cap it to %d", comm->config.maxCTAs, MAXCHANNELS, MAXCHANNELS);
    comm->config.maxCTAs = MAXCHANNELS;
  }

  if (comm->config.minCTAs > comm->config.maxCTAs) {
    WARN("minCTAs %d is larger than maxCTAs %d, set both to %d", comm->config.minCTAs, comm->config.maxCTAs, comm->config.maxCTAs);
    comm->config.minCTAs = comm->config.maxCTAs;
  }

  if (comm->config.splitShare != 1 && comm->config.splitShare != 0) {
    WARN("splitShare %d is not a valid value 0/1, set it to 0\n", comm->config.splitShare);
    comm->config.splitShare = 0;
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

  internalConfigPtr = &internalConfig;
  if (config) {
    memcpy((void*)&realSize, (void*)config, sizeof(size_t));
    realSize = realSize > sizeof(ncclConfig_t) ? sizeof(ncclConfig_t) : realSize;
    memcpy((void*)internalConfigPtr, (void*)config, realSize);
    if (internalConfigPtr->magic != 0xcafebeef) {
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

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, splitShare, NCCL_CONFIG_UNDEF_INT, 0, "Split share", "%d");

  /* assign config to communicator */
  comm->config.blocking = internalConfigPtr->blocking;
  comm->config.cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->config.minCTAs = internalConfigPtr->minCTAs;
  comm->config.maxCTAs = internalConfigPtr->maxCTAs;
  comm->config.netName = internalConfigPtr->netName;
  comm->config.splitShare = internalConfigPtr->splitShare;

  NCCLCHECKGOTO(envConfigOverride(comm), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, ncclConfig_t *config) {
  ncclResult_t res = ncclSuccess;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob *job = NULL;
  const char* env = ncclGetEnv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    INFO(NCCL_ENV, "NCCL_COMM_ID set by environment to %s", env);
    NCCLCHECKGOTO(bootstrapCreateRoot((struct ncclBootstrapHandle*)&commId, true), res, fail);
  }

  NCCLCHECKGOTO(ncclInit(), res, fail);
  if (myrank == 0) showVersion();

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
  NCCLCHECKGOTO(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1), res, fail);
  NCCLCHECKGOTO(ncclCalloc((uint32_t**)&comm->abortFlagRefCount, 1), res, fail);
  *comm->abortFlagRefCount = 1;
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInternalError;
  *newcomm = comm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = comm;
  job->nranks = nranks;
  job->commId = commId; // C++ struct assignment
  job->myrank = myrank;
  job->cudaDev = cudaDev;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, free, comm), res, fail);

exit:
  return ncclGroupErrCheck(res);
fail:
  if (comm) {
    if (comm->abortFlag) ncclCudaHostFree((void *)comm->abortFlag);
    if (comm->abortFlagRefCount) free(comm->abortFlagRefCount);
    free(comm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
}

struct NvtxParamsCommInitRank
{
  int rank;
  int nranks;
  int cudaDev;
};
constexpr nvtxPayloadSchemaEntry_t CommInitRankSchema[] = {
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "Rank"},
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "No. of ranks", nullptr, 0, offsetof(NvtxParamsCommInitRank, nranks)},
  {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "CUDA device", nullptr, 0, offsetof(NvtxParamsCommInitRank, cudaDev)},
};

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  CUDACHECK(cudaGetDevice(&cudaDev));

  NvtxParamsCommInitRank payload{myrank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommInitRank, CommInitRankSchema, payload)

  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, &config));
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  ncclResult_t ret = ncclSuccess;
  int totalnDev;
  int *gpuFlags = NULL;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  constexpr nvtxPayloadSchemaEntry_t CommInitAllSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_INT, "No. of devices"}
  };
  NVTX3_FUNC_WITH_PARAMS(CommInitAll, CommInitAllSchema, ndev)

  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

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
        ret = ncclUnhandledCudaError;
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
  NCCLCHECKGOTO(ncclGroupStart(), ret, fail);
  for (int i=0; i<ndev; i++) {
    // Ignore return codes .. we need to call ncclGroupEnd to clean up anyway
    ncclCommInitRankDev(comms+i, ndev, uniqueId, i, devlist ? devlist[i] : i, &config);
  }
  NCCLCHECKGOTO(ncclGroupEnd(), ret, fail);

fail:
  free(gpuFlags);
  return ret;
}

ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommInitRankConfig, ncclComm_t* comm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config);
ncclResult_t ncclCommInitRankConfig(ncclComm_t *newcomm, int nranks, ncclUniqueId commId, int myrank, ncclConfig_t *config) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  int cudaDev;
  ncclResult_t ret = ncclSuccess;
  ncclConfig_t internalConfig = NCCL_CONFIG_INITIALIZER;
  ncclConfig_t *internalConfigPtr = NULL;
  NCCLCHECK(ncclGroupStartInternal());

  (void)ncclCudaLibraryInit();
  CUDACHECKGOTO(cudaGetDevice(&cudaDev), ret, fail);

  if (config == NULL)
    internalConfigPtr = &internalConfig;
  else
    internalConfigPtr = config;
  NCCLCHECKGOTO(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, internalConfigPtr), ret, fail);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommGetAsyncError(*newcomm, &ret);
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->config.blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
  goto exit;
}

static ncclResult_t commDestroySync(struct ncclAsyncJob* job_) {
  struct ncclCommFinalizeAsyncJob* job = (struct ncclCommFinalizeAsyncJob*) job_;
  ncclComm_t comm = job->comm;
  int savedDevice;
  int commDevice = comm->cudaDev;
  ncclResult_t ret = ncclSuccess;

  CUDACHECKGOTO(cudaGetDevice(&savedDevice), ret, fail);
  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(commDevice), ret, fail);
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d asyncResult %d", comm, comm->rank, *comm->abortFlag, comm->asyncResult);

  if (comm->initState == ncclSuccess) {
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->hostStream), ret, fail);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->sharedRes->deviceStream), ret, fail);
  }
  NCCLCHECKGOTO(ncclCommPollCallbacks(comm, false), ret, fail);
  // And keep polling until all graphs referencing us die.
  while (comm->persistentRefs != 0) {
    NCCLCHECKGOTO(ncclCommPollCallbacks(comm, /*waitSome=*/true), ret, fail);
  }

  if (savedDevice != commDevice) {
    CUDACHECKGOTO(cudaSetDevice(savedDevice), ret, fail);
  }

  comm->finalizeCalled = true;
exit:
  return ret;
fail:
  goto exit;
}

static ncclResult_t commCleanup(ncclComm_t comm) {
  int savedDevice;
  int commDevice = comm->cudaDev;

  CUDACHECK(cudaGetDevice(&savedDevice));
  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  if (comm->tuner != NULL) {
    NCCLCHECK(comm->tuner->destroy());
    NCCLCHECK(ncclCloseTunerPlugin(&comm->tuner));
  }

  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(savedDevice));
  }

  return ncclSuccess;
}

static ncclResult_t commFinalize(ncclComm_t comm, bool userCalled) {
  ncclResult_t ret = ncclSuccess;
  struct ncclCommFinalizeAsyncJob *job = NULL;

  /* launch async thread to finalize comm. */
  NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
  job->comm = comm;

  if (userCalled) {
    NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, commDestroySync, NULL, free, comm), ret, fail);
  } else {
    NCCLCHECKGOTO(commDestroySync(&job->base), ret, fail);
    free(job);
  }

exit:
  return ncclGroupErrCheck(ret);
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommFinalize, ncclComm_t comm);
ncclResult_t ncclCommFinalize(ncclComm_t comm) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

  NCCLCHECK(ncclGroupStartInternal());
  if (comm == NULL) goto exit;

  /* wait comm ready before finalize. */
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), ret, fail);

  /* prevent double finalize. */
  if (comm->finalizeCalled) {
    ret = ncclInvalidArgument;
    goto fail;
  }

  /* finalize comm. */
  ret = commFinalize(comm, true);

exit:
  ncclGroupErrCheck(ret);
  NCCLCHECK(ncclGroupEndInternal());
  if (comm && !comm->config.blocking) { NCCLCHECK(ncclCommGetAsyncError(comm, &ret)) };
  return ret;
fail:
  if (comm && !comm->config.blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  ncclResult_t state;
  int curRank; /* Debug info */

  NCCLCHECKGOTO(ncclCommGetAsyncError(comm, &state), ret, fail);
  TRACE(NCCL_INIT, "commReclaim: reclaim comm %p rank %d state %d", comm, comm->rank, state);
  if (state == ncclSuccess && __atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED) == 0 && comm->finalizeCalled == false) {
    /* user does not call ncclCommFinalize and this is a normal comm destroy. ncclCommDestroy
     * should be nonblocking until last call of ncclCommDestroy. */
    NCCLCHECKGOTO(commFinalize(comm, false), ret, fail);
  }

  if (comm->intraComm0 != NULL) {
    int curRankCnt;
    int intraRanks = comm->intraRanks;
    ncclComm_t intracomm0 = comm->intraComm0;
    int *finalizeRankCnt = &intracomm0->finalizeRankCnt;

    assert(intracomm0 != NULL && finalizeRankCnt != NULL);
    curRankCnt = __atomic_add_fetch(finalizeRankCnt, 1, __ATOMIC_ACQ_REL);
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
            WARN("commReclaim: comm %p (rank = %d) in abort, error %d", curIntraComm, curRank, ret);
        }
      }

      /* ncclProxyStop() loop must be put after commDestroySync() loop. Namely, you cannot do:
       *  while(...) {
       *     commDestroySync(...);
       *     ncclProxyStop(...);
       *  }
       * Considering one process multi-gpu case, we must guarantee all kernels are complete before
       * we free proxy resources; otherwise, we will face invalid memory issues where proxy connection
       * and related intermediate memory from one rank are freed but other ranks are still using it.
       * This is not a problem for multi-process case, since intermediate memory is opened by CUDA IPC
       * or mmap where memory free is guarded by CUDA driver and operating system, so we will not have
       * invalid memory access issue. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        /* free intraprocess proxy resources. */
        if ((ret = ncclProxyStop(curIntraComm)) != ncclSuccess) {
          WARN("commReclaim: comm %p (rank = %d) destroys proxy resource error %d", curIntraComm, curRank, ret);
        }
      }

      /* free local resources. */
      nextIntraComm = intracomm0;
      while (nextIntraComm) {
        curIntraComm = nextIntraComm;
        curRank = curIntraComm->rank;
        nextIntraComm = nextIntraComm->intraNext;

        if ((ret = commCleanup(curIntraComm)) != ncclSuccess) {
          WARN("commReclaim: cleanup comm %p rank %d failed in destroy/abort, error %d", curIntraComm, curRank, ret);
        }
      }
    }
  }

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL) {
    NVTX3_FUNC_RANGE_IN(nccl_domain);
    return ncclSuccess;
  }

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;

  NvtxParamsCommInitRank payload{rank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommDestroy, CommInitRankSchema, payload)

  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);
  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks == -1 || comm->cudaDev == -1 || comm->busId == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  /* init thread must be joined before we destroy the comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  NCCLCHECK(commReclaim(comm));
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Destroy COMPLETE", comm, rank, nranks, cudaDev, busId);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (comm == NULL) {
    NVTX3_FUNC_RANGE_IN(nccl_domain);
    return ncclSuccess;
  }

  volatile uint32_t* childAbortFlag;
  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;

  NvtxParamsCommInitRank payload{rank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommAbort, CommInitRankSchema, payload)

  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);

  // Ask anything that might still be running on the device to quit
  childAbortFlag = __atomic_load_n(&comm->childAbortFlag, __ATOMIC_ACQUIRE);
  if (childAbortFlag != NULL) {
    __atomic_store_n(childAbortFlag, 1, __ATOMIC_RELAXED);
  }
  __atomic_store_n(comm->abortFlag, 1, __ATOMIC_RELAXED);
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  ncclCommEnsureReady(comm);

  (void) commReclaim(comm);
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Abort COMPLETE", comm, rank, nranks, cudaDev, busId);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommSplit, ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config);
ncclResult_t ncclCommSplit(ncclComm_t comm, int color, int key, ncclComm_t *newcomm, ncclConfig_t *config) {
  struct ncclCommInitRankAsyncJob *job = NULL;
  struct ncclComm* childComm = NCCL_COMM_NULL;
  ncclResult_t res = ncclSuccess;

  NCCLCHECK(ncclGroupStartInternal());
  NCCLCHECKGOTO(PtrCheck(comm, "CommSplit", "comm"), res, fail);
  NCCLCHECKGOTO(PtrCheck(newcomm, "CommSplit", "newcomm"), res, fail);
  NCCLCHECKGOTO(ncclCommEnsureReady(comm), res, fail);

  /* *newcomm should be NCCL_COMM_NULL until comm split fully complete. */
  *newcomm = NCCL_COMM_NULL;
  if (color == NCCL_SPLIT_NOCOLOR) {
    INFO(NCCL_INIT, "Rank %d has color with NCCL_SPLIT_NOCOLOR, not creating a new communicator", comm->rank);
  } else {
    NCCLCHECKGOTO(ncclCalloc(&childComm, 1), res, fail);
    if (comm->config.splitShare) {
      childComm->abortFlag = comm->abortFlag;
      childComm->abortFlagRefCount = comm->abortFlagRefCount;
      comm->childAbortFlag = NULL;
      ncclAtomicRefCountIncrement(comm->abortFlagRefCount);
    } else {
      NCCLCHECKGOTO(ncclCudaHostCalloc((uint32_t**)&childComm->abortFlag, 1), res, fail);
      NCCLCHECKGOTO(ncclCalloc((uint32_t**)&childComm->abortFlagRefCount, 1), res, fail);
      /* temporarily used to abort everything during child comm init. */
      comm->childAbortFlag = childComm->abortFlag;
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

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->comm = childComm;
  job->newcomm = newcomm;
  job->parent = comm;
  job->color = color;
  job->key = key;
  job->cudaDev = comm->cudaDev;
  NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, free, comm), res, fail);

exit:
  ncclGroupErrCheck(res);
  NCCLCHECK(ncclGroupEndInternal());
  return res;
fail:
  if (childComm) {
    if (comm && !comm->config.splitShare) {
      if (childComm->abortFlag) ncclCudaHostFree((void*)childComm->abortFlag);
      if (childComm->abortFlagRefCount) free(childComm->abortFlagRefCount);
    }
    free(childComm);
  }
  if (newcomm) *newcomm = NULL;
  goto exit;
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
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));

  /* init thread must be joined before we access the attributes of comm. */
  NCCLCHECK(ncclCommEnsureReady(comm));

  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);

  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));

  NCCLCHECK(ncclCommEnsureReady(comm));

  *rank = comm->rank;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclMemAlloc, void **ptr, size_t size);
ncclResult_t  ncclMemAlloc(void **ptr, size_t size) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;

#if CUDART_VERSION >= 12010
  size_t memGran = 0;
  size_t mcGran = 0;
  CUdevice currentDev;
  CUmemAllocationProp memprop = {};
  CUmulticastObjectProp mcprop = {};
  CUmemAccessDesc accessDesc = {};
  CUmemGenericAllocationHandle handle;
  int cudaDev;
  int flag = 0;
  int dcnt;
  int mcSupport = 0;

  if (ptr == NULL || size == 0) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUDACHECK(cudaGetDevice(&cudaDev));
  CUCHECK(cuDeviceGet(&currentDev, cudaDev));
  if (CUPFN(cuMulticastCreate) != NULL)
    CUCHECK(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, currentDev));

  if (mcSupport) {
    memprop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    memprop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    memprop.requestedHandleTypes = NVLS_CU_MEM_HANDLE_TYPE;
    memprop.location.id = currentDev;
    // Query device to see if RDMA support is available
    CUCHECK(cuDeviceGetAttribute(&flag, CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, currentDev));
    if (flag) memprop.allocFlags.gpuDirectRDMACapable = 1;
    CUCHECK(cuMemGetAllocationGranularity(&memGran, &memprop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

    /* mc property */
    CUDACHECK(cudaGetDeviceCount(&dcnt));
    mcprop.size = size;
    /* device cnt is a dummy value right now, it might affect mc granularity in the future. */
    mcprop.numDevices = dcnt;
    mcprop.handleTypes = NVLS_CU_MEM_HANDLE_TYPE;
    mcprop.flags = 0;
    CUCHECK(cuMulticastGetGranularity(&mcGran, &mcprop, CU_MULTICAST_GRANULARITY_RECOMMENDED));

    /* only size needs to be aligned to mcGran */
    ALIGN_SIZE(size, mcGran);
    /* Allocate the physical memory on the device */
    CUCHECK(cuMemCreate(&handle, size, &memprop, 0));
    /* Reserve a virtual address range */
    CUCHECK(cuMemAddressReserve((CUdeviceptr*)ptr, size, memGran, 0, 0));
    /* Map the virtual address range to the physical allocation */
    CUCHECK(cuMemMap((CUdeviceptr)*ptr, size, 0, handle, 0));
    /* Now allow RW access to the newly mapped memory */
    for (int i = 0; i < dcnt; ++i) {
      int p2p = 0;
      if (i == cudaDev || ((cudaDeviceCanAccessPeer(&p2p, cudaDev, i) == cudaSuccess) && p2p)) {
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id = i;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUCHECK(cuMemSetAccess((CUdeviceptr)*ptr, size, &accessDesc, 1));
      }
    }
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaMalloc(ptr, size), ret, fail);

exit:
  return ret;
fail:
  goto exit;
}

NCCL_API(ncclResult_t, ncclMemFree, void *ptr);
ncclResult_t  ncclMemFree(void *ptr) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  ncclResult_t ret = ncclSuccess;
  int saveDevice;

  CUDACHECK(cudaGetDevice(&saveDevice));
#if CUDART_VERSION >= 12010
  CUdevice ptrDev = 0;
  int mcSupport = 0;

  if (ptr == NULL) goto fallback;

  if (ncclCudaLibraryInit() != ncclSuccess) goto fallback;

  CUCHECKGOTO(cuPointerGetAttribute((void*)&ptrDev, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)ptr), ret, fail);
  if (CUPFN(cuMulticastCreate) != NULL)
    CUCHECKGOTO(cuDeviceGetAttribute(&mcSupport, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, ptrDev), ret, fail);

  CUDACHECKGOTO(cudaSetDevice((int)ptrDev), ret, fail);
  if (mcSupport) {
    NCCLCHECKGOTO(ncclCuMemFree(ptr), ret, fail);
    goto exit;
  }

fallback:
#endif
  CUDACHECKGOTO(cudaFree(ptr), ret, fail);

exit:
  cudaSetDevice(saveDevice);
  return ret;
fail:
  goto exit;
}
