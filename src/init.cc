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
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#define STR2(v) #v
#define STR(v) STR2(v)

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

const char* ncclFuncStr[NCCL_NUM_FUNCTIONS] = { "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce" };
const char* ncclAlgoStr[NCCL_NUM_ALGORITHMS] = { "Tree", "Ring", "CollNetDirect", "CollNetChain", "NVLS" };
const char* ncclProtoStr[NCCL_NUM_PROTOCOLS] = { "LL", "LL128", "Simple" };

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);
NCCL_PARAM(CommBlocking, "COMM_BLOCKING", NCCL_CONFIG_UNDEF_INT);

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
  CUDACHECK(cudaFree(dtor->obj));
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
  if (comm->proxyState.thread)
    pthread_join(comm->proxyState.thread, nullptr);

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

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  for (int channel=0; channel<MAXCHANNELS; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks));

  if (comm->initState == ncclSuccess) {
    NCCLCHECK(ncclStrongStreamDestruct(&comm->hostStream));
    NCCLCHECK(ncclStrongStreamDestruct(&comm->deviceStream));
  }

  if (comm->nvlsSupport) NCCLCHECK(ncclNvlsFree(comm));

  struct ncclDestructor* dtor = comm->destructorHead;
  while (dtor != nullptr) {
    NCCLCHECK(dtor->fn(dtor));
    dtor = dtor->next;
  }

  ncclMemoryStackDestruct(&comm->memScoped);
  ncclMemoryStackDestruct(&comm->memPermanent);

  ncclCudaHostFree((void *)comm->abortFlag);
  free(comm->netName);

  commPoison(comm); // poison comm before free to avoid comm reuse.
  free(comm);

  return ncclSuccess;
}

NCCL_PARAM(AggChannelSize, "AGG_CHANNEL_SIZE", -2);
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

  if (*comm->abortFlag) {
    ncclGroupJobAbort();
  } else {
    NCCLCHECK(ncclCommGetAsyncError(comm, &ret));
    if (ret != ncclSuccess) {
      /* if ret is not ncclInProgress, we just keep it. */
      WARN("Attempt to use communicator before the previous operation returned ncclSuccess");
      if (ret == ncclInProgress) ret = ncclInvalidArgument;
      goto exit;
    }
  }

exit:
  return ret;
}

static ncclResult_t commAlloc(ncclComm_t* comret, int ndev, int rank) {
  if (ndev < 1) {
    WARN("invalid device count (%d) requested", ndev);
    return ncclInvalidArgument;
  }
  if (rank >= ndev || rank < 0) {
    WARN("rank %d exceeds ndev=%d", rank, ndev);
    return ncclInvalidArgument;
  }

  struct ncclComm* comm;
  /* Cuurently we calloc comm in ncclCommInitRankDev for async function support.
   * This 'if' structure is designed to consider the case where commAlloc is called
   * in other cases except ncclCommInitRankDev. */
  if (*comret == NULL) {
    /* user requests a new communicator */
    NCCLCHECK(ncclCalloc(&comm, 1));
    NCCLCHECK(ncclCudaHostCalloc((uint32_t**)&comm->abortFlag, 1));
    NCCLCHECK(ncclCommSetAsyncError(comm, ncclInProgress));
  } else {
    /* We already allocated a communicator in ncclCommInitRankDev. */
    comm = *comret;
  }

  ncclMemoryStackConstruct(&comm->memPermanent);
  ncclMemoryStackConstruct(&comm->memScoped);
  comm->destructorHead = nullptr;
  comm->rank = rank;
  comm->nRanks = ndev;

  NCCLCHECK(ncclNetInit(comm));
  INFO(NCCL_INIT, "Using network %s", ncclNetName(comm));

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  NCCLCHECK(ncclStrongStreamConstruct(&comm->deviceStream));
  NCCLCHECK(ncclStrongStreamConstruct(&comm->hostStream));

  cudaGetDevice(&comm->cudaDev);
  NCCLCHECK(getBusId(comm->cudaDev, &comm->busId));
  comm->compCap = ncclCudaCompCap();
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx compCap %d", comm, rank, ndev, comm->cudaDev, comm->busId, comm->compCap);

  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
  comm->dmaBufSupport = (dmaBufSupported(comm) == ncclSuccess) ? true : false;

  comm->collNetSupport = 0;

  ncclMemoryPoolConstruct(&comm->memPool_ncclKernelPlan);
  ncclMemoryPoolConstruct(&comm->memPool_ncclProxyOp);
  ncclMemoryPoolConstruct(&comm->memPool_ncclPointerList);

  comm->groupNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
  comm->channelSize = ncclParamAggChannelSize();

  static_assert(MAXCHANNELS <= sizeof(*comm->connectSend)*8, "comm->connectSend must have enough bits for all channels");
  static_assert(MAXCHANNELS <= sizeof(*comm->connectRecv)*8, "comm->connectRecv must have enough bits for all channels");
  NCCLCHECK(ncclCalloc(&comm->connectSend, comm->nRanks));
  NCCLCHECK(ncclCalloc(&comm->connectRecv, comm->nRanks));

  // Mark channels as non initialized.
  for (int c=0; c < MAXCHANNELS; c++) comm->channels[c].id = -1;

  ncclIntruQueueMpscConstruct(&comm->callbackQueue);

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  int nRanks = comm->nRanks;
  struct ncclDevCommAndChannels tmpCommAndChans;
  struct ncclDevCommAndChannels *devCommAndChans = NULL;

  NCCLCHECKGOTO(ncclStrongStreamAcquireUncaptured(&comm->deviceStream), ret, fail);
  NCCLCHECKGOTO(ncclCudaCallocAsync(&devCommAndChans, 1, comm->deviceStream.cudaStream), ret, fail);
  ncclCommPushCudaFree(comm, devCommAndChans);
  comm->devComm = &devCommAndChans->comm;
  tmpCommAndChans.comm.rank = comm->rank;
  tmpCommAndChans.comm.nRanks = nRanks;
  tmpCommAndChans.comm.abortFlag = comm->abortFlag;
  for (int p=0; p < NCCL_NUM_PROTOCOLS; p++) {
    tmpCommAndChans.comm.buffSizes[p] = comm->buffSizes[p];
  }
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
      NCCLCHECKGOTO(ncclCudaMemcpyAsync(tmpCommAndChans.channels[c].ring.userRanks, comm->channels[c].ring.userRanks, nRanks, comm->deviceStream.cudaStream), ret, fail);
    }
  }

  NCCLCHECKGOTO(ncclCudaMemcpyAsync(devCommAndChans, &tmpCommAndChans, 1, comm->deviceStream.cudaStream), ret, fail);
exit:
  CUDACHECK(cudaStreamSynchronize(comm->deviceStream.cudaStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &comm->deviceStream));
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
  CUDACHECK(cudaGetDevice(&info->cudaDev));
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
  info->cudaCompCap = ncclCudaCompCap();
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
#define DEFAULT_BUFFSIZE_ARM (1 << 20) /* 1MiB */
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

  if (cpuArch == NCCL_TOPO_CPU_ARCH_ARM) defaults[NCCL_PROTO_SIMPLE] = DEFAULT_BUFFSIZE_ARM;

  for (int p=0; p<NCCL_NUM_PROTOCOLS; p++) {
    comm->buffSizes[p] = envs[p] != -2 ? envs[p] : defaults[p];
  }

  if (comm->nNodes > 1) comm->p2pChunkSize = ncclParamP2pNetChunkSize();
  else if (ncclTopoPathAllNVLink(comm->topo)) comm->p2pChunkSize = ncclParamP2pNvlChunkSize();
  else comm->p2pChunkSize = ncclParamP2pPciChunkSize();
  INFO(NCCL_INIT, "P2P Chunksize set to %d", comm->p2pChunkSize);
  return ncclSuccess;
}

NCCL_PARAM(GraphDumpFileRank, "GRAPH_DUMP_FILE_RANK", 0);
NCCL_PARAM(CollNetNodeThreshold, "COLLNET_NODE_THRESHOLD", 2);
NCCL_PARAM(NvbPreconnect, "NVB_PRECONNECT", 1);
NCCL_PARAM(AllocP2pNetLLBuffers, "ALLOC_P2P_NET_LL_BUFFERS", 0);

static ncclResult_t collNetTrySetup(ncclComm_t comm, struct ncclTopoGraph* collNetGraph) {
  ncclResult_t ret = ncclSuccess;
  int* heads = NULL;
  int rank = comm->rank;
  int collNetSetupFail = 0;
  int highestTypes[NCCL_MAX_LOCAL_RANKS] = { TRANSPORT_P2P };
  // Find all head ranks
  int nHeads = collNetGraph->nChannels;
  int highestTransportType0, highestTransportType1;
  char line[1024];

  NCCLCHECKGOTO(ncclCalloc(&heads, nHeads), ret, fail);
  // Head GPU index is always 0
  for (int c = 0; c < nHeads; c++) {
    heads[c] = collNetGraph->intra[c * comm->localRanks + 0];
  }

  for (int c = 0; c < comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels + c;
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
  NCCLCHECKGOTO(bootstrapIntraNodeAllGather(comm->bootstrap, comm->localRankToRank, comm->localRank, comm->localRanks, highestTypes, sizeof(int)), ret, fail);
  for (int i = 0; i < comm->localRanks; i++) {
    if (highestTypes[i] > comm->intraHighestTransportType)
      comm->intraHighestTransportType = highestTypes[i];
  }

  INFO(NCCL_INIT, "rank %d Connected CollNet", rank);

exit:
  free(heads);
  return ret;
fail:
  ncclTransportCollNetFree(comm);
  comm->collNetSupport = 0;
  goto exit;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 2 AllGathers
  // 1. { peerInfo, comm, compCap}
  // 2. { nChannels, graphInfo, topoRanks }
  ncclResult_t ret = ncclSuccess;
  int rank = comm->rank;
  int nranks = comm->nRanks;
  uint64_t commHash = getHash(commId->internal, NCCL_UNIQUE_ID_BYTES);
  cpu_set_t affinitySave;
  struct ncclTopoGraph ringGraph;
  struct ncclTopoGraph treeGraph;
  struct ncclTopoGraph collNetGraph;

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
    int netDev;
    int collNetSupport;
    struct graphInfo tree;
    struct graphInfo ring;
    struct graphInfo collNet;
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

  TRACE(NCCL_INIT, "comm %p, commHash %lx, rank %d nranks %d - BEGIN", comm, commHash, rank, nranks);
  NCCLCHECKGOTO(bootstrapInit((struct ncclBootstrapHandle*)commId, comm), ret, fail);

  // AllGather1 - begin
  NCCLCHECKGOTO(ncclCalloc(&comm->peerInfo, nranks+1), ret, fail); // Extra rank to represent CollNet root
  NCCLCHECKGOTO(fillInfo(comm, comm->peerInfo+rank, commHash), ret, fail);
  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, comm->peerInfo, sizeof(struct ncclPeerInfo)), ret, fail);

  for (int i = 0; i < nranks; i++) {
    if ((i != rank) && (comm->peerInfo[i].hostHash == comm->peerInfo[rank].hostHash) && (comm->peerInfo[i].busId == comm->peerInfo[rank].busId)) {
      WARN("Duplicate GPU detected : rank %d and rank %d both on CUDA device %lx", rank, i, comm->peerInfo[rank].busId);
      ret = ncclInvalidUsage;
      goto fail;
    }
  }
  // AllGather1 - end

  do {
    // Compute intra-process ranks
    int intraProcRank0 = -1, intraProcRank = -1, intraProcRanks = 0;
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

  // Launch proxy service thread
  NCCLCHECKGOTO(ncclProxyCreate(comm), ret, fail);

  // Get rings and trees
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS/2;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &ringGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &ringGraph), ret, fail);

  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.collNet = 0;
  treeGraph.minChannels = 1;
  treeGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &treeGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &treeGraph), ret, fail);

  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  NCCLCHECKGOTO(ncclTopoCompute(comm->topo, &collNetGraph), ret, fail);
  NCCLCHECKGOTO(ncclTopoPrintGraph(comm->topo, &collNetGraph), ret, fail);

  // Initialize num P2P LL buffers for this communicator
  comm->allocP2pNetLLBuffers = ncclParamAllocP2pNetLLBuffers() == 1;

  if (comm->rank == ncclParamGraphDumpFileRank()) {
    struct ncclTopoGraph* graphs[3] = { &ringGraph, &treeGraph, &collNetGraph };
    NCCLCHECKGOTO(ncclTopoDumpGraphs(comm->topo, 3, graphs), ret, fail);
  }

  // Determine local CollNet support before all-gather
  if (collNetSupport(comm)) {
    char *collNetEnable = getenv("NCCL_COLLNET_ENABLE");
    if (collNetEnable != NULL) {
      INFO(NCCL_ALL, "NCCL_COLLNET_ENABLE set by environment to %s.", collNetEnable);
      if (strcmp(collNetEnable, "1") == 0) {
        comm->collNetSupport = 1;
      }
    }
  }
  if (comm->collNetSupport == 1 && collNetGraph.nChannels <= 0) comm->collNetSupport = 0;

  // AllGather3 - begin
  NCCLCHECKGOTO(ncclCalloc(&allGather3Data, nranks), ret, fail);
  NCCLCHECKGOTO(ncclTopoGetLocalNet(comm->topo, rank, &allGather3Data[rank].netDev), ret, fail);
  allGather3Data[rank].tree.pattern = treeGraph.pattern;
  allGather3Data[rank].tree.nChannels = treeGraph.nChannels;
  allGather3Data[rank].tree.sameChannels = treeGraph.sameChannels;
  allGather3Data[rank].tree.bwIntra = treeGraph.bwIntra;
  allGather3Data[rank].tree.bwInter = treeGraph.bwInter;
  allGather3Data[rank].tree.typeIntra = treeGraph.typeIntra;
  allGather3Data[rank].tree.typeInter = treeGraph.typeInter;
  allGather3Data[rank].ring.pattern = ringGraph.pattern;
  allGather3Data[rank].ring.nChannels = ringGraph.nChannels;
  allGather3Data[rank].ring.sameChannels = ringGraph.sameChannels;
  allGather3Data[rank].ring.bwIntra = ringGraph.bwIntra;
  allGather3Data[rank].ring.bwInter = ringGraph.bwInter;
  allGather3Data[rank].ring.typeIntra = ringGraph.typeIntra;
  allGather3Data[rank].ring.typeInter = ringGraph.typeInter;
  allGather3Data[rank].collNet.pattern = collNetGraph.pattern;
  allGather3Data[rank].collNet.nChannels = collNetGraph.nChannels;
  allGather3Data[rank].collNet.sameChannels = collNetGraph.sameChannels;
  allGather3Data[rank].collNet.bwIntra = collNetGraph.bwIntra;
  allGather3Data[rank].collNet.bwInter = collNetGraph.bwInter;
  allGather3Data[rank].collNet.typeIntra = collNetGraph.typeIntra;
  allGather3Data[rank].collNet.typeInter = collNetGraph.typeInter;
  allGather3Data[rank].collNetSupport = comm->collNetSupport;

  comm->nChannels = std::min(treeGraph.nChannels, ringGraph.nChannels);
  NCCLCHECKGOTO(ncclTopoPreset(comm, &treeGraph, &ringGraph, &collNetGraph, &allGather3Data[rank].topoRanks), ret, fail);

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
      nodesTreePatterns[node] = allGather3Data[r].tree.pattern;
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

  nChannelsOrig = comm->nChannels;
  NCCLCHECKGOTO(ncclCalloc(&allTopoRanks, comm->nRanks), ret, fail);
  for (int i=0; i<nranks; i++) {
    comm->peerInfo[i].netDev = allGather3Data[i].netDev;
    allTopoRanks[i] = &allGather3Data[i].topoRanks;
    // Make sure we align all ranks so that the tuning is consistent across ranks
    treeGraph.nChannels = std::min(allGather3Data[i].tree.nChannels, treeGraph.nChannels);
    treeGraph.sameChannels = std::min(allGather3Data[i].tree.sameChannels, treeGraph.sameChannels);
    treeGraph.bwIntra = std::min(allGather3Data[i].tree.bwIntra, treeGraph.bwIntra);
    treeGraph.bwInter = std::min(allGather3Data[i].tree.bwInter, treeGraph.bwInter);
    treeGraph.typeIntra = std::max(allGather3Data[i].tree.typeIntra, treeGraph.typeIntra);
    treeGraph.typeInter = std::max(allGather3Data[i].tree.typeInter, treeGraph.typeInter);
    ringGraph.nChannels = std::min(allGather3Data[i].ring.nChannels, ringGraph.nChannels);
    ringGraph.sameChannels = std::min(allGather3Data[i].ring.sameChannels, ringGraph.sameChannels);
    ringGraph.bwIntra = std::min(allGather3Data[i].ring.bwIntra, ringGraph.bwIntra);
    ringGraph.bwInter = std::min(allGather3Data[i].ring.bwInter, ringGraph.bwInter);
    ringGraph.typeIntra = std::max(allGather3Data[i].ring.typeIntra, ringGraph.typeIntra);
    ringGraph.typeInter = std::max(allGather3Data[i].ring.typeInter, ringGraph.typeInter);
    collNetGraph.nChannels = std::min(allGather3Data[i].collNet.nChannels, collNetGraph.nChannels);
    collNetGraph.sameChannels = std::min(allGather3Data[i].collNet.sameChannels, collNetGraph.sameChannels);
    collNetGraph.bwIntra = std::min(allGather3Data[i].collNet.bwIntra, collNetGraph.bwIntra);
    collNetGraph.bwInter = std::min(allGather3Data[i].collNet.bwInter, collNetGraph.bwInter);
    collNetGraph.typeIntra = std::max(allGather3Data[i].collNet.typeIntra, collNetGraph.typeIntra);
    collNetGraph.typeInter = std::max(allGather3Data[i].collNet.typeInter, collNetGraph.typeInter);
    comm->collNetSupport = std::min(allGather3Data[i].collNetSupport, comm->collNetSupport);
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
  NCCLCHECKGOTO(ncclTopoPostset(comm, nodesFirstRank, nodesTreePatterns, allTopoRanks, rings, &collNetGraph), ret, fail);
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

  // Check if we can setup CollNet
  if (comm->collNetSupport > 0) collNetTrySetup(comm, &collNetGraph);

  NCCLCHECKGOTO(ncclNvlsSetup(comm), ret, fail);

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, comm->nChannels);

  // Compute time models for algorithm and protocol combinations
  do {
    int myCompCap = comm->peerInfo[rank].cudaCompCap;
    int minCompCap = myCompCap, maxCompCap = myCompCap;
    for (int i = 0; i < nranks; i++) {
      comm->minCompCap = minCompCap = std::min(comm->peerInfo[i].cudaCompCap, minCompCap);
      maxCompCap = std::max(comm->peerInfo[i].cudaCompCap, maxCompCap);
    }
    NCCLCHECKGOTO(ncclTopoTuneModel(comm, minCompCap, maxCompCap, &treeGraph, &ringGraph, &collNetGraph), ret, fail);
  } while(0);

  // Compute nChannels per peer for p2p
  NCCLCHECKGOTO(ncclTopoComputeP2pChannels(comm), ret, fail);

  INFO(NCCL_INIT, "%d coll channels, %d nvls channels, %d p2p channels, %d p2p channels per peer", comm->nChannels, comm->nvlsChannels, comm->p2pnChannels, comm->p2pnChannelsPerPeer);

  do { // Setup p2p structures in comm->tasks
    struct ncclTasks* tasks = &comm->tasks;
    int nRanks = comm->nRanks;
    int node = comm->node;
    int nNodes = comm->nNodes;
    struct ncclNodeRanks *nodeRanks = comm->nodeRanks;
    int localRank = comm->localRank;
    tasks->peers = ncclMemoryStackAlloc<ncclTasks::Peer>(&comm->memPermanent, nRanks);
    tasks->p2pSendOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    tasks->p2pRecvOrder = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
    int s=0, r=0;
    // schedule delta 0, +1, -1, +2, -2, ...
    // also make sure we don't do 0 twice, nor +n/2 and -n/2 if n is even.
    for (int d=0; d <= nNodes/4; d++) {
      int deltas[4] = { d, (nNodes-d)%nNodes, nNodes/2-d, (nNodes-(nNodes/2-d))%nNodes };
      int index = 0;
      int delta = deltas[index];
    sched_delta:
      int recvNode = (node+nNodes-delta)%nNodes;
      int sendNode = (node+delta)%nNodes;
      int steps = comm->maxLocalRanks;
      for (int step=0; step < steps; step++) {
        int recvIndex = (localRank-step+steps)%steps;
        if (recvIndex < nodeRanks[recvNode].localRanks) {
          tasks->p2pRecvOrder[r] = nodeRanks[recvNode].localRankToRank[recvIndex];
          r++;
        }
        int sendIndex = (localRank+step)%steps;
        if (sendIndex < nodeRanks[sendNode].localRanks) {
          tasks->p2pSendOrder[s] = nodeRanks[sendNode].localRankToRank[sendIndex];
          s++;
        }
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
    assert(s == nRanks && r == nRanks);
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
        if (comm->channels[channelId].peers[peer].send[1].connected == 0) {
          comm->connectSend[peer] |= (1UL<<channelId);
        }
      }
      for (int c=0; c<comm->p2pnChannelsPerPeer; c++) {
        NCCLCHECKGOTO(ncclChannelCompute(comm, peer, c, ncclFuncRecv, &channelId), ret, fail);
        if (comm->channels[channelId].peers[peer].recv[1].connected == 0) {
          comm->connectRecv[peer] |= (1UL<<channelId);
        }
      }
    }

    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, NULL, 1), ret, fail);
  }

  // Connect to local net proxy
  NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, comm->rank, &proxyConn), ret, fail);
  NCCLCHECKGOTO(ncclProxyCallBlocking(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);

  // Then to remote ones when using PXN
  if (ncclPxnDisable(comm) == 0) {
    int nranks;
    NCCLCHECKGOTO(ncclTopoGetPxnRanks(comm, &pxnPeers, &nranks), ret, fail);
    for (int r=0; r<nranks; r++) {
      NCCLCHECKGOTO(ncclProxyConnect(comm, TRANSPORT_NET, 1, pxnPeers[r], &proxyConn), ret, fail);
      NCCLCHECKGOTO(ncclProxyCallBlocking(&proxyConn, ncclProxyMsgSharedInit, &comm->p2pnChannels, sizeof(int), NULL, 0), ret, fail);
    }
  }

  if (comm->intraRank == 0) { // Load ncclParamLaunchMode
    char* str = getenv("NCCL_LAUNCH_MODE");
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
  // Unlink proxy shm to make sure it will be properly cleaned up.
  ncclProxyShmUnlink(comm);
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
  ncclComm_t* newcomm;
  int nranks, myrank;
  ncclUniqueId commId;
  int cudaDev;
};

struct ncclCommFinalizeAsyncJob {
  struct ncclAsyncJob base;
  ncclComm_t comm;
};

static ncclResult_t ncclCommInitRankFunc(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclComm_t* newcomm = job->newcomm;
  ncclComm_t comm = *newcomm;
  int nranks = job->nranks;
  ncclUniqueId commId = job->commId; // C++ struct assignment
  int myrank = job->myrank;
  int cudaDev = job->cudaDev;
  int archMajor, archMinor;
  size_t maxLocalSizeBytes = 0;
  ncclResult_t res = ncclSuccess;

  CUDACHECKGOTO(cudaSetDevice(cudaDev), res, fail);
  CUDACHECK(cudaDeviceGetAttribute(&archMajor, cudaDevAttrComputeCapabilityMajor, cudaDev));
  CUDACHECK(cudaDeviceGetAttribute(&archMinor, cudaDevAttrComputeCapabilityMinor, cudaDev));
  comm->cudaArch = 100*archMajor + 10*archMinor;

  NCCLCHECK(ncclInitKernelsForDevice(comm->cudaArch, &maxLocalSizeBytes));
  // Set the maximum kernel stack size of all kernels to avoid
  // a CUDA memory reconfig on load (c.f. NVSHMEM issue)
  if (maxLocalSizeBytes > 0 && ncclParamSetStackSize() == 1) {
    TRACE(NCCL_INIT, "Setting cudaLimitStackSize to %zi", maxLocalSizeBytes);
    CUDACHECKIGNORE(cudaDeviceSetLimit(cudaLimitStackSize, maxLocalSizeBytes));
  }
  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank), res, fail);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, fail);

  // update communicator state
  comm->initState = ncclSuccess;

  // Trace this call for replay tool
  TRACE_CALL("ncclCommInitRank(%p, %d, 0x%llx, %d, %d)",
    *newcomm, nranks, (unsigned long long)hashUniqueId(commId), myrank, (*newcomm)->cudaDev);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx commId 0x%llx - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->busId, (unsigned long long)hashUniqueId(commId));
exit:
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

static ncclResult_t parseCommConfig(ncclComm_t comm, ncclConfig_t *config) {
  ncclResult_t ret = ncclSuccess;
  /* config must not be NULL in this function */
  int blockingEnv;
  int cgaClusterSizeEnv;
  int minCTAsEnv;
  int maxCTAsEnv;
  const char *envNetName, *tmpNetName;
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

  /* default config value can be tuned on different platform. */
  NCCL_CONFIG_DEFAULT(internalConfigPtr, blocking, NCCL_CONFIG_UNDEF_INT, 1, "Blocking", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, cgaClusterSize, NCCL_CONFIG_UNDEF_INT, 4, "CGA cluster size", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, minCTAs, NCCL_CONFIG_UNDEF_INT, 1, "Min CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, maxCTAs, NCCL_CONFIG_UNDEF_INT, MAXCHANNELS, "Max CTAs", "%d");
  NCCL_CONFIG_DEFAULT(internalConfigPtr, netName, NCCL_CONFIG_UNDEF_PTR, NULL, "Net name", "%s");

  tmpNetName = internalConfigPtr->netName;

  /* assign config to communicator */
  comm->blocking = internalConfigPtr->blocking;
  comm->cgaClusterSize = internalConfigPtr->cgaClusterSize;
  comm->minCTAs = internalConfigPtr->minCTAs;
  comm->maxCTAs = internalConfigPtr->maxCTAs;

  /* override configuration from env variable. */
  blockingEnv = ncclParamCommBlocking();
  if (blockingEnv == 0 || blockingEnv == 1)
    comm->blocking = blockingEnv;

  cgaClusterSizeEnv = ncclParamCGAClusterSize();
  if (0 <= cgaClusterSizeEnv && cgaClusterSizeEnv <= NCCL_MAX_CGA_CLUSTER_SIZE) {
    comm->cgaClusterSize = cgaClusterSizeEnv;
  } else if (cgaClusterSizeEnv > NCCL_MAX_CGA_CLUSTER_SIZE) {
    WARN("NCCL_CGA_CLUSTER_SIZE value %d is too big. Limiting value to %d.", cgaClusterSizeEnv, NCCL_MAX_CGA_CLUSTER_SIZE);
    comm->cgaClusterSize = NCCL_MAX_CGA_CLUSTER_SIZE;
  }

  minCTAsEnv = ncclParamMinCTAs();
  if (minCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->minCTAs = minCTAsEnv;
  }

  maxCTAsEnv = ncclParamMaxCTAs();
  if (maxCTAsEnv != NCCL_CONFIG_UNDEF_INT) {
    comm->maxCTAs = maxCTAsEnv;
  }

  /* cap channels if needed */
  if (comm->minCTAs > MAXCHANNELS) {
    WARN("minCTAs %d is larger than #channels upper limit %d", comm->minCTAs, MAXCHANNELS);
    comm->minCTAs = MAXCHANNELS;
  }

  if (comm->maxCTAs > MAXCHANNELS) {
    WARN("maxCTAs %d is larger than #channels upper limit %d", comm->maxCTAs, MAXCHANNELS);
    comm->maxCTAs = MAXCHANNELS;
  }

  if (comm->minCTAs > comm->maxCTAs) {
    WARN("minCTAs %d is larger than maxCTAs %d", comm->minCTAs, comm->maxCTAs);
    ret = ncclInvalidArgument;
    goto fail;
  }

  envNetName = getenv("NCCL_NET");
  if (envNetName)
    tmpNetName = envNetName;
  if (tmpNetName != NULL) {
    int netNameLen = strlen(tmpNetName) + 1;
    comm->netName = (char*)malloc(netNameLen);
    memcpy(comm->netName, tmpNetName, netNameLen);
  } else {
    comm->netName = NULL;
  }

exit:
  return ret;
fail:
  goto exit;
}

static void ncclCommInitRankUndo(struct ncclAsyncJob* job_) {
  struct ncclCommInitRankAsyncJob* job = (struct ncclCommInitRankAsyncJob*)job_;
  ncclCommDestroy(*job->newcomm);
  *job->newcomm = nullptr;
}

static ncclResult_t ncclCommInitRankDev(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank, int cudaDev, ncclConfig_t *config) {
  ncclResult_t res = ncclSuccess;
  ncclComm_t comm = NULL;
  struct ncclCommInitRankAsyncJob *job = NULL;
  char* env = getenv("NCCL_COMM_ID");
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
  NCCLCHECKGOTO(parseCommConfig(comm, config), res, fail);
  /* start with ncclInternalError and will be changed to ncclSuccess if init succeeds. */
  comm->initState = ncclInternalError;
  *newcomm = comm;

  NCCLCHECKGOTO(ncclCalloc(&job, 1), res, fail);
  job->newcomm = newcomm;
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
  if (newcomm && *newcomm && !(*newcomm)->blocking) (void) ncclCommGetAsyncError(*newcomm, &ret);
  return ret;
fail:
  if (newcomm && *newcomm && !(*newcomm)->blocking) (void) ncclCommSetAsyncError(*newcomm, ret);
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
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->hostStream), ret, fail);
    NCCLCHECKGOTO(ncclStrongStreamSynchronize(&comm->deviceStream), ret, fail);
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
  if (comm && !comm->blocking) { NCCLCHECK(ncclCommGetAsyncError(comm, &ret)) };
  return ret;
fail:
  if (comm && !comm->blocking) (void) ncclCommSetAsyncError(comm, ret);
  goto exit;
}

static ncclResult_t commReclaim(ncclComm_t comm) {
  ncclResult_t ret = ncclSuccess;
  ncclResult_t state;
  int curRank; /* Debug info */

  NCCLCHECKGOTO(ncclCommGetAsyncError(comm, &state), ret, fail);
  TRACE(NCCL_INIT, "commReclaim: reclaim comm %p rank %d state %d", comm, comm->rank, state);
  if (state == ncclSuccess && *comm->abortFlag == 0 && comm->finalizeCalled == false) {
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

      /* ncclProxyDestroy() loop must be put after commDestroySync() loop. Namely, you cannot do:
       *  while(...) {
       *     commDestroySync(...);
       *     ncclProxyDestroy(...);
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
        if ((ret = ncclProxyDestroy(curIntraComm)) != ncclSuccess) {
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

  int rank = comm->rank, nranks = comm->nRanks, cudaDev = comm->cudaDev;

  NvtxParamsCommInitRank payload{rank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommAbort, CommInitRankSchema, payload)

  int64_t busId = comm->busId;
  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d busId %lx", comm, rank, nranks, cudaDev, busId);

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;
  /* init thread must be joined before we destroy the comm,
   * and we should ignore the init error here. */
  ncclCommEnsureReady(comm);

  (void) commReclaim(comm);
  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d busId %lx - Abort COMPLETE", comm, rank, nranks, cudaDev, busId);

  return ncclSuccess;
}

NCCL_API(const char*, ncclGetErrorString, ncclResult_t code);
const char* ncclGetErrorString(ncclResult_t code) {
  switch (code) {
    case ncclSuccess                : return "no error";
    case ncclUnhandledCudaError     : return "unhandled cuda error";
    case ncclSystemError            : return "unhandled system error";
    case ncclInternalError          : return "internal error";
    case ncclInvalidArgument        : return "invalid argument";
    case ncclInvalidUsage           : return "invalid usage";
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
