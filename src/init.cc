/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "nccl.h"
#include "core.h"
#include "channel.h"
#include "param.h"
#include "nvmlwrap.h"
#include "rings.h"
#include "trees.h"
#include "bootstrap.h"
#include "transport.h"
#include "group.h"
#include "utils.h"
#include "net.h"
#include "checks.h"
#include "enqueue.h"
#include "topo.h"
#include "nvlink.h"
#include "cpuset.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sched.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <dlfcn.h>

#define STR2(v) #v
#define STR(v) STR2(v)

int ncclDebugLevel;
uint64_t ncclDebugMask = NCCL_INIT; // Default debug sub-system mask is INIT
pthread_mutex_t ncclDebugOutputLock;
FILE *ncclDebugFile = stdout;

#ifdef ENABLE_TRACE
std::chrono::high_resolution_clock::time_point ncclEpoch;
#endif

#if CUDART_VERSION >= 9020
#define NCCL_GROUP_CUDA_STREAM 0 // CGMD: CUDA 9.2,10.X Don't need to use an internal CUDA stream
#else
#define NCCL_GROUP_CUDA_STREAM 1 // CGMD: CUDA 9.0,9.1 Need to use an internal CUDA stream
#endif

NCCL_PARAM(GroupCudaStream, "GROUP_CUDA_STREAM", NCCL_GROUP_CUDA_STREAM);

NCCL_PARAM(CheckPointers, "CHECK_POINTERS", 0);

ncclNet_t* ncclNet = NULL;

// We define this as weak to let tests redefine their own
#pragma weak ncclNvlinkGpu
ncclResult_t ncclNvlinkGpu(int* nvlink) {
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  CUDACHECK(cudaDeviceGetPCIBusId(busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, cudaDev));
  *nvlink = getNvlinkGpu(busId, NULL);
  return ncclSuccess;
}
// We define this as weak to let tests redefine their own
#pragma weak ncclCudaCompCap
int ncclCudaCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  return ccMajor;
}
int ncclCudaFullCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor, ccMinor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev) != cudaSuccess) return 0;
  return ccMajor*10+ccMinor;
}

// Returns ncclInternalError if anything fails, causing that network to be ignored.
ncclResult_t initNet(ncclNet_t* net) {
  int ndev;
  if (net->init(ncclDebugLog) != ncclSuccess) return ncclInternalError;
  if (net->devices(&ndev) != ncclSuccess) return ncclInternalError;
  if (ndev <= 0) return ncclSystemError;
  return ncclSuccess;
}

ncclResult_t initNetPlugin(ncclNet_t** net) {
  void* netPluginLib = dlopen("libnccl-net.so", RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == NULL) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : No plugin found (libnccl-net.so).");
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin load returned %d : %s.", errno, dlerror());
    }
    return ncclSuccess;
  }
  ncclNet_t* extNet = (ncclNet_t*) dlsym(netPluginLib, STR(NCCL_PLUGIN_SYMBOL));
  if (extNet == NULL) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find " STR(NCCL_PLUGIN_SYMBOL) " symbol.");
    goto cleanup;
  }
  if (initNet(extNet) == ncclSuccess) {
    *net = extNet;
    return ncclSuccess;
  }
cleanup:
  if (netPluginLib != NULL) dlclose(netPluginLib);
  return ncclSuccess;
}

ncclResult_t initNet() {
  // Always initialize sockets as we use it for bootstrap
  NCCLCHECK(initNet(&ncclNetSocket));

  NCCLCHECK(initNetPlugin(&ncclNet));
  if (ncclNet != NULL) return ncclSuccess;
  if (initNet(&ncclNetIb) == ncclSuccess) {
    ncclNet = &ncclNetIb;
  } else {
    ncclNet = &ncclNetSocket;
  }
  return ncclSuccess;
}

NCCL_PARAM(LlThreshold, "LL_THRESHOLD", -2);
NCCL_PARAM(ThreadThreshold, "THREAD_THRESHOLD", -2);
NCCL_PARAM(TreeThreshold, "TREE_THRESHOLD", -2);

int ncclThreadThreshold(int minCompCap, int multiNode) {
  int threshold = ncclParamThreadThreshold();
  if (threshold == -2) { // user has not set this env variable
    threshold = (minCompCap <= 6) ? NCCL_THREAD_THRESHOLD_PREVOLTA : NCCL_THREAD_THRESHOLD;
    // multiply by 2 if running on multiple nodes
    if (multiNode) {
      threshold *= 2;
    }
  }
  return threshold;
}

pthread_mutex_t initLock = PTHREAD_MUTEX_INITIALIZER;
static bool initialized = false;
static ncclResult_t ncclInit() {
  if (initialized) return ncclSuccess;
  pthread_mutex_lock(&initLock);
  if (!initialized) {
    initEnv();
    initDebug();
    initNet();
    initialized = true;
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
  return bootstrapGetUniqueId(out);
}

// Prevent compiler from optimizing out these operations
void __attribute__((optimize("O0"))) commPoison(ncclComm_t comm) {
  comm->rank = comm->cudaDev = comm->nvmlDev = comm->nRanks = -1;
}

static ncclResult_t commFree(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  free(comm->peerInfo);

  if (comm->bootstrap)
    NCCLCHECK(bootstrapClose(comm->bootstrap));

  CUDACHECK(cudaFree(comm->hostDevComm.channels));
  CUDACHECK(cudaFree(comm->devComm));

  for (int channel=0; channel<comm->nChannels; channel++)
    NCCLCHECK(freeChannel(comm->channels+channel, comm->nRanks));

  if (comm->doneEvent != NULL)
    CUDACHECK(cudaEventDestroy(comm->doneEvent));

  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamDestroy(comm->groupStream));
  }

  // Last rank frees shared resources between threads
  int isLast;
  NCCLCHECK(ncclCpuBarrierIn(comm, &isLast));
  if (isLast) {
    free(comm->intraBarrier);
    free(comm->intraParams);
    free(comm->intraCudaDevs);
    free(comm->intraCGMode);
    free(comm->intraCC);
  }
  CUDACHECK(cudaFreeHost((void *)comm->abortFlag));
  CUDACHECK(cudaFreeHost((void *)comm->fatalDevError));

  // Poison comm to try and catch a double free
  commPoison(comm);

  free(comm);
  return ncclSuccess;
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

  // Try to create a CUDA object right away. If there is something wrong with
  // the device we're on (failure cause #1) , better know it early.
  cudaEvent_t doneEvent;
  CUDACHECK(cudaEventCreateWithFlags(&doneEvent, cudaEventDisableTiming));

  struct ncclComm* comm;
  NCCLCHECK(ncclCalloc(&comm, 1));

  comm->rank = comm->hostDevComm.rank =rank;
  comm->nRanks = comm->hostDevComm.nRanks = ndev;
  cudaGetDevice(&comm->cudaDev);
  getNvmlDevice(comm->cudaDev, &comm->nvmlDev);
  TRACE(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d nvmlDev %d", comm, rank, ndev, comm->cudaDev, comm->nvmlDev);

  comm->doneEvent = doneEvent;
  comm->llThreshold = ncclParamLlThreshold();
  comm->treeThreshold = ncclParamTreeThreshold();
  comm->checkPointers = ncclParamCheckPointers() == 1 ? true : false;
#if CUDART_VERSION >= 9020
  comm->groupCudaStream = ncclParamGroupCudaStream();
#else
  // Don't allow the user to overload the default setting in older CUDA builds
  comm->groupCudaStream = NCCL_GROUP_CUDA_STREAM;
#endif
  comm->fatalError = ncclSuccess;

  NCCLCHECK(ncclCudaHostAlloc((void**) &comm->fatalDevError, (void**) &comm->hostDevComm.fatalDevError, sizeof(ncclDevError_t)));
  *comm->fatalDevError = ncclDevSuccess;

  NCCLCHECK(ncclCudaHostAlloc((void**) &comm->abortFlag, (void**) &comm->hostDevComm.abortFlag, sizeof(uint32_t)));
  *comm->abortFlag = 0;

  comm->argsptr = &comm->args;

  *comret = comm;
  return ncclSuccess;
}

static ncclResult_t devCommSetup(ncclComm_t comm) {
  // Duplicate the channels on the device
  NCCLCHECK(ncclCudaCalloc(&comm->hostDevComm.channels, comm->nChannels));
  NCCLCHECK(ncclCudaMemcpy(comm->hostDevComm.channels, comm->channels, comm->nChannels));

  // Copy userRanks and peers
  for (int r=0; r<comm->nChannels; r++) {
    NCCLCHECK(ncclCudaMemcpy(comm->channels[r].ring.devUserRanks, comm->channels[r].ring.userRanks, comm->nRanks));
    NCCLCHECK(ncclCudaMemcpy(comm->channels[r].devPeers, comm->channels[r].peers, comm->nRanks));
  }

  // Duplicate the dev comm on the device
  NCCLCHECK(ncclCudaCalloc(&comm->devComm, 1));
  NCCLCHECK(ncclCudaMemcpy(comm->devComm, &comm->hostDevComm, 1));
  return ncclSuccess;
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

static ncclResult_t fillInfo(struct ncclPeerInfo* info, int rank) {
  info->rank = rank;
  CUDACHECK(cudaGetDevice(&info->cudaDev));
  NCCLCHECK(getNvmlDevice(info->cudaDev, &info->nvmlDev))
  info->hostHash=getHostHash();
  info->pidHash=getPidHash();

  // Get PCI Bus Id. We need to get the bus ID through CUDA first, since the
  // cudaDev is a CUDA runtime dev number which could be different from the
  // NVML device number. Then we get the busID from NVML to be sure it is
  // consistent with NVML remote PCI bus Ids.
  CUDACHECK(cudaDeviceGetPCIBusId(info->busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE, info->cudaDev));
  nvmlDevice_t nvmlDevice;
  NCCLCHECK(wrapNvmlDeviceGetHandleByPciBusId(info->busId, &nvmlDevice));
  nvmlPciInfo_t pciInfo;
  NCCLCHECK(wrapNvmlDeviceGetPciInfo(nvmlDevice, &pciInfo));
  strncpy(info->busId, pciInfo.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
  return ncclSuccess;
}

template <int type>
static ncclResult_t selectTransport(struct ncclPeerInfo* myInfo, struct ncclPeerInfo* peerInfo, struct ncclConnect* connect, struct ncclConnector* connector, int buffSize, int channelId) {
  for (int t=0; t<NTRANSPORTS; t++) {
    struct ncclTransport *transport = ncclTransports+t;
    struct ncclTransportComm* transportComm = type == 1 ? &transport->send : &transport->recv;
    ncclTvalue_t ret = 0;
    NCCLCHECK(transport->canConnect(&ret, myInfo, peerInfo));
    if (ret > 0) {
      connector->transportComm = transportComm;
      NCCLCHECK(transportComm->setup(myInfo, peerInfo, connect, connector, buffSize, channelId));
      return ncclSuccess;
    }
  }
  WARN("No transport found !");
  return ncclInternalError;
}

static int log2(int n) {
 int l = 0;
 while (n>>=1) l++;
 return l;
}

static ncclResult_t ncclTreeThreshold(int nnodes, int nranks, int nChannels, ssize_t *treeThreshold) {
  int nvlink;
  NCCLCHECK(ncclNvlinkGpu(&nvlink));
  float ringbw = nvlink ? 5000*nChannels : 5000; // approx, in MB/s or B/us
  float ringlatinter = 6;
  float treelatintra = 4;
  float treelatinter = 15;
  float treebw;
  if (!nvlink) {
    treebw = ringbw * 2 / 3;
  } else {
    treebw = ringbw * 3 / 4;
    if (nnodes == 2) treebw *= 2;
  }
  float ringlat = ringlatinter*(nranks-1);
  float treelat = treelatinter*log2(nnodes)+treelatintra*(nranks/nnodes-1);
  if (nnodes < 2 || ringlat <= treelat)
    *treeThreshold = 0;
  else if (treebw > ringbw)
    *treeThreshold = 0x7fffffffffffffff;
  else
    *treeThreshold = (ssize_t)(((ringbw*treebw/(ringbw-treebw)))*(ringlat-treelat));
  return ncclSuccess;
}

static ncclResult_t setupChannel(struct ncclComm* comm, int channelId, int rank, int nranks, int* ringRanks, int* treeMasters) {
  TRACE(NCCL_INIT, "rank %d nranks %d", rank, nranks);
  NCCLCHECK(initChannel(comm, channelId));

  struct ncclChannel* channel = comm->channels+channelId;
  struct ncclRing* ring = &channel->ring;

  // Reorganize ranks to start with rank.
  int shift;
  for (shift = 0; shift<nranks; shift++) {
    if (ringRanks[shift] == rank) {
      break;
    }
  }
  for (int i=0; i<nranks; i++) {
    ring->userRanks[i] = ringRanks[(i+shift)%nranks];
  }
  int prev = ring->prev = ring->userRanks[nranks-1];
  int next = ring->next = ring->userRanks[1];

  struct ncclTree* tree = &channel->tree;
  tree->up = -1;
  tree->down[0] = tree->down[1] = tree->down[2] = -1;

  //
  // Find per-node masters and connect them via a binary tree
  //

  int nMasters = 0;
  for (int r=0; r<nranks; r++) nMasters += treeMasters[r];
  if (nMasters == 0) {
    nMasters = 1;
    treeMasters[0] = 1;
  }

  if (comm->treeThreshold == -2)
    NCCLCHECK(ncclTreeThreshold(nMasters, comm->nRanks, comm->nChannels, &comm->treeThreshold));

  if (comm->treeThreshold > 0) {
    // Compute tree depth. Not an exact value but a good approximation in most
    // cases and consistent across nodes
    tree->depth = nranks/nMasters + log2(nMasters);

    // Find my master : go backwards in the ring to find my root
    int master = 0;
    for (int i = 0; i<nranks; i++) {
      int r = ring->userRanks[(nranks-i)%nranks];
      if (treeMasters[r]) {
        master = r;
        break;
      }
    }

    int* ranks;
    NCCLCHECK(ncclCalloc(&ranks, nMasters));
    int i = 0, masterIndex = -1;
    // Build binary tree
    for (int r=0; r<nranks; r++) {
      // Create index table
      if (r == master) masterIndex = i;
      if (treeMasters[r]) ranks[i++] = r;
    }
    int btreeUp, btreeDown0, btreeDown1;
    int u0, d0_0, d0_1, u1, d1_0, d1_1;
    NCCLCHECK(ncclGetDtree(nMasters, masterIndex, &u0, &d0_0, &d0_1, &u1, &d1_0, &d1_1));
    if (channelId < DIVUP(comm->nChannels, 2)) {
      btreeUp = u0; btreeDown0 = d0_0; btreeDown1 = d0_1;
    } else {
      btreeUp = u1; btreeDown0 = d1_0; btreeDown1 = d1_1;
    }

    //
    // Now build the full tree, combining the intra-node ring and the
    // inter-node binary tree.
    //

    if (rank == master) {
      int nDown = 0;
      if (btreeUp != -1) tree->up = ranks[btreeUp];
      if (treeMasters[next] == 0) tree->down[nDown++] = next;
      if (btreeDown0 != -1) tree->down[nDown++] = ranks[btreeDown0];
      if (btreeDown1 != -1) tree->down[nDown++] = ranks[btreeDown1];
    } else {
      tree->up = prev;
      if (treeMasters[next] == 0) tree->down[0] = next;
    }
    free(ranks);
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

static ncclResult_t fillConnect(struct ncclPeerInfo* peerInfo, int nranks, int rank, int* connectTransport, ncclTvalue_t* connectValue) {
  for (int r=0; r<nranks; r++) {
    connectTransport[r] = -1;
    for (int t=0; t<NTRANSPORTS; t++) {
      NCCLCHECK(ncclTransports[t].canConnect(connectValue+r, peerInfo+rank, peerInfo+r));
      if (connectValue[r] > 0) {
        connectTransport[r] = t;
        break;
      }
    }
  }
  return ncclSuccess;
}

#define MAXWIDTH 20
#define PREFIXLEN 15
#define STRLENGTH (PREFIXLEN+5*MAXWIDTH)
void dumpMatrix(int* connectMatrix, int nranks) {
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", j);
  INFO(NCCL_INIT,"%s", line);
  for (int i=0; i<nranks; i++) {
    memset(line, ' ', STRLENGTH);
    sprintf(line, "%3d ", i);
    for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+4*j, " %3d", connectMatrix[i*nranks+j]);
    INFO(NCCL_INIT,"%s", line);
  }
}

void dumpMatrixTvalue(ncclTvalue_t* connectMatrix, int nranks) {
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+5*j, " %4d", j);
  INFO(NCCL_INIT,"%s", line);
  for (int i=0; i<nranks; i++) {
    memset(line, ' ', STRLENGTH);
    sprintf(line, "%3d ", i);
    for (int j=0; j<nranks && j<MAXWIDTH; j++) sprintf(4+line+5*j, " %4o", (int)connectMatrix[i*nranks+j]);
    INFO(NCCL_INIT,"%s", line);
  }
}


void dumpLine(int* values, int nranks, const char* prefix) {
  int prefixlen = strlen(prefix);
  char line[STRLENGTH+1];
  line[STRLENGTH] = '\0';
  memset(line, ' ', STRLENGTH);
  strncpy(line, prefix, PREFIXLEN);
  for (int i=0; i<nranks && i<MAXWIDTH; i++) sprintf(line+prefixlen+4*i, " %3d", values[i]);
  INFO(NCCL_INIT,"%s", line);
}

static ncclResult_t buildRings(int nrings, int* rings, int rank, int nranks, int* prev, int* next) {
  for (int r=0; r<nrings; r++) {
    char prefix[30];
    /*sprintf(prefix, "[%d] Channel %d Prev : ", rank, r);
    dumpLine(prev+r*nranks, nranks, prefix);
    sprintf(prefix, "[%d] Channel %d Next : ", rank, r);
    dumpLine(next+r*nranks, nranks, prefix);*/

    int current = rank;
    for (int i=0; i<nranks; i++) {
      rings[r*nranks+i] = current;
      current = next[r*nranks+current];
    }
    sprintf(prefix, "Channel %02d : ", r);
    if (rank == 0) dumpLine(rings+r*nranks, nranks, prefix);
    if (current != rank) {
      WARN("Error : ring %d does not loop back to start (%d != %d)", r, current, rank);
      return ncclInternalError;
    }
    // Check that all ranks are there
    for (int i=0; i<nranks; i++) {
      int found = 0;
      for (int j=0; j<nranks; j++) {
        if (rings[r*nranks+j] == i) {
          found = 1;
          break;
        }
      }
      if (found == 0) {
        WARN("Error : ring %d does not contain rank %d", r, i);
        return ncclInternalError;
      }
    }
  }
  return ncclSuccess;
}

void* waitForNonNullPtr(void* p) {
  volatile void** ptr = (volatile void**) p;
  while (*ptr == NULL) sched_yield();
  return (void*)*ptr;
}

ncclResult_t initParams(struct ncclComm* comm) {
  struct cudaLaunchParams* params = comm->myParams = comm->intraParams+comm->intraRank;
  params->args = &comm->argsptr;
  params->stream = NULL;
  params->sharedMem = 0;
  params->blockDim.x = 0; params->blockDim.y = params->blockDim.z = 1;
  params->gridDim.x = 0; params->gridDim.y = params->gridDim.z = 1;
  return ncclSuccess;
}

// Allocate/Set Intra Process Structures and set CG options
ncclResult_t ncclCommSetIntra(struct ncclComm* comm, int rank, int ranks, struct ncclComm* comm0) {
  comm->intraRank = rank;
  comm->intraRanks = ranks;
  comm->intraPhase = 0;

  // Alloc shared structures
  if (rank == 0) {
    assert(comm == comm0);
    int* bar;
    NCCLCHECK(ncclCalloc(&bar, 2));
    bar[0] = bar[1] = 0;
    comm->intraBarrier = bar;
    NCCLCHECK(ncclCalloc(&comm->intraParams, comm->intraRanks));
    NCCLCHECK(ncclCalloc(&comm->intraCudaDevs, comm->intraRanks));
    int* CGMode;
    NCCLCHECK(ncclCalloc(&CGMode, 1));
    *CGMode = 0x11;
    comm->intraCGMode = CGMode;
    int* CC;
    NCCLCHECK(ncclCalloc(&CC, 1));
    *CC = ncclCudaFullCompCap();
    comm->intraCC = CC;
  } else {
    comm->intraBarrier = (int*)waitForNonNullPtr(&comm0->intraBarrier);
    comm->intraParams = (struct cudaLaunchParams*)waitForNonNullPtr(&comm0->intraParams);
    comm->intraCudaDevs = (int*)waitForNonNullPtr(&comm0->intraCudaDevs);
    comm->intraCGMode = (int*)waitForNonNullPtr(&comm0->intraCGMode);
    comm->intraCC = (int*)waitForNonNullPtr(&comm0->intraCC);
  }
  comm->intraCudaDevs[comm->intraRank] = comm->cudaDev;
  NCCLCHECK(initParams(comm));

  int cgMdLaunch = 0;

  // Set CG Mode
  comm->launchMode = ncclComm::GROUP;
  char* str = getenv("NCCL_LAUNCH_MODE");
  if (comm->intraRanks == 1 || (str && strcmp(str, "PARALLEL") == 0)) {
    comm->launchMode = ncclComm::PARALLEL;
  }
  if (comm->launchMode == ncclComm::GROUP) {
    CUDACHECK(cudaStreamCreateWithFlags(&comm->groupStream, cudaStreamNonBlocking));
#if CUDART_VERSION >= 9000
    if (*comm->intraCC && (ncclCudaFullCompCap() == *comm->intraCC)) {
      // Check whether the GPU supports Cooperative Group Multi Device Launch
      (void) cudaDeviceGetAttribute(&cgMdLaunch, cudaDevAttrCooperativeMultiDeviceLaunch, comm->cudaDev);
    }
#endif
  }

  // Disable cgMdLaunch if any rank does not support it
  if (cgMdLaunch == 0) {
    *comm->intraCGMode = 0x10;
  }
  return ncclSuccess;
}

static ncclResult_t p2pSetup(struct ncclComm* comm, struct ncclChannel* channel, int nrecv, int* peerRecv, int nsend, int* peerSend) {
  TRACE(NCCL_INIT, "nsend %d nrecv %d", nsend, nrecv);
  uint32_t nSkippedSend = 0, nSkippedRecv = 0; /* for tracing */
  struct ncclConnect connect;
  struct ncclConnector* conn;
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].recv;
    if (conn->connected) { ++nSkippedRecv; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<0>(comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn, channel->buffSize, channel->id));
    NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].send;
    if (conn->connected) { ++nSkippedSend; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(selectTransport<1>(comm->peerInfo+comm->rank, comm->peerInfo+peer, &connect, conn, channel->buffSize, channel->id));
    NCCLCHECK(bootstrapSend(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
  }
  for (int i=0; i<nsend; i++) {
    int peer = peerSend[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].send;
    if (conn->connected) {++nSkippedSend; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    NCCLCHECK(conn->transportComm->connect(&connect, conn));
    conn->connected = 1;
  }
  for (int i=0; i<nrecv; i++) {
    int peer = peerRecv[i];
    if (peer == -1) continue;
    conn = &channel->peers[peer].recv;
    if (conn->connected) {++nSkippedRecv; continue; }
    memset(&connect, 0, sizeof(connect));
    NCCLCHECK(bootstrapRecv(comm->bootstrap, peer, &connect, sizeof(struct ncclConnect)));
    NCCLCHECK(conn->transportComm->connect(&connect, conn));
    conn->connected = 1;
  }
  TRACE(NCCL_INIT, "nsend %d nrecv %d nSkippedSend %u nSkippedRecv %u - DONE", nsend, nrecv, nSkippedSend, nSkippedRecv);
  return ncclSuccess;
}

static ncclResult_t initTransportsRank(struct ncclComm* comm, ncclUniqueId* commId) {
  // We use 3 AllGathers
  // 1. { peerInfo, comm }
  // 2. ConnectTransport[nranks], ConnectValue[nranks]
  // 3. { nThreads, nrings, compCap, prev[MAXCHANNELS], next[MAXCHANNELS] }

  int rank = comm->rank;
  int nranks = comm->nRanks;
  TRACE(NCCL_INIT, "rank %d nranks %d - BEGIN", rank, nranks);
  NCCLCHECK(bootstrapInit(commId, rank, nranks, &comm->bootstrap));

  // AllGather1 - begin
  struct {
    struct ncclPeerInfo peerInfo;
    struct ncclComm* comm;
  } *allGather1Data;

  NCCLCHECK(ncclCalloc(&allGather1Data, nranks));
  allGather1Data[rank].comm = comm;
  NCCLCHECK(fillInfo(&allGather1Data[rank].peerInfo, rank));
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather1Data, sizeof(*allGather1Data)));

  NCCLCHECK(ncclCalloc(&comm->peerInfo, nranks));
  for (int i = 0; i < nranks; i++) {
    memcpy(comm->peerInfo+i, &allGather1Data[i].peerInfo, sizeof(struct ncclPeerInfo));
  }
  // AllGather1 data is used again below
  // AllGather1 - end

  // AllGather2 - begin
  size_t allGather2DataRowSize = sizeof(int)*nranks + sizeof(ncclTvalue_t)*nranks;
  void *allGather2Data;
  NCCLCHECK(ncclCalloc((char **)&allGather2Data, allGather2DataRowSize*nranks));
  int *myTransportRow = (int *)((char *)allGather2Data + allGather2DataRowSize*rank);
  ncclTvalue_t *myValueRow = (ncclTvalue_t *)(myTransportRow + nranks);

  NCCLCHECK(fillConnect(comm->peerInfo, nranks, rank, myTransportRow, myValueRow));
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather2Data, allGather2DataRowSize));

  int* connectTransport;
  ncclTvalue_t* connectValue;
  NCCLCHECK(ncclCalloc(&connectTransport, nranks*nranks));
  NCCLCHECK(ncclCalloc(&connectValue, nranks*nranks));
  for (int i = 0; i < nranks; i++) {
    memcpy(connectTransport + i*nranks, (char *)allGather2Data + i*allGather2DataRowSize, sizeof(int)*nranks);
    memcpy(connectValue + i*nranks, (char *)allGather2Data + i*allGather2DataRowSize + nranks*sizeof(int), sizeof(ncclTvalue_t)*nranks);
  }
  free(allGather2Data);
  // AllGather2 - end

  //if (rank == 0) dumpMatrix(connectTransport, nranks);
  //if (rank == 0) dumpMatrixTvalue(connectValue, nranks);

  // Get my rings
  int nrings;
  int* prev, *next, *treeIn, *treeOut;
  NCCLCHECK(ncclCalloc(&prev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&next, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeIn, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeOut, nranks*MAXCHANNELS));
  comm->nThreads = getDefaultThreads();
  NCCLCHECK(ncclGetRings(&nrings, &comm->nThreads, rank, nranks, connectTransport, connectValue, prev, next, treeIn, treeOut));
  TRACE(NCCL_INIT, "rank %d nranks %d - BUILD %d RINGS", rank, nranks, nrings);
  assert(nrings <= MAXCHANNELS);
  free(connectTransport);
  free(connectValue);

  // AllGather3 - begin
  struct {
    int nThreads;
    int nrings;
    int cudaCompCap;
    int prev[MAXCHANNELS];
    int next[MAXCHANNELS];
  } *allGather3Data;

  NCCLCHECK(ncclCalloc(&allGather3Data, nranks));
  allGather3Data[rank].nThreads = comm->nThreads;
  allGather3Data[rank].nrings = nrings;
  allGather3Data[rank].cudaCompCap = ncclCudaCompCap();
  for (int r=0; r<nrings; r++) {
    allGather3Data[rank].prev[r] = *(prev+r*nranks+rank);
    allGather3Data[rank].next[r] = *(next+r*nranks+rank);
  }
  NCCLCHECK(bootstrapAllGather(comm->bootstrap, allGather3Data, sizeof(*allGather3Data)));

  // Find max nThreads
  for (int i=0; i<nranks; i++)
    comm->nThreads = std::max(allGather3Data[i].nThreads, comm->nThreads);

  // Determine the minimum CUDA Compute capability of all GPUs
  int myCompCap = allGather3Data[rank].cudaCompCap;
  int minCompCap = myCompCap;
  for (int i = 0; i < nranks; i++)
    minCompCap = std::min(allGather3Data[i].cudaCompCap, minCompCap);

  // Determine thread threshold across all GPUs
  int nnodes = 0;
  for (int r=0; r<nranks; r++) nnodes += treeIn[r];
  comm->threadThreshold = ncclThreadThreshold(minCompCap, nnodes);

  // Find min nrings across ranks
  for (int i=0; i<nranks; i++)
    nrings = std::min(allGather3Data[i].nrings, nrings);
  comm->nChannels = nrings;

  // Unpack the per ring prev/next arrays
  for (int i = 0; i < nranks; i++) {
    for (int r = 0; r < nrings; r++) {
      prev[r*nranks+i] = allGather3Data[i].prev[r];
      next[r*nranks+i] = allGather3Data[i].next[r];
    }
  }
  free(allGather3Data);
  // AllGather3 - end

  int *rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  NCCLCHECK(buildRings(nrings, rings, rank, nranks, prev, next));
  free(prev);
  free(next);
  TRACE(NCCL_INIT, "rank %d nranks %d - BUILT %d RINGS", rank, nranks, nrings);

  // Connect with prev/next for each ring
  struct ncclConnect *connect;
  NCCLCHECK(ncclCalloc(&connect, 2));
  for (int r=0; r<nrings; r++) {
    struct ncclChannel* channel = comm->channels+r;
    NCCLCHECK(setupChannel(comm, r, rank, nranks, rings+r*nranks, treeIn+r*nranks));
    NCCLCHECK(p2pSetup(comm, channel, 1, &channel->ring.prev, 1, &channel->ring.next));
    NCCLCHECK(p2pSetup(comm, channel, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up));
    NCCLCHECK(p2pSetup(comm, channel, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down));
  }
  if (comm->treeThreshold > 0) {
    char line[1024];
    line[0]='\0';
    for (int c=0; c<nrings; c++) {
      struct ncclTree* tree = &comm->channels[c].tree;
      snprintf(line+strlen(line), 1023-strlen(line), " [%d] %d->%d->%d/%d/%d",
          c, tree->up, rank, tree->down[0], tree->down[1], tree->down[2]);
    }
    line[1023] = '\0';
    INFO(NCCL_INIT, "Trees%s", line);
  }
  if (rank == 0) {
    char treeline[64];
    snprintf(treeline, 64, "enabled up to size %ld", comm->treeThreshold);
    INFO(NCCL_INIT,"Using %d threads, Min Comp Cap %d, Trees %s", comm->nThreads, minCompCap,
       comm->treeThreshold == 0 ? "disabled" :
       comm->treeThreshold == 0x7fffffffffffffff ? "enabled for all sizes" :
       treeline);
  }

  TRACE(NCCL_INIT, "rank %d nranks %d - CONNECTED %d RINGS AND TREES", rank, nranks, nrings);
  free(connect);
  free(rings);
  free(treeIn);
  free(treeOut);

  // Compute intra ranks (using AllGather1 data)
  int intraRank0 = -1, intraRank = -1, intraRanks = 0;
  for (int i = 0; i < nranks; i++) {
    if ((allGather1Data[i].peerInfo.hostHash == allGather1Data[rank].peerInfo.hostHash) &&
        (allGather1Data[i].peerInfo.pidHash == allGather1Data[rank].peerInfo.pidHash)) {
      if (intraRanks == 0) intraRank0 = i;
      if (i == rank) intraRank = intraRanks;
      intraRanks++;
    }
  }
  TRACE(NCCL_INIT,"hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
        rank, allGather1Data[rank].peerInfo.hostHash, intraRank, intraRanks, intraRank0);
  if (intraRank == -1 || intraRank0 == -1 || allGather1Data[intraRank0].comm == NULL) {
    WARN("Failed to determine intra ranks hostHash[%d] %lx intraRank %d intraRanks %d intraRank0 %d",
         rank, allGather1Data[rank].peerInfo.hostHash, intraRank, intraRanks, intraRank0);
    return ncclInternalError;
  }
  NCCLCHECK(ncclCommSetIntra(comm, intraRank, intraRanks, allGather1Data[intraRank0].comm));

  // Done with AllGather1 data
  free(allGather1Data);

  if (nnodes) NCCLCHECK(transportCreateProxy(comm));

  TRACE(NCCL_INIT, "rank %d nranks %d - DONE", rank, nranks);
  return ncclSuccess;
}

static ncclResult_t getCpuGpuAffinity(int cudaDev, cpu_set_t* mask) {
  CPU_ZERO_S(sizeof(cpu_set_t), mask);
  char* cudaPath;
  NCCLCHECK(getCudaPath(cudaDev, &cudaPath));
  char path[PATH_MAX];
  strncpy(path, cudaPath, PATH_MAX-1);
  snprintf(path+strlen(path), PATH_MAX-1-strlen(path), "/local_cpus");
  path[PATH_MAX-1] = '\0';
  int fd;
  SYSCHECKVAL(open(path, O_RDONLY), "open", fd);
  char affinityStr[sizeof(cpu_set_t)*2];
  int r = read(fd, affinityStr, sizeof(cpu_set_t)*2);
  if (r > 0)
    NCCLCHECK(ncclStrToCpuset(affinityStr, mask));
  close(fd);
  free(cudaPath);
  return ncclSuccess;
}

NCCL_PARAM(IgnoreCpuAffinity, "IGNORE_CPU_AFFINITY", 0);

static ncclResult_t setCpuAffinity(int cudaDev) {
  // Query the CPU affinity set we were provided
  cpu_set_t mask;
  SYSCHECK(sched_getaffinity(0, sizeof(cpu_set_t), &mask), "sched_getaffinity");

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&mask, affinityStr));
    TRACE(NCCL_INIT, "Current affinity for GPU %d is %s", cudaDev, affinityStr);
  }
#endif

  // Find the CPUs that are local to the supplied GPU
  cpu_set_t gpuMask;
  NCCLCHECK(getCpuGpuAffinity(cudaDev, &gpuMask));

#ifdef ENABLE_TRACE
  {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&gpuMask, affinityStr));
    TRACE(NCCL_INIT, "CPU GPU affinity for GPU %d is %s", cudaDev, affinityStr);
  }
#endif

  cpu_set_t finalMask;
  if (ncclParamIgnoreCpuAffinity())
    // Ignore the CPU affinity set and use the GPU one instead
    finalMask = gpuMask;
  else
    // Use a subset of the GPU affinity set
    CPU_AND(&finalMask, &mask, &gpuMask);

  // If there is a non empty set, use it to set affinity
  if (CPU_COUNT(&finalMask)) {
    char affinityStr[sizeof(cpu_set_t)*2];
    NCCLCHECK(ncclCpusetToStr(&finalMask, affinityStr));
    INFO(NCCL_INIT, "Setting affinity for GPU %d to %s", cudaDev, affinityStr);
    SYSCHECK(sched_setaffinity(0, sizeof(cpu_set_t), &finalMask), "sched_setaffinity");
  }
  return ncclSuccess;
}

ncclResult_t ncclCommInitRankSync(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);

  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());

  // Make sure all host memory allocation are close to the GPU
  int cudaDev;
  CUDACHECK(cudaGetDevice(&cudaDev));
  NCCLCHECK(setCpuAffinity(cudaDev));
  ncclResult_t res;

  NCCLCHECKGOTO(commAlloc(newcomm, nranks, myrank), res, cleanup);
  NCCLCHECKGOTO(initTransportsRank(*newcomm, &commId), res, cleanup);
  NCCLCHECKGOTO(devCommSetup(*newcomm), res, cleanup);

  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  NCCLCHECKGOTO(wrapNvmlShutdown(), res, cleanup);

  INFO(NCCL_INIT,"comm %p rank %d nranks %d cudaDev %d nvmlDev %d - Init COMPLETE", *newcomm, myrank, nranks, (*newcomm)->cudaDev, (*newcomm)->nvmlDev);

  return ncclSuccess;
cleanup:
  *newcomm = NULL;
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  return res;
}

NCCL_API(ncclResult_t, ncclCommInitRank, ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank);
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  char* env = getenv("NCCL_COMM_ID");
  if (env && myrank == 0) {
    NCCLCHECK(bootstrapCreateRoot(&commId, true));
  }

  NCCLCHECK(ncclInit());
  if (myrank == 0) showVersion();

  // Make sure the CUDA runtime is initialized.
  CUDACHECK(cudaFree(NULL));

  NCCLCHECK(PtrCheck(newcomm, "CommInitRank", "newcomm"));
  if (nranks < 1 || myrank < 0 || myrank >= nranks) {
    WARN("Invalid rank requested : %d/%d", myrank, nranks);
    return ncclInvalidArgument;
  }

  if (ncclAsyncMode()) {
    int cudaDev;
    CUDACHECK(cudaGetDevice(&cudaDev));
    return ncclAsyncInit(ncclCommInitRankSync, cudaDev, newcomm, nranks, commId, myrank);
  } else {
    return ncclCommInitRankSync(newcomm, nranks, commId, myrank);
  }
}

static ncclResult_t initTransportsAll(struct ncclComm** comms, const int* devs, int nranks) {
  struct ncclPeerInfo* allInfo;
  NCCLCHECK(ncclCalloc(&allInfo, nranks));
  for (int rank=0; rank<nranks; rank++) {
    CUDACHECK(cudaSetDevice(devs[rank]));
    NCCLCHECK(fillInfo(allInfo+rank, rank));
  }

  int* connectTransport;
  ncclTvalue_t* connectValue;
  NCCLCHECK(ncclCalloc(&connectTransport, nranks*nranks));
  NCCLCHECK(ncclCalloc(&connectValue, nranks*nranks));
  for (int rank=0; rank<nranks; rank++)
    NCCLCHECK(fillConnect(allInfo, nranks, rank, connectTransport+nranks*rank, connectValue+nranks*rank));

  int* prev, *prevFinal, *next, *nextFinal, *treeIn, *treeOut;
  NCCLCHECK(ncclCalloc(&prev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&prevFinal, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&next, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&nextFinal, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeIn, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeOut, nranks*MAXCHANNELS));
  int nrings = MAXCHANNELS;
  int nthreads=0;
  int myCompCap = ncclCudaCompCap();
  int minCompCap = myCompCap;
  for (int rank=0; rank<nranks; rank++) {
    CUDACHECK(cudaSetDevice(devs[rank]));
    int nringsRank;
    int nthreadsRank = getDefaultThreads();
    myCompCap = ncclCudaCompCap();
    NCCLCHECK(ncclGetRings(&nringsRank, &nthreadsRank, rank, nranks, connectTransport, connectValue, prev, next, treeIn, treeOut));
    nrings = std::min(nrings, nringsRank);
    nthreads = std::max(nthreads, nthreadsRank);
    minCompCap = std::min(minCompCap, myCompCap);
    for (int ring=0; ring<nrings; ring++) {
      int index = ring*nranks+rank;
      prevFinal[index] = prev[index];
      nextFinal[index] = next[index];
    }
  }
  free(connectTransport);
  free(connectValue);
  free(prev);
  free(next);

  INFO(NCCL_INIT,"Using %d threads, Min Comp Cap %d, Trees disabled", nthreads, minCompCap);

  int* rings;
  NCCLCHECK(ncclCalloc(&rings, nranks*MAXCHANNELS));
  NCCLCHECK(buildRings(nrings, rings, 0, nranks, prevFinal, nextFinal));
  free(prevFinal);
  free(nextFinal);

  // Determine thread threshold across all GPUs
  int threadThreshold = ncclThreadThreshold(minCompCap, 0);

  for (int rank=0; rank<nranks; rank++) {
    comms[rank]->nChannels = nrings;
    comms[rank]->nThreads = nthreads;
    comms[rank]->threadThreshold = threadThreshold;
  }

  struct ncclConnect* connect;
  NCCLCHECK(ncclCalloc(&connect, 2*nranks));
  for (int r=0; r<nrings; r++) {
    int* ringRanks = rings+r*nranks;
    for (int rank=0; rank<nranks; rank++) {
      CUDACHECK(cudaSetDevice(devs[rank]));
      struct ncclChannel* channel = comms[rank]->channels+r;
      struct ncclRing *ring = &channel->ring;
      NCCLCHECK(setupChannel(comms[rank], r, rank, nranks, ringRanks, treeIn));
      // Make sure we don't use trees, we cannot use them with initAll
      comms[rank]->treeThreshold = 0;
      int prev = channel->ring.prev = ring->userRanks[nranks-1];
      int next = channel->ring.next = ring->userRanks[1];
      struct ncclConnector* recv = &channel->peers[prev].recv;
      struct ncclConnector* send = &channel->peers[next].send;
      NCCLCHECK(selectTransport<0>(allInfo+rank, allInfo+prev, connect+rank*2+0, recv, channel->buffSize, channel->id));
      NCCLCHECK(selectTransport<1>(allInfo+rank, allInfo+next, connect+rank*2+1, send, channel->buffSize, channel->id));
    }
    for (int rank=0; rank<nranks; rank++) {
      CUDACHECK(cudaSetDevice(devs[rank]));
      struct ncclChannel* channel = comms[rank]->channels+r;
      struct ncclRing *ring = &channel->ring;
      struct ncclConnector* recv = &channel->peers[ring->prev].recv;
      struct ncclConnector* send = &channel->peers[ring->next].send;
      NCCLCHECK(recv->transportComm->connect(connect+ring->prev*2+1, recv));
      NCCLCHECK(send->transportComm->connect(connect+ring->next*2+0, send));
    }
  }
  free(connect);
  free(allInfo);
  free(rings);
  free(treeIn);
  free(treeOut);
  return ncclSuccess;
}


NCCL_API(ncclResult_t, ncclCommInitAll, ncclComm_t* comms, int ndev, const int* devlist);
ncclResult_t ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(wrapNvmlSymbols());
  NCCLCHECK(wrapNvmlInit());
  showVersion();

  INFO(NCCL_INIT,"nranks %d", ndev);

  NCCLCHECK(PtrCheck(comms, "CommInitAll", "comms"));
  if (ndev < 1) {
    WARN("Invalid device count requested : %d", ndev);
    return ncclInvalidArgument;
  }

  ncclResult_t res;
  int savedDevice;
  int rank, cudaDev;
  ncclComm_t comm = NULL;
  int* ncclDevList = NULL;
  NCCLCHECK(ncclCalloc(&ncclDevList, ndev));
  for (int i=0; i<ndev; i++) {
    ncclDevList[i] = devlist ? devlist[i] : i;
  }

  CUDACHECKGOTO(cudaGetDevice(&savedDevice), res, cleanup);

  for(rank=0; rank<ndev; ++rank)
    comms[rank] = NULL;

  cpu_set_t affinitySave;
  sched_getaffinity(0, sizeof(cpu_set_t), &affinitySave);

  for (rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    CUDACHECKGOTO(cudaSetDevice(cudaDev), res, cleanup);

    NCCLCHECK(setCpuAffinity(cudaDev));

    NCCLCHECKGOTO(commAlloc(&comm, ndev, rank), res, cleanup);
    comms[rank] = comm;

    NCCLCHECKGOTO(ncclCommSetIntra(comm, rank, ndev, comms[0]), res, cleanup);
  }

  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);

  NCCLCHECKGOTO(initTransportsAll(comms, ncclDevList, ndev), res, cleanup);

  for(rank=0; rank<ndev; ++rank) {
    cudaDev = ncclDevList[rank];
    CUDACHECKGOTO(cudaSetDevice(cudaDev), res, cleanup);
    NCCLCHECKGOTO(devCommSetup(comms[rank]), res, cleanup);
  }

  res = ncclSuccess;
  goto final;

cleanup:
  for(rank=0; rank<ndev; ++rank) {
    if(comms[rank] != NULL) {
      commFree(comms[rank]);
    }
  }

final:
  free(ncclDevList);
  if(wrapNvmlShutdown() != ncclSuccess)
    INFO(NCCL_INIT,"NCCL did not shutdown nvml properly");
  cudaSetDevice(savedDevice);
  sched_setaffinity(0, sizeof(cpu_set_t), &affinitySave);
  return res;
}


static ncclResult_t commDestroy(ncclComm_t comm) {
  int savedDevice;
#ifdef ENABLE_TRACE
  int rank = comm->rank;
#endif
  CUDACHECK(cudaGetDevice(&savedDevice));
  int commDevice = comm->cudaDev;

  if (savedDevice != commDevice) {
    CUDACHECK(cudaSetDevice(commDevice));
  }

  TRACE(NCCL_INIT, "Destroying comm %p rank %d abortFlag %d fatalError %d", comm, rank, *comm->abortFlag, comm->fatalError);

  CUDACHECK(cudaStreamSynchronize(comm->groupStream));
  NCCLCHECK(transportDestroyProxy(comm));
  NCCLCHECK(commFree(comm));

  if (savedDevice != commDevice)
    CUDACHECK(cudaSetDevice(savedDevice));

  TRACE(NCCL_INIT, "Destroyed comm %p rank %d", comm, rank);

  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommDestroy, ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  TRACE(NCCL_INIT, "comm %p rank %d nRanks %d cudaDev %d nvmlDev %d", comm, comm->rank, comm->nRanks, comm->cudaDev, comm->nvmlDev);

  // Try and prevent a double free of the comm struct (user error)
  if (comm->rank == -1 || comm->nRanks <= 0 || comm->cudaDev == -1 || comm->nvmlDev == -1) {
    WARN("comm %p has already been destroyed", comm);
    return ncclInvalidArgument;
  }

  return commDestroy(comm);
}

NCCL_API(ncclResult_t, ncclCommAbort, ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm) {
  if (comm == NULL)
    return ncclSuccess;

  // Ask anything that might still be running on the device to quit
  *comm->abortFlag = 1;

  return commDestroy(comm);
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
    default                         : return "unknown result code";
  }
}

NCCL_API(ncclResult_t, ncclCommGetAsyncError, ncclComm_t comm, ncclResult_t *asyncError);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(PtrCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  // Check device reported error
  static ncclDevError_t printedDevErr = ncclDevSuccess;
  switch(*comm->fatalDevError) {
    case ncclDevSuccess :
      break;
    case ncclDevAssertedMismatch :
      if (printedDevErr != ncclDevAssertedMismatch) {
        WARN("Mismatched collective detected, please check your collective calls at and around rank %d. You can use NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=COLL to see the collective logs", comm->rank);
        printedDevErr = ncclDevAssertedMismatch;
      }
      if (comm->fatalError == ncclSuccess) {
        comm->fatalError = ncclInvalidUsage;
      }
      break;
    case ncclDevSuspectedMismatch :
      if (printedDevErr != ncclDevSuspectedMismatch) {
        WARN("Your program may be hanging, this may be caused by a collective mismatch around rank %d. Please check your collective calls at and around this rank. You can use NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS=COLL to see the collective logs", comm->rank);
        printedDevErr = ncclDevSuspectedMismatch;
      }
      break;
    default:
      WARN("Unknown device error %d", *comm->fatalDevError);
      return ncclInternalError;
  }
  *asyncError = comm->fatalError;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCount, const ncclComm_t comm, int* count);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count) {
  NCCLCHECK(PtrCheck(comm, "CommCount", "comm"));
  NCCLCHECK(PtrCheck(count, "CommCount", "count"));
  *count = comm->nRanks;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommCuDevice, const ncclComm_t comm, int* devid);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* devid) {
  NCCLCHECK(PtrCheck(comm, "CommCuDevice", "comm"));
  NCCLCHECK(PtrCheck(devid, "CommCuDevice", "devid"));
  *devid = comm->cudaDev;
  return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclCommUserRank, const ncclComm_t comm, int* rank);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank) {
  NCCLCHECK(PtrCheck(comm, "CommUserRank", "comm"));
  NCCLCHECK(PtrCheck(rank, "CommUserRank", "rank"));
  *rank = comm->rank;
  return ncclSuccess;
}
