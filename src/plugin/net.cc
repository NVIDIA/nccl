/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_net.h"
#include <string.h>
#include <errno.h>

extern ncclNet_t* getNcclNet_v6(void* netPluginLib);
extern ncclNet_t* getNcclNet_v7(void* netPluginLib);
extern ncclNet_t* getNcclNet_v8(void* netPluginLib);
extern ncclNet_t* getNcclNet_v9(void* netPluginLib);
extern ncclNet_t* getNcclNet_v10(void* netPluginLib);
extern ncclNet_t* getNcclNet_v11(void* netPluginLib);

extern ncclCollNet_t* getNcclCollNet_v6(void* netPluginLib);
extern ncclCollNet_t* getNcclCollNet_v7(void* netPluginLib);
extern ncclCollNet_t* getNcclCollNet_v8(void* netPluginLib);
extern ncclCollNet_t* getNcclCollNet_v9(void* netPluginLib);
extern ncclCollNet_t* getNcclCollNet_v10(void* netPluginLib);
extern ncclCollNet_t* getNcclCollNet_v11(void* netPluginLib);

extern void* openNetPluginLib(const char* name);
extern void closePluginLib(void* handle);

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
ncclNet_t* ncclNets[NCCL_NET_MAX_PLUGINS] = { nullptr, &ncclNetIb, &ncclNetSocket };
static int ncclNetsVer[NCCL_NET_MAX_PLUGINS] = { -1, NCCL_NET_PLUGIN_VERSION, NCCL_NET_PLUGIN_VERSION };
ncclCollNet_t* ncclCollNets[NCCL_NET_MAX_PLUGINS] = { nullptr, nullptr, nullptr };
enum ncclNetState {
  ncclNetStateInit = 0,
  ncclNetStateEnabled = 1,
  ncclNetStateDisabled = 2
};
enum ncclNetState ncclNetStates[NCCL_NET_MAX_PLUGINS] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };
enum ncclNetState ncclCollNetStates[NCCL_NET_MAX_PLUGINS] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };

NCCL_PARAM(NetPluginRefCount, "NET_PLUGIN_REF_COUNT", 1);
static pthread_mutex_t netPluginLock = PTHREAD_MUTEX_INITIALIZER;
static void* netPluginLib;

static int netPluginRefCount;
static void initNetPluginRefCountOnce(void) { netPluginRefCount = ncclParamNetPluginRefCount();}

enum {
  netPluginLoadFailed  = -1,
  netPluginLoadReady   =  0,
  netPluginLoadSuccess =  1,
};

static int netPluginStatus = netPluginLoadReady;

ncclResult_t ncclNetPluginLoad(struct ncclComm* comm) {
  static pthread_once_t netPluginRefCountOnce = PTHREAD_ONCE_INIT;
  pthread_once(&netPluginRefCountOnce, initNetPluginRefCountOnce);

  pthread_mutex_lock(&netPluginLock);
  if (netPluginLoadFailed == netPluginStatus) {
    goto exit;
  }
  if (netPluginLoadSuccess == netPluginStatus) {
    ++netPluginRefCount;
    goto exit;
  }

  netPluginLib = ncclOpenNetPluginLib(ncclGetEnv("NCCL_NET_PLUGIN"));
  if (netPluginLib == nullptr) {
    goto fail;
  }

  ncclNets[0] = getNcclNet_v11(netPluginLib);
  if (ncclNets[0]) ncclNetsVer[0] = 11;
  if (ncclNets[0] == nullptr) {
    // Try v10 plugin
    ncclNets[0] = getNcclNet_v10(netPluginLib);
    if (ncclNets[0]) ncclNetsVer[0] = 10;
  }
  if (ncclNets[0] == nullptr) {
    // Try v9 plugin
    ncclNets[0] = getNcclNet_v9(netPluginLib);
    if (ncclNets[0]) ncclNetsVer[0] = 9;
  }
  if (ncclNets[0] == nullptr) {
    // Try v8 plugin
    ncclNets[0] = getNcclNet_v8(netPluginLib);
    if (ncclNets[0]) ncclNetsVer[0] = 8;
  }
  if (ncclNets[0] == nullptr) {
    // Try v7 plugin
    ncclNets[0] = getNcclNet_v7(netPluginLib);
    if (ncclNets[0]) ncclNetsVer[0] = 7;
  }
  if (ncclNets[0] == nullptr) {
    // Try v6 plugin
    ncclNets[0] = getNcclNet_v6(netPluginLib);
    if (ncclNets[0]) ncclNetsVer[0] = 6;
  }
  if (ncclNets[0] == nullptr) {
    goto fail;
  }

  // Check for CollNet
  ncclCollNets[0] = getNcclCollNet_v11(netPluginLib);
  if (ncclCollNets[0] == nullptr) {
    ncclCollNets[0] = getNcclCollNet_v10(netPluginLib);
  }
  if (ncclCollNets[0] == nullptr) {
    ncclCollNets[0] = getNcclCollNet_v9(netPluginLib);
  }
  if (ncclCollNets[0] == nullptr) {
    ncclCollNets[0] = getNcclCollNet_v8(netPluginLib);
  }
  if (ncclCollNets[0] == nullptr) {
    ncclCollNets[0] = getNcclCollNet_v7(netPluginLib);
  }
  if (ncclCollNets[0] == nullptr) {
    ncclCollNets[0] = getNcclCollNet_v6(netPluginLib);
  }

  ++netPluginRefCount;
  netPluginStatus = netPluginLoadSuccess;
  comm->netPluginLoaded = 1;

exit:
  pthread_mutex_unlock(&netPluginLock);
  return ncclSuccess;
fail:
  if (netPluginLib) NCCLCHECK(ncclClosePluginLib(netPluginLib));
  netPluginStatus = netPluginLoadFailed;
  ncclNetsVer[0] = -1;
  goto exit;
}

ncclResult_t ncclNetPluginUnload(struct ncclComm* comm) {
  pthread_mutex_lock(&netPluginLock);
  if (comm->netPluginLoaded && 0 == (--netPluginRefCount)) {
    if (ncclNets[0]) {
      INFO(NCCL_NET, "NET/Plugin: Closing net plugin '%s'", ncclNets[0]->name);
    }
    if (ncclCollNets[0]) {
      INFO(NCCL_NET, "NET/Plugin: Closing collnet plugin '%s'", ncclCollNets[0]->name);
    }
    NCCLCHECK(ncclClosePluginLib(netPluginLib));
    netPluginLib = nullptr;
    ncclNets[0] = nullptr;
    ncclCollNets[0] = nullptr;
    netPluginStatus = netPluginLoadReady;
    comm->netPluginLoaded = 0;
    for (int i = 0; i < NCCL_NET_MAX_PLUGINS; ++i)
      ncclCollNetStates[i] = ncclNetStates[i] = ncclNetStateInit;
  }
  pthread_mutex_unlock(&netPluginLock);
  return ncclSuccess;
}

ncclResult_t ncclNetCheckDeviceVersion(struct ncclComm* comm, ncclNet_t* net, int dev) {
  ncclNetProperties_t props;

  NCCLCHECK(net->getProperties(dev, &props));
  ncclNetDeviceType type = props.netDeviceType;
  if (type) switch (type) {
    case NCCL_NET_DEVICE_UNPACK:
      if (props.netDeviceVersion == NCCL_NET_DEVICE_UNPACK_VERSION) {
        INFO(NCCL_INIT, "Using NCCL_NET_DEVICE_UNPACK net plugin version %d",
          props.netDeviceVersion);
        return ncclSuccess;
      } else {
        WARN("NCCL_DEVICE_UNPACK plugin has incompatible version %d, this NCCL build is compatible with %d, not using it",
          props.netDeviceVersion, NCCL_NET_DEVICE_UNPACK_VERSION);
        return ncclInternalError;
      }
    default:
      WARN("Unknown device code index %d \n", type);
      return ncclInternalError;
  }

  return ncclSuccess;
}

static ncclResult_t netGetState(int i, enum ncclNetState* state) {
  pthread_mutex_lock(&netLock);
  if (ncclNetStates[i] == ncclNetStateInit) {
    int ndev;
    if (ncclNets[i]->init(ncclDebugLog, ncclProfilerCallback) != ncclSuccess) ncclNetStates[i] = ncclNetStateDisabled;
    else if (ncclNets[i]->devices(&ndev) != ncclSuccess || ndev <= 0) ncclNetStates[i] = ncclNetStateDisabled;
    else ncclNetStates[i] = ncclNetStateEnabled;
  }
  *state = ncclNetStates[i];
  pthread_mutex_unlock(&netLock);
  return ncclSuccess;
}

static ncclResult_t collNetGetState(int i, enum ncclNetState* state) {
  pthread_mutex_lock(&netLock);
  if (ncclCollNetStates[i] == ncclNetStateInit) {
    int ndev;
    if (ncclCollNets[i]->init(ncclDebugLog) != ncclSuccess) ncclCollNetStates[i] = ncclNetStateDisabled;
    else if (ncclCollNets[i]->devices(&ndev) != ncclSuccess || ndev <= 0) ncclCollNetStates[i] = ncclNetStateDisabled;
    else ncclCollNetStates[i] = ncclNetStateEnabled;
  }
  *state = ncclCollNetStates[i];
  pthread_mutex_unlock(&netLock);
  return ncclSuccess;
}

ncclResult_t ncclNetInit(struct ncclComm* comm) {
  // get the list of user-provided nets, reuse the code for interfaces parsing.
  int nUserIfs;
  const char* netName = comm->config.netName;
  struct netIf userNets[NCCL_NET_MAX_PLUGINS];
  NCCLCHECK(parseIfList(netName, userNets, NCCL_NET_MAX_PLUGINS, &nUserIfs));

  const char* dcNet = ncclGetEnv("NCCL_ALLNET_FASTNET");
  comm->ncclDcNetIndex = 0;
  comm->ncclNetCount = 0;
  bool ok = false;
  for (int i=0; i<NCCL_NET_MAX_PLUGINS; i++) {
    if (ncclNets[i] == nullptr) continue;
    enum ncclNetState state;
    NCCLCHECK(netGetState(i, &state));
    if (state != ncclNetStateEnabled) continue;
    if (!matchIfList(ncclNets[i]->name, /*port=*/-1, userNets, nUserIfs, 1)) continue;
    if (ncclSuccess != ncclNetCheckDeviceVersion(comm, ncclNets[i], 0)) {
      // Mismatched device plugin version
      continue;
    }

    comm->ncclNetPluginIdx[comm->ncclNetCount] = i;
    comm->ncclNetVer[comm->ncclNetCount] = ncclNetsVer[i];
    comm->ncclNet[comm->ncclNetCount] = ncclNets[i];
    if (dcNet && strcmp(dcNet, ncclNets[i]->name) == 0) comm->ncclDcNetIndex = i;
    ok = true;

    // try to load the collNet if we don't use ALLNET and it's the first net to be loaded
    if (comm->ncclNetCount == 0 && ncclCollNets[i]) {
      if (ncclParamAllNet()) {
        INFO(NCCL_INIT | NCCL_NET, "NCCL_ALLNET_ENABLE=1, ignore CollNet. Set NCCL_ALLNET_ENABLE=0 if CollNet is needed.");
      } else {
        NCCLCHECK(collNetGetState(i, &state));
        if (state == ncclNetStateEnabled) {
          comm->ncclCollNetPluginIdx = i;
          comm->ncclCollNet = ncclCollNets[i];
        }
      }
    }
    comm->ncclNetCount++;

    if (!ncclParamAllNet()) break;
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return ncclInvalidUsage;
  }
  return ncclSuccess;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  for (int n = 0; n < comm->ncclNetCount; ++n) {
    comm->ncclNet[n] = nullptr;
  }
  comm->ncclCollNet = nullptr;
  return ncclSuccess;
}

ncclResult_t ncclGpuGdrSupport(struct ncclComm* comm, int* gdrSupport) {
  constexpr int GPU_BUF_SIZE = 2*1024*1024;
#if CUDART_VERSION >= 11030
  // In CUDA 11.3 and later we can now query the cudaDevAttrGPUDirectRDMASupported attribute
  int driverVersion;
  CUDACHECK(cudaDriverGetVersion(&driverVersion));
  if (driverVersion >= 11030) {
    int cudaDev, attr = 0;
    CUDACHECK(cudaGetDevice(&cudaDev));
    CUDACHECK(cudaDeviceGetAttribute(&attr, cudaDevAttrGPUDirectRDMASupported, cudaDev));
    *gdrSupport = attr;
    return ncclSuccess;
  }
#endif
  static int gdrSupportMatrix[32] = {
	  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
	  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };
  if (gdrSupportMatrix[comm->cudaDev] == -1) {
    gdrSupportMatrix[comm->cudaDev] = 0;

    // find any device on any net that is GDR-capable
    int dev = -1;
    ncclNet_t * net = NULL;
    for (int n = 0; n < comm->ncclNetCount; ++n) {
      net = comm->ncclNet[n];
      int netDevs;
      NCCLCHECK(net->devices(&netDevs));
      for (dev = 0; dev < netDevs; dev++) {
        ncclNetProperties_t props;
        NCCLCHECK(net->getProperties(dev, &props));
        // if cuda is supported, select the net and device, then return
        if (props.ptrSupport & NCCL_PTR_CUDA) break;
      }
    }

    // Allocate memory on the GPU and try to register it on the NIC.
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    char* gpuPtr = NULL;
    void* mHandle = NULL;
    ncclResult_t ret;
    ncclDebugNoWarn = NCCL_NET;
    NCCLCHECKGOTO(net->listen(dev, &handle, &lComm), ret, cleanup1);

    bool connected;
    connected = false;
    while (!connected) {
      // If we're aborting now, skip to cleanup
      if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
        goto cleanup2;
      }
      if (sComm == NULL)
        NCCLCHECKGOTO(net->connect(dev, NULL, &handle, &sComm, NULL), ret, cleanup2);
      if (rComm == NULL)
        NCCLCHECKGOTO(net->accept(lComm, &rComm, NULL), ret, cleanup2);
      connected = (rComm != NULL) && (sComm != NULL);
    }

    NCCLCHECKGOTO(ncclCudaMalloc(&gpuPtr, GPU_BUF_SIZE), ret, cleanup2);
    if (net->regMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {
      NCCLCHECK(net->deregMr(sComm, mHandle));
      NCCLCHECK(net->regMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));
      NCCLCHECK(net->deregMr(rComm, mHandle));
      gdrSupportMatrix[comm->cudaDev] = 1;
    }
    ncclDebugNoWarn = 0;
    NCCLCHECK(ncclCudaFree(gpuPtr));
  cleanup2:
    if (rComm != NULL) NCCLCHECK(net->closeRecv(rComm));
    if (sComm != NULL) NCCLCHECK(net->closeSend(sComm));
    NCCLCHECK(net->closeListen(lComm));
  }
cleanup1:
  *gdrSupport = gdrSupportMatrix[comm->cudaDev];
  return ncclSuccess;
}
