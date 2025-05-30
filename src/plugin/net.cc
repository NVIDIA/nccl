/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include "bootstrap.h"
#include "checks.h"
#include "plugin.h"
#include "nccl_net.h"

#include <string.h>
#include <errno.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

typedef ncclNet_t* getNcclNet_t(void* netPluginLib);
typedef ncclCollNet_t* getNcclCollNet_t(void* netPluginLib);

extern getNcclNet_t getNcclNet_v6;
extern getNcclNet_t getNcclNet_v7;
extern getNcclNet_t getNcclNet_v8;
extern getNcclNet_t getNcclNet_v9;
extern getNcclNet_t getNcclNet_v10;
extern getNcclCollNet_t getNcclCollNet_v6;
extern getNcclCollNet_t getNcclCollNet_v7;
extern getNcclCollNet_t getNcclCollNet_v8;
extern getNcclCollNet_t getNcclCollNet_v9;
extern getNcclCollNet_t getNcclCollNet_v10;

NCCL_PARAM(NetPluginRefCount, "NET_PLUGIN_REF_COUNT", 1);
#define NCCL_NET_VERSION_COUNT 5
int ncclNetVersion[NCCL_NET_VERSION_COUNT] = {10, 9, 8, 7, 6};
getNcclNet_t* getNcclNet[NCCL_NET_VERSION_COUNT] = {getNcclNet_v10, getNcclNet_v9, getNcclNet_v8, getNcclNet_v7, getNcclNet_v6};
getNcclCollNet_t* getNcclCollNet[NCCL_NET_VERSION_COUNT] = {getNcclCollNet_v10, getNcclCollNet_v9, getNcclCollNet_v8, getNcclCollNet_v7,  getNcclCollNet_v6};

#define NCCL_NET_NUM_INTERNAL_PLUGINS 2

typedef enum ncclNetPluginState {
  ncclNetPluginStateDisabled        = -2,       // Plugin library failed to initialize
  ncclNetPluginStateLoadFailed      = -1,       // Plugin library failed to load
  ncclNetPluginStateLoadReady       = 0,        // Plugin library is ready to be loaded
  ncclNetPluginStateInitReady       = 1,        // Plugin library is loaded and ready to be initialized
  ncclNetPluginStateEnabled         = 2,        // Plugin library is loaded and initialized
} ncclNetPluginState_t;

#define MAX_STR_LEN 255
typedef struct netPluginLib {
  char name[MAX_STR_LEN];                       // Name of the plugin library
  void* dlHandle;                               // Handle to the plugin library
  ncclNet_t* ncclNet;                           // Pointer to the ncclNet_t structure
  int ncclNetVer;                               // Version of the nccl net plugin
  ncclCollNet_t* ncclCollNet;                   // Pointer to the ncclCollNet_t structure
  ncclNetPluginState_t ncclNetPluginState;      // State of the nccl net plugin
  ncclNetPluginState_t ncclCollNetPluginState;  // State of the nccl coll net plugin
  int ncclNetPluginRefCount;                    // Reference count for the nccl net plugin
} netPluginLib_t;

int pluginCount = 0;
bool netPluginLibsInitialized = false;
netPluginLib_t netPluginLibs[NCCL_NET_MAX_PLUGINS] = { 0 };
static pthread_mutex_t netPluginLock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t initPluginLibsOnceControl = PTHREAD_ONCE_INIT;

static ncclResult_t ncclNetPluginUnload(netPluginLib_t* pluginLib) {
  if ((pluginLib->dlHandle) && ((pluginLib->ncclNetPluginRefCount) == 0)) {
    INFO(NCCL_INIT|NCCL_NET, "Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle));
    memset(pluginLib, 0, sizeof(netPluginLib_t));
  }
  return ncclSuccess;
}

static ncclResult_t ncclNetPluginLoad(netPluginLib_t* pluginLib) {
  pluginLib->dlHandle = ncclOpenNetPluginLib(pluginLib->name);

  if (pluginLib->dlHandle == nullptr) goto fail;
  // load ncclNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
    pluginLib->ncclNetVer = ncclNetVersion[i];
    pluginLib->ncclNet = getNcclNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclNet) break;
  }

  // if we fail to find a net, exit
  if (pluginLib->ncclNet == nullptr) goto fail;

  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;

  // load ncclColNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
    pluginLib->ncclCollNet = getNcclCollNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclCollNet) break;
  }

  if (pluginLib->ncclCollNet == nullptr)
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  else
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateInitReady;

  INFO(NCCL_INIT|NCCL_NET, "Successfully loaded external plugin %s", pluginLib->name);
exit:
  return ncclSuccess;
fail:
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle));
  }
  pluginLib->ncclNetPluginState = ncclNetPluginStateLoadFailed;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  goto exit;
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

static ncclResult_t ncclNetPluginInit(netPluginLib_t* pluginLib) {
  int ndev;
  if (pluginLib->ncclNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    if (pluginLib->ncclNet->init(ncclDebugLog, ncclProfilerCallback) != ncclSuccess) goto fail;
    if (pluginLib->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0) goto fail;
  }
  pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;
  INFO(NCCL_INIT|NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);

  if (pluginLib->ncclCollNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
    if (pluginLib->ncclCollNet->init(ncclDebugLog) != ncclSuccess) pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    else if (pluginLib->ncclCollNet->devices(&ndev) != ncclSuccess || ndev <= 0) pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    else {
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateEnabled;
    }
  }
exit:
  return ncclSuccess;
fail:
  pluginLib->ncclNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  goto exit;
}

static ncclResult_t ncclNetPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
  const char* netName = comm->config.netName;
  if (netName && strcasecmp(netName, netPluginLibs[pluginIndex].ncclNet->name) != 0) goto fail;
  if (ncclSuccess != ncclNetCheckDeviceVersion(comm, netPluginLibs[pluginIndex].ncclNet, 0)) goto fail;

  if (netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateEnabled) {
    comm->ncclNet = netPluginLibs[pluginIndex].ncclNet;
    comm->ncclNetVer = netPluginLibs[pluginIndex].ncclNetVer;
    comm->netPluginIndex = pluginIndex;
    netPluginLibs[pluginIndex].ncclNetPluginRefCount++;
    *isAssigned = true;
    INFO(NCCL_INIT|NCCL_NET, "Assigned NET plugin %s to comm", netPluginLibs[pluginIndex].ncclNet->name);
    if (netPluginLibs[pluginIndex].ncclCollNetPluginState >= ncclNetPluginStateEnabled) {
      comm->ncclCollNet = netPluginLibs[pluginIndex].ncclCollNet;
    }
  }
exit:
  return ncclSuccess;
fail:
  *isAssigned = false;
  netPluginLibs[pluginIndex].ncclNetPluginState = ncclNetPluginStateEnabled;
  netPluginLibs[pluginIndex].ncclCollNetPluginState = ncclNetPluginStateEnabled;
  goto exit;
}

static ncclResult_t ncclNetPluginDisableOtherExternal(int pluginIndex) {
  // Only if an external plugin is enabled, disable other external plugins
  if (pluginIndex >= (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) return ncclSuccess;
  char names[MAX_STR_LEN*(NCCL_NET_MAX_PLUGINS - NCCL_NET_NUM_INTERNAL_PLUGINS)] = { 0 };
  for (int i = 0; i < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS); i++) {
    if (i != pluginIndex) {
      // Append all disabled plugin names to a string
      snprintf(names+strlen(names), sizeof(names)-strlen(names), (strlen(names) == 0) ? "%s" : ", %s", netPluginLibs[i].name);
      netPluginLibs[i].ncclNetPluginState = ncclNetPluginStateDisabled;
    }
  }
  if(strlen(names) > 0) {
    INFO(NCCL_INIT|NCCL_NET, "Disabling external plugins: %s", names);
  }
  return ncclSuccess;
}

static void initPluginLibsOnceFunc() {
  char* netPluginName = nullptr;
  const char* defaultNetPlugin = "libnccl-net.so";
  const char* envNetPlugin = nullptr;
  char* envNetPluginList = nullptr;
  char* savePtr = nullptr;
  int pluginCounter = 0;

  memset(netPluginLibs, 0, NCCL_NET_MAX_PLUGINS * sizeof(netPluginLib_t));
  envNetPlugin = ncclGetEnv("NCCL_NET_PLUGIN");
  if (envNetPlugin) {
    envNetPluginList = strdup(envNetPlugin);
    // Iterate over list until the list is empty
    netPluginName = strtok_r(envNetPluginList, ",", &savePtr);
    while(netPluginName) {
      // We have 2 internal plugins (ib and socket)
      // So, we can have at most( NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS)) in the NCCL_NET_PLUGIN list
      if (pluginCounter >= (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS))) {
        INFO(NCCL_NET|NCCL_INIT,"NCCL_NET_PLUGIN list contains more than %d plugins, ignoring the rest", (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS + 1)));
        break;
      }
      // need to leave space for the name + "\n"
      if((strlen(netPluginName)+1) <= MAX_STR_LEN) {
        netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;
        netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();
        strcpy(netPluginLibs[pluginCounter].name, netPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET|NCCL_INIT,"NCCL_NET_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", netPluginName, MAX_STR_LEN);
      }
      netPluginName = strtok_r(nullptr, ",", &savePtr);
    }
    if (envNetPluginList) free(envNetPluginList);
  } else {
    // Add default net plugin
    netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;
    netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();
    strcpy(netPluginLibs[pluginCounter++].name, defaultNetPlugin);
  }

  // Add 2 internal ib and socket plugins
  netPluginLibs[pluginCounter].ncclNet = &ncclNetIb;
  netPluginLibs[pluginCounter++].ncclNetPluginState = ncclNetPluginStateInitReady;
  netPluginLibs[pluginCounter].ncclNet = &ncclNetSocket;
  netPluginLibs[pluginCounter++].ncclNetPluginState = ncclNetPluginStateInitReady;
  pluginCount = pluginCounter;
}

ncclResult_t ncclNetInit(struct ncclComm* comm) {
  bool ncclNetPluginInitialized = false;
  pthread_once(&initPluginLibsOnceControl, initPluginLibsOnceFunc);
  pthread_mutex_lock(&netPluginLock);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
    if ((pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) && (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady)) {
      NCCLCHECK(ncclNetPluginLoad(&netPluginLibs[pluginIndex]));
    }
    if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateInitReady) {
      NCCLCHECK(ncclNetPluginInit(&netPluginLibs[pluginIndex]));
    }
    if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateEnabled) {
      bool isAssigned = false;
      NCCLCHECK(ncclNetPluginAssignToComm(comm, pluginIndex, &isAssigned));
      if (isAssigned) {
        // If one external plugin is assigned to a comm, then disable all other external plugins
        ncclNetPluginDisableOtherExternal(pluginIndex);
        ncclNetPluginInitialized = true;
        break;
      }
    }
  }
  pthread_mutex_unlock(&netPluginLock);
  if (ncclNetPluginInitialized) return ncclSuccess;
  WARN("Failed to initialize any NET plugin");
  return ncclInvalidUsage;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->netPluginIndex;
  pthread_mutex_lock(&netPluginLock);
  netPluginLibs[pluginIndex].ncclNetPluginRefCount--;
  for (int i = 0; i < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS); i++) {
    NCCLCHECK(ncclNetPluginUnload(&netPluginLibs[i]));
  }
  pthread_mutex_unlock(&netPluginLock);
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
    int netDevs;
    NCCLCHECK(comm->ncclNet->devices(&netDevs));
    gdrSupportMatrix[comm->cudaDev] = 0;
    for (int dev=0; dev<netDevs; dev++) {
      // Find a net device which is GDR-capable
      ncclNetProperties_t props;
      NCCLCHECK(comm->ncclNet->getProperties(dev, &props));
      if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;

    // Allocate memory on the GPU and try to register it on the NIC.
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    char* gpuPtr = NULL;
    void* mHandle = NULL;
    ncclResult_t ret;
    ncclDebugNoWarn = NCCL_NET;
    NCCLCHECKGOTO(comm->ncclNet->listen(dev, &handle, &lComm), ret, cleanup1);

    bool connected;
    connected = false;
    while (!connected) {

      // If we're aborting now, skip to cleanup
      if (__atomic_load_n(comm->abortFlag, __ATOMIC_ACQUIRE)) {
        goto cleanup2;
      }

      if (sComm == NULL)
        NCCLCHECKGOTO(comm->ncclNet->connect(dev, NULL, &handle, &sComm, NULL), ret, cleanup2);

      if (rComm == NULL)
        NCCLCHECKGOTO(comm->ncclNet->accept(lComm, &rComm, NULL), ret, cleanup2);

      connected = (rComm != NULL) && (sComm != NULL);
    }

    NCCLCHECKGOTO(ncclCudaMalloc(&gpuPtr, GPU_BUF_SIZE), ret, cleanup2);
    if (comm->ncclNet->regMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {
      NCCLCHECK(comm->ncclNet->deregMr(sComm, mHandle));
      NCCLCHECK(comm->ncclNet->regMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));
      NCCLCHECK(comm->ncclNet->deregMr(rComm, mHandle));
      gdrSupportMatrix[comm->cudaDev] = 1;
    }
    ncclDebugNoWarn = 0;
    NCCLCHECK(ncclCudaFree(gpuPtr));
cleanup2:
    if (rComm != NULL)
      NCCLCHECK(comm->ncclNet->closeRecv(rComm));
    if (sComm != NULL)
      NCCLCHECK(comm->ncclNet->closeSend(sComm));
    NCCLCHECK(comm->ncclNet->closeListen(lComm));
cleanup1:
      break;
    }
  }
  *gdrSupport = gdrSupportMatrix[comm->cudaDev];
  return ncclSuccess;
}
