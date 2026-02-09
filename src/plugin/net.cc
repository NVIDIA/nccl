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
#include <mutex>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

typedef ncclNet_t* getNcclNet_t(void* netPluginLib);
typedef ncclCollNet_t* getNcclCollNet_t(void* netPluginLib);
typedef ncclGin_t* getNcclGin_t(void* netPluginLib);

extern getNcclNet_t getNcclNet_v6;
extern getNcclNet_t getNcclNet_v7;
extern getNcclNet_t getNcclNet_v8;
extern getNcclNet_t getNcclNet_v9;
extern getNcclNet_t getNcclNet_v10;
extern getNcclNet_t getNcclNet_v11;
extern getNcclCollNet_t getNcclCollNet_v6;
extern getNcclCollNet_t getNcclCollNet_v7;
extern getNcclCollNet_t getNcclCollNet_v8;
extern getNcclCollNet_t getNcclCollNet_v9;
extern getNcclCollNet_t getNcclCollNet_v10;
extern getNcclCollNet_t getNcclCollNet_v11;
extern getNcclGin_t getNcclGin_v11;
NCCL_PARAM(NetPluginRefCount, "NET_PLUGIN_REF_COUNT", 0);
#define NCCL_NET_VERSION_COUNT 6
int ncclNetVersion[NCCL_NET_VERSION_COUNT] = {11, 10, 9, 8, 7, 6};
getNcclNet_t* getNcclNet[NCCL_NET_VERSION_COUNT] = {getNcclNet_v11, getNcclNet_v10, getNcclNet_v9, getNcclNet_v8, getNcclNet_v7, getNcclNet_v6};
getNcclCollNet_t* getNcclCollNet[NCCL_NET_VERSION_COUNT] = {getNcclCollNet_v11, getNcclCollNet_v10, getNcclCollNet_v9, getNcclCollNet_v8, getNcclCollNet_v7, getNcclCollNet_v6};
#define NCCL_GIN_VERSION_COUNT 1
getNcclGin_t* getNcclGin[NCCL_GIN_VERSION_COUNT] = {getNcclGin_v11};

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
  ncclGin_t* ncclGin;                           // Pointer to the ncclGin_t structure
  ncclNetPluginState_t ncclGinPluginState;      // State of the nccl gin plugin
  ncclGin_t* ncclRma;                           // Pointer to the ncclGin_t structure for RMA
  ncclNetPluginState_t ncclRmaPluginState;      // State of the nccl gin rma plugin
  int ncclNetPluginRefCount;                    // Reference count for the nccl net plugin
  int netPhysDevs;                              // ncclNet - number of physical devices
  int netVirtDevs;                              // ncclNet - number of virtual devices
  int collNetPhysDevs;                          // ncclCollNet -  number of physical devices
  int collNetVirtDevs;                          // ncclCollNet -  number of virtual devices
} netPluginLib_t;

int pluginCount = 0;
bool netPluginLibsInitialized = false;
netPluginLib_t netPluginLibs[NCCL_NET_MAX_PLUGINS] = { 0 };
static std::mutex netPluginMutex;
static std::once_flag initPluginLibsOnceFlag;

static ncclResult_t ncclNetPluginUnload(netPluginLib_t* pluginLib) {
  if ((pluginLib->dlHandle) && ((pluginLib->ncclNetPluginRefCount) == 0)) {
    INFO(NCCL_INIT|NCCL_NET, "Unloading plugin %s", pluginLib->name);
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));
    // memset will reset the status to ncllNetPluginStateLoadReady
    memset(pluginLib, 0, sizeof(netPluginLib_t));
    // reset the count of devices to UNDEF_DEV_COUNT
    pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
    pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
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
  if (pluginLib->ncclNet == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "External network plugin %s is unsupported",
         (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
    goto fail;
  }

  pluginLib->ncclNetPluginState = ncclNetPluginStateInitReady;

  // load ncclCollNet
  for (int i = 0; i < NCCL_NET_VERSION_COUNT; i++) {
    pluginLib->ncclCollNet = getNcclCollNet[i](pluginLib->dlHandle);
    if (pluginLib->ncclCollNet) break;
  }

  if (pluginLib->ncclCollNet == nullptr)
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateLoadFailed;
  else
    pluginLib->ncclCollNetPluginState = ncclNetPluginStateInitReady;

  // load gin
  for (int i = 0; i < NCCL_GIN_VERSION_COUNT; i++) {
    pluginLib->ncclGin = getNcclGin[i](pluginLib->dlHandle);
    if (pluginLib->ncclGin) break;
  }

  if (pluginLib->ncclGin == nullptr)
    pluginLib->ncclGinPluginState = ncclNetPluginStateLoadFailed;
  else
    pluginLib->ncclGinPluginState = ncclNetPluginStateInitReady;

  INFO(NCCL_INIT|NCCL_NET, "Successfully loaded external network plugin %s",
       (ncclPluginLibPaths[ncclPluginTypeNet] ? ncclPluginLibPaths[ncclPluginTypeNet] : pluginLib->name));
exit:
  return ncclSuccess;
fail:
  if (pluginLib->dlHandle) {
    NCCLCHECK(ncclClosePluginLib(pluginLib->dlHandle, ncclPluginTypeNet));
  }
  pluginLib->dlHandle = nullptr;
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

static ncclResult_t ncclNetPluginInit(struct ncclComm* comm, netPluginLib_t* pluginLib) {
  int ndev;
  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    ncclNetCommConfig_t commConfig = {};
    commConfig.trafficClass = comm->config.trafficClass == NCCL_CONFIG_UNDEF_INT ? NCCL_NET_TRAFFIC_CLASS_UNDEF : comm->config.trafficClass;
    if (pluginLib->ncclNet->init(&comm->netContext, comm->commHash, &commConfig, ncclDebugLog, ncclProfilerCallback) != ncclSuccess) goto fail;
  }
  // Detection of the devices is only done when the plugin is being initialized the first time
  if (pluginLib->ncclNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclNet) {
    if (pluginLib->ncclNet->devices(&ndev) != ncclSuccess || ndev <= 0) goto fail;
    pluginLib->netPhysDevs = ndev;
    pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  }
  pluginLib->ncclNetPluginState = ncclNetPluginStateEnabled;
  INFO(NCCL_INIT|NCCL_NET, "Initialized NET plugin %s", pluginLib->ncclNet->name);

  // Init must be called for each new comm to set the right context
  if (pluginLib->ncclCollNetPluginState >= ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
    if (pluginLib->ncclCollNet->init(&comm->collNetContext, comm->commHash, ncclDebugLog) != ncclSuccess) pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  }
  // Detection of the devices is only done when the plugin is being initialized the first time
  if (pluginLib->ncclCollNetPluginState == ncclNetPluginStateInitReady && pluginLib->ncclCollNet) {
    if (pluginLib->ncclCollNet->devices(&ndev) != ncclSuccess || ndev <= 0) pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
    else {
      pluginLib->collNetPhysDevs = ndev;
      pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
      pluginLib->ncclCollNetPluginState = ncclNetPluginStateEnabled;
    }
  }

  if (pluginLib->ncclGinPluginState == ncclNetPluginStateInitReady && pluginLib->ncclGin) {
    if ((ncclParamGinType() == -1) && (pluginLib->ncclGin == (ncclGin_t *)-1)) {
      void* throwAwayContext = nullptr;
      if (ncclGinIbGdaki.init(&throwAwayContext, comm->commHash, ncclDebugLog) == ncclSuccess) {
        if (ncclGinIbGdaki.devices(&ndev) == ncclSuccess && ndev > 0) {
          pluginLib->ncclGin = &ncclGinIbGdaki;
        } else {
          pluginLib->ncclGin = &ncclGinIbProxy;
        }
        ncclGinIbGdaki.finalize(throwAwayContext);
      }
      else {
        pluginLib->ncclGin = &ncclGinIbProxy;
      }
    }
    if (pluginLib->ncclGin->init(&comm->ginContext, comm->commHash, ncclDebugLog) != ncclSuccess) pluginLib->ncclGinPluginState = ncclNetPluginStateDisabled;
    else if (pluginLib->ncclGin->devices(&ndev) != ncclSuccess || ndev <= 0) pluginLib->ncclGinPluginState = ncclNetPluginStateDisabled;
    else {
      pluginLib->ncclGinPluginState = ncclNetPluginStateEnabled;
    }
  }

  // Initialize RMA plugin
  if (pluginLib->ncclRmaPluginState == ncclNetPluginStateInitReady && pluginLib->ncclRma) {
    if (pluginLib->ncclRma->init(&comm->netContext, comm->commHash, ncclDebugLog) != ncclSuccess)
      pluginLib->ncclRmaPluginState = ncclNetPluginStateDisabled;
    else if (pluginLib->ncclRma->devices(&ndev) != ncclSuccess || ndev <= 0)
      pluginLib->ncclRmaPluginState = ncclNetPluginStateDisabled;
    else {
      pluginLib->ncclRmaPluginState = ncclNetPluginStateEnabled;
    }
  }
exit:
  return ncclSuccess;
fail:
  INFO(NCCL_INIT|NCCL_NET, "Failed to initialize NET plugin %s", pluginLib->ncclNet->name);
  pluginLib->ncclNet->finalize(comm->netContext);
  pluginLib->netPhysDevs = pluginLib->netVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->collNetPhysDevs = pluginLib->collNetVirtDevs = NCCL_UNDEF_DEV_COUNT;
  pluginLib->ncclNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->ncclCollNetPluginState = ncclNetPluginStateDisabled;
  pluginLib->ncclGinPluginState = ncclNetPluginStateDisabled;
  pluginLib->ncclRmaPluginState = ncclNetPluginStateDisabled;
  goto exit;
}

static ncclResult_t ncclNetPluginAssignToComm(struct ncclComm* comm, int pluginIndex, bool* isAssigned) {
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
    if (netPluginLibs[pluginIndex].ncclGinPluginState >= ncclNetPluginStateEnabled) {
      INFO(NCCL_INIT|NCCL_NET, "Assigned GIN plugin %s to comm", netPluginLibs[pluginIndex].ncclGin->name);
      comm->sharedRes->ginState.ncclGin = netPluginLibs[pluginIndex].ncclGin;
    }
    if (netPluginLibs[pluginIndex].ncclRmaPluginState >= ncclNetPluginStateEnabled) {
      INFO(NCCL_INIT|NCCL_NET, "Assigned RMA plugin %s to comm", netPluginLibs[pluginIndex].ncclRma->name);
      comm->rmaState.rmaProxyState.ncclGin = netPluginLibs[pluginIndex].ncclRma;
    }
  }
exit:
  return ncclSuccess;
fail:
  *isAssigned = false;
  netPluginLibs[pluginIndex].ncclNetPluginState = ncclNetPluginStateEnabled;
  netPluginLibs[pluginIndex].ncclCollNetPluginState = ncclNetPluginStateEnabled;
  netPluginLibs[pluginIndex].ncclGinPluginState = ncclNetPluginStateEnabled;
  netPluginLibs[pluginIndex].ncclRmaPluginState = ncclNetPluginStateEnabled;
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
    INFO(NCCL_ENV|NCCL_NET, "NCCL_NET_PLUGIN set by environment to %s", envNetPlugin);
    if (strcasecmp(envNetPlugin, "none") == 0)
      envNetPlugin = "";
    envNetPluginList = strdup(envNetPlugin);
    // Iterate over list until the list is empty
    netPluginName = strtok_r(envNetPluginList, ",", &savePtr);
    while(netPluginName) {
      // We have 2 internal plugins (ib and socket)
      // So, we can have at most( NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS)) in the NCCL_NET_PLUGIN list
      if (pluginCounter >= (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS))) {
        INFO(NCCL_NET|NCCL_ENV,"NCCL_NET_PLUGIN list contains more than %d plugins, ignoring the rest", (NCCL_NET_MAX_PLUGINS - (NCCL_NET_NUM_INTERNAL_PLUGINS + 1)));
        break;
      }
      // need to leave space for the name + "\n"
      if((strlen(netPluginName)+1) <= MAX_STR_LEN) {
        netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateLoadReady;
        netPluginLibs[pluginCounter].ncclNetPluginRefCount = ncclParamNetPluginRefCount();
        strcpy(netPluginLibs[pluginCounter].name, netPluginName);
        pluginCounter++;
      } else {
        INFO(NCCL_NET|NCCL_ENV,"NCCL_NET_PLUGIN list contains a plugin name %s longer than %d characters, ignoring it.", netPluginName, MAX_STR_LEN);
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
  netPluginLibs[pluginCounter].ncclGin = NULL;
  if (ncclParamGinType() == -1)
    netPluginLibs[pluginCounter].ncclGin = (ncclGin_t *)-1;
  else if (ncclParamGinType() == NCCL_GIN_TYPE_PROXY)
    netPluginLibs[pluginCounter].ncclGin = &ncclGinIbProxy;
  else if (ncclParamGinType() == NCCL_GIN_TYPE_GDAKI)
    netPluginLibs[pluginCounter].ncclGin = &ncclGinIbGdaki;
  netPluginLibs[pluginCounter].ncclNetPluginState = ncclNetPluginStateInitReady;
  netPluginLibs[pluginCounter].ncclGinPluginState = netPluginLibs[pluginCounter].ncclGin ? ncclNetPluginStateInitReady : ncclNetPluginStateLoadFailed;
  netPluginLibs[pluginCounter].ncclRma = &ncclGinIbProxy;
  netPluginLibs[pluginCounter].ncclRmaPluginState = ncclNetPluginStateInitReady;
  ++pluginCounter;
  netPluginLibs[pluginCounter].ncclNet = &ncclNetSocket;
  netPluginLibs[pluginCounter++].ncclNetPluginState = ncclNetPluginStateInitReady;
  pluginCount = pluginCounter;
}

static ncclResult_t ncclNetPluginFinalize(struct ncclComm* comm, int pluginIndex) {
  NCCLCHECK(netPluginLibs[pluginIndex].ncclNet->finalize(comm->netContext));
  if (netPluginLibs[pluginIndex].ncclCollNet && netPluginLibs[pluginIndex].ncclCollNetPluginState == ncclNetPluginStateEnabled) NCCLCHECK(netPluginLibs[pluginIndex].ncclCollNet->finalize(comm->collNetContext));
  if (netPluginLibs[pluginIndex].ncclGin && netPluginLibs[pluginIndex].ncclGinPluginState == ncclNetPluginStateEnabled) NCCLCHECK(netPluginLibs[pluginIndex].ncclGin->finalize(comm->ginContext));
  netPluginLibs[pluginIndex].ncclNetPluginRefCount--;
  if (pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) {
    NCCLCHECK(ncclNetPluginUnload(&netPluginLibs[pluginIndex]));
  }
  return ncclSuccess;
}

ncclResult_t ncclNetInit(struct ncclComm* comm) {
  bool ncclNetPluginInitialized = false;
  std::call_once(initPluginLibsOnceFlag, initPluginLibsOnceFunc);
  std::lock_guard<std::mutex> lock(netPluginMutex);
  for (int pluginIndex = 0; pluginIndex < pluginCount; pluginIndex++) {
    if ((pluginIndex < (pluginCount - NCCL_NET_NUM_INTERNAL_PLUGINS)) && (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateLoadReady)) {
      NCCLCHECK(ncclNetPluginLoad(&netPluginLibs[pluginIndex]));
    }
    if ((netPluginLibs[pluginIndex].ncclNetPluginState >= ncclNetPluginStateInitReady)
        && (!comm->config.netName || (strcasecmp(comm->config.netName, netPluginLibs[pluginIndex].ncclNet->name) == 0))) {
      // plugin init must be done by all comms to setup the context, therefore we use ">="
      NCCLCHECK(ncclNetPluginInit(comm, &netPluginLibs[pluginIndex]));
      if (netPluginLibs[pluginIndex].ncclNetPluginState == ncclNetPluginStateEnabled) {
        bool isAssigned = false;
        NCCLCHECK(ncclNetPluginAssignToComm(comm, pluginIndex, &isAssigned));
        if (isAssigned) {
          // If one external plugin is assigned to a comm, then disable all other external plugins
          ncclNetPluginDisableOtherExternal(pluginIndex);
          ncclNetPluginInitialized = true;
          break;
        }
        else {
          ncclNetPluginFinalize(comm, pluginIndex);
        }
      }
    }
  }
  if (ncclNetPluginInitialized) return ncclSuccess;
  WARN("Failed to initialize any NET plugin");
  return ncclInvalidUsage;
}

ncclResult_t ncclNetInitFromParent(struct ncclComm* comm, struct ncclComm* parent) {
  ncclResult_t ret = ncclSuccess;
  comm->netContext = parent->netContext;
  comm->collNetContext = parent->collNetContext;
  comm->ginContext = parent->ginContext;
  comm->ncclNet = parent->ncclNet;
  comm->ncclCollNet = parent->ncclCollNet;
  comm->netPluginIndex = parent->netPluginIndex;
  if (comm->config.netName != NCCL_CONFIG_UNDEF_PTR && strcasecmp(comm->config.netName, parent->config.netName)) {
    WARN("Comm config netName (%s) does not match the parent (%s)", comm->config.netName, parent->config.netName);
    ret = ncclInvalidUsage;
  }
  if (comm->config.trafficClass != NCCL_CONFIG_UNDEF_INT && comm->config.trafficClass != parent->config.trafficClass) {
    INFO(NCCL_INIT, "Comm config trafficClass (%d) does not match the parent (%d)", comm->config.trafficClass, parent->config.trafficClass);
  }
  return ret;
}

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  int pluginIndex = comm->netPluginIndex;
  std::lock_guard<std::mutex> lock(netPluginMutex);
  NCCLCHECK(ncclNetPluginFinalize(comm, pluginIndex));
  return ncclSuccess;
}

ncclResult_t ncclNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled ||
     netPluginLibs[netPluginIndex].netPhysDevs == NCCL_UNDEF_DEV_COUNT) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = netPluginLibs[netPluginIndex].netPhysDevs;
  *nVirtDevs = netPluginLibs[netPluginIndex].netVirtDevs;
  return ncclSuccess;
fail:
  WARN("%s: trying to access the number of devices of an uninitialized netPlugin[%d]", __func__, netPluginIndex);
  return ncclInternalError;
}

ncclResult_t ncclCollNetGetDevCount(int netPluginIndex, int* nPhysDevs, int* nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled ||
     netPluginLibs[netPluginIndex].collNetPhysDevs == NCCL_UNDEF_DEV_COUNT) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  *nPhysDevs = netPluginLibs[netPluginIndex].collNetPhysDevs;
  *nVirtDevs = netPluginLibs[netPluginIndex].collNetVirtDevs;
  return ncclSuccess;
fail:
  WARN("%s: trying to access the number of devices of an uninitialized netPlugin[%d]", __func__, netPluginIndex);
  return ncclInternalError;
}

ncclResult_t ncclNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  netPluginLibs[netPluginIndex].netVirtDevs = nVirtDevs;
  return ncclSuccess;
fail:
  WARN("%s: failed to set the number of devices for netPlugin[%d] to %d", __func__, netPluginIndex,nVirtDevs);
  return ncclInternalError;
}

ncclResult_t ncclCollNetSetVirtDevCount(int netPluginIndex, int nVirtDevs) {
  if (netPluginLibs[netPluginIndex].ncclCollNetPluginState != ncclNetPluginStateEnabled || nVirtDevs < 0) goto fail;
  // lock not needed as it's called within a lock already in ncclTopoGetSystem
  netPluginLibs[netPluginIndex].collNetVirtDevs = nVirtDevs;
  return ncclSuccess;
fail:
  WARN("%s: failed to set the number of devices for netPlugin[%d] to %d", __func__, netPluginIndex,nVirtDevs);
  return ncclInternalError;
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
    NCCLCHECKGOTONOWARN(comm->ncclNet->listen(comm->netContext, dev, &handle, &lComm), ret, cleanup1, NCCL_NET);

    bool connected;
    connected = false;
    while (!connected) {

      // If we're aborting now, skip to cleanup
      if (COMPILER_ATOMIC_LOAD(comm->abortFlag, std::memory_order_acquire)) {
        goto cleanup2;
      }

      if (sComm == NULL)
        NCCLCHECKGOTONOWARN(comm->ncclNet->connect(comm->netContext, dev, &handle, &sComm, NULL), ret, cleanup2, NCCL_NET);

      if (rComm == NULL)
        NCCLCHECKGOTONOWARN(comm->ncclNet->accept(lComm, &rComm, NULL), ret, cleanup2, NCCL_NET);

      connected = (rComm != NULL) && (sComm != NULL);
    }

    NCCLCHECKGOTONOWARN(ncclCudaMalloc(&gpuPtr, GPU_BUF_SIZE), ret, cleanup2, NCCL_NET);
    NOWARN(ret = comm->ncclNet->regMr(sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), NCCL_NET);
    if (ret == ncclSuccess) {
      NCCLCHECKNOWARN(comm->ncclNet->deregMr(sComm, mHandle), NCCL_NET);
      NCCLCHECKNOWARN(comm->ncclNet->regMr(rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), NCCL_NET);
      NCCLCHECKNOWARN(comm->ncclNet->deregMr(rComm, mHandle), NCCL_NET);
      gdrSupportMatrix[comm->cudaDev] = 1;
    }
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
