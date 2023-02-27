#include "net.h"
#include "bootstrap.h"
#include "checks.h"

#include <string.h>
#include <errno.h>
#include <dlfcn.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

static ncclNet_v6_t ncclNet_v4_as_v6;
static ncclNet_v6_t ncclNet_v5_as_v6;
static ncclNet_v4_t *ncclNet_v4;
static ncclNet_v5_t *ncclNet_v5;
static ncclCollNet_v6_t ncclCollNet_v4_as_v6;
static ncclCollNet_v6_t ncclCollNet_v5_as_v6;
static ncclCollNet_v4_t *ncclCollNet_v4;
static ncclCollNet_v5_t *ncclCollNet_v5;

static ncclResult_t ncclNet_v4_as_v6_getProperties(int dev, ncclNetProperties_v6_t* props) {
  ncclNetProperties_v4_t p4;
  ncclResult_t ans = ncclNet_v4->getProperties(dev, &p4);
  if (ans != ncclSuccess) return ans;
  props->name = p4.name;
  props->pciPath = p4.pciPath;
  props->guid = p4.guid;
  props->ptrSupport = p4.ptrSupport;
  props->speed = p4.speed;
  props->port = p4.port;
  props->maxComms = p4.maxComms;
  props->maxRecvs = 1;
  props->latency = 0;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v4_as_v6_isend(void* sendComm, void* data, int size, int tag, void* mhandle, void** request) {
  return ncclNet_v4->isend(sendComm, data, size, mhandle, request);
}

static ncclResult_t ncclNet_v4_as_v6_irecv(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request) {
  if (n == 0) return ncclSuccess;
  if (n != 1) return ncclInvalidArgument;
  return ncclNet_v4->irecv(recvComm, data[0], sizes[0], mhandles[0], request);
}

static ncclResult_t ncclNet_v4_as_v6_iflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  if (n == 0) return ncclSuccess;
  if (n != 1) return ncclInvalidArgument;
  return ncclNet_v4->iflush(recvComm, data[0], sizes[0], mhandles[0], request);
}

// We use a wrapper around the v4 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclNet_v4_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v4->init(logfn));
  ncclNet_v4_as_v6.name = ncclNet_v4->name;
  ncclNet_v4_as_v6.devices = ncclNet_v4->devices;
  ncclNet_v4_as_v6.getProperties = ncclNet_v4_as_v6_getProperties;
  ncclNet_v4_as_v6.listen = ncclNet_v4->listen;
  ncclNet_v4_as_v6.connect = ncclNet_v4->connect;
  ncclNet_v4_as_v6.accept = ncclNet_v4->accept;
  ncclNet_v4_as_v6.regMr = ncclNet_v4->regMr;
  ncclNet_v4_as_v6.regMrDmaBuf = NULL;
  ncclNet_v4_as_v6.deregMr = ncclNet_v4->deregMr;
  ncclNet_v4_as_v6.isend = ncclNet_v4_as_v6_isend;
  ncclNet_v4_as_v6.irecv = ncclNet_v4_as_v6_irecv;
  ncclNet_v4_as_v6.iflush = ncclNet_v4_as_v6_iflush;
  ncclNet_v4_as_v6.test = ncclNet_v4->test;
  ncclNet_v4_as_v6.closeSend = ncclNet_v4->closeSend;
  ncclNet_v4_as_v6.closeRecv = ncclNet_v4->closeRecv;
  ncclNet_v4_as_v6.closeListen = ncclNet_v4->closeListen;
  return ncclSuccess;
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclNet_v5_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v5->init(logfn));
  ncclNet_v5_as_v6.name = ncclNet_v5->name;
  ncclNet_v5_as_v6.devices = ncclNet_v5->devices;
  ncclNet_v5_as_v6.getProperties = ncclNet_v5->getProperties;
  ncclNet_v5_as_v6.listen = ncclNet_v5->listen;
  ncclNet_v5_as_v6.connect = ncclNet_v5->connect;
  ncclNet_v5_as_v6.accept = ncclNet_v5->accept;
  ncclNet_v5_as_v6.regMr = ncclNet_v5->regMr;
  ncclNet_v5_as_v6.regMrDmaBuf = NULL;
  ncclNet_v5_as_v6.deregMr = ncclNet_v5->deregMr;
  ncclNet_v5_as_v6.isend = ncclNet_v5->isend;
  ncclNet_v5_as_v6.irecv = ncclNet_v5->irecv;
  ncclNet_v5_as_v6.iflush = ncclNet_v5->iflush;
  ncclNet_v5_as_v6.test = ncclNet_v5->test;
  ncclNet_v5_as_v6.closeSend = ncclNet_v5->closeSend;
  ncclNet_v5_as_v6.closeRecv = ncclNet_v5->closeRecv;
  ncclNet_v5_as_v6.closeListen = ncclNet_v5->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v4_as_v6_getProperties(int dev, ncclNetProperties_v6_t* props) {
  ncclNetProperties_v4_t p4;
  ncclResult_t ans = ncclCollNet_v4->getProperties(dev, &p4);
  if (ans != ncclSuccess) return ans;
  props->name = p4.name;
  props->pciPath = p4.pciPath;
  props->guid = p4.guid;
  props->ptrSupport = p4.ptrSupport;
  props->speed = p4.speed;
  props->port = p4.port;
  props->maxComms = p4.maxComms;
  props->maxRecvs = 1;
  props->latency = 0;
  return ncclSuccess;
}

// We use a wrapper around the v4 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v4_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v4->init(logfn));
  ncclCollNet_v4_as_v6.name = ncclCollNet_v4->name;
  ncclCollNet_v4_as_v6.devices = ncclCollNet_v4->devices;
  ncclCollNet_v4_as_v6.getProperties = ncclCollNet_v4_as_v6_getProperties;
  ncclCollNet_v4_as_v6.listen = ncclCollNet_v4->listen;
  ncclCollNet_v4_as_v6.connect = ncclCollNet_v4->connect;
  ncclCollNet_v4_as_v6.reduceSupport = ncclCollNet_v4->reduceSupport;
  ncclCollNet_v4_as_v6.regMr = ncclCollNet_v4->regMr;
  ncclCollNet_v4_as_v6.regMrDmaBuf = NULL;
  ncclCollNet_v4_as_v6.deregMr = ncclCollNet_v4->deregMr;
  ncclCollNet_v4_as_v6.iallreduce = ncclCollNet_v4->iallreduce;
  ncclCollNet_v4_as_v6.iflush = ncclCollNet_v4->iflush;
  ncclCollNet_v4_as_v6.test = ncclCollNet_v4->test;
  ncclCollNet_v4_as_v6.closeColl = ncclCollNet_v4->closeColl;
  ncclCollNet_v4_as_v6.closeListen = ncclCollNet_v4->closeListen;
  return ncclSuccess;
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v5_as_v6_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v5->init(logfn));
  ncclCollNet_v5_as_v6.name = ncclCollNet_v5->name;
  ncclCollNet_v5_as_v6.devices = ncclCollNet_v5->devices;
  ncclCollNet_v5_as_v6.getProperties = ncclCollNet_v5->getProperties;
  ncclCollNet_v5_as_v6.listen = ncclCollNet_v5->listen;
  ncclCollNet_v5_as_v6.connect = ncclCollNet_v5->connect;
  ncclCollNet_v5_as_v6.reduceSupport = ncclCollNet_v5->reduceSupport;
  ncclCollNet_v5_as_v6.regMr = ncclCollNet_v5->regMr;
  ncclCollNet_v5_as_v6.regMrDmaBuf = NULL;
  ncclCollNet_v5_as_v6.deregMr = ncclCollNet_v5->deregMr;
  ncclCollNet_v5_as_v6.iallreduce = ncclCollNet_v5->iallreduce;
  ncclCollNet_v5_as_v6.iflush = ncclCollNet_v5->iflush;
  ncclCollNet_v5_as_v6.test = ncclCollNet_v5->test;
  ncclCollNet_v5_as_v6.closeColl = ncclCollNet_v5->closeColl;
  ncclCollNet_v5_as_v6.closeListen = ncclCollNet_v5->closeListen;
  return ncclSuccess;
}

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
ncclNet_t* ncclNets[3] = { nullptr, &ncclNetIb, &ncclNetSocket };
ncclCollNet_t* ncclCollNets[3] = { nullptr, nullptr, nullptr };
enum ncclNetState {
  ncclNetStateInit = 0,
  ncclNetStateEnabled = 1,
  ncclNetStateDisabled = 2
};
enum ncclNetState ncclNetStates[3] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };
enum ncclNetState ncclCollNetStates[3] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };

ncclResult_t ncclNetPluginInit() {
  char ncclNetPluginName[128];
  const char* envPluginName = getenv("NCCL_NET_PLUGIN");
  if (envPluginName && strlen(envPluginName)) {
    snprintf(ncclNetPluginName, 128, "libnccl-net-%s.so", envPluginName);
    INFO(NCCL_INIT, "Plugin name set by env to %s", ncclNetPluginName);
  } else {
    sprintf(ncclNetPluginName, "libnccl-net.so");
  }
  void* netPluginLib = dlopen(ncclNetPluginName, RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin load (%s) returned %d : %s", ncclNetPluginName, errno, dlerror());
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : No plugin found, using internal implementation");
    return ncclSuccess;
  }

  ncclNets[0] = (ncclNet_v6_t*)dlsym(netPluginLib, "ncclNetPlugin_v6");
  if (ncclNets[0] == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v6 symbol.");
    // Try v5 plugin
    ncclNet_v5 = (ncclNet_v5_t*)dlsym(netPluginLib, "ncclNetPlugin_v5");
    if (ncclNet_v5 == nullptr) {
      ncclNet_v4 = (ncclNet_v4_t*)dlsym(netPluginLib, "ncclNetPlugin_v4");
      if (ncclNet_v4 == nullptr) {
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin symbol (v4 or v5).");
        if (netPluginLib != nullptr) dlclose(netPluginLib);
        return ncclSuccess;
      }
      ncclNets[0] = &ncclNet_v4_as_v6;
      ncclNet_v4_as_v6.init = ncclNet_v4_as_v6_init;
      // Set the name right away to allow for NCCL_NET=... to work
      ncclNet_v4_as_v6.name = ncclNet_v4->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v4)", ncclNets[0]->name);
    } else {
      ncclNets[0] = &ncclNet_v5_as_v6;
      ncclNet_v5_as_v6.init = ncclNet_v5_as_v6_init;
      // Set the name right away to allow for NCCL_NET=... to work
      ncclNet_v5_as_v6.name = ncclNet_v5->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v5)", ncclNets[0]->name);
    }
  }

  // Check for CollNet
  ncclCollNets[0] = (ncclCollNet_v6_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v6");
  if (ncclCollNets[0] == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v6 symbol.");
    ncclCollNet_v5 = (ncclCollNet_v5_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v5");
    if (ncclCollNet_v5 == nullptr) {
      ncclCollNet_v4 = (ncclCollNet_v4_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v4");
      if (ncclCollNet_v4 == nullptr) {
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin symbol (v4 or v5).");
      } else {
        ncclCollNets[0] = &ncclCollNet_v4_as_v6;
        ncclCollNet_v4_as_v6.init = ncclCollNet_v4_as_v6_init;
        ncclCollNet_v4_as_v6.name = ncclCollNet_v4->name;
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v4)", ncclCollNets[0]->name);
      }
    } else {
      ncclCollNets[0] = &ncclCollNet_v5_as_v6;
      ncclCollNet_v5_as_v6.init = ncclCollNet_v5_as_v6_init;
      ncclCollNet_v5_as_v6.name = ncclCollNet_v5->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v5)", ncclCollNets[0]->name);
    }
  }
  return ncclSuccess;
}

static ncclResult_t netGetState(int i, enum ncclNetState* state) {
  pthread_mutex_lock(&netLock);
  if (ncclNetStates[i] == ncclNetStateInit) {
    int ndev;
    if (ncclNets[i]->init(ncclDebugLog) != ncclSuccess) ncclNetStates[i] = ncclNetStateDisabled;
    else if (ncclNets[i]->devices(&ndev) != ncclSuccess || ndev <= 0) ncclNetStates[i] = ncclNetStateDisabled;
    else ncclNetStates[i] = ncclNetStateEnabled;
  }
  *state = ncclNetStates[i];
  pthread_mutex_unlock(&netLock);
  return ncclSuccess;
}

static ncclResult_t collNetGetState(int i, enum ncclNetState* state) {
  if (ncclCollNetStates[i] == ncclNetStateInit) {
    int ndev;
    if (ncclCollNets[i]->init(ncclDebugLog) != ncclSuccess) ncclCollNetStates[i] = ncclNetStateDisabled;
    else if (ncclCollNets[i]->devices(&ndev) != ncclSuccess || ndev <= 0) ncclCollNetStates[i] = ncclNetStateDisabled;
    else ncclCollNetStates[i] = ncclNetStateEnabled;
  }
  *state = ncclCollNetStates[i];
  return ncclSuccess;
}

ncclResult_t ncclNetInit(struct ncclComm* comm) {
  // Initialize main communication network
  char* netName;
  bool ok = false;

  netName = comm->netName;
  for (int i=0; i<3; i++) {
    if (ncclNets[i] == nullptr) continue;
    enum ncclNetState state;
    NCCLCHECK(netGetState(i, &state));
    if (state != ncclNetStateEnabled) continue;
    if (netName && strcasecmp(netName, ncclNets[i]->name) != 0) continue;

    comm->ncclNet = ncclNets[i];
    ok = true;

    if (ncclCollNets[i]) {
      NCCLCHECK(collNetGetState(i, &state));
      if (state == ncclNetStateEnabled) {
        comm->ncclCollNet = ncclCollNets[i];
      }
    }
    break;
  }

  if (!ok) {
    WARN("Error: network %s not found.", netName ? netName : "");
    return ncclInvalidUsage;
  }
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
  int netDevs;
  NCCLCHECK(ncclNetDevices(comm, &netDevs));
  *gdrSupport = 0;
  for (int dev=0; dev<netDevs; dev++) {
    // Find a net device which is GDR-capable
    ncclNetProperties_t props;
    NCCLCHECK(ncclNetGetProperties(comm, dev, &props));
    if ((props.ptrSupport & NCCL_PTR_CUDA) == 0) continue;

    // Allocate memory on the GPU and try to register it on the NIC.
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    void* gpuPtr = NULL;
    void* mHandle = NULL;
    ncclResult_t ret;
    ncclDebugNoWarn = NCCL_NET;
    NCCLCHECKGOTO(ncclNetListen(comm, dev, &handle, &lComm), ret, cleanup1);

    bool connected;
    connected = false;
    while (!connected) {

      // If we're aborting now, skip to cleanup
      if (*comm->abortFlag) {
        goto cleanup2;
      }

      if (sComm == NULL)
        NCCLCHECKGOTO(ncclNetConnect(comm, dev, &handle, &sComm), ret, cleanup2);

      if (rComm == NULL)
        NCCLCHECKGOTO(ncclNetAccept(comm, lComm, &rComm), ret, cleanup2);

      connected = (rComm != NULL) && (sComm != NULL);
    }

    CUDACHECKGOTO(cudaMalloc(&gpuPtr, GPU_BUF_SIZE), ret, cleanup2);
    if (ncclNetRegMr(comm, sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle) == ncclSuccess) {
      NCCLCHECK(ncclNetDeregMr(comm, sComm, mHandle));
      NCCLCHECK(ncclNetRegMr(comm, rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle));
      NCCLCHECK(ncclNetDeregMr(comm, rComm, mHandle));
      *gdrSupport = 1;
    }
    ncclDebugNoWarn = 0;
    CUDACHECK(cudaFree(gpuPtr));
cleanup2:
    if (rComm != NULL)
      NCCLCHECK(ncclNetCloseRecv(comm, rComm));
    if (sComm != NULL)
      NCCLCHECK(ncclNetCloseSend(comm, sComm));
    NCCLCHECK(ncclNetCloseListen(comm, lComm));
cleanup1:
    break;
  }
  return ncclSuccess;
}

int ncclNetVersion(struct ncclComm* comm) {
  return (comm->ncclNet == &ncclNet_v4_as_v6) ? 4 : ((comm->ncclNet == &ncclNet_v5_as_v6) ? 5 : 6);
}
