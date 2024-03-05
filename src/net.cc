/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"
#include "bootstrap.h"
#include "checks.h"

#include <string.h>
#include <errno.h>
#include <dlfcn.h>
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>

static ncclNet_v8_t ncclNet_v5_as_v8;
static ncclNet_v8_t ncclNet_v6_as_v8;
static ncclNet_v8_t ncclNet_v7_as_v8;
static ncclNet_v5_t *ncclNet_v5;
static ncclNet_v6_t *ncclNet_v6;
static ncclNet_v7_t *ncclNet_v7;
static ncclCollNet_v8_t ncclCollNet_v5_as_v8;
static ncclCollNet_v8_t ncclCollNet_v6_as_v8;
static ncclCollNet_v8_t ncclCollNet_v7_as_v8;
static ncclCollNet_v5_t *ncclCollNet_v5;
static ncclCollNet_v6_t *ncclCollNet_v6;
static ncclCollNet_v7_t *ncclCollNet_v7;

static ncclResult_t ncclNet_v7_as_v8_getProperties(int dev, ncclNetProperties_v8_t* props) {
  ncclNetProperties_v7_t p7;
  ncclResult_t ans = ncclNet_v7->getProperties(dev, &p7);
  if (ans != ncclSuccess) return ans;
  props->name = p7.name;
  props->pciPath = p7.pciPath;
  props->guid = p7.guid;
  props->ptrSupport = p7.ptrSupport;
  props->regIsGlobal = 0;
  props->speed = p7.speed;
  props->port = p7.port;
  props->maxComms = p7.maxComms;
  props->maxRecvs = p7.maxRecvs;
  props->latency = p7.latency;
  props->netDeviceType = p7.netDeviceType;
  props->netDeviceVersion = p7.netDeviceVersion;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v7_as_v8_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1<<31) return ncclInternalError;
  return ncclNet_v7->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v7_as_v8_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v7->init(logfn));
  ncclNet_v7_as_v8.name = ncclNet_v7->name;
  ncclNet_v7_as_v8.devices = ncclNet_v7->devices;
  ncclNet_v7_as_v8.getProperties = ncclNet_v7_as_v8_getProperties; // ncclNet_v5->getProperties;
  ncclNet_v7_as_v8.listen = ncclNet_v7->listen;
  ncclNet_v7_as_v8.connect = ncclNet_v7->connect;
  ncclNet_v7_as_v8.accept =  ncclNet_v7->accept;
  ncclNet_v7_as_v8.regMr = ncclNet_v7_as_v8_regMr;
  ncclNet_v7_as_v8.regMrDmaBuf = ncclNet_v7->regMrDmaBuf;
  ncclNet_v7_as_v8.deregMr = ncclNet_v7->deregMr;
  ncclNet_v7_as_v8.isend = ncclNet_v7->isend;
  ncclNet_v7_as_v8.irecv = ncclNet_v7->irecv;
  ncclNet_v7_as_v8.iflush = ncclNet_v7->iflush;
  ncclNet_v7_as_v8.test = ncclNet_v7->test;
  ncclNet_v7_as_v8.closeSend = ncclNet_v7->closeSend;
  ncclNet_v7_as_v8.closeRecv = ncclNet_v7->closeRecv;
  ncclNet_v7_as_v8.closeListen = ncclNet_v7->closeListen;
  ncclNet_v7_as_v8.getDeviceMr = ncclNet_v7->getDeviceMr;
  ncclNet_v7_as_v8.irecvConsumed = ncclNet_v7->irecvConsumed;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v6_as_v8_getProperties(int dev, ncclNetProperties_v8_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v6_as_v8_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1<<31) return ncclInternalError;
  return ncclNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v6_as_v8_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v6->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_v6_as_v8_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v6->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_v6_as_v8_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v6->init(logfn));
  ncclNet_v6_as_v8.name = ncclNet_v6->name;
  ncclNet_v6_as_v8.devices = ncclNet_v6->devices;
  ncclNet_v6_as_v8.getProperties = ncclNet_v6_as_v8_getProperties; // ncclNet_v5->getProperties;
  ncclNet_v6_as_v8.listen = ncclNet_v6->listen;
  ncclNet_v6_as_v8.connect = ncclNet_v6_as_v8_connect;
  ncclNet_v6_as_v8.accept =  ncclNet_v6_as_v8_accept;
  ncclNet_v6_as_v8.regMr = ncclNet_v6_as_v8_regMr;
  ncclNet_v6_as_v8.regMrDmaBuf = ncclNet_v6->regMrDmaBuf;
  ncclNet_v6_as_v8.deregMr = ncclNet_v6->deregMr;
  ncclNet_v6_as_v8.isend = ncclNet_v6->isend;
  ncclNet_v6_as_v8.irecv = ncclNet_v6->irecv;
  ncclNet_v6_as_v8.iflush = ncclNet_v6->iflush;
  ncclNet_v6_as_v8.test = ncclNet_v6->test;
  ncclNet_v6_as_v8.closeSend = ncclNet_v6->closeSend;
  ncclNet_v6_as_v8.closeRecv = ncclNet_v6->closeRecv;
  ncclNet_v6_as_v8.closeListen = ncclNet_v6->closeListen;
  ncclNet_v6_as_v8.getDeviceMr = NULL;
  ncclNet_v6_as_v8.irecvConsumed = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v5_as_v8_getProperties(int dev, ncclNetProperties_v8_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v5->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v5_as_v8_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1<<31) return ncclInternalError;
  return ncclNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v5_as_v8_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v5->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_v5_as_v8_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v5->accept(listenComm, recvComm);
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclNet_v5_as_v8_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v5->init(logfn));
  ncclNet_v5_as_v8.name = ncclNet_v5->name;
  ncclNet_v5_as_v8.devices = ncclNet_v5->devices;
  ncclNet_v5_as_v8.getProperties = ncclNet_v5_as_v8_getProperties;
  ncclNet_v5_as_v8.listen = ncclNet_v5->listen;
  ncclNet_v5_as_v8.connect = ncclNet_v5_as_v8_connect;
  ncclNet_v5_as_v8.accept =  ncclNet_v5_as_v8_accept;
  ncclNet_v5_as_v8.regMr = ncclNet_v5_as_v8_regMr;
  ncclNet_v5_as_v8.regMrDmaBuf = NULL;
  ncclNet_v5_as_v8.deregMr = ncclNet_v5->deregMr;
  ncclNet_v5_as_v8.isend = ncclNet_v5->isend;
  ncclNet_v5_as_v8.irecv = ncclNet_v5->irecv;
  ncclNet_v5_as_v8.iflush = ncclNet_v5->iflush;
  ncclNet_v5_as_v8.test = ncclNet_v5->test;
  ncclNet_v5_as_v8.closeSend = ncclNet_v5->closeSend;
  ncclNet_v5_as_v8.closeRecv = ncclNet_v5->closeRecv;
  ncclNet_v5_as_v8.closeListen = ncclNet_v5->closeListen;
  ncclNet_v5_as_v8.getDeviceMr = NULL;
  ncclNet_v5_as_v8.irecvConsumed = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v5_as_v8_getProperties(int dev, ncclNetProperties_v8_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclCollNet_v5->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v5_as_v8_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1<<31) return ncclInternalError;
  return ncclCollNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v5_as_v8_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v5->init(logfn));
  ncclCollNet_v5_as_v8.name = ncclCollNet_v5->name;
  ncclCollNet_v5_as_v8.devices = ncclCollNet_v5->devices;
  ncclCollNet_v5_as_v8.getProperties = ncclCollNet_v5_as_v8_getProperties;
  ncclCollNet_v5_as_v8.listen = ncclCollNet_v5->listen;
  ncclCollNet_v5_as_v8.connect = ncclCollNet_v5->connect;
  ncclCollNet_v5_as_v8.reduceSupport = ncclCollNet_v5->reduceSupport;
  ncclCollNet_v5_as_v8.regMr = ncclCollNet_v5_as_v8_regMr;
  ncclCollNet_v5_as_v8.regMrDmaBuf = NULL;
  ncclCollNet_v5_as_v8.deregMr = ncclCollNet_v5->deregMr;
  ncclCollNet_v5_as_v8.iallreduce = ncclCollNet_v5->iallreduce;
  ncclCollNet_v5_as_v8.iallgather = nullptr;
  ncclCollNet_v5_as_v8.ireducescatter = nullptr;
  ncclCollNet_v5_as_v8.iflush = ncclCollNet_v5->iflush;
  ncclCollNet_v5_as_v8.test = ncclCollNet_v5->test;
  ncclCollNet_v5_as_v8.closeColl = ncclCollNet_v5->closeColl;
  ncclCollNet_v5_as_v8.closeListen = ncclCollNet_v5->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v6_as_v8_getProperties(int dev, ncclNetProperties_v8_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclCollNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v6_as_v8_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1<<31) return ncclInternalError;
  return ncclCollNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

// We use a wrapper around the v6 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v6_as_v8_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v6->init(logfn));
  ncclCollNet_v6_as_v8.name = ncclCollNet_v6->name;
  ncclCollNet_v6_as_v8.devices = ncclCollNet_v6->devices;
  ncclCollNet_v6_as_v8.getProperties = ncclCollNet_v6_as_v8_getProperties;
  ncclCollNet_v6_as_v8.listen = ncclCollNet_v6->listen;
  ncclCollNet_v6_as_v8.connect = ncclCollNet_v6->connect;
  ncclCollNet_v6_as_v8.reduceSupport = ncclCollNet_v6->reduceSupport;
  ncclCollNet_v6_as_v8.regMr = ncclCollNet_v6_as_v8_regMr;
  ncclCollNet_v6_as_v8.regMrDmaBuf = ncclCollNet_v6->regMrDmaBuf;
  ncclCollNet_v6_as_v8.deregMr = ncclCollNet_v6->deregMr;
  ncclCollNet_v6_as_v8.iallreduce = ncclCollNet_v6->iallreduce;
  ncclCollNet_v6_as_v8.iallgather = nullptr;
  ncclCollNet_v6_as_v8.ireducescatter = nullptr;
  ncclCollNet_v6_as_v8.iflush = ncclCollNet_v6->iflush;
  ncclCollNet_v6_as_v8.test = ncclCollNet_v6->test;
  ncclCollNet_v6_as_v8.closeColl = ncclCollNet_v6->closeColl;
  ncclCollNet_v6_as_v8.closeListen = ncclCollNet_v6->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v7_as_v8_getProperties(int dev, ncclNetProperties_v8_t* props) {
  ncclNetProperties_v7_t p7;
  ncclResult_t ans = ncclCollNet_v7->getProperties(dev, &p7);
  if (ans != ncclSuccess) return ans;
  props->name = p7.name;
  props->pciPath = p7.pciPath;
  props->guid = p7.guid;
  props->ptrSupport = p7.ptrSupport;
  props->regIsGlobal = 0;
  props->speed = p7.speed;
  props->port = p7.port;
  props->maxComms = p7.maxComms;
  props->maxRecvs = p7.maxRecvs;
  props->latency = p7.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v7_as_v8_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1<<31) return ncclInternalError;
  return ncclCollNet_v7->regMr(comm, data, (int) size, type, mhandle);
}

// We use a wrapper around the v7 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v7_as_v8_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v7->init(logfn));
  ncclCollNet_v7_as_v8.name = ncclCollNet_v7->name;
  ncclCollNet_v7_as_v8.devices = ncclCollNet_v7->devices;
  ncclCollNet_v7_as_v8.getProperties = ncclCollNet_v7_as_v8_getProperties;
  ncclCollNet_v7_as_v8.listen = ncclCollNet_v7->listen;
  ncclCollNet_v7_as_v8.connect = ncclCollNet_v7->connect;
  ncclCollNet_v7_as_v8.reduceSupport = ncclCollNet_v7->reduceSupport;
  ncclCollNet_v7_as_v8.regMr = ncclCollNet_v7_as_v8_regMr;
  ncclCollNet_v7_as_v8.regMrDmaBuf = ncclCollNet_v7->regMrDmaBuf;
  ncclCollNet_v7_as_v8.deregMr = ncclCollNet_v7->deregMr;
  ncclCollNet_v7_as_v8.iallreduce = ncclCollNet_v7->iallreduce;
  ncclCollNet_v7_as_v8.iallgather = nullptr;
  ncclCollNet_v7_as_v8.ireducescatter = nullptr;
  ncclCollNet_v7_as_v8.iflush = ncclCollNet_v7->iflush;
  ncclCollNet_v7_as_v8.test = ncclCollNet_v7->test;
  ncclCollNet_v7_as_v8.closeColl = ncclCollNet_v7->closeColl;
  ncclCollNet_v7_as_v8.closeListen = ncclCollNet_v7->closeListen;
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
  const char* envPluginName = ncclGetEnv("NCCL_NET_PLUGIN");
  if (envPluginName && strlen(envPluginName)) {
    snprintf(ncclNetPluginName, 128, "libnccl-net-%s.so", envPluginName);
    INFO(NCCL_INIT, "Plugin name set by env to %s", ncclNetPluginName);
  } else {
    sprintf(ncclNetPluginName, "libnccl-net.so");
  }
  void* netPluginLib = dlopen(ncclNetPluginName, RTLD_NOW | RTLD_LOCAL);
  if (netPluginLib == nullptr) {
    // dlopen does not guarantee to set errno, but dlerror only gives us a
    // string, so checking errno doesn't hurt to try to provide a better
    // error message
    if (errno == ENOENT) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : dlerror=%s No plugin found (%s), using internal implementation", dlerror(), ncclNetPluginName);
      // exit(-1);
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin : Plugin load returned %d : %s.", errno, dlerror());
    }
    return ncclSuccess;
  }

  ncclNets[0] = (ncclNet_v8_t*)dlsym(netPluginLib, "ncclNetPlugin_v8");
  if (ncclNets[0] == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v8 symbol.");
    // Try v7 plugin
    ncclNet_v7 = (ncclNet_v7_t*)dlsym(netPluginLib, "ncclNetPlugin_v7");
    if (ncclNet_v7 == nullptr) {
      // Try v6 plugin
      ncclNet_v6 = (ncclNet_v6_t*)dlsym(netPluginLib, "ncclNetPlugin_v6");
      if (ncclNet_v6 == nullptr) {
        // Try v5 plugin
        ncclNet_v5 = (ncclNet_v5_t*)dlsym(netPluginLib, "ncclNetPlugin_v5");
        if (ncclNet_v5 == nullptr) {
          INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin symbol (>= v5). ncclNetPlugin symbols v4 and lower are not supported.");
          if (netPluginLib != nullptr) dlclose(netPluginLib);
          return ncclSuccess;
        } else {
          ncclNets[0] = &ncclNet_v5_as_v8;
          ncclNet_v5_as_v8.init = ncclNet_v5_as_v8_init;
          // Set the name right away to allow for NCCL_NET=... to work
          ncclNet_v5_as_v8.name = ncclNet_v5->name;
          INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v5)", ncclNets[0]->name);
        }
      } else {
        ncclNets[0] = &ncclNet_v6_as_v8;
        ncclNet_v6_as_v8.init = ncclNet_v6_as_v8_init;
        // Set the name right away to allow for NCCL_NET=... to work
        ncclNet_v6_as_v8.name = ncclNet_v6->name;
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v6)", ncclNets[0]->name);
      }
    } else {
      ncclNets[0] = &ncclNet_v7_as_v8;
      ncclNet_v7_as_v8.init = ncclNet_v7_as_v8_init;
      // Set the name right away to allow for NCCL_NET=... to work
      ncclNet_v7_as_v8.name = ncclNet_v7->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v7)", ncclNets[0]->name);
    }
  }

  // Check for CollNet
  ncclCollNets[0] = (ncclCollNet_v8_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v8");
  if (ncclCollNets[0] == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v8 symbol.");
    ncclCollNet_v7 = (ncclCollNet_v7_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v7");
    if (ncclCollNet_v7 == nullptr) {
      ncclCollNet_v6 = (ncclCollNet_v6_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v6");
      if (ncclCollNet_v6 == nullptr) {
        ncclCollNet_v5 = (ncclCollNet_v5_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v5");
        if (ncclCollNet_v5 == nullptr) {
          INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin symbol (>= v5). ncclCollNetPlugin symbols v4 and lower are not supported.");
        } else {
          ncclCollNets[0] = &ncclCollNet_v5_as_v8;
          ncclCollNet_v5_as_v8.init = ncclCollNet_v5_as_v8_init;
          ncclCollNet_v5_as_v8.name = ncclCollNet_v5->name;
          INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v5)", ncclCollNets[0]->name);
        }
      } else {
        ncclCollNets[0] = &ncclCollNet_v6_as_v8;
        ncclCollNet_v6_as_v8.init = ncclCollNet_v6_as_v8_init;
        ncclCollNet_v6_as_v8.name = ncclCollNet_v6->name;
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v6)", ncclCollNets[0]->name);
      }
    } else {
      ncclCollNets[0] = &ncclCollNet_v7_as_v8;
      ncclCollNet_v7_as_v8.init = ncclCollNet_v7_as_v8_init;
      ncclCollNet_v7_as_v8.name = ncclCollNet_v7->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded coll plugin %s (v7)", ncclCollNets[0]->name);
    }
  }
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
      WARN("Unknown device code index");
      return ncclInternalError;
  }

  INFO(NCCL_INIT, "Using non-device net plugin version %d",
    props.netDeviceVersion);
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
  // Initialize main communication network
  const char* netName;
  bool ok = false;

  netName = comm->config.netName;
  for (int i=0; i<3; i++) {
    if (ncclNets[i] == nullptr) continue;
    enum ncclNetState state;
    NCCLCHECK(netGetState(i, &state));
    if (state != ncclNetStateEnabled) continue;
    if (netName && strcasecmp(netName, ncclNets[i]->name) != 0) continue;
    if (ncclSuccess != ncclNetCheckDeviceVersion(comm, ncclNets[i], 0)) {
      // Mismatched device plugin version
      continue;
    }

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
      if (__atomic_load_n(comm->abortFlag, __ATOMIC_RELAXED)) {
        goto cleanup2;
      }

      if (sComm == NULL)
        NCCLCHECKGOTO(comm->ncclNet->connect(dev, &handle, &sComm, NULL), ret, cleanup2);

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

int ncclNetVersion(struct ncclComm* comm) {
  return
    (comm->ncclNet == &ncclNet_v5_as_v8) ? 5 :
    (comm->ncclNet == &ncclNet_v6_as_v8) ? 6 :
    (comm->ncclNet == &ncclNet_v7_as_v8) ? 7 :
    8;
}
