/*************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net.h"

#define PLUGIN_NAME "Plugin"

#define __hidden __attribute__ ((visibility("hidden")))
#define NCCL_PLUGIN_MAX_RECVS 1

int max_requests = NCCL_NET_MAX_REQUESTS;

__hidden ncclResult_t pluginInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) { return ncclSuccess; }
__hidden ncclResult_t pluginDevices(int* ndev) { *ndev = 0; return ncclSuccess; }
__hidden ncclResult_t pluginPciPath(int dev, char** path) { return ncclInternalError; }
__hidden ncclResult_t pluginPtrSupport(int dev, int* supportedTypes) { return ncclInternalError; }
__hidden ncclResult_t pluginGetProperties(int dev, ncclNetProperties_t* props) {
  // Below are default values, if unsure don't change.

  props->name = "Example";
  // Fill for proper topology detection, e.g. /sys/devices/pci0000:00/0000:00:10.0/0000:0b:00.0
  props->pciPath = NULL;
  // Only used to detect NICs with multiple PCI attachments.
  props->guid = 0;
  // Add NCCL_PTR_CUDA if GPU Direct RDMA is supported and regMr can take CUDA pointers.
  props->ptrSupport = NCCL_PTR_HOST;
  // If you regMr has a fast registration cache, set to 1. If set to 0, user buffer registration may be disabled.
  props->regIsGlobal = 0;
  // Force flush after receive. Needed if the control path and data path use a different path to the GPU
  props->forceFlush = 0;
  // Speed in *Mbps*. 100000 means 100G
  props->speed = 100000;
  // Port number, used in conjunction with guid
  props->port = 0;
  // Custom latency (used to help tuning if latency is high. If set to 0, use default NCCL values.
  props->latency = 0;
  // Maximum number of comm objects we can create.
  props->maxComms = 1024*1024;
  // Maximum number of receive operations taken by irecv().
  props->maxRecvs = NCCL_PLUGIN_MAX_RECVS;
  // Coupling with NCCL network device-side code.
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  // Used to tell NCCL core whether this is a virtual device fusing multiple physical devices.
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  // maximum transfer sizes the plugin can handle
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  return ncclSuccess;
}

__hidden ncclResult_t pluginListen(void* ctx, int dev, void* handle, void** listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginConnect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMr(void* collComm, void* data, size_t size, int type, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginRegMrDmaBuf(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginDeregMr(void* collComm, void* mhandle) { return ncclInternalError;}
__hidden ncclResult_t pluginIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) { return ncclInternalError; }
__hidden ncclResult_t pluginTest(void* request, int* done, int* size) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseSend(void* sendComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseRecv(void* recvComm) { return ncclInternalError; }
__hidden ncclResult_t pluginCloseListen(void* listenComm) { return ncclInternalError; }
__hidden ncclResult_t pluginIrecvConsumed(void* recvComm, int n, void* request) { return ncclInternalError; }
__hidden ncclResult_t pluginGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) { return ncclInternalError; }
__hidden ncclResult_t pluginMakeVDevice(int* d, ncclNetVDeviceProps_t* props) { return ncclInternalError; }
__hidden ncclResult_t pluginFinalize(void* ctx) { return ncclSuccess; }

const ncclNet_v11_t ncclNetPlugin_v11 = {
  .name = PLUGIN_NAME,
  .init = pluginInit,
  .devices = pluginDevices,
  .getProperties = pluginGetProperties,
  .listen = pluginListen,
  .connect = pluginConnect,
  .accept = pluginAccept,
  .regMr = pluginRegMr,
  .regMrDmaBuf = pluginRegMrDmaBuf,
  .deregMr = pluginDeregMr,
  .isend = pluginIsend,
  .irecv = pluginIrecv,
  .iflush = pluginIflush,
  .test = pluginTest,
  .closeSend = pluginCloseSend,
  .closeRecv = pluginCloseRecv,
  .closeListen = pluginCloseListen,
  .getDeviceMr = pluginGetDeviceMr,
  .irecvConsumed = pluginIrecvConsumed,
  .makeVDevice   = pluginMakeVDevice,
  .finalize = pluginFinalize
};

#include "tuner.h"

__hidden ncclResult_t tunerPluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logFunction, ncclNvlDomainInfo_v5_t* nvlDomainInfo, ncclTunerConstants_v5_t* constants) { return ncclSuccess; }

__hidden ncclResult_t tunerPluginGetCollInfo(void* context, ncclFunc_t collType, size_t nBytes,
                              int numPipeOps, float** collCostTable, int numAlgo, int numProto,
                              int regBuff, int* nChannels) {
  // Update NCCL core generated cost table. Updated table will be evaluated by NCCL to pick the best algo/proto combo
  if (collCostTable[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] != NCCL_ALGO_PROTO_IGNORE) {
    collCostTable[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 0.0;
  }
  *nChannels = 1;
  return ncclSuccess;
}

__hidden ncclResult_t tunerPluginFinalize(void* context) { return ncclSuccess; }

const ncclTuner_v5_t ncclTunerPlugin_v5 = {
  .name = PLUGIN_NAME,
  .init = tunerPluginInit,
  .getCollInfo = tunerPluginGetCollInfo,
  .finalize = tunerPluginFinalize
};
