// InfiniBand transport stub for non-Linux platforms
// This provides empty implementations for platforms that don't support InfiniBand

#include "net.h"
#include "core.h"

// Stub ncclNetIb implementation - all functions return ncclInternalError
static ncclResult_t stubInit(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  return ncclInternalError;
}

static ncclResult_t stubDevices(int* ndev) {
  *ndev = 0;
  return ncclSuccess;
}

static ncclResult_t stubGetProperties(int dev, ncclNetProperties_t* props) {
  return ncclInternalError;
}

static ncclResult_t stubListen(void* ctx, int dev, void* handle, void** listenComm) {
  return ncclInternalError;
}

static ncclResult_t stubConnect(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** sendDevComm) {
  return ncclInternalError;
}

static ncclResult_t stubAccept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** recvDevComm) {
  return ncclInternalError;
}

static ncclResult_t stubRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  return ncclInternalError;
}

static ncclResult_t stubRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
  return ncclInternalError;
}

static ncclResult_t stubDeregMr(void* comm, void* mhandle) {
  return ncclInternalError;
}

static ncclResult_t stubIsend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request) {
  return ncclInternalError;
}

static ncclResult_t stubIrecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request) {
  return ncclInternalError;
}

static ncclResult_t stubIflush(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request) {
  return ncclInternalError;
}

static ncclResult_t stubTest(void* request, int* done, int* sizes) {
  return ncclInternalError;
}

static ncclResult_t stubCloseSend(void* sendComm) {
  return ncclSuccess;
}

static ncclResult_t stubCloseRecv(void* recvComm) {
  return ncclSuccess;
}

static ncclResult_t stubCloseListen(void* listenComm) {
  return ncclSuccess;
}

static ncclResult_t stubGetDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) {
  return ncclInternalError;
}

static ncclResult_t stubIrecvConsumed(void* recvComm, int n, void* request) {
  return ncclInternalError;
}

static ncclResult_t stubMakeVDevice(int* d, ncclNetVDeviceProps_t* props) {
  return ncclInternalError;
}

static ncclResult_t stubFinalize(void* ctx) {
  return ncclSuccess;
}

static ncclResult_t stubSetNetAttr(void* ctx, ncclNetAttr_t* netAttr) {
  return ncclInternalError;
}

// Stub ncclNetIb definition
ncclNet_t ncclNetIb = {
  "IB (stub)",
  stubInit,
  stubDevices,
  stubGetProperties,
  stubListen,
  stubConnect,
  stubAccept,
  stubRegMr,
  stubRegMrDmaBuf,
  stubDeregMr,
  stubIsend,
  stubIrecv,
  stubIflush,
  stubTest,
  stubCloseSend,
  stubCloseRecv,
  stubCloseListen,
  stubGetDeviceMr,
  stubIrecvConsumed,
  stubMakeVDevice,
  stubFinalize,
  stubSetNetAttr
};
