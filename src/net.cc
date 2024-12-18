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

static ncclNet_v9_t ncclNet_v5_as_v9;
static ncclNet_v9_t ncclNet_v6_as_v9;
static ncclNet_v9_t ncclNet_v7_as_v9;
static ncclNet_v9_t ncclNet_v8_as_v9;
static ncclNet_v5_t *ncclNet_v5;
static ncclNet_v6_t *ncclNet_v6;
static ncclNet_v7_t *ncclNet_v7;
static ncclNet_v8_t *ncclNet_v8;
static ncclCollNet_v9_t ncclCollNet_v5_as_v9;
static ncclCollNet_v9_t ncclCollNet_v6_as_v9;
static ncclCollNet_v9_t ncclCollNet_v7_as_v9;
static ncclCollNet_v9_t ncclCollNet_v8_as_v9;
static ncclCollNet_v5_t *ncclCollNet_v5;
static ncclCollNet_v6_t *ncclCollNet_v6;
static ncclCollNet_v7_t *ncclCollNet_v7;
static ncclCollNet_v8_t *ncclCollNet_v8;

#define MAX_NET_SIZE (1024*1024*1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.
#define MAX_COLLNET_SIZE (512*1024*1024L) //Set for initial collent plugins when size was not dynamically queried

static ncclResult_t ncclNet_v8_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v8_t p8;
  ncclResult_t ans = ncclNet_v8->getProperties(dev, &p8);
  if (ans != ncclSuccess) return ans;
  props->name = p8.name;
  props->pciPath = p8.pciPath;
  props->guid = p8.guid;
  props->ptrSupport = p8.ptrSupport;
  props->regIsGlobal = p8.regIsGlobal;
  props->forceFlush = 0;
  props->speed = p8.speed;
  props->port = p8.port;
  props->maxComms = p8.maxComms;
  props->maxRecvs = p8.maxRecvs;
  props->latency = p8.latency;
  props->netDeviceType = p8.netDeviceType;
  props->netDeviceVersion = p8.netDeviceVersion;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v8_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
   int sizeInt;
   if (size > MAX_NET_SIZE) return ncclInternalError;
   sizeInt = (int)size;
   ncclResult_t ans = ncclNet_v8->isend(sendComm, data, sizeInt, tag, mhandle, request);
   return ans;
}

static ncclResult_t ncclNet_v8_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
   int sizesInt[NCCL_PROXY_MAX_SUBS];
   //reset to NULL if optional receive completion is set
   if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
   for (int i=0; i<n; i++) {
     if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
     sizesInt[i] = (int) sizes[i];
   }
   ncclResult_t ans = ncclNet_v8->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
   return ans;
}

static ncclResult_t ncclNet_v8_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v8->init(logfn));
  ncclNet_v8_as_v9.name = ncclNet_v8->name;
  ncclNet_v8_as_v9.devices = ncclNet_v8->devices;
  ncclNet_v8_as_v9.getProperties = ncclNet_v8_as_v9_getProperties;
  ncclNet_v8_as_v9.listen = ncclNet_v8->listen;
  ncclNet_v8_as_v9.connect = ncclNet_v8->connect;
  ncclNet_v8_as_v9.accept =  ncclNet_v8->accept;
  ncclNet_v8_as_v9.regMr = ncclNet_v8->regMr;
  ncclNet_v8_as_v9.regMrDmaBuf = ncclNet_v8->regMrDmaBuf;
  ncclNet_v8_as_v9.deregMr = ncclNet_v8->deregMr;
  ncclNet_v8_as_v9.isend = ncclNet_v8_as_v9_isend;
  ncclNet_v8_as_v9.irecv = ncclNet_v8_as_v9_irecv;
  ncclNet_v8_as_v9.iflush = ncclNet_v8->iflush;
  ncclNet_v8_as_v9.test = ncclNet_v8->test;
  ncclNet_v8_as_v9.closeSend = ncclNet_v8->closeSend;
  ncclNet_v8_as_v9.closeRecv = ncclNet_v8->closeRecv;
  ncclNet_v8_as_v9.closeListen = ncclNet_v8->closeListen;
  ncclNet_v8_as_v9.getDeviceMr = ncclNet_v8->getDeviceMr;
  ncclNet_v8_as_v9.irecvConsumed = ncclNet_v8->irecvConsumed;
  ncclNet_v8_as_v9.makeVDevice   = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v7_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v7_t p7;
  ncclResult_t ans = ncclNet_v7->getProperties(dev, &p7);
  if (ans != ncclSuccess) return ans;
  props->name = p7.name;
  props->pciPath = p7.pciPath;
  props->guid = p7.guid;
  props->ptrSupport = p7.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p7.speed;
  props->port = p7.port;
  props->maxComms = p7.maxComms;
  props->maxRecvs = p7.maxRecvs;
  props->latency = p7.latency;
  props->netDeviceType = p7.netDeviceType;
  props->netDeviceVersion = p7.netDeviceVersion;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v7_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v7->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v7_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
   int sizeInt;
   if (size > MAX_NET_SIZE) return ncclInternalError;
   sizeInt = (int)size;
   ncclResult_t ans = ncclNet_v7->isend(sendComm, data, sizeInt, tag, mhandle, request);
   return ans;
}

static ncclResult_t ncclNet_v7_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
   int sizesInt[NCCL_PROXY_MAX_SUBS];
   //reset to NULL if optional receive completion is set
   if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
   for (int i=0; i<n; i++) {
     if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
     sizesInt[i] = (int) sizes[i];
   }
   ncclResult_t ans = ncclNet_v7->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
   return ans;
}

static ncclResult_t ncclNet_v7_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v7->init(logfn));
  ncclNet_v7_as_v9.name = ncclNet_v7->name;
  ncclNet_v7_as_v9.devices = ncclNet_v7->devices;
  ncclNet_v7_as_v9.getProperties = ncclNet_v7_as_v9_getProperties; // ncclNet_v5->getProperties;
  ncclNet_v7_as_v9.listen = ncclNet_v7->listen;
  ncclNet_v7_as_v9.connect = ncclNet_v7->connect;
  ncclNet_v7_as_v9.accept =  ncclNet_v7->accept;
  ncclNet_v7_as_v9.regMr = ncclNet_v7_as_v9_regMr;
  ncclNet_v7_as_v9.regMrDmaBuf = ncclNet_v7->regMrDmaBuf;
  ncclNet_v7_as_v9.deregMr = ncclNet_v7->deregMr;
  ncclNet_v7_as_v9.isend = ncclNet_v7_as_v9_isend;
  ncclNet_v7_as_v9.irecv = ncclNet_v7_as_v9_irecv;
  ncclNet_v7_as_v9.iflush = ncclNet_v7->iflush;
  ncclNet_v7_as_v9.test = ncclNet_v7->test;
  ncclNet_v7_as_v9.closeSend = ncclNet_v7->closeSend;
  ncclNet_v7_as_v9.closeRecv = ncclNet_v7->closeRecv;
  ncclNet_v7_as_v9.closeListen = ncclNet_v7->closeListen;
  ncclNet_v7_as_v9.getDeviceMr = ncclNet_v7->getDeviceMr;
  ncclNet_v7_as_v9.irecvConsumed = ncclNet_v7->irecvConsumed;
  ncclNet_v7_as_v9.makeVDevice  = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v6_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v6_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v6_as_v9_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v6->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_v6_as_v9_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v6->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_v6_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
   int sizeInt;
   if (size > MAX_NET_SIZE) return ncclInternalError;
   sizeInt = (int)size;
   ncclResult_t ans = ncclNet_v6->isend(sendComm, data, sizeInt, tag, mhandle, request);
   return ans;
}

static ncclResult_t ncclNet_v6_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
   int sizesInt[NCCL_PROXY_MAX_SUBS];
   //reset to NULL if optional receive completion is set
   if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
   for (int i=0; i<n; i++) {
     if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
     sizesInt[i] = (int) sizes[i];
   }
   ncclResult_t ans = ncclNet_v6->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
   return ans;
}

static ncclResult_t ncclNet_v6_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v6->init(logfn));
  ncclNet_v6_as_v9.name = ncclNet_v6->name;
  ncclNet_v6_as_v9.devices = ncclNet_v6->devices;
  ncclNet_v6_as_v9.getProperties = ncclNet_v6_as_v9_getProperties;
  ncclNet_v6_as_v9.listen = ncclNet_v6->listen;
  ncclNet_v6_as_v9.connect = ncclNet_v6_as_v9_connect;
  ncclNet_v6_as_v9.accept =  ncclNet_v6_as_v9_accept;
  ncclNet_v6_as_v9.regMr = ncclNet_v6_as_v9_regMr;
  ncclNet_v6_as_v9.regMrDmaBuf = ncclNet_v6->regMrDmaBuf;
  ncclNet_v6_as_v9.deregMr = ncclNet_v6->deregMr;
  ncclNet_v6_as_v9.isend = ncclNet_v6_as_v9_isend;
  ncclNet_v6_as_v9.irecv = ncclNet_v6_as_v9_irecv;
  ncclNet_v6_as_v9.iflush = ncclNet_v6->iflush;
  ncclNet_v6_as_v9.test = ncclNet_v6->test;
  ncclNet_v6_as_v9.closeSend = ncclNet_v6->closeSend;
  ncclNet_v6_as_v9.closeRecv = ncclNet_v6->closeRecv;
  ncclNet_v6_as_v9.closeListen = ncclNet_v6->closeListen;
  ncclNet_v6_as_v9.getDeviceMr = NULL;
  ncclNet_v6_as_v9.irecvConsumed = NULL;
  ncclNet_v6_as_v9.makeVDevice  = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v5_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclNet_v5->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclNet_v5_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclNet_v5_as_v9_connect(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_t** /*sendDevComm*/) {
  return ncclNet_v5->connect(dev, handle, sendComm);
}

static ncclResult_t ncclNet_v5_as_v9_accept(void* listenComm, void** recvComm, ncclNetDeviceHandle_t** /*recvDevComm*/) {
  return ncclNet_v5->accept(listenComm, recvComm);
}

static ncclResult_t ncclNet_v5_as_v9_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle, void** request) {
   int sizeInt;
   if (size > MAX_NET_SIZE) return ncclInternalError;
   sizeInt = (int)size;
   ncclResult_t ans = ncclNet_v5->isend(sendComm, data, sizeInt, tag, mhandle, request);
   return ans;
}

static ncclResult_t ncclNet_v5_as_v9_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** request) {
   int sizesInt[NCCL_PROXY_MAX_SUBS];
   //reset to NULL if optional receive completion is set
   if (*request == (void *)NCCL_NET_OPTIONAL_RECV_COMPLETION) *request = NULL;
   for (int i=0; i<n; i++) {
     if (sizes[i] > MAX_NET_SIZE) return ncclInternalError;
     sizesInt[i] = (int) sizes[i];
   }
   ncclResult_t ans = ncclNet_v5->irecv(recvComm, n, data, sizesInt, tags, mhandles, request);
   return ans;
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclNet_v5_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclNet_v5->init(logfn));
  ncclNet_v5_as_v9.name = ncclNet_v5->name;
  ncclNet_v5_as_v9.devices = ncclNet_v5->devices;
  ncclNet_v5_as_v9.getProperties = ncclNet_v5_as_v9_getProperties;
  ncclNet_v5_as_v9.listen = ncclNet_v5->listen;
  ncclNet_v5_as_v9.connect = ncclNet_v5_as_v9_connect;
  ncclNet_v5_as_v9.accept =  ncclNet_v5_as_v9_accept;
  ncclNet_v5_as_v9.regMr = ncclNet_v5_as_v9_regMr;
  ncclNet_v5_as_v9.regMrDmaBuf = NULL;
  ncclNet_v5_as_v9.deregMr = ncclNet_v5->deregMr;
  ncclNet_v5_as_v9.isend = ncclNet_v5_as_v9_isend;
  ncclNet_v5_as_v9.irecv = ncclNet_v5_as_v9_irecv;
  ncclNet_v5_as_v9.iflush = ncclNet_v5->iflush;
  ncclNet_v5_as_v9.test = ncclNet_v5->test;
  ncclNet_v5_as_v9.closeSend = ncclNet_v5->closeSend;
  ncclNet_v5_as_v9.closeRecv = ncclNet_v5->closeRecv;
  ncclNet_v5_as_v9.closeListen = ncclNet_v5->closeListen;
  ncclNet_v5_as_v9.getDeviceMr = NULL;
  ncclNet_v5_as_v9.irecvConsumed = NULL;
  ncclNet_v5_as_v9.makeVDevice = NULL;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v5_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclCollNet_v5->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v5_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v5->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_v5_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
   int countInt;
   if (count > MAX_NET_SIZE) return ncclInternalError;
   countInt = (int)count;
   ncclResult_t ans = ncclCollNet_v5->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                  sendMhandle, recvMhandle, request);
   return ans;
}

// We use a wrapper around the v5 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v5_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v5->init(logfn));
  ncclCollNet_v5_as_v9.name = ncclCollNet_v5->name;
  ncclCollNet_v5_as_v9.devices = ncclCollNet_v5->devices;
  ncclCollNet_v5_as_v9.getProperties = ncclCollNet_v5_as_v9_getProperties;
  ncclCollNet_v5_as_v9.listen = ncclCollNet_v5->listen;
  ncclCollNet_v5_as_v9.connect = ncclCollNet_v5->connect;
  ncclCollNet_v5_as_v9.reduceSupport = ncclCollNet_v5->reduceSupport;
  ncclCollNet_v5_as_v9.regMr = ncclCollNet_v5_as_v9_regMr;
  ncclCollNet_v5_as_v9.regMrDmaBuf = NULL;
  ncclCollNet_v5_as_v9.deregMr = ncclCollNet_v5->deregMr;
  ncclCollNet_v5_as_v9.iallreduce = ncclCollNet_v5_as_v9_iallreduce;
  ncclCollNet_v5_as_v9.iallgather = nullptr;
  ncclCollNet_v5_as_v9.ireducescatter = nullptr;
  ncclCollNet_v5_as_v9.iflush = ncclCollNet_v5->iflush;
  ncclCollNet_v5_as_v9.test = ncclCollNet_v5->test;
  ncclCollNet_v5_as_v9.closeColl = ncclCollNet_v5->closeColl;
  ncclCollNet_v5_as_v9.closeListen = ncclCollNet_v5->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v6_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v6_t p6;
  ncclResult_t ans = ncclCollNet_v6->getProperties(dev, &p6);
  if (ans != ncclSuccess) return ans;
  props->name = p6.name;
  props->pciPath = p6.pciPath;
  props->guid = p6.guid;
  props->ptrSupport = p6.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p6.speed;
  props->port = p6.port;
  props->maxComms = p6.maxComms;
  props->maxRecvs = p6.maxRecvs;
  props->latency = p6.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v6_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v6->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_v6_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
   int countInt;
   if (count > MAX_NET_SIZE) return ncclInternalError;
   countInt = (int)count;
   ncclResult_t ans = ncclCollNet_v6->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                  sendMhandle, recvMhandle, request);
   return ans;
}

// We use a wrapper around the v6 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v6_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v6->init(logfn));
  ncclCollNet_v6_as_v9.name = ncclCollNet_v6->name;
  ncclCollNet_v6_as_v9.devices = ncclCollNet_v6->devices;
  ncclCollNet_v6_as_v9.getProperties = ncclCollNet_v6_as_v9_getProperties;
  ncclCollNet_v6_as_v9.listen = ncclCollNet_v6->listen;
  ncclCollNet_v6_as_v9.connect = ncclCollNet_v6->connect;
  ncclCollNet_v6_as_v9.reduceSupport = ncclCollNet_v6->reduceSupport;
  ncclCollNet_v6_as_v9.regMr = ncclCollNet_v6_as_v9_regMr;
  ncclCollNet_v6_as_v9.regMrDmaBuf = ncclCollNet_v6->regMrDmaBuf;
  ncclCollNet_v6_as_v9.deregMr = ncclCollNet_v6->deregMr;
  ncclCollNet_v6_as_v9.iallreduce = ncclCollNet_v6_as_v9_iallreduce;
  ncclCollNet_v6_as_v9.iallgather = nullptr;
  ncclCollNet_v6_as_v9.ireducescatter = nullptr;
  ncclCollNet_v6_as_v9.iflush = ncclCollNet_v6->iflush;
  ncclCollNet_v6_as_v9.test = ncclCollNet_v6->test;
  ncclCollNet_v6_as_v9.closeColl = ncclCollNet_v6->closeColl;
  ncclCollNet_v6_as_v9.closeListen = ncclCollNet_v6->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v7_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v7_t p7;
  ncclResult_t ans = ncclCollNet_v7->getProperties(dev, &p7);
  if (ans != ncclSuccess) return ans;
  props->name = p7.name;
  props->pciPath = p7.pciPath;
  props->guid = p7.guid;
  props->ptrSupport = p7.ptrSupport;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = p7.speed;
  props->port = p7.port;
  props->maxComms = p7.maxComms;
  props->maxRecvs = p7.maxRecvs;
  props->latency = p7.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v7_as_v9_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  if (size >= 1UL<<31) return ncclInternalError;
  return ncclCollNet_v7->regMr(comm, data, (int) size, type, mhandle);
}

static ncclResult_t ncclCollNet_v7_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
   int countInt;
   if (count > MAX_NET_SIZE) return ncclInternalError;
   countInt = (int)count;
   ncclResult_t ans = ncclCollNet_v7->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                  sendMhandle, recvMhandle, request);
   return ans;
}

// We use a wrapper around the v7 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v7_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v7->init(logfn));
  ncclCollNet_v7_as_v9.name = ncclCollNet_v7->name;
  ncclCollNet_v7_as_v9.devices = ncclCollNet_v7->devices;
  ncclCollNet_v7_as_v9.getProperties = ncclCollNet_v7_as_v9_getProperties;
  ncclCollNet_v7_as_v9.listen = ncclCollNet_v7->listen;
  ncclCollNet_v7_as_v9.connect = ncclCollNet_v7->connect;
  ncclCollNet_v7_as_v9.reduceSupport = ncclCollNet_v7->reduceSupport;
  ncclCollNet_v7_as_v9.regMr = ncclCollNet_v7_as_v9_regMr;
  ncclCollNet_v7_as_v9.regMrDmaBuf = ncclCollNet_v7->regMrDmaBuf;
  ncclCollNet_v7_as_v9.deregMr = ncclCollNet_v7->deregMr;
  ncclCollNet_v7_as_v9.iallreduce = ncclCollNet_v7_as_v9_iallreduce;
  ncclCollNet_v7_as_v9.iallgather = nullptr;
  ncclCollNet_v7_as_v9.ireducescatter = nullptr;
  ncclCollNet_v7_as_v9.iflush = ncclCollNet_v7->iflush;
  ncclCollNet_v7_as_v9.test = ncclCollNet_v7->test;
  ncclCollNet_v7_as_v9.closeColl = ncclCollNet_v7->closeColl;
  ncclCollNet_v7_as_v9.closeListen = ncclCollNet_v7->closeListen;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v8_as_v9_getProperties(int dev, ncclNetProperties_v9_t* props) {
  ncclNetProperties_v8_t p8;
  ncclResult_t ans = ncclCollNet_v8->getProperties(dev, &p8);
  if (ans != ncclSuccess) return ans;
  props->name = p8.name;
  props->pciPath = p8.pciPath;
  props->guid = p8.guid;
  props->ptrSupport = p8.ptrSupport;
  props->regIsGlobal = p8.regIsGlobal;
  props->forceFlush = 0;
  props->speed = p8.speed;
  props->port = p8.port;
  props->maxComms = p8.maxComms;
  props->maxRecvs = p8.maxRecvs;
  props->latency = p8.latency;
  props->netDeviceType    = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = MAX_NET_SIZE;
  props->maxCollBytes = MAX_COLLNET_SIZE;
  return ncclSuccess;
}

static ncclResult_t ncclCollNet_v8_as_v9_iallreduce(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request) {
   int countInt;
   if (count > MAX_NET_SIZE) return ncclInternalError;
   countInt = (int)count;
   ncclResult_t ans = ncclCollNet_v8->iallreduce(collComm, sendData, recvData, countInt, dataType, redOp,
                  sendMhandle, recvMhandle, request);
   return ans;
}

static ncclResult_t ncclCollNet_v8_as_v9_iallgather (void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v9_t* recvParts,
                           size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                           void* sendMhandle, void** request) {
   ncclNetSGE_v8_t recvPartsInt;
   if (nRecvParts > 1) return ncclInternalError;
   if (recvParts->size > MAX_COLLNET_SIZE) return ncclInternalError;
   recvPartsInt.mhandle = recvParts->mhandle;
   recvPartsInt.address = recvParts->address;
   recvPartsInt.size = (int)recvParts->size;
   ncclResult_t ans = ncclCollNet_v8->iallgather(collComm, sendData, nRecvParts, &recvPartsInt,
                   bytesPerRank, windowOffset, windowBytes,
                   sendMhandle, request);
   return ans;
}

static ncclResult_t ncclCollNet_v8_as_v9_ireducescatter(void* collComm, int nSendParts, ncclNetSGE_v9_t* sendParts, void* recvData,
                               size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                               ncclDataType_t dataType, ncclRedOp_t redOp,
                               void* recvMhandle, void** request) {
   ncclNetSGE_v8_t sendPartsInt;
   if (nSendParts > 1) return ncclInternalError;
   if (sendParts->size > MAX_COLLNET_SIZE) return ncclInternalError;
   sendPartsInt.mhandle = sendParts->mhandle;
   sendPartsInt.address = sendParts->address;
   sendPartsInt.size = (int)sendParts->size;
   ncclResult_t ans = ncclCollNet_v8->ireducescatter(collComm, nSendParts, &sendPartsInt,
                   recvData, bytesPerRank, windowOffset, windowBytes,
                   dataType, redOp,
                  recvMhandle, request);
   return ans;
}

// We use a wrapper around the v8 init to copy over the struct contents
// post-init since they may not be initialized before hand.
static ncclResult_t ncclCollNet_v8_as_v9_init(ncclDebugLogger_t logfn) {
  NCCLCHECK(ncclCollNet_v8->init(logfn));
  ncclCollNet_v8_as_v9.name = ncclCollNet_v8->name;
  ncclCollNet_v8_as_v9.devices = ncclCollNet_v8->devices;
  ncclCollNet_v8_as_v9.getProperties = ncclCollNet_v8_as_v9_getProperties;
  ncclCollNet_v8_as_v9.listen = ncclCollNet_v8->listen;
  ncclCollNet_v8_as_v9.connect = ncclCollNet_v8->connect;
  ncclCollNet_v8_as_v9.reduceSupport = ncclCollNet_v8->reduceSupport;
  ncclCollNet_v8_as_v9.regMr = ncclCollNet_v8->regMr;
  ncclCollNet_v8_as_v9.regMrDmaBuf = ncclCollNet_v8->regMrDmaBuf;
  ncclCollNet_v8_as_v9.deregMr = ncclCollNet_v8->deregMr;
  ncclCollNet_v8_as_v9.iallreduce = ncclCollNet_v8_as_v9_iallreduce;
  ncclCollNet_v8_as_v9.iallgather = ncclCollNet_v8_as_v9_iallgather;
  ncclCollNet_v8_as_v9.ireducescatter = ncclCollNet_v8_as_v9_ireducescatter;
  ncclCollNet_v8_as_v9.iflush = ncclCollNet_v8->iflush;
  ncclCollNet_v8_as_v9.test = ncclCollNet_v8->test;
  ncclCollNet_v8_as_v9.closeColl = ncclCollNet_v8->closeColl;
  ncclCollNet_v8_as_v9.closeListen = ncclCollNet_v8->closeListen;
  return ncclSuccess;
}

static pthread_mutex_t netLock = PTHREAD_MUTEX_INITIALIZER;
ncclNet_t* ncclNets[NCCL_NET_MAX_PLUGINS] = { nullptr, &ncclNetIb, &ncclNetSocket };
ncclCollNet_t* ncclCollNets[NCCL_NET_MAX_PLUGINS] = { nullptr, nullptr, nullptr };
enum ncclNetState {
  ncclNetStateInit = 0,
  ncclNetStateEnabled = 1,
  ncclNetStateDisabled = 2
};
enum ncclNetState ncclNetStates[NCCL_NET_MAX_PLUGINS] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };
enum ncclNetState ncclCollNetStates[NCCL_NET_MAX_PLUGINS] = { ncclNetStateInit, ncclNetStateInit, ncclNetStateInit };

#define MAX_STR_LEN 255

static void* tryOpenLib(char* name, int* err, char* errStr) {
  *err = 0;
  if (nullptr == name || strlen(name) == 0) {
    return nullptr;
  }

  if (strncasecmp(name, "STATIC_PLUGIN", strlen(name)) == 0) {
    name = nullptr;
  }

  void *handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
  if (nullptr == handle) {
    strncpy(errStr, dlerror(), MAX_STR_LEN);
    errStr[MAX_STR_LEN] = '\0';
    // "handle" and "name" won't be NULL at the same time.
    // coverity[var_deref_model]
    if (strstr(errStr, name) && strstr(errStr, "No such file or directory")) {
      *err = ENOENT;
    }
  }
  return handle;
}

static char* tryOpenLibCheck(int openErr, char* openErrStr, char* nameList, int *nameListLen, char* name) {
  if (openErr == ENOENT) {
    snprintf(nameList, *nameListLen, " %s", name);
    nameList += strlen(name) + 1;
    *nameListLen -= strlen(name) + 1;
    return nameList;
  }
  INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: %s", openErrStr);
  return nameList;
}

static void* openNetPluginLib(char* couldNotFindNames, int len) {
  int openErr;
  void *pluginLib;
  char netPluginLibName[PATH_MAX];
  char openErrStr[MAX_STR_LEN + 1] = { 0 };
  const char *envNetPluginName = getenv("NCCL_NET_PLUGIN");
  if (envNetPluginName && strlen(envNetPluginName)) {
    snprintf(netPluginLibName, PATH_MAX, "%s", envNetPluginName);
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Plugin name set by env to %s", netPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, netPluginLibName);

    snprintf(netPluginLibName, PATH_MAX, "libnccl-net-%s.so", envNetPluginName);
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Plugin name set by env to %s", netPluginLibName);
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, netPluginLibName);
  } else {
    snprintf(netPluginLibName, PATH_MAX, "libnccl-net.so");
    pluginLib = tryOpenLib(netPluginLibName, &openErr, openErrStr);
    if (pluginLib) {
      return pluginLib;
    }
    couldNotFindNames = tryOpenLibCheck(openErr, openErrStr, couldNotFindNames, &len, netPluginLibName);
  }
  return nullptr;
}

static pthread_mutex_t netPluginLock = PTHREAD_MUTEX_INITIALIZER;
static int netPluginRefCount;
static void* netPluginLib;

enum {
  netPluginLoadFailed  = -1,
  netPluginLoadReady   =  0,
  netPluginLoadSuccess =  1,
};

static int netPluginStatus = netPluginLoadReady;

#define MAX_PLUGIN_LOAD 2

ncclResult_t ncclNetPluginLoad(struct ncclComm* comm) {
  char couldNotFindNames[MAX_PLUGIN_LOAD * PATH_MAX] = { 0 };
  pthread_mutex_lock(&netPluginLock);
  if (netPluginLoadFailed == netPluginStatus) {
    goto exit;
  }
  if (netPluginLoadSuccess == netPluginStatus) {
    ++netPluginRefCount;
    goto exit;
  }

  netPluginLib = openNetPluginLib(couldNotFindNames, MAX_PLUGIN_LOAD * PATH_MAX);
  if (netPluginLib == nullptr) {
    if (strlen(couldNotFindNames)) {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Could not find:%s. Using internal network plugin.", couldNotFindNames);
    } else {
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Using internal network plugin.");
    }
    goto fail;
  }

  ncclNets[0] = (ncclNet_v9_t*)dlsym(netPluginLib, "ncclNetPlugin_v9");
  if (ncclNets[0] == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclNetPlugin_v9 symbol.");
    ncclNet_v8 = (ncclNet_v8_t*)dlsym(netPluginLib, "ncclNetPlugin_v8");
    if (ncclNet_v8 == nullptr) {
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
            goto fail;
          } else {
            ncclNets[0] = &ncclNet_v5_as_v9;
            ncclNet_v5_as_v9.init = ncclNet_v5_as_v9_init;
            // Set the name right away to allow for NCCL_NET=... to work
            ncclNet_v5_as_v9.name = ncclNet_v5->name;
            INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v5)", ncclNets[0]->name);
          }
        } else {
          ncclNets[0] = &ncclNet_v6_as_v9;
          ncclNet_v6_as_v9.init = ncclNet_v6_as_v9_init;
          // Set the name right away to allow for NCCL_NET=... to work
          ncclNet_v6_as_v9.name = ncclNet_v6->name;
          INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v6)", ncclNets[0]->name);
        }
      } else {
        ncclNets[0] = &ncclNet_v7_as_v9;
        ncclNet_v7_as_v9.init = ncclNet_v7_as_v9_init;
        // Set the name right away to allow for NCCL_NET=... to work
        ncclNet_v7_as_v9.name = ncclNet_v7->name;
        INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v7)", ncclNets[0]->name);
      }
    } else {
      ncclNets[0] = &ncclNet_v8_as_v9;
      ncclNet_v8_as_v9.init = ncclNet_v8_as_v9_init;
      // Set the name right away to allow for NCCL_NET=... to work
      ncclNet_v8_as_v9.name = ncclNet_v8->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v8)", ncclNets[0]->name);
    }
  } else {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded net plugin %s (v9)", ncclNets[0]->name);
  }

  // Check for CollNet
  ncclCollNets[0] = (ncclCollNet_v9_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v9");
  if (ncclCollNets[0] == nullptr) {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin_v9 symbol.");
    ncclCollNet_v8 = (ncclCollNet_v8_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v8");
    if (ncclCollNet_v8 == nullptr) {
      ncclCollNet_v7 = (ncclCollNet_v7_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v7");
      if (ncclCollNet_v7 == nullptr) {
        ncclCollNet_v6 = (ncclCollNet_v6_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v6");
        if (ncclCollNet_v6 == nullptr) {
          ncclCollNet_v5 = (ncclCollNet_v5_t*)dlsym(netPluginLib, "ncclCollNetPlugin_v5");
          if (ncclCollNet_v5 == nullptr) {
            INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Failed to find ncclCollNetPlugin symbol (>= v5). ncclCollNetPlugin symbols v4 and lower are not supported.");
          } else {
            ncclCollNets[0] = &ncclCollNet_v5_as_v9;
            ncclCollNet_v5_as_v9.init = ncclCollNet_v5_as_v9_init;
            ncclCollNet_v5_as_v9.name = ncclCollNet_v5->name;
            INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v5)", ncclCollNets[0]->name);
          }
        } else {
         ncclCollNets[0] = &ncclCollNet_v6_as_v9;
         ncclCollNet_v6_as_v9.init = ncclCollNet_v6_as_v9_init;
         ncclCollNet_v6_as_v9.name = ncclCollNet_v6->name;
         INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v6)", ncclCollNets[0]->name);
        }
      } else {
       ncclCollNets[0] = &ncclCollNet_v7_as_v9;
       ncclCollNet_v7_as_v9.init = ncclCollNet_v7_as_v9_init;
       ncclCollNet_v7_as_v9.name = ncclCollNet_v7->name;
       INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v7)", ncclCollNets[0]->name);
      }
    } else {
      ncclCollNets[0] = &ncclCollNet_v8_as_v9;
      ncclCollNet_v8_as_v9.init = ncclCollNet_v8_as_v9_init;
      ncclCollNet_v8_as_v9.name = ncclCollNet_v8->name;
      INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v8)", ncclCollNets[0]->name);
    }
  } else {
    INFO(NCCL_INIT|NCCL_NET, "NET/Plugin: Loaded collnet plugin %s (v9)", ncclCollNets[0]->name);
  }

  ++netPluginRefCount;
  netPluginStatus = netPluginLoadSuccess;
  comm->netPluginLoaded = 1;

exit:
  pthread_mutex_unlock(&netPluginLock);
  return ncclSuccess;
fail:
  if (netPluginLib) dlclose(netPluginLib);
  netPluginStatus = netPluginLoadFailed;
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
    dlclose(netPluginLib);
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

ncclResult_t ncclNetFinalize(struct ncclComm* comm) {
  comm->ncclNet = nullptr;
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
    (comm->ncclNet == &ncclNet_v5_as_v9) ? 5 :
    (comm->ncclNet == &ncclNet_v6_as_v9) ? 6 :
    (comm->ncclNet == &ncclNet_v7_as_v9) ? 7 :
    (comm->ncclNet == &ncclNet_v8_as_v9) ? 8 :
    9;
}
