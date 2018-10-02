/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <nccl.h>
#include <nccl_net.h>

#define __hidden __attribute__ ((visibility("hidden")))

static int plugin_major = 1;
static int plugin_minor = 0;
static const char *plugin_name = "<a plugged-in network>";
static ncclLoggerFunction_t log;

__hidden ncclResult_t plugin_devices(int* ndev, int** scores) { return ncclSuccess; }
__hidden ncclResult_t plugin_ptrSupport(int dev, int* supportedTypes) { return ncclSuccess; }
__hidden ncclResult_t plugin_listen(int dev, void* handle, void** listenComm) { return ncclSuccess; }
__hidden ncclResult_t plugin_connect(int dev, void* handle, void** sendComm) { return ncclSuccess; }
__hidden ncclResult_t plugin_accept(void* listenComm, void** recvComm) { return ncclSuccess; }
__hidden ncclResult_t plugin_isend(void* sendComm, void* data, int size, int type, void** request) { return ncclSuccess; }
__hidden ncclResult_t plugin_irecv(void* recvComm, void* data, int size, int type, void** request) { return ncclSuccess; }
__hidden ncclResult_t plugin_flush(void* recvComm, void* data, int size) { return ncclSuccess; }
__hidden ncclResult_t plugin_test(void* request, int* done, int* size) { return ncclSuccess; }
__hidden ncclResult_t plugin_closeSend(void* sendComm) { return ncclSuccess; }
__hidden ncclResult_t plugin_closeRecv(void* recvComm) { return ncclSuccess; }
__hidden ncclResult_t plugin_closeListen(void* listenComm) { return ncclSuccess; }
__hidden ncclResult_t plugin_netFini() { return ncclSuccess; }

ncclResult_t ncclNetPluginGetVersion(int *major, int *minor) {
  *major = plugin_major;
  *minor = plugin_minor;
  return ncclSuccess;
}

ncclResult_t ncclNetPluginInit(void *ncclNetParams, void *ncclNet) {
  ncclNetParams_1_0_t *params = (ncclNetParams_1_0_t *)ncclNetParams;
  log = params->loggerFunction;

  ncclNet_1_0_t *net = (ncclNet_1_0_t *)ncclNet;
  net->name = plugin_name;
  net->devices = plugin_devices;
  net->ptrSupport = plugin_ptrSupport;
  net->listen = plugin_listen;
  net->connect = plugin_connect;
  net->accept = plugin_accept;
  net->isend = plugin_isend;
  net->irecv = plugin_irecv;
  net->flush = plugin_flush;
  net->test = plugin_test;
  net->closeSend = plugin_closeSend;
  net->closeRecv = plugin_closeRecv;
  net->closeListen = plugin_closeListen;
  net->netFini = plugin_netFini;
  return ncclSuccess;
}
