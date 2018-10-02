/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"

#ifdef __cplusplus
extern "C" {
#endif

#define NCCL_NET_MAJOR 1
#define NCCL_NET_MINOR 0

#define NCCL_NET_HANDLE_MAXSIZE 64

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2

#define NCCL_MAX_SCORE 0x7

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Return the number of network devices along with their scores relative to the
  // current CUDA device. The per device score should be a value from 1-7 with a
  // higher score representing a better choice for performance.
  // This call should allocate the 'scores' array using malloc(3), and it
  // will then be freed automatically by NCCL.
  ncclResult_t (*devices)(int* ndev, int** scores);
  // Return whether this device supports host pointers and/or CUDA pointers
  // as data from the current GPU. Supported types should be composed with
  // NCCL_PTR_HOST and NCCL_PTR_CUDA.
  ncclResult_t (*ptrSupport)(int dev, int* supportedTypes);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Asynchronous send to a peer. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int type, void** request);
  // Asynchronous recv from a peer. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, int type, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*flush)(void* recvComm, void* data, int size);
  // Test whether a request is complete and return the size received (can be less than requested).
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
  // Called by NCCL when the net will no longer be in use
  ncclResult_t (*netFini)();
} ncclNet_1_0_t;
typedef ncclNet_1_0_t ncclNet_t;

typedef enum {
  NCCL_DEBUG_NONE,
  NCCL_DEBUG_VERSION,
  NCCL_DEBUG_WARN,
  NCCL_DEBUG_INFO,
  NCCL_DEBUG_ABORT,
  NCCL_DEBUG_TRACE} ncclDebugLevel_t;
typedef int (*ncclLoggerFunction_t)(ncclDebugLevel_t debugLevel, const char *format, ...);
typedef struct {
  ncclLoggerFunction_t loggerFunction;
} ncclNetParams_1_0_t;
typedef ncclNetParams_1_0_t ncclNetParams_t;

/* NCCL passes in its current net API version. The plugin returns
 * the API version it can support. NCCL will refuse to use a plugin
 * that uses a newer API version that it was built with.
 * Otherwise, NCCL will use the net plugin's API version. */
ncclResult_t ncclNetPluginGetVersion(int *major, int *minor);
typedef ncclResult_t (*ncclNetPluginGetVersion_t)(int *, int *);

/* NCCL passes in the parameters structure (ncclNetParams). The plugin
 * should fill out the structure pointed to by ncclNet */
ncclResult_t ncclNetPluginInit(void *ncclNetParams, void *ncclNet);
typedef ncclResult_t (*ncclNetPluginInit_t)(void *, void *);

/* TODO remove this */
extern ncclNet_t* ncclNet;

#ifdef __cplusplus
} // extern "C"
#endif

#endif // end include guard
