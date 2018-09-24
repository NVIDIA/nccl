/*************************************************************************
 * Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"

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
} ncclNet_t;

extern
#ifdef __cplusplus
"C"
#endif
ncclNet_t* ncclNet;

#endif // end include guard
