/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_NET_H_
#define NCCL_NET_H_

#include "nccl.h"
#include "nccl_common.h"
#include "net_device.h"
#include <stdint.h>

#define NCCL_NET_HANDLE_MAXSIZE 128

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2
#define NCCL_PTR_DMABUF 0x4

// Maximum number of requests per comm object
#define NCCL_NET_MAX_REQUESTS 32

typedef struct {
  char* name;                      // Used mostly for logging.
  char* pciPath;                   // Path to the PCI device in /sys.
  uint64_t guid;                   // Unique identifier for the NIC chip. Important for
                                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;                  // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int speed;                       // Port speed in Mbps.
  int port;                        // Port number.
  float latency;                   // Network latency
  int maxComms;                    // Maximum number of comms we can create
  int maxRecvs;                    // Maximum number of grouped receives.
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
} ncclNetProperties_v7_t;

typedef ncclNetProperties_v7_t ncclNetProperties_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v7_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then NCCL is requesting device offload for this connection
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v7_t** sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then NCCL is requesting device offload for this connection
  ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v7_t** recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Copy the given mhandle to a dptr in a format usable by this plugin's device code
  ncclResult_t (*getDeviceMr)(void* comm, void* mhandle, void** dptr_mhandle);

  // Notify the plugin that a recv has completed by the device
  ncclResult_t (*irecvConsumed)(void* recvComm, int n, void* request);
} ncclNet_v7_t;

typedef ncclNet_v7_t ncclNet_t;

#define NCCL_NET_PLUGIN_SYMBOL ncclNetPlugin_v7

#define NCCL_COLLNET_PLUGIN_SYMBOL ncclCollNetPlugin_v7

#define NCCL_NET_MAX_REQUESTS_V6 8

// v6 struct for backwards compatibility
typedef struct {
  char* name;     // Used mostly for logging.
  char* pciPath;  // Path to the PCI device in /sys.
  uint64_t guid;  // Unique identifier for the NIC chip. Important for
                  // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport; // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int speed;      // Port speed in Mbps.
  int port;       // Port number.
  float latency;  // Network latency
  int maxComms;   // Maximum number of comms we can create
  int maxRecvs;   // Maximum number of grouped receives.
} ncclNetProperties_v6_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v6_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v7_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_v7_t;

typedef ncclCollNet_v7_t ncclCollNet_t;

// v6 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_v6_t;

// v5 struct for backwards compatibility
typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, int tag, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, int* sizes, int* tags, void** mhandles, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* recvComm, int n, void** data, int* sizes, void** mhandles, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* sizes);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v5_t;

// v5 struct for backwards compatibility
typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  ncclResult_t (*init)(ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v6_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, int size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, int count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclCollNet_v5_t;

#endif // end include guard
