/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NET_V11_H_
#define NET_V11_H_

typedef struct {
  int ndevs;
  int devs[NCCL_NET_MAX_DEVS_PER_NIC];
} ncclNetVDeviceProps_v11_t;

#define NCCL_NET_TRAFFIC_CLASS_UNDEF -1

typedef struct {
  // Plugin-specific TC value
  int trafficClass;
} ncclNetCommConfig_v11_t;

typedef struct {
  char* name;                      // Used mostly for logging.
  char* pciPath;                   // Path to the PCI device in /sys.
  uint64_t guid;                   // Unique identifier for the NIC chip. Important for
                                   // cards with multiple PCI functions (Physical or virtual).
  int ptrSupport;                  // [NCCL_PTR_HOST|NCCL_PTR_CUDA|NCCL_PTR_DMABUF]
  int regIsGlobal;                 // regMr is not tied to a particular comm
  int forceFlush;                  // Force a flush on receives
  int speed;                       // Port speed in Mbps.
  int port;                        // Port number.
  float latency;                   // Network latency
  int maxComms;                    // Maximum number of comms we can create
  int maxRecvs;                    // Maximum number of grouped receives.
  ncclNetDeviceType netDeviceType; // Network offload type
  int netDeviceVersion;            // Version number for network offload
  ncclNetVDeviceProps_v11_t vProps;
  size_t maxP2pBytes;              // Max transfer size for point-to-point operations
  size_t maxCollBytes;             // Max transfer size for collective operations
  int maxMultiRequestSize;         // Maximum number of requests supported in a single multi-request.
} ncclNetProperties_v11_t;

#define NCCL_NET_ATTR_UNDEF -1

#define NCCL_NET_ATTR_INIT { \
  { NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF }, /* sendCommAttr */ \
  { NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF, NCCL_NET_ATTR_UNDEF }, /* recvCommAttr */ \
  (uint32_t)NCCL_NET_ATTR_UNDEF, /* op */ \
  (uint32_t)NCCL_NET_ATTR_UNDEF, /* algo */ \
  (uint32_t)NCCL_NET_ATTR_UNDEF, /* proto */ \
}

typedef struct {
  int32_t maxConcurrentPeers;
  int32_t minConcurrentPeers;
  int32_t maxFlowsPerPeer;
  int32_t minFlowsPerPeer;
} ncclNetCommAttr_v11_t;

typedef struct {
  ncclNetCommAttr_v11_t sendCommAttr;
  ncclNetCommAttr_v11_t recvCommAttr;
  uint32_t op;
  uint32_t algo;
  uint32_t proto;
} ncclNetAttr_v11_t;

typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config, ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction);
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v11_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  // This call must not block for the connection to be established, and instead
  // should return successfully with sendComm == NULL with the expectation that
  // it will be called again until sendComm != NULL.
  // If *sendDevComm points to a valid object, then NCCL is requesting device offload for this connection
  ncclResult_t (*connect)(void* ctx, int dev, void* handle, void** sendComm, ncclNetDeviceHandle_v11_t** sendDevComm);
  // Finalize connection establishment after remote peer has called connect.
  // This call must not block for the connection to be established, and instead
  // should return successfully with recvComm == NULL with the expectation that
  // it will be called again until recvComm != NULL.
  // If *recvDevComm points to a valid object, then NCCL is requesting device offload for this connection
  ncclResult_t (*accept)(void* listenComm, void** recvComm, ncclNetDeviceHandle_v11_t** recvDevComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, size_t size, int tag, void* mhandle, void* phandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, int n, void** data, size_t* sizes, int* tags, void** mhandles, void** phandles, void** request);
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

  // Virtual NIC APIs. makeVDevice will create a virtual NIC given the specified properties, and tell the caller
  // what index this new vNIC exists at
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v11_t* props);
  // Finalize the network.
  ncclResult_t (*finalize)(void* ctx);

  ncclResult_t (*setNetAttr)(void* ctx, ncclNetAttr_v11_t* netAttr);
} ncclNet_v11_t;

typedef struct {
  void* mhandle;
  void* address;
  size_t size;
} ncclNetSGE_v11_t;

typedef struct {
  // Name of the collective network (mainly for logs)
  const char* name;
  // Initialize the collective network.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing collective operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v11_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for collective operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Returns whether a reduction operation on a data type is supported.
  // 1 for supported, 0 otherwise.
  ncclResult_t (*reduceSupport)(ncclDataType_t dataType, ncclRedOp_t redOp, int* supported);
  // Register/Deregister memory. Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* collComm, void* data, size_t size, int type, void** mhandle);
  /* DMA-BUF support */
  ncclResult_t (*regMrDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle);
  ncclResult_t (*deregMr)(void* collComm, void* mhandle);
  // Performs an asynchronous allreduce operation on the collective group.
  // May return request == NULL if the call cannot be performed (or would block).
  ncclResult_t (*iallreduce)(void* collComm, void* sendData, void* recvData, size_t count,
      ncclDataType_t dataType, ncclRedOp_t redOp, void* sendMhandle, void* recvMhandle, void** request);
  ncclResult_t (*iallgather)(void* collComm, void* sendData, int nRecvParts, ncclNetSGE_v11_t* recvParts,
                             size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                             void* sendMhandle, void** request);
  ncclResult_t (*ireducescatter)(void* collComm, int nSendParts, ncclNetSGE_v11_t* sendParts, void* recvData,
                                 size_t bytesPerRank, size_t windowOffset, size_t windowBytes,
                                 ncclDataType_t dataType, ncclRedOp_t redOp,
                                 void* recvMhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*iflush)(void* collComm, void* data, int size, void* mhandle, void** request);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free collective comm objects
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Create a virtual NIC given the specified properties, which can be accessed at device index d
  ncclResult_t (*makeVDevice)(int* d, ncclNetVDeviceProps_v11_t* props);
  // Finalize the collective network.
  ncclResult_t (*finalize)(void* ctx);
} ncclCollNet_v11_t;
#endif // end include guard
