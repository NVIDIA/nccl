/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_V15_H_
#define GIN_V15_H_
#include "nccl_net.h"

typedef struct {
  bool supportsStrongSignals;
  bool supportsVASignals;
  // Barrier preference; see ncclGinBarrierOptions_t in nccl_device/net_device.h. The v14 compatibility layer
  // reports NCCL_GIN_BARRIER_DEFAULT.
  ncclGinBarrierOptions_t barrierOptions;
} ncclGinProperties_v15_t;

typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
  int backendVersion;
  // Sorted list of peerArrayCount peers to connect to. Plugins may choose to connect to only these peers to optimize
  // resource usage. It is still correct to connect to all peers. This field does not affect rank indexing - ranks are always
  // indexed according to the handles list passed to connect. It is undefined behavior for peerArray to request
  // unidirectional connectivity (if rank i is in rank j's peerArray, rank j must be in rank i's peerArray).
  int* peerArray;
  // Number of peers in the peerArray. May be 0 if GIN is requested but this rank requires no GIN connectivity.
  int peerArrayCount;
} ncclGinConfig_v15_t;

typedef struct {
  // Name of the GIN support (mainly for logs)
  const char* name;
  // Initialize the GIN support.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing GIN operations.
  ncclResult_t (*devices)(int* ndev);
  // Get the GIN properties.
  ncclResult_t (*getGinProperties)(ncclGinProperties_v15_t* ginProps);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for GIN operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                          volatile uint32_t* abortFlag);
  // Create device-side GIN context. devHandle will be passed to device code.
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_v15_t* config, void** ginCtx,
                                ncclNetDeviceHandle_v11_t** devHandle);
  // Collective memory registration
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle,
                           void** ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle, void** ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  // Close and free collective comm objects
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Progress function. Will be called if devHandle.needsProxyProgress=1.
  ncclResult_t (*ginProgress)(void* ginCtx);

  // Query the last error for the GIN support. Particularly important when ginProgress is not used, to report errors.
  ncclResult_t (*queryLastError)(void* ginCtx, bool* hasError);

  // Finalize the GIN support
  ncclResult_t (*finalize)(void* ctx);
} ncclGin_v15_t;
#endif // end include guard
