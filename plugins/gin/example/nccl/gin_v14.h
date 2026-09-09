/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_V14_H_
#define GIN_V14_H_

#include <stdbool.h>

typedef struct {
  bool supportsStrongSignals;
  bool supportsVASignals;
} ncclGinProperties_v14_t;

typedef struct {
  int nSignals;
  int nCounters;
  int nContexts;
  int queueDepth;
  int trafficClass;
  int backendVersion;
  int rankStride;
} ncclGinConfig_v14_t;

// NOTE: v14 removes data operations (iput, iputSignal, iget, iflush) from the GIN interface.
// These moved to the separate RMA plugin (ncclRma_v14_t). See plugins/rma/example for a
// reference implementation.
typedef struct {
  // Name of the GIN support (mainly for logs)
  const char* name;
  // Initialize the GIN support.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing GIN operations.
  ncclResult_t (*devices)(int* ndev);
  // Get GIN capabilities (signal type support).
  ncclResult_t (*getGinProperties)(ncclGinProperties_v14_t* ginProps);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_GIN_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for GIN operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Create device-side GIN context. devHandle will be passed to device code.
  ncclResult_t (*createContext)(void* collComm, ncclGinConfig_v14_t* config, void** ginCtx,
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
} ncclGin_v14_t;
#endif // end include guard
