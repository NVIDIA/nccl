/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_V11_H_
#define GIN_V11_H_
typedef struct {
  // Name of the GIN support (mainly for logs)
  const char* name;
  // Initialize the GIN support.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing GIN operations.
  // If ndev returns 0, all other functions might be set to NULL.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v11_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for GIN operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Create device-side GIN context. devHandle will be passed to device code.
  // This function is not used in GIN_PROXY mode.
  ncclResult_t (*createContext)(void* collComm, int nSignals, int nCounters, void** ginCtx, ncclNetDeviceHandle_v11_t** devHandle);
  // Collective memory registration
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle, void **ginHandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd, uint64_t mrFlags, void** mhandle, void **ginHandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  // Close and free collective comm objects
  ncclResult_t (*destroyContext)(void* ginCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Put operations
  ncclResult_t (*iput)(void* collComm, uint64_t srcOff, void* srcMhandle, size_t size,
      uint64_t dstOff, void* dstMhandle, uint32_t rank, void** request);
  ncclResult_t (*iputSignal)(void* collComm, uint64_t srcOff, void* srcMhandle,
      size_t size, uint64_t dstOff, void* dstMhandle,
      uint32_t rank, uint64_t signalOff, void *signalMhandle,
      uint64_t signalValue, uint32_t signalOp, void** request);

  // Test whether a request is complete.
  ncclResult_t (*test)(void* collComm, void* request, int* done);

  // Progress function. Will be called if non-NULL in GIN_PROXY mode, or if devHandle.needsProxyProgress=1.
  ncclResult_t (*ginProgress)(void* collComm);

  // Query the last error for the GIN support. Particularly important when ginProgress is not used, to report errors.
  ncclResult_t (*queryLastError)(void* ginCtx, bool *hasError);

  // Finalize the GIN support
  ncclResult_t (*finalize)(void* ctx);
} ncclGin_v11_t;
#endif // end include guard
