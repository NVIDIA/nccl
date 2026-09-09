/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMA_V14_H_
#define RMA_V14_H_

#include <stdbool.h>

typedef struct {
  int nContexts;
  int trafficClass;
  int rankStride;
} ncclRmaConfig_v14_t;

typedef struct {
  // Name of the RMA support (mainly for logs)
  const char* name;
  // Initialize the RMA support.
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  // Return the number of adapters capable of doing RMA operations.
  ncclResult_t (*devices)(int* ndev);
  // Get various device properties.
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_RMA_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create connections.
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  // Create a group for RMA operations. handles have been created
  // using listen() above. rank indicates caller's rank in the collective network.
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  // Create a set of connections between the group. Config indicates connection properties,
  // like the number of contexts and traffic class.
  ncclResult_t (*createContext)(void* collComm, ncclRmaConfig_v14_t* config, void** rmaCtx);
  // Collective memory registration
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  // Close and free collective comm objects
  ncclResult_t (*destroyContext)(void* rmaCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // Data operations
  ncclResult_t (*iput)(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                       void* dstMhandle, uint32_t rank, void** request);
  ncclResult_t (*iputSignal)(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                             void* dstMhandle, uint32_t rank, uint64_t signalOff, void* signalMhandle,
                             uint64_t signalValue, uint32_t signalOp, bool isStrongSignal, void** request);
  ncclResult_t (*iget)(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                       uint64_t localOff, void* localMhandle, uint32_t rank, void** request);

  ncclResult_t (*iflush)(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request);

  // Test whether a request is complete.
  ncclResult_t (*test)(void* collComm, void* request, int* done);

  // Progress function. Will be called if non-NULL.
  ncclResult_t (*rmaProgress)(void* rmaCtx);

  // Query the last error for the RMA support. Particularly important when rmaProgress is not used, to report errors.
  ncclResult_t (*queryLastError)(void* rmaCtx, bool* hasError);

  // Finalize the RMA support
  ncclResult_t (*finalize)(void* ctx);
} ncclRma_v14_t;
#endif // end include guard
