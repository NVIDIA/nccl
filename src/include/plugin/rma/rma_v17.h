/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMA_V17_H_
#define RMA_V17_H_
#include "nccl_net.h"
#include "rma/rma_v16.h"

typedef struct {
  int nContexts;
  int trafficClass;
  // Sorted list of peerArrayCount peers to connect to. Plugins may choose to connect to only these peers to optimize
  // resource usage. It is still correct to connect to all peers. This field does not affect rank indexing - ranks are always
  // indexed according to the handles list passed to connect. It is undefined behavior for peerArray to request
  // unidirectional connectivity (if rank i is in rank j's peerArray, rank j must be in rank i's peerArray).
  int* peerArray;
  // Number of peers in the peerArray. May be 0 if RMA is requested but this rank requires no RMA connectivity.
  int peerArrayCount;
} ncclRmaConfig_v17_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  // Get the RMA properties for `collComm`.
  ncclResult_t (*getRmaProperties)(void* collComm, ncclRmaProperties_v16_t* rmaProps);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm,
                          volatile uint32_t* abortFlag);
  ncclResult_t (*createContext)(void* collComm, ncclRmaConfig_v17_t* config, void** rmaCtx);
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  ncclResult_t (*destroyContext)(void* rmaCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // optFlags carries ncclRmaOptFlags.
  ncclResult_t (*iput)(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                       void* dstMhandle, uint32_t rank, uint32_t optFlags, void** request);
  ncclResult_t (*iputSignal)(void* rmaCtx, int context, uint64_t srcOff, void* srcMhandle, size_t size, uint64_t dstOff,
                             void* dstMhandle, uint32_t rank, uint64_t signalOff, void* signalMhandle,
                             uint64_t signalValue, uint32_t signalOp, bool isStrongSignal, uint32_t optFlags,
                             void** request);
  ncclResult_t (*iget)(void* rmaCtx, int context, uint64_t remoteOff, void* remoteMhandle, size_t size,
                       uint64_t localOff, void* localMhandle, uint32_t rank, uint32_t optFlags, void** request);

  ncclResult_t (*iflush)(void* rmaCtx, int context, void* mhandle, uint32_t rank, void** request);
  ncclResult_t (*test)(void* collComm, void* request, int* done);
  ncclResult_t (*rmaProgress)(void* rmaCtx);
  ncclResult_t (*queryLastError)(void* rmaCtx, bool* hasError);
  ncclResult_t (*finalize)(void* ctx);
} ncclRma_v17_t;
#endif
