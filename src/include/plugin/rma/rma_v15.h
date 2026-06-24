/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMA_V15_H_
#define RMA_V15_H_
#include "nccl_net.h"
#include "rma/rma_v14.h"

typedef ncclRmaConfig_v14_t ncclRmaConfig_v15_t;

typedef struct {
  const char* name;
  ncclResult_t (*init)(void** ctx, uint64_t commId, ncclDebugLogger_t logFunction);
  ncclResult_t (*devices)(int* ndev);
  ncclResult_t (*getProperties)(int dev, ncclNetProperties_v12_t* props);
  ncclResult_t (*listen)(void* ctx, int dev, void* handle, void** listenComm);
  ncclResult_t (*connect)(void* ctx, void* handles[], int nranks, int rank, void* listenComm, void** collComm);
  ncclResult_t (*createContext)(void* collComm, ncclRmaConfig_v15_t* config, void** rmaCtx);
  ncclResult_t (*regMrSym)(void* collComm, void* data, size_t size, int type, uint64_t mrFlags, void** mhandle);
  ncclResult_t (*regMrSymDmaBuf)(void* collComm, void* data, size_t size, int type, uint64_t offset, int fd,
                                 uint64_t mrFlags, void** mhandle);
  ncclResult_t (*deregMrSym)(void* collComm, void* mhandle);
  ncclResult_t (*destroyContext)(void* rmaCtx);
  ncclResult_t (*closeColl)(void* collComm);
  ncclResult_t (*closeListen)(void* listenComm);

  // optFlags carries ncclGinOptFlags. With ncclGinOptFlagsAggregateRequests the backend may
  // defer the doorbell; the default flushes.
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
} ncclRma_v15_t;
#endif
