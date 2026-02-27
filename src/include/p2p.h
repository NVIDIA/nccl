/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "core.h"
#include "mem_manager.h"

// CUmemFabricHandle compatibility definitions are now in mem_manager.h

typedef union {
  uint64_t data; // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  CUmemFabricHandle handle;
} ncclCuDesc;

typedef union {
  // Legacy CUDA IPC
  cudaIpcMemHandle_t devIpc;
  // cuMem API support
  struct {
    ncclCuDesc cuDesc;
    CUmemGenericAllocationHandle memHandle;
  };
} ncclIpcDesc;

enum ncclIpcRegType {
  NCCL_IPC_SENDRECV = 0,
  NCCL_IPC_COLLECTIVE = 1
};

struct ncclIpcImpInfo {
  void* rmtRegAddr;
  bool legacyIpcCap;
  uintptr_t offset;
  int numSegments;
};

struct ncclIpcRegInfo {
  int peerRank;
  void* baseAddr;
  struct ncclProxyConnector* ipcProxyconn;
  struct ncclIpcImpInfo impInfo;
};

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, int directMap, ncclIpcDesc *ipcDesc, void **ptr, int peerRank = -1, struct ncclMemManager* manager = nullptr, ncclMemType_t memtype = ncclMemPersist);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int peer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr, void* ownerPtr = nullptr, ncclMemType_t memType = ncclMemPersist);
ncclResult_t ncclIpcLocalRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut);
ncclResult_t ncclIpcGraphRegisterBuffer(ncclComm* comm, const void* userbuff, size_t buffSize, int* peerRanks, int nPeers, ncclIpcRegType type, int* regBufFlag, uintptr_t* offsetOut, uintptr_t** peerRmtAddrsOut, void* cleanupQueuePtr, int* nCleanupQueueElts);

ncclResult_t ncclIpcDeregBuffer(struct ncclComm* comm, struct ncclIpcRegInfo* regInfo);

#endif
