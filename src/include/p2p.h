/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdlib.h>

#ifndef NCCL_P2P_H_
#define NCCL_P2P_H_

#include <cuda.h>

#if CUDART_VERSION < 12030
// MNNVL: FABRIC handle support lifted from CUDA 12.3
#define CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED ((CUdevice_attribute)128)
#define CU_MEM_HANDLE_TYPE_FABRIC ((CUmemAllocationHandleType)0x8ULL)
#define CU_IPC_HANDLE_SIZE 64
typedef struct CUmemFabricHandle_st {
    unsigned char data[CU_IPC_HANDLE_SIZE];
} CUmemFabricHandle_v1;
typedef CUmemFabricHandle_v1 CUmemFabricHandle;
#endif

typedef union {
  uint64_t data; // Needs to hold a CUmemGenericAllocationHandle for UDS fd support
  CUmemFabricHandle handle;
} ncclCuDesc;

typedef union {
  // Legacy CUDA IPC
  cudaIpcMemHandle_t devIpc;
  // cuMem API support
  ncclCuDesc cuDesc;
} ncclIpcDesc;

ncclResult_t ncclP2pAllocateShareableBuffer(size_t size, ncclIpcDesc *ipcDesc, void **ptr);
ncclResult_t ncclP2pFreeShareableBuffer(ncclIpcDesc *ipcDesc);
ncclResult_t ncclP2pImportShareableBuffer(struct ncclComm *comm, int tpPeer, size_t size, ncclIpcDesc *ipcDesc, void **devMemPtr);

#endif
