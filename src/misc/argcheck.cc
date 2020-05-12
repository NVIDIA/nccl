/*************************************************************************
 * Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h"
#include "comm.h"

static ncclResult_t CudaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, pointer);
  if (err != cudaSuccess || attr.devicePointer == NULL) {
    WARN("%s : %s is not a valid pointer", opname, ptrname);
    return ncclInvalidArgument;
  }
#if CUDART_VERSION >= 10000
  if (attr.type == cudaMemoryTypeDevice && attr.device != comm->cudaDev) {
#else
  if (attr.memoryType == cudaMemoryTypeDevice && attr.device != comm->cudaDev) {
#endif
    WARN("%s : %s allocated on device %d mismatchs with NCCL device %d", opname, ptrname, attr.device, comm->cudaDev);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t PtrCheck(void* ptr, const char* opname, const char* ptrname) {
  if (ptr == NULL) {
    WARN("%s : %s argument is NULL", opname, ptrname);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t ArgsCheck(struct ncclInfo* info) {
  // First, the easy ones
  if (info->root < 0 || info->root >= info->comm->nRanks) {
    WARN("%s : invalid root %d (root should be in the 0..%d range)", info->opName, info->root, info->comm->nRanks);
    return ncclInvalidArgument;
  }
  if (info->datatype < 0 || info->datatype >= ncclNumTypes) {
    WARN("%s : invalid type %d", info->opName, info->datatype);
    return ncclInvalidArgument;
  }
  // Type is OK, compute nbytes. Convert Allgather/Broadcast/P2P calls to chars.
  info->nBytes = info->count * ncclTypeSize(info->datatype);
  if (info->coll == ncclCollAllGather || info->coll == ncclCollBroadcast) {
    info->count = info->nBytes;
    info->datatype = ncclInt8;
  }
  if (info->coll == ncclCollAllGather || info->coll == ncclCollReduceScatter) info->nBytes *= info->comm->nRanks; // count is per rank

  if (info->op < 0 || info->op >= ncclNumOps) {
    WARN("%s : invalid reduction operation %d", info->opName, info->op);
    return ncclInvalidArgument;
  }

  if (info->comm->checkPointers) {
    if (info->coll == ncclCollSendRecv) {
      if (strcmp(info->opName, "Send") == 0) {
        NCCLCHECK(CudaPtrCheck(info->sendbuff, info->comm, "sendbuff", "Send"));
      } else {
        NCCLCHECK(CudaPtrCheck(info->recvbuff, info->comm, "recvbuff", "Recv"));
      }
    } else {
      // Check CUDA device pointers
      if (info->coll != ncclCollBroadcast || info->comm->rank == info->root) {
        NCCLCHECK(CudaPtrCheck(info->sendbuff, info->comm, "sendbuff", info->opName));
      }
      if (info->coll != ncclCollReduce || info->comm->rank == info->root) {
        NCCLCHECK(CudaPtrCheck(info->recvbuff, info->comm, "recvbuff", info->opName));
      }
    }
  }
  return ncclSuccess;
}
