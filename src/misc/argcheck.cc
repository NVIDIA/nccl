/*************************************************************************
 * Copyright (c) 2019-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "argcheck.h"
#include "comm.h"
#include "bootstrap.h"

ncclResult_t CudaPtrCheck(const void* pointer, struct ncclComm* comm, const char* ptrname, const char* opname) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, pointer);
  if (err != cudaSuccess || attr.devicePointer == NULL) {
    WARN("%s : %s %p is not a valid pointer", opname, ptrname, pointer);
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

ncclResult_t PtrCheck(const void* ptr, const char* opname, const char* ptrname) {
  if (ptr == NULL) {
    WARN("%s : %s argument is NULL", opname, ptrname);
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

ncclResult_t CommCheck(struct ncclComm* comm, const char* opname, const char* ptrname) {
  NCCLCHECK(PtrCheck(comm, opname, ptrname));
  if (comm->startMagic != NCCL_MAGIC || comm->endMagic != NCCL_MAGIC) {
    WARN("Error: corrupted comm object detected");
    return ncclInvalidArgument;
  }
  return ncclSuccess;
}

static ncclResult_t registrationCheck(struct ncclInfo* info) {
  struct symBufInfo {
    bool isSymRegistered;
    uintptr_t bigOffset;
    uintptr_t userOffset;
  };

  ncclResult_t ret = ncclSuccess;
  struct ncclComm *comm = info->comm;
  struct symBufInfo *bufInfo = nullptr; // send and recv buffers
  struct ncclDevrWindow* sendWin = nullptr;
  struct ncclDevrWindow* recvWin = nullptr;
  struct symBufInfo cmpBufInfo[2] = {};
  bool sendWinMismatch = false;
  bool recvWinMismatch = false;
  bool sendUserMismatch = false;
  bool recvUserMismatch = false;
  int sendWinMismatchRank = -1;
  int recvWinMismatchRank = -1;
  int sendUserMismatchRank = -1;
  int recvUserMismatchRank = -1;
  size_t size = info->count * ncclTypeSize(info->datatype);
  int myInfoIdx = comm->rank * 2; // my starting index in bufInfo

  NCCLCHECKGOTO(ncclCalloc(&bufInfo, comm->nRanks * 2), ret, fail);
  NCCLCHECKGOTO(ncclDevrFindWindow(comm, info->sendbuff, &sendWin), ret, fail);
  NCCLCHECKGOTO(ncclDevrFindWindow(comm, info->recvbuff, &recvWin), ret, fail);

  if (sendWin && (sendWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
    bufInfo[myInfoIdx].isSymRegistered = true;
    bufInfo[myInfoIdx].bigOffset = sendWin->bigOffset;
    bufInfo[myInfoIdx].userOffset = (uintptr_t)info->sendbuff - (uintptr_t)sendWin->userPtr;
    INFO(NCCL_COLL, "SymCheck: coll %s size %ld rank %d bigOffset %lx userOffset %lx info->sendbuff %p sendWin->userPtr %p", info->opName, size, comm->rank, bufInfo[myInfoIdx].bigOffset, bufInfo[myInfoIdx].userOffset, info->sendbuff, sendWin->userPtr);
  }

  if (recvWin && (recvWin->winFlags & NCCL_WIN_COLL_SYMMETRIC)) {
    bufInfo[myInfoIdx + 1].isSymRegistered = true;
    bufInfo[myInfoIdx + 1].bigOffset = recvWin->bigOffset;
    bufInfo[myInfoIdx + 1].userOffset = (uintptr_t)info->recvbuff - (uintptr_t)recvWin->userPtr;
    INFO(NCCL_COLL, "SymCheck: coll %s size %ld rank %d bigOffset %lx userOffset %lx info->recvbuff %p recvWin->userPtr %p", info->opName, size, comm->rank, bufInfo[myInfoIdx + 1].bigOffset, bufInfo[myInfoIdx + 1].userOffset, info->recvbuff, recvWin->userPtr);
  }

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, bufInfo, sizeof(struct symBufInfo) * 2), ret, fail);

  cmpBufInfo[0] = bufInfo[0];
  cmpBufInfo[1] = bufInfo[1];
  for (int r = 1; r < comm->nRanks; r++) {
    int infoIdx = r * 2;
    if (cmpBufInfo[0].isSymRegistered != bufInfo[infoIdx].isSymRegistered || cmpBufInfo[1].isSymRegistered != bufInfo[infoIdx + 1].isSymRegistered) {
      if (comm->rank == 0) WARN("Coll %s size %ld symmetric registration check failed on rank %d: sendReg %d recvReg %d mismatch with rank 0 sendReg %d recvReg %d", info->opName, size, r, bufInfo[infoIdx].isSymRegistered, bufInfo[infoIdx + 1].isSymRegistered, cmpBufInfo[0].isSymRegistered, cmpBufInfo[1].isSymRegistered);
      ret = ncclInvalidArgument;
      goto fail;
    }

    if (cmpBufInfo[0].bigOffset != bufInfo[infoIdx].bigOffset) { sendWinMismatch = true; sendWinMismatchRank = r; }
    if (cmpBufInfo[1].bigOffset != bufInfo[infoIdx + 1].bigOffset) { recvWinMismatch = true; recvWinMismatchRank = r; }
    if (cmpBufInfo[0].userOffset != bufInfo[infoIdx].userOffset) { sendUserMismatch = true; sendUserMismatchRank = r; }
    if (cmpBufInfo[1].userOffset != bufInfo[infoIdx + 1].userOffset) { recvUserMismatch = true; recvUserMismatchRank = r; }
  }

  if (info->coll == ncclFuncAllReduce || info->coll == ncclFuncReduceScatter || info->coll == ncclFuncAlltoAll || info->coll == ncclFuncGather) {
    if (cmpBufInfo[0].isSymRegistered) {
      if (sendWinMismatch) {
        if (comm->rank == 0) WARN("Coll %s size %ld symmetric registration check failed on rank %d: send buffer window (0x%lx) mismatch with rank 0 (0x%lx)", info->opName, size, sendWinMismatchRank, bufInfo[sendWinMismatchRank * 2].bigOffset, cmpBufInfo[0].bigOffset);
        ret = ncclInvalidArgument;
        goto fail;
      }
      if (sendUserMismatch) {
        if (comm->rank == 0) WARN("Coll %s size %ld symmetric registration check failed on rank %d: send buffer user offset (0x%lx) mismatch with rank 0 (0x%lx)", info->opName, size, sendUserMismatchRank, bufInfo[sendUserMismatchRank * 2].userOffset, cmpBufInfo[0].userOffset);
        ret = ncclInvalidArgument;
        goto fail;
      }
    }
  }

  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncAllReduce || info->coll == ncclFuncAlltoAll || info->coll == ncclFuncScatter) {
    if (cmpBufInfo[1].isSymRegistered) {
      if (recvWinMismatch) {
        if (comm->rank == 0) WARN("Coll %s size %ld symmetric registration check failed on rank %d: recv buffer window (0x%lx) mismatch with rank 0 (0x%lx)", info->opName, size, recvWinMismatchRank, bufInfo[recvWinMismatchRank * 2 + 1].bigOffset, cmpBufInfo[1].bigOffset);
        ret = ncclInvalidArgument;
        goto fail;
      }
      if (recvUserMismatch) {
        if (comm->rank == 0) WARN("Coll %s size %ld symmetric registration check failed on rank %d: recv buffer user offset (0x%lx) mismatch with rank 0 (0x%lx)", info->opName, size, recvUserMismatchRank, bufInfo[recvUserMismatchRank * 2 + 1].userOffset, cmpBufInfo[1].userOffset);
        ret = ncclInvalidArgument;
        goto fail;
      }
    }
  }

exit:
  free(bufInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclArgsGlobalCheck(struct ncclArgsInfo* argsInfo) {
  struct ncclInfo* info = &argsInfo->info;
  if (info->coll != ncclFuncSend && info->coll != ncclFuncRecv && info->coll != ncclFuncPutSignal && info->coll != ncclFuncSignal && info->coll != ncclFuncWaitSignal) { // exclude one-sided and sendrecv operations
    // Check registration globally
    NCCLCHECK(registrationCheck(info));
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

  // ncclMaxRedOp < info->op will always be false due to the sizes of
  // the datatypes involved, and that's by design.  We keep the check though
  // just as a reminder.
  // coverity[result_independent_of_operands]
  if (info->op < 0 || ncclMaxRedOp < info->op) {
    WARN("%s : invalid reduction operation %d", info->opName, info->op);
    return ncclInvalidArgument;
  }
  int opIx = int(ncclUserRedOpMangle(info->comm, info->op)) - int(ncclNumOps);
  if (ncclNumOps <= info->op &&
      (info->comm->userRedOpCapacity <= opIx || info->comm->userRedOps[opIx].freeNext != -1)) {
    WARN("%s : reduction operation %d unknown to this communicator", info->opName, info->op);
    return ncclInvalidArgument;
  }

  if (info->comm->checkMode != ncclCheckModeDefault) {
    if ((info->coll == ncclFuncSend || info->coll == ncclFuncRecv)) {
      if (info->count >0)
        NCCLCHECK(CudaPtrCheck(info->recvbuff, info->comm, "buff", info->opName));
    } else {
      // Check CUDA device pointers
      if (info->coll != ncclFuncBroadcast || info->comm->rank == info->root) {
        NCCLCHECK(CudaPtrCheck(info->sendbuff, info->comm, "sendbuff", info->opName));
      }
      if (info->coll != ncclFuncReduce || info->comm->rank == info->root) {
        NCCLCHECK(CudaPtrCheck(info->recvbuff, info->comm, "recvbuff", info->opName));
      }
    }

    if (info->comm->checkMode == ncclCheckModeDebugGlobal) {
      struct ncclArgsInfo* argsInfo;
      NCCLCHECK(ncclCalloc(&argsInfo, 1));
      argsInfo->info = *info;
      argsInfo->next = NULL;
      ncclIntruQueueEnqueue(&info->comm->argsInfoQueue, argsInfo);
    }
  }

  return ncclSuccess;
}
