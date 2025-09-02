/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "register.h"
#include "transport.h"

ncclResult_t ncclRegisterP2pNetBuffer(struct ncclComm* comm, void* userbuff, size_t size, struct ncclConnector* conn, int* regFlag, void** handle, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  ncclResult_t ret = ncclSuccess;

  *regFlag = 0;
  if (comm->netDeviceType != NCCL_NET_DEVICE_UNPACK) {
    if (comm->planner.persistent && ncclParamGraphRegister()) {
      ncclNetGraphRegisterBuffer(comm, userbuff, size, &conn, 1, regFlag, handle, cleanupQueue, NULL);
    }
    if (*regFlag == 0 && ncclParamLocalRegister()) {
      ncclNetLocalRegisterBuffer(comm, userbuff, size, &conn, 1, regFlag, handle);
    }
  }
  return ret;
}

ncclResult_t ncclRegisterP2pIpcBuffer(struct ncclComm* comm, void* userbuff, size_t size, int peerRank, int* regFlag, void** regAddr, struct ncclIntruQueue<struct ncclCommCallback, &ncclCommCallback::next>* cleanupQueue) {
  ncclResult_t ret = ncclSuccess;
  uintptr_t offset = 0;
  uintptr_t* peerRmtAddrs = NULL;

  *regFlag = 0;
  if (comm->planner.persistent && ncclParamGraphRegister()) {
    ncclIpcGraphRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs, reinterpret_cast<void*>(cleanupQueue), NULL);
  }
  if (*regFlag == 0 && ncclParamLocalRegister()) {
    ncclIpcLocalRegisterBuffer(comm, userbuff, size, &peerRank, 1, NCCL_IPC_SENDRECV, regFlag, &offset, &peerRmtAddrs);
  }

  if (*regFlag)
    *regAddr = (void*)((uintptr_t)peerRmtAddrs + offset);
  return ret;
}
