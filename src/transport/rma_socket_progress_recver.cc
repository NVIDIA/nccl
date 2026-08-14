/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "rma_socket.h"

#include <limits.h>
#include <string.h>

static ncclResult_t ncclRmaSocketProxyFindMr(struct ncclRmaSocketProxyCollComm* comm, uint64_t mrId,
                                             struct ncclRmaSocketProxyMrHandle** handle) {
  if (comm == NULL || handle == NULL) return ncclInvalidArgument;
  *handle = comm->mrHead;
  while (*handle != NULL) {
    if ((*handle)->mrId == mrId) return ncclSuccess;
    *handle = (*handle)->next;
  }
  return ncclInvalidUsage;
}

static ncclResult_t ncclRmaSocketProxyResolveDataTarget(struct ncclRmaSocketProxyCollComm* comm, int peer,
                                                        struct ncclRmaSocketProxyPeerReceiver* receiver) {
  struct ncclRmaSocketProxyHeader* controlMsg = &receiver->controlMsg;
  struct ncclRmaSocketProxyMrHandle* handle = NULL;
  ncclResult_t ret = ncclRmaSocketProxyFindMr(comm, controlMsg->mrId, &handle);
  if (ret != ncclSuccess) {
    WARN("RMA/Socket : received data for unknown destination mrId=%lu from rank=%d", controlMsg->mrId, peer);
    return ret;
  }
  NCCLCHECK(ncclRmaSocketProxyValidateRange(handle->size, controlMsg->dstOff, (size_t)controlMsg->size,
                                            "destination MR"));
  receiver->dataPtr = (void*)((uintptr_t)handle->data + controlMsg->dstOff);
  receiver->dataType = handle->type;
  receiver->dataHandle = handle;
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyResolveSignalTarget(struct ncclRmaSocketProxyCollComm* comm, int peer,
                                                          struct ncclRmaSocketProxyPeerReceiver* receiver) {
  struct ncclRmaSocketProxyHeader* controlMsg = &receiver->controlMsg;
  struct ncclRmaSocketProxyMrHandle* handle = NULL;
  ncclResult_t ret = ncclRmaSocketProxyFindMr(comm, controlMsg->signalMrId, &handle);
  if (ret != ncclSuccess) {
    WARN("RMA/Socket : received iPutSignal for unknown signal mrId=%lu from rank=%d", controlMsg->signalMrId, peer);
    return ret;
  }
  if (controlMsg->signalOp != NCCL_NET_SIGNAL_OP_INC && controlMsg->signalOp != NCCL_NET_SIGNAL_OP_ADD) {
    WARN("RMA/Socket : received iPutSignal with unsupported signalOp=%u from rank=%d", controlMsg->signalOp, peer);
    return ncclInvalidUsage;
  }
  if (handle->type != NCCL_PTR_CUDA) {
    WARN("RMA/Socket : host VA signals are not supported");
    return ncclInvalidUsage;
  }
  NCCLCHECK(ncclRmaSocketProxyValidateRange(handle->size, controlMsg->signalOff, sizeof(uint64_t), "signal MR"));
  receiver->signalPtr = (void*)((uintptr_t)handle->data + controlMsg->signalOff);
  receiver->signalType = handle->type;
  receiver->signalDelta = controlMsg->signalOp == NCCL_NET_SIGNAL_OP_INC ? 1 : controlMsg->signalValue;
  receiver->signalHandle = handle;
  receiver->hasSignal = 1;
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyApplySignal(struct ncclRmaSocketProxyPeerReceiver* receiver) {
  if (!receiver->hasSignal) return ncclSuccess;
  if (receiver->signalType != NCCL_PTR_CUDA || receiver->signalHandle == NULL) {
    WARN("RMA/Socket : invalid iPutSignal target type=%d handle=%p", receiver->signalType, receiver->signalHandle);
    return ncclInvalidArgument;
  }

  void* signalBarPtr = NULL;
  NCCLCHECK(ncclRmaSocketProxyGdrPtr(receiver->signalHandle, receiver->signalPtr, sizeof(uint64_t), &signalBarPtr));
  if (((uintptr_t)signalBarPtr % sizeof(uint64_t)) != 0) {
    WARN("RMA/Socket : invalid GDRCopy signal mapping %p", signalBarPtr);
    return ncclInternalError;
  }
  uint64_t signalValue;
  NCCLCHECK(wrap_gdr_copy_from_mapping(receiver->signalHandle->gdr.mh, &signalValue, signalBarPtr,
                                       sizeof(signalValue)));
  signalValue += receiver->signalDelta;
  NCCLCHECK(wrap_gdr_copy_to_mapping(receiver->signalHandle->gdr.mh, signalBarPtr, &signalValue, sizeof(signalValue)));
  /* gdr_copy_to_mapping() already fences its write-combined stores. */
  // wc_store_fence();
  return ncclSuccess;
}

static void ncclRmaSocketProxyResetRecv(struct ncclRmaSocketProxyPeerReceiver* receiver) {
  receiver->recvState = ncclRmaSocketProxyRecvStateControlMsg;
  memset(&receiver->controlMsg, 0, sizeof(receiver->controlMsg));
  receiver->controlMsgOffset = 0;
  receiver->recvPayloadOffset = 0;
  receiver->recvChunkSize = 0;
  receiver->recvChunkOffset = 0;
  receiver->dataPtr = NULL;
  receiver->dataType = 0;
  receiver->dataHandle = NULL;
  receiver->signalPtr = NULL;
  receiver->signalType = 0;
  receiver->signalDelta = 0;
  receiver->hasSignal = 0;
  receiver->signalHandle = NULL;
}

static ncclResult_t ncclRmaSocketProxyCompleteRecvPayload(struct ncclRmaSocketProxyCollComm* comm, int peer) {
  struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
  if (receiver->controlMsg.type != ncclRmaSocketProxyMsgTypeIput) {
    WARN("RMA/Socket : completed payload for invalid message type=%u from peer=%d", receiver->controlMsg.type, peer);
    return ncclInternalError;
  }
  /* Each gdr_copy_to_mapping() call already fences its write-combined stores. */
  // if (receiver->controlMsg.size > 0 && receiver->dataType == NCCL_PTR_CUDA) wc_store_fence();
  NCCLCHECK(ncclRmaSocketProxyApplySignal(receiver));
  ncclRmaSocketProxyResetRecv(receiver);
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyStartRecvPut(struct ncclRmaSocketProxyCollComm* comm, int peer) {
  struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
  struct ncclRmaSocketProxyHeader* controlMsg = &receiver->controlMsg;
  if (controlMsg->size > INT_MAX) {
    WARN("RMA/Socket : incoming iPut size %lu exceeds socket API limit", controlMsg->size);
    return ncclRemoteError;
  }
  if (controlMsg->status != ncclSuccess || (controlMsg->opFlags & ~NCCL_RMA_SOCKET_OP_HAS_SIGNAL) != 0) {
    WARN("RMA/Socket : incoming iPut has invalid status=%u opFlags=0x%x size=%lu from peer=%d", controlMsg->status,
         controlMsg->opFlags, controlMsg->size, peer);
    return ncclRemoteError;
  }

  if (controlMsg->size > 0) {
    NCCLCHECK(ncclRmaSocketProxyResolveDataTarget(comm, peer, receiver));
  }
  if (controlMsg->opFlags & NCCL_RMA_SOCKET_OP_HAS_SIGNAL) {
    NCCLCHECK(ncclRmaSocketProxyResolveSignalTarget(comm, peer, receiver));
  }
  if (controlMsg->size > 0 && receiver->buffer == NULL) {
    WARN("RMA/Socket : missing receive staging buffer for peer=%d", peer);
    return ncclInternalError;
  }

  receiver->recvState = ncclRmaSocketProxyRecvStatePayload;
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyProgressRecvControl(struct ncclRmaSocketProxyCollComm* comm, int peer) {
  struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
  int closed = 0;
  NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &receiver->sock, &receiver->controlMsg, sizeof(receiver->controlMsg),
                               &receiver->controlMsgOffset, &closed));
  if (closed) {
    if (receiver->controlMsgOffset == 0) {
      receiver->recvState = ncclRmaSocketProxyRecvStateClosed;
      return ncclSuccess;
    }
    WARN("RMA/Socket : receive socket from peer=%d closed after %d/%zu control message bytes", peer,
         receiver->controlMsgOffset, sizeof(receiver->controlMsg));
    return ncclRemoteError;
  }
  if (receiver->controlMsgOffset < (int)sizeof(receiver->controlMsg)) return ncclInProgress;
  return ncclSuccess;
}

static ncclResult_t ncclRmaSocketProxyProgressRecvPayload(struct ncclRmaSocketProxyCollComm* comm, int peer) {
  struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
  size_t payloadSize = (size_t)receiver->controlMsg.size;
  if (receiver->recvPayloadOffset < payloadSize) {
    int closed = 0;
    if (receiver->recvChunkSize == 0) {
      size_t remaining = payloadSize - receiver->recvPayloadOffset;
      receiver->recvChunkSize = (int)(remaining < NCCL_RMA_SOCKET_CHUNK_SIZE ? remaining : NCCL_RMA_SOCKET_CHUNK_SIZE);
      receiver->recvChunkOffset = 0;
    }
    NCCLCHECK(ncclSocketProgress(NCCL_SOCKET_RECV, &receiver->sock, receiver->buffer, receiver->recvChunkSize,
                                 &receiver->recvChunkOffset, &closed));
    if (closed) {
      WARN("RMA/Socket : receive socket from peer=%d closed while receiving payload", peer);
      return ncclRemoteError;
    }
    if (receiver->recvChunkOffset < receiver->recvChunkSize) return ncclSuccess;
    if (receiver->buffer == NULL || receiver->dataPtr == NULL) {
      WARN("RMA/Socket : cannot apply incoming data chunk from peer=%d", peer);
      return ncclInternalError;
    }
    NCCLCHECK(ncclRmaSocketProxyCopy((char*)receiver->dataPtr + receiver->recvPayloadOffset, receiver->dataType,
                                     receiver->dataHandle, receiver->buffer, NCCL_PTR_HOST, NULL,
                                     receiver->recvChunkSize));
    receiver->recvPayloadOffset += receiver->recvChunkSize;
    receiver->recvChunkSize = 0;
    receiver->recvChunkOffset = 0;

    if (receiver->recvPayloadOffset < payloadSize) return ncclSuccess;
  }
  return ncclRmaSocketProxyCompleteRecvPayload(comm, peer);
}

ncclResult_t ncclRmaSocketProxyProgressPeerRecv(struct ncclRmaSocketProxyCollComm* comm, int peer) {
  ncclResult_t ret = ncclSuccess;
  struct ncclRmaSocketProxyPeerReceiver* receiver = &comm->peerReceiver[peer];
  if (receiver->recvState == ncclRmaSocketProxyRecvStateControlMsg) {
    ret = ncclRmaSocketProxyProgressRecvControl(comm, peer);
    if (ret == ncclInProgress) return ncclSuccess;
    if (ret != ncclSuccess) goto fail;
    if (receiver->recvState == ncclRmaSocketProxyRecvStateClosed) return ncclSuccess;
    switch (receiver->controlMsg.type) {
    case ncclRmaSocketProxyMsgTypeIput:
      NCCLCHECKGOTO(ncclRmaSocketProxyStartRecvPut(comm, peer), ret, fail);
      break;
    default:
      WARN("RMA/Socket : unsupported message type=%u from peer=%d", receiver->controlMsg.type, peer);
      ret = ncclRemoteError;
      goto fail;
    }
  }
  if (receiver->recvState == ncclRmaSocketProxyRecvStatePayload) {
    NCCLCHECKGOTO(ncclRmaSocketProxyProgressRecvPayload(comm, peer), ret, fail);
  }
  return ncclSuccess;
fail:
  ncclRmaSocketProxyResetRecv(receiver);
  return ret;
}
