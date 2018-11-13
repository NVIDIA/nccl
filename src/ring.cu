/*************************************************************************
 * Copyright (c) 2015-2018, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "ring.h"
#include "param.h"

NCCL_PARAM(Buffsize, "BUFFSIZE", DEFAULT_BUFFER_SIZE_BYTES);

ncclResult_t initRing(struct ncclComm* comm, int ringid) {
  struct ncclRing* ring = comm->rings+ringid;
  ring->id = ringid;

  // Setup intermediate buffering
  ring->buffSize = ncclParamBuffsize();

  const int sendSize = ring->devMemSendSize = sizeof(struct ncclSendMem);
  struct ncclSendMem* sendMem;
  NCCLCHECK(ncclCudaCalloc((char**)&sendMem, sendSize));
  ring->devMemSend = sendMem;

  const int recvSize = ring->devMemRecvSize = offsetof(struct ncclRecvMem, buff)+ring->buffSize;
  struct ncclRecvMem* recvMem;
  NCCLCHECK(ncclCudaCalloc((char**)&recvMem, recvSize));
  ring->devMemRecv = recvMem;

  TRACE(NCCL_INIT,"sendMem %p size %d recvMem %p size %d", sendMem, sendSize, recvMem, recvSize);

  // Pre-configure send/recv pointers. Those are the default, they may change later.
  ring->recv.conn.buff = recvMem->buff;
  ring->recv.conn.llBuff = recvMem->llBuff;
  ring->recv.conn.tail = &recvMem->tail;
  ring->recv.conn.opCount = &recvMem->opCount;
  ring->recv.conn.direct = 0;
  ring->send.conn.head = &sendMem->head;
  ring->send.conn.llHead = &sendMem->llHead;
  ring->send.conn.direct = 0;
  ring->send.conn.llStep = 0;
  ring->send.conn.llLastCleaning = 0;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&ring->devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&ring->userRanks, comm->nRanks));

  // Per-ring operation list.
  NCCLCHECK(ncclCudaHostAlloc((void**)&ring->collectives, (void**)&ring->devCollectives, sizeof(struct ncclColl)*NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeRing(struct ncclRing* ring) {
  // Intermediate buffering
  CUDACHECK(cudaFree(ring->devMemSend));
  CUDACHECK(cudaFree(ring->devMemRecv));

  // Index to rank table
  free(ring->userRanks);
  CUDACHECK(cudaFree(ring->devUserRanks));

  // Operation list
  NCCLCHECK(ncclCudaHostFree(ring->collectives));

  // Free transport proxy resources
  if (ring->send.transportResources) NCCLCHECK(ring->send.transport->send.free(ring->send.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->send));
  if (ring->recv.transportResources) NCCLCHECK(ring->recv.transport->recv.free(ring->recv.transportResources));
  NCCLCHECK(transportDestroyProxy(&ring->recv));
  return ncclSuccess;
}
