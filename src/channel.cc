/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"
#include "gdrwrap.h"

// GDRCOPY support: FIFO_ENABLE when enabled locates a workFifo in CUDA memory
NCCL_PARAM(GdrCopyFifoEnable, "GDRCOPY_FIFO_ENABLE", 1);

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  if (channel->id != -1) return ncclSuccess;
  channel->id = channelid;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Communication structures with peers.
  NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks+1)); // The extra one rank is for collnet root (i.e. network)
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks+1));
  for (size_t i=0; i<comm->nRanks+1; ++i) {
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      channel->peers[i].send[b].comm = comm;
      channel->peers[i].recv[b].comm = comm;
    }
  }

  // Per-channel operation list.
  NCCLCHECK(ncclCudaHostCalloc(&channel->workFifo, NCCL_MAX_OPS));
  if (ncclGdrCopy != NULL && ncclParamGdrCopyFifoEnable() == 1) {
    // GDRCOPY support
    // We allocate a workFifo in GDR mapped CUDA memory
    // But we still allocate the Host workFifo so that we
    // can copy the work elements to CUDA memory on kernel launch
    NCCLCHECK(ncclGdrCudaCalloc(&channel->workFifoGdr, &channel->workFifoDev, NCCL_MAX_OPS, &channel->gdrMemDesc));
  } else {
    // The device workFifo is the Host one
    channel->workFifoDev = channel->workFifo;
  }

  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks) {
  if (channel->id == -1) return ncclSuccess;
  // Operation list
  NCCLCHECK(ncclCudaHostFree(channel->workFifo));
  if (channel->gdrMemDesc) {
    // GDRCOPY support
    NCCLCHECK(ncclGdrCudaFree(channel->gdrMemDesc));
  }

  // Free Ring index to rank tables
  free(channel->ring.userRanks);
  CUDACHECK(cudaFree(channel->ring.devUserRanks));

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      if (peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(peer->send+b));
    }
  }
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    for (int b=0; b<NCCL_MAX_CONNS; b++) {
      if (peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(peer->recv+b));
    }
  }

  // Free the peer structures.
  CUDACHECK(cudaFree(channel->devPeers));
  free(channel->peers);

  return ncclSuccess;
}
