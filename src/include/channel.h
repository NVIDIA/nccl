/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CHANNEL_H_
#define NCCL_CHANNEL_H_
#include "comm.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelId, cudaStream_t stream);
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share, cudaStream_t stream);
ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share, cudaStream_t stream);
ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks);

inline ncclResult_t ncclChannelComputeBase(struct ncclComm* comm, int peer, bool isSendNotRecv, int*channelBase) {
  int p2pGroupSize = NCCL_MAX_P2P_FIFO_CONNS_PER_WORK_BATCH;
  int peerNode = comm->rankToNode[peer];
  int peerIndex = comm->rankToLocalRank[peer];
  int nsteps = comm->maxLocalRanks;
  int rankIndex = comm->rankToLocalRank[comm->rank];
  int step, delta;
  if (isSendNotRecv) {
    step = (nsteps + peerIndex - rankIndex)%nsteps;
    delta = (comm->nNodes + peerNode - comm->node) % comm->nNodes;
  } else {
    step = (nsteps + rankIndex - peerIndex)%nsteps;
    delta = (comm->nNodes + comm->node - peerNode) % comm->nNodes;
  }
  *channelBase = comm->nNodes > 1 ? delta+(step/p2pGroupSize) : step;
  return ncclSuccess;
}

inline ncclResult_t ncclChannelComputeFromBase(struct ncclComm* comm, int base, int channelInc, int*channelId) {
  //*channelId = (base+comm->p2pChannels[channelInc]) % comm->p2pnChannels;
  *channelId = (comm->p2pChannels[base%comm->p2pnChannels]+channelInc) % comm->p2pnChannels;
  return ncclSuccess;
}

inline ncclResult_t ncclChannelCompute(struct ncclComm* comm, int peer, int channelInc, bool isSendNotRecv, int*channelId) {
  int base;
  NCCLCHECK(ncclChannelComputeBase(comm, peer, isSendNotRecv, &base));
  NCCLCHECK(ncclChannelComputeFromBase(comm, base, channelInc, channelId));
  return ncclSuccess;
}

#endif
