/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"
#include "graph.h"

#define DEFAULT_BUFFER_SIZE_BYTES (1LL << 22) /* 4MiB */
#define DEFAULT_BUFFER_SIZE_BYTES_ARM (1LL << 20) /* 1MiB */

NCCL_PARAM(Buffsize, "BUFFSIZE", -2);

ncclResult_t initChannel(struct ncclComm* comm, int channelid) {
  struct ncclChannel* channel = comm->channels+channelid;
  channel->id = channelid;

  // Setup intermediate buffering
  int buffSize = ncclParamBuffsize();
  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(comm->topo, &cpuArch, &cpuVendor, &cpuModel));
  channel->buffSize = buffSize != -2 ? buffSize :
	  cpuArch == NCCL_TOPO_CPU_ARCH_ARM ? DEFAULT_BUFFER_SIZE_BYTES_ARM : DEFAULT_BUFFER_SIZE_BYTES;

  // Ring index to user rank table.
  NCCLCHECK(ncclCudaCalloc(&channel->ring.devUserRanks, comm->nRanks));
  NCCLCHECK(ncclCalloc(&channel->ring.userRanks, comm->nRanks));

  // Communication structures with peers.
  NCCLCHECK(ncclCudaCalloc(&channel->devPeers, comm->nRanks+1)); // The extra one rank is for collnet root (i.e. network)
  NCCLCHECK(ncclCalloc(&channel->peers, comm->nRanks+1));
  for (size_t i=0; i<comm->nRanks+1; ++i) {
    channel->peers[i].send.comm = comm;
    channel->peers[i].recv.comm = comm;
  }

  // Per-channel operation list.
  NCCLCHECK(ncclCudaHostAlloc((void**)&channel->collectives, (void**)&channel->devCollectives, sizeof(struct ncclColl)*NCCL_MAX_OPS));
  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks) {
  // Operation list
  NCCLCHECK(ncclCudaHostFree(channel->collectives));

  // Free Ring index to rank tables
  free(channel->ring.userRanks);
  CUDACHECK(cudaFree(channel->ring.devUserRanks));

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->send.transportResources) NCCLCHECK(peer->send.transportComm->free(peer->send.transportResources));
  }
  for (int r=0; r<nRanks+1; r++) {
    struct ncclPeer* peer = channel->peers+r;
    if (peer->recv.transportResources) NCCLCHECK(peer->recv.transportComm->free(peer->recv.transportResources));
  }

  // Free the peer structures.
  CUDACHECK(cudaFree(channel->devPeers));
  free(channel->peers);

  return ncclSuccess;
}
