/*************************************************************************
 * Copyright (c) 2015-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "channel.h"
#include "param.h"
#include "gdrwrap.h"

ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  if (channel->id != -1) return ncclSuccess;

  int nRanks = comm->nRanks;
  int nPeers = nRanks + 1 /* Collnet */ + comm->localRanks /* NVLS */;
  channel->id = channelId;
  channel->workFifoSent = 0;

  struct ncclSharedResources* sharedRes = comm->sharedRes;

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  if (channel->peers == NULL) {
    // The extra on nRanks+1 is for collnet root (i.e. network)
    // Allocate everything related to sharedRes with ncclCalloc as this can be
    // shared between communicators hence should not be tied to comm.
    if (sharedRes->peers[channelId] == NULL) {
      NCCLCHECK(ncclCalloc(sharedRes->peers + channelId, sharedRes->tpNRanks));
    }
    channel->peers = ncclMemoryStackAlloc<struct ncclChannelPeer*>(&comm->memPermanent, nPeers);
    for (int r = 0; r < nRanks; r++) {
      channel->peers[r] = comm->sharedRes->peers[channelId] + comm->topParentRanks[r];
      ncclAtomicRefCountIncrement(&channel->peers[r]->refCount);
    }
  }

  if (channel->devPeers == NULL) {
    if (sharedRes->devPeers[channelId] == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, sharedRes->deviceStream.cudaStream));
    }
    /* channel->devPeers is not shared, so just free it when calling commFree() */
    NCCLCHECK(ncclCudaCallocAsync(&channel->devPeers, nPeers, sharedRes->deviceStream.cudaStream));
    ncclCommPushCudaFree(comm, channel->devPeers);
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }

  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, sharedRes->deviceStream.cudaStream));
  ncclCommPushCudaFree(comm, channel->devRingUserRanks);

  /* guarantee addr has been copied into channel->devPeers */
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;

  if (channel->nvlsPeers != NULL)
    return ncclSuccess;

  if (channel->id == -1)
    NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  if (share) {
    channel->nvlsPeers = parent->channels[channelId].nvlsPeers;
    channel->nvlsDevPeers = parent->channels[channelId].nvlsDevPeers;
    for (int r = 0; r < comm->localRanks; ++r) {
      int tr = comm->topParentLocalRanks[r];
      uintptr_t addr = (uintptr_t)(parent->channels[channelId].nvlsDevPeers + tr);
      channel->peers[comm->nRanks + 1 + r] = parent->channels[channelId].nvlsPeers + tr;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&parent->channels[channelId].nvlsPeers[tr].refCount);
    }
  } else {
    NCCLCHECK(ncclCalloc(&channel->nvlsPeers, comm->localRanks));
    NCCLCHECK(ncclCudaCallocAsync(&channel->nvlsDevPeers, comm->localRanks, sharedRes->deviceStream.cudaStream));
    for (int r = 0; r < comm->localRanks; ++r) {
      uintptr_t addr = (uintptr_t)(channel->nvlsDevPeers + r);
      channel->peers[comm->nRanks + 1 + r] = channel->nvlsPeers + r;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&channel->nvlsPeers[r].refCount);
    }
  }

  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  uintptr_t addr;

  if (channel->collnetPeers != NULL)
    return ncclSuccess;

  if (channel->id == -1)
    NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquireUncaptured(&sharedRes->deviceStream));

  if (share) {
    channel->collnetPeers = parent->channels[channelId].collnetPeers;
    channel->collnetDevPeers = parent->channels[channelId].collnetDevPeers;
    addr = (uintptr_t)parent->channels[channelId].collnetDevPeers;
    channel->peers[comm->nRanks] = parent->channels[channelId].collnetPeers;
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    ncclAtomicRefCountIncrement(&parent->channels[channelId].collnetPeers->refCount);
  } else {
    NCCLCHECK(ncclCalloc(&channel->collnetPeers, 1));
    NCCLCHECK(ncclCudaCallocAsync(&channel->collnetDevPeers, 1, sharedRes->deviceStream.cudaStream));
    addr = (uintptr_t)channel->collnetDevPeers;
    channel->peers[comm->nRanks] = channel->collnetPeers;
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, sharedRes->deviceStream.cudaStream));
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    ncclAtomicRefCountIncrement(&channel->collnetPeers->refCount);
  }

  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(), &sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks) {
  int nPeers = nRanks + collnetNRanks + nvlsNRanks;
  /* channel peers are only valid when async init thread completes commAlloc() and
   * the channel is intialized with initChannel(); if either is not done, this channel
   * should never be free. */
  if (channel->id == -1 || channel->peers == NULL) return ncclSuccess;

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r = 0; r < nPeers; r++) {
    struct ncclChannelPeer* peer = channel->peers[r];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b=0; b<NCCL_MAX_CONNS; b++) {
          if (peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(peer->send+b));
          if (peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(peer->recv+b));
        }
        if (r == nRanks) {
          free(channel->collnetPeers);
          ncclCudaFree(channel->collnetDevPeers);
        } else if (r == nPeers - 1) {
          free(channel->nvlsPeers);
          ncclCudaFree(channel->nvlsDevPeers);
        }
      }
    }
  }

  free(channel->devPeersHostPtr);
  return ncclSuccess;
}
