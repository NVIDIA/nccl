/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "channel.h"
#include "param.h"
#include "gdrwrap.h"
#include "transport.h"

ncclResult_t initChannelMetadata(struct ncclComm* comm, int nChannels) {
  if (nChannels == 0) return ncclSuccess;
  if (nChannels < 0 || nChannels > MAXCHANNELS || comm->channelMetadata != nullptr) return ncclInternalError;

  struct ncclSharedResources* sharedRes = comm->sharedRes;
  size_t nPeers = (size_t)comm->nRanks + 1 + comm->localRanks;
  size_t ringOffset = nPeers * sizeof(struct ncclDevChannelPeer*);
  ALIGN_SIZE(ringOffset, alignof(int));
  size_t stride = ringOffset + (size_t)comm->nRanks * sizeof(int);
  ALIGN_SIZE(stride, alignof(struct ncclDevChannelPeer*));
  char* metadata = nullptr;
  struct ncclDevChannelPeer* sharedPeers = nullptr;
  cudaStream_t deviceStream;
  ncclResult_t result = ncclSuccess;

  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false, &deviceStream));
  NCCLCHECKGOTO(ncclCudaCallocAsync(&metadata, stride * nChannels, deviceStream, comm->memManager, ncclMemOffload),
                result, release);
  comm->channelMetadata = metadata;
  comm->channelMetadataStride = stride;
  comm->channelMetadataRingOffset = ringOffset;
  comm->channelMetadataChannels = nChannels;
  ncclCommPushCudaFree(comm, metadata);

  // The peer slab belongs to the shared resource family.
  if (sharedRes->owner == comm && sharedRes->devPeersSlab == nullptr) {
    NCCLCHECKGOTO(ncclCudaCallocAsync(&sharedPeers, (size_t)nChannels * sharedRes->tpNRanks, deviceStream,
                                      comm->memManager, ncclMemOffload),
                  result, release);
    sharedRes->devPeersSlab = sharedPeers;
    sharedRes->devPeersSlabChannels = nChannels;
  }

release:
  NCCLCHECKIGNORE(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                          /*concurrent=*/false),
                  result);
  if (result == ncclSuccess) NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  return result;
}

ncclResult_t initChannel(struct ncclComm* comm, int channelId) {
  struct ncclChannel* channel = &comm->channels[channelId];
  if (channel->id != -1) return ncclSuccess;

  int nRanks = comm->nRanks;
  int nvlsRanks = comm->localRanks;
  int nPeers = nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */;
  channel->id = channelId;
  channel->workFifoProduced = 0;

  struct ncclSharedResources* sharedRes = comm->sharedRes;
  cudaStream_t deviceStream;
  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false, &deviceStream));

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
      if (sharedRes->devPeersSlab != nullptr && channelId < sharedRes->devPeersSlabChannels) {
        sharedRes->devPeers[channelId] = sharedRes->devPeersSlab + (size_t)channelId * sharedRes->tpNRanks;
        sharedRes->devPeersInSlab[channelId] = true;
      } else {
        NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, deviceStream,
                                      comm->memManager, ncclMemOffload));
      }
    }
    if (comm->channelMetadata != nullptr && channelId < comm->channelMetadataChannels) {
      char* slice = comm->channelMetadata + (size_t)channelId * comm->channelMetadataStride;
      channel->devPeers = reinterpret_cast<struct ncclDevChannelPeer**>(slice);
      channel->devRingUserRanks = reinterpret_cast<int*>(slice + comm->channelMetadataRingOffset);
    } else {
      size_t ringOffset = nPeers * sizeof(*channel->devPeers);
      ALIGN_SIZE(ringOffset, alignof(int));
      size_t bytes = ringOffset + nRanks * sizeof(*channel->devRingUserRanks);
      char* channelMetadata = nullptr;
      NCCLCHECK(ncclCudaCallocAsync(&channelMetadata, bytes, deviceStream, comm->memManager, ncclMemOffload));
      ncclCommPushCudaFree(comm, channelMetadata);
      channel->devPeers = reinterpret_cast<struct ncclDevChannelPeer**>(channelMetadata);
      channel->devRingUserRanks = reinterpret_cast<int*>(channelMetadata + ringOffset);
    }
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, deviceStream));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  } else {
    // Keep a separate ring allocation when the peer array already exists.
    NCCLCHECK(ncclCudaCallocAsync(&channel->devRingUserRanks, nRanks, deviceStream, comm->memManager, ncclMemOffload));
    ncclCommPushCudaFree(comm, channel->devRingUserRanks);
  }

  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  channel->ring.rankToIndex = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);

  /* guarantee addr has been copied into channel->devPeers */
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  return ncclSuccess;
}

ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  cudaStream_t deviceStream;

  if (channel->nvlsPeers != NULL) return ncclSuccess;

  if (channel->id == -1) NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false, &deviceStream));

  int nvlsRanks = comm->localRanks;

  if (share) {
    channel->nvlsPeers = parent->channels[channelId].nvlsPeers;
    channel->nvlsDevPeers = parent->channels[channelId].nvlsDevPeers;
    channel->nvlsDevPeersInSlab = parent->channels[channelId].nvlsDevPeersInSlab;
    for (int r = 0; r < nvlsRanks; ++r) {
      int tr = comm->topParentLocalRanks[r];
      uintptr_t addr = (uintptr_t)(parent->channels[channelId].nvlsDevPeers + tr);
      channel->peers[comm->nRanks + 1 + r] = parent->channels[channelId].nvlsPeers + tr;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1,
                                    deviceStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&parent->channels[channelId].nvlsPeers[tr].refCount);
    }
  } else {
    NCCLCHECK(ncclCalloc(&channel->nvlsPeers, nvlsRanks));
    if (comm->nvlsDevPeersSlab != nullptr) {
      channel->nvlsDevPeers = comm->nvlsDevPeersSlab + (size_t)channelId * nvlsRanks;
      channel->nvlsDevPeersInSlab = true;
    } else {
      NCCLCHECK(ncclCudaCallocAsync(&channel->nvlsDevPeers, nvlsRanks, deviceStream, comm->memManager));
    }
    for (int r = 0; r < nvlsRanks; ++r) {
      uintptr_t addr = (uintptr_t)(channel->nvlsDevPeers + r);
      channel->peers[comm->nRanks + 1 + r] = channel->nvlsPeers + r;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1,
                                    deviceStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&channel->nvlsPeers[r].refCount);
    }
  }

  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t initCollnetChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, bool share) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  uintptr_t addr;
  cudaStream_t deviceStream;

  if (channel->collnetPeers != NULL) return ncclSuccess;

  if (channel->id == -1) NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false, &deviceStream));

  if (share) {
    channel->collnetPeers = parent->channels[channelId].collnetPeers;
    channel->collnetDevPeers = parent->channels[channelId].collnetDevPeers;
    addr = (uintptr_t)parent->channels[channelId].collnetDevPeers;
    channel->peers[comm->nRanks] = parent->channels[channelId].collnetPeers;
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, deviceStream));
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    ncclAtomicRefCountIncrement(&parent->channels[channelId].collnetPeers->refCount);
  } else {
    NCCLCHECK(ncclCalloc(&channel->collnetPeers, 1));
    NCCLCHECK(ncclCudaCallocAsync(&channel->collnetDevPeers, 1, deviceStream, comm->memManager));
    addr = (uintptr_t)channel->collnetDevPeers;
    channel->peers[comm->nRanks] = channel->collnetPeers;
    NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks), (uintptr_t*)&addr, 1, deviceStream));
    channel->devPeersHostPtr[comm->nRanks] = (struct ncclDevChannelPeer*)addr;
    ncclAtomicRefCountIncrement(&channel->collnetPeers->refCount);
  }

  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));

  return ncclSuccess;
}

ncclResult_t freeChannel(struct ncclChannel* channel, int nRanks, int collnetNRanks, int nvlsNRanks,
                         struct ncclComm* comm) {
  int nPeers = nRanks + collnetNRanks + nvlsNRanks;
  /* channel peers are only valid when async init thread completes commAlloc() and
   * the channel is initialized with initChannel(); if either is not done, this channel
   * should never be free. */
  if (channel->id == -1 || channel->peers == NULL) return ncclSuccess;

  // Free transport proxy resources
  // Note: free all send resources first due to CollNet arrangement
  for (int r = 0; r < nPeers; r++) {
    struct ncclChannelPeer* peer = channel->peers[r];
    if (peer) {
      if (ncclAtomicRefCountDecrement(&peer->refCount) == 0) {
        for (int b = 0; b < NCCL_MAX_CONNS; b++) {
          if (peer->send[b].transportComm) NCCLCHECK(peer->send[b].transportComm->free(comm, peer->send + b));
          if (peer->recv[b].transportComm) NCCLCHECK(peer->recv[b].transportComm->free(comm, peer->recv + b));
        }
        if (r == nRanks) {
          free(channel->collnetPeers);
          ncclCudaFree(channel->collnetDevPeers, comm->memManager);
        } else if (r == nPeers - 1) {
          free(channel->nvlsPeers);
          if (!channel->nvlsDevPeersInSlab) ncclCudaFree(channel->nvlsDevPeers, comm->memManager);
        }
      }
    }
  }

  free(channel->devPeersHostPtr);
  return ncclSuccess;
}
