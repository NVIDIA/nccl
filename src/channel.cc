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

int ncclClampChannels(int nChannels, int minChannels, int maxChannels) {
  return std::min(std::max(nChannels, minChannels), maxChannels);
}

int ncclFuncTrafficPerByte(ncclFunc_t func, int nRanks) {
  switch (func) {
  case ncclFuncAllReduce:
    return 2;
  case ncclFuncAllGather:
  case ncclFuncAlltoAll:
  case ncclFuncReduceScatter:
    return nRanks;
  default:
    return 1;
  }
}

size_t ncclCollTrafficPerChannel(size_t trafficBytes, int nChannels) {
  return nChannels > 0 ? DIVUP(trafficBytes / nChannels, 16) * 16 : 0;
}

struct ncclCollChannelLayout ncclComputeCollChannelLayout(size_t count, size_t elementSize, int trafficPerByte,
                                                          size_t trafficPerChannel, size_t currentTraffic,
                                                          int channelId, int maxChannels) {
  struct ncclCollChannelLayout layout = {};
  if (count == 0 || maxChannels <= channelId) return layout;

  size_t cellSize = DIVUP(DIVUP(ncclMinTrafficPerChannel, (size_t)trafficPerByte), 16) * 16;
  layout.elementsPerCell = cellSize / elementSize;
  size_t cells = DIVUP(count * elementSize, cellSize);
  size_t trafficPerCell = cellSize * trafficPerByte;
  size_t cellsPerChannel = std::min(cells, DIVUP(trafficPerChannel, trafficPerCell));

  if (channelId + 1 == maxChannels) {
    layout.cellsLo = cells;
  } else {
    layout.cellsLo = std::min(cells, DIVUP(trafficPerChannel - currentTraffic, trafficPerCell));
  }
  layout.nMidChannels = (cells - layout.cellsLo) / cellsPerChannel;
  layout.cellsHi = (cells - layout.cellsLo) % cellsPerChannel;
  int nChannels = (layout.cellsLo != 0 ? 1 : 0) + layout.nMidChannels + (layout.cellsHi != 0 ? 1 : 0);
  if (maxChannels < channelId + nChannels) {
    layout.nMidChannels = maxChannels - channelId - 2;
    cellsPerChannel = (cells - layout.cellsLo) / (layout.nMidChannels + 1);
    layout.cellsHi = cellsPerChannel + (cells - layout.cellsLo) % (layout.nMidChannels + 1);
  }
  if (layout.cellsHi == 0 && layout.nMidChannels != 0) {
    layout.cellsHi = cellsPerChannel;
    layout.nMidChannels -= 1;
  }
  if (layout.cellsLo == 0) {
    layout.channelOffset = 1;
    if (layout.nMidChannels == 0) {
      layout.cellsLo = layout.cellsHi;
      layout.cellsHi = 0;
    } else {
      layout.cellsLo = cellsPerChannel;
      layout.nMidChannels -= 1;
    }
  }

  layout.countMid = layout.nMidChannels != 0 ? cellsPerChannel * layout.elementsPerCell : 0;
  layout.countLo = layout.cellsLo * layout.elementsPerCell;
  layout.countHi = layout.cellsHi * layout.elementsPerCell;
  (layout.countHi != 0 ? layout.countHi : layout.countLo) -= cells * layout.elementsPerCell - count;
  layout.nChannels = (layout.countLo != 0 ? 1 : 0) + layout.nMidChannels + (layout.cellsHi != 0 ? 1 : 0);
  return layout;
}

// One channel's slice of comm->channelMetadata: devPeers, then devRingUserRanks.
static void channelMetadataLayout(int nRanks, int nvlsRanks, size_t* ringOffset, size_t* stride) {
  size_t offset = (nRanks + 1 /* Collnet */ + nvlsRanks /* NVLS */) * sizeof(struct ncclDevChannelPeer*);
  ALIGN_SIZE(offset, alignof(int));
  size_t size = offset + nRanks * sizeof(int);
  ALIGN_SIZE(size, alignof(struct ncclDevChannelPeer*));
  *ringOffset = offset;
  *stride = size;
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
    // Pack the small per-channel arrays across channels. Channel counts are final here.
    int nSlabChannels = std::max(comm->nChannels, std::max(comm->p2pnChannels, comm->nvlsChannels));
    if (sharedRes->devPeers[channelId] == NULL) {
      if (sharedRes->owner == comm && sharedRes->devPeersSlab == NULL) {
        NCCLCHECK(ncclCudaCallocAsync(&sharedRes->devPeersSlab, nSlabChannels * sharedRes->tpNRanks, deviceStream,
                                      comm->memManager, ncclMemOffload));
        sharedRes->devPeersSlabChannels = nSlabChannels;
      }
      if (channelId < sharedRes->devPeersSlabChannels) {
        sharedRes->devPeers[channelId] = sharedRes->devPeersSlab + channelId * sharedRes->tpNRanks;
      } else {
        // A sharing comm can use more channels than the owner.
        NCCLCHECK(ncclCudaCallocAsync(sharedRes->devPeers + channelId, sharedRes->tpNRanks, deviceStream,
                                      comm->memManager, ncclMemOffload));
      }
    }
    /* channel metadata is not shared, so just free it when calling commFree() */
    size_t ringOffset, stride;
    channelMetadataLayout(nRanks, nvlsRanks, &ringOffset, &stride);
    if (comm->channelMetadata == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(&comm->channelMetadata, nSlabChannels * stride, deviceStream, comm->memManager,
                                    ncclMemOffload));
      ncclCommPushCudaFree(comm, comm->channelMetadata);
      comm->channelMetadataChannels = nSlabChannels;
    }
    if (channelId >= comm->channelMetadataChannels) {
      WARN("Channel %d is outside the %d channels of the channel metadata", channelId, comm->channelMetadataChannels);
      return ncclInternalError;
    }
    char* slice = comm->channelMetadata + channelId * stride;
    channel->devPeers = reinterpret_cast<struct ncclDevChannelPeer**>(slice);
    channel->devRingUserRanks = reinterpret_cast<int*>(slice + ringOffset);
    NCCLCHECK(ncclCalloc(&channel->devPeersHostPtr, nPeers));
    for (int r = 0; r < nRanks; r++) {
      uintptr_t addr = (uintptr_t)(comm->sharedRes->devPeers[channelId] + comm->topParentRanks[r]);
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + r), (uintptr_t*)&addr, 1, deviceStream));
      channel->devPeersHostPtr[r] = (struct ncclDevChannelPeer*)addr;
    }
  }

  channel->ring.userRanks = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);
  channel->ring.rankToIndex = ncclMemoryStackAlloc<int>(&comm->memPermanent, nRanks);

  /* guarantee addr has been copied into channel->devPeers */
  NCCLCHECK(ncclStrongStreamRelease(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false));
  NCCLCHECK(ncclStrongStreamSynchronize(&sharedRes->deviceStream));
  return ncclSuccess;
}

// Set up the channel's NVLS peers, slots nRanks+1 .. nRanks+localRanks. With parentSlots NULL the channel
// allocates its own peers; otherwise it reuses the parent's, with slot r backed by the parent's slot
// parentSlots[r]. Slot h < nHeads is NVLS head h, so a child maps each head to the parent slot of the same GPU.
ncclResult_t initNvlsChannel(struct ncclComm* comm, int channelId, struct ncclComm* parent, const int* parentSlots) {
  struct ncclChannel* channel = &comm->channels[channelId];
  struct ncclSharedResources* sharedRes = comm->sharedRes;
  cudaStream_t deviceStream;

  if (channel->nvlsPeers != NULL) return ncclSuccess;

  if (channel->id == -1) NCCLCHECK(initChannel(comm, channelId));

  NCCLCHECK(ncclStrongStreamAcquire(ncclCudaGraphNone(comm->config.graphUsageMode), &sharedRes->deviceStream,
                                    /*concurrent=*/false, &deviceStream));

  int nvlsRanks = comm->localRanks;

  if (parentSlots != NULL) {
    channel->nvlsPeers = parent->channels[channelId].nvlsPeers;
    channel->nvlsDevPeers = parent->channels[channelId].nvlsDevPeers;
    for (int r = 0; r < nvlsRanks; ++r) {
      // Index of parent slot parentSlots[r] in the resource owner's peer arrays. A parent that shares in turn may
      // hold that slot at a remapped index, which the pointer difference follows.
      int tr = (int)(parent->channels[channelId].peers[parent->nRanks + 1 + parentSlots[r]] -
                     parent->channels[channelId].nvlsPeers);
      uintptr_t addr = (uintptr_t)(parent->channels[channelId].nvlsDevPeers + tr);
      channel->peers[comm->nRanks + 1 + r] = parent->channels[channelId].nvlsPeers + tr;
      NCCLCHECK(ncclCudaMemcpyAsync((uintptr_t*)(channel->devPeers + comm->nRanks + 1 + r), (uintptr_t*)&addr, 1,
                                    deviceStream));
      channel->devPeersHostPtr[comm->nRanks + 1 + r] = (struct ncclDevChannelPeer*)addr;
      ncclAtomicRefCountIncrement(&parent->channels[channelId].nvlsPeers[tr].refCount);
    }
  } else {
    // One allocation for all NVLS channels, freed with nvlsResources.
    struct ncclNvlsSharedRes* resources = comm->nvlsResources;
    if (resources->devPeersSlab == NULL) {
      NCCLCHECK(ncclCudaCallocAsync(&resources->devPeersSlab, resources->nChannels * nvlsRanks, deviceStream,
                                    comm->memManager));
    }
    NCCLCHECK(ncclCalloc(&channel->nvlsPeers, nvlsRanks));
    channel->nvlsDevPeers = resources->devPeersSlab + channelId * nvlsRanks;
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
          // nvlsDevPeers is freed with nvlsResources.
          free(channel->nvlsPeers);
        }
      }
    }
  }

  free(channel->devPeersHostPtr);
  return ncclSuccess;
}
