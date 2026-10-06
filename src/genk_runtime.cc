/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "sym_kernels.h"
#include "alloc.h"
#include "comm.h"
#include "device.h"
#include "nccl_device/core.h"
#include <algorithm>

static ncclResult_t ncclGenkInitRingTopology(struct ncclComm* comm) {
  int const nChannels = comm->symkState.genkComm.nRingChannels;
  struct ncclGenkRing rings[MAXCHANNELS] = {};
  for (int c = 0; c < nChannels; c++) {
    struct ncclRing const& ring = comm->channels[c].ring;
    struct ncclGenkRing& genkRing = rings[c];
    genkRing.prev = ring.prev;
    genkRing.next = ring.next;
    genkRing.index = ring.index;
    genkRing.userRanks = comm->channels[c].devRingUserRanks;
  }
  NCCLCHECK(ncclCudaCalloc(&comm->symkState.genkComm.rings, nChannels, comm->memManager));
  ncclCommPushCudaFree(comm, comm->symkState.genkComm.rings);
  NCCLCHECK(ncclCudaMemcpy(comm->symkState.genkComm.rings, rings, nChannels));
  return ncclSuccess;
}

// Called after ncclDevrInitOnce(), when the LSA peer reachability used by flow processors is final.
static ncclResult_t ncclGenkInitTreeTopology(struct ncclComm* comm) {
  int const nChannels = comm->symkState.genkComm.nTreeChannels;
  struct ncclGenkTree trees[MAXCHANNELS] = {};

  for (int c = 0; c < nChannels; c++) {
    struct ncclTree const& tree = comm->channels[c].tree;
    struct ncclGenkTree& genkTree = trees[c];
    genkTree.upLsa = -1;
    genkTree.upGin = -1;
    genkTree.upSlot = -1;
    genkTree.downLsa = -1;
    genkTree.downLsaSlot = -1;
    genkTree.downGin[0] = -1;
    genkTree.downGin[1] = -1;
    genkTree.downGinSlots[0] = -1;
    genkTree.downGinSlots[1] = -1;

    if (tree.up != -1) {
      genkTree.upSlot = tree.upSlot;
      if (comm->rankToNode[tree.up] == comm->node) genkTree.upLsa = tree.up;
      else genkTree.upGin = tree.up;
    }

    int nDownGin = 0;
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      int const peer = tree.down[i];
      if (peer == -1) continue;
      if (comm->rankToNode[peer] == comm->node) {
        if (genkTree.downLsa != -1) {
          WARN("Symmetric tree channel %d has more than one LSA child", c);
          return ncclInternalError;
        }
        genkTree.downLsa = peer;
        genkTree.downLsaSlot = tree.downSlots[i];
      } else {
        if (nDownGin == 2) {
          WARN("Symmetric tree channel %d has more than two GIN children", c);
          return ncclInternalError;
        }
        genkTree.downGin[nDownGin] = peer;
        genkTree.downGinSlots[nDownGin++] = tree.downSlots[i];
      }
    }
  }
  NCCLCHECK(ncclCudaCalloc(&comm->symkState.genkComm.trees, nChannels, comm->memManager));
  ncclCommPushCudaFree(comm, comm->symkState.genkComm.trees);
  NCCLCHECK(ncclCudaMemcpy(comm->symkState.genkComm.trees, trees, nChannels));
  return ncclSuccess;
}

static bool ncclGenkPeerNeedsGin(struct ncclComm* comm, int peer) {
  return peer != -1 && comm->rankToNode[peer] != comm->node;
}

static bool ncclGenkRingChannelSendsGin(struct ncclComm* comm, int channelId) {
  int const nChannels = comm->symkState.genkComm.nRingChannels;
  if (nChannels == 0) return false;
  return ncclGenkPeerNeedsGin(comm, comm->channels[channelId % nChannels].ring.next);
}

static bool ncclGenkTreeChannelSendsGin(struct ncclComm* comm, int channelId) {
  int const nChannels = comm->symkState.genkComm.nTreeChannels;
  if (nChannels == 0) return false;
  struct ncclTree const& tree = comm->channels[channelId % nChannels].tree;
  if (ncclGenkPeerNeedsGin(comm, tree.up)) return true;
  for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
    if (ncclGenkPeerNeedsGin(comm, tree.down[i])) return true;
  }
  return false;
}

static void ncclGenkAddGinPeer(struct ncclComm* comm, int peer, int* peers, int* nPeers) {
  if (!ncclGenkPeerNeedsGin(comm, peer)) return;
  for (int i = 0; i < *nPeers; i++) {
    if (peers[i] == peer) return;
  }
  peers[(*nPeers)++] = peer;
}

static void ncclGenkGinPeers(struct ncclComm* comm, int* peers, int* nPeers) {
  struct ncclGenkDevComm const* genk = &comm->symkState.genkComm;
  *nPeers = 0;
  for (int c = 0; c < genk->nRingChannels; c++) {
    struct ncclRing const& ring = comm->channels[c].ring;
    ncclGenkAddGinPeer(comm, ring.next, peers, nPeers);
    ncclGenkAddGinPeer(comm, ring.prev, peers, nPeers);
  }
  for (int c = 0; c < genk->nTreeChannels; c++) {
    struct ncclTree const& tree = comm->channels[c].tree;
    ncclGenkAddGinPeer(comm, tree.up, peers, nPeers);
    for (int i = 0; i < NCCL_MAX_TREE_ARITY; i++) {
      ncclGenkAddGinPeer(comm, tree.down[i], peers, nPeers);
    }
  }
}

static ncclResult_t ncclGenkInitDevComm(struct ncclComm* comm, bool enableGin, bool* needGenkDevComm) {
  struct ncclGenkDevComm* genk = &comm->symkState.genkComm;
  // Per-call CTA limits may reduce a launch, but cannot exceed this communicator-wide bound.
  int const nChannels = std::min(comm->nChannels, comm->config.maxCTAs);
  genk->nResourceChannels = nChannels;
  struct ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
  reqs.ginStrongSignalsRequired = enableGin;
  reqs.ginVaSignalsRequired = false;
  int* ginCustomArray = nullptr;
#if !defined(NCCL_OS_WINDOWS)
  int nRingGinBuffers = 0;
  int nTreeGinBuffers = 0;
  genk->ringGinChannelMask = genk->treeGinChannelMask = 0;
  if (enableGin) {
    for (int c = 0; c < nChannels; c++) {
      if (ncclGenkRingChannelSendsGin(comm, c)) nRingGinBuffers++;
      if (ncclGenkTreeChannelSendsGin(comm, c)) nTreeGinBuffers++;
    }
  }

  size_t const ringGinChannelStride = alignUp(ncclGenkRingSimpleChannelSize, (size_t)128);
  size_t const treeGinChannelStride = alignUp(ncclGenkTreeSimpleChannelSize, (size_t)128);
  size_t const ginFlowBufferSize =
    (size_t)nRingGinBuffers * ringGinChannelStride + (size_t)nTreeGinBuffers * treeGinChannelStride;
  struct ncclDevResourceRequirements ginFlowReq = {};
  genk->ginSignal0 = 0;
  if (enableGin) {
    ginFlowReq.bufferSize = alignUp(ginFlowBufferSize, (size_t)128);
    ginFlowReq.bufferAlign = 128;
    ginFlowReq.outBufferHandle = &comm->symkState.genkGinFlow.bufHandle;
    ginFlowReq.ginSignalCount = 2 * nChannels * (ncclGenkRingConnections + ncclGenkTreeConnections);
    ginFlowReq.outGinSignalStart = &comm->symkState.genkGinFlow.signal0;

    int nGinPeers;
    NCCLCHECK(ncclCalloc(&ginCustomArray, comm->nRanks));
    ncclGenkGinPeers(comm, ginCustomArray, &nGinPeers);
    // Fixed LSA-visible resources are prepended below, leaving this
    // rank-dependent local staging at the end of the window.
    ginFlowReq.next = reqs.resourceRequirementsList;
    reqs.resourceRequirementsList = &ginFlowReq;
    // Give each expanded Tree topology group its own context.
    reqs.ginContextCount = DIVUP(genk->nTreeChannels, genk->nTreeSearchChannels);
    reqs.ginConnectionType = NCCL_GIN_CONNECTION_CUSTOM_ARRAY;
    reqs.ginType = NCCL_GIN_TYPE_PROXY;
    reqs.ginCustomArray = ginCustomArray;
    reqs.ginCustomArrayCount = nGinPeers;
  }
#else
  (void)enableGin;
#endif

  struct ncclDevResourceRequirements ringBufferReq = {};
  ringBufferReq.bufferSize = (size_t)nChannels * ncclGenkRingSimpleChannelSize;
  ringBufferReq.bufferAlign = 128;
  ringBufferReq.outBufferHandle = &genk->ringBuffer;
  ringBufferReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &ringBufferReq;

  struct ncclDevResourceRequirements treeBufferReq = {};
  treeBufferReq.bufferSize = (size_t)nChannels * ncclGenkTreeSimpleChannelSize;
  treeBufferReq.bufferAlign = 128;
  treeBufferReq.outBufferHandle = &genk->treeBuffer;
  treeBufferReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &treeBufferReq;

  struct ncclDevResourceRequirements ringLLBufferReq = {};
  ringLLBufferReq.bufferSize = (size_t)nChannels * ncclGenkRingLLChannelWireBytes;
  ringLLBufferReq.bufferAlign = alignof(ncclLLFifoLine);
  ringLLBufferReq.outBufferHandle = &genk->ringLLBuffer;
  ringLLBufferReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &ringLLBufferReq;

  struct ncclDevResourceRequirements treeLLBufferReq = {};
  treeLLBufferReq.bufferSize = (size_t)nChannels * ncclGenkTreeLLChannelWireBytes;
  treeLLBufferReq.bufferAlign = alignof(ncclLLFifoLine);
  treeLLBufferReq.outBufferHandle = &genk->treeLLBuffer;
  treeLLBufferReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &treeLLBufferReq;

  struct ncclDevResourceRequirements ringLL128BufferReq = {};
  ringLL128BufferReq.bufferSize = (size_t)nChannels * ncclGenkRingLL128ChannelWireBytes;
  ringLL128BufferReq.bufferAlign = NCCL_LL128_LINESIZE;
  ringLL128BufferReq.outBufferHandle = &genk->ringLL128Buffer;
  ringLL128BufferReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &ringLL128BufferReq;

  struct ncclDevResourceRequirements treeLL128BufferReq = {};
  treeLL128BufferReq.bufferSize = (size_t)nChannels * ncclGenkTreeLL128ChannelWireBytes;
  treeLL128BufferReq.bufferAlign = NCCL_LL128_LINESIZE;
  treeLL128BufferReq.outBufferHandle = &genk->treeLL128Buffer;
  treeLL128BufferReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &treeLL128BufferReq;

  struct ncclDevResourceRequirements connStateReq = {};
  connStateReq.bufferSize =
    (size_t)nChannels * (ncclGenkRingConnections + ncclGenkTreeConnections) * sizeof(ncclFlowConnState);
  connStateReq.bufferAlign = alignof(ncclFlowConnState);
  connStateReq.outBufferHandle = &genk->connState;
  connStateReq.next = reqs.resourceRequirementsList;
  reqs.resourceRequirementsList = &connStateReq;

  ncclResult_t ret = ncclSuccess;
  if (needGenkDevComm) {
    // Schedule asynchronous creation of the genk devComm.
    NCCLCHECKGOTO(ncclDevrCommCreateAsync(comm, &reqs, &genk->devComm, /*isInternal=*/true,
                                          /*deviceCodeVersion=*/NCCL_VERSION_CODE),
                  ret, exit);
    *needGenkDevComm = true;
  } else {
    // genk devComm can be created immediately.
    NCCLCHECKGOTO(ncclDevrCommCreateInternal(comm, &reqs, &genk->devComm, /*isInternal=*/true,
                                             /*deviceCodeVersion=*/NCCL_VERSION_CODE),
                  ret, exit);
  }
exit:
  free(ginCustomArray);
  return ret;
}

ncclResult_t ncclGenkInitStart(struct ncclComm* comm, bool* needGenkDevComm) {
  // Genk has its own device communicator and does not need Symk resources.
  NCCLCHECK(ncclDevrInitOnce(comm));

  struct ncclSymkState* symk = &comm->symkState;
  if (!symk->genkInitStarted) {
    NCCLCHECK(ncclGenkInitDevComm(comm, /*enableGin=*/comm->nNodes > 1, needGenkDevComm));
    symk->genkInitStarted = true;
  }
  return ncclSuccess;
}

ncclResult_t ncclGenkInitEnd(struct ncclComm* comm) {
  bool enableGin = (comm->nNodes > 1);

  struct ncclSymkState* symk = &comm->symkState;

  if (symk->genkInitStarted && !symk->genkInitialized) {
    struct ncclGenkDevComm* genk = &symk->genkComm;

    genk->workStarted = comm->profiler.symWorkStarted;
    genk->workCompleted = comm->profiler.symWorkCompleted;
    genk->workPhases = comm->profiler.symWorkPhases;

    NCCLCHECK(ncclGenkInitRingTopology(comm));
    NCCLCHECK(ncclGenkInitTreeTopology(comm));

#if !defined(NCCL_OS_WINDOWS)
    if (enableGin) {
      int const nChannels = genk->nResourceChannels;
      genk->ginSignal0 = symk->genkGinFlow.signal0;
      genk->ginFlowBuffer = symk->genkGinFlow.bufHandle;
      // Pack Ring channels followed by Tree channels.
      for (int c = 0; c < nChannels; c++) {
        if (ncclGenkRingChannelSendsGin(comm, c)) {
          genk->ringGinChannelMask |= (1UL << c);
        }
      }
      for (int c = 0; c < nChannels; c++) {
        if (ncclGenkTreeChannelSendsGin(comm, c)) {
          genk->treeGinChannelMask |= (1UL << c);
        }
      }
    }
#endif
    symk->genkInitialized = true;
  }
  return ncclSuccess;
}
