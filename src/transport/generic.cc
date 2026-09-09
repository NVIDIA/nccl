/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "transport.h"
#include "bootstrap.h"

NCCL_PARAM(MultiSegmentRegister, "MULTI_SEGMENT_REGISTER", 1);

ncclResult_t ncclTransportRingConnect(struct ncclComm* comm) {
  struct ringConnInfo {
    bool useNetPXN;
    bool useGdr;
  };
  struct ringConnInfo* ringInfo = NULL;
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    comm->useGdr = true;
    comm->useNetPXN = false;
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->ring.prev, 1, &channel->ring.next, 0), ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_RING], 0), ret, fail);
    if (ncclParamLocalRegister() || ncclParamGraphRegister()) {
      NCCLCHECK(ncclCalloc(&ringInfo, comm->nRanks));
      ringInfo[comm->rank].useGdr = comm->useGdr;
      ringInfo[comm->rank].useNetPXN = comm->useNetPXN;
      NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, ringInfo, sizeof(struct ringConnInfo)), ret, fail);
      for (int i = 0; i < comm->nRanks; ++i) {
        if (!ringInfo[i].useGdr) comm->useGdr = false;
        if (ringInfo[i].useNetPXN) comm->useNetPXN = true;
        if (comm->useGdr == false && comm->useNetPXN == true) break;
      }
    }
    INFO(NCCL_INIT, "Connected all rings, use ring PXN %d GDR %d", comm->useNetPXN, comm->useGdr);
  }
exit:
  free(ringInfo);
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclTransportTreeConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    // Connect Trees
    for (int c = 0; c < comm->nChannels; c++) {
      struct ncclChannel* channel = comm->channels + c;
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, NCCL_MAX_TREE_ARITY, channel->tree.down, 1, &channel->tree.up, 0),
                    ret, fail);
      NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &channel->tree.up, NCCL_MAX_TREE_ARITY, channel->tree.down, 0),
                    ret, fail);
    }
    NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    INFO(NCCL_INIT, "Connected all trees");
  }
exit:
  return ret;
fail:
  goto exit;
}

// Build a node-major rank order with the supplied transport heads first.
ncclResult_t ncclTransportInitRankMap(struct ncclComm* comm, int nHeads, const int* heads) {
  ncclResult_t ret = ncclSuccess;
  if (comm->denseToUserRank != nullptr) return ncclSuccess;

  int rank = comm->rank;
  int* userToDenseRank = NULL;

  NCCLCHECKGOTO(ncclCalloc(&userToDenseRank, comm->nRanks), ret, exit);
  userToDenseRank[rank] = -1;

  for (int h = 0; h < nHeads; h++) {
    if (heads[h] == rank) userToDenseRank[rank] = h;
  }
  if (userToDenseRank[rank] == -1) {
    int denseLocalRank = nHeads;
    for (int localRank = 0; localRank < comm->localRank; localRank++) {
      bool isHead = false;
      for (int h = 0; h < nHeads; h++) isHead |= comm->rankToLocalRank[heads[h]] == localRank;
      if (!isHead) denseLocalRank++;
    }
    userToDenseRank[rank] = denseLocalRank;
  }
  userToDenseRank[rank] += comm->node * comm->localRanks;

  NCCLCHECKGOTO(bootstrapAllGather(comm->bootstrap, userToDenseRank, sizeof(int)), ret, exit);
  comm->denseToUserRank = ncclMemoryStackAlloc<int>(&comm->memPermanent, comm->nRanks);
  for (int r = 0; r < comm->nRanks; r++) {
    comm->denseToUserRank[userToDenseRank[r]] = r;
  }
exit:
  free(userToDenseRank);
  return ret;
}

ncclResult_t ncclTransportPatConnect(struct ncclComm* comm) {
  ncclResult_t ret = ncclSuccess;
  if (comm && comm->nRanks > 1) {
    int denseLocalRank = 0;
    // Connect corresponding NVLS-dense rails across nodes.
    if (comm->localRanks > 1) {
      if (!comm->nvlsSupport || comm->channels[0].nvls.nHeads != comm->localRanks ||
          comm->channels[0].nvls.headRank < 0 || comm->channels[0].nvls.headRank >= comm->localRanks) {
        goto exit;
      }
      denseLocalRank = comm->channels[0].nvls.headRank;
    }

    for (int mask = 1; mask < comm->nNodes; mask <<= 1) {
      int prevNode = (comm->node + mask) % comm->nNodes;
      int numLocalRanks = comm->localRanks;
      int nextNode = (comm->node + comm->nNodes - mask) % comm->nNodes;
      int prevPeer = prevNode;
      int nextPeer = nextNode;
      if (comm->localRanks > 1) {
        prevPeer = comm->denseToUserRank[prevNode * numLocalRanks + denseLocalRank];
        nextPeer = comm->denseToUserRank[nextNode * numLocalRanks + denseLocalRank];
      }
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &prevPeer, 1, &nextPeer, 0), ret, fail); // ReduceScatter
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
      for (int c = 0; c < comm->nChannels; c++) {
        NCCLCHECKGOTO(ncclTransportP2pConnect(comm, c, 1, &nextPeer, 1, &prevPeer, 0), ret, fail); // AllGather
      }
      NCCLCHECKGOTO(ncclTransportP2pSetup(comm, &comm->graphs[NCCL_ALGO_TREE], 0), ret, fail);
    }
    INFO(NCCL_INIT, "Connected binomial trees");
  }
exit:
  return ret;
fail:
  goto exit;
}
