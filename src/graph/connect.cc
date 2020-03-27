/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "trees.h"
#include "rings.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph, struct ncclTopoGraph* collNetGraph,
    struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nChannels = comm->nChannels;

  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->treeUp.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->treeUp.down[i] = -1;
    channel->treeDn.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->treeDn.down[i] = -1;
    channel->collTreeUp.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTreeUp.down[i] = -1;
    channel->collTreeDn.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collTreeDn.down[i] = -1;

    int* ringIntra = ringGraph->intra+c*localRanks;
    int* treeIntra = treeGraph->intra+c*localRanks;
    int* collNetIntra = collNetGraph->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        topoRanks->ringRecv[c] = ringIntra[0];
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        channel->ring.prev = (i == 0) ? -1 : ringIntra[i-1];
        channel->ring.next = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
      if (treeIntra[i] == rank) {
        int recvIndex = 0, sendIndex = treeGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        // Tree loop always flows in the same direction. Other trees are symmetric, i.e.
        // up/down go in reverse directions
        int sym = treeGraph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP ? 0 : 1;

        // Down tree is common
        topoRanks->treeDnRecv[c] = treeIntra[recvIndex];
        topoRanks->treeDnSend[c] = treeIntra[sendIndex];
        channel->treeDn.up       = treeIntra[prev];
        channel->treeDn.down[0]  = treeIntra[next];
        // Up tree depends on the pattern
        topoRanks->treeUpRecv[c] = sym ? topoRanks->treeDnSend[c] : topoRanks->treeDnRecv[c];
        topoRanks->treeUpSend[c] = sym ? topoRanks->treeDnRecv[c] : topoRanks->treeDnSend[c];
        channel->treeUp.down[0]  = sym ? channel->treeDn.down[0]  : channel->treeDn.up ;
        channel->treeUp.up       = sym ? channel->treeDn.up       : channel->treeDn.down[0];
      }
      if (collNetIntra[i] == rank) {
        int prev = (i-1+localRanks)%localRanks, next = (i+1)%localRanks;

        // CollTrees are always symmetric, i.e.
        // up/down go in reverse directions
        channel->collTreeDn.up      = collNetIntra[prev];
        channel->collTreeDn.down[0] = collNetIntra[next];
        channel->collTreeUp.down[0] = channel->collTreeDn.down[0];
        channel->collTreeUp.up      = channel->collTreeDn.up;
      }
    }
    topoRanks->ringPrev[c] = channel->ring.prev;
    topoRanks->ringNext[c] = channel->ring.next;
  }
  // Duplicate channels rings/trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0+nChannels;
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));
  return ncclSuccess;
}

static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext, int* firstRanks) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nRanks;
    int* send = ringSend+c*comm->nRanks;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[firstRanks[n]];
      int prevSendRank = send[firstRanks[(n-1+nNodes)%nNodes]];
      prev[recvRank] = prevSendRank;
      if (comm->rank == recvRank) {
        channel0->ring.prev = prevSendRank;
        channel1->ring.prev = prevSendRank;
      }
      int sendRank = send[firstRanks[n]];
      int nextRecvRank = recv[firstRanks[(n+1)%nNodes]];
      next[sendRank] = nextRecvRank;
      if (comm->rank == sendRank) {
        channel0->ring.next = nextRecvRank;
        channel1->ring.next = nextRecvRank;
      }
    }
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c, channel0->ring.prev, comm->rank, channel0->ring.next);
    TRACE(NCCL_GRAPH, "Ring %d : %d -> %d -> %d", c+nChannels, channel1->ring.prev, comm->rank, channel1->ring.next);
  }
  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes, int* firstRanks) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[firstRanks[n]];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree0, struct ncclTree* tree1, int* indexes, int u0, int u1) {
  if (u0 != -1) tree0->up = indexes[u0];
  if (u1 != -1) tree1->up = indexes[u1];
  return ncclSuccess;
}

static ncclResult_t addRanksDown(int* down, int* indexes, int r0, int r1) {
  int x = 0;
  if (down[x] >= 0) x++;
  if (down[x] >= 0) {
    WARN("Internal error : tree already has more than one child (%d %d %d)\n", down[0], down[1], down[2]);
    return ncclInternalError;
  }
  if (r0 != -1) down[x++] = indexes[r0];
  if (r1 != -1) down[x++] = indexes[r1];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree0, struct ncclTree* tree1, int* indexes, int d0_0, int d0_1, int d1_0, int d1_1) {
  NCCLCHECK(addRanksDown(tree0->down, indexes, d0_0, d0_1));
  NCCLCHECK(addRanksDown(tree1->down, indexes, d1_0, d1_1));
  return ncclSuccess;
}

static ncclResult_t openRing(struct ncclTree* tree, int rank, int upRank) {
  if (tree->down[0] == upRank) tree->down[0] = -1;
  if (rank == upRank) tree->up = -1;
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeUpRecv, int* treeUpSend, int* treeDnRecv, int* treeDnSend, int* firstRanks) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;
  int* indexesSend, *indexesRecv;
  NCCLCHECK(ncclCalloc(&indexesSend, nNodes));
  NCCLCHECK(ncclCalloc(&indexesRecv, nNodes));

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int u0, d0_0, d0_1, u1, d1_0, d1_1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &u0, &d0_0, &d0_1, &u1, &d1_0, &d1_1));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     NCCLCHECK(getIndexes(treeUpSend+c*comm->nRanks, indexesSend, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeUpRecv+c*comm->nRanks, indexesRecv, nNodes, firstRanks));
     NCCLCHECK(openRing(&channel0->treeUp, comm->rank, indexesSend[node]));
     NCCLCHECK(openRing(&channel1->treeUp, comm->rank, indexesSend[node]));
     int root = indexesSend[node];
     if (indexesSend[node] == comm->rank) NCCLCHECK(setTreeUp(&channel0->treeUp, &channel1->treeUp, indexesRecv, u0, u1));
     if (indexesRecv[node] == comm->rank) NCCLCHECK(setTreeDown(&channel0->treeUp, &channel1->treeUp, indexesSend, d0_0, d0_1, d1_0, d1_1));
     NCCLCHECK(getIndexes(treeDnSend+c*comm->nRanks, indexesSend, nNodes, firstRanks));
     NCCLCHECK(getIndexes(treeDnRecv+c*comm->nRanks, indexesRecv, nNodes, firstRanks));
     NCCLCHECK(openRing(&channel0->treeDn, comm->rank, u0 == -1 ? root : indexesRecv[node]));
     NCCLCHECK(openRing(&channel1->treeDn, comm->rank, u1 == -1 ? root : indexesRecv[node]));
     if (indexesSend[node] == comm->rank) NCCLCHECK(setTreeDown(&channel0->treeDn, &channel1->treeDn, indexesRecv, d0_0, d0_1, d1_0, d1_1));
     if (indexesRecv[node] == comm->rank) NCCLCHECK(setTreeUp(&channel0->treeDn, &channel1->treeDn, indexesSend, u0, u1));
     TRACE(NCCL_GRAPH, "TreeUp %d : %d -> %d/%d/%d", c,           channel0->treeUp.up, channel0->treeUp.down[0], channel0->treeUp.down[1], channel0->treeUp.down[2]);
     TRACE(NCCL_GRAPH, "TreeUp %d : %d -> %d/%d/%d", c+nChannels, channel1->treeUp.up, channel1->treeUp.down[0], channel1->treeUp.down[1], channel1->treeUp.down[2]);
     TRACE(NCCL_GRAPH, "TreeDn %d : %d -> %d/%d/%d", c,           channel0->treeDn.up, channel0->treeDn.down[0], channel0->treeDn.down[1], channel0->treeDn.down[2]);
     TRACE(NCCL_GRAPH, "TreeDn %d : %d -> %d/%d/%d", c+nChannels, channel1->treeDn.up, channel1->treeDn.down[0], channel1->treeDn.down[1], channel1->treeDn.down[2]);
     channel0->treeUp.depth = channel1->treeUp.depth = depth;
  }
  free(indexesSend);
  free(indexesRecv);
  return ncclSuccess;
}

ncclResult_t ncclTopoConnectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph, int rank) {
  int nranks = comm->nRanks;
  int depth = nranks/comm->nNodes;
  int sendIndex = collNetGraph->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;  // send GPU index depends on topo pattern
  int sendEndIndex = (sendIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels/2; c++) {
    struct ncclChannel* channel = comm->channels+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[sendIndex+c*comm->localRanks]) { // is master
      channel->collTreeUp.up = channel->collTreeDn.up = nranks;
    }
    if (rank == collNetGraph->intra[sendEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTreeUp.down[0] = channel->collTreeDn.down[0] = -1;
    }
    channel->collTreeUp.depth = channel->collTreeDn.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", c, rank, channel->collTreeUp.up, channel->collTreeUp.down[0]);
  }
  int recvIndex = 0;  // recv GPU index is always 0
  int recvEndIndex = (recvIndex+comm->localRanks-1)%comm->localRanks;
  for (int c=0; c<comm->nChannels/2; c++) {
    struct ncclChannel* channel = comm->channels+comm->nChannels/2+c;
    // Set root of collTree to id nranks
    if (rank == collNetGraph->intra[recvIndex+c*comm->localRanks]) { // is master
      channel->collTreeUp.up = channel->collTreeDn.up = nranks;
    }
    if (rank == collNetGraph->intra[recvEndIndex+c*comm->localRanks]) { // is bottom of intra-node chain
      channel->collTreeUp.down[0] = channel->collTreeDn.down[0] = -1;
    }
    channel->collTreeUp.depth = channel->collTreeDn.depth = depth;
    INFO(NCCL_GRAPH, "CollNet Channel %d rank %d up %d down %d", comm->nChannels/2+c, rank, channel->collTreeDn.up, channel->collTreeDn.down[0]);
  }
  return ncclSuccess;
}

// Legacy naming
NCCL_PARAM(MinNrings, "MIN_NRINGS", -2);
NCCL_PARAM(MaxNrings, "MAX_NRINGS", -2);
// New naming
NCCL_PARAM(MinNchannels, "MIN_NCHANNELS", -2);
NCCL_PARAM(MaxNchannels, "MAX_NCHANNELS", -2);

int ncclMinNchannels() {
  int minNchannels = 0;
  if (ncclParamMinNrings() != -2) minNchannels = ncclParamMinNrings();
  if (ncclParamMinNchannels() != -2) minNchannels = ncclParamMinNchannels();
  if (minNchannels > MAXCHANNELS) {
    WARN("User asked for a minimum of %d channels, limiting to %d\n", minNchannels, MAXCHANNELS);
    minNchannels = MAXCHANNELS;
  }
  if (minNchannels < 0) minNchannels = 0;
  return minNchannels;
}
int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  if (maxNchannels > MAXCHANNELS) maxNchannels = MAXCHANNELS;
  if (maxNchannels < 1) {
    WARN("User asked for a maximum of %d channels, setting it to 1\n", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, struct ncclTopoRanks** allTopoRanks, int* rings) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext, *treeUpRecv, *treeUpSend, *treeDnRecv,*treeDnSend;
  int nranks = comm->nRanks;
  int nChannels = comm->nChannels;
  NCCLCHECK(ncclCalloc(&ringRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeUpSend, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnRecv, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeDnSend, nranks*MAXCHANNELS));
  for (int i=0; i<nranks; i++) {
    for (int c=0; c<nChannels;c++) {
      ringRecv[c*nranks+i] = allTopoRanks[i]->ringRecv[c];
      ringSend[c*nranks+i] = allTopoRanks[i]->ringSend[c];
      ringPrev[c*nranks+i] = allTopoRanks[i]->ringPrev[c];
      ringNext[c*nranks+i] = allTopoRanks[i]->ringNext[c];
      treeUpRecv[c*nranks+i] = allTopoRanks[i]->treeUpRecv[c];
      treeUpSend[c*nranks+i] = allTopoRanks[i]->treeUpSend[c];
      treeDnRecv[c*nranks+i] = allTopoRanks[i]->treeDnRecv[c];
      treeDnSend[c*nranks+i] = allTopoRanks[i]->treeDnSend[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext, firstRanks));
  NCCLCHECK(connectTrees(comm, treeUpRecv, treeUpSend, treeDnRecv, treeDnSend, firstRanks));

  // Duplicate ringPrev/ringNext for ncclBuildRing
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  nChannels = comm->nChannels = std::min((int)ncclMaxNchannels(), nChannels);
  int c;
  for (c=nChannels; c<ncclMinNchannels(); c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-nChannels)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-nChannels)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-nChannels, sizeof(struct ncclChannel));
  }
  nChannels = comm->nChannels = c;

  // Create rings array and check all is fine
  NCCLCHECK(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);
  free(treeUpRecv);
  free(treeUpSend);
  free(treeDnRecv);
  free(treeDnSend);

  return ncclSuccess;
}
