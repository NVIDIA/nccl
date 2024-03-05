/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "graph.h"
#include "trees.h"
#include "rings.h"
#include "topo.h"

/******************************************************************/
/********************* Internode connection ***********************/
/******************************************************************/

ncclResult_t ncclTopoPreset(struct ncclComm* comm, struct ncclTopoGraph** graphs, struct ncclTopoRanks* topoRanks) {
  int rank = comm->rank;
  int localRanks = comm->topo->nodes[GPU].count;
  int nChannels = comm->nChannels;

  topoRanks->nvlsHeadNum = 0;
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->ring.prev = channel->ring.next = -1;
    channel->tree.up = -1;
    channel->collnetChain.up = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->tree.down[i] = -1;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) channel->collnetChain.down[i] = -1;
    channel->collnetDirect.out = -1;
    channel->collnetDirect.headRank = -1;
    channel->collnetDirect.nHeads = 0;
    channel->collnetDirect.shift = 0;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY+1; i++) channel->collnetDirect.heads[i] = -1;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.up[i] = -1;
    for (int i=0; i<NCCL_MAX_DIRECT_ARITY; i++) channel->collnetDirect.down[i] = -1;

    int* ringIntra = graphs[NCCL_ALGO_RING]->intra+c*localRanks;
    int* treeIntra = graphs[NCCL_ALGO_TREE]->intra+c*localRanks;
    int* collNetIntra = graphs[NCCL_ALGO_COLLNET_CHAIN]->intra+c*localRanks;

    for (int i=0; i<localRanks; i++) {
      if (ringIntra[i] == rank) {
        topoRanks->ringRecv[c] = ringIntra[0];
        topoRanks->ringSend[c] = ringIntra[localRanks-1];
        topoRanks->ringPrev[c] = (i == 0) ? -1 : ringIntra[i-1];
        topoRanks->ringNext[c] = (i == localRanks-1) ? -1 : ringIntra[i+1];
      }
      if (treeIntra[i] == rank) {
        int parentIndex = 0;
        int child0Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_TREE ? 0 : 1;
        int child1Index = graphs[NCCL_ALGO_TREE]->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ? 1 : 0;

        topoRanks->treeToParent[c] = treeIntra[parentIndex];
        topoRanks->treeToChild0[c] = treeIntra[child0Index];
        topoRanks->treeToChild1[c] = treeIntra[child1Index];
        channel->tree.up         = i == 0 ? -1 : treeIntra[i-1];
        channel->tree.down[0]    = i == localRanks-1 ? -1 : treeIntra[i+1];
      }
      if (collNetIntra[i] == rank) {
        channel->collnetChain.up      = i == 0 ? comm->nRanks : collNetIntra[i-1];
        channel->collnetChain.down[0] = i == localRanks-1 ? -1 : collNetIntra[i+1];
      }
    }
  }
  // Duplicate channels trees
  struct ncclChannel* channel0 = comm->channels;
  struct ncclChannel* channel1 = channel0+nChannels;
  memcpy(channel1, channel0, nChannels*sizeof(struct ncclChannel));

  // Get nvls heads and the number of heads. Duplicate head is not allowed.
  for (int c = 0; c < graphs[NCCL_ALGO_NVLS]->nChannels; ++c) {
    bool addHead = true;
    int* nvlsIntra = graphs[NCCL_ALGO_NVLS]->intra + c * localRanks;

    for (int dup = 0; dup < topoRanks->nvlsHeadNum; dup++) {
      if (topoRanks->nvlsHeads[dup] == nvlsIntra[0]) {
        addHead = false;
        break;
      }
    }
    if (addHead) {
      topoRanks->nvlsHeads[topoRanks->nvlsHeadNum++] = nvlsIntra[0];
    }
  }

  return ncclSuccess;
}

static ncclResult_t connectRings(struct ncclComm* comm, int* ringRecv, int* ringSend, int* ringPrev, int* ringNext) {
  int nChannels = comm->nChannels;
  int nNodes = comm->nNodes;
  for (int c=0; c<nChannels; c++) {
    int* recv = ringRecv+c*comm->nNodes;
    int* send = ringSend+c*comm->nNodes;
    int* prev = ringPrev+c*comm->nRanks;
    int* next = ringNext+c*comm->nRanks;
    for (int n=0; n<nNodes; n++) {
      int recvRank = recv[n];
      int prevSendRank = send[(n-1+nNodes)%nNodes];
      prev[recvRank] = prevSendRank;
      int sendRank = send[n];
      int nextRecvRank = recv[(n+1)%nNodes];
      next[sendRank] = nextRecvRank;
    }
  }
  return ncclSuccess;
}

static ncclResult_t getIndexes(int* ranks, int* indexes, int nNodes) {
 for (int n=0; n<nNodes; n++) indexes[n] = ranks[n];
 return ncclSuccess;
}

static ncclResult_t setTreeUp(struct ncclTree* tree, int* indexes, int u) {
  if (u == -1) return ncclSuccess;
  tree->up = indexes[u];
  return ncclSuccess;
}

static ncclResult_t setTreeDown(struct ncclTree* tree, int* indexes, int d) {
  if (d == -1) return ncclSuccess;
  int x = 0;
  while (x < NCCL_MAX_TREE_ARITY && tree->down[x] >= 0) x++;
  if (x == NCCL_MAX_TREE_ARITY) {
    WARN("Internal error : tree already has %d children (%d %d %d)", x, tree->down[0], tree->down[1], tree->down[2]);
    return ncclInternalError;
  }
  tree->down[x] = indexes[d];
  return ncclSuccess;
}

static ncclResult_t connectTrees(struct ncclComm* comm, int* treeToParent, int* treeToChild0, int* treeToChild1, int* treePatterns) {
  const int nChannels = comm->nChannels, nNodes = comm->nNodes, node = comm->node;

  // Compute tree depth. Not an exact value but a good approximation in most
  // cases
  int depth = comm->nRanks/nNodes - 1 + log2i(nNodes);

  int t0u, t0d0, t0d1, t0ChildType, t1u, t1d0, t1d1, t1ChildType;
  int* ttp, *ttc0, *ttc1;
  NCCLCHECK(ncclGetDtree(nNodes, node, &t0u, &t0d0, &t0d1, &t0ChildType, &t1u, &t1d0, &t1d1, &t1ChildType));
  for (int c=0; c<nChannels; c++) {
     struct ncclChannel* channel0 = comm->channels+c;
     struct ncclChannel* channel1 = channel0+nChannels;
     ttp = treeToParent+c*comm->nNodes;
     ttc0 = treeToChild0+c*comm->nNodes;
     ttc1 = treeToChild1+c*comm->nNodes;
     if (comm->rank == ttp[node]) {
       NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, t0u));
       NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, t1u));
     }
     if (comm->rank == ttc0[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d0));
       NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d0));
     }
     if (comm->rank == ttc1[node]) {
       NCCLCHECK(setTreeDown(&channel0->tree, ttp, t0d1));
       NCCLCHECK(setTreeDown(&channel1->tree, ttp, t1d1));
     }
     if (comm->rank == ttp[node] ||
         comm->rank == ttc0[node] ||
         comm->rank == ttc1[node]) {
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c,           channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
       INFO(NCCL_GRAPH, "Tree %d : %d -> %d -> %d/%d/%d", c+nChannels, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
     }
     channel0->tree.depth = channel1->tree.depth = depth;
  }
  return ncclSuccess;
}

static ncclResult_t connectCollNet(struct ncclComm* comm, struct ncclTopoGraph* collNetGraph) {
  int rank = comm->rank;
  int localRanks = comm->localRanks;
  int nHeads = 0;
  int *heads;
  NCCLCHECK(ncclCalloc(&heads, localRanks));
  // Find all head ranks
  // Head index is always 0
  for (int c=0; c<collNetGraph->nChannels; c++) {
    int* collNetIntra = collNetGraph->intra+c*localRanks;
    int head = collNetIntra[0];
    for (int h=0; h<nHeads; h++) if (heads[h] == head) head = -1;
    if (head != -1) heads[nHeads++] = collNetIntra[0];
  }
  // For all channels
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    char line[1024];
    sprintf(line, "CollNet channel %d rank %d ", c, rank);
    int nDown = 0;
    for (int i=0; i<nHeads; i++) {
      if (rank == heads[i]) { // is head
        channel->collnetDirect.headRank = i; // Mark the index for deciding offset in the CUDA kernel
        channel->collnetDirect.out = comm->nRanks; // Set root of collnetDirect to id nranks
        int* collNetIntra = collNetGraph->intra+i*localRanks;
        sprintf(line+strlen(line), "down ");
        for (int r=0; r<localRanks; r++) {
          if (collNetIntra[r] == rank) continue;
          channel->collnetDirect.down[nDown++] = collNetIntra[r];  // connect to all peers
          sprintf(line+strlen(line), " %d ", collNetIntra[r]);
        }
        sprintf(line+strlen(line), "nDown %d ", nDown);
        break;
      }
    }
    // Connect to all heads
    int nUp = 0;
    sprintf(line+strlen(line), "up ");
    for (int h=0; h<nHeads; h++) {
      if (rank == heads[h]) continue;
      channel->collnetDirect.up[nUp++] = heads[h];
      sprintf(line+strlen(line), " %d ", heads[h]);
    }
    sprintf(line+strlen(line), "heads ");
    { // heads[] is the list of heads ordered in head order startubg with self
      int h0 = (channel->collnetDirect.headRank == -1) ? 0 : channel->collnetDirect.headRank;
      for (int h1=0; h1 < nHeads; h1++) {
        int h = (h0+h1)%nHeads;
        channel->collnetDirect.heads[h1] = heads[h];
        sprintf(line+strlen(line), " %d ", heads[h]);
      }
    }
    channel->collnetDirect.nHeads = nHeads;
    channel->collnetDirect.shift = (rank%localRanks)%nHeads; // Shift by intraRank so that leaves don't send to same head simultaneously
    channel->collnetDirect.depth = (nUp == 0 && nDown == 0) ? 1 : 2;
    sprintf(line+strlen(line), "nUp %d nHeads %d ", nUp, nHeads);
    sprintf(line+strlen(line), "headRank %d out %d shift %d", channel->collnetDirect.headRank, channel->collnetDirect.out, channel->collnetDirect.shift);
    INFO(NCCL_GRAPH, "%s", line);
    channel->collnetChain.depth = comm->nRanks/comm->nNodes;
  }
  free(heads);
  return ncclSuccess;
}

static ncclResult_t connectNvls(struct ncclComm* comm, int* nvlsHeads, int nHeads) {
  int headRank = -1;
  if (nHeads == 0) {
    comm->nvlsChannels = 0;
    return ncclSuccess;
  }

  for (int h = 0; h < nHeads; h++) {
    if (nvlsHeads[h * comm->nNodes + comm->node] == comm->rank) headRank = h;
  }

  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.nHeads = nHeads;
    for (int h=0; h<nHeads; h++) channel->nvls.up[h] = comm->nRanks+1+h;
    for (int h=nHeads; h<NCCL_MAX_NVLS_ARITY; h++) channel->nvls.up[h] = -1;
    channel->nvls.down = comm->nRanks+1+headRank;
    channel->nvls.out = -1;       // NVLS+SHARP not yet implemented.
    channel->nvls.headRank = headRank;
    channel->nvls.treeUp = channel->nvls.treeDown[0] = channel->nvls.treeDown[1] = channel->nvls.treeDown[2] = -1;
    channel->nvls.node = comm->node;
    channel->nvls.nNodes = comm->nNodes;
    if (comm->collNetSupport && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;
  }
  // MNNVL: NVLS not yet supported
  if (comm->nNodes == 1 || comm->MNNVL) return ncclSuccess;

  // Connect Trees
  int tree0Parent, tree0Child0, tree0Child1, tree1Parent, tree1Child0, tree1Child1;
  int pc0, pc1; // ignored
  NCCLCHECK(ncclGetDtree(comm->nNodes, comm->node,
        &tree0Parent, &tree0Child0, &tree0Child1, &pc0,
        &tree1Parent, &tree1Child0, &tree1Child1, &pc1));

  int* heads = NULL;
  int treeUp[2] = { -1, -1 };
  int treeDown0[2] = { -1, -1 };
  int treeDown1[2] = { -1, -1 };

  if (comm->node == 0) {
    for (int h=0; h<nHeads; h++) {
      char line[1024];
      sprintf(line, "NVLS Head %2d:", h);
      heads = nvlsHeads+h*comm->nNodes;
      for (int n=0; n<comm->nNodes && n<20; n++) {
        sprintf(line+strlen(line), " %2d", heads[n]);
      }
      INFO(NCCL_INIT, "%s", line);
    }
  }

  // Find the heads where I'm the head rank and retain tree up/down
  for (int h=0; h<nHeads; h++) {
    heads = nvlsHeads+h*comm->nNodes;
    if (heads[comm->node] == comm->rank) {
      treeUp[0] = tree0Parent == -1 ? -1: heads[tree0Parent];
      treeDown0[0] = tree0Child0 == -1 ? -1 : heads[tree0Child0];
      treeDown1[0] = tree0Child1 == -1 ? -1 : heads[tree0Child1];
      treeUp[1] = tree1Parent == -1 ? -1 : heads[tree1Parent];
      treeDown0[1] = tree1Child0 == -1 ? -1 : heads[tree1Child0];
      treeDown1[1] = tree1Child1 == -1 ? -1 : heads[tree1Child1];
      break;
    }
  }
  // Set prev/next in all channels (NVLS compute channels work
  // orthogonally to NVLS search channels).
  for (int c=0; c<comm->nChannels; c++) {
    struct ncclChannel* channel = comm->channels+c;
    channel->nvls.treeUp = treeUp[c%2];
    channel->nvls.treeDown[0] = channel->nvls.down;
    int ix = 1;
    if (treeDown0[c%2] != -1) channel->nvls.treeDown[ix++] = treeDown0[c%2];
    if (treeDown1[c%2] != -1) channel->nvls.treeDown[ix] = treeDown1[c%2];
  }

  struct ncclNvls* nvls0 = &comm->channels[0].nvls;
  struct ncclNvls* nvls1 = &comm->channels[1].nvls;
  INFO(NCCL_GRAPH, "NVLS Trees : %d/%d->%d->%d %d/%d->%d->%d",
      nvls0->treeDown[0], nvls0->treeDown[1], comm->rank, nvls0->treeUp,
      nvls1->treeDown[0], nvls1->treeDown[1], comm->rank, nvls1->treeUp);
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
    WARN("User asked for a minimum of %d channels, limiting to %d", minNchannels, MAXCHANNELS);
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
    WARN("User asked for a maximum of %d channels, setting it to 1", maxNchannels);
    maxNchannels = 1;
  }
  return maxNchannels;
}

static int copyChannels(struct ncclComm* comm, int start, int end, int* ringPrev, int* ringNext) {
  int nranks = comm->nRanks;
  int c;
  for (c=start; c<end; c++) {
    memcpy(ringPrev+c*nranks, ringPrev+(c-start)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c-start)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

void exchangeValues(int* v0, int* v1) {
  int tmp = *v1;
  *v1 = *v0;
  *v0 = tmp;
}

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs) {
  // Gather data from all ranks
  int *ringRecv, *ringSend, *ringPrev, *ringNext, *treeToParent, *treeToChild0, *treeToChild1, *nvlsHeads;
  int nranks = comm->nRanks;
  int nNodes = comm->nNodes;
  int nChannels = comm->nChannels;
  int minHeadNum = INT_MAX;
  NCCLCHECK(ncclCalloc(&ringRecv, nNodes*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringSend, nNodes*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringPrev, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&ringNext, nranks*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToParent, nNodes*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToChild0, nNodes*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&treeToChild1, nNodes*MAXCHANNELS));
  NCCLCHECK(ncclCalloc(&nvlsHeads, nNodes*MAXCHANNELS));

  // Alternate rings to avoid crossing rails
  if (graphs[NCCL_ALGO_RING]->crossNic && (comm->nNodes % 2) == 0 && (nChannels % 2) == 0) {
    for (int r=0; r<comm->nRanks; r++) {
      if (comm->rankToNode[r] % 2 == 1) {
        // Exchange rings
        for (int c=0; c<nChannels; c+=2) {
          exchangeValues(allTopoRanks[r]->ringRecv+c, allTopoRanks[r]->ringRecv+(c^1));
          exchangeValues(allTopoRanks[r]->ringSend+c, allTopoRanks[r]->ringSend+(c^1));
          exchangeValues(allTopoRanks[r]->ringPrev+c, allTopoRanks[r]->ringPrev+(c^1));
          exchangeValues(allTopoRanks[r]->ringNext+c, allTopoRanks[r]->ringNext+(c^1));
        }
      }
    }
  }

  for (int c=0; c<nChannels;c++) {
    for (int n=0; n<nNodes; n++) {
      int r = firstRanks[n];
      ringRecv[c*nNodes+n] = allTopoRanks[r]->ringRecv[c];
      ringSend[c*nNodes+n] = allTopoRanks[r]->ringSend[c];
      treeToParent[c*nNodes+n] = allTopoRanks[r]->treeToParent[c];
      treeToChild0[c*nNodes+n] = allTopoRanks[r]->treeToChild0[c];
      treeToChild1[c*nNodes+n] = allTopoRanks[r]->treeToChild1[c];
    }
    for (int r=0; r<nranks; r++) {
      ringPrev[c*nranks+r] = allTopoRanks[r]->ringPrev[c];
      ringNext[c*nranks+r] = allTopoRanks[r]->ringNext[c];
    }
  }

  for (int n = 0; n < nNodes; n++) {
    int r = firstRanks[n];
    if (minHeadNum > allTopoRanks[r]->nvlsHeadNum)
      minHeadNum = allTopoRanks[r]->nvlsHeadNum;
  }

  for (int c = 0; c < minHeadNum; c++) {
    for (int n = 0; n < nNodes; n++) {
      int r = firstRanks[n];
      nvlsHeads[c * nNodes + n] = allTopoRanks[r]->nvlsHeads[c];
    }
  }

  // Connect rings and trees. This should also duplicate the channels.
  NCCLCHECK(connectRings(comm, ringRecv, ringSend, ringPrev, ringNext));
  NCCLCHECK(connectTrees(comm, treeToParent, treeToChild0, treeToChild1, treePatterns));

  // Duplicate ringPrev/ringNext for ncclBuildRing
  memcpy(ringPrev+nChannels*nranks, ringPrev, nChannels*nranks*sizeof(int));
  memcpy(ringNext+nChannels*nranks, ringNext, nChannels*nranks*sizeof(int));

  // Set ring prev/next for my rank
  for (int c=0; c<nChannels; c++) {
    struct ncclChannel* channel0 = comm->channels+c;
    struct ncclChannel* channel1 = channel0+nChannels;
    channel0->ring.prev = channel1->ring.prev = ringPrev[c*nranks+comm->rank];
    channel0->ring.next = channel1->ring.next = ringNext[c*nranks+comm->rank];
  }

  // Duplication should be complete now
  nChannels = comm->nChannels = std::min(MAXCHANNELS,nChannels*2);

  // Setup CollNet
  if (comm->collNetSupport == 1) {
    struct ncclTopoGraph* collNetGraph = graphs[NCCL_ALGO_COLLNET_DIRECT];
    // Add more channels to saturate intra-node bandwidth, except the 1 PPN case
    if (collNetGraph->bwIntra > collNetGraph->bwInter && comm->nRanks > comm->nNodes) {
      int collNetNchannels = std::min(MAXCHANNELS, nChannels+nChannels/2);
      nChannels = comm->nChannels = copyChannels(comm, nChannels, collNetNchannels, ringPrev, ringNext);
    }
    NCCLCHECK(connectCollNet(comm, collNetGraph));
  }

  // Use 4 compute channels per search channel to reach peak BW on <8 PPN
  if (comm->minCompCap == 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && nChannels < 16) {
     nChannels = comm->nChannels = copyChannels(comm, nChannels, 2*nChannels, ringPrev, ringNext);
  }

  // Honor NCCL_MIN_NRINGS/NCCL_MAX_NRINGS.
  // We permit combining max, then min, to only use the first channels, then duplicate them.
  if (comm->sharedRes->owner != comm) {
    /* child comm #channels cannot exceed top parent #channels. */
    nChannels = comm->nChannels = std::min(std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::min(std::max(ncclMinNchannels(), comm->config.minCTAs), comm->sharedRes->tpNChannels), ringPrev, ringNext);
  } else {
    nChannels = comm->nChannels = std::min(std::min(ncclMaxNchannels(), nChannels), comm->config.maxCTAs);
    nChannels = comm->nChannels = copyChannels(comm, nChannels, std::max(ncclMinNchannels(), comm->config.minCTAs), ringPrev, ringNext);
  }

  comm->collChannels = comm->nChannels;
  // Support maximal channel usage for aggregation
  if (comm->nChannels < comm->nvlsChannels) {
    nChannels = comm->nChannels = copyChannels(comm, comm->nChannels, comm->nvlsChannels, ringPrev, ringNext);
  }
  NCCLCHECK(connectNvls(comm, nvlsHeads, minHeadNum));

  // Create rings array and check all is fine
  NCCLCHECK(ncclBuildRings(nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext));

  free(ringRecv);
  free(ringSend);
  free(ringPrev);
  free(ringNext);
  free(treeToParent);
  free(treeToChild0);
  free(treeToChild1);
  free(nvlsHeads);

  return ncclSuccess;
}
