/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "device.h"
#include "graph.h"
#include "transport.h"
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
  memcpy(comm->nvlsHeads, topoRanks->nvlsHeads, sizeof(int) * topoRanks->nvlsHeadNum);

  return ncclSuccess;
}

// ScatterDc = 0 will disable the scattering of the channels on different NICs.
// This way we ensure that all channels will follow the same rank order.
NCCL_PARAM(ScatterXDc, "SCATTER_XDC", 1);

// returns the node that will be used to cross-DC for a given channel
static int channelToNode(struct ncclComm* comm, int nNodes, int channel, int nChannels) {
  int pow2 = 1;
  while ((pow2 << 1) <= nNodes) pow2 <<= 1;
  // In the case of multiple communicators on the same node, they will all have the same value of c, which leads them to use the same node to cross the DC.
  // To avoid this, we offset the channel index by the nvmlDev index on rank 0.
  // For example with 4 GPUs/node and 16 channels/comm (so c = 0 - 15):
  // - comm 0 (GPU 0-3) will have channel index of 0-15,
  // - comm 1 (GPU 4-7) will have channel index of 16-31
  // This way (GPU as the outer index), we make sure that two GPUs need to share a node, it will not be adjacent GPUs.
  // For example, if two out of 8 GPUs need to share node 0, it will be GPU 0 and 4.
  int commId = nChannels * comm->peerInfo[0].nvmlDev + (ncclParamScatterXDc() ? channel : 0);
  return mirrorBits(commId, pow2) % nNodes;
}

static int getCrossNodeForRing(struct ncclComm* comm, struct ncclDcNode* dc, int c, int nChannels) {
  return channelToNode(comm, dc->localNodes, c, nChannels);
}


#define NODE_FROM_DC(dc, i) (dc->localNodeToNode[((i) + dc->localNodes) % dc->localNodes])
// connect the inter-node for nChannels rings. For each ring, ringPrev and ringNext store respectivelly the previous and the next rank in the ring for all the ranks.
// We close the rings inter-node using the search channels information: ringRecv and ringSend.
// For each search channel, ringRecv and ringSend contains the recv and send rank on each node.
static ncclResult_t connectRings(struct ncclComm* comm, int nChannels, struct ncclChannel* channels, int* ringPrev, int* ringNext, int nSearchChannels, int* ringRecv, int* ringSend) {
  INFO(NCCL_GRAPH, "%s: comm 0x%lx connecting %d ring channels using %d search channels", __func__, comm->commHash, nChannels, nSearchChannels);
  int nDc = comm->dcCount;
  for (int c = 0; c < nChannels; c++) {
    int* prev = ringPrev + c * comm->nRanks;
    int* next = ringNext + c * comm->nRanks;
    int* recv = ringRecv + (c % nSearchChannels) * comm->nNodes;
    int* send = ringSend + (c % nSearchChannels) * comm->nNodes;
    for (int dc = 0; dc < nDc; ++dc) {
      struct ncclDcNode* dcNode = &comm->dcNode[dc];
      // we first connect the all the nodes inside the same DC together
      int nNodesInDc = dcNode->localNodes;
      for (int iNode = 0; iNode < nNodesInDc; iNode++) {
        int node = dcNode->localNodeToNode[iNode];
        // recv connects to the prev send
        const int prevNode = dcNode->localNodeToNode[(iNode - 1 + nNodesInDc) % nNodesInDc];
        const int recvRank = recv[node];
        const int prevSendRank = send[prevNode];
        prev[recvRank] = prevSendRank;
        // send connect to the next recv
        const int nextNode = dcNode->localNodeToNode[(iNode + 1) % nNodesInDc];
        const int sendRank = send[node];
        const int nextRecvRank = recv[nextNode];
        next[sendRank] = nextRecvRank;
      }
    }
    // for each DC, we open the rings between node getCrossNodeForRing(dc,c) and getCrossNodeForRing(dc+1,c) + 1
    for (int dc = 0; dc < nDc; ++dc) {
      struct ncclDcNode* currDc = &comm->dcNode[dc];
      struct ncclDcNode* nextDc = &comm->dcNode[(dc + 1) % nDc];
      struct ncclDcNode* prevDc = &comm->dcNode[(dc - 1 + nDc) % nDc];
      int crossNode = getCrossNodeForRing(comm, currDc, c, nChannels);
      INFO(NCCL_GRAPH, "%s: crossNode on DC %d of ring[%d] is node %d", __func__, dc, c, crossNode);
      // recv from the previous DC
      int recvRank = recv[NODE_FROM_DC(currDc, crossNode + 1)];
      int prevSendRank = send[NODE_FROM_DC(prevDc, getCrossNodeForRing(comm, prevDc, c, nChannels))];
      prev[recvRank] = prevSendRank;
      INFO(NCCL_GRAPH, "DC %d - ring %d: connecting rank %d (prev DC) -> rank %d", dc, c, prevSendRank, recvRank);
      // send to the next recv
      int sendRank = send[NODE_FROM_DC(currDc, crossNode)];
      int nextRecvRank = recv[NODE_FROM_DC(nextDc, getCrossNodeForRing(comm, nextDc, c, nChannels) + 1)];
      next[sendRank] = nextRecvRank;
      INFO(NCCL_GRAPH, "DC %d - ring %d: connecting rank %d -> rank %d (next DC)", dc, c, sendRank, nextRecvRank);
    }
    channels[c].ring.prev = prev[comm->rank];
    channels[c].ring.next = next[comm->rank];
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

static int rootNodeTree0(struct ncclComm* comm, struct ncclDcNode* dc, int tree0Id, int nTrees0) {
  // Given a root for tree 0 (R0), the root of tree 1 will either be (R0+1)%nNodes (nNodes odd) or (R0+nNodes-1)%nNodes (nNodes even), see trees.cc
  // Therefore, we distribute the roots of trees 0 on the even nodes only
  int nNodes = dc->localNodes;
  return 2 * channelToNode(comm, nNodes / 2, tree0Id, nTrees0);
}

static int shiftedNodeFromLocalNode(struct ncclComm* comm, struct ncclDcNode* dc, int localNode, int tree0Id, int nTrees0) {
  int root = rootNodeTree0(comm, dc, tree0Id, nTrees0);
  return (localNode - root + dc->localNodes) % dc->localNodes;
}

static int globalNodeFromShiftedNode(struct ncclComm* comm, struct ncclDcNode* dc, int shiftedNode, int tree0Id, int nTrees0) {
  if (shiftedNode == -1) return -1;
  int root = rootNodeTree0(comm, dc, tree0Id, nTrees0);
  int localNodeIndex = (shiftedNode + root) % dc->localNodes;
  return NODE_FROM_DC(dc, localNodeIndex);
}

static struct ncclDcNode* dcIdFromGlobalNode(struct ncclComm* comm, int node, int* dcId, int* localNode) {
  *dcId = -1;
  *localNode = -1;
  for (int dc = 0; dc < comm->dcCount; ++dc) {
    for (int n = 0; n < comm->dcNode[dc].localNodes; ++n) {
      if (comm->dcNode[dc].localNodeToNode[n] == comm->node) {
        *dcId = dc;
        *localNode = n;
        break;
      }
    }
    if (*dcId >= 0 && *localNode >= 0) break;
  }
  return &comm->dcNode[*dcId];
}

// connect a total of nChannels trees (nChannels/2 primary trees and nChannels/2 dual trees) using the channels found in the search. Both trees will use the same search channel.
// For each search channel, treeToParent, treeToChild0, and treeToChild1, contain respectivelly the rank communicating with the parent, the child0, and the child 1 for each node.
static ncclResult_t connectTrees(struct ncclComm* comm, const int nChannels, struct ncclChannel* channels, const int nSearchChannels, int* treeToParent, int* treeToChild0,
                                 int* treeToChild1, int* treePatterns) {
  int dcId = -1, localNode = -1;
  const int nNodes = comm->nNodes, node = comm->node, nDc = comm->dcCount;
  struct ncclDcNode* currDc = dcIdFromGlobalNode(comm, comm->node, &dcId, &localNode);

  // Compute tree depth. Not an exact value but a good approximation in most cases
  int maxDepthDc = 0;
  for (int d = 0; d < nDc; ++d) maxDepthDc = std::max((int)log2i(comm->dcNode[d].localNodes), maxDepthDc);
  int depth = /*intraNode*/ (comm->nRanks / nNodes - 1) + /*intra-DC*/ maxDepthDc + /*inter-DC*/ (nDc - 1);

  int nTrees1 = nChannels / 2;
  int nTrees0 = nTrees1 + (nChannels % 2);
  INFO(NCCL_GRAPH, "%s: comm 0x%lx connecting %d tree channels (%d primal, %d dual) using %d search channels", __func__, comm->commHash, nChannels, nTrees0, nTrees1, nSearchChannels);
  for (int c0 = 0; c0 < nTrees0; c0++) {
    // primal and dual channels, if nChannels is odd, the last dual is not done
    struct ncclChannel* channel0 = channels + c0;
    struct ncclChannel* channel1 = (c0 < nTrees1) ? (channels + nTrees0 + c0) : NULL;
    // dual channel (channel1) has to be the same as channel0. This could not be the case if the number of search channels is higher than the number of desired channels.
    if (channel1) memcpy(&channel1->tree, &channel0->tree, sizeof(struct ncclTree));

    int* ttp = treeToParent + (c0 % nSearchChannels) * comm->nNodes;
    int* ttc0 = treeToChild0 + (c0 % nSearchChannels) * comm->nNodes;
    int* ttc1 = treeToChild1 + (c0 % nSearchChannels) * comm->nNodes;
    //  intraDC tree: each primal/dual trees will get a different root. the shifted ID is the same for both the primal and the dual tree.
    int tt[6];
    int t0ChildType, t1ChildType;
    int nodeShifted = shiftedNodeFromLocalNode(comm, currDc, localNode, c0, nTrees0);
    NCCLCHECK(ncclGetDtree(comm->dcNode[dcId].localNodes, nodeShifted, tt + 0, tt + 1, tt + 2, &t0ChildType, tt + 3, tt + 4, tt + 5, &t1ChildType));
    // we need to restranslate the shifted local indexes into unshifted global index
    int tu[2] = {globalNodeFromShiftedNode(comm, currDc, tt[0], c0, nTrees0), globalNodeFromShiftedNode(comm, currDc, tt[3], c0, nTrees0)};
    int td0[2] = {globalNodeFromShiftedNode(comm, currDc, tt[1], c0, nTrees0), globalNodeFromShiftedNode(comm, currDc, tt[4], c0, nTrees0)};
    int td1[2] = {globalNodeFromShiftedNode(comm, currDc, tt[2], c0, nTrees0), globalNodeFromShiftedNode(comm, currDc, tt[5], c0, nTrees0)};
    if (comm->rank == ttp[node]) {
      NCCLCHECK(setTreeUp(&channel0->tree, t0ChildType == 0 ? ttc0 : ttc1, tu[0]));
      if (channel1) NCCLCHECK(setTreeUp(&channel1->tree, t1ChildType == 0 ? ttc0 : ttc1, tu[1]));
    }
    if (comm->rank == ttc0[node]) {
      NCCLCHECK(setTreeDown(&channel0->tree, ttp, td0[0]));
      if (channel1) NCCLCHECK(setTreeDown(&channel1->tree, ttp, td0[1]));
    }
    if (comm->rank == ttc1[node]) {
      NCCLCHECK(setTreeDown(&channel0->tree, ttp, td1[0]));
      if (channel1) NCCLCHECK(setTreeDown(&channel1->tree, ttp, td1[1]));
    }
    if (comm->rank == ttp[node] || comm->rank == ttc0[node] || comm->rank == ttc1[node]) {
      INFO(NCCL_GRAPH, "Tree %d : %d <-> %d <-> %d/%d/%d", c0, channel0->tree.up, comm->rank, channel0->tree.down[0], channel0->tree.down[1], channel0->tree.down[2]);
      if (channel1)
        INFO(NCCL_GRAPH, "Tree %d : %d <-> %d <-> %d/%d/%d", c0 + nTrees0, channel1->tree.up, comm->rank, channel1->tree.down[0], channel1->tree.down[1], channel1->tree.down[2]);
    }

    /* chain DC roots together to create a single tree
    Example with 3 DCs (R0, R1, and R2 represent the roots of each DC's tree; T0, T1, and T2 the rest of the corresponding tree)
               R2
              /  \
           R1     T2
         /   \
      R0      T1
       \
        T0
    */
    const bool isNodeRoot[2] = {(tu[0] == -1), (tu[1] == -1)};
    if (isNodeRoot[0] || isNodeRoot[1]) {
      int root0, root1;
      // connect rank = ttp[root node of current DC] to the ttc0[root node of next DC]
      if (comm->rank == ttp[node] && dcId < (nDc - 1)) {
        NCCLCHECK(ncclGetDtreeRoots(comm->dcNode[dcId + 1].localNodes, &root0, &root1));
        int nodeR0 = globalNodeFromShiftedNode(comm, &comm->dcNode[dcId + 1], root0, c0, nTrees0);
        int nodeR1 = globalNodeFromShiftedNode(comm, &comm->dcNode[dcId + 1], root1, c0, nTrees0);
        if (isNodeRoot[0]) {
          NCCLCHECK(setTreeUp(&channel0->tree, ttc0, nodeR0));
          INFO(NCCL_GRAPH, "%s: comm 0x%lx primal TREE %d/%d -> rank %d (node %d, DC %d) up to rank %d (node %d, DC %d) ", __func__, comm->commHash, c0, nTrees0, comm->rank, node,
               dcId, ttc0[nodeR0], nodeR0, dcId + 1);
        }
        if (isNodeRoot[1] && channel1) {
          NCCLCHECK(setTreeUp(&channel1->tree, ttc0, nodeR1));
          INFO(NCCL_GRAPH, "%s: comm 0x%lx dual TREE %d/%d -> rank %d (node %d, DC %d) up to rank %d (node %d, DC %d) ", __func__, comm->commHash, c0, nTrees1, comm->rank, node,
               dcId, ttc0[nodeR1], nodeR1, dcId + 1);
        }
      }
      // connect rank = ttc0[root node of current DC] to the ttp[root node of previous DC]
      if (comm->rank == ttc0[node] && dcId > 0) {
        // if I am the rank talking to the child 0, establish connection with the next DC root
        NCCLCHECK(ncclGetDtreeRoots(comm->dcNode[dcId - 1].localNodes, &root0, &root1));
        int nodeR0 = globalNodeFromShiftedNode(comm, &comm->dcNode[dcId - 1], root0, c0, nTrees0);
        int nodeR1 = globalNodeFromShiftedNode(comm, &comm->dcNode[dcId - 1], root1, c0, nTrees0);
        if (isNodeRoot[0]) {
          NCCLCHECK(setTreeDown(&channel0->tree, ttp, nodeR0));
          INFO(NCCL_GRAPH, "%s: comm 0x%lx primal TREE %d/%d -> rank %d (node %d, DC %d) down to rank %d (node %d, DC %d) ", __func__, comm->commHash, c0, nTrees0, comm->rank, node,
               dcId, ttp[nodeR0], nodeR0, dcId - 1);
        }
        if (isNodeRoot[1] && channel1) {
          NCCLCHECK(setTreeDown(&channel1->tree, ttp, nodeR1));
          INFO(NCCL_GRAPH, "%s: comm 0x%lx dual TREE %d/%d -> rank %d (node %d, DC %d) down to rank %d (node %d, DC %d) ", __func__, comm->commHash, c0, nTrees1, comm->rank, node,
               dcId, ttp[nodeR1], nodeR1, dcId - 1);
        }
      }
    }
    channel0->tree.depth = depth;
    if (channel1) channel1->tree.depth = depth;
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
    sprintf(line, "CollNetDirect channel %d rank %d ", c, rank);
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
    // nHeads should always be greater than 0.
    // coverity[divide_by_zero]
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
    if (comm->collNetSupport && channel->nvls.headRank != -1) channel->nvls.out = comm->nRanks;
  }
  if (comm->nNodes == 1) return ncclSuccess;

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
  INFO(NCCL_GRAPH, "NVLS Trees : %d/%d/%d->%d->%d %d/%d/%d->%d->%d",
      nvls0->treeDown[0], nvls0->treeDown[1], nvls0->treeDown[2], comm->rank, nvls0->treeUp,
      nvls1->treeDown[0], nvls1->treeDown[1], nvls1->treeDown[2], comm->rank, nvls1->treeUp);
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

extern int64_t ncclParamWorkArgsBytes();

int ncclMaxNchannels() {
  int maxNchannels = MAXCHANNELS;
  if (ncclParamMaxNrings() != -2) maxNchannels = ncclParamMaxNrings();
  if (ncclParamMaxNchannels() != -2) maxNchannels = ncclParamMaxNchannels();
  maxNchannels = std::min(maxNchannels, ncclDevMaxChannelsForArgsBytes(ncclParamWorkArgsBytes()));
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
    memcpy(ringPrev+c*nranks, ringPrev+(c%start)*nranks, nranks*sizeof(int));
    memcpy(ringNext+c*nranks, ringNext+(c%start)*nranks, nranks*sizeof(int));
    memcpy(comm->channels+c, comm->channels+c-start, sizeof(struct ncclChannel));
  }
  return c;
}

void exchangeValues(int* v0, int* v1) {
  int tmp = *v1;
  *v1 = *v0;
  *v0 = tmp;
}

NCCL_PARAM(UnpackDoubleNChannels, "UNPACK_DOUBLE_NCHANNELS", 1);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks, int* treePatterns, struct ncclTopoRanks** allTopoRanks, int* rings, struct ncclTopoGraph** graphs, struct ncclComm* parent) {
  // Gather data from all ranks
  ncclResult_t ret = ncclSuccess;
  int *ringRecv = NULL, *ringSend = NULL, *ringPrev = NULL, *ringNext = NULL, *treeToParent = NULL, *treeToChild0 = NULL, *treeToChild1 = NULL, *nvlsHeads = NULL;
  int nranks = comm->nRanks;
  int nNodes = comm->nNodes;
  int nSearchChannels = comm->nChannels;
  int maxChannels, minChannels;

  int minHeadNum = INT_MAX;
  int shared = parent && parent->nvlsSupport && parent->config.splitShare;
  NCCLCHECKGOTO(ncclCalloc(&ringPrev, nranks * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringNext, nranks * MAXCHANNELS), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringRecv, nNodes * nSearchChannels), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&ringSend, nNodes * nSearchChannels), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToParent, nNodes * nSearchChannels), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild0, nNodes * nSearchChannels), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&treeToChild1, nNodes * nSearchChannels), ret, fail);
  NCCLCHECKGOTO(ncclCalloc(&nvlsHeads, nNodes * MAXCHANNELS), ret, fail);

  // Alternate rings to avoid crossing rails
  if (graphs[NCCL_ALGO_RING]->crossNic == 2 && (nSearchChannels % 2) == 0) {
    for (int r=0; r<comm->nRanks; r++) {
      if (comm->rankToNode[r] % 2 == 1) {
        // Exchange rings
        for (int c=0; c<nSearchChannels; c+=2) {
          exchangeValues(allTopoRanks[r]->ringRecv+c, allTopoRanks[r]->ringRecv+(c^1));
          exchangeValues(allTopoRanks[r]->ringSend+c, allTopoRanks[r]->ringSend+(c^1));
          exchangeValues(allTopoRanks[r]->ringPrev+c, allTopoRanks[r]->ringPrev+(c^1));
          exchangeValues(allTopoRanks[r]->ringNext+c, allTopoRanks[r]->ringNext+(c^1));
        }
      }
    }
  }

  for (int c = 0; c < nSearchChannels; c++) {
    for (int n = 0; n < nNodes; n++) {
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

  // each search channel gets two compute channels
  comm->nChannels = std::min(MAXCHANNELS, nSearchChannels * 2);

  // Setup CollNet
  if (comm->collNetSupport == 1) {
    struct ncclTopoGraph* collNetChainGraph = graphs[NCCL_ALGO_COLLNET_CHAIN];
    // Add more channels to saturate intra-node bandwidth, except the 1 PPN case
    if (collNetChainGraph->bwIntra > collNetChainGraph->bwInter && comm->nRanks > comm->nNodes) {
      comm->nChannels = std::min(MAXCHANNELS, comm->nChannels + comm->nChannels / 2);
    }
    NCCLCHECKGOTO(connectCollNet(comm, graphs[NCCL_ALGO_COLLNET_DIRECT]), ret, fail);
  }

  // Use 4 compute channels per search channel to reach peak BW on <8 PPN
  if (comm->minCompCap >= 90 && comm->nNodes > 1 && graphs[NCCL_ALGO_RING]->bwIntra > 45.0 && comm->nChannels < 16) {
    comm->nChannels = std::min(MAXCHANNELS, comm->nChannels * 2);
  }

  // Double the number of channels when using unpack networking (greater than 1 node)
  // We won't automatically double past 16 channels, users can specify 32 if they want
  if (comm->netDeviceType == NCCL_NET_DEVICE_UNPACK && comm->nNodes > 1 && comm->nChannels < 16 && ncclParamUnpackDoubleNChannels()) {
    comm->nChannels = std::min(MAXCHANNELS, comm->nChannels * 2);
  }

  // Honor NCCL_MIN/MAX_CTAS and NCCL_MIN/MAX_NCHANNELS
  // child comm #channels cannot exceed top parent #channels.
  if (comm->sharedRes->owner != comm) {
    minChannels = std::min(std::max(ncclMinNchannels(), comm->config.minCTAs), comm->sharedRes->tpNChannels);
    maxChannels = std::min(std::min(ncclMaxNchannels(), comm->config.maxCTAs), comm->sharedRes->tpNChannels);
  } else {
    minChannels = std::max(ncclMinNchannels(), comm->config.minCTAs);
    maxChannels = std::min(ncclMaxNchannels(), comm->config.maxCTAs);
  }
  comm->nChannels = std::max(minChannels, std::min(comm->nChannels, maxChannels));
  if (comm->nChannels > nSearchChannels) comm->nChannels = copyChannels(comm, nSearchChannels, comm->nChannels, ringPrev, ringNext);
  NCCLCHECKGOTO(connectRings(comm, comm->nChannels, comm->channels, ringPrev, ringNext, nSearchChannels, ringRecv, ringSend), ret, fail);
  NCCLCHECKGOTO(connectTrees(comm, comm->nChannels, comm->channels, nSearchChannels, treeToParent, treeToChild0, treeToChild1, treePatterns), ret, fail);

  // We permit combining max, then min, to only use the first max channels, then duplicate them.
  if (maxChannels < minChannels) {
    comm->nChannels = copyChannels(comm, maxChannels, minChannels, ringPrev, ringNext);
  }

  comm->collChannels = comm->nChannels;
#if CUDART_VERSION >= 12010
  // Support maximal channel usage for aggregation
  if (shared && comm->nvlsChannels > parent->nvlsResources->nChannels) {
    comm->nvlsChannels = parent->nvlsResources->nChannels;
  }
  if (comm->nChannels < comm->nvlsChannels) {
    comm->nChannels = copyChannels(comm, comm->nChannels, comm->nvlsChannels, ringPrev, ringNext);
  }
  NCCLCHECKGOTO(connectNvls(comm, nvlsHeads, minHeadNum), ret, fail);
#endif
  if (shared && comm->nChannels > parent->sharedRes->tpNChannels) {
    comm->nChannels = parent->sharedRes->tpNChannels;
    comm->collChannels = std::min(comm->collChannels, comm->nChannels);
  }

  // Create rings array and check all is fine
  NCCLCHECKGOTO(ncclBuildRings(comm->nChannels, rings, comm->rank, comm->nRanks, ringPrev, ringNext), ret, fail);

exit:
  if (ringRecv) free(ringRecv);
  if (ringSend) free(ringSend);
  if (ringPrev) free(ringPrev);
  if (ringNext) free(ringNext);
  if (treeToParent) free(treeToParent);
  if (treeToChild0) free(treeToChild0);
  if (treeToChild1) free(treeToChild1);
  if (nvlsHeads) free(nvlsHeads);
  return ret;
fail:
  goto exit;
}
