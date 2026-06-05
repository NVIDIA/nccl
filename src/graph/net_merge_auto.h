/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_NET_MERGE_AUTO_H_
#define NCCL_NET_MERGE_AUTO_H_

#include "nccl.h"
#include "topo.h"
#include <stdint.h>

#define NCCL_MERGE_AUTO_MAX_RANKS 4096
#define NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE 16
#define NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS 256

struct ncclTopoGraph;
struct ncclTopoSystem;
struct ncclTopoRanks;
struct ncclComm;

enum ncclIbMergeNicsMode {
  NCCL_IB_MERGE_NICS_MODE_UNMERGED = 0,
  NCCL_IB_MERGE_NICS_MODE_MERGED = 1,
  NCCL_IB_MERGE_NICS_MODE_AUTO = 2
};

struct ncclMergeAutoRankToNodeMap {
  int nranks;
  int numNodes;
  int rankToNode[NCCL_MERGE_AUTO_MAX_RANKS];
  int nodeRankCount[2];
  uint64_t nodeHash[2];
  int valid;
};

struct ncclMergeAutoChannelRing {
  int channelId;
  int nRanks;
  const int* ranks;
};

struct ncclMergeAutoChannelSet {
  int nChannels;
  const struct ncclMergeAutoChannelRing* rings;
};

struct ncclMergeAutoCrossEdge {
  int channelId;
  int srcRank;
  int dstRank;
  int srcNode;
  int dstNode;
  int direction; // 0: node0->node1, 1: node1->node0
  int netDev;
  double netBw;
  int nPhysRails;
  int physRails[NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE];
  double physRailBw[NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE];
};

struct ncclMergeAutoMetrics {
  int merge;
  int valid;
  int nEdges;
  int uniqueRails01;
  int uniqueRails10;
  double dirBw01;
  double dirBw10;
  double bidirBw;
  double balance;
  double score;
};

enum ncclMergeAutoTopoCandidateId {
  NCCL_MERGE_AUTO_TOPO_UNMERGED = 0,
  NCCL_MERGE_AUTO_TOPO_MERGED = 1,
  NCCL_MERGE_AUTO_TOPO_COUNT = 2
};

struct ncclMergeAutoTopoCandidate {
  const char* name;
  enum ncclNetMergeView mergeView;
  struct ncclTopoSystem* system;
  struct ncclTopoGraph ringGraph;
  int valid;
};

struct ncclMergeAutoCandidateSummary {
  int valid;
  int nChannels;
};

struct ncclMergeAutoCandidateGlobalMetric {
  int globalValid;
  int globalMinChannels;
  int globalMaxChannels;
  int globalChannelMismatch;
};

struct ncclMergeAutoSelection {
  enum ncclNetMergeView selectedView;
  const char* reason;
};

int ncclIbMergeNicsMode();
bool ncclIbMergeNicsAutoEnabled();
int ncclIbMergeNicsAutoThresholdPct();
bool ncclIbMergeNicsAutoDumpEnabled();
ncclResult_t ncclIbMergeNicsAutoLogEnv();
const char* ncclMergeAutoViewName(enum ncclNetMergeView mergeView);
ncclResult_t ncclMergeAutoCheckTwoNode(struct ncclComm* comm, struct ncclMergeAutoRankToNodeMap* rankToNodeMap, int* isTwoNode);
ncclResult_t ncclMergeAutoExtractGraphChannelRings(
    struct ncclTopoSystem* system,
    const struct ncclTopoGraph* graph,
    struct ncclMergeAutoChannelRing* rings,
    int* rankStorage,
    int maxChannels,
    int maxRanksPerChannel,
    struct ncclMergeAutoChannelSet* out);
ncclResult_t ncclMergeAutoDumpGraphChannelRings(const char* label, struct ncclTopoSystem* system, const struct ncclTopoGraph* graph);
ncclResult_t ncclMergeAutoResolveNetDevForEdge(struct ncclComm* comm, const struct ncclTopoGraph* graph, struct ncclMergeAutoCrossEdge* edge);
ncclResult_t ncclMergeAutoDumpGraphCrossEdges(
    const char* label,
    struct ncclComm* comm,
    struct ncclTopoSystem* system,
    const struct ncclTopoGraph* graph,
    const struct ncclMergeAutoRankToNodeMap* rankToNodeMap);
ncclResult_t ncclMergeAutoDumpGraphCrossEdgesFromComm(const char* label, struct ncclComm* comm, const struct ncclTopoGraph* graph);
ncclResult_t ncclMergeAutoDumpPostsetRingEdges(
    const char* label,
    struct ncclComm* comm,
    const struct ncclTopoGraph* graph,
    struct ncclTopoRanks** allTopoRanks,
    const int* firstRanks,
    int nChannels);
ncclResult_t ncclMergeAutoMapRanksToNodes(int nranks, const uint64_t* rankHostHash, struct ncclMergeAutoRankToNodeMap* map);
ncclResult_t ncclMergeAutoMapRanksToNodesFromComm(struct ncclComm* comm, struct ncclMergeAutoRankToNodeMap* map);
ncclResult_t ncclMergeAutoCheckRuntime(
    struct ncclComm* comm,
    int minNetDeviceCount,
    struct ncclMergeAutoRankToNodeMap* rankToNodeMap,
    int* shouldRun);
ncclResult_t ncclMergeAutoGetCrossNodeEdges(
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoRankToNodeMap* rankToNodeMap,
    struct ncclMergeAutoCrossEdge* edges,
    int maxEdges,
    int* nEdges);
ncclResult_t ncclMergeAutoEvaluateCandidate(
    int merge,
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoCrossEdge* edges,
    int nEdges,
    struct ncclMergeAutoMetrics* metrics);
int ncclMergeAutoPickMergeMode(const struct ncclMergeAutoMetrics* merge0, const struct ncclMergeAutoMetrics* merge1, int thresholdPct);
// Must be called by all ranks in the same fixed order. Candidate system builds may
// run topology XML exchange/fusion internally. ringGraphTemplate must be a clean
// uncomputed ring graph template.
ncclResult_t ncclMergeAutoBuildChannelCandidates(
    struct ncclComm* comm,
    const struct ncclTopoGraph* ringGraphTemplate,
    struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT]);
ncclResult_t ncclMergeAutoSelectView(
    struct ncclComm* comm,
    const struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT],
    struct ncclMergeAutoCandidateGlobalMetric metrics[NCCL_MERGE_AUTO_TOPO_COUNT],
    struct ncclMergeAutoSelection* selection);
void ncclMergeAutoFreeChannelCandidates(struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT]);

#endif
