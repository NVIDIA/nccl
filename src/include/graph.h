/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_GRAPH_H_
#define NCCL_GRAPH_H_

#include "nccl.h"
#include "devcomm.h"
#include <limits.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdio.h>

enum ncclPathDist {
  PATH_PIX  = 0,
  PATH_PXB  = 1,
  PATH_PHB  = 2,
  PATH_NODE = 3,
  PATH_SYS  = 4,
  PATH_ARRAY_SIZE = 5
};

extern const char* pathDists[PATH_ARRAY_SIZE];

ncclResult_t ncclTopoCudaPath(int cudaDev, char** path);

struct ncclTopoSystem;
// Build the topology
ncclResult_t ncclTopoGetSystem(struct ncclComm* comm, struct ncclTopoSystem** system);
ncclResult_t ncclTopoSortSystem(struct ncclTopoSystem* system);
ncclResult_t ncclTopoPrint(struct ncclTopoSystem* system);

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclPeerInfo* info);
void ncclTopoFree(struct ncclTopoSystem* system);
ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm);
ncclResult_t ncclTopoGetMaxSpeed(struct ncclTopoSystem* system);

// Query topology
ncclResult_t ncclTopoGetNvlink(struct ncclTopoSystem* system, int64_t busId1, int64_t busId2, int* nvlink);
ncclResult_t ncclTopoHasNvlink(struct ncclTopoSystem* system, int64_t busId, int* nvlink);
ncclResult_t ncclTopoGpuDistance(struct ncclTopoSystem* system, int64_t busId1, int64_t busId2, int* distance);
ncclResult_t ncclTopoGetNetDev(struct ncclTopoGraph* graph, int dir, int channelId, int* net);
ncclResult_t ncclTopoNetDistance(struct ncclTopoSystem* system, int64_t busId, int netDev, int* distance);
ncclResult_t ncclTopoCpuCount(struct ncclTopoSystem* system, int* count);

#define NCCL_TOPO_MAX_NODES 256

#define NCCL_TOPO_PATTERN_SPLIT_TREE_LOOP 1 // Split tree (send/recv from different ranks) always flowing in the same direction
#define NCCL_TOPO_PATTERN_SPLIT_TREE 2      // Split tree (send/recv from different ranks) flowing in both directions
#define NCCL_TOPO_PATTERN_TREE 3            // Simple tree (send/recv from same rank) flowing in both directions
#define NCCL_TOPO_PATTERN_RING 4            // Ring
struct ncclTopoGraph {
  // Input / output
  int pattern;
  int crossNic;
  // Output
  int nChannels;
  int speedIntra;
  int speedInter;
  int type;
  int nvlink;
  int sameChannels;
  int nHops;
  int intra[MAXCHANNELS*NCCL_TOPO_MAX_NODES];
  int inter[MAXCHANNELS*2];
};
ncclResult_t ncclTopoCompute(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph);

struct ncclTopoRanks {
  int ringRecv[MAXCHANNELS];
  int ringSend[MAXCHANNELS];
  int ringPrev[MAXCHANNELS];
  int ringNext[MAXCHANNELS];
  int treeUpRecv[MAXCHANNELS];
  int treeUpSend[MAXCHANNELS];
  int treeDnRecv[MAXCHANNELS];
  int treeDnSend[MAXCHANNELS];
};

ncclResult_t ncclTopoPreset(struct ncclComm* comm,
    struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph,
    struct ncclTopoRanks* topoRanks);

ncclResult_t ncclTopoPostset(struct ncclComm* comm, int* firstRanks,
    struct ncclTopoRanks** allTopoRanks, int* rings);

ncclResult_t ncclSetThresholds(struct ncclComm* comm, int minCompCap, int maxCompCap, struct ncclTopoGraph* treeGraph, struct ncclTopoGraph* ringGraph);

#endif
