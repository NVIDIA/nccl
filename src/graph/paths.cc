/*************************************************************************
 * Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "core.h"
#include "graph.h"
#include "topo.h"
#include "comm.h"
#include "net.h"

// Pre-compute GPU->NIC, GPU->GPU and NIC->GPU paths

struct ncclTopoNodeList {
  struct ncclTopoNode* list[NCCL_TOPO_MAX_NODES];
  int count;
};

static ncclResult_t getPath(struct ncclTopoSystem* system, struct ncclTopoNode* node, int t, int64_t id, struct ncclTopoLinkList** path) {
  for (int i=0; i<system->nodes[t].count; i++) {
    if (system->nodes[t].nodes[i].id == id) {
      *path = node->paths[t]+i;
      return ncclSuccess;
    }
  }
  WARN("Could not find node of type %d id %lx\n", t, id);
  return ncclInternalError;
}

static ncclResult_t ncclTopoSetPaths(struct ncclTopoNode* baseNode, struct ncclTopoSystem* system) {
  if (baseNode->paths[baseNode->type] == NULL) {
    NCCLCHECK(ncclCalloc(baseNode->paths+baseNode->type, system->nodes[baseNode->type].count));
  }

  // breadth-first search to set all paths to that node in the system
  struct ncclTopoNodeList nodeList;
  struct ncclTopoNodeList nextNodeList;
  nodeList.count = 1; nodeList.list[0] = baseNode;
  nextNodeList.count = 0;
  struct ncclTopoLinkList* basePath;
  NCCLCHECK(getPath(system, baseNode, baseNode->type, baseNode->id, &basePath));
  basePath->count = 0;
  basePath->width = LOC_WIDTH;
  basePath->type = LINK_LOC;

  while (nodeList.count) {
    nextNodeList.count = 0;
    for (int n=0; n<nodeList.count; n++) {
      struct ncclTopoNode* node = nodeList.list[n];
      struct ncclTopoLinkList* path;
      NCCLCHECK(getPath(system, node, baseNode->type, baseNode->id, &path));
      for (int l=0; l<node->nlinks; l++) {
        struct ncclTopoLink* link = node->links+l;
        struct ncclTopoNode* remNode = link->remNode;
        if (remNode->paths[baseNode->type] == NULL) {
          NCCLCHECK(ncclCalloc(remNode->paths+baseNode->type, system->nodes[baseNode->type].count));
        }
        struct ncclTopoLinkList* remPath;
        NCCLCHECK(getPath(system, remNode, baseNode->type, baseNode->id, &remPath));
        int width = std::min(path->width, link->width);
        if (remPath->width < width) {
          // Find reverse link
          for (int l=0; l<remNode->nlinks; l++) {
            if (remNode->links[l].remNode == node) {
              remPath->list[0] = remNode->links+l;
              break;
            }
          }
          if (remPath->list[0] == NULL) {
            WARN("Failed to find reverse path from remNode id %d type %d nlinks %d to node id %d type %d",
                 remNode->id, remNode->type, remNode->nlinks, node->id, node->type);
            return ncclInternalError;
          }
          // Copy the rest of the path
          for (int i=0; i<path->count; i++) remPath->list[i+1] = path->list[i];
          remPath->count = path->count + 1;
          remPath->width = width;

          // Consider the path is QPI when going through the CPU
          // Also don't consider LINK_NET as we only care about the NIC->GPU path.
          int type = remNode->type == CPU ? LINK_QPI : link->type == LINK_NET ? 0 : link->type;
          remPath->type = std::max(path->type, type);

          // Add to the list for the next iteration if not already in the list
          // Disallow GPUs as intermediate steps for now
          if (remNode->type != GPU) {
            int i;
            for (i=0; i<nextNodeList.count; i++) if (nextNodeList.list[i] == remNode) break;
            if (i == nextNodeList.count) nextNodeList.list[nextNodeList.count++] = remNode;
          }
        }
      }
    }
    memcpy(&nodeList, &nextNodeList, sizeof(nodeList));
  }
  return ncclSuccess;
}

static void printNodePaths(struct ncclTopoSystem* system, struct ncclTopoNode* node) {
  char line[1024];
#ifdef ENABLE_TRACE
  INFO(NCCL_GRAPH, "Paths from %s/%lX :", topoNodeTypeStr[node->type], node->id);
#else
  sprintf(line, "%s/%lX :", topoNodeTypeStr[node->type], node->id);
  int offset = strlen(line);
#endif
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    if (node->paths[t] == NULL) continue;
    for (int n = 0; n<system->nodes[t].count; n++) {
#ifdef ENABLE_TRACE
      line[0] = 0;
      int offset = 0;
      for (int i=0; i<node->paths[t][n].count; i++) {
        struct ncclTopoLink* link = node->paths[t][n].list[i];
        struct ncclTopoNode* remNode = link->remNode;
        sprintf(line+offset, "--%s->%s/%lX", topoLinkTypeStr[link->type], topoNodeTypeStr[remNode->type], remNode->id);
        offset = strlen(line);
      }
      INFO(NCCL_GRAPH, "%s (%d)", line, node->paths[t][n].width);
#else
      sprintf(line+offset, "%s/%lX (%d/%d/%d) ", topoNodeTypeStr[t], system->nodes[t].nodes[n].id, node->paths[t][n].count, node->paths[t][n].width, node->paths[t][n].type);
      offset = strlen(line);
#endif
    }
  }
#ifndef ENABLE_TRACE
  INFO(NCCL_GRAPH, "%s", line);
#endif
}

ncclResult_t ncclTopoPrintPaths(struct ncclTopoSystem* system) {
  for (int i=0; i<system->nodes[GPU].count; i++) {
    printNodePaths(system, system->nodes[GPU].nodes+i);
  }
  for (int i=0; i<system->nodes[NET].count; i++) {
    printNodePaths(system, system->nodes[NET].nodes+i);
  }
  return ncclSuccess;
}

static ncclResult_t getLocalCpu(struct ncclTopoSystem* system, int gpu, int* retCpu) {
  // Find the closest CPU to a GPU
  int minHops = 0;
  int localCpu = -1;
  struct ncclTopoLinkList* paths = system->nodes[GPU].nodes[gpu].paths[CPU];
  for (int c=0; c<system->nodes[CPU].count; c++) {
    int hops = paths[c].count;
    if (minHops == 0 || hops < minHops) {
      localCpu = c;
      minHops = hops;
    }
  }
  if (localCpu == -1) {
    WARN("Error : could not find CPU close to GPU %d", gpu);
    return ncclInternalError;
  }
  *retCpu = localCpu;
  return ncclSuccess;
}

static ncclResult_t addCpuStep(struct ncclTopoSystem* system, int c, int t1, int i1, int t2, int i2) {
  struct ncclTopoNode* cpuNode = system->nodes[CPU].nodes+c;
  struct ncclTopoNode* srcNode = system->nodes[t1].nodes+i1;

  int l=0;
  // Node 1 -> CPU
  for (int i=0; i<srcNode->paths[CPU][c].count; i++) srcNode->paths[t2][i2].list[l++] = srcNode->paths[CPU][c].list[i];
  // CPU -> Node 2
  for (int i=0; i<cpuNode->paths[t2][i2].count; i++) srcNode->paths[t2][i2].list[l++] = cpuNode->paths[t2][i2].list[i];

  // Update path characteristics
  srcNode->paths[t2][i2].count = l;
  srcNode->paths[t2][i2].type = LINK_QPI;
  srcNode->paths[t2][i2].width = std::min(srcNode->paths[CPU][c].width, cpuNode->paths[t2][i2].width);
  return ncclSuccess;
}

// Remove/free paths for a given type
static void ncclTopoRemovePathType(struct ncclTopoSystem* system, int nodeType) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
    // Remove links _to_ the given type
    for (int n=0; n<system->nodes[t].count; n++) {
      struct ncclTopoNode* node = system->nodes[t].nodes+n;
      free(node->paths[nodeType]);
      node->paths[nodeType] = NULL;
    }
    // Remove links _from_ the given type
    for (int n=0; n<system->nodes[nodeType].count; n++) {
      struct ncclTopoNode* node = system->nodes[nodeType].nodes+n;
      free(node->paths[t]);
      node->paths[t] = NULL;
    }
  }
}

ncclResult_t ncclTopoComputePaths(struct ncclTopoSystem* system, struct ncclPeerInfo* peerInfos) {
  // Precompute paths between GPUs/NICs.

  // Remove everything in case we're re-computing
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) ncclTopoRemovePathType(system, t);

  // Set direct paths from/to CPUs. We need them in many cases.
  for (int c=0; c<system->nodes[CPU].count; c++) {
    NCCLCHECK(ncclTopoSetPaths(system->nodes[CPU].nodes+c, system));
  }

  // Set direct paths from/to GPUs.
  for (int g=0; g<system->nodes[GPU].count; g++) {
    // Compute paths to GPU g
    NCCLCHECK(ncclTopoSetPaths(system->nodes[GPU].nodes+g, system));

    if (peerInfos == NULL) continue;
    // Update paths from GPUs p to GPU g when we can't or don't want to use P2P or even SHM
    struct ncclPeerInfo* dstInfo = peerInfos+system->nodes[GPU].nodes[g].rank;
    for (int p=0; p<system->nodes[GPU].count; p++) {
      if (p == g) continue;
      struct ncclPeerInfo* srcInfo = peerInfos+system->nodes[GPU].nodes[p].rank;
      int p2p;
      NCCLCHECK(ncclTransports[TRANSPORT_P2P].canConnect(&p2p, system, NULL, srcInfo, dstInfo));
      if (p2p == 0) {
        int shm;
        NCCLCHECK(ncclTransports[TRANSPORT_SHM].canConnect(&shm, system, NULL, srcInfo, dstInfo));
        if (shm == 1) {
          // We cannot use GPU Direct, so we need all traffic to go through a CPU
          int cpu;
          NCCLCHECK(getLocalCpu(system, g, &cpu));
          NCCLCHECK(addCpuStep(system, cpu, GPU, p, GPU, g));
        } else {
          // We cannot communicate with that peer.
          system->nodes[GPU].nodes[p].paths[GPU][g].count = 0;
        }
      }
    }
  }

  // Set direct paths from/to NICs.
  for (int n=0; n<system->nodes[NET].count; n++) {
    struct ncclTopoNode* netNode = system->nodes[NET].nodes+n;
    NCCLCHECK(ncclTopoSetPaths(netNode, system));

    if (peerInfos == NULL) continue;
    for (int g=0; g<system->nodes[GPU].count; g++) {
      if ((peerInfos[system->nodes[GPU].nodes[g].rank].gdrSupport & (1 << n)) == 0) {
        // We cannot use GPU Direct RDMA, so we need all NIC<->GPU paths
        // to go through a CPU
        int localCpu;
        NCCLCHECK(getLocalCpu(system, g, &localCpu));
        NCCLCHECK(addCpuStep(system, localCpu, NET, n, GPU, g));
        NCCLCHECK(addCpuStep(system, localCpu, GPU, g, NET, n));
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t ncclTopoTrimSystem(struct ncclTopoSystem* system, struct ncclComm* comm) {
  int *domains;
  int64_t *ids;
  NCCLCHECK(ncclCalloc(&domains, system->nodes[GPU].count));
  NCCLCHECK(ncclCalloc(&ids, system->nodes[GPU].count));
  int myDomain = 0;
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    domains[g] = g;
    ids[g] = gpu->id;
    for (int p=0; p<g; p++) {
      if (gpu->paths[GPU][p].count > 0) {
        domains[g] = std::min(domains[g], domains[p]);
      }
    }
    if (gpu->rank == comm->rank) myDomain = domains[g];
  }

  int ngpus = system->nodes[GPU].count;
  for (int i=0; i<ngpus; i++) {
    if (domains[i] == myDomain) continue;
    struct ncclTopoNode* gpu = NULL;
    int g;
    for (g=0; g<system->nodes[GPU].count /* This one varies over the loops */; g++) {
      gpu = system->nodes[GPU].nodes+g;
      if (gpu->id == ids[i]) break; else gpu=NULL;
    }
    if (gpu == NULL) {
      WARN("Could not find id %lx", ids[i]);
      free(domains);
      free(ids);
      return ncclInternalError;
    }

    // Remove GPUs I can't access (even indirectly) from my view of the node
    for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
      for (int n=0; n<system->nodes[t].count; n++) {
        struct ncclTopoNode* node = system->nodes[t].nodes+n;
        if (node == gpu) continue;
        for (int l=0; l<node->nlinks; l++) {
          while (l<node->nlinks && node->links[l].remNode == gpu) {
            if (l<node->nlinks-1)
              memmove(node->links+l, node->links+l+1, (node->nlinks-l-1)*sizeof(struct ncclTopoLink));
            node->nlinks--;
          }
          if (l<node->nlinks && node->links[l].remNode->type == GPU && node->links[l].remNode >= gpu) {
            node->links[l].remNode--;
          }
        }
      }
    }
    if (g != system->nodes[GPU].count-1)
      memmove(gpu, gpu+1, (system->nodes[GPU].count-g-1)*sizeof(struct ncclTopoNode));
    system->nodes[GPU].count--;
  }

  comm->localRanks = system->nodes[GPU].count;
  if (system->nodes[GPU].count == comm->nRanks) {
    // Trim network
    ncclTopoRemovePathType(system, NET);
    system->nodes[NET].count = 0;
    for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) {
      for (int n=0; n<system->nodes[t].count; n++) {
        struct ncclTopoNode* node = system->nodes[t].nodes+n;
        for (int l=0; l<node->nlinks; l++) {
          struct ncclTopoLink* link = &(node->links[l]);
          if (link->remNode->type == NET) {
            // Remove the link
            for (int i=l; i<(node->nlinks-1); i++) {
              memcpy(&(node->links[i]), &(node->links[i+1]), sizeof(ncclTopoLink));
            }
            node->nlinks--;
            l--;  // revisit the same value of "l" for the next iteration, since we edited the list in the middle of the loop
          }
        }
      }
    }
  }
  free(domains);
  free(ids);
  return ncclSuccess;
}

static ncclResult_t getGpuSpeed(struct ncclTopoNode* node, int* speed) {
  int nvlSpeed = 0;
  int nvlPeers = 0;
  int pciSpeed = 0;
  for (int l=0; l<node->nlinks; l++) {
    if (node->links[l].type == LINK_NVL) nvlSpeed += node->links[l].width;
    if (node->links[l].remNode->type == GPU) nvlPeers++; else nvlPeers = 2;
    if (node->links[l].type == LINK_PCI) pciSpeed = node->links[l].width;
  }
  *speed = std::min(*speed, std::max(nvlSpeed, pciSpeed));
  return ncclSuccess;
}

ncclResult_t ncclTopoGetMaxSpeed(struct ncclTopoSystem* system) {
  // Compute max speed to try to accelerate the search.
  system->maxSpeed = LOC_WIDTH;

  for (int g=0; g<system->nodes[GPU].count; g++) {
    NCCLCHECK(getGpuSpeed(system->nodes[GPU].nodes+g, &system->maxSpeed));
  }
  if (system->nodes[NET].count) {
    // Try to assign one NIC per GPU
    int netMaxSpeed = 0;
    int netMaxSpeedCount = 0;
    for (int n=0; n<system->nodes[NET].count; n++) {
      int maxSpeed = 0;
      struct ncclTopoNode* net = system->nodes[NET].nodes+n;
      for (int g=0; g<system->nodes[GPU].count; g++) {
        maxSpeed = std::max(maxSpeed, net->paths[GPU][g].width);
      }
      if (maxSpeed > netMaxSpeed) {
        netMaxSpeed = maxSpeed;
        netMaxSpeedCount = 1;
      } else if (maxSpeed == netMaxSpeed) {
        netMaxSpeedCount++;
      }
    }
    system->maxSpeed = std::min(system->maxSpeed, netMaxSpeedCount*NET_WIDTH);
  }
  return ncclSuccess;
}

void ncclTopoFree(struct ncclTopoSystem* system) {
  for (int t=0; t<NCCL_TOPO_NODE_TYPES; t++) ncclTopoRemovePathType(system, t);
  free(system);
}
