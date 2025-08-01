/*************************************************************************
 * Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "comm.h"
#include "core.h"
#include "graph.h"
#include "topo.h"
#include "transport.h"
#include "xml.h"
#include <math.h>

NCCL_PARAM(CrossNic, "CROSS_NIC", 2);

// Initialize system->maxBw. This is the per-channel (i.e. per-SM)
// max bw.
static float getMaxBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu, int type) {
  float maxBw = 0.0;
  for (int i=0; i<system->nodes[type].count; i++) {
    struct ncclTopoLinkList* path = gpu->paths[type]+i;
    float bw = path->bw;
    if (path->count == 0) continue;
    maxBw = std::max(maxBw, bw);
  }
  return maxBw;
}
static float getTotalBw(struct ncclTopoSystem* system, struct ncclTopoNode* gpu) {
  float nvlinkBw = 0.0, pciBw = 0.0;
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* link = gpu->links+l;
    if (link->type == LINK_NVL) nvlinkBw += link->bw;
    if (link->type == LINK_PCI) pciBw = link->bw;
  }
  return std::max(pciBw, nvlinkBw);
}
ncclResult_t ncclTopoSearchInit(struct ncclTopoSystem* system) {
  system->maxBw = 0.0;
  system->totalBw = 0.0;
  int inter = system->nodes[NET].count;
  if (inter == 0 && system->nodes[GPU].count == 1) {
    system->maxBw = LOC_BW;
    system->totalBw = LOC_BW;
    return ncclSuccess;
  }
  for (int g=0; g<system->nodes[GPU].count; g++) {
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
    system->maxBw = std::max(system->maxBw, getMaxBw(system, gpu, inter ? NET : GPU));
    system->totalBw = std::max(system->totalBw, getTotalBw(system, gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoComputeCommCPU(struct ncclComm* comm) {
  // We assume there is at least one CPU and that the CPUs have the same
  // architecture and vendor.
  const struct ncclTopoNodeSet* cpus = &comm->topo->nodes[CPU];
  comm->cpuArch = cpus->nodes[0].cpu.arch;
  comm->cpuVendor = cpus->nodes[0].cpu.vendor;
  return ncclSuccess;
}

static ncclResult_t findRevLink(struct ncclTopoNode* node1, struct ncclTopoNode* node2, int type, struct ncclTopoLink** revLink) {
  for (int l=0; l<node2->nlinks; l++) {
    struct ncclTopoLink* link = node2->links+l;
    if (link->remNode == node1 && link->type == type) {
      *revLink = link;
      return ncclSuccess;
    }
  }
  WARN("Could not find rev link for %d/%ld -> %d/%ld", node1->type, node1->id, node2->type, node2->id);
  return ncclInternalError;
}

// This is unfortunately needed since manipulating floats often results in rounding errors.
#define SUB_ROUND(a, b) (a = roundf((a-b)*1000)/1000)

static ncclResult_t followPath(struct ncclTopoLinkList* path, struct ncclTopoNode* start, int maxSteps, float bw, int* steps) {
  float pciBw = bw;
  for (int step=0; step<path->count; step++) {
    struct ncclTopoNode* node = path->list[step]->remNode;
    if (node->type == CPU) {
      // Account for P2P inefficiency through Intel CPU RC
      if (path->type == PATH_PHB && start->type == GPU &&
          node->cpu.arch == NCCL_TOPO_CPU_ARCH_X86 &&
          node->cpu.vendor == NCCL_TOPO_CPU_VENDOR_INTEL) {
        pciBw = INTEL_P2P_OVERHEAD(bw);
      }
    }
  }

  struct ncclTopoNode* node = start;
  for (int step=0; step<maxSteps; step++) {
    struct ncclTopoLink* link = path->list[step];
    struct ncclTopoLink* revLink = NULL;
    float fwBw = link->type == LINK_PCI ? pciBw : bw;
    float revBw = 0;
    if (link->remNode->type == GPU && link->remNode->gpu.cudaCompCap < 80 && start->type != GPU) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw/8;
    }
    if (link->remNode->type == CPU && link->remNode->cpu.arch == NCCL_TOPO_CPU_ARCH_POWER && link->type == LINK_NVL) {
      if (revLink == NULL) NCCLCHECK(findRevLink(node, link->remNode, link->type, &revLink));
      revBw += fwBw;
    }
    // Coverity thinks that revLink could be NULL below.  However, we access it only if revBw is non-0, and the
    // logic of the code is that revBw can become non-0 only if revLink is non-NULL (see the "if" statement right above).
    // coverity[var_deref_op]
    if (link->bw < fwBw || (revBw && revLink->bw < revBw)) { *steps = step; return ncclSuccess; }
    SUB_ROUND(link->bw, fwBw);
    if (revBw) SUB_ROUND(revLink->bw, revBw);
    node = link->remNode;
  }
  *steps = maxSteps;
  return ncclSuccess;
}

// Try to go from node type1/index1 to no type2/index2. mult indicates whether we are counting the bandwidth (1) or undoing (-1).
static ncclResult_t ncclTopoFollowPath(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int type1, int index1, int type2, int index2, float mult, struct ncclTopoNode** node) {
  // First handle easy cases
  *node = system->nodes[type2].nodes+index2;
  if (type1 == -1) return ncclSuccess;
  struct ncclTopoNode* node1 = system->nodes[type1].nodes+index1;
  struct ncclTopoLinkList* path = node1->paths[type2]+index2;
  struct ncclTopoNode* node2 = system->nodes[type2].nodes+index2;
  struct ncclTopoLinkList* revPath = node2->paths[type1]+index1;

  if (path == NULL) {
    WARN("No path computed to go from %s/%d to %s/%d", topoNodeTypeStr[type1], index1, topoNodeTypeStr[type2], index2);
    return ncclInternalError;
  }

  // Now check link type
  *node = NULL;
  int intra = (type1 == GPU || type1 == NVS) && (type2 == GPU || type2 == NVS);
  float bw = intra ? graph->bwIntra : graph->bwInter;
  int type = intra ? graph->typeIntra : graph->typeInter;

  if (path->type >= PATH_DIS) return ncclSuccess;
  if (mult == 1 && (path->type > type)) return ncclSuccess;
  if (mult == 1 && (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) &&
      (revPath->type > type)) return ncclSuccess;

  bw *= mult;

  // Check there is enough bandwidth on paths.
  int step = 0;
  NCCLCHECK(followPath(path, node1, path->count, bw, &step));
  if (step < path->count) goto rewind;

  // Enough bandwidth : return destination node.
  graph->nHops += mult*path->count;
  *node = system->nodes[type2].nodes+index2;
  return ncclSuccess;

rewind:
  // Not enough bandwidth : rewind and exit.
  NCCLCHECK(followPath(path, node1, step, -bw, &step));
  return ncclSuccess;
}

static int gpuPciBw(struct ncclTopoNode* gpu) {
  for (int l=0; l<gpu->nlinks; l++) {
    struct ncclTopoLink* gpuLink = gpu->links+l;
    if (gpuLink->type != LINK_PCI) continue;
    struct ncclTopoNode* pci = gpuLink->remNode;
    for (int l=0; l<pci->nlinks; l++) {
      struct ncclTopoLink* pciLink = pci->links+l;
      if (pciLink->remNode != gpu) continue;
      return std::min(gpuLink->bw, pciLink->bw);
    }
  }
  return -1;
}

/* Choose the order in which we try next GPUs. This is critical for the search
   to quickly converge to the best solution even if it eventually times out. */
struct ncclGpuScore {
  int g;             // Retain the index
  int startIndex;    // Least important
  int intraNhops;
  int intraBw;
  int interNhops;
  int interPciBw;
  int interBw;    // Most important
};

static int cmpScore(const void * g1, const void * g2) {
   struct ncclGpuScore *s1 = (struct ncclGpuScore*)g1;
   struct ncclGpuScore *s2 = (struct ncclGpuScore*)g2;
   int d;
   if ((d = (s2->interBw - s1->interBw))) return d;
   if ((d = (s2->interPciBw - s1->interPciBw))) return d;
   if ((d = (s1->interNhops - s2->interNhops))) return d;
   if ((d = (s2->intraBw - s1->intraBw))) return d;
   if ((d = (s1->intraNhops - s2->intraNhops))) return d;
   return s1->startIndex - s2->startIndex;
}

static int cmpIntraScores(struct ncclGpuScore* scores, int count) {
  int intraBw = scores[0].intraBw;
  int intraNhops = scores[0].intraNhops;
  for (int i=1; i<count; i++) {
    if (scores[i].intraBw != intraBw || scores[i].intraNhops != intraNhops) return 1;
  }
  return 0;
}

static ncclResult_t getGpuIndex(struct ncclTopoSystem* system, int rank, int* index) {
  for (int g=0; g<system->nodes[GPU].count; g++) {
    if (system->nodes[GPU].nodes[g].gpu.rank == rank) {
      *index = g;
      return ncclSuccess;
    }
  }
  WARN("Could not find gpu rank %d", rank);
  return ncclInternalError;
}

static ncclResult_t getNetIndex(struct ncclTopoSystem* system, int64_t id, int* index) {
  for (int n=0; n<system->nodes[NET].count; n++) {
    if (system->nodes[NET].nodes[n].id == id) {
      *index = n;
      return ncclSuccess;
    }
  }
  WARN("Could not find net id %lx", id);
  return ncclInternalError;
}

static ncclResult_t getNetPaths(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoLinkList** netPaths) {
  int64_t netId = graph->inter[graph->nChannels*2];
  int n;
  NCCLCHECK(getNetIndex(system, netId, &n));
  *netPaths=system->nodes[NET].nodes[n].paths[GPU];
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchNextGpuSort(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoNode* gpu, int* next, int* countPtr, int sortNet) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  int ngpus = system->nodes[GPU].count;
  struct ncclTopoLinkList* paths = gpu->paths[GPU];
  struct ncclTopoLinkList* netPaths = NULL;
  if (sortNet) NCCLCHECK(getNetPaths(system, graph, &netPaths));

  struct ncclGpuScore scores[NCCL_TOPO_MAX_NODES];
  memset(scores, 0, ngpus*sizeof(struct ncclGpuScore));
  int start = gpu-system->nodes[GPU].nodes;
  int count = 0;
  for (int i=1; i<ngpus; i++) {
    int g = (start+i)%ngpus;
    if (paths[g].count == 0) continue; // There is no path to that GPU
    if (system->nodes[GPU].nodes[g].used & flag) continue;
    scores[count].g = g;
    scores[count].startIndex = i;
    scores[count].intraNhops = paths[g].count;
    scores[count].intraBw = paths[g].bw;
    if (netPaths) {
      scores[count].interNhops = netPaths[g].count;
      scores[count].interPciBw = gpuPciBw(system->nodes[GPU].nodes+g);
      scores[count].interBw = netPaths[g].bw;
    }
    count++;
  }

  // Sort GPUs
  qsort(scores, count, sizeof(struct ncclGpuScore), cmpScore);

  // Check if all have the same intra-node score in which case we go reverse for sortNet = -1
  if (sortNet == -1 && cmpIntraScores(scores, count) == 0) {
    for (int i=0; i<count; i++) next[i] = scores[count-1-i].g;
  } else {
    for (int i=0; i<count; i++) next[i] = scores[i].g;
  }

  *countPtr = count;

  if (system->nodes[NVS].count) {
    // NVSwitches prefer when we talk to a limited set of peers. Try to use neighbors first.
    int index = gpu-system->nodes[GPU].nodes;
    int i;
    int prevGpu = (index-1+ngpus)%ngpus;
    int nextGpu = (index+1)%ngpus;
    int firstGpus[2];
    int firstGpuCount = 0;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
      firstGpus[0] = nextGpu; firstGpus[1] = prevGpu; firstGpuCount = 2;
    } else if (graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE ||
        graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      firstGpus[0] = prevGpu; firstGpus[1] = nextGpu; firstGpuCount = 2;
    } else {
      firstGpus[0] = nextGpu; firstGpuCount = 1;
    }
    if (nextGpu == prevGpu && firstGpuCount == 2) firstGpuCount = 1;
    int firstGpuRealCount = 0;
    for (int g=0; g<firstGpuCount; g++) {
      for (i=0; i<count && next[i] != firstGpus[g]; i++);
      if (i<count) {
        for (; i>0; i--) next[i] = next[i-1];
        next[0] = firstGpus[g];
        firstGpuRealCount++;
      }
    }
    *countPtr = firstGpuRealCount;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time);

// Try to keep all searchs within one second
#define NCCL_SEARCH_GLOBAL_TIMEOUT (1ULL<<19)
#define NCCL_SEARCH_TIMEOUT (1<<14)
#define NCCL_SEARCH_TIMEOUT_TREE (1<<14)
#define NCCL_SEARCH_TIMEOUT_SAMECHANNELS (1<<8)

#define FORCED_ORDER_PCI 1
#define FORCED_ORDER_REPLAY 2

ncclResult_t ncclTopoReplayGetGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int step, int* g) {
  *g = -1;
  if (graph->nChannels == 0) return ncclInternalError;
  int ngpus = system->nodes[GPU].count;
  int nextRank = graph->intra[(graph->nChannels-1)*ngpus+step+1];
  for (int i=0; i<ngpus; i++) if (system->nodes[GPU].nodes[i].gpu.rank == nextRank) {
    *g = i;
    return ncclSuccess;
  }
  return ncclInternalError;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time);

ncclResult_t ncclTopoSearchTryGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time, int type, int index, int g) {
  const uint64_t flag = 1ULL<<(graph->nChannels);
  struct ncclTopoNode* gpu;
  NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, 1, &gpu));
  if (gpu) {
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, backToNet, backToFirstRank, forcedOrder, time));
    gpu->used ^= flag;
    NCCLCHECK(ncclTopoFollowPath(system, graph, type, index, GPU, g, -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchTryCollnetDirect(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int g, int ngpus, int *time) {
  int fwdg = 0;
  int bwdg = 0;
  struct ncclTopoNode* gpu = NULL;
  float mul = 1.0 / (float)(system->nodes[GPU].count - 1);
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, mul, &gpu));
  } while (gpu && ++fwdg < system->nodes[GPU].count);

  if (gpu != NULL) {
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, mul, &gpu));
    } while (gpu && ++bwdg < system->nodes[GPU].count);
    if (gpu != NULL) {
      // Both directions worked. Now we already have head, so pop the all other intra ranks.
      int step = 1;
      for (int index = 0; index < ngpus; ++index) {
        if (index != g) {
          graph->intra[graph->nChannels * ngpus + step] = system->nodes[GPU].nodes[index].gpu.rank;
          step++;
        }
      }
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (bwdg) {
      bwdg--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, bwdg, GPU, g, -mul, &gpu));
    }
  }
  while (fwdg) {
    fwdg--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, fwdg, -mul, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchTryNvls(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int g, int ngpus, int *time) {
  struct ncclTopoNode* nvs;
  struct ncclTopoNode* gpu;
  int d0=0; // See if there is enough bandwidth for NVS->GPU traffic
  do {
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? 2 : 1, &gpu));
    d0++;
  } while (gpu && d0 < system->nodes[GPU].count);
  if (gpu == NULL) {
    d0--;
  } else {
    int d1=0; // See if there is enough bandwidth for GPU->NVS traffic
    do {
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? 2 : 1, &nvs));
      d1++;
    } while (nvs && d1 < system->nodes[GPU].count);
    if (nvs == NULL) {
      d1--;
    } else { // Both directions worked. Move on to the next path.
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, NULL, ngpus, -1, -1, 0, time));
    }
    while (d1) {
      d1--;
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, d1, NVS, 0, d1 == g ? -2 : -1, &nvs));
    }
  }
  while (d0) {
    d0--;
    NCCLCHECK(ncclTopoFollowPath(system, graph, NVS, 0, GPU, d0, d0 == g ? -2 : -1, &gpu));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoCompareGraphs(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* refGraph, int* copy) {
  // 1. Try to get the same nChannels between Rings and Trees
  if (graph->nChannels < graph->minChannels) return ncclSuccess;

  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) { // NVLS channels correspond to GPUs pulling from NVLS. So the more the better.
    if (graph->nChannels > refGraph->nChannels && graph->nChannels <= system->nodes[GPU].count) *copy = 1;
    if (graph->nChannels*graph->bwInter > refGraph->nChannels*refGraph->bwInter) *copy = 1;
    return ncclSuccess;
  }
  // 2. Try to get better bandwidth
  if (graph->nChannels*graph->bwIntra > refGraph->nChannels*refGraph->bwIntra) {
    *copy = 1;
    return ncclSuccess;
  }
  if (graph->nChannels*graph->bwIntra < refGraph->nChannels*refGraph->bwIntra) return ncclSuccess;

  // 3. Less hops
  if (graph->pattern == refGraph->pattern && graph->crossNic == refGraph->crossNic && graph->nHops < refGraph->nHops) *copy = 1;
  return ncclSuccess;
}

// Add the preferred NICs ordered by GPU first
static ncclResult_t ncclTopoPrefNetsGpuFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES], int* netCount) {
  const int nGpus = (gpu == -1) ? system->nodes[GPU].count : 1;
  int gpuCount = nGpus;
  int gpuIds[NCCL_TOPO_MAX_NODES] = {gpu};
  int firstNets[NCCL_TOPO_MAX_NODES];
  if (gpu == -1)
    for (int g = 0; g < nGpus; g++) gpuIds[g] = g;

  for (int c = 0; c < MAXCHANNELS; c++) {
    for (int g = 0; g < nGpus; g++) {
      if (gpuIds[g] == -1) continue;
      int localNet;
      int64_t netId;
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes + gpuIds[g];
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, &localNet));
      // store the first net found for each GPU in case of duplicates
      if(c == 0) firstNets[g] = localNet;
      // if the NET has already been returned for channel 0, that GPU is done
      if (c > 0 && firstNets[g] == localNet) {
        gpuIds[g] = -1;
        gpuCount--;
        continue;
      }
      // only add it to the list if it doesn't already exist
      int found = 0;
      while (found < (*netCount) && nets[found] != localNet) found++;
      if (found == (*netCount)) nets[(*netCount)++] = localNet;
    }
    if (gpuCount == 0) break;
  }
  return ncclSuccess;
}

// Add the preferred NICs ordered by channels first
static ncclResult_t ncclTopoPrefNetsChannelFirst(struct ncclTopoSystem* system, int gpu, int nets[NCCL_TOPO_MAX_NODES], int* netCount) {
  for (int g = 0; g < system->nodes[GPU].count; g++) {
    if (gpu != -1 && gpu != g) continue;
    int localNetCount = 0, localNets[MAXCHANNELS];
    struct ncclTopoNode* gpu = system->nodes[GPU].nodes + g;
    for (int c = 0; c < MAXCHANNELS; c++) {
      int64_t netId;
      NCCLCHECK(ncclTopoGetLocalNet(system, gpu->gpu.rank, c, &netId, NULL));
      NCCLCHECK(ncclTopoIdToIndex(system, NET, netId, localNets + localNetCount));
      if (localNetCount > 0 && localNets[localNetCount] == localNets[0]) break;
      localNetCount++;
    }
    // Append NICs to list
    for (int i = 0; i < localNetCount; i++) {
      int n = localNets[i];
      int found = 0;
      while (found < (*netCount) && nets[found] != n) found++;
      if (found == (*netCount)) nets[(*netCount)++] = n;
    }
  }
  return ncclSuccess;
}

// Build a sorted list of the NETs to try.
//
// "gpu" can be set to -1 to build a list suitable for all GPUs (search start) or to a given gpu
//  index when trying to get back to the NIC.
//
// The list is built the following way:
// 1. Select NETs starting with those close to GPU(s), based on paths[n].type.
// 2. add other NETs satisfying typeInter but not already in the list.
NCCL_PARAM(ScatterEnable, "MNNVL_SCATTER_NETS_ENABLE", 1);
ncclResult_t ncclTopoSelectNets(struct ncclTopoSystem* system, int typeInter, int gpu, int nets[NCCL_TOPO_MAX_NODES], int* netCountRet) {
  ncclResult_t ret = ncclSuccess;
  int netCount = 0;

  // First add the preferred NETs.
  if (system->nHosts > 1 && ncclParamScatterEnable()) {
    // For MNNVL systems, we sort the devices by GPU first, then by channel
    NCCLCHECK(ncclTopoPrefNetsGpuFirst(system, gpu, nets, &netCount));
  } else {
    // For other systems, we sort the devices by channel first, then by GPU
    NCCLCHECK(ncclTopoPrefNetsChannelFirst(system, gpu, nets, &netCount));
  }

  // Then add others satisfying typeInter
  for (int t=0; t <= typeInter; t++) {
    for (int g = 0; g < system->nodes[GPU].count; g++) {
      if (gpu != -1 && gpu != g) continue;
      int localNetCount = 0, localNets[MAXCHANNELS];
      struct ncclTopoNode* gpu = system->nodes[GPU].nodes+g;
      struct ncclTopoLinkList* paths = gpu->paths[NET];
      for (int n=0; n<system->nodes[NET].count && n<MAXCHANNELS; n++) {
        if (paths[n].type == t) localNets[localNetCount++] = n;
      }
      // Append NICs to list
      for (int i=0; i<localNetCount; i++) {
        int n = localNets[i];
        int found = 0;
        while (found<netCount && nets[found] != n) found++;
        if (found == netCount) nets[netCount++] = n;
      }
    }
  }

  *netCountRet = netCount;
  return ret;
}

ncclResult_t ncclTopoSearchRecGpu(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, struct ncclTopoNode* gpu, int step, int backToNet, int backToFirstRank, int forcedOrder, int *time) {
  if ((*time) <= 0) return ncclSuccess;
  (*time)--;

  int ngpus = system->nodes[GPU].count;
  if (step == ngpus) {
    // Determine whether we found a better solution or not
    int copy = 0;
    graph->nChannels++;
    NCCLCHECK(ncclTopoCompareGraphs(system, graph, saveGraph, &copy));
    if (copy) {
      memcpy(saveGraph, graph, sizeof(struct ncclTopoGraph));
      if (graph->nChannels == graph->maxChannels) *time = -1;
    }
    if (graph->nChannels < graph->maxChannels) {
      NCCLCHECK(ncclTopoSearchRec(system, graph, saveGraph, time));
    }
    graph->nChannels--;
    return ncclSuccess;
  }
  graph->intra[graph->nChannels*ngpus+step] = gpu->gpu.rank;
  int g = gpu - system->nodes[GPU].nodes;
  int nets[NCCL_TOPO_MAX_NODES];
  if (step == backToNet) {
    // first get back to NIC
    if (system->nodes[NET].count) {
      int startNetIndex;
      NCCLCHECK(getNetIndex(system, graph->inter[graph->nChannels*2], &startNetIndex));
      struct ncclTopoNode* startNet = system->nodes[NET].nodes+startNetIndex;
      int netCount;
      NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, g, nets, &netCount));
      for (int i=0; i<netCount; i++) {
        int n = nets[i];
        struct ncclTopoNode* net = system->nodes[NET].nodes+n;
        if (graph->pattern == NCCL_TOPO_PATTERN_TREE && net->id != startNet->id) continue; // Trees are symmetric
        if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2) {
          if (graph->nChannels & 1 && net->id != graph->inter[(graph->nChannels-1)*2]) continue;
        } else {
          if (graph->crossNic == 0 && (net->net.asic != startNet->net.asic || net->net.port != startNet->net.port)) continue;
        }

        // Balanced Tree : count half of the bandwidth on first two GPUs
        int nextBackToNet = -1;
        float bwInterSave = graph->bwInter;
        if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
          // Count half of the bandwidth on each of the first two GPUs
          if (step == 0) nextBackToNet = 1;
          else if (net->id != graph->inter[graph->nChannels*2+1]) continue;
          graph->bwInter /= 2;
        }

        NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, 1, &net));
        graph->bwInter = bwInterSave;
        if (net) {
          graph->inter[graph->nChannels*2+1] = net->id;
          NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, step, nextBackToNet, backToFirstRank, forcedOrder, time));

          if (graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) graph->bwInter /= 2;
          NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, NET, n, -1, &net));
          graph->bwInter = bwInterSave;
        }
      }
    }
  } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    NCCLCHECK(ncclTopoSearchTryNvls(system, graph, saveGraph, g, ngpus, time));
  } else if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
    NCCLCHECK(ncclTopoSearchTryCollnetDirect(system, graph, saveGraph, g, ngpus, time));
  } else if (step < system->nodes[GPU].count-1) {
    // Go to next GPU
    int next[NCCL_TOPO_MAX_NODES];
    int count;
    if (forcedOrder == FORCED_ORDER_PCI) { // Try the PCI order
      next[0] = step+1;
      count = 1;
    } else if (forcedOrder == FORCED_ORDER_REPLAY) { // Try last channel order
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, step, next));
      count = 1;
    } else { // Normal search
      NCCLCHECK(ncclTopoSearchNextGpuSort(system, graph, gpu, next, &count, backToNet == -1 ? 0 : backToNet == step+1 ? 1 : -1 ));
    }
    for (int i=0; i<count; i++) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, step+1, backToNet, backToFirstRank, forcedOrder, time, GPU, g, next[i]));
    }
  } else if (step == backToFirstRank) {
    // Find first GPU and loop back to it
    int p;
    NCCLCHECK(getGpuIndex(system, graph->intra[graph->nChannels*ngpus], &p));
    struct ncclTopoNode* firstGpu;
    NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, 1, &firstGpu));
    if (firstGpu) {
      NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, firstGpu, step+1, backToNet, -1, forcedOrder, time));
      NCCLCHECK(ncclTopoFollowPath(system, graph, GPU, g, GPU, p, -1, &firstGpu));
    }
  } else {
    // Next path
    NCCLCHECK(ncclTopoSearchRecGpu(system, graph, saveGraph, gpu, ngpus, -1, -1, forcedOrder, time));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRecNet(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int backToNet, int backToFirstRank, int* time) {
  const int bw = graph->bwInter;
  int nets[NCCL_TOPO_MAX_NODES];
  int netCount;
  int graphFound = 0;
  NCCLCHECK(ncclTopoSelectNets(system, graph->typeInter, -1, nets, &netCount));
  for (int i=0; i<netCount; i++) {
    if ((graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) && graphFound) break;
    int n = nets[(graph->nChannels+i)%netCount];
    struct ncclTopoNode* net = system->nodes[NET].nodes+n;
    if (graph->collNet && net->net.collSupport == 0) continue;
    if (net->net.bw < bw) continue;
    if (graph->pattern == NCCL_TOPO_PATTERN_RING && graph->crossNic == 2
        && (graph->nChannels & 1) && net->id != graph->inter[(graph->nChannels-1)*2+1]) continue;

    graph->inter[graph->nChannels*2] = net->id;
    graph->latencyInter = net->net.latency;

    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw -= bw;
      }
    }

    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS || graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) {
      // NVLS search only tries to find NIC:GPU combinations to compute the heads.
      if (graph->nChannels < netCount) {
        int gpu = net->net.localGpu;
        if (gpu != -1) {
          int duplicate = 0;
          // check whether there is duplicate head when one GPU connects with multiple NICs
          for (int gc = 0; gc < graph->nChannels; gc++) {
            if (graph->intra[gc * system->nodes[GPU].count] == system->nodes[GPU].nodes[gpu].gpu.rank) {
              duplicate = 1;
              break;
            }
          }
          if (!duplicate) {
            NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, gpu));
            graphFound = 1;
          }
        }
      }
    } else {
      if (graph->nChannels > 0 && graph->sameChannels == 1) {
        // Try to replay the last channel
        int g;
        NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, NET, n, g));
      } else {
        if (graph->nChannels == 0 && system->nodes[NVS].count == 0) {
          // Always try the PCI order first to set a reference, but don't count in the timeout nor let it run for long
          int t = 1 << 10;
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, &t, NET, n, 0));
          if (t == -1) *time = -1;
        }

        // Then try the most local GPUs
        int localGpu = net->net.localGpu;
        if (localGpu != -1) {
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpu));
        }
        int localGpus[NCCL_TOPO_MAX_NODES], localGpuCount, pathType;
        NCCLCHECK(ncclTopoGetLocal(system, NET, n, GPU, localGpus, &localGpuCount, &pathType));
        // if no GPUs are connected, skip this net
        if (pathType == PATH_DIS) continue;
        for (int g = 0; g < localGpuCount; ++g) {
          if (localGpus[g] == localGpu) continue; // We already tried this one
          NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, NET, n, localGpus[g]));
        }
      }
    }

    for (int i=0; i<system->nodes[NET].count; i++) {
      if ((system->nodes[NET].nodes[i].net.asic == net->net.asic) &&
          (system->nodes[NET].nodes[i].net.port == net->net.port)) {
        system->nodes[NET].nodes[i].net.bw += bw;
      }
    }
  }
  return ncclSuccess;
}

/* Search Patterns
 *
 *     Intra-node
 * Ring            : GPU a -> GPU b -> .. -> GPU x -> GPU a
 * (=Split Tree Loop)
 * Tree            : GPU a -> GPU b -> .. -> GPU x
 * (=Split Tree)
 *
 *     Inter-node
 * Ring            : NET n -> GPU a -> GPU b -> .. -> GPU x -> NET n (or m if crossNic)
 * Tree            : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                              `--> NET n (or m if crossNic)
 * Split Tree      : NET n -> GPU a -> GPU b -> .. -> GPU x
 *                                       `--> NET n (or m if crossNic)
 * Split Tree Loop : NET n -> GPU a -> GPU b -> .. -> GPU x -> GPU a
 *                                       `--> NET n (or m if crossNic)
 */
ncclResult_t ncclTopoSearchParams(struct ncclTopoSystem* system, int pattern, int* backToNet, int* backToFirstRank) {
  if (system->nodes[NET].count) {
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToNet = system->nodes[GPU].count-1;
    else if (pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) *backToNet = 1;
    else *backToNet = 0;
    *backToFirstRank = -1;
  } else {
    *backToNet = -1;
    if (pattern == NCCL_TOPO_PATTERN_RING) *backToFirstRank = system->nodes[GPU].count-1;
    else *backToFirstRank = -1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoSearchRec(struct ncclTopoSystem* system, struct ncclTopoGraph* graph, struct ncclTopoGraph* saveGraph, int* time) {
  int backToNet, backToFirstRank;
  NCCLCHECK(ncclTopoSearchParams(system, graph->pattern, &backToNet, &backToFirstRank));
  if (system->nodes[NET].count) {
    // Start from NET
    ncclTopoSearchRecNet(system, graph, saveGraph, backToNet, backToFirstRank, time);
  } else {
    // Intra-node only.
    if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, graph->nChannels));
      return ncclSuccess;
    } else if (graph->nChannels == 0) {
      // Try PCI order first
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_PCI, time, -1, -1, 0));
    } else {
      // Also try to replay previous channel
      int g;
      NCCLCHECK(ncclTopoReplayGetGpu(system, graph, -1, &g));
      NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, FORCED_ORDER_REPLAY, time, -1, -1, g));
    }
    if (graph->sameChannels == 0 || graph->nChannels == 0) {
      // Finally, try all other possibilities unless we are forced to use the same channels
      for (int g=0; g<system->nodes[GPU].count; g++) {
        NCCLCHECK(ncclTopoSearchTryGpu(system, graph, saveGraph, 0, backToNet, backToFirstRank, 0, time, -1, -1, g));
      }
    }
  }
  return ncclSuccess;
}

/************************************/
/* User defined graph from XML file */
/************************************/

struct kvDict kvDictLinkType[] = {
  { "LOC", PATH_LOC },
  { "NVL", PATH_NVL },
  { "NVB", PATH_NVB },
  { "PIX", PATH_PIX },
  { "PXB", PATH_PXB },
  { "P2C", PATH_P2C },
  { "PXN", PATH_PXN },
  { "PHB", PATH_PHB },
  { "SYS", PATH_SYS },
  { NULL, 0 }
};

ncclResult_t ncclTopoGetChannelFromXml(struct ncclXmlNode *xmlChannel, int c, struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  int n=0, g=0;
  for (int s=0; s<xmlChannel->nSubs; s++) {
    struct ncclXmlNode* sub = xmlChannel->subs[s];
    int64_t dev;
    const char* str;
    NCCLCHECK(xmlGetAttrStr(sub, "dev", &str));
    dev = strtol(str, NULL, 16);
    if (strcmp(sub->name, "net") == 0) {
      inter[n++] = dev;
    } else if (strcmp(sub->name, "gpu") == 0) {
      int rank = -1;
      for (int g=0; g<ngpus; g++) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[g].id);
        if (NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[g].gpu.dev) == dev) rank = system->nodes[GPU].nodes[g].gpu.rank;
      }
      if (rank == -1) {
        WARN("XML Import Channel : dev %ld not found.", dev);
        return ncclSystemError;
      }
      intra[g++] = rank;
    }
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXmlSub(struct ncclXmlNode *xmlGraph, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  int id;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "id", &id));
  if (graph->id != id) return ncclSuccess;

  int crossNic;
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "crossnic", &crossNic));
  if (ncclParamCrossNic() == 0 && crossNic == 1) return ncclSuccess;
  graph->crossNic = crossNic;

  NCCLCHECK(xmlGetAttrInt(xmlGraph, "pattern", &graph->pattern));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "nchannels", &graph->nChannels));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedintra", &graph->bwIntra));
  NCCLCHECK(xmlGetAttrFloat(xmlGraph, "speedinter", &graph->bwInter));
  const char* str;
  NCCLCHECK(xmlGetAttr(xmlGraph, "latencyinter", &str));
  if (!str) INFO(NCCL_GRAPH, "latencyinter not found in graph, using 0.0");
  graph->latencyInter = str ? strtof(str, NULL) : 0.0;
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeintra", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeIntra, kvDictLinkType));
  NCCLCHECK(xmlGetAttr(xmlGraph, "typeinter", &str));
  NCCLCHECK(kvConvertToInt(str, &graph->typeInter, kvDictLinkType));
  NCCLCHECK(xmlGetAttrInt(xmlGraph, "samechannels", &graph->sameChannels));
  for (int s=0; s<xmlGraph->nSubs; s++) {
    NCCLCHECK(ncclTopoGetChannelFromXml(xmlGraph->subs[s], s, system, graph));
  }
  *nChannels = xmlGraph->nSubs;
  return ncclSuccess;
}
ncclResult_t ncclTopoGetGraphFromXml(struct ncclXmlNode *xmlGraphs, struct ncclTopoSystem* system, struct ncclTopoGraph* graph, int* nChannels) {
  for (int s=0; s<xmlGraphs->nSubs; s++) {
    NCCLCHECK(ncclTopoGetGraphFromXmlSub(xmlGraphs->subs[s], system, graph, nChannels));
  }
  return ncclSuccess;
}

/* And the reverse : graph->xml */
ncclResult_t ncclTopoGetXmlFromChannel(struct ncclTopoGraph* graph, int c, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlChannel;
  int ngpus = system->nodes[GPU].count;
  int64_t* inter = graph->inter+2*c;
  int* intra = graph->intra+ngpus*c;
  NCCLCHECK(xmlAddNode(xml, parent, "channel", &xmlChannel));
  struct ncclXmlNode* node;
  if (system->nodes[NET].count) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[0]));
  }
  for (int g=0; g<ngpus; g++) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "gpu", &node));
    int64_t dev = -1;
    for (int i=0; i<ngpus; i++) {
      if (system->nodes[GPU].nodes[i].gpu.rank == intra[g]) {
        int systemId = NCCL_TOPO_ID_SYSTEM_ID(system->nodes[GPU].nodes[i].id);
        dev = NCCL_TOPO_ID(systemId, system->nodes[GPU].nodes[i].gpu.dev);
      }
    }
    if (dev == -1) {
      WARN("XML Export Channel : rank %d not found.", intra[g]);
      return ncclInternalError;
    }
    NCCLCHECK(xmlSetAttrLong(node, "dev", dev));
    if (graph->id == 3) break; // NVLS graphs only use the first GPU
  }
  if (system->nodes[NET].count) {
    NCCLCHECK(xmlAddNode(xml, xmlChannel, "net", &node));
    NCCLCHECK(xmlSetAttrLong(node, "dev", inter[1]));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraph(struct ncclTopoGraph* graph, struct ncclTopoSystem* system, struct ncclXml *xml, struct ncclXmlNode* parent) {
  struct ncclXmlNode* xmlGraph;
  NCCLCHECK(xmlAddNode(xml, parent, "graph", &xmlGraph));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "id", graph->id));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "pattern", graph->pattern));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "crossnic", graph->crossNic));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "nchannels", graph->nChannels));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedintra", graph->bwIntra));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "speedinter", graph->bwInter));
  NCCLCHECK(xmlSetAttrFloat(xmlGraph, "latencyinter", graph->latencyInter));
  const char* str;
  NCCLCHECK(kvConvertToStr(graph->typeIntra, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeintra", str));
  NCCLCHECK(kvConvertToStr(graph->typeInter, &str, kvDictLinkType));
  NCCLCHECK(xmlSetAttr(xmlGraph, "typeinter", str));
  NCCLCHECK(xmlSetAttrInt(xmlGraph, "samechannels", graph->sameChannels));
  for (int c=0; c<graph->nChannels; c++) {
    NCCLCHECK(ncclTopoGetXmlFromChannel(graph, c, system, xml, xmlGraph));
  }
  return ncclSuccess;
}
ncclResult_t ncclTopoGetXmlFromGraphs(int ngraphs, struct ncclTopoGraph** graphs, struct ncclTopoSystem* system, struct ncclXml *xml) {
  xml->maxIndex = 0;
  struct ncclXmlNode* xmlGraphs;
  NCCLCHECK(xmlAddNode(xml, NULL, "graphs", &xmlGraphs));
  NCCLCHECK(xmlSetAttrInt(xmlGraphs, "version", NCCL_GRAPH_XML_VERSION));
  for (int g=0; g<ngraphs; g++) {
    NCCLCHECK(ncclTopoGetXmlFromGraph(graphs[g], system, xml, xmlGraphs));
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDupChannels(struct ncclTopoGraph* graph, int ccMin, int ngpus) {
  if (graph->nChannels == 0) return ncclSuccess;
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) return ncclSuccess;
  if (graph->bwIntra < 25.0) return ncclSuccess;
  if (ccMin > 80 && graph->bwIntra < 50.0 && graph->nChannels > 4) return ncclSuccess;

  int dupChannels = std::min(graph->nChannels*2, graph->maxChannels);
  memcpy(graph->intra+graph->nChannels*ngpus, graph->intra, (dupChannels-graph->nChannels)*ngpus*sizeof(int));
  memcpy(graph->inter+graph->nChannels*2,graph->inter, (dupChannels-graph->nChannels)*2*sizeof(int64_t));
  graph->bwIntra /= DIVUP(dupChannels, graph->nChannels);
  graph->bwInter /= DIVUP(dupChannels, graph->nChannels);
  graph->nChannels = dupChannels;
  return ncclSuccess;
}

float speedArrayIntra[] = { 40.0, 30.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0 };
float speedArrayInter[] = { 48.0, 30.0, 28.0, 24.0, 20.0, 18.0, 15.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#define NSPEEDSINTRA (sizeof(speedArrayIntra)/sizeof(float))
#define NSPEEDSINTER (sizeof(speedArrayInter)/sizeof(float))

float sm90SpeedArrayIntra[] = { 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 15.0, 12.0, 11.0, 6.0, 3.0 };
float sm90SpeedArrayInter[] = { 48.0, 45.0, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#define NSPEEDSINTRA_SM90 (sizeof(sm90SpeedArrayIntra)/sizeof(float))
#define NSPEEDSINTER_SM90 (sizeof(sm90SpeedArrayInter)/sizeof(float))

float sm100SpeedArrayIntra[] = { 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 24.0, 20.0, 19.0, 18.0 };
float sm100SpeedArrayInter[] = { 96.0, 48.0, 45.1, 42.0, 40.0, 30.0, 24.0, 22.0, 20.0, 17.5, 15.0, 12.0, 6.0, 3.0, 2.4, 1.2, 0.24, 0.12 };
#define NSPEEDSINTRA_SM100 (sizeof(sm100SpeedArrayIntra)/sizeof(float))
#define NSPEEDSINTER_SM100 (sizeof(sm100SpeedArrayInter)/sizeof(float))

ncclResult_t ncclTopoCompute(ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  int ngpus = system->nodes[GPU].count;
  int crossNic = (system->nodes[NET].count > 1) &&
	 (graph->pattern == NCCL_TOPO_PATTERN_RING ||
          graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE ||
          graph->pattern == NCCL_TOPO_PATTERN_SPLIT_TREE) ? ncclParamCrossNic() : 0;
  graph->crossNic = crossNic == 1 ? 1 : 0;
  graph->bwIntra = graph->bwInter = 0;
  graph->latencyInter = 0;
  int minTypeIntra = PATH_LOC, minTypeInter = PATH_PIX;
  int maxTypeIntra = PATH_SYS, maxTypeInter = PATH_SYS;
  if (ngpus > 1) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, GPU, &minTypeIntra));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, GPU, &maxTypeIntra));
  }
  if (system->nodes[NET].count > 0) {
    NCCLCHECK(ncclTopoGetGpuMinPath(system, NET, &minTypeInter));
    NCCLCHECK(ncclTopoGetGpuMaxPath(system, NET, &maxTypeInter));
    maxTypeIntra = maxTypeInter;
  }

  graph->typeIntra = minTypeIntra;
  graph->typeInter = minTypeInter;
  graph->nChannels = 0;
  int trySameChannels = graph->pattern == NCCL_TOPO_PATTERN_NVLS ? 0 : 1;
  graph->sameChannels = trySameChannels;

  int cpuArch, cpuVendor, cpuModel;
  NCCLCHECK(ncclTopoCpuType(system, &cpuArch, &cpuVendor, &cpuModel));

  const char* str = ncclGetEnv("NCCL_GRAPH_FILE");
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_FILE set by environment to %s", str);
    struct ncclXml* xml;
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECK(ncclTopoGetXmlGraphFromFile(str, xml));
    int nChannels;
    NCCLCHECK(ncclTopoGetGraphFromXml(xml->nodes, system, graph, &nChannels));
    INFO(NCCL_GRAPH, "Search %d : %d channels loaded from XML graph", graph->id, nChannels);
    free(xml);
    if (graph->nChannels > 0) return ncclSuccess;
  }

  int ccMin;
  NCCLCHECK(ncclTopoGetCompCap(system, &ccMin, NULL));
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && (system->nodes[NVS].count == 0 || ccMin < 90)) return ncclSuccess;
  // NVLS and COLLNET_DIRECT search must have ngpus heads at most.
  if (graph->pattern == NCCL_TOPO_PATTERN_NVLS) graph->maxChannels = std::min(NCCL_MAX_NVLS_ARITY, system->nodes[GPU].count);
  if (graph->pattern == NCCL_TOPO_PATTERN_COLLNET_DIRECT) graph->maxChannels = std::min(NCCL_MAX_DIRECT_ARITY+1, system->nodes[GPU].count);

  if (ngpus == 1) if (graph->pattern != NCCL_TOPO_PATTERN_RING) graph->pattern = NCCL_TOPO_PATTERN_TREE;

  if (system->nodes[NET].count == 0 && graph->pattern == NCCL_TOPO_PATTERN_NVLS) {
    // Force intra-node NVLS algorithm to pull evenly from all GPUs.
    graph->minChannels = graph->maxChannels;
  }

  int splitNvLink;
  NCCLCHECK(ncclTopoSplitNvLink(system, &splitNvLink));
  if (graph->pattern == NCCL_TOPO_PATTERN_RING && splitNvLink) {
    // We have two sockets with NVLink and a slower link in between (typically QPI).
    // Tree is likely going to work better but it needs at least 2 channels.
    // Since Tree needs to have the same number of channels as Ring, also force Ring to use 2 channels.
    if (graph->maxChannels >= 2 && graph->minChannels == 1) graph->minChannels = 2;
  }

  struct ncclTopoGraph tmpGraph;
  memcpy(&tmpGraph, graph, sizeof(struct ncclTopoGraph));

  // First try crossnic, then decrease bw and finally increase bwIntra.
  int nspeeds = 0;
  float* speedArray = NULL;
  if (system->nodes[NET].count == 0) {
    nspeeds = ccMin >= 100 ? NSPEEDSINTRA_SM100 : (ccMin >= 90 ? NSPEEDSINTRA_SM90 : NSPEEDSINTRA);
    speedArray = ccMin >= 100 ? sm100SpeedArrayIntra : (ccMin >= 90 ? sm90SpeedArrayIntra : speedArrayIntra);
  } else {
    nspeeds = ccMin >= 100 ? NSPEEDSINTER_SM100 : (ccMin >= 90 ? NSPEEDSINTER_SM90 : NSPEEDSINTER);
    speedArray = ccMin >= 100 ? sm100SpeedArrayInter : (ccMin >= 90 ? sm90SpeedArrayInter : speedArrayInter);
  }
  int pass = 1;
  int speedIndex = 0;
  float maxBw = system->maxBw;
  float totalBw = system->totalBw;
  if (ngpus > 1 && graph->pattern != NCCL_TOPO_PATTERN_RING) totalBw *= ngpus*1.0/(ngpus-1);
  while ((speedArray[speedIndex] > maxBw || speedArray[speedIndex]*graph->minChannels > totalBw) && speedIndex < nspeeds-1) speedIndex++;
  tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
  int64_t globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;

search:
  int time = tmpGraph.sameChannels ? NCCL_SEARCH_TIMEOUT_SAMECHANNELS :
    tmpGraph.pattern == NCCL_TOPO_PATTERN_TREE ? NCCL_SEARCH_TIMEOUT_TREE : NCCL_SEARCH_TIMEOUT;
  tmpGraph.nChannels = 0;
  globalTimeout -= time;

  NCCLCHECK(ncclTopoSearchRec(system, &tmpGraph, graph, &time));
#if 0
  printf("Id %d Pattern %d, crossNic %d, Bw %g/%g, type %d/%d, channels %d-%d sameChannels %d -> nChannels %dx%g/%g %s\n", tmpGraph.id, tmpGraph.pattern, tmpGraph.crossNic, tmpGraph.bwInter, tmpGraph.bwIntra, tmpGraph.typeInter, tmpGraph.typeIntra, tmpGraph.minChannels, tmpGraph.maxChannels, tmpGraph.sameChannels, graph->nChannels, graph->bwInter, graph->bwIntra, time == 0 ? "TIMEOUT" : time == -1 ? "PERFECT" : "");
  for (int c=0; c<graph->nChannels; c++) {
    printf("%2d : ", c);
    for (int g=0; g<ngpus; g++) {
      printf("%d ", graph->intra[c*ngpus+g]);
    }
    printf("[%lx %lx]", graph->inter[c*2+0], graph->inter[c*2+1]);
    printf("\n");
  }
#endif
  // Optimal solution, stop here
  if (time == -1) goto done;
  if (graph->nChannels*graph->bwInter >= system->totalBw) goto done;

  if (pass == 1) {
    // First pass, we don't have a solution yet ; try other options

    // Try having different channels (except when going through AMD CPUs)
    if (tmpGraph.sameChannels == 1 &&
        !(cpuArch == NCCL_TOPO_CPU_ARCH_X86 && cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD && tmpGraph.typeIntra == PATH_SYS)) {
      tmpGraph.sameChannels = 0;
      goto search;
    }
    tmpGraph.sameChannels = trySameChannels;

    if (time != -1) globalTimeout += time;
    else globalTimeout = NCCL_SEARCH_GLOBAL_TIMEOUT;
    if (globalTimeout < 0 && graph->nChannels) goto done;

    // Try a simpler tree
    if (ccMin >= 90 && tmpGraph.pattern == NCCL_TOPO_PATTERN_BALANCED_TREE) {
      tmpGraph.pattern = NCCL_TOPO_PATTERN_TREE;
      goto search;
    }
    tmpGraph.pattern = graph->pattern;

    int maxIntra = system->nodes[NET].count > 0 ? tmpGraph.typeInter : maxTypeIntra;
    if (tmpGraph.typeIntra < maxIntra && (graph->nChannels == 0 || tmpGraph.typeIntra < graph->typeIntra)) {
      tmpGraph.typeIntra += 1;
      if (tmpGraph.typeIntra < PATH_DIS) goto search;
    }
    tmpGraph.typeIntra = minTypeIntra;

    if (system->nodes[NET].count > 0 && tmpGraph.typeInter < maxTypeInter && (graph->nChannels == 0 || tmpGraph.typeInter < graph->typeInter || tmpGraph.typeInter < PATH_PXN)) {
      tmpGraph.typeInter += 1;
      if (tmpGraph.typeInter < PATH_DIS) goto search;
    }
    tmpGraph.typeInter = minTypeInter;

    if (crossNic == 2 && tmpGraph.crossNic == 0
        && (graph->pattern == NCCL_TOPO_PATTERN_RING || graph->pattern == NCCL_TOPO_PATTERN_BALANCED_TREE)) {
      // Try again with crossNic if permitted
      tmpGraph.crossNic = 2;
      goto search;
    }
    tmpGraph.crossNic = crossNic == 1 ? 1 : 0;

    // Decrease bw until we find a solution
    if ((speedIndex < nspeeds-1) && (graph->nChannels == 0 || (speedArray[speedIndex+1]/graph->bwInter > .49))) {
      tmpGraph.bwInter = tmpGraph.bwIntra = speedArray[++speedIndex];
      goto search;
    }
    speedIndex = 0;
    while (speedArray[speedIndex] > maxBw && speedIndex < nspeeds-1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];

  }

done:
  // We have a solution. Start from that solution and move to pass 2.
  if (pass == 1) {
    time = -1;
    NCCLCHECK(ncclTopoDupChannels(graph, ccMin, ngpus));
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
    speedIndex = 0;
    while (speedArray[speedIndex] > graph->bwInter && speedIndex < nspeeds-1) speedIndex++;
    tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[speedIndex];
    tmpGraph.minChannels = graph->nChannels;
    pass = 2;
  }

  if (pass == 2) {
    // See if we can increase bw
    if (time != 0 && speedIndex > 0) {
      if (graph->pattern == NCCL_TOPO_PATTERN_RING) {
        // increase bw for Ring
        tmpGraph.bwIntra = tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (graph->pattern == NCCL_TOPO_PATTERN_NVLS && tmpGraph.bwInter == graph->bwInter && tmpGraph.bwInter < tmpGraph.bwIntra*2) {
        tmpGraph.minChannels = tmpGraph.maxChannels = graph->nChannels;
        tmpGraph.bwInter = speedArray[--speedIndex];
        goto search;
      } else if (tmpGraph.bwIntra == graph->bwIntra && tmpGraph.bwIntra < tmpGraph.bwInter*2) {
        // increase bwIntra for trees (2 nodes or collnet)
        tmpGraph.bwIntra = speedArray[--speedIndex];
        goto search;
      }
    }
    time = -1;
    memcpy(&tmpGraph, graph, sizeof(tmpGraph));
  }

  if (graph->nChannels == 0 && graph->collNet == 0 && graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
    INFO(NCCL_GRAPH, "Could not find a path for pattern %d, falling back to simple order", graph->pattern);
    for (int i=0; i<ngpus; i++) graph->intra[i] = system->nodes[GPU].nodes[i].gpu.rank;
    graph->inter[0] = graph->inter[1] = 0;
    graph->bwIntra = graph->bwInter = 0.1;
    graph->typeIntra = graph->typeInter = PATH_SYS;
    graph->nChannels = 1;
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoPrintGraph(struct ncclTopoSystem* system, struct ncclTopoGraph* graph) {
  INFO(NCCL_GRAPH, "Pattern %d, crossNic %d, nChannels %d, bw %f/%f, type %s/%s, sameChannels %d", graph->pattern, graph->crossNic, graph->nChannels, graph->bwIntra, graph->bwInter, topoPathTypeStr[graph->typeIntra], topoPathTypeStr[graph->typeInter], graph->sameChannels);
  int ngpus = system->nodes[GPU].count;

  char line[1024];
  for (int c=0; c<graph->nChannels; c++) {
    sprintf(line, "%2d :", c);
    int offset = strlen(line);
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c]));
      offset = strlen(line);
    }
    for (int i=0; i<ngpus; i++) {
      int g;
      ncclTopoRankToIndex(system, graph->intra[ngpus * c + i], &g, true);
      int64_t topoId = system->nodes[GPU].nodes[g].id;
      sprintf(line + offset, " %s/%lx-%lx", topoNodeTypeStr[GPU], NCCL_TOPO_ID_SYSTEM_ID(topoId), NCCL_TOPO_ID_LOCAL_ID(topoId));
      offset = strlen(line);
      if (graph->id == 3) break; // NVLS graphs only use the first GPU
    }
    if (system->nodes[NET].count > 0) {
      sprintf(line+offset, " %s/%lx-%lx", topoNodeTypeStr[NET], NCCL_TOPO_ID_SYSTEM_ID(graph->inter[2*c+1]), NCCL_TOPO_ID_LOCAL_ID(graph->inter[2*c+1]));
      offset = strlen(line);
    }
    INFO(NCCL_GRAPH, "%s", line);
  }
  return ncclSuccess;
}

ncclResult_t ncclTopoDumpGraphs(struct ncclTopoSystem* system, int ngraphs, struct ncclTopoGraph** graphs) {
  ncclResult_t ret = ncclSuccess;
  const char* str = ncclGetEnv("NCCL_GRAPH_DUMP_FILE");
  struct ncclXml* xml = NULL;
  if (str) {
    INFO(NCCL_ENV, "NCCL_GRAPH_DUMP_FILE set by environment to %s", str);
    NCCLCHECK(xmlAlloc(&xml, NCCL_GRAPH_XML_MAX_NODES));
    NCCLCHECKGOTO(ncclTopoGetXmlFromGraphs(ngraphs, graphs, system, xml), ret, fail);
    NCCLCHECKGOTO(ncclTopoDumpXmlToFile(str, xml), ret, fail);
  }
exit:
  if (xml) free(xml);
  return ret;
fail:
  goto exit;
}

#include "comm.h"
// NVLS channels aren't compute channels. Find which NIC corresponds to our rank being the head
ncclResult_t getNvlsNetDev(struct ncclComm* comm, struct ncclTopoGraph* graph, int channelId, int64_t* netId) {
  ncclResult_t ret = ncclSuccess;
  int localRanks = comm->topo->nodes[GPU].count;
  int netNum = 0;
  int64_t net[MAXCHANNELS];

  for (int c = 0; c < graph->nChannels; c++) {
    if (graph->intra[c * localRanks] == comm->rank) {
      net[netNum++] = graph->inter[c * 2];
    }
  }
  if (netNum) {
    *netId = net[channelId % netNum];
  } else {
    ret = ncclInternalError;
    goto fail;
  }

exit:
  return ret;
fail:
  WARN("Could not find NIC for rank %d in NVLS graph", comm->rank);
  goto exit;
}

// 0: don't use PXN for P2P, 1: use PXN if needed, 2: use PXN as much as possible to maximize aggregation
NCCL_PARAM(P2pPxnLevel, "P2P_PXN_LEVEL", 2);

ncclResult_t ncclTopoGetNetDev(struct ncclComm* comm, int rank, struct ncclTopoGraph* graph, int channelId, int peerRank, int64_t* id, int* dev, int* proxyRank) {
  int64_t netId = -1;
  int netDev = -1;
  if (graph) {
    // Honor the net device in the graph
    int channel = channelId%graph->nChannels;
    int ngpus = comm->topo->nodes[GPU].count;
    int index = graph->intra[channel*ngpus] == rank ? 0 : 1;
    if (graph->pattern != NCCL_TOPO_PATTERN_NVLS) {
      netId = graph->inter[channel*2+index];
    } else {
      NCCLCHECK(getNvlsNetDev(comm, graph, channelId, &netId));
    }
    NCCLCHECK(ncclTopoIdToNetDev(comm->topo, netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, netId, proxyRank));
  } else if (peerRank == -1) {
    return ncclInternalError;
  } else {
    // Start with our local NIC and local Rank
    NCCLCHECK(ncclTopoGetLocalNet(comm->topo, rank, channelId, &netId, &netDev));
    if (dev) *dev = netDev;
    if (id) *id = netId;
    *proxyRank = rank;

    int pxnLevel = ncclPxnDisable(comm) == 1 ? 0 : ncclParamP2pPxnLevel();
    // See whether we can use the remote rank preferred device.
    if (ncclParamCrossNic() == 0 || (pxnLevel != 0)) {
      // Find local NIC number close to local nvmlDev
      int nvmlDev = comm->peerInfo[peerRank].nvmlDev;
      int localRank;
      if (ncclTopoDevToRank(comm->topo, nvmlDev, &localRank) != ncclSuccess) return ncclSuccess;
      NCCLCHECK(ncclTopoGetLocalNet(comm->topo, localRank, channelId, &netId, &netDev));

      // Check that device exists on our node
      if (ncclParamCrossNic() == 0) {
        if (dev) *dev = netDev;
        if (id) *id = netId;
      }
      if (pxnLevel == 1) {
        int g, n;
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g, /*showWarn=*/true));
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        struct ncclTopoNode* gpu = comm->topo->nodes[GPU].nodes+g;
        if (gpu->paths[NET][n].type <= PATH_PXN) {
          if (dev) *dev = netDev;
          if (id) *id = netId;
          NCCLCHECK(ncclTopoGetIntermediateRank(comm->topo, rank, *dev, proxyRank));
        }
      } else if (pxnLevel == 2) {
        // Check which local GPU corresponds to that NIC and see if we can use PXN.
        int n, g1, g2;
        NCCLCHECK(ncclTopoIdToIndex(comm->topo, NET, netId, &n));
        NCCLCHECK(ncclTopoRankToIndex(comm->topo, rank, &g1, /*showWarn=*/true));
        NCCLCHECK(ncclTopoGetLocalGpu(comm->topo, netId, &g2));
        if (g2 != -1) {
          struct ncclTopoNode* peerGpu = comm->topo->nodes[GPU].nodes+g2;
          int pxnType = ncclParamPxnC2c() ? PATH_P2C : PATH_PXB;
          if (peerGpu->paths[GPU][g1].type <= PATH_NVL && peerGpu->paths[NET][n].type <= pxnType) {
            *proxyRank = peerGpu->gpu.rank;
            if (dev) *dev = netDev;
            if (id) *id = netId;
            return ncclSuccess;
          }
        }
      }
    }
  }
  return ncclSuccess;
}
