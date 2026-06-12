/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "net_merge_auto.h"
#include "comm.h"
#include "bootstrap.h"
#include "debug.h"
#include "graph.h"
#include "param.h"
#include "topo.h"
#include "transport.h"
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern int64_t ncclParamIbMergeNics();

NCCL_PARAM(IbMergeNicsAutoThreshold, "IB_MERGE_NICS_AUTO_THRESHOLD", 110);
NCCL_PARAM(IbMergeNicsAutoDump, "IB_MERGE_NICS_AUTO_DUMP", 0);

int ncclIbMergeNicsMode() {
  return (int)ncclParamIbMergeNics();
}

bool ncclIbMergeNicsAutoEnabled() {
  return ncclIbMergeNicsMode() == NCCL_IB_MERGE_NICS_MODE_AUTO;
}

int ncclIbMergeNicsAutoThresholdPct() {
  return (int)ncclParamIbMergeNicsAutoThreshold();
}

bool ncclIbMergeNicsAutoDumpEnabled() {
  return ncclIbMergeNicsAutoEnabled() && ncclParamIbMergeNicsAutoDump() != 0;
}

ncclResult_t ncclIbMergeNicsAutoLogEnv() {
  if (ncclIbMergeNicsAutoEnabled()) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: env mode=2 threshold=%d dump=%d",
      ncclIbMergeNicsAutoThresholdPct(), ncclIbMergeNicsAutoDumpEnabled() ? 1 : 0);
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoMapRanksToNodesFromComm(struct ncclComm* comm, struct ncclMergeAutoRankToNodeMap* map) {
  if (comm == NULL || map == NULL || comm->peerInfo == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  uint64_t* rankHostHash = (uint64_t*)malloc(sizeof(*rankHostHash) * comm->nRanks);
  if (rankHostHash == NULL) return ncclSystemError;
  for (int r = 0; r < comm->nRanks; r++) rankHostHash[r] = comm->peerInfo[r].hostHash;

  ncclResult_t ret = ncclMergeAutoMapRanksToNodes(comm->nRanks, rankHostHash, map);
  free(rankHostHash);
  return ret;
}

ncclResult_t ncclMergeAutoCheckTwoNode(struct ncclComm* comm, struct ncclMergeAutoRankToNodeMap* rankToNodeMap, int* isTwoNode) {
  if (isTwoNode == NULL) return ncclInvalidArgument;
  *isTwoNode = 0;
  if (!ncclIbMergeNicsAutoEnabled()) return ncclSuccess;
  if (comm == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclSuccess;

  struct ncclMergeAutoRankToNodeMap localRankToNodeMap;
  struct ncclMergeAutoRankToNodeMap* map = rankToNodeMap != NULL ? rankToNodeMap : &localRankToNodeMap;
  NCCLCHECK(ncclMergeAutoMapRanksToNodesFromComm(comm, map));
  *isTwoNode = (map->valid && map->numNodes == 2) ? 1 : 0;
  return ncclSuccess;
}

static bool ncclMergeAutoIsIbNet(struct ncclComm* comm) {
  if (comm == NULL || comm->ncclNet == NULL || comm->ncclNet->name == NULL) return false;
  return strcmp(comm->ncclNet->name, "IB") == 0;
}

ncclResult_t ncclMergeAutoCheckRuntime(
    struct ncclComm* comm,
    int minNetDeviceCount,
    struct ncclMergeAutoRankToNodeMap* rankToNodeMap,
    int* shouldRun) {
  if (shouldRun != NULL) *shouldRun = 0;
  if (!ncclIbMergeNicsAutoEnabled()) return ncclSuccess;
  if (comm == NULL || rankToNodeMap == NULL) return ncclInvalidArgument;

  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) {
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=nranks_unsupported nranks=%d max=%d", comm->nRanks, NCCL_MERGE_AUTO_MAX_RANKS);
    return ncclSuccess;
  }

  ncclResult_t ret = ncclMergeAutoMapRanksToNodesFromComm(comm, rankToNodeMap);
  if (ret != ncclSuccess) return ret;

  if (!rankToNodeMap->valid || rankToNodeMap->numNodes != 2) {
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=not_two_nodes nodes=%d nranks=%d", rankToNodeMap->numNodes, comm->nRanks);
    return ncclSuccess;
  }

  if (!ncclMergeAutoIsIbNet(comm)) {
    const char* netName = (comm->ncclNet != NULL && comm->ncclNet->name != NULL) ? comm->ncclNet->name : "none";
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=non_ib_net net=%s", netName);
    return ncclSuccess;
  }

  if (minNetDeviceCount < 2) {
    if (comm->rank == 0) INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: skipped reason=num_net_devs_lt_2 count=%d", minNetDeviceCount);
    return ncclSuccess;
  }

  if (shouldRun != NULL) *shouldRun = 1;
  if (comm->rank == 0) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: enabled mode=two_node nodes=2 nranks=%d minNetDevs=%d threshold=%d",
      comm->nRanks, minNetDeviceCount, ncclIbMergeNicsAutoThresholdPct());
  }
  return ncclSuccess;
}

const char* ncclMergeAutoViewName(enum ncclNetMergeView mergeView) {
  switch (mergeView) {
  case NCCL_NET_MERGE_VIEW_UNMERGED: return "UNMERGED";
  case NCCL_NET_MERGE_VIEW_MERGED_DEFAULT: return "MERGED_DEFAULT";
  case NCCL_NET_MERGE_VIEW_SUPERSET: return "SUPERSET";
  default: return "UNKNOWN";
  }
}

static void ncclMergeAutoInitChannelCandidates(struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT]) {
  memset(candidates, 0, sizeof(*candidates) * NCCL_MERGE_AUTO_TOPO_COUNT);
  candidates[NCCL_MERGE_AUTO_TOPO_UNMERGED].name = "unmerged";
  candidates[NCCL_MERGE_AUTO_TOPO_UNMERGED].mergeView = NCCL_NET_MERGE_VIEW_UNMERGED;
  candidates[NCCL_MERGE_AUTO_TOPO_MERGED].name = "merged";
  candidates[NCCL_MERGE_AUTO_TOPO_MERGED].mergeView = NCCL_NET_MERGE_VIEW_MERGED_DEFAULT;
}

static ncclResult_t ncclMergeAutoInitCandidateRingGraph(
    const struct ncclTopoGraph* ringGraphTemplate,
    struct ncclTopoGraph* ringGraph) {
  if (ringGraphTemplate == NULL || ringGraph == NULL) return ncclInvalidArgument;
  if (ringGraphTemplate->pattern != NCCL_TOPO_PATTERN_RING || ringGraphTemplate->nChannels != 0) return ncclInvalidArgument;

  memset(ringGraph, 0, sizeof(*ringGraph));
  ringGraph->id = ringGraphTemplate->id;
  ringGraph->pattern = ringGraphTemplate->pattern;
  ringGraph->crossNic = ringGraphTemplate->crossNic;
  ringGraph->collNet = ringGraphTemplate->collNet;
  ringGraph->minChannels = ringGraphTemplate->minChannels;
  ringGraph->maxChannels = ringGraphTemplate->maxChannels;
  return ncclSuccess;
}

void ncclMergeAutoFreeChannelCandidates(struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT]) {
  if (candidates == NULL) return;
  for (int c = 0; c < NCCL_MERGE_AUTO_TOPO_COUNT; c++) {
    if (candidates[c].system != NULL) {
      ncclTopoFree(candidates[c].system);
      candidates[c].system = NULL;
    }
    memset(&candidates[c].ringGraph, 0, sizeof(candidates[c].ringGraph));
    candidates[c].valid = 0;
  }
}

static ncclResult_t ncclMergeAutoComputeOneChannelCandidate(
    struct ncclComm* comm,
    const struct ncclTopoGraph* ringGraphTemplate,
    struct ncclMergeAutoTopoCandidate* candidate) {
  if (comm == NULL || ringGraphTemplate == NULL || candidate == NULL || candidate->system == NULL) return ncclInvalidArgument;

  ncclResult_t ret = ncclTopoComputePaths(candidate->system, comm);
  if (ret != ncclSuccess) return ret;
  ret = ncclTopoTrimSystem(candidate->system, comm);
  if (ret != ncclSuccess) return ret;
  ret = ncclTopoComputePaths(candidate->system, comm);
  if (ret != ncclSuccess) return ret;
  ret = ncclTopoSearchInit(candidate->system);
  if (ret != ncclSuccess) return ret;

  // Build from scalar search constraints only. Candidate graph output must not
  // reuse or depend on the live graph that the formal NCCL path will compute.
  NCCLCHECK(ncclMergeAutoInitCandidateRingGraph(ringGraphTemplate, &candidate->ringGraph));
  ret = ncclTopoCompute(candidate->system, &candidate->ringGraph);
  if (ret != ncclSuccess) return ret;
  candidate->valid = 1;

  if (comm->rank == 0) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: candidate=%s valid=1 pattern=%d channels=%d bwIntra=%.1f bwInter=%.1f typeIntra=%d typeInter=%d crossNic=%d",
      candidate->name, candidate->ringGraph.pattern, candidate->ringGraph.nChannels,
      candidate->ringGraph.bwIntra, candidate->ringGraph.bwInter,
      candidate->ringGraph.typeIntra, candidate->ringGraph.typeInter, candidate->ringGraph.crossNic);
  }
  NCCLCHECK(ncclMergeAutoDumpGraphChannelRings(candidate->name, candidate->system, &candidate->ringGraph));
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoBuildChannelCandidates(
    struct ncclComm* comm,
    const struct ncclTopoGraph* ringGraphTemplate,
    struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT]) {
  if (comm == NULL || ringGraphTemplate == NULL || candidates == NULL) return ncclInvalidArgument;
  if (ringGraphTemplate->pattern != NCCL_TOPO_PATTERN_RING || ringGraphTemplate->nChannels != 0) return ncclInvalidArgument;
  ncclMergeAutoInitChannelCandidates(candidates);
  if (!ncclIbMergeNicsAutoEnabled()) return ncclSuccess;
  if (!ncclMergeAutoIsIbNet(comm)) return ncclSuccess;

  for (int c = 0; c < NCCL_MERGE_AUTO_TOPO_COUNT; c++) {
    struct ncclMergeAutoTopoCandidate* candidate = candidates + c;
    if (comm->rank == 0) {
      INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: build candidate=%s view=%s",
        candidate->name, ncclMergeAutoViewName(candidate->mergeView));
    }

    ncclResult_t ret = ncclTopoGetSystemWithMergeView(comm, &candidate->system, NULL, candidate->mergeView);
    if (ret != ncclSuccess) return ret;

    ret = ncclMergeAutoComputeOneChannelCandidate(comm, ringGraphTemplate, candidate);
    if (ret != ncclSuccess) {
      if (comm->rank == 0) {
        INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: candidate=%s valid=0 reason=channel_build_failed ret=%d",
          candidate->name, ret);
      }
      if (candidate->system != NULL) {
        ncclTopoFree(candidate->system);
        candidate->system = NULL;
      }
      memset(&candidate->ringGraph, 0, sizeof(candidate->ringGraph));
      candidate->valid = 0;
    }
  }
  return ncclSuccess;
}

static void ncclMergeAutoFillCandidateSummary(
    const struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT],
    struct ncclMergeAutoCandidateSummary summaries[NCCL_MERGE_AUTO_TOPO_COUNT]) {
  memset(summaries, 0, sizeof(*summaries) * NCCL_MERGE_AUTO_TOPO_COUNT);
  for (int c = 0; c < NCCL_MERGE_AUTO_TOPO_COUNT; c++) {
    summaries[c].valid = candidates[c].valid ? 1 : 0;
    summaries[c].nChannels = candidates[c].valid ? candidates[c].ringGraph.nChannels : 0;
  }
}

static void ncclMergeAutoComputeGlobalMetrics(
    int nranks,
    const struct ncclMergeAutoCandidateSummary* allSummaries,
    struct ncclMergeAutoCandidateGlobalMetric metrics[NCCL_MERGE_AUTO_TOPO_COUNT]) {
  memset(metrics, 0, sizeof(*metrics) * NCCL_MERGE_AUTO_TOPO_COUNT);
  for (int c = 0; c < NCCL_MERGE_AUTO_TOPO_COUNT; c++) {
    int globalValid = 1;
    int minChannels = INT_MAX;
    int maxChannels = 0;
    for (int r = 0; r < nranks; r++) {
      const struct ncclMergeAutoCandidateSummary* summary = allSummaries + r * NCCL_MERGE_AUTO_TOPO_COUNT + c;
      if (!summary->valid) globalValid = 0;
      if (summary->nChannels < minChannels) minChannels = summary->nChannels;
      if (summary->nChannels > maxChannels) maxChannels = summary->nChannels;
    }
    if (minChannels == INT_MAX) minChannels = 0;
    metrics[c].globalValid = globalValid;
    metrics[c].globalMinChannels = minChannels;
    metrics[c].globalMaxChannels = maxChannels;
    metrics[c].globalChannelMismatch = minChannels != maxChannels ? 1 : 0;
  }
}

static void ncclMergeAutoPickSelectedView(
    const struct ncclMergeAutoCandidateGlobalMetric metrics[NCCL_MERGE_AUTO_TOPO_COUNT],
    struct ncclMergeAutoSelection* selection) {
  const struct ncclMergeAutoCandidateGlobalMetric* unmerged = metrics + NCCL_MERGE_AUTO_TOPO_UNMERGED;
  const struct ncclMergeAutoCandidateGlobalMetric* merged = metrics + NCCL_MERGE_AUTO_TOPO_MERGED;

  // TODO(MergeAuto metric tuning):
  // This is intentionally the simplest first-version metric.
  // Current selection only compares globalMinChannels.
  // Future versions may add:
  //   1. bwInter / bwIntra based score
  //   2. merge penalty for MERGED_DEFAULT
  //   3. mismatch penalty when minChannels != maxChannels
  //   4. rail coverage / HCA usage balance
  //   5. GPU-HCA affinity cost
  //   6. runtime benchmark feedback
  //   7. channel-level or edge-level mixed merge selection
  //
  // Do not add these in the first version.
  // Keep the first implementation deterministic and easy to validate.
  if (unmerged->globalValid && !merged->globalValid) {
    selection->selectedView = NCCL_NET_MERGE_VIEW_UNMERGED;
    selection->reason = "only_unmerged_valid";
  } else if (!unmerged->globalValid && merged->globalValid) {
    selection->selectedView = NCCL_NET_MERGE_VIEW_MERGED_DEFAULT;
    selection->reason = "only_merged_valid";
  } else if (!unmerged->globalValid && !merged->globalValid) {
    selection->selectedView = NCCL_NET_MERGE_VIEW_MERGED_DEFAULT;
    selection->reason = "both_invalid_fallback_default";
  } else if (unmerged->globalMinChannels >= merged->globalMinChannels) {
    selection->selectedView = NCCL_NET_MERGE_VIEW_UNMERGED;
    selection->reason = "unmerged_min_channels_ge_merged";
  } else {
    selection->selectedView = NCCL_NET_MERGE_VIEW_MERGED_DEFAULT;
    selection->reason = "merged_min_channels_gt_unmerged";
  }
}

ncclResult_t ncclMergeAutoSelectView(
    struct ncclComm* comm,
    const struct ncclMergeAutoTopoCandidate candidates[NCCL_MERGE_AUTO_TOPO_COUNT],
    struct ncclMergeAutoCandidateGlobalMetric metrics[NCCL_MERGE_AUTO_TOPO_COUNT],
    struct ncclMergeAutoSelection* selection) {
  if (comm == NULL || candidates == NULL || metrics == NULL || selection == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  memset(selection, 0, sizeof(*selection));
  struct ncclMergeAutoCandidateSummary* allSummaries = (struct ncclMergeAutoCandidateSummary*)malloc(
      sizeof(*allSummaries) * comm->nRanks * NCCL_MERGE_AUTO_TOPO_COUNT);
  if (allSummaries == NULL) return ncclSystemError;
  memset(allSummaries, 0, sizeof(*allSummaries) * comm->nRanks * NCCL_MERGE_AUTO_TOPO_COUNT);
  ncclMergeAutoFillCandidateSummary(candidates, allSummaries + comm->rank * NCCL_MERGE_AUTO_TOPO_COUNT);

  ncclResult_t ret = bootstrapAllGather(comm->bootstrap, allSummaries, sizeof(*allSummaries) * NCCL_MERGE_AUTO_TOPO_COUNT);
  if (ret != ncclSuccess) {
    free(allSummaries);
    return ret;
  }

  ncclMergeAutoComputeGlobalMetrics(comm->nRanks, allSummaries, metrics);
  ncclMergeAutoPickSelectedView(metrics, selection);

  if (comm->rank == 0) {
    for (int c = 0; c < NCCL_MERGE_AUTO_TOPO_COUNT; c++) {
      INFO(NCCL_GRAPH|NCCL_NET, "MergeAutoMetric: candidate=%s globalValid=%d minChannels=%d maxChannels=%d mismatch=%d",
        ncclMergeAutoViewName(candidates[c].mergeView), metrics[c].globalValid,
        metrics[c].globalMinChannels, metrics[c].globalMaxChannels, metrics[c].globalChannelMismatch);
      if (metrics[c].globalChannelMismatch) {
        INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: WARNING candidate=%s channel mismatch minChannels=%d maxChannels=%d",
          ncclMergeAutoViewName(candidates[c].mergeView), metrics[c].globalMinChannels, metrics[c].globalMaxChannels);
      }
    }
    if (!metrics[NCCL_MERGE_AUTO_TOPO_UNMERGED].globalValid && !metrics[NCCL_MERGE_AUTO_TOPO_MERGED].globalValid) {
      INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: WARNING both candidates invalid, fallback view=MERGED_DEFAULT");
    }
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAuto: selected view=%s reason=%s",
      ncclMergeAutoViewName(selection->selectedView), selection->reason);
  }

  free(allSummaries);
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoExtractGraphChannelRings(
    struct ncclTopoSystem* system,
    const struct ncclTopoGraph* graph,
    struct ncclMergeAutoChannelRing* rings,
    int* rankStorage,
    int maxChannels,
    int maxRanksPerChannel,
    struct ncclMergeAutoChannelSet* out) {
  if (system == NULL || graph == NULL || rings == NULL || rankStorage == NULL || out == NULL) return ncclInvalidArgument;
  if (graph->nChannels < 0 || graph->nChannels > maxChannels) return ncclInvalidArgument;
  if (graph->pattern != NCCL_TOPO_PATTERN_RING) return ncclInvalidArgument;
  int ngpus = system->nodes[GPU].count;
  if (ngpus <= 0 || ngpus > maxRanksPerChannel || ngpus > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  out->nChannels = 0;
  out->rings = rings;
  for (int c = 0; c < graph->nChannels; c++) {
    int* ranks = rankStorage + c * maxRanksPerChannel;
    for (int i = 0; i < ngpus; i++) {
      int rank = graph->intra[c * ngpus + i];
      int gpuIndex;
      ncclResult_t ret = ncclTopoRankToIndex(system, rank, &gpuIndex, /*showWarn=*/false);
      if (ret != ncclSuccess) return ret;
      ranks[i] = rank;
    }
    rings[c].channelId = c;
    rings[c].nRanks = ngpus;
    rings[c].ranks = ranks;
    out->nChannels++;
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoDumpGraphChannelRings(const char* label, struct ncclTopoSystem* system, const struct ncclTopoGraph* graph) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (label == NULL) label = "graph";
  if (system == NULL || graph == NULL || graph->nChannels < 0 || graph->nChannels > MAXCHANNELS) return ncclInvalidArgument;
  if (graph->nChannels == 0) return ncclSuccess;

  int ngpus = system->nodes[GPU].count;
  if (ngpus <= 0 || ngpus > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  struct ncclMergeAutoChannelRing* rings = (struct ncclMergeAutoChannelRing*)malloc(sizeof(*rings) * graph->nChannels);
  int* rankStorage = (int*)malloc(sizeof(*rankStorage) * graph->nChannels * ngpus);
  if (rings == NULL || rankStorage == NULL) {
    free(rings);
    free(rankStorage);
    return ncclSystemError;
  }
  struct ncclMergeAutoChannelSet channels;
  ncclResult_t ret = ncclMergeAutoExtractGraphChannelRings(system, graph, rings, rankStorage, graph->nChannels, ngpus, &channels);
  if (ret != ncclSuccess) {
    free(rings);
    free(rankStorage);
    return ret;
  }

  for (int c = 0; c < channels.nChannels; c++) {
    const struct ncclMergeAutoChannelRing* ring = channels.rings + c;
    int lineSize = 64 + ring->nRanks * 16;
    char* line = (char*)malloc(lineSize);
    if (line == NULL) {
      ret = ncclSystemError;
      break;
    }
    int offset = snprintf(line, lineSize, "MergeAutoDump: cand=%s ch=%02d ring=", label, ring->channelId);
    if (offset < 0) {
      free(line);
      ret = ncclSystemError;
      break;
    }
    for (int r = 0; r < ring->nRanks && offset < lineSize; r++) {
      int written = snprintf(line + offset, lineSize - offset, "%s%d", r == 0 ? "" : " ", ring->ranks[r]);
      if (written < 0) {
        free(line);
        ret = ncclSystemError;
        break;
      }
      offset += written;
    }
    if (ret == ncclSuccess) {
      if (offset >= lineSize) snprintf(line + lineSize - 4, 4, "...");
      INFO(NCCL_GRAPH|NCCL_NET, "%s", line);
    }
    free(line);
    if (ret != ncclSuccess) break;
  }
  free(rings);
  free(rankStorage);
  return ret;
}

ncclResult_t ncclMergeAutoResolveNetDevForEdge(struct ncclComm* comm, const struct ncclTopoGraph* graph, struct ncclMergeAutoCrossEdge* edge) {
  if (comm == NULL || graph == NULL || edge == NULL) return ncclInvalidArgument;
  edge->netDev = -1;
  edge->netBw = 1.0;
  edge->nPhysRails = 0;

  int proxyRank;
  int64_t netId;
  int netDev = -1;
  ncclResult_t ret = ncclTopoGetNetDev(comm, edge->srcRank, (struct ncclTopoGraph*)graph, edge->channelId, edge->dstRank, &netId, &netDev, &proxyRank);
  if (ret != ncclSuccess || netDev < 0) return ncclSuccess;
  edge->netDev = netDev;

  if (comm->ncclNet == NULL || comm->ncclNet->getProperties == NULL) return ncclSuccess;
  ncclNetProperties_t props;
  ret = comm->ncclNet->getProperties(netDev, &props);
  if (ret != ncclSuccess) return ncclSuccess;

  int nPhysRails = props.vProps.ndevs;
  if (nPhysRails <= 0) {
    edge->nPhysRails = 1;
    edge->physRails[0] = netDev;
    edge->physRailBw[0] = 1.0;
    return ncclSuccess;
  }
  if (nPhysRails > NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE) nPhysRails = NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE;
  edge->nPhysRails = nPhysRails;
  for (int r = 0; r < nPhysRails; r++) {
    edge->physRails[r] = props.vProps.devs[r];
    edge->physRailBw[r] = 1.0;
  }
  return ncclSuccess;
}

static void ncclMergeAutoFormatPhysRails(const struct ncclMergeAutoCrossEdge* edge, char* buffer, int bufferSize) {
  if (buffer == NULL || bufferSize <= 0) return;
  if (bufferSize < 4) {
    buffer[0] = '\0';
    return;
  }
  int offset = snprintf(buffer, bufferSize, "{");
  if (offset < 0) {
    buffer[0] = '\0';
    return;
  }
  for (int r = 0; r < edge->nPhysRails && offset < bufferSize; r++) {
    int written = snprintf(buffer + offset, bufferSize - offset, "%s%d", r == 0 ? "" : ",", edge->physRails[r]);
    if (written < 0) {
      buffer[0] = '\0';
      return;
    }
    offset += written;
  }
  if (offset < bufferSize) {
    snprintf(buffer + offset, bufferSize - offset, "}");
  } else {
    snprintf(buffer + bufferSize - 4, 4, "...");
  }
}

static void ncclMergeAutoDumpResolvedEdge(const char* label, struct ncclComm* comm, const struct ncclMergeAutoCrossEdge* edge) {
  char phys[256];
  ncclMergeAutoFormatPhysRails(edge, phys, sizeof(phys));
  const char* netName = (comm != NULL && comm->ncclNet != NULL && comm->ncclNet->name != NULL) ? comm->ncclNet->name : "unknown";
  if (edge->netDev >= 0) {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAutoDump: cand=%s ch=%02d edge=%d->%d dir=%d->%d net=%s/%d phys=%s bw=1",
      label, edge->channelId, edge->srcRank, edge->dstRank, edge->srcNode, edge->dstNode, netName, edge->netDev, phys);
  } else {
    INFO(NCCL_GRAPH|NCCL_NET, "MergeAutoDump: cand=%s ch=%02d edge=%d->%d dir=%d->%d net=unknown phys=%s bw=1",
      label, edge->channelId, edge->srcRank, edge->dstRank, edge->srcNode, edge->dstNode, phys);
  }
}

ncclResult_t ncclMergeAutoDumpGraphCrossEdges(
    const char* label,
    struct ncclComm* comm,
    struct ncclTopoSystem* system,
    const struct ncclTopoGraph* graph,
    const struct ncclMergeAutoRankToNodeMap* rankToNodeMap) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (label == NULL) label = "graph";
  if (system == NULL || graph == NULL || rankToNodeMap == NULL || graph->nChannels < 0 || graph->nChannels > MAXCHANNELS) return ncclInvalidArgument;
  if (!rankToNodeMap->valid || rankToNodeMap->numNodes != 2) return ncclSuccess;
  if (graph->nChannels == 0) return ncclSuccess;

  int ngpus = system->nodes[GPU].count;
  if (ngpus <= 0 || ngpus > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  struct ncclMergeAutoChannelRing* rings = (struct ncclMergeAutoChannelRing*)malloc(sizeof(*rings) * graph->nChannels);
  int* rankStorage = (int*)malloc(sizeof(*rankStorage) * graph->nChannels * ngpus);
  struct ncclMergeAutoCrossEdge* edges = (struct ncclMergeAutoCrossEdge*)malloc(sizeof(*edges) * graph->nChannels * ngpus);
  if (rings == NULL || rankStorage == NULL || edges == NULL) {
    free(rings);
    free(rankStorage);
    free(edges);
    return ncclSystemError;
  }

  struct ncclMergeAutoChannelSet channels;
  ncclResult_t ret = ncclMergeAutoExtractGraphChannelRings(system, graph, rings, rankStorage, graph->nChannels, ngpus, &channels);
  int nEdges = 0;
  if (ret == ncclSuccess) {
    ret = ncclMergeAutoGetCrossNodeEdges(&channels, rankToNodeMap, edges, graph->nChannels * ngpus, &nEdges);
  }
  if (ret == ncclSuccess) {
    for (int e = 0; e < nEdges; e++) {
      struct ncclMergeAutoCrossEdge* edge = edges + e;
      if (comm != NULL) {
        ncclResult_t resolveRet = ncclMergeAutoResolveNetDevForEdge(comm, graph, edge);
        if (resolveRet != ncclSuccess) ret = resolveRet;
      }
      if (ret != ncclSuccess) break;
      ncclMergeAutoDumpResolvedEdge(label, comm, edge);
    }
  }

  free(rings);
  free(rankStorage);
  free(edges);
  return ret;
}

ncclResult_t ncclMergeAutoDumpGraphCrossEdgesFromComm(const char* label, struct ncclComm* comm, const struct ncclTopoGraph* graph) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (comm == NULL || graph == NULL) return ncclInvalidArgument;
  if (comm->nRanks <= 0 || comm->nRanks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclSuccess;

  struct ncclMergeAutoRankToNodeMap rankToNodeMap;
  ncclResult_t ret = ncclMergeAutoMapRanksToNodesFromComm(comm, &rankToNodeMap);
  if (ret != ncclSuccess) return ret;
  return ncclMergeAutoDumpGraphCrossEdges(label, comm, comm->topo, graph, &rankToNodeMap);
}

ncclResult_t ncclMergeAutoDumpPostsetRingEdges(
    const char* label,
    struct ncclComm* comm,
    const struct ncclTopoGraph* graph,
    struct ncclTopoRanks** allTopoRanks,
    const int* firstRanks,
    int nChannels) {
  if (!ncclIbMergeNicsAutoDumpEnabled()) return ncclSuccess;
  if (label == NULL) label = "postset";
  if (comm == NULL || graph == NULL || allTopoRanks == NULL || firstRanks == NULL) return ncclInvalidArgument;
  if (nChannels < 0 || nChannels > MAXCHANNELS) return ncclInvalidArgument;
  if (comm->nNodes != 2 || comm->rankToNode == NULL || comm->node < 0 || comm->node >= comm->nNodes) return ncclSuccess;

  int node = comm->node;
  if (firstRanks[node] != comm->rank) return ncclSuccess;
  int nextNode = (node + 1) % comm->nNodes;
  int srcFirstRank = firstRanks[node];
  int dstFirstRank = firstRanks[nextNode];
  if (srcFirstRank < 0 || srcFirstRank >= comm->nRanks || dstFirstRank < 0 || dstFirstRank >= comm->nRanks) return ncclInvalidArgument;
  if (allTopoRanks[srcFirstRank] == NULL || allTopoRanks[dstFirstRank] == NULL) return ncclInvalidArgument;

  for (int c = 0; c < nChannels; c++) {
    int src = allTopoRanks[srcFirstRank]->ringSend[c];
    int dst = allTopoRanks[dstFirstRank]->ringRecv[c];
    if (src < 0 || src >= comm->nRanks || dst < 0 || dst >= comm->nRanks) return ncclInvalidArgument;
    int srcNode = comm->rankToNode[src];
    int dstNode = comm->rankToNode[dst];
    if (srcNode < 0 || srcNode >= comm->nNodes || dstNode < 0 || dstNode >= comm->nNodes) return ncclInvalidArgument;
    if (srcNode == dstNode) continue;

    struct ncclMergeAutoCrossEdge edge;
    memset(&edge, 0, sizeof(edge));
    edge.channelId = c;
    edge.srcRank = src;
    edge.dstRank = dst;
    edge.srcNode = srcNode;
    edge.dstNode = dstNode;
    edge.direction = (srcNode == 0 && dstNode == 1) ? 0 : 1;
    edge.netDev = -1;
    edge.netBw = 1.0;
    NCCLCHECK(ncclMergeAutoResolveNetDevForEdge(comm, graph, &edge));
    ncclMergeAutoDumpResolvedEdge(label, comm, &edge);
  }
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoMapRanksToNodes(int nranks, const uint64_t* rankHostHash, struct ncclMergeAutoRankToNodeMap* map) {
  if (rankHostHash == NULL || map == NULL || nranks <= 0 || nranks > NCCL_MERGE_AUTO_MAX_RANKS) return ncclInvalidArgument;

  memset(map, 0, sizeof(*map));
  map->nranks = nranks;
  map->valid = 0;
  for (int r = 0; r < nranks; r++) map->rankToNode[r] = -1;

  for (int r = 0; r < nranks; r++) {
    int node = -1;
    for (int n = 0; n < map->numNodes; n++) {
      if (map->nodeHash[n] == rankHostHash[r]) {
        node = n;
        break;
      }
    }
    if (node == -1) {
      if (map->numNodes == 2) {
        map->numNodes = 3;
        return ncclSuccess;
      }
      node = map->numNodes++;
      map->nodeHash[node] = rankHostHash[r];
    }
    map->rankToNode[r] = node;
    map->nodeRankCount[node]++;
  }

  map->valid = (map->numNodes == 2 && map->nodeRankCount[0] > 0 && map->nodeRankCount[1] > 0);
  return ncclSuccess;
}

ncclResult_t ncclMergeAutoGetCrossNodeEdges(
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoRankToNodeMap* rankToNodeMap,
    struct ncclMergeAutoCrossEdge* edges,
    int maxEdges,
    int* nEdges) {
  if (channels == NULL || rankToNodeMap == NULL || edges == NULL || nEdges == NULL || maxEdges < 0) return ncclInvalidArgument;
  if (channels->nChannels < 0 || (channels->nChannels > 0 && channels->rings == NULL)) return ncclInvalidArgument;
  if (!rankToNodeMap->valid || rankToNodeMap->numNodes != 2) return ncclInvalidArgument;

  *nEdges = 0;
  for (int c = 0; c < channels->nChannels; c++) {
    const struct ncclMergeAutoChannelRing* ring = channels->rings + c;
    if (ring->nRanks < 0 || (ring->nRanks > 0 && ring->ranks == NULL)) return ncclInvalidArgument;
    if (ring->nRanks < 2) continue;

    for (int i = 0; i < ring->nRanks; i++) {
      int src = ring->ranks[i];
      int dst = ring->ranks[(i + 1) % ring->nRanks];
      if (src < 0 || src >= rankToNodeMap->nranks || dst < 0 || dst >= rankToNodeMap->nranks) return ncclInvalidArgument;
      int srcNode = rankToNodeMap->rankToNode[src];
      int dstNode = rankToNodeMap->rankToNode[dst];
      if (srcNode < 0 || dstNode < 0) return ncclInvalidArgument;
      if (srcNode == dstNode) continue;
      if (srcNode > 1 || dstNode > 1) return ncclInvalidArgument;
      if (*nEdges == maxEdges) return ncclInvalidArgument;

      struct ncclMergeAutoCrossEdge* edge = edges + (*nEdges)++;
      memset(edge, 0, sizeof(*edge));
      edge->channelId = ring->channelId;
      edge->srcRank = src;
      edge->dstRank = dst;
      edge->srcNode = srcNode;
      edge->dstNode = dstNode;
      edge->direction = (srcNode == 0 && dstNode == 1) ? 0 : 1;
      edge->netDev = -1;
      edge->netBw = 1.0;
    }
  }
  return ncclSuccess;
}

struct ncclMergeAutoRailSet {
  int rails[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  double bw[NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS];
  int n;
};

static ncclResult_t ncclMergeAutoRailSetAdd(struct ncclMergeAutoRailSet* set, int rail, double bw) {
  if (set == NULL) return ncclInvalidArgument;
  if (rail < 0) return ncclSuccess;
  if (bw <= 0.0) bw = 1.0;
  for (int i = 0; i < set->n; i++) {
    if (set->rails[i] == rail) {
      if (set->bw[i] < bw) set->bw[i] = bw;
      return ncclSuccess;
    }
  }
  if (set->n == NCCL_MERGE_AUTO_MAX_UNIQUE_RAILS) return ncclInvalidArgument;
  set->rails[set->n] = rail;
  set->bw[set->n] = bw;
  set->n++;
  return ncclSuccess;
}

static double ncclMergeAutoRailSetBw(const struct ncclMergeAutoRailSet* set) {
  double sum = 0.0;
  for (int i = 0; i < set->n; i++) sum += set->bw[i];
  return sum;
}

ncclResult_t ncclMergeAutoEvaluateCandidate(
    int merge,
    const struct ncclMergeAutoChannelSet* channels,
    const struct ncclMergeAutoCrossEdge* edges,
    int nEdges,
    struct ncclMergeAutoMetrics* metrics) {
  if (channels == NULL || edges == NULL || metrics == NULL || nEdges < 0) return ncclInvalidArgument;
  if (channels->nChannels < 0 || (channels->nChannels > 0 && channels->rings == NULL)) return ncclInvalidArgument;
  memset(metrics, 0, sizeof(*metrics));
  metrics->merge = merge;
  metrics->valid = 1;
  metrics->nEdges = nEdges;

  struct ncclMergeAutoRailSet rails[2];
  memset(rails, 0, sizeof(rails));

  for (int e = 0; e < nEdges; e++) {
    const struct ncclMergeAutoCrossEdge* edge = edges + e;
    if (edge->direction != 0 && edge->direction != 1) return ncclInvalidArgument;
    if (edge->nPhysRails < 0 || edge->nPhysRails > NCCL_MERGE_AUTO_MAX_PHYS_RAILS_PER_EDGE) return ncclInvalidArgument;

    int dir = edge->direction;
    if (edge->nPhysRails == 0) {
      int rail = edge->netDev >= 0 ? edge->netDev : edge->channelId * 2 + dir;
      NCCLCHECK(ncclMergeAutoRailSetAdd(&rails[dir], rail, edge->netBw));
      continue;
    }

    for (int r = 0; r < edge->nPhysRails; r++) {
      NCCLCHECK(ncclMergeAutoRailSetAdd(&rails[dir], edge->physRails[r], edge->physRailBw[r]));
    }
  }

  metrics->uniqueRails01 = rails[0].n;
  metrics->uniqueRails10 = rails[1].n;
  metrics->dirBw01 = ncclMergeAutoRailSetBw(&rails[0]);
  metrics->dirBw10 = ncclMergeAutoRailSetBw(&rails[1]);
  double minBw = metrics->dirBw01 < metrics->dirBw10 ? metrics->dirBw01 : metrics->dirBw10;
  double maxBw = metrics->dirBw01 > metrics->dirBw10 ? metrics->dirBw01 : metrics->dirBw10;
  metrics->bidirBw = 2.0 * minBw;
  metrics->balance = maxBw > 0.0 ? minBw / maxBw : 0.0;
  metrics->score = metrics->bidirBw;
  return ncclSuccess;
}

int ncclMergeAutoPickMergeMode(const struct ncclMergeAutoMetrics* merge0, const struct ncclMergeAutoMetrics* merge1, int thresholdPct) {
  if (thresholdPct <= 0) thresholdPct = 110;
  if ((merge0 == NULL || !merge0->valid) && (merge1 == NULL || !merge1->valid)) return NCCL_IB_MERGE_NICS_MODE_MERGED;
  if (merge0 == NULL || !merge0->valid) return NCCL_IB_MERGE_NICS_MODE_MERGED;
  if (merge1 == NULL || !merge1->valid) return NCCL_IB_MERGE_NICS_MODE_UNMERGED;

  if (merge0->score * 100.0 > merge1->score * (double)thresholdPct) return NCCL_IB_MERGE_NICS_MODE_UNMERGED;
  if (merge1->score * 100.0 > merge0->score * (double)thresholdPct) return NCCL_IB_MERGE_NICS_MODE_MERGED;
  return NCCL_IB_MERGE_NICS_MODE_MERGED;
}
