/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "comm.h"

// NVLS efficiency factor.
static const float nvlsEfficiency[NCCL_NUM_COMPCAPS] = {
  0.0f, // Volta
  0.0f, // Ampere
  0.85f, // Hopper
  0.74f, // Blackwell
};

ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!comm->nvlsSupport) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr));

  if ((algo == NCCL_ALGO_NVLS || algo == NCCL_ALGO_NVLS_TREE) && (proto != NCCL_PROTO_SIMPLE)) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  if (comm->nNodes == 1 && algo == NCCL_ALGO_NVLS_TREE) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  if (comm->config.collnetEnable == 0 && algo == NCCL_ALGO_NVLS && comm->nNodes > 1) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  int compCapIndex = ncclTuningGetCompCapIndex(comm);
  float bw = comm->nNodes <= 2 ? comm->graphs[algo].bwIntra : comm->graphs[algo].bwInter;
  if (comm->graphs[algo].nChannels < 2) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
  int index1, index2;
  ncclTuningGetConstantsIndexes(comm, &index1, &index2);
  double perChMaxNVLSTreeBw = comm->tuningContext.tuningConstants.perChMaxNVLSTreeBws[compCapIndex][index2];
  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    comm->tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
    // NVLS is supported for AR, AG, and RS.
    // NVLSTree is only supported for AR.
    if (!(c == ncclFuncAllReduce ||
          ((c == ncclFuncAllGather || c == ncclFuncReduceScatter) && algo == NCCL_ALGO_NVLS))) {
      enabled[c] = 0; // Hard disable
      continue;
    }
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float intraBw = comm->graphs[algo].bwIntra * nvlsEfficiency[compCapIndex] * (comm->graphs[algo].nChannels - 1) /
                    comm->graphs[algo].nChannels;
    if (c == ncclFuncAllReduce) {
      intraBw *= 2.0f;
    } else {
      float ppn = comm->minLocalRanks;
      intraBw *= (ppn - 1) / ppn;
    }
    float interBw = comm->graphs[algo].bwInter * ((comm->nNodes <= 2 && algo == NCCL_ALGO_NVLS_TREE) ? 2 : 1);
    bw = std::min({intraBw, interBw,
                   algo == NCCL_ALGO_NVLS_TREE ? (float)perChMaxNVLSTreeBw : std::numeric_limits<float>::max()});
    bw = bw * comm->graphs[algo].nChannels;

    if (comm->nNodes > 1 && algo == NCCL_ALGO_NVLS && (c == ncclFuncAllGather || c == ncclFuncReduceScatter)) {
      int nHeads = 0;
      if (c == ncclFuncAllGather && (!comm->ncclCollNet || !comm->ncclCollNet->iallgather)) {
        bw = -1.0;
        enabled[c] = 0; // Hard disable
      }
      if (c == ncclFuncReduceScatter && (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter)) {
        bw = -1.0;
        enabled[c] = 0; // Hard disable
      }
      if (comm->config.collnetEnable) {
        nHeads = comm->collNetHeadsNum;
      } else {
        bw = -1.0;
        enabled[c] = 0; // Hard disable
      }
      if (bw > 0.0f) {
        for (int r = 0; r < comm->nRanks; r++) {
          int node = comm->rankToNode[r];
          if (comm->nodeRanks[node].localRanks > nHeads) {
            bw = -1.0;
            enabled[c] = 0; // Hard disable
            break;
          }
        }
      }
    }

    comm->tuningContext.generalBandwidths[c][algo][proto] = bw * comm->nRanks / nSteps;
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];

    int intraHw, interHw;
    ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
    float intraLat = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
    // With ppn=1 latencies are fully exposed, use the Tree network latency
    float interLat =
      comm->nNodes == 1 ? intraLat : comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
    interLat += comm->graphs[algo].latencyInter;
    // Also add the flush extra latency
    if (proto == NCCL_PROTO_SIMPLE) interLat += comm->graphs[algo].latencyInter;
    if (algo == NCCL_ALGO_NVLS) {
      comm->tuningContext.generalLatencies[c][algo][proto] = intraLat;
      if (comm->nNodes > 1) comm->tuningContext.generalLatencies[c][algo][proto] += interLat;
    } else if (algo == NCCL_ALGO_NVLS_TREE) {
      comm->tuningContext.generalLatencies[c][algo][proto] += intraLat + 2 * log2i(comm->nNodes) * interLat;
    }
  }
  return ret;
}

ncclResult_t ncclTuningNvlsModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  if (bw == -1.0f) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  int nvlsSupport = inputs->nvlsSupport;
  if (!nvlsSupport) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (inputs->func != ncclFuncAllReduce && inputs->comm->localRanks > NCCL_MAX_NVLS_ARITY) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (tuning->algo == NCCL_ALGO_NVLS && !inputs->collNetSupport && inputs->comm->nNodes > 1) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }

  int logSize = log2i(inputs->nBytes >> 6);
  if (tuning->algo == NCCL_ALGO_NVLS_TREE && inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize < 24 &&
      inputs->comm->minCompCap >= 100 && inputs->comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86)
    bw *= treeCorrectionFactor[tuning->proto][logSize];
  tuning->timeUs = ncclTuningGetTime(inputs, tuning->algo, &lat, &bw);
  return ret;
}
