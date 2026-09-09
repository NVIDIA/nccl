/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "comm.h"

ncclResult_t ncclTuningCollnetModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (!comm->ncclCollNet) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr, nullptr));

  // Disable CollNet+Direct on non-NVSwitch systems with multiple local ranks.
  // Keep CollNet+Chain disabled on non-NVSwitch systems.
  if (comm->config.collnetEnable == 0) {
    int nvsCount = 0;
    NCCLCHECK(ncclTopoGetNvsCount(comm->topo, &nvsCount));
    if (nvsCount == 0 &&
        (algo == NCCL_ALGO_COLLNET_CHAIN || (algo == NCCL_ALGO_COLLNET_DIRECT && comm->maxLocalRanks > 1))) {
      memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
      return ncclSuccess;
    }
  }
  if (algo == NCCL_ALGO_COLLNET_CHAIN && comm->collNetChainSupport == 0) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  float busBw = 0.0;
  int nNodes = comm->nNodes;
  int nRanks = comm->nRanks;
  float ppn = (float)nRanks / nNodes;

  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    comm->tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
    if (c != ncclFuncAllGather && c != ncclFuncReduceScatter && c != ncclFuncAllReduce) {
      enabled[c] = 0; // Hard disable
      continue;
    }

    if (c == ncclFuncAllGather || c == ncclFuncReduceScatter) {
      busBw = ppn * std::min(comm->graphs[algo].bwIntra, comm->graphs[algo].bwInter * 0.9f);
    } else {
      busBw = comm->graphs[algo].nChannels * comm->graphs[algo].bwIntra;
      if (algo == NCCL_ALGO_COLLNET_DIRECT) {
        // Collnet+Direct requires all GPUs to have a local NIC to work at full speed
        float factor = ppn / (1.0 * comm->graphs[algo].nChannels); // GPU/NIC ratio
        factor -= (factor - 1) / 2;
        busBw /= factor;
        if (comm->minCompCap >= 90) busBw *= .85;
      }
    }

    if (comm->nNodes > 1) {
      int nHeads = 0;
      if (c == ncclFuncAllGather && (!comm->ncclCollNet || !comm->ncclCollNet->iallgather)) {
        busBw = -1.0;
        enabled[c] = 0; // Hard disable
      }
      if (c == ncclFuncReduceScatter && (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter)) {
        busBw = -1.0;
        enabled[c] = 0; // Hard disable
      }
      if (comm->config.collnetEnable) {
        nHeads = comm->collNetHeadsNum;
      } else {
        busBw = -1.0;
        enabled[c] = 0; // Hard disable
      }
      if (busBw > 0.0f) {
        for (int r = 0; r < comm->nRanks; r++) {
          int node = comm->rankToNode[r];
          if (comm->nodeRanks[node].localRanks > nHeads) {
            comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
            enabled[c] = 0; // Hard disable
            continue;
          }
        }
      } else {
        comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
        enabled[c] = 0; // Hard disable
        continue;
      }
    }

    comm->tuningContext.generalBandwidths[c][algo][proto] = c == ncclFuncAllReduce ? busBw * 0.5 : busBw;
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];

    int intraHw, interHw;
    ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);

    float intraLat = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
    // With ppn=1 latencies are fully exposed, use the Tree network latency
    float interLat =
      comm->nNodes == 1 ? intraLat : comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
    interLat += comm->graphs[algo].latencyInter;
    if (proto == NCCL_PROTO_SIMPLE) interLat += comm->graphs[algo].latencyInter;

    if (algo == NCCL_ALGO_COLLNET_DIRECT) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * (std::min(1, (nRanks / nNodes - 1)) * intraLat + (nRanks / nNodes - 1) * 0.4) +
        interLat;  // Add 0.4 us arity serialization latency
    } else if (algo == NCCL_ALGO_COLLNET_CHAIN) {
      comm->tuningContext.generalLatencies[c][algo][proto] += 2 * (nRanks / nNodes - 1) * intraLat + interLat;
    }
  }

  return ret;
}

ncclResult_t ncclTuningCollnetModelSim(struct ncclTuningInput_t* const inputs,
                                       struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;
  int collnetSupport = inputs->collNetSupport;
  if (!collnetSupport) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  // CollNetDirect is only supported for up to 8 local GPUs
  // Disable CollNet Chain for more than 8 local GPUs
  if (inputs->comm->maxLocalRanks > NCCL_MAX_DIRECT_ARITY + 1) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }

  tuning->timeUs =
    ncclTuningGetTime(inputs, tuning->algo,
                      &inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto],
                      &inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto]);
  return ret;
}
