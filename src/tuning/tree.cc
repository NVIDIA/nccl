 /*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "comm.h"

ncclResult_t ncclTuningTreeModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr));
  int compCapIndex = ncclTuningGetCompCapIndex(comm);
  int index1, index2;
  ncclTuningGetConstantsIndexes(comm, &index1, &index2);
  double llMaxBw = comm->tuningContext.tuningConstants.llMaxBws[index1][index2];
  double perChMaxTreeBw = comm->tuningContext.tuningConstants.perChMaxTreeBws[compCapIndex][index2];
  double perChMaxTreeLL128Bw = comm->tuningContext.tuningConstants.perChMaxTreeLL128Bws[compCapIndex][index2];
  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    if (c != ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
    float busBw = std::min(comm->graphs[algo].bwInter, comm->graphs[algo].bwIntra) * comm->graphs[algo].nChannels;
    if (c == ncclFuncAllReduce) busBw = std::min(busBw * .92, comm->graphs[algo].nChannels * perChMaxTreeBw);
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(busBw * 1.0 / 3.8, llMaxBw);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (comm->nNodes == 1 ? 7.0 / 9.0 : 120.0 / 128.0),
                       comm->graphs[algo].nChannels * perChMaxTreeLL128Bw);
    if (comm->maxTreePattern == NCCL_TOPO_PATTERN_TREE) busBw *= .85;

    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * 0.5;

    int intraHw, interHw;
    ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);

    float intraLat = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
    // With ppn=1 latencies are fully exposed, use the Tree network latency
    float interLat =
      comm->nNodes == 1 ? intraLat : comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
    interLat += comm->graphs[algo].latencyInter;
    if (proto == NCCL_PROTO_SIMPLE) interLat += comm->graphs[algo].latencyInter;

    if (c == ncclFuncAllReduce) {
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        2 * ((comm->nRanks / comm->nNodes - 1) * intraLat + log2i(comm->nNodes) * interLat);
    }
  }
  return ret;
}

ncclResult_t ncclTuningTreeModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;
  if (inputs->func != ncclFuncAllReduce) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  int logSize = log2i(inputs->nBytes >> 6);
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  if (inputs->func == ncclFuncAllReduce && logSize >= 0 && logSize < 23)
    bw *= treeCorrectionFactor[tuning->proto][logSize];
  tuning->timeUs = ncclTuningGetTime(inputs, tuning->algo, &lat, &bw);
  return ret;
}
