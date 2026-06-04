 /*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "comm.h"

// Network post overhead in ns (1000 = 1 us)
NCCL_PARAM(NetOverhead, "NET_OVERHEAD", -2);

static float getNetOverhead(struct ncclComm* comm) {
  if (ncclParamNetOverhead() != -2) return ncclParamNetOverhead() * .001;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_INTEL) return 1.0;
  if (comm->cpuArch == NCCL_TOPO_CPU_ARCH_X86 && comm->cpuVendor == NCCL_TOPO_CPU_VENDOR_AMD) return 2.0;
  return 1.0;
}

ncclResult_t ncclTuningRingModelInit(struct ncclComm* comm, int id, int* /*enabled[NCCL_NUM_FUNCTIONS]*/) {
  ncclResult_t ret = ncclSuccess;
  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr));

  int compCapIndex = ncclTuningGetCompCapIndex(comm);
  int index1, index2;
  ncclTuningGetConstantsIndexes(comm, &index1, &index2);
  double llMaxBw = comm->tuningContext.tuningConstants.llMaxBws[index1][index2];
  double perChMaxRingLL128Bw = comm->tuningContext.tuningConstants.perChMaxRingLL128Bws[compCapIndex][index2];
  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    comm->tuningContext.generalLatencies[c][algo][proto] = -1.0;
    comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
    int nSteps = ncclTuningGetNsteps(c, comm->nRanks);
    float busBw =
      (comm->nNodes > 1 ? comm->graphs[algo].bwInter : comm->graphs[algo].bwIntra) * comm->graphs[algo].nChannels;
    if (proto == NCCL_PROTO_LL) {
      busBw = std::min(llMaxBw, busBw * .5);
    }
    if (proto == NCCL_PROTO_LL128)
      busBw = std::min(busBw * (0.92 /*120.0/128.0*/), comm->graphs[algo].nChannels * perChMaxRingLL128Bw);

    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * comm->nRanks / nSteps;

    int intraHw, interHw;
    ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);

    float intraLat = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
    // With ppn=1 latencies are fully exposed, use the Tree network latency
    float interLat =
      comm->nNodes == 1 ? intraLat : comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
    interLat += comm->graphs[algo].latencyInter;
    if (proto == NCCL_PROTO_SIMPLE) interLat += comm->graphs[algo].latencyInter;

    if ((c == ncclFuncReduce || c == ncclFuncBroadcast)) {
      float lat = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
      if (comm->graphs[algo].sameChannels) {
        comm->tuningContext.generalLatencies[c][algo][proto] += lat;
      } else {
        if (proto == NCCL_PROTO_SIMPLE)
          lat =
            comm->tuningContext.tuningConstants
              .hwLatencies[intraHw][NCCL_ALGO_TREE][proto]; // Add some chunk latency, waiting for proper chunk modeling
        comm->tuningContext.generalLatencies[c][algo][proto] += nSteps * lat;
      }
    } else {
      // Inter-node rings still have to launch nsteps * net overhead.
      float netOverhead = 0.0;
      if (comm->nNodes > 1) {
        netOverhead = getNetOverhead(comm);
        if (proto == NCCL_PROTO_SIMPLE) netOverhead *= 3;
      }
      intraLat = std::max(intraLat, netOverhead);
      int nInterSteps = comm->nNodes == 1 ? 0 : c == ncclFuncAllReduce ? 2 * (comm->nNodes - 1) : comm->nNodes - 1;
      comm->tuningContext.generalLatencies[c][algo][proto] +=
        (nSteps - nInterSteps) * intraLat + nInterSteps * interLat;
    }
  }
  return ret;
}

ncclResult_t ncclTuningRingModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ncclSuccess;
  }
  float lat = inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto];
  float bw = inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto];
  if (bw == -1.0f) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ret;
  }
  if (tuning->algo == NCCL_ALGO_RING && tuning->proto == NCCL_PROTO_SIMPLE && inputs->comm->nNodes > 1 &&
      inputs->func == ncclFuncAllReduce && inputs->nBytes / (inputs->comm->nChannels * inputs->comm->nRanks) >= 64) {
    lat *= inputs->comm->minCompCap < 80 ? 1.9 : 1.4; // Plateau effect of ring
  }
  tuning->timeUs = ncclTuningGetTime(inputs, tuning->algo, &lat, &bw);
  return ret;
}
