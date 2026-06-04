/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "comm.h"
#include "cost_model.h"

NCCL_PARAM(PatEnable, "PAT_ENABLE", 2);
static int ncclPatEnable(struct ncclComm* comm) {
  int patEnable = ncclParamPatEnable();
  if (comm->minCompCap < 60) return 0; // Need SM60 or higher for CUDA atomics
  if (patEnable != 2) return patEnable;
  if (comm->nNodes != comm->nRanks) return 0; // PAT only supports 1 GPU per node
  if (comm->netDeviceType != NCCL_NET_DEVICE_HOST) return 0;   // PAT doesn't support net device offload
  return 1;
}

ncclResult_t ncclTuningPatModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]) {
  ncclResult_t ret = ncclSuccess;
  if (ncclPatEnable(comm) == 0) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr));
  if (proto != NCCL_PROTO_SIMPLE) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }
  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    if (c != ncclFuncReduceScatter && c != ncclFuncAllGather) {
      comm->tuningContext.generalLatencies[c][algo][proto] = -1.0;
      comm->tuningContext.generalBandwidths[c][algo][proto] = -1.0;
      enabled[c] = 0; // Hard disable
      continue;
    }
    float busBw = std::min(comm->graphs[algo].bwInter, comm->graphs[algo].bwIntra) * comm->graphs[algo].nChannels;
    comm->tuningContext.generalBandwidths[c][algo][proto] = busBw * 0.75;
    comm->tuningContext.generalLatencies[c][algo][proto] =
      comm->tuningContext.tuningConstants.baseLatencies[algo][proto];
    int interHw;
    ncclTuningGetHwIndexes(comm, algo, nullptr, &interHw);
    float interLat = comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
    if (proto == NCCL_PROTO_SIMPLE) interLat += comm->graphs[algo].latencyInter;
    comm->tuningContext.generalLatencies[c][algo][proto] +=
      log2i(comm->nNodes) * (interLat / 3.5) // Log latency
      + comm->nRanks * 2.8; // Still a linear part; hopefully we'll manage to remove it at some point.
  }
  return ret;
}

ncclResult_t ncclTuningPatModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning) {
  ncclResult_t ret = ncclSuccess;
  if (inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto] == -1.0f) {
    tuning->valid = 0;
    return ret;
  }
  tuning->timeUs =
    ncclTuningGetTime(inputs, tuning->algo,
                      &inputs->comm->tuningContext.generalLatencies[inputs->func][tuning->algo][tuning->proto],
                      &inputs->comm->tuningContext.generalBandwidths[inputs->func][tuning->algo][tuning->proto]);
  return ret;
}
