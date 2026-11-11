 /*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "tuning.h"
#include "../cost_model.h"
#include "../model.h"
#include "../tuning_int.h"
#include "pipeline_model.h"

static float computeNVLSMaxBusBw(struct ncclComm* comm, ncclFunc_t func, int algo, int proto) {
  const struct ncclTopoGraph* graph = &comm->graphs[algo];
  int compCapIndex = ncclTuningGetCompCapIndex(comm);

  int ctasPerChannel = comm->nvlsChannels / graph->nChannels;
  float intraBw =
    graph->bwIntra * nvlsEfficiency[compCapIndex] * (graph->nChannels - 1) / (graph->nChannels * ctasPerChannel);
  if (func == ncclFuncAllReduce) {
    intraBw *= 2.0f;
  } else {
    float ppn = comm->minLocalRanks;
    intraBw *= (ppn - 1) / ppn;
  }
  float interBw = graph->bwInter * ((comm->nNodes <= 2 && algo == NCCL_ALGO_NVLS_TREE) ? 2 : 1) / ctasPerChannel;
  intraBw = ncclTuningProtoBW(comm, algo, proto, intraBw);
  interBw = ncclTuningProtoBW(comm, algo, proto, interBw);
  float bw = std::min({
    intraBw,
    interBw,
  });
  return bw * graph->nChannels * ctasPerChannel;
}

static float nvlsMaxBusBw(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning) {
  return computeNVLSMaxBusBw(inputs->comm, inputs->func, tuning->algo, tuning->proto);
}

static int getNVLSSteps(int coll) {
  return coll == ncclFuncAllReduce ? 2 : 1;
}

static ncclResult_t ncclTuningPipelineNvlsGetAlgoStep(struct ncclComm* comm, int proto, int algo, ncclFunc_t func,
                                                      /*output*/ struct ncclTuningPipelineStep* intra,
                                                      struct ncclTuningPipelineStep* inter) {
  int nSteps = getNVLSSteps(func);

  const struct ncclTopoGraph* graph = &comm->graphs[algo];
  int compCapIndex = ncclTuningGetCompCapIndex(comm);
  float intraBw = graph->bwIntra * nvlsEfficiency[compCapIndex];
  TRACE(NCCL_TUNING, "NVLS intraBw=%f ( graphBw=%f * eff=%f [compCapIndex %d]) with graphchannels=%d", intraBw,
        graph->bwIntra, nvlsEfficiency[compCapIndex], compCapIndex, graph->nChannels);
  int intraHw, interHw;
  ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
  float intraRtt = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];

  *intra = {nSteps, intraRtt, intraBw * 1000};
  *inter = {0, 0., 1.}; // no inter traffic

  return ncclSuccess;
}

static ncclResult_t ncclTuningPipelineNvlsTreeGetAlgoStep(struct ncclComm* comm, int proto, int algo, ncclFunc_t func,
                                                          /*output*/ struct ncclTuningPipelineStep* intra,
                                                          struct ncclTuningPipelineStep* inter) {
  int nStepsNVLS = getNVLSSteps(func);

  const struct ncclTopoGraph* graph = &comm->graphs[algo];
  int compCapIndex = ncclTuningGetCompCapIndex(comm);
  float intraBw = graph->bwIntra * nvlsEfficiency[compCapIndex];
  float interBw = ncclTuningProtoBWFactor(proto) * graph->bwInter;
  int intraHw, interHw;
  ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
  float intraRtt = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
  float interRtt = comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];

  *intra = {nStepsNVLS, intraRtt, intraBw * 1000};
  *inter = {(int)(2 * log2i(comm->nNodes)), interRtt, interBw * 1000}; // only allreduce

  return ncclSuccess;
}

static int ncclTuningPipelineNVLSIsValid(struct ncclTuningInput_t* const inputs,
                                         struct ncclTuningResult_t* const tuning) {
  if (tuning->algo != NCCL_ALGO_NVLS) return 0;

  if (!inputs->nvlsSupport) return 0;
  if (inputs->func != ncclFuncAllReduce && inputs->comm->graphs[tuning->algo].nChannels > NCCL_MAX_NVLS_ARITY) return 0;
  if (inputs->func != ncclFuncAllReduce && inputs->comm->localRanks > NCCL_MAX_NVLS_ARITY) return 0;
  if (!inputs->collNetSupport && inputs->comm->nNodes > 1) return 0;

  return 1;
}

static int ncclTuningPipelineNVLSTreeIsValid(struct ncclTuningInput_t* const inputs,
                                             struct ncclTuningResult_t* const tuning) {
  if (tuning->algo != NCCL_ALGO_NVLS_TREE) return 0;

  if (!inputs->nvlsSupport) return 0;
  if (inputs->func != ncclFuncAllReduce) return 0;

  return 1;
}

static inline ncclResult_t ncclTuningPipelineNVLSComputePipeline(struct ncclTuningInput_t* const inputs,
                                                                 struct ncclTuningResult_t* tuning,
                                                                 struct ncclTuningPipeline* pipeline /*output*/) {
  NCCLCHECK(ncclTuningPreComputePipeline(inputs, tuning, pipeline));

  // Counts the number of slices transferred through a link.
  size_t sliceCount = (pipeline->sliceSize == 0) ? 0 : DIVUP(pipeline->channelSize, pipeline->sliceSize);
  pipeline->nSlices = sliceCount;

  return ncclSuccess;
}

ncclResult_t ncclTuningPipelineNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS],
                                             struct ncclTuningModelState* internal) {
  if (internal == nullptr) return ncclInvalidArgument;

  if (!comm->nvlsSupport) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr, nullptr));

  if (proto != NCCL_PROTO_SIMPLE) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  if (comm->config.collnetEnable == 0 && comm->nNodes > 1) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    // NVLS is supported for AR, AG, and RS.
    if (c != ncclFuncAllReduce && c != ncclFuncAllGather && c != ncclFuncReduceScatter) {
      enabled[c] = 0; // Hard disable
      continue;
    }

    if (comm->nNodes > 1 && (c == ncclFuncAllGather || c == ncclFuncReduceScatter)) {
      int nHeads = 0;
      if (c == ncclFuncAllGather && (!comm->ncclCollNet || !comm->ncclCollNet->iallgather)) {
        enabled[c] = 0; // Hard disable
      }
      if (c == ncclFuncReduceScatter && (!comm->ncclCollNet || !comm->ncclCollNet->ireducescatter)) {
        enabled[c] = 0; // Hard disable
      }
      if (comm->config.collnetEnable) {
        nHeads = comm->collNetHeadsNum;
      } else {
        enabled[c] = 0; // Hard disable
      }
      for (int r = 0; r < comm->nRanks; r++) {
        int node = comm->rankToNode[r];
        if (comm->nodeRanks[node].localRanks > nHeads) {
          enabled[c] = 0; // Hard disable
          break;
        }
      }
    }
    comm->tuningContext.generalBandwidths[c][algo][proto] = computeNVLSMaxBusBw(comm, (ncclFunc_t)c, algo, proto);
  }

  internal->pipeline.getAlgoStep = ncclTuningPipelineNvlsGetAlgoStep;
  internal->pipeline.isValid = ncclTuningPipelineNVLSIsValid;
  internal->pipeline.computePipeline = ncclTuningPipelineNVLSComputePipeline;
  internal->pipeline.maxBusBw = nvlsMaxBusBw;
  return ncclSuccess;
}

ncclResult_t ncclTuningPipelineNvlsTreeModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS],
                                                 struct ncclTuningModelState* internal) {
  if (internal == nullptr) return ncclInvalidArgument;

  if (!comm->nvlsSupport) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  int algo, proto;
  NCCLCHECK(ncclTuningExpandId(id, &algo, &proto, nullptr, nullptr));

  if (proto != NCCL_PROTO_SIMPLE) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  if (comm->nNodes == 1) {
    memset(enabled, 0, NCCL_NUM_FUNCTIONS * sizeof(int));
    return ncclSuccess;
  }

  for (int c = 0; c < NCCL_NUM_FUNCTIONS; c++) {
    // NVLSTree is supported for AR only
    if (c != ncclFuncAllReduce) {
      enabled[c] = 0; // Hard disable
      continue;
    }
    comm->tuningContext.generalBandwidths[c][algo][proto] = computeNVLSMaxBusBw(comm, (ncclFunc_t)c, algo, proto);
  }

  internal->pipeline.isValid = ncclTuningPipelineNVLSTreeIsValid;
  internal->pipeline.getAlgoStep = ncclTuningPipelineNvlsTreeGetAlgoStep;
  internal->pipeline.computePipeline =
    ncclTuningPipelineNVLSComputePipeline; // the NVLS pipeline setup is correct for NVLSTree
  internal->pipeline.maxBusBw = nvlsMaxBusBw;
  return ncclSuccess;
}
