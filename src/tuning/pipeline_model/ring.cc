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

static float ringMaxBusBw(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning) {
  struct ncclComm* comm = inputs->comm;
  const struct ncclTopoGraph* graph = &comm->graphs[tuning->algo];
  float bwFactor = 1.0f / graph->nCtasPerChannel;
  float intraBw = bwFactor * graph->bwIntra, interBw = bwFactor * graph->bwInter;
  intraBw = ncclTuningProtoBW(comm, tuning->algo, tuning->proto, intraBw);
  interBw = ncclTuningProtoBW(comm, tuning->algo, tuning->proto, interBw);
  return (comm->nNodes > 1 ? interBw : intraBw) * graph->nChannels * graph->nCtasPerChannel;
}

static ncclResult_t ncclTuningPipelineRingGetAlgoStep(struct ncclComm* comm, int proto, int algo, ncclFunc_t func,
                                                      /*output*/ struct ncclTuningPipelineStep* intra,
                                                      struct ncclTuningPipelineStep* inter) {
  const struct ncclTopoGraph* graph = &comm->graphs[algo];
  int nSteps = ncclTuningGetNsteps(func, comm->nRanks);
  int nInterSteps = (comm->nNodes == 1) ? 0 : (func == ncclFuncAllReduce) ? (2 * (comm->nNodes - 1)) : comm->nNodes - 1;
  float bwFactor = ncclTuningProtoBWFactor(proto) / graph->nCtasPerChannel;
  float intraBw = bwFactor * graph->bwIntra, interBw = bwFactor * graph->bwInter;
  int intraHw, interHw;
  ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
  float intraRtt = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
  float interRtt = comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
  *intra = {nSteps - nInterSteps, intraRtt, intraBw * 1000};
  *inter = {nInterSteps, interRtt, interBw * 1000};

  return ncclSuccess;
}

static int ncclTuningPipelineRingIsValid(struct ncclTuningInput_t* const inputs,
                                         struct ncclTuningResult_t* const tuning) {
  return tuning->algo == NCCL_ALGO_RING;
}

static inline ncclResult_t ncclTuningPipelineRingComputePipeline(struct ncclTuningInput_t* const inputs,
                                                                 struct ncclTuningResult_t* tuning,
                                                                 struct ncclTuningPipeline* pipeline /*output*/) {
  NCCLCHECK(ncclTuningPreComputePipeline(inputs, tuning, pipeline));

  struct ncclComm* comm = inputs->comm;
  // Counts the number of slices transferred through a link.
  size_t sliceCount = (pipeline->sliceSize == 0) ? 0 : DIVUP(pipeline->channelSize, pipeline->sliceSize);

  // All links are active throughout the nSteps, the total bytes through a link are (channelSize/nRanks)*nSteps.
  // The total number of slices is then C = ceil(channelSlices*nSteps/nRanks).
  int nSteps = ncclTuningGetNsteps(inputs->func, comm->nRanks);
  pipeline->nSlices = comm->nRanks == 0 ? 0 : DIVUP(sliceCount * nSteps, comm->nRanks);
  if (inputs->func == ncclFuncAllReduce && comm->nRanks > 2 && pipeline->nSlices > 0)
    pipeline->nSlices -= 1; // ghost slice accounting

  return ncclSuccess;
}

ncclResult_t ncclTuningPipelineRingModelInit(
  struct ncclComm* /*comm*/, int /*id*/, int* /*enabled[NCCL_NUM_FUNCTIONS]*/, struct ncclTuningModelState* internal) {
  if (internal == nullptr) return ncclInvalidArgument;
  internal->pipeline.getAlgoStep = ncclTuningPipelineRingGetAlgoStep;
  internal->pipeline.isValid = ncclTuningPipelineRingIsValid;
  internal->pipeline.computePipeline = ncclTuningPipelineRingComputePipeline;
  internal->pipeline.maxBusBw = ringMaxBusBw;
  return ncclSuccess;
}
