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

static ncclResult_t getTreeNsteps(struct ncclComm* comm, ncclFunc_t func, int* nStepsIntra, int* nStepsInter) {
  if (func != ncclFuncAllReduce) return ncclInternalError;
  *nStepsIntra = 2 * (comm->maxLocalRanks - 1);
  *nStepsInter = 2 * log2i(comm->nNodes);
  return ncclSuccess;
}

static float treeMaxBusBw(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning) {
  struct ncclComm* comm = inputs->comm;
  const struct ncclTopoGraph* graph = &comm->graphs[tuning->algo];
  float algofactor = 1.;
  if (inputs->func == ncclFuncAllReduce || inputs->func == ncclFuncReduceScatter || inputs->func == ncclFuncAllGather)
    algofactor = ((float)(comm->nRanks - 1)) / comm->nRanks;
  return (comm->nNodes > 1 ? graph->bwInter : graph->bwIntra) * graph->nChannels *
         ncclTuningProtoBWFactor(tuning->proto) * algofactor;
}

static ncclResult_t ncclTuningPipelineTreeGetAlgoStep(struct ncclComm* comm, int proto, int algo, ncclFunc_t func,
                                                      /*output*/ struct ncclTuningPipelineStep* intra,
                                                      struct ncclTuningPipelineStep* inter) {
  int nStepsInter, nStepsIntra;
  NCCLCHECK(getTreeNsteps(comm, func, &nStepsIntra, &nStepsInter));
  const struct ncclTopoGraph* graph = &comm->graphs[algo];
  float bwFactor = ncclTuningProtoBWFactor(proto) / graph->nCtasPerChannel;
  float intraBw = bwFactor * graph->bwIntra, interBw = bwFactor * graph->bwInter;
  int intraHw, interHw;
  ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
  float intraRtt = comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto];
  float interRtt = comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto];
  *intra = {nStepsIntra, intraRtt, intraBw * 1000};
  *inter = {nStepsInter, interRtt, interBw * 1000};
  return ncclSuccess;
}

static int ncclTuningPipelineTreeIsValid(struct ncclTuningInput_t* const inputs,
                                         struct ncclTuningResult_t* const tuning) {
  if (tuning->algo != NCCL_ALGO_TREE) return 0;
  if (inputs->func != ncclFuncAllReduce) return 0;
  return 1;
}

static inline ncclResult_t ncclTuningPipelineTreeComputePipeline(struct ncclTuningInput_t* const inputs,
                                                                 struct ncclTuningResult_t* tuning,
                                                                 struct ncclTuningPipeline* pipeline /*output*/) {
  NCCLCHECK(ncclTuningPreComputePipeline(inputs, tuning, pipeline));

  // Counts the number of slices transferred through a link.
  size_t sliceCount = (pipeline->sliceSize == 0) ? 0 : DIVUP(pipeline->channelSize, pipeline->sliceSize);
  if (inputs->func == ncclFuncAllReduce) {
    // Each tree link transfers every slice in the channel twice for an all-reduce.
    pipeline->nSlices = 2 * sliceCount;
    if (pipeline->nSlices > 0) pipeline->nSlices -= 1; // ghost slice accounting
  } else {
    // unsupported
    pipeline->nSlices = 0;
  }

  return ncclSuccess;
}

ncclResult_t ncclTuningPipelineTreeModelInit(struct ncclComm* comm, int id, int* /*enabled[NCCL_NUM_FUNCTIONS]*/,
                                             struct ncclTuningModelState* internal) {
  if (internal == nullptr) return ncclInvalidArgument;
  internal->pipeline.getAlgoStep = ncclTuningPipelineTreeGetAlgoStep;
  internal->pipeline.isValid = ncclTuningPipelineTreeIsValid;
  internal->pipeline.computePipeline = ncclTuningPipelineTreeComputePipeline;
  internal->pipeline.maxBusBw = treeMaxBusBw;
  return ncclSuccess;
}
