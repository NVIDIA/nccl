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
#include "../platforms/platforms.h"

static float ringMaxBusBw(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning) {
  struct ncclComm* comm = inputs->comm;
  const struct ncclTopoGraph* graph = &comm->graphs[tuning->algo];
  float bwFactor = 1.0f / graph->nCtasPerChannel;
  float intraBw = bwFactor * graph->bwIntra, interBw = bwFactor * graph->bwInter;
  intraBw = ncclTuningProtoBW(comm, tuning->algo, tuning->proto, intraBw);
  interBw = ncclTuningProtoBW(comm, tuning->algo, tuning->proto, interBw);
  return (comm->nNodes > 1 ? interBw : intraBw) * graph->nChannels * graph->nCtasPerChannel;
}

static float getIntraLatScaling(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning) {
  ncclTuningLatFactor lf;

  ncclTuningPlatformGetIntraLatFactor(inputs->comm, tuning->algo, tuning->proto, &lf);

  int ctas = tuning->nChannels;
  if (ctas <= lf.channel_saturation_point)
    return 1 + (lf.saturation_scaling - 1) / (float)lf.channel_saturation_point * ctas;
  else return ctas / (float)lf.channel_saturation_point * lf.saturation_scaling;
}

static float getInterLatScaling(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning) {
  ncclTuningLatFactor lf;

  ncclTuningPlatformGetInterLatFactor(inputs->comm, tuning->algo, tuning->proto, &lf);

  int ctas = tuning->nChannels;
  if (ctas <= lf.channel_saturation_point)
    return 1 + (lf.saturation_scaling - 1) / (float)lf.channel_saturation_point * ctas;
  else return ctas / (float)lf.channel_saturation_point * lf.saturation_scaling;
}

static ncclResult_t ncclTuningPipelineRingGetAlgoStep(struct ncclTuningInput_t* const inputs,
                                                      struct ncclTuningResult_t* tuning,
                                                      struct ncclTuningPipeline* pipeline /*output*/) {
  struct ncclComm* comm = inputs->comm;
  int proto = tuning->proto;
  int algo = tuning->algo;
  ncclFunc_t func = inputs->func;
  struct ncclTuningPipelineStep* intra = &(pipeline->intraStep);
  struct ncclTuningPipelineStep* inter = &(pipeline->interStep);
  const struct ncclTopoGraph* graph = &comm->graphs[algo];
  int nSteps = ncclTuningGetNsteps(func, comm->nRanks);
  int nInterSteps = (comm->nNodes == 1) ? 0 : (func == ncclFuncAllReduce) ? (2 * (comm->nNodes)) : comm->nNodes;
  if (comm->nRanks == comm->nNodes) nInterSteps = nSteps;
  float bwFactor = ncclTuningProtoBWFactor(proto) / graph->nCtasPerChannel;
  float intraBw = bwFactor * graph->bwIntra, interBw = bwFactor * graph->bwInter;
  int intraHw, interHw;
  ncclTuningGetHwIndexes(comm, algo, &intraHw, &interHw);
  float intraRtt =
    comm->tuningContext.tuningConstants.hwLatencies[intraHw][algo][proto] * getIntraLatScaling(inputs, tuning);
  float interRtt =
    comm->tuningContext.tuningConstants.hwLatencies[interHw][algo][proto] * getInterLatScaling(inputs, tuning);
  *intra = {nSteps - nInterSteps, intraRtt, intraBw * 1000};
  *inter = {nInterSteps, interRtt, interBw * 1000};

  ncclTuningMinFifoTimeData mftd;
  ncclTuningPlatformMinFifoTime(inputs->comm, tuning->algo, tuning->proto, &mftd);
  if (tuning->nChannels < mftd.CTAsaturationPoint)
    pipeline->minFifoStepTime = mftd.minFifoTime + (mftd.saturatedFifoTime - mftd.minFifoTime) *
                                                     ((float)tuning->nChannels / mftd.CTAsaturationPoint);
  else pipeline->minFifoStepTime = mftd.saturatedFifoTime * ((float)tuning->nChannels / mftd.CTAsaturationPoint);

  return ncclSuccess;
}

static int ncclTuningPipelineRingIsValid(struct ncclTuningInput_t* const inputs,
                                         struct ncclTuningResult_t* const tuning) {
  return tuning->algo == NCCL_ALGO_RING;
}

static inline ncclResult_t ncclTuningPipelineRingComputePipeline(struct ncclTuningInput_t* const inputs,
                                                                 struct ncclTuningResult_t* tuning,
                                                                 struct ncclTuningPipeline* pipeline /*output*/) {
  struct ncclComm* comm = inputs->comm;
  NCCLCHECK(ncclTuningPipelineComputeChunkSize(inputs, tuning, pipeline));

  size_t chunkWorkSize = pipeline->channelSize;

  if (tuning->algo == NCCL_ALGO_RING && (inputs->func == ncclFuncAllReduce || inputs->func == ncclFuncAllGather ||
                                         inputs->func == ncclFuncReduceScatter)) {
    size_t elementSize = ncclTypeSize(inputs->datatype);

    size_t elements = DIVUP(pipeline->channelSize, elementSize);
    size_t alignedRankElements = ncclElementAlignedDivUp(elements, comm->nRanks, elementSize, (size_t)16);
    chunkWorkSize = std::min(pipeline->channelSize, alignedRankElements * elementSize);
  }

  pipeline->chunkSize = std::min(pipeline->nominalChunkSize, chunkWorkSize);

  NCCLCHECK(ncclTuningPipelineComputeSliceSize(inputs, tuning, pipeline));

  pipeline->fifoStepsNeeded = DIVUP(pipeline->channelSize, pipeline->chunkSize);
  if (inputs->func == ncclFuncAllReduce)
    pipeline->fifoStepsNeeded *= (float)ncclTuningGetNsteps(inputs->func, comm->nRanks) / comm->nRanks;

  if (pipeline->fifoStepsNeeded < NCCL_STEPS) pipeline->fifoStepsNeeded = 0;

  // LL and LL128 behave like a full pipeline in 1 domains
  if (inputs->comm->nNodes == 1 && (tuning->proto == NCCL_PROTO_LL || tuning->proto == NCCL_PROTO_LL128)) {
    pipeline->sliceSize = std::min(pipeline->sliceSize, (size_t)ncclProtoGrainSize(tuning->proto));
  }

  // Counts the number of slices transferred through a link.
  size_t sliceCount = (pipeline->sliceSize == 0) ? 0 : DIVUP(pipeline->channelSize, pipeline->sliceSize);

  // All links are active throughout the nSteps, the total bytes through a link are (channelSize/nRanks)*nSteps.
  // The total number of slices is then C = ceil(channelSlices*nSteps/nRanks).
  int nSteps = ncclTuningGetNsteps(inputs->func, comm->nRanks);
  pipeline->nSlices = comm->nRanks == 0 ? 0 : DIVUP(sliceCount * nSteps, comm->nRanks);
  if (inputs->func == ncclFuncAllReduce && comm->nRanks > 2 && pipeline->nSlices > 0)
    pipeline->nSlices -= 1; // ghost slice accounting

  if (tuning->proto == NCCL_PROTO_SIMPLE) pipeline->maxMode = true;

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
