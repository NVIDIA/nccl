 /*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "comm.h"
#include "../cost_model.h"
#include "../model.h"
#include "nccl_tuner.h"
#include "pipeline_model.h"

#include <cmath>

// Time to complete a pipeline of "count" chunks of size s.
// The time is measured as the sum of 2 terms:
// - T_first: the time for the first chunk to traverse the whole pipeline (intra.count + inter.count steps)
// - T_delay: the time between 2 chunks at the bottleneck link in the pipeline.
// The total time is T_first + (count - 1) * T_delay
static float pipelineModel(size_t s, int count, struct ncclTuningPipelineStep intra,
                           struct ncclTuningPipelineStep inter) {
  // T_first
  float first =
    intra.stepCount * (intra.lat * 0.5 + s / intra.busBw) + inter.stepCount * (inter.lat * 0.5 + s / inter.busBw);
  // T_delay: we assume s/min(bw) is the largest time
  float delay = s * std::max((intra.stepCount > 0) / intra.busBw, (inter.stepCount > 0) / inter.busBw);
  float totaltime = first + (count - 1) * delay;
  return totaltime;
}

ncclResult_t ncclTuningPipelineModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning,
                                        struct ncclTuningModelState* internal) {
  ncclResult_t ret = ncclSuccess;

  if (internal == nullptr || internal->pipeline.isValid == nullptr) return ncclInternalError;
  if (!internal->pipeline.isValid(inputs, tuning)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    TRACE(NCCL_TUNING, "pipeline a/p/c/s/n %s/%s/%s/%ld/%d is not valid", ncclAlgoToString(tuning->algo),
          ncclProtoToString(tuning->proto), ncclFuncToString(inputs->func), inputs->nBytes, inputs->comm->nRanks);
    return ncclSuccess;
  }

  float lat = 0.;
  float algBw = 0.;

  struct ncclComm* comm = inputs->comm;

  // The number of active channels might not be the number of channels provisioned
  NCCLCHECK(ncclTuningGetChannels(inputs, tuning));
  tuning->nChannels = ncclTuningGetActiveChannels(inputs, tuning);
  int nChannels = tuning->nChannels;
  if (nChannels == 0) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    TRACE(NCCL_TUNING, "a/p/c/s/n %s/%s/%s/%ld/%d pipeline has 0 channels", ncclAlgoToString(tuning->algo),
          ncclProtoToString(tuning->proto), ncclFuncToString(inputs->func), inputs->nBytes, inputs->comm->nRanks);
    return ret;
  }

  TRACE(NCCL_TUNING, "a/p/c/s/n %s/%s/%s/%ld/%d topology info graph={.bwIntra=%f, .bwInter=%f .nChannels=%d}",
        ncclAlgoToString(tuning->algo), ncclProtoToString(tuning->proto), ncclFuncToString(inputs->func),
        inputs->nBytes, inputs->comm->nRanks, comm->graphs[tuning->algo].bwIntra, comm->graphs[tuning->algo].bwInter,
        comm->graphs[tuning->algo].nChannels);

  // Get one channel time using pipeline model
  struct ncclTuningPipeline pipeline;
  if (internal == nullptr || internal->pipeline.computePipeline == nullptr) return ncclInternalError;
  NCCLCHECK(internal->pipeline.computePipeline(inputs, tuning, &pipeline));
  if (pipeline.sliceSize == 0 || pipeline.nSlices == 0) return ncclInternalError;

  struct ncclTuningPipelineStep intra, inter;
  if (internal == nullptr || internal->pipeline.getAlgoStep == nullptr) return ncclInternalError;
  NCCLCHECK(internal->pipeline.getAlgoStep(comm, tuning->proto, tuning->algo, inputs->func, &intra, &inter));

  float pipelineTime = pipelineModel(pipeline.sliceSize, pipeline.nSlices, intra, inter);
  if (pipelineTime <= 0 || !std::isfinite(pipelineTime)) return ncclInternalError;

  // Convert the system bus bandwidth to algorithm bandwidth using algBw = busBw * nRanks / nSteps.
  // nSteps/nRanks is 2*(nRanks-1)/nRanks for AllReduce, (nRanks-1)/nRanks for AG/RS, and 1 for Bcast/Reduce.
  int nSteps = ncclTuningGetNsteps(inputs->func, comm->nRanks);
  float maxBusBw = internal->pipeline.maxBusBw(inputs, tuning);
  float maxAlgBw = maxBusBw * comm->nRanks / nSteps;
  // The factor 1000.0 is used to convert bytes/us to GB/s.
  float channelAlgBw = pipeline.channelSize / pipelineTime / 1000.0f;
  algBw = ncclSoftMin(channelAlgBw * nChannels, maxAlgBw, /*smoothness=*/.1 * maxAlgBw);

  // Keep launch latency outside the pipeline model
  lat = comm->tuningContext.tuningConstants.baseLatencies[tuning->algo][tuning->proto];
  tuning->timeUs = ncclTuningGetTime(inputs, tuning->algo, lat, algBw);

  TRACE(
    NCCL_TUNING,
    "a/p/c/s/n %s/%s/%s/%ld/%d maxBusBw %f, maxAlgBw %f, channelAlgBw %f * %d channels [%d warps] = algBw %f timeUs=%f",
    ncclAlgoToString(tuning->algo), ncclProtoToString(tuning->proto), ncclFuncToString(inputs->func), inputs->nBytes,
    inputs->comm->nRanks, maxBusBw, maxAlgBw, channelAlgBw, nChannels, tuning->nWarps, algBw, tuning->timeUs);

  return ret;
}
