/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TUNING_PIPELINE_MODEL_H_
#define NCCL_TUNING_PIPELINE_MODEL_H_

#include "comm.h"
#include "coll_sizes.h"
#include "tuning.h"

#include <algorithm>

struct ncclTuningPipeline;

typedef ncclResult_t (*ncclTuningGetAlgoStepFn_t)(struct ncclComm* comm, int proto, int algo, ncclFunc_t func,
                                                  struct ncclTuningPipelineStep* intra,
                                                  struct ncclTuningPipelineStep* inter);

typedef int (*ncclTuningIsValidFn_t)(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

typedef float (*ncclTuningMaxBusBw_t)(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

typedef ncclResult_t (*ncclTuningPipelineCompute_t)(struct ncclTuningInput_t* const, struct ncclTuningResult_t*,
                                                    struct ncclTuningPipeline*);

struct ncclTuningPipeline {
  size_t channelSize;
  size_t nominalChunkSize;
  size_t chunkSize;
  size_t sliceSize;
  size_t nSlices;
};

struct ncclTuningPipelineState {
  ncclTuningGetAlgoStepFn_t getAlgoStep;
  ncclTuningIsValidFn_t isValid;
  ncclTuningPipelineCompute_t computePipeline;
  ncclTuningMaxBusBw_t maxBusBw;
};

struct ncclTuningPipelineStep {
  int stepCount;
  float lat, busBw;
};

inline ncclResult_t ncclTuningPipelineComputeSliceSize(
  struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* tuning, struct ncclTuningPipeline* pipeline) {
  int protocol = tuning->proto;
  size_t elementSize = ncclTypeSize(inputs->datatype);

  int chunkSteps = ncclGetChunkSteps(protocol, tuning->algo, inputs->chunkSteps);
  int sliceSteps = ncclGetSliceSteps(protocol, tuning->algo, inputs->sliceSteps);

  size_t nominalSliceSize = ncclNominalSliceSize(pipeline->nominalChunkSize, chunkSteps, sliceSteps);

  if (protocol == NCCL_PROTO_SIMPLE && pipeline->chunkSize != 0) {
    size_t chunkElements = DIVUP(pipeline->chunkSize, elementSize);
    size_t nominalSliceElements = nominalSliceSize / elementSize;
    int slicesPerChunk = chunkSteps / sliceSteps;
    pipeline->sliceSize = ncclSimpleSliceSize(chunkElements, slicesPerChunk, nominalSliceElements) * elementSize;
    if (pipeline->sliceSize > pipeline->chunkSize) pipeline->sliceSize = pipeline->chunkSize;
  } else {
    pipeline->sliceSize = pipeline->chunkSize < nominalSliceSize ? pipeline->chunkSize : nominalSliceSize;
  }

  return ncclSuccess;
}

inline ncclResult_t ncclTuningPipelineComputeChunkSize(struct ncclTuningInput_t* const inputs,
                                                       struct ncclTuningResult_t* tuning,
                                                       struct ncclTuningPipeline* pipeline /*output*/) {
  struct ncclComm* comm = inputs->comm;
  int protocol = tuning->proto;
  int chunkSteps = inputs->chunkSteps;
  chunkSteps = ncclGetChunkSteps(protocol, tuning->algo, chunkSteps);
  size_t stepSize = comm->buffSizes[protocol] / NCCL_STEPS;
  pipeline->channelSize = ncclSizePerChannel(inputs->nBytes, tuning->nChannels);
  pipeline->nominalChunkSize = ncclGetChunkSize(protocol, stepSize, chunkSteps);
  pipeline->chunkSize = std::min(pipeline->nominalChunkSize, pipeline->channelSize);

  return ncclSuccess;
}

inline ncclResult_t ncclTuningPreComputePipeline(struct ncclTuningInput_t* const inputs,
                                                 struct ncclTuningResult_t* tuning,
                                                 struct ncclTuningPipeline* pipeline /*output*/) {
  NCCLCHECK(ncclTuningPipelineComputeChunkSize(inputs, tuning, pipeline));
  NCCLCHECK(ncclTuningPipelineComputeSliceSize(inputs, tuning, pipeline));

  return ncclSuccess;
}

#endif // NCCL_TUNING_PIPELINE_MODEL_H_
