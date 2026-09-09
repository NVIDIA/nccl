/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_COST_MODEL_H_
#define NCCL_INT_COST_MODEL_H_

#include "tuning_int.h"

// Init and finalize to support context memory allocation and deallocaiton on comm creation/finalize
typedef ncclResult_t (*ncclTuningModelInitFn_t)(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]);
typedef ncclResult_t (*ncclTuningModelSimFn_t)(struct ncclTuningInput_t* const inputs,
                                               struct ncclTuningResult_t* const tuning);
typedef ncclResult_t (*ncclTuningModelFinalizeFn_t)(struct ncclComm* comm, int id);

struct ncclTuningModelEntry_t {
  ncclTuningModelInitFn_t init; // Initilize Model
  ncclTuningModelSimFn_t model; // Simulate Model
  ncclTuningModelFinalizeFn_t finalize; // Finalize Model, useful for memory de-allocations
  int enabled[NCCL_NUM_FUNCTIONS]; // Hard disable of the model/capability.
};

// Trees are not perfectly sticking to the model for medium sizes. Applying a static correction
// factor is not ideal but works quite well. Powers of two, 64 B to 256MB.
extern float treeCorrectionFactor[NCCL_NUM_PROTOCOLS][24];

ncclResult_t ncclTuningCostModelInit(struct ncclComm* comm);
ncclResult_t ncclTuningCostModelFinalize(struct ncclComm* comm);
ncclResult_t ncclTuningCostModelSimModel(int id, struct ncclTuningInput_t* const input,
                                         struct ncclTuningResult_t* const result);

//////////
// General Models
// Ring model
ncclResult_t ncclTuningRingModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]);
ncclResult_t ncclTuningRingModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// Tree model
ncclResult_t ncclTuningTreeModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]);
ncclResult_t ncclTuningTreeModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// Collnet model
ncclResult_t ncclTuningCollnetModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]);
ncclResult_t ncclTuningCollnetModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// NVLS model
ncclResult_t ncclTuningNvlsModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]);
ncclResult_t ncclTuningNvlsModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// PAT model
ncclResult_t ncclTuningPatModelInit(struct ncclComm* comm, int id, int enabled[NCCL_NUM_FUNCTIONS]);
ncclResult_t ncclTuningPatModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// Symm Kernel Model Wrapper
ncclResult_t ncclTuningSymkModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// CE (Copy Engine) collective model
ncclResult_t ncclTuningCeModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning);

// Tuning general
int ncclTuningGetNsteps(int coll, int nRanks);
int ncclTuningGetCompCapIndex(struct ncclComm* comm);
void ncclTuningGetConstantsIndexes(struct ncclComm* comm, int* index1, int* index2);
void ncclTuningGetHwIndexes(struct ncclComm* comm, int a, int* intraHw, int* interHw);
float ncclTuningGetTime(struct ncclTuningInput_t* const inputs, int a, float* lat, float* bw);

#endif
