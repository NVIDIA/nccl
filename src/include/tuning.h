/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_INT_TUNING_H_
#define NCCL_INT_TUNING_H_

#include "nccl.h"
#include "nccl_tuner.h"
#include "sym_kernels.h"

#define NCCL_TUNING_COUNT (NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS + ncclSymkKernelId_Count)
#define NCCL_TUNING_SYM_KERNEL_ID_OFFSET (NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS)

#define NCCL_TUNING_MASK_ALL ((1ul << NCCL_TUNING_COUNT) - 1ul)
#define NCCL_TUNING_MASK_GENERAL_KERNELS ((1ul << (NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS)) - 1ul)
#define NCCL_TUNING_MASK_SYM_KERNELS \
  ((1ul << (NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS + ncclSymkKernelId_Count)) - 1ul - \
   (NCCL_TUNING_MASK_GENERAL_KERNELS))

#define NCCL_TUNING_ENTRY_INIT_VALUE -1
#define NCCL_TUNING_RESULT_INIT \
  {/*.id =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.valid =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.timeUs =*/NCCL_TUNING_IGNORE, \
   /*.algo =*/NCCL_ALGO_UNDEF, \
   /*.proto =*/NCCL_PROTO_UNDEF, \
   /*.symKernelId =*/ncclSymkKernelId_Count, \
   /*.nChannels =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.maxChannels =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.nWarps =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.forced =*/0}

struct ncclTuningResult_t {
  int id;
  int valid;
  float timeUs;
  int algo;
  int proto;
  int symKernelId;
  int nChannels;
  int maxChannels;
  int nWarps;
  int forced;
};

struct ncclTuningInput_t {
  struct ncclComm* comm;
  uint64_t tuningMask;
  ncclFunc_t func;
  ncclRedOp_t redOp;
  ncclDevRedOp_t devRedOp;
  ncclDataType_t datatype;
  size_t nBytes;
  int numPipeOps;
  size_t count;
  size_t countMax;
  int nWorks;
  ncclSymRegType_t winRegType;
  int regBuff;
  int collNetSupport;
  int nvlsSupport;
};

struct ncclTuningContext_t {
  // Persistant tuning parameters tied to a communicator.
  ncclTunerConstants_t tuningConstants;
  // State of the tuning models
  // Forced function is set via env var
  int forced[NCCL_NUM_FUNCTIONS];
  // Disabled tuning models are not execute and excluded from implemetation selection.
  int enabled[NCCL_TUNING_COUNT][NCCL_NUM_FUNCTIONS];
  // Store of model contexts per communicator.
  float generalLatencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float generalBandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
};

ncclResult_t ncclTuningInit(struct ncclComm* comm);
ncclResult_t ncclTuningCompute(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result);
ncclResult_t ncclTuningFinalize(struct ncclComm* comm);

// Symm Kernel Model Helper Functions
double ncclTuningGetLsaBw(struct ncclComm* comm);
double ncclTuningGetGinLat(struct ncclComm* comm);
double ncclTuningGetGinBw(struct ncclComm* comm);
void ncclTuningGetBusMulReduceScatterRailA2A(struct ncclComm* comm, bool ldmc, double* out_smMul, double* out_lsaMul,
                                             double* out_ginMul);
double ncclTuningGetSmLatReduceScatterRailA2A(struct ncclComm* comm, bool ldmc);
int ncclTuningCalcSatBlocksReduceScatterRailA2A(struct ncclComm* comm, bool ldmc);

#endif