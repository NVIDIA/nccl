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
#include "ce_coll.h"

#define NCCL_TUNING_SYM_KERNEL_ID_OFFSET (NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS)
#define NCCL_TUNING_CE_METHOD_ID_OFFSET (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + ncclSymkKernelId_Count)
#define NCCL_TUNING_COUNT (NCCL_TUNING_CE_METHOD_ID_OFFSET + ncclCeMethodId_Count)

// The general collective tables cover the five functions in NCCL_NUM_FUNCTIONS.
// Symmetric tuning additionally supports AlltoAll without changing that global
// count, which is also used by unrelated operation counters and plugin tables.
#define NCCL_NUM_TUNING_FUNCTIONS (NCCL_NUM_FUNCTIONS + 1)

static inline int ncclTuningFuncIndex(ncclFunc_t func) {
  if ((unsigned)func < NCCL_NUM_FUNCTIONS) return (int)func;
  return func == ncclFuncAlltoAll ? NCCL_NUM_FUNCTIONS : -1;
}

#define NCCL_TUNING_MASK_GENERAL_KERNELS ((1ULL << NCCL_TUNING_SYM_KERNEL_ID_OFFSET) - 1ULL)
#define NCCL_TUNING_MASK_SYM_KERNELS \
  ((1ULL << NCCL_TUNING_CE_METHOD_ID_OFFSET) - 1ULL - NCCL_TUNING_MASK_GENERAL_KERNELS)
#define NCCL_TUNING_MASK_CE ((1ULL << NCCL_TUNING_COUNT) - (1ULL << NCCL_TUNING_CE_METHOD_ID_OFFSET))
#define NCCL_TUNING_MASK_ALL ((1ULL << NCCL_TUNING_COUNT) - 1ULL)
static_assert(NCCL_TUNING_COUNT < 64, "Tuning masks cannot represent every tuning entry");

#define NCCL_TUNING_MASK_GENERAL_ALGO(algo) (((1ULL << NCCL_NUM_PROTOCOLS) - 1ULL) << ((algo) * NCCL_NUM_PROTOCOLS))
#define NCCL_TUNING_MASK_NVLS_GENERAL \
  (NCCL_TUNING_MASK_GENERAL_ALGO(NCCL_ALGO_NVLS) | NCCL_TUNING_MASK_GENERAL_ALGO(NCCL_ALGO_NVLS_TREE))
#define NCCL_TUNING_SYM_KERNEL_BIT(k) (1ULL << (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + (k)))
#define NCCL_TUNING_MASK_NVLS_SYM_KERNELS \
  (NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_AllReduce_AGxLLMC_R) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_AllGather_LLMC) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_AllGather_TmaSTMC) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_AllGather_STMC) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_AllGather_RailRing_LsaSTMC) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_ReduceScatter_LDMC) | \
   NCCL_TUNING_SYM_KERNEL_BIT(ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC))

#define NCCL_TUNING_ENTRY_INIT_VALUE -1
#define NCCL_TUNING_RESULT_INIT \
  {/*.id =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.valid =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.timeUs =*/NCCL_TUNING_IGNORE, \
   /*.selectionTimeUs =*/NCCL_TUNING_IGNORE, \
   /*.algo =*/NCCL_ALGO_UNDEF, \
   /*.proto =*/NCCL_PROTO_UNDEF, \
   /*.symKernelId =*/ncclSymkKernelId_Count, \
   /*.ceMethodId =*/ncclCeMethodId_Count, \
   /*.nChannels =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.maxChannels =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.nWarps =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.minChunkPayloadBytes =*/NCCL_TUNING_ENTRY_INIT_VALUE, \
   /*.forced =*/0}

struct ncclTuningResult_t {
  int id;
  int valid;
  float timeUs;
  float selectionTimeUs;
  int algo;
  int proto;
  int symKernelId;
  int ceMethodId;
  int nChannels;
  int maxChannels;
  int nWarps;
  int minChunkPayloadBytes;
  int forced;
};

struct ncclTuningInput_t {
  struct ncclComm* comm;
  uint64_t tuningMask;
  // Per-call algorithm selection, 0 when the call made none. The tuner narrows tuningMask to it
  // unless the environment forced this function, and resolves an unmet selection itself: it
  // retunes with the full tuningMask, then fails when forceAlgSelection is set and falls back
  // otherwise. Callers that resolve the selection themselves leave this at 0.
  uint64_t algSelection;
  // 1 == tuningMask is a hard constraint: error out instead of returning a selection outside it.
  // Callers that implement their own fallback policy on an unmet mask must leave this at 0.
  int forceAlgSelection;
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
  bool symAligned16B; // Check for TMA eligibility in sym scheduler
  int captured;
  int inPlace;
  int minCTAs;  // resolved lower CTA bound (env > per-call > comm)
  int maxCTAs;  // resolved upper CTA bound (env > per-call > comm)
  int CTAPolicy;   // resolved effective CTAPolicy (per-call)
  int chunkSteps;
  int sliceSteps;
};

struct ncclTuningContext_t {
  // Persistant tuning parameters tied to a communicator.
  ncclTunerConstants_t tuningConstants;
  // State of the tuning models
  // Forced function is set via env var
  int forced[NCCL_NUM_TUNING_FUNCTIONS];
  // Disabled tuning models are not execute and excluded from implemetation selection.
  int enabled[NCCL_TUNING_COUNT][NCCL_NUM_TUNING_FUNCTIONS];
  // Store of model contexts per communicator.
  float generalLatencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  float generalBandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  int maxThreads[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
};

ncclResult_t ncclTuningInit(struct ncclComm* comm);
ncclResult_t ncclTuningCompute(struct ncclTuningInput_t* const input, struct ncclTuningResult_t* const result);
ncclResult_t ncclTuningFinalize(struct ncclComm* comm);

// Higher CE threshold for AllGather since TMA kernels continue to perform better till higher message sizes.
int64_t ncclSymCeAllGatherThreshold(struct ncclComm* comm);

// Symm Kernel Model Helper Functions
double ncclTuningGetLsaBw(struct ncclComm* comm);
double ncclTuningGetGinLat(struct ncclComm* comm);
double ncclTuningGetGinBw(struct ncclComm* comm);
void ncclTuningGetBusMulReduceScatterRailA2A(struct ncclComm* comm, bool ldmc, double* out_smMul, double* out_lsaMul,
                                             double* out_ginMul);
double ncclTuningGetSmLatReduceScatterRailA2A(struct ncclComm* comm, bool ldmc);
int ncclTuningCalcSatBlocksReduceScatterRailA2A(struct ncclComm* comm, bool ldmc);

int64_t ncclParamSymNoWinEnable();

#endif
