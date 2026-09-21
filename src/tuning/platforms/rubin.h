/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TUNING_PLATFORM_RUBIN_H_
#define NCCL_TUNING_PLATFORM_RUBIN_H_

#include "tuner.h"
#include "../cost_model.h"

static struct ncclTuningModelEntry_t ncclTuningRubinModelMap[] = {
  /*
Initialize default, static models here
{mod_init, mod_sim, mod_final, enabled}
Enable order: Broadcast, Reduce, AllGather, ReduceScatter, AllReduce
*/
  {ncclTuningPipelineTreeModelInit, ncclTuningPipelineModelSim, nullptr, {0, 0, 0, 0, 1}, {}},       // Tree/LL
  {ncclTuningPipelineTreeModelInit, ncclTuningPipelineModelSim, nullptr, {0, 0, 0, 0, 1}, {}},       // Tree/LL128
  {ncclTuningPipelineTreeModelInit, ncclTuningPipelineModelSim, nullptr, {0, 0, 0, 0, 1}, {}},       // Tree/Simple
  {ncclTuningPipelineRingModelInit, ncclTuningPipelineModelSim, nullptr, {1, 1, 1, 1, 1}, {}},       // Ring/LL
  {ncclTuningPipelineRingModelInit, ncclTuningPipelineModelSim, nullptr, {1, 1, 1, 1, 1}, {}},       // Ring/LL128
  {ncclTuningPipelineRingModelInit, ncclTuningPipelineModelSim, nullptr, {1, 1, 1, 1, 1}, {}},       // Ring/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetDirect/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetDirect/LL128, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0, 0, 1, 1, 1}, {}}, // CollNetDirect/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetChain/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetChain/LL128, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // CollNetChain/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLS/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLS/LL128, disabled as there is no implementation
  {ncclTuningPipelineNvlsModelInit, ncclTuningPipelineModelSim, nullptr, {0, 0, 1, 1, 1}, {}}, // NVLS/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLSTree/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLSTree/LL128, disabled as there is no implementation
  {ncclTuningPipelineNvlsTreeModelInit, ncclTuningPipelineModelSim, nullptr, {0, 0, 1, 1, 1}, {}}, // NVLSTree/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // PAT/LL
  {nullptr, nullptr, nullptr, {0}, {}}, // PAT/LL128
  {nullptr, nullptr, nullptr, {0, 0, 1, 1, 0}, {}}, // PAT/Simple
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // AllReduce_AGxLL_R
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // AllReduce_AGxLLMC_R
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // AllReduce_RSxTmaLD_AGxTmaST
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // AllReduce_RSxLD_AGxST
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // AllReduce_RSxLDMC_AGxSTMC
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_LL
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_LLMC
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_TmaST
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_ST
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_TmaSTMC
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_STMC
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // AllGather_RailRing_LsaSTMC
  {nullptr, nullptr, nullptr, {0, 0, 0, 1, 0}, {}}, // ReduceScatter_LL
  {nullptr, nullptr, nullptr, {0, 0, 0, 1, 0}, {}}, // ReduceScatter_TmaLD
  {nullptr, nullptr, nullptr, {0, 0, 0, 1, 0}, {}}, // ReduceScatter_LD
  {nullptr, nullptr, nullptr, {0, 0, 0, 1, 0}, {}}, // ReduceScatter_LDMC
  {nullptr, nullptr, nullptr, {0, 0, 0, 1, 0}, {}}, // ReduceScatter_RailA2A_LsaLD
  {nullptr, nullptr, nullptr, {0, 0, 0, 1, 0}, {}}, // ReduceScatter_RailA2A_LsaLDMC
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // CE AllGather Unicast
  {nullptr, nullptr, nullptr, {0, 0, 1, 0, 0}, {}}, // CE AllGather Multicast
};

static void ncclTuningRubinTuningConstants(ncclTunerConstants_t* constants) {
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_LL] = 3.49;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_LL128] = 4.88;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 10.99;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_RING][NCCL_PROTO_LL] = 4.03;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_RING][NCCL_PROTO_LL128] = 5.46;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 14.26;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 41.35;
  constants->hwLatencies[NCCL_HW_NVLINK][NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = 40.48;

  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][NCCL_PROTO_LL] = 19.56;
  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][NCCL_PROTO_LL128] = 27.64;
  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_TREE][NCCL_PROTO_SIMPLE] = 36.32;
  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_LL] = 8.19;
  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_LL128] = 11.29;
  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 15.59;
  constants->hwLatencies[NCCL_HW_NET][NCCL_ALGO_NVLS_TREE][NCCL_PROTO_SIMPLE] = 34.37;
}

static void ncclTuningRubinPreInit(struct ncclComm* comm) {
  ncclTuningRubinTuningConstants(&comm->tuningContext.tuningConstants);
}

static ncclResult_t ncclTuningRubinGetModelEntry(int compCap, int id, struct ncclTuningModelEntry_t** entry) {
  if (ncclTuningRubinModelMap[id].model != nullptr) {
    *entry = &ncclTuningRubinModelMap[id];
  }
  return ncclSuccess;
}

#endif // NCCL_TUNING_PLATFORM_RUBIN_H_
