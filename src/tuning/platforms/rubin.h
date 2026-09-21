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
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}},       // Tree/LL
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}},       // Tree/LL128
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}},       // Tree/Simple
  {nullptr, nullptr, nullptr, {1, 1, 1, 1, 1}, {}},       // Ring/LL
  {nullptr, nullptr, nullptr, {1, 1, 1, 1, 1}, {}},       // Ring/LL128
  {nullptr, nullptr, nullptr, {1, 1, 1, 1, 1}, {}},       // Ring/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetDirect/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetDirect/LL128, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0, 0, 1, 1, 1}, {}}, // CollNetDirect/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetChain/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // CollNetChain/LL128, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0, 0, 0, 0, 1}, {}}, // CollNetChain/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLS/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLS/LL128, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0, 0, 1, 1, 1}, {}}, // NVLS/Simple
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLSTree/LL, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0}, {}}, // NVLSTree/LL128, disabled as there is no implementation
  {nullptr, nullptr, nullptr, {0, 0, 1, 1, 1}, {}}, // NVLSTree/Simple
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

static void ncclTuningRubinPreInit(struct ncclComm* comm) {}

static ncclResult_t ncclTuningRubinGetModelEntry(int compCap, int id, struct ncclTuningModelEntry_t** entry) {
  if (ncclTuningRubinModelMap[id].model != nullptr) {
    *entry = &ncclTuningRubinModelMap[id];
  }
  return ncclSuccess;
}

#endif // NCCL_TUNING_PLATFORM_RUBIN_H_
