/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_DEVCOMM_V22902_H_
#define NCCL_DEVCOMM_V22902_H_

#include "dev_runtime.h"

struct ncclWindow_vidmem_v22902 {
  void* winHost;
  char* lsaFlatBase;
  int lsaRank;
  int worldRank;
  uint32_t stride4G;
  uint32_t mcOffset4K;
  uint32_t ginOffset4K;
  ncclGinWindow_t ginWins[NCCL_GIN_MAX_CONNECTIONS];
};

static_assert(sizeof(struct ncclWindow_vidmem_v22902) == 72);

template <typename Dst, typename Src>
static void ncclDevCommCopyResourceWindow_v22902(Dst* dst, Src const& src) {
  dst->lsaFlatBase = src.lsaFlatBase;
  dst->stride4G = src.stride4G;
  dst->mcOffset4K = src.mcOffset4K;
}

#endif // NCCL_DEVCOMM_V22902_H_
