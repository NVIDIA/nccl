/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "nccl_device/core.h"
#include "nccl_device/impl/cft_barrier__types.h"
#include <cstring>

ncclResult_t ncclCftBarrierCreateRequirement(ncclTeam_t team, int nBarriers, ncclCftBarrierHandle_t* outHandle,
                                             ncclDevResourceRequirements_t* outReq) {
  memset(outReq, 0, sizeof(*outReq));
  outHandle->nBarriers = nBarriers;
  outReq->bufferSize = (3 * nBarriers + nBarriers * team.nRanks) * NCCL_CFT_BARRIER_GRAN;
  outReq->bufferAlign = NCCL_CFT_BARRIER_ALIGN;
  outReq->outBufferHandle = &outHandle->bufHandle;
  return ncclSuccess;
}
