/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "core.h"
#include "nccl_device/host.h"
#include "nccl_device/impl/gin_barrier__funcs.h"

NCCL_API(ncclResult_t, ncclGinBarrierCreateRequirement, ncclComm_t comm, ncclTeam_t team, int nBarriers,
         ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq);
ncclResult_t ncclGinBarrierCreateRequirement(ncclComm_t comm, ncclTeam_t team, int nBarriers,
                                             ncclGinBarrierHandle_t* outHandle, ncclDevResourceRequirements_t* outReq) {
  memset(outReq, 0, sizeof(*outReq));
  // Per-peer slots, clamped to 2 so that device code running the two-signal barrier always fits.
  outReq->ginSignalCount = nBarriers * ncclGinBarrierSlots(NCCL_GIN_BARRIER_DEFAULT, team.nRanks);
  outReq->outGinSignalStart = &outHandle->signal0;
  return ncclSuccess;
}
