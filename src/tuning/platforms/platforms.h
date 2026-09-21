/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TUNING_PLATFORMS_H_
#define NCCL_TUNING_PLATFORMS_H_

#include "cudawrap.h"
#include "../cost_model.h"
#include "rubin.h"

static ncclResult_t ncclTuningGetModelPlatformEntry(int compCap, int id, struct ncclTuningModelEntry_t** entry) {
  if (RUBIN_AND_LATER(compCap)) {
    NCCLCHECK(ncclTuningRubinGetModelEntry(compCap, id, entry));
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuningPlatformPreInit(struct ncclComm* comm) {
  if (RUBIN_AND_LATER(comm->minCompCap)) {
    ncclTuningRubinPreInit(comm);
  }
  return ncclSuccess;
}

#endif // NCCL_TUNING_PLATFORMS_H_