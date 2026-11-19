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
#include "tuning.h"
#include "../cost_model.h"
#include "rubin.h"

#define NCCL_TUNING_PLATFORM_CTA_UPPERBOUND 128

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

static ncclResult_t ncclTuningPlatformChannelTuning(struct ncclTuningInput_t* const input,
                                                    const struct ncclTuningResult_t* result, int* maxChannels) {
  if (RUBIN_AND_LATER(input->comm->minCompCap)) {
    NCCLCHECK(ncclTuningRubinChannelTuning(input, result, maxChannels));
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuningPlatformMinFifoTime(const struct ncclComm* comm, int algo, int proto,
                                                  struct ncclTuningMinFifoTimeData* minFifoTime) {
  *minFifoTime = ncclTuningMinFifoTimeData({0, 0, 1});
  if (RUBIN_AND_LATER(comm->minCompCap)) {
    NCCLCHECK(ncclTuningRubinMinFifoTime(comm, algo, proto, minFifoTime));
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuningPlatformGetIntraLatFactor(const struct ncclComm* comm, int algo, int proto,
                                                        struct ncclTuningLatFactor* lf) {
  lf->saturation_scaling = 1.;
  lf->channel_saturation_point = NCCL_TUNING_PLATFORM_CTA_UPPERBOUND; // disables lat scaling by default
  if (RUBIN_AND_LATER(comm->minCompCap)) {
    NCCLCHECK(ncclTuningRubinGetIntraLatFactor(comm, algo, proto, lf));
  }
  return ncclSuccess;
}

static ncclResult_t ncclTuningPlatformGetInterLatFactor(const struct ncclComm* comm, int algo, int proto,
                                                        struct ncclTuningLatFactor* lf) {
  lf->saturation_scaling = 1.;
  lf->channel_saturation_point = NCCL_TUNING_PLATFORM_CTA_UPPERBOUND; // disables lat scaling by default
  return ncclSuccess;
}

#endif // NCCL_TUNING_PLATFORMS_H_
