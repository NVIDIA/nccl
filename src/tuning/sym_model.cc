/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "cost_model.h"
#include "sym_kernels.h"
#include "sym_model/model.h"

#include "comm.h"
#include "core.h"

#include <cfloat>
#include <cmath>

NCCL_PARAM(SymCTAs, "SYM_CTAS", 0)

static constexpr float disableTime = 1.e30f;

int ncclSymkModelCtasEnvOverride() {
  int64_t nUserCTAs = ncclParamSymCTAs();
  if (nUserCTAs < 1) return 0;
  if (nUserCTAs > ncclSymkMaxBlocks) return ncclSymkMaxBlocks;
  return static_cast<int>(nUserCTAs);
}

static ncclResult_t queryModel(struct ncclTuningInput_t* input, enum ncclSymkKernelId kernelId, size_t nBytes,
                               float* timeUs, float* selectionTimeUs, int* nBlocks) {
  if (ncclSymkGinKernelMask() >> kernelId & 1) {
    NCCLCHECK(ncclSymkGinModel(input, kernelId, nBytes, timeUs, nBlocks));
    *selectionTimeUs = *timeUs;
  } else {
    NCCLCHECK(ncclSymkLsaModel(input, kernelId, nBytes, timeUs, selectionTimeUs, nBlocks));
  }
  return ncclSuccess;
}

ncclResult_t ncclTuningSymkModelSim(struct ncclTuningInput_t* const inputs, struct ncclTuningResult_t* const tuning,
                                    struct ncclTuningModelState* /*internal*/) {
  ncclResult_t ret = ncclSuccess;
  tuning->selectionTimeUs = NCCL_TUNING_IGNORE;

  if (tuning->symKernelId == ncclSymkKernelId_Count) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if (!ncclSymkAvailable(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype, inputs->count)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  ncclSymkKernelMask tuning_kmask = 1ull << tuning->symKernelId;
  ncclSymkKernelMask valid_kmask = ncclSymkMask(inputs->comm, inputs->func, inputs->devRedOp, inputs->datatype,
                                                inputs->countMax, inputs->symAligned16B);
  ncclSymkKernelMask window_optional_kmask = ncclSymkLLKernelMask() | ncclGenkKernelMask();
  ncclSymkKernelMask ungrouped_ll_kmask = ncclSymkLLKernelMask() & ~ncclGenkKernelMask();
  if ((tuning_kmask & valid_kmask) == 0) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  // The specialized LL kernels do not support grouping; Flow protocols do.
  if ((inputs->nWorks > 1 && ((tuning_kmask & ungrouped_ll_kmask) != 0)) ||
      (inputs->func == ncclFuncAllReduce && inputs->winRegType != ncclSymSendRegRecvReg &&
       (tuning_kmask & window_optional_kmask) == 0) ||
      ((inputs->func == ncclFuncBroadcast || inputs->func == ncclFuncAllGather) &&
       inputs->winRegType != ncclSymSendRegRecvReg && inputs->winRegType != ncclSymSendNonregRecvReg &&
       (tuning_kmask & window_optional_kmask) == 0) ||
      ((inputs->func == ncclFuncReduce || inputs->func == ncclFuncReduceScatter) &&
       inputs->winRegType != ncclSymSendRegRecvReg && inputs->winRegType != ncclSymSendRegRecvNonreg &&
       (tuning_kmask & window_optional_kmask) == 0) ||
      (inputs->func == ncclFuncAllGather && inputs->winRegType != ncclSymSendRegRecvReg && inputs->comm->nNodes > 1 &&
       (tuning_kmask & ncclSymkGinKernelMask()) != 0)) {
    tuning->valid = 0;
    tuning->timeUs = -1.0;
    return ncclSuccess;
  }

  if ((tuning_kmask & ncclGenkKernelMask()) != 0) return ncclTuningGenkModelSim(inputs, tuning);

  float kTime = FLT_MAX;
  float kSelectionTime = FLT_MAX;
  int kBlocks = 0;
  NCCLCHECK(queryModel(inputs, (enum ncclSymkKernelId)tuning->symKernelId, inputs->nBytes, &kTime, &kSelectionTime,
                       &kBlocks));
  if (kBlocks <= 0 || !std::isfinite(kTime) || kTime >= disableTime) {
    tuning->valid = 0;
    tuning->timeUs = -1.0f;
    tuning->nChannels = 0;
    return ncclSuccess;
  }

  tuning->timeUs = kTime;
  tuning->selectionTimeUs = kSelectionTime;
  tuning->nChannels = kBlocks;
  tuning->nWarps = ncclSymkMaxThreads / WARP_SIZE;
  return ret;
}
