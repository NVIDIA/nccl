/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_TUNER_H_
#define NCCL_INT_TUNER_H_

#include "nccl_tuner.h"

// Tuning plugin to override NCCL's default algorithm/protocol tuning.

// Attempts to load NCCL tuner from environmental variable.
// Returns ncclSuccess if the correct tuner symbol has been found and
// successully loaded.  Otherwise returns an error and also logs the error.
ncclResult_t ncclTunerPluginLoad(ncclTuner_t** tuner);

// Cleans up NCCL tuner plugin.
ncclResult_t ncclTunerPluginUnload(ncclTuner_t** tuner);
#endif
