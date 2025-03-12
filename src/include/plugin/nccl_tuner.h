/*************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include "nccl.h"
#include "nccl_common.h"

#include "tuner/tuner_v4.h"
#include "tuner/tuner_v3.h"
#include "tuner/tuner_v2.h"

typedef ncclTuner_v4_t ncclTuner_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v4"

#endif
