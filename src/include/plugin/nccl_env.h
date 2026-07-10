/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_ENV_H_
#define NCCL_ENV_H_

#include "env/env_v1.h"
#include "env/env_v2.h"

typedef ncclEnv_v2_t ncclEnv_t;

#define NCCL_ENV_PLUGIN_SYMBOL ncclEnvPlugin_v2

#endif // end include guard
