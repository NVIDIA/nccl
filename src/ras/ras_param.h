/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_RAS_PARAM_H_
#define NCCL_RAS_PARAM_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float ncclParamRasTimeoutFactor(void);
int64_t rasTimeoutFactorNs(int64_t baseSeconds);
double rasTimeoutFactorSec(int baseSeconds);

#ifdef __cplusplus
}
#endif

#endif // NCCL_RAS_PARAM_H_
