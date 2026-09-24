/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMA_V17_H_
#define RMA_V17_H_
#include "rma/rma_v16.h"

// v17 is identical to v16; it is aliased so the plugin version can be bumped without duplicating the definitions.
typedef ncclRmaConfig_v16_t ncclRmaConfig_v17_t;
typedef ncclRma_v16_t ncclRma_v17_t;
#endif
