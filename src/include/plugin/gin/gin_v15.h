/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef GIN_V15_H_
#define GIN_V15_H_
#include "gin/gin_v14.h"

// v15 is identical to v14; it is aliased so the plugin version can be bumped without duplicating the definitions.
typedef ncclGinProperties_v14_t ncclGinProperties_v15_t;
typedef ncclGinConfig_v14_t ncclGinConfig_v15_t;
typedef ncclGin_v14_t ncclGin_v15_t;
#endif // end include guard
