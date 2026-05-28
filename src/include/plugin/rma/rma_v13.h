/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2017-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef RMA_V13_H_
#define RMA_V13_H_

#if !defined(NCCL_OS_WINDOWS)
#include "gin/gin_v13.h"
typedef ncclGin_v13_t ncclRma_v13_t;
#endif

#endif
