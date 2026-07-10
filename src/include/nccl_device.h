/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef _NCCL_DEVICE_H_
#define _NCCL_DEVICE_H_

#include "nccl_device/coop.h"
#include "nccl_device/impl/barrier__funcs.h"
#include "nccl_device/impl/comm__funcs.h"
#include "nccl_device/impl/core__funcs.h"
#include "nccl_device/impl/ll_a2a__funcs.h"
#include "nccl_device/impl/lsa_barrier__funcs.h"
#if !defined(NCCL_OS_WINDOWS)
#include "nccl_device/impl/gin__funcs.h"
#include "nccl_device/impl/gin_barrier__funcs.h"
#endif
#include "nccl_device/impl/ptr__funcs.h"
#include "nccl_device/impl/reduce_copy__funcs.h"

#endif // _NCCL_DEVICE_H_
