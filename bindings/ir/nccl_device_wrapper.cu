/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * NCCL Device API force instantiation and C style APIs for LLVM IR generation.
 */

// nccl_device_wrapper.h must come first: it installs the build-specific
// inlining overrides that the device-API definitions in nccl_device.h below
// are then compiled with, so the library emits real symbols.
#include "nccl_device_wrapper.h"
// The bitcode library is device-only: suppress "nccl_device/host.h" by
// pre-defining its include guard, so the host entrypoints never enter this TU.
#define _NCCL_DEVICE_HOST_H_
#include "nccl_device.h"
#ifdef NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER
#error "nccl_device/host.h leaked in: its include guard was renamed, update the #define above"
#endif
#include "util.h"

#include <new>

#include "impl/core.h"
#include "impl/coop.h"
#include "impl/gin.h"
#include "impl/barrier.h"
#include "impl/reduce_copy.h"
