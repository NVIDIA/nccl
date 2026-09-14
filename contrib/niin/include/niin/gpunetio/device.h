/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_DEVICE_H_
#define NIIN_GPUNETIO_DEVICE_H_

#include "niin/gpunetio/context.h"

// The device side of the NIIN GPUNetIO backend is header-only and always
// compiled into CUDA consumers: NCCL ships the GPUNetIO sources and NIIN's
// build links their include tree into its own include/ directory, so
// `-I$NIIN_HOME/include` is all a consumer needs. NIIN_GPUNETIO_ENABLE=0 opts
// out and leaves network AMOs unimplemented.
#if defined(__CUDACC__) && (!defined(NIIN_GPUNETIO_ENABLE) || NIIN_GPUNETIO_ENABLE)
#if defined(__has_include) && !__has_include(<gpunetio/doca_gpunetio_device.h>)
#if !__has_include(<doca_gpunetio_device.h>)
#error "GPUNetIO device headers not found. Build NIIN from an NCCL source checkout so include/gpunetio exists, or compile with -DNIIN_GPUNETIO_ENABLE=0."
#endif
// Some GPUNetIO SDK layouts put the public headers directly on the include
// path. Keep the NIIN source a customer of either published include layout.
#include <doca_gpunetio_device.h>
#else
#include <gpunetio/doca_gpunetio_device.h>
#endif
#if !defined(DOCA_GPUNETIO_VERSION_MAJOR) || DOCA_GPUNETIO_VERSION_MAJOR != 4
#error "NIIN direct atomics require the GPUNetIO 4.x device ABI used by NCCL GIN"
#endif
#define NIIN_GPUNETIO_HAS_DEVICE_API 1
#else
#define NIIN_GPUNETIO_HAS_DEVICE_API 0
#endif

#endif  // NIIN_GPUNETIO_DEVICE_H_
