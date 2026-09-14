/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_GPUNETIO_DEVICE_H_
#define NIIN_GPUNETIO_DEVICE_H_

#include "niin/gpunetio/context.h"

// The NIIN GPUNetIO backend is opt-in because its CUDA targets need the
// GPUNetIO device headers on their include path.  Its host-side provider owns
// that build dependency; ordinary NIIN users never include this header.
#if defined(__CUDACC__) && defined(NIIN_GPUNETIO_ENABLE) && NIIN_GPUNETIO_ENABLE
#if defined(__has_include)
#if __has_include(<gpunetio/doca_gpunetio_device.h>)
#include <gpunetio/doca_gpunetio_device.h>
#elif __has_include(<doca_gpunetio_device.h>)
// Some GPUNetIO SDK layouts put the public headers directly on the include
// path. Keep the NIIN source a customer of either published include layout.
#include <doca_gpunetio_device.h>
#else
#error "NIIN_GPUNETIO_ENABLE requires the GPUNetIO device headers"
#endif
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
