/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include <stdlib.h>
#include <stdint.h>
#include <algorithm> // For std::min/std::max
#include "nccl.h"

#ifdef NCCL_OS_WINDOWS
  // On Windows, nccl.h already provides extern "C" declarations for all API functions.
  // Exports are controlled via nccl.def (module definition file) to avoid C2375 linkage
  // conflicts that arise when mixing extern "C" __declspec(dllexport) forward-declarations
  // with un-attributed function definitions in the same translation unit.
  // NCCL_API is used only as a redundant forward-declaration (no-op here).
  #define NCCL_API(ret, func, ...) ret func(__VA_ARGS__)
#elif defined(PROFAPI)
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI / NCCL_OS_WINDOWS

#include "debug.h"
#include "checks.h"
#include "cudawrap.h"
#include "alloc.h"
#include "utils.h"
#include "param.h"
#include "nvtx.h"

#endif // end include guard
