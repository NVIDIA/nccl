/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_PORTABLE_INTRINSICS_H
#define NCCL_PORTABLE_INTRINSICS_H

#ifdef __cplusplus
  extern "C++" {
#endif

#include <atomic>

#ifdef __cplusplus
  }
#endif

// Compiler detection macros
#if defined(__GNUC__) || defined(__clang__)
  #define NCCL_COMPILER_GCC 1
  #include "compiler/gcc.h"
#elif defined(_MSC_VER)
  #define NCCL_COMPILER_MSVC 1
  #include "compiler/msvc.h"
#else
  #error "Unsupported compiler"
#endif

#endif // NCCL_PORTABLE_INTRINSICS_H
