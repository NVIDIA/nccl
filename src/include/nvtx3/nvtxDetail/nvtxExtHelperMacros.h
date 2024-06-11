/*
* Copyright 2023  NVIDIA Corporation.  All rights reserved.
*
* Licensed under the Apache License v2.0 with LLVM Exceptions.
* See https://llvm.org/LICENSE.txt for license information.
* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
*/

#ifndef NVTX_EXT_HELPER_MACROS_H
#define NVTX_EXT_HELPER_MACROS_H

/* Combine tokens */
#define _NVTX_EXT_CONCAT(a, b) a##b
#define NVTX_EXT_CONCAT(a, b) _NVTX_EXT_CONCAT(a, b)

/* Resolves to the number of arguments passed. */
#define NVTX_EXT_NUM_ARGS(...) \
    NVTX_EXT_SELECTA16(__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, throwaway)
#define NVTX_EXT_SELECTA16(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, ...) a16

/* Cast argument(s) to void to prevent unused variable warnings. */
#define _NVTX_EXT_VOIDIFY1(a1) (void)a1;
#define _NVTX_EXT_VOIDIFY2(a1, a2) (void)a1; (void)a2;
#define _NVTX_EXT_VOIDIFY3(a1, a2, a3) (void)a1; (void)a2; (void)a3;
#define _NVTX_EXT_VOIDIFY4(a1, a2, a3, a4) (void)a1; (void)a2; (void)a3; (void)a4;

/* Mark function arguments as unused. */
#define NVTX_EXT_HELPER_UNUSED_ARGS(...) \
    NVTX_EXT_CONCAT(_NVTX_EXT_VOIDIFY, NVTX_EXT_NUM_ARGS(__VA_ARGS__))(__VA_ARGS__)

#endif /* NVTX_EXT_HELPER_MACROS_H */