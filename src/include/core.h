/*************************************************************************
 * Copyright (c) 2015-2021, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_CORE_H_
#define NCCL_CORE_H_

#include "platform.h"

#if NCCL_PLATFORM_WINDOWS
#include "platform/win32_thread.h"
#include "platform/win32_misc.h"
#else
#include <pthread.h>
#include <unistd.h>
#endif

#include <stdlib.h>
#include <stdint.h>
#include <algorithm> // For std::min/std::max
#include "nccl.h"

#if NCCL_PLATFORM_WINDOWS
#ifdef PROFAPI
#define NCCL_API(ret, func, ...) \
    extern "C"                   \
        __declspec(dllexport)    \
        ret                      \
        func(__VA_ARGS__)
#else
#define NCCL_API(ret, func, ...) \
    extern "C"                   \
        __declspec(dllexport)    \
        ret                      \
        func(__VA_ARGS__)
#endif // end PROFAPI
#else
#ifdef PROFAPI
#define NCCL_API(ret, func, args...)           \
    extern "C"                                 \
        __attribute__((visibility("default"))) \
        __attribute__((alias(#func)))          \
        ret p##func(args);                     \
    extern "C"                                 \
        __attribute__((visibility("default"))) \
        __attribute__((weak))                  \
        ret                                    \
        func(args)
#else
#define NCCL_API(ret, func, args...)           \
    extern "C"                                 \
        __attribute__((visibility("default"))) \
        ret                                    \
        func(args)
#endif // end PROFAPI
#endif // NCCL_PLATFORM_WINDOWS

#include "debug.h"
#include "checks.h"
#include "cudawrap.h"
#include "alloc.h"
#include "utils.h"
#include "param.h"
#include "nvtx.h"

#endif // end include guard
