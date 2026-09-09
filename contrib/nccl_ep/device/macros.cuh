/*
 * Portions of this file are adapted from DeepEP (https://github.com/deepseek-ai/DeepEP).
 * Copyright (c) 2025 DeepSeek. Licensed under the MIT License.
 * SPDX-License-Identifier: MIT
 */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 */

#pragma once

#include <exception>
#include <string>
#include <cuda_runtime.h>

//==============================================================================
// Static Assert
//==============================================================================

#ifndef EP_STATIC_ASSERT
#define EP_STATIC_ASSERT(cond, reason) static_assert(cond, reason)
#endif

//==============================================================================
// Exception Handling
//==============================================================================

class EPException : public std::exception {
private:
    std::string message = {};

public:
    explicit EPException(const char* name, const char* file, const int line, const std::string& error) {
        message = std::string("Failed: ") + name + " error " + file + ":" + std::to_string(line) + " '" + error + "'";
    }

    const char* what() const noexcept override {
        return message.c_str();
    }
};

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd) \
    do { \
        cudaError_t e = (cmd); \
        if (e != cudaSuccess) { \
            throw EPException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
        } \
    } while (0)
#endif

#ifndef EP_HOST_ASSERT
#define EP_HOST_ASSERT(cond) \
    do { \
        if (not(cond)) { \
            throw EPException("Assertion", __FILE__, __LINE__, #cond); \
        } \
    } while (0)
#endif

#ifndef EP_DEVICE_ASSERT
#define EP_DEVICE_ASSERT(cond) \
    do { \
        if (not(cond)) { \
            printf("Assertion failed: %s:%d, condition: %s\n", __FILE__, __LINE__, #cond); \
            asm("trap;"); \
        } \
    } while (0)
#endif

//==============================================================================
// Kernel Launch Configuration
//==============================================================================

#ifndef SETUP_LAUNCH_CONFIG
#ifndef DISABLE_SM90_FEATURES
#define SETUP_LAUNCH_CONFIG(num_sms, num_threads, stream) \
    cudaLaunchConfig_t cfg = {(num_sms), (num_threads), 0, stream, nullptr, 0}; \
    cudaLaunchAttribute attr[2]; \
    attr[0].id = cudaLaunchAttributeCooperative; \
    attr[0].val.cooperative = 1; \
    attr[1].id = cudaLaunchAttributeClusterDimension; \
    attr[1].val.clusterDim.x = (num_sms % 2 == 0 ? 2 : 1); \
    attr[1].val.clusterDim.y = 1; \
    attr[1].val.clusterDim.z = 1; \
    cfg.attrs = attr; \
    cfg.numAttrs = 2
#else
#define SETUP_LAUNCH_CONFIG(sms, threads, stream) \
    int __num_sms = (sms); \
    int __num_threads = (threads); \
    auto __stream = (stream)
#endif
#endif

#ifndef LAUNCH_KERNEL
#ifndef DISABLE_SM90_FEATURES
#define LAUNCH_KERNEL(config, kernel, ...) CUDA_CHECK(cudaLaunchKernelEx(config, kernel, ##__VA_ARGS__))
#else
#define LAUNCH_KERNEL(config, kernel, ...) \
    do { \
        kernel<<<__num_sms, __num_threads, 0, __stream>>>(__VA_ARGS__); \
        cudaError_t e = cudaGetLastError(); \
        if (e != cudaSuccess) { \
            EPException cuda_exception("CUDA", __FILE__, __LINE__, cudaGetErrorString(e)); \
            fprintf(stderr, "%s\n", cuda_exception.what()); \
            throw cuda_exception; \
        } \
    } while (0)
#endif
#endif
