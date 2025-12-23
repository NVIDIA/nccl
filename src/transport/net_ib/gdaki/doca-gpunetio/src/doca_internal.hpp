/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <vector>

#include <stdint.h>
#include <stddef.h>
#include <string>
#include <cmath>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <syslog.h>
#include <linux/types.h>
#include <cuda_runtime.h>

#include "host/doca_error.h"
#include "doca_gpunetio_config.h"
#include "doca_gpunetio_log.hpp"

#ifndef CUDA_ROUND_UP
#define CUDA_ROUND_UP(unaligned_mapping_size, align_val) \
    ((unaligned_mapping_size) + (align_val) - 1) & (~((align_val) - 1))
#endif

#ifndef CUDA_ROUND_DOWN
#define CUDA_ROUND_DOWN(unaligned_mapping_size, align_val) \
    ((unaligned_mapping_size) & ~((align_val) - 1))
#endif

#define DOCA_VERBS_PAGE_SIZE 4096
#define DOCA_VERBS_CACHELINE_SIZE (64)

#define DOCA_VERBS_DB_UAR_SIZE 8

static inline cudaError_t doca_verbs_cuda_clear_error(cudaError_t cuda_result) {
    if (cuda_result != cudaSuccess) cudaGetLastError();
    return cuda_result;
}

#define DOCA_VERBS_CUDA_CALL_CLEAR_ERROR(cmd) doca_verbs_cuda_clear_error(cmd)

/**
 * @brief This method checks if a number is a power of 2
 *
 * @param [in] x
 * The number to check
 * @return true if x is a power of 2, false if not.
 */
inline bool doca_internal_utils_is_power_of_two(uint64_t x) { return x && (x & (x - 1)) == 0; }

inline uint64_t doca_internal_utils_next_power_of_two(uint64_t x) {
    x--;

    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;

    return x + 1;
}

struct doca_internal_mlx5_wqe_data_seg {
    __be32 byte_count;
    __be32 lkey;
    __be64 addr;
};

struct doca_internal_mlx5_wqe_mprq_next_seg {
    uint8_t rsvd0[2];
    __be16 next_wqe_index;
    uint8_t signature;
    uint8_t rsvd1[11];
};

template <typename T>
T doca_internal_utils_log2(T x) {
    if (x == 0) /* log(0) is undefined */
        return 0;

    return static_cast<T>(std::log2(x));
}

inline uint64_t doca_internal_utils_align_up_uint64(uint64_t value, uint64_t alignment) {
    uint64_t remainder = (value % alignment);

    if (remainder == 0) return value;

    return value + (alignment - remainder);
}

inline uint32_t doca_internal_utils_align_up_uint32(uint32_t value, uint32_t alignment) {
    return (uint32_t)doca_internal_utils_align_up_uint64(value, alignment);
}
