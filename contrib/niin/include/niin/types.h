/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NIIN_TYPES_H_
#define NIIN_TYPES_H_

#include <stdint.h>
#include <stddef.h>

// NVSHMEM comparison operators
#define NVSHMEM_CMP_EQ 0
#define NVSHMEM_CMP_NE 1
#define NVSHMEM_CMP_GT 2
#define NVSHMEM_CMP_GE 3
#define NVSHMEM_CMP_LT 4
#define NVSHMEM_CMP_LE 5

// NVSHMEM signal operators
#define NVSHMEM_SIGNAL_SET 0
#define NVSHMEM_SIGNAL_ADD 1

// NVSHMEM team type -- opaque handle
typedef int nvshmem_team_t;
#define NVSHMEM_TEAM_WORLD              0
#define NVSHMEM_TEAM_SHARED             1
#define NVSHMEMX_TEAM_NODE              2
#define NVSHMEMX_TEAM_SAME_MYPE_NODE    3  // Same local rank across nodes (rail)
#define NVSHMEMI_TEAM_SAME_GPU          4  // PEs sharing same GPU (size=1 in NCCL)
#define NVSHMEMI_TEAM_GPU_LEADERS       5  // One PE per GPU (=WORLD in NCCL)
#define NVSHMEM_TEAM_INVALID           (-1)

// X-macro for standard RMA types (matching NVSHMEM's type set)
// Format: X(TYPENAME, TYPE)
#define NIIN_STANDARD_RMA_TYPES(X) \
  X(float,       float)            \
  X(double,      double)           \
  X(char,        char)             \
  X(schar,       signed char)      \
  X(short,       short)            \
  X(int,         int)              \
  X(long,        long)             \
  X(longlong,    long long)        \
  X(uchar,       unsigned char)    \
  X(ushort,      unsigned short)   \
  X(uint,        unsigned int)     \
  X(ulong,       unsigned long)    \
  X(ulonglong,   unsigned long long) \
  X(int8,        int8_t)           \
  X(int16,       int16_t)          \
  X(int32,       int32_t)          \
  X(int64,       int64_t)          \
  X(uint8,       uint8_t)          \
  X(uint16,      uint16_t)        \
  X(uint32,      uint32_t)        \
  X(uint64,      uint64_t)        \
  X(size,        size_t)           \
  X(ptrdiff,     ptrdiff_t)

// Types valid for AMO (atomic memory operations)
#define NIIN_AMO_STANDARD_TYPES(X) \
  X(int,         int)              \
  X(long,        long)             \
  X(longlong,    long long)        \
  X(uint,        unsigned int)     \
  X(ulong,       unsigned long)    \
  X(ulonglong,   unsigned long long) \
  X(int32,       int32_t)          \
  X(int64,       int64_t)          \
  X(uint32,      uint32_t)        \
  X(uint64,      uint64_t)        \
  X(size,        size_t)

// Types valid for bitwise AMOs (and, or, xor)
#define NIIN_AMO_BITWISE_TYPES(X) \
  X(uint,        unsigned int)     \
  X(ulong,       unsigned long)    \
  X(ulonglong,   unsigned long long) \
  X(int32,       int32_t)          \
  X(int64,       int64_t)          \
  X(uint32,      uint32_t)        \
  X(uint64,      uint64_t)

// Types for wait/test operations
#define NIIN_WAIT_TYPES(X) \
  X(short,       short)            \
  X(int,         int)              \
  X(long,        long)             \
  X(longlong,    long long)        \
  X(ushort,      unsigned short)   \
  X(uint,        unsigned int)     \
  X(ulong,       unsigned long)    \
  X(ulonglong,   unsigned long long) \
  X(int32,       int32_t)          \
  X(int64,       int64_t)          \
  X(uint32,      uint32_t)        \
  X(uint64,      uint64_t)        \
  X(size,        size_t)

// Sized RMA type list (put8, put16, put32, put64, put128)
// Format: X(SIZE_SUFFIX, BYTE_COUNT)
#define NIIN_SIZED_RMA(X) \
  X(8,   1)               \
  X(16,  2)               \
  X(32,  4)               \
  X(64,  8)               \
  X(128, 16)

// Helper to evaluate comparison operators
__device__ __forceinline__ bool niin_cmp_eval(int cmp, long long a, long long b) {
  switch (cmp) {
    case NVSHMEM_CMP_EQ: return a == b;
    case NVSHMEM_CMP_NE: return a != b;
    case NVSHMEM_CMP_GT: return a > b;
    case NVSHMEM_CMP_GE: return a >= b;
    case NVSHMEM_CMP_LT: return a < b;
    case NVSHMEM_CMP_LE: return a <= b;
    default: return false;
  }
}

// Unsigned version for unsigned types
__device__ __forceinline__ bool niin_cmp_eval_u(int cmp, unsigned long long a, unsigned long long b) {
  switch (cmp) {
    case NVSHMEM_CMP_EQ: return a == b;
    case NVSHMEM_CMP_NE: return a != b;
    case NVSHMEM_CMP_GT: return a > b;
    case NVSHMEM_CMP_GE: return a >= b;
    case NVSHMEM_CMP_LT: return a < b;
    case NVSHMEM_CMP_LE: return a <= b;
    default: return false;
  }
}

#endif // NIIN_TYPES_H_
