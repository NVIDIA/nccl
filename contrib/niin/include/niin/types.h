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

// ---------------------------------------------------------------------------
// TMA (Tensor Memory Accelerator) shared-memory registration
//
// NIIN mirrors NVSHMEM's TMA API: applications hand a slice of their CTA's
// shared memory to the runtime with nvshmemx_give_smem(), and the LSA (NVLink)
// put/get paths then route through cp.async.bulk instead of vectorized
// load/store. Sizing comes from nvshmemx_ask_smem().
// ---------------------------------------------------------------------------

// Static carve at the base of every nvshmemx_give_smem() buffer, reserved for
// NIIN internal use (mbarriers used to track async-proxy completion). User data
// for TMA staging starts at smem + NIIN_SMEM_DATA_REGION_OFFSET.
// Layout: 32 slots of 16 bytes each; slot 0 is at offset 0.
#define NIIN_TMA_BARRIER_REGION_BYTES 512
#define NIIN_TMA_NUM_BARRIER_SLOTS    32
#define NIIN_SMEM_DATA_REGION_OFFSET  NIIN_TMA_BARRIER_REGION_BYTES

// CTAs with a linear block id at or beyond this bound cannot register shared
// memory and fall back to the vectorized load/store path.
#define NIIN_TMA_MAX_BLOCKS 4096

typedef enum {
  NVSHMEMX_TMA_DISABLE = 0,  // Do not use TMA for transfers (default)
  NVSHMEMX_TMA_ENABLE  = 1,  // Use TMA when smem is given and the arch supports it
  NVSHMEMX_TMA_FORCE   = 2,  // Require TMA; initialization fails if unavailable
  NVSHMEMX_TMA_POLICY_MAX = 0x7fffffff
} nvshmemx_tma_policy_t;

typedef enum {
  NVSHMEMX_SMEM_RECOMMENDED   = 0,
  NVSHMEMX_SMEM_MINIMUM       = 1,
  NVSHMEMX_SMEM_BARRIERS_ONLY = 2,
  NVSHMEMX_SMEM_AMOUNT_MAX = 0x7fffffff
} nvshmemx_smem_amount_t;

// nvshmemx_ask_smem: shared memory (in bytes) NIIN wants for TMA transfers.
//
// Available on host and device. On the host the return value sizes the dynamic
// shared memory of a kernel launch; on the device it sizes the buffer passed to
// nvshmemx_give_smem(). Every returned value includes
// NIIN_SMEM_DATA_REGION_OFFSET at the base of the buffer; TMA data tiles live
// in the remainder.
//
//   NVSHMEMX_SMEM_RECOMMENDED   - 64 KiB. The largest single cp.async.bulk
//                                 transfer on Hopper/Blackwell, sized so the
//                                 gmem->gmem staging path gets two full tiles.
//   NVSHMEMX_SMEM_MINIMUM       - 32 KiB. Enough for single-buffered staging
//                                 and for smem->gmem puts.
//   NVSHMEMX_SMEM_BARRIERS_ONLY - Barriers only, no data tile. The gmem->gmem
//                                 staging path stays off, but smem->gmem puts
//                                 still work when the application manages its
//                                 own staging buffer.
__host__ __device__ __forceinline__ int nvshmemx_ask_smem(nvshmemx_smem_amount_t flag) {
  switch (flag) {
    case NVSHMEMX_SMEM_RECOMMENDED:   return 65536;  // 64 KiB
    case NVSHMEMX_SMEM_MINIMUM:       return 32768;  // 32 KiB
    case NVSHMEMX_SMEM_BARRIERS_ONLY: return NIIN_SMEM_DATA_REGION_OFFSET;
    default:                          return 65536;
  }
}

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
  X(size,        size_t)          \
  X(ptrdiff,     ptrdiff_t)

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
