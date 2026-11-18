/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *************************************************************************/

#ifndef NCCL_SYM_PARTITION_H_
#define NCCL_SYM_PARTITION_H_

#include <stdint.h>

#if defined(__CUDACC__)
#define NCCL_SYM_PARTITION_HD __host__ __device__
#else
#define NCCL_SYM_PARTITION_HD
#endif

#define NCCL_SYM_KERNEL_CELL_SIZE 1024 // no less than 16 bytes minimal cell size
constexpr uint64_t ncclSymkFractionScale = 1 << 16;

NCCL_SYM_PARTITION_HD constexpr uint64_t ncclSymkDivUp(uint64_t count, uint64_t divisor) {
  return count / divisor + (count % divisor != 0);
}

// Keep the scheduler's two rounding steps, including grouped work in element units.
NCCL_SYM_PARTITION_HD constexpr uint64_t ncclSymkCellsPerChannel(uint64_t count, int channels, uint64_t cellCount) {
  return ncclSymkDivUp(ncclSymkDivUp(count, channels), cellCount);
}

// The work descriptor stores the inclusive high fraction as this value minus one.
NCCL_SYM_PARTITION_HD constexpr uint64_t ncclSymkCellFraction(uint64_t cell, uint64_t cells) {
  return ncclSymkDivUp(ncclSymkFractionScale * cell, cells);
}

NCCL_SYM_PARTITION_HD constexpr uint64_t ncclSymkFractionCell(uint64_t fraction, uint64_t cells) {
  return fraction * cells / ncclSymkFractionScale;
}

#undef NCCL_SYM_PARTITION_HD
#endif // NCCL_SYM_PARTITION_H_
