/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MC_ARENA_H_
#define NCCL_MC_ARENA_H_

#include "multicast.h"
#include "allocator.h"
#include "utils.h"

// One registration in the arena; a bind and its matching unbind must use the same
// (offset, bindSize).
struct ncclMcArenaReg {
  size_t offset;               // offset within the partition
  size_t bindSize;             // bound extent
  CUdeviceptr mcBaseAddress;   // partition.ptr + offset
  bool bound;                  // a bind at (offset, bindSize) is live on this rank
};

// Sub-allocator over one MC partition. Its state is bit-identical across ranks because it
// mutates only on converged decisions.
struct ncclMcArena {
  struct ncclMcPartition partition; // value copy; the group keeps ownership
  size_t ucGranularity;             // UC allocation granularity; eligibility alignment
  struct ncclSpace space;           // free space over [0, partition.size)
};

#if CUDART_VERSION >= 12010

ncclResult_t ncclMcArenaInit(struct ncclComm* comm, struct ncclMcArena* arena, const struct ncclMcPartition* partition);
void ncclMcArenaDestroy(struct ncclMcArena* arena);

// Reserve and bind one range per nonzero request; a zero size leaves that outRegs entry NULL.
// The reservation is all or nothing: a full arena unwinds it entirely and leaves every outRegs
// entry NULL (a benign error, uniform across ranks). A bind failure keeps the records and their
// reservations, with reg->bound telling which binds took: the caller must dispose of them once
// the outcome has converged. outStatus reports each bind's verdict, whatever the return value.
ncclResult_t ncclMcArenaRegister(struct ncclMcArena* arena, int n, const void** buffs, const size_t* sizes,
                                 struct ncclMcArenaReg** outRegs, enum ncclMcBindStatus* outStatus);

// Roll a record back into free space without unbinding. Only legal when the caller knows no
// rank ever bound the range.
ncclResult_t ncclMcArenaDeregister(struct ncclMcArena* arena, struct ncclMcArenaReg* reg);

#endif // CUDART_VERSION >= 12010

#endif // NCCL_MC_ARENA_H_
