/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#include "mc_arena.h"
#include "alloc.h"
#include "bitops.h"
#include "comm.h"
#include "cudawrap.h"

#if CUDART_VERSION >= 12010

// The arena's reservation for a bind. Every alloc and free must derive the same reserved
// extent, or a free returns a different range than was reserved.
static size_t reserveSize(const struct ncclMcArena* arena, size_t bindSize) {
  return alignUp(bindSize, arena->partition.minGranularity);
}

ncclResult_t ncclMcArenaInit(struct ncclComm* comm, struct ncclMcArena* arena,
                             const struct ncclMcPartition* partition) {
  CUmemAllocationProp prop;

  arena->partition = *partition;

  memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = comm->cudaDev;
  prop.requestedHandleTypes = ncclCuMemHandleType;
  CUCHECK(cuMemGetAllocationGranularity(&arena->ucGranularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));

  ncclSpaceConstruct(&arena->space);
  return ncclSuccess;
}

void ncclMcArenaDestroy(struct ncclMcArena* arena) {
  ncclSpaceDestruct(&arena->space);
}

ncclResult_t ncclMcArenaRegister(struct ncclMcArena* arena, int n, const void** buffs, const size_t* sizes,
                                 struct ncclMcArenaReg** outRegs, enum ncclMcBindStatus* outStatus) {
  ncclResult_t ret = ncclSuccess;
  bool allBound = true;
  int nReserved = 0; // requests already holding arena space; exactly what the unwind returns
  int i;

  for (i = 0; i < n; i++) {
    outRegs[i] = nullptr;
    outStatus[i] = ncclMcBindStatusTransient;
  }

  // Deterministic first fit for every request: the offsets are a pure function of the
  // converged request sizes and the arena state, so every rank picks the same ones. All
  // ranges are held before anything binds, so a full arena unwinds without an unbind.
  for (i = 0; i < n; i++) {
    int64_t offset = 0;
    if (sizes[i] == 0) continue;
    NCCLCHECKGOTO(ncclCalloc(&outRegs[i], 1), ret, fail);
    outRegs[i]->bindSize = sizes[i];
    // A full arena is benign: the caller falls back to its unregistered path.
    if (ncclSpaceTryAlloc(&arena->space, (int64_t)arena->partition.size, (int64_t)reserveSize(arena, sizes[i]),
                          (int)arena->partition.minGranularity, &offset) != ncclSuccess) {
      ret = ncclInternalError;
      goto fail;
    }
    outRegs[i]->offset = (size_t)offset;
    outRegs[i]->mcBaseAddress = (CUdeviceptr)arena->partition.ptr + outRegs[i]->offset;
    nReserved = i + 1;
  }

  // Past the first bind nothing is undone locally: a range a peer may have bound must not
  // be freed here, so the records stay with the caller until the outcome has converged.
  for (i = 0; i < n; i++) {
    if (outRegs[i] == nullptr) continue;
    NCCLCHECKIGNORE(ncclMcPartitionTryBindAddr(&arena->partition, outRegs[i]->offset, (CUdeviceptr)buffs[i],
                                               outRegs[i]->bindSize, &outStatus[i]),
                    ret);
    outRegs[i]->bound = (outStatus[i] == ncclMcBindStatusOk);
    allBound &= outRegs[i]->bound;
  }
  if (!allBound && ret == ncclSuccess) ret = ncclInternalError;
  return ret;

fail:
  // Reached only before any bind, so the whole set goes straight back. A record past
  // nReserved holds no arena space yet, so only its memory is returned.
  for (i = 0; i < n; i++) {
    if (outRegs[i] == nullptr) continue;
    if (i < nReserved)
      NCCLCHECKIGNORE(ncclSpaceFree(&arena->space, (int64_t)outRegs[i]->offset,
                                    (int64_t)reserveSize(arena, outRegs[i]->bindSize)),
                      ret);
    free(outRegs[i]);
    outRegs[i] = nullptr;
  }
  return ret;
}

ncclResult_t ncclMcArenaDeregister(struct ncclMcArena* arena, struct ncclMcArenaReg* reg) {
  ncclResult_t ret = ncclSuccess;

  NCCLCHECKIGNORE(ncclSpaceFree(&arena->space, (int64_t)reg->offset, (int64_t)reserveSize(arena, reg->bindSize)), ret);
  free(reg);
  return ret;
}

#endif // CUDART_VERSION >= 12010
