/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstdio>
#include "cuda_runtime.h"
#include "nccl.h"
#include "reshard_types.h"
#include "reshard_checks.h"
#include "reshard_log.h"
#include "reshard_internal.h"

/* ======================================================================
 * Per-comm symmetric transpose buffer pool.
 *
 * One buffer per ncclComm_t, reused across sequential collective calls.
 * When a different stream reuses the same comm's buffer, a per-entry
 * cudaEvent_t + cudaStreamWaitEvent serializes access.
 *
 * Growth (high-water-mark): the old buffer is parked in gRetired and
 * freed only at finalization.  This avoids any cudaDeviceSynchronize
 * or cudaStreamSynchronize on the hot path.
 * ====================================================================*/

static TransposeBufferEntry gPool[MAX_TRANSPOSE_BUFFER_ENTRIES];
static int gPoolCount = 0;

#define MAX_RETIRED_BUFFERS (MAX_TRANSPOSE_BUFFER_ENTRIES * 2)
static void* gRetired[MAX_RETIRED_BUFFERS];
static int gRetiredCount = 0;

static TransposeBufferEntry* findPoolEntry(ncclComm_t comm) {
  for (int i = 0; i < gPoolCount; i++)
    if (gPool[i].comm == comm && gPool[i].allocated) return &gPool[i];
  return nullptr;
}

ncclResult_t ensureTransposeBuffer(ncclComm_t comm, size_t requiredBytes, cudaStream_t stream) {
  TransposeBufferEntry* entry = findPoolEntry(comm);

  if (entry != nullptr) {
    if (entry->stream != stream) {
      NCCLXFER_CUDACHECK(cudaStreamWaitEvent(stream, entry->event, 0));
      entry->stream = stream;
    }

    if (entry->capacity >= requiredBytes) return ncclSuccess;

    /* Growth: retire the old buffer and allocate a larger one. */
    if (gRetiredCount >= MAX_RETIRED_BUFFERS) {
      fprintf(stderr,
              "[nccl-reshard] Transpose retired-buffer list full (%d); "
              "too many buffer growths.\n",
              MAX_RETIRED_BUFFERS);
      return ncclInternalError;
    }
    RESHARD_DEBUG(-1,
                  "Transpose buffer growing for comm %p: %zu -> %zu bytes "
                  "(retiring %p)",
                  (void*)comm, entry->capacity, requiredBytes, entry->buffer);
    gRetired[gRetiredCount++] = entry->buffer;
    entry->buffer = nullptr;
    entry->capacity = 0;

    NCCLXFER_CHECK(ncclMemAlloc(&entry->buffer, requiredBytes));
    entry->capacity = requiredBytes;
    return ncclSuccess;
  }

  /* New comm — allocate a fresh slot. */
  if (gPoolCount >= MAX_TRANSPOSE_BUFFER_ENTRIES) {
    fprintf(stderr,
            "[nccl-reshard] Transpose buffer pool full (%d entries); "
            "increase MAX_TRANSPOSE_BUFFER_ENTRIES.\n",
            MAX_TRANSPOSE_BUFFER_ENTRIES);
    return ncclInvalidArgument;
  }

  TransposeBufferEntry& e = gPool[gPoolCount++];
  e.comm = comm;
  e.stream = stream;
  e.event = nullptr;
  e.buffer = nullptr;
  e.capacity = 0;
  e.allocated = true;

  NCCLXFER_CUDACHECK(cudaEventCreateWithFlags(&e.event, cudaEventDisableTiming));

  NCCLXFER_CHECK(ncclMemAlloc(&e.buffer, requiredBytes));
  e.capacity = requiredBytes;
  RESHARD_DEBUG(-1, "Transpose buffer allocated for comm %p: %zu bytes (%p) [slot %d]", (void*)comm, requiredBytes,
                e.buffer, gPoolCount - 1);
  return ncclSuccess;
}

void* getTransposeBuffer(ncclComm_t comm) {
  TransposeBufferEntry* e = findPoolEntry(comm);
  return (e != nullptr) ? e->buffer : nullptr;
}

size_t getTransposeBufferCapacity(ncclComm_t comm) {
  TransposeBufferEntry* e = findPoolEntry(comm);
  return (e != nullptr) ? e->capacity : 0;
}

ncclResult_t transposeBufferRecordEvent(ncclComm_t comm, cudaStream_t stream) {
  TransposeBufferEntry* e = findPoolEntry(comm);
  if (e != nullptr) NCCLXFER_CUDACHECK(cudaEventRecord(e->event, stream));
  return ncclSuccess;
}

void transposeBufferFinalize() {
  for (int i = 0; i < gPoolCount; i++) {
    if (gPool[i].event != nullptr) cudaEventDestroy(gPool[i].event);
    if (gPool[i].buffer != nullptr) ncclMemFree(gPool[i].buffer);
    gPool[i] = {};
  }
  gPoolCount = 0;

  for (int i = 0; i < gRetiredCount; i++) {
    if (gRetired[i] != nullptr) {
      ncclMemFree(gRetired[i]);
      gRetired[i] = nullptr;
    }
  }
  gRetiredCount = 0;
}

bool shouldTransposeForCrossDim(const size_t* srcDimsBytes, const size_t* dstDimsBytes, int ndims, int srcShardDim,
                                int dstShardDim, int srcShardCount, int dstShardCount, int* swapDimA, int* swapDimB) {
  // 2D case: replicated src (or shard on dim 0) -> dst shards innermost dim
  if (ndims == 2 && dstShardDim == 1 && srcShardDim != 1) {
    const size_t* dims = (dstDimsBytes[0] > 0) ? dstDimsBytes : srcDimsBytes;
    size_t innerSize = dims[1];
    if (innerSize < CROSS_DIM_TRANSPOSE_THRESHOLD) {
      *swapDimA = 0;
      *swapDimB = 1;
      return true;
    }
    return false;
  }

  if (ndims != 3) return false;
  if (srcShardDim < 0 || dstShardDim < 0) return false;
  if (srcShardDim == dstShardDim) return false;

  const size_t* dims = (srcDimsBytes[0] > 0) ? srcDimsBytes : dstDimsBytes;

  size_t globalInner = dims[ndims - 1];
  if (srcShardDim == ndims - 1) globalInner *= srcShardCount;
  if (dstShardDim == ndims - 1) globalInner *= dstShardCount;

  size_t innerSize = globalInner;
  if (dstShardDim == ndims - 1) innerSize = globalInner / dstShardCount;
  if (srcShardDim == ndims - 1) innerSize = globalInner / srcShardCount;

  if (innerSize >= CROSS_DIM_TRANSPOSE_THRESHOLD) return false;

  int freeDim = -1;
  for (int d = 0; d < ndims; d++) {
    if (d != srcShardDim && d != dstShardDim) {
      freeDim = d;
      break;
    }
  }
  if (freeDim < 0) return false;

  if (dstShardDim == ndims - 1 && srcShardDim != ndims - 2) {
    *swapDimA = ndims - 2;
    *swapDimB = ndims - 1;
    return true;
  }

  return false;
}
