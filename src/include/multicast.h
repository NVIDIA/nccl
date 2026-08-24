/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2016-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_MULTICAST_H_
#define NCCL_MULTICAST_H_

#include "nccl.h"
#include <cuda.h>
#include <stddef.h>

// A shared NVSwitch multicast (MC) group: one MC object whose full VA is mapped
// once, then handed out to consumers as immutable, non-overlapping slices. This
// keeps a whole NVLS domain to a single MC slot instead of one per consumer.

// One consumer's capacity request. A zero size marks the consumer absent: Build
// zeroes its partition, so callers can pass a fixed-shape request array and
// fill entries conditionally.
struct ncclMcRequest {
  size_t size;       // logical capacity; Build rounds the slice up to the group granularity
  size_t alignment;  // consumer alignment; effective = max(alignment, group granularity)
};

// A resolved slice within the group's mapped VA; immutable for the group's life.
// It carries everything a bind needs, so consumers never see the group itself.
struct ncclMcPartition {
  size_t offset;  // byte offset within the group's mapped VA
  size_t size;    // resolved size, aligned to the bind granularity
  void* ptr;     // groupBase + offset
  CUmemGenericAllocationHandle mcHandle;  // the group's MC object
  size_t minGranularity;                  // the group's minimum bind granularity
  int dev;                                // local device, for bind/unbind
};

struct ncclMcGroup;

#if CUDART_VERSION >= 12010

// Create one MC object and export its shareable handle (root rank), or import
// that handle on the other ranks.
ncclResult_t ncclMcCreate(struct ncclComm* comm, CUmulticastObjectProp* prop, int rank, unsigned int nranks,
                          CUmemGenericAllocationHandle* mcHandle, char* shareableHandle);
ncclResult_t ncclMcImport(struct ncclComm* comm, char* shareableHandle, int rank,
                          CUmemGenericAllocationHandle* mcHandle);

// Collectively create one MC object over the comm's NVLS local ranks, sized to
// the aligned sum of requests, map its full VA once, and return one resolved
// slice per request. On failure everything is unwound and *outGroup stays NULL.
ncclResult_t ncclMcGroupBuildPartitions(struct ncclComm* comm, const struct ncclMcRequest* requests, int nRequests,
                                        struct ncclMcGroup** outGroup, struct ncclMcPartition* outPartitions);

// Unmap the VA and release the MC handle (drops any bindings still present).
ncclResult_t ncclMcGroupDestroy(struct ncclMcGroup** group);

// Locally bind a UC handle into a partition at offsetInPartition; the bind is
// bounds-checked against the partition.
ncclResult_t ncclMcPartitionBindMem(const struct ncclMcPartition* partition, size_t offsetInPartition,
                                    CUmemGenericAllocationHandle mem, size_t memOffset, size_t bindSize);
// Unbind; a binding surviving a failed unbind lasts only until ncclMcGroupDestroy,
// so callers may keep tearing down.
ncclResult_t ncclMcPartitionUnbind(const struct ncclMcPartition* partition, size_t offsetInPartition, size_t bindSize);

#endif // CUDART_VERSION >= 12010

#endif // NCCL_MULTICAST_H_
