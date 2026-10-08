/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_COLL_SIZES_H_
#define NCCL_COLL_SIZES_H_

#include "device.h"

#define NCCL_MAX_NET_SIZE (1024 * 1024 * 1024L) // Rather than send INT_MAX which is 2G-1, send a power of two.

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS / 4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS / 2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS / 4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS / 2)
#define ALLTOALL_SLICESTEPS 1
#define ALLTOALL_CHUNKSTEPS 1
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS / 4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS / 2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define GATHER_SLICESTEPS 1
#define GATHER_CHUNKSTEPS 1
#define SCATTER_SLICESTEPS 1
#define SCATTER_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
  case ncclFloat8e4m3:
  case ncclFloat8e5m2:
    return 1;
  case ncclFloat16:
  case ncclBfloat16:
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

inline int ncclDefaultChunkStep(ncclFunc_t func) {
  switch (func) {
  case ncclFuncAllReduce:
    return ALLREDUCE_CHUNKSTEPS;
  case ncclFuncAllGather:
    return ALLGATHER_CHUNKSTEPS;
  case ncclFuncAlltoAll:
    return ALLTOALL_CHUNKSTEPS;
  case ncclFuncReduceScatter:
    return REDUCESCATTER_CHUNKSTEPS;
  case ncclFuncBroadcast:
    return BROADCAST_CHUNKSTEPS;
  case ncclFuncGather:
    return GATHER_CHUNKSTEPS;
  case ncclFuncScatter:
    return SCATTER_CHUNKSTEPS;
  case ncclFuncReduce:
    return REDUCE_CHUNKSTEPS;
  default:
    return 1;
  }
}

inline int ncclDefaultSliceStep(ncclFunc_t func) {
  switch (func) {
  case ncclFuncAllReduce:
    return ALLREDUCE_SLICESTEPS;
  case ncclFuncAllGather:
    return ALLGATHER_SLICESTEPS;
  case ncclFuncAlltoAll:
    return ALLTOALL_SLICESTEPS;
  case ncclFuncReduceScatter:
    return REDUCESCATTER_SLICESTEPS;
  case ncclFuncBroadcast:
    return BROADCAST_SLICESTEPS;
  case ncclFuncGather:
    return GATHER_SLICESTEPS;
  case ncclFuncScatter:
    return SCATTER_SLICESTEPS;
  case ncclFuncReduce:
    return REDUCE_SLICESTEPS;
  default:
    return 1;
  }
}

inline int ncclGetChunkSteps(int protocol, int algorithm, int chunkSteps) {
  return protocol == NCCL_PROTO_SIMPLE && algorithm == NCCL_ALGO_RING ? chunkSteps : 1;
}

inline int ncclGetSliceSteps(int protocol, int algorithm, int sliceSteps) {
  return protocol == NCCL_PROTO_SIMPLE && algorithm == NCCL_ALGO_RING ? sliceSteps : 1;
}

inline size_t ncclSizePerChannel(size_t nBytes, int nChannels) {
  return nChannels > 0 ? DIVUP(nBytes, nChannels) : 0;
}

inline size_t ncclGetChunkSize(int protocol, size_t stepSize, int chunkSteps) {
  size_t chunkSize = stepSize * chunkSteps;
  if (protocol == NCCL_PROTO_LL) chunkSize /= 2;
  if (protocol == NCCL_PROTO_LL128) chunkSize = (chunkSize / NCCL_LL128_LINEELEMS) * NCCL_LL128_DATAELEMS;
  return chunkSize;
}

inline size_t ncclNominalSliceSize(size_t chunkSize, int chunkSteps, int sliceSteps) {
  return chunkSteps > 0 ? chunkSize / chunkSteps * sliceSteps : 0;
}

// Return the slice size used by Simple primitives for a chunk containing
// nelem elements. All size arguments and the result are expressed in elements.
template <typename Int>
__host__ __device__ inline Int ncclSimpleSliceSize(Int nelem, int slicesPerChunk, Int nominalSliceSize) {
  Int aligned = DIVUP(nelem, 16 * slicesPerChunk) * 16;
  Int minimum = nominalSliceSize / 32;
  return aligned > minimum ? aligned : minimum;
}

// Divides elements by divider while maintaining alignment
template <typename Int>
__host__ __device__ inline Int ncclElementAlignedDivUp(Int elements, int divider, Int elementSize, Int alignment) {
  Int alignmentElements = alignment / elementSize;
  Int rankElements = DIVUP(elements, static_cast<Int>(divider));
  Int alignedRankElements = DIVUP(rankElements, alignmentElements) * alignmentElements;

  return alignedRankElements;
}

#endif // NCCL_COLL_SIZES_H_
