/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <cstring>
#include <algorithm>
#include "reshard_types.h"
#include "reshard_internal.h"

void computeStrides(const size_t dims[], int ndims, size_t strides[]) {
  strides[ndims - 1] = 1;
  for (int d = ndims - 2; d >= 0; d--) strides[d] = strides[d + 1] * dims[d + 1];
}

void computeMeshGroupInfo(const ncclXferReshardMesh_t* mesh, int worldRank, ncclXferMeshGroupInfo* info) {
  memset(info, 0, sizeof(*info));
  info->shardMeshDim = -1;
  info->repMeshDim = -1;
  info->shardTensorDim = -1;
  for (int d = 0; d < 2; d++) {
    if (mesh->placement[d] == NCCLXFER_RESHARD_REPLICATE) {
      info->repMeshDim = d;
    } else if (IS_SHARD_PLACEMENT(mesh->placement[d])) {
      info->shardMeshDim = d;
      info->shardTensorDim = GET_SHARD_TENSOR_DIM(mesh->placement[d]);
    }
  }
  info->shardCount = (info->shardMeshDim >= 0) ? mesh->dims[info->shardMeshDim] : 1;
  info->repCount = (info->repMeshDim >= 0) ? mesh->dims[info->repMeshDim] : 1;
  int localRank = worldRank - mesh->startRank;
  info->meshPos[0] = localRank / mesh->dims[1];
  info->meshPos[1] = localRank % mesh->dims[1];
  info->shardIdx = (info->shardMeshDim >= 0) ? info->meshPos[info->shardMeshDim] : 0;
  info->repIdx = (info->repMeshDim >= 0) ? info->meshPos[info->repMeshDim] : 0;
  if (info->shardMeshDim == 1) {
    info->shardGroupStart = mesh->startRank + info->meshPos[0] * mesh->dims[1];
    info->shardGroupStride = 1;
    info->repGroupStart = mesh->startRank + info->meshPos[1];
    info->repGroupStride = mesh->dims[1];
  } else if (info->shardMeshDim == 0) {
    info->shardGroupStart = mesh->startRank + info->meshPos[1];
    info->shardGroupStride = mesh->dims[1];
    info->repGroupStart = mesh->startRank + info->meshPos[0] * mesh->dims[1];
    info->repGroupStride = 1;
  } else {
    info->shardGroupStart = mesh->startRank;
    info->shardGroupStride = 1;
    info->repGroupStart = worldRank;
    info->repGroupStride = 0;
  }
}

int getMeshRank(const ncclXferReshardMesh_t* mesh, const ncclXferMeshGroupInfo* info, int shardIdx, int repIdx) {
  int meshPos[2] = {0, 0};
  if (info->shardMeshDim >= 0) meshPos[info->shardMeshDim] = shardIdx;
  if (info->repMeshDim >= 0) meshPos[info->repMeshDim] = repIdx;
  if (info->shardMeshDim < 0) {
    meshPos[0] = repIdx;
    meshPos[1] = 0;
  }
  if (info->repMeshDim < 0) {
    if (info->shardMeshDim == 0) meshPos[1] = 0;
    else meshPos[0] = 0;
  }
  return mesh->startRank + meshPos[0] * mesh->dims[1] + meshPos[1];
}

void computeGlobalRange(const size_t localDims[], int ndims, int shardTensorDim, int shardIdx, size_t globalStart[],
                        size_t globalEnd[]) {
  for (int d = 0; d < ndims; d++) {
    if (d == shardTensorDim) {
      globalStart[d] = shardIdx * localDims[d];
      globalEnd[d] = globalStart[d] + localDims[d];
    } else {
      globalStart[d] = 0;
      globalEnd[d] = localDims[d];
    }
  }
}

bool computeOverlap(const size_t srcStart[], const size_t srcEnd[], const size_t dstStart[], const size_t dstEnd[],
                    int ndims, size_t overlapStart[], size_t overlapEnd[]) {
  for (int d = 0; d < ndims; d++) {
    overlapStart[d] = std::max(srcStart[d], dstStart[d]);
    overlapEnd[d] = std::min(srcEnd[d], dstEnd[d]);
    if (overlapStart[d] >= overlapEnd[d]) return false;
  }
  return true;
}

void computeTransferPlan(const size_t srcDims[], const size_t srcStrides[], int srcShardDim, int srcShardIdx,
                         const size_t dstDims[], const size_t dstStrides[], int dstShardDim, int dstShardIdx, int ndims,
                         size_t elementsPerChunk, ncclXferTransferPlan* plan) {
  (void)elementsPerChunk;
  memset(plan, 0, sizeof(*plan));
  if (ndims < 1 || ndims > MAX_TENSOR_DIMS) {
    plan->totalInnerTransfers = 0;
    return;
  }
  size_t srcGlobalStart[MAX_TENSOR_DIMS], srcGlobalEnd[MAX_TENSOR_DIMS];
  size_t dstGlobalStart[MAX_TENSOR_DIMS], dstGlobalEnd[MAX_TENSOR_DIMS];
  computeGlobalRange(srcDims, ndims, srcShardDim, srcShardIdx, srcGlobalStart, srcGlobalEnd);
  computeGlobalRange(dstDims, ndims, dstShardDim, dstShardIdx, dstGlobalStart, dstGlobalEnd);
  if (!computeOverlap(srcGlobalStart, srcGlobalEnd, dstGlobalStart, dstGlobalEnd, ndims, plan->overlapStart,
                      plan->overlapEnd)) {
    plan->totalInnerTransfers = 0;
    return;
  }
  size_t overlapSize[MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) overlapSize[d] = plan->overlapEnd[d] - plan->overlapStart[d];
  int innerContigStart = ndims - 1;
  size_t innerSize = 1;
  for (int d = ndims - 1; d >= 0; d--) {
    if (d != srcShardDim && d != dstShardDim && overlapSize[d] == srcDims[d] && overlapSize[d] == dstDims[d]) {
      innerSize *= overlapSize[d];
      innerContigStart = d;
    } else {
      innerSize *= overlapSize[d];
      innerContigStart = d;
      break;
    }
  }
  plan->innerSize = innerSize;
  plan->numOuterLoops = innerContigStart;
  plan->totalInnerTransfers = 1;
  for (int d = 0; d < innerContigStart; d++) {
    plan->outerCounts[d] = overlapSize[d];
    plan->outerSrcStrides[d] = srcStrides[d];
    plan->outerDstStrides[d] = dstStrides[d];
    plan->totalInnerTransfers *= overlapSize[d];
  }
  plan->srcBaseOffset = 0;
  plan->dstBaseOffset = 0;
  for (int d = 0; d < ndims; d++) {
    size_t srcLocalStart = plan->overlapStart[d] - srcGlobalStart[d];
    size_t dstLocalStart = plan->overlapStart[d] - dstGlobalStart[d];
    plan->srcBaseOffset += srcLocalStart * srcStrides[d];
    plan->dstBaseOffset += dstLocalStart * dstStrides[d];
  }
}
