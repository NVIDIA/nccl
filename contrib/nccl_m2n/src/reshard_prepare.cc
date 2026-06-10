/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information.
 ************************************************************************/

/*************************************************************************
 * Tensor Reshard — Kernel Parameter Builders
 *
 * Contains:
 *   - debugPrintMeshGroupInfo / debugPrintLoadBalancer /
 *     debugPrintTransferPlan          (static, TRACE-level helpers)
 *   - prepareReshardParams            (ring / hierarchical algorithm)
 *   - prepareDirectReshardParams      (direct algorithm)
 ************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include "reshard_types.h"
#include "m2n_checks.h"
#include "m2n_log.h"
#include "reshard_internal.h"

// ============================================================================
// Local helpers
// ============================================================================

static int fmtSizes(char* buf, size_t bufsz, const size_t* arr, int n) {
  int pos = 0;
  for (int i = 0; i < n && pos < (int)bufsz - 1; i++)
    pos += snprintf(buf + pos, bufsz - pos, "%s%zu", (i != 0) ? ", " : "", arr[i]);
  return pos;
}

// ============================================================================
// Debug Print Helpers (TRACE level)
// ============================================================================

static void debugPrintMeshGroupInfo(int worldRank, const char* meshName, const ncclMesh_t* mesh,
                                    const ncclReshardMeshGroupInfo* info) {
  if (reshardGetLogLevel() < RESHARD_LOG_TRACE) return;

  const char* p0 = mesh->placement[0] == NCCL_RESHARD_REPLICATE ?
                     "REPLICATE" :
                     (mesh->placement[0] == 0 ? "SHARD(0)" : (mesh->placement[0] == 1 ? "SHARD(1)" : "SHARD(2)"));
  const char* p1 = mesh->placement[1] == NCCL_RESHARD_REPLICATE ?
                     "REPLICATE" :
                     (mesh->placement[1] == 0 ? "SHARD(0)" : (mesh->placement[1] == 1 ? "SHARD(1)" : "SHARD(2)"));

  RESHARD_TRACE(worldRank, "=== %s Mesh Analysis ===", meshName);
  RESHARD_TRACE(worldRank, "  Mesh: dims=[%d, %d], startRank=%d", mesh->dims[0], mesh->dims[1], mesh->startRank);
  RESHARD_TRACE(worldRank, "  Placement: [%s, %s]", p0, p1);
  RESHARD_TRACE(worldRank, "  Derived: shardMeshDim=%d, repMeshDim=%d, shardTensorDim=%d", info->shardMeshDim,
                info->repMeshDim, info->shardTensorDim);
  RESHARD_TRACE(worldRank, "  Counts: shardCount=%d, repCount=%d", info->shardCount, info->repCount);
  RESHARD_TRACE(worldRank, "  My position: meshPos=[%d, %d], shardIdx=%d, repIdx=%d", info->meshPos[0],
                info->meshPos[1], info->shardIdx, info->repIdx);

  char shardBuf[512];
  int pos = 0;
  for (int i = 0; i < info->shardCount && pos < (int)sizeof(shardBuf) - 16; i++) {
    pos += snprintf(shardBuf + pos, sizeof(shardBuf) - pos, "%s%d", (i != 0) ? ", " : "",
                    info->shardGroupStart + i * info->shardGroupStride);
  }
  RESHARD_TRACE(worldRank, "  Shard group: start=%d, stride=%d (members: %s)", info->shardGroupStart,
                info->shardGroupStride, shardBuf);

  char repBuf[512];
  pos = 0;
  for (int i = 0; i < info->repCount && pos < (int)sizeof(repBuf) - 16; i++) {
    pos += snprintf(repBuf + pos, sizeof(repBuf) - pos, "%s%d", (i != 0) ? ", " : "",
                    info->repGroupStart + i * info->repGroupStride);
  }
  RESHARD_TRACE(worldRank, "  Rep group: start=%d, stride=%d (members: %s)", info->repGroupStart, info->repGroupStride,
                repBuf);
}

static void debugPrintLoadBalancer(int worldRank, const ncclReshardRepLoadBalancer* lb) {
  if (reshardGetLogLevel() < RESHARD_LOG_TRACE) return;

  RESHARD_TRACE(worldRank, "=== Load Balancer ===");
  RESHARD_TRACE(worldRank, "  mode=%s", lb->mode == RESHARD_LB_NODE_AWARE ? "NODE_AWARE" : "UNIFORM");
  RESHARD_TRACE(worldRank, "  srcRepCount=%d, dstRepCount=%d", lb->srcRepCount, lb->dstRepCount);
  RESHARD_TRACE(worldRank, "  dstGpusPerDomain=%d", lb->dstGpusPerDomain);
  RESHARD_TRACE(worldRank, "  dstRepStartRank=%d, dstRepStride=%d", lb->dstRepStartRank, lb->dstRepStride);

  if (lb->mode == RESHARD_LB_NODE_AWARE) {
    int numDstNodes = getNumDestNodes(lb);
    RESHARD_TRACE(worldRank, "  numDstNodes=%d", numDstNodes);
    RESHARD_TRACE(worldRank, "  Node assignment:");
    for (int n = 0; n < numDstNodes && n < 8; n++) {
      int firstDstNode = getNodeOfDestRep(lb, 0);
      int repStart, repEnd;
      getDestRepsOnNode(lb, firstDstNode + n, &repStart, &repEnd);
      RESHARD_TRACE(worldRank, "    node %d: dst_reps [%d, %d)", firstDstNode + n, repStart, repEnd);
    }
    if (numDstNodes > 8) RESHARD_TRACE(worldRank, "    ... (%d more nodes)", numDstNodes - 8);
  }

  RESHARD_TRACE(worldRank, "  Source Rep -> Dest Rep Range mapping:");
  int activeSenders = 0;
  for (int srcRep = 0; srcRep < lb->srcRepCount; srcRep++) {
    int repStart, repEnd;
    getTargetRepRange(lb, srcRep, &repStart, &repEnd);
    if (repStart < repEnd) {
      RESHARD_TRACE(worldRank, "    srcRep %d -> dst_reps [%d, %d) (SENDER)", srcRep, repStart, repEnd);
      activeSenders++;
    } else {
      RESHARD_TRACE(worldRank, "    srcRep %d -> (IDLE - no targets)", srcRep);
    }
  }
  RESHARD_TRACE(worldRank, "  Active senders: %d / %d source reps", activeSenders, lb->srcRepCount);

  RESHARD_TRACE(worldRank, "  Dest Rep -> Source Rep mapping:");
  for (int dstRep = 0; dstRep < lb->dstRepCount; dstRep++) {
    int srcRep = getSourceRepForDest(lb, dstRep);
    RESHARD_TRACE(worldRank, "    dstRep %d <- srcRep %d (leader)", dstRep, srcRep);
  }
}

static void debugPrintTransferPlan(int worldRank, int srcShardIdx, int dstShardIdx, int ndims,
                                   const ncclReshardTransferPlan* plan, const size_t srcDims[], const size_t dstDims[]) {
  if (reshardGetLogLevel() < RESHARD_LOG_TRACE) return;

  RESHARD_TRACE(worldRank, "    Transfer Plan: srcShard=%d -> dstShard=%d", srcShardIdx, dstShardIdx);
  RESHARD_TRACE(worldRank, "      numOuterLoops=%d, totalInnerTransfers=%zu, innerSize=%zu", plan->numOuterLoops,
                plan->totalInnerTransfers, plan->innerSize);
  RESHARD_TRACE(worldRank, "      isContiguous=%s", plan->totalInnerTransfers == 1 ? "YES" : "NO");
  RESHARD_TRACE(worldRank, "      srcBaseOffset=%zu, dstBaseOffset=%zu", plan->srcBaseOffset, plan->dstBaseOffset);

  {
    char osBuf[128], oeBuf[128];
    fmtSizes(osBuf, sizeof(osBuf), plan->overlapStart, ndims);
    fmtSizes(oeBuf, sizeof(oeBuf), plan->overlapEnd, ndims);
    RESHARD_TRACE(worldRank, "      overlapStart=[%s], overlapEnd=[%s]", osBuf, oeBuf);
  }

  if (plan->numOuterLoops > 0) {
    char oc[128], oss[128], ods[128];
    fmtSizes(oc, sizeof(oc), plan->outerCounts, plan->numOuterLoops);
    fmtSizes(oss, sizeof(oss), plan->outerSrcStrides, plan->numOuterLoops);
    fmtSizes(ods, sizeof(ods), plan->outerDstStrides, plan->numOuterLoops);
    RESHARD_TRACE(worldRank,
                  "      outerCounts=[%s], outerSrcStrides=[%s], "
                  "outerDstStrides=[%s]",
                  oc, oss, ods);
  }

  size_t maxSrcOffset = plan->srcBaseOffset;
  size_t maxDstOffset = plan->dstBaseOffset;

  for (int d = 0; d < plan->numOuterLoops; d++) {
    if (plan->outerCounts[d] == 0) {
      printf("[ERROR][Rank %d]       outerCounts[%d] is ZERO - will "
             "cause div by zero!\n",
             worldRank, d);
    }
    if (plan->outerCounts[d] > 0) {
      maxSrcOffset += (plan->outerCounts[d] - 1) * plan->outerSrcStrides[d];
      maxDstOffset += (plan->outerCounts[d] - 1) * plan->outerDstStrides[d];
    }
  }

  size_t srcBufferSize = 1;
  size_t dstBufferSize = 1;
  for (int d = 0; d < ndims; d++) {
    srcBufferSize *= srcDims[d];
    dstBufferSize *= dstDims[d];
  }

  size_t srcMaxAccess = maxSrcOffset + plan->innerSize;
  size_t dstMaxAccess = maxDstOffset + plan->innerSize;

  RESHARD_TRACE(worldRank,
                "      BOUNDS CHECK: maxSrcOffset=%zu, +innerSize=%zu, "
                "total_access=%zu, srcBuffer=%zu",
                maxSrcOffset, plan->innerSize, srcMaxAccess, srcBufferSize);
  RESHARD_TRACE(worldRank,
                "      BOUNDS CHECK: maxDstOffset=%zu, +innerSize=%zu, "
                "total_access=%zu, dstBuffer=%zu",
                maxDstOffset, plan->innerSize, dstMaxAccess, dstBufferSize);

  if (srcMaxAccess > srcBufferSize) {
    printf("[ERROR][Rank %d]       *** SOURCE BUFFER OVERFLOW! %zu > %zu "
           "***\n",
           worldRank, srcMaxAccess, srcBufferSize);
  }
  if (dstMaxAccess > dstBufferSize)
    printf("[ERROR][Rank %d]       *** DEST BUFFER OVERFLOW! %zu > %zu ***\n", worldRank, dstMaxAccess, dstBufferSize);

  if (plan->srcBaseOffset > srcBufferSize) {
    printf("[ERROR][Rank %d]       *** srcBaseOffset looks corrupted: "
           "%zu > buffer_size %zu ***\n",
           worldRank, plan->srcBaseOffset, srcBufferSize);
  }
  if (plan->dstBaseOffset > dstBufferSize) {
    printf("[ERROR][Rank %d]       *** dstBaseOffset looks corrupted: "
           "%zu > buffer_size %zu ***\n",
           worldRank, plan->dstBaseOffset, dstBufferSize);
  }
  fflush(stdout);
}

// ============================================================================
// Prepare Kernel Parameters (Ring / Hierarchical)
// ============================================================================

ncclReshardParams prepareReshardParams(
  int worldRank, const void* srcBuffer, const size_t srcTensorDims[], int ndims, const ncclMesh_t* srcMesh,
  const void* dstBuffer, const size_t dstTensorDims[], const ncclMesh_t* dstMesh, ncclWindow_t window,
  size_t elementsPerChunk, int numCtas, int srcGpusPerDomain, int dstGpusPerDomain, const size_t* allWindowOffsets) {
  ncclReshardParams params;
  memset(&params, 0, sizeof(params));

  {
    char sd[128], dd[128];
    fmtSizes(sd, sizeof(sd), srcTensorDims, ndims);
    fmtSizes(dd, sizeof(dd), dstTensorDims, ndims);
    RESHARD_DEBUG(worldRank, "========================================");
    RESHARD_DEBUG(worldRank, "prepareReshardParams() START");
    RESHARD_DEBUG(worldRank, "========================================");
    RESHARD_DEBUG(worldRank, "Input parameters:");
    RESHARD_DEBUG(worldRank, "  srcBuffer=%p, ndims=%d", srcBuffer, ndims);
    RESHARD_DEBUG(worldRank, "  srcTensorDims=[%s]", sd);
    RESHARD_DEBUG(worldRank, "  dstBuffer=%p", dstBuffer);
    RESHARD_DEBUG(worldRank, "  dstTensorDims=[%s]", dd);
    RESHARD_DEBUG(worldRank, "  elementsPerChunk=%zu, numCtas=%d", elementsPerChunk, numCtas);
    RESHARD_DEBUG(worldRank, "  srcGpusPerDomain=%d, dstGpusPerDomain=%d", srcGpusPerDomain, dstGpusPerDomain);
    RESHARD_DEBUG(worldRank, "Struct sizes:");
    RESHARD_DEBUG(worldRank, "  sizeof(ncclReshardParams) = %zu bytes", sizeof(ncclReshardParams));
    RESHARD_DEBUG(worldRank, "  sizeof(ncclReshardSourceInfo) = %zu bytes", sizeof(ncclReshardSourceInfo));
    RESHARD_DEBUG(worldRank, "  sizeof(ncclReshardTargetInfo) = %zu bytes", sizeof(ncclReshardTargetInfo));
    RESHARD_DEBUG(worldRank, "  sizeof(ncclReshardTransferPlan) = %zu bytes", sizeof(ncclReshardTransferPlan));
    RESHARD_DEBUG(worldRank, "  MAX_SOURCES=%d, MAX_TARGETS=%d", MAX_SOURCES, MAX_TARGETS);
    RESHARD_DEBUG(worldRank, "  sources array size = %zu bytes", (size_t)MAX_SOURCES * sizeof(ncclReshardSourceInfo));
    RESHARD_DEBUG(worldRank, "  targets array size = %zu bytes", (size_t)MAX_TARGETS * sizeof(ncclReshardTargetInfo));
  }

  params.window = window;
  params.elementsPerChunk = elementsPerChunk;
  /* Chunk size is parsed once at init from NCCL_RESHARD_CHUNK_SIZE
   * into gReshardChunkSizeBytes; 0 means "no override". Avoids
   * touching getenv on the per-call hot path. */
  params.chunkSizeBytes = gReshardChunkSizeBytes > 0 ? gReshardChunkSizeBytes : CHUNK_SIZE_BYTES;
  params.totalCtas = numCtas;
  params.myWorldRank = worldRank;
  params.ndims = ndims;

  int srcMeshSize = srcMesh->dims[0] * srcMesh->dims[1];
  int dstMeshSize = dstMesh->dims[0] * dstMesh->dims[1];

  params.isSource = (worldRank >= srcMesh->startRank && worldRank < srcMesh->startRank + srcMeshSize);
  params.isDest = (worldRank >= dstMesh->startRank && worldRank < dstMesh->startRank + dstMeshSize);

  RESHARD_DEBUG(worldRank, "Role determination:");
  RESHARD_DEBUG(worldRank, "  srcMesh: ranks [%d, %d), size=%d", srcMesh->startRank, srcMesh->startRank + srcMeshSize,
                srcMeshSize);
  RESHARD_DEBUG(worldRank, "  dstMesh: ranks [%d, %d), size=%d", dstMesh->startRank, dstMesh->startRank + dstMeshSize,
                dstMeshSize);
  RESHARD_DEBUG(worldRank, "  isSource=%d, isDest=%d", params.isSource, params.isDest);

  ncclReshardMeshGroupInfo srcInfo, dstInfo;
  ncclReshardMeshGroupInfo fullSrcInfo, fullDstInfo;

  computeMeshGroupInfo(srcMesh, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstMesh, dstMesh->startRank, &fullDstInfo);

  params.srcShardTensorDim = fullSrcInfo.shardTensorDim;
  params.dstShardTensorDim = fullDstInfo.shardTensorDim;
  params.srcShardCount = fullSrcInfo.shardCount;
  params.dstShardCount = fullDstInfo.shardCount;
  params.sameShardDim = (params.srcShardTensorDim == params.dstShardTensorDim);

  RESHARD_DEBUG(worldRank, "Sharding configuration:");
  RESHARD_DEBUG(worldRank, "  srcShardTensorDim=%d, dstShardTensorDim=%d", params.srcShardTensorDim,
                params.dstShardTensorDim);
  RESHARD_DEBUG(worldRank, "  srcShardCount=%d, dstShardCount=%d", params.srcShardCount, params.dstShardCount);
  RESHARD_DEBUG(worldRank, "  sameShardDim=%d (%s)", params.sameShardDim,
                params.sameShardDim ? "SAME-DIM sharding" : "CROSS-DIM sharding");

  if (params.isSource) {
    computeMeshGroupInfo(srcMesh, worldRank, &srcInfo);
    debugPrintMeshGroupInfo(worldRank, "Source", srcMesh, &srcInfo);

    params.mySrcShardIdx = srcInfo.shardIdx;
    params.mySrcRepIdx = srcInfo.repIdx;
    for (int d = 0; d < ndims; d++) params.srcDims[d] = srcTensorDims[d];
    computeStrides(params.srcDims, ndims, params.srcStrides);

    {
      char db[128], sb[128];
      fmtSizes(db, sizeof(db), params.srcDims, ndims);
      fmtSizes(sb, sizeof(sb), params.srcStrides, ndims);
      RESHARD_DEBUG(worldRank, "Source local info:");
      RESHARD_DEBUG(worldRank, "  mySrcShardIdx=%d, mySrcRepIdx=%d", params.mySrcShardIdx, params.mySrcRepIdx);
      RESHARD_DEBUG(worldRank, "  srcDims=[%s], srcStrides=[%s]", db, sb);
    }
  } else {
    params.mySrcShardIdx = -1;
    params.mySrcRepIdx = -1;
  }

  if (params.isDest) {
    computeMeshGroupInfo(dstMesh, worldRank, &dstInfo);
    debugPrintMeshGroupInfo(worldRank, "Dest", dstMesh, &dstInfo);

    params.myDstShardIdx = dstInfo.shardIdx;
    params.myDstRepIdx = dstInfo.repIdx;
    for (int d = 0; d < ndims; d++) params.dstDims[d] = dstTensorDims[d];
    computeStrides(params.dstDims, ndims, params.dstStrides);

    {
      char db[128], sb[128];
      fmtSizes(db, sizeof(db), params.dstDims, ndims);
      fmtSizes(sb, sizeof(sb), params.dstStrides, ndims);
      RESHARD_DEBUG(worldRank, "Dest local info:");
      RESHARD_DEBUG(worldRank, "  myDstShardIdx=%d, myDstRepIdx=%d", params.myDstShardIdx, params.myDstRepIdx);
      RESHARD_DEBUG(worldRank, "  dstDims=[%s], dstStrides=[%s]", db, sb);
    }
  } else {
    params.myDstShardIdx = -1;
    params.myDstRepIdx = -1;
  }

  // Infer missing dimensions
  if (params.isSource && !params.isDest) {
    for (int d = 0; d < ndims; d++) {
      size_t globalSize;
      if (d == params.srcShardTensorDim) globalSize = srcTensorDims[d] * params.srcShardCount;
      else globalSize = srcTensorDims[d];

      if (d == params.dstShardTensorDim) params.dstDims[d] = globalSize / params.dstShardCount;
      else params.dstDims[d] = globalSize;
    }
    computeStrides(params.dstDims, ndims, params.dstStrides);

    {
      char db[128], sb[128];
      fmtSizes(db, sizeof(db), params.dstDims, ndims);
      fmtSizes(sb, sizeof(sb), params.dstStrides, ndims);
      RESHARD_DEBUG(worldRank, "Inferred dstDims from src:");
      RESHARD_DEBUG(worldRank, "  dstDims=[%s], dstStrides=[%s]", db, sb);
    }
  }

  if (params.isDest && !params.isSource) {
    for (int d = 0; d < ndims; d++) {
      size_t globalSize;
      if (d == params.dstShardTensorDim) globalSize = dstTensorDims[d] * params.dstShardCount;
      else globalSize = dstTensorDims[d];

      if (d == params.srcShardTensorDim) params.srcDims[d] = globalSize / params.srcShardCount;
      else params.srcDims[d] = globalSize;
    }
    computeStrides(params.srcDims, ndims, params.srcStrides);

    {
      char db[128], sb[128];
      fmtSizes(db, sizeof(db), params.srcDims, ndims);
      fmtSizes(sb, sizeof(sb), params.srcStrides, ndims);
      RESHARD_DEBUG(worldRank, "Inferred srcDims from dst:");
      RESHARD_DEBUG(worldRank, "  srcDims=[%s], srcStrides=[%s]", db, sb);
    }
  }

  ncclReshardRepLoadBalancer lb = {.srcRepCount = fullSrcInfo.repCount,
                                .dstRepCount = fullDstInfo.repCount,
                                .dstGpusPerDomain = dstGpusPerDomain,
                                .dstRepStartRank = dstMesh->startRank,
                                .dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1,
                                .mode = reshardGetLoadBalanceMode()};

  debugPrintLoadBalancer(worldRank, &lb);

  // SOURCE: Compute targets (with hierarchical target selection)
  if (params.isSource) {
    params.numTargets = 0;

    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);

    RESHARD_TRACE(worldRank, "=== SOURCE: Computing Targets (Hierarchical) ===");
    RESHARD_TRACE(worldRank, "  mySrcShardIdx=%d, mySrcRepIdx=%d", params.mySrcShardIdx, params.mySrcRepIdx);
    RESHARD_TRACE(worldRank, "  target_rep_range=[%d, %d)", targetRepStart, targetRepEnd);
    RESHARD_TRACE(worldRank, "  Checking %d dest shards for overlap...", params.dstShardCount);

    for (int dstShard = 0; dstShard < params.dstShardCount; dstShard++) {
      ncclReshardTransferPlan plan;
      computeTransferPlan(params.srcDims, params.srcStrides, params.srcShardTensorDim, params.mySrcShardIdx,
                          params.dstDims, params.dstStrides, params.dstShardTensorDim, dstShard, ndims,
                          elementsPerChunk, &plan);

      if (plan.totalInnerTransfers == 0) {
        RESHARD_TRACE(worldRank, "  dstShard %d: NO OVERLAP", dstShard);
        continue;
      }

      if (params.numTargets >= MAX_TARGETS) {
        RESHARD_FATAL(worldRank,
                      "prepareReshardParams: target list TRUNCATED at "
                      "dstShard %d! "
                      "numTargets=%d >= MAX_TARGETS=%d. Remaining dst "
                      "shards dropped. "
                      "Some dest ranks will NEVER receive data — "
                      "kernel WILL HANG. "
                      "Fix: increase MAX_TARGETS in reshard_limits.h.",
                      dstShard, params.numTargets, MAX_TARGETS);
      }

      if (targetRepStart < targetRepEnd) {
        int numSourcesToDstShard = 0;
        int myPosition = 0;

        for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
          ncclReshardTransferPlan checkPlan;
          computeTransferPlan(params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard, params.dstDims,
                              params.dstStrides, params.dstShardTensorDim, dstShard, ndims, elementsPerChunk,
                              &checkPlan);

          if (checkPlan.totalInnerTransfers > 0) {
            if (srcShard < params.mySrcShardIdx) myPosition++;
            numSourcesToDstShard++;
          }
        }

        int firstRepRank = getMeshRank(dstMesh, &fullDstInfo, dstShard, targetRepStart);
        int firstRepNode = firstRepRank / dstGpusPerDomain;

        int localRepsOnTargetNode[MAX_LOCAL_FOLLOWERS + 1];
        int numLocalRepsOnTargetNode = 0;

        for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
          int repRank = getMeshRank(dstMesh, &fullDstInfo, dstShard, rep);
          int repNode = repRank / dstGpusPerDomain;

          if (repNode == firstRepNode) {
            if (numLocalRepsOnTargetNode < MAX_LOCAL_FOLLOWERS + 1) {
              localRepsOnTargetNode[numLocalRepsOnTargetNode++] = rep;
            } else {
              RESHARD_FATAL(worldRank,
                            "prepareReshardParams: "
                            "localRepsOnTargetNode overflow! "
                            "numLocalRepsOnTargetNode=%d >= "
                            "%d (MAX_LOCAL_FOLLOWERS+1). "
                            "Fix: increase MAX_LOCAL_FOLLOWERS "
                            "in reshard_limits.h.",
                            numLocalRepsOnTargetNode, MAX_LOCAL_FOLLOWERS + 1);
            }
          }
        }

        int targetLocalRepIdx = 0;

        if (numLocalRepsOnTargetNode > 0 && numSourcesToDstShard > 0) {
          int sourcesPerRep = numSourcesToDstShard / numLocalRepsOnTargetNode;
          int extraSources = numSourcesToDstShard % numLocalRepsOnTargetNode;

          int threshold = extraSources * (sourcesPerRep + 1);
          if (myPosition < threshold) targetLocalRepIdx = myPosition / (sourcesPerRep + 1);
          else targetLocalRepIdx = extraSources + (myPosition - threshold) / sourcesPerRep;

          if (targetLocalRepIdx >= numLocalRepsOnTargetNode) targetLocalRepIdx = numLocalRepsOnTargetNode - 1;
        }

        int leaderRep = localRepsOnTargetNode[targetLocalRepIdx];
        int leaderRank = getMeshRank(dstMesh, &fullDstInfo, dstShard, leaderRep);

        RESHARD_TRACE(worldRank, "  dstShard %d: Hierarchical target selection:", dstShard);
        RESHARD_TRACE(worldRank, "    numSourcesToDstShard=%d, myPosition=%d", numSourcesToDstShard, myPosition);
        RESHARD_TRACE(worldRank, "    firstRepNode=%d, numLocalRepsOnTargetNode=%d", firstRepNode,
                      numLocalRepsOnTargetNode);
        RESHARD_TRACE(worldRank,
                      "    targetLocalRepIdx=%d -> leaderRep=%d, "
                      "leaderRank=%d",
                      targetLocalRepIdx, leaderRep, leaderRank);

        ncclReshardTargetInfo* target = &params.targets[params.numTargets++];
        target->dstShardIdx = dstShard;
        target->dstWorldRank = leaderRank;
        target->windowOffset = (allWindowOffsets != nullptr) ? allWindowOffsets[leaderRank] : 0;
        for (int d = 0; d < ndims; d++) {
          target->overlapStart[d] = plan.overlapStart[d];
          target->overlapEnd[d] = plan.overlapEnd[d];
        }
        target->plan = plan;

        RESHARD_TRACE(worldRank,
                      "    target[%d]: dstRank=%d, dstShard=%d, "
                      "windowOffset=%zu, srcBaseOffset=%zu, "
                      "dstBaseOffset=%zu",
                      params.numTargets - 1, leaderRank, dstShard, target->windowOffset, plan.srcBaseOffset,
                      plan.dstBaseOffset);

        target->isContiguous = (plan.totalInnerTransfers == 1);
        target->totalBytes = plan.totalInnerTransfers * plan.innerSize;

        RESHARD_TRACE(worldRank, "  dstShard %d: OVERLAP -> target[%d]", dstShard, params.numTargets - 1);
        RESHARD_TRACE(worldRank, "    leaderRep=%d, leaderRank=%d", leaderRep, leaderRank);
        RESHARD_TRACE(worldRank, "    isContiguous=%d, totalBytes=%zu", target->isContiguous, target->totalBytes);
        debugPrintTransferPlan(worldRank, params.mySrcShardIdx, dstShard, ndims, &plan, params.srcDims, params.dstDims);
      }
    }

    RESHARD_TRACE(worldRank, "  Total targets: %d", params.numTargets);
  }

  // DEST: Compute sources and replication
  if (params.isDest) {
    params.numSources = 0;

    int sourceRep = getSourceRepForDest(&lb, dstInfo.repIdx);

    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, sourceRep, &targetRepStart, &targetRepEnd);

    int myNode = worldRank / dstGpusPerDomain;

    RESHARD_TRACE(worldRank, "=== DEST: Computing Sources (Hierarchical) ===");
    RESHARD_TRACE(worldRank, "  myDstShardIdx=%d, myDstRepIdx=%d", params.myDstShardIdx, params.myDstRepIdx);
    RESHARD_TRACE(worldRank, "  sourceRep (who sends to me)=%d", sourceRep);
    RESHARD_TRACE(worldRank, "  target_rep_range=[%d, %d), myNode=%d", targetRepStart, targetRepEnd, myNode);

    // Step 1: Find local reps on the same domain and determine my position
    int localReps[MAX_LOCAL_FOLLOWERS + 1];
    int numLocalReps = 0;
    int myLocalRepIdx = 0;

    for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
      int repRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, rep);
      int repNode = repRank / dstGpusPerDomain;

      if (repNode == myNode) {
        if (numLocalReps >= MAX_LOCAL_FOLLOWERS + 1) {
          RESHARD_FATAL(worldRank,
                        "prepareReshardParams: localReps overflow! "
                        "numLocalReps=%d >= %d (MAX_LOCAL_FOLLOWERS+1). "
                        "Fix: increase MAX_LOCAL_FOLLOWERS in "
                        "reshard_limits.h.",
                        numLocalReps, MAX_LOCAL_FOLLOWERS + 1);
        }
        if (rep == dstInfo.repIdx) myLocalRepIdx = numLocalReps;
        localReps[numLocalReps++] = rep;
      }
    }

    params.localRepIdx = myLocalRepIdx;
    params.numLocalReps = numLocalReps;

    if (reshardGetLogLevel() >= RESHARD_LOG_TRACE) {
      char lrBuf[2048];
      int pos = 0;
      for (int i = 0; i < numLocalReps && pos < (int)sizeof(lrBuf) - 32; i++) {
        int repRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, localReps[i]);
        pos +=
          snprintf(lrBuf + pos, sizeof(lrBuf) - pos, "%srep%d(rank%d)", (i != 0) ? ", " : "", localReps[i], repRank);
      }
      RESHARD_TRACE(worldRank, "  Local rep discovery: numLocalReps=%d, myLocalRepIdx=%d", numLocalReps, myLocalRepIdx);
      RESHARD_TRACE(worldRank, "  Local reps on node %d: [%s]", myNode, lrBuf);
    }

    int firstNodeLocalReps = 0;
    int firstRepNode = -1;
    if (targetRepStart < targetRepEnd) {
      int firstRepRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, targetRepStart);
      firstRepNode = firstRepRank / dstGpusPerDomain;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int repRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, rep);
        int repNode = repRank / dstGpusPerDomain;
        if (repNode == firstRepNode) firstNodeLocalReps++;
      }
    }

    // Step 2: First pass - count all overlapping sources
    int allSourceShards[MAX_SOURCES];
    int numAllSources = 0;

    RESHARD_TRACE(worldRank, "  Checking %d src shards for overlap...", params.srcShardCount);

    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      ncclReshardTransferPlan plan;
      computeTransferPlan(params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard, params.dstDims,
                          params.dstStrides, params.dstShardTensorDim, params.myDstShardIdx, ndims, elementsPerChunk,
                          &plan);

      if (plan.totalInnerTransfers > 0 && numAllSources < MAX_SOURCES) {
        allSourceShards[numAllSources++] = srcShard;
        RESHARD_TRACE(worldRank, "    srcShard %d: OVERLAP (all_source[%d])", srcShard, numAllSources - 1);
      } else if (plan.totalInnerTransfers > 0 && numAllSources >= MAX_SOURCES) {
        RESHARD_FATAL(worldRank,
                      "prepareReshardParams: allSourceShards "
                      "TRUNCATED at srcShard %d! "
                      "numAllSources=%d >= MAX_SOURCES=%d. "
                      "Overlapping src shards dropped. "
                      "Dest rank will miss data from source shards — "
                      "kernel WILL HANG. "
                      "Fix: increase MAX_SOURCES in reshard_limits.h.",
                      srcShard, numAllSources, MAX_SOURCES);
      } else {
        RESHARD_TRACE(worldRank, "    srcShard %d: NO OVERLAP", srcShard);
      }
    }

    RESHARD_TRACE(worldRank, "  Total overlapping sources: %d", numAllSources);

    // Step 3: Distribute sources across local reps
    int mySourceStart = 0, mySourceEnd = 0;
    int sourceRepSlots = firstNodeLocalReps > 0 ? firstNodeLocalReps : numLocalReps;
    int activeSourceSlots = sourceRepSlots;
    /* Sources are initially sent to leaders on the first target node.
       Once a later node has fewer reps, the collapsed source slots stay
       collapsed for downstream nodes because each rank has only one ring
       successor. */
    for (int node = firstRepNode; node >= 0 && node <= myNode; node++) {
      int nodeLocalReps = 0;
      for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
        int repRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, rep);
        int repNode = repRank / dstGpusPerDomain;
        if (repNode == node) nodeLocalReps++;
      }
      if (nodeLocalReps > 0 && nodeLocalReps < activeSourceSlots) activeSourceSlots = nodeLocalReps;
    }

    int mySourceSlotStart = myLocalRepIdx;
    int mySourceSlotEnd = myLocalRepIdx + 1;
    if (activeSourceSlots > 0 && myLocalRepIdx == activeSourceSlots - 1 && sourceRepSlots > activeSourceSlots)
      mySourceSlotEnd = sourceRepSlots;

    if (sourceRepSlots > 0 && mySourceSlotStart < activeSourceSlots && numAllSources > 0) {
      int sourcesPerRep = numAllSources / sourceRepSlots;
      int extraSources = numAllSources % sourceRepSlots;
      int threshold = extraSources * (sourcesPerRep + 1);

      if (mySourceSlotStart < extraSources) mySourceStart = mySourceSlotStart * (sourcesPerRep + 1);
      else mySourceStart = threshold + (mySourceSlotStart - extraSources) * sourcesPerRep;
      if (mySourceSlotEnd >= sourceRepSlots) mySourceEnd = numAllSources;
      else if (mySourceSlotEnd < extraSources) mySourceEnd = mySourceSlotEnd * (sourcesPerRep + 1);
      else mySourceEnd = threshold + (mySourceSlotEnd - extraSources) * sourcesPerRep;
    }

    params.isLeaderForSources = (mySourceEnd > mySourceStart);

    RESHARD_TRACE(worldRank,
                  "  Source distribution: firstNodeLocalReps=%d, "
                  "sourceRepSlots=%d, "
                  "activeSourceSlots=%d, my_source_slots=[%d, %d), "
                  "my_source_range=[%d, %d), is_leader=%d",
                  firstNodeLocalReps, sourceRepSlots, activeSourceSlots, mySourceSlotStart, mySourceSlotEnd,
                  mySourceStart, mySourceEnd, params.isLeaderForSources);

    if (reshardGetLogLevel() >= RESHARD_LOG_TRACE && sourceRepSlots > 1) {
      RESHARD_TRACE(worldRank, "  Hierarchical distribution across %d source slots:", sourceRepSlots);
      for (int lr = 0; lr < sourceRepSlots; lr++) {
        int lrSourcesPerRep = numAllSources / sourceRepSlots;
        int lrExtraSources = numAllSources % sourceRepSlots;
        int lrStart, lrEnd;
        if (lr < lrExtraSources) {
          lrStart = lr * (lrSourcesPerRep + 1);
          lrEnd = lrStart + lrSourcesPerRep + 1;
        } else {
          lrStart = lrExtraSources * (lrSourcesPerRep + 1) + (lr - lrExtraSources) * lrSourcesPerRep;
          lrEnd = lrStart + lrSourcesPerRep;
        }
        int mappedLr = lr;
        if (activeSourceSlots > 0 && mappedLr >= activeSourceSlots) mappedLr = activeSourceSlots - 1;
        int repRank =
          numLocalReps > 0 ? getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, localReps[mappedLr]) : -1;
        RESHARD_TRACE(worldRank,
                      "    source_slot[%d] -> local_rep[%d] (rank %d): "
                      "sources [%d, %d)%s",
                      lr, mappedLr, repRank, lrStart, lrEnd, mappedLr == myLocalRepIdx ? " <- ME" : "");
      }
    }

    // Step 4: Second pass - only add MY sources to params.sources[]
    for (int idx = mySourceStart; idx < mySourceEnd; idx++) {
      int srcShard = allSourceShards[idx];

      ncclReshardTransferPlan plan;
      computeTransferPlan(params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard, params.dstDims,
                          params.dstStrides, params.dstShardTensorDim, params.myDstShardIdx, ndims, elementsPerChunk,
                          &plan);

      if (params.numSources >= MAX_SOURCES) {
        RESHARD_FATAL(worldRank,
                      "prepareReshardParams: source list TRUNCATED at "
                      "srcShard %d (all_idx=%d)! "
                      "numSources=%d >= MAX_SOURCES=%d. Remaining "
                      "sources dropped. "
                      "Dest rank will wait for data that never arrives "
                      "— kernel WILL HANG. "
                      "Fix: increase MAX_SOURCES in reshard_limits.h.",
                      srcShard, idx, params.numSources, MAX_SOURCES);
      }
      {
        int srcRank = getMeshRank(srcMesh, &fullSrcInfo, srcShard, sourceRep);

        ncclReshardSourceInfo* source = &params.sources[params.numSources++];
        source->signalBase = srcRank * numCtas;
        source->plan = plan;

        source->isContiguous = (plan.totalInnerTransfers == 1);
        source->totalBytes = plan.totalInnerTransfers * plan.innerSize;

        RESHARD_TRACE(worldRank, "  srcShard %d: OVERLAP -> source[%d] (all_idx=%d)", srcShard, params.numSources - 1,
                      idx);
        RESHARD_TRACE(worldRank, "    srcRank=%d, signalBase=%u", srcRank, source->signalBase);
        RESHARD_TRACE(worldRank, "    isContiguous=%d, totalBytes=%zu", source->isContiguous, source->totalBytes);
        debugPrintTransferPlan(worldRank, srcShard, params.myDstShardIdx, ndims, &plan, params.srcDims, params.dstDims);

        if (reshardGetLogLevel() >= RESHARD_LOG_TRACE) {
          if (!source->isContiguous) {
            size_t itersPerCta = (plan.totalInnerTransfers + numCtas - 1) / numCtas;
            RESHARD_TRACE(worldRank,
                          "    CTA work distribution: totalIters=%zu, "
                          "itersPerCta=%zu, chunkSizeBytes=%zu",
                          plan.totalInnerTransfers, itersPerCta, params.chunkSizeBytes);
            for (int cta = 0; cta < numCtas && cta < 4; cta++) {
              size_t myStart = cta * itersPerCta;
              size_t myEnd = std::min(myStart + itersPerCta, plan.totalInnerTransfers);
              size_t myBytes = (myEnd - myStart) * plan.innerSize;
              size_t myChunks = (myBytes > 0) ? ((myBytes + params.chunkSizeBytes - 1) / params.chunkSizeBytes) : 0;
              RESHARD_TRACE(worldRank,
                            "      CTA %d: iters [%zu, %zu), "
                            "bytes=%zu, signals=%zu",
                            cta, myStart, myEnd, myBytes, myChunks);
            }
            if (numCtas > 4) RESHARD_TRACE(worldRank, "      ... (%d more CTAs)", numCtas - 4);
          } else {
            RESHARD_TRACE(worldRank, "    CTA work distribution: CONTIGUOUS, "
                                     "1 signal per CTA");
            size_t bytesPerCta = (source->totalBytes + numCtas - 1) / numCtas;
            RESHARD_TRACE(worldRank, "      bytesPerCta=%zu", bytesPerCta);
          }
        }
      }
    }

    RESHARD_TRACE(worldRank, "  Total sources (my portion): %d", params.numSources);

    // Step 5: Compute replication (followers and ring next)
    RESHARD_TRACE(worldRank, "=== DEST: Computing Replication ===");
    RESHARD_TRACE(worldRank, "  sourceRep=%d sends to dst_reps [%d, %d)", sourceRep, targetRepStart, targetRepEnd);

    params.numLocalFollowers = 0;
    params.ringNextWorldRank = -1;
    params.isRingLast = true;

    RESHARD_TRACE(worldRank, "  myNode=%d (dstGpusPerDomain=%d)", myNode, dstGpusPerDomain);
    RESHARD_TRACE(worldRank, "  Scanning reps in range for replication:");

    int ringNextNode = -1;
    for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
      int repRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, rep);
      int repNode = repRank / dstGpusPerDomain;
      if (repNode > myNode && (ringNextNode == -1 || repNode < ringNextNode)) ringNextNode = repNode;
    }
    RESHARD_TRACE(worldRank, "  ringNextNode=%d", ringNextNode);

    for (int rep = targetRepStart; rep < targetRepEnd; rep++) {
      int repRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, rep);
      int repRankInMesh = repRank - dstMesh->startRank;
      int repNode = repRank / dstGpusPerDomain;

      if (repRank == worldRank) {
        RESHARD_TRACE(worldRank,
                      "    rep %d: comm_rank=%d, rank_in_mesh=%d, "
                      "node=%d -> SELF (skip)",
                      rep, repRank, repRankInMesh, repNode);
        continue;
      }

      if (repNode == myNode) {
        if (params.numLocalFollowers < MAX_LOCAL_FOLLOWERS) {
          params.localFollowerWorldRanks[params.numLocalFollowers++] = repRank;
          RESHARD_TRACE(worldRank,
                        "    rep %d: comm_rank=%d, rank_in_mesh=%d, "
                        "node=%d -> LSA follower[%d]",
                        rep, repRank, repRankInMesh, repNode, params.numLocalFollowers - 1);
        } else {
          RESHARD_FATAL(worldRank,
                        "prepareReshardParams: localFollowerWorldRanks "
                        "overflow! "
                        "numLocalFollowers=%d >= MAX_LOCAL_FOLLOWERS=%d. "
                        "Fix: increase MAX_LOCAL_FOLLOWERS in "
                        "reshard_limits.h.",
                        params.numLocalFollowers, MAX_LOCAL_FOLLOWERS);
        }
      } else if (rep > dstInfo.repIdx && params.ringNextWorldRank == -1) {
        bool foundRingNext = false;
        if (repNode == ringNextNode) {
          int ringNodeLocalReps = 0;
          for (int r = targetRepStart; r < targetRepEnd; r++) {
            int rRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, r);
            int rNode = rRank / dstGpusPerDomain;
            if (rNode == repNode) ringNodeLocalReps++;
          }
          int targetRingLocalRepIdx = params.localRepIdx;
          int nextActiveSourceSlots = activeSourceSlots;
          if (ringNodeLocalReps > 0 && nextActiveSourceSlots > ringNodeLocalReps)
            nextActiveSourceSlots = ringNodeLocalReps;
          if (nextActiveSourceSlots > 0 && targetRingLocalRepIdx >= nextActiveSourceSlots)
            targetRingLocalRepIdx = nextActiveSourceSlots - 1;

          int nextLocalRepIdx = 0;
          for (int r = targetRepStart; r <= rep; r++) {
            int rRank = getMeshRank(dstMesh, &fullDstInfo, params.myDstShardIdx, r);
            int rNode = rRank / dstGpusPerDomain;
            if (rNode == repNode) {
              if (r == rep) break;
              nextLocalRepIdx++;
            }
          }
          if (nextLocalRepIdx == targetRingLocalRepIdx) {
            params.ringNextWorldRank = repRank;
            params.isRingLast = false;
            RESHARD_TRACE(worldRank,
                          "    rep %d: comm_rank=%d, "
                          "rank_in_mesh=%d, node=%d -> "
                          "RING NEXT (localRepIdx=%d, "
                          "targetLocalRepIdx=%d)",
                          rep, repRank, repRankInMesh, repNode, params.localRepIdx, targetRingLocalRepIdx);
            foundRingNext = true;
          }
        }
        if (foundRingNext) break;
        RESHARD_TRACE(worldRank,
                      "    rep %d: comm_rank=%d, rank_in_mesh=%d, "
                      "node=%d -> (not relevant)",
                      rep, repRank, repRankInMesh, repNode);
      } else {
        RESHARD_TRACE(worldRank,
                      "    rep %d: comm_rank=%d, rank_in_mesh=%d, "
                      "node=%d -> (not relevant)",
                      rep, repRank, repRankInMesh, repNode);
      }
    }

    if (reshardGetLogLevel() >= RESHARD_LOG_TRACE) {
      if (params.numLocalFollowers > 0) {
        char fbuf[1024];
        int pos = 0;
        for (int i = 0; i < params.numLocalFollowers && pos < (int)sizeof(fbuf) - 16; i++) {
          pos +=
            snprintf(fbuf + pos, sizeof(fbuf) - pos, "%s%d", (i != 0) ? ", " : "", params.localFollowerWorldRanks[i]);
        }
        RESHARD_TRACE(worldRank, "  Replication summary:");
        RESHARD_TRACE(worldRank, "    numLocalFollowers=%d (ranks: %s)", params.numLocalFollowers, fbuf);
      } else {
        RESHARD_TRACE(worldRank, "  Replication summary:");
        RESHARD_TRACE(worldRank, "    numLocalFollowers=0");
      }
      RESHARD_TRACE(worldRank, "    ringNextWorldRank=%d, isRingLast=%d", params.ringNextWorldRank, params.isRingLast);
    }
  }

  {
    char sd[128], dd[128];
    fmtSizes(sd, sizeof(sd), params.srcDims, ndims);
    fmtSizes(dd, sizeof(dd), params.dstDims, ndims);
    RESHARD_DEBUG(worldRank, "========================================");
    RESHARD_DEBUG(worldRank, "prepareReshardParams() COMPLETE");
    RESHARD_DEBUG(worldRank, "  Final params summary:");
    RESHARD_DEBUG(worldRank, "    isSource=%d, isDest=%d", params.isSource, params.isDest);
    RESHARD_DEBUG(worldRank, "    srcDims=[%s], dstDims=[%s]", sd, dd);
    RESHARD_DEBUG(worldRank, "    srcShardTensorDim=%d, dstShardTensorDim=%d (%s)", params.srcShardTensorDim,
                  params.dstShardTensorDim, params.sameShardDim ? "SAME-DIM" : "CROSS-DIM");
    RESHARD_DEBUG(worldRank, "    srcShardCount=%d, dstShardCount=%d", params.srcShardCount, params.dstShardCount);
    RESHARD_DEBUG(worldRank, "    numTargets=%d, numSources=%d", params.numTargets, params.numSources);
    if (params.isDest) {
      RESHARD_DEBUG(worldRank,
                    "    hierarchical: localRepIdx=%d, "
                    "numLocalReps=%d, is_leader=%d",
                    params.localRepIdx, params.numLocalReps, params.isLeaderForSources);
      RESHARD_DEBUG(worldRank,
                    "    replication: numFollowers=%d, ring_next=%d, "
                    "isRingLast=%d",
                    params.numLocalFollowers, params.ringNextWorldRank, params.isRingLast);
    }
    RESHARD_DEBUG(worldRank, "========================================");
  }

  return params;
}

// ============================================================================
// Prepare Direct Algorithm Parameters
// ============================================================================

ncclReshardDirectParams prepareDirectReshardParams(
  int worldRank, const size_t* srcTensorDims, const size_t* dstTensorDims, int ndims,
  const ncclMesh_t* srcMesh, const ncclMesh_t* dstMesh, ncclWindow_t window,
  size_t elementsPerChunk, int numCtas, const size_t* allWindowOffsets) {
  ncclReshardDirectParams params;
  memset(&params, 0, sizeof(params));

  RESHARD_DEBUG(worldRank, "================================================");
  RESHARD_DEBUG(worldRank, "prepareDirectReshardParams() ENTER");
  RESHARD_DEBUG(worldRank, "  elementsPerChunk=%zu, numCtas=%d", elementsPerChunk, numCtas);

  params.window = window;
  params.elementsPerChunk = elementsPerChunk;
  params.totalCtas = numCtas;
  params.myWorldRank = worldRank;
  params.ndims = ndims;

  int srcMeshSize = srcMesh->dims[0] * srcMesh->dims[1];
  int dstMeshSize = dstMesh->dims[0] * dstMesh->dims[1];

  params.isSource = (worldRank >= srcMesh->startRank && worldRank < srcMesh->startRank + srcMeshSize);
  params.isDest = (worldRank >= dstMesh->startRank && worldRank < dstMesh->startRank + dstMeshSize);

  /* Mesh-level views (shardTensorDim, shardCount, repCount, etc.)
   * are identical for every rank in a mesh — compute them once at
   * startRank.  Per-rank fields (meshPos, shardIdx, repIdx) only
   * matter for participating ranks; defer those until isSource /
   * isDest are checked below. */
  ncclReshardMeshGroupInfo fullSrcInfo, fullDstInfo;
  computeMeshGroupInfo(srcMesh, srcMesh->startRank, &fullSrcInfo);
  computeMeshGroupInfo(dstMesh, dstMesh->startRank, &fullDstInfo);

  params.srcShardTensorDim = fullSrcInfo.shardTensorDim;
  params.dstShardTensorDim = fullDstInfo.shardTensorDim;
  params.srcShardCount = fullSrcInfo.shardCount;
  params.dstShardCount = fullDstInfo.shardCount;

  size_t globalDims[MAX_TENSOR_DIMS];
  for (int d = 0; d < ndims; d++) {
    if (params.isSource && srcTensorDims != nullptr)
      if (d == params.srcShardTensorDim) globalDims[d] = srcTensorDims[d] * params.srcShardCount;
      else globalDims[d] = srcTensorDims[d];
    else if (params.isDest && dstTensorDims != nullptr)
      if (d == params.dstShardTensorDim) globalDims[d] = dstTensorDims[d] * params.dstShardCount;
      else globalDims[d] = dstTensorDims[d];
    else globalDims[d] = 1;
  }

  for (int d = 0; d < ndims; d++) {
    if (d == params.srcShardTensorDim) params.srcDims[d] = globalDims[d] / params.srcShardCount;
    else params.srcDims[d] = globalDims[d];
    if (d == params.dstShardTensorDim) params.dstDims[d] = globalDims[d] / params.dstShardCount;
    else params.dstDims[d] = globalDims[d];
  }

  computeStrides(params.srcDims, ndims, params.srcStrides);
  computeStrides(params.dstDims, ndims, params.dstStrides);

  /* Per-rank position info — only computed for participating ranks. */
  params.mySrcShardIdx = -1;
  params.mySrcRepIdx = -1;
  params.myDstShardIdx = -1;
  params.myDstRepIdx = -1;
  ncclReshardMeshGroupInfo srcInfo{}, dstInfo{};
  if (params.isSource) {
    computeMeshGroupInfo(srcMesh, worldRank, &srcInfo);
    params.mySrcShardIdx = srcInfo.shardIdx;
    params.mySrcRepIdx = srcInfo.repIdx;
  }
  if (params.isDest) {
    computeMeshGroupInfo(dstMesh, worldRank, &dstInfo);
    params.myDstShardIdx = dstInfo.shardIdx;
    params.myDstRepIdx = dstInfo.repIdx;
  }

  int srcRepCount = fullSrcInfo.repCount;
  int dstRepCount = fullDstInfo.repCount;

  int dstGpusPerDomain = (reshardGetDstDomainSize() > 0) ? reshardGetDstDomainSize() : reshardGetGpusPerNode();

  ncclReshardRepLoadBalancer lb = {.srcRepCount = srcRepCount,
                                .dstRepCount = dstRepCount,
                                .dstGpusPerDomain = dstGpusPerDomain,
                                .dstRepStartRank = dstMesh->startRank,
                                .dstRepStride = (fullDstInfo.repMeshDim == 0) ? dstMesh->dims[1] : 1,
                                .mode = reshardGetLoadBalanceMode()};

  RESHARD_DEBUG(worldRank, "  isSource=%d, isDest=%d", params.isSource, params.isDest);
  RESHARD_DEBUG(worldRank, "  srcShardCount=%d, dstShardCount=%d", params.srcShardCount, params.dstShardCount);
  RESHARD_DEBUG(worldRank, "  srcRepCount=%d, dstRepCount=%d", srcRepCount, dstRepCount);

  // SOURCE: Compute targets - send to ALL overlapping dests in target rep
  // range.
  if (params.isSource) {
    params.numTargets = 0;

    int targetRepStart, targetRepEnd;
    getTargetRepRange(&lb, srcInfo.repIdx, &targetRepStart, &targetRepEnd);

    RESHARD_TRACE(worldRank, "=== DIRECT SOURCE: Computing Targets ===");
    RESHARD_TRACE(worldRank, "  mySrcShardIdx=%d, mySrcRepIdx=%d", params.mySrcShardIdx, params.mySrcRepIdx);
    RESHARD_TRACE(worldRank, "  target_rep_range=[%d, %d)", targetRepStart, targetRepEnd);

    for (int dstShard = 0; dstShard < params.dstShardCount; dstShard++) {
      ncclReshardTransferPlan plan;
      computeTransferPlan(params.srcDims, params.srcStrides, params.srcShardTensorDim, params.mySrcShardIdx,
                          params.dstDims, params.dstStrides, params.dstShardTensorDim, dstShard, ndims,
                          elementsPerChunk, &plan);

      if (plan.totalInnerTransfers == 0) continue;

      for (int dstRep = targetRepStart; dstRep < targetRepEnd; dstRep++) {
        if (params.numTargets >= MAX_DIRECT_TARGETS) {
          RESHARD_FATAL(worldRank,
                        "prepareDirectReshardParams: target list TRUNCATED at "
                        "dstShard %d, dstRep %d! "
                        "numTargets=%d >= MAX_DIRECT_TARGETS=%d. Remaining "
                        "(dstShard, dstRep) pairs dropped. "
                        "Some dest ranks will NEVER receive data - kernel "
                        "WILL HANG. "
                        "Fix: increase MAX_DIRECT_TARGETS in reshard_limits.h.",
                        dstShard, dstRep, params.numTargets, MAX_DIRECT_TARGETS);
        }

        int dstRank = getMeshRank(dstMesh, &fullDstInfo, dstShard, dstRep);

        ncclReshardDirectTargetInfo* target = &params.targets[params.numTargets++];
        target->dstShardIdx = dstShard;
        target->dstRepIdx = dstRep;
        target->dstWorldRank = dstRank;
        for (int d = 0; d < ndims; d++) {
          target->overlapStart[d] = plan.overlapStart[d];
          target->overlapEnd[d] = plan.overlapEnd[d];
        }
        target->plan = plan;
        target->isContiguous = (plan.totalInnerTransfers == 1);
        target->totalBytes = plan.totalInnerTransfers * plan.innerSize;
        target->windowOffset = (allWindowOffsets != nullptr) ? allWindowOffsets[dstRank] : 0;

        RESHARD_TRACE(worldRank,
                      "  target[%d]: dstShard=%d, dstRep=%d, "
                      "dstRank=%d, bytes=%zu, windowOffset=%zu",
                      params.numTargets - 1, dstShard, dstRep, dstRank, target->totalBytes, target->windowOffset);
      }
    }

    RESHARD_TRACE(worldRank, "  Total targets: %d", params.numTargets);
  }

  // DEST: Compute sources - receive from ALL overlapping sources
  if (params.isDest) {
    params.numSources = 0;

    int sourceRep = getSourceRepForDest(&lb, dstInfo.repIdx);

    RESHARD_TRACE(worldRank, "=== DIRECT DEST: Computing Sources ===");
    RESHARD_TRACE(worldRank, "  myDstShardIdx=%d, myDstRepIdx=%d", params.myDstShardIdx, params.myDstRepIdx);
    RESHARD_TRACE(worldRank, "  sourceRep=%d", sourceRep);

    for (int srcShard = 0; srcShard < params.srcShardCount; srcShard++) {
      ncclReshardTransferPlan plan;
      computeTransferPlan(params.srcDims, params.srcStrides, params.srcShardTensorDim, srcShard, params.dstDims,
                          params.dstStrides, params.dstShardTensorDim, params.myDstShardIdx, ndims, elementsPerChunk,
                          &plan);

      if (plan.totalInnerTransfers == 0) continue;

      if (params.numSources >= MAX_DIRECT_SOURCES) {
        RESHARD_FATAL(worldRank,
                      "prepareDirectReshardParams: source list TRUNCATED at "
                      "srcShard %d! "
                      "numSources=%d >= MAX_DIRECT_SOURCES=%d. Remaining "
                      "sources dropped. "
                      "Dest rank will wait for data that never arrives - kernel "
                      "WILL HANG. "
                      "Fix: increase MAX_DIRECT_SOURCES in reshard_limits.h.",
                      srcShard, params.numSources, MAX_DIRECT_SOURCES);
      }

      int srcRank = getMeshRank(srcMesh, &fullSrcInfo, srcShard, sourceRep);

      ncclReshardDirectSourceInfo* source = &params.sources[params.numSources++];
      source->signalBase = srcRank * numCtas;
      source->plan = plan;
      source->isContiguous = (plan.totalInnerTransfers == 1);
      source->totalBytes = plan.totalInnerTransfers * plan.innerSize;

      RESHARD_TRACE(worldRank,
                    "  source[%d]: srcShard=%d, srcRep=%d, "
                    "srcRank=%d, bytes=%zu",
                    params.numSources - 1, srcShard, sourceRep, srcRank, source->totalBytes);
    }

    RESHARD_TRACE(worldRank, "  Total sources: %d", params.numSources);
  }

  RESHARD_DEBUG(worldRank, "prepareDirectReshardParams() EXIT");
  RESHARD_DEBUG(worldRank, "================================================");

  return params;
}
