/*************************************************************************
 * Copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/


#ifndef NCCL_ALLGATHERV_SCHED_H_
#define NCCL_ALLGATHERV_SCHED_H_

#include "scheduler.h"

ncclResult_t ncclScheduleBcastTasksToPlan(
    struct ncclComm* comm, struct ncclKernelPlan* plan, struct ncclKernelPlanBudget* budget
) {
  struct ncclKernelPlanner* planner = &comm->planner;
  int nChannels = comm->nChannels;
  int nRanks = comm->nRanks;
  int maxitem = ncclMaxDevWorkBatchBytes(comm->cudaArch) / sizeof(ncclDevWorkBcast);
  bool newBatch = true;
  if (planner->nTasksBcast > 0) {
    uint32_t batchTasks = 0;
    size_t sumBcastBytes = 0;
    size_t maxBcastBytes = 0;

    // Make a batch consisting of one bcast from each peer.
    if (plan->nWorkBatches != 0) return ncclSuccess;
    for (int peer=planner->bcast_info.minBcastPeer; peer <= planner->bcast_info.maxBcastPeer; peer++) {
      struct ncclTaskBcast* t = ncclIntruQueueHead(&planner->peers[peer].bcastQueue);
      if (t == nullptr) continue;
      // see if we can fit another batch to args, and a bunch of bcast to workStorage
      // Each batch can fit 64 bcast, if batchTasks > 64 we use nextExtends to extend the batch.
      if (!ncclTestBudget(budget, nChannels * DIVUP(batchTasks+1, 64), (batchTasks+1) * sizeof(struct ncclDevWorkBcast)) || batchTasks+1 == maxitem) {
        break;
      }
      sumBcastBytes += t->count;
      maxBcastBytes = std::max(maxBcastBytes, t->count);
      batchTasks += 1;
    }

    if (batchTasks == 0) {
      return ncclSuccess;
    }

    // find best protocol
    struct ncclTaskColl tcoll;
    memset(&tcoll, 0, sizeof(tcoll));
    tcoll.func = ncclFuncAllGather;
    tcoll.count = maxBcastBytes;
    tcoll.datatype = ncclInt8;
    tcoll.algorithm = NCCL_ALGO_RING;
    tcoll.protocol = NCCL_PROTO_UNDEF;
    NCCLCHECK(ncclGetAlgoInfo(comm, &tcoll, /*collNetSupport=*/0, /*nvlsSupport=*/0, /*nTasksPerChannel=*/1, /*simInfo=*/nullptr));

    // calculate chunk size
    int proto = tcoll.protocol;
    int chunkSteps = 1;
    int sliceSteps = 1;
    int stepSize = comm->buffSizes[proto]/NCCL_STEPS;
    int chunkSize = chunkSteps*stepSize;
    if (proto == NCCL_PROTO_LL) chunkSize = chunkSize/2;
    if (proto == NCCL_PROTO_LL128) chunkSize = (chunkSize/NCCL_LL128_LINEELEMS)*NCCL_LL128_DATAELEMS;
    size_t grainSize = ncclProtoGrainSize(proto);
    nChannels = tcoll.nMaxChannels;
    chunkSize = chunkSize / grainSize * grainSize;


    // Determine thread count per block
    int threadPerBlock = std::max((unsigned long)(tcoll.nWarps * WARP_SIZE), 64 * sizeof(ncclDevWorkBcast) / 16 + 3 * WARP_SIZE);
    plan->threadPerBlock = threadPerBlock;

    // Choose kernel for plan. Based on proto, algo=ring
    int funcIndex = ncclDevFuncId(ncclFuncAllGatherV, /*devRedOp,type=*/0,0, NCCL_ALGO_RING, proto);
    if (!plan->kernelSpecialized) {
      plan->kernelFn = ncclDevKernelForFunc[funcIndex];
      plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[funcIndex];
    }

    // Compute opCount for proxy work.
    uint64_t proxyOpCount = uint64_t(comm->collOpCount++)<<1 | /*bcast=*/0;

    // Break each bcast into nParts evenly, each part assigned to a channel.
    int nParts = nChannels;
    uint32_t channelWorkBytes[MAXCHANNELS] = {0};
    for (int part=0; part < nParts; part++) {
      // Sort tasks according to ring depth upstream from us.
      int nTasks = batchTasks;
      int channelId = part;

      // reset comm->ringTasks
      struct ncclTaskBcast** ringTasks = (struct ncclTaskBcast**)comm->ringTasks;
      for (int r=0; r < nRanks; r++) ringTasks[r] = nullptr;

      // calculate, and find min and max ring depth among this plan's tasks
      int minRingDepth = INT_MAX;
      int maxRingDepth = INT_MIN;
      struct ncclTaskBcast* t = nullptr;
      for (int peer=planner->bcast_info.minBcastPeer; nTasks != 0; peer++) {
        t = ncclIntruQueueHead(&planner->peers[peer].bcastQueue);
        if (t != nullptr) {
          nTasks -= 1;
          int index = comm->channels[channelId].ring.rankToIndex[peer];
          // Need to flip from "downstream from us" to "upstream from us".
          int ringDepth = (index == 0) ? 0 : nRanks-index;
          ringTasks[ringDepth] = t;
          t->ringDepth = ringDepth;
          minRingDepth = std::min(minRingDepth, ringDepth);
          maxRingDepth = std::max(maxRingDepth, ringDepth);
        }
      }

      // Start an empty dev work batch.
      int sendSlices=0, recvSlices=0;
      int maxSendSlices=0, maxRecvSlices=0;
      // Add each task to the batch in ring depth order.
      int nBcasts = 0;
      int slices = 0;

      for (int ringDepth=minRingDepth; ringDepth <= maxRingDepth; ringDepth++) {
        t = ringTasks[ringDepth];
        if (t != nullptr) {
          size_t partBytes = divUp(t->count, nParts);
          size_t offset_lo = std::min<size_t>(alignUp((part+0)*partBytes, 256), t->count);
          size_t offset_hi = std::min<size_t>(alignUp((part+1)*partBytes, 256), t->count);
          if (offset_hi == offset_lo) {
            continue;
          }
          plan->channelMask |= uint64_t(1)<<channelId;
          struct ncclWorkList* workNode = ncclMemoryStackAllocInlineArray<ncclWorkList, ncclDevWorkBcast>(&comm->memScoped, 1);
          workNode->workType = ncclDevWorkTypeBcast;
          workNode->size = sizeof(struct ncclDevWorkBcast);
          ncclIntruQueueEnqueue(&plan->workQueue, workNode);
          struct ncclDevWorkBcast* work = (struct ncclDevWorkBcast*)(workNode+1);
          work->recvbuff = (char*)t->recvbuff + offset_lo;
          work->bytes = offset_hi-offset_lo;
          work->ringDepth = ringDepth;
          work->chunkSize = chunkSize;
          if (work->bytes != 0) {
            slices = divUp(work->bytes, chunkSize);
            if (ringDepth != 0) {
              recvSlices += slices;
              maxRecvSlices = std::max(maxRecvSlices, slices);
            }
            if (ringDepth != nRanks-1) {
              sendSlices += slices;
              maxSendSlices = std::max(maxSendSlices, slices);
            }
            if (ringDepth == 0) {
              work->sendbuff = (char*)t->sendbuff + offset_lo;
            }
            channelWorkBytes[channelId] += sizeof(ncclDevWorkBcast);
          }
          nBcasts += 1;
          ncclAddWorkBatchToPlan(comm, plan, channelId, ncclDevWorkTypeBcast, funcIndex, plan->workBytes, /*p2pEpoch=*/-1, /*p2pRound=*/-1, newBatch);
          newBatch = false;
          plan->workBytes += sizeof(ncclDevWorkBcast);
      }
      }

      // calculate proxy for this channel
      if (sendSlices + recvSlices != 0) {
        struct ncclProxyOp proxyOp = {};
        proxyOp.channelId = channelId;
        proxyOp.opCount = proxyOpCount;
        proxyOp.rank = comm->rank;

        proxyOp.coll = ncclFuncAllGatherV;
        proxyOp.pattern = ncclPatternRing;
        proxyOp.nsteps = 0;
        proxyOp.specifics.bcast.sendSlices = sendSlices;
        proxyOp.specifics.bcast.recvSlices = recvSlices;
        proxyOp.specifics.bcast.stepSize = stepSize;

        proxyOp.sliceSteps = sliceSteps;
        proxyOp.chunkSteps = chunkSteps;
        proxyOp.dtype = ncclInt8;
        proxyOp.redOp = ncclSum;
        proxyOp.protocol = proto;

        proxyOp.chunkSize = chunkSize;
        proxyOp.sliceSize = chunkSize / chunkSteps * sliceSteps;
        proxyOp.nbytes = stepSize * sliceSteps;
        proxyOp.reg = 0;

        proxyOp.sendbuff = nullptr;
        proxyOp.recvbuff = nullptr;
        proxyOp.channelSize = 0;
        proxyOp.ringAlgo = nullptr;
        proxyOp.loopOffset = 0;
        proxyOp.loopSize = 0;

        // profiler support
        proxyOp.eActivationMask = 0;
        proxyOp.nChannels = nChannels;
        NCCLCHECK(ncclAddProxyOpIfNeeded(comm, plan, &proxyOp));
      }
    }
    int nTasks = batchTasks;
    planner->nTasksBcast -= nTasks;
    for (int peer=planner->bcast_info.minBcastPeer; nTasks != 0; peer++) {
      struct ncclTaskBcast* t = ncclIntruQueueTryDequeue(&planner->peers[peer].bcastQueue);
      if (t != nullptr) {
        --nTasks;
        ncclIntruQueueEnqueue(&plan->bcastTaskQueue, t);
        plan->nTasksBcast += 1;
      }
    }
  }
  return ncclSuccess;
}

#endif // NCCL_ALLGATHERV_SCHED_H_
