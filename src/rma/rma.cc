#include <assert.h>
#include "nccl.h"
#include "alloc.h"
#include "checks.h"
#include "comm.h"
#include "rma/rma.h"

static bool isLsaAccessible(struct ncclComm* comm, int rank) {
  for (int i = 0; i < comm->devrState.lsaSize; i++) {
    if (comm->devrState.lsaRankList[i] == rank) {
      return true;
    }
  }
  return false;
}

ncclResult_t ncclRmaWaitSignal(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // If we have both proxy and CE tasks, execute them in parallel
  if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
    cudaStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
    cudaEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;

    // Record event on input stream first to establish dependency
    CUDACHECKGOTO(cudaEventRecord(ceEvent, stream), ret, fail);

    // Set up CE stream for parallel execution
    CUDACHECKGOTO(cudaStreamWaitEvent(ceStream, ceEvent, 0), ret, fail);

    // Launch both operations
    NCCLCHECKGOTO(ncclRmaWaitSignalProxy(comm, plan, stream), ret, fail);
    NCCLCHECKGOTO(ncclRmaWaitSignalCe(comm, plan, ceStream), ret, fail);

    // Synchronize streams
    CUDACHECKGOTO(cudaEventRecord(ceEvent, ceStream), ret, fail);
    CUDACHECKGOTO(cudaStreamWaitEvent(stream, ceEvent, 0), ret, fail);
  }
  else if (plan->rmaArgs->nRmaTasksProxy > 0) {
    NCCLCHECKGOTO(ncclRmaWaitSignalProxy(comm, plan, stream), ret, fail);
  }
  else if (plan->rmaArgs->nRmaTasksCe > 0) {
    NCCLCHECKGOTO(ncclRmaWaitSignalCe(comm, plan, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}


ncclResult_t ncclRmaPut(struct ncclComm* comm, struct ncclKernelPlan* plan, cudaStream_t stream){
  ncclResult_t ret = ncclSuccess;

  // If we have both proxy and CE tasks, execute them in parallel
  if (plan->rmaArgs->nRmaTasksProxy > 0 && plan->rmaArgs->nRmaTasksCe > 0) {
    cudaStream_t ceStream = comm->rmaState.rmaCeState.ceStream;
    cudaEvent_t ceEvent = comm->rmaState.rmaCeState.ceEvent;

    // Record event on input stream first to establish dependency
    CUDACHECKGOTO(cudaEventRecord(ceEvent, stream), ret, fail);

    // Set up CE stream for parallel execution
    CUDACHECKGOTO(cudaStreamWaitEvent(ceStream, ceEvent, 0), ret, fail);

    // Launch both operations
    NCCLCHECKGOTO(ncclRmaPutProxy(comm, plan, stream), ret, fail);
    NCCLCHECKGOTO(ncclRmaPutCe(comm, plan, ceStream), ret, fail);

    // Synchronize streams
    CUDACHECKGOTO(cudaEventRecord(ceEvent, ceStream), ret, fail);
    CUDACHECKGOTO(cudaStreamWaitEvent(stream, ceEvent, 0), ret, fail);
  }
  else if (plan->rmaArgs->nRmaTasksProxy > 0) {
    NCCLCHECKGOTO(ncclRmaPutProxy(comm, plan, stream), ret, fail);
  }
  else if (plan->rmaArgs->nRmaTasksCe > 0) {
    NCCLCHECKGOTO(ncclRmaPutCe(comm, plan, stream), ret, fail);
  }

exit:
  return ret;
fail:
  goto exit;
}

ncclResult_t ncclLaunchRma(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  cudaStream_t stream = comm->planner.streams->stream;

  switch (plan->rmaArgs->func) {
    case ncclFuncPutSignal:
      NCCLCHECKGOTO(ncclRmaPut(comm, plan, stream), ret, fail);
      break;
    case ncclFuncSignal:
      NCCLCHECKGOTO(ncclRmaPut(comm, plan, stream), ret, fail);
      break;
    case ncclFuncWaitSignal:
      NCCLCHECKGOTO(ncclRmaWaitSignal(comm, plan, stream), ret, fail);
      break;
    default:
      ret = ncclInvalidUsage;
  }

exit:
  return ret;
fail:
  goto exit;
}

static inline bool isRmaPutOrSignal(ncclFunc_t func) {
  return (func == ncclFuncPutSignal || func == ncclFuncSignal);
}

// Check if two RMA tasks can be batched together
static inline bool canBatchRmaTasks(struct ncclTaskRma* task1, struct ncclTaskRma* task2) {
  // Check if the tasks are in the same context
  if (task1->ctx != task2->ctx) return false;

  // Check if the tasks are the same function
  if (task1->func == task2->func) return true;

  // Put/Signal tasks can be batched together
  if (isRmaPutOrSignal(task1->func) && isRmaPutOrSignal(task2->func)) {
    return true;
  }

  return false;
}

// Schedule comm->planner RMA tasks to the plan and split the RMA tasks into CE and Proxy tasks
// Then seek opportunities to batch tasks, batching checked for consecutive operations targeting the same context
// - ncclFuncWaitSignal does not perform further batching as the API can already batch waitSignal from multiple peers
// - Consecutive put/signal operation can be batched into the same plan
ncclResult_t scheduleRmaTasksToPlan(struct ncclComm* comm, struct ncclKernelPlan* plan) {
  ncclResult_t ret = ncclSuccess;
  struct ncclKernelPlanner* planner = &comm->planner;

  // Find the first non-empty context queue
  int ctx = -1;
  for (int i = 0; i < comm->config.numRmaCtx; i++) {
    if (!ncclIntruQueueEmpty(&planner->rmaTaskQueues[i])) {
      ctx = i;
      break;
    }
  }

  // No RMA tasks to schedule
  if (ctx == -1) return ncclSuccess;

  struct ncclIntruQueue<struct ncclTaskRma, &ncclTaskRma::next>* ctxQueue = &planner->rmaTaskQueues[ctx];

  // Get the first task to determine the operation category
  struct ncclTaskRma* firstTask = ncclIntruQueueDequeue(ctxQueue);

  // Initialize plan
  plan->isRma = true;
  plan->rmaArgs = ncclMemoryStackAlloc<struct ncclRmaArgs>(&comm->memScoped);
  plan->rmaArgs->ctx = ctx;
  plan->rmaArgs->func = firstTask->func;
  plan->rmaArgs->nRmaTasks = 0;
  plan->rmaArgs->nRmaTasksProxy = 0;
  plan->rmaArgs->nRmaTasksCe = 0;

  // WaitSignal tasks
  if (firstTask->func == ncclFuncWaitSignal) {
    // Allocate temporary arrays to hold peers and nsignals for both proxy and CE paths
    int* peersCe = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* nsignalsCe = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* peersProxy = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);
    int* nsignalsProxy = ncclMemoryStackAlloc<int>(&comm->memScoped, firstTask->npeers);

    int npeersCe = 0;
    int npeersProxy = 0;

    // Go over the firstTask->peers and split them based on LSA accessibility
    for (int i = 0; i < firstTask->npeers; i++) {
      int peerRank = firstTask->peers[i];
      bool lsaAccessible = isLsaAccessible(comm, peerRank);

      if (lsaAccessible) {
        // Add to CE list
        peersCe[npeersCe] = peerRank;
        nsignalsCe[npeersCe] = firstTask->nsignals[i];
        npeersCe++;
      } else {
        // Add to Proxy list
        peersProxy[npeersProxy] = peerRank;
        nsignalsProxy[npeersProxy] = firstTask->nsignals[i];
        npeersProxy++;
      }
    }

    // Initialize the CE task if there are CE peers
    if (npeersCe > 0) {
      struct ncclTaskRma* waitSignalTaskCe = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
      waitSignalTaskCe->func = ncclFuncWaitSignal;
      waitSignalTaskCe->ctx = firstTask->ctx;
      waitSignalTaskCe->signalMode = firstTask->signalMode;
      waitSignalTaskCe->peers = peersCe;
      waitSignalTaskCe->nsignals = nsignalsCe;
      waitSignalTaskCe->npeers = npeersCe;
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, waitSignalTaskCe);
      plan->rmaArgs->nRmaTasksCe = 1;
    } else {
      plan->rmaArgs->nRmaTasksCe = 0;
    }

    // Initialize the Proxy task if there are Proxy peers
    if (npeersProxy > 0) {
      struct ncclTaskRma* waitSignalTaskProxy = ncclMemoryPoolAlloc<struct ncclTaskRma>(&comm->memPool_ncclTaskRma, &comm->memPermanent);
      waitSignalTaskProxy->func = ncclFuncWaitSignal;
      waitSignalTaskProxy->ctx = firstTask->ctx;
      waitSignalTaskProxy->signalMode = firstTask->signalMode;
      waitSignalTaskProxy->peers = peersProxy;
      waitSignalTaskProxy->nsignals = nsignalsProxy;
      waitSignalTaskProxy->npeers = npeersProxy;
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, waitSignalTaskProxy);
      plan->rmaArgs->nRmaTasksProxy = 1;
    } else {
      plan->rmaArgs->nRmaTasksProxy = 0;
    }

    plan->rmaArgs->nRmaTasks = (npeersCe > 0 ? 1 : 0) + (npeersProxy > 0 ? 1 : 0);
    planner->nTasksRma -= 1;
    // Free the original WaitSignal task (split into CE and Proxy tasks)
    ncclMemoryPoolFree(&comm->memPool_ncclTaskRma, firstTask);
  }
  // Put/Signal tasks
  else {
    // Check if the first task is LSA accessible
    bool lsaAccessible = isLsaAccessible(comm, firstTask->peer);

    plan->rmaArgs->nRmaTasks = 1;
    plan->rmaArgs->nRmaTasksProxy = lsaAccessible ? 0 : 1;
    plan->rmaArgs->nRmaTasksCe = lsaAccessible ? 1 : 0;

    if (lsaAccessible) {
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, firstTask);
    } else {
      ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, firstTask);
    }

    planner->nTasksRma -= 1;

    // Batch consecutive tasks from the same context that match operation category
    while (!ncclIntruQueueEmpty(ctxQueue)) {
      struct ncclTaskRma* task = ncclIntruQueueHead(ctxQueue);

      // Check if this task can be batched with the first task
      if (!canBatchRmaTasks(firstTask, task)) {
        break;
      }

      bool lsaAccessible = isLsaAccessible(comm, task->peer);

      // If the task can be batched, remove from context queue and add to plan
      ncclIntruQueueDequeue(ctxQueue);
      if (lsaAccessible) {
        ncclIntruQueueEnqueue(&plan->rmaTaskQueueCe, task);
        plan->rmaArgs->nRmaTasksCe++;
      } else {
        ncclIntruQueueEnqueue(&plan->rmaTaskQueueProxy, task);
        plan->rmaArgs->nRmaTasksProxy++;
      }
      plan->rmaArgs->nRmaTasks++;
      planner->nTasksRma -= 1;
    }
  }

  INFO(NCCL_COLL, "scheduleRmaTasksToPlan: rank=%d ctx=%d func=%d nRmaTasks=%d nRmaTasksProxy=%d nRmaTasksCe=%d",
    comm->rank, ctx, plan->rmaArgs->func, plan->rmaArgs->nRmaTasks, plan->rmaArgs->nRmaTasksProxy, plan->rmaArgs->nRmaTasksCe);

  return ret;
}
