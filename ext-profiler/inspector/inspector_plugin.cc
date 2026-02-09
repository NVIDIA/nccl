/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <linux/limits.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "profiler.h"
#include "inspector.h"

#define __hidden __attribute__ ((visibility("hidden")))

static int gInitialized;

static pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;


/*
 * Description:
 *   Records an event trace with timestamp and sequence number
 *
 * Thread Safety:
 *   Not thread-safe - must be called with proper locking. This function
 *   is designed to be called from within locked sections where the
 *   collective info structure is already protected.
 *
 * Input:
 *   struct inspectorEventTraceInfo* evtTrace - event trace array
 *   int eventIndex - index in the event trace array (must be valid)
 *   struct inspectorCollInfo* collInfo - collective info structure (must not be NULL)
 *
 * Output:
 *   Event trace is updated with current timestamp and next sequence
 *   number from collective
 *
 * Return:
 *   uint64_t - the sequence number assigned to this event
 *
 * Preconditions:
 *   - collInfo must not be NULL
 *   - eventIndex must be within valid bounds for evtTrace array
 *   - Function must be called from within a locked section
 */
static uint64_t inspectorRecordEventTrace(struct inspectorEventTraceInfo* evtTrace,
                                          int eventIndex,
                                          struct inspectorCollInfo* collInfo) {
  evtTrace[eventIndex].ts = inspectorGetTime();
  evtTrace[eventIndex].sn = ++collInfo->collEvtTrk.sn; // Increment coll sequence counter

  return evtTrace[eventIndex].sn;
}

/*
 * Description:
 *
 *   Initializes the NCCL Inspector plugin and global state for a
 *   communicator.
 *
 * Thread Safety:
 *   Thread-safe (uses mutex for initialization).
 *
 * Input:
 *   void** context - pointer to plugin context.
 *   int* eActivationMask - pointer to activation mask output.
 *   const char* commName - communicator name.
 *   uint64_t commHash - communicator hash.
 *   int nNodes - number of nodes.
 *   int nranks - number of ranks.
 *   int rank - rank.
 *   ncclDebugLogger_t logfn - logger function pointer.
 *
 * Output:
 *   context is set to plugin context; eActivationMask is set.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginInit(void** context, uint64_t commHash,
                                          int* eActivationMask,
                                          const char* commName,
                                          int nNodes, int nranks, int rank,
                                          ncclDebugLogger_t logfn) {
  inspectorResult_t res = inspectorSuccess;
  *context = nullptr;
  logFn = logfn;

  pthread_mutex_lock(&gLock);
  if (++gInitialized == 1) {
    res = inspectorGlobalInit(rank);
    if (res != inspectorSuccess) {
      INFO_INSPECTOR("Inspector Init Failed %s:%d -> error %d: %s",__FILE__, __LINE__, res,
           inspectorErrorString(res));
      gInitialized = 0;
      pthread_mutex_unlock(&gLock);
      return ncclSuccess;
    }
  }
  pthread_mutex_unlock(&gLock);

  res = inspectorAddComm((struct inspectorCommInfo **)context,
                         commName, commHash,
                         nNodes, nranks, rank);
  if (res != inspectorSuccess) {
    INFO_INSPECTOR("%s:%d -> error %d: %s", __FILE__, __LINE__, res,
                   inspectorErrorString(res));
    return ncclSuccess;
  }
  *eActivationMask = ncclProfileColl | ncclProfileKernelCh;
  INFO(NCCL_INIT, "PROFILER/Plugin: init commName: %s commHash: %lu nranks: %d rank: %d",
       commName ? commName : "", commHash, nranks, rank);
  return ncclSuccess;
}

/*
 * Description:
 *
 *   Finalizes the NCCL Inspector plugin and global state for a
 *   communicator.
 *
 * Thread Safety:
 *   Thread-safe (uses mutex for finalization).
 *
 * Input:
 *   void* context - plugin context.
 *
 * Output:
 *   Plugin context is finalized and cleaned up.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginFinalize(void* context) {
  inspectorDelComm((struct inspectorCommInfo *)context);
  pthread_mutex_lock(&gLock);
  if (--gInitialized == 0) {
    inspectorGlobalFinalize();
  }
  pthread_mutex_unlock(&gLock);
  return ncclSuccess;
}

inspectorResult_t inspectorPluginCollInfoRef(struct inspectorCollInfo *collInfo) {
  collInfo->refCount += 1;
  return inspectorSuccess;
}

inspectorResult_t inspectorPluginCollInfoRefSafe(struct inspectorCollInfo *collInfo) {
  inspectorLockWr(&collInfo->guard);
  inspectorPluginCollInfoRef(collInfo);
  inspectorUnlockRWLock(&collInfo->guard);
  return inspectorSuccess;
}

inspectorResult_t inspectorPluginCollInfoDeRef(struct inspectorCollInfo *collInfo) {
  collInfo->refCount -= 1;
  if (collInfo->refCount == 0) {
    inspectorLockDestroy(&collInfo->guard);
    memset(collInfo, 0, sizeof(struct inspectorCollInfo));
    free(collInfo);
    return inspectorReturn;
  }
  return inspectorSuccess;
}

inspectorResult_t inspectorPluginCollInfoDeRefSafe(struct inspectorCollInfo *collInfo) {
  inspectorLockWr(&collInfo->guard);
  inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
  inspectorUnlockRWLock(&collInfo->guard);
  return res;
}

/*
 * Description:
 *   Initializes a new inspectorCollInfo structure for a collective
 *   event.
 *
 * Thread Safety:
 *   Not thread-safe (allocates and initializes a new collective info
 *   structure).
 *
 * Input:
 *
 *   struct inspectorCollInfo **collInfo - pointer to output
 *   collective info struct.
 *   ncclProfilerEventDescr_t *eDescr - event descriptor.
 *
 * Output:
 *   collInfo is set to the new collective info struct.
 *
 * Return:
 *   None.
 */
static void inspectorPluginCollInfoInit(struct inspectorCollInfo **collInfo,
                                        ncclProfilerEventDescr_t *eDescr,
                                        struct inspectorCommInfo *commInfo) {
  struct inspectorCollInfo *collInfoPtr
    = (struct inspectorCollInfo*)calloc(1, sizeof(struct inspectorCollInfo));
  if (collInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for collective info structure");
    *collInfo = nullptr;
    return;
  }
  collInfoPtr->type = ncclProfileColl;
  collInfoPtr->refCount = 0;
  inspectorPluginCollInfoRef(collInfoPtr); //self ref; no locks needed
  collInfoPtr->func = eDescr->coll.func;
  collInfoPtr->sn = eDescr->coll.seqNumber;
  collInfoPtr->nChannels = eDescr->coll.nChannels;
  if (collInfoPtr->nChannels > 0) {
    inspectorPluginCollInfoRef(collInfoPtr); //extra ref for kernel completion
  }
  collInfoPtr->tsStartUsec = inspectorGetTime();
  collInfoPtr->msgSizeBytes =
    ncclTypeSize(inspectorStringToDatatype(eDescr->coll.datatype)) * eDescr->coll.count;


  collInfoPtr->commInfo = commInfo;
  collInfoPtr->collEvtTrk.sn = 0;
  collInfoPtr->collEvtTrk.nChannels = collInfoPtr->nChannels;
  inspectorRecordEventTrace(collInfoPtr->collEvtTrk.evntTrace,
                            NCCL_INSP_EVT_TRK_COLL_START, collInfoPtr);

  inspectorLockInit(&collInfoPtr->guard);
  *collInfo = collInfoPtr;
}

/*
 * Description:
 *
 *   Initializes a new inspectorKernelChInfo structure for a kernel
 *   channel event.
 *
 * Thread Safety:
 *   Not thread-safe (initializes kernel channel info within a
 *   collective info structure).
 *
 * Input:
 *   struct inspectorKernelChInfo **kernelChInfo - pointer to output
 *   kernel channel info struct.
 *   ncclProfilerEventDescr_t *eDescr - event descriptor.
 *
 * Output:
 *
 *   kernelChInfo is set to the new kernel channel info struct.
 *
 * Return:
 *   None.
 */
static void inspectorPluginKernelChInfoInit(struct inspectorKernelChInfo **kernelChInfo,
                                            ncclProfilerEventDescr_t *eDescr) {
  if (eDescr->parentObj) {
    uint64_t parentType=*(uint64_t*)eDescr->parentObj;
    if (parentType == ncclProfileColl) {
      struct inspectorCollInfo *collInfo = (struct inspectorCollInfo*)eDescr->parentObj;
      if (collInfo && collInfo->type == ncclProfileColl) {
        inspectorLockWr(&collInfo->guard);
        struct inspectorEventTraceInfo *krnlEvtTrk =
          collInfo->collEvtTrk.kernelCh[eDescr->kernelCh.channelId].evntTrace;
        inspectorRecordEventTrace(krnlEvtTrk,
                                  NCCL_INSP_EVT_TRK_KERNEL_START,
                                  collInfo);
        struct inspectorKernelChInfo *kernelChInfoPtr
          = &collInfo->kernelCh[eDescr->kernelCh.channelId];
        kernelChInfoPtr->type = ncclProfileKernelCh;
        kernelChInfoPtr->channelId = eDescr->kernelCh.channelId;
        kernelChInfoPtr->startGpuClk = eDescr->kernelCh.pTimer;
        if (kernelChInfoPtr->stopGpuClk == 0) {
          inspectorPluginCollInfoRef(collInfo); //Pairs with Record Kernel Stop event
        }
        kernelChInfoPtr->tsStartUsec = inspectorGetTime();
        if (collInfo->nKernelChStarted == 0) {
          collInfo->tsStartUsec = kernelChInfoPtr->tsStartUsec;
        }
        collInfo->nKernelChStarted += 1;
        inspectorPluginCollInfoRef(collInfo); //Pairs with Stop Kernel Event
        kernelChInfoPtr->collInfo = collInfo;

        *kernelChInfo = kernelChInfoPtr;
        inspectorUnlockRWLock(&collInfo->guard);
      }
    }
  }
}
/*
 * Description:
 *
 *   Starts a profiling event for the NCCL Inspector plugin.
 *
 * Thread Safety:
 *   Thread-safe (allocates and initializes event structures).
 *
 * Input:
 *   void* context - plugin context.
 *   void** eHandle - pointer to event handle output.
 *   ncclProfilerEventDescr_t* eDescr - event descriptor.
 *
 * Output:
 *   eHandle is set to the new event structure.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginStartEvent(void* context,
                                                void** eHandle,
                                                ncclProfilerEventDescr_t* eDescr) {
  if (context == nullptr || eDescr == nullptr) {
    INFO(NCCL_INIT, "Profiler/Plugin: context/eDescr NULL for start event %s", __func__);
    return ncclSuccess;
  }
  *eHandle = nullptr;
  if (eDescr->type == ncclProfileColl) {
    struct inspectorCollInfo *collEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginCollInfoInit(&collEvent, eDescr, commInfoCtx);
    *eHandle = collEvent;
  } else if (eDescr->type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChEvent = nullptr;
    inspectorPluginKernelChInfoInit(&kernelChEvent, eDescr);
    *eHandle = kernelChEvent;
  } else {
    return ncclSuccess;
  }
  return ncclSuccess;
}

/*
 * Description:
 *
 *   Stops a profiling event for the NCCL Inspector plugin.
 *
 * Thread Safety:
 *
 *   Thread-safe (updates event state and performance info).
 *
 * Input:
 *
 *   void *eHandle - event handle.
 *
 * Output:
 *
 *   Event is stopped and performance info may be updated.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginStopEvent(void *eHandle) {

  if (eHandle == nullptr) {
    INFO(NCCL_INIT,
         "Profiler/Plugin: Event Handle NULL for start event %s", __func__);
    return ncclSuccess;
  }
  uint64_t type = *(uint64_t *)eHandle;
  inspectorResult_t res = inspectorSuccess;

  if (type == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = (struct inspectorCollInfo *)eHandle;
    // Record collective stop event
    inspectorLockWr(&collInfo->guard);
    inspectorRecordEventTrace(collInfo->collEvtTrk.evntTrace,
                              NCCL_INSP_EVT_TRK_COLL_STOP,
                              collInfo);
    res = inspectorPluginCollInfoDeRef(collInfo);
    if (res == inspectorReturn) {
      // WARN("NCCL Inspector unnatural return: inspectorPluginStopEvent:ncclProfileColl");
      return ncclSuccess;
    }
    inspectorUnlockRWLock(&collInfo->guard);
    return ncclSuccess;
  } else if (type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChInfo
      = (struct inspectorKernelChInfo *)eHandle;
    struct inspectorCollInfo *collInfo = kernelChInfo->collInfo;
    if (collInfo && collInfo->type == ncclProfileColl) {
      inspectorLockWr(&collInfo->guard);
      struct inspectorEventTraceInfo *krnlEvtTrk =
        collInfo->collEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
      inspectorRecordEventTrace(krnlEvtTrk,
                                NCCL_INSP_EVT_TRK_KERNEL_STOP,
                                collInfo);
      kernelChInfo->tsCompletedUsec = inspectorGetTime();
      collInfo->nKernelChCompleted += 1;

      res = inspectorPluginCollInfoDeRef(collInfo);
      if (res == inspectorReturn) {
        INFO_INSPECTOR("NCCL Inspector unnatural return: inspectorPluginStopEvent:ncclProfileKernelCh");
        return ncclSuccess;
      }
      if ((collInfo->nKernelChCompleted == collInfo->nKernelChStarted)
          && (collInfo->nKernelChCompleted == collInfo->nChannels)) {
        struct inspectorCompletedCollInfo completedColl;
        struct inspectorCommInfo *commInfo = collInfo->commInfo;
        collInfo->tsCompletedUsec = kernelChInfo->tsCompletedUsec;
        inspectorUpdateCollPerf(&completedColl, collInfo);

        res = inspectorPluginCollInfoDeRef(collInfo);
        if (res != inspectorReturn) {
          inspectorUnlockRWLock(&collInfo->guard);
        }
        if (commInfo != nullptr) {
          inspectorLockWr(&commInfo->guard);
          inspectorComputeCollBw(commInfo,
                                 &completedColl,
                                 completedColl.func);
          memcpy(&commInfo->completedCollInfo,
                 &completedColl,
                 sizeof(struct inspectorCompletedCollInfo));
          commInfo->dump = true;
          inspectorUnlockRWLock(&commInfo->guard);
        }
        return ncclSuccess;
      }
      inspectorUnlockRWLock(&collInfo->guard);
    }
    return ncclSuccess;
  }
  return ncclSuccess;
}

/*
 * Description:
 *
 *   Records the state of a profiling event for the NCCL Inspector
 *   plugin.
 *
 * Thread Safety:
 *
 *   Thread-safe (updates event state as needed).
 *
 * Input:
 *   void* eHandle - event handle.
 *   ncclProfilerEventState_t eState - event state.
 *   ncclProfilerEventStateArgs_t* eStateArgs - event state arguments.
 *
 * Output:
 *   Event state is updated as needed.
 *
 * Return:
 *   ncclResult_t - success or error code.
 *
 */
__hidden ncclResult_t inspectorPluginRecordEventState(void* eHandle,
                                                      ncclProfilerEventState_t eState,
                                                      ncclProfilerEventStateArgs_t* eStateArgs) {
  if (eHandle == nullptr || eStateArgs == nullptr)
    return ncclSuccess;

  uint64_t type = *(uint64_t *)eHandle;

  if (type == ncclProfileKernelCh && eState == ncclProfilerKernelChStop) {
    struct inspectorKernelChInfo *kernelChInfo = (struct inspectorKernelChInfo *)eHandle;
    struct inspectorCollInfo *collInfo = kernelChInfo->collInfo;
    inspectorResult_t res = inspectorSuccess;
    if (collInfo && collInfo->type == ncclProfileColl) {
      inspectorLockWr(&collInfo->guard);
      struct inspectorEventTraceInfo *krnlEvtTrk
        = collInfo->collEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
      inspectorRecordEventTrace(krnlEvtTrk,
                                NCCL_INSP_EVT_TRK_KERNEL_RECORD,
                                collInfo);
      kernelChInfo->stopGpuClk = eStateArgs->kernelCh.pTimer;
      if (kernelChInfo->startGpuClk != 0) {
        res = inspectorPluginCollInfoDeRef(collInfo);
        if (res == inspectorReturn) {
          INFO_INSPECTOR("NCCL Inspector unnatural return: inspectorPluginRecordEventState");
          return ncclSuccess;
        }
      }
      inspectorUnlockRWLock(&collInfo->guard);
    }
  }
  return ncclSuccess;
}

ncclProfiler_t ncclProfiler_v5 = {
  "Inspector",
  inspectorPluginInit,
  inspectorPluginStartEvent,
  inspectorPluginStopEvent,
  inspectorPluginRecordEventState,
  inspectorPluginFinalize,
};
