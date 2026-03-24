/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

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
#include "inspector_ring.h"
#include "inspector_event_pool.h"

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
 *   Records an event trace with timestamp and sequence number for P2P operations
 */
static uint64_t inspectorRecordP2pEventTrace(struct inspectorEventTraceInfo* evtTrace,
                                             int eventIndex,
                                             struct inspectorP2pInfo* p2pInfo) {
  evtTrace[eventIndex].ts = inspectorGetTime();
  evtTrace[eventIndex].sn = ++p2pInfo->p2pEvtTrk.sn;
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
      INFO_INSPECTOR("Inspector Init Failed %s:%d -> error %d: %s",
                     __FILE__, __LINE__, res,
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
  if (enableNcclInspectorP2p) {
    *eActivationMask |= ncclProfileP2p;
  }

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
    inspectorEventPoolReleaseColl(collInfo);
    return inspectorReturn;
  }
  return inspectorSuccess;
}

static void inspectorUpdateCommOpInfo(struct inspectorCommInfo *commInfo,
                                      struct inspectorCompletedOpInfo *completedOp) {
  struct inspectorCompletedRing *ring =
    completedOp->isP2p ? &commInfo->completedP2pRing : &commInfo->completedCollRing;
  inspectorLockWr(&commInfo->guard);
  inspectorComputeOpBw(commInfo, completedOp);
  inspectorRingEnqueue(ring, completedOp);
  if (completedOp->isP2p) {
    commInfo->dump_p2p = inspectorRingNonEmpty(&commInfo->completedP2pRing);
  } else {
    commInfo->dump_coll = inspectorRingNonEmpty(&commInfo->completedCollRing);
  }
  inspectorUnlockRWLock(&commInfo->guard);
}

inspectorResult_t inspectorPluginP2pInfoRef(struct inspectorP2pInfo *p2pInfo) {
  p2pInfo->refCount += 1;
  return inspectorSuccess;
}

inspectorResult_t inspectorPluginP2pInfoDeRef(struct inspectorP2pInfo *p2pInfo) {
  p2pInfo->refCount -= 1;
  if (p2pInfo->refCount == 0) {
    inspectorLockDestroy(&p2pInfo->guard);
    inspectorEventPoolReleaseP2p(p2pInfo);
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
  struct inspectorCollInfo *collInfoPtr = inspectorEventPoolAllocColl();
  if (collInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for collective info structure");
    *collInfo = nullptr;
    return;
  }
  collInfoPtr->type = ncclProfileColl;
  collInfoPtr->refCount = 0;
  inspectorPluginCollInfoRef(collInfoPtr); //self ref; no locks needed
  collInfoPtr->func = eDescr->coll.func;
  collInfoPtr->algo = eDescr->coll.algo;
  collInfoPtr->proto = eDescr->coll.proto;
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
                            NCCL_INSP_EVT_TRK_OP_START, collInfoPtr);

  inspectorLockInit(&collInfoPtr->guard);
  *collInfo = collInfoPtr;
}

static void inspectorPluginP2pInfoInit(struct inspectorP2pInfo **p2pInfo,
                                       ncclProfilerEventDescr_t *eDescr,
                                       struct inspectorCommInfo *commInfo) {
  struct inspectorP2pInfo *p2pInfoPtr = inspectorEventPoolAllocP2p();
  if (p2pInfoPtr == nullptr) {
    INFO_INSPECTOR("Inspector: Failed to allocate memory for P2P info structure");
    *p2pInfo = nullptr;
    return;
  }
  p2pInfoPtr->type = ncclProfileP2p;
  p2pInfoPtr->refCount = 0;
  inspectorPluginP2pInfoRef(p2pInfoPtr); // self ref
  p2pInfoPtr->func = eDescr->p2p.func;
  p2pInfoPtr->nChannels = eDescr->p2p.nChannels;
  p2pInfoPtr->peer = eDescr->p2p.peer;
  if (p2pInfoPtr->nChannels > 0) {
    inspectorPluginP2pInfoRef(p2pInfoPtr); // extra ref for kernel completion
  }
  p2pInfoPtr->tsStartUsec = inspectorGetTime();
  p2pInfoPtr->msgSizeBytes =
    ncclTypeSize(inspectorStringToDatatype(eDescr->p2p.datatype)) * eDescr->p2p.count;

  p2pInfoPtr->commInfo = commInfo;
  p2pInfoPtr->sn = __atomic_add_fetch(&commInfo->p2pSeqNum, 1, __ATOMIC_RELAXED);
  p2pInfoPtr->p2pEvtTrk.nChannels = p2pInfoPtr->nChannels;
  p2pInfoPtr->p2pEvtTrk.sn = p2pInfoPtr->sn;
  inspectorRecordP2pEventTrace(p2pInfoPtr->p2pEvtTrk.evntTrace,
                               NCCL_INSP_EVT_TRK_OP_START, p2pInfoPtr);

  inspectorLockInit(&p2pInfoPtr->guard);
  *p2pInfo = p2pInfoPtr;
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
static struct inspectorCollInfo* getKernelChCollInfo(struct inspectorKernelChInfo *kernelChInfo) {
  if (kernelChInfo && kernelChInfo->parentType == ncclProfileColl) {
    return (struct inspectorCollInfo*)kernelChInfo->parentObj;
  }
  return nullptr;
}

static struct inspectorP2pInfo* getKernelChP2pInfo(struct inspectorKernelChInfo *kernelChInfo) {
  if (kernelChInfo && kernelChInfo->parentType == ncclProfileP2p) {
    return (struct inspectorP2pInfo*)kernelChInfo->parentObj;
  }
  return nullptr;
}

static void inspectorPluginKernelChInfoInitColl(struct inspectorKernelChInfo **kernelChInfo,
                                                ncclProfilerEventDescr_t *eDescr,
                                                struct inspectorCollInfo *collInfo) {
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
  kernelChInfoPtr->parentType = ncclProfileColl;
  kernelChInfoPtr->parentObj = collInfo;
  if (kernelChInfoPtr->stopGpuClk == 0) {
    inspectorPluginCollInfoRef(collInfo); //Pairs with Record Kernel Stop event
  }
  kernelChInfoPtr->tsStartUsec = inspectorGetTime();
  if (collInfo->nKernelChStarted == 0) {
    collInfo->tsStartUsec = kernelChInfoPtr->tsStartUsec;
  }
  collInfo->nKernelChStarted += 1;
  inspectorPluginCollInfoRef(collInfo); //Pairs with Stop Kernel Event

  *kernelChInfo = kernelChInfoPtr;
  inspectorUnlockRWLock(&collInfo->guard);
}

static void inspectorPluginKernelChInfoInitP2p(struct inspectorKernelChInfo **kernelChInfo,
                                               ncclProfilerEventDescr_t *eDescr,
                                               struct inspectorP2pInfo *p2pInfo) {
  inspectorLockWr(&p2pInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk =
    p2pInfo->p2pEvtTrk.kernelCh[eDescr->kernelCh.channelId].evntTrace;
  inspectorRecordP2pEventTrace(krnlEvtTrk,
                               NCCL_INSP_EVT_TRK_KERNEL_START,
                               p2pInfo);
  struct inspectorKernelChInfo *kernelChInfoPtr
    = &p2pInfo->kernelCh[eDescr->kernelCh.channelId];
  kernelChInfoPtr->type = ncclProfileKernelCh;
  kernelChInfoPtr->channelId = eDescr->kernelCh.channelId;
  kernelChInfoPtr->startGpuClk = eDescr->kernelCh.pTimer;
  kernelChInfoPtr->parentType = ncclProfileP2p;
  kernelChInfoPtr->parentObj = p2pInfo;
  if (kernelChInfoPtr->stopGpuClk == 0) {
    inspectorPluginP2pInfoRef(p2pInfo); //Pairs with Record Kernel Stop event
  }
  kernelChInfoPtr->tsStartUsec = inspectorGetTime();
  if (p2pInfo->nKernelChStarted == 0) {
    p2pInfo->tsStartUsec = kernelChInfoPtr->tsStartUsec;
  }
  p2pInfo->nKernelChStarted += 1;
  inspectorPluginP2pInfoRef(p2pInfo); //Pairs with Stop Kernel Event

  *kernelChInfo = kernelChInfoPtr;
  inspectorUnlockRWLock(&p2pInfo->guard);
}

static void inspectorPluginKernelChInfoInit(struct inspectorKernelChInfo **kernelChInfo,
                                            ncclProfilerEventDescr_t *eDescr) {
  if (eDescr->parentObj) {
    uint64_t parentType = *(uint64_t*)eDescr->parentObj;
    if (parentType == ncclProfileColl) {
      struct inspectorCollInfo *collInfo = (struct inspectorCollInfo*)eDescr->parentObj;
      if (collInfo && collInfo->type == ncclProfileColl) {
        inspectorPluginKernelChInfoInitColl(kernelChInfo, eDescr, collInfo);
      }
    } else if (parentType == ncclProfileP2p) {
      struct inspectorP2pInfo *p2pInfo = (struct inspectorP2pInfo*)eDescr->parentObj;
      if (p2pInfo && p2pInfo->type == ncclProfileP2p) {
        inspectorPluginKernelChInfoInitP2p(kernelChInfo, eDescr, p2pInfo);
      }
    }
  }
}

static bool inspectorShouldTrackColl(const ncclProfilerEventDescr_t* eDescr) {
  if (!eDescr) {
    return false;
  }
  int typeSize = ncclTypeSize(inspectorStringToDatatype(eDescr->coll.datatype));
  if (typeSize <= 0) {
    return true;
  }
  if (eDescr->coll.count == 0) {
    return false;
  }
  if (eDescr->coll.count > (SIZE_MAX / (size_t)typeSize)) {
    return true;
  }
  size_t msgSizeBytes = (size_t)typeSize * eDescr->coll.count;
  return msgSizeBytes >= ncclInspectorDumpMinSizeBytes;
}

static bool inspectorShouldTrackP2p(const ncclProfilerEventDescr_t* eDescr) {
  if (!eDescr) {
    return false;
  }
  int typeSize = ncclTypeSize(inspectorStringToDatatype(eDescr->p2p.datatype));
  if (typeSize <= 0) {
    return true;
  }
  if (eDescr->p2p.count == 0) {
    return false;
  }
  if (eDescr->p2p.count > (SIZE_MAX / (size_t)typeSize)) {
    return true;
  }
  size_t msgSizeBytes = (size_t)typeSize * eDescr->p2p.count;
  return msgSizeBytes >= ncclInspectorDumpMinSizeBytes;
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
    if (!inspectorShouldTrackColl(eDescr)) return ncclSuccess;
    struct inspectorCollInfo *collEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginCollInfoInit(&collEvent, eDescr, commInfoCtx);
    *eHandle = collEvent;
  } else if (eDescr->type == ncclProfileP2p) {
    if (!enableNcclInspectorP2p) return ncclSuccess;
    if (!inspectorShouldTrackP2p(eDescr)) return ncclSuccess;
    struct inspectorP2pInfo *p2pEvent = nullptr;
    struct inspectorCommInfo *commInfoCtx = (struct inspectorCommInfo*)context;
    inspectorPluginP2pInfoInit(&p2pEvent, eDescr, commInfoCtx);
    *eHandle = p2pEvent;
  } else if (eDescr->type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChEvent = nullptr;
    inspectorPluginKernelChInfoInit(&kernelChEvent, eDescr);
    *eHandle = kernelChEvent;
  } else {
    return ncclSuccess;
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventColl(struct inspectorCollInfo *collInfo) {
  inspectorLockWr(&collInfo->guard);
  inspectorRecordEventTrace(collInfo->collEvtTrk.evntTrace,
                            NCCL_INSP_EVT_TRK_OP_STOP,
                            collInfo);
  inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
  if (res == inspectorReturn) {
    return ncclSuccess;
  }
  inspectorUnlockRWLock(&collInfo->guard);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventP2p(struct inspectorP2pInfo *p2pInfo) {
  inspectorLockWr(&p2pInfo->guard);
  inspectorRecordP2pEventTrace(p2pInfo->p2pEvtTrk.evntTrace,
                               NCCL_INSP_EVT_TRK_OP_STOP,
                               p2pInfo);
  inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
  if (res == inspectorReturn) {
    return ncclSuccess;
  }
  inspectorUnlockRWLock(&p2pInfo->guard);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventKernelChColl(struct inspectorKernelChInfo *kernelChInfo,
                                                         struct inspectorCollInfo *collInfo) {
  struct inspectorCompletedOpInfo completedOp;

  inspectorLockWr(&collInfo->guard);
  struct inspectorCommInfo *commInfo = collInfo->commInfo;
  struct inspectorEventTraceInfo *krnlEvtTrk =
    collInfo->collEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordEventTrace(krnlEvtTrk,
                            NCCL_INSP_EVT_TRK_KERNEL_STOP,
                            collInfo);
  kernelChInfo->tsCompletedUsec = inspectorGetTime();
  collInfo->nKernelChCompleted += 1;

  inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
  if (res == inspectorReturn) {
    return ncclSuccess;
  }

  if ((collInfo->nKernelChCompleted == collInfo->nKernelChStarted)
      && (collInfo->nKernelChCompleted == collInfo->nChannels)) {

    collInfo->tsCompletedUsec = kernelChInfo->tsCompletedUsec;
    inspectorUpdateCollPerf(&completedOp, collInfo);

    // Discard if GPU-based kernel timing is not available and kernel timing is required.
    if (requireKernelTiming &&
        completedOp.timingSource != inspectorTimingSourceKernelGpu) {
      res = inspectorPluginCollInfoDeRef(collInfo);
      if (res != inspectorReturn) {
        inspectorUnlockRWLock(&collInfo->guard);
      }
      return ncclSuccess;
    }

    res = inspectorPluginCollInfoDeRef(collInfo);
    if (res != inspectorReturn) {
      inspectorUnlockRWLock(&collInfo->guard);
    }
    if (commInfo != nullptr) {
      inspectorUpdateCommOpInfo(commInfo, &completedOp);
    }
    return ncclSuccess;
  }
  inspectorUnlockRWLock(&collInfo->guard);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventKernelChP2p(struct inspectorKernelChInfo *kernelChInfo,
                                                        struct inspectorP2pInfo *p2pInfo) {
  struct inspectorCompletedOpInfo completedOp;

  inspectorLockWr(&p2pInfo->guard);
  struct inspectorCommInfo *commInfo = p2pInfo->commInfo;
  struct inspectorEventTraceInfo *krnlEvtTrk =
    p2pInfo->p2pEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordP2pEventTrace(krnlEvtTrk,
                               NCCL_INSP_EVT_TRK_KERNEL_STOP,
                               p2pInfo);
  kernelChInfo->tsCompletedUsec = inspectorGetTime();
  p2pInfo->nKernelChCompleted += 1;

  inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
  if (res == inspectorReturn) {
    return ncclSuccess;
  }

  if ((p2pInfo->nKernelChCompleted == p2pInfo->nKernelChStarted)
      && (p2pInfo->nKernelChCompleted == p2pInfo->nChannels)) {

    p2pInfo->tsCompletedUsec = kernelChInfo->tsCompletedUsec;
    inspectorUpdateP2pPerf(&completedOp, p2pInfo);

    // Discard if GPU-based kernel timing is not available and kernel timing is required.
    if (requireKernelTiming &&
        completedOp.timingSource != inspectorTimingSourceKernelGpu) {
      res = inspectorPluginP2pInfoDeRef(p2pInfo);
      if (res != inspectorReturn) {
        inspectorUnlockRWLock(&p2pInfo->guard);
      }
      return ncclSuccess;
    }

    res = inspectorPluginP2pInfoDeRef(p2pInfo);
    if (res != inspectorReturn) {
      inspectorUnlockRWLock(&p2pInfo->guard);
    }
    if (commInfo != nullptr) {
      inspectorUpdateCommOpInfo(commInfo, &completedOp);
    }
    return ncclSuccess;
  }
  inspectorUnlockRWLock(&p2pInfo->guard);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginStopEventKernelCh(struct inspectorKernelChInfo *kernelChInfo) {
  if (kernelChInfo->parentType == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = getKernelChCollInfo(kernelChInfo);
    if (collInfo) return inspectorPluginStopEventKernelChColl(kernelChInfo, collInfo);
  } else if (kernelChInfo->parentType == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = getKernelChP2pInfo(kernelChInfo);
    if (p2pInfo) return inspectorPluginStopEventKernelChP2p(kernelChInfo, p2pInfo);
  }
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelChColl(struct inspectorKernelChInfo *kernelChInfo,
                                                                struct inspectorCollInfo *collInfo,
                                                                ncclProfilerEventStateArgs_t* eStateArgs) {
  inspectorLockWr(&collInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk
    = collInfo->collEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordEventTrace(krnlEvtTrk,
                            NCCL_INSP_EVT_TRK_KERNEL_RECORD,
                            collInfo);
  kernelChInfo->stopGpuClk = eStateArgs->kernelCh.pTimer;
  if (kernelChInfo->startGpuClk != 0) {
    inspectorResult_t res = inspectorPluginCollInfoDeRef(collInfo);
    if (res == inspectorReturn) {
      return ncclSuccess;
    }
  }
  inspectorUnlockRWLock(&collInfo->guard);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelChP2p(struct inspectorKernelChInfo *kernelChInfo,
                                                               struct inspectorP2pInfo *p2pInfo,
                                                               ncclProfilerEventStateArgs_t* eStateArgs) {
  inspectorLockWr(&p2pInfo->guard);
  struct inspectorEventTraceInfo *krnlEvtTrk
    = p2pInfo->p2pEvtTrk.kernelCh[kernelChInfo->channelId].evntTrace;
  inspectorRecordP2pEventTrace(krnlEvtTrk,
                               NCCL_INSP_EVT_TRK_KERNEL_RECORD,
                               p2pInfo);
  kernelChInfo->stopGpuClk = eStateArgs->kernelCh.pTimer;
  if (kernelChInfo->startGpuClk != 0) {
    inspectorResult_t res = inspectorPluginP2pInfoDeRef(p2pInfo);
    if (res == inspectorReturn) {
      return ncclSuccess;
    }
  }
  inspectorUnlockRWLock(&p2pInfo->guard);
  return ncclSuccess;
}

static ncclResult_t inspectorPluginRecordEventStateKernelCh(struct inspectorKernelChInfo *kernelChInfo,
                                                            ncclProfilerEventStateArgs_t* eStateArgs) {
  if (kernelChInfo->parentType == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = getKernelChCollInfo(kernelChInfo);
    if (collInfo) return inspectorPluginRecordEventStateKernelChColl(kernelChInfo, collInfo, eStateArgs);
  } else if (kernelChInfo->parentType == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = getKernelChP2pInfo(kernelChInfo);
    if (p2pInfo) return inspectorPluginRecordEventStateKernelChP2p(kernelChInfo, p2pInfo, eStateArgs);
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
  if (type == ncclProfileColl) {
    struct inspectorCollInfo *collInfo = (struct inspectorCollInfo *)eHandle;
    return inspectorPluginStopEventColl(collInfo);
  } else if (type == ncclProfileP2p) {
    struct inspectorP2pInfo *p2pInfo = (struct inspectorP2pInfo *)eHandle;
    return inspectorPluginStopEventP2p(p2pInfo);
  } else if (type == ncclProfileKernelCh) {
    struct inspectorKernelChInfo *kernelChInfo
      = (struct inspectorKernelChInfo *)eHandle;
    return inspectorPluginStopEventKernelCh(kernelChInfo);
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

    struct inspectorKernelChInfo *kernelChInfo
      = (struct inspectorKernelChInfo *)eHandle;

    return inspectorPluginRecordEventStateKernelCh(kernelChInfo,
                                                   eStateArgs);

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
