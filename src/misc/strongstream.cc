/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "strongstream.h"
#include "cudawrap.h"
#include "checks.h"
#include "param.h"

// Tracks the captured work a given graph captured identified by its graph id.
struct ncclStrongStreamCapture {
  struct ncclStrongStreamCapture* next;
  cudaGraph_t graph;
  unsigned long long graphId;
  cudaStream_t captureStream;
  cudaGraphNode_t lastRecord;
  void* acquiredBy;
};

////////////////////////////////////////////////////////////////////////////////

static ncclCudaContext* cxtListHead = nullptr;
static pthread_mutex_t cxtListLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t ncclCudaContextTrack(struct ncclCudaContext** out) {
  ncclResult_t result = ncclSuccess;
  CUcontext hcontext;
  CUCHECK(cuCtxGetCurrent(&hcontext));

  pthread_mutex_lock(&cxtListLock);
  struct ncclCudaContext* p = cxtListHead;
  while (1) {
    if (p == nullptr) {
      p = (struct ncclCudaContext*)calloc(1, sizeof(struct ncclCudaContext));
      p->refCount = 1;
      p->hcontext = hcontext;
      p->next = cxtListHead;
      cxtListHead = p;
      NCCLCHECKGOTO(ncclStrongStreamConstruct(&p->launchOrder), result, leave);
      break;
    }
    if (p->hcontext == hcontext) {
      p->refCount += 1;
      break;
    }
    p = p->next;
  }
leave:
  pthread_mutex_unlock(&cxtListLock);
  *out = p;
  return ncclSuccess;
}

void ncclCudaContextDrop(struct ncclCudaContext* cxt) {
  pthread_mutex_lock(&cxtListLock);
  if (0 == --cxt->refCount) {
    struct ncclCudaContext** pp = &cxtListHead;
    while (*pp != cxt) pp = &(*pp)->next;
    *pp = cxt->next; // remove from list
    // Destroy resources held in cxt
    ncclStrongStreamDestruct(&cxt->launchOrder);
    free(cxt);
  }
  pthread_mutex_unlock(&cxtListLock);
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, cudaStream_t stream
  ) {
  #if CUDART_VERSION >= 10000 // cudaStreamGetCaptureInfo
    int driver;
    NCCLCHECK(ncclCudaDriverVersion(&driver));
    if (CUDART_VERSION < 11030 || driver < 11030) {
      cudaStreamCaptureStatus status;
      CUDACHECK(cudaStreamGetCaptureInfo(stream, &status, nullptr));
      #if CUDART_VERSION >= 11030
        graph->origin = nullptr;
        graph->graph = nullptr;
        graph->graphId = ULLONG_MAX;
      #endif
      if (status != cudaStreamCaptureStatusNone) {
        WARN("NCCL cannot be captured in a graph if either it wasn't built with CUDA runtime >= 11.3 or if the installed CUDA driver < R465.");
        return ncclInvalidUsage;
      }
    } else {
      #if CUDART_VERSION >= 11030
        cudaStreamCaptureStatus status;
        CUDACHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &graph->graphId, &graph->graph, nullptr, nullptr));
        if (status != cudaStreamCaptureStatusActive) {
          graph->origin = nullptr;
          graph->graph = nullptr;
          graph->graphId = ULLONG_MAX;
        } else {
          graph->origin = stream;
        }
      #endif
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg) {
  #if CUDART_VERSION >= 11030
    cudaUserObject_t object;
    CUDACHECK(cudaUserObjectCreate(
      &object, arg, fn, /*initialRefcount=*/1, cudaUserObjectNoDestructorSync
    ));
    // Hand over ownership to CUDA Graph
    CUDACHECK(cudaGraphRetainUserObject(graph.graph, object, 1, cudaGraphUserObjectMove));
    return ncclSuccess;
  #else
    return ncclInvalidUsage;
  #endif
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss) {
  CUDACHECK(cudaStreamCreateWithFlags(&ss->liveStream, cudaStreamNonBlocking));
  #if CUDART_VERSION >= 11030
    ss->everCaptured = false;
    ss->captureHead = nullptr;
    pthread_mutex_init(&ss->lock, nullptr);
    CUDACHECK(cudaEventCreateWithFlags(&ss->serialEvent, cudaEventDisableTiming));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) {
  CUDACHECK(cudaStreamDestroy(ss->liveStream));
  #if CUDART_VERSION >= 11030
    struct ncclStrongStreamCapture* cap = ss->captureHead;
    while (cap) {
      struct ncclStrongStreamCapture* next = cap->next;
      CUDACHECK(cudaStreamDestroy(cap->captureStream));
      free(cap);
      cap = next;
    }
    CUDACHECK(cudaEventDestroy(ss->serialEvent));
    pthread_mutex_destroy(&ss->lock);
  #endif
  return ncclSuccess;
}

NCCL_PARAM(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", 1)
NCCL_PARAM(LaunchRaceFatal, "LAUNCH_RACE_FATAL", 1);
constexpr char const* launchRaceFatalMsg = "Fatal: host threads racing to launch NCCL on same device.";

static __thread char threadIdMarker;
static void* localThreadId() { return &threadIdMarker; }

ncclResult_t ncclStrongStreamAcquire(
   struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent,
   cudaStream_t* workStream
  ) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (graph.graphId == ULLONG_MAX) {
      *workStream = ss->liveStream;
      ss->liveAcquiredBy = localThreadId();
      if (mixing && __atomic_load_n(&ss->everCaptured, __ATOMIC_RELAXED)) {
        CUDACHECK(cudaStreamWaitEvent(ss->liveStream, ss->serialEvent, 0));
      }
    } else {
      bool firstCapture = !ss->everCaptured;
      __atomic_store_n(&ss->everCaptured, true, __ATOMIC_RELAXED);

      ncclResult_t ret = ncclSuccess;
      if (concurrent) pthread_mutex_lock(&ss->lock);

      // Look for capture in our list of active captures.
      struct ncclStrongStreamCapture** pcap = &ss->captureHead;
      struct ncclStrongStreamCapture* cap;
      struct ncclStrongStreamCapture* spare = nullptr;
      while (*pcap != nullptr) {
        cap = *pcap;
        if (cap->graphId == graph.graphId) { // Capture node already exists.
          *workStream = cap->captureStream;
          cap->acquiredBy = localThreadId();
          if (concurrent) pthread_mutex_unlock(&ss->lock);
          return ncclSuccess;
        } else {
          cudaStreamCaptureStatus status;
          CUDACHECKGOTO(cudaStreamIsCapturing(cap->captureStream, &status), ret, do_unlock);
          if (status == cudaStreamCaptureStatusActive) {
            pcap = &cap->next; // Active capture doesn't match, on to next.
          } else { // Capture no longer active
            *pcap = cap->next; // Remove from current list
            if (spare == nullptr) { // Keep one spare to reuse below.
              spare = cap;
            } else {
              cudaStreamDestroy(cap->captureStream);
              free(cap);
            }
          }
        }
      }
      // No matching capture, need a new entry.
      cap = spare;
      if (cap == nullptr) {
        cap = (struct ncclStrongStreamCapture*)calloc(1, sizeof(struct ncclStrongStreamCapture));
        CUDACHECKGOTO(cudaStreamCreateWithFlags(&cap->captureStream, cudaStreamNonBlocking), ret, do_unlock);
      }
      cap->graphId = graph.graphId;
      cap->lastRecord = nullptr;
      cap->acquiredBy = localThreadId();
      // Push to capturing list.
      cap->next = ss->captureHead;
      ss->captureHead = cap;

    do_unlock:
      if (concurrent) pthread_mutex_unlock(&ss->lock);
      if (ret != ncclSuccess) return ret;

      *workStream = cap->captureStream;

      // Bring captureStream into the graph but without any dependencies.
      cudaEvent_t scratch;
      CUDACHECK(cudaEventCreateWithFlags(&scratch, cudaEventDisableTiming));
      CUDACHECK(cudaEventRecord(scratch, graph.origin));
      CUDACHECK(cudaStreamWaitEvent(cap->captureStream, scratch, 0));
      CUDACHECK(cudaEventDestroy(scratch));
      CUDACHECK(cudaStreamUpdateCaptureDependencies(cap->captureStream, nullptr, 0, cudaStreamSetCaptureDependencies));

      if (mixing && firstCapture) {
        CUDACHECK(cudaEventRecord(ss->serialEvent, ss->liveStream));
      }
      if (mixing) {
        // First dependency is to wait on serialEvent
        CUDACHECK(cudaStreamWaitEvent(cap->captureStream, ss->serialEvent, cudaEventWaitExternal));
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamAcquiredWorkStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent,
    cudaStream_t* workStream
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graphId == ULLONG_MAX) {
      *workStream = ss->liveStream;
    } else {
      if (concurrent) pthread_mutex_lock(&ss->lock);
      struct ncclStrongStreamCapture* cap = ss->captureHead;
      while (cap->graphId != graph.graphId) cap = cap->next;
      *workStream = cap->captureStream;
      if (concurrent) pthread_mutex_unlock(&ss->lock);
    }
  #else
    *workStream = ss->liveStream
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamRelease(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent
  ) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (mixing) {
      if (graph.graphId == ULLONG_MAX) {
        if (__atomic_load_n(&ss->everCaptured, __ATOMIC_RELAXED)) {
          CUDACHECK(cudaEventRecord(ss->serialEvent, ss->liveStream));
        }
        if (ss->liveAcquiredBy != localThreadId() && ncclParamLaunchRaceFatal()) {
          WARN("%s", launchRaceFatalMsg);
          return ncclInvalidUsage;
        }
      } else {
        if (concurrent) pthread_mutex_lock(&ss->lock);
        struct ncclStrongStreamCapture* cap = ss->captureHead;
        while (cap->graphId != graph.graphId) cap = cap->next;
        if (concurrent) pthread_mutex_unlock(&ss->lock);

        // Add event record node with dependencies added further down.
        cudaGraphNode_t recordNode;
        CUDACHECK(cudaGraphAddEventRecordNode(&recordNode, graph.graph, nullptr, 0, ss->serialEvent));

        // Make this record order after previous record on this stream.
        if (cap->lastRecord != nullptr) {
          CUDACHECK(cudaGraphAddDependencies(graph.graph, &cap->lastRecord, &recordNode, 1));
        }
        cap->lastRecord = recordNode;

        // Get current nodes from work stream so we can add them as dependencies.
        cudaStreamCaptureStatus status;
        cudaGraphNode_t const* nodes;
        size_t count = 0;
        cudaError_t res = cudaStreamGetCaptureInfo_v2(cap->captureStream, &status, nullptr, nullptr, &nodes, &count);

        #if CUDART_VERSION >= 12030
        if (res == cudaErrorLossyQuery) { // CUDA is telling us the dependencies have edge annotations.
          cudaGraphEdgeData const* edges;
          CUDACHECK(cudaStreamGetCaptureInfo_v3(cap->captureStream, &status, nullptr, nullptr, &nodes, &edges, &count));
          for (int i=0; i < (int)count; i++) {
            CUDACHECK(cudaGraphAddDependencies_v2(graph.graph, &nodes[i], &recordNode, &edges[i], 1));
          }
        }
        #else
        if (false) {}
        #endif
        else {
          CUDACHECK(res /* = cudaStreamGetCaptureInfo_v2(...)*/);
          for (int i=0; i < (int)count; i++) {
            CUDACHECK(cudaGraphAddDependencies(graph.graph, &nodes[i], &recordNode, 1));
          }
        }

        if (cap->acquiredBy != localThreadId() && ncclParamLaunchRaceFatal()) {
          WARN("%s", launchRaceFatalMsg);
          return ncclInvalidUsage;
        }
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStreamWaitStream(cudaStream_t a, cudaStream_t b, cudaEvent_t scratchEvent) {
  CUDACHECK(cudaEventRecord(scratchEvent, b));
  CUDACHECK(cudaStreamWaitEvent(a, scratchEvent, 0));
  return ncclSuccess;
}

ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph g, cudaStream_t s, cudaEvent_t e) {
  if (g.graphId == ULLONG_MAX) {
    CUDACHECK(cudaStreamWaitEvent(s, e, 0));
  } else {
    cudaStream_t tmp;
    CUDACHECK(cudaStreamCreateWithFlags(&tmp, cudaStreamNonBlocking));
    CUDACHECK(cudaStreamWaitEvent(tmp, e, 0));

    cudaStreamCaptureStatus status;
    cudaGraphNode_t const* nodes;
    size_t count = 0;
    cudaError_t res = cudaStreamGetCaptureInfo_v2(tmp, &status, nullptr, nullptr, &nodes, &count);

    #if CUDART_VERSION >= 12030
    if (res == cudaErrorLossyQuery) { // CUDA is telling us the dependencies have edge annotations.
      cudaGraphEdgeData const* edges;
      CUDACHECK(cudaStreamGetCaptureInfo_v3(tmp, &status, nullptr, nullptr, &nodes, &edges, &count));
      CUDACHECK(cudaStreamUpdateCaptureDependencies_v2(s, (cudaGraphNode_t*)nodes, edges, count, cudaStreamSetCaptureDependencies));
    }
    #else
    if (false) {}
    #endif
    else {
      CUDACHECK(res /* = cudaStreamGetCaptureInfo_v2(...)*/);
      CUDACHECK(cudaStreamUpdateCaptureDependencies(s, (cudaGraphNode_t*)nodes, count, cudaStreamSetCaptureDependencies));
    }

    CUDACHECK(cudaStreamDestroy(tmp));
  }
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaStreamWaitEvent(ss->liveStream, ss->serialEvent, 0));
  #endif
  CUDACHECK(cudaStreamSynchronize(ss->liveStream));
  return ncclSuccess;
}
