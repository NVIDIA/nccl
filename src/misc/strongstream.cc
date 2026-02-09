/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "strongstream.h"
#include "cudawrap.h"
#include "checks.h"
#include "param.h"
#include <mutex>
#include <memory>

#if CUDART_VERSION >= 13000
#define cudaStreamGetCaptureInfo_v3 cudaStreamGetCaptureInfo
#define cudaGraphAddDependencies_v2 cudaGraphAddDependencies
#define cudaStreamUpdateCaptureDependencies_v2 cudaStreamUpdateCaptureDependencies
#endif

// Tracks the captured work a given graph captured identified by its graph id.
struct ncclStrongStreamCapture {
  struct ncclStrongStreamCapture* next;
  cudaGraph_t graph;
  unsigned long long graphId;
  cudaStream_t captureStream;
  void* acquiredBy;
};

////////////////////////////////////////////////////////////////////////////////

static ncclCudaContext* cxtListHead = nullptr;
static std::mutex cxtListMutex;

ncclResult_t ncclCudaContextTrack(struct ncclCudaContext** out) {
  ncclResult_t result = ncclSuccess;
  CUcontext hcontext;
  CUCHECK(cuCtxGetCurrent(&hcontext));

  std::lock_guard<std::mutex> lock(cxtListMutex);
  struct ncclCudaContext* p = cxtListHead;
  while (1) {
    if (p == nullptr) {
      p = new ncclCudaContext{};
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
  *out = p;
  return ncclSuccess;
}

void ncclCudaContextDrop(struct ncclCudaContext* cxt) {
  std::lock_guard<std::mutex> lock(cxtListMutex);
  if (0 == --cxt->refCount) {
    struct ncclCudaContext** pp = &cxtListHead;
    while (*pp != cxt) pp = &(*pp)->next;
    *pp = cxt->next; // remove from list
    // Destroy resources held in cxt
    ncclStrongStreamDestruct(&cxt->launchOrder);
    delete cxt;
  }
}

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, cudaStream_t stream, int graphUsageMode
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
        graph->graphUsageMode = graphUsageMode;
      #endif
      if (status != cudaStreamCaptureStatusNone) {
        WARN("NCCL cannot be captured in a graph if either it wasn't built with CUDA runtime >= 11.3 or if the installed CUDA driver < R465.");
        return ncclInvalidUsage;
      }
    } else {
      #if CUDART_VERSION >= 11030
        cudaStreamCaptureStatus status;
      #if CUDART_VERSION >= 13000
        CUDACHECK(cudaStreamGetCaptureInfo_v3(stream, &status, &graph->graphId, &graph->graph, nullptr, nullptr, nullptr));
      #else
        CUDACHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &graph->graphId, &graph->graph, nullptr, nullptr));
      #endif
        if (status != cudaStreamCaptureStatusActive) {
          graph->origin = nullptr;
          graph->graph = nullptr;
          graph->graphId = ULLONG_MAX;
        } else {
          graph->origin = stream;
        }
        graph->graphUsageMode = graphUsageMode;
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
  #endif
  return ncclSuccess;
}

NCCL_PARAM(LaunchRaceFatal, "LAUNCH_RACE_FATAL", 1);
constexpr char const* launchRaceFatalMsg = "Fatal: host threads racing to launch NCCL on same device.";

static thread_local char threadIdMarker;
static void* localThreadId() { return &threadIdMarker; }

ncclResult_t ncclStrongStreamAcquire(
   struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent,
   cudaStream_t* workStream
  ) {
  #if CUDART_VERSION >= 11030
    bool mixing = graph.graphUsageMode == 2;
    if (graph.graphId == ULLONG_MAX) {
      *workStream = ss->liveStream;
      ss->liveAcquiredBy = localThreadId();
      if (mixing && COMPILER_ATOMIC_LOAD(&ss->everCaptured, std::memory_order_relaxed)) {
        CUDACHECK(cudaStreamWaitEvent(ss->liveStream, ss->serialEvent, 0));
      }
    } else {
      bool firstCapture = !ss->everCaptured;
      COMPILER_ATOMIC_STORE(&ss->everCaptured, true, std::memory_order_relaxed);

      ncclResult_t ret = ncclSuccess;
      std::unique_lock<std::mutex> lock(ss->mutex, std::defer_lock);
      if (concurrent) lock.lock();

      // Look for capture in our list of active captures.
      struct ncclStrongStreamCapture** pcap = &ss->captureHead;
      struct ncclStrongStreamCapture* cap;
      struct ncclStrongStreamCapture* spare = nullptr;
      while (*pcap != nullptr) {
        cap = *pcap;
        if (cap->graphId == graph.graphId) { // Capture node already exists.
          *workStream = cap->captureStream;
          cap->acquiredBy = localThreadId();
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
      cap->acquiredBy = localThreadId();
      // Push to capturing list.
      cap->next = ss->captureHead;
      ss->captureHead = cap;

    do_unlock:
      if (concurrent) lock.unlock();
      if (ret != ncclSuccess) return ret;

      *workStream = cap->captureStream;

      // Bring captureStream into the graph but without any dependencies.
      cudaEvent_t scratch;
      CUDACHECK(cudaEventCreateWithFlags(&scratch, cudaEventDisableTiming));
      CUDACHECK(cudaEventRecord(scratch, graph.origin));
      CUDACHECK(cudaStreamWaitEvent(cap->captureStream, scratch, 0));
      CUDACHECK(cudaEventDestroy(scratch));
      #if CUDART_VERSION >= 13000
      CUDACHECK(cudaStreamUpdateCaptureDependencies_v2(cap->captureStream, nullptr, nullptr, 0, cudaStreamSetCaptureDependencies));
      #else
      CUDACHECK(cudaStreamUpdateCaptureDependencies(cap->captureStream, nullptr, 0, cudaStreamSetCaptureDependencies));
      #endif

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
      std::unique_lock<std::mutex> lock(ss->mutex, std::defer_lock);
      if (concurrent) lock.lock();
      struct ncclStrongStreamCapture* cap = ss->captureHead;
      while (cap->graphId != graph.graphId) cap = cap->next;
      *workStream = cap->captureStream;
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
    bool mixing = graph.graphUsageMode == 2;
    if (mixing) {
      if (graph.graphId == ULLONG_MAX) {
        if (COMPILER_ATOMIC_LOAD(&ss->everCaptured, std::memory_order_relaxed)) {
          CUDACHECK(cudaEventRecord(ss->serialEvent, ss->liveStream));
        }
        if (ss->liveAcquiredBy != localThreadId() && ncclParamLaunchRaceFatal()) {
          WARN("%s", launchRaceFatalMsg);
          return ncclInvalidUsage;
        }
      } else {
        std::unique_lock<std::mutex> lock(ss->mutex, std::defer_lock);
        if (concurrent) lock.lock();
        struct ncclStrongStreamCapture* cap = ss->captureHead;
        while (cap->graphId != graph.graphId) cap = cap->next;
        if (concurrent) lock.unlock();

        // Add event record node with dependencies added further down.
        cudaGraphNode_t recordNode;
        CUDACHECK(cudaGraphAddEventRecordNode(&recordNode, graph.graph, nullptr, 0, ss->serialEvent));

        // Get current nodes from work stream so we can add them as dependencies.
        cudaStreamCaptureStatus status;
        cudaGraphNode_t const* nodes;
        size_t count = 0;
        #if CUDART_VERSION >= 13000
        cudaError_t res = cudaStreamGetCaptureInfo_v3(cap->captureStream, &status, nullptr, nullptr, &nodes, nullptr, &count);
        #else
        cudaError_t res = cudaStreamGetCaptureInfo_v2(cap->captureStream, &status, nullptr, nullptr, &nodes, &count);
        #endif

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
          #if CUDART_VERSION >= 13000
            CUDACHECK(cudaGraphAddDependencies_v2(graph.graph, &nodes[i], &recordNode, nullptr, 1));
          #else
            CUDACHECK(cudaGraphAddDependencies(graph.graph, &nodes[i], &recordNode, 1));
          #endif
          }
        }

	// Make every future operation captured on cap->captureStream depend on 'recordNode'.
        #if CUDART_VERSION >= 13000
        CUDACHECK(cudaStreamUpdateCaptureDependencies_v2(
                    cap->captureStream,
                    &recordNode,          /* dependencies                */
                    /*edges =*/ nullptr,  /* no edge annotations         */
                    1,                    /* count                       */
                    cudaStreamSetCaptureDependencies));
        #else
        CUDACHECK(cudaStreamUpdateCaptureDependencies(
                    cap->captureStream,
                    &recordNode,
                    1,
                    cudaStreamSetCaptureDependencies));
        #endif

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
    #if CUDART_VERSION >= 13000
    cudaError_t res = cudaStreamGetCaptureInfo_v3(tmp, &status, nullptr, nullptr, &nodes, nullptr, &count);
    #else
    cudaError_t res = cudaStreamGetCaptureInfo_v2(tmp, &status, nullptr, nullptr, &nodes, &count);
    #endif

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
    #if CUDART_VERSION >= 13000
      CUDACHECK(cudaStreamUpdateCaptureDependencies_v2(s, (cudaGraphNode_t*)nodes, nullptr, count, cudaStreamSetCaptureDependencies));
    #else
      CUDACHECK(cudaStreamUpdateCaptureDependencies(s, (cudaGraphNode_t*)nodes, count, cudaStreamSetCaptureDependencies));
    #endif
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
