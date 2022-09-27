/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "strongstream.h"
#include "cudawrap.h"
#include "checks.h"
#include "param.h"

////////////////////////////////////////////////////////////////////////////////

ncclResult_t ncclCudaGetCapturingGraph(
    struct ncclCudaGraph* graph, cudaStream_t stream
  ) {
  #if CUDART_VERSION >= 11030
    int driver;
    NCCLCHECK(ncclCudaDriverVersion(&driver));
    if (driver < 11030) {
      cudaStreamCaptureStatus status;
      unsigned long long gid;
      graph->graph = nullptr;
      CUDACHECK(cudaStreamGetCaptureInfo(stream, &status, &gid));
      if (status != cudaStreamCaptureStatusNone) {
        WARN("The installed CUDA driver is older than the minimum version (R465) required for NCCL's CUDA Graphs support");
        return ncclInvalidUsage;
      }
    } else {
      cudaStreamCaptureStatus status;
      unsigned long long gid;
      CUDACHECK(cudaStreamGetCaptureInfo_v2(stream, &status, &gid, &graph->graph, nullptr, nullptr));
      if (status != cudaStreamCaptureStatusActive) {
        graph->graph = nullptr;
        gid = ULLONG_MAX;
      }
      graph->graphId = gid;
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
  CUDACHECK(cudaStreamCreateWithFlags(&ss->stream, cudaStreamNonBlocking));
  CUDACHECK(cudaEventCreateWithFlags(&ss->event, cudaEventDisableTiming));
  #if CUDART_VERSION >= 11030
    ss->node = nullptr;
    ss->graphId = (1ull<<(8*sizeof(long long)-1))-1;
    ss->eventIsLagging = 0;
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaEventDestroy(ss->event));
  #endif
  CUDACHECK(cudaStreamDestroy(ss->stream));
  return ncclSuccess;
}

NCCL_PARAM(GraphMixingSupport, "GRAPH_MIXING_SUPPORT", 1)

ncclResult_t ncclStrongStreamAcquire(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss
  ) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (graph.graph == nullptr) {
      if (mixing && ncclStrongStreamEverCaptured(ss)) {
        CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event, 0));
        ss->eventIsLagging = 0;
      }
    } else {
      if (ss->graphId != graph.graphId) {
        if (mixing && ss->eventIsLagging) {
          // Can only be here if previous release was for uncaptured work that
          // elided updating the event because no capture had yet occurred.
          CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event, 0));
          CUDACHECK(cudaEventRecord(ss->event, ss->stream));
        }
        ss->graphId = graph.graphId;
        ss->eventIsLagging = 0;
        if (mixing) {
          CUDACHECK(cudaGraphAddEventWaitNode(&ss->node, graph.graph, nullptr, 0, ss->event));
        } else {
          CUDACHECK(cudaGraphAddEmptyNode(&ss->node, graph.graph, nullptr, 0));
        }
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (mixing && ncclStrongStreamEverCaptured(ss)) {
      CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event, 0));
    }
    ss->eventIsLagging = 1; // Assume the caller is going to add work to stream.
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    bool mixing = ncclParamGraphMixingSupport();
    if (mixing && ss->eventIsLagging) {
      if (graph.graph == nullptr) {
        if (ncclStrongStreamEverCaptured(ss)) {
          CUDACHECK(cudaEventRecord(ss->event, ss->stream));
          ss->eventIsLagging = 0;
        }
      } else {
        CUDACHECK(cudaGraphAddEventRecordNode(&ss->node, graph.graph, &ss->node, 1, ss->event));
        ss->eventIsLagging = 0;
      }
    }
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamLaunchHost(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss, cudaHostFn_t fn, void* arg
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaLaunchHostFunc(ss->stream, fn, arg));
    } else {
      cudaHostNodeParams p;
      p.fn = fn;
      p.userData = arg;
      CUDACHECK(cudaGraphAddHostNode(&ss->node, graph.graph, &ss->node, 1, &p));
    }
    ss->eventIsLagging = 1;
  #else
    CUDACHECK(cudaLaunchHostFunc(ss->stream, fn, arg));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamLaunchKernel(
    struct ncclCudaGraph graph, struct ncclStrongStream* ss,
    void* fn, dim3 grid, dim3 block, void* args[], size_t sharedMemBytes
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->stream));
    } else {
      cudaGraphNode_t tip = ss->node;
      cudaKernelNodeParams p;
      p.func = fn;
      p.gridDim = grid;
      p.blockDim = block;
      p.kernelParams = args;
      p.sharedMemBytes = sharedMemBytes;
      p.extra = nullptr;
      CUDACHECK(cudaGraphAddKernelNode(&ss->node, graph.graph, &tip, 1, &p));
    }
    ss->eventIsLagging = 1;
  #else
    CUDACHECK(cudaLaunchKernel(fn, grid, block, args, sharedMemBytes, ss->stream));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->eventIsLagging) {
        b->eventIsLagging = 0;
        CUDACHECK(cudaEventRecord(b->event, b->stream));
      }
      CUDACHECK(cudaStreamWaitEvent(a->stream, b->event, 0));
    } else {
      cudaGraphNode_t pair[2] = {a->node, b->node};
      CUDACHECK(cudaGraphAddEmptyNode(&a->node, graph.graph, pair, 2));
    }
    a->eventIsLagging = 1;
  #else
    CUDACHECK(cudaEventRecord(b->event, b->stream));
    CUDACHECK(cudaStreamWaitEvent(a->stream, b->event, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      CUDACHECK(cudaEventRecord(a->event, b));
      CUDACHECK(cudaStreamWaitEvent(a->stream, a->event, 0));
      // We used a->event to record b so it no longer reflects anything about a.
      a->eventIsLagging = 1;
    } else {
      cudaStreamCaptureStatus status;
      unsigned long long gid1;
      cudaGraphNode_t const* deps;
      size_t depN = 0;
      CUDACHECK(cudaStreamGetCaptureInfo_v2(b, &status, &gid1, nullptr, &deps, &depN));
      if (status != cudaStreamCaptureStatusActive || graph.graphId != gid1) {
        WARN("Stream is not being captured by the expected graph.");
        return ncclInvalidUsage;
      }
      if (depN > 0 && (depN > 1 || deps[0] != a->node)) {
        cudaGraphNode_t tie;
        if (depN == 1) {
          tie = deps[0];
        } else {
          CUDACHECK(cudaGraphAddEmptyNode(&tie, graph.graph, deps, depN));
        }
        cudaGraphNode_t pair[2] = {a->node, tie};
        CUDACHECK(cudaGraphAddEmptyNode(&a->node, graph.graph, pair, 2));
        a->eventIsLagging = 1;
      }
    }
  #else
    CUDACHECK(cudaEventRecord(a->event, b));
    CUDACHECK(cudaStreamWaitEvent(a->stream, a->event, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamWaitStream(
    struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b
  ) {
  #if CUDART_VERSION >= 11030
    if (graph.graph == nullptr) {
      if (b->eventIsLagging) {
        b->eventIsLagging = 0;
        CUDACHECK(cudaEventRecord(b->event, b->stream));
      }
      CUDACHECK(cudaStreamWaitEvent(a, b->event, 0));
    } else {
      CUDACHECK(cudaStreamUpdateCaptureDependencies(a, &b->node, 1, cudaStreamAddCaptureDependencies));
    }
  #else
    CUDACHECK(cudaEventRecord(b->event, b->stream));
    CUDACHECK(cudaStreamWaitEvent(a, b->event, 0));
  #endif
  return ncclSuccess;
}

ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    CUDACHECK(cudaStreamWaitEvent(ss->stream, ss->event, 0));
  #endif
  CUDACHECK(cudaStreamSynchronize(ss->stream));
  return ncclSuccess;
}
