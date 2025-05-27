/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "nccl.h"
#include "checks.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ncclCudaContext: wraps a CUDA context with per-context state.
struct ncclCudaContext;

// Get a ncclCudaContext to track the currently active CUDA context.
ncclResult_t ncclCudaContextTrack(struct ncclCudaContext** out);
// Drop reference.
void ncclCudaContextDrop(struct ncclCudaContext* cxt);

/* ncclCudaGraph: Wraps a cudaGraph_t so that we can support pre-graph CUDA runtimes
 * easily.
 */
struct ncclCudaGraph {
#if CUDART_VERSION >= 11030
  cudaStream_t origin;
  cudaGraph_t graph;
  unsigned long long graphId;
#endif
};

inline struct ncclCudaGraph ncclCudaGraphNone() {
  struct ncclCudaGraph tmp;
  #if CUDART_VERSION >= 11030
    tmp.origin = nullptr;
    tmp.graph = nullptr;
    tmp.graphId = ULLONG_MAX;
  #endif
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
  #if CUDART_VERSION >= 11030
    return graph.graphId != ULLONG_MAX;
  #else
    return false;
  #endif
}

inline bool ncclCudaGraphSame(struct ncclCudaGraph a, struct ncclCudaGraph b) {
  #if CUDART_VERSION >= 11030
    return a.graphId == b.graphId;
  #else
    return true;
  #endif
}

ncclResult_t ncclCudaGetCapturingGraph(struct ncclCudaGraph* graph, cudaStream_t stream);
ncclResult_t ncclCudaGraphAddDestructor(struct ncclCudaGraph graph, cudaHostFn_t fn, void* arg);

/* ncclStrongStream: An abstraction over CUDA streams that do not lose their
 * identity while being captured. Regular streams have the deficiency that the
 * captured form of a stream in one graph launch has no relation to the
 * uncaptured stream or to the captured form in other graph launches. This makes
 * streams unfit for the use of serializing access to a persistent resource.
 * Strong streams have been introduced to address this need.
 *
 * All updates to a strong stream must be enclosed by a Acquire/Release pair.
 *
 * Acquire retrieves a "work" stream (cudaStream_t) which may be used to add
 * work.
 *
 * Release publishes the work streams work into the strong stream. The Release
 * must be issued by the same thread that did the Acquire.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// Acquire the strong stream. Upon return `*workStream` will be usable to add work.
// `concurrent` indicates if other threads may be using the strong stream.
ncclResult_t ncclStrongStreamAcquire(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent, cudaStream_t* workStream
);

// Get the workStream for an already acquired strong stream.
// `concurrent` indicates if other threads may be using the strong stream.
ncclResult_t ncclStrongStreamAcquiredWorkStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent, cudaStream_t* workStream
);

// Release of the strong stream.
// `concurrent` indicates if other threads may be using the strong stream.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss, bool concurrent);

ncclResult_t ncclStreamWaitStream(
  cudaStream_t a, cudaStream_t b, cudaEvent_t scratchEvent
);

// Like cudaStreamWaitEvent except `e` must be strictly ahead of everything in `s`.
ncclResult_t ncclStreamAdvanceToEvent(struct ncclCudaGraph g, cudaStream_t s, cudaEvent_t e);

// Synchrnoization does not need the strong stream to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

struct ncclStrongStreamCapture; // internal to ncclStrongStream

struct ncclStrongStream {
  // The stream to use for non-captured work.
  cudaStream_t liveStream;
  void* liveAcquiredBy;
#if CUDART_VERSION >= 11030
  // This stream ever appeared in a graph capture.
  bool everCaptured;
  pthread_mutex_t lock;
  struct ncclStrongStreamCapture* captureHead;
  // The event used to establish order between graphs and streams. During acquire
  // this event is waited on, during release it is recorded to.
  cudaEvent_t serialEvent;
#endif
};

struct ncclCudaContext {
  struct ncclCudaContext* next;
  CUcontext hcontext;
  int refCount;
  struct ncclStrongStream launchOrder;
};

#endif
