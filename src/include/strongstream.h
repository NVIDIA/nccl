/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_STRONGSTREAM_H_
#define NCCL_STRONGSTREAM_H_

#include "nccl.h"
#include "checks.h"

#include <stdint.h>

/* ncclCudaGraph: Wraps a cudaGraph_t so that we can support pre-graph CUDA runtimes
 * easily.
 */
struct ncclCudaGraph {
#if CUDART_VERSION >= 11030
  cudaGraph_t graph;
  uint64_t graphId;
#endif
};

inline struct ncclCudaGraph ncclCudaGraphNull() {
  struct ncclCudaGraph tmp;
  #if CUDART_VERSION >= 11030
    tmp.graph = nullptr;
    tmp.graphId = ULLONG_MAX;
  #endif
  return tmp;
}

inline bool ncclCudaGraphValid(struct ncclCudaGraph graph) {
  #if CUDART_VERSION >= 11030
    return graph.graph != nullptr;
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
 * Constraints of using strong streams:
 *
 * - Operations that enqueue work to the strong stream need to be enclosed by
 *   ncclStrongStream[Acquire/Release] pairs. Acquire/release act like fences,
 *   the strong stream is not stateful so there is no harm in redundant acquire
 *   or releases.
 *
 * - An {Acquire; ...; Release} sequence must not be concurrent with any
 *   other operations against the strong stream including graph launches which
 *   reference this stream.
 *
 * - All strong stream functions take a "graph" parameter which must reference
 *   the currently capturing graph, or null if none.
 */
struct ncclStrongStream;

ncclResult_t ncclStrongStreamConstruct(struct ncclStrongStream* ss);
ncclResult_t ncclStrongStreamDestruct(struct ncclStrongStream* ss);

// Has this strong stream ever been captured in a graph.
bool ncclStrongStreamEverCaptured(struct ncclStrongStream* ss);

// Acquire-fence the strong stream.
ncclResult_t ncclStrongStreamAcquire(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss
);

// Acquire-fence the strong stream assuming no graph is capturing. This permits
// the caller to enqueue directly to the `ss->stream` member using native CUDA
// calls. Strong stream must be released via:
//   ncclStrongStreamRelease(ncclCudaGraphNull(), graphRefs, ss);
ncclResult_t ncclStrongStreamAcquireUncaptured(struct ncclStrongStream* ss);

// Release-fence of the strong stream.
ncclResult_t ncclStrongStreamRelease(struct ncclCudaGraph graph, struct ncclStrongStream* ss);

// Add a host launch to the stream.
ncclResult_t ncclStrongStreamLaunchHost(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  cudaHostFn_t fn, void* arg
);
// Add a kernel launch to the stream.
ncclResult_t ncclStrongStreamLaunchKernel(
  struct ncclCudaGraph graph, struct ncclStrongStream* ss,
  void* fn, dim3 grid, dim3 block, void** args, size_t sharedMemBytes
);
// Cause `a` to wait for the current state `b`. Both `a` and `b` must be acquired.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, struct ncclStrongStream* b
);
// `b` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, struct ncclStrongStream* a, cudaStream_t b
);
// `a` must be capturing within `graph`.
ncclResult_t ncclStrongStreamWaitStream(
  struct ncclCudaGraph graph, cudaStream_t a, struct ncclStrongStream* b
);

// Synchrnoization does not need the strong stream to be acquired.
ncclResult_t ncclStrongStreamSynchronize(struct ncclStrongStream* ss);

////////////////////////////////////////////////////////////////////////////////

struct ncclStrongStream {
  cudaStream_t stream;
  cudaEvent_t event;
  #if CUDART_VERSION >= 11030
  cudaGraphNode_t node; // null if never captured, otherwise never null again
  uint64_t graphId:63, eventIsLagging:1;
  #endif
};

inline bool ncclStrongStreamEverCaptured(struct ncclStrongStream* ss) {
  #if CUDART_VERSION >= 11030
    return ss->node != nullptr;
  #else
    return false;
  #endif
}

#endif
