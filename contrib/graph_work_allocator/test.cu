// SPDX-License-Identifier: Apache-2.0
// Author: 0z5a
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

static int rank;
#define CHECK(call, ok) do { int error = (call); if (error != (ok)) { \
  fprintf(stderr, "rank=%d %s error=%d\n", rank, #call, error); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CUDA(call) CHECK(call, cudaSuccess)
#define NCCL(call) CHECK(call, ncclSuccess)
#define MPI(call) CHECK(call, MPI_SUCCESS)

struct Pool {
  cudaMemPool_t pool;
  std::mutex mutex;
  std::unordered_map<void*, size_t> live;
  size_t allocations = 0, frees = 0, bytes = 0;
  bool fail = false;
};
#ifdef WORK_ALLOCATOR
static ncclResult_t allocate(void* context, void** ptr, size_t bytes, int device, cudaStream_t stream) {
  auto& pool = *static_cast<Pool*>(context);
  if (pool.fail) return ncclUnhandledCudaError;
  int current;
  if (cudaGetDevice(&current) != cudaSuccess || current != device) return ncclInternalError;
  if (cudaMallocFromPoolAsync(ptr, bytes, pool.pool, stream) != cudaSuccess) return ncclUnhandledCudaError;
  std::lock_guard<std::mutex> lock(pool.mutex);
  pool.live.emplace(*ptr, bytes);
  pool.allocations++;
  pool.bytes += bytes;
  return ncclSuccess;
}
static ncclResult_t release(void* context, void* ptr, size_t bytes, int device) {
  auto& pool = *static_cast<Pool*>(context);
  int current;
  if (cudaGetDevice(&current) != cudaSuccess || current != device) return ncclInternalError;
  std::lock_guard<std::mutex> lock(pool.mutex);
  auto entry = pool.live.find(ptr);
  if (entry == pool.live.end() || entry->second != bytes) return ncclInternalError;
  if (cudaFree(ptr) != cudaSuccess) return ncclUnhandledCudaError;
  pool.live.erase(entry);
  pool.frees++;
  return ncclSuccess;
}
#endif

int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  MPI(MPI_Init(&argc, &argv)); MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int ranks; MPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  const char* mode = argc > 1 ? argv[1] : "default";
  const bool external = strcmp(mode, "default") != 0;
  CUDA(cudaSetDevice(rank));
  Pool pool;
  pool.fail = !strcmp(mode, "fail");
  cudaMemPoolProps props = {};
  props.allocType = cudaMemAllocationTypePinned;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = rank;
  CUDA(cudaMemPoolCreate(&pool.pool, &props));
  uint64_t threshold = 1 << 20;
  CUDA(cudaMemPoolSetAttribute(pool.pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  ncclUniqueId id;
  if (!rank) NCCL(ncclGetUniqueId(&id));
  MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.maxCTAs = 1;
#ifdef WORK_ALLOCATOR
  if (external) config.workAllocator = {&pool, allocate, release};
  if (!strcmp(mode, "invalid")) config.workAllocator.free = nullptr;
  if (!strcmp(mode, "legacy")) config.size = offsetof(ncclConfig_t, workAllocator);
#else
  if (external) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
  ncclComm_t comm = nullptr;
  ncclResult_t initialized = ncclCommInitRankConfig(&comm, ranks, id, rank, &config);
  if (!strcmp(mode, "invalid")) {
    if (initialized != ncclInvalidArgument) MPI_Abort(MPI_COMM_WORLD, 1);
    printf("RESULT,invalid,rank=%d,PASS\n", rank);
    CUDA(cudaMemPoolDestroy(pool.pool)); MPI(MPI_Finalize()); return 0;
  }
  NCCL(initialized);
  constexpr int calls = 64, count = 128;
  const size_t bytes = calls * count * sizeof(float);
  std::vector<float> host(calls * count, float(rank + 1));
  float *send, *recv;
  CUDA(cudaMalloc(&send, bytes)); CUDA(cudaMalloc(&recv, bytes));
  CUDA(cudaMemcpy(send, host.data(), bytes, cudaMemcpyHostToDevice));
  cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  auto enqueue = [&]() {
    ncclResult_t result = ncclGroupStart();
    if (result != ncclSuccess) return result;
    for (int call = 0; call < calls; call++) {
      ncclResult_t next = ncclAllReduce(send + call * count, recv + call * count, count,
                                       ncclFloat, ncclSum, comm, stream);
      if (next != ncclSuccess) result = next;
    }
    ncclResult_t end = ncclGroupEnd();
    return result == ncclSuccess ? end : result;
  };
  NCCL(enqueue()); CUDA(cudaStreamSynchronize(stream));
  ncclComm_t parent = comm;
  double captureUs = 0, replayUs = 0;
  for (int iteration = 0; iteration < 6; iteration++) {
    if (iteration == 3 && !strcmp(mode, "split")) {
      NCCL(ncclCommSplit(parent, 0, rank, &comm, nullptr));
      NCCL(enqueue()); CUDA(cudaStreamSynchronize(stream));
    }
    cudaGraph_t graph = nullptr;
    auto start = std::chrono::steady_clock::now();
    CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    ncclResult_t result = enqueue();
    cudaError_t end = cudaStreamEndCapture(stream, &graph);
    if (pool.fail) {
      if (result != ncclUnhandledCudaError) MPI_Abort(MPI_COMM_WORLD, 1);
      if (end != cudaSuccess && end != cudaErrorStreamCaptureInvalidated) MPI_Abort(MPI_COMM_WORLD, 1);
      if (graph) CUDA(cudaGraphDestroy(graph));
      NCCL(ncclCommAbort(comm));
      printf("RESULT,allocation-failure,rank=%d,PASS\n", rank);
      CUDA(cudaStreamDestroy(stream)); CUDA(cudaFree(recv)); CUDA(cudaFree(send));
      CUDA(cudaMemPoolDestroy(pool.pool)); MPI(MPI_Finalize()); return 0;
    }
    NCCL(result); CUDA(end);
    cudaGraphExec_t executable;
    CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    double captured = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count();
    if (iteration) captureUs += captured;
    for (int repeat = 0; repeat < 3; repeat++) {
      CUDA(cudaMemsetAsync(recv, 0xff, bytes, stream));
      CUDA(cudaStreamSynchronize(stream));
      start = std::chrono::steady_clock::now();
      CUDA(cudaGraphLaunch(executable, stream)); CUDA(cudaStreamSynchronize(stream));
      if (iteration) replayUs += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-start).count();
      CUDA(cudaMemcpy(host.data(), recv, bytes, cudaMemcpyDeviceToHost));
      for (float value : host) if (value != float(ranks * (ranks + 1) / 2)) MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph));
  }
  NCCL(ncclCommDestroy(comm));
  if (comm != parent) NCCL(ncclCommDestroy(parent));
  bool usesCallback = external && strcmp(mode, "legacy");
  if (!pool.live.empty() || pool.allocations != pool.frees || (usesCallback != (pool.allocations > 0)))
    MPI_Abort(MPI_COMM_WORLD, 1);
  printf("RESULT,%s,rank=%d,capture_us=%.3f,replay_us=%.3f,alloc=%zu,free=%zu,bytes=%zu,PASS\n",
         mode, rank, captureUs/5, replayUs/15, pool.allocations, pool.frees, pool.bytes);
  CUDA(cudaStreamDestroy(stream)); CUDA(cudaFree(recv)); CUDA(cudaFree(send));
  CUDA(cudaMemPoolDestroy(pool.pool)); MPI(MPI_Finalize());
}
