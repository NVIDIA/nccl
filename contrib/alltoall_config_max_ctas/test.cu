// SPDX-License-Identifier: Apache-2.0
// Author: 0z5a
#include <cuda_runtime.h>
#include <cupti.h>
#include <mpi.h>
#include <nccl.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call, ok) do { int error = (call); if (error != (ok)) { \
  fprintf(stderr, "rank=%d %s error=%d\n", rank, #call, error); MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define CUDA(call) CHECK(call, cudaSuccess)
#define NCCL(call) CHECK(call, ncclSuccess)
#define MPI(call) CHECK(call, MPI_SUCCESS)
#define CUPTI(call) CHECK(call, CUPTI_SUCCESS)
static int rank, limit, violations, launches;
static const char* scenario;
static bool measuring;
static void recordGrid(unsigned int blocks, const char* source) {
  printf("GRID,%s,%s,%d,%d,%u\n", scenario, source, rank, limit, blocks);
  launches++;
  if (limit > 0 && blocks > static_cast<unsigned int>(limit)) violations++;
}
static void CUPTIAPI trace(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data) {
  const auto* cb = static_cast<const CUpti_CallbackData*>(data);
  if (!measuring || domain != CUPTI_CB_DOMAIN_DRIVER_API || cb->callbackSite != CUPTI_API_ENTER) return;
  if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
    const auto* p = static_cast<const cuLaunchKernel_params*>(cb->functionParams);
    recordGrid(p->gridDimX * p->gridDimY * p->gridDimZ, "eager");
  } else if (id == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
    const auto* p = static_cast<const cuLaunchKernelEx_params*>(cb->functionParams);
    recordGrid(p->config->gridDimX * p->config->gridDimY * p->config->gridDimZ, "eager");
  }
}
static void inspectGraph(cudaGraph_t graph) {
  size_t count = 0;
  CUDA(cudaGraphGetNodes(graph, nullptr, &count));
  std::vector<cudaGraphNode_t> nodes(count);
  CUDA(cudaGraphGetNodes(graph, nodes.data(), &count));
  for (auto node : nodes) {
    cudaGraphNodeType type;
    CUDA(cudaGraphNodeGetType(node, &type));
    if (type == cudaGraphNodeTypeKernel) {
      CUDA_KERNEL_NODE_PARAMS params;
      CHECK(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(node), &params), CUDA_SUCCESS);
      recordGrid(params.gridDimX * params.gridDimY * params.gridDimZ, "graph-node");
    } else if (type == cudaGraphNodeTypeGraph) {
      cudaGraph_t child;
      CUDA(cudaGraphChildGraphNodeGetGraph(node, &child));
      inspectGraph(child);
    }
  }
}
static float value(int call, int src, int dst, size_t i) {
  return float(call * 10000 + src * 1000 + dst * 100 + i % 97);
}
int main(int argc, char** argv) {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  MPI(MPI_Init(&argc, &argv));
  MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  int ranks;
  MPI(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
  if (argc != 3 || (strcmp(argv[1], "eager") && strcmp(argv[1], "graph"))) {
    if (!rank) fprintf(stderr, "usage: mpirun -np N ./alltoall_config_mpi eager|graph COUNT_PER_PEER\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  const bool graphMode = !strcmp(argv[1], "graph");
  size_t count = strtoull(argv[2], nullptr, 10);
  CUDA(cudaSetDevice(rank));
  ncclUniqueId id;
  if (!rank) NCCL(ncclGetUniqueId(&id));
  MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm;
  NCCL(ncclCommInitRank(&comm, ranks, id, rank));
  cudaStream_t stream;
  CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const size_t elements = count * ranks;
  std::vector<float> host(std::max<size_t>(1, elements * 3));
  for (int call = 0; call < 3; call++)
    for (int dst = 0; dst < ranks; dst++)
      for (size_t i = 0; i < count; i++) host[call * elements + dst * count + i] = value(call, rank, dst, i);
  float *send, *recv;
  CUDA(cudaMalloc(&send, host.size() * sizeof(float)));
  CUDA(cudaMalloc(&recv, host.size() * sizeof(float)));
  CUDA(cudaMemcpy(send, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUpti_SubscriberHandle subscriber;
  CUPTI(cuptiSubscribe(&subscriber, trace, nullptr));
  CUPTI(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
  CUPTI(cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx));
  int failures = 0;
  for (int test = 0; test < 10; test++) {
    const int limits[] = {-1, 1, 3, 7, 1, 1, 1, -1, -1, -1};
    limit = limits[test];
    scenario = test == 4 ? "mixed-default-3-1" : test == 5 ? "mixed-sendrecv" :
        test == 6 ? "mixed-allreduce" : test == 7 ? "restored-sendrecv" : test == 8 ? "restored-default" : test == 9 ? "mixed-wide-allreduce" : "single";
    const int calls = test == 4 ? 3 : (test == 5 || test == 6 || test == 9) ? 2 : 1;
    auto enqueue = [&]() {
      NCCL(ncclGroupStart());
      for (int call = 0; call < calls; call++) {
        ncclCollConfig_t config = NCCL_COLLCONFIG_INITIALIZER;
        int bound = test == 4 ? (call == 0 ? -1 : call == 1 ? 3 : 1) : limit;
        if (test == 9) bound = call == 0 ? 1 : 7;
        if (bound > 0) config.maxCTAs = bound;
        if ((test == 5 && call == 1) || test == 7) {
          NCCL(ncclSend(send + call * elements, count, ncclFloat, (rank + 1) % ranks, comm, stream));
          NCCL(ncclRecv(recv + call * elements, count, ncclFloat, (rank + ranks - 1) % ranks, comm, stream));
        } else if ((test == 6 || test == 9) && call == 1) {
          NCCL(ncclAllReduceConfig(send + elements, recv + elements, count, ncclFloat,
                                   ncclSum, comm, stream, &config));
        } else {
          NCCL(ncclAlltoAllConfig(send + call * elements, recv + call * elements, count,
                                 ncclFloat, comm, stream, &config));
        }
      }
      NCCL(ncclGroupEnd());
    };
    // Warm up connections before capture. Inspect the selected execution mode only.
    measuring = false;
    enqueue();
    CUDA(cudaStreamSynchronize(stream));
    violations = launches = 0;
    measuring = !graphMode;
    CUDA(cudaMemset(recv, 0xff, host.size() * sizeof(float)));
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    if (graphMode) {
      CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      enqueue();
      CUDA(cudaStreamEndCapture(stream, &graph));
      inspectGraph(graph);
      CUDA(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    }
    for (int repeat = 0; repeat < 3; repeat++) {
      CUDA(cudaMemsetAsync(recv, 0xff, host.size() * sizeof(float), stream));
      if (graphMode) CUDA(cudaGraphLaunch(executable, stream)); else enqueue();
      CUDA(cudaStreamSynchronize(stream));
      CUDA(cudaMemcpy(host.data(), recv, host.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (int call = 0; call < calls; call++) {
        size_t validElements = (call == 1 && (test == 5 || test == 6 || test == 9)) || test == 7 ? count : elements;
        for (size_t i = 0; i < validElements; i++) {
          float expected = value(call, i / count, rank, i % count);
          if ((test == 5 && call == 1) || test == 7) expected = value(call, (rank + ranks - 1) % ranks, 0, i);
          if ((test == 6 || test == 9) && call == 1) {
            expected = 0;
            for (int src = 0; src < ranks; src++) expected += value(call, src, 0, i);
          }
          if (host[call * elements + i] != expected) {
            fprintf(stderr, "MISMATCH,rank=%d,call=%d,index=%zu\n", rank, call, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
        }
      }
    }
    if (graphMode) { CUDA(cudaGraphExecDestroy(executable)); CUDA(cudaGraphDestroy(graph)); }
    if (count && !launches) MPI_Abort(MPI_COMM_WORLD, 1);
    printf("RESULT,%s,%s,%d,%d,%zu,correctness=PASS,bounds=%s,observations=%d\n",
           argv[1], scenario, rank, limit, count, violations ? "FAIL" : "PASS", launches);
    failures += violations;
  }
  CUPTI(cuptiUnsubscribe(subscriber));
  CUDA(cudaFree(recv)); CUDA(cudaFree(send)); CUDA(cudaStreamDestroy(stream));
  NCCL(ncclCommDestroy(comm));
  int total;
  MPI(MPI_Allreduce(&failures, &total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  MPI(MPI_Finalize());
  return total ? 2 : 0;
}
