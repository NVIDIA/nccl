/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NCCL Windows port — unit tests
 * Tests: AllReduce, AllGather, Broadcast, Reduce, ReduceScatter,
 *        Scatter, Gather, AlltoAll, Send/Recv (P2P), multi-op stream.
 *************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include "nccl.h"

/* ---------- helpers ---------------------------------------------------- */

#define CUDACHECK(cmd) do {                                          \
  cudaError_t e = (cmd);                                            \
  if (e != cudaSuccess) {                                           \
    fprintf(stderr, "CUDA error %s:%d '%s'\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(e));             \
    exit(1);                                                        \
  }                                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                                          \
  ncclResult_t r = (cmd);                                           \
  if (r != ncclSuccess) {                                           \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",                     \
            __FILE__, __LINE__, ncclGetErrorString(r));             \
    exit(1);                                                        \
  }                                                                 \
} while(0)

static int gPassed = 0, gFailed = 0;

#define TEST_ASSERT(cond, msg) do {             \
  if (!(cond)) {                                \
    fprintf(stderr, "FAIL [%s]: %s\n", testName, msg); \
    ++gFailed; return;                          \
  }                                             \
} while(0)

#define TEST_PASS() do { printf("PASS [%s]\n", testName); ++gPassed; } while(0)

/* Build a flat communicator group across all visible GPUs.
 * Returns nGPUs (>=2 required for P2P tests). */
static int buildComms(std::vector<ncclComm_t>& comms,
                      std::vector<cudaStream_t>& streams) {
  int nGPUs = 0;
  CUDACHECK(cudaGetDeviceCount(&nGPUs));
  if (nGPUs < 1) { fprintf(stderr, "No CUDA devices found\n"); exit(1); }

  comms.resize(nGPUs);
  streams.resize(nGPUs);
  for (int i = 0; i < nGPUs; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }
  NCCLCHECK(ncclCommInitAll(comms.data(), nGPUs, nullptr));
  return nGPUs;
}

static void destroyComms(std::vector<ncclComm_t>& comms,
                         std::vector<cudaStream_t>& streams) {
  for (int i = 0; i < (int)comms.size(); ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    CUDACHECK(cudaStreamDestroy(streams[i]));
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }
}

/* ---------- test 1: AllReduce (sum of floats) --------------------------- */

static void testAllReduce() {
  const char* testName = "AllReduce";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int count = 1024;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], count * sizeof(float)));
    std::vector<float> h(count, 1.0f);
    CUDACHECK(cudaMemcpy(send[i], h.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclAllReduce(send[i], recv[i], count, ncclFloat, ncclSum, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(count);
    CUDACHECK(cudaMemcpy(h.data(), recv[i], count * sizeof(float), cudaMemcpyDeviceToHost));
    if (h[0] != (float)n) fprintf(stderr, "  rank %d: got %.1f expected %.1f\n", i, h[0], (float)n);
    TEST_ASSERT(h[0] == (float)n, "AllReduce sum mismatch");
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 2: AllGather ------------------------------------------ */

static void testAllGather() {
  const char* testName = "AllGather";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int perRank = 256;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], perRank * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], n * perRank * sizeof(float)));
    std::vector<float> h(perRank, (float)(i + 1));
    CUDACHECK(cudaMemcpy(send[i], h.data(), perRank * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclAllGather(send[i], recv[i], perRank, ncclFloat, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(n * perRank);
    CUDACHECK(cudaMemcpy(h.data(), recv[i], n * perRank * sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < n; ++r)
      TEST_ASSERT(h[r * perRank] == (float)(r + 1), "AllGather rank data mismatch");
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 3: Broadcast ------------------------------------------ */

static void testBroadcast() {
  const char* testName = "Broadcast";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int count = 512;
  const float kVal = 42.0f;
  const int root = 0;

  std::vector<float*> buf(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&buf[i], count * sizeof(float)));
    std::vector<float> h(count, i == root ? kVal : 0.0f);
    CUDACHECK(cudaMemcpy(buf[i], h.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclBroadcast(buf[i], buf[i], count, ncclFloat, root, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(count);
    CUDACHECK(cudaMemcpy(h.data(), buf[i], count * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h[0] == kVal, "Broadcast value mismatch");
    CUDACHECK(cudaFree(buf[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 4: Reduce --------------------------------------------- */

static void testReduce() {
  const char* testName = "Reduce";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int count = 512;
  const int root = 0;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], count * sizeof(float)));
    std::vector<float> h(count, 1.0f);
    CUDACHECK(cudaMemcpy(send[i], h.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclReduce(send[i], recv[i], count, ncclFloat, ncclSum, root, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  CUDACHECK(cudaSetDevice(root));
  CUDACHECK(cudaStreamSynchronize(streams[root]));
  std::vector<float> h(count);
  CUDACHECK(cudaMemcpy(h.data(), recv[root], count * sizeof(float), cudaMemcpyDeviceToHost));
  TEST_ASSERT(h[0] == (float)n, "Reduce sum mismatch at root");

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }
  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 5: ReduceScatter -------------------------------------- */

static void testReduceScatter() {
  const char* testName = "ReduceScatter";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int recvCount = 256;
  const int sendCount = recvCount * n;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], sendCount * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], recvCount * sizeof(float)));
    std::vector<float> h(sendCount, 1.0f);
    CUDACHECK(cudaMemcpy(send[i], h.data(), sendCount * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclReduceScatter(send[i], recv[i], recvCount, ncclFloat, ncclSum, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(recvCount);
    CUDACHECK(cudaMemcpy(h.data(), recv[i], recvCount * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h[0] == (float)n, "ReduceScatter sum mismatch");
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 6: Scatter -------------------------------------------- */

static void testScatter() {
  const char* testName = "Scatter";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int perRank = 128;
  const int root = 0;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], n * perRank * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], perRank * sizeof(float)));
    if (i == root) {
      std::vector<float> h(n * perRank);
      for (int r = 0; r < n; ++r)
        for (int j = 0; j < perRank; ++j) h[r * perRank + j] = (float)(r + 10);
      CUDACHECK(cudaMemcpy(send[i], h.data(), n * perRank * sizeof(float), cudaMemcpyHostToDevice));
    }
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclScatter(send[i], recv[i], perRank, ncclFloat, root, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(perRank);
    CUDACHECK(cudaMemcpy(h.data(), recv[i], perRank * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h[0] == (float)(i + 10), "Scatter value mismatch");
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 7: Gather --------------------------------------------- */

static void testGather() {
  const char* testName = "Gather";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int perRank = 128;
  const int root = 0;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], perRank * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], n * perRank * sizeof(float)));
    std::vector<float> h(perRank, (float)(i + 1));
    CUDACHECK(cudaMemcpy(send[i], h.data(), perRank * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclGather(send[i], recv[i], perRank, ncclFloat, root, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  CUDACHECK(cudaSetDevice(root));
  CUDACHECK(cudaStreamSynchronize(streams[root]));
  std::vector<float> h(n * perRank);
  CUDACHECK(cudaMemcpy(h.data(), recv[root], n * perRank * sizeof(float), cudaMemcpyDeviceToHost));
  for (int r = 0; r < n; ++r)
    TEST_ASSERT(h[r * perRank] == (float)(r + 1), "Gather rank data mismatch");

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }
  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 8: AlltoAll ------------------------------------------- */

static void testAlltoAll() {
  const char* testName = "AlltoAll";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int perPair = 64;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], n * perPair * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], n * perPair * sizeof(float)));
    std::vector<float> h(n * perPair);
    for (int r = 0; r < n; ++r)
      for (int j = 0; j < perPair; ++j) h[r * perPair + j] = (float)(i * 100 + r);
    CUDACHECK(cudaMemcpy(send[i], h.data(), n * perPair * sizeof(float), cudaMemcpyHostToDevice));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclAlltoAll(send[i], recv[i], perPair, ncclFloat, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(n * perPair);
    CUDACHECK(cudaMemcpy(h.data(), recv[i], n * perPair * sizeof(float), cudaMemcpyDeviceToHost));
    for (int r = 0; r < n; ++r)
      TEST_ASSERT(h[r * perPair] == (float)(r * 100 + i), "AlltoAll value mismatch");
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 9: Send/Recv (P2P ring) -------------------------------- */

static void testSendRecv() {
  const char* testName = "SendRecv";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  if (n < 2) { printf("SKIP [%s]: need >=2 GPUs\n", testName); return; }
  const int count = 256;

  std::vector<float*> send(n), recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&send[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&recv[i], count * sizeof(float)));
    std::vector<float> h(count, (float)i);
    CUDACHECK(cudaMemcpy(send[i], h.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemset(recv[i], 0, count * sizeof(float)));
  }

  /* Ring: rank i sends to (i+1)%n, receives from (i-1+n)%n */
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i) {
    int dst = (i + 1) % n;
    int src = (i - 1 + n) % n;
    NCCLCHECK(ncclSend(send[i], count, ncclFloat, dst, comms[i], streams[i]));
    NCCLCHECK(ncclRecv(recv[i], count, ncclFloat, src, comms[i], streams[i]));
  }
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    int src = (i - 1 + n) % n;
    std::vector<float> h(count);
    CUDACHECK(cudaMemcpy(h.data(), recv[i], count * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h[0] == (float)src, "SendRecv ring value mismatch");
    CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- test 10: multi-op on same stream (AllReduce + AllGather) ---- */

static void testMultiOpStream() {
  const char* testName = "MultiOpStream";
  std::vector<ncclComm_t> comms; std::vector<cudaStream_t> streams;
  int n = buildComms(comms, streams);
  const int count = 128;

  std::vector<float*> ar_send(n), ar_recv(n), ag_send(n), ag_recv(n);
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(&ar_send[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&ar_recv[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&ag_send[i], count * sizeof(float)));
    CUDACHECK(cudaMalloc(&ag_recv[i], n * count * sizeof(float)));
    std::vector<float> h(count, 1.0f);
    CUDACHECK(cudaMemcpy(ar_send[i], h.data(), count * sizeof(float), cudaMemcpyHostToDevice));
    CUDACHECK(cudaMemcpy(ag_send[i], h.data(), count * sizeof(float), cudaMemcpyHostToDevice));
  }

  /* First collective */
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclAllReduce(ar_send[i], ar_recv[i], count, ncclFloat, ncclSum, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  /* Second collective on the same streams (implicitly sequenced) */
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < n; ++i)
    NCCLCHECK(ncclAllGather(ag_send[i], ag_recv[i], count, ncclFloat, comms[i], streams[i]));
  NCCLCHECK(ncclGroupEnd());

  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
    std::vector<float> h(count);
    CUDACHECK(cudaMemcpy(h.data(), ar_recv[i], count * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_ASSERT(h[0] == (float)n, "MultiOpStream AllReduce mismatch");
    std::vector<float> g(n * count);
    CUDACHECK(cudaMemcpy(g.data(), ag_recv[i], n * count * sizeof(float), cudaMemcpyDeviceToHost));
    TEST_ASSERT(g[0] == 1.0f, "MultiOpStream AllGather mismatch");
    CUDACHECK(cudaFree(ar_send[i])); CUDACHECK(cudaFree(ar_recv[i]));
    CUDACHECK(cudaFree(ag_send[i])); CUDACHECK(cudaFree(ag_recv[i]));
  }

  destroyComms(comms, streams);
  TEST_PASS();
}

/* ---------- main ------------------------------------------------------- */

int main() {
  int nGPUs = 0;
  CUDACHECK(cudaGetDeviceCount(&nGPUs));
  printf("NCCL unit tests — %d GPU(s) detected\n\n", nGPUs);

  testAllReduce();
  testAllGather();
  testBroadcast();
  testReduce();
  testReduceScatter();
  testScatter();
  testGather();
  testAlltoAll();
  testSendRecv();
  testMultiOpStream();

  printf("\n%d passed, %d failed\n", gPassed, gFailed);
  return gFailed ? 1 : 0;
}
