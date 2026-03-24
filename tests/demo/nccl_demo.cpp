/*
 * NCCL Windows port — standalone demo
 *
 * Demonstrates how to:
 *   1. Create a communicator over all available GPUs
 *   2. AllGather:  each GPU contributes a chunk, all receive the whole array
 *   3. AllReduce:  distributed sum across GPUs
 *   4. ReduceScatter: each GPU ends up with one reduced chunk
 *   5. Send/Recv (P2P): ring exchange
 *
 * The demo prints the data layout before and after each operation so you can
 * see exactly how data moves between GPUs.
 */

#include <nccl.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// ─── Macros ──────────────────────────────────────────────────────────────────

#define CUDACHECK(cmd) do {                                           \
  cudaError_t _e = (cmd);                                            \
  if (_e != cudaSuccess) {                                           \
    fprintf(stderr, "[CUDA] %s:%d  %s\n",                           \
            __FILE__, __LINE__, cudaGetErrorString(_e));             \
    exit(EXIT_FAILURE);                                              \
  }                                                                  \
} while(0)

#define NCCLCHECK(cmd) do {                                           \
  ncclResult_t _r = (cmd);                                           \
  if (_r != ncclSuccess) {                                           \
    fprintf(stderr, "[NCCL] %s:%d  %s\n",                           \
            __FILE__, __LINE__, ncclGetErrorString(_r));             \
    exit(EXIT_FAILURE);                                              \
  }                                                                  \
} while(0)

// ─── Utility ─────────────────────────────────────────────────────────────────

static void printBanner(const char* title) {
  printf("\n╔═══════════════════════════════════════════╗\n");
  printf("║  %-41s  ║\n", title);
  printf("╚═══════════════════════════════════════════╝\n");
}

static void printRow(int gpu, const float* h, int n, const char* label) {
  printf("  GPU %d [%-12s] :", gpu, label);
  for (int i = 0; i < n && i < 8; i++) printf(" %6.1f", h[i]);
  if (n > 8) printf(" ...");
  printf("\n");
}

static void fillHost(float* h, int n, float val) {
  for (int i = 0; i < n; i++) h[i] = val;
}

static void toDevice(int gpu, float* dev, const float* host, int n) {
  CUDACHECK(cudaSetDevice(gpu));
  CUDACHECK(cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice));
}

static void fromDevice(int gpu, float* host, const float* dev, int n) {
  CUDACHECK(cudaSetDevice(gpu));
  CUDACHECK(cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost));
}

// ─── Main demo ───────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
  // ── Find devices ──────────────────────────────────────────────────────────
  int ndev = 0;
  CUDACHECK(cudaGetDeviceCount(&ndev));
  if (ndev == 0) { fprintf(stderr, "No CUDA devices.\n"); return 1; }

  int N = ndev;
  for (int i = 1; i < argc - 1; i++)
    if (strcmp(argv[i], "--ngpus") == 0)
      N = std::min(atoi(argv[i + 1]), ndev);

  printf("NCCL Demo — %d GPU(s) detected, using %d\n", ndev, N);
  printf("NCCL build version: %d\n", NCCL_VERSION_CODE);

  // ── Create communicator ───────────────────────────────────────────────────
  printBanner("Step 1: Create communicator");

  std::vector<int>          devs(N);
  std::vector<ncclComm_t>   comms(N);
  std::vector<cudaStream_t> streams(N);

  for (int i = 0; i < N; i++) devs[i] = i;
  NCCLCHECK(ncclCommInitAll(comms.data(), N, devs.data()));

  for (int i = 0; i < N; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }
  printf("  Communicator ready across %d GPUs.\n", N);
  for (int i = 0; i < N; i++) {
    int rank, size;
    NCCLCHECK(ncclCommUserRank(comms[i], &rank));
    NCCLCHECK(ncclCommCount(comms[i], &size));
    printf("  GPU %d: rank=%d / %d\n", i, rank, size);
  }

  // ── AllGather ─────────────────────────────────────────────────────────────
  //  Each GPU contributes 4 elements. After AllGather every GPU has N*4 elems.
  {
    printBanner("Step 2: AllGather");
    const int chunk = 4;
    const int recvTotal = N * chunk;

    std::vector<float*> dSend(N), dRecv(N);
    std::vector<std::vector<float>> hSend(N), hRecv(N, std::vector<float>(recvTotal, 0));

    printf("  Before: each GPU holds %d elements (its own chunk)\n", chunk);
    for (int i = 0; i < N; i++) {
      hSend[i].resize(chunk);
      for (int j = 0; j < chunk; j++) hSend[i][j] = (float)(i * 10 + j);
      printRow(i, hSend[i].data(), chunk, "send");

      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&dSend[i], chunk * sizeof(float)));
      CUDACHECK(cudaMalloc(&dRecv[i], recvTotal * sizeof(float)));
      toDevice(i, dSend[i], hSend[i].data(), chunk);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < N; i++)
      NCCLCHECK(ncclAllGather(dSend[i], dRecv[i], chunk,
                              ncclFloat, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < N; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\n  After: every GPU holds all %d elements\n", recvTotal);
    for (int i = 0; i < N; i++) {
      fromDevice(i, hRecv[i].data(), dRecv[i], recvTotal);
      printRow(i, hRecv[i].data(), recvTotal, "recv");
      cudaFree(dSend[i]); cudaFree(dRecv[i]);
    }
  }

  // ── AllReduce ─────────────────────────────────────────────────────────────
  //  Each GPU starts with value (rank+1). After AllReduce every GPU has sum.
  {
    printBanner("Step 3: AllReduce (SUM)");
    const int n = 8;

    std::vector<float*> dBuf(N);
    std::vector<std::vector<float>> h(N, std::vector<float>(n));

    printf("  Before: GPU i holds (i+1) in all slots\n");
    for (int i = 0; i < N; i++) {
      fillHost(h[i].data(), n, (float)(i + 1));
      printRow(i, h[i].data(), n, "input");
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&dBuf[i], n * sizeof(float)));
      toDevice(i, dBuf[i], h[i].data(), n);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < N; i++)
      NCCLCHECK(ncclAllReduce(dBuf[i], dBuf[i], n,
                              ncclFloat, ncclSum, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < N; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    float expected = (float)(N * (N + 1) / 2);
    printf("\n  After: every GPU should have sum = %.0f\n", expected);
    for (int i = 0; i < N; i++) {
      fromDevice(i, h[i].data(), dBuf[i], n);
      printRow(i, h[i].data(), n, "result");
      cudaFree(dBuf[i]);
    }
  }

  // ── ReduceScatter ─────────────────────────────────────────────────────────
  //  Total = N * chunk elements per GPU, reduced → each GPU gets one chunk.
  {
    printBanner("Step 4: ReduceScatter (SUM)");
    const int chunk = 4;
    const int sendTotal = N * chunk;

    std::vector<float*> dSend(N), dRecv(N);
    std::vector<std::vector<float>> hRecv(N, std::vector<float>(chunk));

    printf("  Before: each GPU holds %d elements (all set to rank+1)\n", sendTotal);
    for (int i = 0; i < N; i++) {
      std::vector<float> hs(sendTotal, (float)(i + 1));
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&dSend[i], sendTotal * sizeof(float)));
      CUDACHECK(cudaMalloc(&dRecv[i], chunk * sizeof(float)));
      toDevice(i, dSend[i], hs.data(), sendTotal);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < N; i++)
      NCCLCHECK(ncclReduceScatter(dSend[i], dRecv[i], chunk,
                                  ncclFloat, ncclSum, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < N; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    float expected = (float)(N * (N + 1) / 2);
    printf("\n  After: each GPU holds its reduced chunk (expected %.0f)\n", expected);
    for (int i = 0; i < N; i++) {
      fromDevice(i, hRecv[i].data(), dRecv[i], chunk);
      printRow(i, hRecv[i].data(), chunk, "recv chunk");
      cudaFree(dSend[i]); cudaFree(dRecv[i]);
    }
  }

  // ── P2P Send/Recv (ring) ──────────────────────────────────────────────────
  if (N >= 2) {
    printBanner("Step 5: P2P Send/Recv (ring)");
    const int n = 4;

    std::vector<float*> dSend(N), dRecv(N);
    std::vector<std::vector<float>> hSend(N), hRecv(N, std::vector<float>(n, -1));

    printf("  Ring: GPU i sends its value (i+1) to GPU (i+1)%%N\n");
    for (int i = 0; i < N; i++) {
      hSend[i].assign(n, (float)(i + 1));
      printRow(i, hSend[i].data(), n, "sending");
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&dSend[i], n * sizeof(float)));
      CUDACHECK(cudaMalloc(&dRecv[i], n * sizeof(float)));
      toDevice(i, dSend[i], hSend[i].data(), n);
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < N; i++) {
      int dst = (i + 1) % N;
      int src = (i + N - 1) % N;
      NCCLCHECK(ncclSend(dSend[i], n, ncclFloat, dst, comms[i], streams[i]));
      NCCLCHECK(ncclRecv(dRecv[i], n, ncclFloat, src, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    for (int i = 0; i < N; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }

    printf("\n  After ring exchange:\n");
    for (int i = 0; i < N; i++) {
      fromDevice(i, hRecv[i].data(), dRecv[i], n);
      int src = (i + N - 1) % N;
      printf("  GPU %d received from GPU %d:", i, src);
      for (int j = 0; j < n; j++) printf(" %.1f", hRecv[i][j]);
      printf("\n");
      cudaFree(dSend[i]); cudaFree(dRecv[i]);
    }
  } else {
    printf("\n  (Skipping P2P demo — need >= 2 GPUs)\n");
  }

  // ── Cleanup ──────────────────────────────────────────────────────────────
  for (int i = 0; i < N; i++) {
    CUDACHECK(cudaSetDevice(i));
    cudaStreamDestroy(streams[i]);
    ncclCommDestroy(comms[i]);
  }

  printf("\n=== Demo complete ===\n");
  return 0;
}
