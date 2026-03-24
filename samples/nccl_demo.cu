/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * NCCL Windows port — standalone demo
 *
 * Shows:
 *   1. Communicator creation across all visible GPUs
 *   2. AllGather  — each rank contributes a unique slice; all ranks end
 *                   up with the full concatenated buffer
 *   3. AllReduce  — sum across all ranks; each element == nGPUs
 *   4. Broadcast  — rank 0 seeds a value; all others receive it
 *   5. ReduceScatter — complement of AllGather; each rank holds a
 *                      partial sum
 *   6. P2P Send/Recv — ring shift: rank i receives from (i-1+n)%n
 *************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "nccl.h"

/* ---- error helpers ---------------------------------------------------- */

#define CUDACHECK(cmd) do {                                          \
  cudaError_t e = (cmd);                                            \
  if (e != cudaSuccess) {                                           \
    fprintf(stderr, "CUDA error %s:%d '%s'\n",                     \
            __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);    \
  }                                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                                          \
  ncclResult_t r = (cmd);                                           \
  if (r != ncclSuccess) {                                           \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",                     \
            __FILE__, __LINE__, ncclGetErrorString(r)); exit(1);    \
  }                                                                 \
} while(0)

/* ---- utility ---------------------------------------------------------- */

static void banner(const char* title) {
  printf("\n═══ %s ═══\n", title);
}

static void syncAll(int n, const std::vector<cudaStream_t>& streams) {
  for (int i = 0; i < n; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(streams[i]));
  }
}

/* ======================================================================= */
int main() {
  int nGPUs = 0;
  CUDACHECK(cudaGetDeviceCount(&nGPUs));
  if (nGPUs < 1) { fprintf(stderr, "No CUDA devices.\n"); return 1; }
  printf("NCCL demo — %d GPU(s)\n", nGPUs);

  /* ---- 1. Build communicators ---------------------------------------- */
  banner("1. Communicator init");
  std::vector<ncclComm_t>    comms(nGPUs);
  std::vector<cudaStream_t>  streams(nGPUs);
  for (int i = 0; i < nGPUs; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }
  NCCLCHECK(ncclCommInitAll(comms.data(), nGPUs, nullptr));
  printf("  Communicator created for %d GPU(s)\n", nGPUs);

  const int COUNT = 8;   /* small so we can print the full array */

  /* ---- 2. AllGather -------------------------------------------------- */
  banner("2. AllGather  (each GPU contributes COUNT floats)");
  {
    std::vector<float*> send(nGPUs), recv(nGPUs);
    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&send[i], COUNT * sizeof(float)));
      CUDACHECK(cudaMalloc(&recv[i], nGPUs * COUNT * sizeof(float)));
      std::vector<float> h(COUNT);
      for (int j = 0; j < COUNT; ++j) h[j] = (float)(i * 100 + j);
      CUDACHECK(cudaMemcpy(send[i], h.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice));
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; ++i)
      NCCLCHECK(ncclAllGather(send[i], recv[i], COUNT, ncclFloat, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    syncAll(nGPUs, streams);

    /* Print result from GPU 0 */
    std::vector<float> h(nGPUs * COUNT);
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(h.data(), recv[0], nGPUs * COUNT * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Result on GPU 0 (all %d × %d = %d elements):\n  ", nGPUs, COUNT, nGPUs * COUNT);
    for (int i = 0; i < nGPUs * COUNT; ++i) printf("%.0f ", h[i]);
    printf("\n  (pattern: GPU-rank × 100 + element-index)\n");

    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
    }
  }

  /* ---- 3. AllReduce -------------------------------------------------- */
  banner("3. AllReduce  (sum, all GPUs contribute 1.0 per element)");
  {
    std::vector<float*> buf(nGPUs);
    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&buf[i], COUNT * sizeof(float)));
      std::vector<float> h(COUNT, 1.0f);
      CUDACHECK(cudaMemcpy(buf[i], h.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice));
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; ++i)
      NCCLCHECK(ncclAllReduce(buf[i], buf[i], COUNT, ncclFloat, ncclSum, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    syncAll(nGPUs, streams);

    std::vector<float> h(COUNT);
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(h.data(), buf[0], COUNT * sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Result (each element should be %d): ", nGPUs);
    for (int i = 0; i < COUNT; ++i) printf("%.0f ", h[i]);
    printf("\n");

    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i)); CUDACHECK(cudaFree(buf[i]));
    }
  }

  /* ---- 4. Broadcast -------------------------------------------------- */
  banner("4. Broadcast  (GPU 0 broadcasts the value 7.0)");
  {
    const float BVAL = 7.0f;
    std::vector<float*> buf(nGPUs);
    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&buf[i], COUNT * sizeof(float)));
      std::vector<float> h(COUNT, i == 0 ? BVAL : 0.0f);
      CUDACHECK(cudaMemcpy(buf[i], h.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice));
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; ++i)
      NCCLCHECK(ncclBroadcast(buf[i], buf[i], COUNT, ncclFloat, 0, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    syncAll(nGPUs, streams);

    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      std::vector<float> h(COUNT);
      CUDACHECK(cudaMemcpy(h.data(), buf[i], COUNT * sizeof(float), cudaMemcpyDeviceToHost));
      printf("  GPU %d received: %.0f  %s\n", i, h[0], h[0] == BVAL ? "✓" : "✗");
      CUDACHECK(cudaFree(buf[i]));
    }
  }

  /* ---- 5. ReduceScatter ---------------------------------------------- */
  banner("5. ReduceScatter  (sum; each rank owns 1 chunk of COUNT elements)");
  {
    const int perRank = COUNT;
    const int total   = perRank * nGPUs;
    std::vector<float*> send(nGPUs), recv(nGPUs);
    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&send[i], total * sizeof(float)));
      CUDACHECK(cudaMalloc(&recv[i], perRank * sizeof(float)));
      std::vector<float> h(total, 1.0f);
      CUDACHECK(cudaMemcpy(send[i], h.data(), total * sizeof(float), cudaMemcpyHostToDevice));
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; ++i)
      NCCLCHECK(ncclReduceScatter(send[i], recv[i], perRank, ncclFloat, ncclSum, comms[i], streams[i]));
    NCCLCHECK(ncclGroupEnd());
    syncAll(nGPUs, streams);

    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      std::vector<float> h(perRank);
      CUDACHECK(cudaMemcpy(h.data(), recv[i], perRank * sizeof(float), cudaMemcpyDeviceToHost));
      printf("  GPU %d chunk (should be %d each): ", i, nGPUs);
      for (int j = 0; j < perRank; ++j) printf("%.0f ", h[j]);
      printf("\n");
      CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
    }
  }

  /* ---- 6. P2P Send/Recv ring ----------------------------------------- */
  if (nGPUs >= 2) {
    banner("6. P2P Send/Recv  (ring: rank i receives from (i-1+n)%n)");
    std::vector<float*> send(nGPUs), recv(nGPUs);
    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMalloc(&send[i], COUNT * sizeof(float)));
      CUDACHECK(cudaMalloc(&recv[i], COUNT * sizeof(float)));
      std::vector<float> h(COUNT, (float)i);
      CUDACHECK(cudaMemcpy(send[i], h.data(), COUNT * sizeof(float), cudaMemcpyHostToDevice));
      CUDACHECK(cudaMemset(recv[i], 0, COUNT * sizeof(float)));
    }

    NCCLCHECK(ncclGroupStart());
    for (int i = 0; i < nGPUs; ++i) {
      int dst = (i + 1) % nGPUs;
      int src = (i - 1 + nGPUs) % nGPUs;
      NCCLCHECK(ncclSend(send[i], COUNT, ncclFloat, dst, comms[i], streams[i]));
      NCCLCHECK(ncclRecv(recv[i], COUNT, ncclFloat, src, comms[i], streams[i]));
    }
    NCCLCHECK(ncclGroupEnd());
    syncAll(nGPUs, streams);

    for (int i = 0; i < nGPUs; ++i) {
      CUDACHECK(cudaSetDevice(i));
      std::vector<float> h(COUNT);
      CUDACHECK(cudaMemcpy(h.data(), recv[i], COUNT * sizeof(float), cudaMemcpyDeviceToHost));
      int src = (i - 1 + nGPUs) % nGPUs;
      printf("  GPU %d received from GPU %d: first value = %.0f  %s\n",
             i, src, h[0], h[0] == (float)src ? "✓" : "✗");
      CUDACHECK(cudaFree(send[i])); CUDACHECK(cudaFree(recv[i]));
    }
  } else {
    printf("\n  (P2P demo skipped — need ≥2 GPUs)\n");
  }

  /* ---- cleanup -------------------------------------------------------- */
  banner("Done");
  for (int i = 0; i < nGPUs; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamDestroy(streams[i]));
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }
  printf("All done.\n");
  return 0;
}
