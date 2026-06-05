/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// main.cu
//
// Host-side driver for the LSA AllToAll sample kernel.
//
// What it demonstrates:
//   1. Initializing NCCL and creating an ncclDevComm with LSA support.
//   2. Registering a symmetric buffer as an ncclWindow.
//   3. Passing ncclDevComm and ncclWindow by value to a custom CUDA kernel.
//   4. lsa_simple_alltoall_kernel — AllToAll using direct NVLink load/store within
//      the local NVLink island (LSA team), entirely from the GPU.
//   5. ncclAlltoAll — NCCL's built-in AllToAll over symmetric memory.
//   6. Side-by-side performance comparison across a range of message sizes.
//
// Only public NCCL headers are used:
//   nccl.h             — ncclComm, ncclWindow, ncclMemAlloc, ncclCommWindowRegister
//   nccl_device.h      — ncclDevComm, ncclDevCommCreate/Destroy, ncclDevCommRequirements_t
//
// Run with MPI:
//   mpirun -np <N> ./custom_algos [options]
//
// Options:
//   -c, --min-cta  N   smallest CTA count (power of 2, default 4)
//   -C, --max-cta  N   largest  CTA count (power of 2, default 4)
//   -m, --min-msg  B   smallest message size in bytes (power of 2, default 4)
//   -M, --max-msg  B   largest  message size in bytes (power of 2, default 128 MB)
//   -h, --help         print this help and exit
//       --no-graph     disable CUDA graph capture (default: enabled)
//
// Prerequisites:
//   - At least 2 MPI ranks on NVLink-connected GPUs (same NVLink island).
//   - CUDA >= 12.2, GPU compute capability >= 7.0.
//   - lsaSize == 1 means no NVLink peers; the correctness test will still
//     pass (rank reads its own send slot) but bandwidth numbers are trivial.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

// Public NCCL headers only.
#include "nccl.h"
#include "nccl_device.h"

#include "lsa_simple_alltoall_kernel.cuh"
#include "lsa_poison_alltoall_kernel.cuh"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Buffer is allocated once to cover the largest message size in the sweep.
// Total symmetric allocation per rank: (nRanks + 1) * PERF_MAX_COUNT * sizeof(int).
// With nRanks=8 and PERF_MAX_COUNT=33554432: ~1 GB — ensure sufficient GPU memory.
#define PERF_MAX_COUNT  (1 << 25)   // 33554432 ints = 128 MB per rank-pair

// Maximum number of distinct message-size steps in the sweep (log2 of range).
#define MAX_MSG_STEPS   64

// Maximum number of CTA/channel steps in the sweep (log2 of max CTA count).
#define MAX_CTA_STEPS   32

// CTA count range and message size range are accepted as runtime arguments.
// lsaBarrierCount is set to maxCtaCount so every slot 0..maxCtaCount-1
// is pre-allocated.

// Size of the L2 flush buffer.  Must exceed the GPU L2 cache (H100: ~50 MB)
// to guarantee all relevant cache lines are evicted before each benchmark.
#define L2_FLUSH_BYTES  (128 << 20)   // 128 MB

#define WARMUP_ITERS    5

// Timed iterations scale down for large messages to keep each sweep point fast.
// Small messages need more iterations for timing resolution; large ones are slow
// enough that even 10 iterations give a stable reading.
static inline int timedIters(int count) {
    if (count <= (1 << 14)) return 500;   // <= 64 KB  : 500 iters
    if (count <= (1 << 18)) return 100;   // <= 1 MB   : 100 iters
    if (count <= (1 << 21)) return  20;   // <= 8 MB   :  20 iters
    return 10;                             //  > 8 MB   :  10 iters
}

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

#define MPICHECK(cmd) do {                                          \
    int _e = (cmd);                                                 \
    if (_e != MPI_SUCCESS) {                                        \
        fprintf(stderr, "MPI error %s:%d code=%d\n",               \
                __FILE__, __LINE__, _e);                            \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

#define CUDACHECK(cmd) do {                                         \
    cudaError_t _e = (cmd);                                         \
    if (_e != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(_e));        \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

#define NCCLCHECK(cmd) do {                                         \
    ncclResult_t _r = (cmd);                                        \
    if (_r != ncclSuccess) {                                        \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",                  \
                __FILE__, __LINE__, ncclGetErrorString(_r));        \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

// ---------------------------------------------------------------------------
// Utility: hostname hash for localRank detection
// ---------------------------------------------------------------------------
static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: mpirun -np <N> %s [options]\n"
        "\n"
        "Options:\n"
        "  -c, --min-cta  N   smallest LSA CTA count to test (power of 2, default 4)\n"
        "  -C, --max-cta  N   largest  LSA CTA count to test (power of 2, default 4)\n"
        "  -m, --min-msg  B   smallest message size in bytes  (power of 2, default 4)\n"
        "  -M, --max-msg  B   largest  message size in bytes  (power of 2, default 134217728 = 128 MB)\n"
        "  -h, --help         print this help and exit\n"
        "      --no-graph     disable CUDA graph capture (default: enabled)\n"
        "\n"
        "Both CTA and message-size ranges are swept in powers of 2 (min, 2*min, …, max).\n",
        prog);
}

static uint64_t hostHash(const char* s) {
    uint64_t h = 5381;
    for (int i = 0; s[i]; i++) h = ((h << 5) + h) + s[i];
    return h;
}

// ---------------------------------------------------------------------------
// runTimed — capture or directly record nTimed operations and return
// per-iteration latency in microseconds.
//
// useGraph=true:  captures fn() into a CUDA graph, instantiates it, then
//   records evStart/evStop around a single cudaGraphLaunch (all nTimed ops
//   execute in one launch, eliminating per-kernel launch overhead).
// useGraph=false: records evStart, calls fn() directly, records evStop.
// ---------------------------------------------------------------------------
template<typename F>
static float runTimed(bool useGraph, int nTimed,
                      cudaStream_t stream,
                      cudaEvent_t evStart, cudaEvent_t evStop,
                      F fn) {
    float ms = 0.f;
    if (useGraph) {
        cudaGraph_t     graph;
        cudaGraphExec_t exec;
        CUDACHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        fn();
        CUDACHECK(cudaStreamEndCapture(stream, &graph));
        CUDACHECK(cudaGraphInstantiate(&exec, graph, 0));
        CUDACHECK(cudaGraphDestroy(graph));
        CUDACHECK(cudaEventRecord(evStart, stream));
        CUDACHECK(cudaGraphLaunch(exec, stream));
        CUDACHECK(cudaEventRecord(evStop, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        CUDACHECK(cudaGraphExecDestroy(exec));
    } else {
        CUDACHECK(cudaEventRecord(evStart, stream));
        fn();
        CUDACHECK(cudaEventRecord(evStop, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
    }
    CUDACHECK(cudaEventElapsedTime(&ms, evStart, evStop));
    return ms * 1000.f / nTimed;
}

// ---------------------------------------------------------------------------
// fill_buffer — initialise a flat int array with a uniform value.
// Used to fill the shared LSA / UB input buffer (sendBuff) with myRank so
// that every per-dest slab carries the sender's rank index.
// ---------------------------------------------------------------------------
__global__ void fill_buffer(int* buf, int val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

// poison_sentinel_buffer — set .w of each uint4 slot to LSA_SENTINEL_POISON.
// Cannot use cudaMemset since 0xFFFAFFFA is not a uniform byte pattern.
__global__ void poison_sentinel_buffer(uint4* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i].w = LSA_SENTINEL_POISON;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // ------------------------------------------------------------------
    // 1. MPI setup
    // ------------------------------------------------------------------
    MPICHECK(MPI_Init(&argc, &argv));

    int myRank, nRanks;
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    if (nRanks < 2) {
        if (myRank == 0)
            fprintf(stderr, "This sample requires at least 2 MPI ranks.\n");
        MPI_Finalize();
        return 1;
    }

    // Parse command-line options with getopt_long.
    int minCtaCount  = 4;
    int maxCtaCount  = 4;
    long minMsgBytes = 4;
    long maxMsgBytes = (long)PERF_MAX_COUNT * sizeof(int);  // 128 MB default
    bool useGraph    = true;

    static const struct option longOpts[] = {
        { "min-cta",  required_argument, nullptr, 'c' },
        { "max-cta",  required_argument, nullptr, 'C' },
        { "min-msg",  required_argument, nullptr, 'm' },
        { "max-msg",  required_argument, nullptr, 'M' },
        { "help",     no_argument,       nullptr, 'h' },
        { "no-graph", no_argument,       nullptr, 'G' },
        { nullptr,    0,                 nullptr,  0  }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:C:m:M:h", longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'c': minCtaCount  = atoi(optarg);  break;
            case 'C': maxCtaCount  = atoi(optarg);  break;
            case 'm': minMsgBytes  = atol(optarg);  break;
            case 'M': maxMsgBytes  = atol(optarg);  break;
            case 'G': useGraph     = false;          break;
            case 'h':
                if (myRank == 0) printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            default:
                if (myRank == 0) printUsage(argv[0]);
                MPI_Finalize();
                return 1;
        }
    }

    auto isPow2L = [](long n){ return n >= 1 && (n & (n - 1)) == 0; };
    auto isPow2  = [](int  n){ return n >= 1 && (n & (n - 1)) == 0; };

    if (!isPow2(minCtaCount) || !isPow2(maxCtaCount) || minCtaCount > maxCtaCount) {
        if (myRank == 0)
            fprintf(stderr, "Error: --min-cta and --max-cta must be powers of 2"
                            " with min-cta <= max-cta.\n");
        MPI_Finalize();
        return 1;
    }
    if (!isPow2L(minMsgBytes) || !isPow2L(maxMsgBytes) || minMsgBytes > maxMsgBytes) {
        if (myRank == 0)
            fprintf(stderr, "Error: --min-msg and --max-msg must be powers of 2"
                            " with min-msg <= max-msg.\n");
        MPI_Finalize();
        return 1;
    }
    if (maxMsgBytes > (long)PERF_MAX_COUNT * (long)sizeof(int)) {
        if (myRank == 0)
            fprintf(stderr, "Error: --max-msg exceeds compiled-in PERF_MAX_COUNT (%ld bytes).\n",
                    (long)PERF_MAX_COUNT * (long)sizeof(int));
        MPI_Finalize();
        return 1;
    }

    // Determine localRank (GPU index on this host) via hostname hashing.
    uint64_t hostHashes[nRanks];
    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    for (int i = 0; i < (int)sizeof(hostname) && hostname[i]; i++)
        if (hostname[i] == '.') { hostname[i] = '\0'; break; }

    hostHashes[myRank] = hostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           hostHashes, sizeof(uint64_t), MPI_BYTE,
                           MPI_COMM_WORLD));
    int localRank = 0;
    for (int p = 0; p < myRank; p++)
        if (hostHashes[p] == hostHashes[myRank]) localRank++;

    printf("[Rank %d/%d] localRank=%d hostname=%s\n",
           myRank, nRanks, localRank, hostname);
    fflush(stdout);

    // ------------------------------------------------------------------
    // 2. Select GPU
    // ------------------------------------------------------------------
    CUDACHECK(cudaSetDevice(localRank));

    // ------------------------------------------------------------------
    // 3. Initialize NCCL communicator
    // ------------------------------------------------------------------
    ncclUniqueId id;
    if (myRank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));

    // ------------------------------------------------------------------
    // 4. Allocate shared buffers
    //
    //    recvBuff/recvBuff2/recvBuff3 — triple symmetric buffers for poison
    //              AllToAll (triple-buffering with separate clear pass).
    //              recvBuff is also used as the output for LSA and ncclAlltoAll
    //              benchmarks.
    //    sendBuff — local input buffer, nRanks * PERF_MAX_COUNT ints.
    //              Filled with myRank; shared by NCCL, LSA, and poison.
    // ------------------------------------------------------------------
    const size_t bufLen   = (size_t)nRanks * PERF_MAX_COUNT;
    const size_t bufBytes = bufLen * sizeof(int);

    int* recvBuff;
    NCCLCHECK(ncclMemAlloc((void**)&recvBuff, bufBytes));

    ncclWindow_t memWin;
    NCCLCHECK(ncclCommWindowRegister(comm, recvBuff, bufBytes,
                                     &memWin, NCCL_WIN_COLL_SYMMETRIC));

    // Two additional symmetric buffers for poison AllToAll triple-buffering.
    // Three buffers rotate each iteration (iter%3). The kernel re-poisons
    // bufs[(iter+2)%3] after polling bufs[iter%3], leaving the recv buffer
    // intact for the caller to read.  By the time a buffer is reused
    // (3 iterations later), it was cleared 2 iterations prior.
    int* recvBuff2;
    ncclWindow_t memWin2;
    NCCLCHECK(ncclMemAlloc((void**)&recvBuff2, bufBytes));
    NCCLCHECK(ncclCommWindowRegister(comm, recvBuff2, bufBytes,
                                     &memWin2, NCCL_WIN_COLL_SYMMETRIC));

    int* recvBuff3;
    ncclWindow_t memWin3;
    NCCLCHECK(ncclMemAlloc((void**)&recvBuff3, bufBytes));
    NCCLCHECK(ncclCommWindowRegister(comm, recvBuff3, bufBytes,
                                     &memWin3, NCCL_WIN_COLL_SYMMETRIC));

    int* sendBuff;                   // local input shared by NCCL, LSA, and poison (not peer-accessible)
    NCCLCHECK(ncclMemAlloc((void**)&sendBuff,
                           (size_t)nRanks * PERF_MAX_COUNT * sizeof(int)));

    // L2 flush buffer: written before each kernel's warmup to evict stale cache
    // lines and give every benchmark section the same cold-cache starting point.
    char* flushBuf;
    CUDACHECK(cudaMalloc((void**)&flushBuf, L2_FLUSH_BYTES));

    // ------------------------------------------------------------------
    // 5. Create LSA device communicator
    //
    //    lsaBarrierCount = 1  — one CTA-scoped barrier slot (slot 0).
    //    lsaMultimem     = false — unicast NVLink only.
    //
    //    ncclDevComm is passed by value to the kernel; CUDA copies it into
    //    the kernel parameter area.  All device pointers inside remain valid.
    // ------------------------------------------------------------------
    ncclDevComm_t devComm;
    {
        ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount = maxCtaCount;
        reqs.lsaMultimem     = false;
        NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    }

    int lsaSize = devComm.lsaSize;

    // Thread count is chosen dynamically per message size (see perf loop).
    // Ideal minimum: lsaSize warps (one per destination), but CUDA caps
    // blockDim.x at 1024 (32 warps).  When lsaSize > 32 each warp covers
    // multiple destinations via the outer destOff loop in the kernel.
    // Maximum: 1024 threads (32 warps) for large messages where extra warps
    // per destination improve HBM read parallelism.
    const int lsaMaxThreads = 1024;
    const int lsaMinThreads = min(lsaSize * 32, lsaMaxThreads);

    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (myRank == 0)
        printf("\nLSA team size: %d rank(s) per NVLink island\n\n", lsaSize);
    fflush(stdout);

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Fill sendBuff with myRank — shared source buffer for LSA, NCCL, and poison.
    // sendBuff[dest * count + i] = myRank for all dest/i,
    // so after AllToAll: recvBuff[r * count + i] == r (correctness check holds).
    {
        int fillN = (int)((size_t)nRanks * PERF_MAX_COUNT);
        fill_buffer<<<(fillN + 255) / 256, 256, 0, stream>>>(sendBuff, myRank, fillN);
    }
    CUDACHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------
    // 6. Correctness tests
    //
    //    sendBuff is filled with myRank (== lsaRank when all ranks share one
    //    NVLink island), so after AllToAll recvBuff[r*count+i].x == r.
    //
    //    TEST_COUNT uint4 elements per rank-pair — large enough to exercise
    //    multi-element paths and catch off-by-one indexing bugs.
    // ------------------------------------------------------------------
    static const int TEST_COUNT = 256;   // uint4 elements per rank-pair
    static const int TEST_ITERS = 10;    // poison iterations

    int pass = 1;

    // ---- 6a. lsa_simple_alltoall_kernel ----
    {
        CUDACHECK(cudaMemset(recvBuff, 0xff, (size_t)lsaSize * TEST_COUNT * sizeof(uint4)));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        lsa_simple_alltoall_kernel<<<maxCtaCount, lsaMinThreads, 0, stream>>>(
            (const uint4*)sendBuff, (uint4*)recvBuff, memWin, TEST_COUNT, devComm);
        CUDACHECK(cudaStreamSynchronize(stream));

        size_t checkBytes = (size_t)lsaSize * TEST_COUNT * sizeof(uint4);
        std::vector<uint4> h(lsaSize * TEST_COUNT);
        CUDACHECK(cudaMemcpy(h.data(), recvBuff, checkBytes, cudaMemcpyDeviceToHost));

        for (int r = 0; r < lsaSize && pass; r++) {
            for (int i = 0; i < TEST_COUNT && pass; i++) {
                uint4 slot = h[r * TEST_COUNT + i];
                if ((int)slot.x != r) {
                    fprintf(stderr,
                        "[Rank %d] LSA FAIL: recvBuff[r=%d][i=%d].x = %d, expected %d\n",
                        myRank, r, i, (int)slot.x, r);
                    pass = 0;
                }
            }
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (pass && myRank == 0) printf("[Correctness LSA     ] PASS\n");
        fflush(stdout);
    }

    // ---- 6b. lsa_poison_alltoall_kernel ----
    //
    // Run TEST_ITERS iterations with 3-buffer rotation.  After the last
    // iteration verify three things:
    //   (1) Data correct: recvBuff[r*count+i].x == r.
    //   (2) recvBuff.w != POISON: kernel left the recv buffer intact.
    //   (3) clearBuf.w == POISON: kernel re-poisoned the clear buffer.
    {
        uint4*       sentBufs[3] = {(uint4*)recvBuff, (uint4*)recvBuff2, (uint4*)recvBuff3};
        ncclWindow_t sentWins[3] = {memWin, memWin2, memWin3};
        int          sentSlots   = lsaSize * TEST_COUNT;
        int          poisonBlks  = (sentSlots + 255) / 256;

        // Pre-poison all three buffers.
        for (int b = 0; b < 3; b++)
            poison_sentinel_buffer<<<poisonBlks, 256, 0, stream>>>(sentBufs[b], sentSlots);

        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

        for (int si = 0; si < TEST_ITERS; si++)
            lsa_poison_alltoall_kernel<<<maxCtaCount, lsaMinThreads, 0, stream>>>(
                (const uint4*)sendBuff,
                sentBufs[si % 3], sentWins[si % 3],
                TEST_COUNT, devComm,
                sentBufs[(si + 2) % 3], /*skip_barrier=*/si >= 2);
        CUDACHECK(cudaStreamSynchronize(stream));

        // Which buffer received the last iteration?
        int lastIdx   = (TEST_ITERS - 1) % 3;
        int clearIdx  = (TEST_ITERS - 1 + 2) % 3;  // == (lastIdx + 2) % 3

        size_t checkBytes = (size_t)sentSlots * sizeof(uint4);
        std::vector<uint4> hRecv(sentSlots), hClear(sentSlots);
        CUDACHECK(cudaMemcpy(hRecv.data(),  sentBufs[lastIdx],  checkBytes, cudaMemcpyDeviceToHost));
        CUDACHECK(cudaMemcpy(hClear.data(), sentBufs[clearIdx], checkBytes, cudaMemcpyDeviceToHost));

        for (int r = 0; r < lsaSize && pass; r++) {
            for (int i = 0; i < TEST_COUNT && pass; i++) {
                uint4 slot = hRecv[r * TEST_COUNT + i];

                // (1) Data correct.
                if ((int)slot.x != r) {
                    fprintf(stderr,
                        "[Rank %d] POISON FAIL (data): recvBuff[r=%d][i=%d].x = %d, expected %d\n",
                        myRank, r, i, (int)slot.x, r);
                    pass = 0;
                }
                // (2) recvBuff.w not poisoned (kernel left recv buffer intact).
                if (slot.w == LSA_SENTINEL_POISON) {
                    fprintf(stderr,
                        "[Rank %d] POISON FAIL (intact): recvBuff[r=%d][i=%d].w == POISON"
                        " (kernel should not have cleared recvBuff)\n",
                        myRank, r, i);
                    pass = 0;
                }
            }
        }
        // (3) clearBuf.w == POISON (kernel re-poisoned it).
        for (int r = 0; r < lsaSize && pass; r++) {
            for (int i = 0; i < TEST_COUNT && pass; i++) {
                if (hClear[r * TEST_COUNT + i].w != LSA_SENTINEL_POISON) {
                    fprintf(stderr,
                        "[Rank %d] POISON FAIL (clear): clearBuf[r=%d][i=%d].w != POISON"
                        " (kernel should have re-poisoned clearBuf)\n",
                        myRank, r, i);
                    pass = 0;
                }
            }
        }
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
        if (pass && myRank == 0) printf("[Correctness Poison  ] PASS\n");
        fflush(stdout);
    }

    if (pass && myRank == 0) printf("\n");
    fflush(stdout);

    // ------------------------------------------------------------------
    // 7. Performance sweep
    //
    //    For each step N in [minCtaCount..maxCtaCount] (powers of 2):
    //      - LSA kernel launched with N CTAs
    //      - ncclAlltoAll run on a communicator created with nChannels = N
    //    Both are measured at each message size and printed side by side.
    //
    //    ncclConfig_t.minCTAs/maxCTAs (set equal) pins the number of
    //    channels NCCL uses — this is the closest knob that maps to the
    //    LSA CTA count.
    //
    //    Communicators are pre-created before the timed loop to avoid
    //    including init overhead in measurements.
    //
    //    Bandwidth follows the nccl-tests algbw convention:
    //      LSA  : lsaSize * count * sizeof(int) per rank
    //      NCCL : nRanks  * count * sizeof(int) per rank
    // ------------------------------------------------------------------

    // ---- 7a. Pre-create one ncclComm_t per channel count ----
    ncclComm_t ncomms[MAX_CTA_STEPS];
    int ncommCount = 0;
    if (myRank == 0) {
        printf("Initializing per-channel-count communicators...\n\n");
        fflush(stdout);
    }
    for (int n = minCtaCount; n <= maxCtaCount; n *= 2) {
        ncclUniqueId uid;
        if (myRank == 0) NCCLCHECK(ncclGetUniqueId(&uid));
        MPICHECK(MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD));
        ncclConfig_t ncfg  = NCCL_CONFIG_INITIALIZER;
        ncfg.blocking      = 1;
        ncfg.minCTAs       = n;
        ncfg.maxCTAs       = n;
        NCCLCHECK(ncclCommInitRankConfig(&ncomms[ncommCount++], nRanks, uid, myRank, &ncfg));
    }

    cudaEvent_t evStart, evStop;
    CUDACHECK(cudaEventCreate(&evStart));
    CUDACHECK(cudaEventCreate(&evStop));

    // ---- 7b. Sweep over CTA / channel counts ----
    int commIdx = 0;
    for (int nBlocks = minCtaCount; nBlocks <= maxCtaCount; nBlocks *= 2, commIdx++) {
        ncclComm_t ncomm = ncomms[commIdx];

        if (myRank == 0) {
            printf("=== CTAs / nChannels: %d ===\n", nBlocks);
            printf("%-12s  %-14s  %-14s  %-14s  %-14s  %-16s  %-16s\n",
                   "msg/pair",
                   "lsa_simple lat", "lsa_simple BW", "lsa_poison lat", "lsa_poison BW",
                   "NCCL_baseline lat", "NCCL_baseline BW");
            printf("%-12s  %-14s  %-14s  %-14s  %-14s  %-16s  %-16s\n",
                   "(bytes)",
                   "(us)", "(GB/s)", "(us)", "(GB/s)", "(us)", "(GB/s)");
            printf("-----------------------------------------------------------------------------------------------------------\n");
            fflush(stdout);
        }

        for (long msgBytes = minMsgBytes; msgBytes <= maxMsgBytes; msgBytes *= 2) {
            int count  = (int)(msgBytes / sizeof(int));
            if (count < 1) count = 1;
            int nTimed = timedIters(count);
            // UB transfers uint4 (16 B) units; skip if message < 16 B.
            int nlines = (int)(msgBytes / sizeof(uint4));

            // Use minimum threads (one warp per dest) until the per-block
            // per-dest slice exceeds 512 uint4 (8 KB).  Below that threshold
            // the copy is sub-µs and the 2 µs barrier saving from fewer warps
            // outweighs the lost NVLink parallelism.  Above it, 4 warps per
            // dest are needed to keep the NVLink pipes saturated.
            int lsaBlockDim = (nlines > 512 * nBlocks) ? lsaMaxThreads : lsaMinThreads;

            // ---- ncclAlltoAll with nChannels = nBlocks ----
            CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
            CUDACHECK(cudaMemset(recvBuff, 0xff,
                                 (size_t)nRanks * count * sizeof(int)));
            for (int i = 0; i < WARMUP_ITERS; i++)
                NCCLCHECK(ncclAlltoAll(sendBuff, recvBuff, count, ncclInt32, ncomm, stream));
            CUDACHECK(cudaStreamSynchronize(stream));

            MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
            float ncclLat = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                for (int i = 0; i < nTimed; i++)
                    NCCLCHECK(ncclAlltoAll(sendBuff, recvBuff, count, ncclInt32, ncomm, stream));
            });
            MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &ncclLat, 1, MPI_FLOAT,
                                   MPI_MAX, MPI_COMM_WORLD));

            // ---- LSA kernel with nBlocks CTAs ----
            // Uses uint4 (128-bit) transfers; skip if message < 16 B.
            float lsaLat = 0.f;
            if (nlines > 0) {
                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                CUDACHECK(cudaMemset(recvBuff, 0xff,
                                     (size_t)lsaSize * nlines * sizeof(uint4)));
                for (int i = 0; i < WARMUP_ITERS; i++)
                    lsa_simple_alltoall_kernel<<<nBlocks, lsaBlockDim, 0, stream>>>(
                        (const uint4*)sendBuff, (uint4*)recvBuff, memWin, nlines, devComm);
                CUDACHECK(cudaStreamSynchronize(stream));

                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                lsaLat = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                    for (int i = 0; i < nTimed; i++)
                        lsa_simple_alltoall_kernel<<<nBlocks, lsaBlockDim, 0, stream>>>(
                            (const uint4*)sendBuff, (uint4*)recvBuff, memWin, nlines, devComm);
                });
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &lsaLat, 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            // ---- Poison AllToAll (triple-buffer, separate clear pass) ----
            //
            // Three symmetric windows rotate each iteration (iter % 3).
            // At iteration si:
            //   recvBuff = bufs[si % 3]        — receives incoming data here
            //   clearBuf = bufs[(si+2) % 3]    — re-poisons the buffer used 2
            //                                    iterations ago
            // recvBuff is left intact so the caller can read the result after
            // the kernel returns.  A peer at most 1 iter ahead writes to
            // bufs[(si+1)%3], so clearBuf is never written concurrently.
            // skip_barrier=true in steady state (iter >= 2).
            float sentLat = 0.f;
            if (nlines > 0) {
                uint4*       sentBufs[3] = {(uint4*)recvBuff, (uint4*)recvBuff2, (uint4*)recvBuff3};
                ncclWindow_t sentWins[3] = {memWin, memWin2, memWin3};

                // Pre-poison all three buffers before the first call.
                int sentSlots    = lsaSize * nlines;
                int poisonBlocks = (sentSlots + 255) / 256;
                for (int b = 0; b < 3; b++)
                    poison_sentinel_buffer<<<poisonBlocks, 256, 0, stream>>>(sentBufs[b], sentSlots);

                // Warmup.
                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                int sentIter = 0;
                for (int w = 0; w < WARMUP_ITERS; w++, sentIter++)
                    lsa_poison_alltoall_kernel<<<nBlocks, lsaBlockDim, 0, stream>>>(
                        (const uint4*)sendBuff,
                        sentBufs[sentIter % 3], sentWins[sentIter % 3],
                        nlines, devComm,
                        sentBufs[(sentIter + 2) % 3], /*skip_barrier=*/sentIter >= 2);
                CUDACHECK(cudaStreamSynchronize(stream));

                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                int sentIterBase = sentIter;
                sentLat = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                    int si = sentIterBase;
                    for (int i = 0; i < nTimed; i++, si++)
                        lsa_poison_alltoall_kernel<<<nBlocks, lsaBlockDim, 0, stream>>>(
                            (const uint4*)sendBuff,
                            sentBufs[si % 3], sentWins[si % 3],
                            nlines, devComm,
                            sentBufs[(si + 2) % 3], /*skip_barrier=*/true);
                });
                sentIter = sentIterBase + nTimed;
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &sentLat, 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            if (myRank == 0) {
                double ncclBw = (double)nRanks  * count * sizeof(int) / (ncclLat * 1e-6) / 1e9;
                if (nlines > 0) {
                    double lsaBw  = (double)lsaSize * nlines * sizeof(uint4) / (lsaLat  * 1e-6) / 1e9;
                    double sentBw = (double)lsaSize * nlines * sizeof(uint4) / (sentLat * 1e-6) / 1e9;
                    printf("%-12ld  %-14.2f  %-14.3f  %-14.2f  %-14.3f  %-16.2f  %-16.3f\n",
                           msgBytes, lsaLat, lsaBw, sentLat, sentBw, ncclLat, ncclBw);
                } else {
                    printf("%-12ld  %-14s  %-14s  %-14s  %-14s  %-16.2f  %-16.3f\n",
                           msgBytes, "N/A", "N/A", "N/A", "N/A", ncclLat, ncclBw);
                }
                fflush(stdout);
            }
        }
        if (myRank == 0) printf("\n");
    }

    // ------------------------------------------------------------------
    // 8. Cleanup
    // ------------------------------------------------------------------
    CUDACHECK(cudaEventDestroy(evStart));
    CUDACHECK(cudaEventDestroy(evStop));
    CUDACHECK(cudaStreamDestroy(stream));
    for (int i = 0; i < ncommCount; i++) {
        NCCLCHECK(ncclCommFinalize(ncomms[i]));
        NCCLCHECK(ncclCommDestroy(ncomms[i]));
    }
    NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
    NCCLCHECK(ncclCommWindowDeregister(comm, memWin));
    NCCLCHECK(ncclCommWindowDeregister(comm, memWin2));
    NCCLCHECK(ncclCommWindowDeregister(comm, memWin3));
    CUDACHECK(cudaFree(flushBuf));
    NCCLCHECK(ncclMemFree(recvBuff));
    NCCLCHECK(ncclMemFree(recvBuff2));
    NCCLCHECK(ncclMemFree(recvBuff3));
    NCCLCHECK(ncclMemFree(sendBuff));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Finalize());

    return pass ? 0 : 1;
}
