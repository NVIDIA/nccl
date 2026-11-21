/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 *************************************************************************/

// main.cu
//
// Host-side driver for the custom AllReduce sample kernels.
//
// What it demonstrates:
//   1. Initializing NCCL and creating an ncclDevComm (lsaMultimem=true when
//      NVLink multicast is available).
//   2. Registering symmetric buffers as ncclWindows with NCCL_WIN_COLL_SYMMETRIC.
//   3. Passing ncclDevComm and ncclWindow by value to custom CUDA kernels.
//   4. oneshot_mc_allreduce_kernel   — AllReduce via multimem.ld_reduce.
//   5. twoshot_mc_simple_allreduce_kernel — RS+AG via multimem.ld_reduce + multimem.st.
//   6. twoshot_mc_poison_allreduce_kernel — twoshot with Lamport sentinel completion.
//   7. oneshot_lsa_push_allreduce_kernel:    one-shot push over LSA unicast
//      pointers with an LSA barrier (PCIe P2P, no multicast needed).
//   8. oneshot_lsa_push_ll_allreduce_kernel: same push with LL flags
//      (ncclLLA2ASession) instead of a barrier.
//   9. ncclAllReduce baseline.
//  10. Side-by-side performance comparison across CTA counts and message sizes.
//
// The three MC kernels require sm >= 9.0 (Hopper / Blackwell) for the
// multimem PTX instructions.  The driver checks device capability at startup
// and skips MC kernels on unsupported hardware.  The two LSA push kernels run
// on any GPUs whose communicator reports Device API support with a single LSA
// team, including PCIe-only systems with CUDA P2P.
//
// Only public NCCL headers are used:
//   nccl.h          — ncclComm, ncclWindow, ncclMemAlloc, ncclCommWindowRegister
//   nccl_device.h   — ncclDevComm, ncclDevCommCreate/Destroy, ncclDevCommRequirements_t
//
// Run with MPI:
//   mpirun -np <N> ./allreduce [options]
//
// Options:
//   -c, --min-cta  N   smallest CTA count (power of 2, default 1)
//   -C, --max-cta  N   largest  CTA count (power of 2, default 1)
//   -m, --min-msg  B   smallest message size in bytes (default 16)
//   -M, --max-msg  B   largest  message size in bytes (default 128 MB)
//       --no-graph     disable CUDA graph capture (default: enabled)
//   -d, --dtype    T   float | half | bf16: element type of the push kernels
//                      and the NCCL baseline in the sweep (default float;
//                      MC kernels are float only and are skipped otherwise)
//   -h, --help         print this help and exit
//
// Prerequisites:
//   - At least 2 MPI ranks.
//   - MC kernels: NVLink-connected GPUs with multicast support,
//     CUDA >= 12.2, GPU compute capability >= 9.0.
//   - LSA push kernels: Device API support with a single LSA team
//     (on PCIe: working CUDA P2P between all GPUs).

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <getopt.h>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>

#include "nccl.h"
#include "nccl_device.h"

#include "oneshot_mc_allreduce_kernel.cuh"
#include "twoshot_mc_simple_allreduce_kernel.cuh"
#include "twoshot_mc_poison_allreduce_kernel.cuh"
#include "oneshot_lsa_push_allreduce_kernel.cuh"
#include "oneshot_lsa_push_ll_allreduce_kernel.cuh"

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

// Maximum float4 elements per buffer; sets the default max message size.
// 1 << 23 float4 = 8 M * 16 B = 128 MB.
#define PERF_MAX_NLINES  (1 << 23)

// L2 flush buffer size.  Must exceed the GPU L2 cache (H100: ~50 MB).
#define L2_FLUSH_BYTES   (128 << 20)

#define WARMUP_ITERS     5

// One-shot push kernels copy the whole message to every peer, so they are
// only measured up to this size.  It also sizes their scratch buffers.
#define PUSH_MAX_BYTES   (1L << 20)

// Threads per CTA of the push kernels (tuned on 2 x T4, see README).
#define LSA_PUSH_THREADS     512
#define LSA_PUSH_LL_THREADS  512

// Correctness test: number of twoshot_mc_poison iterations.
static const int TEST_ITERS = 50;

// Scale timed iterations with message size for stable but fast sweeps.
static inline int timedIters(long nlines) {
    long bytes = nlines * (long)sizeof(float4);
    if (bytes <= (1 << 14)) return 500;
    if (bytes <= (1 << 18)) return 100;
    if (bytes <= (1 << 21)) return  20;
    return 10;
}

// ---------------------------------------------------------------------------
// Error-checking macros
// ---------------------------------------------------------------------------

#define MPICHECK(cmd) do {                                              \
    int _e = (cmd);                                                     \
    if (_e != MPI_SUCCESS) {                                            \
        fprintf(stderr, "MPI error %s:%d code=%d\n",                   \
                __FILE__, __LINE__, _e);                                \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#define CUDACHECK(cmd) do {                                             \
    cudaError_t _e = (cmd);                                             \
    if (_e != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d '%s'\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

#define NCCLCHECK(cmd) do {                                             \
    ncclResult_t _r = (cmd);                                            \
    if (_r != ncclSuccess) {                                            \
        fprintf(stderr, "NCCL error %s:%d '%s'\n",                      \
                __FILE__, __LINE__, ncclGetErrorString(_r));            \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

static void printUsage(const char* prog) {
    fprintf(stderr,
        "Usage: mpirun -np <N> %s [options]\n"
        "\n"
        "Options:\n"
        "  -c, --min-cta  N   smallest CTA count (power of 2, default 1)\n"
        "  -C, --max-cta  N   largest  CTA count (power of 2, default 1)\n"
        "  -m, --min-msg  B   smallest message size in bytes (default 16)\n"
        "  -M, --max-msg  B   largest  message size in bytes (default %ld)\n"
        "      --no-graph     disable CUDA graph capture (default: enabled)\n"
        "  -d, --dtype    T   float | half | bf16 for push kernels + NCCL (default float)\n"
        "  -h, --help         print this help and exit\n",
        prog, (long)PERF_MAX_NLINES * (long)sizeof(float4));
}

static uint64_t hostHash(const char* s) {
    uint64_t h = 5381;
    for (int i = 0; s[i]; i++) h = ((h << 5) + h) + (uint8_t)s[i];
    return h;
}

// runTimed — capture or directly record nTimed kernel launches and return
// per-iteration latency in microseconds.
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
        // Upload the graph before timing.  Otherwise the first launch also
        // pays the upload of every node, which inflates the per-iteration
        // time, most for kernels with large parameter blocks (ncclAllReduce).
        CUDACHECK(cudaGraphUpload(exec, stream));
        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
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
    return ms * 1000.f / (float)nTimed;
}

// fill_float — initialize a flat float buffer with a uniform value.
__global__ void fill_float(float* buf, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = val;
}

// Element types of the push kernels.  Data moves as 16-byte lines for every
// type; only the add differs (lsa_push_reduce.cuh).
enum { DT_FLOAT = 0, DT_HALF = 1, DT_BF16 = 2, DT_COUNT = 3 };
static const char*    dtypeName(int d) { return d == DT_HALF ? "half" : d == DT_BF16 ? "bf16" : "float"; }
static size_t         dtypeSize(int d) { return d == DT_FLOAT ? 4 : 2; }
static ncclDataType_t dtypeNccl(int d) { return d == DT_HALF ? ncclHalf : d == DT_BF16 ? ncclBfloat16 : ncclFloat; }

template <typename T> __device__ __forceinline__ T     cvtFromFloat(float v);
template <> __device__ __forceinline__ float           cvtFromFloat<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half          cvtFromFloat<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16   cvtFromFloat<__nv_bfloat16>(float v) { return __float2bfloat16(v); }
__device__ __forceinline__ float cvtToFloat(float v)         { return v; }
__device__ __forceinline__ float cvtToFloat(__half v)        { return __half2float(v); }
__device__ __forceinline__ float cvtToFloat(__nv_bfloat16 v) { return __bfloat162float(v); }

// fill_pattern — buf[i] = base * 2^(i % 8) (shift 0 gives a uniform fill).
// Powers of two keep every partial sum exact in half and bf16 for small rank
// counts, so the 16-bit correctness test can compare bit-exactly, and the
// pattern differs across the 8 lanes of a 16-byte line.
template <typename T>
__global__ void fill_pattern(T* buf, float base, int shift, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = cvtFromFloat<T>(base * (float)(1 << ((i & 7) * shift)));
}

// to_float — widen a typed buffer to float for host-side checks.
template <typename T>
__global__ void to_float(const T* in, float* out, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = cvtToFloat(in[i]);
}

template <typename T>
static void fillTyped(void* buf, float base, int shift, size_t n, cudaStream_t s) {
    fill_pattern<T><<<(unsigned)((n + 255) / 256), 256, 0, s>>>((T*)buf, base, shift, n);
}
static void fillDtype(int dt, void* buf, float base, int shift, size_t bytes, cudaStream_t s) {
    const size_t n = bytes / dtypeSize(dt);
    if (dt == DT_HALF)      fillTyped<__half>(buf, base, shift, n, s);
    else if (dt == DT_BF16) fillTyped<__nv_bfloat16>(buf, base, shift, n, s);
    else                    fillTyped<float>(buf, base, shift, n, s);
}

// launchPushTyped — launch one push kernel instantiated for element type T.
template <typename T>
static void launchPushTyped(int ll, int nBlocks, cudaStream_t stream,
                            const float4* in, float4* out, size_t nlines,
                            ncclLLA2AHandle_t pushLL, ncclWindow_t pushWin,
                            size_t slotLines, int slot, ncclDevComm_t devComm) {
    if (ll)
        oneshot_lsa_push_ll_allreduce_kernel<T>
            <<<nBlocks, LSA_PUSH_LL_THREADS, 0, stream>>>(in, out, pushLL, nlines, devComm);
    else
        oneshot_lsa_push_allreduce_kernel<T>
            <<<nBlocks, LSA_PUSH_THREADS, 0, stream>>>(in, out, pushWin, slotLines, nlines,
                                                       slot, devComm);
}

// poison_float4_buffer — set .w of each float4 slot to LSA_POISON.
// Cannot use cudaMemset since 0xFFFAFFFA is not a uniform byte pattern.
__global__ void poison_float4_buffer(float4* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i].w = __int_as_float((int)LSA_POISON);
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

    // ------------------------------------------------------------------
    // 2. Argument parsing
    // ------------------------------------------------------------------
    int  minCtaCount = 1;
    int  maxCtaCount = 1;
    long minMsgBytes = (long)sizeof(float4);
    long maxMsgBytes = (long)PERF_MAX_NLINES * (long)sizeof(float4);
    bool useGraph    = true;
    int  dtype       = DT_FLOAT;

    static const struct option longOpts[] = {
        { "min-cta",  required_argument, nullptr, 'c' },
        { "max-cta",  required_argument, nullptr, 'C' },
        { "min-msg",  required_argument, nullptr, 'm' },
        { "max-msg",  required_argument, nullptr, 'M' },
        { "no-graph", no_argument,       nullptr, 'G' },
        { "dtype",    required_argument, nullptr, 'd' },
        { "help",     no_argument,       nullptr, 'h' },
        { nullptr,    0,                 nullptr,  0  }
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "c:C:m:M:d:h", longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'c': minCtaCount = atoi(optarg); break;
            case 'C': maxCtaCount = atoi(optarg); break;
            case 'm': minMsgBytes = atol(optarg); break;
            case 'M': maxMsgBytes = atol(optarg); break;
            case 'G': useGraph    = false;         break;
            case 'd':
                if      (!strcmp(optarg, "float")) dtype = DT_FLOAT;
                else if (!strcmp(optarg, "half"))  dtype = DT_HALF;
                else if (!strcmp(optarg, "bf16"))  dtype = DT_BF16;
                else {
                    if (myRank == 0) fprintf(stderr, "Error: --dtype must be float, half or bf16.\n");
                    MPI_Finalize(); return 1;
                }
                break;
            case 'h': if (myRank == 0) printUsage(argv[0]); MPI_Finalize(); return 0;
            default:  if (myRank == 0) printUsage(argv[0]); MPI_Finalize(); return 1;
        }
    }

    auto isPow2 = [](int n) { return n >= 1 && (n & (n - 1)) == 0; };
    if (!isPow2(minCtaCount) || !isPow2(maxCtaCount) || minCtaCount > maxCtaCount) {
        if (myRank == 0)
            fprintf(stderr, "Error: --min-cta and --max-cta must be powers of 2"
                            " with min-cta <= max-cta.\n");
        MPI_Finalize(); return 1;
    }
    if (minMsgBytes < (long)sizeof(float4) || maxMsgBytes < (long)sizeof(float4) ||
        minMsgBytes > maxMsgBytes ||
        maxMsgBytes > (long)PERF_MAX_NLINES * (long)sizeof(float4)) {
        if (myRank == 0)
            fprintf(stderr, "Error: message sizes must be >= 16 bytes and <= %ld bytes.\n",
                    (long)PERF_MAX_NLINES * (long)sizeof(float4));
        MPI_Finalize(); return 1;
    }

    // ------------------------------------------------------------------
    // 3. GPU + NCCL setup
    // ------------------------------------------------------------------
    uint64_t hostHashes[nRanks];
    char hostname[1024];
    gethostname(hostname, sizeof(hostname));
    for (int i = 0; i < (int)sizeof(hostname) && hostname[i]; i++)
        if (hostname[i] == '.') { hostname[i] = '\0'; break; }
    hostHashes[myRank] = hostHash(hostname);
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                           hostHashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    int localRank = 0;
    for (int p = 0; p < myRank; p++)
        if (hostHashes[p] == hostHashes[myRank]) localRank++;

    printf("[Rank %d/%d] localRank=%d hostname=%s\n", myRank, nRanks, localRank, hostname);
    fflush(stdout);

    CUDACHECK(cudaSetDevice(localRank));

    // MC kernels require sm >= 9.0 (Hopper / Blackwell).
    int smMajor = 0, smMinor = 0;
    CUDACHECK(cudaDeviceGetAttribute(&smMajor, cudaDevAttrComputeCapabilityMajor, localRank));
    CUDACHECK(cudaDeviceGetAttribute(&smMinor, cudaDevAttrComputeCapabilityMinor, localRank));
    bool hasMC = (smMajor >= 9);
    if (!hasMC && myRank == 0)
        fprintf(stderr, "Warning: GPU sm_%d%d < 9.0 — MC kernels will be skipped.\n",
                smMajor, smMinor);

    ncclUniqueId id;
    if (myRank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPICHECK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t comm;
    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.blocking = 1;
    NCCLCHECK(ncclCommInitRankConfig(&comm, nRanks, id, myRank, &config));

    // The LSA push kernels need Device API support with one LSA team (all
    // ranks reachable by load/store).  On PCIe this means working CUDA P2P.
    ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;
    NCCLCHECK(ncclCommQueryProperties(comm, &props));
    bool hasLsa = props.deviceApiSupport && props.nLsaTeams == 1;
    hasMC = hasMC && hasLsa && props.multimemSupport;
    if (!hasLsa && myRank == 0)
        fprintf(stderr, "Warning: no Device API support with a single LSA team"
                        "; all custom kernels will be skipped.\n");
    else if (!hasMC && myRank == 0)
        fprintf(stderr, "Warning: no NVLink multicast; MC kernels will be skipped.\n");

    // ------------------------------------------------------------------
    // 4. Allocate symmetric buffers
    //
    //   sendBuff  — input for all custom kernels.  Registered with
    //               NCCL_WIN_COLL_SYMMETRIC so ncclGetLsaMultimemPointer
    //               returns its multicast VA (used by multimem.ld_reduce).
    //   recvBuff  — primary output / twoshot_mc_poison rotation buffer 0.
    //               Symmetric window for multimem.st (twoshot kernels).
    //               Also used as plain output pointer for oneshot_mc and
    //               as send/recv for ncclAllReduce.
    //   recvBuff2 / recvBuff3 — additional rotation buffers for
    //               twoshot_mc_poison triple-buffering.
    // ------------------------------------------------------------------
    const size_t bufBytes = (size_t)PERF_MAX_NLINES * sizeof(float4);

    float4* sendBuff;
    NCCLCHECK(ncclMemAlloc((void**)&sendBuff, bufBytes));
    ncclWindow_t sendWin;
    NCCLCHECK(ncclCommWindowRegister(comm, sendBuff, bufBytes,
                                     &sendWin, NCCL_WIN_COLL_SYMMETRIC));

    float4* recvBuff;
    NCCLCHECK(ncclMemAlloc((void**)&recvBuff, bufBytes));
    ncclWindow_t recvWin;
    NCCLCHECK(ncclCommWindowRegister(comm, recvBuff, bufBytes,
                                     &recvWin, NCCL_WIN_COLL_SYMMETRIC));

    float4* recvBuff2;
    NCCLCHECK(ncclMemAlloc((void**)&recvBuff2, bufBytes));
    ncclWindow_t recvWin2;
    NCCLCHECK(ncclCommWindowRegister(comm, recvBuff2, bufBytes,
                                     &recvWin2, NCCL_WIN_COLL_SYMMETRIC));

    float4* recvBuff3;
    NCCLCHECK(ncclMemAlloc((void**)&recvBuff3, bufBytes));
    ncclWindow_t recvWin3;
    NCCLCHECK(ncclCommWindowRegister(comm, recvBuff3, bufBytes,
                                     &recvWin3, NCCL_WIN_COLL_SYMMETRIC));

    // Scratch for oneshot_lsa_push: 2 slots x nRanks source ranks.
    const long   pushMaxBytes  = maxMsgBytes < PUSH_MAX_BYTES ? maxMsgBytes : PUSH_MAX_BYTES;
    const size_t pushSlotLines = (size_t)pushMaxBytes / sizeof(float4);
    const size_t pushBytes     = 2 * (size_t)nRanks * pushSlotLines * sizeof(float4);
    void* pushScratch = nullptr;
    ncclWindow_t pushWin;
    if (hasLsa) {
        NCCLCHECK(ncclMemAlloc(&pushScratch, pushBytes));
        NCCLCHECK(ncclCommWindowRegister(comm, pushScratch, pushBytes,
                                         &pushWin, NCCL_WIN_COLL_SYMMETRIC));
    }

    char* flushBuf;
    CUDACHECK(cudaMalloc((void**)&flushBuf, L2_FLUSH_BYTES));

    // ------------------------------------------------------------------
    // 5. Create LSA device communicator
    //
    //   lsaMultimem       : set when multicast is available; activates the
    //                       multicast VA for sendWin/recvWins,
    //                       required by multimem.ld_reduce and multimem.st.
    //   lsaBarrierCount   — pre-allocates barrier slots 0..maxCtaCount-1,
    //                       one per CTA (each CTA uses slot blockIdx.x).
    //   LL buffer         : ncclLLA2A resource for oneshot_lsa_push_ll: one
    //                       region per CTA, sized for the largest chunk
    //                       (smallest CTA count, largest push message).
    // ------------------------------------------------------------------
    ncclDevComm_t     devComm;
    ncclLLA2AHandle_t pushLL;
    if (hasLsa) {
        ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
        reqs.lsaBarrierCount = maxCtaCount;
        reqs.lsaMultimem     = hasMC;
        ncclDevResourceRequirements_t llReq;
        int llSlots = ncclLLA2ACalcSlots(
            (nRanks - 1) * (int)lsaPushLLChunkLines(pushSlotLines, minCtaCount), sizeof(float4));
        NCCLCHECK(ncclLLA2ACreateRequirement(maxCtaCount, llSlots, &pushLL, &llReq));
        llReq.next = reqs.resourceRequirementsList;
        reqs.resourceRequirementsList = &llReq;
        NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    }

    cudaStream_t stream;
    CUDACHECK(cudaStreamCreate(&stream));

    // Fill sendBuff with a per-rank value: rank r uses (r+1) for all elements.
    // Expected AllReduce sum = nRanks*(nRanks+1)/2, a non-trivial value that
    // catches data-mix bugs (e.g. a rank reading its own buffer instead of
    // peers', or one rank's contribution being dropped or double-counted).
    {
        int nFloats = (int)((size_t)PERF_MAX_NLINES * 4);
        fill_float<<<(nFloats + 255) / 256, 256, 0, stream>>>(
            (float*)sendBuff, (float)(myRank + 1), nFloats);
    }
    CUDACHECK(cudaStreamSynchronize(stream));

    // ------------------------------------------------------------------
    // 6. Correctness tests
    //
    //   Tests run over three message sizes:
    //     { nRanks, nRanks*64, nRanks*1024 } float4 lines
    //   All are exact multiples of nRanks so twoshot and poison kernels
    //   are always valid.
    //
    //   Tests run over up to two CTA counts: { 1, maxCtaCount }.
    //   When maxCtaCount == 1 only the single-CTA path is tested.
    //
    //   twoshot_mc_poison is iterated TEST_ITERS times per case.  After
    //   completion the following are verified for each rank locally:
    //     (1) recvBuf: all four float components == EXPECTED everywhere.
    //     (2) clearBuf.{x,y,z}: == EXPECTED everywhere (written by the
    //         AllGather two iterations before this iteration, never
    //         subsequently overwritten by re-poisoning, which only touches .w).
    //     (3) clearBuf.w in myRank's ReduceScatter chunk: == LSA_POISON
    //         (this rank re-poisoned its own chunk inline).
    //     (4) clearBuf.w outside myRank's chunk: == EXPECTED (bit-exact),
    //         confirming the AllGather result is intact and no spurious
    //         poisoning occurred.
    //
    //   The pass flag is reduced across all MPI ranks (MPI_MIN) after each
    //   sub-test so the final result reflects the global outcome.
    // ------------------------------------------------------------------
    const float EXPECTED = (float)(nRanks * (nRanks + 1) / 2);

    const int testNlines[] = { nRanks, nRanks * 64, nRanks * 1024 };
    const int nTestSizes   = (int)(sizeof(testNlines) / sizeof(testNlines[0]));

    int testCtaCounts[2] = { 1, maxCtaCount };
    int nCtaTests        = (maxCtaCount > 1) ? 2 : 1;

    // checkAllEqual — verify all four float4 components are EXPECTED for
    // every element in h[0..n-1].  Returns false and prints on first mismatch.
    auto checkAllEqual = [&](const std::vector<float4>& h, int n,
                              const char* tag) -> bool {
        for (int i = 0; i < n; i++) {
            float4 s = h[i];
            if (s.x != EXPECTED || s.y != EXPECTED ||
                s.z != EXPECTED || s.w != EXPECTED) {
                fprintf(stderr,
                    "[Rank %d] %s FAIL: recvBuf[%d]={%g,%g,%g,%g},"
                    " expected all %g\n",
                    myRank, tag, i, s.x, s.y, s.z, s.w, EXPECTED);
                return false;
            }
        }
        return true;
    };

    // checkClearBuf — verify the clearBuf invariants after twoshot_mc_poison:
    //   .x/.y/.z == EXPECTED everywhere (from the preceding AllGather).
    //   .w == LSA_POISON  in [chunkLo, chunkHi)  (re-poisoned by this rank).
    //   .w == EXPECTED    outside [chunkLo, chunkHi) (AllGather result intact).
    auto checkClearBuf = [&](const std::vector<float4>& h, int n,
                              size_t chunkLo, size_t chunkHi,
                              const char* tag) -> bool {
        uint32_t expBits; memcpy(&expBits, &EXPECTED, sizeof(uint32_t));
        for (int i = 0; i < n; i++) {
            float4 s = h[i];
            if (s.x != EXPECTED || s.y != EXPECTED || s.z != EXPECTED) {
                fprintf(stderr,
                    "[Rank %d] %s FAIL (clearBuf xyz): clearBuf[%d]={%g,%g,%g},"
                    " expected all %g\n",
                    myRank, tag, i, s.x, s.y, s.z, EXPECTED);
                return false;
            }
            uint32_t wBits; memcpy(&wBits, &s.w, sizeof(uint32_t));
            bool inMyChunk = ((size_t)i >= chunkLo && (size_t)i < chunkHi);
            if (inMyChunk) {
                if (wBits != LSA_POISON) {
                    fprintf(stderr,
                        "[Rank %d] %s FAIL (clearBuf my chunk): clearBuf[%d].w"
                        "=0x%08X, expected LSA_POISON\n",
                        myRank, tag, i, wBits);
                    return false;
                }
            } else {
                if (wBits != expBits) {
                    fprintf(stderr,
                        "[Rank %d] %s FAIL (clearBuf other chunk): clearBuf[%d].w"
                        "=0x%08X, expected 0x%08X\n",
                        myRank, tag, i, wBits, expBits);
                    return false;
                }
            }
        }
        return true;
    };

    // checkAllValue: like checkAllEqual, for an arbitrary expected value.
    auto checkAllValue = [&](const std::vector<float4>& h, int n, float want,
                             const char* tag) -> bool {
        for (int i = 0; i < n; i++) {
            float4 s = h[i];
            if (s.x != want || s.y != want || s.z != want || s.w != want) {
                fprintf(stderr,
                    "[Rank %d] %s FAIL: recvBuf[%d]={%g,%g,%g,%g},"
                    " expected all %g\n",
                    myRank, tag, i, s.x, s.y, s.z, s.w, want);
                return false;
            }
        }
        return true;
    };

    // The push kernel alternates its scratch slot on every call.
    unsigned pushSlot = 0;
    auto launchPush = [&](int ll, int nBlocks, const float4* in, float4* out,
                          size_t nlines, int dt) {
        const int slot = ll ? 0 : (int)(pushSlot++ & 1);
        if (dt == DT_HALF)
            launchPushTyped<__half>(ll, nBlocks, stream, in, out, nlines,
                                    pushLL, pushWin, pushSlotLines, slot, devComm);
        else if (dt == DT_BF16)
            launchPushTyped<__nv_bfloat16>(ll, nBlocks, stream, in, out, nlines,
                                           pushLL, pushWin, pushSlotLines, slot, devComm);
        else
            launchPushTyped<float>(ll, nBlocks, stream, in, out, nlines,
                                   pushLL, pushWin, pushSlotLines, slot, devComm);
    };

    int pass = 1;

    for (int ti = 0; ti < nTestSizes && pass; ti++) {
        int tNlines = testNlines[ti];
        if ((size_t)tNlines > PERF_MAX_NLINES) continue;

        for (int ci = 0; ci < nCtaTests && pass; ci++) {
            int  tCtaCount = testCtaCounts[ci];
            char tag[128];

            // ---- 6a. oneshot_mc ----
            if (hasMC) {
                snprintf(tag, sizeof(tag), "oneshot_mc[nlines=%d,ctas=%d]",
                         tNlines, tCtaCount);
                CUDACHECK(cudaMemset(recvBuff, 0, (size_t)tNlines * sizeof(float4)));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                oneshot_mc_allreduce_kernel<<<tCtaCount, MC_AR_MAXTHREADS, 0, stream>>>(
                    recvBuff, sendWin, (size_t)tNlines, devComm);
                CUDACHECK(cudaStreamSynchronize(stream));

                std::vector<float4> h(tNlines);
                CUDACHECK(cudaMemcpy(h.data(), recvBuff,
                                     (size_t)tNlines * sizeof(float4),
                                     cudaMemcpyDeviceToHost));
                if (!checkAllEqual(h, tNlines, tag)) pass = 0;
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT,
                                       MPI_MIN, MPI_COMM_WORLD));
                if (pass && myRank == 0) printf("[Correctness %s] PASS\n", tag);
                fflush(stdout);
            }

            // ---- 6b. twoshot_mc_simple ----
            if (hasMC && pass) {
                snprintf(tag, sizeof(tag), "twoshot_mc[nlines=%d,ctas=%d]",
                         tNlines, tCtaCount);
                CUDACHECK(cudaMemset(recvBuff, 0, (size_t)tNlines * sizeof(float4)));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                twoshot_mc_simple_allreduce_kernel<<<tCtaCount, TWOSHOT_MC_MAXTHREADS, 0, stream>>>(
                    sendWin, recvWin, (size_t)tNlines, devComm);
                CUDACHECK(cudaStreamSynchronize(stream));

                std::vector<float4> h(tNlines);
                CUDACHECK(cudaMemcpy(h.data(), recvBuff,
                                     (size_t)tNlines * sizeof(float4),
                                     cudaMemcpyDeviceToHost));
                if (!checkAllEqual(h, tNlines, tag)) pass = 0;
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT,
                                       MPI_MIN, MPI_COMM_WORLD));
                if (pass && myRank == 0) printf("[Correctness %s] PASS\n", tag);
                fflush(stdout);
            }

            // ---- 6c. twoshot_mc_poison ----
            //
            // Triple-buffer rotation:
            //   Iter si: recvBuf = poisBufs[si%3], clearBuf = poisBufs[(si+2)%3].
            //   skip_barrier = false for si < 2 (entry barrier until steady state).
            //
            // After TEST_ITERS iterations, clearBuf is the buffer that was
            // last the recvBuf two iters ago.  The AllGather of that iter wrote
            // EXPECTED to all elements; then the current iter's re-poison pass
            // wrote LSA_POISON to .w only within this rank's ReduceScatter chunk.
            if (hasMC && pass) {
                snprintf(tag, sizeof(tag),
                         "twoshot_mc_poison[nlines=%d,ctas=%d]",
                         tNlines, tCtaCount);
                float4*      poisBufs[3] = { recvBuff,  recvBuff2,  recvBuff3  };
                ncclWindow_t poisWins[3] = { recvWin,   recvWin2,   recvWin3   };
                int poisonBlocks = ((int)tNlines + 255) / 256;

                for (int b = 0; b < 3; b++)
                    poison_float4_buffer<<<poisonBlocks, 256, 0, stream>>>(
                        poisBufs[b], tNlines);
                CUDACHECK(cudaStreamSynchronize(stream));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

                for (int si = 0; si < TEST_ITERS; si++)
                    twoshot_mc_poison_allreduce_kernel<<<tCtaCount, TWOSHOT_MC_POISON_MAXTHREADS, 0, stream>>>(
                        sendWin,
                        poisBufs[si % 3], poisWins[si % 3],
                        poisBufs[(si + 2) % 3],
                        (size_t)tNlines, devComm,
                        /*skip_barrier=*/si >= 2);
                CUDACHECK(cudaStreamSynchronize(stream));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));

                int lastIdx  = (TEST_ITERS - 1) % 3;
                int clearIdx = (TEST_ITERS - 1 + 2) % 3;

                size_t chunkLines = (size_t)tNlines / nRanks;
                size_t myChunkLo  = (size_t)myRank * chunkLines;
                size_t myChunkHi  = myChunkLo + chunkLines;

                std::vector<float4> hRecv(tNlines), hClear(tNlines);
                CUDACHECK(cudaMemcpy(hRecv.data(), poisBufs[lastIdx],
                                     (size_t)tNlines * sizeof(float4),
                                     cudaMemcpyDeviceToHost));
                CUDACHECK(cudaMemcpy(hClear.data(), poisBufs[clearIdx],
                                     (size_t)tNlines * sizeof(float4),
                                     cudaMemcpyDeviceToHost));

                // (1) All 4 components of recvBuf == EXPECTED.
                if (!checkAllEqual(hRecv, tNlines, tag)) pass = 0;
                // (2–4) clearBuf invariants.
                if (pass && !checkClearBuf(hClear, tNlines,
                                           myChunkLo, myChunkHi, tag))
                    pass = 0;

                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT,
                                       MPI_MIN, MPI_COMM_WORLD));
                if (pass && myRank == 0) printf("[Correctness %s] PASS\n", tag);
                fflush(stdout);
            }

            // ---- 6d/6e. oneshot_lsa_push and oneshot_lsa_push_ll ----
            //
            // A chain of PUSH_TEST_ITERS calls with no host sync in between,
            // each call reducing the previous call's output (the first one
            // reads sendBuff, the last one runs in place).  After call k every
            // rank holds the same value v_k, so v_{k+1} = nRanks * v_k, summed
            // in rank order exactly as the kernels do.  A call that reads a
            // stale scratch slot or LL line breaks the chain.
            for (int ll = 0; ll < 2 && hasLsa && pass; ll++) {
                if ((size_t)tNlines > pushSlotLines) continue;
                const int PUSH_TEST_ITERS = 6;
                snprintf(tag, sizeof(tag), "%s[nlines=%d,ctas=%d]",
                         ll ? "oneshot_lsa_push_ll" : "oneshot_lsa_push",
                         tNlines, tCtaCount);
                float4* bufs[2] = { recvBuff, recvBuff2 };
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                for (int k = 0; k < PUSH_TEST_ITERS; k++) {
                    const float4* in  = (k == 0) ? sendBuff : bufs[(k - 1) % 2];
                    float4*       out = (k == PUSH_TEST_ITERS - 1) ? bufs[(k - 1) % 2]
                                                                   : bufs[k % 2];
                    launchPush(ll, tCtaCount, in, out, (size_t)tNlines, DT_FLOAT);
                }
                CUDACHECK(cudaStreamSynchronize(stream));

                float want = EXPECTED;
                for (int k = 1; k < PUSH_TEST_ITERS; k++) {
                    float v = want;
                    for (int r = 1; r < nRanks; r++) v += want;
                    want = v;
                }
                std::vector<float4> h(tNlines);
                CUDACHECK(cudaMemcpy(h.data(), bufs[(PUSH_TEST_ITERS - 2) % 2],
                                     (size_t)tNlines * sizeof(float4),
                                     cudaMemcpyDeviceToHost));
                if (!checkAllValue(h, tNlines, want, tag)) pass = 0;
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT,
                                       MPI_MIN, MPI_COMM_WORLD));
                if (pass && myRank == 0) printf("[Correctness %s] PASS\n", tag);
                fflush(stdout);
            }

            // ---- 6f. oneshot_lsa_push(_ll) in half and bf16 ----
            //
            // The protocol is type-independent (the chain above covers it),
            // so this checks the typed add: rank r sends (r+1) * 2^(i % 8),
            // two calls back to back (both scratch slots / LL parities), and
            // every element must equal nRanks*(nRanks+1)/2 * 2^(i % 8)
            // exactly.  Exact for up to 22 ranks in bf16, 31 in half.
            for (int dt = DT_HALF; dt < DT_COUNT && hasLsa && pass; dt++) {
                if ((size_t)tNlines > pushSlotLines) continue;
                const size_t lineBytes = (size_t)tNlines * sizeof(float4);
                const size_t nElem     = lineBytes / dtypeSize(dt);
                fillDtype(dt, sendBuff, (float)(myRank + 1), 1, lineBytes, stream);
                for (int ll = 0; ll < 2 && pass; ll++) {
                    snprintf(tag, sizeof(tag), "%s %s[nlines=%d,ctas=%d]",
                             ll ? "oneshot_lsa_push_ll" : "oneshot_lsa_push",
                             dtypeName(dt), tNlines, tCtaCount);
                    float4* bufs[2] = { recvBuff, recvBuff2 };
                    MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                    for (int k = 0; k < 2; k++)
                        launchPush(ll, tCtaCount, sendBuff, bufs[k], (size_t)tNlines, dt);
                    for (int k = 0; k < 2 && pass; k++) {
                        if (dt == DT_HALF)
                            to_float<__half><<<(unsigned)((nElem + 255) / 256), 256, 0, stream>>>(
                                (const __half*)bufs[k], (float*)recvBuff3, nElem);
                        else
                            to_float<__nv_bfloat16><<<(unsigned)((nElem + 255) / 256), 256, 0, stream>>>(
                                (const __nv_bfloat16*)bufs[k], (float*)recvBuff3, nElem);
                        std::vector<float> h(nElem);
                        CUDACHECK(cudaMemcpyAsync(h.data(), recvBuff3, nElem * sizeof(float),
                                                  cudaMemcpyDeviceToHost, stream));
                        CUDACHECK(cudaStreamSynchronize(stream));
                        for (size_t i = 0; i < nElem; i++) {
                            const float want = EXPECTED * (float)(1 << (i & 7));
                            if (h[i] != want) {
                                fprintf(stderr, "[Rank %d] %s FAIL: call %d elem %zu = %g,"
                                        " expected %g\n", myRank, tag, k, i, h[i], want);
                                pass = 0;
                                break;
                            }
                        }
                    }
                    MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT,
                                           MPI_MIN, MPI_COMM_WORLD));
                    if (pass && myRank == 0) printf("[Correctness %s] PASS\n", tag);
                    fflush(stdout);
                }
                // Restore the float input the other tests expect.
                fillDtype(DT_FLOAT, sendBuff, (float)(myRank + 1), 0, lineBytes, stream);
                CUDACHECK(cudaStreamSynchronize(stream));
            }
        }
    }

    if (!hasMC && myRank == 0)
        printf("[Correctness MC kernels       ] SKIPPED (sm < 9.0)\n");
    if (myRank == 0) printf("\n");
    fflush(stdout);

    // Re-fill sendBuff with 1.0 (in the sweep's dtype) for the performance
    // sweep so bandwidth numbers reflect a uniform-data workload.  The MC
    // kernels are float only, so they are skipped for half and bf16.
    fillDtype(dtype, sendBuff, 1.0f, 0, bufBytes, stream);
    CUDACHECK(cudaStreamSynchronize(stream));
    const bool mcPerf = hasMC && dtype == DT_FLOAT;
    if (myRank == 0) printf("dtype: %s\n\n", dtypeName(dtype));

    // ------------------------------------------------------------------
    // 7. Performance sweep
    //
    //   Outer loop: CTA count (minCtaCount..maxCtaCount, powers of 2).
    //   Inner loop: message size (minMsgBytes..maxMsgBytes, powers of 2).
    //
    //   For each (nBlocks, msgBytes):
    //     nlines    = msgBytes / sizeof(float4)
    //     twoNlines = nlines rounded down to the nearest multiple of nRanks
    //                 (twoshot constraint: nlines % nRanks == 0)
    //
    //   oneshot_mc   measured when nlines >= 1 and hasMC.
    //   twoshot_mc_* measured when twoNlines >= nRanks and hasMC.
    //   ncclAllReduce always measured as baseline.
    //
    //   Bandwidth: algbw = msgBytes / lat_us (GB/s).
    //   Latency printed in µs; bandwidth in GB/s.
    // ------------------------------------------------------------------
    cudaEvent_t evStart, evStop;
    CUDACHECK(cudaEventCreate(&evStart));
    CUDACHECK(cudaEventCreate(&evStop));

    for (int nBlocks = minCtaCount; nBlocks <= maxCtaCount; nBlocks *= 2) {
        if (myRank == 0) {
            printf("=== CTAs: %d ===\n", nBlocks);
            printf("%-12s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s\n",
                   "msg(bytes)",
                   "oneshot_mc",   "(GB/s)",
                   "2shot_mc",     "(GB/s)",
                   "2shot_mc_pois","(GB/s)",
                   "push_lsa",     "(GB/s)",
                   "push_lsa_ll",  "(GB/s)",
                   "nccl_ar",      "(GB/s)");
            printf("%-12s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s\n",
                   "", "(us)", "", "(us)", "", "(us)", "", "(us)", "", "(us)", "", "(us)", "");
            printf("------------------------------------------------------------------------------------"
                   "------------------------------------------------------------------------------\n");
            fflush(stdout);
        }

        for (long msgBytes = minMsgBytes; msgBytes <= maxMsgBytes; msgBytes *= 2) {
            long nlines    = msgBytes / (long)sizeof(float4);
            if (nlines < 1) continue;
            long twoNlines = (nlines / nRanks) * nRanks;
            int  nTimed    = timedIters(nlines);

            // ---- oneshot_mc ----
            float oneLat = 0.f;
            if (mcPerf) {
                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                for (int i = 0; i < WARMUP_ITERS; i++)
                    oneshot_mc_allreduce_kernel<<<nBlocks, MC_AR_MAXTHREADS, 0, stream>>>(
                        recvBuff, sendWin, (size_t)nlines, devComm);
                CUDACHECK(cudaStreamSynchronize(stream));

                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                oneLat = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                    for (int i = 0; i < nTimed; i++)
                        oneshot_mc_allreduce_kernel<<<nBlocks, MC_AR_MAXTHREADS, 0, stream>>>(
                            recvBuff, sendWin, (size_t)nlines, devComm);
                });
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &oneLat, 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            // ---- twoshot_mc_simple ----
            float twoLat = 0.f;
            if (mcPerf && twoNlines >= nRanks) {
                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                for (int i = 0; i < WARMUP_ITERS; i++)
                    twoshot_mc_simple_allreduce_kernel<<<nBlocks, TWOSHOT_MC_MAXTHREADS, 0, stream>>>(
                        sendWin, recvWin, (size_t)twoNlines, devComm);
                CUDACHECK(cudaStreamSynchronize(stream));

                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                twoLat = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                    for (int i = 0; i < nTimed; i++)
                        twoshot_mc_simple_allreduce_kernel<<<nBlocks, TWOSHOT_MC_MAXTHREADS, 0, stream>>>(
                            sendWin, recvWin, (size_t)twoNlines, devComm);
                });
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &twoLat, 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            // ---- twoshot_mc_poison ----
            //
            // Pre-poison all three rotation buffers before warmup.
            // Warmup runs WARMUP_ITERS + 3 iterations so every slot in the
            // rotation has been used at least once before timing starts.
            // The timed loop always uses skip_barrier=true (steady state).
            // nTimed is rounded up to a multiple of 3 for a balanced rotation.
            float poisLat = 0.f;
            if (mcPerf && twoNlines >= nRanks) {
                float4*      poisBufs[3] = { recvBuff,  recvBuff2,  recvBuff3  };
                ncclWindow_t poisWins[3] = { recvWin,   recvWin2,   recvWin3   };
                int poisonBlocks = ((int)twoNlines + 255) / 256;
                for (int b = 0; b < 3; b++)
                    poison_float4_buffer<<<poisonBlocks, 256, 0, stream>>>(
                        poisBufs[b], (int)twoNlines);

                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                int sentIter = 0;
                for (int w = 0; w < WARMUP_ITERS + 3; w++, sentIter++)
                    twoshot_mc_poison_allreduce_kernel<<<nBlocks, TWOSHOT_MC_POISON_MAXTHREADS, 0, stream>>>(
                        sendWin,
                        poisBufs[sentIter % 3], poisWins[sentIter % 3],
                        poisBufs[(sentIter + 2) % 3],
                        (size_t)twoNlines, devComm,
                        /*skip_barrier=*/sentIter >= 2);
                CUDACHECK(cudaStreamSynchronize(stream));

                int itersInGraph = ((nTimed + 2) / 3) * 3;
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                int sentIterBase = sentIter;
                poisLat = runTimed(useGraph, itersInGraph, stream, evStart, evStop, [&]() {
                    int si = sentIterBase;
                    for (int i = 0; i < itersInGraph; i++, si++)
                        twoshot_mc_poison_allreduce_kernel<<<nBlocks, TWOSHOT_MC_POISON_MAXTHREADS, 0, stream>>>(
                            sendWin,
                            poisBufs[si % 3], poisWins[si % 3],
                            poisBufs[(si + 2) % 3],
                            (size_t)twoNlines, devComm,
                            /*skip_barrier=*/true);
                });
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &poisLat, 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            // ---- oneshot_lsa_push / oneshot_lsa_push_ll ----
            float pushLat[2] = { 0.f, 0.f };
            bool  hasPush = hasLsa && msgBytes <= pushMaxBytes;
            for (int ll = 0; ll < 2 && hasPush; ll++) {
                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                for (int i = 0; i < WARMUP_ITERS; i++)
                    launchPush(ll, nBlocks, sendBuff, recvBuff, (size_t)nlines, dtype);
                CUDACHECK(cudaStreamSynchronize(stream));

                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                pushLat[ll] = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                    for (int i = 0; i < nTimed; i++)
                        launchPush(ll, nBlocks, sendBuff, recvBuff, (size_t)nlines, dtype);
                });
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &pushLat[ll], 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            // ---- ncclAllReduce baseline ----
            float ncclLat = 0.f;
            {
                CUDACHECK(cudaMemset(flushBuf, 0, L2_FLUSH_BYTES));
                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                for (int i = 0; i < WARMUP_ITERS; i++)
                    NCCLCHECK(ncclAllReduce(sendBuff, recvBuff,
                                           (size_t)nlines * sizeof(float4) / dtypeSize(dtype), dtypeNccl(dtype), ncclSum,
                                           comm, stream));
                CUDACHECK(cudaStreamSynchronize(stream));

                MPICHECK(MPI_Barrier(MPI_COMM_WORLD));
                ncclLat = runTimed(useGraph, nTimed, stream, evStart, evStop, [&]() {
                    for (int i = 0; i < nTimed; i++)
                        NCCLCHECK(ncclAllReduce(sendBuff, recvBuff,
                                               (size_t)nlines * sizeof(float4) / dtypeSize(dtype), dtypeNccl(dtype), ncclSum,
                                               comm, stream));
                });
                MPICHECK(MPI_Allreduce(MPI_IN_PLACE, &ncclLat, 1, MPI_FLOAT,
                                       MPI_MAX, MPI_COMM_WORLD));
            }

            if (myRank == 0) {
                // algbw = msgBytes / lat_us  (converts to GB/s).
                auto fmtCols = [](char lat_s[32], char bw_s[32],
                                  float lat, long bytes) {
                    double algbw = (double)bytes / (lat * 1e-6) / 1e9;
                    snprintf(lat_s, 32, "%.2f", lat);
                    snprintf(bw_s,  32, "%.3f", algbw);
                };

                char oneLat_s[32]  = "N/A", oneBw_s[32]  = "N/A";
                char twoLat_s[32]  = "N/A", twoBw_s[32]  = "N/A";
                char poisLat_s[32] = "N/A", poisBw_s[32] = "N/A";
                char pushLat_s[2][32] = { "N/A", "N/A" }, pushBw_s[2][32] = { "N/A", "N/A" };
                char ncclLat_s[32], ncclBw_s[32];

                if (mcPerf)
                    fmtCols(oneLat_s, oneBw_s, oneLat,
                            nlines * (long)sizeof(float4));
                if (mcPerf && twoNlines >= nRanks) {
                    fmtCols(twoLat_s, twoBw_s, twoLat,
                            twoNlines * (long)sizeof(float4));
                    fmtCols(poisLat_s, poisBw_s, poisLat,
                            twoNlines * (long)sizeof(float4));
                }
                if (hasPush)
                    for (int ll = 0; ll < 2; ll++)
                        fmtCols(pushLat_s[ll], pushBw_s[ll], pushLat[ll],
                                nlines * (long)sizeof(float4));
                fmtCols(ncclLat_s, ncclBw_s, ncclLat,
                        nlines * (long)sizeof(float4));

                printf("%-12ld  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s  %-13s %-10s\n",
                       msgBytes,
                       oneLat_s, oneBw_s,
                       twoLat_s, twoBw_s,
                       poisLat_s, poisBw_s,
                       pushLat_s[0], pushBw_s[0],
                       pushLat_s[1], pushBw_s[1],
                       ncclLat_s, ncclBw_s);
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
    if (hasLsa) {
        NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
        NCCLCHECK(ncclCommWindowDeregister(comm, pushWin));
        NCCLCHECK(ncclMemFree(pushScratch));
    }
    NCCLCHECK(ncclCommWindowDeregister(comm, sendWin));
    NCCLCHECK(ncclCommWindowDeregister(comm, recvWin));
    NCCLCHECK(ncclCommWindowDeregister(comm, recvWin2));
    NCCLCHECK(ncclCommWindowDeregister(comm, recvWin3));
    CUDACHECK(cudaFree(flushBuf));
    NCCLCHECK(ncclMemFree(sendBuff));
    NCCLCHECK(ncclMemFree(recvBuff));
    NCCLCHECK(ncclMemFree(recvBuff2));
    NCCLCHECK(ncclMemFree(recvBuff3));
    NCCLCHECK(ncclCommFinalize(comm));
    NCCLCHECK(ncclCommDestroy(comm));
    MPICHECK(MPI_Finalize());

    return pass ? 0 : 1;
}
