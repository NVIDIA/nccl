/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NVSHMEM-compatible host API for NIIN.
//
// Provides the standard NVSHMEM host-side functions (init, malloc, query,
// barrier, finalize) implemented on top of NCCL. Uses a global singleton
// to match NVSHMEM's implicit-global-state programming model.
//
// For multi-PE operation, launch with mpirun and call:
//   nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr)
// or use NVSHMEMX_INIT_WITH_UNIQUEID after distributing a unique ID with the
// application's bootstrap mechanism. nvshmem_init() can also auto-detect
// rank/size metadata from supported launchers.
//
// For single-PE operation (testing), just call nvshmem_init().

#ifndef NIIN_NVSHMEM_HOST_H_
#define NIIN_NVSHMEM_HOST_H_

#include <nccl.h>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "niin/host.h"

#ifdef NIIN_HAS_MPI
#include <mpi.h>
#endif

// Default symmetric heap size (256 MB), overridable via NVSHMEM_SYMMETRIC_SIZE env var
#ifndef NIIN_DEFAULT_HEAP_SIZE
#define NIIN_DEFAULT_HEAP_SIZE (256ULL << 20)
#endif

// Allocation alignment within the symmetric heap
#define NIIN_ALLOC_ALIGN 256

// ---------------------------------------------------------------------------
// Global state singleton
// ---------------------------------------------------------------------------
namespace niin {
namespace detail {

// Free-list heap block
struct HeapBlock {
  size_t offset;
  size_t size;
  bool free;
  bool valid;  // slot in use
};

// Max number of concurrent allocations (each malloc uses one slot;
// free + coalesce returns slots). Typically NVSHMEM programs have < 100.
#define NIIN_MAX_HEAP_BLOCKS 256

struct GlobalState {
  bool initialized;
  ncclComm_t comm;
  int rank;
  int nRanks;
  int lsaRank;
  int lsaSize;
  int cudaDev;
  void* heapBase;
  size_t heapSize;
  HeapBlock heapBlocks[NIIN_MAX_HEAP_BLOCKS];
  int heapBlockCount;
  niinContext_host hostCtx;
  niinContext* devCtx;
  cudaStream_t stream;
  int rmaCtx;       // Host RMA context index for ncclPutSignal
  int rmaSigIdx;     // Signal index for host RMA operations
  bool hostRmaAvail; // Whether host RMA (ncclPutSignal) is available
  bool ginAvail;     // Whether this communicator supports GIN device resources
  void* barrierScratch; // Scratch buffer for host-side barrier (separate from heap)
};

inline GlobalState& state() {
  static GlobalState s = {};
  return s;
}

// Parse heap size from environment, matching NVSHMEM's NVSHMEM_SYMMETRIC_SIZE
inline size_t parseHeapSize() {
  const char* env = getenv("NVSHMEM_SYMMETRIC_SIZE");
  if (!env) return NIIN_DEFAULT_HEAP_SIZE;
  size_t val = 0;
  char suffix = 0;
  if (sscanf(env, "%zu%c", &val, &suffix) >= 1) {
    switch (suffix) {
      case 'k': case 'K': val <<= 10; break;
      case 'm': case 'M': val <<= 20; break;
      case 'g': case 'G': val <<= 30; break;
      default: break;
    }
  }
  return val > 0 ? val : NIIN_DEFAULT_HEAP_SIZE;
}

inline bool parseForceSeparatePutSignal() {
  const char* env = getenv("NIIN_PUT_SIGNAL_MODE");
  if (env == nullptr) return false;
  if (strcmp(env, "separate") == 0 || strcmp(env, "split") == 0 ||
      strcmp(env, "fence_signal") == 0) {
    return true;
  }
  if (strcmp(env, "fused") == 0 || strcmp(env, "auto") == 0) {
    return false;
  }
  if (strcmp(env, "1") == 0 || strcmp(env, "true") == 0 ||
      strcmp(env, "TRUE") == 0 || strcmp(env, "yes") == 0) {
    return true;
  }
  return false;
}

// Detect rank/nRanks from common MPI/PMI environment variables.
// Returns true if detection succeeded.
inline bool detectRankFromEnv(int* rank, int* nRanks) {
  const char* envPairs[][2] = {
    {"OMPI_COMM_WORLD_RANK",  "OMPI_COMM_WORLD_SIZE"},   // OpenMPI
    {"PMI_RANK",              "PMI_SIZE"},                 // PMI
    {"SLURM_PROCID",          "SLURM_NTASKS"},            // SLURM
    {"MV2_COMM_WORLD_RANK",   "MV2_COMM_WORLD_SIZE"},     // MVAPICH
    {"PMIX_RANK",             "PMIX_SIZE"},                // PMIx
  };
  for (auto& pair : envPairs) {
    const char* r = getenv(pair[0]);
    const char* s = getenv(pair[1]);
    if (r && s) {
      *rank = atoi(r);
      *nRanks = atoi(s);
      if (*nRanks > 0) return true;
    }
  }
  return false;
}

// Detect which local GPU to use. Tries LOCAL_RANK-style env vars, falls back
// to rank % deviceCount.
inline int detectLocalDevice(int rank) {
  const char* localRankEnvs[] = {
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "MPI_LOCALRANKID",
    "SLURM_LOCALID",
    "MV2_COMM_WORLD_LOCAL_RANK",
  };
  for (auto& env : localRankEnvs) {
    const char* val = getenv(env);
    if (val) return atoi(val);
  }
  int nDevs;
  cudaGetDeviceCount(&nDevs);
  return nDevs > 0 ? rank % nDevs : 0;
}

// Common init logic shared by nvshmem_init() and nvshmemx_init_attr()
inline int initCommon(ncclComm_t comm) {
  auto& s = state();
  ncclCommProperties_t props = NCCL_COMM_PROPERTIES_INITIALIZER;

  ncclCommUserRank(comm, &s.rank);
  ncclCommCount(comm, &s.nRanks);
  s.comm = comm;
  s.ginAvail = false;
  s.hostRmaAvail = false;
  if (ncclCommQueryProperties(comm, &props) == ncclSuccess) {
    s.ginAvail = props.ginType != NCCL_GIN_TYPE_NONE;
    if (props.hostRmaSupport) {
      s.hostRmaAvail = true;
      s.rmaCtx = 0;
      s.rmaSigIdx = 0;
    }
  }

  // Heap
  s.heapSize = parseHeapSize();
  ncclResult_t r = ncclMemAlloc(&s.heapBase, s.heapSize);
  if (r != ncclSuccess) {
    fprintf(stderr, "NIIN: ncclMemAlloc(%zu) failed: %s\n",
            s.heapSize, ncclGetErrorString(r));
    return -1;
  }
  cudaMemset(s.heapBase, 0, s.heapSize);
  memset(s.heapBlocks, 0, sizeof(s.heapBlocks));
  s.heapBlocks[0] = {0, s.heapSize, true, true};
  s.heapBlockCount = 1;

  // Device context and scratch buffers
  cudaMalloc(&s.devCtx, sizeof(niinContext));
  cudaMalloc(&s.barrierScratch, 16);  // Separate scratch for host barrier
  cudaMemset(s.barrierScratch, 0, 16);
  cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking);

  // Detect native atomic support for peer GPUs
  {
    int curDev;
    cudaGetDevice(&curDev);
    int nativeAtomic = 1;
    // Check all peers — if any lack native atomics, flag it
    for (int i = 0; i < s.nRanks; i++) {
      // In MPI mode, each process has one GPU. We check device 0 against device 0
      // on the peer. For multi-GPU-per-process, check the actual peer device.
      int peerDev = curDev; // Simplified — only matters for same-process multi-GPU
      if (peerDev != curDev) {
        int val = 0;
        cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrNativeAtomicSupported, curDev, peerDev);
        if (!val) nativeAtomic = 0;
      }
    }
    // For MPI mode (different processes), query the first other visible GPU
    int nDevs;
    cudaGetDeviceCount(&nDevs);
    for (int d = 0; d < nDevs; d++) {
      if (d == curDev) continue;
      int val = 0;
      cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrNativeAtomicSupported, curDev, d);
      if (!val) { nativeAtomic = 0; break; }
    }
    s.hostCtx.peerNativeAtomic = (nativeAtomic != 0);
  }
  s.hostCtx.forceSeparatePutSignal = parseForceSeparatePutSignal();

  // Two-phase init
  ncclGroupStart();
  r = niinInit(comm, s.heapBase, s.heapSize, &s.hostCtx, s.ginAvail);
  ncclGroupEnd();
  if (r != ncclSuccess) return -1;

  r = niinCommit(&s.hostCtx, s.devCtx);
  if (r != ncclSuccess) return -1;

  // Set the __device__ global so kernels don't need manual context setup.
  // niin_g_ctx is declared in niin/context.h.
  niinContext* devCtxPtr = s.devCtx;
  cudaMemcpyToSymbol(niin_g_ctx, &devCtxPtr, sizeof(niinContext*));

  // Extract LSA info from the devComm
  s.lsaRank = s.hostCtx.devComm.lsaRank;
  s.lsaSize = s.hostCtx.devComm.lsaSize;

  // Initialize predefined teams
  niin::teams::initPredefined(s.rank, s.nRanks, s.lsaRank, s.lsaSize);

  s.initialized = true;
  return 0;
}

} // namespace detail
} // namespace niin

// ---------------------------------------------------------------------------
// Kernel context setup helper.
// Call from one thread at kernel entry to wire the global device context.
// ---------------------------------------------------------------------------
#define NIIN_KERNEL_INIT()                        \
  do {                                             \
    extern __device__ niinContext* niin_g_ctx;      \
    if (threadIdx.x == 0)                          \
      niin_g_ctx = niin::detail::state().devCtx;   \
    __syncthreads();                               \
  } while (0)

// This doesn't work directly because devCtx is a host variable. The actual
// approach: pass devCtx as a kernel argument. We provide a helper:
__device__ inline void niin_set_context(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
}

// ---------------------------------------------------------------------------
// nvshmem_init / nvshmem_finalize
// ---------------------------------------------------------------------------

// nvshmem_init: initialize NIIN with auto-detected rank/size.
// Multi-PE: detects rank from supported launcher metadata and bootstraps
// ncclUniqueId via MPI_Bcast when NIIN_HAS_MPI is available. Single-PE
// fallback otherwise.
inline int nvshmem_init(void) {
  auto& s = niin::detail::state();
  if (s.initialized) {
    fprintf(stderr, "NIIN: already initialized\n");
    return -1;
  }

  int rank = 0, nRanks = 1;
  bool multiPE = niin::detail::detectRankFromEnv(&rank, &nRanks);

  int localDev = niin::detail::detectLocalDevice(rank);
  cudaSetDevice(localDev);
  s.cudaDev = localDev;

  // Request host RMA support via ncclConfig
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.numRmaCtx = 1;

  ncclComm_t comm;
  if (multiPE && nRanks > 1) {
#ifdef NIIN_HAS_MPI
    // Use MPI for ncclUniqueId distribution
    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRankConfig(&comm, nRanks, id, rank, &config);
#else
    fprintf(stderr, "NIIN: multi-PE detected (rank=%d nRanks=%d) but NIIN_HAS_MPI not defined. "
            "Use nvshmemx_init_attr() with MPI or compile with -DNIIN_HAS_MPI.\n",
            rank, nRanks);
    return -1;
#endif
  } else {
    // Single-PE mode
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclCommInitRankConfig(&comm, 1, id, 0, &config);
  }

  return niin::detail::initCommon(comm);
}

// nvshmem_finalize: tear down NIIN global state.
inline void nvshmem_finalize(void) {
  auto& s = niin::detail::state();
  if (!s.initialized) return;

  cudaSetDevice(s.cudaDev);
  cudaDeviceSynchronize();

  ncclGroupStart();
  niinFinalize(s.comm, s.devCtx);
  ncclGroupEnd();

  cudaFree(s.devCtx);
  cudaFree(s.barrierScratch);
  ncclMemFree(s.heapBase);
  cudaStreamDestroy(s.stream);
  ncclCommDestroy(s.comm);

  s.initialized = false;
  s.heapBase = nullptr;
  s.devCtx = nullptr;
}

// ---------------------------------------------------------------------------
// nvshmemx_init_attr
// ---------------------------------------------------------------------------
#define NVSHMEMX_INIT_WITH_MPI_COMM   (1u << 1)
#define NVSHMEMX_INIT_WITH_UNIQUEID   (1u << 3)
#define NIIN_UNIQUEID_ARGS_INVALID    (-1)

typedef struct {
  int version;
  char internal[124];
} nvshmemx_uniqueid_t;

typedef struct {
  int version;
  nvshmemx_uniqueid_t* id;
  int myrank;
  int nranks;
} nvshmemx_uniqueid_args_t;

typedef struct {
  int version;
  nvshmemx_uniqueid_args_t uid_args;
  int cuda_device_id;
  char content[88];
} nvshmemx_init_args_t;

typedef struct {
  int version;
  void* mpi_comm;
  nvshmemx_init_args_t args;
} nvshmemx_init_attr_t;

static_assert(sizeof(nvshmemx_uniqueid_t) == NCCL_UNIQUE_ID_BYTES,
              "NIIN unique ID must fit an NCCL unique ID");
static_assert(sizeof(nvshmemx_uniqueid_args_t) == 24,
              "NIIN unique ID args should match NVSHMEM v1 size");
static_assert(sizeof(nvshmemx_init_args_t) == 128,
              "NIIN init args should match NVSHMEM v2 size");
static_assert(sizeof(nvshmemx_init_attr_t) == 144,
              "NIIN init attr should match NVSHMEM v2 size");

#define NVSHMEMX_UNIQUEID_INITIALIZER \
  { (1 << 16) + (int)sizeof(nvshmemx_uniqueid_t), {0} }
#define NVSHMEMX_UNIQUEID_ARGS_INITIALIZER \
  { (1 << 16) + (int)sizeof(nvshmemx_uniqueid_args_t), nullptr, \
    NIIN_UNIQUEID_ARGS_INVALID, NIIN_UNIQUEID_ARGS_INVALID }
#define NVSHMEMX_INIT_ARGS_INITIALIZER \
  { (1 << 16) + (int)sizeof(nvshmemx_init_args_t), \
    NVSHMEMX_UNIQUEID_ARGS_INITIALIZER, NIIN_UNIQUEID_ARGS_INVALID, {0} }
#define NVSHMEMX_INIT_ATTR_INITIALIZER \
  { (1 << 16) + (int)sizeof(nvshmemx_init_attr_t), nullptr, \
    NVSHMEMX_INIT_ARGS_INITIALIZER }

inline int nvshmemx_init_attr(unsigned int flags, nvshmemx_init_attr_t* attr) {
  auto& s = niin::detail::state();
  if (s.initialized) {
    fprintf(stderr, "NIIN: already initialized\n");
    return -1;
  }

  if (flags & NVSHMEMX_INIT_WITH_UNIQUEID) {
    if (attr == nullptr || attr->args.uid_args.id == nullptr ||
        attr->args.uid_args.myrank < 0 || attr->args.uid_args.nranks <= 0) {
      fprintf(stderr, "NIIN: NVSHMEMX_INIT_WITH_UNIQUEID requires rank, nranks, and unique ID args\n");
      return -1;
    }

    int rank = attr->args.uid_args.myrank;
    int nRanks = attr->args.uid_args.nranks;
    int localDev = niin::detail::detectLocalDevice(rank);
    cudaSetDevice(localDev);
    s.cudaDev = localDev;

    ncclUniqueId id;
    memcpy(&id, attr->args.uid_args.id, sizeof(id));

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 1;
    ncclComm_t comm;
    ncclCommInitRankConfig(&comm, nRanks, id, rank, &config);
    return niin::detail::initCommon(comm);
  }

  if (flags & NVSHMEMX_INIT_WITH_MPI_COMM) {
#ifdef NIIN_HAS_MPI
    MPI_Comm mpi_comm = (attr != nullptr && attr->mpi_comm != nullptr) ?
                        *(MPI_Comm*)attr->mpi_comm : MPI_COMM_WORLD;
    int rank, nRanks;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &nRanks);

    int localDev = niin::detail::detectLocalDevice(rank);
    cudaSetDevice(localDev);
    s.cudaDev = localDev;

    ncclUniqueId id;
    if (rank == 0) ncclGetUniqueId(&id);
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

    ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
    config.numRmaCtx = 1;
    ncclComm_t comm;
    ncclCommInitRankConfig(&comm, nRanks, id, rank, &config);
    return niin::detail::initCommon(comm);
#else
    fprintf(stderr, "NIIN: NVSHMEMX_INIT_WITH_MPI_COMM requires -DNIIN_HAS_MPI\n");
    return -1;
#endif
  }

  // Fallback: treat as nvshmem_init()
  return nvshmem_init();
}

// nvshmemx_set_attr_uniqueid_args: populate init attr with rank/nranks/uniqueid
inline int nvshmemx_set_attr_uniqueid_args(int rank, int nranks,
                                            const nvshmemx_uniqueid_t* id,
                                            nvshmemx_init_attr_t* attr) {
  if (id == nullptr || attr == nullptr) return -1;
  attr->args.uid_args.id = const_cast<nvshmemx_uniqueid_t*>(id);
  attr->args.uid_args.myrank = rank;
  attr->args.uid_args.nranks = nranks;
  return 0;
}

inline int nvshmemx_get_uniqueid(nvshmemx_uniqueid_t* id) {
  if (id == nullptr) return -1;
  ncclUniqueId ncclId;
  ncclResult_t ret = ncclGetUniqueId(&ncclId);
  if (ret != ncclSuccess) return -1;
  memcpy(id, &ncclId, sizeof(ncclId));
  return 0;
}

// ---------------------------------------------------------------------------
// Host-side memory allocation (free-list allocator within symmetric heap)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Free-list helpers (insert/remove for fixed-size block array)
// ---------------------------------------------------------------------------
namespace niin { namespace detail {

inline bool heapInsert(GlobalState& s, int pos, HeapBlock blk) {
  if (s.heapBlockCount >= NIIN_MAX_HEAP_BLOCKS) return false;
  for (int j = s.heapBlockCount; j > pos; j--)
    s.heapBlocks[j] = s.heapBlocks[j - 1];
  s.heapBlocks[pos] = blk;
  s.heapBlockCount++;
  return true;
}

inline void heapRemove(GlobalState& s, int pos) {
  for (int j = pos; j < s.heapBlockCount - 1; j++)
    s.heapBlocks[j] = s.heapBlocks[j + 1];
  s.heapBlockCount--;
}

// Allocate from the free list with a given alignment.
inline void* heapAlloc(GlobalState& s, size_t size, size_t alignment) {
  if (size == 0) return nullptr;
  size_t align = alignment > NIIN_ALLOC_ALIGN ? alignment : NIIN_ALLOC_ALIGN;
  size_t alignedSize = (size + align - 1) & ~(align - 1);

  for (int i = 0; i < s.heapBlockCount; i++) {
    auto& blk = s.heapBlocks[i];
    if (!blk.free) continue;

    size_t alignedOffset = (blk.offset + align - 1) & ~(align - 1);
    size_t padding = alignedOffset - blk.offset;
    if (padding + alignedSize > blk.size) continue;

    // Split off leading padding as a free fragment
    if (padding > 0) {
      if (!heapInsert(s, i, {blk.offset, padding, true, true})) return nullptr;
      i++;
      s.heapBlocks[i].offset = alignedOffset;
      s.heapBlocks[i].size -= padding;
    }

    auto& alloc = s.heapBlocks[i];
    size_t remainder = alloc.size - alignedSize;

    if (remainder >= align) {
      if (!heapInsert(s, i + 1, {alloc.offset + alignedSize, remainder, true, true}))
        return nullptr;
      alloc.size = alignedSize;
    }
    alloc.free = false;
    return (char*)s.heapBase + alloc.offset;
  }
  return nullptr;
}

}} // namespace niin::detail

// ---------------------------------------------------------------------------
// Host-side memory allocation (free-list allocator within symmetric heap)
// ---------------------------------------------------------------------------

inline void* nvshmem_malloc(size_t size) {
  auto& s = niin::detail::state();
  if (!s.initialized) { fprintf(stderr, "NIIN: not initialized\n"); return nullptr; }
  void* ptr = niin::detail::heapAlloc(s, size, NIIN_ALLOC_ALIGN);
  if (!ptr && size > 0)
    fprintf(stderr, "NIIN: symmetric heap exhausted (requested %zu, heap %zu)\n",
            size, s.heapSize);
  return ptr;
}

inline void* nvshmem_calloc(size_t count, size_t size) {
  size_t total = count * size;
  void* ptr = nvshmem_malloc(total);
  if (ptr) cudaMemset(ptr, 0, total);
  return ptr;
}

inline void* nvshmem_align(size_t alignment, size_t size) {
  auto& s = niin::detail::state();
  if (!s.initialized) { fprintf(stderr, "NIIN: not initialized\n"); return nullptr; }
  return niin::detail::heapAlloc(s, size, alignment);
}

// nvshmem_free: returns memory to the free list with coalescing.
inline void nvshmem_free(void* ptr) {
  if (!ptr) return;
  auto& s = niin::detail::state();
  if (!s.initialized) return;

  size_t offset = (size_t)((char*)ptr - (char*)s.heapBase);

  for (int i = 0; i < s.heapBlockCount; i++) {
    if (s.heapBlocks[i].offset == offset && !s.heapBlocks[i].free) {
      s.heapBlocks[i].free = true;

      // Coalesce with next block
      if (i + 1 < s.heapBlockCount && s.heapBlocks[i + 1].free) {
        s.heapBlocks[i].size += s.heapBlocks[i + 1].size;
        niin::detail::heapRemove(s, i + 1);
      }
      // Coalesce with previous block
      if (i > 0 && s.heapBlocks[i - 1].free) {
        s.heapBlocks[i - 1].size += s.heapBlocks[i].size;
        niin::detail::heapRemove(s, i);
      }
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Host-side trampoline functions.
// These are called by the __host__ __device__ wrappers in query.h,
// collectives.h, and sync.h via extern "C" declarations. They read from
// the global singleton state.
// ---------------------------------------------------------------------------

inline int niin_host_my_pe() { return niin::detail::state().rank; }
inline int niin_host_n_pes() { return niin::detail::state().nRanks; }

inline void* niin_host_ptr(void* ptr, int pe) {
  auto& s = niin::detail::state();
  // Check that ptr is within the symmetric heap
  uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t base = reinterpret_cast<uintptr_t>(s.heapBase);
  if (p < base || p >= base + s.heapSize) return nullptr;
  if (pe == s.rank) return ptr;
  size_t offset = (size_t)(p - base);
  void* peerPtr = nullptr;
  ncclResult_t r = ncclGetPeerDevicePointer(s.hostCtx.heapWindow, offset, pe, &peerPtr);
  return (r == ncclSuccess) ? peerPtr : nullptr;
}

inline int niin_host_team_my_pe(int team) {
  return niin_teams_my_pe(team);
}

inline int niin_host_team_n_pes(int team) {
  return niin_teams_n_pes(team);
}

inline void niin_host_barrier_all() {
  auto& s = niin::detail::state();
  if (!s.initialized) return;
  ncclAllReduce(s.barrierScratch, s.barrierScratch, 1, ncclInt32, ncclSum, s.comm, s.stream);
  cudaStreamSynchronize(s.stream);
}

inline void niin_host_fence() {
  cudaDeviceSynchronize();
}

inline void niin_host_quiet() {
  niin_host_fence();
}

// ---------------------------------------------------------------------------
// Host-side info
// ---------------------------------------------------------------------------
inline void nvshmem_info_get_name(char* name) {
  snprintf(name, 256, "NIIN (NVSHMEM Implemented In NCCL)");
}

inline void nvshmem_info_get_version(int* major, int* minor) {
  *major = 3; *minor = 0;  // Report as NVSHMEM 3.0 compatible
}

// ---------------------------------------------------------------------------
// Convenience: get the device context pointer for passing to kernels.
// Usage: myKernel<<<...>>>(niin_get_device_ctx(), ...);
// Then in kernel: niin_set_context(ctx);
// ---------------------------------------------------------------------------
inline niinContext* niin_get_device_ctx() {
  return niin::detail::state().devCtx;
}

#endif // NIIN_NVSHMEM_HOST_H_
