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
// For multi-PE operation, nvshmem_init() auto-detects supported launcher
// rank/size metadata and distributes an NCCL unique ID through a small record
// in a shared bootstrap directory. Alternatively, launch with MPI and call
// nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr), or use
// NVSHMEMX_INIT_WITH_UNIQUEID after distributing a unique ID with the
// application's bootstrap mechanism.
//
// For single-PE operation (testing), just call nvshmem_init().

#ifndef NIIN_NVSHMEM_HOST_H_
#define NIIN_NVSHMEM_HOST_H_

#include <nccl.h>
#include <cuda_runtime.h>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <limits.h>
#include <strings.h>  // strcasecmp, for NVSHMEM_TMA_POLICY parsing
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "niin/host.h"
#include "niin/sync.h"

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

// Publish the device context from device code rather than through the CUDA
// runtime's host-side symbol lookup.  The latter is not reliable for an
// externally-owned RDC device symbol in every consumer module: a successful
// NIIN host initialization can otherwise leave the consumer kernel's
// niin_g_ctx null.  This kernel and all device API wrappers resolve the same
// device-link symbol.
static __global__ void publishDeviceContext(niinContext* ctx) {
  niin_g_ctx = ctx;
}

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
  int nodeRank;
  int nodeSize;
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
  struct niinGpunetioAtomicHostContext* gpunetioAtomics; // Network AMO provider, when enabled
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

// Parse NVSHMEM_TMA_POLICY, matching NVSHMEM's DISABLE/ENABLE/FORCE spelling.
// Unrecognized values fall back to DISABLE with a diagnostic, as NVSHMEM does.
inline nvshmemx_tma_policy_t parseTmaPolicy() {
  const char* env = getenv("NVSHMEM_TMA_POLICY");
  if (env == nullptr) return NVSHMEMX_TMA_DISABLE;
  if (strcasecmp(env, "ENABLE") == 0) return NVSHMEMX_TMA_ENABLE;
  if (strcasecmp(env, "FORCE") == 0) return NVSHMEMX_TMA_FORCE;
  if (strcasecmp(env, "DISABLE") == 0) return NVSHMEMX_TMA_DISABLE;
  fprintf(stderr, "NIIN: invalid NVSHMEM_TMA_POLICY value \"%s\"; using DISABLE\n", env);
  return NVSHMEMX_TMA_DISABLE;
}

// Resolve the requested TMA policy against the current device's compute
// capability. TMA needs sm_90 or newer: FORCE reports failure there, ENABLE
// degrades to DISABLE with a warning. Returns false when init should fail.
inline bool resolveTmaPolicy(nvshmemx_tma_policy_t* policy) {
  if (*policy == NVSHMEMX_TMA_DISABLE) return true;

  int dev = 0;
  cudaGetDevice(&dev);
  int capMajor = 0, capMinor = 0;
  cudaDeviceGetAttribute(&capMajor, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&capMinor, cudaDevAttrComputeCapabilityMinor, dev);
  if (capMajor >= 9) return true;

  if (*policy == NVSHMEMX_TMA_FORCE) {
    fprintf(stderr, "NIIN: NVSHMEM_TMA_POLICY=FORCE requires sm_90 or newer; "
                    "device %d is sm_%d%d\n", dev, capMajor, capMinor);
    return false;
  }
  fprintf(stderr, "NIIN: NVSHMEM_TMA_POLICY=ENABLE requires sm_90 or newer; "
                  "device %d is sm_%d%d, disabling TMA\n", dev, capMajor, capMinor);
  *policy = NVSHMEMX_TMA_DISABLE;
  return true;
}

// NIIN_GPUNETIO_ATOMICS selects how hard NIIN tries to bring up the network
// AMO provider: 0 disables it, 1 requires it and reports why it did not come
// up, and the default enables it whenever the provider is linked and the
// fabric supports it.
inline int parseGpunetioAtomicsMode() {
  const char* env = getenv("NIIN_GPUNETIO_ATOMICS");
  if (env == nullptr || env[0] == '\0') return -1;
  if (strcmp(env, "0") == 0) return 0;
  if (strcmp(env, "1") == 0) return 1;
  fprintf(stderr, "NIIN: NIIN_GPUNETIO_ATOMICS must be 0 or 1\n");
  return -1;
}

// Bring up the network atomic provider. Its initialization is collective and
// all-gathers QP metadata internally, so every PE has to agree before any of
// them enters it: a PE that bailed out early while the others proceeded would
// hang them. Agree first, then initialize.
inline void enableGpunetioAtomics(GlobalState& s) {
  s.gpunetioAtomics = nullptr;
  const int mode = parseGpunetioAtomicsMode();
  if (mode == 0) return;

  const bool linked = (niinGpunetioAtomicInit != nullptr && niinGpunetioAtomicBind != nullptr &&
                       niinGpunetioAtomicFinalize != nullptr);
  // Network AMOs only mean something when some peer is off this LSA domain.
  int local = (linked && s.nRanks > s.lsaSize) ? 1 : 0;

  int agreed = local;
  if (s.nRanks > 1) {
    int* dev = static_cast<int*>(s.barrierScratch);
    if (dev == nullptr) return;
    if (cudaMemcpy(dev, &local, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return;
    if (ncclAllReduce(dev, dev, 1, ncclInt, ncclMin, s.comm, s.stream) != ncclSuccess) return;
    if (cudaStreamSynchronize(s.stream) != cudaSuccess) return;
    if (cudaMemcpy(&agreed, dev, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return;
  }
  if (agreed == 0) {
    if (mode == 1 && s.rank == 0) {
      fprintf(stderr, "NIIN: NIIN_GPUNETIO_ATOMICS=1 but the provider is %s\n",
              linked ? "unusable on this job's topology" : "not linked into this application");
    }
    return;
  }

  niinGpunetioAtomicOptions options = NIIN_GPUNETIO_ATOMIC_OPTIONS_INITIALIZER;
  struct niinGpunetioAtomicHostContext* provider = nullptr;
  ncclResult_t r = niinGpunetioAtomicInit(s.comm, s.heapBase, s.heapSize, &options, &provider);
  if (r == ncclSuccess) r = niinGpunetioAtomicBind(provider, s.devCtx);
  if (r != ncclSuccess) {
    if (provider != nullptr) niinGpunetioAtomicFinalize(provider);
    if (mode == 1 && s.rank == 0)
      fprintf(stderr, "NIIN: network atomic provider unavailable (%s); network AMOs stay unimplemented\n",
              ncclGetErrorString(r));
    return;
  }
  s.gpunetioAtomics = provider;
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

// Generate a launcher-scoped path in a shared filesystem.  A caller can set
// NIIN_BOOTSTRAP_DIR and/or NIIN_BOOTSTRAP_ID to override the defaults.  The
// HOME fallback is appropriate for the common Slurm configuration where home
// directories are shared between allocated nodes.
inline bool bootstrapPath(char* path, size_t pathSize) {
  char directory[PATH_MAX];
  const char* configuredDirectory = getenv("NIIN_BOOTSTRAP_DIR");
  if (configuredDirectory != nullptr && configuredDirectory[0] != '\0') {
    if (snprintf(directory, sizeof(directory), "%s", configuredDirectory) >=
        static_cast<int>(sizeof(directory))) {
      fprintf(stderr, "NIIN: NIIN_BOOTSTRAP_DIR is too long\n");
      return false;
    }
  } else {
    const char* home = getenv("HOME");
    if (home == nullptr || home[0] == '\0' ||
        snprintf(directory, sizeof(directory), "%s/.niin", home) >=
          static_cast<int>(sizeof(directory))) {
      fprintf(stderr, "NIIN: set NIIN_BOOTSTRAP_DIR to a shared directory\n");
      return false;
    }
  }

  if (mkdir(directory, 0700) != 0 && errno != EEXIST) {
    fprintf(stderr, "NIIN: cannot create bootstrap directory %s: %s\n",
            directory, strerror(errno));
    return false;
  }

  char generatedId[128];
  const char* id = getenv("NIIN_BOOTSTRAP_ID");
  if (id == nullptr || id[0] == '\0') {
    const char* jobId = getenv("SLURM_JOB_ID");
    if (jobId == nullptr || jobId[0] == '\0') jobId = getenv("PMI_JOBID");
    if (jobId == nullptr || jobId[0] == '\0') jobId = getenv("PMIX_NAMESPACE");
    if (jobId == nullptr || jobId[0] == '\0') {
      fprintf(stderr, "NIIN: set NIIN_BOOTSTRAP_ID when the launcher has no job ID\n");
      return false;
    }
    const char* stepId = getenv("SLURM_STEP_ID");
    if (snprintf(generatedId, sizeof(generatedId), "%s.%s", jobId,
                 stepId != nullptr ? stepId : "0") >= static_cast<int>(sizeof(generatedId))) {
      fprintf(stderr, "NIIN: launcher bootstrap ID is too long\n");
      return false;
    }
    id = generatedId;
  }

  for (const char* c = id; *c != '\0'; ++c) {
    const bool safe = (*c >= 'a' && *c <= 'z') || (*c >= 'A' && *c <= 'Z') ||
                      (*c >= '0' && *c <= '9') || *c == '.' || *c == '_' || *c == '-';
    if (!safe) {
      fprintf(stderr, "NIIN: NIIN bootstrap ID contains an unsafe character\n");
      return false;
    }
  }

  if (snprintf(path, pathSize, "%s/niin-bootstrap-%s", directory, id) >=
      static_cast<int>(pathSize)) {
    fprintf(stderr, "NIIN: bootstrap path is too long\n");
    return false;
  }
  return true;
}

inline bool writeBootstrapRecord(int fd, const void* buffer, size_t size) {
  const char* cursor = static_cast<const char*>(buffer);
  while (size > 0) {
    ssize_t written = write(fd, cursor, size);
    if (written < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    cursor += written;
    size -= static_cast<size_t>(written);
  }
  return true;
}

inline bool readBootstrapRecord(int fd, void* buffer, size_t size) {
  char* cursor = static_cast<char*>(buffer);
  while (size > 0) {
    ssize_t readCount = read(fd, cursor, size);
    if (readCount < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    if (readCount == 0) return false;
    cursor += readCount;
    size -= static_cast<size_t>(readCount);
  }
  return true;
}

struct BootstrapRecord {
  uint64_t magic;
  int nRanks;
  ncclUniqueId id;
};

constexpr uint64_t NIIN_BOOTSTRAP_MAGIC = UINT64_C(0x4e49494e424f4f54);

// Distribute an NCCL unique ID without requiring the application to initialize
// MPI.  Rank zero publishes one small record atomically; other ranks wait for
// it and validate the expected world size before joining the communicator.
inline int bootstrapUniqueId(int rank, int nRanks, ncclUniqueId* id,
                             char* path, size_t pathSize) {
  if (!bootstrapPath(path, pathSize)) return -1;

  if (rank == 0) {
    ncclResult_t result = ncclGetUniqueId(id);
    if (result != ncclSuccess) {
      fprintf(stderr, "NIIN: ncclGetUniqueId failed: %s\n", ncclGetErrorString(result));
      return -1;
    }

    const int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) {
      fprintf(stderr, "NIIN: cannot create bootstrap record %s: %s\n", path, strerror(errno));
      return -1;
    }
    const BootstrapRecord record = {NIIN_BOOTSTRAP_MAGIC, nRanks, *id};
    const bool written = writeBootstrapRecord(fd, &record, sizeof(record)) && fsync(fd) == 0;
    close(fd);
    if (!written) {
      fprintf(stderr, "NIIN: cannot write bootstrap record %s: %s\n", path, strerror(errno));
      unlink(path);
      return -1;
    }
    return 0;
  }

  constexpr int kRetries = 6000;  // 60 seconds at 10 ms per attempt.
  for (int attempt = 0; attempt < kRetries; ++attempt) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
      if (errno == ENOENT) {
        usleep(10000);
        continue;
      }
      fprintf(stderr, "NIIN: cannot open bootstrap record %s: %s\n", path, strerror(errno));
      return -1;
    }
    BootstrapRecord record = {};
    const bool readOk = readBootstrapRecord(fd, &record, sizeof(record));
    close(fd);
    if (!readOk) {
      usleep(10000);
      continue;
    }
    if (record.magic != NIIN_BOOTSTRAP_MAGIC || record.nRanks != nRanks) {
      fprintf(stderr, "NIIN: bootstrap record %s does not match this job\n", path);
      return -1;
    }
    *id = record.id;
    return 0;
  }

  fprintf(stderr, "NIIN: timed out waiting for bootstrap record %s\n", path);
  return -1;
}

// Detect this PE's rank on its physical node. Tries LOCAL_RANK-style env
// vars, falling back to rank % deviceCount.
inline int detectLocalRank(int rank) {
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

inline int detectLocalDevice(int rank) {
  return detectLocalRank(rank);
}

inline int detectNodeSize() {
  const char* localSizeEnvs[] = {
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MPI_LOCALNRANKS",
    "SLURM_NTASKS_PER_NODE",
  };
  for (const char* env : localSizeEnvs) {
    const char* value = getenv(env);
    if (value == nullptr || value[0] == '\0') continue;
    char* end = nullptr;
    const long size = strtol(value, &end, 10);
    if (end != value && size > 0 && size <= INT_MAX) return static_cast<int>(size);
  }
  return 1;
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
  r = niinInit(comm, s.heapBase, s.heapSize, &s.hostCtx, s.ginAvail,
               /*barrierCount=*/1, /*ginSignalCount=*/0,
               /*ginContextCount=*/0, s.nodeRank, s.nodeSize);
  ncclGroupEnd();
  if (r != ncclSuccess) return -1;

  // TMA registration tables, if the policy asks for them. niinInit left the
  // context with TMA off, so a DISABLE policy needs nothing further.
  {
    nvshmemx_tma_policy_t policy = parseTmaPolicy();
    if (!resolveTmaPolicy(&policy)) return -1;
    if (niinTmaEnable(&s.hostCtx, policy) != ncclSuccess) return -1;
  }

  r = niinCommit(&s.hostCtx, s.devCtx);
  if (r != ncclSuccess) return -1;

  // Publish the context used by every device-side NIIN wrapper.  Do this in a
  // device kernel so the update resolves through the same RDC symbol as a
  // consumer kernel, including consumers that own the external context
  // symbol.  Check completion here: proceeding with a null context turns a
  // setup failure into an unrelated illegal memory access in the consumer.
  publishDeviceContext<<<1, 1>>>(s.devCtx);
  cudaError_t cudaResult = cudaGetLastError();
  if (cudaResult != cudaSuccess) {
    fprintf(stderr, "NIIN: device-context publication launch failed: %s\n",
            cudaGetErrorString(cudaResult));
    return -1;
  }
  cudaResult = cudaDeviceSynchronize();
  if (cudaResult != cudaSuccess) {
    fprintf(stderr, "NIIN: device-context publication failed: %s\n",
            cudaGetErrorString(cudaResult));
    return -1;
  }

  // Extract LSA info from the devComm
  s.lsaRank = s.hostCtx.devComm.lsaRank;
  s.lsaSize = s.hostCtx.devComm.lsaSize;

  // Initialize predefined teams
  niin::teams::initPredefined(s.rank, s.nRanks, s.lsaRank, s.lsaSize,
                              s.nodeRank, s.nodeSize);

  // Network AMOs are on by default wherever the provider can come up. A
  // failure here leaves them fail-closed, exactly as before the provider
  // existed, rather than failing initialization.
  enableGpunetioAtomics(s);

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
    if (threadIdx.x == 0)                          \
      niin_g_ctx = niin::detail::state().devCtx;   \
    __syncthreads();                               \
  } while (0)

// This doesn't work directly because devCtx is a host variable. The actual
// approach: pass devCtx as a kernel argument. We provide a helper:
__device__ inline void niin_set_context(niinContext* ctx) {
  if (threadIdx.x == 0) {
    niin_g_ctx = ctx;
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_device);
  }
  __syncthreads();
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_device);
}

// ---------------------------------------------------------------------------
// nvshmem_init / nvshmem_finalize
// ---------------------------------------------------------------------------

// nvshmem_init: initialize NIIN with auto-detected rank/size.
// Multi-PE: detects rank from supported launcher metadata and bootstraps the
// NCCL unique ID internally.  This preserves the ordinary NVSHMEM contract:
// an application can call nvshmem_init() directly without initializing MPI.
inline int nvshmem_init(void) {
  auto& s = niin::detail::state();
  if (s.initialized) {
    fprintf(stderr, "NIIN: already initialized\n");
    return -1;
  }

  int rank = 0, nRanks = 1;
  bool multiPE = niin::detail::detectRankFromEnv(&rank, &nRanks);

  s.nodeRank = niin::detail::detectLocalRank(rank);
  s.nodeSize = niin::detail::detectNodeSize();
  int localDev = s.nodeRank;
  cudaSetDevice(localDev);
  s.cudaDev = localDev;

  // Request host RMA support via ncclConfig
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;
  config.numRmaCtx = 1;

  ncclComm_t comm;
  if (multiPE && nRanks > 1) {
    ncclUniqueId id;
    char bootstrapFile[PATH_MAX];
    if (niin::detail::bootstrapUniqueId(rank, nRanks, &id, bootstrapFile,
                                        sizeof(bootstrapFile)) != 0) {
      return -1;
    }
    const ncclResult_t result = ncclCommInitRankConfig(&comm, nRanks, id, rank, &config);
    if (rank == 0) unlink(bootstrapFile);
    if (result != ncclSuccess) {
      fprintf(stderr, "NIIN: ncclCommInitRankConfig failed: %s\n", ncclGetErrorString(result));
      return -1;
    }
  } else {
    // Single-PE mode
    ncclUniqueId id;
    ncclResult_t result = ncclGetUniqueId(&id);
    if (result == ncclSuccess) result = ncclCommInitRankConfig(&comm, 1, id, 0, &config);
    if (result != ncclSuccess) {
      fprintf(stderr, "NIIN: single-PE communicator initialization failed: %s\n",
              ncclGetErrorString(result));
      return -1;
    }
  }

  return niin::detail::initCommon(comm);
}

// nvshmem_finalize: tear down NIIN global state.
inline void nvshmem_finalize(void) {
  auto& s = niin::detail::state();
  if (!s.initialized) return;

  cudaSetDevice(s.cudaDev);
  cudaDeviceSynchronize();

  // Tear the atomic provider down while its device context and the NCCL
  // communicator it all-gathered over are both still alive.
  if (s.gpunetioAtomics != nullptr) {
    niinGpunetioAtomicFinalize(s.gpunetioAtomics);
    s.gpunetioAtomics = nullptr;
  }

  ncclGroupStart();
  niinFinalize(s.comm, s.devCtx);
  ncclGroupEnd();

  niinTmaDisable(&s.hostCtx);
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
    s.nodeRank = niin::detail::detectLocalRank(rank);
    s.nodeSize = niin::detail::detectNodeSize();
    int localDev = s.nodeRank;
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

    s.nodeRank = niin::detail::detectLocalRank(rank);
    s.nodeSize = niin::detail::detectNodeSize();
    int localDev = s.nodeRank;
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
  auto& s = niin::detail::state();
  // Wait for in-flight kernels to retire first -- they may still be issuing puts.
  niin_host_fence();
  if (!s.initialized || !s.ginAvail) return;
  // A put can still be in flight on a GIN context after the kernel that issued it
  // has retired, so cudaDeviceSynchronize() alone does not complete it.
  niin_quiet_kernel<><<<1, 1, 0, s.stream>>>();
  cudaStreamSynchronize(s.stream);
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
