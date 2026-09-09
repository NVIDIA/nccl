/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN-compatible shim for NVSHMEM's test/common/utils.h.
// Provides the same symbols so NVSHMEM test .cu files compile unmodified.

#ifndef NIIN_COMPAT_UTILS_H_
#define NIIN_COMPAT_UTILS_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <inttypes.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>

#include "nvshmem.h"
#include "nvshmemx.h"

#ifdef NIIN_HAS_MPI
#include <mpi.h>
#endif

// ---------------------------------------------------------------------------
// Error checking macros (matching NVSHMEM test harness)
// ---------------------------------------------------------------------------
#undef CUDA_CHECK
#define CUDA_CHECK(stmt) do {                                                  \
  cudaError_t r = (stmt);                                                      \
  if (r != cudaSuccess) {                                                      \
    fprintf(stderr, "[%s:%d] CUDA error: %s\n", __FILE__, __LINE__,           \
            cudaGetErrorString(r));                                             \
    exit(1);                                                                    \
  }                                                                            \
} while(0)

#define CU_CHECK(stmt) do {                                                    \
  CUresult r = (stmt);                                                         \
  if (r != CUDA_SUCCESS) {                                                     \
    const char* str;                                                           \
    cuGetErrorString(r, &str);                                                 \
    fprintf(stderr, "[%s:%d] CUDA driver error: %s\n", __FILE__, __LINE__, str); \
    exit(1);                                                                    \
  }                                                                            \
} while(0)

#define ERROR_EXIT(...) do {                                                   \
  fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);                              \
  fprintf(stderr, __VA_ARGS__);                                                \
  exit(1);                                                                      \
} while(0)

#define ERROR_PRINT(...) do {                                                  \
  fprintf(stderr, "%s:%d: ", __FILE__, __LINE__);                              \
  fprintf(stderr, __VA_ARGS__);                                                \
} while(0)

#ifdef _NVSHMEM_DEBUG
#define DEBUG_PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define DEBUG_PRINT(...) do {} while(0)
#endif

#define NVSHMEMI_TEST_STRINGIFY(x) #x

// ---------------------------------------------------------------------------
// Return code
// ---------------------------------------------------------------------------
#define NVSHMEMX_SUCCESS 0
#define NVSHMEM_MAX_NAME_LEN 256
#define NVSHMEM_VENDOR_STRING "NIIN"
#define NVSHMEM_MAJOR_VERSION 3
#define NVSHMEM_MINOR_VERSION 0
#define NVSHMEM_PATCH_VERSION 0
#define NVSHMEM_VENDOR_MAJOR_VERSION 3
#define NVSHMEM_VENDOR_MINOR_VERSION 0
#define NVSHMEM_VENDOR_PATCH_VERSION 0
#define NVSHMEM_VENDOR_VERSION ((NVSHMEM_VENDOR_MAJOR_VERSION << 16) | (NVSHMEM_VENDOR_MINOR_VERSION << 8) | NVSHMEM_VENDOR_PATCH_VERSION)

// Printf format specifiers (matching NVSHMEM)
#define NVSHPRI_half "%f"
#define NVSHPRI_bfloat16 "%f"
#define NVSHPRI_float "%.2f"
#define NVSHPRI_double "%.2f"
#define NVSHPRI_char "%hhd"
#define NVSHPRI_schar "%hhd"
#define NVSHPRI_short "%hd"
#define NVSHPRI_int "%d"
#define NVSHPRI_long "%ld"
#define NVSHPRI_longlong "%lld"
#define NVSHPRI_uchar "%hhu"
#define NVSHPRI_ushort "%hu"
#define NVSHPRI_uint "%u"
#define NVSHPRI_ulong "%lu"
#define NVSHPRI_ulonglong "%llu"
#define NVSHPRI_int8 "%" PRIi8
#define NVSHPRI_int16 "%" PRIi16
#define NVSHPRI_int32 "%" PRIi32
#define NVSHPRI_int64 "%" PRIi64
#define NVSHPRI_uint8 "%" PRIu8
#define NVSHPRI_uint16 "%" PRIu16
#define NVSHPRI_uint32 "%" PRIu32
#define NVSHPRI_uint64 "%" PRIu64
#define NVSHPRI_size "%zu"
#define NVSHPRI_ptrdiff "%td"

// ---------------------------------------------------------------------------
// Global variables (matching NVSHMEM test harness)
// ---------------------------------------------------------------------------
static int mype = 0, mype_node = 0, npes = 0, npes_node = 0;
static int use_cubin = 0;
static bool use_mmap = false;
static bool use_egm = false;
static size_t _mem_handle_type = 0;

// From perftest-style args
static size_t min_size = 4;
static size_t max_size = 4 * 1024 * 1024;
static size_t step_factor = 2;
static size_t iters = 200;
static size_t warmup_iters = 50;
static size_t threads_per_block = 256;
static size_t num_blocks = 4;

// ---------------------------------------------------------------------------
// nvshmemx_signal_op — device-side signal set/add to remote PE.
// Uses niin_deliver_signal which routes through GIN on PCIe GPUs.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void nvshmemx_signal_op(
    uint64_t* sig_addr, uint64_t value, int sig_op, int pe) {
  niin_deliver_signal(sig_addr, value, sig_op, pe);
}

// ---------------------------------------------------------------------------
// nvshmemx_barrier_all_block/warp — device-side cooperative barriers.
// On NVLink (native atomics): uses ncclBarrierSession (fast LSA barrier).
// On PCIe (no native atomics): uses GIN-based signal exchange to avoid
// peer pointer atomics that corrupt subsequent cooperative stores.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void niin_gin_barrier_all() {
  // GIN-based barrier: each PE signals all others via GIN, waits for signals back.
  // Avoids peer pointer atomics entirely.
  int pe = niin_device_my_pe();
  int npes = niin_device_n_pes();
  ncclDevComm const& comm = niin_comm();
  ncclGin gin(comm, niin_gin_context_index());
  ncclTeam world = ncclTeamWorld(comm);

  // Use a dedicated signal in the heap for barrier (last 64 bytes)
  uint64_t* bar_sig = (uint64_t*)((char*)niin_heap_base() + niin_heap_size() - 64);
  uint64_t epoch = *bar_sig;

  // Signal all other PEs via GIN
  for (int i = 0; i < npes; i++) {
    if (i != pe) {
      size_t sigOff = niin_sym_offset(bar_sig);
      gin.putValue<uint64_t>(world, i, niin_heap_window(), sigOff, epoch + 1);
    }
  }
  gin.flush(ncclCoopThread{});

  // Wait for all peers to signal us
  volatile uint64_t* v = (volatile uint64_t*)bar_sig;
  while (*v < epoch + 1) {}
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
  *bar_sig = epoch + 1;
}

__device__ __forceinline__ void nvshmemx_barrier_all_block() {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncthreads();
  if (threadIdx.x == 0) {
    if (niin_peer_native_atomic())
      nvshmem_barrier_all();
    else
      niin_gin_barrier_all();
  }
  __syncthreads();
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
}

__device__ __forceinline__ void nvshmemx_barrier_all_warp() {
  cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);
  __syncwarp();
  if (nccl::utility::lane() == 0) {
    if (niin_peer_native_atomic())
      nvshmem_barrier_all();
    else
      niin_gin_barrier_all();
  }
  __syncwarp();
  cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
}

// ---------------------------------------------------------------------------
// nvshmtest threadgroup scope helpers (used by put_signal tests)
// ---------------------------------------------------------------------------
// Thread scope
__device__ __forceinline__ int nvshmtest_thread_id_in_thread() { return 0; }
__device__ __forceinline__ int nvshmtest_thread_size() { return 1; }
__device__ __forceinline__ void nvshmtest_thread_sync() {}

// Warp scope
__device__ __forceinline__ int nvshmtest_thread_id_in_warp() {
  return nccl::utility::lane();
}
__device__ __forceinline__ int nvshmtest_warp_size() { return 32; }
__device__ __forceinline__ void nvshmtest_warp_sync() { __syncwarp(); }

// Block scope
__device__ __forceinline__ int nvshmtest_thread_id_in_block() {
  return threadIdx.x;
}
__device__ __forceinline__ int nvshmtest_block_size() { return blockDim.x; }
__device__ __forceinline__ void nvshmtest_block_sync() { __syncthreads(); }

// ---------------------------------------------------------------------------
// Init / finalize wrappers
// ---------------------------------------------------------------------------
static void init_wrapper(int* argc, char*** argv) {
#ifdef NIIN_HAS_MPI
  MPI_Init(argc, argv);
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  MPI_Comm comm = MPI_COMM_WORLD;
  attr.mpi_comm = &comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
#else
  nvshmem_init();
#endif
  mype = nvshmem_my_pe();
  npes = nvshmem_n_pes();
  mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  npes_node = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
}

static void finalize_wrapper() {
  nvshmem_finalize();
#ifdef NIIN_HAS_MPI
  MPI_Finalize();
#endif
}

// ---------------------------------------------------------------------------
// read_args — parse CLI args (simplified, matching NVSHMEM conventions)
// ---------------------------------------------------------------------------
static void read_args(int argc, char** argv) {
  static struct option long_options[] = {
    {"egm", no_argument, 0, 0},
    {"mmap", no_argument, 0, 0},
    {"min_size", required_argument, 0, 'b'},
    {"max_size", required_argument, 0, 'e'},
    {"step", required_argument, 0, 'f'},
    {"iters", required_argument, 0, 'i'},
    {"warmup", required_argument, 0, 'w'},
    {"threads", required_argument, 0, 't'},
    {"blocks", required_argument, 0, 'n'},
    {"help", no_argument, 0, 'h'},
    {0, 0, 0, 0}
  };
  int c, option_index = 0;
  optind = 1;  // reset getopt
  while ((c = getopt_long(argc, argv, "b:e:f:i:w:t:n:h", long_options, &option_index)) != -1) {
    switch (c) {
      case 0:
        if (strcmp(long_options[option_index].name, "egm") == 0) use_egm = true;
        else if (strcmp(long_options[option_index].name, "mmap") == 0) use_mmap = true;
        break;
      case 'b': min_size = atol(optarg); break;
      case 'e': max_size = atol(optarg); break;
      case 'f': step_factor = atol(optarg); break;
      case 'i': iters = atol(optarg); break;
      case 'w': warmup_iters = atol(optarg); break;
      case 't': threads_per_block = atol(optarg); break;
      case 'n': num_blocks = atol(optarg); break;
      case 'h': break;
      default: break;
    }
  }
}

// ---------------------------------------------------------------------------
// CUBIN module stubs (NIIN doesn't use CUBIN modules)
// ---------------------------------------------------------------------------
static CUmodule mymodule = NULL;

static void init_cumodule(const char* str) {
  (void)str;
  // NIIN doesn't support CUBIN module testing
}

static void init_test_case_kernel(CUfunction* kernel, const char* name) {
  (void)kernel; (void)name;
}

// ---------------------------------------------------------------------------
// mmap buffer stubs (not supported in NIIN)
// ---------------------------------------------------------------------------
static void* allocate_mmap_buffer(size_t size, int mem_handle_type,
                                   bool egm = false, bool reset_zero = false) {
  (void)mem_handle_type; (void)egm;
  // Fall back to nvshmem_malloc
  void* ptr = nvshmem_malloc(size);
  if (ptr && reset_zero) cudaMemset(ptr, 0, size);
  return ptr;
}

static void free_mmap_buffer(void* ptr) {
  nvshmem_free(ptr);
}

// ---------------------------------------------------------------------------
// Vendor version info
// ---------------------------------------------------------------------------
inline void nvshmemx_vendor_get_version_info(int* major, int* minor, int* patch) {
  *major = NVSHMEM_VENDOR_MAJOR_VERSION;
  *minor = NVSHMEM_VENDOR_MINOR_VERSION;
  *patch = NVSHMEM_VENDOR_PATCH_VERSION;
}

// ---------------------------------------------------------------------------
// NVML stubs
// ---------------------------------------------------------------------------
static bool is_mnnvl_supported(int dev_id) {
  (void)dev_id;
  return false;  // Conservative: assume no MNNVL
}

// ---------------------------------------------------------------------------
// Table allocation / printing (for perftests that some tests reference)
// ---------------------------------------------------------------------------
static void alloc_tables(void*** table_mem, int num_tables, int num_entries) {
  *table_mem = (void**)malloc(num_tables * sizeof(void*));
  for (int i = 0; i < num_tables; i++)
    (*table_mem)[i] = calloc(num_entries, sizeof(uint64_t));
}

static void free_tables(void** tables, int num_tables) {
  for (int i = 0; i < num_tables; i++) free(tables[i]);
  free(tables);
}

static void print_table_basic(const char* job, const char* scope,
                               const char* var, const char* out,
                               const char* units, char pm,
                               uint64_t* sizes, double* values, int n) {
  (void)pm;
  printf("\n#%s\n", job);
  printf("%-16s %-10s %s (%s)\n", var, "scope", out, units);
  for (int i = 0; i < n; i++)
    printf("%-16lu %-10s %.6f\n", (unsigned long)sizes[i], scope, values[i]);
  printf("\n");
  fflush(stdout);
}

#endif // NIIN_COMPAT_UTILS_H_
