/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN perftest common harness.
// Replaces NVSHMEM's perftest/common/utils.h with a self-contained
// version that uses NIIN's nvshmem_init/nvshmem_malloc API.

#ifndef NIIN_PERFTEST_H_
#define NIIN_PERFTEST_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>

#include "nvshmem.h"
#include "nvshmemx.h"

#ifdef NIIN_HAS_MPI
#include <mpi.h>
#endif

// ---------------------------------------------------------------------------
// Error checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(stmt) do {                                                  \
  cudaError_t r = (stmt);                                                      \
  if (r != cudaSuccess) {                                                      \
    fprintf(stderr, "[%s:%d] CUDA error: %s\n", __FILE__, __LINE__,           \
            cudaGetErrorString(r));                                             \
    exit(1);                                                                    \
  }                                                                            \
} while(0)

// ---------------------------------------------------------------------------
// Global perftest parameters
// ---------------------------------------------------------------------------
static size_t min_size = 4;
static size_t max_size = 4 * 1024 * 1024;
static size_t step_factor = 2;
static size_t iters = 200;
static size_t warmup_iters = 50;
static size_t threads_per_block = 256;
static size_t num_blocks = 4;
static size_t max_size_log = 0;

// ---------------------------------------------------------------------------
// CLI argument parsing (matching NVSHMEM perftest conventions)
// ---------------------------------------------------------------------------
static void read_args(int argc, char** argv) {
  static struct option long_options[] = {
    {"min_size",   required_argument, 0, 'b'},
    {"max_size",   required_argument, 0, 'e'},
    {"step",       required_argument, 0, 'f'},
    {"iters",      required_argument, 0, 'i'},
    {"warmup",     required_argument, 0, 'w'},
    {"threads",    required_argument, 0, 't'},
    {"blocks",     required_argument, 0, 'n'},
    {"help",       no_argument,       0, 'h'},
    {0, 0, 0, 0}
  };
  int c, option_index = 0;
  while ((c = getopt_long(argc, argv, "b:e:f:i:w:t:n:h", long_options, &option_index)) != -1) {
    switch (c) {
      case 'b': min_size = atol(optarg); break;
      case 'e': max_size = atol(optarg); break;
      case 'f': step_factor = atol(optarg); break;
      case 'i': iters = atol(optarg); break;
      case 'w': warmup_iters = atol(optarg); break;
      case 't': threads_per_block = atol(optarg); break;
      case 'n': num_blocks = atol(optarg); break;
      case 'h':
        printf("Usage: [options]\n"
               "  -b, --min_size <bytes>    Minimum message size (default: 4)\n"
               "  -e, --max_size <bytes>    Maximum message size (default: 4M)\n"
               "  -f, --step <factor>       Step factor (default: 2)\n"
               "  -i, --iters <n>           Iterations (default: 200)\n"
               "  -w, --warmup <n>          Warmup iterations (default: 50)\n"
               "  -t, --threads <n>         Threads per block (default: 256)\n"
               "  -n, --blocks <n>          Number of blocks (default: 4)\n");
        exit(0);
      default: break;
    }
  }
  // Compute log2 of max_size for table allocation
  size_t tmp = max_size;
  max_size_log = 0;
  while (tmp > 0) { max_size_log++; tmp /= step_factor; }
}

// ---------------------------------------------------------------------------
// Table allocation and printing (matching NVSHMEM format)
// ---------------------------------------------------------------------------
static void alloc_tables(void*** table_mem, int num_tables, int num_entries) {
  *table_mem = (void**)malloc(num_tables * sizeof(void*));
  for (int i = 0; i < num_tables; i++) {
    (*table_mem)[i] = calloc(num_entries, sizeof(uint64_t) > sizeof(double)
                              ? sizeof(uint64_t) : sizeof(double));
  }
}

static void free_tables(void** tables, int num_tables) {
  for (int i = 0; i < num_tables; i++) free(tables[i]);
  free(tables);
}

static void print_table_basic(const char* job_name, const char* scope_name,
                               const char* var_name, const char* output_name,
                               const char* units, char /*plus_minus*/,
                               uint64_t* sizes, double* values, int n) {
  printf("\n%s (%s)\n", job_name, scope_name);
  printf("%-16s %12s (%s)\n", var_name, output_name, units);
  printf("%-16s %12s\n", "----------------", "------------");
  for (int i = 0; i < n; i++) {
    printf("%-16lu %12.3f\n", (unsigned long)sizes[i], values[i]);
  }
  printf("\n");
  fflush(stdout);
}

// BW table: prints bandwidth in GB/s
static void print_table_bw(const char* job_name, const char* scope_name,
                            uint64_t* sizes, double* bw, int n) {
  printf("\n%s (%s)\n", job_name, scope_name);
  printf("%-16s %12s\n", "size (Bytes)", "BW (GB/s)");
  printf("%-16s %12s\n", "----------------", "------------");
  for (int i = 0; i < n; i++) {
    printf("%-16lu %12.3f\n", (unsigned long)sizes[i], bw[i]);
  }
  printf("\n");
  fflush(stdout);
}

// ---------------------------------------------------------------------------
// Init/finalize wrappers
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
}

static void finalize_wrapper() {
  nvshmem_finalize();
#ifdef NIIN_HAS_MPI
  MPI_Finalize();
#endif
}

#endif // NIIN_PERFTEST_H_
