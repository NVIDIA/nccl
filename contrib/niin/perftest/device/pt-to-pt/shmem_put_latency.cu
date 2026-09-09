/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Put latency: measures nvshmem_int_put_nbi + nvshmem_quiet round-trip
// for thread, warp, and block scopes across message sizes.
//
// Build:
//   nvcc shmem_put_latency.cu -o shmem_put_latency -DNIIN_HAS_MPI \
//       -I ../../common -I ../../../include \
//       -I <nccl>/build/include \
//       -I <mpi>/include -L <nccl>/build/lib -lnccl -L <mpi>/lib -lmpi \
//       --expt-relaxed-constexpr -std=c++17 -arch=sm_90
//
// Run:
//   mpirun -np 2 ./shmem_put_latency

#include "perftest.h"

// --- Kernels ---

__global__ void latency_kern(int* data_d, int len, int pe, int iter) {
  int peer = !pe;
  for (int i = 0; i < iter; i++) {
    nvshmem_int_put_nbi(data_d, data_d, len, peer);
    nvshmem_quiet();
  }
}

__global__ void latency_kern_warp(int* data_d, int len, int pe, int iter) {
  int peer = !pe;
  for (int i = 0; i < iter; i++) {
    nvshmemx_int_put_nbi_warp(data_d, data_d, len, peer);
    __syncthreads();
    if (!threadIdx.x) nvshmem_quiet();
    __syncthreads();
  }
}

__global__ void latency_kern_block(int* data_d, int len, int pe, int iter) {
  int peer = !pe;
  for (int i = 0; i < iter; i++) {
    nvshmemx_int_put_nbi_block(data_d, data_d, len, peer);
    __syncthreads();
    if (!threadIdx.x) nvshmem_quiet();
    __syncthreads();
  }
}

// --- Host ---

int main(int argc, char* argv[]) {
  read_args(argc, argv);
  int iter = iters;
  int skip = warmup_iters;

  init_wrapper(&argc, &argv);

  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();

  if (npes != 2) {
    if (!mype) fprintf(stderr, "This test requires exactly 2 PEs\n");
    finalize_wrapper();
    return 1;
  }

  int* data_d = (int*)nvshmem_malloc(max_size);
  CUDA_CHECK(cudaMemset(data_d, 0, max_size));

  void** h_tables;
  alloc_tables(&h_tables, 2, max_size_log + 1);
  uint64_t* h_size = (uint64_t*)h_tables[0];
  double* h_lat = (double*)h_tables[1];

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  nvshmem_barrier_all();

  // --- Thread scope ---
  int idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    int nelems = size / sizeof(int);
    h_size[idx] = size;

    if (!mype) {
      latency_kern<<<1, 1>>>(data_d, nelems, mype, skip);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      latency_kern<<<1, 1>>>(data_d, nelems, mype, iter);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      h_lat[idx] = (ms * 1000.0) / iter;  // us
    }
    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_basic("shmem_put_latency", "Thread", "size (Bytes)", "latency", "us", '-', h_size, h_lat, idx);

  // --- Warp scope ---
  idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    int nelems = size / sizeof(int);
    h_size[idx] = size;

    if (!mype) {
      latency_kern_warp<<<1, 32>>>(data_d, nelems, mype, skip);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      latency_kern_warp<<<1, 32>>>(data_d, nelems, mype, iter);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      h_lat[idx] = (ms * 1000.0) / iter;
    }
    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_basic("shmem_put_latency", "Warp", "size (Bytes)", "latency", "us", '-', h_size, h_lat, idx);

  // --- Block scope ---
  idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    int nelems = size / sizeof(int);
    h_size[idx] = size;

    if (!mype) {
      latency_kern_block<<<1, threads_per_block>>>(data_d, nelems, mype, skip);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      latency_kern_block<<<1, threads_per_block>>>(data_d, nelems, mype, iter);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      h_lat[idx] = (ms * 1000.0) / iter;
    }
    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_basic("shmem_put_latency", "Block", "size (Bytes)", "latency", "us", '-', h_size, h_lat, idx);

  nvshmem_free(data_d);
  free_tables(h_tables, 2);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  finalize_wrapper();
  return 0;
}
