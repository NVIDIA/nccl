/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Put-signal ping-pong latency: PE 0 sends data+signal to PE 1,
// PE 1 waits for signal, sends back. Measures half round-trip.

#include "perftest.h"

__global__ void ping_pong(int* data_d, uint64_t* flag_d, int len, int pe, int iter) {
  int peer = !pe;

  for (int i = 0; i < iter; i++) {
    if (pe) {
      if (!threadIdx.x) nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
      __syncthreads();
      nvshmem_int_put_signal(data_d, data_d, len, flag_d, i + 1, NVSHMEM_SIGNAL_SET, peer);
    } else {
      nvshmem_int_put_signal(data_d, data_d, len, flag_d, i + 1, NVSHMEM_SIGNAL_SET, peer);
      __syncthreads();
      if (!threadIdx.x) nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
    }
    __syncthreads();
  }
  nvshmem_quiet();
}

__global__ void ping_pong_block(int* data_d, uint64_t* flag_d, int len, int pe, int iter) {
  int peer = !pe;

  for (int i = 0; i < iter; i++) {
    if (pe) {
      if (!threadIdx.x) nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
      __syncthreads();
      nvshmemx_int_put_signal_block(data_d, data_d, len, flag_d, i + 1, NVSHMEM_SIGNAL_SET, peer);
    } else {
      nvshmemx_int_put_signal_block(data_d, data_d, len, flag_d, i + 1, NVSHMEM_SIGNAL_SET, peer);
      __syncthreads();
      if (!threadIdx.x) nvshmem_uint64_wait_until(flag_d, NVSHMEM_CMP_EQ, (uint64_t)(i + 1));
    }
    __syncthreads();
  }
  nvshmem_quiet();
}

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
  uint64_t* flag_d = (uint64_t*)nvshmem_malloc(sizeof(uint64_t));
  CUDA_CHECK(cudaMemset(data_d, 0, max_size));
  CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));

  void** h_tables;
  alloc_tables(&h_tables, 2, max_size_log + 1);
  uint64_t* h_size = (uint64_t*)h_tables[0];
  double* h_lat = (double*)h_tables[1];

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  nvshmem_barrier_all();

  // Thread scope
  int idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    int nelems = size / sizeof(int);
    h_size[idx] = size;

    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    ping_pong<<<1, 1>>>(data_d, flag_d, nelems, mype, skip);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(start));
    ping_pong<<<1, 1>>>(data_d, flag_d, nelems, mype, iter);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    h_lat[idx] = (ms * 1000.0) / (2.0 * iter);  // half round-trip

    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_basic("shmem_put_signal_latency", "Thread", "size (Bytes)", "latency", "us", '-', h_size, h_lat, idx);

  // Block scope
  idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    int nelems = size / sizeof(int);
    h_size[idx] = size;

    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    ping_pong_block<<<1, threads_per_block>>>(data_d, flag_d, nelems, mype, skip);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(flag_d, 0, sizeof(uint64_t)));
    nvshmem_barrier_all();

    CUDA_CHECK(cudaEventRecord(start));
    ping_pong_block<<<1, threads_per_block>>>(data_d, flag_d, nelems, mype, iter);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    h_lat[idx] = (ms * 1000.0) / (2.0 * iter);

    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_basic("shmem_put_signal_latency", "Block", "size (Bytes)", "latency", "us", '-', h_size, h_lat, idx);

  nvshmem_free(data_d);
  nvshmem_free(flag_d);
  free_tables(h_tables, 2);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  finalize_wrapper();
  return 0;
}
