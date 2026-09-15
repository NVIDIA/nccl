/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// Put bandwidth: measures NVLink store bandwidth using cooperative block puts.
// Each block copies its chunk of data iter times with no inter-block sync.
// Only a single __threadfence_system at the end to ensure visibility.
//
// Both kernels lend their dynamic shared memory to NIIN, so running with
// NVSHMEM_TMA_POLICY=ENABLE on sm_90+ measures the TMA path instead of
// vectorized stores. Without that variable tma_smem_bytes() is 0, give_smem is
// inert, and the run is unchanged.

#include "perftest.h"

#define B_TO_GB (1000.0 * 1000.0 * 1000.0)
#define MS_TO_S 1000.0

// int4 element type pins the dynamic shared memory base to 16-byte alignment,
// which nvshmemx_give_smem() requires.
extern __shared__ int4 bw_smem[];

// --- Kernels ---

// Block-cooperative put: each block handles len/nblocks elements.
// No inter-block sync — all blocks blast data independently.
__global__ void bw_block(double* data_d, size_t len, int pe, int iter, size_t smem_bytes) {
  nvshmemx_give_smem(bw_smem, smem_bytes);
  __syncthreads();

  int peer = !pe;
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  size_t chunk = len / nblocks;
  double* my_src = data_d + bid * chunk;
  double* my_dst = data_d + bid * chunk;

  for (int i = 0; i < iter; i++) {
    nvshmemx_double_put_nbi_block(my_dst, my_src, chunk, peer);
    __syncthreads();
  }

  // Flush all outstanding operations (GIN + NVLink)
  __syncthreads();
  if (threadIdx.x == 0) nvshmem_quiet();
  __syncthreads();

  nvshmemx_release_smem();
  __syncthreads();
}

// Thread-scope put: one thread per CTA puts the CTA's chunk.
__global__ void bw_thread(double* data_d, size_t len, int pe, int iter, size_t smem_bytes) {
  nvshmemx_give_smem(bw_smem, smem_bytes);
  __syncthreads();

  int peer = !pe;
  int bid = blockIdx.x;
  int nblocks = gridDim.x;
  size_t per_block = len / nblocks;

  // One issuer per CTA: the thread-scoped TMA staging tile is per-CTA state.
  if (threadIdx.x == 0) {
    for (int i = 0; i < iter; i++) {
      nvshmem_double_put_nbi(data_d + bid * per_block,
                              data_d + bid * per_block,
                              per_block, peer);
    }
    nvshmem_quiet();
  }

  __syncthreads();
  nvshmemx_release_smem();
  __syncthreads();
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

  double* data_d = (double*)nvshmem_malloc(max_size);
  CUDA_CHECK(cudaMemset(data_d, 0, max_size));

  void** h_tables;
  alloc_tables(&h_tables, 2, max_size_log + 1);
  uint64_t* h_size = (uint64_t*)h_tables[0];
  double* h_bw = (double*)h_tables[1];

  size_t smem_bytes = tma_smem_bytes();

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  nvshmem_barrier_all();

  // --- Block scope BW ---
  int idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    size_t nelems = size / sizeof(double);
    if (nelems < (size_t)num_blocks) continue;

    h_size[idx] = size;

    if (!mype) {
      bw_block<<<num_blocks, threads_per_block, smem_bytes>>>(data_d, nelems, mype, skip, smem_bytes);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      bw_block<<<num_blocks, threads_per_block, smem_bytes>>>(data_d, nelems, mype, iter, smem_bytes);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      double seconds = ms / MS_TO_S;
      h_bw[idx] = ((double)size * iter) / seconds / B_TO_GB;
    }
    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_bw("shmem_put_bw", "Block", h_size, h_bw, idx);

  // --- Thread scope BW ---
  idx = 0;
  for (size_t size = min_size; size <= max_size; size *= step_factor) {
    size_t nelems = size / sizeof(double);
    if (nelems < (size_t)num_blocks) continue;

    h_size[idx] = size;

    if (!mype) {
      bw_thread<<<num_blocks, threads_per_block, smem_bytes>>>(data_d, nelems, mype, skip, smem_bytes);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaEventRecord(start));
      bw_thread<<<num_blocks, threads_per_block, smem_bytes>>>(data_d, nelems, mype, iter, smem_bytes);
      CUDA_CHECK(cudaEventRecord(stop));
      CUDA_CHECK(cudaEventSynchronize(stop));
      float ms;
      CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
      double seconds = ms / MS_TO_S;
      h_bw[idx] = ((double)size * iter) / seconds / B_TO_GB;
    }
    nvshmem_barrier_all();
    idx++;
  }
  if (!mype) print_table_bw("shmem_put_bw", "Thread", h_size, h_bw, idx);

  nvshmem_free(data_d);
  free_tables(h_tables, 2);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  finalize_wrapper();
  return 0;
}
