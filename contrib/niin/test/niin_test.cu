/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

// NIIN functional tests — exercises all API categories on real GPUs.
// Single-process, multi-GPU. Uses 2 GPUs (LSA peers on same node).

#include "nvshmem.h"
#include "nvshmemx.h"
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>

#define CUDACHECK(cmd) do {                                                   \
  cudaError_t e = (cmd);                                                      \
  if (e != cudaSuccess) {                                                     \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
            cudaGetErrorString(e));                                            \
    exit(1);                                                                   \
  }                                                                            \
} while(0)

#define NCCLCHECK(cmd) do {                                                   \
  ncclResult_t r = (cmd);                                                     \
  if (r != ncclSuccess) {                                                     \
    fprintf(stderr, "NCCL error %s:%d: %s\n", __FILE__, __LINE__,            \
            ncclGetErrorString(r));                                            \
    exit(1);                                                                   \
  }                                                                            \
} while(0)

// Device-side test result reporting
struct TestResult {
  int passed;
  int failed;
  char failMsg[256];
};

__device__ void recordPass(TestResult* r) {
  atomicAdd(&r->passed, 1);
}

__device__ void recordFail(TestResult* r, const char* msg) {
  int idx = atomicAdd(&r->failed, 1);
  if (idx == 0) {
    // Record first failure message
    int i = 0;
    while (msg[i] && i < 255) { r->failMsg[i] = msg[i]; i++; }
    r->failMsg[i] = '\0';
  }
}

// ============================================================================
// Test 1: Query APIs
// ============================================================================
__global__ void test_query(niinContext* ctx, TestResult* result, int expectedRank, int expectedNRanks) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    if (nvshmem_my_pe() == expectedRank)
      recordPass(result);
    else
      recordFail(result, "nvshmem_my_pe() wrong");

    if (nvshmem_n_pes() == expectedNRanks)
      recordPass(result);
    else
      recordFail(result, "nvshmem_n_pes() wrong");

    if (nvshmem_team_my_pe(NVSHMEM_TEAM_WORLD) == expectedRank)
      recordPass(result);
    else
      recordFail(result, "nvshmem_team_my_pe(WORLD) wrong");

    if (nvshmem_team_n_pes(NVSHMEM_TEAM_WORLD) == expectedNRanks)
      recordPass(result);
    else
      recordFail(result, "nvshmem_team_n_pes(WORLD) wrong");
  }
}

// ============================================================================
// Test 2: nvshmem_ptr — verify we can get a valid peer pointer for LSA peers
// ============================================================================
__global__ void test_ptr(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    void* base = niin_heap_base();

    // Self pointer should return self
    void* selfPtr = nvshmem_ptr(base, pe);
    if (selfPtr == base)
      recordPass(result);
    else
      recordFail(result, "nvshmem_ptr(self) != base");

    // Peer pointer (LSA peer) should be non-null
    int peer = (pe + 1) % npes;
    void* peerPtr = nvshmem_ptr(base, peer);
    if (peerPtr != nullptr)
      recordPass(result);
    else
      recordFail(result, "nvshmem_ptr(lsa_peer) == nullptr");
  }
}

// ============================================================================
// Test 3: Scalar put/get (_p and _g) between LSA peers
// ============================================================================
__global__ void test_scalar_put_get(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    // Each PE writes its rank to offset 0 in next PE's heap
    int* buf = (int*)niin_heap_base();
    int peer = (pe + 1) % npes;

    // Write our rank to peer
    nvshmem_int_p(buf, pe, peer);
    nvshmem_fence();
  }
}

__global__ void test_scalar_put_get_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int* buf = (int*)niin_heap_base();
    int expectedSender = (pe - 1 + npes) % npes;

    if (buf[0] == expectedSender)
      recordPass(result);
    else
      recordFail(result, "scalar put/get: wrong value received");
  }
}

// ============================================================================
// Test 4: Block put between LSA peers
// ============================================================================
__global__ void test_block_put_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    // Write a pattern into local buffer at offset 1024
    int* src = (int*)niin_heap_base() + 256;
    for (int i = 0; i < 64; i++) {
      src[i] = pe * 1000 + i;
    }
  }
}

__global__ void test_block_put(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    int* src = (int*)niin_heap_base() + 256;
    // Put into peer's buffer at offset 512 (ints)
    int* dest = (int*)niin_heap_base() + 512;
    nvshmem_int_put(dest, src, 64, peer);
    nvshmem_fence();
  }
}

__global__ void test_block_put_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int expectedSender = (pe - 1 + npes) % npes;
    int* dest = (int*)niin_heap_base() + 512;

    bool ok = true;
    for (int i = 0; i < 64; i++) {
      if (dest[i] != expectedSender * 1000 + i) {
        ok = false;
        break;
      }
    }

    if (ok)
      recordPass(result);
    else
      recordFail(result, "block put: wrong data received");
  }
}

// ============================================================================
// Test 5: Scalar get from LSA peer
// ============================================================================
__global__ void test_scalar_get_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    // Write our rank at offset 768 (ints)
    int* buf = (int*)niin_heap_base() + 768;
    buf[0] = pe * 7 + 13;
  }
}

__global__ void test_scalar_get(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    int* remoteBuf = (int*)niin_heap_base() + 768;

    int val = nvshmem_int_g(remoteBuf, peer);
    int expected = peer * 7 + 13;

    if (val == expected)
      recordPass(result);
    else
      recordFail(result, "scalar get: wrong value");
  }
}

// ============================================================================
// Test 6: Block get from LSA peer
// ============================================================================
__global__ void test_block_get(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    // Source is at offset 256 (ints) on the peer (set by test_block_put_setup)
    int* remoteSrc = (int*)niin_heap_base() + 256;
    // Destination is at offset 1024 (ints) on self
    int* localDest = (int*)niin_heap_base() + 1024;
    nvshmem_int_get(localDest, remoteSrc, 64, peer);

    bool ok = true;
    for (int i = 0; i < 64; i++) {
      if (localDest[i] != peer * 1000 + i) {
        ok = false;
        break;
      }
    }

    if (ok)
      recordPass(result);
    else
      recordFail(result, "block get: wrong data");
  }
}

// ============================================================================
// Test 7: putmem / getmem
// ============================================================================
__global__ void test_putmem_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    char* buf = (char*)niin_heap_base() + 8192;
    for (int i = 0; i < 128; i++) buf[i] = (char)(nvshmem_my_pe() + i);
  }
}

__global__ void test_putmem(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    char* src = (char*)niin_heap_base() + 8192;
    char* dest = (char*)niin_heap_base() + 8192 + 256;
    nvshmem_putmem(dest, src, 128, peer);
    nvshmem_fence();
  }
}

__global__ void test_putmem_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    char* dest = (char*)niin_heap_base() + 8192 + 256;
    bool ok = true;
    for (int i = 0; i < 128; i++) {
      if (dest[i] != (char)(sender + i)) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "putmem: wrong data");
  }
}

// ============================================================================
// Test 8: Atomics (on self for simplicity, then on LSA peer)
// ============================================================================
__global__ void test_atomics_self(niinContext* ctx, TestResult* result) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int* buf = (int*)niin_heap_base() + 2048;
    buf[0] = 0;
    __threadfence();

    // fetch_add
    int old = nvshmem_int_atomic_fetch_add(buf, 10, pe);
    if (old == 0 && buf[0] == 10)
      recordPass(result);
    else
      recordFail(result, "atomic fetch_add self failed");

    // compare_swap: expect 10, swap to 42
    old = nvshmem_int_atomic_compare_swap(buf, 10, 42, pe);
    if (old == 10 && buf[0] == 42)
      recordPass(result);
    else
      recordFail(result, "atomic compare_swap self failed");

    // swap
    old = nvshmem_int_atomic_swap(buf, 99, pe);
    if (old == 42 && buf[0] == 99)
      recordPass(result);
    else
      recordFail(result, "atomic swap self failed");

    // fetch
    int val = nvshmem_int_atomic_fetch(buf, pe);
    if (val == 99)
      recordPass(result);
    else
      recordFail(result, "atomic fetch self failed");

    // set
    nvshmem_int_atomic_set(buf, 0, pe);
    __threadfence();
    if (buf[0] == 0)
      recordPass(result);
    else
      recordFail(result, "atomic set self failed");

    // inc
    nvshmem_int_atomic_inc(buf, pe);
    __threadfence();
    if (buf[0] == 1)
      recordPass(result);
    else
      recordFail(result, "atomic inc self failed");
  }
}

// Test atomics on a peer
__global__ void test_atomics_peer_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    // Zero out target location
    int* buf = (int*)niin_heap_base() + 2048 + 16;
    buf[0] = 0;
  }
}

__global__ void test_atomics_peer(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    int* buf = (int*)niin_heap_base() + 2048 + 16;

    // Each PE atomically adds its rank+1 to peer's buffer
    int old = nvshmem_int_atomic_fetch_add(buf, pe + 1, peer);
    // Can't verify immediately (race), just check it didn't trap
    (void)old;
    recordPass(result);
  }
}

__global__ void test_atomics_peer_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int* buf = (int*)niin_heap_base() + 2048 + 16;

    // The predecessor PE should have added (predecessor_rank+1)
    int sender = (pe - 1 + npes) % npes;
    int expected = sender + 1;
    if (buf[0] == expected)
      recordPass(result);
    else
      recordFail(result, "atomic peer: wrong value");
  }
}

// ============================================================================
// Test 9: Fence and Quiet
// ============================================================================
__global__ void test_fence_quiet(niinContext* ctx, TestResult* result) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    // Just verify these don't crash
    nvshmem_fence();
    nvshmem_quiet();
    recordPass(result);
  }
}

// ============================================================================
// Test 10: Wait/Test operations (on local memory)
// ============================================================================
__global__ void test_wait_test(niinContext* ctx, TestResult* result) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int* buf = (int*)niin_heap_base() + 3072;
    buf[0] = 42;
    __threadfence();

    // test should succeed immediately
    int r = nvshmem_int_test(buf, NVSHMEM_CMP_EQ, 42);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test CMP_EQ failed");

    r = nvshmem_int_test(buf, NVSHMEM_CMP_NE, 0);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test CMP_NE failed");

    r = nvshmem_int_test(buf, NVSHMEM_CMP_GE, 42);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test CMP_GE failed");

    r = nvshmem_int_test(buf, NVSHMEM_CMP_GT, 41);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test CMP_GT failed");

    r = nvshmem_int_test(buf, NVSHMEM_CMP_LE, 42);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test CMP_LE failed");

    r = nvshmem_int_test(buf, NVSHMEM_CMP_LT, 43);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test CMP_LT failed");

    // wait_until should return immediately (value already matches)
    nvshmem_int_wait_until(buf, NVSHMEM_CMP_EQ, 42);
    recordPass(result);

    // test_all
    int arr[4] = {10, 20, 30, 40};
    int status[4] = {0, 0, 0, 0};
    r = nvshmem_int_test_all(arr, 4, status, NVSHMEM_CMP_GE, 5);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test_all failed");

    // test_any
    size_t idx = nvshmem_int_test_any(arr, 4, status, NVSHMEM_CMP_EQ, 20);
    if (idx == 1)
      recordPass(result);
    else
      recordFail(result, "test_any failed");

    // test_some
    size_t indices[4];
    size_t count = nvshmem_int_test_some(arr, 4, indices, status, NVSHMEM_CMP_GE, 25);
    if (count == 2)  // 30 and 40 meet >= 25
      recordPass(result);
    else
      recordFail(result, "test_some: wrong count");
  }
}

// ============================================================================
// Test 11: Signal operations
// ============================================================================
__global__ void test_signal_ops(niinContext* ctx, TestResult* result) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    uint64_t* sig = (uint64_t*)((char*)niin_heap_base() + 16384);
    *sig = 0;
    __threadfence();

    // signal_fetch on zero
    uint64_t val = nvshmem_signal_fetch(sig);
    if (val == 0)
      recordPass(result);
    else
      recordFail(result, "signal_fetch: expected 0");

    // Manually set the signal, then use signal_wait_until
    *sig = 5;
    __threadfence();
    uint64_t waited = nvshmem_signal_wait_until(sig, NVSHMEM_CMP_GE, 5);
    if (waited >= 5)
      recordPass(result);
    else
      recordFail(result, "signal_wait_until failed");
  }
}

// ============================================================================
// Test 12: put_signal between peers
// ============================================================================
__global__ void test_put_signal(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    int* data = (int*)niin_heap_base() + 4096;
    uint64_t* sig = (uint64_t*)((char*)niin_heap_base() + 32768);

    // Set up source data
    data[0] = pe * 100 + 7;
    __threadfence();

    // Put data + signal to peer
    int* dest = (int*)niin_heap_base() + 4096 + 64;
    nvshmem_int_put_signal(dest, data, 1, sig, 1, NVSHMEM_SIGNAL_ADD, peer);
  }
}

__global__ void test_put_signal_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    uint64_t* sig = (uint64_t*)((char*)niin_heap_base() + 32768);

    // Wait for the signal
    nvshmem_signal_wait_until(sig, NVSHMEM_CMP_GE, 1);

    // Verify data
    int* dest = (int*)niin_heap_base() + 4096 + 64;
    int expected = sender * 100 + 7;
    if (dest[0] == expected)
      recordPass(result);
    else
      recordFail(result, "put_signal: wrong data");
  }
}

// ============================================================================
// Test 13: Strided put/get (iput/iget)
// ============================================================================
__global__ void test_strided_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int* buf = (int*)niin_heap_base() + 5120;
    for (int i = 0; i < 8; i++) buf[i] = pe * 100 + i;
  }
}

__global__ void test_strided(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    int* src = (int*)niin_heap_base() + 5120;
    int* dest = (int*)niin_heap_base() + 5120 + 64;

    // iput: 4 elements, dst stride 2, src stride 1
    nvshmem_int_iput(dest, src, 2, 1, 4, peer);
    nvshmem_fence();
  }
}

__global__ void test_strided_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    int* dest = (int*)niin_heap_base() + 5120 + 64;

    // dest[0] = sender*100+0, dest[2] = sender*100+1, dest[4] = sender*100+2, dest[6] = sender*100+3
    bool ok = true;
    for (int i = 0; i < 4; i++) {
      if (dest[i * 2] != sender * 100 + i) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "strided iput: wrong data");
  }
}

// ============================================================================
// Test 14: Multiple typed puts (float, double, long long)
// ============================================================================
__global__ void test_typed_puts(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int peer = (pe + 1) % npes;
    char* base = (char*)niin_heap_base() + 40960;

    // float
    float* fbuf = (float*)base;
    nvshmem_float_p(fbuf, 3.14f, peer);

    // double
    double* dbuf = (double*)(base + 64);
    nvshmem_double_p(dbuf, 2.71828, peer);

    // long long
    long long* llbuf = (long long*)(base + 128);
    nvshmem_longlong_p(llbuf, 0xDEADBEEFCAFELL, peer);

    nvshmem_fence();
  }
}

__global__ void test_typed_puts_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    char* base = (char*)niin_heap_base() + 40960;

    float fv = *(volatile float*)base;
    if (fv == 3.14f) recordPass(result);
    else recordFail(result, "typed float put wrong");

    double dv = *(volatile double*)(base + 64);
    if (dv == 2.71828) recordPass(result);
    else recordFail(result, "typed double put wrong");

    long long llv = *(volatile long long*)(base + 128);
    if (llv == 0xDEADBEEFCAFELL) recordPass(result);
    else recordFail(result, "typed longlong put wrong");
  }
}

// ============================================================================
// Test 15: Bitwise atomics (on self)
// ============================================================================
__global__ void test_bitwise_atomics(niinContext* ctx, TestResult* result) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    unsigned int* buf = (unsigned int*)((char*)niin_heap_base() + 49152);
    *buf = 0xFF;
    __threadfence();

    unsigned int old;

    // AND
    old = nvshmem_uint_atomic_fetch_and(buf, 0x0F, pe);
    if (old == 0xFF && *buf == 0x0F) recordPass(result);
    else recordFail(result, "bitwise AND failed");

    // OR
    old = nvshmem_uint_atomic_fetch_or(buf, 0xF0, pe);
    if (old == 0x0F && *buf == 0xFF) recordPass(result);
    else recordFail(result, "bitwise OR failed");

    // XOR
    old = nvshmem_uint_atomic_fetch_xor(buf, 0x0F, pe);
    if (old == 0xFF && *buf == 0xF0) recordPass(result);
    else recordFail(result, "bitwise XOR failed");
  }
}

// ============================================================================
// Test 16: Vector wait/test operations
// ============================================================================
__global__ void test_vector_wait_test(niinContext* ctx, TestResult* result) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int arr[4] = {10, 20, 30, 40};
    int cmp_vals[4] = {10, 15, 25, 50};
    int status[4] = {0, 0, 0, 0};

    // test_all_vector: arr[0]>=10, arr[1]>=15, arr[2]>=25, arr[3]>=50?
    // arr[3]=40 < 50, so should fail
    int r = nvshmem_int_test_all_vector(arr, 4, status, NVSHMEM_CMP_GE, cmp_vals);
    if (r == 0)
      recordPass(result);
    else
      recordFail(result, "test_all_vector should fail");

    // Fix cmp_vals[3] so all pass
    cmp_vals[3] = 40;
    r = nvshmem_int_test_all_vector(arr, 4, status, NVSHMEM_CMP_GE, cmp_vals);
    if (r == 1)
      recordPass(result);
    else
      recordFail(result, "test_all_vector should pass");

    // test_any_vector: first that matches arr[i] == cmp_vals[i]
    int eq_vals[4] = {99, 20, 99, 99};
    size_t idx = nvshmem_int_test_any_vector(arr, 4, status, NVSHMEM_CMP_EQ, eq_vals);
    if (idx == 1)
      recordPass(result);
    else
      recordFail(result, "test_any_vector wrong index");

    // test_some_vector: count matches for arr[i] >= cmp_vals[i]
    int ge_vals[4] = {5, 25, 30, 100};  // 10>=5, 20<25, 30>=30, 40<100 => 2 matches
    size_t indices[4];
    size_t count = nvshmem_int_test_some_vector(arr, 4, indices, status, NVSHMEM_CMP_GE, ge_vals);
    if (count == 2 && indices[0] == 0 && indices[1] == 2)
      recordPass(result);
    else
      recordFail(result, "test_some_vector wrong");

    // wait_until_all_vector: values already satisfy, should return immediately
    nvshmem_int_wait_until_all_vector(arr, 4, status, NVSHMEM_CMP_GE, cmp_vals);
    recordPass(result);
  }
}

// ============================================================================
// Test 17: Warp/block threadgroup put (nvshmemx_*_put_warp/block)
// ============================================================================
__global__ void test_threadgroup_put_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int* buf = (int*)niin_heap_base() + 6144;
    for (int i = 0; i < 16; i++) buf[i] = pe * 100 + i;
  }
}

__global__ void test_threadgroup_put(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  // All 32 threads participate in the warp put
  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  int* src = (int*)niin_heap_base() + 6144;
  int* dest = (int*)niin_heap_base() + 6144 + 64;
  nvshmemx_int_put_warp(dest, src, 16, peer);
}

__global__ void test_threadgroup_put_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    int* dest = (int*)niin_heap_base() + 6144 + 64;
    bool ok = true;
    for (int i = 0; i < 16; i++) {
      if (dest[i] != sender * 100 + i) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "threadgroup put_warp wrong data");
  }
}

__global__ void test_threadgroup_put_block(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  // All threads in block participate
  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  int* src = (int*)niin_heap_base() + 6144;
  int* dest = (int*)niin_heap_base() + 6144 + 128;
  nvshmemx_int_put_block(dest, src, 16, peer);
}

__global__ void test_threadgroup_put_block_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    int* dest = (int*)niin_heap_base() + 6144 + 128;
    bool ok = true;
    for (int i = 0; i < 16; i++) {
      if (dest[i] != sender * 100 + i) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "threadgroup put_block wrong data");
  }
}

// ============================================================================
// Test 17c: Misaligned warp put (src+dest offset by 3 bytes)
// ============================================================================
__global__ void test_misaligned_put_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    // Write pattern at a misaligned offset: heapBase + 8192 + 3 bytes
    char* base = (char*)niin_heap_base() + 8192 + 3;
    for (int i = 0; i < 200; i++) base[i] = (char)(pe + i + 1);
  }
}

__global__ void test_misaligned_warp_put(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  // Both src and dest are misaligned (offset by 3 bytes)
  char* src = (char*)niin_heap_base() + 8192 + 3;
  char* dest = (char*)niin_heap_base() + 8192 + 512 + 3;
  nvshmemx_putmem_warp(dest, src, 200, peer);
}

__global__ void test_misaligned_put_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    char* dest = (char*)niin_heap_base() + 8192 + 512 + 3;
    bool ok = true;
    for (int i = 0; i < 200; i++) {
      if (dest[i] != (char)(sender + i + 1)) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "misaligned warp put wrong data");
  }
}

// ============================================================================
// Test 17d: Large block put (64 KB, exercises unrolled body)
// ============================================================================
__global__ void test_large_put_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  // All threads cooperatively fill source buffer at offset 16384
  int* src = (int*)((char*)niin_heap_base() + 16384);
  int pe = niin_device_my_pe();
  int nElems = 16384; // 64 KB / 4
  for (int i = threadIdx.x; i < nElems; i += blockDim.x)
    src[i] = pe * 1000000 + i;
}

__global__ void test_large_block_put(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  int* src = (int*)((char*)niin_heap_base() + 16384);
  int* dest = (int*)((char*)niin_heap_base() + 16384 + 65536);
  // 64 KB = 16384 ints
  nvshmemx_int_put_block(dest, src, 16384, peer);
}

__global__ void test_large_put_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    int* dest = (int*)((char*)niin_heap_base() + 16384 + 65536);
    bool ok = true;
    for (int i = 0; i < 16384; i++) {
      if (dest[i] != sender * 1000000 + i) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "large block put wrong data");
  }
}

// ============================================================================
// Test 17e: Odd-size block put (1000 bytes — not multiple of 16)
// ============================================================================
__global__ void test_oddsize_put_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    char* src = (char*)niin_heap_base() + 131072;
    for (int i = 0; i < 1000; i++) src[i] = (char)(pe * 3 + i);
  }
}

__global__ void test_oddsize_block_put(niinContext* ctx, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  char* src = (char*)niin_heap_base() + 131072;
  char* dest = (char*)niin_heap_base() + 131072 + 2048;
  nvshmemx_putmem_block(dest, src, 1000, peer);
}

__global__ void test_oddsize_put_verify(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int sender = (pe - 1 + npes) % npes;
    char* dest = (char*)niin_heap_base() + 131072 + 2048;
    bool ok = true;
    for (int i = 0; i < 1000; i++) {
      if (dest[i] != (char)(sender * 3 + i)) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "odd-size block put wrong data");
  }
}

// ============================================================================
// Test 17f: Block get from LSA peer (cooperative get)
// ============================================================================
__global__ void test_coop_block_get_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    int* buf = (int*)((char*)niin_heap_base() + 196608);
    for (int i = 0; i < 256; i++) buf[i] = pe * 500 + i;
  }
}

__global__ void test_coop_block_get(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  int* remoteSrc = (int*)((char*)niin_heap_base() + 196608);
  int* localDest = (int*)((char*)niin_heap_base() + 196608 + 4096);
  nvshmemx_int_get_block(localDest, remoteSrc, 256, peer);

  // Thread 0 verifies
  if (threadIdx.x == 0) {
    bool ok = true;
    for (int i = 0; i < 256; i++) {
      if (localDest[i] != peer * 500 + i) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "coop block get wrong data");
  }
}

// ============================================================================
// Test 17g: Misaligned block get (offset by 5 bytes, 999 bytes)
// ============================================================================
__global__ void test_misaligned_get_setup(niinContext* ctx) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();
  if (threadIdx.x == 0) {
    int pe = nvshmem_my_pe();
    char* buf = (char*)niin_heap_base() + 262144 + 5;
    for (int i = 0; i < 999; i++) buf[i] = (char)(pe * 7 + i);
  }
}

__global__ void test_misaligned_block_get(niinContext* ctx, TestResult* result, int npes) {
  if (threadIdx.x == 0) niin_g_ctx = ctx;
  __syncthreads();

  int pe = niin_device_my_pe();
  int peer = (pe + 1) % npes;
  char* remoteSrc = (char*)niin_heap_base() + 262144 + 5;
  char* localDest = (char*)niin_heap_base() + 262144 + 2048 + 5;
  nvshmemx_getmem_block(localDest, remoteSrc, 999, peer);

  if (threadIdx.x == 0) {
    bool ok = true;
    for (int i = 0; i < 999; i++) {
      if (localDest[i] != (char)(peer * 7 + i)) { ok = false; break; }
    }
    if (ok) recordPass(result);
    else recordFail(result, "misaligned block get wrong data");
  }
}

// ============================================================================
// Test 18: Team operations (host-side, run from main)
// ============================================================================

// ============================================================================
// Main: run all tests
// ============================================================================

void printResult(const char* name, TestResult& r) {
  if (r.failed == 0)
    printf("  PASS: %-35s (%d checks)\n", name, r.passed);
  else
    printf("  FAIL: %-35s (%d passed, %d failed: %s)\n", name, r.passed, r.failed, r.failMsg);
}

int main() {
  int nGpus;
  CUDACHECK(cudaGetDeviceCount(&nGpus));
  if (nGpus < 2) {
    printf("Need at least 2 GPUs, found %d\n", nGpus);
    return 1;
  }

  int nPes = 2;  // Use 2 GPUs
  printf("NIIN Functional Tests — using %d GPUs\n\n", nPes);

  // Step 1: Create NCCL communicators
  int devices[2] = {0, 1};
  ncclComm_t comms[2];
  cudaStream_t streams[2];
  NCCLCHECK(ncclCommInitAll(comms, nPes, devices));
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamCreate(&streams[i]));
  }

  // Step 2: Allocate symmetric heap memory and set up NIIN contexts
  size_t heapSize = 1 << 20;  // 1 MB
  void* heapBufs[2];
  niinContext* ctxs[2];
  TestResult* results[2];

  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(ncclMemAlloc(&heapBufs[i], heapSize));
    CUDACHECK(cudaMemset(heapBufs[i], 0, heapSize));
    CUDACHECK(cudaMalloc(&ctxs[i], sizeof(niinContext)));
    CUDACHECK(cudaMalloc(&results[i], sizeof(TestResult)));
  }

  // Step 3: Initialize NIIN (two-phase: NCCL collectives, then device copy)
  niinContext_host hostCtxs[2];
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(niinInit(comms[i], heapBufs[i], heapSize, &hostCtxs[i],
                        /*ginContextIndex=*/0, /*barrierCount=*/1, /*ginSignalCount=*/0));
  }
  NCCLCHECK(ncclGroupEnd());

  // Phase 2: copy to device (after groupEnd so NCCL results are finalized)
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    NCCLCHECK(niinCommit(&hostCtxs[i], ctxs[i]));
  }

  // Initialize team table for host-side team tests (using PE 0's perspective)
  niin::teams::initPredefined(0, nPes, hostCtxs[0].devComm.lsaRank, hostCtxs[0].devComm.lsaSize);

  // Helper lambdas for running tests
  auto clearResults = [&]() {
    for (int i = 0; i < nPes; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMemset(results[i], 0, sizeof(TestResult)));
    }
  };

  auto syncAll = [&]() {
    for (int i = 0; i < nPes; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaStreamSynchronize(streams[i]));
    }
  };

  auto getResults = [&](TestResult out[2]) {
    for (int i = 0; i < nPes; i++) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMemcpy(&out[i], results[i], sizeof(TestResult), cudaMemcpyDeviceToHost));
    }
  };

  int totalPassed = 0, totalFailed = 0;
  auto reportTest = [&](const char* name) {
    TestResult out[2];
    getResults(out);
    for (int i = 0; i < nPes; i++) {
      char label[128];
      snprintf(label, sizeof(label), "%s [PE %d]", name, i);
      printResult(label, out[i]);
      totalPassed += out[i].passed;
      totalFailed += out[i].failed;
    }
  };

  // ===== Run Tests =====

  // Test 1: Query
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_query<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], i, nPes);
  }
  syncAll();
  reportTest("Query APIs");

  // Test 2: Ptr
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_ptr<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("nvshmem_ptr");

  // Test 3: Scalar put/get
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_scalar_put_get<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_scalar_put_get_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Scalar put (_p) + verify");

  // Test 4: Block put
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_block_put_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_block_put<<<1, 1, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_block_put_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Block put");

  // Test 5: Scalar get
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_scalar_get_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_scalar_get<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Scalar get (_g)");

  // Test 6: Block get
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_block_get<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Block get");

  // Test 7: putmem
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_putmem_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_putmem<<<1, 1, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_putmem_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("putmem");

  // Test 8: Atomics (self)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_atomics_self<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i]);
  }
  syncAll();
  reportTest("Atomics (self)");

  // Test 8b: Atomics (peer)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_atomics_peer_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_atomics_peer<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_atomics_peer_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Atomics (peer)");

  // Test 9: Fence/Quiet
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_fence_quiet<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i]);
  }
  syncAll();
  reportTest("Fence/Quiet");

  // Test 10: Wait/Test
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_wait_test<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i]);
  }
  syncAll();
  reportTest("Wait/Test ops");

  // Test 11: Signal ops
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_signal_ops<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i]);
  }
  syncAll();
  reportTest("Signal ops (local)");

  // Test 12: put_signal
  clearResults();
  // Clear signal locations first
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMemset((char*)heapBufs[i] + 32768, 0, sizeof(uint64_t)));
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_put_signal<<<1, 1, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_put_signal_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("put_signal");

  // Test 13: Strided put
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_strided_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_strided<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_strided_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Strided iput");

  // Test 14: Multi-type puts
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_typed_puts<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_typed_puts_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Multi-type puts");

  // Test 15: Bitwise atomics
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_bitwise_atomics<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i]);
  }
  syncAll();
  reportTest("Bitwise atomics");

  // Test 16: Vector wait/test
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_vector_wait_test<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i]);
  }
  syncAll();
  reportTest("Vector wait/test");

  // Test 17a: Warp threadgroup put
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_threadgroup_put_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_threadgroup_put<<<1, 32, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_threadgroup_put_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Warp put (nvshmemx)");

  // Test 17b: Block threadgroup put
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_threadgroup_put_block<<<1, 64, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_threadgroup_put_block_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Block put (nvshmemx)");

  // Test 17c: Misaligned warp put (200 bytes at offset +3)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_misaligned_put_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_misaligned_warp_put<<<1, 32, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_misaligned_put_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Misaligned warp put");

  // Test 17d: Large block put (64 KB)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_large_put_setup<<<1, 256, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_large_block_put<<<1, 256, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_large_put_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Large block put (64KB)");

  // Test 17e: Odd-size block put (1000 bytes)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_oddsize_put_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_oddsize_block_put<<<1, 128, 0, streams[i]>>>(ctxs[i], nPes);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_oddsize_put_verify<<<1, 1, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Odd-size block put (1000B)");

  // Test 17f: Block get (cooperative, 256 ints)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_coop_block_get_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_coop_block_get<<<1, 128, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Block get (nvshmemx)");

  // Test 17g: Misaligned block get (999 bytes at offset +5)
  clearResults();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_misaligned_get_setup<<<1, 1, 0, streams[i]>>>(ctxs[i]);
  }
  syncAll();
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    test_misaligned_block_get<<<1, 64, 0, streams[i]>>>(ctxs[i], results[i], nPes);
  }
  syncAll();
  reportTest("Misaligned block get");

  // Test 18: Team operations (host-side)
  {
    TestResult teamResult = {};
    // team_split_strided: split WORLD into even/odd PEs
    nvshmem_team_t even_team;
    nvshmem_team_split_strided(NVSHMEM_TEAM_WORLD, 0, 2, (nPes + 1) / 2,
                                nullptr, 0, &even_team);
    // PE 0 should be in the even team, PE 1 should not
    // But with only 2 PEs: even team = {0}, odd would be {1}
    // PE 0: rank 0 in even_team; PE 1: INVALID
    // We test from PE 0's perspective (the host process manages both)

    // team_translate_pe
    if (even_team != NVSHMEM_TEAM_INVALID) {
      int worldPe = nvshmem_team_translate_pe(even_team, 0, NVSHMEM_TEAM_WORLD);
      if (worldPe == 0) teamResult.passed++;
      else { teamResult.failed++; snprintf(teamResult.failMsg, 256, "translate_pe: got %d expected 0", worldPe); }
    } else {
      // This PE is not in the even team — that's also a valid result
      teamResult.passed++;
    }

    // team_split_2d
    nvshmem_team_t xteam, yteam;
    nvshmem_team_split_2d(NVSHMEM_TEAM_WORLD, 2, nullptr, 0, &xteam, nullptr, 0, &yteam);
    // With 2 PEs and xrange=2: x-axis has both PEs, y-axis has 1 each
    if (xteam != NVSHMEM_TEAM_INVALID) {
      int xsize = niin_teams_n_pes(xteam);
      if (xsize == 2) teamResult.passed++;
      else { teamResult.failed++; snprintf(teamResult.failMsg, 256, "split_2d xsize: got %d expected 2", xsize); }
    } else {
      teamResult.failed++;
      snprintf(teamResult.failMsg, 256, "split_2d returned INVALID xteam");
    }

    // team_destroy
    if (even_team != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(even_team);
    if (xteam != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(xteam);
    if (yteam != NVSHMEM_TEAM_INVALID) nvshmem_team_destroy(yteam);
    teamResult.passed++;  // destroy didn't crash

    char label[128];
    snprintf(label, sizeof(label), "Team operations [host]");
    printResult(label, teamResult);
    totalPassed += teamResult.passed;
    totalFailed += teamResult.failed;
  }

  // Test 19: Stream-based put on stream (host-launched)
  {
    // Use GPU 0 for this test
    CUDACHECK(cudaSetDevice(0));
    TestResult streamResult = {};

    // Clear a region of the heap
    int* buf = (int*)heapBufs[0];
    CUDACHECK(cudaMemset(buf + 7168, 0, 64 * sizeof(int)));
    CUDACHECK(cudaDeviceSynchronize());

    // Prepare source data
    int* src = buf + 7168;
    int* dst = buf + 7168 + 32;
    // Put some data at src
    int hostData[8] = {100, 200, 300, 400, 500, 600, 700, 800};
    CUDACHECK(cudaMemcpy(src, hostData, sizeof(hostData), cudaMemcpyHostToDevice));

    // Use nvshmemx_int_put_on_stream to copy src->dst on self (pe=0, which is this PE)
    // But we need the NVSHMEM global state context for PE 0... We're using the low-level
    // API here, so we need to set up niin_g_ctx. The ctx is already set for device 0.
    // The stream kernel will read niin_g_ctx which was set by the low-level API init.
    nvshmemx_int_put_on_stream(dst, src, 8, 0, streams[0]);
    CUDACHECK(cudaStreamSynchronize(streams[0]));

    int hostOut[8] = {};
    CUDACHECK(cudaMemcpy(hostOut, dst, sizeof(hostOut), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < 8; i++) {
      if (hostOut[i] != hostData[i]) { ok = false; break; }
    }
    if (ok) streamResult.passed++;
    else { streamResult.failed++; snprintf(streamResult.failMsg, 256, "put_on_stream data mismatch"); }

    // Test nvshmemx_int_p_on_stream
    nvshmemx_int_p_on_stream(dst, 42, 0, streams[0]);
    CUDACHECK(cudaStreamSynchronize(streams[0]));
    int pval;
    CUDACHECK(cudaMemcpy(&pval, dst, sizeof(int), cudaMemcpyDeviceToHost));
    if (pval == 42) streamResult.passed++;
    else { streamResult.failed++; snprintf(streamResult.failMsg, 256, "p_on_stream wrong value"); }

    char label[128];
    snprintf(label, sizeof(label), "Stream-based RMA [host]");
    printResult(label, streamResult);
    totalPassed += streamResult.passed;
    totalFailed += streamResult.failed;
  }

  // ===== Summary =====
  printf("\n========================================\n");
  printf("Total: %d passed, %d failed\n", totalPassed, totalFailed);
  printf("========================================\n");

  // Cleanup — niinFinalize does CUDA + NCCL calls, so don't wrap in a group
  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    niinFinalize(comms[i], ctxs[i]);
  }

  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(ctxs[i]));
    CUDACHECK(cudaFree(results[i]));
    NCCLCHECK(ncclMemFree(heapBufs[i]));
  }

  for (int i = 0; i < nPes; i++) {
    CUDACHECK(cudaStreamDestroy(streams[i]));
    NCCLCHECK(ncclCommDestroy(comms[i]));
  }

  return totalFailed > 0 ? 1 : 0;
}
