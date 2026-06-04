/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdio.h>

#include <nccl.h>
// Phase 4: pull in the device-API headers so __device__ kernels can call
// ncclGetLsaPointer / ncclGetLsaMultimemPointer / ncclGetLocalPointer.
// nccl_device.h also forward-declares the host-side getters we use in the
// launcher (e.g. ncclGetLsaMultimemDevicePointer).
#include <nccl_device.h>

#include <ubx/common.h>
#include <ubx/ubx.h>

#define CUDACHECK(cmd) UBX_CHECK_CUDA(cmd)

// Resolve the LSA multicast device pointer for `(window, off)` host-side.
// Returns nullptr when the window has no multicast handle (multicast-unavailable
// mode, no-MC configurations).
//
// Phase-4 perf note: kernels keep using ncclGetLsaPointer device-side for
// LSA *peer* pointers (per-peer, cheap, fits the kernel's per-thread
// access pattern). For LSA *multicast* pointers we host-resolve in the
// launcher and pass the resulting void* as a kernel arg — the multicast
// pointer is the same regardless of peer, so a single host call avoids
// the per-launch device-side ncclGetLsaMultimemPointer overhead that
// otherwise shows up at small messages (~0.15 µs on Lamport@1KB).
static inline void* nccl_lsa_mc_ptr(ncclWindow_t window, size_t off) {
  if (window == nullptr) return nullptr;
  void* p = nullptr;
  if (ncclGetLsaMultimemDevicePointer(window, off, &p) != ncclSuccess) {
    return nullptr;
  }
  return p;
}

#define UBX_MAXTHREADS 1024
#define UBX_MAX_SMS 128
#define UBX_LAMPORT_INT 0xFFFAFFFA

#ifdef UB_TIMEOUT_ENABLED
// Kernel-side polling-loop timeout, in GPU clocks. Default ~1s @ 2 GHz; the
// host-side `ubx_set_timeout()` overwrites it from UBX_TIMEOUT_SEC at
// SymmAllocator construction. Constant memory is broadcast-cached, so each
// kernel iteration costs essentially nothing.
__device__ __constant__ unsigned long long g_ubx_timeout = 2000000000ull;
#define UBX_TIMEOUT_CLOCKS g_ubx_timeout

extern "C" void ubx_set_timeout(unsigned long long clocks) {
  CUDACHECK(cudaMemcpyToSymbol(g_ubx_timeout, &clocks, sizeof(clocks)));
}
#else
// Build without UB_TIMEOUT_ENABLED: kernels have no polling timeout, but the
// setter still exists so the Python ABI is stable.  No-op.
extern "C" void ubx_set_timeout(unsigned long long clocks) { (void)clocks; }
#endif

// SM-count resolution for kernel launches:
//   sms = default_sms (forwarded from SymmAllocator, populated once from
//                      UBX_MAXSM at allocator construction)
//             ?: per-kernel compiled default below
//      → smlimit (per-call cap, downward only)
// The hot path never touches getenv — see SymmAllocator.__init__ in
// ubx/allocator.py.

//REG0 flags in use
#define UBX_FLAG_NVLS2_LAMPORT_ID 0
#define UBX_FLAG_NVLS2_LAMPORT_SM_SYNC 1
#define UBX_FLAG_NVLS2_LAMPORT_RS_BAR 2
#define UBX_FLAG_NVLS2_ID 3
#define UBX_FLAG_NVLS2_SM_SYNC 4
#define UBX_FLAG_NVLS2_RS_BAR 5
#define UBX_FLAG_NVLS2_AG_BAR 6
#define UBX_FLAG_NVLS1_ID 7
#define UBX_FLAG_NVLS1_SM_SYNC 8
#define UBX_FLAG_NVLS1_BAR 9
// Async a2av: separate counters so send+wait can overlap with other NVLS1 ops
#define UBX_FLAG_A2AV_ID 10
#define UBX_FLAG_A2AV_SM_SYNC 11
#define UBX_FLAG_A2AV_BAR 12
// Combine: separate flag set from a2av so dispatch and combine pipeline
// without flag-slot races (consecutive layers on one stream).
#define UBX_FLAG_COMBINE_ID 13
#define UBX_FLAG_COMBINE_SM_SYNC 14
#define UBX_FLAG_COMBINE_BAR 15
// Persistent (chunked) a2av: per-chunk SM_SYNC scratch slots.
// Launcher memsets these to 0 before each persistent launch.
#define UBX_FLAG_A2AV_PERSISTENT_SM_SYNC_BASE 16
#define UBX_MAX_PERSISTENT_CHUNKS 32
// Standalone barrier: separate flag set for ubx_barrier kernel, lets us
// surround dispatch/combine calls with a sync that uses DIFFERENT flag
// slots from the data-movement kernels — useful for isolating whether
// the combine/dispatch flag protocol has a race.
#define UBX_FLAG_BARRIER_ID 48
#define UBX_FLAG_BARRIER_SM_SYNC 49
#define UBX_FLAG_BARRIER_BAR 50
// E22 (combine v2 — true 2-kernel split): own BAR slot, host-tracked reduce_id.
// Phase 1 is pure data copy (no sync). Phase 2 does cross-rank barrier (one
// thread of CTA 0 issues all peer UCINCs, all CTAs poll), then summation.
#define UBX_FLAG_COMBINE2_BAR 51
// Minimal cross-rank peer atomic-inc test: each rank's thread i atomically
// increments peer i's flag. Then each rank spins polling own flag. No
// lastSM logic, no Phase-1 data, no PDL — just the raw peer-write pattern
// that combine kernel relies on.
#define UBX_FLAG_DIAG_PEER_TEST 52
// push3 (3-kernel PUSH combine): own BAR slot, host-tracked reduce_id.
// Phase 1: many-SM kernel pushes expert outputs to peer ranks' dest bufs.
// Phase 2: 1-SM 1-CTA signaler does N peer UCINCs and polls own flag (only
//          place that can hang; has compile-in timeout).
// Phase 3: many-SM kernel reads OWN dest buf, sums across topk, writes
//          output (purely local; no NVLink → cannot hang).
#define UBX_FLAG_PUSH3_BAR 53
// Vidmem-stored monotonic reduce_id for push3 phase 2. The kernel reads
// `*UBX_FLAG_PUSH3_ID + 1`, writes it back, and uses that for the BAR
// expected count — this matches the existing NVLS2/NVLS1 pattern at the
// top of this file and is the only way to keep flag and id state in lock-
// step across CUDA graph replays. A host-passed reduce_id would be baked
// into the captured kernel and stay frozen at the capture-time value
// while the BAR flag advanced on each replay → instant pass → race.
#define UBX_FLAG_PUSH3_ID  54

#define xhalf __nv_bfloat16

#define ATOMIC_MCINC(ptr)                                          \
  asm volatile("multimem.red.add.u32 [%0], %1;" ::"l"(ptr), "r"(1) \
               : "memor"                                           \
                 "y");
#define ATOMIC_UCINC(ptr)                                        \
  asm volatile("red.global.add.u32 [%0], %1;" ::"l"(ptr), "r"(1) \
               : "memor"                                         \
                 "y");
#define MULTIMEM_ST(val, ptr)                                                           \
  asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr), "r"(val.x), \
               "r"(val.y), "r"(val.z), "r"(val.w)                                       \
               : "memory");

#define MULTIMEM_LD(val, ptr)                                                 \
  asm("multimem.ld_reduce.global.add.v4.bf16x2.acc::f32 {%0,%1,%2,%3}, [%4];" \
      : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                    \
      : "l"(ptr)                                                              \
      : "memory");

// Return true if producer > consumer, otherwise false while preventing integer overflow
// If we expect that producer will be 2B+ messages behind consumer
#define CHECK_IDS(producer, consumer) (((unsigned)(producer) - (unsigned)(consumer)) & (~INT_MAX))

#define FINAL_MASK 0xffffffff
template <typename T, int NUM>
__inline__ __device__ T warpReduceSumV2(T *val) {
#pragma unroll
  for (int i = 0; i < NUM; i++) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
      val[i] += __shfl_xor_sync(FINAL_MASK, val[i], mask, 32);
  }
  return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceSumV2(T *val) {
  static __shared__ T shared[NUM][33];
  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  warpReduceSumV2<T, NUM>(val);

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < NUM; i++) {
      shared[i][wid] = val[i];
    }
  }

  __syncthreads();

  bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
  for (int i = 0; i < NUM; i++) {
    val[i] = is_mask ? shared[i][lane] : (T)(0.0f);
  }
  warpReduceSumV2<T, NUM>(val);
  return (T)0.0f;
}

template <int UNROLL>
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_allreduce_2shot_mc(const int RANKS, const int myrank, const int mylines,
                                        int* uc_flagptr, int* mc_flagptr,
                                        uint4* mc_ptr_in, uint4* mc_ptr_out,
                                        uint4 *residual_in, uint4 *residual_out,
                                        xhalf *gamma, float eps, const int hidden_size,
                                        bool fuse_layernorm) {
  // flags[3,4,5,6]: reduce_id, sm_sync-local, flag-barrier-1, flag-barrier-2.
  // uc_flagptr / mc_flagptr / mc_ptr_in / mc_ptr_out are pre-resolved
  // host-side by the launcher (mc-pointers via ncclGetLsaMultimemDevicePointer,
  // uc_flagptr via pool_ptr + REG0_FLAG_OFFSET). Avoiding device-side
  // ncclGetLsaMultimem* calls reclaims small-message latency.
  int reduce_id;
  __shared__ float s_variance;

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    if (blockIdx.x == 0) ATOMIC_MCINC(mc_flagptr + UBX_FLAG_NVLS2_RS_BAR);

    reduce_id = uc_flagptr[UBX_FLAG_NVLS2_ID] + 1;

    volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS2_RS_BAR]);

    const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
               *flag);
        break;
      }
#endif
    }
  }

  __syncthreads();

  const int loop_step0 = blockDim.x;
  const int loop_step = loop_step0 * UNROLL * gridDim.x;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL;
  const int end_elem = max(start_elem, mylines);

  for (int line = start_elem; line < end_elem; line += loop_step) {
    uint4 val[UNROLL];
    xhalf *x = reinterpret_cast<xhalf *>(&val[0]);
#pragma unroll
    for (int i = 0; i < UNROLL; i++) MULTIMEM_LD(val[i], mc_ptr_in + (line + i * loop_step0))

    if (residual_in != nullptr) {
      for (int i = 0; i < UNROLL; i++) {
        uint4 resval = residual_in[line + i * loop_step0];
        xhalf *y = reinterpret_cast<xhalf *>(&resval);
#pragma unroll
        for (int j = 0; j < 8; j++) x[i * 8 + j] += y[j];
        if (residual_out != nullptr) residual_out[line + i * loop_step0] = val[i];
      }
    }
    if (fuse_layernorm) {
      float local_var_sum = 0.0f;
      for (int j = 0; j < UNROLL * sizeof(int4) / sizeof(xhalf); j++)
        local_var_sum += (float)(x[j]) * (float)(x[j]);

      float packed[1] = {local_var_sum};
      blockReduceSumV2<float, 1>(packed);
      float variance = packed[0];

      if (threadIdx.x == 0) {
        variance = (variance / hidden_size);  // Var[x] = E[x^2]
        s_variance = rsqrtf(variance + eps);
      }
      __syncthreads();
    }

    int i = 0;
#pragma unroll
    for (int g = 0; g < UNROLL; g++) {
      if (fuse_layernorm) {
#pragma unroll
        for (int j = 0; j < sizeof(int4) / sizeof(xhalf); j++) {
          x[i] =
              (xhalf)((float)(x[i]) * s_variance *
                      (float)
                          gamma[(threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(xhalf) + j]);
          i++;
        }
      }
      MULTIMEM_ST(val[g], mc_ptr_out + (line + g * loop_step0))
    }
  }

  __syncthreads();
  if (threadIdx.x != 0) return;

  __threadfence();
  const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
  const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_NVLS2_SM_SYNC, value_to_add);

  const int lastSM =
      (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
  if (!lastSM) return;
  __threadfence_system();
  ATOMIC_MCINC(mc_flagptr + UBX_FLAG_NVLS2_AG_BAR);
  uc_flagptr[UBX_FLAG_NVLS2_ID] = reduce_id;
  cudaTriggerProgrammaticLaunchCompletion();
  volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS2_AG_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
             *flag);
      break;
    }
#endif
  }
}  // fp16 inplace reduce kernel (Hopper) MC

template <int RANKS>
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_allreduce_2shot_uc(const int myrank, const int numlines,
                                        const int64_t lineoffset_in, const int64_t lineoffset_out,
                                        uint4 *residual_in, uint4 *residual_out,
                                        xhalf *gamma, float eps,
                                        const int hidden_size, bool fuse_layernorm,
                                        ncclWindow_t window) {
  // flags[3,4,5,6]: reduce_id, sm_sync-local, flag-barrier-1, flag-barrier-2.
  // Phase 4: peer / local pointers come from the NCCL device API on the
  // registered window. REG0 layout matches ubx_kernel_a2a — see comment
  // there. lineoffset_{in,out} are uint4-line offsets within the symmetric
  // window (relative to pool base).
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(
      ncclGetLocalPointer(window, REG0_FLAG_OFFSET));

  __shared__ uint4 *userptr[RANKS];
  __shared__ int lastSM;
  int reduce_id;

  if (threadIdx.x < RANKS) {
    // Was: int *rem_flagptr = (int*)commbuff[threadIdx.x];
    int *rem_pool_ptr = reinterpret_cast<int*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));
    int *rem_flagptr = reinterpret_cast<int*>(
        reinterpret_cast<char*>(rem_pool_ptr) + REG0_FLAG_OFFSET);
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    if (blockIdx.x == 0) ATOMIC_UCINC(rem_flagptr + UBX_FLAG_NVLS2_RS_BAR);

    reduce_id = uc_flagptr[UBX_FLAG_NVLS2_ID] + 1;

    userptr[threadIdx.x] = reinterpret_cast<uint4*>(rem_pool_ptr);
  }

  if (threadIdx.x == 0) {
    volatile int *flag = uc_flagptr + UBX_FLAG_NVLS2_RS_BAR;
    lastSM = 0;
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
               *flag);
        break;
      }
#endif
    }
  }

  __syncthreads();

  int warp = blockIdx.x + (threadIdx.x >> 5);
  int dest[RANKS];
#pragma unroll
  for (int i = 0; i < RANKS; i++) dest[i] = (i + myrank + warp) & (RANKS - 1);

  __syncthreads();
  for (int line = threadIdx.x + blockDim.x * (myrank + RANKS * blockIdx.x); line < numlines;
       line += blockDim.x * gridDim.x * RANKS) {
    uint4 val[RANKS];

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      val[i] = userptr[dest[i]][lineoffset_in + line];
    }

    uint4 sum = val[0];
    xhalf *s = reinterpret_cast<xhalf *>(&sum);

#pragma unroll
    for (int i = 1; i < RANKS; i++) {
      xhalf *x = reinterpret_cast<xhalf *>(&val[i]);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += x[j];
    }

    if (residual_in != nullptr) {
      uint4 resval = residual_in[lineoffset_in + line];
      xhalf *y = reinterpret_cast<xhalf *>(&resval);
#pragma unroll
      for (int j = 0; j < 8; j++) s[j] += y[j];
      if (residual_out != nullptr) residual_out[lineoffset_in + line] = sum;
    }

#pragma unroll
    for (int i = 0; i < RANKS; i++) {
      userptr[dest[i]][lineoffset_out + line] = sum;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_NVLS2_SM_SYNC, value_to_add);
    lastSM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM) uc_flagptr[UBX_FLAG_NVLS2_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
  }
  if (threadIdx.x >= RANKS) return;
  __syncthreads();
  if (!lastSM) return;
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  // Was: ATOMIC_UCINC((int *)(userptr[threadIdx.x]) + UBX_FLAG_NVLS2_AG_BAR + RANKS*2);
  // userptr[i] points to peer i's pool base (offset 0); shift by REG0_FLAG_OFFSET
  // to land in the flag region, then index by UBX_FLAG_NVLS2_AG_BAR (int32).
  int* peer_flag = reinterpret_cast<int*>(
      reinterpret_cast<char*>(userptr[threadIdx.x]) + REG0_FLAG_OFFSET);
  ATOMIC_UCINC(peer_flag + UBX_FLAG_NVLS2_AG_BAR);
  if (threadIdx.x != 0) return;
  volatile int *flag = uc_flagptr + UBX_FLAG_NVLS2_AG_BAR;
  const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
             *flag);
      break;
    }
#endif
  }
}  // UC 2shot kernel (non-lamport)

// Standalone barrier kernel — does NO data movement. Same atomic-flag
// protocol as the combine/dispatch barriers but uses a SEPARATE flag
// set (UBX_FLAG_BARRIER_*). Lets us surround other UBX calls with a
// hard sync that doesn't share flag slots with the data-movement
// kernels — useful for isolating flag-protocol races.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_barrier(
        const int RANKS,
        const int myrank,
        int* uc_flagptr,
        int* mc_flagptr,
        ncclWindow_t window) {
  __shared__ int lastSM_shmem;
  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_BARRIER_ID] + 1;
  }
  __syncthreads();
  // SM-sync: all CTAs atomicAdd to SM_SYNC; the "last SM" is the one
  // whose atomic crosses reduce_id * UBX_MAX_SMS.
  if (threadIdx.x == 0) {
    __threadfence_system();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val = atomicAdd(uc_flagptr + UBX_FLAG_BARRIER_SM_SYNC, value_to_add);
    lastSM_shmem = (gridDim.x == 1 || old_val + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM_shmem) {
      uc_flagptr[UBX_FLAG_BARRIER_ID] = reduce_id;
    }
  }
  __syncthreads();
  // Last SM signals peers: MC path (single inc) or UC fallback (RANKS
  // threads each unicast atomic-inc a peer flag).
  if (lastSM_shmem) {
    if (mc_flagptr != nullptr) {
      if (threadIdx.x == 0) {
        ATOMIC_MCINC(mc_flagptr + UBX_FLAG_BARRIER_BAR);
      }
    } else {
      if (threadIdx.x < (unsigned)RANKS) {
        const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
        int* peer_flag = reinterpret_cast<int*>(
            ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
        ATOMIC_UCINC(peer_flag + UBX_FLAG_BARRIER_BAR);
      }
    }
  }
  // Spin-wait for all peers to have signaled this rank.
  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_BARRIER_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("UBX_BARRIER:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Diagnostic: dump per-rank peer LSA pointers via ncclGetLsaPointer for the
// pool base (offset 0). Used to verify NCCL's per-rank peer-ptr cache is
// populated and consistent across the EP group BEFORE the first combine
// fires. Each thread i in [0, RANKS) writes ncclGetLsaPointer(window, 0, i)
// to out_ptrs[i] (uint64).
__global__ void ubx_kernel_peer_ptr_dump(
    const int RANKS,
    ncclWindow_t window,
    unsigned long long* out_ptrs) {
  if (blockIdx.x != 0) return;
  if (threadIdx.x < RANKS) {
    void* p = ncclGetLsaPointer(window, 0, threadIdx.x);
    out_ptrs[threadIdx.x] = reinterpret_cast<unsigned long long>(p);
  }
}

// Minimal peer atomic-inc test: each thread i in [0, RANKS) issues
// ATOMIC_UCINC on peer i's UBX_FLAG_DIAG_PEER_TEST slot, then thread 0
// spins polling THIS rank's flag until it reaches `test_id * RANKS`.
// Same cross-rank UC atomic pattern as combine's lastSM signaling, but
// with NO lastSM determination, NO Phase 1 data copy, NO PDL. If group
// 0's combine hangs and this test ALSO hangs in isolation, the bug is
// at the level of cross-rank UC atomic writes.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_peer_atomic_test(
        const int RANKS, const int myrank, const int test_id,
        int* uc_flagptr,
        ncclWindow_t window) {
  if (blockIdx.x != 0) return;
  // Phase 1: cross-rank UC atomic-inc on each peer's flag.
  if (threadIdx.x < (unsigned)RANKS) {
    const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
    int* peer_flag = reinterpret_cast<int*>(
        ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
    ATOMIC_UCINC(peer_flag + UBX_FLAG_DIAG_PEER_TEST);
  }
  __syncthreads();
  // Phase 2: thread 0 polls own flag until peers' atomics all arrived.
  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_DIAG_PEER_TEST]);
    const int expected = test_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("DIAG_PEER_TEST:rank %d test_id %d:expecting %d got %d\n",
               myrank, test_id, expected, *flag);
        break;
      }
#endif
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

__global__ void memset_int(uint32_t *data, int n, uint32_t val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] = val;
  }
}

template <int UNROLL>
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_allreduce_2shot_mc_lamport(const int RANKS, const int myrank,
                                                const int mylines, const int numlines,
                                                int* uc_flagptr, int* mc_flagptr,
                                                uint4* mc_ptr_in, uint4* mc_ptr_out,
                                                uint4 *uc_ptr_out,
                                                uint4 *clear_ptr, uint4 *residual_in,
                                                uint4 *residual_out, xhalf *gamma, float eps,
                                                const int hidden_size, bool fuse_layernorm) {
  // flags[0,1,2]: reduce_id, sm_sync-local, flag-barrier.
  // uc_flagptr / mc_flagptr / mc_ptr_in / mc_ptr_out are pre-resolved
  // host-side by the launcher (host-side ncclGetLsaMultimemDevicePointer
  // for the multimem pointers, pool_ptr + REG0_FLAG_OFFSET for
  // uc_flagptr). uc_ptr_out (Lamport poll target) and clear_ptr
  // (next-call poison target) remain pool-resident raw pointers.
  int reduce_id;
  __shared__ float s_variance;

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    if (blockIdx.x == 0) ATOMIC_MCINC(mc_flagptr + UBX_FLAG_NVLS2_LAMPORT_RS_BAR);
    reduce_id = uc_flagptr[UBX_FLAG_NVLS2_LAMPORT_ID];
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync =
        atomicAdd(uc_flagptr + UBX_FLAG_NVLS2_LAMPORT_SM_SYNC, value_to_add);
    volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS2_LAMPORT_RS_BAR]);
    reduce_id++;
    const int lastSM =
        (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);

    if (lastSM) uc_flagptr[UBX_FLAG_NVLS2_LAMPORT_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();

    const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("NVONLY RSBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
               *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  const int loop_step0 = blockDim.x;
  const int loop_step = loop_step0 * UNROLL * gridDim.x;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x * UNROLL;
  const int end_elem = max(start_elem, mylines);

  for (int line = start_elem; line < end_elem; line += loop_step) {
    uint4 val[UNROLL];
    xhalf *x = reinterpret_cast<xhalf *>(&val[0]);
#pragma unroll
    for (int i = 0; i < UNROLL; i++) MULTIMEM_LD(val[i], mc_ptr_in + (line + i * loop_step0))

    if (residual_in != nullptr) {
      for (int i = 0; i < UNROLL; i++) {
        uint4 resval = residual_in[line + i * loop_step0];
        xhalf *y = reinterpret_cast<xhalf *>(&resval);
#pragma unroll
        for (int j = 0; j < 8; j++) x[i * 8 + j] += y[j];
        if (residual_out != nullptr) residual_out[line + i * loop_step0] = val[i];
      }
    }
    if (fuse_layernorm) {
      float local_var_sum = 0.0f;
      for (int j = 0; j < UNROLL * sizeof(int4) / sizeof(xhalf); j++)
        local_var_sum += (float)(x[j]) * (float)(x[j]);

      float packed[1] = {local_var_sum};
      blockReduceSumV2<float, 1>(packed);
      float variance = packed[0];

      if (threadIdx.x == 0) {
        variance = (variance / hidden_size);  // Var[x] = E[x^2]
        s_variance = rsqrtf(variance + eps);
      }
      __syncthreads();
    }

    int i = 0;
#pragma unroll
    for (int g = 0; g < UNROLL; g++) {
      if (fuse_layernorm) {
#pragma unroll
        for (int j = 0; j < sizeof(int4) / sizeof(xhalf); j++) {
          x[i] =
              (xhalf)((float)(x[i]) * s_variance *
                      (float)
                          gamma[(threadIdx.x + g * loop_step0) * sizeof(int4) / sizeof(xhalf) + j]);
          i++;
        }
      }
      MULTIMEM_ST(val[g], mc_ptr_out + (line + g * loop_step0))
    }
  }

  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < numlines;
       line += blockDim.x * gridDim.x) {
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (true) {
      uint4 result;

      asm volatile("ld.volatile.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
                   : "l"(&uc_ptr_out[line])
                   : "memory");
      if (result.w != UBX_LAMPORT_INT) {
        if (clear_ptr) clear_ptr[line].w = UBX_LAMPORT_INT;
        break;
      }
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("Lamport POLL:SM %d [%d]:expecting %d got (%d,%d,%d) %d\n", blockIdx.x, threadIdx.x,
               UBX_LAMPORT_INT, result.x, result.y, result.z, result.w);
        break;
      }
#endif
    }
  }

}  // two-shot NVLS + lamport sync instead of last membar

__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_allgather_mc(const int RANKS, const int myrank, const int mylines,
                      int* uc_flagptr, int* mc_flagptr,
                      uint4* uc_ptr_in, uint4* mc_ptr_out) {
  // flags[7,8,9]: reduce_id, sm_sync-local, flag-barrier.
  // uc_flagptr / mc_flagptr / mc_ptr_out are pre-resolved host-side by
  // the launcher; mc_ptr_out includes the per-rank slice offset.
  int reduce_id;

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_NVLS1_ID] + 1;
  }

  __syncthreads();

  const int loop_step0 = blockDim.x*gridDim.x;
  const int start_elem = threadIdx.x + blockDim.x * blockIdx.x;
  const int end_elem = max(start_elem, mylines);

  for (int line = start_elem; line < end_elem; line += loop_step0)
    MULTIMEM_ST(uc_ptr_in[line], mc_ptr_out + line);

  __syncthreads();
  if (threadIdx.x != 0) return;

  __threadfence();
  const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
  const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_NVLS1_SM_SYNC, value_to_add);

  const int lastSM =
      (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
  if (!lastSM) return;
  __threadfence_system();
  ATOMIC_MCINC(mc_flagptr + UBX_FLAG_NVLS1_BAR);
  uc_flagptr[UBX_FLAG_NVLS1_ID] = reduce_id;
  cudaTriggerProgrammaticLaunchCompletion();
  volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS1_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
             *flag);
      break;
    }
#endif
  }
}  // fp16 out of place MC Allgather kernel

// UC out-of-place allgather. Required when multicast isn't available
// (NVLS-disabled EP groups, large EP groups >36 ranks, etc).
// Each warp writes the FULL local input to ONE peer's slot. Warps are
// distributed round-robin across peers (same pattern as ubx_kernel_a2a),
// then a UC barrier (per-peer ATOMIC_UCINC + spin-on-own-flag) finishes
// the kernel. Uses the same NVLS1 flag set as the MC variant.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_allgather_uc(
        const int RANKS, const int myrank, const int mylines,
        const int64_t lineoffset_out,
        uint4* ptr_in, ncclWindow_t window) {
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(
      ncclGetLocalPointer(window, REG0_FLAG_OFFSET));

  int reduce_id;
  __shared__ int lastSM;

  const int globalthread = threadIdx.x + blockDim.x * blockIdx.x;
  const int numwarps    = blockDim.x * gridDim.x / 32;
  const int globalwarp  = globalthread / 32;
  const int mydest      = globalwarp % RANKS;
  const int mywarp      = globalwarp / RANKS;

  // Destination on peer `mydest`: my slice starts at lineoffset_out + myrank*mylines.
  uint4* uc_dst_ptr = reinterpret_cast<uint4*>(
      ncclGetLsaPointer(window, 0, mydest));
  uc_dst_ptr += lineoffset_out + myrank * mylines;

  const int mythreadidx = mywarp * 32 + globalthread % 32;
  const int myblockdim  = (numwarps / RANKS + (mydest < numwarps % RANKS ? 1 : 0)) * 32;

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_NVLS1_ID] + 1;
  }
  __syncthreads();

  // Every warp pushes the FULL local input (mylines lines) to its assigned peer.
  for (int line = mythreadidx; line < mylines; line += myblockdim)
    uc_dst_ptr[line] = ptr_in[line];

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_NVLS1_SM_SYNC, value_to_add);
    lastSM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM) uc_flagptr[UBX_FLAG_NVLS1_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
  }
  if (threadIdx.x >= (unsigned)RANKS) return;
  __syncthreads();
  if (!lastSM) return;
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  int* peer_flag = reinterpret_cast<int*>(
      ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
  ATOMIC_UCINC(peer_flag + UBX_FLAG_NVLS1_BAR);

  if (threadIdx.x != 0) return;
  volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS1_BAR]);
  const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("UC AGBAR:SM %d [%d]:expecting %d got %d\n",
             blockIdx.x, threadIdx.x, expected, *flag);
      break;
    }
#endif
  }
}  // UC out-of-place Allgather kernel

__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_a2a(const int RANKS, const int myrank, const int nlines,
                    const int64_t lineoffset_out, uint4 *ptr_in, ncclWindow_t window) {
  // flags[7,8,9]: reduce_id, sm_sync-local, flag-barrier
  // Exit barrier uses UC atomics (not MC) so this kernel works with multicast
  // disabled where multicast is unavailable. Pattern matches UC allreduce barrier
  // at lines 304-332.
  //
  // Phase 4: peer / local pointers come from the NCCL device API on the
  // registered window. REG0 layout (transitional during the migration):
  // first RANKS*sizeof(void*) bytes were the legacy commbuff[] region —
  // the allocator stops populating them in Phase 5; the flag region lives
  // immediately after.
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(
      ncclGetLocalPointer(window, REG0_FLAG_OFFSET));

  int reduce_id;
  __shared__ int lastSM;

  const int globalthread = threadIdx.x + blockDim.x * blockIdx.x;
  const int numwarps = blockDim.x * gridDim.x / 32;
  const int globalwarp = globalthread / 32;
  const int mydest = globalwarp % RANKS;
  const int mywarp = globalwarp / RANKS;
  // Was: uint4* uc_dst_ptr = (uint4*)commbuff[mydest];
  uint4* uc_dst_ptr = reinterpret_cast<uint4*>(
      ncclGetLsaPointer(window, 0, mydest));
  uc_dst_ptr += lineoffset_out+myrank*nlines;
  uint4* ptr_in_rank = ptr_in+(mydest*nlines);
  const int mythreadidx = mywarp * 32 + globalthread % 32;
  const int myblockdim = (numwarps / RANKS + (mydest<numwarps%RANKS?1:0)) * 32;

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_NVLS1_ID] + 1;
  }

  __syncthreads();

  // Issue 4 loads back-to-back then 4 stores so the load pipeline + NVLink
  // injection overlap, lifting the per-thread issue rate when peer NVLink
  // injection is the bottleneck.
  // ptxas: 64 regs/thread, 0 spill bytes — full occupancy preserved.
  // 8-way was empirically tested and did NOT pay off (injection ceiling, not
  // a register-spill cliff).
  {
    constexpr int UNROLL = 4;
    int line = mythreadidx;
    const int step = myblockdim;
    for (; line + (UNROLL - 1) * step < nlines; line += UNROLL * step) {
      uint4 v[UNROLL];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) v[i] = ptr_in_rank[line + i * step];
#pragma unroll
      for (int i = 0; i < UNROLL; i++) uc_dst_ptr[line + i * step] = v[i];
    }
    for (; line < nlines; line += step)
      uc_dst_ptr[line] = ptr_in_rank[line];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_NVLS1_SM_SYNC, value_to_add);
    lastSM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM) uc_flagptr[UBX_FLAG_NVLS1_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
  }
  // UC barrier: threads 0..RANKS-1 each atomically increment their peer's flag.
  // Was: ATOMIC_UCINC((int *)(commbuff[threadIdx.x]) + UBX_FLAG_NVLS1_BAR + RANKS*2);
  // The "+ RANKS * 2" was the in-int32 skip past the legacy commbuff region
  // (RANKS * 8 bytes == RANKS * 2 int32s). NCCL's LSA peer-pointer fetch
  // with an explicit byte offset replaces both steps cleanly.
  if (threadIdx.x >= (unsigned)RANKS) return;
  __syncthreads();
  if (!lastSM) return;
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  int* peer_flag = reinterpret_cast<int*>(
      ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
  ATOMIC_UCINC(peer_flag + UBX_FLAG_NVLS1_BAR);
  if (threadIdx.x != 0) return;
  volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS1_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("NVONLY AGBAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected,
             *flag);
      break;
    }
#endif
  }
}  // out of place UC Alltoall kernel

// Variable-length alltoall: each rank sends/receives different amounts.
// send_offsets[r]: start line in ptr_in for data going to rank r
// send_lines[r]:   number of uint4 lines to send to rank r
// dest_offsets[r]:  absolute line offset in rank r's pool where data from myrank lands
//                   (includes both pool base offset and per-source recv offset)
// Variable-length alltoallv kernel. Offsets and per-destination counts
// are all in BYTES. For each (this-rank → mydest) stream the kernel does
// a bulk uint4 (16 B) loop for the aligned prefix, then a small ushort
// (2 B) tail for the remaining 0..14 bytes. This keeps the fast path
// (wide-row tensors with per-dest send_bytes a multiple of 16, e.g.
// MoE bf16 [N,hidden=2688]) at full speed, while correctly handling
// narrow rows (fp32 1D dispatch_probs etc.) without overshooting into
// the next source's slice.
//
// All four supported element sizes (bf16, fp16, fp32, fp64) are 2-byte
// aligned, so 2 B tail writes are always sufficient — no 1 B path needed.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_a2av(const int RANKS, const int myrank,
                     const long long* __restrict__ send_byte_offsets,
                     const long long* __restrict__ send_byte_counts,
                     const long long* __restrict__ dest_byte_offsets,
                     const void *ptr_in_void, ncclWindow_t window) {
  // Phase 4: peer / local pointers come from the NCCL device API on the
  // registered window. REG0 layout matches ubx_kernel_a2a — see comment
  // there.
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(
      ncclGetLocalPointer(window, REG0_FLAG_OFFSET));

  int reduce_id;
  __shared__ int lastSM;

  const int globalthread = threadIdx.x + blockDim.x * blockIdx.x;
  const int numwarps = blockDim.x * gridDim.x / 32;
  const int globalwarp = globalthread / 32;
  const int mydest = globalwarp % RANKS;
  const int mywarp = globalwarp / RANKS;
  const int mythreadidx = mywarp * 32 + globalthread % 32;
  const int myblockdim = (numwarps / RANKS + (mydest < numwarps % RANKS ? 1 : 0)) * 32;

  const long long sb_ll = send_byte_counts[mydest];
  const int sb = static_cast<int>(sb_ll);  // per-dest count fits in 31 bits (≤ a few hundred MB)
  const char* my_src = static_cast<const char*>(ptr_in_void) + send_byte_offsets[mydest];
  char* my_dst = reinterpret_cast<char*>(
      ncclGetLsaPointer(window, 0, mydest)) + dest_byte_offsets[mydest];

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_NVLS1_ID] + 1;
  }
  __syncthreads();

  // Two paths, selected at runtime per destination (uniform within a warp
  // since all threads in this warp have the same `mydest`):
  //   (a) FAST: src and dst both 16-aligned -> bulk uint4 + ushort tail
  //   (b) SAFE: misaligned pointers -> pure 2 B stores
  // Path (a) covers the common wide-row case (e.g. bf16 [N, hidden=2688]
  // where row_bytes=5376 keeps every per-dest offset 16-aligned). Path (b)
  // handles the narrow-row case (fp32 1D dispatch_probs etc.) where
  // per-dest send_bytes are only 4 B aligned and uint4 stores would trap
  // with cudaErrorMisalignedAddress on Blackwell.
  const bool can_uint4 = (sb >= 16)
      && ((reinterpret_cast<uintptr_t>(my_src) & 15U) == 0)
      && ((reinterpret_cast<uintptr_t>(my_dst) & 15U) == 0);
  if (can_uint4) {
    const int bulk_lines = sb / 16;
    const uint4* src_u4 = reinterpret_cast<const uint4*>(my_src);
    uint4* dst_u4 = reinterpret_cast<uint4*>(my_dst);
    for (int line = mythreadidx; line < bulk_lines; line += myblockdim)
      dst_u4[line] = src_u4[line];
    const int tail_bytes = sb - bulk_lines * 16;
    if (tail_bytes > 0) {
      const int tail_start_b2 = bulk_lines * 8;       // 8 ushorts per uint4
      const int tail_units = tail_bytes / 2;
      const unsigned short* src_u2 = reinterpret_cast<const unsigned short*>(my_src);
      unsigned short* dst_u2 = reinterpret_cast<unsigned short*>(my_dst);
      if (mythreadidx < tail_units)
        dst_u2[tail_start_b2 + mythreadidx] = src_u2[tail_start_b2 + mythreadidx];
    }
  } else {
    // 2 B safe path — all our element types (bf16/fp16/fp32/fp64) are 2 B
    // aligned, so this always works.
    const int n_units = sb / 2;
    const unsigned short* src_u2 = reinterpret_cast<const unsigned short*>(my_src);
    unsigned short* dst_u2 = reinterpret_cast<unsigned short*>(my_dst);
    for (int i = mythreadidx; i < n_units; i += myblockdim)
      dst_u2[i] = src_u2[i];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_NVLS1_SM_SYNC, value_to_add);
    lastSM = (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM) uc_flagptr[UBX_FLAG_NVLS1_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
  }
  // UC barrier
  if (threadIdx.x >= (unsigned)RANKS) return;
  __syncthreads();
  if (!lastSM) return;
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  int* peer_flag = reinterpret_cast<int*>(
      ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
  ATOMIC_UCINC(peer_flag + UBX_FLAG_NVLS1_BAR);
  if (threadIdx.x != 0) return;
  volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS1_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("A2AV BAR:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x, expected, *flag);
      break;
    }
#endif
  }
}  // out of place UC Alltoallv kernel

// Phase 1 kernel: optional UC entry barrier + UC write data to peers.
// Split from the monolithic alltoall_lamport kernel so phase 2's poll
// (which busy-waits on .w sentinel) cannot run concurrently with phase 1
// on the same SM. Hang-tests at 4 ranks + 4M graph mode showed a drift
// that didn't trace to PDL or buffer-gap — splitting the kernels means
// phase 2 only begins once phase 1 has completed at this rank (kernel
// boundary serializes), reducing the kernel's resource pressure during
// phase 2 polling.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_alltoall_lamport_write(const int RANKS, const int myrank, const int nlines,
                            const int64_t lineoffset_out,
                            uint4 *ptr_in,
                            const bool skip_barrier,
                            ncclWindow_t window) {
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(
      ncclGetLocalPointer(window, REG0_FLAG_OFFSET));

  __shared__ int lastSM;
  __shared__ int s_reduce_id;

  const int globalthread = threadIdx.x + blockDim.x * blockIdx.x;
  const int numwarps = blockDim.x * gridDim.x / 32;
  const int globalwarp = globalthread / 32;
  const int mydest = globalwarp % RANKS;
  const int mywarp = globalwarp / RANKS;
  uint4 *uc_dst_ptr = reinterpret_cast<uint4*>(
      ncclGetLsaPointer(window, 0, mydest));
  uc_dst_ptr += lineoffset_out + myrank * nlines;
  uint4 *ptr_in_rank = ptr_in + (mydest * nlines);
  const int mythreadidx = mywarp * 32 + globalthread % 32;
  const int myblockdim = (numwarps / RANKS + (mydest < numwarps % RANKS ? 1 : 0)) * 32;

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    if (skip_barrier) {
      cudaTriggerProgrammaticLaunchCompletion();
    } else {
      int reduce_id = uc_flagptr[UBX_FLAG_NVLS1_ID];
      const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
      const int old_val_sm_sync =
          atomicAdd(uc_flagptr + UBX_FLAG_NVLS1_SM_SYNC, value_to_add);
      reduce_id++;
      s_reduce_id = reduce_id;
      lastSM =
          (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);

      if (lastSM) uc_flagptr[UBX_FLAG_NVLS1_ID] = reduce_id;
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
  __syncthreads();

  if (!skip_barrier) {
    if (threadIdx.x == 0 && lastSM) __threadfence_system();
    __syncthreads();
    if (threadIdx.x < (unsigned)RANKS && lastSM) {
      int* peer_flag = reinterpret_cast<int*>(
          ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
      ATOMIC_UCINC(peer_flag + UBX_FLAG_NVLS1_BAR);
    }
    if (threadIdx.x == 0) {
      volatile int *flag = (volatile int *)&(uc_flagptr[UBX_FLAG_NVLS1_BAR]);
      const int expected = s_reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
      clock_t s = clock64();
#endif
      while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
        if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
          printf("A2A Lamport READY:SM %d [%d]:expecting %d got %d\n", blockIdx.x, threadIdx.x,
                 expected, *flag);
          break;
        }
#endif
      }
    }
    __syncthreads();
  }

  // Phase 1: UC write data to peers — single NVLink latency
  for (int line = mythreadidx; line < nlines; line += myblockdim)
    uc_dst_ptr[line] = ptr_in_rank[line];
}  // write phase: barrier + UC write to peers

// Phase 2 kernel (large-size variant): Lamport poll on own output buffer
// with per-line poison-write to clear_ptr interleaved into the poll loop.
// Runs after the write-phase kernel completes at this rank. Best when
// nlines is large enough that the per-line poison-write overlaps with
// the poll busy-wait latency (default cutoff: bytes_per_rank * ranks >=
// UBX_LAMPORT_POISON_FIRST_CUTOFF, default 1 MiB).
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_alltoall_lamport_poll(const int RANKS, const int nlines,
                            uint4 *uc_ptr_out,
                            uint4 *clear_ptr) {
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    cudaTriggerProgrammaticLaunchCompletion();
  }
  __syncthreads();

  const int total_nlines = nlines * RANKS;
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < total_nlines;
       line += blockDim.x * gridDim.x) {
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (true) {
      uint4 result;

      asm volatile("ld.volatile.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
                   : "l"(&uc_ptr_out[line])
                   : "memory");
      if (result.w != UBX_LAMPORT_INT) {
        if (clear_ptr) clear_ptr[line].w = UBX_LAMPORT_INT;
        break;
      }
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("A2A Lamport POLL:SM %d [%d]:expecting != %08x got (%08x,%08x,%08x) %08x\n",
               blockIdx.x, threadIdx.x, UBX_LAMPORT_INT, result.x, result.y, result.z,
               result.w);
        break;
      }
#endif
    }
  }
}  // poll phase: Lamport poll + per-line poison clear_ptr (large-size)

// Phase 2 kernel (small-size variant): poison clear_ptr in bulk first,
// then Lamport poll. Two separate sweeps over the data. For small
// nlines this is faster than the interleaved variant because the
// poison-write loop avoids the branch + has better warp coherence,
// and the poll loop is hot/tight without a side-store every iteration.
// Cost: 2x memory traffic — pays off only below the cutoff.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_alltoall_lamport_poll_poison_first(const int RANKS, const int nlines,
                            uint4 *uc_ptr_out,
                            uint4 *clear_ptr) {
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    cudaTriggerProgrammaticLaunchCompletion();
  }
  __syncthreads();

  const int total_nlines = nlines * RANKS;

  // Phase 2a: bulk poison clear_ptr.w.
  if (clear_ptr) {
    for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < total_nlines;
         line += blockDim.x * gridDim.x) {
      clear_ptr[line].w = UBX_LAMPORT_INT;
    }
  }

  // Phase 2b: Lamport poll for peer data.
  for (int line = threadIdx.x + blockDim.x * blockIdx.x; line < total_nlines;
       line += blockDim.x * gridDim.x) {
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (true) {
      uint4 result;
      asm volatile("ld.volatile.v4.u32 {%0, %1, %2, %3}, [%4];"
                   : "=r"(result.x), "=r"(result.y), "=r"(result.z), "=r"(result.w)
                   : "l"(&uc_ptr_out[line])
                   : "memory");
      if (result.w != UBX_LAMPORT_INT) break;
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("A2A Lamport POLL2:SM %d [%d]:expecting != %08x got (%08x,%08x,%08x) %08x\n",
               blockIdx.x, threadIdx.x, UBX_LAMPORT_INT, result.x, result.y, result.z,
               result.w);
        break;
      }
#endif
    }
  }
}  // poll phase: bulk poison clear_ptr first, then Lamport poll (small-size)
// Maximum ranks supported in the dest_ptr shared-mem cache.
#define MAXRANKS 72

// Out-of-place token-dispatch kernel: routes bf16 tokens to remote ranks with
// mxfp8 quantization fused in.  Each token is independently routed to zero or
// more experts (determined by token_offsets).  Quantization uses one E8M0
// scale per 32-element block (mxfp8 microscaling).
//
// Memory layout
// -------------
// Input:  ptr_in_bf16[token * blocks_per_token * 4 + blk * 4 + line_in_blk]
//   bfloat_lines_per_block = 4  (32 bf16 × 2 B / 16 B per uint4)
//
// Output (per dest rank R, output slot S):
//   fp8 data:  (uint4*)(commbuff[R])[lineoffset_out + S * blocks_per_token*2 + blk*2 + fp8_line]
//   scales:    ((uint8_t*)(commbuff[R]))[lineoffset_scales*16 + S*blocks_per_token + blk]
//
// token_offsets[token * RANKS * experts_per_rank + expert] = dest slot (≥ 0) or -1.
//
// Thread assignment: one warp (32 lanes) per 32-element block.
//   lane i → element i, where line_in_blk = i/8, elem_in_line = i%8.
// Expert writes are distributed across warp lanes: each lane checks one expert
// per outer loop iteration, so only the lanes assigned to a live expert issue
// remote stores.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_a2av_token_bf16_mxfp8(
        const int RANKS,
        const int myrank,
        const int ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int* __restrict__ token_offsets,
        const int64_t lineoffset_out,
        const int64_t lineoffset_scales,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        const int do_sync,
        const int expert_start,
        const int expert_count,
        ncclWindow_t window) {
  // 4-thread sub-warp layout: each warp = 8 sub-warps × 4 threads, each
  // sub-warp processes one 32-element block. Each thread loads a single
  // 16 B uint4 (8 bf16 elements), runs an intra-thread amax over its 8
  // elements, and the 4-thread sub-warp does a 2-step shfl_xor reduction
  // for the per-block amax. fp8 output (32 B / block) is produced as 4 ×
  // 8 B uint64 stores at consecutive offsets — the four threads coalesce
  // into one 32 B transaction per active expert. E8M0 scale (1 B) is
  // emitted by lane 0 of each sub-warp.

  const int fp8_lines_per_block = 2;  // 32 * 1B / 16B per uint4 = 2
  const int total_experts       = RANKS * experts_per_rank;
  const int exp_begin = expert_start;
  const int exp_end   = expert_count > 0 ? min(expert_start + expert_count, total_experts)
                                         : total_experts;

  __shared__ uint4* dest_ptr[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sw_base_lane  = lane & ~3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int tokens_per_cta = (ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, ntokens);

  if (threadIdx.x < RANKS)
    dest_ptr[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_A2AV_ID] + 1;
  }
  __syncthreads();

  const int total_cta_blocks = (my_token_end - my_token_start) * blocks_per_token;
  // All 32 lanes of a warp must enter the in-loop shfl_sync calls in lockstep,
  // so we iterate a uniform number of times per sub-warp and gate per-iter
  // work on a `valid` flag rather than `continue;`-ing the loop early.
  const int max_iters = (total_cta_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_cta_blocks;

    int blk_in_token = 0, global_token = 0, global_line = 0;
    if (valid) {
      const int local_token = blk_local / blocks_per_token;
      blk_in_token = blk_local % blocks_per_token;
      global_token = local_token + my_token_start;
      global_line  = global_token * blocks_per_token * 4 + blk_in_token * 4 + lane_in_sub;
    }

    // Each thread loads one uint4 = 8 bf16 elements. Out-of-range threads load
    // index 0 — value is unused.
    const uint4 v = ptr_in_bf16[valid ? global_line : 0];
    const xhalf* my_elems = reinterpret_cast<const xhalf*>(&v);

    // Intra-thread amax over 8 elements.
    float my_max = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; e++)
      my_max = fmaxf(my_max, fabsf(__bfloat162float(my_elems[e])));

    // 4-thread sub-warp amax reduction (2 shfl_xor steps).
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFFU, my_max, 1));
    my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFFU, my_max, 2));
    const float amax = my_max;

    // E8M0 scale: smallest power-of-2 S ≥ amax/448 (ceil to avoid saturation).
    const float log2_scale = (amax > 0.0f) ? ceilf(log2f(amax / 448.0f)) : -127.0f;
    const int   e8m0       = max(0, min(255, (int)log2_scale + 127));
    const float scale_f    = exp2f((float)(e8m0 - 127));

    // Quantize my 8 bf16 elements to fp8 e4m3, pack into uint64 (8 bytes).
    uint64_t fp8_packed = 0;
#pragma unroll
    for (int e = 0; e < 8; e++) {
      const float fv = (scale_f > 0.0f) ? __bfloat162float(my_elems[e]) / scale_f : 0.0f;
      const __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(fv);
      const uint8_t b = *reinterpret_cast<const uint8_t*>(&fp8_v);
      fp8_packed |= (static_cast<uint64_t>(b)) << (e * 8);
    }

    // Per-expert loop: each sub-warp's 4 threads check 4 experts per outer
    // iter (stride 4); the inner t-loop serialises over the 4 candidates,
    // and each accepted candidate triggers a 32 B coalesced write across
    // the sub-warp + 1 B scale write from lane 0.
    for (int expert_base = exp_begin; expert_base < exp_end; expert_base += 4) {
      int slot   = -1;
      int expert = 0;
      if (valid) {
        expert = expert_base + lane_in_sub;
        if (expert < exp_end)
          slot = token_offsets[global_token * total_experts + expert];
      }

#pragma unroll
      for (int t = 0; t < 4; t++) {
        const int peer_slot   = __shfl_sync(0xFFFFFFFFU, slot,   sw_base_lane | t);
        const int peer_expert = __shfl_sync(0xFFFFFFFFU, expert, sw_base_lane | t);
        if (valid && peer_slot >= 0) {
          const int dest_rank           = peer_expert / experts_per_rank;
          const int fp8_lines_per_token = blocks_per_token * fp8_lines_per_block;
          uint64_t* fp8_dst = reinterpret_cast<uint64_t*>(
              dest_ptr[dest_rank] + lineoffset_out
              + peer_slot * fp8_lines_per_token + blk_in_token * fp8_lines_per_block);
          fp8_dst[lane_in_sub] = fp8_packed;

          if (lane_in_sub == 0) {
            uint8_t* dst_scale = reinterpret_cast<uint8_t*>(dest_ptr[dest_rank])
                               + lineoffset_scales * 16
                               + peer_slot * blocks_per_token + blk_in_token;
            *dst_scale = static_cast<uint8_t>(e8m0);
          }
        }
      }
    }
  }

  __syncthreads();

  // SM-sync: each SM's thread 0 atomically increments SM_SYNC and checks
  // whether it is the last SM to arrive. Share lastSM via shared memory so
  // RANKS threads (needed for the UC barrier fanout below) can branch on it.
  __shared__ int lastSM_shmem;
  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_A2AV_SM_SYNC, value_to_add);
    lastSM_shmem =
        (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
  }
  __syncthreads();

  if (!lastSM_shmem) return;  // non-last SMs are done

  // Last SM: thread 0 publishes A2AV_ID, triggers PDL completion, fences for
  // visibility, then drives the cross-rank barrier — MC if available, else UC.
  if (threadIdx.x == 0) {
    uc_flagptr[UBX_FLAG_A2AV_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
    __threadfence_system();
  }
  __syncthreads();

  if (mc_flagptr != nullptr) {
    // MC barrier (intra-rack): one multicast atomic-inc fans out to all peers.
    if (threadIdx.x == 0) {
      ATOMIC_MCINC(mc_flagptr + UBX_FLAG_A2AV_BAR);
    }
  } else {
    // UC barrier (NVLS-disabled): threads 0..RANKS-1 each do one
    // unicast atomic-inc on a peer's BAR flag. Matches the bf16_bf16 kernel's
    // UC path (mc_flagptr == nullptr signals UC).
    const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
    if (threadIdx.x < (unsigned)RANKS) {
      int* peer_flag = reinterpret_cast<int*>(
          ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
      ATOMIC_UCINC(peer_flag + UBX_FLAG_A2AV_BAR);
    }
  }

  if (threadIdx.x != 0) return;

  if (!do_sync) return;  // async mode: signal sent, skip polling

  volatile int* flag     = (volatile int*)&(uc_flagptr[UBX_FLAG_A2AV_BAR]);
  const int expected     = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("A2AV_MXFP8 BAR:SM %d [%d]:expecting %d got %d\n",
             blockIdx.x, threadIdx.x, expected, *flag);
      break;
    }
#endif
  }
}  // out-of-place token-dispatch bf16→mxfp8 kernel

// Out-of-place token-dispatch bf16→bf16 (no quantization).
//
// Same routing scheme as the mxfp8 variant — token_offsets[t,e] = slot or -1,
// same NVLS1 flag protocol — but writes the raw bf16 block (4 uint4 lines per
// 32-element block) to the destination instead of fp8 + E8M0 scale.
//
// Memory layout
// -------------
// Input:   ptr_in_bf16[token * blocks_per_token * 4 + blk * 4 + line_in_blk]
// Output (per dest rank R, output slot S):
//   bf16 data: (uint4*)(commbuff[R])[lineoffset_out + S * blocks_per_token*4
//                                     + blk*4 + line_in_blk]   (line ∈ {0,1,2,3})
//
// Thread assignment: 4-thread sub-warp per 32-element block (8 sub-warps per
// warp). Each thread loads a single 16 B uint4 (8 bf16). For each routed
// expert, the 4 threads cooperatively write 4 contiguous uint4 lines to
// dst[slot] — the writes coalesce into one 64 B transaction per active expert.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_a2av_token_bf16_bf16(
        const int RANKS,
        const int myrank,
        const int ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int* __restrict__ token_offsets,
        const int64_t lineoffset_out,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        const int do_sync,
        const int expert_start,
        const int expert_count,
        ncclWindow_t window) {
  const int total_experts = RANKS * experts_per_rank;
  const int exp_begin = expert_start;
  const int exp_end   = expert_count > 0 ? min(expert_start + expert_count, total_experts)
                                         : total_experts;
  // For the UC barrier fallback at the end of the kernel.
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);

  __shared__ uint4* dest_ptr[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sw_base_lane  = lane & ~3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int tokens_per_cta = (ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, ntokens);

  if (threadIdx.x < RANKS)
    dest_ptr[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_A2AV_ID] + 1;
  }
  __syncthreads();

  const int total_cta_blocks = (my_token_end - my_token_start) * blocks_per_token;
  const int max_iters = (total_cta_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_cta_blocks;

    int blk_in_token = 0, global_token = 0, global_line = 0;
    if (valid) {
      const int local_token = blk_local / blocks_per_token;
      blk_in_token = blk_local % blocks_per_token;
      global_token = local_token + my_token_start;
      global_line  = global_token * blocks_per_token * 4 + blk_in_token * 4 + lane_in_sub;
    }

    // Each thread loads one uint4 = 8 bf16 elements (its line of the block).
    // OOB threads load index 0 (value unused).
    const uint4 v = ptr_in_bf16[valid ? global_line : 0];

    const int bf16_lines_per_token = blocks_per_token * 4;
    for (int expert_base = exp_begin; expert_base < exp_end; expert_base += 4) {
      int slot   = -1;
      int expert = 0;
      if (valid) {
        expert = expert_base + lane_in_sub;
        if (expert < exp_end)
          slot = token_offsets[global_token * total_experts + expert];
      }

#pragma unroll
      for (int t = 0; t < 4; t++) {
        const int peer_slot   = __shfl_sync(0xFFFFFFFFU, slot,   sw_base_lane | t);
        const int peer_expert = __shfl_sync(0xFFFFFFFFU, expert, sw_base_lane | t);
        if (valid && peer_slot >= 0) {
          const int dest_rank = peer_expert / experts_per_rank;
          uint4* dst_data = dest_ptr[dest_rank] + lineoffset_out
                          + peer_slot * bf16_lines_per_token + blk_in_token * 4;
          // Each thread writes its own line at offset lane_in_sub — 4
          // contiguous uint4 stores coalesce into one 64 B transaction.
          dst_data[lane_in_sub] = v;
        }
      }
    }
  }

  __syncthreads();

  // SM-sync: each SM's thread 0 atomically increments SM_SYNC and checks
  // whether it is the last SM to arrive. Share lastSM via shared memory so
  // RANKS threads (needed for the UC barrier fanout below) can branch on it.
  __shared__ int lastSM_shmem;
  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_A2AV_SM_SYNC, value_to_add);
    lastSM_shmem =
        (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
  }
  __syncthreads();

  if (!lastSM_shmem) return;  // non-last SMs are done

  // Last SM: thread 0 publishes A2AV_ID, triggers PDL completion, fences for
  // visibility, then drives the cross-rank barrier — MC if available, else UC.
  if (threadIdx.x == 0) {
    uc_flagptr[UBX_FLAG_A2AV_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
    __threadfence_system();
  }
  __syncthreads();

  if (mc_flagptr != nullptr) {
    // MC barrier (intra-rack): one multicast atomic-inc fan-outs to all peers.
    if (threadIdx.x == 0) {
      ATOMIC_MCINC(mc_flagptr + UBX_FLAG_A2AV_BAR);
    }
  } else {
    // UC barrier (NVLS-disabled): threads 0..RANKS-1 each do one
    // unicast atomic-inc on a peer's BAR flag. Matches the a2a kernel's UC
    // path (ub-x supports both barriers; mc_flagptr == nullptr signals UC).
    if (threadIdx.x < (unsigned)RANKS) {
      int* peer_flag = reinterpret_cast<int*>(
          ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
      ATOMIC_UCINC(peer_flag + UBX_FLAG_A2AV_BAR);
    }
  }

  if (threadIdx.x != 0) return;

  if (!do_sync) return;

  volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_A2AV_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("A2AV_BF16 BAR:SM %d [%d]:expecting %d got %d\n",
             blockIdx.x, threadIdx.x, expected, *flag);
      break;
    }
#endif
  }
}  // out-of-place token-dispatch bf16→bf16 kernel

// Top-K variant of the bf16→bf16 fused dispatch.
//
// The base kernel above iterates *every* (token, global-expert) pair —
// for total_experts=128 and top-K=6 that's 95 % dead checks. This variant
// takes a pre-computed [ntokens, topk_max] LUT of (expert_id, slot) for
// each routed entry, so the inner loop runs only K times per token
// instead of total_experts times.
//
// Pre-computed inputs (host-side via compute_dispatch_topk_map):
//   topk_expert[t, k]  : global expert id of t's k-th routed expert,
//                        sorted ascending, or -1 for unused k slots.
//   topk_slot[t, k]    : destination slot index for (t, expert), or -1.
//                        Identical -1 mask as topk_expert by construction.
// On a -1 entry we `break` out of the k loop — topk_expert is sorted
// ascending, so unused slots are guaranteed to come last.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_a2av_token_bf16_bf16_topk(
        const int RANKS,
        const int myrank,
        const int ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int topk_max,
        const int* __restrict__ topk_expert,  // [ntokens, topk_max]
        const int* __restrict__ topk_slot,    // [ntokens, topk_max]
        const int64_t lineoffset_out,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        const int do_sync,
        ncclWindow_t window) {
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);

  __shared__ uint4* dest_ptr[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int tokens_per_cta = (ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, ntokens);

  if (threadIdx.x < RANKS)
    dest_ptr[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_A2AV_ID] + 1;
  }
  __syncthreads();

  const int bf16_lines_per_token = blocks_per_token * 4;
  const int total_cta_blocks = (my_token_end - my_token_start) * blocks_per_token;
  const int max_iters = (total_cta_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_cta_blocks;

    int blk_in_token = 0, global_token = 0, global_line = 0;
    if (valid) {
      const int local_token = blk_local / blocks_per_token;
      blk_in_token = blk_local % blocks_per_token;
      global_token = local_token + my_token_start;
      global_line  = global_token * blocks_per_token * 4 + blk_in_token * 4 + lane_in_sub;
    }

    const uint4 v = ptr_in_bf16[valid ? global_line : 0];

    // K-loop: 1 iteration per routed expert (vs total_experts in base kernel).
    // All 4 lanes in a sub-warp read the same (token, k) entry — coalesced.
    for (int k = 0; k < topk_max; k++) {
      int expert = -1, slot = -1;
      if (valid) {
        expert = topk_expert[global_token * topk_max + k];
        slot   = topk_slot  [global_token * topk_max + k];
      }
      // -1 means "no more routes for this token" (topk_expert sorted ascending).
      // Different lanes in the same sub-warp share global_token so they hit
      // the same `-1` together — but neighbouring sub-warps may operate on
      // *different* tokens with different K counts. Use `continue` rather
      // than `break` so a sub-warp that's done doesn't desync the others.
      if (expert < 0) continue;
      const int dest_rank = expert / experts_per_rank;
      uint4* dst_data = dest_ptr[dest_rank] + lineoffset_out
                      + slot * bf16_lines_per_token + blk_in_token * 4;
      dst_data[lane_in_sub] = v;
    }
  }

  __syncthreads();

  // Same SM-sync + cross-rank barrier as the base kernel.
  __shared__ int lastSM_shmem;
  if (threadIdx.x == 0) {
    __threadfence();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val_sm_sync = atomicAdd(uc_flagptr + UBX_FLAG_A2AV_SM_SYNC, value_to_add);
    lastSM_shmem =
        (gridDim.x == 1 || old_val_sm_sync + value_to_add == reduce_id * UBX_MAX_SMS);
  }
  __syncthreads();
  if (!lastSM_shmem) return;

  if (threadIdx.x == 0) {
    uc_flagptr[UBX_FLAG_A2AV_ID] = reduce_id;
    cudaTriggerProgrammaticLaunchCompletion();
    __threadfence_system();
  }
  __syncthreads();

  if (mc_flagptr != nullptr) {
    if (threadIdx.x == 0) {
      ATOMIC_MCINC(mc_flagptr + UBX_FLAG_A2AV_BAR);
    }
  } else {
    if (threadIdx.x < (unsigned)RANKS) {
      int* peer_flag = reinterpret_cast<int*>(
          ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
      ATOMIC_UCINC(peer_flag + UBX_FLAG_A2AV_BAR);
    }
  }

  if (threadIdx.x != 0) return;
  if (!do_sync) return;

  volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_A2AV_BAR]);
  const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("A2AV_TOPK BAR:SM %d [%d]:expecting %d got %d\n",
             blockIdx.x, threadIdx.x, expected, *flag);
      break;
    }
#endif
  }
}  // bf16→bf16 token-dispatch (top-K LUT variant)

// Persistent (chunked) variant: one kernel launch runs N chunks, each with its
// own cross-rank barrier. Avoids the N-launch overhead of calling the
// non-persistent kernel N times.
//
// Chunking is by LOCAL experts per rank: chunk c covers local experts
// [c*nexperts_per_chunk, min((c+1)*nexperts_per_chunk, experts_per_rank)) on
// every rank. Global expert subset per chunk is non-contiguous (spans all
// ranks); the inner loop filters by `expert % experts_per_rank`.
//
// Per-chunk synchronization:
//   (1) All SMs perform remote writes for this chunk's experts.
//   (2) __threadfence_system ensures NVLink writes are globally visible.
//   (3) Each SM's thread 0 atomicAdds to a per-chunk scratch slot; the SM
//       whose addition hits UBX_MAX_SMS is "last" and issues ATOMIC_MCINC
//       on BAR + bumps A2AV_ID.  Non-last SMs proceed to next chunk with
//       zero wait.
//   (4) The last-SM path serializes MCINC in chunk order: last-SM of chunk c
//       waits until A2AV_ID has reached (reduce_id_start + c) before
//       issuing its own MCINC.  This keeps BAR increments ordered so the
//       receiver's per-chunk a2av_wait sees correct chunk-granular arrivals.
//
// Quantize is re-done per chunk (the inner token/block loop runs once per
// chunk, filtered by expert subset). This is the main cost we pay vs the
// monolithic kernel; in exchange we get per-chunk barriers with essentially
// no launch overhead.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_a2av_token_bf16_mxfp8_persistent(
        const int RANKS,
        const int myrank,
        const int ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int* __restrict__ token_offsets,
        const int64_t lineoffset_out,
        const int64_t lineoffset_scales,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        const int nchunks,
        const int nexperts_per_chunk,
        ncclWindow_t window) {
  // uc_flagptr / mc_flagptr are pre-resolved host-side by the launcher.
  // dest_ptr[r] (peer pool base) is still resolved device-side via
  // ncclGetLsaPointer — that's a per-peer call.

  const int bfloat_lines_per_block = 4;
  const int fp8_lines_per_block    = 2;
  const int total_experts          = RANKS * experts_per_rank;

  __shared__ uint4*  dest_ptr[MAXRANKS];
  __shared__ uint8_t fp8_pack[UBX_MAXTHREADS / 32][32];
  __shared__ int     reduce_id_start_shmem;

  const int warp_id  = threadIdx.x >> 5;
  const int lane     = threadIdx.x & 31;
  const int num_warps = blockDim.x >> 5;

  const int tokens_per_cta = (ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, ntokens);

  if (threadIdx.x < RANKS)
    // Was: dest_ptr[threadIdx.x] = (uint4*)commbuff[threadIdx.x];
    dest_ptr[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id_start_shmem = uc_flagptr[UBX_FLAG_A2AV_ID];
  }
  __syncthreads();
  const int reduce_id_start = reduce_id_start_shmem;

  const int total_cta_blocks = (my_token_end - my_token_start) * blocks_per_token;

  for (int chunk = 0; chunk < nchunks; chunk++) {
    const int exp_start_local = chunk * nexperts_per_chunk;
    const int exp_end_local   = min(exp_start_local + nexperts_per_chunk, experts_per_rank);

    // ---- Chunk write phase (quantize + remote stores for this chunk's experts) ----
    for (int blk_local = warp_id; blk_local < total_cta_blocks; blk_local += num_warps) {
      const int local_token  = blk_local / blocks_per_token;
      const int blk_in_token = blk_local % blocks_per_token;
      const int global_token = local_token + my_token_start;

      const int line_in_blk  = lane >> 3;
      const int elem_in_line = lane & 7;
      const int global_line  = global_token * blocks_per_token * bfloat_lines_per_block
                             + blk_in_token * bfloat_lines_per_block + line_in_blk;

      const uint4 v       = ptr_in_bf16[global_line];
      const xhalf my_elem = reinterpret_cast<const xhalf*>(&v)[elem_in_line];

      float my_abs = fabsf(__bfloat162float(my_elem));
#pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1)
        my_abs = fmaxf(my_abs, __shfl_xor_sync(0xFFFFFFFF, my_abs, delta));
      const float amax = my_abs;

      const float log2_scale = (amax > 0.0f) ? ceilf(log2f(amax / 448.0f)) : -127.0f;
      const int   e8m0       = max(0, min(255, (int)log2_scale + 127));
      const float scale_f    = exp2f((float)(e8m0 - 127));

      const float quant_f   = (scale_f > 0.0f) ? __bfloat162float(my_elem) / scale_f : 0.0f;
      const __nv_fp8_e4m3 fp8_val = __nv_fp8_e4m3(quant_f);
      fp8_pack[warp_id][lane] = *reinterpret_cast<const uint8_t*>(&fp8_val);
      __syncwarp();

      const uint4 fp8_line0 = *reinterpret_cast<const uint4*>(&fp8_pack[warp_id][0]);
      const uint4 fp8_line1 = *reinterpret_cast<const uint4*>(&fp8_pack[warp_id][16]);

      // Filter expert iteration by chunk's local expert range.
      for (int expert_base = 0; expert_base < total_experts; expert_base += 32) {
        const int expert = expert_base + lane;
        int slot = -1;
        if (expert < total_experts) {
          const int local_e = expert - (expert / experts_per_rank) * experts_per_rank;
          if (local_e >= exp_start_local && local_e < exp_end_local)
            slot = token_offsets[global_token * total_experts + expert];
        }

        if (slot >= 0) {
          const int dest_rank           = expert / experts_per_rank;
          const int fp8_lines_per_token = blocks_per_token * fp8_lines_per_block;

          uint4* dst_data = dest_ptr[dest_rank] + lineoffset_out
                          + slot * fp8_lines_per_token + blk_in_token * fp8_lines_per_block;
          dst_data[0] = fp8_line0;
          dst_data[1] = fp8_line1;

          uint8_t* dst_scale = reinterpret_cast<uint8_t*>(dest_ptr[dest_rank])
                             + lineoffset_scales * 16
                             + slot * blocks_per_token + blk_in_token;
          *dst_scale = (uint8_t)e8m0;
        }
      }
    }

    // ---- Per-chunk cross-rank barrier ----
    __syncthreads();
    // Compute lastSM into shared mem so RANKS threads (needed for UC barrier
    // fanout when mc_flagptr == nullptr) can branch on it.
    __shared__ int lastSM_shmem;
    if (threadIdx.x == 0) {
      __threadfence();
      const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
      const int sync_slot    = UBX_FLAG_A2AV_PERSISTENT_SM_SYNC_BASE + chunk;
      const int old_val      = atomicAdd(uc_flagptr + sync_slot, value_to_add);
      lastSM_shmem           = (old_val + value_to_add == UBX_MAX_SMS) ? 1 : 0;
    }
    __syncthreads();

    if (lastSM_shmem) {
      const int target_id = reduce_id_start + chunk + 1;
      if (threadIdx.x == 0) {
        // Serialize MCINC/UCINC fanout across chunks: wait until the previous
        // chunk's last-SM has bumped A2AV_ID to (target_id - 1) before signalling.
        if (chunk > 0) {
          volatile int* id_ptr  = (volatile int*)&uc_flagptr[UBX_FLAG_A2AV_ID];
          const int prev_id     = target_id - 1;
#ifdef UB_TIMEOUT_ENABLED
          clock_t s = clock64();
#endif
          while (*id_ptr < prev_id) {
#ifdef UB_TIMEOUT_ENABLED
            if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
              printf("A2AV_PERSISTENT ORDER:SM %d chunk %d waiting for id %d got %d\n",
                     blockIdx.x, chunk, prev_id, *id_ptr);
              break;
            }
#endif
          }
        }
        __threadfence_system();
      }
      __syncthreads();

      if (mc_flagptr != nullptr) {
        if (threadIdx.x == 0) {
          ATOMIC_MCINC(mc_flagptr + UBX_FLAG_A2AV_BAR);
        }
      } else {
        // UC barrier (NVLS-disabled): threads 0..RANKS-1 each do
        // one unicast atomic-inc on a peer's BAR flag.
        const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
        if (threadIdx.x < (unsigned)RANKS) {
          int* peer_flag = reinterpret_cast<int*>(
              ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
          ATOMIC_UCINC(peer_flag + UBX_FLAG_A2AV_BAR);
        }
      }

      if (threadIdx.x == 0) {
        uc_flagptr[UBX_FLAG_A2AV_ID] = target_id;
      }
    }
    // Non-last SMs (and last SM after signalling) fall through to next chunk
    // with no grid-level wait. __syncthreads below just keeps threads in a
    // block in lockstep for the next iteration of the outer loop.
    __syncthreads();
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // persistent chunked token-dispatch bf16→mxfp8 kernel

// Wait kernel: polls A2AV barrier flag until all ranks have completed dispatch.
// Launched with 1 block, 1 thread. Pairs with a2av_token_bf16_mxfp8(sync=false).
__global__ void ubx_kernel_a2av_wait(
    const int RANKS,
    ncclWindow_t window) {
  // Phase 4: derive the local-VA flag pointer from the registered window.
  const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(
      ncclGetLocalPointer(window, REG0_FLAG_OFFSET));

  #ifndef UBX_SKIP_GRID_DEP_SYNC
  cudaGridDependencySynchronize();
  #endif
  const int reduce_id = uc_flagptr[UBX_FLAG_A2AV_ID];
  volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_A2AV_BAR]);
  const int expected = reduce_id * RANKS;

#ifdef UB_TIMEOUT_ENABLED
  clock_t s = clock64();
#endif
  while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
    if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
      printf("A2AV_WAIT BAR:[%d]:expecting %d got %d\n",
             threadIdx.x, expected, *flag);
      break;
    }
#endif
  }
  cudaTriggerProgrammaticLaunchCompletion();
}  // a2av wait kernel

// ============================================================================
// MoE token-combine kernels (reverse of a2av_token dispatch).
//
// Two flavors share a two-phase PULL design:
//   Phase 1 (local stage): each rank reads its bf16 expert outputs from
//     ptr_in_bf16 ([n_recv, hidden]) and writes them into the temp symm
//     buffer at lineoffset_temp using the SAME slot indexing as a dispatch
//     output. bf16-wire = bulk memcpy; mxfp8-wire = quantize per 32-elem
//     block to fp8 + E8M0 scale (lifted from ubx_kernel_a2av_token_bf16_mxfp8).
//   Cross-rank barrier on UBX_FLAG_COMBINE_BAR (last-SM ATOMIC_MCINC; every
//     block spin-waits until BAR == reduce_id * RANKS).
//   Phase 2 (remote PULL + reduce + write): each rank iterates its
//     local_ntokens local tokens, walks the K experts via its own
//     token_offsets[t, e] row (dest_rank = e/experts_per_rank, slot = the
//     entry), PULLs from peer's temp symm, dequantizes (mxfp8 only),
//     multiplies by gate_weights[t, e] if non-NULL, fp32-accumulates,
//     downcasts to bf16, writes the local output.
//
// PULL gives single-writer per-output-element (no atomics) and reuses
// compute_token_offsets unchanged — no inverse map needed.
//
// Sync vs async:
//   sync=1 (default): main kernel does Phase 1 + barrier-wait + Phase 2 + write.
//                     Returns with output ready; combine_wait is a no-op.
//   sync=0:           main kernel does ONLY Phase 1 (writes temp symm, signals
//                     COMBINE_BAR, returns). Caller must invoke combine_wait()
//                     afterwards — which polls BAR and runs Phase 2 itself
//                     (read remote, sum, write output). Lets the caller
//                     overlap host or device work between Phase 1 and Phase 2.
// ============================================================================

// Phase 2 helpers: shared between the main combine kernels (sync=1 path) and
// the combine_wait kernels (sync=0 path). Each iterates local tokens × blocks,
// walks the K experts, PULLs from peer temp symm, applies gate weights, and
// writes the bf16 output. The mxfp8 variant additionally dequantizes via the
// per-block E8M0 scale at lineoffset_scales.
__device__ __forceinline__ void combine_phase2_bf16(
    const int RANKS,
    const int local_ntokens,
    const int blocks_per_token,
    const int experts_per_rank,
    const int* __restrict__ token_offsets,
    const float* __restrict__ gate_weights,
    const int64_t lineoffset_temp,
    uint4* const* __restrict__ peer_temp,
    uint4* __restrict__ ptr_out_bf16) {
  const int total_experts        = RANKS * experts_per_rank;
  const int bf16_lines_per_token = blocks_per_token * 4;

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sw_base_lane  = lane & ~3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int tokens_per_cta = (local_ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, local_ntokens);
  const int total_blocks   = (my_token_end - my_token_start) * blocks_per_token;
  const int max_iters      = (total_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_blocks;

    int local_token = 0, blk_in_token = 0, out_line = 0;
    if (valid) {
      const int local_t = blk_local / blocks_per_token;
      blk_in_token      = blk_local % blocks_per_token;
      local_token       = local_t + my_token_start;
      out_line          = local_token * bf16_lines_per_token
                          + blk_in_token * 4 + lane_in_sub;
    }

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int expert_base = 0; expert_base < total_experts; expert_base += 4) {
      int   slot   = -1;
      int   expert = 0;
      float gate_w = 1.0f;
      if (valid) {
        expert = expert_base + lane_in_sub;
        if (expert < total_experts) {
          slot = token_offsets[local_token * total_experts + expert];
          if (gate_weights != nullptr && slot >= 0)
            gate_w = gate_weights[local_token * total_experts + expert];
        }
      }

#pragma unroll
      for (int t = 0; t < 4; t++) {
        const int   peer_slot   = __shfl_sync(0xFFFFFFFFU, slot,   sw_base_lane | t);
        const int   peer_expert = __shfl_sync(0xFFFFFFFFU, expert, sw_base_lane | t);
        const float peer_gate   = __shfl_sync(0xFFFFFFFFU, gate_w, sw_base_lane | t);
        if (valid && peer_slot >= 0) {
          const int dest_rank = peer_expert / experts_per_rank;
          const uint4* src = peer_temp[dest_rank] + lineoffset_temp
                           + peer_slot * bf16_lines_per_token + blk_in_token * 4;
          const uint4 v = src[lane_in_sub];
          const xhalf* vh = reinterpret_cast<const xhalf*>(&v);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            acc[i] += __bfloat162float(vh[i]) * peer_gate;
          }
        }
      }
    }

    if (valid) {
      uint4 out_v;
      xhalf* outh = reinterpret_cast<xhalf*>(&out_v);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        outh[i] = __float2bfloat16(acc[i]);
      }
      ptr_out_bf16[out_line] = out_v;
    }
  }
}

__device__ __forceinline__ void combine_phase2_mxfp8(
    const int RANKS,
    const int local_ntokens,
    const int blocks_per_token,
    const int experts_per_rank,
    const int* __restrict__ token_offsets,
    const float* __restrict__ gate_weights,
    const int64_t lineoffset_temp,
    const int64_t lineoffset_scales,
    uint4* const* __restrict__ peer_temp,
    uint4* __restrict__ ptr_out_bf16) {
  const int total_experts        = RANKS * experts_per_rank;
  const int bf16_lines_per_token = blocks_per_token * 4;
  const int fp8_lines_per_block  = 2;
  const int fp8_lines_per_token  = blocks_per_token * fp8_lines_per_block;

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sw_base_lane  = lane & ~3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int tokens_per_cta = (local_ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, local_ntokens);
  const int total_blocks   = (my_token_end - my_token_start) * blocks_per_token;
  const int max_iters      = (total_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_blocks;

    int local_token = 0, blk_in_token = 0, out_line = 0;
    if (valid) {
      const int local_t = blk_local / blocks_per_token;
      blk_in_token      = blk_local % blocks_per_token;
      local_token       = local_t + my_token_start;
      out_line          = local_token * bf16_lines_per_token
                          + blk_in_token * 4 + lane_in_sub;
    }

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int expert_base = 0; expert_base < total_experts; expert_base += 4) {
      int   slot   = -1;
      int   expert = 0;
      float gate_w = 1.0f;
      if (valid) {
        expert = expert_base + lane_in_sub;
        if (expert < total_experts) {
          slot = token_offsets[local_token * total_experts + expert];
          if (gate_weights != nullptr && slot >= 0)
            gate_w = gate_weights[local_token * total_experts + expert];
        }
      }

#pragma unroll
      for (int t = 0; t < 4; t++) {
        const int   peer_slot   = __shfl_sync(0xFFFFFFFFU, slot,   sw_base_lane | t);
        const int   peer_expert = __shfl_sync(0xFFFFFFFFU, expert, sw_base_lane | t);
        const float peer_gate   = __shfl_sync(0xFFFFFFFFU, gate_w, sw_base_lane | t);
        if (valid && peer_slot >= 0) {
          const int dest_rank = peer_expert / experts_per_rank;
          const uint64_t* fp8_src = reinterpret_cast<const uint64_t*>(
              peer_temp[dest_rank] + lineoffset_temp
              + peer_slot * fp8_lines_per_token + blk_in_token * fp8_lines_per_block);
          const uint64_t fp8_packed = fp8_src[lane_in_sub];

          const uint8_t* scale_src = reinterpret_cast<const uint8_t*>(
              peer_temp[dest_rank]) + lineoffset_scales * 16
              + peer_slot * blocks_per_token + blk_in_token;
          const uint8_t e8m0       = *scale_src;
          const float   scale_f    = exp2f((float)((int)e8m0 - 127));
          const float   block_gate = scale_f * peer_gate;

#pragma unroll
          for (int i = 0; i < 8; i++) {
            __nv_fp8_e4m3 fp8_v;
            fp8_v.__x = static_cast<__nv_fp8_storage_t>((fp8_packed >> (i * 8)) & 0xFFu);
            acc[i] += static_cast<float>(fp8_v) * block_gate;
          }
        }
      }
    }

    if (valid) {
      uint4 out_v;
      xhalf* outh = reinterpret_cast<xhalf*>(&out_v);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        outh[i] = __float2bfloat16(acc[i]);
      }
      ptr_out_bf16[out_line] = out_v;
    }
  }
}

// Combine kernel, bf16 wire. ptr_in_bf16 = [n_recv, hidden] bf16 LOCAL
// (caller's regular tensor). ptr_out_bf16 = [local_ntokens, hidden] bf16
// LOCAL. Temp symm buffer at lineoffset_temp on every rank; first dim
// max_tokens_per_rank.
//
// Threading: same v2 4-thread sub-warp layout as dispatch — 4 threads
// per 32-elem block, one uint4 per thread, expert iteration with
// __shfl_sync over expert_base += 4.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_bf16_bf16(
        const int RANKS,
        const int myrank,
        const int local_ntokens,
        const int n_recv,
        const int blocks_per_token,
        const int experts_per_rank,
        const int max_tokens_per_rank,
        const int* __restrict__ token_offsets,
        const float* __restrict__ gate_weights,  // may be NULL
        const int64_t lineoffset_temp,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        uint4* __restrict__ ptr_out_bf16,
        const int do_sync,
        ncclWindow_t window) {
  const int bf16_lines_per_token = blocks_per_token * 4;

  __shared__ uint4* peer_temp[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  if (threadIdx.x < RANKS)
    peer_temp[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_COMBINE_ID] + 1;
  }
  __syncthreads();

  // ---- Phase 1: copy local bf16 expert outputs into temp symm buffer ----
  // Per-CTA range over [0, n_recv).
  {
    const int slots_per_cta = (n_recv + gridDim.x - 1) / gridDim.x;
    const int my_slot_start = slots_per_cta * blockIdx.x;
    const int my_slot_end   = min(my_slot_start + slots_per_cta, n_recv);
    const int total_blocks  = (my_slot_end - my_slot_start) * blocks_per_token;
    const int max_iters     = (total_blocks + num_sub_warps - 1) / num_sub_warps;

    uint4* my_temp = peer_temp[myrank] + lineoffset_temp;

    for (int iter = 0; iter < max_iters; iter++) {
      const int blk_local = iter * num_sub_warps + sub_warp_id;
      const bool valid    = blk_local < total_blocks;

      int local_line = 0, dst_line = 0;
      if (valid) {
        const int slot_local   = blk_local / blocks_per_token;
        const int blk_in_token = blk_local % blocks_per_token;
        const int slot         = slot_local + my_slot_start;
        local_line = slot * bf16_lines_per_token + blk_in_token * 4 + lane_in_sub;
        dst_line   = local_line;  // same indexing — we use the same stride in temp
      }
      const uint4 v = ptr_in_bf16[valid ? local_line : 0];
      if (valid) my_temp[dst_line] = v;
    }
  }

  // ---- Phase 1 → Phase 2 barrier: signal own completion ----
  // Cross-rank barrier path:
  //   MC (multicast):    last-SM thread 0 issues one multimem atomic-inc
  //   UC (NVLS-disabled): last-SM RANKS threads each unicast atomic-inc a peer
  __syncthreads();
  __shared__ int lastSM_shmem;
  if (threadIdx.x == 0) {
    __threadfence_system();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val = atomicAdd(uc_flagptr + UBX_FLAG_COMBINE_SM_SYNC, value_to_add);
    lastSM_shmem =
        (gridDim.x == 1 || old_val + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM_shmem) {
      uc_flagptr[UBX_FLAG_COMBINE_ID] = reduce_id;
    }
  }
  __syncthreads();

  if (lastSM_shmem) {
    if (mc_flagptr != nullptr) {
      if (threadIdx.x == 0) {
        ATOMIC_MCINC(mc_flagptr + UBX_FLAG_COMBINE_BAR);
      }
    } else {
      // UC fallback: RANKS threads each increment one peer's BAR flag.
      if (threadIdx.x < (unsigned)RANKS) {
        const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
        int* peer_flag = reinterpret_cast<int*>(
            ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
        ATOMIC_UCINC(peer_flag + UBX_FLAG_COMBINE_BAR);
      }
    }
  }

  // Async path (do_sync==0): caller will invoke combine_wait_bf16 to do the
  // spin-wait + Phase 2 itself. We exit cleanly here so the caller can
  // overlap host or device work between Phase 1 and Phase 2.
  if (!do_sync) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
    return;
  }

  // Sync path: spin until BAR == reduce_id * RANKS, then Phase 2.
  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("COMBINE_BF16 BAR:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  combine_phase2_bf16(
      RANKS, local_ntokens, blocks_per_token, experts_per_rank,
      token_offsets, gate_weights, lineoffset_temp,
      peer_temp, ptr_out_bf16);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // combine bf16-wire kernel

// ----------------------------------------------------------------------------
// E22: true 2-kernel combine (bf16 wire).
//
// Phase 1 (ubx_kernel_combine_v2_phase1_bf16): pure data-copy kernel.
//   Each CTA copies its slice of `expert_outputs` into the local symm temp
//   buffer. NO atomicAdd, NO threadfence, NO peer atomics, NO barrier.
//   Kernel exit + CUDA stream ordering is the only synchronization.
//
// Phase 2 (ubx_kernel_combine_v2_phase2_bf16): cross-rank barrier + Phase 2.
//   PDL on entry: thread 0 of CTA 0 calls cudaGridDependencySynchronize,
//   then issues `RANKS` ATOMIC_UCINCs (one per peer) to UBX_FLAG_COMBINE2_BAR.
//   ALL CTAs (thread 0 each) spin polling BAR until >= reduce_id * RANKS.
//   Then __syncthreads + combine_phase2_bf16 (peer-PULL + fp32 sum).
//
// `reduce_id` is host-tracked (a per-rank counter incremented per call) and
// passed as a kernel argument. No GPU-side ID flag is needed; the kernels
// don't depend on any prior call's GPU state for ID.
//
// Why this avoids the EP=32 SM-hold deadlock that v1 had:
//   v1 sync=1 has every CTA do atomicAdd on SM_SYNC, then non-last CTAs SPIN
//   on BAR holding their SMs. If gridDim.x CTAs don't all schedule together,
//   the held SMs prevent the rest from scheduling → no rank ever signals →
//   deadlock.
//   v2 Phase 1 has zero spin, every CTA returns after its copy → SMs always
//   free up. Phase 2's spin only depends on PEER atomics, not on local
//   gridDim co-residence (CTA 0 alone signals; if it schedules first, peers
//   make progress; the rest of this rank's CTAs can come and go in any order
//   because they don't gate each other).
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_v2_phase1_bf16(
        const int n_recv,
        const int blocks_per_token,
        const uint4* __restrict__ ptr_in_bf16,
        uint4* __restrict__ my_temp) {
  const int bf16_lines_per_token = blocks_per_token * 4;
  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int slots_per_cta = (n_recv + gridDim.x - 1) / gridDim.x;
  const int my_slot_start = slots_per_cta * blockIdx.x;
  const int my_slot_end   = min(my_slot_start + slots_per_cta, n_recv);
  const int total_blocks  = (my_slot_end - my_slot_start) * blocks_per_token;
  const int max_iters     = (total_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_blocks;
    int line = 0;
    if (valid) {
      const int slot_local   = blk_local / blocks_per_token;
      const int blk_in_token = blk_local % blocks_per_token;
      const int slot         = slot_local + my_slot_start;
      line = slot * bf16_lines_per_token + blk_in_token * 4 + lane_in_sub;
    }
    const uint4 v = ptr_in_bf16[valid ? line : 0];
    if (valid) my_temp[line] = v;
  }
  // PDL signal so Phase 2 can start without waiting full kernel-completion.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_v2_phase2_bf16(
        const int RANKS,
        const int reduce_id,
        const int local_ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int* __restrict__ token_offsets,
        const float* __restrict__ gate_weights,
        const int64_t lineoffset_temp,
        int* uc_flagptr,
        const ncclWindow_t window,
        uint4* __restrict__ ptr_out_bf16) {
  __shared__ uint4* peer_temp[MAXRANKS];
  if (threadIdx.x < RANKS) {
    peer_temp[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));
  }
  // Wait for Phase 1 (own data copy) to finish on THIS rank.
#ifndef UBX_SKIP_GRID_DEP_SYNC
  if (threadIdx.x == 0) cudaGridDependencySynchronize();
#endif
  __syncthreads();

  // CTA 0 alone issues the cross-rank barrier signal: RANKS threads each
  // ATOMIC_UCINC one peer's UBX_FLAG_COMBINE2_BAR. (UC fallback only — MC
  // path is unused for v2 to keep the protocol minimal.)
  if (blockIdx.x == 0 && threadIdx.x < (unsigned)RANKS) {
    const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
    int* peer_flag = reinterpret_cast<int*>(
        ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
    ATOMIC_UCINC(peer_flag + UBX_FLAG_COMBINE2_BAR);
  }

  // Spin: every CTA's thread 0 polls own BAR until it reaches reduce_id*RANKS.
  // This depends ONLY on peer ranks' CTA-0 signals — local CTA scheduling is
  // irrelevant. Even if only K of gridDim.x CTAs schedule first, they'll
  // observe BAR transition independently and proceed.
  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE2_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("COMBINE_V2 BAR:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  combine_phase2_bf16(
      RANKS, local_ntokens, blocks_per_token, experts_per_rank,
      token_offsets, gate_weights, lineoffset_temp,
      peer_temp, ptr_out_bf16);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// Combine kernel, mxfp8 wire. Phase 1 quantizes bf16 → fp8 + E8M0 (same
// packing as ubx_kernel_a2av_token_bf16_mxfp8). Phase 2 PULLs fp8 + scale
// from peers, dequantizes, gate-mul, fp32-accumulates, downcasts to bf16.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_mxfp8_bf16(
        const int RANKS,
        const int myrank,
        const int local_ntokens,
        const int n_recv,
        const int blocks_per_token,
        const int experts_per_rank,
        const int max_tokens_per_rank,
        const int* __restrict__ token_offsets,
        const float* __restrict__ gate_weights,  // may be NULL
        const int64_t lineoffset_temp,
        const int64_t lineoffset_scales,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        uint4* __restrict__ ptr_out_bf16,
        const int do_sync,
        ncclWindow_t window) {
  const int bf16_lines_per_token = blocks_per_token * 4;
  const int fp8_lines_per_block  = 2;
  const int fp8_lines_per_token  = blocks_per_token * fp8_lines_per_block;

  __shared__ uint4* peer_temp[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  if (threadIdx.x < RANKS)
    peer_temp[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_COMBINE_ID] + 1;
  }
  __syncthreads();

  // ---- Phase 1: bf16 → fp8 + E8M0, write to local temp symm ----
  // Same quantization math as ubx_kernel_a2av_token_bf16_mxfp8 (lines 998-
  // 1027); the destination is THIS rank's own temp symm rather than a remote
  // peer's pool.
  {
    const int slots_per_cta = (n_recv + gridDim.x - 1) / gridDim.x;
    const int my_slot_start = slots_per_cta * blockIdx.x;
    const int my_slot_end   = min(my_slot_start + slots_per_cta, n_recv);
    const int total_blocks  = (my_slot_end - my_slot_start) * blocks_per_token;
    const int max_iters     = (total_blocks + num_sub_warps - 1) / num_sub_warps;

    uint4*   my_temp_data  = peer_temp[myrank] + lineoffset_temp;
    uint8_t* my_temp_scale = reinterpret_cast<uint8_t*>(peer_temp[myrank])
                             + lineoffset_scales * 16;

    for (int iter = 0; iter < max_iters; iter++) {
      const int blk_local = iter * num_sub_warps + sub_warp_id;
      const bool valid    = blk_local < total_blocks;

      int blk_in_token = 0, slot = 0, src_line = 0;
      if (valid) {
        const int slot_local = blk_local / blocks_per_token;
        blk_in_token = blk_local % blocks_per_token;
        slot         = slot_local + my_slot_start;
        src_line     = slot * bf16_lines_per_token + blk_in_token * 4 + lane_in_sub;
      }
      const uint4 v = ptr_in_bf16[valid ? src_line : 0];
      const xhalf* my_elems = reinterpret_cast<const xhalf*>(&v);

      // Intra-thread amax over 8 elements.
      float my_max = 0.0f;
#pragma unroll
      for (int e = 0; e < 8; e++)
        my_max = fmaxf(my_max, fabsf(__bfloat162float(my_elems[e])));

      // 4-thread sub-warp amax reduction (2 shfl_xor steps).
      my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFFU, my_max, 1));
      my_max = fmaxf(my_max, __shfl_xor_sync(0xFFFFFFFFU, my_max, 2));
      const float amax = my_max;

      const float log2_scale = (amax > 0.0f) ? ceilf(log2f(amax / 448.0f)) : -127.0f;
      const int   e8m0       = max(0, min(255, (int)log2_scale + 127));
      const float scale_f    = exp2f((float)(e8m0 - 127));

      uint64_t fp8_packed = 0;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        const float fv = (scale_f > 0.0f) ? __bfloat162float(my_elems[e]) / scale_f : 0.0f;
        const __nv_fp8_e4m3 fp8_v = __nv_fp8_e4m3(fv);
        const uint8_t b = *reinterpret_cast<const uint8_t*>(&fp8_v);
        fp8_packed |= (static_cast<uint64_t>(b)) << (e * 8);
      }

      if (valid) {
        uint64_t* fp8_dst = reinterpret_cast<uint64_t*>(
            my_temp_data + slot * fp8_lines_per_token
                          + blk_in_token * fp8_lines_per_block);
        fp8_dst[lane_in_sub] = fp8_packed;
        if (lane_in_sub == 0) {
          my_temp_scale[slot * blocks_per_token + blk_in_token] =
              static_cast<uint8_t>(e8m0);
        }
      }
    }
  }

  // ---- Phase 1 → Phase 2 barrier: signal own completion via last-SM MCINC ----
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val = atomicAdd(uc_flagptr + UBX_FLAG_COMBINE_SM_SYNC, value_to_add);
    const int lastSM =
        (gridDim.x == 1 || old_val + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM) {
      ATOMIC_MCINC(mc_flagptr + UBX_FLAG_COMBINE_BAR);
      uc_flagptr[UBX_FLAG_COMBINE_ID] = reduce_id;
    }
  }

  // Async path: caller invokes combine_wait_mxfp8 to do Phase 2.
  if (!do_sync) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      cudaTriggerProgrammaticLaunchCompletion();
    }
    return;
  }

  // Sync path: spin until BAR == reduce_id * RANKS, then Phase 2.
  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("COMBINE_MXFP8 BAR:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  combine_phase2_mxfp8(
      RANKS, local_ntokens, blocks_per_token, experts_per_rank,
      token_offsets, gate_weights, lineoffset_temp, lineoffset_scales,
      peer_temp, ptr_out_bf16);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // combine mxfp8-wire kernel

// Combine wait kernels (bf16 / mxfp8 wire). Pair with combine_*(sync=0):
// the main combine kernel did Phase 1 + signal, then exited. These kernels
// poll UBX_FLAG_COMBINE_BAR and run Phase 2 (read peer temp symm, dequant
// for mxfp8, gate-mul, fp32 accumulate, downcast bf16, write local output).
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_wait_bf16(
        const int RANKS,
        const int local_ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int* __restrict__ token_offsets,
        const float* __restrict__ gate_weights,
        const int64_t lineoffset_temp,
        int* uc_flagptr,
        uint4* __restrict__ ptr_out_bf16,
        ncclWindow_t window) {
  __shared__ uint4* peer_temp[MAXRANKS];
  if (threadIdx.x < RANKS)
    peer_temp[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_COMBINE_ID];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("COMBINE_WAIT_BF16 BAR:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  combine_phase2_bf16(
      RANKS, local_ntokens, blocks_per_token, experts_per_rank,
      token_offsets, gate_weights, lineoffset_temp,
      peer_temp, ptr_out_bf16);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_wait_mxfp8(
        const int RANKS,
        const int local_ntokens,
        const int blocks_per_token,
        const int experts_per_rank,
        const int* __restrict__ token_offsets,
        const float* __restrict__ gate_weights,
        const int64_t lineoffset_temp,
        const int64_t lineoffset_scales,
        int* uc_flagptr,
        uint4* __restrict__ ptr_out_bf16,
        ncclWindow_t window) {
  __shared__ uint4* peer_temp[MAXRANKS];
  if (threadIdx.x < RANKS)
    peer_temp[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_COMBINE_ID];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("COMBINE_WAIT_MXFP8 BAR:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  combine_phase2_mxfp8(
      RANKS, local_ntokens, blocks_per_token, experts_per_rank,
      token_offsets, gate_weights, lineoffset_temp, lineoffset_scales,
      peer_temp, ptr_out_bf16);

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}

// ============================================================================
// PUSH-semantics Lamport combine (bf16 wire only).
//
// Mirrors alltoall_lamport's race-free pattern: each rank PUSHES its data
// to peer's destination buf; readers poll their OWN destination buf. With
// triple-buffer rotation, my call N+1's clear_ptr re-poisons MY OWN
// bufs[(N+2)%3] — never the buf any peer is currently polling. No
// cross-rank race like the PULL variant above has.
//
// Routing (inverse compared to dispatch):
//   For each slot s ∈ [0, max_tokens_per_rank) on this rank, the inverse
//   routing map (compute_combine_push_map) returns:
//     (origin_rank, origin_local_token, k_idx, valid)
//   so this rank pushes its bf16 expert output at slot s into peer
//   `origin_rank`'s combine destination buf at row
//   [origin_local_token, k_idx, *].
//
// Destination buf layout (per rank, per triple-buf slot):
//   [local_ntokens, topk_max, hidden] bf16
// = local_ntokens * topk_max * blocks_per_token * 4 uint4 lines.
//
// Phase 2 (poll OWN dest, dequant, gate-mul, sum, write output):
//   For each (local_token t, blk_in_token blk):
//     For each k ∈ [0, topk_max):
//       expert_id = topk_idx[t, k];           // -1 = unrouted (skip)
//       gate      = gate_weights[t, expert_id]; // 1.0 if NULL
//       v         = my_dest[t, k, blk, lane_in_sub];  // Lamport poll
//       acc[i] += __bfloat162float(v[i]) * gate;
//     downcast acc → bf16, write output[t, blk, lane_in_sub].
//
// Re-poison: clear_ptr is MY OWN bufs[(N+2)%3]. Phase 1 of THIS call
// re-poisons the FULL [local_ntokens × topk_max × blocks_per_token × 4]
// line range so by call N+2 (when this rank's combine kernel reads from
// MY bufs[(N+2)%3] = it's MY own out_idx for that future call) it's
// fully poisoned.  Same warmup-on-calls-0-1 pattern as PULL variant.
// ============================================================================
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_bf16_bf16_lamport_push(
        const int RANKS,
        const int myrank,
        const int local_ntokens,
        const int n_recv,
        const int blocks_per_token,
        const int topk_max,
        const int total_experts,
        const int* __restrict__ inverse_map_flat,  // [max_tokens_per_rank * 4] flat int32
        const int* __restrict__ topk_idx,        // [local_ntokens, topk_max]
        const float* __restrict__ gate_weights,  // [local_ntokens, total_experts] or NULL
        const int max_tokens_per_rank,
        const int64_t lineoffset_dest,
        int* uc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,   // [n_recv, hidden] my expert outputs
        uint4* __restrict__ ptr_out_bf16,        // [local_ntokens, hidden] output
        uint4* __restrict__ clear_ptr,           // pool-VA, my own bufs[(N+2)%3]
        const int do_warmup_barrier,
        ncclWindow_t window) {
  const int bf16_lines_per_token = blocks_per_token * 4;
  // Per-token destination stride in uint4 lines = topk_max * bf16_lines_per_token.
  const int dest_stride_per_token = topk_max * bf16_lines_per_token;
  // Total dest buf lines per rank = local_ntokens * dest_stride_per_token.
  const int total_dest_lines = local_ntokens * dest_stride_per_token;

  __shared__ uint4* peer_dest[MAXRANKS];
  __shared__ int    lastSM;
  __shared__ int    s_reduce_id;

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  if (threadIdx.x < RANKS)
    peer_dest[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  // ---- Warmup barrier (calls 0-1) ---- (mirrors PULL variant exactly)
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    if (do_warmup_barrier) {
      int reduce_id = uc_flagptr[UBX_FLAG_COMBINE_ID];
      const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
      const int old_val =
          atomicAdd(uc_flagptr + UBX_FLAG_COMBINE_SM_SYNC, value_to_add);
      reduce_id++;
      s_reduce_id = reduce_id;
      lastSM = (gridDim.x == 1 ||
                old_val + value_to_add == reduce_id * UBX_MAX_SMS);
      if (lastSM) uc_flagptr[UBX_FLAG_COMBINE_ID] = reduce_id;
    } else {
      cudaTriggerProgrammaticLaunchCompletion();
    }
  }
  __syncthreads();

  if (do_warmup_barrier) {
    if (threadIdx.x == 0 && lastSM) __threadfence_system();
    __syncthreads();
    if (threadIdx.x < (unsigned)RANKS && lastSM) {
      const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
      int* peer_flag = reinterpret_cast<int*>(
          ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
      ATOMIC_UCINC(peer_flag + UBX_FLAG_COMBINE_BAR);
    }
    if (threadIdx.x == 0) {
      volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE_BAR]);
      const int expected = s_reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
      clock_t s = clock64();
#endif
      while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
        if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
          printf("COMBINE_LAMPORT_PUSH WARMUP:SM %d:expecting %d got %d\n",
                 blockIdx.x, expected, *flag);
          break;
        }
#endif
      }
    }
    __syncthreads();
  }

  // ---- Phase 1: PUSH local expert outputs to peers + re-poison clear_ptr ----
  // Per sub-warp processes one (slot, blk_in_token) per iter.
  // 4 threads of sub-warp write 4 uint4 lines coalesced (= 64 B per dest).
  {
    const int slots_per_cta = (max_tokens_per_rank + gridDim.x - 1) / gridDim.x;
    const int my_slot_start = slots_per_cta * blockIdx.x;
    const int my_slot_end   = min(my_slot_start + slots_per_cta,
                                   max_tokens_per_rank);
    const int total_blocks  = (my_slot_end - my_slot_start) * blocks_per_token;
    const int max_iters     = (total_blocks + num_sub_warps - 1) / num_sub_warps;

    for (int iter = 0; iter < max_iters; iter++) {
      const int blk_local = iter * num_sub_warps + sub_warp_id;
      const bool valid    = blk_local < total_blocks;

      int slot = 0, blk_in_token = 0, src_line = 0;
      int orig_rank = 0, orig_token = 0, k_idx = 0, valid_slot = 0;
      if (valid) {
        const int slot_local = blk_local / blocks_per_token;
        blk_in_token = blk_local % blocks_per_token;
        slot         = slot_local + my_slot_start;
        src_line     = slot * bf16_lines_per_token + blk_in_token * 4 + lane_in_sub;

        // inverse_map_flat[slot*4+0..3] = (orig_rank, orig_token, k_idx, valid).
        const int* inv = inverse_map_flat + slot * 4;
        orig_rank  = inv[0];
        orig_token = inv[1];
        k_idx      = inv[2];
        valid_slot = inv[3];
      }

      // PUSH my data to peer's dest at (orig_token, k_idx, blk, lane_in_sub).
      if (valid && valid_slot && slot < n_recv) {
        const uint4 v = ptr_in_bf16[src_line];
        const int64_t dst_line = lineoffset_dest
                             + orig_token * dest_stride_per_token
                             + k_idx * bf16_lines_per_token
                             + blk_in_token * 4 + lane_in_sub;
        peer_dest[orig_rank][dst_line] = v;
      }
    }
  }

  // ---- Re-poison clear_ptr (MY OWN bufs[(N+2)%3]) over its full extent ----
  // Iterate every line of the dest buf and write .w = poison.  Distribute
  // across all CTAs.  This runs in parallel with peers' Phase 1 pushes —
  // since clear_ptr is MY OWN buf and peers don't write here, no race.
  {
    for (int line = threadIdx.x + blockDim.x * blockIdx.x;
         line < total_dest_lines;
         line += blockDim.x * gridDim.x) {
      if (clear_ptr) clear_ptr[line].w = UBX_LAMPORT_INT;
    }
  }
  __syncthreads();

  // ---- Phase 2: poll OWN dest at (t, k), dequant gate, sum, write output ---
  {
    const int tokens_per_cta = (local_ntokens + gridDim.x - 1) / gridDim.x;
    const int my_token_start = tokens_per_cta * blockIdx.x;
    const int my_token_end   = min(my_token_start + tokens_per_cta, local_ntokens);
    const int total_blocks   = (my_token_end - my_token_start) * blocks_per_token;
    const int max_iters      = (total_blocks + num_sub_warps - 1) / num_sub_warps;

    uint4* my_dest = peer_dest[myrank] + lineoffset_dest;

    for (int iter = 0; iter < max_iters; iter++) {
      const int blk_local = iter * num_sub_warps + sub_warp_id;
      const bool valid    = blk_local < total_blocks;

      int local_token = 0, blk_in_token = 0, out_line = 0;
      if (valid) {
        const int local_t = blk_local / blocks_per_token;
        blk_in_token      = blk_local % blocks_per_token;
        local_token       = local_t + my_token_start;
        out_line          = local_token * bf16_lines_per_token
                            + blk_in_token * 4 + lane_in_sub;
      }

      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = 0.0f;

      for (int k = 0; k < topk_max; k++) {
        if (!valid) continue;  // sub-warp idle this iter

        const int expert_id = topk_idx[local_token * topk_max + k];
        if (expert_id < 0) continue;  // token routes to <topk_max experts

        float gate_w = 1.0f;
        if (gate_weights != nullptr)
          gate_w = gate_weights[local_token * total_experts + expert_id];

        // Read my own dest at (local_token, k, blk_in_token, lane_in_sub).
        const int dst_line = local_token * dest_stride_per_token
                             + k * bf16_lines_per_token
                             + blk_in_token * 4 + lane_in_sub;

        uint4 v;
#ifdef UB_TIMEOUT_ENABLED
        clock_t spin_s = clock64();
#endif
        while (true) {
          asm volatile("ld.volatile.v4.u32 {%0,%1,%2,%3}, [%4];"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                       : "l"(&my_dest[dst_line])
                       : "memory");
          if (v.w != UBX_LAMPORT_INT) break;
#ifdef UB_TIMEOUT_ENABLED
          if (clock64() - spin_s > UBX_TIMEOUT_CLOCKS) {
            printf("COMBINE_LAMPORT_PUSH POLL:SM %d [%d] t=%d k=%d:"
                   "expecting != %08x got %08x\n",
                   blockIdx.x, threadIdx.x, local_token, k,
                   UBX_LAMPORT_INT, v.w);
            break;
          }
#endif
        }

        const xhalf* vh = reinterpret_cast<const xhalf*>(&v);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          acc[i] += __bfloat162float(vh[i]) * gate_w;
        }
      }

      if (valid) {
        uint4 out_v;
        xhalf* outh = reinterpret_cast<xhalf*>(&out_v);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          outh[i] = __float2bfloat16(acc[i]);
        }
        ptr_out_bf16[out_line] = out_v;
      }
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // combine bf16-wire Lamport PUSH kernel

// ============================================================================
// PUSH-semantics combine, barrier-based (NOT Lamport).
//
// Same PUSH routing as ubx_kernel_combine_bf16_bf16_lamport_push but Phase 2
// uses a single cross-rank ATOMIC_MCINC barrier + spin, then a single bulk
// read of the own dest buf — saves Lamport's per-element poll overhead at
// large sizes.  Trade-off: at small messages the barrier latency dominates
// (Lamport variant wins there).
//
// Double-buffered (caller rotates 2 dest bufs via combine_push_double_buf
// in Python): call N writes/reads bufs[N%2], call N+1 writes/reads
// bufs[(N+1)%2].  This avoids the cross-call race where rank A's call N+1
// Phase 1 push to B's dest conflicts with B's call N Phase 2 read of same
// dest — the differing buf indices keep the writes and reads separated.
// One cross-rank barrier per call (Phase 1 → Phase 2 transition).  No
// warmup/poison logic needed since reads observe a fully-completed buffer.
// ============================================================================
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_bf16_bf16_push(
        const int RANKS,
        const int myrank,
        const int local_ntokens,
        const int n_recv,
        const int blocks_per_token,
        const int topk_max,
        const int total_experts,
        const int* __restrict__ inverse_map_flat,  // [max_tpr * 4]
        const int* __restrict__ topk_idx,           // [local_n, topk_max]
        const float* __restrict__ gate_weights,     // optional
        const int max_tokens_per_rank,
        const int64_t lineoffset_dest,
        int* uc_flagptr,
        int* mc_flagptr,
        const uint4* __restrict__ ptr_in_bf16,
        uint4* __restrict__ ptr_out_bf16,
        ncclWindow_t window) {
  const int bf16_lines_per_token = blocks_per_token * 4;
  const int dest_stride_per_token = topk_max * bf16_lines_per_token;

  __shared__ uint4* peer_dest[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  if (threadIdx.x < RANKS)
    peer_dest[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));

  int reduce_id;
  if (threadIdx.x == 0) {
    #ifndef UBX_SKIP_GRID_DEP_SYNC
    cudaGridDependencySynchronize();
    #endif
    reduce_id = uc_flagptr[UBX_FLAG_COMBINE_ID] + 1;
  }
  __syncthreads();

  // ---- Phase 1: PUSH local expert outputs to peers ----
  {
    const int slots_per_cta = (max_tokens_per_rank + gridDim.x - 1) / gridDim.x;
    const int my_slot_start = slots_per_cta * blockIdx.x;
    const int my_slot_end   = min(my_slot_start + slots_per_cta,
                                   max_tokens_per_rank);
    const int total_blocks  = (my_slot_end - my_slot_start) * blocks_per_token;
    const int max_iters     = (total_blocks + num_sub_warps - 1) / num_sub_warps;

    for (int iter = 0; iter < max_iters; iter++) {
      const int blk_local = iter * num_sub_warps + sub_warp_id;
      const bool valid    = blk_local < total_blocks;

      int slot = 0, blk_in_token = 0, src_line = 0;
      int orig_rank = 0, orig_token = 0, k_idx = 0, valid_slot = 0;
      if (valid) {
        const int slot_local = blk_local / blocks_per_token;
        blk_in_token = blk_local % blocks_per_token;
        slot         = slot_local + my_slot_start;
        src_line     = slot * bf16_lines_per_token + blk_in_token * 4 + lane_in_sub;

        const int* inv = inverse_map_flat + slot * 4;
        orig_rank  = inv[0];
        orig_token = inv[1];
        k_idx      = inv[2];
        valid_slot = inv[3];
      }

      if (valid && valid_slot && slot < n_recv) {
        const uint4 v = ptr_in_bf16[src_line];
        const int64_t dst_line = lineoffset_dest
                             + orig_token * dest_stride_per_token
                             + k_idx * bf16_lines_per_token
                             + blk_in_token * 4 + lane_in_sub;
        peer_dest[orig_rank][dst_line] = v;
      }
    }
  }

  // ---- Cross-rank barrier (last-SM ATOMIC_MCINC + every-block spin) ----
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();
    const int value_to_add = blockIdx.x == 0 ? UBX_MAX_SMS - gridDim.x + 1 : 1;
    const int old_val = atomicAdd(uc_flagptr + UBX_FLAG_COMBINE_SM_SYNC, value_to_add);
    const int lastSM =
        (gridDim.x == 1 || old_val + value_to_add == reduce_id * UBX_MAX_SMS);
    if (lastSM) {
      ATOMIC_MCINC(mc_flagptr + UBX_FLAG_COMBINE_BAR);
      uc_flagptr[UBX_FLAG_COMBINE_ID] = reduce_id;
    }
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_COMBINE_BAR]);
    const int expected = reduce_id * RANKS;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("COMBINE_PUSH BAR:SM %d:expecting %d got %d\n",
               blockIdx.x, expected, *flag);
        break;
      }
#endif
    }
  }
  __syncthreads();

  // ---- Phase 2: read OWN dest at (t, k) — no poll, just bulk read ----
  {
    const int tokens_per_cta = (local_ntokens + gridDim.x - 1) / gridDim.x;
    const int my_token_start = tokens_per_cta * blockIdx.x;
    const int my_token_end   = min(my_token_start + tokens_per_cta, local_ntokens);
    const int total_blocks   = (my_token_end - my_token_start) * blocks_per_token;
    const int max_iters      = (total_blocks + num_sub_warps - 1) / num_sub_warps;

    uint4* my_dest = peer_dest[myrank] + lineoffset_dest;

    for (int iter = 0; iter < max_iters; iter++) {
      const int blk_local = iter * num_sub_warps + sub_warp_id;
      const bool valid    = blk_local < total_blocks;

      int local_token = 0, blk_in_token = 0, out_line = 0;
      if (valid) {
        const int local_t = blk_local / blocks_per_token;
        blk_in_token      = blk_local % blocks_per_token;
        local_token       = local_t + my_token_start;
        out_line          = local_token * bf16_lines_per_token
                            + blk_in_token * 4 + lane_in_sub;
      }

      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; i++) acc[i] = 0.0f;

      for (int k = 0; k < topk_max; k++) {
        if (!valid) continue;

        const int expert_id = topk_idx[local_token * topk_max + k];
        if (expert_id < 0) continue;

        float gate_w = 1.0f;
        if (gate_weights != nullptr)
          gate_w = gate_weights[local_token * total_experts + expert_id];

        const int dst_line = local_token * dest_stride_per_token
                             + k * bf16_lines_per_token
                             + blk_in_token * 4 + lane_in_sub;
        const uint4 v = my_dest[dst_line];
        const xhalf* vh = reinterpret_cast<const xhalf*>(&v);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          acc[i] += __bfloat162float(vh[i]) * gate_w;
        }
      }

      if (valid) {
        uint4 out_v;
        xhalf* outh = reinterpret_cast<xhalf*>(&out_v);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          outh[i] = __float2bfloat16(acc[i]);
        }
        ptr_out_bf16[out_line] = out_v;
      }
    }
  }

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // combine bf16-wire PUSH-barrier kernel

// ============================================================================
// push3 (3-kernel PUSH combine, bf16 wire).
//
// Splits the PUSH combine into three kernels chained on the same stream:
//   Kernel 1 (phase1_write): many CTAs, each pushes its slice of expert
//     outputs to peer ranks' dest bufs at (origin_token, k_idx) coords.
//     Pure cross-rank NVLink WRITE. No flags. No barrier. Just data writes.
//   Kernel 2 (phase2_signal): 1 CTA, RANKS threads. Each thread i issues
//     ATOMIC_UCINC on peer i's UBX_FLAG_PUSH3_BAR. Then thread 0 spin-polls
//     OWN flag until BAR == reduce_id * RANKS. This is the ONLY kernel
//     that can hang; it has the compile-in timeout.
//   Kernel 3 (phase3_sum): many CTAs, each owns a slice of local_ntokens.
//     For each local token t, sums across topk slots in OWN dest buf
//     (zero NVLink), applies gate weights, writes output. Cannot hang —
//     purely local reads and writes.
//
// reduce_id is host-tracked (passed as a kernel arg to phase2_signal).
// ============================================================================

__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_push3_phase1_write(
        const int RANKS,
        const int n_recv,
        const int blocks_per_token,
        const int topk_max,
        const int* __restrict__ inverse_map_flat,  // [max_tpr * 4]
        const int max_tokens_per_rank,
        const int64_t lineoffset_dest,
        const uint4* __restrict__ ptr_in_bf16,
        ncclWindow_t window) {
  const int bf16_lines_per_token = blocks_per_token * 4;
  const int dest_stride_per_token = topk_max * bf16_lines_per_token;

  __shared__ uint4* peer_dest[MAXRANKS];

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  if (threadIdx.x < RANKS)
    peer_dest[threadIdx.x] = reinterpret_cast<uint4*>(
        ncclGetLsaPointer(window, 0, threadIdx.x));
  __syncthreads();

  const int slots_per_cta = (max_tokens_per_rank + gridDim.x - 1) / gridDim.x;
  const int my_slot_start = slots_per_cta * blockIdx.x;
  const int my_slot_end   = min(my_slot_start + slots_per_cta,
                                 max_tokens_per_rank);
  const int total_blocks  = (my_slot_end - my_slot_start) * blocks_per_token;
  const int max_iters     = (total_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_blocks;

    int slot = 0, blk_in_token = 0, src_line = 0;
    int orig_rank = 0, orig_token = 0, k_idx = 0, valid_slot = 0;
    if (valid) {
      const int slot_local = blk_local / blocks_per_token;
      blk_in_token = blk_local % blocks_per_token;
      slot         = slot_local + my_slot_start;
      src_line     = slot * bf16_lines_per_token + blk_in_token * 4 + lane_in_sub;

      const int* inv = inverse_map_flat + slot * 4;
      orig_rank  = inv[0];
      orig_token = inv[1];
      k_idx      = inv[2];
      valid_slot = inv[3];
    }

    if (valid && valid_slot && slot < n_recv) {
      const uint4 v = ptr_in_bf16[src_line];
      const int64_t dst_line = lineoffset_dest
                           + orig_token * dest_stride_per_token
                           + k_idx * bf16_lines_per_token
                           + blk_in_token * 4 + lane_in_sub;
      peer_dest[orig_rank][dst_line] = v;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // push3 phase1

// 1 SM, 1 CTA. Each thread i in [0, RANKS) ATOMIC_UCINCs peer i's flag,
// then thread 0 polls own flag until peers' atomics arrived. Has timeout.
//
// reduce_id is derived from the device-resident UBX_FLAG_PUSH3_ID slot
// (NOT passed from host). This keeps id and BAR state in lockstep across
// CUDA graph replays: each replay re-runs the kernel, which reads the
// current id from vidmem, increments, and uses that for `expected`. The
// BAR flag grows by RANKS per call, so `expected = new_id * RANKS` always
// matches the cumulative state. A host-passed id would be baked into the
// captured kernel arg and stay frozen at capture-time value while the BAR
// kept advancing → spin condition already true → instant pass → race.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_push3_phase2_signal(
        const int RANKS,
        int* uc_flagptr,
        ncclWindow_t window) {
  if (blockIdx.x != 0) return;
  __shared__ int s_expected;
  if (threadIdx.x == 0) {
    int rid = uc_flagptr[UBX_FLAG_PUSH3_ID] + 1;
    uc_flagptr[UBX_FLAG_PUSH3_ID] = rid;
    s_expected = rid * RANKS;
  }
  __syncthreads();
  if (threadIdx.x < (unsigned)RANKS) {
    const size_t REG0_FLAG_OFFSET = (size_t)RANKS * sizeof(void*);
    int* peer_flag = reinterpret_cast<int*>(
        ncclGetLsaPointer(window, REG0_FLAG_OFFSET, threadIdx.x));
    ATOMIC_UCINC(peer_flag + UBX_FLAG_PUSH3_BAR);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    volatile int* flag = (volatile int*)&(uc_flagptr[UBX_FLAG_PUSH3_BAR]);
    const int expected = s_expected;
#ifdef UB_TIMEOUT_ENABLED
    clock_t s = clock64();
#endif
    while (CHECK_IDS(*flag, expected)) {
#ifdef UB_TIMEOUT_ENABLED
      if (clock64() - s > UBX_TIMEOUT_CLOCKS) {
        printf("PUSH3 BAR: expected=%d got=%d\n", expected, *flag);
        break;
      }
#endif
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // push3 phase2

// Many CTAs. Purely local: reads OWN dest buf at (t, k) slots, sums across
// k_idx using gate_weights, writes output. No NVLink. Cannot hang.
__global__ void __launch_bounds__(UBX_MAXTHREADS)
    ubx_kernel_combine_push3_phase3_sum(
        const int local_ntokens,
        const int blocks_per_token,
        const int topk_max,
        const int total_experts,
        const int* __restrict__ topk_idx,           // [local_n, topk_max]
        const float* __restrict__ gate_weights,     // optional [local_n, total_experts]
        const int64_t lineoffset_dest,                  // pool-relative uint4 offset
        const uint4* __restrict__ my_dest,          // already = pool_base + lineoffset_dest
        uint4* __restrict__ ptr_out_bf16) {
  const int bf16_lines_per_token = blocks_per_token * 4;
  const int dest_stride_per_token = topk_max * bf16_lines_per_token;

  const int lane          = threadIdx.x & 31;
  const int lane_in_sub   = lane & 3;
  const int sub_warp_id   = threadIdx.x >> 2;
  const int num_sub_warps = blockDim.x >> 2;

  const int tokens_per_cta = (local_ntokens + gridDim.x - 1) / gridDim.x;
  const int my_token_start = tokens_per_cta * blockIdx.x;
  const int my_token_end   = min(my_token_start + tokens_per_cta, local_ntokens);
  const int total_blocks   = (my_token_end - my_token_start) * blocks_per_token;
  const int max_iters      = (total_blocks + num_sub_warps - 1) / num_sub_warps;

  for (int iter = 0; iter < max_iters; iter++) {
    const int blk_local = iter * num_sub_warps + sub_warp_id;
    const bool valid    = blk_local < total_blocks;

    int local_token = 0, blk_in_token = 0, out_line = 0;
    if (valid) {
      const int local_t = blk_local / blocks_per_token;
      blk_in_token      = blk_local % blocks_per_token;
      local_token       = local_t + my_token_start;
      out_line          = local_token * bf16_lines_per_token
                          + blk_in_token * 4 + lane_in_sub;
    }

    float acc[8];
#pragma unroll
    for (int i = 0; i < 8; i++) acc[i] = 0.0f;

    for (int k = 0; k < topk_max; k++) {
      if (!valid) continue;
      const int expert_id = topk_idx[local_token * topk_max + k];
      if (expert_id < 0) continue;
      float gate_w = 1.0f;
      if (gate_weights != nullptr)
        gate_w = gate_weights[local_token * total_experts + expert_id];
      const int dst_line = local_token * dest_stride_per_token
                           + k * bf16_lines_per_token
                           + blk_in_token * 4 + lane_in_sub;
      const uint4 v = my_dest[dst_line];
      const xhalf* vh = reinterpret_cast<const xhalf*>(&v);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] += __bfloat162float(vh[i]) * gate_w;
      }
    }

    if (valid) {
      uint4 out_v;
      xhalf* outh = reinterpret_cast<xhalf*>(&out_v);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        outh[i] = __float2bfloat16(acc[i]);
      }
      ptr_out_bf16[out_line] = out_v;
    }
  }
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
}  // push3 phase3

// Cap gridDim.x to what the device can actually schedule concurrently.
//
// Background: combine kernels have an intra-kernel barrier (Phase 1 signal →
// all CTAs spin in Phase 2). If the GPU cannot resident all gridDim.x CTAs at
// once, scheduled CTAs spin on the BAR holding SMs while unscheduled CTAs
// wait for an SM that will never free → deadlock. On B200 (132 SMs) we
// observed deadlock at gridDim.x=32+ when other resident kernels (NVTE,
// driver) hold ~16+ SMs.
//
// Returns min(requested_sms, deviceSMs * maxBlocksPerSM - safety_margin).
// The safety margin reserves SMs for the CUDA driver and other resident
// kernels (NVTE persistent ops, etc.).
static int safe_resident_sms(const void* kernel_fn, int threads, int requested_sms) {
  static int cached_device = -1;
  static cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  if (device != cached_device) {
    cudaGetDeviceProperties(&prop, device);
    cached_device = device;
  }
  int max_blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &max_blocks_per_sm, kernel_fn, threads, /*dynamic_shmem=*/0);
  if (max_blocks_per_sm <= 0) return requested_sms;  // fall back
  int safety_margin = 16;
  const char* env_margin = getenv("UBX_SM_SAFETY_MARGIN");
  if (env_margin) safety_margin = atoi(env_margin);
  int cap = prop.multiProcessorCount * max_blocks_per_sm - safety_margin;
  if (cap < 1) cap = 1;
  int result = requested_sms > cap ? cap : requested_sms;
  if (getenv("UBX_DEBUG_SM_CAP")) {
    fprintf(stderr,
            "[UBX_SM_CAP] requested=%d devSMs=%d maxBlocksPerSM=%d margin=%d cap=%d => %d\n",
            requested_sms, prop.multiProcessorCount, max_blocks_per_sm,
            safety_margin, cap, result);
  }
  return result;
}

#define SETUP_LAUNCH_CONFIG(sms, threads, stream, cga_size, pdl_launch)    \
  cudaLaunchConfig_t cfg = {sms, threads, 0, stream, NULL, 0};             \
  cudaLaunchAttribute attribute_ub[3];                                     \
  attribute_ub[2].id = cudaLaunchAttributeClusterDimension;                \
  attribute_ub[2].val.clusterDim.x = sms % cga_size == 0 ? cga_size : 1;   \
  attribute_ub[2].val.clusterDim.y = 1;                                    \
  attribute_ub[2].val.clusterDim.z = 1;                                    \
  attribute_ub[1].id = cudaLaunchAttributeCooperative;                     \
  attribute_ub[0].id = cudaLaunchAttributeProgrammaticStreamSerialization; \
  { const char* _env_pdl = getenv("UBX_DISABLE_PDL");                      \
    attribute_ub[0].val.programmaticStreamSerializationAllowed =           \
      (_env_pdl && _env_pdl[0] == '1') ? 0 : (pdl_launch); }               \
  cfg.attrs = attribute_ub;                                                \
  cfg.numAttrs = 3;

#define split_tokens(x)                                                              \
  const int elements = bytes / sizeof(half);                                         \
  const int elements_per_thread = sizeof(uint4) / sizeof(half);                      \
  int nthreads = 128, nlines = 1;                                                   \
  size_t total_bytes = bytes / ranks, start_bytes = myrank * total_bytes;            \
  int sms = x;                                                                       \
  if (hidden_size) {                                                                 \
    assert(hidden_size <= 32768);                                                    \
    assert(elements % hidden_size == 0);                                             \
    assert(hidden_size % elements_per_thread == 0);                                  \
    int ntokens = elements / hidden_size;                                            \
    int my_tokens = ntokens / ranks;                                                 \
    int extra_tokens = ntokens % ranks;                                              \
    int first_token = myrank * my_tokens;                                            \
    first_token += myrank < extra_tokens ? myrank : extra_tokens;                    \
    if (myrank < extra_tokens) my_tokens++;                                          \
    start_bytes = first_token * hidden_size * sizeof(half);                          \
    total_bytes = my_tokens * hidden_size * sizeof(half);                            \
    nthreads = hidden_size / elements_per_thread;                                    \
    nlines = 1;                                                                      \
    while (nthreads > 1024) {                                                        \
      nlines++;                                                                      \
      assert(nlines <= 4);                                                           \
      if ((hidden_size / elements_per_thread) % nlines == 0)                         \
        nthreads = ((hidden_size / elements_per_thread)) / nlines;                   \
    }                                                                                \
    if (sms > my_tokens) sms = my_tokens;                                            \
    if (sms == 0) sms = 1;                                                           \
  }                                                                                  \
  /* residual_in/out are uintptr_t under the new ABI — compare to 0, not nullptr */ \
  bool residual_in_global = residual_in != 0 && residual_in != residual_out && \
                            residual_out != 0;  // out residual is always local

// ============================================================================
// Public ABI launchers.
//
// Each public entry point now accepts (devcomm, window, pool_ptr) plus
// per-tensor byte offsets, instead of pre-resolved (uc0ptr, mc0ptr,
// mcptr_in/out) pointers. The launcher resolves the legacy pointer-form
// arguments host-side via NCCL host getters and feeds them to the kernel
// exactly as before — kernel bodies are unchanged in Phase 3 and will be
// rewritten in Phase 4 to call ncclGetLsa*Pointer() directly inside the
// kernel.
// ============================================================================

extern "C" void ubx_allreduce_2shot_mc(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    size_t in_offset, size_t out_offset, size_t bytes,
    uintptr_t residual_in, uintptr_t residual_out, bool fuse_layernorm,
    uintptr_t gamma, float eps, const int hidden_size,
    int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel,
    cudaStream_t stream) {
  // Host-resolve all multimem pointers + the local flag pointer once per
  // launch and pass the results as kernel args. The kernel can then start
  // its work immediately without any per-launch ncclGetLsaMultimem* calls.
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  // mc_ptr_in/out fold in the per-rank reduce-scatter start offset that
  // split_tokens() computes below, mirroring the legacy ABI shape.

  split_tokens(default_sms ? default_sms : 64);
  if (smlimit && sms>smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, (cgasize?cgasize:1), 1);

  void* mc_ptr_in  = nccl_lsa_mc_ptr(window, in_offset  + start_bytes);
  void* mc_ptr_out = nccl_lsa_mc_ptr(window, out_offset + start_bytes);

  int   arg1 = ranks, arg2 = myrank, arg3 = total_bytes / sizeof(uint4);
  void *arg4 = uc_flagptr, *arg5 = mc_flag,
       *arg6 = mc_ptr_in,  *arg7 = mc_ptr_out;
  void *arg8 = residual_in_global ? reinterpret_cast<void*>(residual_in + start_bytes)
                                  : reinterpret_cast<void*>(residual_in),
       *arg9  = reinterpret_cast<void*>(residual_out),
       *arg10 = reinterpret_cast<void*>(gamma);
  float arg11 = eps;
  int   arg12 = hidden_size;
  bool  arg13 = fuse_layernorm;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2,  (void *)&arg3,  (void *)&arg4,
                        (void *)&arg5, (void *)&arg6,  (void *)&arg7,  (void *)&arg8,
                        (void *)&arg9, (void *)&arg10, (void *)&arg11, (void *)&arg12,
                        (void *)&arg13};
#define call_mc_kernel(x, cond)                                                                   \
  if (x == nlines || cond) {                                                                      \
    CUDACHECK(                                                                                    \
        cudaLaunchKernelExC(&cfg, (void *)(ubx_kernel_allreduce_2shot_mc<x>), kernelArgs)); \
    return;                                                                                       \
  }
  call_mc_kernel(1, false);
  call_mc_kernel(2, false);
  call_mc_kernel(3, false);
  call_mc_kernel(4, true);
}

extern "C" void ubx_allreduce_2shot_uc(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t /*pool_ptr*/,
    size_t in_offset, size_t out_offset, size_t bytes,
    uintptr_t residual_in, uintptr_t residual_out, bool fuse_layernorm,
    uintptr_t gamma, float eps, const int hidden_size,
    int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel,
    cudaStream_t stream) {
  // Phase 4: kernel resolves uc_flagptr / peer pool pointers via the
  // NCCL device API on the registered window.
  int sms = default_sms ? default_sms : 64;
  int nthreads = 1024;
  if (smlimit && sms>smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, (cgasize?cgasize:1), 1);

  // lineoffset_{in,out} are uint4-line offsets relative to the pool base.
  // arg3/arg4 must be int64_t to match the kernel's int64_t lineoffset_*
  // signature — cudaLaunchKernelExC passes args by pointer and the kernel
  // reads 8 bytes per arg slot, so a 4-byte int here puts garbage in the
  // upper half and corrupts lineoffset → OOB write → cudaErrorIllegalAddress.
  int arg1 = myrank, arg2 = bytes / 16;
  int64_t arg3 = (int64_t)(in_offset / sizeof(uint4)),
          arg4 = (int64_t)(out_offset / sizeof(uint4));
  void *arg5 = reinterpret_cast<void*>(residual_in),
       *arg6 = reinterpret_cast<void*>(residual_out),
       *arg7 = reinterpret_cast<void*>(gamma);
  float arg8 = eps;
  int arg9 = hidden_size;
  bool arg10 = fuse_layernorm;
  ncclWindow_t arg11 = window;
  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4,
                        (void *)&arg5, (void *)&arg6, (void *)&arg7, (void *)&arg8,
                        (void *)&arg9, (void *)&arg10, (void *)&arg11};
#define call_uc_kernel(x) \
  if (x == ranks)         \
    CUDACHECK(            \
        cudaLaunchKernelExC(&cfg, (void *)(ubx_kernel_allreduce_2shot_uc<x>), kernelArgs));
  call_uc_kernel(2);
  call_uc_kernel(4);
  call_uc_kernel(8);
}

extern "C" void ubx_allreduce_2shot_mc_lamport(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ucptr_out_arg, size_t in_offset, size_t out_offset,
    uintptr_t clear_ptr_arg, size_t bytes, bool poisoned,
    uintptr_t residual_in, uintptr_t residual_out, bool fuse_layernorm,
    uintptr_t gamma, float eps, const int hidden_size,
    int default_sms, int smlimit, int cgasize, int nchunk, bool multi_kernel,
    cudaStream_t stream) {
  // Host-resolve mc + uc flag pointers once; pass to the kernel as args
  // to avoid per-launch device-side ncclGetLsaMultimem* overhead (this is
  // the kernel that showed +7% small-message Lamport latency in the
  // first device-side-getter pass).
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* ucptr_out = reinterpret_cast<void*>(ucptr_out_arg);
  void* clear_ptr = reinterpret_cast<void*>(clear_ptr_arg);


  if (!poisoned) {
    //user tells us destination was not pre-poisoned, so we need to do it before calling allreduce
    int threadsPerBlock = 512;
    int blocks = (bytes / 4 + threadsPerBlock - 1) / threadsPerBlock;
    memset_int<<<blocks, threadsPerBlock, 0, stream>>>((uint32_t *)ucptr_out, bytes / 4,
                                                       UBX_LAMPORT_INT);
  }
  split_tokens(default_sms ? default_sms : 64);
  if (smlimit && sms>smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, (cgasize?cgasize:1), 1);

  void* mc_ptr_in  = nccl_lsa_mc_ptr(window, in_offset  + start_bytes);
  void* mc_ptr_out = nccl_lsa_mc_ptr(window, out_offset + start_bytes);

  int   arg1 = ranks, arg2 = myrank, arg3 = total_bytes / sizeof(uint4),
        arg3a = bytes / sizeof(uint4);
  void *arg4 = uc_flagptr, *arg5 = mc_flag,
       *arg6 = mc_ptr_in,  *arg7 = mc_ptr_out,
       *arg8 = ucptr_out, *arg9 = clear_ptr,
       *arg10 = residual_in_global ? reinterpret_cast<void*>(residual_in + start_bytes)
                                   : reinterpret_cast<void*>(residual_in),
       *arg11 = reinterpret_cast<void*>(residual_out),
       *arg12 = reinterpret_cast<void*>(gamma);
  float arg13 = eps;
  int   arg14 = hidden_size;
  bool  arg15 = fuse_layernorm;
  void *kernelArgs[] = {(void *)&arg1,  (void *)&arg2,  (void *)&arg3,  (void *)&arg3a,
                        (void *)&arg4,  (void *)&arg5,  (void *)&arg6,  (void *)&arg7,
                        (void *)&arg8,  (void *)&arg9,  (void *)&arg10, (void *)&arg11,
                        (void *)&arg12, (void *)&arg13, (void *)&arg14, (void *)&arg15};

#define call_mc_lamport_kernel(x, cond)                                                           \
  if (x == nlines || cond) {                                                                      \
    CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)(ubx_kernel_allreduce_2shot_mc_lamport<x>), \
                                  kernelArgs));                                                   \
    return;                                                                                       \
  }

  call_mc_lamport_kernel(1, false);
  call_mc_lamport_kernel(2, false);
  call_mc_lamport_kernel(3, false);
  call_mc_lamport_kernel(4, true);
}

extern "C" void ubx_allgather_mc(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_arg, size_t out_offset, size_t bytes,
    int default_sms, int smlimit, cudaStream_t stream) {
  // Host-resolve uc/mc flag and mc_ptr_out (with per-rank slice offset).
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* mc_ptr_out = nccl_lsa_mc_ptr(window, out_offset + (size_t)myrank * bytes);
  void* ptr_in     = reinterpret_cast<void*>(ptr_in_arg);

  int sms = default_sms ? default_sms : 32;
  int nthreads = 1024;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int   arg1 = ranks, arg2 = myrank, arg3 = bytes / sizeof(uint4);
  void *arg4 = uc_flagptr, *arg5 = mc_flag,
       *arg6 = ptr_in,     *arg7 = mc_ptr_out;

  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4,
                        (void *)&arg5, (void *)&arg6, (void *)&arg7};

  CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)(ubx_kernel_allgather_mc), kernelArgs));
}

extern "C" void ubx_allgather_uc(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t /*pool_ptr*/,
    uintptr_t ptr_in_arg, size_t out_offset, size_t bytes,
    int default_sms, int smlimit, cudaStream_t stream) {
  void* ptr_in = reinterpret_cast<void*>(ptr_in_arg);

  int sms = default_sms ? default_sms : 32;
  int nthreads = 1024;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int arg1 = ranks, arg2 = myrank, arg3 = bytes / sizeof(uint4);
  // arg4 is int64_t to match kernel's int64_t lineoffset_out — see comment
  // in ubx_allreduce_2shot_uc for why a 4-byte arg here causes IMA.
  int64_t arg4 = (int64_t)(out_offset / sizeof(uint4));
  void* arg5 = ptr_in;
  ncclWindow_t arg6 = window;

  void* kernelArgs[] = {(void*)&arg1, (void*)&arg2, (void*)&arg3,
                        (void*)&arg4, (void*)&arg5, (void*)&arg6};
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_allgather_uc, kernelArgs));
}

extern "C" void ubx_alltoall(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t /*pool_ptr*/,
    uintptr_t ptr_in_arg, size_t out_offset, size_t bytes,
    int default_sms, int smlimit, int nthreads, cudaStream_t stream) {
  // Phase 4: kernel resolves uc_flagptr / peer pool pointers from the
  // NCCL device API on the registered window. No more host-side
  // mc-pointer pre-resolution or commbuff threading.
  void* ptr_in = reinterpret_cast<void*>(ptr_in_arg);

  int sms = default_sms ? default_sms : 32;
  int threads = 1024;
  // smlimit set-exact, nthreads set-exact (0 = use default_sms / launcher
  // default). Enables runtime SM/threads sweeps from the bench
  // (--smlimit / --nthreads) without a rebuild.
  if (smlimit) sms = smlimit;
  if (nthreads) threads = nthreads;
  SETUP_LAUNCH_CONFIG(sms, threads, stream, 1, 1);


  int arg1 = ranks, arg2 = myrank, arg3 = bytes / sizeof(uint4);
  // arg4 is int64_t to match kernel's int64_t lineoffset_out — see comment
  // in ubx_allreduce_2shot_uc for why a 4-byte arg here causes IMA.
  int64_t arg4 = (int64_t)(out_offset / sizeof(uint4));
  void* arg5 = ptr_in;
  ncclWindow_t arg6 = window;

  void *kernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3,
                        (void *)&arg4, (void *)&arg5, (void *)&arg6};

  CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)(ubx_kernel_a2a), kernelArgs));
}

extern "C" void ubx_alltoallv(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t /*pool_ptr*/,
    uintptr_t ptr_in_arg,
    uintptr_t send_byte_offsets_arg, uintptr_t send_byte_counts_arg,
    uintptr_t dest_byte_offsets_arg,
    int default_sms, int smlimit, cudaStream_t stream) {
  // Phase 4: kernel resolves uc_flagptr / peer pool pointers via the
  // NCCL device API on the registered window.
  void* ptr_in            = reinterpret_cast<void*>(ptr_in_arg);
  void* send_byte_offsets = reinterpret_cast<void*>(send_byte_offsets_arg);
  void* send_byte_counts  = reinterpret_cast<void*>(send_byte_counts_arg);
  void* dest_byte_offsets = reinterpret_cast<void*>(dest_byte_offsets_arg);

  int sms = default_sms ? default_sms : 32;
  int nthreads = 1024;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int          arg_ranks       = ranks;
  int          arg_myrank      = myrank;
  void*        arg_send_off    = send_byte_offsets;
  void*        arg_send_cnt    = send_byte_counts;
  void*        arg_dest_off    = dest_byte_offsets;
  void*        arg_in          = ptr_in;
  ncclWindow_t arg_window      = window;

  void *kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_send_off, &arg_send_cnt, &arg_dest_off,
    &arg_in, &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)ubx_kernel_a2av, kernelArgs));
}

extern "C" void ubx_alltoall_lamport(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_arg, size_t out_offset, uintptr_t clear_ptr_arg,
    size_t bytes, bool poisoned,
    int default_sms, int smlimit, int nthreads, bool skip_barrier,
    cudaStream_t stream) {
  void* ptr_in    = reinterpret_cast<void*>(ptr_in_arg);
  void* ucptr_out = reinterpret_cast<void*>(pool_ptr + out_offset);
  void* clear_ptr = reinterpret_cast<void*>(clear_ptr_arg);


  if (!poisoned) {
    int threadsPerBlock = 512;
    size_t total_bytes = bytes * ranks;
    int blocks = (total_bytes / 4 + threadsPerBlock - 1) / threadsPerBlock;
    memset_int<<<blocks, threadsPerBlock, 0, stream>>>((uint32_t *)ucptr_out, total_bytes / 4,
                                                       UBX_LAMPORT_INT);
  }

  int default_sms_local = default_sms ? default_sms : 32;
  int threads = 1024;
  // smlimit is cap-only here (consistent with allreduce/allgather/a2av_mxfp8);
  // nthreads sets threads-per-block exactly when non-zero.
  if (smlimit && default_sms_local > smlimit) default_sms_local = smlimit;
  if (nthreads) threads = nthreads;

  // Per-kernel SM count overrides (one-shot env reads, cached).
  //   UBX_LAMPORT_WRITE_SMS — SMs for the write-phase kernel.
  //   UBX_LAMPORT_POLL_SMS  — SMs for the poll-phase kernel.
  // 0 / unset => use the size-aware default (write) / default_sms_local (poll).
  static int s_write_sms = -1, s_poll_sms = -1;
  if (s_write_sms < 0) {
    const char* env = getenv("UBX_LAMPORT_WRITE_SMS");
    s_write_sms = env ? atoi(env) : 0;
  }
  if (s_poll_sms < 0) {
    const char* env = getenv("UBX_LAMPORT_POLL_SMS");
    s_poll_sms = env ? atoi(env) : 0;
  }

  // Size-aware defaults (4-GPU sweeps, jobs 165530-165545):
  //   write: 32 SMs suffice when the per-rank chunk is small (kernel is
  //     launch-overhead bound). At total tensor >= 1 MiB the kernel
  //     becomes UC-write bandwidth-bound; 64 SMs win by 4-10%.
  //   poll: same crossover at total tensor 1 MiB, but the poll kernel
  //     is local-memory bandwidth bound at large sizes and scales
  //     linearly to high SM counts. 128 SMs (≈ device SM count on
  //     B200/GB200) wins by 27% over 32 SMs at 32 MiB.
  const size_t total_bytes_for_sms = (size_t)bytes * (size_t)ranks;
  int default_write_sms = default_sms_local;
  int default_poll_sms  = default_sms_local;
  if (total_bytes_for_sms >= (1 * 1024 * 1024)) {
    default_write_sms = 64;
    default_poll_sms  = 128;
  }

  const int write_sms = s_write_sms > 0 ? s_write_sms : default_write_sms;
  const int poll_sms  = s_poll_sms  > 0 ? s_poll_sms  : default_poll_sms;

  // Phase 1 kernel args.
  int arg1 = ranks, arg2 = myrank, arg3 = bytes / sizeof(uint4);
  // arg4 is int64_t to match kernel's int64_t lineoffset_out — see comment
  // in ubx_allreduce_2shot_uc for why a 4-byte arg here causes IMA.
  int64_t arg4 = (int64_t)(out_offset / sizeof(uint4));
  void *arg5 = ptr_in;
  bool arg6 = skip_barrier;
  ncclWindow_t arg7 = window;
  void *writeKernelArgs[] = {(void *)&arg1, (void *)&arg2, (void *)&arg3, (void *)&arg4,
                             (void *)&arg5, (void *)&arg6, (void *)&arg7};
  {
    SETUP_LAUNCH_CONFIG(write_sms, threads, stream, 1, 1);
    CUDACHECK(cudaLaunchKernelExC(&cfg, (void *)(ubx_kernel_alltoall_lamport_write),
                                  writeKernelArgs));
  }

  // Phase 2 kernel: pick variant based on total tensor size. Default
  // cutoff 2 MiB; overridable via UBX_LAMPORT_POISON_FIRST_CUTOFF
  // (interpreted as bytes of the total tensor = bytes_per_rank * ranks).
  // Below cutoff: bulk-poison-then-poll (avoids the per-iteration side
  // store in the poll loop). At/above cutoff: per-line interleaved.
  //
  // Sweep results (bf16, ranks->crossover): 4 ranks crosses between
  // 512K and 1M; 32 ranks crosses between 1M and 2M. 2 MiB is the
  // safe default that wins at 32 ranks and is within noise at 4 ranks
  // for sizes 1-2M (where both variants are <5% apart). Hang-tests
  // 165520 / 165521 / 165523 / 165524.
  static int s_poison_first_cutoff = -1;
  if (s_poison_first_cutoff < 0) {
    const char* env = getenv("UBX_LAMPORT_POISON_FIRST_CUTOFF");
    s_poison_first_cutoff = env ? atoi(env) : (2 * 1024 * 1024);
  }
  const size_t total_bytes = (size_t)bytes * (size_t)ranks;
  const bool poison_first = total_bytes < (size_t)s_poison_first_cutoff;

  void *pollArg1 = ucptr_out, *pollArg2 = clear_ptr;
  void *pollKernelArgs[] = {(void *)&arg1, (void *)&arg3, (void *)&pollArg1, (void *)&pollArg2};
  void* poll_kernel_fn = poison_first
      ? (void *)(ubx_kernel_alltoall_lamport_poll_poison_first)
      : (void *)(ubx_kernel_alltoall_lamport_poll);
  {
    SETUP_LAUNCH_CONFIG(poll_sms, threads, stream, 1, 1);
    CUDACHECK(cudaLaunchKernelExC(&cfg, poll_kernel_fn, pollKernelArgs));
  }
}

extern "C" void ubx_a2av_token_bf16_mxfp8(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, uintptr_t token_offsets_arg,
    int64_t lineoffset_out, int64_t lineoffset_scales,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg,
    int default_sms, int smlimit, int sync,
    int expert_start, int expert_count,
    cudaStream_t stream) {
  // Host-resolve uc/mc flag pointers; kernel resolves per-rank pool
  // bases device-side via ncclGetLsaPointer (per-peer, cheap).
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* token_offsets = reinterpret_cast<void*>(token_offsets_arg);
  void* ptr_in_bf16   = reinterpret_cast<void*>(ptr_in_bf16_arg);

  // Per-kernel default for mxfp8 dispatch is 128 SMs (the v2 4-thread
  // sub-warp layout keeps scaling all the way up: avg latency 181 → 113 →
  // 84 μs at sm=32/64/128). UBX_MAXSM env var overrides via default_sms.
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_ntokens     = ntokens;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  void*  arg_offsets     = token_offsets;
  int64_t    arg_loff_out    = lineoffset_out;
  int64_t    arg_loff_scales = lineoffset_scales;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  void*  arg_in          = ptr_in_bf16;
  int    arg_sync        = sync;
  int    arg_exp_start   = expert_start;
  int    arg_exp_count   = expert_count;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_ntokens, &arg_bpt, &arg_epr,
    &arg_offsets, &arg_loff_out, &arg_loff_scales,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_sync, &arg_exp_start, &arg_exp_count,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_a2av_token_bf16_mxfp8, kernelArgs));
}

extern "C" void ubx_a2av_token_bf16_bf16(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, uintptr_t token_offsets_arg,
    int64_t lineoffset_out,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg,
    int default_sms, int smlimit, int sync,
    int expert_start, int expert_count,
    cudaStream_t stream) {
  // Mirrors ubx_a2av_token_bf16_mxfp8 minus lineoffset_scales — bf16 output
  // has no per-block scale region.
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* token_offsets = reinterpret_cast<void*>(token_offsets_arg);
  void* ptr_in_bf16   = reinterpret_cast<void*>(ptr_in_bf16_arg);

  // Per-kernel default for bf16 dispatch is 64 SMs — the v2 4-thread
  // sub-warp layout saturates NVLink bandwidth at 64 SMs (avg latency
  // 165 → 94 → 98 μs at sm=32/64/128; sm=128 sees marginal contention).
  // UBX_MAXSM env var overrides via default_sms.
  int sms      = default_sms ? default_sms : 64;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_ntokens     = ntokens;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  void*  arg_offsets     = token_offsets;
  int64_t    arg_loff_out    = lineoffset_out;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  void*  arg_in          = ptr_in_bf16;
  int    arg_sync        = sync;
  int    arg_exp_start   = expert_start;
  int    arg_exp_count   = expert_count;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_ntokens, &arg_bpt, &arg_epr,
    &arg_offsets, &arg_loff_out,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_sync, &arg_exp_start, &arg_exp_count,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_a2av_token_bf16_bf16, kernelArgs));
}

// Top-K variant launcher. Same arg shape as ubx_a2av_token_bf16_bf16 but
// takes topk_expert + topk_slot LUTs (shape [ntokens, topk_max]) instead
// of token_offsets[ntokens, total_experts]. The kernel's inner k-loop
// runs only topk_max iterations instead of total_experts iterations.
// No expert-range chunking — drop expert_start/expert_count.
extern "C" void ubx_a2av_token_bf16_bf16_topk(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, int topk_max,
    uintptr_t topk_expert_arg, uintptr_t topk_slot_arg,
    int64_t lineoffset_out,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg,
    int default_sms, int smlimit, int sync,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* topk_exp   = reinterpret_cast<void*>(topk_expert_arg);
  void* topk_sl    = reinterpret_cast<void*>(topk_slot_arg);
  void* ptr_in_bf16 = reinterpret_cast<void*>(ptr_in_bf16_arg);

  int sms      = default_sms ? default_sms : 64;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int   arg_ranks    = ranks;
  int   arg_myrank   = myrank;
  int   arg_ntokens  = ntokens;
  int   arg_bpt      = blocks_per_token;
  int   arg_epr      = experts_per_rank;
  int   arg_topk     = topk_max;
  void* arg_topk_exp = topk_exp;
  void* arg_topk_sl  = topk_sl;
  int64_t   arg_loff_out = lineoffset_out;
  void* arg_uc_flag  = uc_flagptr;
  void* arg_mc_flag  = mc_flag;
  void* arg_in       = ptr_in_bf16;
  int   arg_sync     = sync;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_ntokens, &arg_bpt, &arg_epr, &arg_topk,
    &arg_topk_exp, &arg_topk_sl, &arg_loff_out,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_sync, &arg_window,
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_a2av_token_bf16_bf16_topk, kernelArgs));
}

extern "C" void ubx_a2av_token_bf16_mxfp8_persistent(
    int ranks, int myrank, int ntokens, int blocks_per_token,
    int experts_per_rank, uintptr_t token_offsets_arg,
    int64_t lineoffset_out, int64_t lineoffset_scales,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg,
    int default_sms, int smlimit,
    int nchunks, int nexperts_per_chunk,
    cudaStream_t stream) {
  // Host-resolve uc/mc flag pointers. pool_ptr is still required for the
  // host-side cudaMemsetAsync of the per-chunk SM_SYNC scratch slots
  // (we can't issue cudaMemset against a pure window handle).
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* token_offsets = reinterpret_cast<void*>(token_offsets_arg);
  void* ptr_in_bf16   = reinterpret_cast<void*>(ptr_in_bf16_arg);

  if (nchunks < 1 || nchunks > UBX_MAX_PERSISTENT_CHUNKS) {
    printf("a2av_persistent: nchunks=%d out of range [1,%d]\n",
           nchunks, UBX_MAX_PERSISTENT_CHUNKS);
    return;
  }

  // Reset per-chunk SM_SYNC scratch slots to 0 before launch.
  CUDACHECK(cudaMemsetAsync(
      uc_flagptr + UBX_FLAG_A2AV_PERSISTENT_SM_SYNC_BASE,
      0, nchunks * sizeof(int), stream));

  // Per-kernel default 128 SMs (matches non-persistent mxfp8 launcher;
  // UBX_MAXSM env var overrides via default_sms).
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_ntokens     = ntokens;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  void*  arg_offsets     = token_offsets;
  int64_t    arg_loff_out    = lineoffset_out;
  int64_t    arg_loff_scales = lineoffset_scales;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  void*  arg_in          = ptr_in_bf16;
  int    arg_nchunks     = nchunks;
  int    arg_nec         = nexperts_per_chunk;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_ntokens, &arg_bpt, &arg_epr,
    &arg_offsets, &arg_loff_out, &arg_loff_scales,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_nchunks, &arg_nec,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg,
                                (void*)ubx_kernel_a2av_token_bf16_mxfp8_persistent,
                                kernelArgs));
}

extern "C" void ubx_a2av_wait(
    int ranks,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t /*pool_ptr*/,
    cudaStream_t stream) {
  // Phase 4: kernel resolves the flag pointer via window.
  SETUP_LAUNCH_CONFIG(1, 1, stream, 1, 1);

  int          arg_ranks  = ranks;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = { &arg_ranks, &arg_window };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_a2av_wait, kernelArgs));
}

// ----------------------------------------------------------------------------
// Standalone barrier launcher — no data movement, just the atomic-flag
// barrier protocol. Uses UBX_FLAG_BARRIER_* slots (separate from
// combine/dispatch). Useful for surrounding other UBX calls with a
// hard sync without sharing flag slots with the data kernels.
// ----------------------------------------------------------------------------
// E22: combine v2 — two separate kernels (phase1 = pure copy, phase2 = barrier+sum).
extern "C" void ubx_combine_v2_phase1_bf16(
    int n_recv, int blocks_per_token,
    uintptr_t pool_ptr, int64_t lineoffset_temp,
    uintptr_t ptr_in_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const void* ptr_in_bf16 = reinterpret_cast<const void*>(ptr_in_bf16_arg);
  uint4* my_temp = reinterpret_cast<uint4*>(pool_ptr + (size_t)lineoffset_temp * sizeof(uint4));
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms((const void*)ubx_kernel_combine_v2_phase1_bf16, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);
  int arg_n_recv = n_recv;
  int arg_bpt = blocks_per_token;
  const void* arg_in = ptr_in_bf16;
  void* arg_temp = my_temp;
  void* kernelArgs[] = { &arg_n_recv, &arg_bpt, &arg_in, &arg_temp };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_combine_v2_phase1_bf16, kernelArgs));
}

extern "C" void ubx_combine_v2_phase2_bf16(
    int ranks, int reduce_id, int local_ntokens, int blocks_per_token,
    int experts_per_rank,
    uintptr_t token_offsets_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_temp,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* token_offsets  = reinterpret_cast<void*>(token_offsets_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms((const void*)ubx_kernel_combine_v2_phase2_bf16, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int arg_ranks = ranks;
  int arg_rid = reduce_id;
  int arg_local_n = local_ntokens;
  int arg_bpt = blocks_per_token;
  int arg_epr = experts_per_rank;
  void* arg_offsets = token_offsets;
  void* arg_gates = gate_weights;
  int64_t arg_loff_temp = lineoffset_temp;
  void* arg_uc_flag = uc_flagptr;
  ncclWindow_t arg_window = window;
  void* arg_out = ptr_out_bf16;
  void* kernelArgs[] = {
    &arg_ranks, &arg_rid, &arg_local_n, &arg_bpt, &arg_epr,
    &arg_offsets, &arg_gates, &arg_loff_temp,
    &arg_uc_flag, &arg_window, &arg_out
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_combine_v2_phase2_bf16, kernelArgs));
}

// Diagnostic launcher: dump peer pointers via the GPU kernel. Caller
// supplies a device-resident uint64 buffer of size >= ranks; on return
// (after stream sync) buf[i] = ncclGetLsaPointer(window, 0, i) for i in
// [0, ranks).
extern "C" void ubx_peer_ptr_dump(
    int ranks,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t out_ptrs_arg,
    cudaStream_t stream) {
  unsigned long long* out_ptrs =
      reinterpret_cast<unsigned long long*>(out_ptrs_arg);
  int sms = 1;
  int nthreads = ranks > 32 ? ranks : 32;  // need >= ranks threads, round up
  if (nthreads > UBX_MAXTHREADS) nthreads = UBX_MAXTHREADS;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);
  int    arg_ranks = ranks;
  ncclWindow_t arg_window = window;
  void*  arg_out = out_ptrs;
  void* kernelArgs[] = { &arg_ranks, &arg_window, &arg_out };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_peer_ptr_dump, kernelArgs));
}

// ============================================================================
// push3 (3-kernel PUSH combine) launchers.
// ============================================================================
extern "C" void ubx_combine_push3_phase1_write(
    int ranks, int n_recv, int blocks_per_token, int topk_max,
    uintptr_t inverse_map_arg, int max_tokens_per_rank, int64_t lineoffset_dest,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window,
    uintptr_t ptr_in_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const int* inverse_map = reinterpret_cast<const int*>(inverse_map_arg);
  const void* ptr_in_bf16 = reinterpret_cast<const void*>(ptr_in_bf16_arg);
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms(
      (const void*)ubx_kernel_combine_push3_phase1_write, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);
  int arg_ranks = ranks;
  int arg_n_recv = n_recv;
  int arg_bpt = blocks_per_token;
  int arg_topk_max = topk_max;
  const void* arg_inv = inverse_map;
  int arg_max_tpr = max_tokens_per_rank;
  int64_t arg_loff_dest = lineoffset_dest;
  const void* arg_in = ptr_in_bf16;
  ncclWindow_t arg_window = window;
  void* kernelArgs[] = {
    &arg_ranks, &arg_n_recv, &arg_bpt, &arg_topk_max,
    &arg_inv, &arg_max_tpr, &arg_loff_dest,
    &arg_in, &arg_window,
  };
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, (void*)ubx_kernel_combine_push3_phase1_write, kernelArgs));
}

extern "C" void ubx_combine_push3_phase2_signal(
    int ranks,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  int sms = 1;
  int nthreads = ranks > 32 ? ranks : 32;
  if (nthreads > UBX_MAXTHREADS) nthreads = UBX_MAXTHREADS;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);
  int arg_ranks = ranks;
  void* arg_uc = uc_flagptr;
  ncclWindow_t arg_window = window;
  void* kernelArgs[] = { &arg_ranks, &arg_uc, &arg_window };
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, (void*)ubx_kernel_combine_push3_phase2_signal, kernelArgs));
}

extern "C" void ubx_combine_push3_phase3_sum(
    int local_ntokens, int blocks_per_token, int topk_max, int total_experts,
    uintptr_t topk_idx_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_dest, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const int*   topk_idx     = reinterpret_cast<const int*>(topk_idx_arg);
  const float* gate_weights = reinterpret_cast<const float*>(gate_weights_arg);
  const uint4* my_dest      = reinterpret_cast<const uint4*>(
      pool_ptr + (size_t)lineoffset_dest * sizeof(uint4));
  void* ptr_out_bf16        = reinterpret_cast<void*>(ptr_out_bf16_arg);

  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms(
      (const void*)ubx_kernel_combine_push3_phase3_sum, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int arg_local_n = local_ntokens;
  int arg_bpt = blocks_per_token;
  int arg_topk_max = topk_max;
  int arg_total_e = total_experts;
  const void* arg_tk = topk_idx;
  const void* arg_gw = gate_weights;
  int64_t arg_loff_dest = lineoffset_dest;
  const void* arg_dest = my_dest;
  void* arg_out = ptr_out_bf16;
  void* kernelArgs[] = {
    &arg_local_n, &arg_bpt, &arg_topk_max, &arg_total_e,
    &arg_tk, &arg_gw, &arg_loff_dest, &arg_dest, &arg_out,
  };
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, (void*)ubx_kernel_combine_push3_phase3_sum, kernelArgs));
}

// Diagnostic launcher: cross-rank peer atomic-inc test. Returns once kernel
// completes (or is killed by watchdog). Caller should torch.cuda.synchronize
// to wait + see whether the test passed or hung.
extern "C" void ubx_peer_atomic_test(
    int ranks, int myrank, int test_id,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int* uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  int sms = 1;
  int nthreads = ranks > 32 ? ranks : 32;
  if (nthreads > UBX_MAXTHREADS) nthreads = UBX_MAXTHREADS;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);
  int    arg_ranks = ranks;
  int    arg_myrank = myrank;
  int    arg_test_id = test_id;
  void*  arg_uc = uc_flagptr;
  ncclWindow_t arg_window = window;
  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_test_id, &arg_uc, &arg_window,
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_peer_atomic_test, kernelArgs));
}

extern "C" void ubx_barrier(
    int ranks, int myrank,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag    = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  int sms      = default_sms ? default_sms : 1;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);
  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  ncclWindow_t arg_window = window;
  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_uc_flag, &arg_mc_flag, &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_barrier, kernelArgs));
}

// ----------------------------------------------------------------------------
// Combine launchers (reverse of a2av_token dispatch).
// ----------------------------------------------------------------------------

extern "C" void ubx_combine_bf16_bf16(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token,
    int experts_per_rank, int max_tokens_per_rank,
    uintptr_t token_offsets_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_temp,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg, uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit, int sync,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag        = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* token_offsets  = reinterpret_cast<void*>(token_offsets_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);  // may be 0
  void* ptr_in_bf16    = reinterpret_cast<void*>(ptr_in_bf16_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);

  // Per-kernel default 128 SMs. Combine Phase 2 is read+fp32-sum bound
  // (not write-bound like dispatch), so it scales with SM count: a 4-rank
  // GB200 SM sweep at α=2 (hidden=7168, topk=8) showed bf16 combine
  // bus_bw 93→158→184→226 GB/s at sm=32/64/96/128 — a +43% jump from 64
  // to 128. UBX_MAXSM env var still overrides via default_sms.
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  // Cap to what the device can actually run concurrently. Combine has an
  // intra-kernel barrier — over-subscribing SMs causes scheduling deadlock.
  sms = safe_resident_sms((const void*)ubx_kernel_combine_bf16_bf16, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_local_n     = local_ntokens;
  int    arg_n_recv      = n_recv;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  int    arg_max_tpr     = max_tokens_per_rank;
  void*  arg_offsets     = token_offsets;
  void*  arg_gates       = gate_weights;
  int64_t    arg_loff_temp   = lineoffset_temp;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  void*  arg_in          = ptr_in_bf16;
  void*  arg_out         = ptr_out_bf16;
  int    arg_sync        = sync;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_local_n, &arg_n_recv,
    &arg_bpt, &arg_epr, &arg_max_tpr,
    &arg_offsets, &arg_gates, &arg_loff_temp,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_out, &arg_sync,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_combine_bf16_bf16, kernelArgs));
}

extern "C" void ubx_combine_mxfp8_bf16(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token,
    int experts_per_rank, int max_tokens_per_rank,
    uintptr_t token_offsets_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_temp, int64_t lineoffset_scales,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg, uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit, int sync,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag        = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* token_offsets  = reinterpret_cast<void*>(token_offsets_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);
  void* ptr_in_bf16    = reinterpret_cast<void*>(ptr_in_bf16_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);

  // Per-kernel default 128 SMs (mirrors a2av_token_bf16_mxfp8 dispatch).
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms((const void*)ubx_kernel_combine_mxfp8_bf16, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_local_n     = local_ntokens;
  int    arg_n_recv      = n_recv;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  int    arg_max_tpr     = max_tokens_per_rank;
  void*  arg_offsets     = token_offsets;
  void*  arg_gates       = gate_weights;
  int64_t    arg_loff_temp   = lineoffset_temp;
  int64_t    arg_loff_scales = lineoffset_scales;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  void*  arg_in          = ptr_in_bf16;
  void*  arg_out         = ptr_out_bf16;
  int    arg_sync        = sync;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_local_n, &arg_n_recv,
    &arg_bpt, &arg_epr, &arg_max_tpr,
    &arg_offsets, &arg_gates, &arg_loff_temp, &arg_loff_scales,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_out, &arg_sync,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_combine_mxfp8_bf16, kernelArgs));
}

extern "C" void ubx_combine_wait_bf16(
    int ranks, int local_ntokens, int blocks_per_token,
    int experts_per_rank,
    uintptr_t token_offsets_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_temp,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* token_offsets  = reinterpret_cast<void*>(token_offsets_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);

  // Match the bf16 main kernel default (128 SMs) — see SM-sweep comment
  // in ubx_combine_bf16_bf16.
  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms((const void*)ubx_kernel_combine_wait_bf16, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_local_n     = local_ntokens;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  void*  arg_offsets     = token_offsets;
  void*  arg_gates       = gate_weights;
  int64_t    arg_loff_temp   = lineoffset_temp;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_out         = ptr_out_bf16;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_local_n, &arg_bpt, &arg_epr,
    &arg_offsets, &arg_gates, &arg_loff_temp,
    &arg_uc_flag, &arg_out, &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_combine_wait_bf16, kernelArgs));
}

extern "C" void ubx_combine_wait_mxfp8(
    int ranks, int local_ntokens, int blocks_per_token,
    int experts_per_rank,
    uintptr_t token_offsets_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_temp, int64_t lineoffset_scales,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* token_offsets  = reinterpret_cast<void*>(token_offsets_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);

  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms((const void*)ubx_kernel_combine_wait_mxfp8, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int    arg_ranks       = ranks;
  int    arg_local_n     = local_ntokens;
  int    arg_bpt         = blocks_per_token;
  int    arg_epr         = experts_per_rank;
  void*  arg_offsets     = token_offsets;
  void*  arg_gates       = gate_weights;
  int64_t    arg_loff_temp   = lineoffset_temp;
  int64_t    arg_loff_scales = lineoffset_scales;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_out         = ptr_out_bf16;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_local_n, &arg_bpt, &arg_epr,
    &arg_offsets, &arg_gates, &arg_loff_temp, &arg_loff_scales,
    &arg_uc_flag, &arg_out, &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(&cfg, (void*)ubx_kernel_combine_wait_mxfp8, kernelArgs));
}

// ----------------------------------------------------------------------------
// Lamport combine PUSH launcher (bf16 wire) — race-free counterpart to the
// PULL variant.  Each rank pushes its expert outputs to peer destination
// bufs at (origin_token, k_idx) coordinates; readers poll OWN buf.  See
// ubx_kernel_combine_bf16_bf16_lamport_push docstring.
//
// poisoned/skip_warmup_barrier: caller-side flags. Caller must arrange
// triple-buffer rotation in Python (combine_push_triple_buf).
// Memset target on warmup is the LOCAL dest buf (peers will write into
// it; we must initialize it to poison so the readers' Lamport polls see
// "not yet" until peers' writes land).
// ----------------------------------------------------------------------------
extern "C" void ubx_combine_bf16_bf16_lamport_push(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
    int topk_max,
    uintptr_t inverse_map_arg, uintptr_t topk_idx_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_dest, uintptr_t clear_ptr_arg,
    bool poisoned, bool skip_warmup_barrier,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg, uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* inverse_map    = reinterpret_cast<void*>(inverse_map_arg);
  void* topk_idx       = reinterpret_cast<void*>(topk_idx_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);
  void* clear_ptr      = reinterpret_cast<void*>(clear_ptr_arg);
  void* ptr_in_bf16    = reinterpret_cast<void*>(ptr_in_bf16_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);

  uint4* my_dest_buf =
      reinterpret_cast<uint4*>(pool_ptr + (size_t)lineoffset_dest * sizeof(uint4));

  // Warmup memset: poison every uint32 of MY dest buf so peers' writes
  // landing here transition poison→data, and my Phase 2 polls observe it.
  if (!poisoned) {
    int threadsPerBlock = 512;
    size_t total_bytes = (size_t)local_ntokens * topk_max
                         * blocks_per_token * 4 * sizeof(uint4);
    size_t total_uint32 = total_bytes / 4;
    int blocks = (int)((total_uint32 + threadsPerBlock - 1) / threadsPerBlock);
    memset_int<<<blocks, threadsPerBlock, 0, stream>>>(
        (uint32_t*)my_dest_buf, (int)total_uint32, UBX_LAMPORT_INT);
  }

  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms(
      (const void*)ubx_kernel_combine_bf16_bf16_lamport_push, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int total_experts = ranks * experts_per_rank;
  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_local_n     = local_ntokens;
  int    arg_n_recv      = n_recv;
  int    arg_bpt         = blocks_per_token;
  int    arg_topk_max    = topk_max;
  int    arg_total_e     = total_experts;
  void*  arg_inv         = inverse_map;
  void*  arg_topk_idx    = topk_idx;
  void*  arg_gates       = gate_weights;
  int    arg_max_tpr     = max_tokens_per_rank;
  int64_t    arg_loff_dest   = lineoffset_dest;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_in          = ptr_in_bf16;
  void*  arg_out         = ptr_out_bf16;
  void*  arg_clear       = clear_ptr;
  int    arg_warmup      = skip_warmup_barrier ? 0 : 1;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_local_n, &arg_n_recv,
    &arg_bpt, &arg_topk_max, &arg_total_e,
    &arg_inv, &arg_topk_idx, &arg_gates,
    &arg_max_tpr, &arg_loff_dest,
    &arg_uc_flag, &arg_in, &arg_out, &arg_clear, &arg_warmup,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, (void*)ubx_kernel_combine_bf16_bf16_lamport_push, kernelArgs));
}

// ----------------------------------------------------------------------------
// PUSH non-Lamport launcher (barrier-based combine).
//
// No memset/poison logic — Phase 2 reads after a cross-rank barrier so the
// dest buf is fully populated.  Caller arranges double buffering in Python
// (combine_push_double_buf): call N writes/reads bufs[N%2]; call N+1 uses
// the alternate buf so peer's call N+1 Phase 1 push doesn't race with my
// call N Phase 2 read.
// ----------------------------------------------------------------------------
extern "C" void ubx_combine_bf16_bf16_push(
    int ranks, int myrank, int local_ntokens, int n_recv,
    int blocks_per_token, int experts_per_rank, int max_tokens_per_rank,
    int topk_max,
    uintptr_t inverse_map_arg, uintptr_t topk_idx_arg, uintptr_t gate_weights_arg,
    int64_t lineoffset_dest,
    ncclDevComm_t const* /*devcomm*/, ncclWindow_t window, uintptr_t pool_ptr,
    uintptr_t ptr_in_bf16_arg, uintptr_t ptr_out_bf16_arg,
    int default_sms, int smlimit,
    cudaStream_t stream) {
  const size_t REG0_FLAG_OFFSET = (size_t)ranks * sizeof(void*);
  int*  uc_flagptr     = reinterpret_cast<int*>(pool_ptr + REG0_FLAG_OFFSET);
  void* mc_flag        = nccl_lsa_mc_ptr(window, REG0_FLAG_OFFSET);
  void* inverse_map    = reinterpret_cast<void*>(inverse_map_arg);
  void* topk_idx       = reinterpret_cast<void*>(topk_idx_arg);
  void* gate_weights   = reinterpret_cast<void*>(gate_weights_arg);
  void* ptr_in_bf16    = reinterpret_cast<void*>(ptr_in_bf16_arg);
  void* ptr_out_bf16   = reinterpret_cast<void*>(ptr_out_bf16_arg);

  int sms      = default_sms ? default_sms : 128;
  int nthreads = UBX_MAXTHREADS;
  if (smlimit && sms > smlimit) sms = smlimit;
  sms = safe_resident_sms(
      (const void*)ubx_kernel_combine_bf16_bf16_push, nthreads, sms);
  SETUP_LAUNCH_CONFIG(sms, nthreads, stream, 1, 1);

  int total_experts = ranks * experts_per_rank;
  int    arg_ranks       = ranks;
  int    arg_myrank      = myrank;
  int    arg_local_n     = local_ntokens;
  int    arg_n_recv      = n_recv;
  int    arg_bpt         = blocks_per_token;
  int    arg_topk_max    = topk_max;
  int    arg_total_e     = total_experts;
  void*  arg_inv         = inverse_map;
  void*  arg_topk_idx    = topk_idx;
  void*  arg_gates       = gate_weights;
  int    arg_max_tpr     = max_tokens_per_rank;
  int64_t    arg_loff_dest   = lineoffset_dest;
  void*  arg_uc_flag     = uc_flagptr;
  void*  arg_mc_flag     = mc_flag;
  void*  arg_in          = ptr_in_bf16;
  void*  arg_out         = ptr_out_bf16;
  ncclWindow_t arg_window = window;

  void* kernelArgs[] = {
    &arg_ranks, &arg_myrank, &arg_local_n, &arg_n_recv,
    &arg_bpt, &arg_topk_max, &arg_total_e,
    &arg_inv, &arg_topk_idx, &arg_gates,
    &arg_max_tpr, &arg_loff_dest,
    &arg_uc_flag, &arg_mc_flag,
    &arg_in, &arg_out,
    &arg_window
  };
  CUDACHECK(cudaLaunchKernelExC(
      &cfg, (void*)ubx_kernel_combine_bf16_bf16_push, kernelArgs));
}
