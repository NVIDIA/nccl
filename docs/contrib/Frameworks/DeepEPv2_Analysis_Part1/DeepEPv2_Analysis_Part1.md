# DeepEPv2 Analysis (Part 1)

*Original by zartbot · Translated to English*

## TL;DR

There was a previous analysis of the older DeepEP — "An Analysis of EP Parallelism and DeepSeek's
Open-Source DeepEP Code". DeepEPv2 has updated quite a lot. On the communication side, it now
extends to support Engram, PP, CP, and AGRS (All-Gather and Reduce-Scatter), so the name has changed
to DeepEveryParallel. V1 used NVSHMEM; V2 switches to the NCCL GIN backend, dramatically reducing SM
occupancy and improving scalability, while introducing `ElasticBuffer`. It currently supports a
pure-GPU backend, with CPU and hybrid backends planned.

Because the writeup is long, it is split in two. Part 1 covers the `ElasticBuffer` structure, the
memory layout, and other non-EP features; Part 2 will dive into dispatch/combine in EP parallelism.
Outline below:

1. [`ElasticBuffer`](#1-elasticbuffer)
   1. [Buffer initialization](#11-buffer-initialization)
   2. [Memory layout](#12-memory-layout)
2. [Barrier implementation](#2-barrier-implementation)
   1. [Function call](#21-function-call)
   2. [Kernel body](#22-kernel-body)
   3. [Scale-up barrier](#23-scale-up-barrier)
   4. [Scale-out barrier](#24-scale-out-barrier)
   5. [Hybrid barrier](#25-hybrid-barrier)
3. [PP parallel communication](#3-pp-parallel-communication)
   1. [Initial configuration](#31-initial-configuration)
   2. [Function call](#32-function-call)
   3. [Buffer layout](#33-buffer-layout)
   4. [Send flow](#34-send-flow)
   5. [Recv flow](#35-recv-flow)
4. [Engram](#4-engram)
   1. [Buffer layout](#41-buffer-layout)
   2. [`engram_write`](#42-engram_write)
   3. [`engram_fetch`](#43-engram_fetch)
5. [AGRS](#5-agrs)
   1. [Buffer layout](#51-buffer-layout)
   2. [Session-based context management](#52-session-based-context-management)
   3. [all-gather flow](#53-all-gather-flow)

## 1. `ElasticBuffer`

### 1.1 Buffer initialization

DeepEPv2 unifies its initialization on top of Symmetric Memory and the NCCL Gin backend. The whole
flow is summarized below; for the underlying NCCL Gin material, see the companion piece
*"[NCCL Gin & Symmetric Memory](../../GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md)".*

Concretely: the code first calls `get_nccl_comm_handle(group)` to obtain the raw NCCL comm pointer,
then uses `calculate_elastic_buffer_size` to estimate the byte size required for the buffer. In the
C++ backend, `ElasticBuffer` instantiates an `NCCLSymmetricMemoryContext` to construct the symmetric
memory. It calls `ncclCommQueryProperties` to check Gin availability, then chooses whether to support
multi-plane and multi-rail topologies based on the actual physical layout.

On the RDMA side, the code computes `num_allocated_qps`. First, one dedicated notify QP is required.
In hybrid mode with fast-RDMA-atomic support, `64 channel QPs + 1 notify QP = 65` QPs total are
needed. Without fast-RDMA-atomic, the channel QPs double, giving `64×2 + 1 = 129`. Looking at
`check_fast_rdma_atomic_support`, only CX7 (MT4131) and newer NICs qualify. These results then drive
the `devComm` creation:

```cpp
ncclDevCommRequirements_t reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
if (num_ranks > 1 and !EP_DISABLE_GIN) {
  reqs.ginContextCount      = num_allocated_qps;        // QP pool size
  reqs.ginExclusiveContexts = true;                     // exclusive QPs
  reqs.ginQueueDepth        = 1024;
  reqs.ginTrafficClass      = sl_idx;                   // RDMA Service Level
  reqs.ginSignalCount       = num_ranks + 2 * 2;        // extra signals used by custom barriers
  reqs.ginConnectionType    = allow_hybrid_mode ? NCCL_GIN_CONNECTION_RAIL : NCCL_GIN_CONNECTION_FULL;
}
ncclDevCommCreate(comm, &reqs, &dev_comm);
```

Next, scale-up and scale-out rank distributions are derived from LSA rank and size:

```cpp
// Physical
num_nvl_ranks  = dev_comm.lsaSize;
nvl_rank_idx   = dev_comm.lsaRank;
num_rdma_ranks = num_ranks / num_nvl_ranks;
rdma_rank_idx  = rank_idx / num_nvl_ranks;
EP_HOST_ASSERT(num_ranks % num_nvl_ranks == 0
               and nvl_rank_idx == rank_idx % num_nvl_ranks);
EP_HOST_ASSERT(rank_idx == rdma_rank_idx * num_nvl_ranks + nvl_rank_idx);

// Logical
if (allow_hybrid_mode) {
  num_scaleout_ranks = num_rdma_ranks;   scaleout_rank_idx = rdma_rank_idx;
  num_scaleup_ranks  = num_nvl_ranks;    scaleup_rank_idx  = nvl_rank_idx;
} else {
  num_scaleout_ranks = 1;                scaleout_rank_idx = 0;
  num_scaleup_ranks  = num_ranks;        scaleup_rank_idx  = rank_idx;
}
is_scaleup_nvlink = num_scaleup_ranks == num_nvl_ranks;
```

Symmetric-memory allocation and `CommWindow` registration are then delegated directly to NCCL:

```cpp
NCCL_CHECK(ncclMemAlloc(&raw_window_ptr, size));                        // allocate physical memory
NCCL_CHECK(ncclCommWindowRegister(comm, raw_window_ptr, size, &window,  // register cross-rank window
                                  NCCL_WIN_DEFAULT));
```

Then the NVLink-peer pointers are resolved:

```cpp
ncclGetLsaDevicePointer(window, 0, nvl_rank_idx, &mapped_window_ptr);   // local rank's window base
nvl_window_ptrs.resize(num_nvl_ranks);
for (int i = 0; i < num_nvl_ranks; ++i)
    ncclGetLsaDevicePointer(window, 0, i, &nvl_window_ptrs[i]);         // every NVLink peer base
```

* `mapped_window_ptr`: the base address used by every subsequent operation on this rank.
* `nvl_window_ptrs[i]`: the local-VA address that maps to NVL peer `i`'s window base. Subsequent
  `get_sym_ptr(local, peer)` simply computes `peer_base + (local - mapped_window_ptr)`.

The resulting fields:

| Field                                      | Type            | Purpose                                                            |
| ------------------------------------------ | --------------- | ------------------------------------------------------------------ |
| `comm`                                     | `ncclComm_t`    | host-side NCCL comm.                                               |
| `dev_comm`                                 | `ncclDevComm`   | device-side comm (carries the GIN context pool).                   |
| `rank_idx` / `num_ranks`                   | `int`           | global coordinates.                                                |
| `nvl_rank_idx` / `num_nvl_ranks`           | `int`           | NVLink physical-domain coordinates (from `dev_comm.lsaRank/Size`). |
| `rdma_rank_idx` / `num_rdma_ranks`         | `int`           | RDMA physical-domain coordinates.                                  |
| `scaleout_rank_idx` / `num_scaleout_ranks` | `int`           | logical scale-out domain.                                          |
| `scaleup_rank_idx` / `num_scaleup_ranks`   | `int`           | logical scale-up domain.                                           |
| `is_scaleup_nvlink`                        | `bool`          | whether scale-up is exactly the NVLink domain.                     |
| `num_allocated_qps`                        | `int`           | number of reserved QPs.                                            |
| `raw_window_ptr`                           | `void*`         | raw address returned by `ncclMemAlloc`.                            |
| `window`                                   | `ncclWindow_t`  | registered window handle.                                          |
| `mapped_window_ptr`                        | `void*`         | base used by this rank to access the window.                       |
| `nvl_window_ptrs`                          | `vector<void*>` | same-offset bases for every NVL peer (used by `get_sym_ptr`).      |

### 1.2 Memory layout

The memory is split between `workspace` and `buffer`:

```cpp
workspace = this->nccl_context->mapped_window_ptr;
workspace_layout_wo_expert = std::make_shared<layout::WorkspaceLayout>(
    workspace, nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks, /*num_experts=*/0);
buffer = static_cast<uint8_t*>(workspace) + layout::WorkspaceLayout::get_num_bytes();
```

`workspace_layout_wo_expert` is a "layout without expert info" (`num_experts=0`). At construction time
the MoE expert count is unknown; APIs that need expert-related offsets (e.g. dispatch) will build a
`WorkspaceLayout` with experts at runtime. All non-expert regions (barrier, rank count, channel
metadata, PP, AGRS signals) are already available here. The 11 regions are:

| #   | Offset | region | bytes                                     |
| --- | -- | ----------------------------- | ---------------------- |
| 1   | 0  | NVL barrier counter + signals | 16                     |
| 2   | 16 | notify reduction `workspace` | `(1024+2048)*8 = 24576` |
| 3   | …  | scaleup rank count send/recv | `1024*8*2 = 16384`      |
| 4   | …  | scaleup expert count send/recv | `2048*8*2 = 32768`    |
| 5   | …  | scaleup atomic sender counter | `1024*4 = 4096`        |
| 6   | …  | scaleout rank count send/recv | `1024*4*2 = 8192`      |
| 7   | …  | scaleout expert count send/recv | `2048*4*2 = 16384`   |
| 8   | …  | scaleout channel metadata | `1024*1280*8 ≈ 10 MB`      |
| 9   | …  | channel scaleup tail | `1024*1280*4 ≈ 5 MB`            |
| 10  | …  | PP send/recv counts | `2*2*8 = 32`                     |
| 11  | …  | AGRS signals | `(32+1)*1024*4 = 135168`                |
| —   |    | align to 32 B |                                        |

In addition, `WorkspaceLayout` creates a host-side mapping:

```cpp
// Allocate host workspaces
CUDA_RUNTIME_CHECK(cudaMallocHost(&host_workspace, layout::WorkspaceLayout::get_num_bytes()));
CUDA_RUNTIME_CHECK(cudaHostGetDevicePointer(&mapped_host_workspace, host_workspace, 0));
```

The main `buffer` is shared by several primitives, partitioned into dynamically-sized sub-regions
whose carving is determined by which API is currently running.

## 2. Barrier implementation

### 2.1 Function call

The Python entry point is `ElasticBuffer.barrier`:

```python
def barrier(self, use_comm_stream: bool = True, with_cpu_sync: bool = False) -> None:
    self.runtime.barrier(use_comm_stream, with_cpu_sync)
```

Two switches:

* `use_comm_stream`: when `True`, the barrier kernel runs on `comm_stream`; otherwise on the current
  compute stream.
* `with_cpu_sync`: when `True`, `cudaDeviceSynchronize()` is added before and after the barrier.

The C++ implementation: this layer manages stream-level coordination only; the actual cross-rank
synchronization happens entirely inside the kernel.

```cpp
void barrier(const bool& use_comm_stream, const bool& with_cpu_sync) const {
  const auto compute_stream = at::cuda::getCurrentCUDAStream();
  const auto stream = use_comm_stream ? comm_stream : compute_stream;

  // Make comm_stream wait for already-queued work on compute_stream
  if (use_comm_stream)
    stream_wait(comm_stream, compute_stream);

  // CPU sync
  if (with_cpu_sync)
    cudaDeviceSynchronize();

  // Launch barrier kernel
  launch_barrier(nccl_context->dev_comm, nccl_context->window,
                 workspace,
                 nccl_context->scaleout_rank_idx, nccl_context->scaleup_rank_idx,
                 nccl_context->num_scaleout_ranks, nccl_context->num_scaleup_ranks,
                 num_gpu_timeout_cycles,
                 nccl_context->is_scaleup_nvlink,
                 stream);

  // CPU sync
  if (with_cpu_sync)
    cudaDeviceSynchronize();

  // compute_stream waits for comm_stream on return
  if (use_comm_stream)
    stream_wait(compute_stream, comm_stream);
}
```

### 2.2 Kernel body

When launching the barrier kernel, the SM count and thread count are determined as follows:

* 1 or 2 SMs: pure scale-up (single node) needs only 1 SM; hybrid (multi-node) needs 2 SMs to run
  scale-up + scale-out barriers in parallel.
* 512 threads/block: satisfies `kNumRanks <= kNumThreads` (one thread per rank to write a signal).
* `cooperative=true`: the kernel uses `this_grid().sync()` for cross-SM sync, so a cooperative launch is
  required.
* JIT compile: every dimension
  (`num_scaleout_ranks`/`num_scaleup_ranks`/`is_scaleup_nvlink`/`num_timeout_cycles`) is a template
  parameter, so the same (shape, topology) is compiled exactly once.

```cpp
constexpr auto kNumThreads = 512;
const auto num_sms = num_scaleout_ranks > 1 ? 2 : 1;  // hybrid needs 2 SMs
const BarrierRuntime::Args args = {
  .is_scaleup_nvlink = is_scaleup_nvlink,
  .num_scaleout_ranks = num_scaleout_ranks,
  .num_scaleup_ranks = num_scaleup_ranks,
  .num_timeout_cycles = num_timeout_cycles,
  .nccl_dev_comm = nccl_dev_comm,
  .nccl_window = nccl_window,
  .workspace = workspace,
  .scaleout_rank_idx = scaleout_rank_idx,
  .scaleup_rank_idx = scaleup_rank_idx,
  // cluster_dim=1, cooperative=true (need grid.sync())
  .launch_args = jit::LaunchArgs(num_sms, kNumThreads, 0, 1, true),
};
const auto code = BarrierRuntime::generate(args);
const auto runtime = jit::compiler->build("barrier", code);
BarrierRuntime::launch(runtime, args, stream);
```

The actual implementation lives in `/deep_ep/include/deep_ep/impls/barrier.cuh`:

```cpp
template <bool kIsScaleupNVLink, int kNumSMs, int kNumThreads,
          int kNumScaleoutRanks, int kNumScaleupRanks,
          int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(kNumThreads, 1)
barrier_impl(...) {
  const auto workspace_layout = layout::WorkspaceLayout(workspace, kNumScaleoutRanks, kNumScaleupRanks, 0);
  const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0);
  comm::gpu_barrier<kIsScaleupNVLink, kNumScaleoutRanks, kNumScaleupRanks,
                    kNumSMs, kNumThreads,
                    comm::kFlushAllAllocatedQPs,  // flush every QP
                    kNumTimeoutCycles,
                    comm::kKernelBarrierTag,      // tag=1
                    false, false, false>(         // kFlushStores/kSyncAtStart/kSyncAtEnd
      gin, workspace_layout, scaleout_rank_idx, scaleup_rank_idx, sm_idx, thread_idx);
}
```

This calls `comm::gpu_barrier`. Note three key template parameters:

* `kFlushStores = false`: the barrier itself does not write business data, so no `tma_store_commit`/`wait`
  is needed.
* `kSyncAtStart = false`: kernel entry doesn't need `grid.sync()` (no pending stores to publish).
* `kSyncAtEnd = false`: no enforced grid sync at exit either (CPU-side `cudaDeviceSynchronize` or stream
  events take over).

```cpp
do_scaleout &= kNumScaleoutRanks > 1;  // multi-node only
do_scaleup  &= kNumScaleupRanks  > 1;  // not needed for a single GPU

if (do_scaleup && do_scaleout) {
  // Hybrid: two SMs run in parallel
  if (sm_idx == 0) scaleup_barrier_wo_local_sync(...);   // SM0 runs scale-up
  else             scaleout_barrier_wo_local_sync(...);  // SM1 runs scale-out
} else if (do_scaleup) { // single node / pure NVLink domain
  scaleup_barrier_wo_local_sync(...);
} else if (do_scaleout) { // pure RDMA
  scaleout_barrier_wo_local_sync(...);
}
```

So:

* The scale-up barrier uses symmetric signal variables in `workspace` (NVLink LD/ST) or the same-team
  GIN signal.
* The scale-out barrier uses cross-node team signals via GIN.
* They run over different physical channels (NVLink memory vs RDMA QP), and not interfering with
  each other is the key to maximizing throughput.

### 2.3 Scale-up barrier

There are two implementations:

```cpp
template <bool kIsScaleupNVLink, ...>
void scaleup_barrier_wo_local_sync(...) {
  if (kIsScaleupNVLink)
    nvlink_barrier_wo_local_sync<kNumScaleupRanks, ...>(...);   // use NVLink LD/ST
  else
    gin_barrier_wo_local_sync<..., ncclTeamTagWorld, ...>(...); // use GIN
}
```

First, `nvlink_barrier_wo_local_sync`, which uses the dedicated fields reserved in `WorkspaceLayout`:

```cpp
// First 8 B: phase counter (u64)
unsigned long long* get_nvl_barrier_counter_ptr() { return workspace; }
// Two 4 B signal slots immediately after: one for phase=0, one for phase=1
int* get_nvl_barrier_signal_ptr(int phase) { return workspace + (2 + phase) * sizeof(int); }
```

Algorithm flow:

```cpp
// Use only 1 SM
if (kNumSMs > 1 && sm_idx > 0) return;

// Read state: low bit is phase (0/1), next bit is sign
const int status = (*counter_ptr) & 3;
const int phase  = status & 1;
const int sign   = status >> 1;

// Each thread handles one rank: atomic add/sub into that peer's signal slot
if (thread_idx < kNumRanks) {
  auto dst_ptr = gin.get_sym_ptr<ncclTeamTagLsa>(
      workspace.get_nvl_barrier_signal_ptr(phase), thread_idx);
  ptx::red_add_rel_sys(dst_ptr, sign ? -1 : +1);  // release-semantic atomic add
}
__syncthreads();

// thread_idx=0 bumps counter by 1 (phase, sign rotate: 0->1->2->3->0)
if (thread_idx == 0)
  atomicAdd(counter_ptr, 1);

// Spin until this rank's signal slot reaches the target
const auto target = sign ? 0 : kNumRanks;
timeout_while<kNumTimeoutCycles>(thread_idx == 0, [=] {
  auto signal = ptx::ld_acquire_sys<int>(workspace.get_nvl_barrier_signal_ptr(phase));
  return signal == target;  // pass when target reached
});
```

First it uses Symmetric Memory LSA addressing: `gin.get_sym_ptr<ncclTeamTagLsa>(local_ptr, peer_rank)`
translates the local signal address into the virtual address of the corresponding offset in the
peer's LSA domain, allowing direct LD/ST over NVLink. N threads in parallel atomically add into the
same signal slot of N peers, so each peer sees its slot incremented `kNumRanks` times. A double-phase
scheme avoids ABA on consecutive barrier calls: the previous barrier used phase 0, the next one uses
phase 1.

As for sign reversal: previous round used +1, next round uses -1, so slots can be reused without
zeroing. Concretely:

* `sign=0`: add 1, wait until `== kNumRanks`.
* `sign=1`: add -1, wait until `== 0`.

The slot value bounces between `[0, kNumRanks]`, and the next barrier counts in reverse from the
current value — zero-cost reset. Memory ordering uses release/acquire semantics: `red_add_rel_sys` is
a system-scoped release atomic, `ld_acquire_sys` is a system-scoped acquire load. Combined they form a
happens-before relation that guarantees pre-barrier stores are visible to post-barrier loads.

The scale-up barrier can also operate in GIN mode using `gin_barrier_wo_local_sync`, primarily when
`is_scaleup_nvlink=false` — i.e. RDMA-only scenarios such as RTX 6000 Pro PCIe cards. These setups are
uncommon. The GIN call path is identical to the scale-out barrier described next.

### 2.4 Scale-out barrier

Scale-out runs through NCCL GIN's (GPU-Initiated Networking) signal API for cross-node RDMA
synchronization. `gin_barrier_wo_local_sync` begins with a flush phase:

```cpp
if constexpr (kFlushStores) {
  for (int i = global_warp_idx; i < num_qps; i += kNumSMs * kNumWarps) {
    ncclGin(dev_comm, i, NCCL_GIN_RESOURCE_SHARING_CTA)
        .flush(ncclCoopWarp());  // each warp flushes a batch of QP doorbells
  }
  (gridDim.x > 1) ? this_grid().sync() : __syncthreads();
}
```

`kNumQPs == kFlushAllAllocatedQPs(-1)` → at runtime takes `nccl_dev_comm.ginContextCount`, flushing
every QP. `ncclGin.flush(ncclCoopWarp())` lets a warp cooperatively force previously-posted work
(`puts`/`atomics`) to land at the destination, after which a grid-wide sync ensures every SM is done
flushing before entering the signal phase.

Note that the barrier itself runs with `kFlushStores=false` and skips this block. It's the EP
`dispatch`/`combine` paths that need it when calling `gpu_barrier` at the end.

The signal phase only runs on SM0:

```cpp
if (sm_idx == 0) {
  // team: world (fully-connected scale-out) or rail (multi-rail mode)
  const auto team = is_world ? ncclTeamWorld(dev_comm) : ncclTeamRail(dev_comm);
  const ncclGin gin(dev_comm, 0, NCCL_GIN_RESOURCE_SHARING_CTA);  // QP 0 dedicated

  // (a) send: each thread issues a signal to rank i in the team; signal_id = local rank_idx
  for (int i = thread_idx; i < kNumRanks; i += kNumThreads)
    gin.signal(team, i, ncclGin_SignalInc{static_cast<ncclGinSignal_t>(rank_idx)});

  // (b) wait: this rank checks the signals from each peer rank i in the team
  for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
    const auto shadow_ptr = gin.getSignalShadowPtr(i);
    const auto target = ++(*shadow_ptr);  // shadow accumulates locally; +1 per barrier

    const auto gdaki = static_cast<ncclGinGdakiGPUContext*>(gin._ginHandle) + gin.contextIndex;
    const auto signal_ptr = reinterpret_cast<uint64_t*>(
        __ldg(&gdaki->signals_table.buffer)) + i;
    timeout_while<kNumTimeoutCycles>([=](bool is_last_check) {
      auto signal = ptx::ld_acquire_sys<uint64_t>(signal_ptr);
      return signal >= target;  // pass when >= shadow
    });
  }
}
```

Team selection ties to the actual physical RDMA topology:

* `ncclTeamTagWorld`: for standard RDMA fully-connected scale-out → all ranks within scale-out.
* `ncclTeamTagRail`: for multi-rail deployments → all ranks within the same rail of scale-out.

On the send side, the dedicated notify QP (QP 0) is used, avoiding mixing with data QPs that would
cause stalls and latency. The barrier sends to all ranks, parallelized across threads.

On the receive side a shadow counter is used: `getSignalShadowPtr(i)` is the local shadow, incremented
before each comparison. The load uses `ld_acquire_sys` so that, once the signal reaches the target,
all preceding remote RDMA puts are guaranteed globally visible.

### 2.5 Hybrid barrier

On a typical cluster with both scale-up and scale-out connectivity, the design uses two SMs sending
in parallel:

```cpp
if (do_scaleup && do_scaleout) {
  if (sm_idx == 0) {
    // SM0: scale-up barrier (NVLink symmetric memory phase+sign protocol)
    scaleup_barrier_wo_local_sync<kIsScaleupNVLink, kNumScaleupRanks,
                                   kNumSMs, ...>(
        gin, workspace, scaleup_rank_idx, sm_idx, thread_idx);
    if constexpr (kFlushStores)
      this_grid().sync();  // align with SM1's flush->signal via grid sync
  } else {
    // SM1: scale-out barrier (GIN rail team signal)
    scaleout_barrier_wo_local_sync<kNumScaleoutRanks, kNumSMs - 1, ...>(
        gin, scaleout_rank_idx, scaleup_rank_idx, sm_idx - 1, thread_idx);
  }
}
```

## 3. PP parallel communication

### 3.1 Initial configuration

First, `get_pp_buffer_size_hint` estimates the buffer size. The two factors of 2 correspond to
*send/recv double-buffering and prev/next directions*, so each rank has `4 × inflight` slots in total:

```python
@staticmethod
def get_pp_buffer_size_hint(
    num_max_tensor_bytes: int, num_max_inflight_tensors: int,
) -> int:
    # Align with LDG.256
    num_max_tensor_bytes = align(num_max_tensor_bytes, 32)

    # PP ring communication, each rank's buffer needs 4 groups:
    # (send, recv) x (prev, next) x num_max_inflight_tensors x aligned bytes
    return num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2
```

Then `pp_set_config` performs configuration with (`num_max_tensor_bytes`, `num_max_inflight_tensors`), and
computes prev/next rank indexes at the same time:

```cpp
void pp_set_config(const int64_t &num_max_tensor_bytes, const int &num_max_inflight_tensors) {
  // Flush previous operations
  barrier(false, true);

  EP_HOST_ASSERT(num_max_tensor_bytes > 0 and num_max_inflight_tensors > 0);
  EP_HOST_ASSERT(num_max_tensor_bytes * num_max_inflight_tensors * 2 * 2 <= num_buffer_bytes);
  this->prev_rank_idx = (nccl_context->rank_idx + nccl_context->num_ranks - 1) % nccl_context->num_ranks;
  this->next_rank_idx = (nccl_context->rank_idx + 1) % nccl_context->num_ranks;
  this->num_max_pp_tensor_bytes = math::align<int64_t>(num_max_tensor_bytes, 32);
  this->num_max_pp_inflight_tensors = num_max_inflight_tensors;
}
```

`barrier(false, true)` (i.e. `use_comm_stream=false`, `with_cpu_sync=true`) runs on the current compute
stream and inserts `cudaDeviceSynchronize` before and after the barrier kernel. This drains every
previously inflight kernel and RDMA, pushing send/recv counters and Gin signals into a consistent
state. `pp_set_config` therefore must be called before the first PP communication, and every parameter
change pays the cost of a global barrier.

### 3.2 Function call

Once configuration is done, send/recv can be called:

```python
def pp_send(self, t: torch.Tensor, dst_rank_idx: int, num_sms: int = 0) -> None:
    self.runtime.pp_send(t, dst_rank_idx, num_sms)

def pp_recv(self, t: torch.Tensor, src_rank_idx: int, num_sms: int = 0) -> None:
    self.runtime.pp_recv(t, src_rank_idx, num_sms)
```

Three pre-flight checks: `pp_set_config` must have been called; the tensor must be valid and within
limits; the peer must be a ring-adjacent rank. `pp_recv` handling is fully symmetric, so we'll walk
through send below.

```cpp
void pp_send(const torch::Tensor& x, const int& dst_rank_idx, const int& num_sms) const {
  EP_HOST_ASSERT(num_max_pp_tensor_bytes > 0 and num_max_pp_inflight_tensors > 0);
  EP_HOST_ASSERT(x.is_cuda() and x.is_contiguous() and x.nbytes() <= num_max_pp_tensor_bytes);
  EP_HOST_ASSERT(dst_rank_idx == prev_rank_idx or dst_rank_idx == next_rank_idx);

  launch_pp_send(nccl_context->dev_comm, nccl_context->window,
                 x.data_ptr(), x.nbytes(),
                 buffer, workspace,
                 nccl_context->rank_idx, dst_rank_idx, nccl_context->num_ranks,
                 num_max_pp_tensor_bytes, num_max_pp_inflight_tensors,
                 // num_sms == 0 means use device_runtime->get_num_sms() -- all SMs
                 num_sms == 0 ? jit::device_runtime->get_num_sms() : num_sms,
                 num_gpu_timeout_cycles,
                 jit::device_runtime->get_num_smem_bytes(),
                 at::cuda::getCurrentCUDAStream());
}
```

`launch_pp_send` then JIT-compiles, instantiating the kernel with four template constants:
(`num_sms`, `num_ranks`, `smem_bytes`, `timeout_cycles`). `launch_args` is
`LaunchArgs(num_sms, 32, smem_bytes, 1, true)` — i.e. `num_sms` blocks × 32 threads (1 warp),
dynamic shared memory = `smem_bytes`, `cooperative=true`.

```cpp
static std::string generate_impl(const Args& args) {
  return fmt::format(R"(

#include <deep_ep/impls/pp_send_recv.cuh>
using namespace deep_ep::elastic;

static void __instantiate_kernel() {{
  auto ptr = reinterpret_cast<void*>(&pp_send_impl<{}, {}, {}, {}>);
}}
)",
                     args.launch_args.grid_dim.first,
                     args.num_ranks,
                     args.num_smem_bytes,
                     args.num_timeout_cycles);
}

// ...
const PPSendRuntime::Args args = {
  // ...
  .launch_args = jit::LaunchArgs(num_sms, 32, num_smem_bytes, 1, true),
};
const auto code = PPSendRuntime::generate(args);
const auto runtime = jit::compiler->build("pp_send", code);
PPSendRuntime::launch(runtime, args, stream);
```

### 3.3 Buffer layout

First, the helper below returns (`local_idx_in_dst`, `dst_idx_in_local`):

* When `dst == next` → `(0, 1)`: this rank looks like prev from the peer's perspective (uses slot 0 to
  receive); the peer looks like next from this rank's perspective (occupies local slot 1).
* When `dst == prev` → `(1, 0)`: symmetric.

```cpp
template <int kNumRanks>
__device__ __forceinline__ std::pair<int, int> get_buffer_offset(
  const int &src_rank_idx,
  const int &dst_rank_idx) {
  const auto next_rank_idx = (src_rank_idx + 1) % kNumRanks;
  return dst_rank_idx == next_rank_idx ? std::make_pair(0, 1) : std::make_pair(1, 0);
}
```

The return value drives the four `buffer` segments:

| Segment / coefficient | Expression | meaning                                                                               |
| --------------------- | ------------------------------------------------ | ----------------------------------------------- |
| 0                     | `(local_idx_in_dst + 0) * inflight` (next view 0) | recv area for data this rank receives from next. |
| 1                     | `(local_idx_in_dst + 0) * inflight` (prev view 1) | recv area for data this rank receives from prev. |
| 2                     | `(dst_idx_in_local + 2) * inflight` (prev view 0+2) | send buffer for data this rank sends to prev.  |
| 3                     | `(dst_idx_in_local + 2) * inflight` (next view 1+2) | send buffer for data this rank sends to next.  |

In addition, two counters live in `WorkspaceLayout`:

```cpp
__forceinline__ __device__ __host__ int64_t* get_pp_send_count_ptr(const int& offset) const {
  const auto base_ptr = math::advance_ptr<int64_t>(
    get_channel_scaleup_tail_ptr(0, 0),
    kNumMaxRanks * kNumMaxChannels * sizeof(int));
  return base_ptr + offset;
}
__forceinline__ __device__ __host__ int64_t* get_pp_recv_count_ptr(const int& offset) const {
  const auto base_ptr = math::advance_ptr<int64_t>(
    get_pp_send_count_ptr(0), 2 * sizeof(int64_t));
  return base_ptr + offset;
}
```

Two `int64`s each: `offset=0/1` correspond to prev/next sent/received counts. NCCL Gin signals are split
into two groups:

* `signal = kNumRanks + offset`: data-ready (the put on the sender carries a `SignalInc`; once raised,
  the receiver may consume).
* `signal = kNumRanks + offset + 2`: slot-release (after consumption the receiver issues `gin.signal`
  telling the sender the slot is free).

### 3.4 Send flow

The implementation is in `deep_ep/include/deep_ep/impls/pp_send_recv.cuh`:

```cpp
template <int kNumSMs, int kNumRanks, int kNumSmemBytes, int64_t kNumTimeoutCycles>
// __launch_bounds__(32, 1) means 32 threads per block (1 warp), matching launch_args' 32
__global__ void __launch_bounds__(32, 1)
pp_send_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,
             void* x, const int64_t num_x_bytes,
             void* buffer, void* workspace,
             const int rank_idx, const int dst_rank_idx,
             const int64_t num_max_tensor_bytes,
             const int num_max_inflight_tensors) {
  const auto sm_idx = static_cast<int>(blockIdx.x);
  const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);
  const auto [local_idx_in_dst, dst_idx_in_local] = get_buffer_offset<kNumRanks>(rank_idx, dst_rank_idx);

  // Gin handle
  const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

  // Buffer offsets
  const auto send_count_ptr = workspace_layout.get_pp_send_count_ptr(dst_idx_in_local);
  const auto send_count = __ldg(send_count_ptr);
  const auto slot_idx = send_count % num_max_inflight_tensors;
  auto send_buffer_ptr = math::advance_ptr(
      buffer, ((dst_idx_in_local + 2) * num_max_inflight_tensors + slot_idx) *
                  num_max_tensor_bytes);
  auto recv_buffer_ptr = math::advance_ptr(
      buffer, ((local_idx_in_dst + 0) * num_max_inflight_tensors + slot_idx) *
                  num_max_tensor_bytes);

  // Wait for buffer slot release and then do TMA
  if (ptx::elect_one_sync()) {
    check_signal<kNumTimeoutCycles>(
        gin,
        static_cast<ncclGinSignal_t>(kNumRanks + dst_idx_in_local + 2),
        send_count - num_max_inflight_tensors + 1,
        []() { printf("DeepEP PP send timeout, recv buffer is full"); }
    );
    tma_copy<kNumSMs, kNumSmemBytes>(x, send_buffer_ptr, num_x_bytes, sm_idx);
  }
  cooperative_groups::this_grid().sync();

  // Issue RDMA put
  if (sm_idx == 0 and ptx::elect_one_sync()) {
    gin.put<ncclTeamTagWorld>(
        recv_buffer_ptr,
        send_buffer_ptr,
        num_x_bytes, dst_rank_idx,
        0,
        ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));
    *send_count_ptr += 1;
  }
}
```

Step-by-step:

1. Locate the slot. `__ldg(send_count_ptr)` reads the local sent count (incremented after every
   successful put), `slot_idx = send_count % inflight` gives the ring-buffer write position.
2. Compute pointers on both sides:

   * `send_buffer_ptr` points to the local send staging segment (segment 2 or 3), chosen via
     `dst_idx_in_local + 2`.
   * `recv_buffer_ptr` points to the peer's recv segment (segment 0 or 1), chosen via
     `local_idx_in_dst + 0`. The `slot_idx` is the same on both sides, ensuring cross-rank alignment.
3. Reverse flow control (wait for slot release): `ptx::elect_one_sync()` picks a single warp lane to
   call `check_signal`:

   * `signal_idx = kNumRanks + dst_idx_in_local + 2` — the release signal.
   * `target = send_count - inflight + 1` — the slot is reusable only when the peer has released at
     least `send_count - inflight + 1` times.
   * Timeout prints `"DeepEP PP send timeout, recv buffer is full"`.

```cpp
template <int64_t kNumTimeoutCycles, typename timeout_print_t>
__device__ __forceinline__ void check_signal(
    const handle::NCCLGin& gin,
    const ncclGinSignal_t& signal_idx,
    const int64_t& target,
    const timeout_print_t& timeout_print) {
  const auto gdaki = static_cast<struct ncclGinGdakiGPUContext*>(gin.gin._ginHandle);
  const auto signal_ptr = reinterpret_cast<int64_t*>(
      __ldg(reinterpret_cast<int64_t*>(&gdaki->signals_table.buffer))) + signal_idx;
  comm::timeout_while<kNumTimeoutCycles>([=](const bool &is_last_check) {
    const auto signal = ptx::ld_acquire_sys<int64_t>(signal_ptr);
    if (signal >= target)
      return true;
    if (is_last_check)
      timeout_print();
    return false;
  });
}
```

4. Local TMA copy. `tma_copy` uses a `kNumStages=2` `mbarrier` pipeline to move the user tensor into
   `send_buffer_ptr` in stages. Work split: `num_tma_blocks = num_bytes / kNumTMAAlignBytes`, per-SM
   split `num_tma_blocks_per_sm = ceil_div(num_tma_blocks, kNumSMs)`; each SM independently pipelines
   its share.

```cpp
for (int64_t iter_idx = 0; iter_idx < num_iterations; ++iter_idx) {
  const auto stage_idx = static_cast<int>(iter_idx % kNumStages);
  const auto [store_offset, num_store_bytes] = get_iter_info(iter_idx);

  if (iter_idx < kNumStages) { // prime the pipeline: first kNumStages iters only issue load
    ptx::tma_load_1d(tma_buffers + stage_idx * kNumTMABytesPerStage,
                     math::advance_ptr(src_ptr, store_offset),
                     mbarriers + stage_idx, num_store_bytes);
    ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_store_bytes);
  }
  ptx::mbarrier_wait_and_flip_phase(mbarriers + stage_idx, phases[stage_idx]);
  ptx::tma_store_1d(math::advance_ptr(dst_ptr, store_offset),
                    tma_buffers + stage_idx * kNumTMABytesPerStage,
                    num_store_bytes);
  ptx::tma_store_commit();

  const auto next_iter_idx = iter_idx + kNumStages;
  if (next_iter_idx < num_iterations) { // prefetch: as soon as a stage frees, dispatch the next load
    ptx::tma_store_wait<kNumStages - 1>();
    const auto [load_offset, num_load_bytes] = get_iter_info(next_iter_idx);
    ptx::tma_load_1d(tma_buffers + stage_idx * kNumTMABytesPerStage,
                     math::advance_ptr(src_ptr, load_offset),
                     mbarriers + stage_idx, num_load_bytes);
    ptx::mbarrier_arrive_and_set_tx(mbarriers + stage_idx, num_load_bytes);
  }
}
ptx::tma_store_wait();
```

5. Barrier. Because TMA moves data asynchronously, `cooperative_groups::this_grid().sync()` waits for
   every SM's TMA stores to be globally visible before issuing the RDMA put — preventing the put
   from reading half-copied data.
6. RDMA PUT + signal.

   * The single thread on `sm_idx == 0` calls `gin.put(recv_buffer_ptr, send_buffer_ptr, num_x_bytes,
     dst_rank_idx, 0, SignalInc(local_idx_in_dst + kNumRanks))`.
   * NCCL Gin writes the local send segment into the peer's recv segment, then atomically bumps the
     data-ready signal (`signal_idx = kNumRanks + local_idx_in_dst`), which the receiver blocks on.
     Finally `*send_count_ptr += 1` advances the local count.

```cpp
// Issue RDMA put
if (sm_idx == 0 and ptx::elect_one_sync()) {
  gin.put<ncclTeamTagWorld>(
      recv_buffer_ptr,
      send_buffer_ptr,
      num_x_bytes, dst_rank_idx,
      0,
      // TODO: is this signal highly optimized?
      ncclGin_SignalInc(static_cast<ncclGinSignal_t>(local_idx_in_dst + kNumRanks)));
  *send_count_ptr += 1;
}
```

### 3.5 Recv flow

The receive side runs symmetrically, with a TMA copy after the RDMA arrival:

```cpp
// Template constants:
//   kNumSMs = blocks in the grid (each block 1 warp)
//   kNumRanks = ranks in the comm
//   kNumSmemBytes = dynamic shared memory available for the TMA pipeline
//   kNumTimeoutCycles = upper bound for the spin-wait timeout (in cycles)
// __launch_bounds__(32, 1): 32 threads per block = 1 warp; at least 1 resident block.
// Matches the 32-thread warp launch.
template <int kNumSMs, int kNumRanks, int kNumSmemBytes, int64_t kNumTimeoutCycles>
__global__ void __launch_bounds__(32, 1)
pp_recv_impl(const ncclDevComm_t nccl_dev_comm, const ncclWindow_t nccl_window,   // NCCL device handles
             void* x, int64_t num_x_bytes,                                        // user receive tensor
             void* buffer, void* workspace,                                       // buffer + workspace base
             const int rank_idx, const int src_rank_idx,                          // own rank, source rank
             const int64_t num_max_tensor_bytes,                                  // max single-slot bytes
             const int num_max_inflight_tensors) {                                // ring depth
  const auto sm_idx = static_cast<int>(blockIdx.x);

  // WorkspaceLayout helps locate counter offsets in workspace (here just the two pp_recv_count int64 slots).
  const auto workspace_layout = layout::WorkspaceLayout(workspace, 1, kNumRanks, 0);

  // Resolve "from src's perspective": the local recv segment (src_idx_in_local) and which segment we occupy on src
  //   src == next -> (0, 1): receive from next -> use local segment 0; we occupy segment 1 on src
  //   src == prev -> (1, 0): receive from prev -> use local segment 1; we occupy segment 0 on src
  const auto [src_idx_in_local, local_idx_in_src] = get_buffer_offset<kNumRanks>(src_rank_idx, rank_idx);

  // Build the NCCL Gin handle: CTA-level resource sharing; context 0 = default QP
  const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, 0, NCCL_GIN_RESOURCE_SHARING_CTA);

  // Read the local "received from src" count. Bumped by 1 per successful recv (see the function tail);
  // determines slot and expected signal value.
  const auto recv_count_ptr = workspace_layout.get_pp_recv_count_ptr(src_idx_in_local);
  const auto recv_count = __ldg(recv_count_ptr);
  // Ring-buffer modulo locates this slot, exactly matching the sender's send_count % inflight.
  const auto slot_idx = recv_count % num_max_inflight_tensors;
  // Local recv segment base: segment = src_idx_in_local + 0 (segments 0 or 1 are recv-only).
  // Offset = (segment * inflight + slot_idx) * aligned bytes.
  const auto recv_buffer_ptr = math::advance_ptr(
      buffer, ((src_idx_in_local + 0) * num_max_inflight_tensors + slot_idx) * num_max_tensor_bytes);

  // elect_one_sync: pick a single elected lane out of the 32-thread warp to drive single-thread logic.
  // The data-ready wait and the TMA dispatch are both driven by this lane (one thread is enough to issue a TMA descriptor).
  if (ptx::elect_one_sync()) {
    // Wait for the sender's put on this slot to complete:
    //   signal_idx = kNumRanks + src_idx_in_local (data-ready signal group)
    //   target     = recv_count + 1 (the sender bumps this signal by 1 per put)
    // Spin via ld_acquire_sys; on timeout, call timeout_print and continue.
    check_signal<kNumTimeoutCycles>(
        gin,
        static_cast<ncclGinSignal_t>(src_idx_in_local + kNumRanks),
        recv_count + 1,
        []() { printf("DeepEP PP recv timeout, recv buffer is empty\n"); });
    // TMA 1D copy: move the recv segment in the symmetric buffer into user tensor x.
    //   Internally chunked by kNumTMAAlignBytes and split across kNumSMs;
    //   each SM uses a 2-stage mbarrier pipeline (load -> wait -> store -> commit -> prefetch next),
    //   saturating throughput.
    tma_copy<kNumSMs, kNumSmemBytes>(recv_buffer_ptr, x, num_x_bytes, sm_idx);
  }

  // Grid-level barrier: wait for every block's TMA store to complete and become globally visible.
  // Must precede the release-signal so the sender can't reuse the slot before data has landed.
  cooperative_groups::this_grid().sync();

  // Only the leader of block 0 sends the "slot consumed, free to reuse" notification to the sender.
  if (sm_idx == 0 and ptx::elect_one_sync()) {
    // Atomically bump the release-signal to src_rank_idx:
    //   signal_idx = kNumRanks + local_idx_in_src + 2 (+2 distinguishes from data-ready)
    // The sender's pp_send_impl waits for this counter >= send_count - inflight + 1.
    gin.signal<ncclTeamTagWorld>(
        src_rank_idx, ncclGin_SignalInc(static_cast<ncclGinSignal_t>(kNumRanks + local_idx_in_src + 2)));
    // Advance the local count; next pp_recv computes slot/expected signal from the new value.
    *recv_count_ptr += 1;
  }
}
```

## 4. Engram

### 4.1 Buffer layout

`get_engram_storage_size_hint` computes the buffer size, parameterized by entry count
`num_engram_entries` and input-token hidden dim `engram_hidden`:

```python
num_sf_packs = ceil_div(hidden, 32) if dtype.itemsize <= 1 else 0  # FP8: reserve scale
num_bytes_per_entry = align(hidden * dtype.itemsize + num_sf_packs * 4, 32)
return num_bytes_per_entry * (num_entries + num_max_tokens_per_rank)
#                             ^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^
#                             storage area  + reserved fetch receive area
```

The `buffer` is split into two parts:

* The storage area, which depending on `dtype` reserves space for FP8 scale.
* The recv area is the rest of the `buffer`, and must satisfy
  `num_tokens * hidden * 2 <= num_engram_recv_bytes`.

| Field                             | Meaning                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------- |
| `num_buffer_bytes`                | Total bytes in the buffer area.                                               |
| `buffer`                          | Start of the buffer (symmetric address; same offset on local and every peer). |
| `num_engram_entries`, `engram_hidden` | Number of entries on this rank, plus hidden.                              |
| `num_engram_storage_bytes`        | Bytes occupied by storage written (aligned to 32 B).                          |
| `num_engram_recv_bytes`           | `num_buffer_bytes - num_engram_storage_bytes` — used for fetch writes.        |

### 4.2 `engram_write`

`engram_write` is straightforward — an asynchronous copy, in four steps:

1. `barrier(false, true)` — i.e. `use_comm_stream=false`, `with_cpu_sync=true` — runs on `compute_stream`
   and inserts `cudaDeviceSynchronize` before/after the barrier kernel, ensuring all RDMA gets from
   the previous `engram_fetch` are complete and that no peer is still reading the old storage when it
   gets overwritten.
2. Validate storage:

```cpp
EP_HOST_ASSERT(storage.scalar_type() == torch::kBFloat16);
EP_HOST_ASSERT(storage.is_cuda() and storage.is_contiguous());
num_engram_entries = num_entries;
engram_hidden = hidden;
EP_HOST_ASSERT(storage.nbytes() <= num_buffer_bytes);
```

3. Async copy. Because `buffer` was registered as symmetric memory by `ncclCommWindowRegister`, other
   ranks can read this address through `gin.get`:

```cpp
cudaMemcpyAsync(buffer, storage.data_ptr(), storage.nbytes(), cudaMemcpyDeviceToDevice,
                compute_stream);
num_engram_storage_bytes = align<int64_t>(storage.nbytes(), 32);
num_engram_recv_bytes    = num_buffer_bytes - num_engram_storage_bytes;
```

4. Another barrier, ensuring the symmetric-window contents are fresh and observed by all peers.

### 4.3 `engram_fetch`

First validate parameters:

```cpp
EP_HOST_ASSERT(indices.scalar_type() == torch::kInt);
EP_HOST_ASSERT(num_tokens * engram_hidden * sizeof(nv_bfloat16)
               <= num_engram_recv_bytes); // recv area large enough
if (num_qps == 0) num_qps = nccl_context->num_allocated_qps;
EP_HOST_ASSERT(num_engram_entries > 0); // engram_write must have happened first
```

Then construct the fetched tensor view:

```cpp
const auto fetched = torch::from_blob(
    math::advance_ptr(buffer, num_engram_storage_bytes),  // start of recv area
    {num_tokens, engram_hidden},
    torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCUDA));
```

Allocate `last_gin_requests`, sized `num_ranks × num_qps` — each (QP, peer) pair holds the handle of the
last aggregated request issued, used for wait:

```cpp
const auto last_gin_requests = torch::empty(
    {nccl_context->num_ranks * num_qps, sizeof(ncclGinRequest_t)}, ...);
```

Issue kernel via `launch_engram_fetch` — JIT generates and builds `engram_fetch_impl`, launched on the
current CUDA stream with `grid = num_qps`, `threads = 1024`. The template parameters of
`engram_fetch_impl` are
`<kNumQPs, kNumEntriesPerRank, kHidden, kNumRanks, kNumThreads=1024, kNumWarps = kNumThreads / 32>`,
and the launch is `grid=kNumQPs`, `block=kNumThreads` — i.e. one block per QP, 32
blocks total. Execution flow:

1. Initialization:

```cpp
const auto qp_idx = blockIdx.x;
const auto warp_idx = ptx::get_warp_idx();
const auto global_warp_idx = qp_idx * kNumWarps + warp_idx;
const auto gin = handle::NCCLGin(nccl_dev_comm, nccl_window, qp_idx,
                                 NCCL_GIN_RESOURCE_SHARING_CTA);

__shared__ bool sent_to_rank[kNumRanks];  // tracks which peers this QP issued a get to
```

2. Issue read requests via the leader thread. `indices[i]` is a global entry id (range
   `[0, num_ranks * num_entries)`), split via integer divide and mod into (`src_rank`, `src_entry`):

```cpp
if (ptx::elect_one_sync()) {
  for (int i = global_warp_idx; i < num_tokens; i += kNumQPs * kNumWarps) {
    const auto global_idx    = __ldg(indices + i);
    const auto src_rank_idx  = global_idx / kNumEntriesPerRank;
    const auto src_entry_idx = global_idx % kNumEntriesPerRank;

    // src: peer buffer's storage segment, entry src_entry_idx
    // dst: local buffer's recv segment, slot i
    gin.get<team_t>(
        advance_ptr(storage, src_entry_idx * kNumHiddenBytes),
        advance_ptr(fetched, i             * kNumHiddenBytes),
        kNumHiddenBytes, src_rank_idx,
        ncclGinOptFlagsAggregateRequests);  // aggregate; don't ring DB immediately
    sent_to_rank[src_rank_idx] = true;
  }
}
__syncthreads();
```

3. Because these are reads, run `flush_async` per peer next. It posts every previously aggregated
   read to the NIC and writes the handle of "the last request" into `last_gin_requests`, triggering
   the aggregated DB ring:

```cpp
if (ptx::elect_one_sync()) {
  for (int i = warp_idx; i < kNumRanks; i += kNumWarps) {
    auto* request_ptr = last_gin_requests + qp_idx * kNumRanks + i;
    if (sent_to_rank[i]) {
      gin.flush_async<team_t>(i, request_ptr);  // returns a waitable request
    } else {
      *reinterpret_cast<int4*>(request_ptr) = make_int4(0, 0, 0, 0);
    }
  }
}
```

Finally, wait for the data to come back:

```cpp
for (int i = thread_idx; i < kNumRanks; i += kNumThreads) {
  auto v4 = __ldg(reinterpret_cast<int4*>(last_gin_requests + qp_idx * kNumRanks + i));
  if (v4.x | v4.y | v4.z | v4.w) {  // peer has a real request
    auto req = *reinterpret_cast<ncclGinRequest_t*>(&v4);
    gin.wait(req);  // block until complete
  }
}
```

## 5. AGRS

Despite the name ("all-gather reduce-scatter"), AGRS confines itself to the NVLink domain in
exchange for DMA-only, zero-SM peak performance. For cross-node scenarios that include RDMA, fall
back to NCCL all-gather directly.

### 5.1 Buffer layout

First, `get_agrs_buffer_size_hint` computes the buffer size and returns `num_max_session_bytes` — the
sum of "`per-rank session bytes × num_ranks × num_max_inflight_agrs`":

```python
num_max_session_bytes = deep_ep.ElasticBuffer.get_agrs_buffer_size_hint(
    group, num_bytes_per_tensor * group.size() * num_max_inflight_agrs
)
buffer = deep_ep.ElasticBuffer(group, explicitly_destroy=True, num_bytes=num_max_session_bytes)
buffer.agrs_set_config(num_max_session_bytes, num_max_inflight_agrs)
```

The buffer is created via `deep_ep.ElasticBuffer`. `WorkspaceLayout` contains:

* AGRS recv signals (`kNumMaxInflightAGRS × kNumMaxRanks = 32 × 1024` entries).
* AGRS session signals (`kNumMaxRanks` entries). The device side fetches pointers via:

```cpp
// AGRS recv signal: each all_gather owns a slot for slot-level sync
__forceinline__ __device__ __host__ int* get_agrs_recv_signal_ptr const int& slot, const int& rank_idx) const {
  const auto base_ptr = math::advance_ptr<int>(
      get_pp_recv_count_ptr(0), 2 * sizeof(int64_t));
  return base_ptr + slot * kNumMaxRanks + rank_idx;
}

// AGRS session signal: one int per rank for session-level sync
__forceinline__ __device__ __host__ int* get_agrs_session_signal_ptr(const int& rank_idx) const {
  const auto base_ptr = math::advance_ptr<int>(
      get_agrs_recv_signal_ptr(0, 0), kNumMaxInflightAGRS * kNumMaxRanks * sizeof(int));
  return base_ptr + rank_idx;
}
```

AGRS uses a buffer slice (`= num_max_session_bytes`). The session buffer is advanced by two
cooperating fields: `agrs_buffer_offset` (byte cursor) and `agrs_buffer_slot_idx` (slot counter).
`agrs_set_config` runs:

```cpp
barrier(true, true);  // flush every prior op
EP_HOST_ASSERT(num_max_session_bytes > 0 and new_num_max_agrs_per_session > 0);
EP_HOST_ASSERT(num_max_session_bytes <= num_buffer_bytes);
// Up to 32 all-gathers per session, sets the AGRS signal-slot count in workspace
EP_HOST_ASSERT(new_num_max_agrs_per_session <= layout::WorkspaceLayout::kNumMaxInflightAGRS);

// AGRS is single-node NVLink only
EP_HOST_ASSERT(nccl_context->num_nvl_ranks == nccl_context->num_ranks);
this->num_max_agrs_session_bytes = math::align<int64_t>(num_max_session_bytes, 32);  // LDG.256 alignment
this->num_max_agrs_per_session = new_num_max_agrs_per_session;
```

### 5.2 Session-based context management

`create_agrs_session` only mutates host-side state and issues no GPU op. `agrs_session_idx` is
monotonically increasing and never resets — that lets the recv signal value itself distinguish stale
writes from previous sessions. There is no barrier here either: it relies on the previous
`destroy_agrs_session`'s session-completion signal to ensure peers have consumed their buffer.

```cpp
void create_agrs_session() {
  EP_HOST_ASSERT(not agrs_in_session);  // no nesting
  agrs_in_session      = true;
  agrs_buffer_offset   = 0;  // restart partitioning at the buffer start
  agrs_buffer_slot_idx = 0;  // reset slot counter
  agrs_session_idx    += 1;  // monotonically increasing global session id
}
```

`destroy_agrs_session` works as follows:

```cpp
void destroy_agrs_session() {
  EP_HOST_ASSERT(agrs_in_session);
  agrs_in_session = false;

  // 1. comm_stream waits for compute_stream -- ensures user-side consume is done
  stream_wait(comm_stream, at::cuda::getCurrentCUDAStream());

  // 2. for the N-1 peers, do a "write peer's signal + wait for peer's signal" batch
  std::vector<void*> write_ptrs(num_ranks - 1), wait_ptrs(num_ranks - 1);
  for (int i = 0; i < num_ranks - 1; ++i) {
    int dst = (rank_idx + i + 1) % num_ranks;
    // write into peer's workspace at "this rank's session signal"
    write_ptrs[i] = get_sym_ptr(session_signal_ptr(rank_idx), dst);
    // wait on peer's session signal in our local workspace
    wait_ptrs[i] = session_signal_ptr(dst);
  }
  cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, agrs_session_idx);
}
```

The last line uses `cuStreamBatchMemOp` to submit a group of operations as a single driver-level
command:

```cpp
void batched_write_and_wait(CUstream stream, const std::vector<void*>& write_ptrs,
                            const std::vector<void*>& wait_ptrs, uint32_t value) {
  std::vector<CUstreamBatchMemOpParams> ops(write_ptrs.size() + wait_ptrs.size());
  for (int i = 0; i < write_ptrs.size(); ++i)
    ops[i] = create_mem_op(write_ptrs[i], value, CU_STREAM_MEM_OP_WRITE_VALUE_32);
  for (int i = 0; i < wait_ptrs.size(); ++i)
    ops[write_ptrs.size() + i] = create_mem_op(wait_ptrs[i], value, CU_STREAM_MEM_OP_WAIT_VALUE_32_GEQ);
  CUDA_DRIVER_CHECK(lazy_cuStreamBatchMemOp(stream, ops.size(), ops.data(), 0));
}
```

* `CU_STREAM_MEM_OP_WRITE_VALUE_32`: writes `agrs_session_idx` into the peer's session-signal location.
* `CU_STREAM_MEM_OP_WAIT_VALUE_32 + WAIT_VALUE_GEQ`: spin-waits on `comm_stream` until the local signal
  is `>= agrs_session_idx` (`GEQ` accommodates higher session ids that arrive later).

destroy is effectively a session-level barrier:

1. Tell every peer "I'm no longer reading session X's `buffer`".
2. Wait for every peer to tell me the same.

From this point onward, the `buffer` area used by session X is reusable for every rank. This is why
`create_agrs_session` doesn't need a barrier itself — the previous `destroy` already provides the "safe
to reuse after write" guarantee, which is precisely what makes resetting `agrs_buffer_offset=0` safe.

### 5.3 all-gather flow

Take a typical caller from `test_agrs.py`:

```python
with buffer.agrs_new_session():
    out_tensors, handle = do_all_gather(buffer, is_inplace, is_batched, tensors)
    for h in handle: h()
```

Inside the AGRS all-gather, every rank uses NCCL symmetric-memory LSA pointers to put its local slot
directly into the matching slot in each peer. Optionally, `agrs_get_inplace_tensor` returns a tensor
whose pointer lives in this rank's recv slot of the session `buffer`. Once the user has written into
it, subsequent `all_gather` calls skip the copy when `src==dst`:

```cpp
for (const auto& num_bytes : num_bytes_list) {
  out.push_back(torch::from_blob(
      buffer + offset + num_bytes * rank_idx,  // this rank's slot
      {num_bytes}, uint8));
  offset += num_bytes * num_ranks;             // reserve slots for every rank
}
```

The C++ `all_gather` implementation:

1. Plan slot offsets and detect `inplace`.

```cpp
for (int i = 0; i < num_tensors; ++i) {
  const auto& x = tensors[i];
  // Inputs must be contiguous, on CUDA, with nbytes 32-byte aligned (LDG.256)
  EP_HOST_ASSERT(x.is_contiguous());
  EP_HOST_ASSERT(x.is_cuda() and x.nbytes() % 32 == 0);

  // Inplace detection: if x came from agrs_get_inplace_tensor (its pointer is inside session buffer),
  // we can skip the local-rank -> local-rank copy.
  const auto x_offset = math::ptr_diff(x.data_ptr(), buffer);
  const bool is_inplace = 0 <= x_offset and x_offset < num_max_agrs_session_bytes;
  offset[i] = agrs_buffer_offset;
  // Each tensor is sent num_ranks times (one to each peer); inplace saves one copy
  num_copies += nccl_context->num_ranks - is_inplace;
  // Advance cursor: reserve num_ranks slots for this tensor
  agrs_buffer_offset += x.nbytes() * nccl_context->num_ranks;
  // Sanity check: an inplace tensor must land exactly at "this rank's slot" offset,
  // otherwise alignment is wrong
  EP_HOST_ASSERT(not is_inplace or x.data_ptr() == math::advance_ptr(buffer, offset[i] + x.nbytes() * nccl_context->rank_idx));
}
// Session capacity check: exceeding it means config is too small or destroy_session was forgotten
EP_HOST_ASSERT(agrs_buffer_offset <= num_max_agrs_session_bytes and agrs_buffer_slot_idx < kNumMaxInflightAGRS,
               "Not enough session buffer size. Did you forget to flush session?");
```

2. Stream sync + build batched-copy parameters.

```cpp
// ========== Stage B: stream sync + build batched-copy parameters ==========
// Wait compute stream
// comm_stream waits for compute_stream so user writes to x on compute_stream are visible.
const auto compute_stream = at::cuda::getCurrentCUDAStream();
stream_wait(comm_stream, compute_stream);

// Send data to all ranks
// Build (src, dst, size) triples and send them in one cudaMemcpyBatchAsync.
//   - src: this rank's raw data (or its inplace slot in the session buffer)
//   - dst: NVLink P2P address of "this rank's slot" in the peer's buffer (resolved via get_sym_ptr)
//   - Outer loop iterates peers using (rank_idx + i) offset to stagger destination order, avoiding all-out-egress collisions
std::vector<size_t> sizes(num_copies);
std::vector<void*> dst_ptrs(num_copies), src_ptrs(num_copies);
int count = 0;
for (int i = 0; i < nccl_context->num_ranks; ++i) {
  for (int j = 0; j < num_tensors; ++j) {
    const auto& x = tensors[j];
    const auto dst_rank_idx = (nccl_context->rank_idx + i) % nccl_context->num_ranks;
    void* src_ptr = x.data_ptr();
    // get_sym_ptr translates a local address to the peer's NVLink P2P address
    // (offset is preserved; only the base swaps to nvl_window_ptrs[dst]).
    void* dst_ptr =
        nccl_context->get_sym_ptr(math::advance_ptr(buffer, offset[j] + x.nbytes() * nccl_context->rank_idx), dst_rank_idx);
    // src == dst happens when inplace and dst_rank_idx == rank_idx -- skip the no-op copy
    if (src_ptr != dst_ptr) {
      src_ptrs[count] = src_ptr;
      dst_ptrs[count] = dst_ptr;
      sizes[count] = x.nbytes();
      count += 1;
    }
  }
}
// SrcAccessOrderStream: tells the runtime sources are accessed in stream order, removing conservative sync
// PreferOverlapWithCompute: prefer the dedicated copy engine to overlap with compute kernels
cudaMemcpyAttributes attrs = {.srcAccessOrder = cudaMemcpySrcAccessOrderStream, ...};

// One driver call dispatches every copy to comm_stream -- 0 SM occupancy, pure NVLink DMA concurrency
CUDA_RUNTIME_CHECK(cudaMemcpyBatchAsync(dst_ptrs.data(), src_ptrs.data(), sizes.data(), ...));
```

3. Slot-level signal exchange (wait until every peer's data has landed).

```cpp
// Wait for data from other ranks
// Snapshot session id as the signal value; slot_idx is the signal slot owned by this all_gather
const int current_session = agrs_session_idx;
const int slot_idx = agrs_buffer_slot_idx;
agrs_buffer_slot_idx += 1;
// Build (write, wait) pairs for each peer (excluding self)
// write: store current_session into peer's recv_signal[slot, this rank]
//        meaning: "I have completed every prior copy through slot_idx"
// wait : spin-wait until local recv_signal[slot, peer] >= current_session
//        meaning: "the peer has finished its copy through slot_idx; it's safe to read what it wrote into my slot"
// Stream ordering guarantees that cudaMemcpyBatchAsync copies finish before signal writes,
// so once a peer sees the signal, the data is already visible -- no extra fence needed.
std::vector<void*> write_ptrs(nccl_context->num_ranks - 1);
std::vector<void*> wait_ptrs(nccl_context->num_ranks - 1);
for (int i = 0; i < nccl_context->num_ranks - 1; ++i) {
  const auto dst_rank_idx = (nccl_context->rank_idx + i + 1) % nccl_context->num_ranks;
  // Write address is translated into peer's symmetric memory via get_sym_ptr;
  // wait address is just the local workspace (no translation needed).
  write_ptrs[i] = nccl_context->get_sym_ptr(
      workspace_layout_wo_expert->get_agrs_recv_signal_ptr( slot_idx, nccl_context->rank_idx), dst_rank_idx);
  wait_ptrs[i] = workspace_layout_wo_expert->get_agrs_recv_signal_ptr(slot_idx, dst_rank_idx);
}
// batched_write_and_wait uses CU_STREAM_MEM_OP_WAIT_VALUE_32 + WAIT_VALUE_GEQ:
//  ">= current_session" prevents stale signals from older sessions from triggering a false wakeup
//  (agrs_session_idx is monotonically increasing, so older session signals always have smaller values).
//  No write/wait against self: comm_stream's strict in-stream ordering already guarantees the local order.
cuda_driver::batched_write_and_wait(comm_stream, write_ptrs, wait_ptrs, current_session);
```

4. Build zero-copy output views and return an async wait handle.

```cpp
// Build output tensors eagerly
// Insert a num_ranks dimension at the front of each output's shape; torch::from_blob points
// directly at offset[i] inside the session buffer (which spans num_ranks contiguous slots) -- no allocation, no copy.
std::vector<torch::Tensor> out(num_tensors);
for (int i = 0; i < num_tensors; ++i) {
  auto shape = tensors[i].sizes().vec();
  shape.insert(shape.begin(), nccl_context->num_ranks);
  out[i] = torch::from_blob(math::advance_ptr(buffer, offset[i]), shape, tensors[i].options());
}

// Return tensors and a handle for waiting on data arrival
// The event on comm_stream marks "data fully ready"
const auto event = EventHandle(comm_stream);
// Return a closure handle. The caller must call h() before consuming out:
//   stream_wait(compute_stream, event): make compute_stream catch up to comm_stream
// Two assertions: must be called on the original compute_stream; the session must not have been recreated
// (handles from a previous session are invalid because the buffer would have been overwritten).
auto handle = [=, this]() {
  EP_HOST_ASSERT(compute_stream == at::cuda::getCurrentCUDAStream());
  EP_HOST_ASSERT(agrs_in_session and current_session == this->agrs_session_idx);
  stream_wait(compute_stream, event);
};
return {std::move(out), std::move(handle)};
```

The overall call timeline matches the diagram in the original article.
