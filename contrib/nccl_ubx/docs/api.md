# UB-X API Guide

How to use the `ubx` Python package: install, allocate symmetric memory,
run collectives, and dispatch MoE tokens.

UB-X is a low-latency NVLink collectives library. It backs `torch.Tensor`
subclasses with NCCL symmetric-memory windows so collectives execute with
single-kernel device-side primitives (multimem, Lamport polling, UC peer
writes) instead of host-driven ring algorithms. Hardware target: SM 9.0+
(Hopper, Blackwell).

---

## 1. Install

```bash
pip install -e .                                    # core + bench
pip install -e ".[dev]"                             # with pytest, ruff
TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e .  # override SM targets
```

Defaults: SM 9.0a (Hopper), 10.0a (Blackwell). Always use the `a` suffix —
plain `9.0`/`10.0` builds may silently lose access to multimem variants
adopted by future kernels.

Runnable end-to-end examples live under [`examples/`](../examples) and can
be launched with `torchrun --nproc-per-node=N <example>.py`.

---

## 2. Two ways to use the library

### Convenience API (`ubx.ops`)

A global registry maps each `torch.distributed` process group to a lazily
created `SymmAllocator`. You declare the largest tensor you'll need, ask
for symmetric tensors as you go, and call free functions for collectives.
Best for drop-in use inside an existing model.

```python
from ubx.ops import request_allocator, get_sym_tensor, allreduce

request_allocator(group, shape=(4096,), dtype=torch.bfloat16)
sym = get_sym_tensor((4096,), torch.bfloat16, group)
sym.copy_(some_activation)
out = allreduce(sym)            # auto-selects MC or Lamport
```

### Direct allocator API (`ubx.SymmAllocator`)

You construct the allocator yourself and call methods on it. Better when
you want explicit control over pool size, lifecycle, or want to mix
multiple groups. All collectives below are documented as methods on
`SymmAllocator`.

```python
from ubx import SymmAllocator

allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)
sym = allocator.create_tensor(shape, torch.bfloat16)
out = allocator.allreduce(sym)
```

---

## 3. Symmetric memory: `SymmAllocator` and `SymmTensor`

### `SymmAllocator(size_bytes, device, dist_group)`

Allocates a single contiguous symmetric pool registered with NCCL. The
pool is split internally into a graph region (default 90%, controlled by
`UBX_GRAPH_POOL_SHARE`) for stable addresses across CUDA-graph replays
and a non-graph region for eager allocations. Free-list management is
first-fit with refcounting; adjacent free segments merge on free.

Construction is collective: every rank in `dist_group` must call it with
the same `size_bytes`. The first construction broadcasts an NCCL unique
ID over the existing process group if a fresh communicator is needed.

```python
allocator = SymmAllocator(64 * 1024 * 1024, device, group)
allocator.close()           # explicit teardown; also runs at __del__
```

Useful properties:

| Attribute | Meaning |
|---|---|
| `rank`, `world_size` | as in `torch.distributed` |
| `multicast_available` | `True` if NVLink multicast (MC) is usable on this fabric |
| `pool_ptr`, `mc0_ptr` | UC and MC base pointers into the symmetric window |

### `allocator.create_tensor(shape, dtype, blocked=None) -> SymmTensor`

Allocates a `SymmTensor` (a `torch.Tensor` subclass) backed by a slice
of the pool. The tensor is reference-counted and returns its memory to
the free list when the last view is dropped.

`blocked='mxfp8'` reserves an aligned data region followed by E8M0 scale
bytes (1 byte per 32 elements). The visible shape and dtype describe
only the data; metadata is reached via `tensor.metadata_ptr`.

```python
x = allocator.create_tensor((1024, 4096), torch.bfloat16)
y = allocator.create_tensor((slots, hidden), torch.float8_e4m3fn, blocked='mxfp8')
y_scales_ptr = y.metadata_ptr        # raw int pointer to scale bytes
```

`SymmTensor.view()` returns a `SymmTensor` view of the same memory.
`SymmTensor.clone()` falls back to a plain `torch.Tensor` because the
clone is not symmetric. Blocked tensors do not support `view()`.

---

## 4. Collectives on `SymmAllocator`

All collectives operate on `SymmTensor` inputs allocated from the same
allocator. `smlimit` caps SMs (0 = launcher default); `cgasize` is the
CGA cluster size for the kernel-internal grid (0 = no clustering).

### AllReduce

| Method | When to use |
|---|---|
| `allreduce(t, smlimit=0, cgasize=0)` | Auto: Lamport ≤ 0.25 MB, otherwise MC |
| `allreduce_mc(t, ...)` | Force multicast — best for large tensors |
| `allreduce_lamport(t, ...)` | Force Lamport polling — best for small tensors; falls back to MC if pool exhausted, or to UC if multicast unavailable |
| `allreduce_uc(t, ...)` | Force unicast software reduction — fallback on fabrics without multicast |

All variants are in-place. The MC and Lamport methods accept optional
`residual_in`, `residual_out`, `gamma`, `eps`, `hidden_size` arguments to
fuse residual addition and RMSNorm into the same kernel — see
[Section 5](#5-fused-residual--rmsnorm) for the dedicated wrapper.

```python
out = allocator.allreduce(sym)                    # in-place, returns sym
```

### AllGather

```python
out = allocator.allgather(t, smlimit=0)
```

Gathers `t` from every rank along axis 0. Returns a fresh
`SymmTensor` of shape `(world_size * t.shape[0], *t.shape[1:])`.

### AllToAll

| Method | When to use |
|---|---|
| `alltoall_auto(t, smlimit=0, nthreads=0)` | Auto: Lamport ≤ 0.25 MB, otherwise UC |
| `alltoall(t, ...)` | UC + barrier |
| `alltoall_lamport(t, ...)` | Triple-buffered, barrier-free in steady state. 1.5–2× faster than UC at small sizes |

Both variants split `t` into `world_size` equal chunks along axis 0 and
return a new `SymmTensor` of the same shape. For `alltoall`, `smlimit`
sets the SM count *exactly* (not as a cap) so callers can sweep up and
down; `alltoall_lamport`'s `smlimit` is a cap, matching allreduce.

When NVLink multicast is unavailable, set `NCCL_NVLS_ENABLE=0` before
constructing the allocator.

### AllToAllV (variable-length)

Two-step API for the hot path, plus a one-call convenience wrapper:

```python
state = allocator.alltoallv_prepare(output_split_sizes, input_split_sizes,
                                    dtype=torch.bfloat16)
for step in steps:
    out = allocator.alltoallv_run(input_tensor, state)        # repeatable

# Or, when split sizes change every call:
out = allocator.alltoallv(input_tensor, output_split_sizes, input_split_sizes)
```

Use `prepare`/`run` when split sizes are stable across iterations — the
prepare step does an `all_gather` to compute remote receive offsets and
allocates the output buffer; only `run` is on the hot path. The
convenience `alltoallv()` wraps both for one-shot use.

---

## 5. Fused residual + RMSNorm

`ubx.fused.allreduce_fused` calls the same MC/Lamport kernels as
`SymmAllocator.allreduce` but exposes the fusion arguments directly:

```python
from ubx.fused import allreduce_fused

out = allreduce_fused(
    allocator, sym, hidden_size,
    residual_in=residual,            # optional
    residual_out=new_residual,       # optional
    gamma=rmsnorm_weight,            # enables RMSNorm fusion when set with eps
    eps=1e-6,
)
```

When `gamma` and `eps` are both provided, RMSNorm is computed as part of
the allreduce kernel. When `residual_in` is provided, it is added before
the norm. This is detachable from the core API — removing
`ubx/fused.py` does not affect plain collectives.

The convenience wrapper `ubx.ops.allreduce` also picks up fusion when
called with `gamma`/`eps`, and additionally manages an internal residual
buffer on the allocator (used across transformer FFN/attn blocks).
Companion helpers:

```python
from ubx.ops import allreduce, free_residual

out = allreduce(sym, gamma=g, eps=1e-6)
free_residual(sym)                   # release the cached residual buffer
```

---

## 6. MoE token dispatch (`a2av_token_bf16_mxfp8`)

Routes bf16 tokens to remote ranks and quantizes them to mxfp8 in a
single kernel. Three steps:

```python
from ubx import SymmAllocator, compute_token_offsets

# 1. Compute slot assignments from a routing matrix that's identical on every rank
token_offsets, max_tokens_per_rank, tokens_per_expert, expert_offsets = \
    compute_token_offsets(routing, experts_per_rank, myrank=rank, nranks=world_size)

# 2. Allocate the receive buffer (symmetric, mxfp8-blocked, identical size on every rank)
output = allocator.create_tensor(
    [max_tokens_per_rank, hidden], torch.float8_e4m3fn, blocked='mxfp8',
)

# 3. Dispatch
output = allocator.a2av_token_bf16_mxfp8(
    tokens_bf16,            # [local_ntokens, hidden] bfloat16
    token_offsets,          # [local_ntokens, total_experts] int32, -1 = not routed
    experts_per_rank,
    output,                 # mxfp8 SymmTensor allocated above
)
```

`hidden` must be a multiple of 32 (one mxfp8 block). After the call,
fp8 data is at `output.data_ptr()` and E8M0 scale bytes are at
`output.metadata_ptr`. Quantization tolerance is roughly
`max_block_scale × 2^-4`.

### Async dispatch

```python
allocator.a2av_token_bf16_mxfp8(..., sync=False)
# ... do unrelated CPU/GPU work on a different stream ...
allocator.a2av_wait()
```

`sync=False` makes the dispatch kernel return as soon as the barrier flag
is signalled; you must call `a2av_wait()` before reading the output.

### Pipelined dispatch (`a2av_token_bf16_mxfp8_persistent`)

When the consumer is a per-expert GEMM and you'd otherwise launch the
dispatch kernel `experts_per_rank` times, use the persistent variant:
one kernel performs `nchunks = ceil(experts_per_rank / nexperts_per_chunk)`
internal chunks, each with its own barrier matching the non-persistent
protocol. The caller still calls `a2av_wait()` once per chunk.

```python
allocator.a2av_token_bf16_mxfp8_persistent(
    tokens_bf16, token_offsets, experts_per_rank, output,
    nexperts_per_chunk=2,
)
```

`nchunks` is capped at 32.

### bf16 → bf16 variant

`a2av_token_bf16_bf16` is the same routing/slot scheme without
quantization — useful when the receiver expects raw bf16. The output
must be a plain (non-blocked) bf16 `SymmTensor`.

---

## 7. MoE token combine

The reverse of dispatch: each rank holds the FFN expert outputs for the
tokens it received in dispatch, and combine routes those outputs back to
the originating ranks where they are gate-weighted, summed, and produced
as `[local_ntokens, hidden]` bf16. There are two families:

- **PULL-barrier** (`combine_bf16_bf16`, `combine_mxfp8_bf16`). Each rank
  stages its expert outputs into a temp symmetric buffer (Phase 1),
  ranks barrier, then every rank pulls peer contributions for its own
  tokens, gate-weights them, sums in fp32, and downcasts (Phase 2).
  Reuses the dispatch `token_offsets` matrix. Supports a 2-phase async
  mode via `sync=False` for overlapping Phase 1 with unrelated work.
- **PUSH** (`combine_bf16_bf16_push`, `combine_bf16_bf16_lamport_push`).
  Each rank pushes its FFN outputs directly into peer destination buffers
  at known `(origin_token, k_idx)` coordinates. Receivers then read their
  *own* buffer — same race-free pattern as `alltoall_lamport`. Requires
  the inverse-routing tables built by `compute_combine_push_map`.

### PULL-barrier

```python
out = allocator.combine_bf16_bf16(
    expert_outputs,        # [n_recv, hidden] bf16, in dispatch slot order
    token_offsets,         # same matrix from compute_token_offsets
    experts_per_rank,
    max_tokens_per_rank,
    gate_weights=g,        # optional [local_ntokens, total_experts] fp32
)
# → [local_ntokens, hidden] bf16
```

`combine_mxfp8_bf16` is identical except it quantizes each rank's expert
outputs to mxfp8 before the cross-rank exchange (halving NVLink payload).
The accumulation and the returned tensor stay bf16.

Async (Phase 1 / Phase 2 overlap):

```python
out = allocator.combine_bf16_bf16(..., sync=False)
# ... unrelated work on a different stream ...
allocator.combine_wait()       # runs Phase 2; out is now valid
```

### PUSH

PUSH variants need a different routing artifact:

```python
from ubx import compute_combine_push_map

inverse_map, topk_idx, max_tokens_per_rank = compute_combine_push_map(
    routing, experts_per_rank, myrank=rank, nranks=world_size,
)

out = allocator.combine_bf16_bf16_lamport_push(
    expert_outputs, inverse_map, topk_idx,
    experts_per_rank, max_tokens_per_rank,
    gate_weights=g,
)
```

| Method | Sync | When to use |
|---|---|---|
| `combine_bf16_bf16_lamport_push` | barrier-free in steady state (triple-buffered Lamport) | Small messages — wins when polling cost beats barrier latency |
| `combine_bf16_bf16_push` | single MC barrier, double-buffered | Large messages — barrier amortizes over the bigger payload |

PUSH variants do not currently provide an async/`sync=False` form;
`combine_wait()` only applies to the PULL-barrier kernels.

`gate_weights` (when provided) is the full `[local_ntokens,
total_experts]` fp32 router output — the kernel indexes it per
contribution. Pass `None` for an unweighted sum.

---

## 8. Module-level helpers (`ubx.ops`)

| Function | Use |
|---|---|
| `request_allocator(group, shape=None, dtype=torch.bfloat16)` | Declare a process group + max tensor size for the registry. Idempotent; max size accumulates. |
| `get_sym_tensor(shape, dtype, group)` | Allocate a symmetric tensor; lazily creates the underlying `SymmAllocator` on first call. Returns `None` for unsupported dtypes (currently only bf16 is supported by this path). |
| `allreduce(sym, gamma=None, eps=None, smlimit=0, cgasize=0)` | Auto-selecting allreduce with optional fused RMSNorm + internal residual handling. |
| `free_residual(sym)` | Drop the allocator's cached residual buffer. |
| `restore(tensor, group)` | Rewrap a `torch.Tensor` whose `data_ptr` lies inside the registered pool as a `SymmTensor`. No-op otherwise. |
| `compute_token_offsets(routing, experts_per_rank, myrank, nranks)` | Build `token_offsets` and `max_tokens_per_rank` from a routing matrix (used by dispatch and PULL combine). |
| `compute_combine_push_map(routing, experts_per_rank, myrank, nranks)` | Build the inverse routing tables (`inverse_map`, `topk_idx`) needed by PUSH combine. |
| `mem_stats()` | Print used/free pool sizes for every registered allocator. |

---

## 9. CUDA graphs

The pool's graph region holds stable addresses across replays. Allocate
the tensors you'll capture *before* `cudaStreamBeginCapture`, run the
collectives during capture, and the same `SymmTensor` objects can be
replayed any number of times. The Lamport allreduce and alltoall paths
keep separate "next poison" and triple-buffer state for graph vs eager
mode so the two don't interfere.

---

## 10. Environment variables

| Variable | Default | Effect |
|---|---|---|
| `UBX_SYMM_POOL_SIZE` | `6 × max_tensor_size` MB | Pool size when the registry creates the allocator lazily |
| `UBX_GRAPH_POOL_SHARE` | `0.9` | Fraction of pool reserved for graph mode |
| `UBX_BLOCK_ALIGN` | `4096` | Data-region alignment for blocked tensors |
| `UBX_MAXSM` | `0` | Override per-kernel SM grid (1–128); 0 = compiled default |
| `UBX_TIMEOUT_SEC` | `30` | Kernel polling-loop timeout (only effective when built with `UBX_BUILD_TIMEOUT=1`) |
| `UBX_DEBUG` | `0` | Enable debug prints |
| `UBX_DUMMY` | `0` | Bypass UB-X (returns input unchanged) |
| `NCCL_NVLS_ENABLE` | unset | Set to `0` to disable NVLink multicast (UC-only barriers) |
| `TORCH_CUDA_ARCH_LIST` | `9.0a;10.0a` | Target SM architectures (use the `a` suffix) |

---

## 11. Common pitfalls

- **Pool too small.** Lamport allreduce and alltoall keep two or three
  buffers of the input shape live at once. Size the pool to at least
  `6 × max_tensor_size` (the default heuristic) for steady-state use.
- **Multicast left enabled on a fabric that doesn't support it.**
  `mc0_ptr` will be 0 but NCCL will still try to perform MC rendezvous.
  Set `NCCL_NVLS_ENABLE=0` before constructing the allocator.
- **`hidden % 32 != 0`** in mxfp8 dispatch. Pad to a multiple of 32 or
  reshape upstream.
- **Mismatched `routing` matrices across ranks.** `compute_token_offsets`
  expects every rank to produce the *same* matrix, so use a shared seed
  or an explicit broadcast.
- **Calling `a2av_wait()` after `sync=True`.** Harmless, but unnecessary —
  the kernel already polled to completion.
- **Dropping the last `SymmTensor` view while a kernel is still running.**
  Refcounting is host-side; rely on `torch.cuda.synchronize()` or stream
  ordering before deallocation in lifecycle-sensitive code.
