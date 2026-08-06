# PACE — Parallelism-Aware Collective Engine

PACE is a parallelism-aware collective engine for LLM training and inference.
It fuses the memory operations that frameworks otherwise insert around
communication — layout conversion, dtype casting, scatter/gather — directly
into the collective kernels, and exposes a *parallelism* interface rather than
the traditional collective interface: a compute kernel's output feeds straight
into PACE, and PACE's output is consumed directly by the next kernel. PACE is
built on top of NCCL's **GIN (GPU Interface for Networking)** device API.

This `contrib/` entry contains the **All-Gather (AG)**, **Reduce-Scatter
(RS)** and **Scatter-Gather (SG / Ulysses-CP All-to-All)** collectives of
PACE.

## Maintainers

| Maintainer | Area |
|---|---|
| @Chaser-wind | All |
| @Gaojiaqi | All |

## Motivation

In LLM training and inference, a surprisingly large share of the compute
cycles is spent *gluing* compute kernels and communication kernels together,
rather than on useful work inside either.

The mismatch lies in the memory laytout. On one side, the output of a compute
kernel often spans multiple, non-contiguous regions (e.g. a per-rank or
QKV-split view), while a collective like all-gather or all-to-all expects a
single contiguous buffer. On the other side, the output of a communication
kernel is arranged in rank order — each rank's chunk a contiguous block,
concatenated along the rank axis — while the downstream compute operator often
needs the data with a different axis ordering (e.g. token-major for attention
rather than rank-major), forcing a transpose/permute/reshape. Bridging these
gaps forces the framework to insert memory operations — `transpose`,
`contiguous`, `concatenate` (and their inverses `split`, `view`, `chunk`) —
around every communication step. These are pure memory traffic: they move bytes
and perform no FLOPs, yet they add kernel launches, extra passes over HBM/L2,
and synchronization points.

In practice, these glue operations can account for **up to ~10% of end-to-end
training time**.

The key observation is that the data movement these glue ops perform is exactly
what a collective kernel must do anyway to stage and deliver tensor chunks. So
they are natural fusion candidates: fold the layout/scatter/gather semantics
into the communication kernel and eliminate the extra memory passes and
launches entirely.

That is the design bet behind PACE — a *parallelism-aware* collective engine.
Instead of the traditional collective interface (`all_reduce`, `all_gather`,
`all_to_all`, …), which expects contiguous in/out tensors, PACE exposes a
**parallelism** interface: the output of a compute kernel — however it is laid
out — can be fed *directly* into a PACE operation, and PACE's output can be
consumed *directly* by the next compute kernel. The memory ops, dtype
conversions, and scatter/gather semantics that the framework would otherwise
implement as glue are absorbed into the fused communication kernel. The
collectives in this `contrib/` implement exactly that for three common
parallelism patterns: FSDP All-Gather and Reduce-Scatter (which skip the
copy-in/copy-out and can convert dtypes on the wire), and Ulysses-CP
All-to-All (which fuses the pre/post all-to-all memory ops).

## Collectives

Every PACE collective is structured as three fused stages — **prologue**,
**communication**, and **epilogue** — so the framework never has to touch the
tensor between them:

* **Prologue** prepares the compute kernel's output for communication:
  multi-tensor (list) input, on-the-wire dtype conversion, strided input read
  in place, and scaling factors.
* **Communication** moves the data. We provide SM-resident consume kernels,
  plus resource-lean **0-SM (intranode)** and **1-SM (internode)** variants to
  minimize the SM footprint (Scatter-Gather and All-Gather only). Both variants
  use pipelined staging, keeping the shared buffer bounded — independent of the
  tensor size — without sacrificing bandwidth. For All-Gather collective, CUDA
  Graph is supported to reduce CPU launching overhead.
* **Epilogue** is the reverse of the prologue: it lays the received data out so
  the next compute kernel can consume it directly.

Prologue support across the three collectives in this `contrib/`:

| Prologue op | FSDP All-Gather | FSDP Reduce-Scatter | Ulysses-CP All-to-All |
|---|---|---|---|
| Multi-tensor (list) input | ✅ | ✅ | ✅ |
| Dtype conversion | ✅ `fp32→bf16` | ✅ `fp32`/`bf16` | — |
| Strided input | — | — | ✅ QKV-split |
| Scaling factors | — | ✅ `pre`/`postdivide` | — |

## Software prerequisites

* PyTorch ≥ 2.10 (Python ≥ 3.12)
* CUDA Toolkit ≥ 12.9
* **NCCL ≥ 2.31** with GIN support

## Build

PACE is a PyTorch C++ extension. The build links the GIN-capable NCCL
**statically** (`libnccl_static.a`) into `pace_cpp` so that calls resolve to
the NCCL version it was built against.

```bash
NCCL_DIR=/path/to/nccl-install \
TORCH_CUDA_ARCH_LIST="9.0" \       # or: 10.0
python setup.py bdist_wheel        # or: pip install .
```

Environment variables consumed by `setup.py`:

| Variable | Default | Effect |
|---|---|---|
| `NCCL_DIR` | system NCCL | Path to an NCCL install (must contain `include/` and `lib/libnccl_static.a`). Required unless the system `libnccl.so` is ≥ 2.30 with GIN. |
| `TORCH_CUDA_ARCH_LIST` | torch default | Target SM architectures, e.g. `9.0` for Hopper. |
| `PACE_TIMEOUT_DEBUG` | on | Kernel-side spinloop timeouts + arrival debug. |
| `PACE_FAST_DEBUG` | off | Shorter timeout constants for hang triage. |
| `PACE_KERNEL_DEBUG` | off | `-lineinfo` + ptxas spill warnings. |

## Configuration

Every collective takes a `CommConfig` dataclass (defined in `pace/utils.py`,
re-exported from `pace.ag`, `pace.rs`, and `pace.sg`). Fields not relevant to a
given collective are ignored by it — `ag_zero_sm` is AG-only, `use_wg` is
RS-only, and SG always uses the data-ring.

| Field | Default | Applies to | Effect |
|---|---|---|---|
| `slot_unroll` | `16` | AG / RS / SG | Copy slot length (unrolls of 1024 float4). |
| `nvl_ring_size` | `4` | AG / RS / SG | NVLink ring depth of the staging ring buffer. |
| `rdma_ring_size` | `4` | AG / RS / SG | RDMA ring depth of the staging ring buffer. |
| `num_sms` | `32` | AG / RS / SG | SM-resident kernel grid size. `0` selects the 0-SM (stream) path, `1` the 1-SM lean variant. |
| `use_wg` | `False` | RS | Hint to prefer warp-group primitives for small messages. |
| `ag_zero_sm` | `AGZeroSMConfig()` | AG | Knobs for the 0-SM (`num_sms == 0`) all-gather cord path (below). |

`AGZeroSMConfig` (from `pace.utils`):

| Field | Default | Effect |
|---|---|---|
| `use_ring` | `True` | Inter-node data shape: ring-forward to the next rail peer vs broadcast to every other node's rail peer. |
| `use_graph` | `False` | Capture the 0-SM AG into a CUDA graph; off = direct submission each call. |
| `force_graph_capture` | `False` | Re-capture + re-instantiate the graph on every call instead of using the cache. |
| `min_graph_nodes` | `32` | Minimum estimated graph-node count before graph mode is used (tiny ops fall back to direct submission). |

## Usage

See the `test/` directory for the AG / RS / SG correctness and performance
test harnesses:

```bash
python test/test_ag.py --num-processes 2 --seq-len 1048576
python test/test_rs.py --num-processes 2 --num-sms 32 --seq-len 1048576
python test/test_sg.py --num-processes 2 --num-sms 32
python test/test_sg.py --num-processes 2 --num-sms 32 --strided   # + strided-S (QKV-view) inputs
```

For multi-node, run one process per node with the usual `MASTER_ADDR`,
`MASTER_PORT`, `WORLD_SIZE` and `RANK` environment variables (see
[`test/utils.py`](test/utils.py) `init_dist`).

## License

Apache-2.0
