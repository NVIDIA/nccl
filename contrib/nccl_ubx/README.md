# UB-X

UB-X (**U**ltra **B**andwidth — e**X**perimental) is a prototype GPU
communication library exploring the newest ideas for achieving lowest
collective latency and highest sustained NVLink bandwidth. Its focus is
**compute fusion** — folding pre/post-collective
work (residual add, RMSNorm, mxfp8 quantization) into the collective
kernel — and **efficient overlap with compute kernels**, so collectives
can run concurrently with model compute on the same GPU.

A central piece of the design is a custom **symmetric allocator** that
provides zero-copy collective input/output buffers while remaining easy
to plug into existing PyTorch code: tensors are ordinary `torch.Tensor`
instances backed by an NCCL-managed symmetric window.

> **API stability:** UB-X is new code. The public API is expected to
> evolve and will not be stable for some time.

## Hardware prerequisites

- **GPU:** SM 9.0+ — Hopper (H100, H200), Blackwell (B200, GB200, GB300).
  NVLink multicast hardware is required for the MC kernel paths.
- **SM 8.0 (A100) is not supported** — there is no NVLink multicast
  hardware on Ampere; the inline PTX (`multimem.*`) won't assemble for
  arch `8.0`.

## Software prerequisites

- PyTorch ≥ 2.1
- CUDA Toolkit ≥ 12.0
- **NCCL ≥ 2.30** — UB-X calls into the NCCL device API
  (`ncclDevCommCreate`, `ncclMemAlloc`, `ncclCommWindowRegister`,
  `ncclGetLsaPointer`, `ncclGetLsaMultimemPointer`).
- **nccl4py** — Python bindings for the NCCL device API. Build from the
  matching NCCL release (`bindings/nccl4py` in the NCCL source tree) so
  the binding ABI matches the linked `libnccl`.
- A C++17 compiler matching your PyTorch build

## Install

```bash
pip install -e .                                       # default: SM 9.0a;10.0a
pip install -e ".[dev]"                                # adds pytest, ruff
TORCH_CUDA_ARCH_LIST="9.0a;10.0a" pip install -e .     # explicit arch list
```

The benchmark suite (`ubx_bench`) is installed by the same command from
the in-tree `bench/` directory.

## Compilation options

Build-time environment variables consumed by `setup.py`:

| Variable | Default | Effect |
|---|---|---|
| `TORCH_CUDA_ARCH_LIST` | `9.0a;10.0a` | Target SM architectures. Use the `a` suffix to ensure access to the full `multimem.*` instruction set — some accelerated-only variants are unavailable on plain `9.0` / `10.0`, so future kernels using those variants would silently lose performance or fail to assemble. Recent NVCC tolerates non-`a` for the instructions UB-X uses today, but the `a` form is the supported path. |
| `UBX_BUILD_TIMEOUT` | `0` (off) | When set to `1`, compiles in kernel-side spinloop timeouts. Adds runtime overhead (extra `clock64()` checks and `printf` on expiry); useful for triaging hangs. Runtime threshold is then tunable via `UBX_TIMEOUT_SEC` (read once at `SymmAllocator` construction). |
| `NCCL_HOME` | `/usr/local` | NCCL install root, used to derive `include` and `lib` paths. |
| `NCCL_INCLUDE_DIR` | `$NCCL_HOME/include` | Override include path for `nccl.h`. |
| `NCCL_LIBRARY_DIR` | `$NCCL_HOME/lib` | Override link path for `libnccl`. |

## Quick start

```python
import torch
import torch.distributed as dist
from ubx.ops import request_allocator, get_sym_tensor, allreduce

# During model init: declare the allocator's process group + max shape
dist.init_process_group(backend="nccl")
group = dist.group.WORLD
request_allocator(group, shape=(1024, 2048), dtype=torch.bfloat16)

# During forward pass: allocate a symmetric tensor and run the collective
tensor = get_sym_tensor((1024, 2048), torch.bfloat16, group)
tensor.fill_(1.0)
output = allreduce(tensor)        # auto-selects Lamport / MC by size
```

A full runnable version is in [`examples/allreduce.py`](examples/allreduce.py).

## Available collectives

| Op           | Variants                                | Auto-select                    |
|--------------|-----------------------------------------|--------------------------------|
| AllReduce    | `mc`, `uc`, `lamport`, **`auto`**       | Lamport ≤ 0.25 MB, else MC     |
| AllToAll     | `uc`, `lamport`, **`auto`**             | Lamport ≤ 0.25 MB, else UC     |
| AllToAllV    | `uc`                                    | —                              |
| AllGather    | `mc`                                    | —                              |
| Token dispatch (MoE) | bf16→bf16, bf16→mxfp8           | —                              |

`SymmAllocator.allreduce()` and `SymmAllocator.alltoall_auto()` pick the
fastest variant based on total tensor size. Direct calls to the explicit
methods (`allreduce_mc`, `alltoall_lamport`, etc.) bypass auto selection.

`SymmAllocator.allreduce_mc()` and `allreduce_lamport()` accept optional
`gamma`/`residual_in` parameters to fuse residual addition + RMSNorm into
the same kernel.

## MoE token dispatch + mxfp8

`a2av_token_bf16_mxfp8` is a single GPU kernel that routes bf16 tokens to
remote ranks while quantizing them to mxfp8 (E8M0 scale per 32 elements)
on the fly. Use `compute_token_offsets` to convert a routing matrix into
the slot-assignment tensor the kernel expects.

```python
from ubx import compute_token_offsets, SymmAllocator

token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
    routing,           # [ntokens, total_experts] (uint8/bool/int)
    experts_per_rank,
    myrank=rank,
    nranks=world_size,
)
output = allocator.create_tensor(
    [max_tokens_per_rank, hidden], torch.float8_e4m3fn, blocked="mxfp8",
)
output = allocator.a2av_token_bf16_mxfp8(
    tokens_bf16, token_offsets, experts_per_rank, output,
)
```

A full runnable version is in [`examples/moe_dispatch.py`](examples/moe_dispatch.py).

## Examples

The `examples/` directory contains minimal multi-GPU programs that use the
public API end-to-end. Launch each with two or more GPUs:

```bash
torchrun --nproc-per-node=2 examples/allreduce.py
torchrun --nproc-per-node=2 examples/alltoall.py
torchrun --nproc-per-node=2 examples/moe_dispatch.py
```

| Example | What it shows |
|---|---|
| [`examples/allreduce.py`](examples/allreduce.py) | `request_allocator` + `get_sym_tensor` + `allreduce`, validated against `torch.distributed.all_reduce` |
| [`examples/alltoall.py`](examples/alltoall.py) | Direct `SymmAllocator` use; UC and Lamport alltoall, validated against `torch.distributed.all_to_all` |
| [`examples/moe_dispatch.py`](examples/moe_dispatch.py) | Routing matrix → `compute_token_offsets` → `a2av_token_bf16_mxfp8`, with a dequantize check |

## Benchmarking

`ubx_bench` provides an nccl-tests-compatible CLI for measuring UB-X and
NCCL collectives side-by-side. CUDA graph mode is the default; pass
`--no-cudagraph` for eager mode.

```bash
# AllReduce, UB-X vs NCCL
torchrun --nproc-per-node=8 -m ubx_bench all_reduce \
    -b 1K -e 128M -f 2 --backend ubx,nccl -n 5 -w 2

# AllToAll
torchrun --nproc-per-node=8 -m ubx_bench all_to_all \
    -b 1K -e 8M -f 2 --backend ubx,nccl -n 5 -w 2
```

See [`bench/README.md`](bench/README.md) for the full CLI reference and
benchmarking environment variables.

## Tests

```bash
# Single-GPU unit tests (allocator, tensor, backend detection)
pytest tests/

# Multi-GPU correctness tests (requires 2+ GPUs)
torchrun --nproc-per-node=2 -m pytest tests/distributed/
```

Distributed tests launch worker subprocesses (under `tests/distributed/_workers/`)
and compare results against `torch.distributed` (NCCL) reference output.
Tolerances: bf16 `atol=0.0625, rtol=0.02`; FP8 `atol=0.0625, rtol=0.125`.

## Repository structure

```
ubx/                     Python package (allocator, tensor, ops, fused)
csrc/                    C++/CUDA kernels and pybind11 bindings
tests/                   Single-GPU + distributed test suites
bench/                   In-tree ubx_bench benchmarking harness
examples/                Minimal runnable programs using the public API
docs/                    User-facing API documentation
```

## Documentation

- [`docs/api.md`](docs/api.md) — API guide: install, `SymmAllocator` /
  `SymmTensor`, collectives, fused residual + RMSNorm, MoE token
  dispatch, environment variables

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `UBX_SYMM_POOL_SIZE` | `6 × max_tensor_size` | Total symmetric pool size in bytes |
| `UBX_GRAPH_POOL_SHARE` | `0.9` | Fraction of pool reserved for CUDA-graph allocations |
| `UBX_DEBUG` | `0` | Enable internal debug prints |
| `UBX_BLOCK_ALIGN` | `4096` | Data-region alignment for blocked tensor formats (e.g. mxfp8) |
| `UBX_TIMEOUT_SEC` | unset | Kernel-spinloop timeout in seconds. Effective only when the extension was built with `UBX_BUILD_TIMEOUT=1` — see [Compilation options](#compilation-options). |
| `NCCL_NVLS_ENABLE` | unset (= enabled) | Set to `0` to operate without NVLink multicast |

## Maintainers

| GitHub | Areas |
|--------|-------|
| @nv-akorzh | PyTorch integration and UBX kernels |
| @nduvdevani | General collectives and UBX kernels |
| @itamar-rauch | Framework MoE integration and collectives |

## License

Apache 2.0
