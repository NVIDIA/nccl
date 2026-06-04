# ubx_bench

Benchmark harness for UB-X collectives. Reports latency and bandwidth in
the same column layout as nccl-tests, so you can paste UBX numbers into
the same comparison sheets you already use.

CUDA graph mode is the default (`-G 10000`). Pass `--no-cudagraph` for
eager mode when you want per-op launch + sync overhead measured as well.

## Install

`ubx_bench` is installed by the same `pip install -e .` as the core `ubx`
package — no separate setup.

## Quick start

```bash
# AllReduce, UB-X vs NCCL — graph mode (default)
torchrun --nproc-per-node=8 -m ubx_bench all_reduce \
    -b 1K -e 128M -f 2 --backend ubx,nccl -n 5 -w 2

# AllToAll — UB-X auto-selects Lamport (small) vs UC (large)
torchrun --nproc-per-node=8 -m ubx_bench all_to_all \
    -b 1K -e 8M -f 2 --backend ubx,nccl -n 5 -w 2

# Eager mode (each op individually synchronized; useful for debugging)
torchrun --nproc-per-node=8 -m ubx_bench all_reduce \
    -b 1K -e 1M -f 2 --backend ubx --no-cudagraph -n 20 -w 5

# Sweep specific UB-X allreduce kernels
torchrun --nproc-per-node=8 -m ubx_bench all_reduce \
    -b 1K -e 128M -f 2 --backend ubx --kernel mc,uc,lamport,auto

# MoE token dispatch — UB-X bf16 vs UB-X mxfp8
srun -n 4 python -m ubx_bench a2av_dispatch \
    --hidden 7168 --topk 8 --experts-per-rank 32 \
    --min-tokens 128 --max-tokens 8192 \
    --backend ubx_bf16,ubx_mxfp8 -n 5 -w 2
```

## Available collectives

| Sub-command | Backends | Notes |
|---|---|---|
| `all_reduce` | `ubx` (mc/uc/lamport/auto), `nccl` | UB-X auto switches at 0.25 MB total bytes |
| `all_to_all` | `ubx` (uc/lamport/auto), `nccl` | UB-X auto switches at 0.25 MB total bytes |
| `all_gather` | `ubx` (mc), `nccl` | |
| `alltoallv` | `ubx` (uc), `nccl` | `--alpha` controls split-size skew (Zipfian) |
| `reduce_scatter` | `nccl` | No UB-X kernel |
| `a2av_mxfp8` | `ubx` | Sweeps token counts; bf16→mxfp8 quantized dispatch |
| `a2av_dispatch` | `ubx_bf16`, `ubx_mxfp8` | MoE token-dispatch latency sweep |
| `a2av_combine` | `ubx_bf16`, `ubx_mxfp8`, `ubx_bf16_async`, `ubx_mxfp8_async` | MoE token-combine; `--kernel lamport_push`/`push` selects PUSH variants (bf16) |

## Common flags

| Flag | Default | Description |
|---|---|---|
| `-b SIZE` / `-e SIZE` | `32M` | Min / max message size. Accepts `K`/`M`/`G` suffixes (e.g. `8K`, `128M`). |
| `-f N` | unset | Multiplicative step factor for the size sweep (overrides `-i`). |
| `-i SIZE` | none | Additive step. |
| `-n N` | `20` | Timed iterations per measurement. |
| `-w N` | `5` | Warmup iterations. |
| `-G N` | `10000` | Number of CUDA-graph replays per measurement. |
| `--no-cudagraph` | (off) | Disable CUDA graph capture; run in eager mode. |
| `--backend NAME[,NAME…]` | `all` | `ubx`, `nccl`, `all`, or comma-separated list. |
| `--kernel NAME` | `auto` | UB-X kernel: `mc`, `uc`, `lamport`, `auto` (collective-dependent). |
| `-d TYPE` | `bf16` | Datatype: `bf16`, `fp16`, `fp32`. |
| `-J FILE.json` | none | Also write a JSON record of the run. |

For the full list of arguments: `python -m ubx_bench --help`.

## Output

Per measurement, the harness reports the standard nccl-tests columns
(`size`, `count`, `type`, `redop`, `time`, `algbw`, `busbw`) for each
selected backend, side by side.

Bandwidth correction factors match nccl-tests:

* AllReduce: `2 × (n-1) / n`
* AllGather, ReduceScatter, AllToAll, AllToAllV, A2AV: `(n-1) / n`

## Environment

| Variable | Recommended | Purpose |
|---|---|---|
| `UBX_GRAPH_POOL_SHARE` | auto-set per call | Pool split between graph and eager allocations. The bench sets `0.5` in graph mode and `0.1` in eager mode automatically. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | Reduces GPU scheduling overhead in graph mode. |
| `NCCL_NVLS_ENABLE` | unset (= enabled) | Set to `0` to disable NVLink multicast. |

