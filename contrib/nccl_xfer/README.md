# NCCL Xfer

NCCL Xfer is an experimental, standalone NCCL-based library for cross-group
GPU data movement. This preview contains the **reshard** functionality: redistribute a
global tensor between two disjoint groups of GPU processes (the source group
holds one sharding / replication layout, the destination group holds
another), with a single call that moves the data with no host involvement.

The library is built on NCCL's user-window API (`ncclWindow_t` +
`ncclMemAlloc`), so transfers are zero-copy and one-sided. The single public
entry point for this release is `ncclXferReshardWithWindow`. The shared library is
installed as `libnccl_xfer.so`; the public header is `nccl_xfer.h`, while the
`ncclXferReshard*` symbols carry both the library prefix (`ncclXfer`) and the
functional scope (`Reshard`).

> **Status.** Experimental — see [RELEASE.md](RELEASE.md) for the full list of
> known limitations and the supported tensor-rank / mesh-size envelope. Depends
> on NCCL window APIs in `nccl_device.h`, which are still evolving; pin a
> known-good NCCL build for production use.

## Maintainers

| GitHub | Areas |
|--------|-------|
| @kaushik-ks | Design, kernels, performance |
| @spotluri | Design, kernels, performance |
| @kingchc | APIs, integration |
| @kwen2501 | APIs, integration |

---

## Concepts

A typical caller has two disjoint sets of ranks (e.g. trainer ranks and
inference / generator ranks) inside one NCCL communicator. Each side owns a
local tile of the same logical tensor under a different layout. One call to
`ncclXferReshardWithWindow` reshapes the tile on every destination rank to match
the destination layout.

**Mesh** (`ncclXferReshardMesh_t`) — describes one side's rank topology. It is
always 2-axis:

```c
typedef struct {
    int dims[2];        // axis-0 size × axis-1 size = number of ranks on this side
    int startRank;     // first global rank that belongs to this mesh
    int placement[2];   // per-axis role: REPLICATE or SHARD(tensor_dim)
} ncclXferReshardMesh_t;
```

`placement[i]` is one of:

- `NCCLXFER_RESHARD_REPLICATE` — the mesh axis replicates the tensor.
- `NCCLXFER_RESHARD_SHARD(d)`  — the mesh axis shards along tensor dim `d`.

Exactly **one** axis per mesh should be a SHARD (the other a REPLICATE) for a
sharded layout. For full replication, encode it as a 1-shard layout (mesh axis
of size 1) — both `REPLICATE` is a degenerate case not exercised by the test
suite.

**Roles** — by convention, ranks `[0 .. src_total)` belong to the source mesh
(`startRank = 0`) and ranks `[src_total .. world_size)` belong to the
destination mesh (`startRank = src_total`). The two halves are disjoint and
their union is the world.

**Same-dim vs cross-dim** — when `src_shard_dim == dst_shard_dim` the source
and destination shard the tensor along the same axis (a partial-overlap copy
between rank groups). Otherwise it is **cross-dim sharding** — effectively an
all-to-all between groups along different axes.

**Single-offset window contract** — each participating pointer must live at a
non-negative offset inside the registered window. Both `src->dataPtr` and
`dst->dataPtr` must have the same offset within that window. A zero-offset
buffer is the common case, but a matching non-zero offset is accepted.

---

## Quick Start

Prerequisites: an NCCL build from source (with `nccl_device.h` exported), CUDA,
and an MPI runtime for the benchmarks.

Two build paths are shipped side-by-side — pick either; both produce the
same artifacts under `build/lib/` and `build/bin/`.

```bash
git clone <repo-url> nccl-xfer
cd nccl-xfer

# 1. Point at your NCCL build
export NCCL_HOME=/path/to/nccl/build
```

### Option A — Make

```bash
# 2a. Build the shared library
make                                       # → build/lib/libnccl_xfer.so

# 3a. Build the canonical single-layer benchmark (also a worked example)
make reshard                               # → build/bin/reshard_bench
```

`make help` lists all targets. `make` (no target) builds only the library.

### Option B — CMake

```bash
# 2b. Configure + build everything (library + bench + tests)
cmake -B build -DNCCL_HOME="$NCCL_HOME" \
      -DNCCL_XFER_BUILD_BENCH=ON \
      -DNCCL_XFER_BUILD_TESTS=ON
cmake --build build -j

# Library only (faster):
# cmake -B build -DNCCL_HOME="$NCCL_HOME" && cmake --build build -j
```

Use `ctest --test-dir build` to run the host-only test subset.

---

## Usage

A complete walk-through lives in `benchmarks/reshard_bench.cc`; a minimal
sketch:

```cpp
#include "nccl_xfer.h"

// Caller-allocated, communicator-wide symmetric buffer (sized to the worst
// case across all reshard calls that will use this comm).
void* buffer;
ncclMemAlloc(&buffer, max_local_bytes);

// Register one window on the comm; reuse for many reshards.
ncclWindow_t window;
ncclCommWindowRegister(comm, buffer, max_local_bytes, &window,
                       NCCL_WIN_COLL_SYMMETRIC);

// Initialize the reshard library. Optional config struct lets you set
// maxCta; everything else is env-driven (see "Tuning" below).
ncclXferReshardConfig_t cfg = NCCLXFER_RESHARD_CONFIG_INITIALIZER;
cfg.maxCta = 8;
ncclResult_t r = ncclXferReshardInit(&cfg);  // or ncclXferReshardInit(NULL) for defaults
if (r != ncclSuccess) { /* handle */ }

// Describe the two layouts. Example: 8 ranks split 4 src / 4 dst, both
// 1×4 1-D meshes that shard the outer tensor dim.
ncclXferReshardMesh_t src_mesh = {
    .dims = {1, 4}, .startRank = 0,
    .placement = {NCCLXFER_RESHARD_REPLICATE, NCCLXFER_RESHARD_SHARD(0)},
};
ncclXferReshardMesh_t dst_mesh = {
    .dims = {1, 4}, .startRank = 4,
    .placement = {NCCLXFER_RESHARD_REPLICATE, NCCLXFER_RESHARD_SHARD(0)},
};

// Pack the per-side tensor descriptors. localShape entries are in
// **elements** and only the first ndims slots are read.  dataPtr is
// NULL on the side this rank doesn't participate in (mirroring PyTorch
// DTensor's size-0 local tensor for non-participating ranks).  mesh is
// always required.
ncclXferDistTensor_t src = {
    .dataPtr    = is_source ? buffer : NULL,
    .localShape = {256, 1024, 0},
    .ndims       = 2,
    .dtype       = ncclFloat32,
    .mesh        = &src_mesh,
};
ncclXferDistTensor_t dst = {
    .dataPtr    = is_dest ? buffer : NULL,
    .localShape = {256, 1024, 0},
    .ndims       = 2,
    .dtype       = ncclFloat32,
    .mesh        = &dst_mesh,
};

ncclXferReshardWithWindow(comm, window, &src, &dst, stream);

// Tear down.
ncclCommWindowDeregister(comm, window);
ncclMemFree(buffer);
ncclXferReshardFinalize();    // releases internal caches + transpose buffer
```

The window must be registered on the **full** communicator regardless
of whether this rank's `dataPtr` is NULL on either side.

---

## Public API Reference

```c
#include "nccl_xfer.h"
```

### DistTensor

`ncclXferDistTensor_t` ("Distributed Tensor") names the same abstraction
training frameworks use: a logical tensor that is split across many
GPUs under a per-rank layout (sharded along one or more dimensions,
replicated on others), where each rank only ever holds and operates on
its local tile.  The closest public analogues are **PyTorch DTensor**
([torch.distributed.tensor]) and **JAX** sharded `jax.Array` (via
`NamedSharding(mesh, PartitionSpec)`); the placement vocabulary used
here (`REPLICATE`, `SHARD(d)`) maps 1-to-1 onto PyTorch's `Replicate()`
and `Shard(d)`.  Both PyTorch and JAX bundle the topology with the
per-rank tile (DTensor's `_spec.mesh`, JAX's `Array.sharding.mesh`),
and `ncclXferDistTensor_t` follows that convention: data + shape + dtype +
mesh in one descriptor.

```c
typedef struct {
    void*                     dataPtr;        // local buffer; NULL if this rank
                                               //   doesn't participate on this side
    size_t                    localShape[NCCLXFER_RESHARD_MAX_TENSOR_DIMS];  // elements
    int                       ndims;           // 1..3
    ncclDataType_t            dtype;           // element type
    const ncclXferReshardMesh_t*  mesh;            // topology + placement (caller-owned)
} ncclXferDistTensor_t;
```

`dtype` selects the element size. Supported: `ncclInt8`, `ncclUint8`,
`ncclFloat8e4m3`, `ncclFloat8e5m2`, `ncclFloat16`, `ncclBfloat16`,
`ncclInt32`, `ncclUint32`, `ncclFloat32`, `ncclInt64`, `ncclUint64`,
`ncclFloat64`.

A rank that doesn't participate on a given side passes a fully-formed
descriptor with `dataPtr = NULL` (mirroring DTensor's size-0 local
tensor for non-participating ranks).  `mesh` is required on every
rank — the library reads both meshes everywhere to compute who-holds-
which-shard.

[torch.distributed.tensor]: https://pytorch.org/docs/stable/distributed.tensor.html

### Reshard

```c
ncclResult_t ncclXferReshardWithWindow(
    ncclComm_t                comm,    // contains all ranks (src + dst)
    ncclWindow_t              window,  // registered on comm
    const ncclXferDistTensor_t*   src,     // source-side descriptor
    const ncclXferDistTensor_t*   dst,     // destination-side descriptor
    cudaStream_t              stream   // explicit stream, or default-stream sentinel
);
```

CTA count defaults to `DEFAULT_NUM_CTAS = 8` and can be capped with
`config.maxCta` / `NCCLXFER_RESHARD_MAX_CTA`. Chunking defaults to
`DEFAULT_ELEMENTS_PER_CHUNK = 32`; the RING path also honors
`NCCLXFER_RESHARD_CHUNK_SIZE` as a byte-level chunk override.

**Preconditions** (return `ncclInvalidArgument` if violated, except where
noted otherwise):

- `comm`, `window`, `src`, `dst`, `src->mesh`, `dst->mesh` are non-NULL.
- `src->ndims == dst->ndims`, both in `1..NCCLXFER_RESHARD_MAX_TENSOR_DIMS`
  (currently 3; 4-D is not supported).
- `src->dtype == dst->dtype` and is a supported dtype (see list above).
- `window` is registered on `comm` itself with `NCCL_WIN_COLL_SYMMETRIC`.
- `stream` is either an explicit CUDA stream or a default-stream sentinel
  (`NULL`, `cudaStreamLegacy`, or `cudaStreamPerThread`). Default-stream callers
  run on an internal non-blocking stream pool when enabled.
- Per-rank `src->dataPtr` / `dst->dataPtr` is either `NULL` for an inactive
  side or lies inside the registered window. If both pointers are present on
  one rank, their window offsets must match. Window-offset checks require
  NCCL ≥ 2.29.2 (`ncclWinGetUserPtr`); on older NCCL the offset symmetry is
  trusted. Violations of the single-offset contract abort the process via
  `RESHARD_FATAL` rather than returning `ncclInvalidArgument`.
- Each `localShape[shard_dim]` × shard count divides cleanly into the
  global tensor dim.

**Returns** `ncclSuccess` on success, otherwise an `ncclResult_t` from NCCL or
`ncclInvalidArgument` from the preconditions above.

**Threading** — the call is collective and follows CUDA stream semantics.
Issue a single reshard at a time per `(comm, effective stream)`. Use separate
communicators for concurrent transfers; the batched benchmark does this with
`--num-comms`.

### Lifecycle

```c
ncclResult_t ncclXferReshardInit(ncclXferReshardConfig_t* config); // idempotent; NULL = defaults
ncclResult_t ncclXferReshardFinalize(void); // releases caches + transpose buffer
```

`Finalize` should be called before the process exits to release the device
caches; running without `Finalize` is not a correctness bug but may show up as
leaked device memory at process teardown.

### Library configuration

Modeled after `ncclConfig_t`. Fill an `ncclXferReshardConfig_t` with
`NCCLXFER_RESHARD_CONFIG_INITIALIZER`, override the fields you care about,
and pass a pointer to `ncclXferReshardInit()`. `NULL` means "all defaults".
Fields left at `NCCLXFER_RESHARD_CONFIG_UNDEF_INT` keep the library default.

| Field    | Purpose |
|---|---|
| `maxCta` | Max number of CTAs used by reshard kernel. |

---

## Building

Both Make and CMake are supported; pick the one that fits your toolchain.

### Make targets

| Target | Output | Notes |
|---|---|---|
| `make` / `make lib`             | `build/lib/libnccl_xfer.so`                  | Library only; no MPI link. |
| `make reshard`                  | `build/bin/reshard_bench`                | Single-layer bench (links MPI). |
| `make reshard_batch_user_window` | `build/bin/reshard_batch_bench_user_window` | Batched/concurrent comm sweep. |
| `make reshard_model`            | `build/bin/reshard_model_bench`           | Config-driven model transfer bench (links MPI). |
| `make bench`                    | All bench binaries above                    | |
| `make bench reshard`            | Equivalent to `make reshard`                | Sub-name picker, see `make help`. |
| `make tests`                    | `basic_api_test_{mpi,local}`                | C-level functional matrix. |
| `make install`                  | Copies `lib` + `nccl_xfer.h` to `$PREFIX`  | Defaults `PREFIX=/usr/local`. |
| `make clean`                    | `rm -rf build/`                             | |

### CMake targets

```bash
cmake -B build -DNCCL_HOME="$NCCL_HOME" \
      [-DNCCL_XFER_BUILD_BENCH=ON] [-DNCCL_XFER_BUILD_TESTS=ON]
cmake --build build -j [--target <name>]
```

| `--target` | Output | Notes |
|---|---|---|
| *(default)*           | `build/lib/libnccl_xfer.{so,a}`            | Builds all configured targets. |
| `nccl_xfer_shared`    | `build/lib/libnccl_xfer.so`                | Library only. |
| `nccl_xfer_static`    | `build/lib/libnccl_xfer.a`                 | Static archive. |
| `reshard_bench` *etc.* | `build/bin/<name>`                        | Requires `-DNCCL_XFER_BUILD_BENCH=ON`. |
| `basic_api_test_*`    | `build/bin/<name>`                         | Requires `-DNCCL_XFER_BUILD_TESTS=ON`. |
| `unit_tests`          | `build/bin/unit_tests`                     | Private CI gtest suite; links library SRCs directly. |
| `install`             | Copies `lib` + headers to `CMAKE_INSTALL_PREFIX` | Defaults `/usr/local`. |

`ctest --test-dir build` runs the host-only subset of `unit_tests` +
`basic_api_test_local`.

### Required environment

| Variable | Default | Purpose |
|---|---|---|
| `NCCL_HOME` | *(unset; required)* | Path to a from-source NCCL build (`$NCCL_HOME/include/nccl_device.h` must exist). Make reads the env var directly; pass `-DNCCL_HOME=...` to `cmake`. |

### Optional environment / cache vars

| Make var | CMake equivalent | Default | Purpose |
|---|---|---|---|
| `CUDA_HOME` | auto-detected by `find_package(CUDAToolkit)` | `/usr/local/cuda` | CUDA install. CUDA ≥ 12.4 is needed for the default `sm_100` arch. |
| `MPI_HOME` | auto-detected by `find_package(MPI)` | system MPI | Used by benchmarks only (the library does not link MPI). |
| `PREFIX` | `CMAKE_INSTALL_PREFIX` | `/usr/local` | `install` destination. |
| `NVCC_GENCODE` | `CMAKE_CUDA_ARCHITECTURES` | `sm_80, sm_90, sm_100` (Make) / `80;90;100` (CMake) | Target GPU arch. |
| `DEBUG=1` / `DEBUG=full` | `-DCMAKE_BUILD_TYPE=Debug` (+ `-DCMAKE_CUDA_FLAGS_DEBUG=...`) | unset / Release | Line info / device debug. |
| `BUILDDIR` | `cmake -B <dir>` | `build/` | Output directory. |

---

## Benchmarks

All benches link MPI for NCCL bootstrap; rank 0 broadcasts the unique ID.

### Single layer — `reshard_bench`

Drives one reshard with the configuration given on the command line; runs
warmup + timed iterations, optionally validates the byte pattern.

```bash
# 8 GPUs, 2-D, same-dim sharding
mpirun -np 8 ./build/bin/reshard_bench \
    --src-mesh-dims 1,4 --dst-mesh-dims 1,4 \
    --tensor-dims 1024,1024 \
    --src-shard-dim 0 --dst-shard-dim 0 \
    --algorithm ring --validate

# 3-D cross-dim
mpirun -np 8 ./build/bin/reshard_bench \
    --src-mesh-dims 1,4 --dst-mesh-dims 1,4 \
    --tensor-dims 256,128,64 \
    --src-shard-dim 0 --dst-shard-dim 2 \
    --algorithm ring --validate
```

`--help` lists all flags (`--lb-mode`, `--print-all-ranks`,
`--verbose`, ...).

### Batched — `reshard_batch_bench_user_window`

Sweeps tensor sizes × shard-dim pairs and compares **sequential** vs
**concurrent** issue across `--num-comms` independent NCCL communicators.

```bash
mpirun -np 16 ./build/bin/reshard_batch_bench_user_window \
    --src-mesh-dims 1,8 --dst-mesh-dims 1,8 \
    --tensor-dims '256,256:1024,1024:4096,4096' \
    --src-shard-dims 0,0 --dst-shard-dims 0,1 \
    --num-comms 2 --num-tensors 4 \
    --validate
```

### Model transfer — `reshard_model_bench`

A config-driven benchmark that measures disaggregated model resharding using
real model parameter shapes, dtypes, and parallelism configs. Unlike the
synthetic `reshard_bench` (single-layer), this benchmark reads HuggingFace-style model config
and system config JSON files, automatically computing placement rules, expert
grouping, PP stage mapping, and deduplication.

#### How It Works

```
                                  ┌──────────────────────────────┐
                                  │  model config JSON           │
                                  │  (per-param shape + dtype)   │
                                  └──────────┬───────────────────┘
                                             │
                         ┌───────────────────▼────────────────────┐
                         │  1. Group expert params into 3D tensors│
                         │  2. Deduplicate across layers (opt.)   │
                         │  3. Compute placement rules per param  │
                         │  4. Build transfer descriptors          │
                         └───────────────────┬────────────────────┘
                                             │
                  ┌──────────────────────────▼──────────────────────────┐
                  │  system config JSON                                  │
                  │  (train/gen: num_gpus, TP, CP, EP, DP, PP)          │
                  └──────────────────────────┬──────────────────────────┘
                                             │
            ┌────────────────────────────────▼────────────────────────────────┐
            │  Per (train_stage, gen_stage) PP pair:                           │
            │    - Create NCCL communicator (ncclCommInitRank)                 │
            │    - Create dedicated CUDA stream                               │
            │                                                                 │
            │  Per transfer descriptor (param):                               │
            │    - Allocate symmetric buffer (ncclMemAlloc)                   │
            │    - Register ncclWindow_t (ncclCommWindowRegister)              │
            └────────────────────────────────┬────────────────────────────────┘
                                             │
                    ┌────────────────────────▼─────────────────────────┐
                    │  Warmup → Per-pattern: [Validation] → Timed runs │
                    │  → Aggregate summary                             │
                    └──────────────────────────────────────────────────┘
```

**Pipeline steps in detail:**

1. **Parse configs** -- model config gives per-parameter global shapes and dtypes; system config gives train/gen parallelism (TP, CP, EP, DP, PP, num_gpus).

2. **Expert grouping** -- individual per-expert 2D weight matrices (e.g. `layers.0.experts.0.gate_proj.weight`) are combined into a single 3D tensor (e.g. `layers.0.experts.gate_proj.weight` with shape `[num_experts, ...]`).

3. **Deduplication** -- with PP, many layers share the same (shape, placement, PP-stage-pair) pattern. By default, only one representative per pattern is benchmarked, reducing runtime without affecting accuracy. Disable with `--no-dedup`.

4. **Placement rules** --
   - **Column-parallel (Shard dim 0)**: q/k/v/gate/up projections, embed_tokens, lm_head
   - **Row-parallel (Shard dim 1)**: o/down projections
   - **Expert-parallel (Shard dim 0)**: `.experts.` params sharded across EP
   - **Replicated**: layernorms, biases, MoE router gate, 1D params

5. **Transfer descriptors** -- for each param: local shapes (after sharding), mesh specs (rep count, shard count, shard tensor dim), PP stage assignment, byte sizes.

6. **NCCL communicators** -- one `ncclComm_t` and one `cudaStream_t` per (train_stage, gen_stage) pair. Trainer ranks occupy `[0, trainStageSize)` and generator ranks occupy `[trainStageSize, trainStageSize + genStageSize)` within each sub-communicator.

7. **Buffer allocation** -- one symmetric buffer per transfer descriptor (i.e. per parameter). Each buffer is registered as an `ncclWindow_t` on the corresponding PP communicator. This per-param allocation ensures that multiple parameters within the same pattern group can be in flight simultaneously without data corruption, which is critical for validation correctness.

8. **Execution** -- all transfers in a pattern group are launched asynchronously on per-PP-comm CUDA streams (NCCL serializes ops on the same communicator), then synchronized once. Per-pattern and aggregate bandwidth/latency are reported.

#### Input File Formats

**Model config JSON** (generated by `hf_converter.py` from a HuggingFace repo):

```json
{
  "lm_head.weight": { "shape": [129280, 7168], "dtype": "BF16" },
  "model.layers.0.self_attn.q_proj.weight": { "shape": [7168, 7168], "dtype": "F8_E4M3" },
  "model.layers.0.input_layernorm.weight": { "shape": [7168], "dtype": "BF16" }
}
```

**System config JSON**:

```json
{
  "train": {
    "num_gpus": 32,
    "tp_size": 1,
    "cp_size": 1,
    "ep_size": 16,
    "dp_size": 1,
    "pp_size": 2
  },
  "generation": {
    "num_gpus": 32,
    "tp_size": 8,
    "cp_size": 1,
    "ep_size": 1,
    "dp_size": 4,
    "pp_size": 1
  }
}
```

Total MPI world size must equal `train.num_gpus + generation.num_gpus`.
Ranks `[0, train.num_gpus)` are trainers; ranks
`[train.num_gpus, total)` are generators.

#### Building

```bash
make bench reshard_model
make bench reshard_model NVCC_GENCODE="-gencode=arch=compute_100,code=sm_100"
```

#### Running

```bash
mpirun -np <total_gpus> ./build/bin/reshard_model_bench \
    --model-config benchmarks/configs/model_configs/<model>.json \
    --system-config benchmarks/configs/system_configs/<system>.json \
    --iterations 10 --warmup 2 \
    --algorithm auto --lb-mode node \
    --validate
```

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--model-config <file>` | required | Model config JSON with HuggingFace-style per-parameter shapes and dtypes. |
| `--system-config <file>` | required | System config JSON with train/generation parallelism. |
| `--iterations <N>` | 10 | Timed iterations per pattern. |
| `--warmup <N>` | 2 | Warmup iterations. |
| `--gpus-per-node <N>` | 8 | GPUs per node for load balancing. |
| `--algorithm <auto\|ring\|direct>` | auto | Reshard algorithm selection. |
| `--lb-mode <uniform\|node>` | uniform | Load-balance mode. |
| `--no-dedup` | off | Benchmark all layers instead of one representative per repeated pattern. |
| `--validate` | off | Run correctness validation before timing. |
| `--validate-iterations <N>` | 3 | Validation iterations per pattern. |

#### Validation Mode

With `--validate`, the benchmark runs a per-pattern correctness check before
the timed iterations for each pattern group:

1. Trainer ranks initialize each parameter's dedicated buffer with a
   deterministic byte pattern based on global coordinates; generator ranks zero
   their buffers.
2. All transfers in the pattern group execute in byte mode (`element_size=1`,
   last tensor dimension scaled by the actual element size).
3. Streams synchronize and an MPI barrier ensures all transfers complete.
4. Generator ranks verify each parameter's received data matches the expected
   pattern.
5. Per-pattern pass/fail is reported, and a global `VALIDATION PASSED` or
   `VALIDATION FAILED` summary is printed after all patterns.

Since each parameter has its own buffer, multiple parameters within a pattern
group are validated without data conflicts. Validation operates in byte space
to work with multi-byte dtypes like BF16 and FP8.

#### Reference Performance

Measured end-to-end model transfer latency from `reshard_model_bench` on a
GB200 NVL72 cluster (4 GPUs/node, NVLink intra-rack). Both runs use
`--validate --no-dedup`, so every layer's transfers are timed (not just one
representative per pattern). "Max latency" is the final-aggregate
`Latency Max` line from the benchmark summary: the slowest rank's wall-clock
summed across all per-pattern timings.

The two columns compare the hierarchical algorithm (intra-NVL fan-out plus
cross-NVL ring) against the direct point-to-point algorithm (every source rank
issues GIN puts to every destination rank), holding everything else fixed:

- **Hierarchical**: `--algorithm ring --lb-mode node`
- **Direct P2P**: `--algorithm direct --lb-mode uniform`

| Model | Cluster | GPUs (train + gen) | Trainer parallelism | Generator parallelism | Hierarchical max latency (ms) | Direct P2P max latency (ms) | Speedup |
|---|---|---|---|---|---:|---:|---:|
| DeepSeek-V3 (`dsv3.model.json`) | GB200 NVL72 | 256 (128T / 2 NVL + 128G / 2 NVL) | `TP=1, CP=1, EP=16, DP=2, PP=4` | `TP=8, CP=1, EP=1, DP=16, PP=1` | 1447.44 | 3837.74 | 2.65x |
| Qwen3-235B (`qwen3-235b.model.json`) | GB200 NVL72 | 128 (64T / 1 NVL + 64G / 1 NVL) | `TP=2, CP=2, EP=16, DP=1, PP=4` | `TP=8, CP=1, EP=1, DP=8, PP=1` | 988.75 | 2175.16 | 2.20x |

The `dsv3.model.json` and `qwen3-235b.model.json` files are derived from the
public HuggingFace model cards
([deepseek-ai/DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json),
[Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B/blob/main/config.json))
and are not bundled here; the tree ships only
`dsv3-toy.model.json` (see the "Shipped configs" subsection below).

#### Shipped configs

This tree ships a toy DeepSeek-style model config at
`benchmarks/configs/model_configs/dsv3-toy.model.json` plus system config
examples under `benchmarks/configs/system_configs/`:

- `dsv3-128gpus-gb200.json`
- `dsv3-256gpus-gb200.json`
- `dsv3-pp-64gpus-gb200.json`
- `qwen235b-TP2CP2PP4EP16-128gpus-gb200.json`

---

## Tests

A C-level functional matrix lives in [`tests/`](tests/README.md), exposed as
two gtest binaries that share a single descriptor table:

| Binary | Bootstrap | Use when |
|---|---|---|
| `basic_api_test_mpi`   | MPI (`mpirun`) | Cluster runs, large rank counts. |
| `basic_api_test_local` | Single-process pthreads via `ncclCommInitAll` | Dev workstation, no MPI install. |

Build:

```bash
make tests
```

Run:

```bash
# Single-host, no MPI
./build/bin/basic_api_test_local --filter full_sharding

# Multi-host
mpirun -np 8 ./build/bin/basic_api_test_mpi --filter 2d_placement
```

Case groups: `full_replication`, `full_sharding`, `2d_placement`,
`uneven_ratio`, `tensor_size_sensitivity`, `nd_tensors`, `cross_dim_regression`,
plus 1-D analogues. See [`tests/README.md`](tests/README.md) for the full
matrix and the `--list` / `--min-world` / `--max-world` flags used to bin a CI
run into rank tiers. The C-level matrix covers RING and DIRECT; the MPI binary
also accepts `--algorithm all`.

Exit code is `1` if any case reports `FAIL`, `0` otherwise. `SKIP` does
not fail the run.

---

## Tuning

**Algorithm**

- `NCCLXFER_RESHARD_ALGORITHM=AUTO` (default) — currently falls through to `RING`; no
  NVL-domain auto-detection in this build.
- `NCCLXFER_RESHARD_ALGORITHM=RING` — hierarchical ring + intra-NVL fan-out via the input
  comm's window. Best for cross-NVL transfers and scales linearly with
  bandwidth.
- `NCCLXFER_RESHARD_ALGORITHM=DIRECT` — every src rank issues GIN puts directly to every
  dst rank. Lower latency for small transfers; higher pressure on the NIC
  and on the prepare-time fan-out.

**Load balance**

- `NCCLXFER_RESHARD_LB_MODE=UNIFORM` (default) — splits work evenly by rank count.
- `NCCLXFER_RESHARD_LB_MODE=NODE_AWARE` — bias the assignment so each NVL domain serves
  its local peers first; benefits cross-NVL fan-in.

**CTA count and chunk granularity** — CTA count resolves once during
`ncclXferReshardInit`: built-in default 8, then optional `config.maxCta`, then
`NCCLXFER_RESHARD_MAX_CTA` if set. `pickElementsPerChunk` currently returns the
compile-time default (`DEFAULT_ELEMENTS_PER_CHUNK = 32`). The RING prepare path
also uses `CHUNK_SIZE_BYTES` (256 KB) as a byte-level chunk size, overridable
per-process via `NCCLXFER_RESHARD_CHUNK_SIZE` (bytes).

**Cross-dim transpose** — when cross-dim sharding would produce per-rank
inner strides below `CROSS_DIM_TRANSPOSE_THRESHOLD` (256 KB), the library
transparently transposes dimensions into a private buffer to keep GIN puts
large. Applies to both 2-D and 3-D tensors; transparent to callers.

---

## Runtime environment variables

Most env vars are read once in `ncclXferReshardInit`. Env vars always override
matching fields of `ncclXferReshardConfig_t` (matches upstream NCCL's
`envConfigOverride` precedence). `NCCLXFER_RESHARD_CHUNK_SIZE` is read by the RING
prepare path for each call.

| Variable | Effect |
|---|---|
| `NCCLXFER_RESHARD_LOG_LEVEL`        | One of `NONE`, `WARN` (default), `INFO`, `DEBUG`, `TRACE`. |
| `NCCLXFER_RESHARD_ALGORITHM`        | `AUTO` (default), `RING`, or `DIRECT`. |
| `NCCLXFER_RESHARD_LB_MODE`          | `UNIFORM` (default) or `NODE_AWARE`. |
| `NCCLXFER_RESHARD_MAX_CTA`          | Overrides `config.maxCta`. |
| `NCCLXFER_RESHARD_STREAM_POOL_SIZE` | Max distinct `(comm, dev)` entries in the internal stream pool (default 4). |
| `NCCLXFER_RESHARD_CHUNK_SIZE`       | Override the library's default chunk size in **bytes**. |

---

## Repository layout

```
.
├── src/                                  # Library
│   ├── nccl_xfer.h                    # Unified public C API (the only header callers need)
│   ├── reshard_internal.h                # Cross-TU function declarations (internal)
│   ├── reshard_types.h                   # Internal struct definitions
│   ├── reshard_limits.h                  # Compile-time constants (MAX_*, defaults)
│   ├── reshard_log.h                     # RESHARD_LOG / RESHARD_DEBUG tier macros
│   ├── reshard_checks.h                  # Error-checking macros
│   ├── reshard_kernels.cuh               # Header-only device helpers
│   ├── reshard_config.cc                 # Config/env parsing                  (host)
│   ├── reshard_init.cc                   # Init / Finalize                     (host)
│   ├── reshard_cache.cc                  # DevComm + Window caches             (host)
│   ├── reshard_mesh.cc                   # Mesh analysis helpers               (host)
│   ├── reshard_loadbalance.cc            # Replication load balancer           (host)
│   ├── reshard_prepare.cc                # Kernel-parameter builders           (host)
│   ├── reshard_transpose.cc              # Cross-dim transpose buffer mgmt     (host)
│   └── reshard_user_window.cu            # ncclXferReshardWithWindow + CUDA kernels
├── benchmarks/
│   ├── Makefile                          # Bench build rules (delegated to from root)
│   ├── bench_common.h                    # Host-only macros + arg parsing
│   ├── bench_common_kernels.{h,cu}       # Shared validation kernels + launchers
│   ├── reshard_bench.cc                  # Single-layer bench (host-only driver)
│   ├── reshard_batch_bench_user_window.cu  # Batched / concurrent-comms bench
│   ├── reshard_model_bench.cu            # Config-driven model transfer bench
│   └── configs/                          # Bench config JSONs
│       ├── model_configs/                # Per-model config JSONs
│       ├── system_configs/               # Per-system topology JSONs
│       ├── sample_layer_config.json      # Single-stage example config
│       ├── sample_layer_config_multi.json  # Multi-stage example config
│       └── moe.json                      # MoE-shaped tensor config
├── tests/                                # C-level API tests + helpers
│   ├── Makefile                          # Test build rules (delegated to from root)
│   ├── README.md                         # Case matrix + CLI flags
│   ├── basic_api_test_core.h             # Shared test descriptor table
│   ├── basic_api_test_mpi.cc             # MPI-bootstrapped test binary
│   ├── basic_api_test_local.cc           # Single-process pthreads test binary
│   └── pytorch/                          # PyTorch mxn_cast binding tests
├── third_party/                          # Vendored deps (nlohmann/json for benchmark JSON parsers)
├── Makefile
├── README.md
├── RELEASE.md
├── ThirdPartyNotices.txt
└── Makefile.common
```

---

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ncclInvalidArgument` from `ncclXferReshardWithWindow` | One of the preconditions failed: NULL comm/window/descriptor/mesh, mismatched `ndims`/dtype, `ndims` outside 1..3, or an unsupported dtype. |
| Validation fails with destination still containing the pre-call bytes | The kernel did not write to dest. Re-run with `NCCLXFER_RESHARD_LOG_LEVEL=DEBUG` to see the prepared plan, then file an issue with the `reshard_bench` command line that reproduces the failure. |
| `nccl_device.h: No such file or directory` at compile time | `NCCL_HOME` points at a binary install rather than a from-source build. Build NCCL from source or point at one. |
| Fast-but-wrong: `make` succeeds yet runtime crashes with "illegal instruction" | Often a downstream symptom of a kernel that completed with corrupt state on the previous reshard call. Re-run with `NCCLXFER_RESHARD_LOG_LEVEL=DEBUG` to see the prepared plan. |

---

## Contributing

Issues and merge requests are tracked on this same GitLab project. When
reporting a bug, include:

- Cluster / GPU model and NCCL version.
- The mesh shape, tensor shape, shard dims, algorithm, and load-balance mode
  used.
- The minimal `reshard_bench` command line that reproduces the issue.

See `RELEASE.md` for release history.

---

## License

Apache-2.0 in the NCCL contrib drop, inherited from the parent `nccl/nccl`
`LICENSE.txt`. Third-party notices for vendored dependencies are in
[`ThirdPartyNotices.txt`](ThirdPartyNotices.txt). © NVIDIA Corporation.
