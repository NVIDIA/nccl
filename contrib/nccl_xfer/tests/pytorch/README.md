# `mxn_cast` PyTorch tests

Standalone PyTorch-level correctness tests for
`torch.distributed._symmetric_memory.mxn_cast`, the binding that wraps
`ncclXferReshardWithWindow`.

The main test runner is `basic_api_test.py`. It mirrors the C-level
`tests/basic_api_test_core.h` matrix and a PyTorch-level pytest matrix
as closely as the binding permits.

## Files

| File | Purpose |
|---|---|
| `basic_api_test.py` | Distributed correctness matrix for `symm_mem.mxn_cast`. |

## Matrix

| Group | Minimum world | Tensor shape | dtype | Notes |
|---|---|---|---|---|
| `full_replication` | `world >= 4`, even | `(200, 200)` | fp32 / bf16 / uint8 | Semantically full replicate on both sides. The runner encodes this as a one-shard placement; pure `{REPLICATE, REPLICATE}` is a degenerate prepare-time case the test suite does not exercise. |
| `full_sharding` | `world >= 4`, even | `(200, 200)` | fp32 / bf16 / uint8 | 1D mesh per side, `sharding_dims` in `{(0,0),(0,1),(1,0),(1,1)}`. |
| `2d_placement` | `world >= 4`, even | `(200, 200)` | bf16 | `n_shards` in `{(2,4),(4,2),(2,2)}`, placement in `{sr/sr,rs/rs,sr/rs,rs/sr}`. |
| `uneven_ratio` | `world >= 4`, `world % 4 == 0` | `(240, 240)` | bf16 | Ratio in `{(3,1),(1,3)}`, `n_shards=(2,2)`. |
| `tensor_size_sensitivity` | `world >= 4`, even | `576^2`, `3072x6144`, `3072^2` | bf16 | `n_shards=(4,4)`. Preserves the external pytest skip for large `rs/rs` cases below `world_size=32`. |
| `nd_tensors` | `world >= 4`, even | `(64,128,128)`, `(128,64,64)` | bf16 | 3D shapes with the xferdtensor sharding-dim matrix; the `(64,128,128), sd=(0,1)` historical case lives in `cross_dim_regression`. |
| `1d_*` | `world >= 4`, even | `8192`, `16384`, `1048576`, `4194304` | fp32 / bf16 / uint8 for `1d_full_sharding`, bf16 for placement groups | C-level matrix extension for 1D tensor coverage. |
| `cross_dim_regression` | `world >= 8`, even | `(200,200)`, `(64,128,128)` | bf16 | Targeted historical cross-dim regression cases from the current C-level suite. |

## Local syntax/list checks

These do not require GPUs:

```bash
python3 -m py_compile tests/pytorch/basic_api_test.py
python3 tests/pytorch/basic_api_test.py --list
```

## Distributed run

Useful overrides:

| Var | Default | Notes |
|---|---|---|
| `FILTER` | empty | Substring filter over case names. |
| `START_INDEX` | `0` | Skip the first N selected cases. |
| `MIN_WORLD` / `MAX_WORLD` | empty | Select by minimum feasible world. |
| `LIMIT` | empty | Stop after N selected cases. |
| `CASE_CHUNK_SIZE` | `60` | Cases per Python process; keeps the NCCL Xfer DevComm cache below its 64-entry limit. Set `0` to run one process. |
| `STREAM_MODE` | `non-default` | `default`, `non-default`, or `both`. |
| `FAIL_FAST` | `0` | Stop at first failure. |
| `REUSE_WORLD_GROUP` | `0` | Reuse `dist.WORLD`; by default the runner creates a fresh all-rank NCCL group per case to avoid cross-case `mxn_cast` cache reuse. |

Launch the runner with any distributed launcher that sets the standard
`RANK`, `WORLD_SIZE`, and `LOCAL_RANK` environment variables, for example:

```bash
torchrun --nproc-per-node 8 tests/pytorch/basic_api_test.py
```
