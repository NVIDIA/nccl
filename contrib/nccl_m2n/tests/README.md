# `tests/` — basic_api correctness suite

Two gtest binaries exercise `ncclReshardWithWindow` across a parameter
matrix mirroring an external pytest reference suite. Both share the test
descriptor table and per-case logic in `basic_api_test_core.h`; only the
bootstrap differs.

| Binary | Bootstrap | Use when |
|---|---|---|
| `build/bin/basic_api_test_mpi`   | MPI (`mpirun`)              | Multi-host MPI runs, large rank counts. |
| `build/bin/basic_api_test_local` | Single-process pthreads via `ncclCommInitAll` | Dev workstation, no MPI install required. |

Both use only the public API in `src/nccl_m2n.h`.

## Building

```sh
make tests
```

`make tests` builds the public basic_api binaries when a googletest source
tree is available. `tests/gtest.mk` looks for one in this order:

1. `$(GTEST_DIR)` — caller-supplied (highest priority).
2. `tests/googletest/` — vendored alongside the tests.

If neither resolves, `make tests` prints a friendly skip message and
returns success — the library and benchmarks build either way. To build
the tests against a googletest checkout
(<https://github.com/google/googletest>):

```sh
make tests GTEST_DIR=/path/to/googletest
```

## Test groups (mirror of pytest matrix)

| Group | Minimum world | Tensor shape | dtype | Notes |
|---|---|---|---|---|
| `full_replication`         | `world ≥ 4`, even           | (200, 200) | fp32 / bf16 / uint8 | 1D mesh per side, both replicate |
| `full_sharding`            | `world ≥ 4`, even           | (200, 200) | fp32 / bf16 / uint8 | 1D mesh per side, sharding_dims ∈ {(0,0),(0,1),(1,0),(1,1)} |
| `2d_placement`             | `world ≥ 4`, even           | (200, 200) | bf16 | n_shards ∈ {(2,4),(4,2),(2,2)}, placement ∈ {sr/sr, rs/rs, sr/rs, rs/sr} |
| `uneven_ratio`             | `world ≥ 4`, `world % 4 == 0` | (240, 240) | bf16 | ratio ∈ {(3,1),(1,3)}, n_shards (2,2) |
| `tensor_size_sensitivity`  | `world ≥ 4`, even           | 576², 3072², 3072×6144 | bf16 | n_shards (4,4) |
| `nd_tensors`               | `world ≥ 4`, even           | (64,128,128), (128,64,64) | bf16 | 3D only; the `(64,128,128), sd=(0,1)` historical case lives in `cross_dim_regression` |
| `1d_full_sharding`            | `world ≥ 4`, even           | (8192,)            | fp32 / bf16 / uint8 | 1D mesh, sd=(0,0) (only one tensor axis) |
| `1d_2d_placement`             | `world ≥ 4`, even           | (8192,)            | bf16 | n_shards ∈ {(2,4),(4,2),(2,2)}, placement ∈ {sr/sr, rs/rs, sr/rs, rs/sr} |
| `1d_uneven_ratio`             | `world ≥ 4`, `world % 4 == 0` | (16384,)         | bf16 | ratio ∈ {(3,1),(1,3)}, n_shards (2,2) |
| `1d_tensor_size_sensitivity`  | `world ≥ 4`, even           | (16384,), (1048576,), (4194304,) | bf16 | n_shards (4,4) |
| `cross_dim_regression`        | `world ≥ 8`, even           | (200, 200), (64,128,128) | bf16 | Targeted historical cross-dim regressions for transpose and non-transpose paths |

Each case carries additional **runtime feasibility checks** (n_shards
must divide `src_total` / `dst_total`, the chosen tensor dim must divide
by the shard count). Cases that don't fit at the current world size are
reported as `SKIP` and don't fail the run. `world_min` is set to the
smallest world where *some* case in the group can possibly run; the
pytest source uses higher floors (8 / 32) but those are arbitrary —
the C kernel has no inherent 8-rank requirement.

## Sample invocations

### Single-host, no MPI

```sh
# List the case matrix without running anything.
./build/bin/basic_api_test_local --list

# Default: ranks = cudaGetDeviceCount(); skip cases that need more.
./build/bin/basic_api_test_local

# Smaller rank count (e.g., a 2-GPU box).
./build/bin/basic_api_test_local -N 2

# Run only one group.
./build/bin/basic_api_test_local --filter full_replication

# Switch to the direct-put kernel.
./build/bin/basic_api_test_local --algorithm direct
```

### Multi-host, MPI

```sh
# Wrapper script (defaults to N=8):
tests/run_basic_api_tests.sh
tests/run_basic_api_tests.sh -N 32 --filter nd_tensors

# Or directly:
mpirun -np 8 ./build/bin/basic_api_test_mpi --filter 2d_placement
```

## CLI flags (both binaries)

| Flag | Effect |
|---|---|
| `--list`                              | Print `idx min_world name` per case and exit (skips NCCL init). Honors `--filter` / `--max-world` / `--min-world`. |
| `--filter <substr>`                   | Only run cases whose name contains `<substr>`. |
| `--gtest_filter=<pattern>`            | Native gtest filtering over sanitized parameter names, after the custom matrix filters above. |
| `--max-world <N>`                     | Skip cases whose minimum required world size is `> N`. |
| `--min-world <N>`                     | Skip cases whose minimum required world size is `< N`. |
| `--algorithm ring\|direct`            | Algorithm selection (default ring). `basic_api_test_mpi` also accepts `all`, registering one parameter per algorithm. |
| `--lb-mode uniform\|node`             | Load-balance mode (default uniform). |
| `--verbose`                           | Library verbose + per-rank diagnostics. |
| `-N <ranks>` *(local only)*           | Number of ranks/threads (clamped to device count). |

### Min-world introspection (CI / scheduler integration)

Each test case carries dynamic preconditions (n_shards must divide src/dst
totals, global tensor dim must divide the resulting shard count). The
helper `computeMinWorldForCase` walks those preconditions to find the
smallest world size at which a case actually runs, and `--list` exposes it
so a CI script can request just enough ranks per batch:

```
$ ./build/bin/basic_api_test_local --list | head
# total_cases=226
# columns: idx min_world name
[   0]    4 full_replication[gd=200x200,esz=4]
[   5]    4 full_sharding[gd=200x200,sd=0/0,esz=4]
[  25]    8 2d_placement[gd=200x200,m=2x4_sr/sr,sd=0/0,esz=2]
[  73]    8 uneven_ratio[gd=240x240,m=2x2_sr/sr,sd=0/0,esz=2,ratio=3:1]
[ 105]    8 tensor_size_sensitivity[gd=576x576,m=4x4_sr/sr,sd=0/0,esz=2]
```

Combine `--max-world` and `--min-world` to bin a CI run into rank tiers
and provision exactly the allocation each tier needs:

```bash
# Tier 1: smoke at 4 ranks
mpirun -np 4  basic_api_test_mpi --max-world 4

# Tier 2: cases that need 5..8 ranks, run at world=8
mpirun -np 8  basic_api_test_mpi --min-world 5 --max-world 8

# Tier 3: cases that need >8 ranks, run at world=32
mpirun -np 32 basic_api_test_mpi --min-world 9 --max-world 32
```

## Exit code

Returns `1` if any case reports `FAIL`, `0` otherwise. `SKIP` does not
fail the run.

## What the validator checks

Every byte in the global tensor takes value `(global_idx % 256)`.
Source ranks initialize their slice with that pattern; destination
ranks read the post-reshard contents and confirm each byte matches the
expected value at its global index. The validator is exact (no
`atol` tolerance), works for any `element_size`, and is correct for
both same-dim and cross-dim sharding because it depends only on each
byte's global position — not on where the data was sourced from.

## Case naming

`<group>[gd=DxD,m=AxB_<sr|rs>/<sr|rs>,sd=S/D,esz=N,ratio=R:R]`

- `gd` — global tensor dims
- `m` — mesh axis-0 sizes (src x dst)
- placement strings are `sr` (Shard outer, Replicate inner) or `rs` (Replicate outer, Shard inner)
- `sd` — sharding dims (src/dst)
- `esz` — element size in bytes (1=uint8, 2=bf16, 4=fp32)
- `ratio` only present for uneven src/dst splits

Filter substrings target any portion of the name, so
`--filter "esz=2"` runs every bf16 case across all groups, and
`--filter "sr/rs"` runs every "sr→rs" placement combo.

---

# Smoke test

A practical pre-merge smoke is to run the broad C-level functional
matrix plus the RING/DIRECT x same-dim/cross-dim acceptance combos with
`--validate`, then check the PASS/FAIL and bandwidth summary. Exits are
non-zero on any failure so the run can be used as a single CI gate.

## What it runs

| § | Check | Expected output |
|---|---|---|
| §1 | `basic_api_test_mpi` — full C-level matrix | Cases run/skip, `0` exit if no FAIL |
| §2a | `reshard_bench --validate` RING cross-dim (`sd=0 dd=1`) | `*** VALIDATION PASSED ***`, bandwidth |
| §2b | `reshard_bench --validate` RING same-dim (`sd=0 dd=0`) | same |
| §2c | `reshard_bench --validate` DIRECT cross-dim (`sd=0 dd=1`) | same |
| §2d | `reshard_bench --validate` DIRECT same-dim (`sd=0 dd=0`) | same |
| §3 | Summary table | one row per check with PASS/FAIL + bandwidth |

Logs for each check land under a per-job directory printed at the end of
the run.

Cluster-specific launch wrappers and perf-regression workflows are not
shipped with this contrib; copy `run_basic_api_tests.sh` and adapt its
environment for your system.
