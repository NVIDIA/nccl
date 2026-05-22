# NCCL Xfer — Release Notes

NCCL Xfer is an experimental, NCCL-based library for cross-group GPU data
movement. This preview scopes the surface to the **reshard** functionality: redistribute a
global tensor between two disjoint groups of GPU processes (the source group
holds one sharding / replication layout, the destination group holds
another). Future releases may extend the same library to other cross-group
transfer primitives under the same `NCCL Xfer` umbrella. The reshard
functionality is built on NCCL's user-window API (`ncclWindow_t` +
`ncclMemAlloc`) and on the NCCL Device API (LSA load/store and GIN
put/signal), so transfers are zero-copy, one-sided, and have no host
involvement on the critical path.

Install artifacts are shared library `libnccl_xfer.so` and public header
`include/nccl_xfer.h`. The `ncclXferReshard*` API surface combines the library
prefix (`ncclXfer`) with the functional scope (`Reshard`) for this release.

## v0.1

Initial preview of NCCL Xfer — reshard functionality.

- **Public C API** in `src/nccl_xfer.h`:
  - `ncclXferReshardWithWindow` — single-shot reshard against a caller-registered
    `ncclWindow_t`.
  - `ncclXferDistTensor_t` descriptor — bundles per-rank tile, dtype, and mesh
    in one struct, modeled after PyTorch DTensor / JAX `NamedSharding`.
  - `ncclXferReshardConfig_t` with `NCCLXFER_RESHARD_CONFIG_INITIALIZER`; currently
    exposes `maxCta`.
  - Lifecycle helpers `ncclXferReshardInit(config|NULL)` /
    `ncclXferReshardFinalize`. Algorithm, load-balance mode, stream-pool size,
    logging, and chunk sizing are env-driven.
- **Resharding kernels:**
  - **Ring** — hierarchical ring + intra-NVL fan-out via the user window.
  - **Direct** — per-rank GIN puts.
  - Transparent cross-dim transpose for 2-D and 3-D layouts when cross-dim
    sharding would otherwise create small innermost transfers.
- **Window contract:** participating pointers must be inside the supplied
  symmetric-memory window; source and destination pointers on the same rank
  must share a single window offset. Matching non-zero offsets are supported.
- **Default stream support:** callers may pass `NULL`, `cudaStreamLegacy`, or
  `cudaStreamPerThread`; the library uses an internal non-blocking stream pool
  and records a back-edge event so later default-stream work observes the
  result.
- **Benchmarks** under `benchmarks/`:
  - `reshard_bench` — single-layer bench (canonical worked example).
  - `reshard_batch_bench_user_window` — batched sequential vs concurrent
    comms sweep.
  - `reshard_model_bench` — config-driven model transfer bench.
- **Tests** under `tests/`:
  - `basic_api_test_mpi` and `basic_api_test_local` — C-level functional
    matrix mirroring an external pytest reference suite. Groups:
    `full_replication`, `full_sharding`, `2d_placement`, `uneven_ratio`,
    `tensor_size_sensitivity`, `nd_tensors`, `cross_dim_regression`, plus 1-D
    analogues and FP8 coverage where the NCCL enum is available.
  - `--list` / `--min-world` / `--max-world` introspection for binning a
    CI run into rank tiers.
  - PyTorch-level `mxn_cast` binding matrix under `tests/pytorch/`.
- **Runtime environment variables:**
  - `NCCLXFER_RESHARD_LOG_LEVEL` — `NONE` / `WARN` / `INFO` / `DEBUG` / `TRACE`.
  - `NCCLXFER_RESHARD_ALGORITHM` — `AUTO`, `RING`, or `DIRECT`.
  - `NCCLXFER_RESHARD_LB_MODE` — `UNIFORM` or `NODE_AWARE`.
  - `NCCLXFER_RESHARD_MAX_CTA` — overrides `config.maxCta`.
  - `NCCLXFER_RESHARD_STREAM_POOL_SIZE` — caps internal default-stream pool
    entries; `0` disables the pool.
  - `NCCLXFER_RESHARD_CHUNK_SIZE` — override the default 256 KB byte-level
    chunk size in the RING prepare path.

## Known Limitations

This preview is experimental.

| Limitation | Description |
|---|---|
| **Limited QA coverage**           | Functional matrix is the C-level basic_api suite × {RING, DIRECT} × dtype mix. Large multi-node coverage is still cluster-limited and workload-specific. |
| **Tensor rank ≤ 3**               | `NCCLXFER_RESHARD_MAX_TENSOR_DIMS = 3`. 4-D and higher are not supported. |
| **Both-REPLICATE meshes unsupported** | `placement = {REPLICATE, REPLICATE}` falls into a degenerate prepare-time branch that the test suite does not exercise. Encode full replication as a 1-shard layout (mesh axis of size 1). |
| **Single-offset window contract** | Per-rank source and destination pointers must have the same offset within the registered window when both are present; asymmetric source/destination offsets are rejected. |
| **Cross-rank offset symmetry trusted** | The API validates local offset consistency but does not currently perform a cross-rank collective check that every rank uses the same offset. |
| **Algorithm auto-select unimplemented** | `NCCLXFER_RESHARD_ALGORITHM=AUTO` currently aliases to `RING`; no input/topology-aware algorithm picker in this build. |
| **Single in-flight reshard per `(comm, effective stream)`** | The internal DevComm/window/transpose caches are designed for sequential use on a comm. Use separate communicators for concurrent transfers, as in `reshard_batch_bench_user_window --num-comms`. |
| **Not thread-safe — process-wide single-thread access** | The init-time globals and the internal caches (DevComm cache, window cache, stream pool) are process-wide shared state. Caller is responsible for serializing every `ncclXferReshardInit` / `ncclXferReshardFinalize` / `ncclXferReshardWithWindow` call on the host side — including calls on different `ncclComm_t` handles. Device-side concurrency (issuing successive reshards on separate CUDA streams from a single host thread) is supported. |
| **Static mesh-size caps**         | Compile-time array bounds in `src/reshard_limits.h` cap supported mesh sizes: `MAX_SOURCES = 16`, `MAX_TARGETS = 64`, `MAX_LOCAL_FOLLOWERS = 128` (RING); `MAX_DIRECT_SOURCES = 32`, `MAX_DIRECT_TARGETS = 64` (DIRECT). Larger meshes require recompiling. |
| **Single-shot public API**        | The public API exposes one `ncclXferReshardWithWindow` collective per call. Batched/concurrent behavior is built by callers with multiple descriptors/comms, not by a persistent public API. |

## References

- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [GPU-Initiated Networking Paper](https://arxiv.org/abs/2511.15076)
- [NCCL Xfer README](README.md)

## License

The NCCL contrib drop inherits the parent `nccl/nccl` license. Third-party
dependencies are listed in [`ThirdPartyNotices.txt`](ThirdPartyNotices.txt).
