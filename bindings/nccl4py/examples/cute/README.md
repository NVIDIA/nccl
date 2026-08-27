# CuTeDSL device API examples

Runnable examples for `nccl.core.device.cute`, the CuTeDSL binding over the
NCCL device API. Together they exercise its whole public surface.

Install the dependencies for your CUDA major version, and use NCCL 2.30.7
or newer with one GPU per rank:

```sh
pip install -r requirements-cu13.txt    # or requirements-cu12.txt
```

`libnccl_device.bc` is resolved from `$NCCL_HOME/lib/` first, then the
installed `nvidia-nccl-cuXX` wheel.

## The examples

| File | Topic | Ranks | Needs |
|---|---|---|---|
| `00_basic.py` | Start here: 1 MiB `Gin.put` with a completion signal, launched through both `@cute.jit` argument forms | **exactly 2** | GIN network |
| `01_coop_and_teams.py` | Cooperative groups, devcomm fields, teams and rank translation | any | — |
| `02_lsa_allreduce.py` | Window address translation + LSA barriers, as a device-side all-reduce | ≥2, one node | peer access |
| `03_multimem.py` | Multimem handles, multimem window addresses, multimem barrier arrive | ≥2, one node | multicast |
| `04_gin_ops.py` | Every `Gin` operation: put / put_value / get / signal / counters / flush | **exactly 2** | GIN network, 2.31.1+ for `get` |
| `05_barriers.py` | All three barrier session types, both factory forms, all fence levels | ≥2 | GIN network, 2.31.0+ for the `Gin` factories |
| `06_resource_buffers.py` | The five `resource_buffer_*` address translations | ≥2, one node | peer access, multicast |
| `07_compile_with_fake_args.py` | Compile with type-only arguments before creating NCCL resources, then invoke with real resources | **exactly 2** | GIN network |

"Exactly 2" means the example hardcodes a rank 0 → rank 1 transfer and
rejects any other count rather than idling the extra ranks.
"One node" means every rank must be in the same LSA team, since those
examples read peer memory directly. GIN examples want two nodes for a real
network path, but a single node works when NCCL can set up loopback GIN
connections.

Run under MPI, one rank per GPU — substitute `srun --mpi=pmix -n <ranks>`
under Slurm. Recommended command per example:

```sh
mpirun -n 2 -N 1 python 00_basic.py              # two nodes, one rank each
mpirun -n 2      python 01_coop_and_teams.py    # any rank count
mpirun -n 8      python 02_lsa_allreduce.py     # one node, all GPUs
mpirun -n 8      python 03_multimem.py          # one node, all GPUs
mpirun -n 2 -N 1 python 04_gin_ops.py           # two nodes, one rank each
mpirun -n 2 -N 1 python 05_barriers.py          # two nodes, one rank each
mpirun -n 8      python 06_resource_buffers.py  # one node, all GPUs
mpirun -n 2 -N 1 python 07_compile_with_fake_args.py  # two nodes, one rank each
```

`-n 8` stands for however many GPUs the node has; the one-node examples
scale to any count ≥ 2.

Nothing faults on an unsuitable machine: a wrong rank count or topology is
an `ERROR` with a non-zero exit, and a missing capability is reported as
skipped.

## Compile before creating NCCL resources

`DevComm` and `Window` are dynamic arguments, so an application may compile a
`@cute.jit` function before it creates a communicator or registers memory:

```python
compiled = cute.compile(
    fn,
    nccl_cute.runtime.make_fake_dev_comm(),
    nccl_cute.runtime.make_fake_window(),
)

# Create the real NCCL resources later, then supply their runtime values.
compiled(nccl_cute.DevComm(dev_comm_resource), nccl_cute.Window(window_resource))
```

The fake objects describe only the JIT argument types. They carry no NCCL
resource or runtime pointer, are not `Constexpr` values, and must only be
passed to `cute.compile`. Passing one to the compiled callable contributes no
runtime argument, which shifts every argument after it and launches with
garbage rather than raising, so invoke with host-mode views as above — or with
the resources themselves, if the parameters are unannotated (see 00). The same
module provides compile-only factories for `MultimemHandle`,
`LsaBarrierHandle`, and `GinBarrierHandle` arguments.

## Output

The numbered examples name themselves on startup, warn when an API they
would use is unavailable, and print their own results:

```
===== 05_barriers.py =====
WARNING: NCCL 2.30.7 passes ncclGin_C by value; skipping world_gin,
rail_gin, gin_session, world_hybrid and hybrid_session
[rank 0] [SUCCESS] LSA barriers and the GIN_ALL_CONTEXTS barrier completed
```

A machine that lacks a capability entirely warns and exits 0; a wrong rank
count or topology is an `ERROR` with a non-zero exit.

Two device symbols changed after 2.30.7, in different releases, so the
affected sections check `nccl.get_version()` and skip rather than fail:
`ncclGinGet` only gained C linkage in **2.31.1**, and the barrier-session
entry points started taking `ncclGin_C` by pointer in **2.31.0**. The check
reads the loaded `libnccl.so` and assumes `libnccl_device.bc` came from the
same install — which is how the wheel and a source build ship them. If you
`LD_PRELOAD` one, point `NCCL_HOME` at the other's build.

## API coverage

| API | Example |
|---|---|
| `cta`, `warp`, `thread`, `lanes`, `warp_span`; `Coop.thread_rank` / `size` / `num_threads` / `sync` | 01, 04 |
| `DevComm.rank` / `n_ranks` / `lsa_rank` / `lsa_size` | 01 |
| `DevComm.team_world` / `team_lsa` / `team_rail`, `team_rank_to_world` / `team_rank_to_lsa` | 01 |
| `DevComm.gin` | 04, 05 |
| `DevComm.lsa_barrier` / `rail_gin_barrier` / `world_gin_barrier` / `hybrid_*` | 05 |
| `DevComm.lsa_multimem` | 03, 06 |
| `DevComm.resource_buffer_*` (all five) | 06 |
| `Window.tensor` | 00, 02, 04, 07 |
| `Window.local_pointer` / `lsa_pointer` / `peer_pointer` | 02 |
| `Window.multimem_pointer` / `lsa_multimem_pointer` | 03 |
| `Gin.put` | 00, 04, 07 |
| `Gin.put_value` / `get` / `signal` / `flush` | 04 |
| `Gin.wait_signal` | 00, 04, 07 |
| `Gin.read_signal` / `reset_signal` / `signal_shadow_pointer` | 04 |
| `Gin.read_counter` / `wait_counter` / `reset_counter` | 04 |
| `lsa_session` / `lsa_default`, `arrive` / `wait` / `sync` | 02, 03, 05, 06 |
| `gin_session` / `world_gin` / `rail_gin`, `GIN_ALL_CONTEXTS` | 05 |
| `hybrid_session` / `world_hybrid` | 05 |
| `MultimemHandle` / `LsaBarrierHandle` / `GinBarrierHandle` and their fields | 03, 05, 06 |
| `MemoryOrder` | 02–06 |
| `ThreadScope` / `GinResourceSharingMode` | 04 |
| `GinFenceLevel` | 05 |
| `GinBackendMask` | 00, 04, 05, 07 |

Not exercised: `Gin.value()` and the raw `.ptr` accessors (binding
plumbing); the `is_descriptor` / `descriptor_ptr` arguments, which need an
`ncclGinDescriptorSmem` this layer does not wrap; and pass-through knobs
left at their defaults — `opt_flags`, `signal_op=1` (Add), the non-GPU
resource-sharing modes, and the individual `GinBackendMask` bits.
