# CuTeDSL device API examples

Runnable examples for `nccl.core.device.cute`, the CuTeDSL binding over the
NCCL device API. Together they cover its main public interfaces.

Install the dependencies for your CUDA major version, and use NCCL 2.32.6
or newer with one GPU per rank:

```sh
pip install -r requirements-cu13.txt    # or requirements-cu12.txt
```

`libnccl_device.bc` is resolved from `$NCCL_HOME/lib/` first, then the
installed `nvidia-nccl-cuXX` wheel.

The TMA copy in example 08 requires SM100+ GPUs and `libnccl_device.bc` built
for SM100+; set `IR_ARCH=sm_100` when building the IR library (see
[`bindings/ir/README.md`](../../../ir/README.md)).

## The examples

| File | Topic | Ranks | Needs |
|---|---|---|---|
| `00_basic.py` | Start here: 1 MiB `Gin.put` with a completion signal, launched through both `@cute.jit` argument forms | **exactly 2** | GIN network |
| `01_coop_and_teams.py` | Cooperative groups, devcomm fields, teams and rank translation | any | - |
| `02_lsa_allreduce.py` | Window address translation + LSA barriers, as a device-side all-reduce | >=2, one node | peer access |
| `03_multimem.py` | Multimem handles, multimem window addresses, multimem barrier arrive | >=2, one node | multicast |
| `04_gin_ops.py` | GIN transfers, signal/counter actions, descriptors, signal shadows and completion requests | **exactly 2** | GIN network |
| `05_barriers.py` | Barrier sessions, built-in teams and explicit handles, all fence levels | >=2 | GIN network |
| `06_resource_buffers.py` | The five `resource_buffer_*` address translations | >=2, one node | peer access, multicast |
| `07_compile_with_fake_args.py` | Compile with type-only arguments before creating NCCL resources, then invoke with real resources | **exactly 2** | GIN network |
| `08_reduce_copy.py` | Device-side `ReduceCopy`: `lsa_reduce_sum`, `multimem_reduce_sum`, `lsa_copy`, `multimem_copy`, `lsa_reduce_sum_copy`, `multimem_reduce_sum_copy`, `local_reduce_sum_copy`, and `lsa_copy_tma` | >=2, one node | peer access; multimem needs multicast; TMA needs SM100+ |

"Exactly 2" means the example hardcodes a rank 0 -> rank 1 transfer and
rejects any other count rather than idling the extra ranks.
"One node" means every rank must be in the same LSA team, since those
examples read peer memory directly. GIN examples want two nodes for a real
network path, but a single node works when NCCL can set up loopback GIN
connections.

Run under MPI, one rank per GPU -- substitute `srun --mpi=pmix -n <ranks>`
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
mpirun -n 8      python 08_reduce_copy.py      # one node, all GPUs
```

`-n 8` stands for however many GPUs the node has; the one-node examples
scale to any count >= 2.

Nothing faults on an unsuitable machine: a wrong rank count or topology is
an `ERROR` with a non-zero exit, and a missing capability is reported as
skipped.

## Check the extension compiler preview

CuTeDSL 4.8 provides an opt-in extension compiler. Compare the default runs
with these runs to check teams, fake arguments, and ReduceCopy on that path:

```sh
mpirun -n 2 env CUTE_DSL_USE_EXTENSION_COMPILER=1 python 01_coop_and_teams.py
mpirun -n 2 -N 1 env CUTE_DSL_USE_EXTENSION_COMPILER=1 python 07_compile_with_fake_args.py
mpirun -n 8 env CUTE_DSL_USE_EXTENSION_COMPILER=1 python 08_reduce_copy.py
```

`env` sets the option in every MPI rank before Python starts. Keep the same
hardware and topology requirements listed above; run 08 on one node and
adjust its rank count to the available GPUs. Capability skips do not verify
the skipped operations. The library does not change the compiler setting.

## Host object conversion

Use `nccl_cute.runtime.to_cute()` for explicit host-to-CuTeDSL conversion.
It supports `NCCLTeam`, `DevCommResource`, `RegisteredWindowHandle`,
`MultimemHandle`, `LsaBarrierHandle`, and `GinBarrierHandle`. Unsupported types
raise `TypeError`.

```python
dev_comm = nccl_cute.runtime.to_cute(dev_comm_resource)
window = nccl_cute.runtime.to_cute(window_resource)
team = nccl_cute.runtime.to_cute(nccl_comm.team_world)
```

Resource views retain the host object without allocating NCCL resources or
device storage, and can be reused across launches. Keep the owning resources
valid until GPU work completes. Direct constructors such as
`DevComm(dev_comm_resource)` and `Window(window_resource)` remain supported;
`to_cute()` delegates to the corresponding constructor for resource objects.

Host `NCCLTeam` objects require explicit conversion. `to_cute()` copies their
three fields into a resource-free `Team`. Construct Teams directly in Python
or traced code with `Team(n_ranks=..., rank=..., stride=...)`; the read-only
fields accept Python `int` or `cutlass.Int32` values. As ordinary JIT arguments,
all three fields become dynamic `Int32` values.

Inside a kernel, `dev_comm.team_world`, `dev_comm.team_lsa`, and
`dev_comm.team_rail` already return CuTeDSL Teams. Example 01 demonstrates
host conversion and direct construction; example 08 obtains the LSA team
from `DevComm`.

The five host resource types also support implicit JIT conversion. Use a
CuTeDSL parameter annotation with explicit conversion, or leave the parameter
unannotated (or host-typed) for implicit conversion:

```python
@cute.jit
def launch_explicit(handle: nccl_cute.MultimemHandle):
    kernel(handle).launch(grid=[1], block=[128])


@cute.jit
def launch_implicit(handle):
    kernel(handle).launch(grid=[1], block=[128])


launch_explicit(nccl_cute.runtime.to_cute(host_handle))
launch_implicit(host_handle)
```

Handles are opaque. Both conversion paths borrow existing NCCL resources;
keep their owning `DevCommResource` valid until GPU work completes. Example
06 passes host-readable resource metadata as separate scalar arguments.
Example 00 demonstrates both conversion paths for `DevComm` and `Window`.

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
compiled(
    nccl_cute.runtime.to_cute(dev_comm_resource),
    nccl_cute.runtime.to_cute(window_resource),
)
```

The fake objects describe only the JIT argument types. They carry no NCCL
resource or runtime pointer, are not `Constexpr` values, and must only be
passed to `cute.compile`. Replace them with real arguments before execution;
CuTeDSL does not guarantee a runtime error if a placeholder is passed to a
compiled callable. Use converted views as above, or the resources themselves
if the parameters are unannotated (see 00). The same module provides
compile-only factories for `MultimemHandle`, `LsaBarrierHandle`, and
`GinBarrierHandle` arguments.

`Team` owns no NCCL resource and needs no fake placeholder. Use
`nccl_cute.Team(n_ranks=2, rank=0, stride=1)` as a representative runtime
argument.

## GIN transfer endpoints

GIN transfers take `SymmetricPointer` endpoints and an element count:

```python
src = nccl_cute.SymmetricPointer(send_window, cutlass.Float32)
dst = nccl_cute.SymmetricPointer(recv_window, cutlass.Float32)
gin.put(
    coop,
    team,
    peer,
    dst,
    src,
    count,
    remote_action=nccl_cute.GinSignalInc(signal_id, strong=False),
)
```

`Gin.put_value` supports `Uint8`, `Uint16`, `Uint32`, and `Uint64` destinations.
Python integers must fit the destination range; CuTeDSL values must have the
same dtype as the destination.

Choose the access form for the operation:

| Expression | Use |
|---|---|
| `ptr + i` | Offset the symmetric endpoint by `i` elements, for example before passing it to GIN. |
| `ptr.local_pointer()[i]` | Read or write a scalar in the current rank's local buffer. |
| `ptr.tensor(layout)` | Create a local tensor view for layouts, multidimensional indexing, and slicing. |

Reuse the returned local pointer for multiple scalar accesses. When starting
from a Window, use `window.local_pointer(dtype)` or `window.tensor(dtype, layout)`.

The weak signal covers this put. A strong signal also orders earlier puts to
the same peer/context, as in example 04. `GinCounterInc` reports consumption
of the attached put's local source. Actions describe notifications; constructing
one does not submit it.

Example 04 uses blocking `flush` on EFA GDA, whose current backend does not
support completion requests; other backends demonstrate `flush_async` / `wait`.

## Output

The numbered examples name themselves on startup, warn when an API they
would use is unavailable, and print their own results:

```
===== 05_barriers.py =====
[rank 0] [SUCCESS] LSA, GIN and hybrid barriers completed
```

A machine that lacks a capability entirely warns and exits 0; a wrong rank
count or topology is an `ERROR` with a non-zero exit.

Use `libnccl.so` and `libnccl_device.bc` from the same build. If you `LD_PRELOAD`
one, point `NCCL_HOME` at that build as well.

## API coverage

| API | Example |
|---|---|
| `Coop.cta()`, `Coop.warp()`, `Coop.thread()`, `Coop.lanes()`, `Coop.warp_span()`; `thread_rank` / `num_threads` / `sync()` | 01, 04 |
| `runtime.to_cute(dev_comm_resource)` / `runtime.to_cute(window_resource)` | 00, 07 |
| `DevComm.rank` / `n_ranks` / `lsa_rank` / `lsa_size` | 01 |
| `runtime.to_cute(host_team)`, `Team(n_ranks=..., rank=..., stride=...)`; `Team.n_ranks` / `rank` / `stride` | 01 |
| `DevComm.team_world` / `team_lsa` / `team_rail`, `team_rank_to_world` / `team_rank_to_lsa` | 01, 08 |
| `DevComm.gin()` / `gin_all_contexts()` | 04, 05 |
| Built-in barrier resources: `LsaBarrierSession.lsa`, `GinBarrierSession.world` / `.rail`, `HybridBarrierSession.world` / `.lsa` / `.rail` | 05 |
| Default LSA multimem mapping and host-passed multimem handles | 03, 06, 08 |
| `DevComm.resource_buffer_*` (all five) | 06 |
| `Window.local_pointer` / `lsa_pointer` / `peer_pointer` | 02 |
| `Window.multimem_pointer` / `lsa_multimem_pointer` | 03, 08 |
| `SymmetricPointer`, element arithmetic, `local_pointer` / `lsa_pointer` | 00, 02, 04, 07, 08 |
| `Gin.put` | 00, 04, 07 |
| `Gin.put_value` / `get` / `signal` / `flush` | 04 |
| `GinSignalInc` / `GinSignalAdd` / `GinCounterInc` | 00, 04, 07 |
| `GinDescriptorSmem`, `Gin.flush_async` / `wait` | 04 |
| `Gin.wait_signal` | 00, 07 |
| `Gin.read_signal` / `reset_signal` / `signal_shadow_pointer` / `wait_signal_follow_shadow` | 04 |
| `Gin.read_counter` / `wait_counter` / `reset_counter` | 04 |
| `LsaBarrierSession` / `.lsa()`, `arrive` / `wait` / `sync` / `close` | 02, 03, 05, 06 |
| `GinBarrierSession` / `.world()` / `.rail()`, `GinAllContexts` | 05 |
| `HybridBarrierSession` / `.world()` / `.lsa()` / `.rail()` | 05 |
| `runtime.to_cute(host_handle)` for opaque `MultimemHandle` / `LsaBarrierHandle` / `GinBarrierHandle` views | 03, 05, 06 |
| `MemoryOrder` | 02, 04 |
| `ThreadScope` / `GinResourceSharingMode` | 04 |
| `GinFenceLevel` | 05 |
| `lsa_reduce_sum` / `multimem_reduce_sum` / `lsa_copy` / `multimem_copy` / `lsa_reduce_sum_copy` / `multimem_reduce_sum_copy` / `local_reduce_sum_copy` / `lsa_copy_tma` | 08 |

Not exercised: timeout overloads, `increase_signal_shadow` /
`wait_signal_meet_shadow`, non-default optimization flags and memory segments,
and the non-GPU resource-sharing modes.
