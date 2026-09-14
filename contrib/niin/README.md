# NIIN: NVSHMEM Implemented In NCCL

NIIN implements the NVSHMEM API on top of NCCL's communication infrastructure
(LSA peer pointers, GIN network transport, barriers). The device API is
header-only, while the host API keeps process-wide state and links against NCCL.
NIIN lets you write kernels using `nvshmem_*` names while using NCCL underneath
-- no NVSHMEM build or installation required.

> **Status:** NIIN is experimental contrib software maintained separately from
> NCCL core. It is not a complete NVSHMEM replacement and does not support the
> full NVSHMEM API surface area; see [Limitations](#limitations) for the current
> compatibility boundaries. NIIN has been validated for correctness on the
> supported paths but has not yet been tuned for performance.

NIIN is an experimental version of NVSHMEM, not a product. It is intended for
applications that can use the supported subset of NVSHMEM host and device APIs
described below.

## Maintainers

| GitHub | Areas |
|--------|------|
| @benjaming | All |

## Basic Usage

```cpp
#include "nvshmem.h"    // drop-in replacement
#include "nvshmemx.h"   // extended API (threadgroup, stream, teams)

__global__ void my_kernel(int *buf) {
    int pe   = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    int dst  = (pe + 1) % npes;

    if (threadIdx.x == 0) {
        nvshmem_int_p(buf, 42, dst);             // scalar put
        int *peer = (int *)nvshmem_ptr(buf, dst); // direct pointer for LSA peers
        if (peer != nullptr) peer[1] = pe;
        nvshmem_fence();
        nvshmem_quiet();
    }
}

int main() {
    nvshmem_init();                       // sets up NCCL comm, heap, device context
    int *buf = (int *)nvshmem_malloc(4096);
    my_kernel<<<1, 32>>>(buf);
    cudaDeviceSynchronize();
    nvshmem_barrier_all();
    nvshmem_free(buf);
    nvshmem_finalize();
}
```

## Installation

### Dependencies and Build Configuration

- NCCL 2.29+ with device API support for windows, GIN, and barriers
- CUDA 12.2+ with `--expt-relaxed-constexpr` and C++17
- A shared filesystem for multi-PE `nvshmem_init()` (the default is
  `$HOME/.niin`; set `NIIN_BOOTSTRAP_DIR` when home is not shared). The
  launcher must provide rank and size metadata plus a job ID, or set
  `NIIN_BOOTSTRAP_ID` explicitly.
- MPI for multi-PE applications that call `nvshmemx_init_attr()` with
  `NVSHMEMX_INIT_WITH_MPI_COMM`; compile those applications with `-DNIIN_HAS_MPI`.
  UID-based initialization can use the application's own bootstrap mechanism.

NIIN's device-side API is header-only. The host API is stateful and is exposed
through `libnvshmem_host.so`. Build NCCL first, then prepare NIIN so its
`include/` and `lib/` directories contain everything an application needs.

Build NIIN from the NCCL tree:

```bash
make -C <nccl>/contrib/niin NCCL_HOME=<nccl>
export NIIN_HOME=<nccl>/contrib/niin
```

This creates `$NIIN_HOME/lib/libnvshmem_host.so` and wires the generated backend
headers into `$NIIN_HOME/include`.

Compile an application with `NIIN_HOME` pointing to NIIN:

```bash
NIIN_HOME=<nccl>/contrib/niin

nvcc my_app.cu -o my_app \
    -I ${NIIN_HOME}/include \
    -L ${NIIN_HOME}/lib -lnvshmem_host \
    --expt-relaxed-constexpr -std=c++17 -arch=sm_89
```

For device code spanning multiple CUDA translation units, compile those units
with `-rdc=true` and perform one device link. This lets NIIN share its device
context across the consumer's translation units.

Set the runtime library path or link with an rpath:

```bash
export LD_LIBRARY_PATH=${NIIN_HOME}/lib:${LD_LIBRARY_PATH}
```

NIIN does not build or require a separate device archive. Device-side NVSHMEM
functions are compiled from NIIN's headers into the application's CUDA
translation units. The GIN transport routines -- `niin_gin_put`/`get`/`put_value`,
the remote signal delivery, the GIN flush, and both put-with-signal remote forms
-- are marked `__noinline__`, so they compile to one device function per program
instead of expanding a network descriptor build into every caller: kernels that
only ever reach NVLink peers do not pay for that code or its registers. The
NVLink and self paths stay force-inlined, being a peer-pointer copy plus a store
where a call would cost more than it saves. During
`nvshmem_init()`/`nvshmemx_init_attr()`, NIIN creates the NCCL device
communicator and symmetric heap state, then publishes the device context pointer
to NIIN's `__device__` global with `cudaMemcpyToSymbol`. Multi-translation-unit
CUDA applications must use relocatable device code (`-rdc=true` or CMake CUDA
separable compilation); a single-TU non-RDC program is supported as well.

## Source Layout

```
contrib/niin/include/
  nvshmem.h              # Drop-in replacement master include
  nvshmemx.h             # Extended API (threadgroup, stream, module stubs, buffer register)
  niin/
    config.h             # NIIN_NOT_IMPLEMENTED error policy
    types.h              # NVSHMEM constants, X-macro type generation
    context.h            # niinContext struct, __device__ global, helpers
    query.h              # nvshmem_my_pe, nvshmem_n_pes, nvshmem_ptr (__host__ __device__)
    rma.h                # All put/get variants (typed, sized, mem, nbi, strided)
    signaling.h          # put_signal, signal_fetch, signal_wait_until
    sync.h               # fence, quiet, wait_until, test (+ all/any/some + vector)
    atomics.h            # Atomic fetch_add, CAS, swap, and/or/xor
    collectives.h        # barrier_all, sync_all (rest = NOT_IMPLEMENTED)
    threadgroup.h        # nvshmemx_*_put_warp/block, get_warp/block, etc.
    stream.h             # nvshmemx_*_on_stream (host-launched kernel wrappers)
    teams.h              # team_split_strided, team_split_2d, team_translate_pe
    host.h               # Low-level niinInit/niinCommit/niinFinalize
    nvshmem_host.h       # NVSHMEM-compatible host API (init, malloc, query, barrier)
```

## Design

### Initialization

`nvshmem_init()` performs the following:

1. Detects rank and world size from supported launcher-provided metadata. Falls back to single-PE mode if none is found.
2. Selects the local GPU from launcher-provided local rank metadata or `rank % nDevices`.
3. For multi-PE jobs, has PE 0 publish an NCCL unique ID in a small record in
   `NIIN_BOOTSTRAP_DIR` (default `$HOME/.niin`) and waits for it on the other
   PEs. The record name is derived from the Slurm, PMI, or PMIx job ID; set
   `NIIN_BOOTSTRAP_ID` to override it. This needs no MPI initialization.
4. Creates an NCCL communicator with `ncclCommInitRankConfig`.
5. Allocates the symmetric heap via `ncclMemAlloc` (default 256 MB, configurable via `NVSHMEM_SYMMETRIC_SIZE`).
6. Registers the heap as an NCCL window (`ncclCommWindowRegister`) and creates a device communicator (`ncclDevCommCreate`) with GIN resources.
7. Sets NIIN's `__device__` global context pointer via `cudaMemcpyToSymbol` so all subsequent kernel launches have access to the NIIN context automatically.

For multi-PE with MPI, use:

```cpp
MPI_Init(&argc, &argv);
nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
MPI_Comm comm = MPI_COMM_WORLD;
attr.mpi_comm = &comm;
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
```

Compile with `-DNIIN_HAS_MPI` and link MPI. The NIIN perftest Makefile shows
the complete pattern:

```bash
make -C ${NIIN_HOME}/perftest NCCL_HOME=${NCCL_HOME} MPI_HOME=${MPI_HOME}
```

`shmem_atomic_bw` links NIIN's GPUNetIO atomic provider, which the NIIN build
produces by default from NCCL's vendored GPUNetIO sources, so the commands above
build it too. Keep the provider's GPU architecture and the perftest `ARCH` in
step -- the `89-real`/`sm_89` pair below is for L40S:

```bash
make -C ${NIIN_HOME} NCCL_HOME=${NCCL_HOME} GPUNETIO_CUDA_ARCHITECTURES=89-real
make -C ${NIIN_HOME}/perftest NCCL_HOME=${NCCL_HOME} MPI_HOME=${MPI_HOME} ARCH=sm_89
```

Pass `GPUNETIO_HOME=` to build against a different unmodified GPUNetIO tree, and
`IBVERBS_LIBRARY=` when the verbs library is not on the default search path.

Run `build/shmem_atomic_bw` with two PEs on different LSA domains (normally
one GPU rank on each of two nodes), for example with
`-a fetch_add`. It accepts the upstream operation names `inc`, `fetch_inc`,
`set`, `add`, `fetch_add`, `and`, `fetch_and`, `or`, `fetch_or`, `xor`,
`fetch_xor`, `swap`, and `compare_swap`. The current provider serializes each
destination PE's response slot, so this reports serialized direct-AMO payload
throughput rather than NIC wire bandwidth; its table includes payload GB/s and
millions of atomic operations per second. Its defaults are a 8 B--64 KiB sweep
with 10 warmup and 10 timed iterations; `--blocks` and `--threads` must remain
one. Make the selected NCCL, CUDA, MPI, and ibverbs libraries available at
runtime (or preserve the matching `LD_LIBRARY_PATH` used for the build).

For UID-based bootstrap, generate a NIIN/NCCL unique ID on one PE, distribute it
with the application's bootstrap mechanism, and pass the rank metadata through
`nvshmemx_set_attr_uniqueid_args()`:

```cpp
nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
nvshmemx_uniqueid_t id = NVSHMEMX_UNIQUEID_INITIALIZER;

if (rank == 0) nvshmemx_get_uniqueid(&id);
MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
nvshmemx_set_attr_uniqueid_args(rank, nranks, &id, &attr);
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
```

### RMA Routing (LSA vs GIN)

Every RMA operation checks whether the target PE is an LSA (Load-Store Accessible) peer as reported by the NCCL device communicator:

- **LSA path**: Direct load/store via `ncclGetPeerPointer()`.
- **Network path (GIN)**: `ncclGin::put()` / `ncclGin::putValue()` for puts
  and `ncclGin::get()` for block gets into the symmetric heap.

Scalar `nvshmem_*_g()` over non-LSA peers is out of scope for now because it
returns a value directly rather than writing into a symmetric destination
buffer. Use block get APIs for network GET paths.

### Multi-QP (GIN contexts)

A NCCL GIN context owns one QP per peer, so a single context serializes all of
a PE's network traffic onto one QP. NIIN requests several contexts at init
(`NIIN_NUM_QPS`, default 4) and round-robins operations across them: every
network put, get, or signal takes the next context from a rotating cursor, the
way NVSHMEM's IBGDA and IBRC transports pick a QP. Because NCCL derives a
context's `connectionId` from its index, consecutive contexts also land on
different NICs on multi-rail nodes.

NCCL rounds the requested count up to a multiple of the connection count, so the
effective number of QPs can exceed `NIIN_NUM_QPS`. NIIN reads the real value back
from `ncclDevComm::ginContextCount` rather than assuming its request was honored
verbatim.

Selection is per operation rather than per CTA, so any grid shape -- a single
CTA included -- spreads across every QP. An operation that has to stay ordered
internally reserves one context and reuses it: a `put_signal` issues its
payload, the flush that orders it, and its signal on the same QP, and the fused
GIN path drives the whole descriptor from one context.

Two QPs are not ordered against each other at the receiving NIC, so ordering an
earlier put ahead of a later one means completing it:

- **`nvshmem_fence()` and `nvshmem_quiet()`** drain every context. Both are
  O(contexts) per call, which is what per-operation rotation costs.
- **Barriers** run on context 0 on every PE: GIN barrier signals are
  per-context state, so a barrier only converges if all PEs drive it from the
  same index. The barrier drains the rotating contexts before synchronizing.
- **`nvshmemx_quiet_on_stream()` and the host `nvshmem_quiet()`** drain every
  context as well. Their kernel never issued the puts it is being asked to
  complete, and a put can still be in flight on a context after the kernel that
  issued it has retired, so waiting on the kernel alone does not complete it.

#### Tuning `NIIN_NUM_QPS`

| Situation | Guidance |
|---|---|
| Large single transfers (>= 1 MB) already at line rate | Extra QPs do not help; the link, not the QP, is the limit |
| Many mid-sized transfers (4 KB - 512 KB) | The sweet spot. This is where per-QP issue rate dominates |
| Fence- or quiet-heavy code | Every fence and quiet drains all contexts, so fewer QPs can win |
| Debugging a suspected ordering bug | Set `NIIN_NUM_QPS=1` to collapse onto one QP and compare |

Raising the count is not free: each context is a real QP per peer, so memory and
connection-setup cost grow with it, and the drains above are O(contexts x peers).

Note that `NIIN_NUM_QPS=1` does not necessarily produce one context. NCCL rounds
the request up to a multiple of the GIN connection count, so on a 2-connection
node the smallest achievable count is 2.
### Memory Allocation

`nvshmem_malloc()` returns pointers into a single pre-allocated symmetric heap registered as an NCCL window. A free-list allocator manages sub-allocations (aligned to 256 bytes) with first-fit search, block splitting, and coalescing on `nvshmem_free()`. Freed memory is immediately reusable — freeing all allocations coalesces the heap back into a single block.

The allocator supports up to 256 concurrent allocation records. The heap size
defaults to 256 MB and can be configured via the `NVSHMEM_SYMMETRIC_SIZE`
environment variable (supports K/M/G suffixes).

### Memory Model

NIIN preserves the normal
[NVSHMEM symmetric heap memory model](https://docs.nvidia.com/nvshmem/api/gen/mem-model.html)
for memory returned by `nvshmem_malloc()`, `nvshmem_calloc()`, and
`nvshmem_align()`: symmetric allocations are remotely accessible by the same
addressing pattern on every PE. Device RMA operations should use pointers into
this symmetric heap.

### Signaling

`nvshmem_*_put_signal` uses peer-pointer atomics for signals on native-atomic
LSA peers. For LSA peers without native atomics, and optionally for any
transport, NIIN can split the operation into `put -> threadfence_system ->
signal` by setting `NIIN_PUT_SIGNAL_MODE=separate` (aliases: `split`,
`fence_signal`). The default `auto` mode keeps fused signaling where safe and
uses the split path where peer pointer atomics are not reliable.

## API Coverage

### Host-Side API

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmem_init()` | Full | Auto-detects launcher metadata and self-bootstraps multi-PE jobs through a shared directory; single-PE fallback |
| `nvshmemx_init_attr(MPI_COMM)` | Full | Requires `-DNIIN_HAS_MPI` |
| `nvshmemx_init_attr(UNIQUEID)` | Full | Uses `nvshmemx_get_uniqueid()` and caller-distributed UID |
| `nvshmem_finalize()` | Full | |
| `nvshmem_malloc(size)` | Full | Free-list allocator from symmetric heap |
| `nvshmem_calloc(count, size)` | Full | malloc + cudaMemset |
| `nvshmem_align(alignment, size)` | Full | |
| `nvshmem_free(ptr)` | Full | Returns memory to free list with coalescing |
| `nvshmem_my_pe()` | Full | Host and device |
| `nvshmem_n_pes()` | Full | Host and device |
| `nvshmem_team_my_pe(team)` | Full | WORLD, SHARED, NODE, and constrained host custom teams |
| `nvshmem_team_n_pes(team)` | Full | WORLD, SHARED, NODE, and constrained host custom teams |
| `nvshmem_ptr(ptr, pe)` | Full | Host and device; nullptr for non-LSA peers |
| `nvshmem_barrier_all()` | Full | Host: ncclAllReduce barrier; Device: ncclBarrierSession |
| `nvshmem_sync_all()` | Full | Alias for barrier_all |
| `nvshmem_fence()` | Full | Host: cudaDeviceSynchronize; Device: drains every GIN context + `__threadfence_system()`, since operations rotate across QPs |
| `nvshmem_quiet()` | Full | Host: `cudaDeviceSynchronize` then an all-contexts GIN drain (a put can outlive the kernel that issued it); Device: drains every GIN context + threadfence |
| `nvshmem_info_get_name()` | Full | Returns "NIIN (NVSHMEM Implemented In NCCL)" |
| `nvshmem_info_get_version()` | Full | Reports 3.0 |
| `nvshmemx_barrier_all_on_stream()` | WORLD only | Stream-ordered NCCL world barrier |
| `nvshmemx_sync_all_on_stream()` | WORLD only | Stream-ordered NCCL world sync |
| `nvshmemx_quiet_on_stream()` | Full | Stream-ordered device quiet kernel: GIN flush + `__threadfence_system()` |
| `nvshmemx_collective_launch()` | Stub | Redirects to cudaLaunchKernel |
| `nvshmemx_cumodule_init/finalize()` | Stub | No-op (compatibility) |
| `nvshmem_global_exit()` | Full | finalize + exit |

### Host Stream-Ordered APIs (`nvshmemx.h`)

The supported `nvshmemx_*_on_stream` variants below match NVSHMEM's host
on-stream contract: callers provide a `cudaStream_t`, but do not choose grid or
CTA geometry. NIIN owns the internal launch geometry; block put/get and
put-signal fallbacks use a size-based number of eight-warp CTAs for self and
LSA transfers, capped by `NVSHMEM_MAX_CTAS`, so the LSA path can use the
cooperative block RMA implementation. Non-LSA GIN transfers remain single-CTA,
so a single on-stream transfer issues from one CTA; its operations still rotate
across QPs as described in
[Multi-QP (GIN contexts)](#multi-qp-gin-contexts). Scalar, strided, signal, and
wait wrappers use compact control kernels.

NIIN currently owns one host NCCL communicator, for `NVSHMEM_TEAM_WORLD`.
The host collectives below reject non-WORLD teams rather than issue a partial
collective on that communicator; per-team NCCL communicators are required to
extend this support safely.

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmemx_<TYPE>_p_on_stream()` | Full | 24 types |
| `nvshmemx_<TYPE>_g_on_stream()` | Full | 24 types (synchronous — syncs stream to return value) |
| `nvshmemx_<TYPE>_put_on_stream()` | Full | 24 typed + 5 sized + putmem |
| `nvshmemx_<TYPE>_get_on_stream()` | Full | 24 typed + 5 sized + getmem |
| `nvshmemx_<TYPE>_put/get_nbi_on_stream()` | Full | Same as blocking |
| `nvshmemx_<TYPE>_iput/iget_on_stream()` | Full | 24 typed strided |
| `nvshmemx_<TYPE>_put_signal_on_stream()` | Full | 24 typed + 5 sized + putmem |
| `nvshmemx_<TYPE>_put_signal_nbi_on_stream()` | Full | Same as blocking |
| `nvshmemx_signal_op_on_stream()` | Full | |
| `nvshmemx_signal_wait_until_on_stream()` | Full | Synchronous |
| `nvshmemx_<TYPE>_wait_until_on_stream()` | Full | 13 wait types |
| `nvshmemx_<TYPE>_wait_until_all_on_stream()` | Full | 13 wait types |
| `nvshmemx_<TYPE>_wait_until_all_vector_on_stream()` | Full | 13 wait types |
| `nvshmemx_int32/int64_<sum/min/max>_reduce_on_stream()` | WORLD only | NCCL AllReduce |
| `nvshmemx_alltoallmem_on_stream()` | WORLD only | NCCL AlltoAll, byte count |

### Device-Side Query

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmem_my_pe()` | Full | `comm->rank` |
| `nvshmem_n_pes()` | Full | `comm->nRanks` |
| `nvshmem_ptr(ptr, pe)` | Full | `ncclGetPeerPointer` for LSA; nullptr for network |
| `nvshmem_team_my_pe(team)` | Full | WORLD, SHARED, NODE (device); + custom teams (host) |
| `nvshmem_team_n_pes(team)` | Full | WORLD, SHARED, NODE (device); + custom teams (host) |

### Device-Side RMA

| NVSHMEM API | LSA | Network | Notes |
|---|---|---|---|
| `nvshmem_<TYPE>_p(dest, val, pe)` | Full | Full | Scalar put; GIN `putValue` for network |
| `nvshmem_<TYPE>_put(dest, src, n, pe)` | Full | Full | Block put; GIN `put` for network |
| `nvshmem_putmem(dest, src, bytes, pe)` | Full | Full | Untyped block put |
| `nvshmem_put8/16/32/64/128()` | Full | Full | Sized block puts |
| `nvshmem_<TYPE>_put_nbi()` | Full | Full | Same as blocking (already async) |
| `nvshmem_putmem_nbi()` | Full | Full | |
| `nvshmem_<TYPE>_iput()` | Full | **Not impl** | Strided put; LSA only |
| `nvshmem_<TYPE>_g(src, pe)` | Full | **Not impl** | Scalar get returns by value; network path has no symmetric destination buffer |
| `nvshmem_<TYPE>_get(dest, src, n, pe)` | Full | Full | Block get; GIN `get` for network |
| `nvshmem_getmem()` | Full | Full | Untyped block get |
| `nvshmem_get8/16/32/64/128()` | Full | Full | Sized block gets |
| `nvshmem_<TYPE>_get_nbi()` | Full | Full | Same as blocking |
| `nvshmem_<TYPE>_iget()` | Full | **Not impl** | Strided get; LSA only |

All typed variants are generated via X-macros for 24 standard RMA types: `float`, `double`, `char`, `schar`, `short`, `int`, `long`, `longlong`, `uchar`, `ushort`, `uint`, `ulong`, `ulonglong`, `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `size`, `ptrdiff`.

### Device-Side Signaling

| NVSHMEM API | LSA | Network | Notes |
|---|---|---|---|
| `nvshmem_<TYPE>_put_signal(SIGNAL_ADD)` | Full | Full | Fused GIN `VASignalAdd` by default; split path available via `NIIN_PUT_SIGNAL_MODE=separate` |
| `nvshmem_<TYPE>_put_signal(SIGNAL_SET)` | Full | Full | Implemented via split `put -> flush/fence -> signal` path |
| `nvshmem_putmem_signal(SIGNAL_ADD)` | Full | Full | |
| `nvshmem_<TYPE>_put_signal_nbi()` | Full | Full | Same as blocking variant |
| `nvshmem_signal_fetch(sig_addr)` | Full | N/A | Local atomic read |
| `nvshmem_signal_wait_until(sig, cmp, val)` | Full | N/A | Local spin-wait |

### Device-Side Synchronization

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmem_fence()` | Full | All-contexts GIN drain + `__threadfence_system()`; ordering across rotating QPs means completing the earlier operation. Not a collective barrier |
| `nvshmem_quiet()` | Full | All-contexts GIN drain + `__threadfence_system()` |
| `nvshmem_barrier_all()` | Full | All-contexts `ncclGinBarrierSession` (world team) with acquire/release ordering and a `Put\|Get` fence, so barrier_all implies quiet as NVSHMEM requires |
| `nvshmem_sync_all()` | Full | Collective synchronization; currently uses the same NCCL barrier path as barrier_all |
| `nvshmem_barrier(TEAM_WORLD)` | Full | Redirects to barrier_all |
| `nvshmem_barrier(other team)` | Not impl | **Team mapping gap** |
| `nvshmem_<TYPE>_wait_until()` | Full | Spin-wait on local memory |
| `nvshmem_<TYPE>_test()` | Full | Single-shot check |
| `nvshmem_<TYPE>_wait_until_all/any/some()` | Full | Loop over wait_until/test |
| `nvshmem_<TYPE>_test_all/any/some()` | Full | Loop over test |
| `nvshmem_<TYPE>_wait_until_all/any/some_vector()` | Full | Per-element comparison arrays |
| `nvshmem_<TYPE>_test_all/any/some_vector()` | Full | Per-element comparison arrays |

Wait/test operations are generated for 13 types: `short`, `int`, `long`, `longlong`, `ushort`, `uint`, `ulong`, `ulonglong`, `int32`, `int64`, `uint32`, `uint64`, `size`.

### Device-Side Atomics

| NVSHMEM API | LSA | Network | Notes |
|---|---|---|---|
| `nvshmem_<TYPE>_atomic_fetch_add()` | Full | Optional native | Requires the NIIN GPUNetIO provider to be bound at init; not a GIN op |
| `nvshmem_<TYPE>_atomic_add()` | Full | Optional native | Same provider |
| `nvshmem_<TYPE>_atomic_compare_swap()` | Full | Optional native | Same provider |
| `nvshmem_<TYPE>_atomic_swap()` | Full | Optional native | Same provider |
| `nvshmem_<TYPE>_atomic_fetch()` | Full | Optional native | Same provider |
| `nvshmem_<TYPE>_atomic_set()` | Full | Optional native | Same provider |
| `nvshmem_<TYPE>_atomic_inc/fetch_inc()` | Full | Optional native | Same provider |
| `nvshmem_<TYPE>_atomic_fetch_and/or/xor()` | Full | Optional native | Native mode uses masked 4/8-byte WQEs; bitwise types only |
| `nvshmem_<TYPE>_atomic_and/or/xor()` | Full | Optional native | Same provider |
| `nvshmem_float/double_atomic_{swap,fetch,set}()` | Full | Experimental only | Remote support is an unsupported SRQ-proxy prototype |
| `nvshmemx_{half,float,double}_atomic_{fetch_add,add}()` | Full | Experimental only | Remote support is an unsupported SRQ-proxy prototype |

Standard AMO types (12): `int`, `long`, `longlong`, `uint`, `ulong`, `ulonglong`, `int32`, `int64`, `uint32`, `uint64`, `size`, `ptrdiff`.
Bitwise AMO types (7): `uint`, `ulong`, `ulonglong`, `int32`, `int64`, `uint32`, `uint64`.

GIN does not expose atomic operations, so the provider in
[`gpunetio/`](gpunetio/README.md) creates NIIN-owned atomic-only QPs and a
separate remote-atomic registration of the symmetric heap. On systems without
NIC atomic support, and for operations the NIC cannot express such as
floating-point atomics, a GDRCopy-based implementation is available.

### Device-Side Collectives

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmem_barrier_all()` | Full | `ncclBarrierSession` |
| `nvshmem_sync_all()` | Full | Same as barrier_all |
| `nvshmem_<TYPE>_sum/max/min/prod_reduce()` | Not impl | **No device-side collectives in NCCL** |
| `nvshmem_broadcastmem()` | Not impl | **No device-side collectives in NCCL** |
| `nvshmem_alltoallmem()` | Not impl | **No device-side collectives in NCCL** |
| `nvshmem_fcollectmem()` | Not impl | **No device-side collectives in NCCL** |

### Warp/Block Threadgroup RMA (`nvshmemx.h`)

All `nvshmemx_*_warp` and `nvshmemx_*_block` variants are implemented with cooperative vectorized memcpy for LSA peer bandwidth:

- **LSA path**: All threads cooperatively copy data using 4x-unrolled `int4` (16-byte) loads/stores. A 3-phase approach handles arbitrary alignment: head bytes until 16-byte aligned, vectorized body, tail bytes. With a 512-thread block this produces 32 KB per iteration, all coalesced.
- **Network (GIN) path**: Thread 0 issues the GIN put/get (single-thread API).
- **Strided iput/iget**: Thread 0 only (non-contiguous access doesn't benefit from cooperative copy).

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmemx_<TYPE>_put_warp/block()` | Full | 24 typed + 5 sized + putmem |
| `nvshmemx_<TYPE>_get_warp/block()` | Full | 24 typed + 5 sized + getmem |
| `nvshmemx_<TYPE>_put_nbi_warp/block()` | Full | Same as blocking |
| `nvshmemx_<TYPE>_get_nbi_warp/block()` | Full | Same as blocking |
| `nvshmemx_<TYPE>_iput_warp/block()` | Full | 24 typed strided |
| `nvshmemx_<TYPE>_iget_warp/block()` | Full | 24 typed strided |
| `nvshmemx_<TYPE>_put_signal_warp/block()` | Full | 24 typed + putmem |
| `nvshmemx_<TYPE>_put_signal_nbi_warp/block()` | Full | 24 typed + putmem |
| `nvshmemx_sync_all_block()` | Full | One matching CTA collective per PE at a time |
| `nvshmemx_barrier_all_block()` | Full | CTA release/acquire ordering; one matching CTA collective per PE at a time |

### Team Management

#### Predefined Teams

All 6 NVSHMEM predefined teams are supported on both host and device:

| ID | Team | NCCL Mapping | Device Impl |
|---|---|---|---|
| 0 | `NVSHMEM_TEAM_WORLD` | All PEs | `comm.rank` / `comm.nRanks` |
| 1 | `NVSHMEM_TEAM_SHARED` | Same node (LSA) | `comm.lsaRank` / `comm.lsaSize` |
| 2 | `NVSHMEMX_TEAM_NODE` | Alias for SHARED | Same as SHARED |
| 3 | `NVSHMEMX_TEAM_SAME_MYPE_NODE` | Same local rank across nodes (rail) | `rank/lsaSize` / `ceil(nRanks/lsaSize)` |
| 4 | `NVSHMEMI_TEAM_SAME_GPU` | PEs sharing same GPU | Always rank=0, size=1 (NCCL is 1-PE-per-GPU) |
| 5 | `NVSHMEMI_TEAM_GPU_LEADERS` | One PE per GPU | Same as WORLD (NCCL is 1-PE-per-GPU) |

#### Custom Teams and Operations

NIIN supports predefined teams and a constrained host-side custom-team model for
regular strided and 2D split patterns. It does not support arbitrary NVSHMEM team
creation with user-selected PE membership, arbitrary layouts, or team-specific
context configuration. User-created teams are host-side handles for query,
translation, and destroy operations; device queries support only predefined
teams.

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmem_team_split_strided()` | Partial | Stores regular strided teams as {start, stride, size}; no arbitrary PE membership; up to 58 user teams |
| `nvshmem_team_split_2d()` | Partial | Builds X-axis and Y-axis teams from regular strided row/column splits; no arbitrary layouts |
| `nvshmem_team_destroy()` | Full | |
| `nvshmem_team_translate_pe()` | Full | World rank as intermediate for predefined and NIIN-created teams |
| `nvshmem_team_get_config()` | Stub | Returns num_contexts=0 |
| `nvshmem_team_my_pe(custom)` | Partial | Host: team table lookup for NIIN-created teams; Device: predefined teams only |
| `nvshmem_team_n_pes(custom)` | Partial | Host: team table lookup for NIIN-created teams; Device: predefined teams only |

### Buffer Registration

| NVSHMEM API | NIIN Status | Notes |
|---|---|---|
| `nvshmemx_buffer_register()` | Stub | No-op (heap model handles common case) |
| `nvshmemx_buffer_unregister()` | Stub | No-op |
| `nvshmemx_buffer_register_symmetric()` | Stub | Returns input pointer |
| `nvshmemx_buffer_unregister_symmetric()` | Stub | No-op |

## Limitations

NIIN is experimental NVSHMEM implemented over NCCL, not the NVSHMEM runtime. It
does not support the full NVSHMEM API surface area. The most important
operational difference is that NCCL owns transport selection, debugging, memory
registration, and network behavior.

### NOT_IMPLEMENTED Policy

Operations outside NIIN's supported API subset invoke the NOT_IMPLEMENTED
handler. The behavior is controlled at compile time by defining
`NIIN_ON_NOT_IMPLEMENTED` before including `nvshmem.h`:

| Value       | Behavior                                    |
|-------------|---------------------------------------------|
| `NIIN_TRAP` | (default) `printf` + `__trap()` — hard stop |
| `NIIN_NOOP` | Silent no-op; return 0 for value-returning  |
| `NIIN_WARN` | `printf` warning + no-op                    |

### Runtime and configuration limitations

| Limitation | Impact | Notes |
|---|---|---|
| Unsupported NVSHMEM-specific environment variables are not interpreted | NVSHMEM transport, bootstrap, affinity, debug, and tuning variables do not configure NIIN | NIIN honors only the NVSHMEM-named variables listed in [Environment and Runtime Configuration](#environment-and-runtime-configuration). Use NCCL environment variables such as `NCCL_DEBUG`, `NCCL_IB_HCA`, `NCCL_IB_DISABLE`, `NCCL_P2P_DISABLE`, and `NCCL_SHM_DISABLE` for NCCL runtime behavior. |
| `nvshmem_init()` bootstrap needs shared storage | Normal multi-PE initialization cannot proceed if `$HOME/.niin` is not shared | Set `NIIN_BOOTSTRAP_DIR` to a job-writable shared directory. The launcher must set rank/size and a job ID, or set `NIIN_BOOTSTRAP_ID` explicitly. MPI and caller-managed UNIQUEID initialization remain available alternatives. |
| One NCCL PE per GPU | Multi-process-per-GPU NVSHMEM use cases are not represented | `NVSHMEMI_TEAM_SAME_GPU` is always a single-PE team. |
| No arbitrary NVSHMEM team creation | Applications cannot create teams with arbitrary PE membership, layouts, or team-specific contexts | NIIN represents predefined teams plus constrained host-side strided/2D split teams only. User-created teams are not available to device APIs. |
| NCCL memory/window model | Symmetric allocations come from one NCCL-registered heap | Set `NVSHMEM_SYMMETRIC_SIZE` before initialization when a larger heap is required. |
| NCCL device transport constraints | API support depends on whether the target PE is LSA-accessible or reachable through GIN | Block gets use GIN; scalar network gets remain unimplemented. Network integral AMOs require NIIN's optional native GPUNetIO provider; network floating AMOs remain experimental because GIN has no AMO API. |
| Performance is not yet tuned | Supported APIs are currently optimized for correctness first | Expect performance to change as NIIN's NCCL-backed paths and launch heuristics are tuned. The main knob available today is `NIIN_NUM_QPS`; see [Multi-QP (GIN contexts)](#multi-qp-gin-contexts). |

### Cannot be implemented with current NCCL APIs

| Feature | Reason |
|---|---|
| GIN-native network atomics | NCCL GIN does not expose remote atomic operations. NIIN's optional direct GPUNetIO sidecar covers integral AMOs without adding a GIN operation. A separate SRQ/GDRCopy floating-AMO prototype is not part of NIIN's supported coverage. |
| Device-side collectives | NCCL does not expose device-kernel-callable NVSHMEM collectives such as reductions, broadcast, alltoall, or fcollect |

### Out of scope for now

| Feature | Notes |
|---|---|
| Scalar network get (`nvshmem_<TYPE>_g`) | Block gets use GIN; scalar get returns a value directly and has no symmetric destination buffer |
| Tile put/get APIs | Not part of NIIN's current supported RMA subset |

### Not yet implemented (could be added)

| Feature | Notes |
|---|---|
| Device-side custom (user-created) team query | Device supports all 6 predefined teams but not `team_split_strided` results |

### NCCL infrastructure limitations (non-API-surface)

These are fundamental differences between NCCL and NVSHMEM's underlying infrastructure that affect behavior even for implemented APIs:

| Limitation | Impact | Notes |
|---|---|---|
| **PCIe device atomics** | Device atomics, including LSA barrier atomics, are not supported over PCIe | This follows the same practical restriction users must consider with NVSHMEM device atomics. |
| **No multi-process-per-GPU (MPG)** | `NVSHMEMI_TEAM_SAME_GPU` is always size 1 | NIIN uses one PE per GPU. |
| **Symmetric heap is a single NCCL window** | All `nvshmem_malloc` allocations come from one pre-registered window | NVSHMEM can register multiple heaps; NIIN uses a fixed 256MB (configurable) bump+free-list allocator |
| **No NVLS multicast on non-Hopper** | `nvshmemx_mc_ptr` returns nullptr on systems without NVLS support | This follows hardware and NCCL capability availability. |

### Out of scope

| Feature | Reason |
|---|---|
| Queue Pair (QP) APIs | NVSHMEM-specific API; out of scope at this time. NIIN does use multiple QPs internally -- see [Multi-QP (GIN contexts)](#multi-qp-gin-contexts) -- but does not expose NVSHMEM's QP-management calls. |
| SHMEM interop (`nvshmemx_init_attr(SHMEM)`) | No OpenSHMEM runtime in NCCL |
| EGM / fabric memory handles | NCCL handles its own memory registration |

## Environment and Runtime Configuration

NIIN uses NCCL for communication, so NCCL environment variables control transport
selection, logging, and tuning. NIIN recognizes only the NVSHMEM-named
compatibility variables listed in the next table; this list is complete for the
current release. Other NVSHMEM-specific environment variables are not interpreted
by NIIN.

| NVSHMEM-named variable | Description |
|---|---|
| `NVSHMEM_SYMMETRIC_SIZE` | Symmetric heap size; default is `256M`; supports K/M/G suffixes |
| `NVSHMEM_MAX_CTAS` | Maximum CTAs for NIIN's internal host on-stream block RMA launch heuristic; default is `16` |

NIIN also provides the NIIN-specific `NIIN_PUT_SIGNAL_MODE` variable to select
put-signal routing: `auto`, `separate`, `split`, or `fence_signal`.

`NIIN_NUM_QPS` sets the number of GIN contexts (QPs per peer) NIIN round-robins
operations across; the default is `4`, NCCL rounds it up to a multiple of the GIN
connection count, and `1` restores single-QP behavior. See
[Multi-QP (GIN contexts)](#multi-qp-gin-contexts).

For NCCL runtime behavior, use NCCL environment variables. Common examples:

| NCCL variable | Description |
|---|---|
| `NCCL_DEBUG` | NCCL logging and diagnostics |
| `NCCL_IB_HCA` | Selects IB HCAs used by NCCL |
| `NCCL_IB_DISABLE` | Disables NCCL IB transport when set |
| `NCCL_P2P_DISABLE` | Disables NCCL P2P transport when set |
| `NCCL_SHM_DISABLE` | Disables NCCL shared-memory transport when set |

Example launch using NCCL transport controls:

```bash
NCCL_DEBUG=INFO NCCL_IB_HCA=mlx5_1 CUDA_VISIBLE_DEVICES=0,1 \
mpirun -np 2 ./my_app
```

## Testing

### Multi-QP coverage

The network (GIN) paths only engage when the peer is not LSA-accessible, so
multi-QP has to be exercised across nodes. On a multi-node NVLink system such as
GB200 NVL72 that also means disabling multi-node NVLink, or peers on separate
trays remain LSA and never reach GIN at all:

```bash
# 2 nodes x 1 PE; NCCL_MNNVL_ENABLE=0 forces the IB/GIN path
NIIN_NUM_QPS=8 NCCL_MNNVL_ENABLE=0 \
  srun -N2 --ntasks-per-node=1 --mpi=pmix \
  ./build/shmem_put_bw -n 16 -t 256 -b 4096 -e 8388608
```

Confirm the contexts were actually created -- NCCL logs the real count, which
may be rounded up from `NIIN_NUM_QPS`:

```bash
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT ... 2>&1 | grep "creating .* contexts"
# devCommCreate: creating 8 contexts: 2 GIN connections with 4 contexts each
```

Sweep `NIIN_NUM_QPS` over 1, 2, 4, 8 and compare; run the counts interleaved
rather than in blocks so drift cannot masquerade as a trend. Operations rotate
across QPs independently of the grid, so the CTA count does not have to track
the QP count.
### Functional tests (`contrib/niin/test/niin_test.cu`)

Multi-GPU test using 2 GPUs via the low-level `niinInit`/`niinCommit` API. 24 test categories, 109 checks:

| Test | What it validates |
|---|---|
| Query APIs | `nvshmem_my_pe()`, `nvshmem_n_pes()`, team queries |
| nvshmem_ptr | Self returns base, LSA peer returns non-null |
| Scalar put + verify | `nvshmem_int_p()` cross-GPU |
| Block put | `nvshmem_int_put()` 64 elements |
| Scalar get | `nvshmem_int_g()` from LSA peer |
| Block get | `nvshmem_int_get()` 64 elements |
| putmem | 128 bytes untyped |
| Atomics (self) | fetch_add, compare_swap, swap, fetch, set, inc |
| Atomics (peer) | fetch_add on remote PE |
| Fence/Quiet | Doesn't crash |
| Wait/Test | All 6 CMP operators, test_all, test_any, test_some |
| Signal ops | signal_fetch, signal_wait_until |
| put_signal | Cross-GPU put with SIGNAL_ADD + signal wait |
| Strided iput | dst_stride=2, src_stride=1, 4 elements |
| Multi-type puts | float, double, long long cross-GPU |
| Bitwise atomics | fetch_and, fetch_or, fetch_xor |
| Vector wait/test | test_all_vector, test_any_vector, test_some_vector with per-element values |
| Warp put | `nvshmemx_int_put_warp()` cross-GPU, 32 threads cooperative |
| Block put | `nvshmemx_int_put_block()` cross-GPU, 64 threads cooperative |
| Misaligned warp put | 200 bytes at +3 byte offset, exercises head/body/tail phases |
| Large block put (64KB) | 256-thread cooperative vectorized copy cross-GPU |
| Odd-size block put | 1000 bytes (not multiple of 16), exercises tail handling |
| Block get | `nvshmemx_int_get_block()` 256 ints cooperative get cross-GPU |
| Misaligned block get | 999 bytes at +5 byte offset cross-GPU |
| Team operations | `team_split_strided`, `team_split_2d`, `team_translate_pe`, `team_destroy` |
| Stream-based RMA | `nvshmemx_int_put_on_stream()`, `nvshmemx_int_p_on_stream()` |

Build and run:

```bash
make -C contrib/niin NCCL_HOME=$PWD
export NIIN_HOME=$PWD/contrib/niin

nvcc contrib/niin/test/niin_test.cu -o niin_test \
    -I ${NIIN_HOME}/include \
    -L ${NIIN_HOME}/lib -lnvshmem_host \
    --expt-relaxed-constexpr -std=c++17 -arch=sm_89 \
    -Xlinker -rpath,${NIIN_HOME}/lib

./niin_test    # requires >= 2 GPUs
```

### NVSHMEM-compatible API test

Single-PE test using the pure NVSHMEM API surface (no `niin_*` calls):

```cpp
#include "nvshmem.h"

__global__ void kernel() {
    int pe   = nvshmem_my_pe();
    int npes = nvshmem_n_pes();
    // ... use nvshmem_* device APIs ...
}

int main() {
    nvshmem_init();
    int *buf = (int *)nvshmem_malloc(4096);
    nvshmem_barrier_all();
    kernel<<<1, 32>>>();
    cudaDeviceSynchronize();
    nvshmem_barrier_all();
    nvshmem_free(buf);
    nvshmem_finalize();
}
```

## Additional Documentation

- [Testing instructions](test/TESTING.md)

## License

NIIN is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt)
for the license text and [ThirdPartyNotices.txt](ThirdPartyNotices.txt) for
third-party dependency notes.
