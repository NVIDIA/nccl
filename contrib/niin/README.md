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

Set the runtime library path or link with an rpath:

```bash
export LD_LIBRARY_PATH=${NIIN_HOME}/lib:${LD_LIBRARY_PATH}
```

NIIN does not build or require a separate `libnvshmem_device.a`. Device-side
NVSHMEM functions are compiled from NIIN's headers into the application's CUDA
translation units. During `nvshmem_init()`/`nvshmemx_init_attr()`, NIIN creates
the NCCL device communicator and symmetric heap state, then publishes the device
context pointer to NIIN's `__device__` global with `cudaMemcpyToSymbol`.

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

1. Detects rank and world size from an MPI communicator or supported launcher-provided rank metadata. Falls back to single-PE mode if none found.
2. Selects the local GPU from launcher-provided local rank metadata or `rank % nDevices`.
3. Creates an NCCL communicator (`ncclCommInitRank` for multi-PE with MPI, `ncclCommInitAll` for single-PE).
4. Allocates the symmetric heap via `ncclMemAlloc` (default 256 MB, configurable via `NVSHMEM_SYMMETRIC_SIZE`).
5. Registers the heap as an NCCL window (`ncclCommWindowRegister`) and creates a device communicator (`ncclDevCommCreate`) with GIN resources.
6. Sets NIIN's `__device__` global context pointer via `cudaMemcpyToSymbol` so all subsequent kernel launches have access to the NIIN context automatically.

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
| `nvshmem_init()` | Full | Auto-detects rank metadata; single-PE fallback |
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
| `nvshmem_fence()` | Full | Host: cudaDeviceSynchronize; Device: __threadfence_system |
| `nvshmem_quiet()` | Full | Host: same as fence (`cudaDeviceSynchronize`); Device: GIN flush + threadfence |
| `nvshmem_info_get_name()` | Full | Returns "NIIN (NVSHMEM Implemented In NCCL)" |
| `nvshmem_info_get_version()` | Full | Reports 3.0 |
| `nvshmemx_barrier_all_on_stream()` | Full | |
| `nvshmemx_quiet_on_stream()` | Full | Stream-ordered device quiet kernel: GIN flush + `__threadfence_system()` |
| `nvshmemx_collective_launch()` | Stub | Redirects to cudaLaunchKernel |
| `nvshmemx_cumodule_init/finalize()` | Stub | No-op (compatibility) |
| `nvshmem_global_exit()` | Full | finalize + exit |

### Host Stream-Ordered APIs (`nvshmemx.h`)

All `nvshmemx_*_on_stream` variants are implemented. These APIs match
NVSHMEM's host on-stream contract: callers provide a `cudaStream_t`, but do not
choose grid or CTA geometry. NIIN owns the internal launch geometry; block
put/get and put-signal fallbacks use a size-based number of eight-warp CTAs for
self and LSA transfers, capped by `NVSHMEM_MAX_CTAS`, so the LSA path can use
the cooperative block RMA implementation. Non-LSA GIN transfers remain
single-CTA until NIIN can assign independent GIN contexts per CTA. Scalar,
strided, signal, and wait wrappers use compact control kernels.

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
| `nvshmem_fence()` | Full | Local system-scope ordering via `__threadfence_system()`; not a collective barrier |
| `nvshmem_quiet()` | Full | `gin.flush()` + `__threadfence_system()` |
| `nvshmem_barrier_all()` | Full | `ncclBarrierSession` (world team) with acquire/release barrier ordering |
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
| `nvshmem_<TYPE>_atomic_fetch_add()` | Full | **Not impl** | **No network atomics in NCCL GIN** |
| `nvshmem_<TYPE>_atomic_add()` | Full | **Not impl** | |
| `nvshmem_<TYPE>_atomic_compare_swap()` | Full | **Not impl** | |
| `nvshmem_<TYPE>_atomic_swap()` | Full | **Not impl** | |
| `nvshmem_<TYPE>_atomic_fetch()` | Full | **Not impl** | `atomicAdd(ptr, 0)` |
| `nvshmem_<TYPE>_atomic_set()` | Full | **Not impl** | `atomicExch` |
| `nvshmem_<TYPE>_atomic_inc/fetch_inc()` | Full | **Not impl** | |
| `nvshmem_<TYPE>_atomic_fetch_and/or/xor()` | Full | **Not impl** | Bitwise types only |
| `nvshmem_<TYPE>_atomic_and/or/xor()` | Full | **Not impl** | |

Standard AMO types (11): `int`, `long`, `longlong`, `uint`, `ulong`, `ulonglong`, `int32`, `int64`, `uint32`, `uint64`, `size`.
Bitwise AMO types (7): `uint`, `ulong`, `ulonglong`, `int32`, `int64`, `uint32`, `uint64`.

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
| No standalone NVSHMEM bootstrap runtime | Multi-PE setup is driven by MPI, UID-based initialization, or launcher-provided rank metadata | Use `nvshmemx_init_attr(...MPI_COMM...)` with `-DNIIN_HAS_MPI`, or use `nvshmemx_get_uniqueid()` with `NVSHMEMX_INIT_WITH_UNIQUEID` and an application-provided bootstrap. |
| One NCCL PE per GPU | Multi-process-per-GPU NVSHMEM use cases are not represented | `NVSHMEMI_TEAM_SAME_GPU` is always a single-PE team. |
| No arbitrary NVSHMEM team creation | Applications cannot create teams with arbitrary PE membership, layouts, or team-specific contexts | NIIN represents predefined teams plus constrained host-side strided/2D split teams only. User-created teams are not available to device APIs. |
| NCCL memory/window model | Symmetric allocations come from one NCCL-registered heap | Set `NVSHMEM_SYMMETRIC_SIZE` before initialization when a larger heap is required. |
| NCCL device transport constraints | API support depends on whether the target PE is LSA-accessible or reachable through GIN | Block gets can use GIN; scalar network gets and network atomics are not implemented. |
| Performance is not yet tuned | Supported APIs are currently optimized for correctness first | Expect performance to change as NIIN's NCCL-backed paths and launch heuristics are tuned. |

### Cannot be implemented with current NCCL APIs

| Feature | Reason |
|---|---|
| Network atomics | NCCL GIN does not expose remote atomic operations |
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
| Queue Pair (QP) APIs | NVSHMEM-specific API; out of scope at this time |
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
