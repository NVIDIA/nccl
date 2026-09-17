# NIIN GPUNetIO atomics and SRQ proxy

This is a deliberately narrow, NIIN-owned escape hatch for atomic operations
that GIN does not currently expose. It is not a new GIN API and it does not
replace any existing NIIN transport path:

- NIIN puts, gets, signals, barriers, and all existing RMA continue to use
  GIN exactly as before.
- In native-direct mode, only remote integral AMOs use the GPUNetIO WQE
  sidecar; self and LSA AMOs remain CUDA atomics.
- An optional raw-verbs/SRQ proxy source tree is retained for development of
  floating AMOs; it is not part of the supported native-direct provider.
- GPUNetIO is consumed as an external dependency. No GPUNetIO source is
  modified, and there are no NCCL core or GIN changes.

Keeping this code here means it can be removed cleanly when GIN exposes native
atomic operations.

> **SRQ proxy status:** `NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS` is
> experimental. The native direct provider below is the supported result of
> this work. The proxy has no peer-visible recovery when a target response
> fails, and a CPU GDRCopy BAR update cannot be ordered with a target GPU
> kernel that is already running. Do not enable it for normal applications.

## What it owns

The provider creates a small private atomic fabric:

- one outbound RC QP per remote PE, configured for
  `DOCA_VERBS_QP_ATOMIC_MODE_UP_TO_8BYTES`;
- one independent registration of the existing NIIN symmetric heap, with
  remote-atomic access only;
- one registered 16-byte response slot and device lock per destination PE;
- a public-NCCL all-gather of only its own QP and heap-MR metadata;
- a collective setup gate that requires GPUNetIO GPU-SM doorbells.

The response slot serialization is intentional for the first functional
implementation: a PE's dedicated QP and response buffer are never reused
until its completion is observed. It can later grow into a ticketed ring
without changing the public NIIN AMO API.

The provider verifies the selected HCA supports standard and masked
fetch-add/compare-swap, plus 4- and 8-byte RC-QP atomics, before exposing any
endpoint. It fails closed rather than advertise extended WQEs on unsupported
hardware.

## Native-direct coverage

The direct GPUNetIO provider covers 4- or 8-byte integral AMOs when it is
bound:

- fetch-add/add, compare-swap, swap/set, fetch, increment/fetch-increment;
- fetch and non-fetch AND, OR, and XOR;
- masked 32-bit operations and the two-WQEBB masked 64-bit compare-swap
  encodings needed by set/swap/AND/OR.

Self and LSA peers still use CUDA atomics. When no provider is bound, remote
AMOs keep NIIN's previous fail-closed behavior.

`ncclMemAlloc` normally returns a CUDA VMM heap, for which CUDA rejects the
legacy `CU_POINTER_ATTRIBUTE_SYNC_MEMOPS` pointer attribute. The sidecar does
not require that attribute: its direct and proxy paths use explicit completion
and system-fence ordering. This still does not make unsafely concurrent target
kernel access valid. As with other GPUDirect RDMA updates, applications must
use their normal CUDA/NVSHMEM phase synchronization before a target kernel
consumes a remote atomic update.

## Experimental SRQ proxy

Native NIC atomics cannot express floating-point addition. When built with
`NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS=ON`, the same library also includes a
separate NIIN-owned raw-verbs/GDRCopy SRQ prototype. Its current code explores
proxy-all routing for the integral API, raw-bit float/double fetch/swap/set,
and the six
`nvshmemx_{half,float,double}_atomic_{fetch_add,add}` operations.

Initialize and bind either `niinGpunetioAtomic*` or
`niinGpunetioProxyAtomic*` for a given device context, never both. The
GPUNetIO `CPU_PROXY` handler remains unrelated: it only forwards GPUNetIO
doorbells and is not the semantic SRQ AMO fallback. The native direct provider
therefore rejects CPU-doorbell QPs collectively; request GPU-SM doorbells when
bring-up requires a deterministic choice.

The proxy maps each target's existing symmetric heap through GDRCopy and
serializes AMOs on that target's NIIN progress thread. It requires a working
GDRCopy kernel driver and runtime library on every PE. It is intentionally
kept outside the supported coverage matrix until it has a safe target-GPU
ordering handoff and source-visible transport-failure recovery. In particular,
do not use proxy-all self or LSA AMOs: the host BAR update cannot be ordered
with the waiting GPU kernel.

## Build

Point the optional target at an unmodified GPUNetIO Open source tree and an
NCCL build/install prefix. Its version must exactly match the GPUNetIO device
headers used to build NCCL; the CMake configuration rejects an ABI mismatch:

```bash
make -C contrib/niin gpunetio-atomics \
  NCCL_HOME=/path/to/nccl \
  GPUNETIO_HOME=/path/to/gpunetio-open \
  GPUNETIO_CUDA_ARCHITECTURES=89-real
```

Set `GPUNETIO_CUDA_ARCHITECTURES` to the deployment GPU architecture (the
example is an L40S-only `sm_89` artifact). It is forwarded as CMake's
`CMAKE_CUDA_ARCHITECTURES`; choosing a `-real` architecture embeds native SASS
and avoids relying on a driver to JIT PTX emitted by a newer CUDA toolkit.

For current NCCL development trees, GPUNetIO may be vendored without its own
`CMakeLists.txt`. That is also supported without changing its sources; point
the build at the matching NCCL checkout as well:

```bash
cmake -S contrib/niin/gpunetio -B build/niin-gpunetio \
  -DNCCL_HOME=/path/to/nccl-build-or-prefix \
  -DNCCL_SOURCE_ROOT=/path/to/nccl-source \
  -DGPUNETIO_HOME=/path/to/nccl-source/src/transport/net_ib/gdaki/doca-gpunetio
cmake --build build/niin-gpunetio --target niin_gpunetio_compile_check
```

For development-only compilation of the SRQ proxy, enable it when GDRCopy
headers are available:

```bash
cmake -S contrib/niin/gpunetio -B build/niin-gpunetio \
  -DNCCL_HOME=/path/to/nccl-build-or-prefix \
  -DNCCL_SOURCE_ROOT=/path/to/nccl-source \
  -DGPUNETIO_HOME=/path/to/nccl-source/src/transport/net_ib/gdaki/doca-gpunetio \
  -DNIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS=ON \
  -DGDRCOPY_INCLUDE_DIR=/path/to/gdrcopy/include
cmake --build build/niin-gpunetio --target niin_gpunetio_proxy_atomic_smoke
```

The Make helper forwards the same options:

```bash
make -C contrib/niin gpunetio-atomics \
  NCCL_HOME=/path/to/nccl \
  GPUNETIO_HOME=/path/to/gpunetio-open \
  NIIN_GPUNETIO_ENABLE_SRQ_PROXY_ATOMICS=ON \
  GDRCOPY_INCLUDE_DIR=/path/to/gdrcopy/include
```

If rdma-core is not installed in a standard compiler location, the Make helper
also accepts `IBVERBS_INCLUDE_DIR=/path/to/include` and
`IBVERBS_LIBRARY=/path/to/libibverbs.so`.

The target builds `libniin_gpunetio_atomics.a` and a CUDA compile-only check
that instantiates every current public NIIN AMO, including the masked WQEs.
The Make target also exposes the archive as
`contrib/niin/lib/libniin_gpunetio_atomics.a`. Prefer consuming the CMake
target so its CUDA, ibverbs, thread, dynamic-loader, and NCCL link
dependencies propagate correctly. It is intentionally separate from
`make -C contrib/niin all`, so ordinary NIIN users do not acquire a GPUNetIO
dependency.

Applications that use the provider must compile their CUDA code with
`NIIN_GPUNETIO_ENABLE=1` and the GPUNetIO public include directory. The CMake
target exports both requirements to CMake consumers.

## Lifecycle

Every PE calls the atomic initialization collectively, after the normal NIIN
two-phase setup has completed and outside `ncclGroupStart/End`:

```cpp
niinContext_host hostCtx;
niinContext* deviceCtx;
// niinInit(...), ncclGroupEnd(), and niinCommit(&hostCtx, deviceCtx) first.

niinGpunetioAtomicOptions opts = NIIN_GPUNETIO_ATOMIC_OPTIONS_INITIALIZER;
niinGpunetioAtomicHostContext* atomics = nullptr;
NCCLCHECK(niinGpunetioAtomicInit(comm, heapBase, heapBytes, &opts, &atomics));
NCCLCHECK(niinGpunetioAtomicBind(atomics, deviceCtx));

// Launch kernels using deviceCtx.

cudaDeviceSynchronize();
NCCLCHECK(niinGpunetioAtomicFinalize(atomics));
NCCLCHECK(ncclGroupStart());
NCCLCHECK(niinFinalize(comm, deviceCtx));
NCCLCHECK(ncclGroupEnd());
```

The experimental proxy has a parallel lifecycle API, but is deliberately not
shown as an application recipe until its target-GPU ordering and failure
recovery semantics are complete.

`niinGpunetioAtomicInit` uses only public NCCL/CUDA/ibverbs/GPUNetIO APIs. It
does not inspect a GIN context, borrow GIN QPs, reuse a GIN MR, or issue any
direct put/get/RMA operation.

Select the HCA/port/GID when automatic selection is not appropriate:

- `NIIN_GPUNETIO_IB_DEV=mlx5_0`
- `NIIN_GPUNETIO_IB_PORT=1`
- `NIIN_GPUNETIO_GID_INDEX=<index>`

The experimental SRQ proxy has separate selection variables:

- `NIIN_GPUNETIO_PROXY_IB_DEV=mlx5_0`
- `NIIN_GPUNETIO_PROXY_IB_PORT=1`
- `NIIN_GPUNETIO_PROXY_GID_INDEX=<index>`

The selected GPU allocation must be registerable by ibverbs (for example via
the site GPU peer-memory driver or DMA-BUF support). A direct atomic provider
cannot reuse GIN's private MR registration without new NCCL-core plumbing.

## Validation

The compile-only target instantiates the public AMO API. The direct smoke test
calls the native adapter directly and covers standard/masked 4/8-byte
operations and concurrent callers to one destination PE. The optional proxy
smoke is a compile-time development harness only. A forced GPUNetIO
CPU-doorbell handler is an expected collective rejection, rather than a kernel
launch.
