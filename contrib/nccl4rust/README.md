# nccl4rust

`nccl4rust` is an experimental Rust interface to NCCL's public host and device
APIs. It follows the CUDA-Oxide boundary prototyped by
[NVSHMEM MR !1878](https://gitlab-master.nvidia.com/nvshmem/nvshmem/-/merge_requests/1878),
while splitting raw bindings from Rust-style wrappers and adapting NCCL's C++
device templates through a small C ABI shim.

This contribution is a prototype. It is not part of NCCL's supported Rust API
and does not inherit NCCL core's release-quality guarantees.

## Maintainers

| Maintainer | Area |
|---|---|
| @benjaming | All |

## Layout

| Path | Purpose |
|---|---|
| `crates/nccl-sys` | Bindgen-generated raw host ABI, including public device-communicator setup APIs |
| `crates/nccl` | Rust-style host wrappers and RAII ownership |
| `crates/nccl-device-sys` | `no_std` CUDA-Oxide declarations for the device shim |
| `crates/nccl-device` | Typed `DevComm`, `Team`, `Window`, pointer, barrier, and reduce/copy wrappers |
| `shim/` | CUDA C++ C-ABI shim built exclusively from public `nccl.h` and `nccl_device.h` |
| `tests/cuda-oxide-smoke` | Compile-and-link smoke kernels for the Rust/device-shim boundary |
| `tests/runtime-smoke` | One-process, two-GPU host collective and device-kernel runtime validation |

The split is intentional. Host applications can use `nccl` without a Rust GPU
compiler. CUDA-Oxide kernels use `nccl-device`; raw consumers can opt into the
corresponding `-sys` crate.

## Current surface

The raw host crate generates the exported public host ABI visible in the
selected NCCL headers. Profiling interposition symbols and header-only inline
helpers are intentionally outside that generated FFI. The initial Rust host
layer covers version and unique-ID handling, communicator ownership and
queries, typed collectives and point-to-point operations, grouped calls, NCCL
memory, registered windows, and creation of a public device communicator.
The current collective and point-to-point wrappers accept raw device pointers
and are declared `unsafe`. Their contracts require correctly sized,
CUDA-accessible buffers to remain valid until the stream completes; a
stream-aware buffer abstraction can encode those requirements in safe APIs.

Safe communicator initialization consumes and retains its `Config`, forces the
initial blocking mode, and rejects explicitly nonblocking configuration. Safe
output/resource constructors also reject wrapper-managed `Group` scopes, where
NCCL is allowed to defer writes to caller-owned output storage. If an external
configuration override still produces `ncclInProgress`, those constructors
keep their output storage alive while polling to a terminal state. Enqueue and
wrapper-managed group paths likewise poll asynchronous launch setup to
completion when an external override forces nonblocking behavior; this does
not wait for the CUDA stream's communication work to finish.

The initial device layer covers:

- world, LSA, and rail communicator/team queries;
- local, LSA-peer, world-peer, and multimem window pointer translation;
- thread, warp, and CTA LSA barriers; and
- a CTA `f32` LSA reduce-sum-copy instantiation.

The compile-and-link CUDA-Oxide smoke covers all 16 raw device-shim calls and
the complete initial Rust wrapper surface without requiring a GPU. The runtime
smoke exercises scalar and typed team queries, rank translation, window pointer
translation, LSA barriers, and an offset/tail-sensitive registered-window
reduce-sum-copy in addition to a host `f32` all-reduce. GIN, hybrid barriers,
low-latency all-to-all, additional reduce/copy types, and runtime multimem
coverage are follow-on work.

## Requirements

- Matching NCCL 2.31 headers and runtime. This prototype is developed against
  NCCL 2.31.0 and directly initializes fields that differ in earlier NCCL
  device-API versions.
- CUDA Toolkit 12.2 or newer for the Hopper flow. The optional `sm_100+`
  Blackwell flow requires CUDA 12.8 or newer; current checks use CUDA 13.2.
- Rust 1.85+ for host crates.
- A CUDA-Oxide-compatible nightly toolchain and LLVM 21+ (`llc`) for device
  kernels.
- Clang/libclang for `bindgen`.
- CMake and a C++17-capable host compiler for NCCL headers.

Multimem requires appropriate Hopper-or-newer NVLink hardware. GIN will add
its own transport and version restrictions when that surface is implemented.

## Prepare matching NCCL headers and library

The checked-out NCCL source and any previously existing `build/` directory may
be at different versions. Configure a fresh build and generate its public
headers before building Rust bindings:

```bash
cmake -S ../.. -B ../../build-nccl4rust \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90
cmake --build ../../build-nccl4rust --target nccl_header -j
cmake --build ../../build-nccl4rust --target nccl -j
```

All device code and the runtime `libnccl.so` must come from compatible NCCL
versions. Rebuild the device shim whenever the NCCL device API version changes.
Select `CMAKE_CUDA_ARCHITECTURES` for the test GPUs so the NCCL library contains
native SASS. Otherwise a newer toolkit's embedded PTX may require a newer CUDA
driver than the node provides.

## Build and check host crates

Point binding generation at the public include and library directories:

```bash
export NCCL_INCLUDE_DIR="$PWD/../../build-nccl4rust/include"
export NCCL_LIB_DIR="$PWD/../../build-nccl4rust/lib"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="$NCCL_LIB_DIR:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cargo check --workspace
cargo test -p nccl-sys -p nccl
```

`nccl-sys` also recognizes `NCCL_SOURCE_DIR` when the installed include tree is
not self-contained. Repository-relative `build/include`, `src/include`, and
`build/lib` are fallbacks for in-tree development.

## Minimal host use

One bootstrap rank generates a unique ID and transports its bytes to every
other rank using an out-of-band channel. After selecting the local CUDA device,
each rank reconstructs that ID and initializes its communicator:

```rust,no_run
use nccl::{Communicator, UniqueId};

fn create_bootstrap_id() -> nccl::Result<[u8; UniqueId::BYTE_LEN]> {
    Ok(UniqueId::generate()?.into_bytes())
}

fn initialize_rank(
    rank_count: i32,
    rank: i32,
    id_bytes: [u8; UniqueId::BYTE_LEN],
) -> nccl::Result<Communicator> {
    let id = UniqueId::from_bytes(id_bytes);
    Communicator::init_rank(rank_count, &id, rank)
}
```

Collective and point-to-point methods currently take raw device pointers and a
`CudaStream`. Their documented contracts require buffers to remain valid until
the CUDA stream completes.

## Build the device LTOIR

The device shim uses only installed/public NCCL headers. `ARCH` is a numeric SM
architecture and defaults to `90`:

```bash
make device \
  NCCL_INCLUDE_DIR="$NCCL_INCLUDE_DIR" \
  CUDA_HOME="$CUDA_HOME" \
  ARCH=90
```

The result is `build/libnccl4rust_device_sm_90.ltoir`. The default Hopper flow
adds CUDA-Oxide's Rust PTX and this shim LTOIR to nvJitLink with a matching
`-arch=sm_90` target. An optional all-LTO flow is available for `sm_100+` with
CUDA 12.8 or newer.

The `cuda-oxide` feature applies CUDA-Oxide's `#[device]` annotation to the raw
`nccl-device-sys` shim declarations; `nccl-device` supplies inline Rust
wrappers over those declarations:

```bash
cargo check -p nccl-device-sys --features cuda-oxide
cargo check -p nccl-device --features cuda-oxide
```

The standalone smoke package lowers Rust kernels to PTX and links that PTX with
the shim LTOIR into a native cubin. It does not load CUDA or require a GPU:

```bash
cd tests/cuda-oxide-smoke
cargo oxide build nccl4rust-cuda-oxide-smoke --arch=sm_90
CUDA_HOME="$CUDA_HOME" cargo run --offline
```

See `tests/cuda-oxide-smoke/README.md` for the optional Blackwell all-LTO mode
and input/output overrides.

## Run the multi-GPU runtime smoke

After producing the `sm_90` cubin above, run the standalone runtime package on
a node with two visible compatible GPUs:

```bash
cd ../runtime-smoke
export NCCL4RUST_CUBIN=../cuda-oxide-smoke/nccl4rust_cuda_oxide_smoke_sm_90.cubin
timeout 240s cargo run --release --offline
```

The test uses one Rust thread and CUDA device per rank. It validates a typed
host `f32` all-reduce, device-communicator creation and byte copying, scalar and
typed world/LSA/rail queries, rank translation, self-pointer translation at two
window offsets, and a 17-element device LSA reduce-sum-copy. The device reduction
runs for two barrier epochs over registered source and destination windows and
checks surrounding guards before explicit device-communicator teardown and host
finalization. See `tests/runtime-smoke/README.md` for the complete environment
and launch contract.

## Host/device ownership boundary

`ncclDevCommCreate` produces a versioned public structure in host memory. The
host `DeviceCommunicator` wrapper owns that structure and destroys it before
its parent communicator. CUDA-Oxide remains responsible for allocating device
memory, copying those bytes, and keeping the copy alive while kernels execute.
Kernels construct `nccl_device::DevComm` from a pointer to that device copy.
Using a pointer rather than a by-value Rust mirror keeps the versioned C struct
layout out of the kernel argument ABI.

The NCCL runtime-version device-communicator mode is intentionally not exposed
yet. It requires querying and allocating a runtime-selected image size, while
the current wrapper owns the compile-time `ncclDevComm_t` representation.

Likewise, nccl4rust does not select devices, create CUDA contexts, create
streams, copy memory, or load modules. Those remain responsibilities of the
application's CUDA host library. Unlike NVSHMEM's module API, NCCL device
communicators do not require per-module registration.

## Current API contracts

- The raw `-sys` crates mirror the C ABI without adding ownership or lifetime
  validation; calls must follow NCCL and CUDA preconditions.
- Communicator, allocation, window, and device-communicator wrappers enforce
  destruction order through Rust lifetimes and RAII where the host API permits.
- NCCL retains `ncclConfig_t::commName`; the owned communicator therefore keeps
  its `Config` and string storage alive through final destruction.
- Safe initialization and output-producing management calls must not be mixed
  with raw `nccl-sys` group state, which the wrapper cannot observe.
- Current raw-pointer enqueue methods do not tie buffer borrows to CUDA
  completion; callers maintain the documented stream and resource lifetimes.
- Current pointer-translation methods return raw device pointers and cannot
  validate offset bounds, alignment, peer membership, aliasing, or window
  lifetime; callers supply those invariants.
- Current barrier methods do not encode participation and convergence in their
  types; callers coordinate all required threads and ranks with matching
  arguments. Reduce/copy invocations are not rank-collective: callers assign
  each output region to one issuer or to issuers with disjoint regions, while
  arranging any required barriers separately.

## Validation status

The public-header CUDA shim is compile-checked for `sm_90` with CUDA 13.2, and
its output has the expected raw LTOIR magic (`ed 43 4e 7f`). CUDA-Oxide's Rust
PTX and that shim LTOIR also link successfully into a native `sm_90` cubin. The
link smoke verifies that all 16 shim identifiers survive code generation. Host
binding and wrapper tests cover raw ABI signatures, public-structure layout,
every initial device wrapper's argument mapping, and invalid or overflowing
device-communicator requirements. They pass against matching freshly built
NCCL 2.31 headers and `libnccl.so`, including a loaded-library version check.

The runtime smoke passes repeatedly on two directly connected H100 GPUs with
NCCL 2.31.0 and CUDA 13.2. Both host and device reductions return `3.0` on
both ranks; the kernels also validate scalar queries, world/LSA/rail teams and
rank translation, local/LSA/world/team self-pointer mappings at two offsets,
two reused LSA barrier epochs, a 17-element non-16-byte-aligned reduction tail,
and untouched guard regions. Runtime multimem is intentionally disabled. The
NCCL library was built with native `sm_90` SASS for compatibility with the
node's driver.

## License

Apache License 2.0. See `LICENSE.txt`. Direct third-party dependencies are
listed in `ThirdPartyNotices.txt`.
