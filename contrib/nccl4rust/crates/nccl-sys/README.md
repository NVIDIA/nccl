# nccl-sys

`nccl-sys` provides raw, build-time-generated Rust bindings to NCCL's public
host-callable API. It parses only `nccl.h` and `nccl_device.h`; it does not
include NCCL implementation headers or provide safe ownership wrappers.

## Build requirements

- Rust 1.85 or newer.
- libclang usable by `bindgen` (set `LIBCLANG_PATH` when it is not in a standard
  location).
- CUDA headers. Set `CUDA_HOME` to the toolkit root, or put that toolkit's
  `nvcc` in `PATH`.
- NCCL public headers and `libnccl`.

For an installed NCCL tree, set both directories explicitly:

```bash
export NCCL_INCLUDE_DIR=/opt/nccl/include
export NCCL_LIB_DIR=/opt/nccl/lib
export CUDA_HOME=/opt/cuda
export LD_LIBRARY_PATH=/opt/nccl/lib:$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
```

The CUDA runtime path is needed when the selected toolkit is not already in the
system dynamic-linker configuration.

`NCCL_INCLUDE_DIR` must contain both public entry points (`nccl.h` and
`nccl_device.h`) and the `nccl_device/` public include subtree.
When `NCCL_INCLUDE_DIR` is set without `NCCL_LIB_DIR`, the build deliberately
uses the platform linker's standard search paths for `-lnccl`; it will not mix
those explicit headers with `NCCL_SOURCE_DIR/build/lib` or this checkout's
`build/lib`. The caller must ensure that the library found by the linker matches
the selected headers.
If libclang cannot discover a nonstandard host C++ toolchain, pass its root with
the standard bindgen override, for example
`BINDGEN_EXTRA_CLANG_ARGS=--gcc-toolchain=/opt/gcc`.

For an NCCL source checkout, set `NCCL_SOURCE_DIR` to its root after generating
`build/include/nccl.h`. The build script finds host declarations in
`build/include`, device declarations in `build/include` or `src/include`, and
the library in `build/lib`. When built in the NCCL repository containing this
crate, that checkout is used as the final default.
If no explicit or source-tree library directory is available, the platform
linker's standard search paths are used for `-lnccl`.

The generated crate is a direct C-ABI layer: it retains the exact C names and
adds no ownership or lifetime checks. Calls must follow NCCL's documented
preconditions. Profiling interposition entry points named `pnccl*` are
deliberately excluded. The crate's tests cover NCCL's central opaque handles
and the current public device-setup structure layouts. Bindgen's automatic
layout-test mode is intentionally off: bindgen 0.71.1 otherwise treats these
complete CUDA C++ structures as opaque.
