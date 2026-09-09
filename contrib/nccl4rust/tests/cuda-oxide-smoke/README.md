# CUDA-Oxide compile-and-link smoke

This package ensures CUDA-Oxide can lower the complete Rust-facing NCCL device
surface, then resolves every raw shim call through nvJitLink. Its compile-only
kernels cover:

- communicator queries;
- world, LSA, and rail teams derived from scalar communicator queries, plus
  typed rank translation;
- local, LSA, world, team, explicit-multimem, and LSA-multimem pointers with
  several Rust pointee types; multimem calls remain link-covered behind a
  runtime flag so the kernel can also execute without multimem support;
- thread-, warp-, and CTA-cooperative LSA barriers, including runtime `u32`
  indices and false, true, and runtime boolean argument mapping; and
- the CTA `f32` LSA reduce-sum-copy building block.

Before linking, the host driver also verifies that the emitted PTX or NVVM IR
references every symbol in the raw device shim. This guards against a wrapper
or compile-only kernel silently dropping out of code generation. The link-only
smoke does not load CUDA or require a working GPU.

## Hopper-native PTX + LTOIR link (default)

CUDA-Oxide's normal backend lowers Rust to PTX for `sm_90`. The PTX keeps the
NCCL shim calls as external `.func` declarations; nvJitLink resolves those
against the matching NCCL device LTOIR module.

```bash
cd contrib/nccl4rust
CUDA_HOME=/path/to/cuda make device ARCH=90 NCCL_INCLUDE_DIR=/path/to/nccl/include

cd tests/cuda-oxide-smoke
CUDA_HOME=/path/to/cuda cargo oxide build nccl4rust-cuda-oxide-smoke --arch=sm_90
CUDA_HOME=/path/to/cuda cargo run --offline
```

Use the CUDA-Oxide codegen backend from the revision pinned in `Cargo.toml`.
The dependency revision and backend must agree; a newer backend may reject or
lower device-extern signatures differently.

The defaults are:

- Link mode: `ptx`
- Rust input: `nccl4rust_cuda_oxide_smoke.ptx`
- NCCL shim input: `../../build/libnccl4rust_device_sm_90.ltoir`
- Link target: `sm_90`
- Output: `nccl4rust_cuda_oxide_smoke_sm_90.cubin`

## Blackwell NVVM IR + LTOIR link

The all-LTO path compiles CUDA-Oxide's emitted NVVM IR to LTOIR with libNVVM,
then gives both matching LTOIR modules to nvJitLink. Building for `sm_100`
requires CUDA Toolkit 12.8 or newer:

```bash
cd contrib/nccl4rust
CUDA_HOME=/path/to/cuda make device ARCH=100 NCCL_INCLUDE_DIR=/path/to/nccl/include

cd tests/cuda-oxide-smoke
CUDA_HOME=/path/to/cuda cargo oxide build nccl4rust-cuda-oxide-smoke \
  --emit-nvvm-ir --arch=sm_100
CUDA_HOME=/path/to/cuda \
  NCCL4RUST_LINK_MODE=nvvm-lto \
  NCCL4RUST_ARCH=sm_100 \
  cargo run --offline
```

The pinned CUDA-Oxide revision's NVVM-IR exporter currently supports the
opaque-pointer NVVM 20 dialect for Blackwell and newer (`sm_100+`). Its
typed-pointer NVVM 7 exporter for pre-Blackwell targets is not yet supported,
so the all-LTO mode rejects an `sm_90` target with a focused diagnostic. The
default PTX path avoids that limitation and produces a native `sm_90` cubin.

Set `NCCL4RUST_LINK_MODE` to `ptx` or `nvvm-lto`. The input/output overrides
are `NCCL4RUST_RUST_PTX`, `NCCL4RUST_RUST_NVVM_IR`,
`NCCL4RUST_DEVICE_LTOIR`, and `NCCL4RUST_CUBIN`; `NCCL4RUST_ARCH` selects the
matching target and defaults to `sm_90`. Prefer a device shim built for the
same architecture as the Rust input and final cubin. `CUDA_HOME` (or
`CUDA_PATH`) selects the toolkit containing libNVVM, libdevice, and nvJitLink.
Runtime kernel execution remains a separate test that requires matching NCCL
host setup and a multi-GPU worker.
