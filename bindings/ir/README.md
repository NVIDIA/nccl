# NCCL device IR bindings

`bindings/ir` provides a higher-level IR ABI for language bindings on top of
NCCL device APIs. Its C-linkage functions are compiled into LLVM bitcode or
NVIDIA LTO-IR, letting GPU compilers and DSLs link NCCL communication operations
into their kernels.

[nccl4py's CuTeDSL API](../nccl4py/nccl/core/device/cute/) uses the LLVM bitcode
library, `libnccl_device.bc`, to provide NCCL device operations in CuTeDSL kernels.

## Build

Requires a CUDA Toolkit with NVCC. LLVM bitcode additionally requires Clang
with NVPTX support, `opt`, `llvm-dis`, and `llvm-as`.

From the NCCL repository root, build both formats (adjust CUDA path and GPU
architecture as needed):

```sh
make -j ir.build CUDA_HOME=/usr/local/cuda IR_ARCH=sm_90
```

This also builds native NCCL. Use `ir.llvm_ir` or `ir.ltoir` instead of
`ir.build` to select one format.

Outputs under `BUILDDIR` (`build/` by default):

- `include/nccl_device_ir.h`: device API declarations.
- `lib/libnccl_device.bc`: LLVM bitcode library.
- `lib/libnccl_device.ltoir`: NVIDIA LTO-IR library.

See [nccl_device_ir.h](nccl_device_ir.h) for the available APIs.
