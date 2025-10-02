# Eugo Notes
This is a CMake specific port of NVIDIA's NCCL library. We exclusively leverage Clang for both host and device code compilation.

This is unlikely to work with NVIDIA NVCC compiler.

## IMPORTANT: Eugo Maintenance guide about syncing with upstream:

1. For any `.cu`, `.cc`, in `src/device`, we must add/edit their file extensions:
    1. In the `colldevice` library, if there are other files besides `.cu` (like `host_table.cc`) we need to ensure they are compiled using the proper language (for example, `host_table.cc` is actually a host only cuda file, and to ensure it gets compiled correctly, we renamed it to `host_table.cu` so it gets treated as a cuda file and not pure/usual C++ file).
    2. For any new `.cu`, `.cc`, or other source files, we need to ensure they are compiled using the proper language (e.g., `.cu` files should be compiled as CUDA files, while `.cc` files should be compiled as C++ files).
2. All new files, targets, and libraries:
    1. Wherever NCCL adds new source files, headers, etc, we need to add them to the appropriate existing CMake targets, or introduce brand new ones.
    2. We need to compare (`diff` or `BeyondCompare`) the new `src/` structure with the one prior syncing with upstream repo.
        1. RETODO: Script to automatically compare the new `src/` structure with the one prior syncing with upstream repo.

## NCCLRas

Testing ncclras

```bash
bash-5.2# ncclras --version
NCCL RAS client version 2.28.3
```

# NCCL

Optimized primitives for inter-GPU communication.

This fork enables building NCCL with CMake, Clang for both host and device, and using system nvtx instead of the bundled NVTX, and is based on the original NCCL repository. It is designed to be compatible with the original NCCL library, while providing additional build features and improvements.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

More NCCL resources:
• https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#downloadnccl
• https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
• https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy


## Inspiration

The inspiration for this cmake port came from (thank you @cyyever):
• https://github.com/NVIDIA/nccl/pull/664/files
• https://github.com/cyyever/nccl/tree/cmake
• https://github.com/NVIDIA/nccl/issues/1287

The xla patch for clang came from:
• https://github.dev/openxla/xla/blob/main/third_party/nccl/archive.patch


## Original NCCL repo resources:
• `export NVCC_APPEND_FLAGS='-allow-unsupported-compiler'`: https://github.com/NVIDIA/cuda-samples/issues/179
• Provenance of NVTX headers in NCCL: https://github.com/NVIDIA/nccl/issues/1270


## Clang specific CUDA resources:
• https://github.com/llvm/llvm-project/blob/main/clang/lib/Basic/Targets/NVPTX.cpp
• https://llvm.org/docs/LangRef.html#:~:text=x%3A%20Invalid.-,NVPTX,-%3A
• https://llvm.org/docs/CompileCudaWithLLVM.html


## NVCC Identification Macros
• https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#compilation-phases