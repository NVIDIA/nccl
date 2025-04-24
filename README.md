# NCCL

Optimized primitives for inter-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

More NCCL resources:
• https://docs.nvidia.com/deeplearning/nccl/install-guide/index.html#downloadnccl
• https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
• https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy


## Inspiration

The inspiration for this cmake port came from:
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