# Symmetric Memory Preview branch

This is a preview branch for symmetric memory registration, along with optimized kernels for symmetric memory.

## New API

This adds new symmetric registration and de-registration APIs. Their usage is similar to `ncclCommRegister`,
except those are collective calls, i.e. they need to be called by all ranks with the same size.

```
ncclResult_t  ncclCommSymmetricRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t  ncclCommSymmetricDeregister(const ncclComm_t comm, void* handle);
```

Memory passed to `ncclCommsymmetricRegister` needs to be allocated with `cuMem*` APIs. Using `ncclMemAlloc`
to allocate that memory is recommended. Memory allocated through `cudaMalloc` is not supported.

## Collective calls

Collective operations can be called on symmetrically registered memory to improve their performance,
leveraging the symmetric nature of memory.
This implies extra constraints for collective calls, as each rank has to call collective operations with
aligned offsets between ranks.

## Current limitations

The current implementation only provides symmetric kernels optimized for NVLink operations. Network
operations are not yet implemented and communicators with a network dimension will therefore fallback to
non-symmetric kernels.

The code also requires sm70 or later.

## Running NCCL perf tests with symmetric memory

To use the new kernels, we need the performance tests to register memory using the new registration APIs.
Make sure to checkout the `v2.27_sym_memory` branch of the NCCL perf tests, and run with `-R 2` to enable
symmetric registration.

# NCCL

Optimized primitives for inter-GPU communication.

## Introduction

NCCL (pronounced "Nickel") is a stand-alone library of standard communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, as well as any send/receive based communication pattern. It has been optimized to achieve high bandwidth on platforms using PCIe, NVLink, NVswitch, as well as networking using InfiniBand Verbs or TCP/IP sockets. NCCL supports an arbitrary number of GPUs installed in a single node or across multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

For more information on NCCL usage, please refer to the [NCCL documentation](https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html).

## Build

Note: the official and tested builds of NCCL can be downloaded from: https://developer.nvidia.com/nccl. You can skip the following build steps if you choose to use the official builds.

To build the library :

```shell
$ cd nccl
$ make -j src.build
```

If CUDA is not installed in the default /usr/local/cuda path, you can define the CUDA path with :

```shell
$ make src.build CUDA_HOME=<path to cuda install>
```

NCCL will be compiled and installed in `build/` unless `BUILDDIR` is set.

By default, NCCL is compiled for all supported architectures. To accelerate the compilation and reduce the binary size, consider redefining `NVCC_GENCODE` (defined in `makefiles/common.mk`) to only include the architecture of the target platform :
```shell
$ make -j src.build NVCC_GENCODE="-gencode=arch=compute_70,code=sm_70"
```

## Install

To install NCCL on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package
$ make pkg.debian.build
$ ls build/pkg/deb/
```

RedHat/CentOS :
```shell
$ # Install tools to create rpm packages
$ sudo yum install rpm-build rpmdevtools
$ # Build NCCL rpm package
$ make pkg.redhat.build
$ ls build/pkg/rpm/
```

OS-agnostic tarball :
```shell
$ make pkg.txz.build
$ ls build/pkg/txz/
```

## Tests

Tests for NCCL are maintained separately at https://github.com/nvidia/nccl-tests.

```shell
$ git clone https://github.com/NVIDIA/nccl-tests.git
$ cd nccl-tests
$ make
$ ./build/all_reduce_perf -b 8 -e 256M -f 2 -g <ngpus>
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
