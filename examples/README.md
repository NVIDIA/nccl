<!-- Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

See LICENSE.txt for license information -->

# NCCL Library Examples

Welcome to the NCCL examples directory. This collection of NCCL (NVIDIA
Collective Communications Library) examples is designed to teach developers how
to effectively use NCCL in their applications. The examples progress from basic
concepts to advanced usage patterns, with each example featuring a detailed
README file. The APIs and features covered here are far from the complete set of
what NCCL provides. The [NCCL
Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
includes a detailed description of NCCL features and APIs.

These examples showcase individual features but are not intended to maximize the
performance for an individual communication pattern. For a performance
implementation please refer to the
[nccl-tests](https://github.com/NVIDIA/nccl-tests/) GitHub repository.

## Basic Examples
We start with the most basic NCCL operations. All examples in this section are
self-contained, meaning you can copy-paste one file and compile it on its own.
The only dependencies are the NCCL library itself and, in the MPI case, an MPI
implementation. These templates tend to new users coming up to speed with NCCL
for GPU communication.

### [Communicators](01_communicators/)

This section teaches you how to create, test, and destroy a communicator. We
have provided 3 examples using a single thread, multiple threads, and multiple
processes. This section shows the different options of launching an NCCL
application.

### [Point 2 Point](02_point_to_point/)

This sample send/recv implementation uses point to point communication to
communicate in a simple ring pattern.

### [Collectives](03_collectives/)

This sample implementation shows the most basic NCCL collective communication
call.

## Advanced Features

These examples are intended for experienced users looking for best practices to
use a specific feature. For complete end-to-end templates please use the basic
examples.

Since NCCL does not include its own launcher, we have provided two popular
bootstrap mechanisms. By default these examples will be launched as separate
threads, one thread per GPU. Users can set `MPI=1` to build an MPI parallel
version which can run across multiple compute nodes. Users can optionally
provide a valid MPI installation under `MPI_HOME`.

Each example can be run individually. By default you can run each executable via
```
[NTHREADS=<number of threads>] ./<example_name>
```

If `NTHREADS` is unset, the examples will use the number of visible GPUs as
number threads. If the applications are built with `MPI` support, you can run
each executable as

```
mpirun -np <number of MPI ranks> ./<example_name>
```

To ease the readability of these examples we have moved the bootstrap and
broadcast part to the [common](common/) directory. Completely self-contained
examples are provided in the sections above.

### [User Buffer Registration](04_user_buffer_registration/)

User Buffer Registration eliminates the overhead of copying data between user
buffers and internal NCCL buffers. This folder provides sample implementation
using User Buffer Registration with common collectives.

### [Symmetric Memory Registration](05_symmetric_memory/)

Since 2.27, NCCL supports window registration, which allows users to register
local buffers into NCCL window and enables extreme low latency and high
bandwidth communication in NCCL. This folder provides sample implementation
using Symmetric Memory Registration with common collectives.

### [Device APIs](06_device_api/)

Device API enables GPU kernels to directly perform inter-GPU communication. This
enables applications to perform communication from within CUDA kernels and fuse
computation and communication, and fine-grained control over collective
implementation. This folder demonstrates how to implement collectives using
device-side kernels.


## Prerequisites

- The same prerequisites as building NCCL from source.
- Users can optionally add `MPI_HOME` for an MPI library in a non-standard
  location.

## Build Steps
The examples can be built while building the NCCL library from source. Users can
choose to build the examples with MPI support (`MPI=1`).

```
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j examples [MPI=1]
```

or, if NCCL has already been built, the user can optionally add a non-standard
NCCL installation location:

```
cd examples
make NCCL_HOME=<path-to-nccl> [MPI=1]
```
## Environment Variables

### Build Stage
Users can use these optional variables to choose which libraries are used to
build these examples:
- `NCCL_HOME=<path>`: Local base directory of a NCCL installation.
- `MPI` : [0,1] Build the examples with MPI support.
- `MPI_HOME=<path>` : Local base directory of a MPI installation.
- `CUDA_HOME=<path>` : Local base directory of a CUDA installation.

### Run Stage
- `NTHREADS=<n>`: Number of threads to create for the threaded examples.
  Defaults to number of visible GPUs.
- `CUDA_VISIBLE_DEVICES`: Comma delimited list of GPUs visible to the
  application.
- All other NCCL [environment
  variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
  apply.

## Supported OS
Linux

## Troubleshooting
Each example includes a /Common Issues and Solutions/ section for the individual
tests. For general runtime issues use the debug output enabled by setting
`NCCL_DEBUG=INFO` for detailed logging. The
[Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
section of the NCCL documentation also includes many helpful tips.
