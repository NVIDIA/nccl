<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

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

Python variants are provided where `nccl4py` exposes the required host-side
APIs. Each Python example includes `requirements-cu12.txt` and
`requirements-cu13.txt` in its `python/` directory. Examples that depend on
pthread-specific coordination or CUDA device-side programming remain C/CUDA-only.

## Basic Examples
We start with the most basic NCCL operations. These examples are intended as
approachable starting points for new users coming up to speed with NCCL for GPU
communication. Each example is organized as a small directory, with C sources
under `c/` and, where available, a Python variant under `python/`. Language-
specific build and run instructions are documented in those directories.

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
bootstrap mechanisms. Depending on the example, execution may use a single
process managing multiple GPUs, one thread per GPU, or one process per GPU with
MPI. Users can set `MPI=1` to build an MPI parallel version which can run
across multiple compute nodes. Users can optionally provide a valid MPI
installation under `MPI_HOME`.

To ease the readability of these examples we have moved shared bootstrap and
broadcast helper code to the [common](common/) directory. Build and run commands
for each example are documented in the corresponding category README and in each
example's `c/README.md` or, where available, `python/README.md`.

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

### [Fused Computation & Communication](07_kernel_fusion/)

This section demonstrates fusing computation and communication within GPU kernels
using NCCL's Device API. Examples include distributed RMSNorm implementations
that perform three phases (reduction, computation, distribution) in a single
kernel launch without CPU involvement. Implementations are provided for
**Load Store Accessible (LSA)**, **Multimem** (SM 9.0+), **GIN**
(GPU-Initiated Networking), and **hybrid GIN/LSA** for optimal multi-node
communication.


### [RAS (Reliability, Availability, Serviceability)](08_ras/)

RAS is NCCL's built-in health monitoring subsystem. It starts automatically
when a communicator is created and maintains a cluster-wide view of every
process, communicator, and GPU in the job. The example here injects a process
fault and pauses at key moments so you can query the RAS daemon yourself
(via `nc` or `ncclras`) and watch the full `RUNNING → INCOMPLETE → DEAD`
progression in real time. Both fault modes are covered: fast detection via
socket close (`--type exit`) and keep-alive timeout detection via `SIGSTOP`
(`--type suspend`).

## Prerequisites

- The same prerequisites as building NCCL from source.
- Users can optionally add `MPI_HOME` for an MPI library in a non-standard
  location.
- Python 3 is required if you want to run the `python/` variants.

## Build Steps
The examples can be built while building the NCCL library from source. Users can
choose to build the examples with MPI support (`MPI=1`).

```shell
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make -j examples [MPI=1]
```

or, if NCCL has already been built, the user can optionally add a non-standard
NCCL installation location:

```shell
cd docs/examples
make NCCL_HOME=<path-to-nccl> [MPI=1]
```

Python examples are run from the corresponding example's `python/` directory.
Each Python README includes exact setup and run instructions. In most cases the
workflow is:

```shell
cd <example>/python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt   # or requirements-cu13.txt
python <example>.py
```

To install all Python dependencies at once and run any example without
per-directory setup, use the top-level requirements file:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r docs/examples/requirements-cu12.txt   # or requirements-cu13.txt
```

MPI-enabled Python examples additionally require `mpi4py` (already included in
the top-level requirements).

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
