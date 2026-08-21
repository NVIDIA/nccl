<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API AllGather with CFT counted

This example shows a simple allgather implemented inside a CUDA kernel using
CFT counted puts and wait.

Each rank owns one contiguous input slice. The kernel copies the local slice to
the local output slot, sends the same slice to every other rank with
`ncclCft::putCounted`, and then waits for the expected counted bytes from every
peer selected by a bit mask.

## Build

```shell
cd c
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

```shell
./allgather_cft_counted
```

or with MPI:

```shell
mpirun -np <num_processes> ./allgather_cft_counted
```

The example requires CFT-capable hardware and CUDA Toolkit support for CFT
fabric instructions.
