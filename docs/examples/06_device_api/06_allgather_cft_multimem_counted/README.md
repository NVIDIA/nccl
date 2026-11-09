<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# NCCL Example: Device API AllGather with CFT multimem counted

This example shows a simple allgather implemented inside a CUDA kernel using
CFT multimem counted puts and wait.

Each rank owns one contiguous input slice. The kernel multicasts that slice to
all ranks, including itself, with `ncclCft::putMultimemCounted`, then waits on
one shared counter for `count * sizeof(float) * nRanks` bytes.

## Build

```shell
cd c
make [MPI=1] [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

```shell
./allgather_cft_multimem_counted
```

or with MPI:

```shell
mpirun -np <num_processes> ./allgather_cft_multimem_counted
```

The example requires CFT multimem-capable hardware and CUDA Toolkit support for
CFT fabric multimem instructions.
