<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# C: RAS Fault Detection

C implementation of the NCCL RAS fault-detection example.  It runs a live
AllReduce workload, injects a configurable fault, and stays queryable so you
can watch the RAS report change while it runs.

See [../README.md](../README.md) for the full walkthrough — the fault modes,
the RAS query workflow (`nc` / `ncclras`), and the expected output at each
stage.

This example **requires MPI** and is meant for **interactive** use: launch it,
then query RAS yourself.  It is intentionally not driven by automated test
runners.

## Build

From this directory:

```shell
make [MPI_HOME=<path-to-mpi>] [NCCL_HOME=<path-to-nccl>] [CUDA_HOME=<path-to-cuda>]
```

## Run

```shell
mpirun -np 4 ./ras_fault_detection --type exit
```

Options: `--type exit|suspend|sleep`, `--ranks R,...`, `--wait S`,
`--observe S`.  While it runs, query RAS on the job's node (find it with
`squeue --me`; port `28028` by default):

```shell
echo "verbose status" | nc <hostname> 28028
```

See [../README.md](../README.md#observing) for the full set of query options
(`ncclras -v`, live event stream, JSON).
