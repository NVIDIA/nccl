<!--
  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

  See LICENSE.txt for more license information
-->

# Python (nccl4py + mpi4py): One Device per Process (MPI)

This is the nccl4py implementation using `mpi4py` for process management and
`nccl.core` for GPU communication.

## Run

From this directory:

```shell
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-cu12.txt  # or: requirements-cu13.txt
mpirun -np <num_processes> python one_device_per_process_mpi.py
```

## Code walk-through

- **MPI initialization**: `mpi4py.MPI.COMM_WORLD` provides rank, size, and broadcast.
- **Local rank detection**: `MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)` determines
  which GPU each process on a node should use (matches C's `MPI_Comm_split_type`).
- **Unique ID distribution**: Rank 0 calls `nccl.get_unique_id()` and broadcasts the result
  via `comm_world.bcast()`. The `UniqueId` object is picklable and can be broadcast directly.
- **Communicator init**: `nccl.Communicator.init(nranks=mpi_size, rank=mpi_rank, unique_id=unique_id)`
  (matches C's `ncclCommInitRank`).
- **Cleanup**: `comm.finalize()` + `comm.destroy()` releases this rank's communicator.
