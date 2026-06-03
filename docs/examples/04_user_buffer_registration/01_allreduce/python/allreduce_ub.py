# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information

"""
NCCL Example (nccl4py + mpi4py): User Buffer Registration AllReduce
===================================================================

Demonstrates registering user buffers for zero-copy collective operations,
using one GPU per MPI process.

Usage:
```shell
mpirun -np <num_processes> python allreduce_ub.py
```
"""

import sys

import nccl.core as nccl
import cupy as cp
import numpy as np
from mpi4py import MPI

EXAMPLE_NAME = "User Buffer Registration AllReduce"


def get_local_rank(comm: MPI.Comm) -> int:
    """Determine the local rank of this process on its physical node.

    Uses MPI_Comm_split_type with MPI_COMM_TYPE_SHARED to split ranks by
    shared-memory domain (i.e., by node), then returns the rank within
    the local communicator.  This rank is used as the GPU device index.
    """
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    node_comm.Free()
    return local_rank


def main() -> int:
    # =========================================================================
    # STEP 1: Initialize MPI and determine process layout
    # =========================================================================

    comm_world = MPI.COMM_WORLD
    mpi_rank = comm_world.Get_rank()
    mpi_size = comm_world.Get_size()

    if mpi_rank == 0:
        print(f"Starting {EXAMPLE_NAME} example with {mpi_size} processes")

    local_rank = get_local_rank(comm_world)

    print(f"  MPI initialized - Process {mpi_rank} of {mpi_size} total processes")

    print()

    # =========================================================================
    # STEP 2: Detect Available GPUs
    # =========================================================================

    num_gpus = cp.cuda.runtime.getDeviceCount()

    if num_gpus < 1:
        print("ERROR: No CUDA devices found on this node!", file=sys.stderr)
        return 1

    if local_rank >= num_gpus:
        print(
            f"ERROR: Process {mpi_rank} needs GPU {local_rank} "
            f"but only {num_gpus} devices available",
            file=sys.stderr,
        )
        return 1

    cp.cuda.Device(local_rank).use()

    print(f"  MPI rank {mpi_rank} assigned to CUDA device {local_rank}")

    print()

    # =========================================================================
    # STEP 3: Initialize NCCL Communicator
    # =========================================================================

    unique_id = nccl.get_unique_id() if mpi_rank == 0 else None
    if mpi_rank == 0:
        print("Rank 0 generated NCCL unique ID for all processes")

    unique_id = comm_world.bcast(unique_id, root=0)

    # Each process joins the distributed communicator on its assigned GPU.
    nccl_comm = nccl.Communicator.init(nranks=mpi_size, rank=mpi_rank, unique_id=unique_id)
    print(f"  Rank {mpi_rank} created NCCL communicator")

    print()

    # =========================================================================
    # STEP 4: Allocate and register user buffers
    # =========================================================================

    count = 1024 * 1024  # 1M floats
    dtype = cp.float32

    # Allocate NCCL-backed arrays so they can be registered for zero-copy.
    send_buf = nccl.cupy.empty(count, dtype=dtype)
    recv_buf = nccl.cupy.empty(count, dtype=dtype)

    # register_buffer pre-registers each rank's buffers (matches ncclCommRegister).
    send_handle = nccl_comm.register_buffer(send_buf)
    recv_handle = nccl_comm.register_buffer(recv_buf)
    print(f"  Rank {mpi_rank} registered user buffers")

    print()

    # =========================================================================
    # STEP 5: Initialize data and perform AllReduce
    # =========================================================================

    # Each rank contributes its rank value.
    send_buf.fill(float(mpi_rank))

    nccl_comm.allreduce(send_buf, recv_buf, nccl.SUM)
    cp.cuda.Device(local_rank).synchronize()

    print(f"  Rank {mpi_rank} completed AllReduce")

    print()

    # =========================================================================
    # STEP 6: Verify results
    # =========================================================================

    expected = float(mpi_size * (mpi_size - 1)) / 2  # 0 + 1 + ... + (nranks-1)
    got = float(cp.asnumpy(recv_buf[:1])[0])
    local_ok = bool(np.isclose(got, expected, atol=0.001))
    all_ok = comm_world.allreduce(local_ok, op=MPI.LAND)

    print(f"  MPI rank {mpi_rank} verification - Expected: {expected:.1f}, Got: {got:.1f}")

    comm_world.Barrier()

    print()

    # =========================================================================
    # STEP 7: Clean shutdown and resource cleanup
    # =========================================================================

    print("Cleaning up resources")

    # Registered buffers are automatically deregistered when the communicator
    # is destroyed.
    nccl_comm.finalize()
    nccl_comm.destroy()
    print(f"  Rank {mpi_rank} destroyed NCCL communicator")

    if mpi_rank == 0 and all_ok:
        print("=============================================================")
        print(f"SUCCESS: {EXAMPLE_NAME} example completed!")
        print("=============================================================")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
