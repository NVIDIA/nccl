# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# See LICENSE.txt for more license information

"""
NCCL Example (nccl4py + mpi4py): One Device per Process (MPI)
==============================================================

Demonstrates how to initialize NCCL communicators in a single process.

Usage:
```shell
mpirun -np <num_processes> python one_device_per_process_mpi.py
```
"""

import sys

import nccl.core as nccl
from cuda.core import Device, system
from mpi4py import MPI

EXAMPLE_NAME = "one_devices_per_process"


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
        print(f"Starting NCCL communicator lifecycle example with {mpi_size} processes")

    local_rank = get_local_rank(comm_world)

    print(f"  MPI initialized - Process {mpi_rank} of {mpi_size} total processes")

    print()

    # =========================================================================
    # STEP 2: Detect Available GPUs
    # =========================================================================

    num_gpus = system.get_num_devices()

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

    device = Device(local_rank)
    device.set_current()

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
    # STEP 4: Verify communicator setup
    # =========================================================================

    comm_rank = nccl_comm.rank
    comm_size = nccl_comm.nranks
    comm_device = nccl_comm.device.device_id

    print(f"  MPI rank {mpi_rank} -> NCCL rank {comm_rank}/{comm_size} on GPU device {comm_device}")

    comm_world.Barrier()

    print()

    # =========================================================================
    # STEP 5: Clean shutdown and resource cleanup
    # =========================================================================
    
    print("Cleaning up resources")

    if mpi_rank == 0:
        print("\nAll communicators initialized successfully! Beginning cleanup...")

    nccl_comm.finalize()
    nccl_comm.destroy()
    print(f"  Rank {mpi_rank} destroyed NCCL communicator")

    if mpi_rank == 0:
        print("=============================================================")
        print(f"SUCCESS: {EXAMPLE_NAME} example completed!")
        print("=============================================================")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
