#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NCCL4Py Basic Example: AllReduce
==================================

The simplest possible example showing how to use NCCL4Py with AllReduce.
Each rank contributes its rank number, and all ranks receive the sum.

USAGE:
mpirun -np 4 python 01_allreduce.py
"""

import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. Install with: pip install mpi4py")
    sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

import nccl.core as nccl


def main():
    # Initialize MPI
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()

    # Set rank 0 as the root
    root = 0

    # Assign GPU to each process
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    # [NCCL4Py] Generate unique ID on the root rank
    unique_id = nccl.get_unique_id() if rank == root else None

    # Broadcast unique ID to all ranks
    unique_id = comm_mpi.bcast(unique_id, root=root)

    # [NCCL4Py] Initialize NCCL communicator
    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    if rank == root:
        print(f"Running AllReduce with {nranks} ranks...")

    # Create PyTorch tensor with rank value
    data = torch.tensor([float(rank)], dtype=torch.float32, device=device)

    # [NCCL4Py] AllReduce: Sum all rank values
    nccl_comm.reduce(data, data, nccl.SUM)

    torch.cuda.synchronize()

    # Verify result
    expected = float(nranks * (nranks - 1) // 2)
    actual = float(data[0].item())

    print(f"Rank {rank}: AllReduce result = {actual:.0f} (expected {expected:.0f})")

    # [NCCL4Py] Destroy NCCL communicator (collective call)
    nccl_comm.destroy()

    if rank == root:
        if actual == expected:
            print("SUCCESS!")
        else:
            print("FAILED!")

    return 0 if actual == expected else 1


if __name__ == "__main__":
    sys.exit(main())
