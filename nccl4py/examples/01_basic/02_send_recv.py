#!/usr/bin/env python3
"""
Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

NCCL4Py Basic Example: Send/Recv
==================================

Simple point-to-point communication example showing two ranks exchanging data.
Demonstrates the use of group() to avoid deadlocks.

USAGE:
mpirun -np 2 python 02_send_recv.py
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

    if nranks != 2:
        if rank == root:
            print("ERROR: This example requires exactly 2 ranks")
            print("Usage: mpirun -np 2 python 02_send_recv.py")
        sys.exit(1)

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
        print("Running Send/Recv between 2 ranks...")

    # Create tensors
    send_data = torch.tensor([float(100 + rank)], dtype=torch.float32, device=device)
    recv_data = torch.zeros(1, dtype=torch.float32, device=device)

    other_rank = 1 - rank  # Rank 0 <-> Rank 1

    # Exchange data using group() to avoid deadlock
    # IMPORTANT: Using group() allows send() and recv() to be called in any order.
    # Without group(), you must ensure send() and recv() are ordered carefully to avoid deadlock.
    # For example:
    #   - All ranks call send() first, then all call recv(), OR
    #   - Even ranks send then recv, odd ranks recv then send

    # [NCCL4Py] Use group() to avoid deadlock
    with nccl.group():
        # [NCCL4Py] Send data to the other rank
        nccl_comm.send(send_data, peer=other_rank)
        # [NCCL4Py] Receive data from the other rank
        nccl_comm.recv(recv_data, peer=other_rank)

    torch.cuda.synchronize()

    # Verify result
    expected = float(100 + (1 - rank))
    actual = float(recv_data[0].item())

    print(f"Rank {rank}: Sent {100 + rank:.0f}, Received {actual:.0f} (expected {expected:.0f})")

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
