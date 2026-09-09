#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""Cooperative groups, device-communicator fields and teams.

The starting point for the CuTeDSL device API: no windows, no data
movement, nothing that needs more than one GPU. It builds every kind of
coop, reads the devcomm's scalar fields, and prints the three teams and
the rank translations between them.

Runs on any rank count, one node::

    mpirun -n 2 python 01_coop_and_teams.py
"""

import os
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. Install with: pip install mpi4py", flush=True)
    sys.exit(1)

try:
    from cuda.core import Device, system
except ImportError:
    print("ERROR: cuda.core required. Install with: pip install cuda-core", flush=True)
    sys.exit(1)

import cutlass.cute as cute
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


NAME = os.path.basename(__file__)



N_WARPS = 2
THREADS = 32 * N_WARPS


@cute.kernel
def coop_and_teams_kernel(dev_comm: nccl_cute.DevComm):
    """Print coop shapes and team identities from a two-warp CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
    """
    tidx, _, _ = cute.arch.thread_idx()

    # Each coop allocates its own ncclCoopAny on the stack; they are cheap.
    block = nccl_cute.cta()
    lane_group = nccl_cute.warp()
    single = nccl_cute.thread()
    even_lanes = nccl_cute.lanes(0x55555555)
    # id distinguishes concurrent spans that share a CTA; 0..15.
    span = nccl_cute.warp_span(warp0=0, n_warps=N_WARPS, id=0)

    if 0 == tidx:
        cute.printf(
            f"coop cta        rank={block.thread_rank} size={block.size} threads={block.num_threads}")
        cute.printf(
            f"coop warp       rank={lane_group.thread_rank} size={lane_group.size} threads={lane_group.num_threads}")
        cute.printf(
            f"coop thread     rank={single.thread_rank} size={single.size} threads={single.num_threads}")
        cute.printf(
            f"coop lanes      rank={even_lanes.thread_rank} size={even_lanes.size} threads={even_lanes.num_threads}")
        cute.printf(
            f"coop warp_span  rank={span.thread_rank} size={span.size} threads={span.num_threads}")

    # The coop-scoped barrier; on a CTA coop this is __syncthreads().
    block.sync()

    # Teams are addressing domains, not communicators: (nRanks, rank, stride)
    # over the world ranks.
    world = dev_comm.team_world
    lsa = dev_comm.team_lsa
    rail = dev_comm.team_rail

    if 0 == tidx:
        cute.printf(
            f"devcomm rank={dev_comm.rank}/{dev_comm.n_ranks} lsa={dev_comm.lsa_rank}/{dev_comm.lsa_size}")
        cute.printf(f"team world  rank={world.rank} nRanks={world.nRanks} stride={world.stride}")
        cute.printf(f"team lsa    rank={lsa.rank} nRanks={lsa.nRanks} stride={lsa.stride}")
        cute.printf(f"team rail   rank={rail.rank} nRanks={rail.nRanks} stride={rail.stride}")

        # Translate a rank expressed in one team to world or LSA numbering.
        cute.printf(f"lsa   rank 0 -> world rank {dev_comm.team_rank_to_world(lsa, 0)}")
        cute.printf(f"rail  rank 0 -> world rank {dev_comm.team_rank_to_world(rail, 0)}")
        # team_rank_to_lsa requires the rank to be local; a non-local rank
        # yields an out-of-range result rather than an error.
        cute.printf(f"world rank {dev_comm.rank} -> lsa rank {dev_comm.team_rank_to_lsa(world, dev_comm.rank)}")


@cute.jit
def coop_and_teams(dev_comm: nccl_cute.DevComm):
    """Launch :func:`coop_and_teams_kernel` on a single two-warp CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
    """
    coop_and_teams_kernel(dev_comm).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


def main():
    """Create a device communicator and print its coops and teams.

    Returns:
        Exit code; 0 on success.
    """
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    root = 0

    if rank == root:
        print(f"\n===== {NAME} =====", flush=True)

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    # Without device API support no devcomm can be created at all.
    if not nccl_comm.device_api_support:
        if rank == root:
            print("WARNING: this platform has no device API support; "
                  "nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    # No windows, barriers or GIN here, so default requirements suffice.
    dev_comm_resource = nccl_comm.create_dev_comm(
        requirements=nccl.NCCLDevCommRequirements())
    assert dev_comm_resource.is_valid

    coop_and_teams(nccl_cute.DevComm(dev_comm_resource))
    device.sync()


    dev_comm_resource.close()
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
