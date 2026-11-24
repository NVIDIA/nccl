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
coop, reads the devcomm's scalar fields, and prints predefined teams and
the rank translations between them. It also demonstrates host Team
conversion and device-side construction.

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
def coop_and_teams_kernel(dev_comm: nccl_cute.DevComm, team: nccl_cute.Team):
    """Print coop shapes and team identities from a two-warp CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        team: World Team passed from the host.
    """
    tidx, _, _ = cute.arch.thread_idx()

    # Each Coop factory allocates its own thread-local storage.
    block = nccl_cute.Coop.cta()
    lane_group = nccl_cute.Coop.warp()
    single = nccl_cute.Coop.thread()
    even_lanes = nccl_cute.Coop.lanes(0x55555555)
    # IDs 0..14 distinguish concurrent spans that share a CTA.
    span = nccl_cute.Coop.warp_span(warp0=0, n_warps=N_WARPS, id=0)

    if 0 == tidx:
        cute.printf(f"coop cta        rank={block.thread_rank} threads={block.num_threads}")
        cute.printf(
            f"coop warp       rank={lane_group.thread_rank} threads={lane_group.num_threads}"
        )
        cute.printf(f"coop thread     rank={single.thread_rank} threads={single.num_threads}")
        cute.printf(
            f"coop lanes      rank={even_lanes.thread_rank} threads={even_lanes.num_threads}"
        )
        cute.printf(f"coop warp_span  rank={span.thread_rank} threads={span.num_threads}")

    # The coop-scoped barrier; on a CTA coop this is __syncthreads().
    block.sync()

    # Teams are addressing domains, not communicators: (n_ranks, rank, stride)
    # over the world ranks.
    world = dev_comm.team_world
    lsa = dev_comm.team_lsa
    rail = dev_comm.team_rail
    explicit_world = nccl_cute.Team(n_ranks=dev_comm.n_ranks, rank=dev_comm.rank, stride=1)

    if 0 == tidx:
        cute.printf(
            f"devcomm rank={dev_comm.rank}/{dev_comm.n_ranks} lsa={dev_comm.lsa_rank}/{dev_comm.lsa_size}"
        )
        cute.printf(f"team world  rank={world.rank} n_ranks={world.n_ranks} stride={world.stride}")
        cute.printf(f"team lsa    rank={lsa.rank} n_ranks={lsa.n_ranks} stride={lsa.stride}")
        cute.printf(f"team rail   rank={rail.rank} n_ranks={rail.n_ranks} stride={rail.stride}")
        cute.printf(f"team host   rank={team.rank} n_ranks={team.n_ranks} stride={team.stride}")
        cute.printf(
            f"team made   rank={explicit_world.rank} n_ranks={explicit_world.n_ranks} stride={explicit_world.stride}"
        )

        # Translate a rank expressed in one team to world or LSA numbering.
        cute.printf(f"lsa   rank 0 -> world rank {dev_comm.team_rank_to_world(lsa, 0)}")
        cute.printf(f"rail  rank 0 -> world rank {dev_comm.team_rank_to_world(rail, 0)}")
        # team_rank_to_lsa requires the rank to be local; a non-local rank
        # yields an out-of-range result rather than an error.
        cute.printf(
            f"world rank {dev_comm.rank} -> lsa rank {dev_comm.team_rank_to_lsa(world, dev_comm.rank)}"
        )


@cute.jit
def coop_and_teams(
    dev_comm: nccl_cute.DevComm,
    team: nccl_cute.Team,
):
    """Launch :func:`coop_and_teams_kernel` on a single two-warp CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        team: World Team passed from the host.
    """
    coop_and_teams_kernel(dev_comm, team).launch(
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
            print("WARNING: this platform has no device API support; nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    # No windows, barriers or GIN here, so default requirements suffice.
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=nccl.NCCLDevCommRequirements())
    assert dev_comm_resource.is_valid

    dev_comm = nccl_cute.runtime.to_cute(dev_comm_resource)
    # Host Teams require explicit conversion before entering JIT.
    team = nccl_cute.runtime.to_cute(nccl_comm.team_world)
    coop_and_teams(dev_comm, team)
    device.sync()

    dev_comm_resource.close()
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
