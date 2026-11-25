#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""Every barrier session type, built both ways.

LSA (intra-node), GIN (inter-node) and hybrid barriers. Named constructors
select built-in teams and barrier resources; explicit constructors take
teams and host-requested handles. This runs both forms of all three.

Needs a GIN-capable network; two nodes exercise it for real::

    mpirun -n 2 -N 1 python 05_barriers.py
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


THREADS = 128
INDEX = 0
N_EXTRA_BARRIERS = 2


@cute.kernel
def barriers_kernel(
    dev_comm: nccl_cute.DevComm,
    lsa_handle: nccl_cute.LsaBarrierHandle,
    gin_handle: nccl_cute.GinBarrierHandle,
):
    """Run each barrier session type with built-in and explicit resources.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        lsa_handle: separately requested LSA barrier handle.
        gin_handle: separately requested GIN barrier handle.
    """
    tidx, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.Coop.cta()
    gin = dev_comm.gin()

    # === LSA ===

    # Explicit form on the separately requested handle. Only the LSA session
    # exposes arrive and wait as separate phases.
    lsa_explicit = nccl_cute.LsaBarrierSession(
        coop, dev_comm, dev_comm.team_lsa, lsa_handle, index=INDEX
    )
    lsa_explicit.arrive()
    lsa_explicit.wait()
    lsa_explicit.close()

    # Convenience form selects the built-in LSA team and barrier.
    lsa_default = nccl_cute.LsaBarrierSession.lsa(coop, dev_comm, index=INDEX)
    lsa_default.sync()
    lsa_default.close()

    # === GIN ===

    # The fence level says what the barrier drains besides
    # synchronizing: NONE nothing, PUT inbound puts, GET this rank's gets.

    # Explicit form: a team and its host-requested GIN barrier handle.
    gin_explicit = nccl_cute.GinBarrierSession(
        coop, gin, dev_comm.team_world, gin_handle, index=INDEX
    )
    gin_explicit.sync(fence=nccl_cute.GinFenceLevel.NONE)
    gin_explicit.close()

    gin_rail = nccl_cute.GinBarrierSession.rail(coop, gin, index=INDEX)
    gin_rail.sync()
    gin_rail.close()

    gin_world = nccl_cute.GinBarrierSession.world(coop, gin, index=INDEX)
    gin_world.sync(fence=nccl_cute.GinFenceLevel.PUT)
    gin_world.close()

    # Fence all contexts when puts span several contexts.
    gin_explicit_all_contexts = nccl_cute.GinBarrierSession(
        coop, dev_comm.gin_all_contexts(), dev_comm.team_world, gin_handle, index=INDEX
    )
    gin_explicit_all_contexts.sync()
    gin_explicit_all_contexts.close()

    gin_rail_all_contexts = nccl_cute.GinBarrierSession.rail(
        coop, dev_comm.gin_all_contexts(), index=INDEX
    )
    gin_rail_all_contexts.sync()
    gin_rail_all_contexts.close()

    gin_world_all_contexts = nccl_cute.GinBarrierSession.world(
        coop, dev_comm.gin_all_contexts(), index=INDEX
    )
    gin_world_all_contexts.sync(fence=nccl_cute.GinFenceLevel.PUT)
    gin_world_all_contexts.close()

    # === Hybrid ===

    # Both stages named explicitly, using the separately requested handles.
    hybrid_explicit = nccl_cute.HybridBarrierSession(
        coop,
        dev_comm.team_lsa,
        dev_comm.team_world,
        gin,
        lsa_handle,
        gin_handle,
        index=INDEX,
    )
    hybrid_explicit.sync(fence=nccl_cute.GinFenceLevel.NONE)
    hybrid_explicit.close()

    # Built-in world barrier selects its LSA/GIN algorithm based on the
    # connections, backend capabilities and requested fence.
    hybrid_world = nccl_cute.HybridBarrierSession.world(coop, gin, index=INDEX)
    hybrid_world.sync(fence=nccl_cute.GinFenceLevel.PUT)
    hybrid_world.close()

    # Select a single stage using the communicator's hybrid barrier resources.
    hybrid_lsa = nccl_cute.HybridBarrierSession.lsa(coop, dev_comm, index=INDEX)
    hybrid_lsa.sync()
    hybrid_lsa.close()

    hybrid_rail = nccl_cute.HybridBarrierSession.rail(coop, gin, index=INDEX)
    hybrid_rail.sync()
    hybrid_rail.close()

    if 0 == tidx:
        cute.printf(f"rank {dev_comm.rank}: barrier sessions completed")


@cute.jit
def barriers(
    dev_comm: nccl_cute.DevComm,
    lsa_handle: nccl_cute.LsaBarrierHandle,
    gin_handle: nccl_cute.GinBarrierHandle,
):
    """Launch :func:`barriers_kernel` on a single CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        lsa_handle: separately requested LSA barrier handle.
        gin_handle: separately requested GIN barrier handle.
    """
    barriers_kernel(dev_comm, lsa_handle, gin_handle).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


def main():
    """Request every barrier resource the kernel uses and run it.

    Returns:
        Exit code; 0 on success, 1 on a wrong rank count.
    """
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    root = 0

    if nranks < 2:
        if rank == root:
            print(f"\n[{NAME}] ERROR: needs at least 2 ranks, got {nranks}")
        return 1

    if rank == root:
        print(f"\n===== {NAME} =====", flush=True)

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    # The GIN and hybrid sections need a GIN transport.
    if not nccl_comm.device_api_support or nccl_comm.gin_type == nccl.NcclGinType.NONE:
        if rank == root:
            print(
                "WARNING: no GIN transport on this system "
                f"(gin_type={nccl_comm.gin_type.name}); nothing to run",
                flush=True,
            )
        nccl_comm.destroy()
        return 0

    # Counts request built-in barrier resources; `resources` requests
    # standalone ones, returned in resource_handles in order.
    reqs = nccl.NCCLDevCommRequirements(
        lsa_barrier_count=1,  # built-in LSA barrier
        rail_gin_barrier_count=1,  # built-in rail GIN barrier
        world_gin_barrier_count=1,  # built-in world GIN barrier
        barrier_count=1,  # built-in world/hybrid barrier resources
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        resources=(
            nccl.LsaBarrierRequirement(team=nccl_comm.team_lsa, n_barriers=N_EXTRA_BARRIERS),
            nccl.GinBarrierRequirement(team=nccl_comm.team_world, n_barriers=N_EXTRA_BARRIERS),
        ),
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)

    lsa_handle, gin_handle = dev_comm_resource.resource_handles
    if rank == root:
        print(
            f"host: lsa buf_handle={lsa_handle.buf_handle} "
            f"n_barriers={lsa_handle.n_barriers}, gin signal0={gin_handle.signal0}"
        )

    barriers(
        nccl_cute.runtime.to_cute(dev_comm_resource),
        nccl_cute.runtime.to_cute(lsa_handle),
        nccl_cute.runtime.to_cute(gin_handle),
    )
    device.sync()
    comm_mpi.Barrier()

    print(f"[rank {rank}] [SUCCESS] LSA, GIN and hybrid barriers completed")

    dev_comm_resource.close()
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
