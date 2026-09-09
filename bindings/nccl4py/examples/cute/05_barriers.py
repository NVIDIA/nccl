#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""Every barrier session type, built both ways.

LSA (intra-node), GIN (inter-node) and hybrid (LSA inner stage + GIN
outer stage). Each has a convenience factory that pulls the team and
handle out of the devcomm, and an explicit factory that takes them as
arguments; this runs both spellings of all three.

The factories that take a :class:`Gin` need NCCL 2.31.0, where the C
entry points behind them changed from taking ``ncclGin_C`` by value to
taking a pointer. Below that they are skipped — passing a pointer to the
older entry point corrupts memory instead of failing to link.
``GIN_ALL_CONTEXTS`` goes through a different, unchanged entry point and
always runs.

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


# Always available.

# Need the 2.31.0 pointer ABI for ncclGin_C.

# ncclGinBarrierSessionInit / ncclBarrierSessionInit took ncclGin_C by value up
# to 2.30.7 and take a pointer from 2.31.0. The bindings pass a pointer, which
# an older library reads as the struct's first field — silent corruption, so it
# must be guarded rather than caught. ncclGinBarrierSessionInitAllContexts is
# unchanged, so the GIN_ALL_CONTEXTS form works either way.
# get_version() reports the loaded libnccl, which is what the device
# bitcode is expected to match.
NCCL_VERSION = nccl.get_version().libnccl.version
NCCL_GIN_SESSION_TAKES_PTR = NCCL_VERSION.release >= (2, 31, 0)

THREADS = 128
INDEX = 0
N_EXTRA_BARRIERS = 2


@cute.kernel
def barriers_kernel(
    dev_comm: nccl_cute.DevComm,
    lsa_handle: nccl_cute.LsaBarrierHandle,
    gin_handle: nccl_cute.GinBarrierHandle,
):
    """Run each barrier session type in both its factory forms.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        lsa_handle: separately requested LSA barrier handle.
        gin_handle: separately requested GIN barrier handle.
    """
    tidx, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.cta()
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)

    if 0 == tidx:
        # The LSA handle names a slice of the devcomm's resource buffer; the
        # GIN handle names the first of its signals.
        cute.printf(
            f"requested lsa handle: buf_handle={lsa_handle.buf_handle} "
            f"n_barriers={lsa_handle.n_barriers}")
        cute.printf(f"requested gin handle: signal0={gin_handle.signal0}")
        cute.printf(
            f"embedded lsa handle:  buf_handle={dev_comm.lsa_barrier.buf_handle} "
            f"n_barriers={dev_comm.lsa_barrier.n_barriers}")

    # === LSA ===

    # Convenience form: team_lsa + dev_comm.lsa_barrier.
    nccl_cute.lsa_default(coop, dev_comm, index=INDEX).sync(
        coop, nccl_cute.MemoryOrder.ACQ_REL)

    # Explicit form on the separately requested handle. Only the LSA session
    # exposes arrive and wait as separate phases.
    explicit_lsa = nccl_cute.lsa_session(
        coop, dev_comm, dev_comm.team_lsa, lsa_handle, index=INDEX)
    explicit_lsa.arrive(coop, nccl_cute.MemoryOrder.RELEASE)
    explicit_lsa.wait(coop, nccl_cute.MemoryOrder.ACQUIRE)

    # === GIN ===

    # GIN_ALL_CONTEXTS fences every context on the comm rather than the one
    # `gin` is bound to — needed when puts span several, at a flush per
    # (context, peer). The fence level says what the barrier drains besides
    # synchronizing: NONE nothing, PUT inbound puts, GET this rank's gets.
    nccl_cute.world_gin(
        coop, nccl_cute.GIN_ALL_CONTEXTS, dev_comm, index=INDEX
    ).sync(coop, nccl_cute.MemoryOrder.ACQ_REL, nccl_cute.GinFenceLevel.PUT)

    if NCCL_GIN_SESSION_TAKES_PTR:
        nccl_cute.world_gin(coop, gin, dev_comm, index=INDEX).sync(
            coop, nccl_cute.MemoryOrder.ACQ_REL, nccl_cute.GinFenceLevel.PUT)

        nccl_cute.rail_gin(coop, gin, dev_comm, index=INDEX).sync(
            coop, nccl_cute.MemoryOrder.ACQ_REL,
            nccl_cute.GinFenceLevel.PUT | nccl_cute.GinFenceLevel.GET)

        # Explicit form: any team, any GIN barrier handle.
        nccl_cute.gin_session(
            coop, gin, dev_comm, dev_comm.team_world, gin_handle, index=INDEX
        ).sync(coop, nccl_cute.MemoryOrder.ACQ_REL, nccl_cute.GinFenceLevel.NONE)

        # === Hybrid ===

        # LSA within the node, GIN across nodes: one rank per node carries
        # the outer stage.
        nccl_cute.world_hybrid(coop, gin, dev_comm, index=INDEX).sync(
            coop, nccl_cute.MemoryOrder.ACQ_REL, nccl_cute.GinFenceLevel.PUT)

        # Both stages named explicitly; the handles come from the devcomm's
        # hybrid pair, sized by requirements.barrier_count.
        nccl_cute.hybrid_session(
            coop,
            dev_comm.team_lsa,
            dev_comm.team_world,
            gin,
            dev_comm.hybrid_lsa_barrier,
            dev_comm.hybrid_rail_gin_barrier,
            index=INDEX,
        ).sync(coop, nccl_cute.MemoryOrder.ACQ_REL, nccl_cute.GinFenceLevel.NONE)

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
        if not NCCL_GIN_SESSION_TAKES_PTR:
            print(f"WARNING: NCCL {NCCL_VERSION} passes "
                  "ncclGin_C by value; skipping world_gin, rail_gin, "
                  "gin_session, world_hybrid and hybrid_session", flush=True)

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    # The GIN and hybrid sections need a GIN transport.
    if not nccl_comm.device_api_support or nccl_comm.gin_type == nccl.NcclGinType.NONE:
        if rank == root:
            print("WARNING: no GIN transport on this system "
                  f"(gin_type={nccl_comm.gin_type.name}); nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    # The counts size the handles NCCL embeds in the devcomm; `resources`
    # asks for standalone ones, returned in resource_handles in order.
    reqs = nccl.NCCLDevCommRequirements(
        lsa_barrier_count=1,          # dev_comm.lsa_barrier
        rail_gin_barrier_count=1,     # dev_comm.rail_gin_barrier
        world_gin_barrier_count=1,    # dev_comm.world_gin_barrier
        barrier_count=1,              # dev_comm.hybrid_{lsa,rail_gin}_barrier
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        resources=(
            nccl.LsaBarrierRequirement(
                team=nccl_comm.team_lsa, n_barriers=N_EXTRA_BARRIERS),
            nccl.GinBarrierRequirement(
                team=nccl_comm.team_world, n_barriers=N_EXTRA_BARRIERS),
        ),
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)

    lsa_handle, gin_handle = dev_comm_resource.resource_handles
    if rank == root:
        print(f"host: lsa buf_handle={lsa_handle.buf_handle} "
              f"n_barriers={lsa_handle.n_barriers}, gin signal0={gin_handle.signal0}")

    barriers(
        nccl_cute.DevComm(dev_comm_resource),
        nccl_cute.LsaBarrierHandle(lsa_handle),
        nccl_cute.GinBarrierHandle(gin_handle),
    )
    device.sync()
    comm_mpi.Barrier()

    ran = ("LSA, GIN and hybrid barriers" if NCCL_GIN_SESSION_TAKES_PTR
           else "LSA barriers and the GIN_ALL_CONTEXTS barrier")
    print(f"[rank {rank}] [SUCCESS] {ran} completed")


    dev_comm_resource.close()
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
