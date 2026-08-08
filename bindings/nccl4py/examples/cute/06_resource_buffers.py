#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""Resource-buffer address translation.

Barriers and low-latency all-to-all resources live in a buffer NCCL
allocates inside the device communicator, not in a user window. The
``buf_handle`` field of a resource handle names a slice of it, and the
``DevComm.resource_buffer_*`` accessors translate that handle into an
address, once per addressing mode.

The buffer belongs to NCCL: this example only reads through the
translated addresses, to show that the local, LSA and peer translations
agree for the calling rank. Writing into a live barrier's storage
corrupts it.

Multimem is intra-node, so run on a single node::

    mpirun -n <gpus-on-node> python 06_resource_buffers.py
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

import cutlass
import cutlass.cute as cute
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


NAME = os.path.basename(__file__)



THREADS = 128
INDEX = 0


@cute.kernel
def resource_buffers_kernel(
    dev_comm: nccl_cute.DevComm,
    lsa_handle: nccl_cute.LsaBarrierHandle,
):
    """Translate one resource handle through every addressing mode.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        lsa_handle: LSA barrier handle whose ``buf_handle`` is translated.
    """
    tidx, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.cta()

    # An offset into the devcomm's resource window, assigned at creation.
    handle = lsa_handle.buf_handle
    me = dev_comm.lsa_rank

    if 0 == tidx:
        local = cute.make_ptr(
            cutlass.Int8, dev_comm.resource_buffer_local_pointer(handle)).toint()
        # Same slice as seen on an LSA peer; peer == me gives the local one.
        lsa = cute.make_ptr(
            cutlass.Int8, dev_comm.resource_buffer_lsa_pointer(handle, me)).toint()
        # The team-relative form: addresses within any team, not just LSA.
        peer = cute.make_ptr(
            cutlass.Int8,
            dev_comm.resource_buffer_peer_pointer(handle, dev_comm.team_lsa, me)).toint()
        # Multicast addresses: one store reaches every peer's copy.
        mm = cute.make_ptr(
            cutlass.Int8,
            dev_comm.resource_buffer_multimem_pointer(handle, dev_comm.lsa_multimem)).toint()
        lsa_mm = cute.make_ptr(
            cutlass.Int8,
            dev_comm.resource_buffer_lsa_multimem_pointer(handle)).toint()

        cute.printf(f"resource buffer local={local} lsa={lsa} peer={peer}")
        cute.printf(f"resource buffer multimem={mm} lsa_multimem={lsa_mm}")
        cute.printf(f"local == lsa(self): {local == lsa}, local == peer(self): {local == peer}")

    # Live barrier storage: reading it around a barrier shows NCCL updating
    # it. Never write through it.
    state = cute.make_tensor(
        cute.make_ptr(cutlass.Uint64, dev_comm.resource_buffer_local_pointer(handle)),
        cute.make_layout(1))
    before = state[0]

    nccl_cute.lsa_session(
        coop, dev_comm, dev_comm.team_lsa, lsa_handle, index=INDEX
    ).sync(coop, nccl_cute.MemoryOrder.ACQ_REL)

    if 0 == tidx:
        cute.printf(f"barrier state before={before} after={state[0]}")


@cute.jit
def resource_buffers(
    dev_comm: nccl_cute.DevComm,
    lsa_handle: nccl_cute.LsaBarrierHandle,
):
    """Launch :func:`resource_buffers_kernel` on a single CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        lsa_handle: LSA barrier handle whose ``buf_handle`` is translated.
    """
    resource_buffers_kernel(dev_comm, lsa_handle).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


def main():
    """Request an LSA barrier resource and translate its handle.

    Returns:
        Exit code; 0 on success, 1 if the topology or multimem is missing.
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

    if nccl_comm.team_lsa.n_ranks != nranks:
        if rank == root:
            print(f"\n[{NAME}] ERROR: all {nranks} ranks must be LSA peers "
                  f"(lsa team has {nccl_comm.team_lsa.n_ranks}); run on one node", flush=True)
        nccl_comm.destroy()
        return 1

    # The two multimem translations need a multicast mapping, which is also
    # what lets the team requirement below succeed.
    if not nccl_comm.device_api_support or not nccl_comm.multimem_support:
        if rank == root:
            print("WARNING: no multicast support on this system, so the "
                  "two multimem translations cannot run; nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    # lsa_multimem backs the multimem translations; the LSA barrier
    # requirement produces the resource handle itself.
    reqs = nccl.NCCLDevCommRequirements(
        lsa_multimem=True,
        teams=(nccl.TeamRequirement(team=nccl_comm.team_lsa, multimem=True),),
        resources=(
            nccl.LsaBarrierRequirement(team=nccl_comm.team_lsa, n_barriers=1),
        ),
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)
    (lsa_handle,) = dev_comm_resource.resource_handles

    if rank == root:
        print(f"host: buf_handle={lsa_handle.buf_handle} "
              f"n_barriers={lsa_handle.n_barriers}")

    resource_buffers(
        nccl_cute.DevComm(dev_comm_resource),
        nccl_cute.LsaBarrierHandle(lsa_handle),
    )
    device.sync()
    comm_mpi.Barrier()

    print(f"[rank {rank}] [SUCCESS] resource handle translated in all five modes")


    dev_comm_resource.close()
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
