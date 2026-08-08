#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""Multimem handles, multimem window addresses and multimem barriers.

A multimem (multicast) mapping lets one store reach every LSA peer's copy
of a window. This requests one for the LSA team and shows the two places
the handle comes from — embedded in the devcomm, or returned to the host
by ``create_dev_comm`` — and the three places it is consumed.

The CuTeDSL layer exposes multimem *addresses*; the multimem load-reduce
and store instructions themselves are PTX and are not wrapped here.

Multimem is an intra-node mapping, so run on a single node::

    mpirun -n <gpus-on-node> python 03_multimem.py
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



NUM_ELEMS = 1024
THREADS = 128
BARRIER_INDEX = 0


@cute.kernel
def multimem_kernel(
    dev_comm: nccl_cute.DevComm,
    win: nccl_cute.Window,
    host_mm: nccl_cute.MultimemHandle,
):
    """Report multimem addresses and run a multimem-arrive LSA barrier.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        win: registered window covered by the multimem mapping.
        host_mm: the LSA team's multimem handle, passed from the host.
    """
    tidx, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.cta()

    # Two sources for the same mapping: the devcomm's embedded copy, and the
    # handle the host got back from create_dev_comm.
    embedded_mm = dev_comm.lsa_multimem

    if 0 == tidx:
        embedded_mc = cute.make_ptr(cutlass.Int8, embedded_mm.mc_base_pointer).toint()
        host_mc = cute.make_ptr(cutlass.Int8, host_mm.mc_base_pointer).toint()
        cute.printf(f"mc_base embedded={embedded_mc} host={host_mc}")

        # multimem_pointer takes an explicit handle, so it addresses non-LSA
        # teams too; lsa_multimem_pointer is the LSA shorthand.
        via_handle = cute.make_ptr(cutlass.Int8, win.multimem_pointer(0, host_mm)).toint()
        via_lsa = cute.make_ptr(cutlass.Int8, win.lsa_multimem_pointer(0, dev_comm)).toint()
        unicast = cute.make_ptr(cutlass.Int8, win.local_pointer(0)).toint()
        cute.printf(f"window mm(handle)={via_handle} mm(lsa)={via_lsa} unicast={unicast}")

    # multimem=True publishes the barrier flag with one multicast store
    # instead of one per peer; mm_handle must cover the barrier's team.
    bar = nccl_cute.lsa_default(
        coop, dev_comm, index=BARRIER_INDEX, multimem=True, mm_handle=embedded_mm)
    bar.sync(coop, nccl_cute.MemoryOrder.ACQ_REL)

    # Same barrier, with team and handle named by the caller.
    explicit = nccl_cute.lsa_session(
        coop, dev_comm, dev_comm.team_lsa, dev_comm.lsa_barrier,
        index=BARRIER_INDEX, multimem=True, mm_handle=host_mm)
    explicit.sync(coop, nccl_cute.MemoryOrder.ACQ_REL)

    if 0 == tidx:
        cute.printf(f"rank {dev_comm.rank}: multimem barriers passed")


@cute.jit
def multimem_demo(
    dev_comm: nccl_cute.DevComm,
    win: nccl_cute.Window,
    host_mm: nccl_cute.MultimemHandle,
):
    """Launch :func:`multimem_kernel` on a single CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        win: registered window covered by the multimem mapping.
        host_mm: the LSA team's multimem handle.
    """
    multimem_kernel(dev_comm, win, host_mm).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


def main():
    """Request an LSA multimem mapping and exercise it from a kernel.

    Returns:
        Exit code; 0 on success, 1 if multimem is unavailable.
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

    # Requesting multimem without multimem_support fails devcomm creation.
    if not nccl_comm.device_api_support or not nccl_comm.multimem_support:
        if rank == root:
            print("WARNING: no multicast support on this system, so no "
                  "multimem mapping; nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    buf = nccl.cupy.empty(NUM_ELEMS, dtype='float32')
    buf[:] = float(rank + 1)
    device.sync()
    win_resource = nccl_comm.register_window(buf)

    # lsa_multimem populates DevComm.lsa_multimem; the TeamRequirement is what
    # makes the handle retrievable host-side.
    reqs = nccl.NCCLDevCommRequirements(
        lsa_multimem=True,
        lsa_barrier_count=1,
        teams=(nccl.TeamRequirement(team=nccl_comm.team_lsa, multimem=True),),
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)

    mm_handle = dev_comm_resource.multimem_handle(nccl_comm.team_lsa)
    print(f"[rank {rank}] host mc_base_pointer=0x{mm_handle.mc_base_pointer:x}")

    multimem_demo(
        nccl_cute.DevComm(dev_comm_resource),
        nccl_cute.Window(win_resource),
        nccl_cute.MultimemHandle(mm_handle),
    )
    device.sync()

    print(f"[rank {rank}] [SUCCESS] multimem addresses resolved and barriers completed")


    dev_comm_resource.close()
    win_resource.close()
    nccl_comm.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
