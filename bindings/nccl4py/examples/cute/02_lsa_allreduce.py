#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""LSA window pointers and LSA barriers: a device-side all-reduce.

Every rank stores ``rank + 1`` into its send window; each rank then reads
every LSA peer's send window directly over NVLink/peer-access and writes
the sum into its own receive window. Two LSA barriers bracket the
reduction — the first split into its ``arrive`` / ``wait`` phases, the
second a single ``sync``.

All ranks must be LSA peers, so run on a single node::

    mpirun -n <gpus-on-node> python 02_lsa_allreduce.py
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

try:
    import cupy as cp
except ImportError:
    print("ERROR: cupy required. Install with: pip install cupy-cuda13x (or cupy-cuda12x)", flush=True)
    sys.exit(1)

import cutlass
import cutlass.cute as cute
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


NAME = os.path.basename(__file__)



THREADS = 256
ELEMS_PER_THREAD = 16
NUM_ELEMS = THREADS * ELEMS_PER_THREAD
BARRIER_INDEX = 0


@cute.kernel
def lsa_allreduce_kernel(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
):
    """Sum every LSA peer's send window into this rank's receive window.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        send_win: registered source window, one per rank.
        recv_win: registered destination window, one per rank.
    """
    tidx, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.cta()
    layout = cute.make_layout(NUM_ELEMS)

    # team_lsa + the devcomm's embedded handle; needs lsa_barrier_count >
    # BARRIER_INDEX in the requirements.
    bar = nccl_cute.lsa_default(coop, dev_comm, index=BARRIER_INDEX)

    # Split into its two phases, which leaves room for local work between
    # publishing this rank's stores and waiting for its peers.
    bar.arrive(coop, nccl_cute.MemoryOrder.RELEASE)
    bar.wait(coop, nccl_cute.MemoryOrder.ACQUIRE)

    if 0 == tidx:
        # The ways to name memory in a window; for the local rank they all
        # resolve to the same address.
        me = dev_comm.lsa_rank
        local = cute.make_ptr(cutlass.Int8, send_win.local_pointer(0)).toint()
        lsa = cute.make_ptr(cutlass.Int8, send_win.lsa_pointer(0, me)).toint()
        peer = cute.make_ptr(cutlass.Int8, send_win.peer_pointer(0, dev_comm.rank)).toint()
        teamed = cute.make_ptr(
            cutlass.Int8, send_win.peer_pointer(0, me, dev_comm.team_lsa)).toint()
        cute.printf(f"local={local} lsa={lsa} peer={peer} peer(team_lsa)={teamed}")

    recv = recv_win.tensor(cutlass.Float32, layout)
    lsa_size = dev_comm.lsa_size

    # Each thread owns ELEMS_PER_THREAD strided elements, so only the peer
    # loop is dynamic.
    for k in range(ELEMS_PER_THREAD):
        i = tidx + k * THREADS
        acc = cutlass.Float32(0.0)
        for peer in cutlass.range(lsa_size):
            # A peer's address for this window offset; the load is ordinary.
            peer_send = cute.make_tensor(
                cute.make_ptr(cutlass.Float32, send_win.lsa_pointer(0, peer)), layout)
            acc += peer_send[i]
        recv[i] = acc

    # arrive + wait in one call; release makes the reduction visible first.
    bar.sync(coop, nccl_cute.MemoryOrder.RELEASE)


@cute.jit
def lsa_allreduce(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
):
    """Launch :func:`lsa_allreduce_kernel` on a single CTA.

    A single CTA keeps the barrier bookkeeping to one slot. Using N CTAs
    requires lsa_barrier_count >= N and a per-CTA barrier index.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        send_win: registered source window.
        recv_win: registered destination window.
    """
    lsa_allreduce_kernel(dev_comm, send_win, recv_win).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


def main():
    """Run the LSA all-reduce and validate the result host-side.

    Returns:
        Exit code; 0 on success, 1 if the payload mismatched.
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

    # Without device API support no devcomm can be created at all.
    if not nccl_comm.device_api_support:
        if rank == root:
            print("WARNING: this platform has no device API support; "
                  "nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    send_buf = nccl.cupy.empty(NUM_ELEMS, dtype='float32')
    recv_buf = nccl.cupy.empty(NUM_ELEMS, dtype='float32')
    send_buf[:] = float(rank + 1)
    recv_buf[:] = 0.0
    device.sync()

    send_win_resource = nccl_comm.register_window(send_buf)
    recv_win_resource = nccl_comm.register_window(recv_buf)

    # One LSA barrier slot for BARRIER_INDEX.
    dev_comm_resource = nccl_comm.create_dev_comm(
        requirements=nccl.NCCLDevCommRequirements(lsa_barrier_count=1))

    lsa_allreduce(
        nccl_cute.DevComm(dev_comm_resource),
        nccl_cute.Window(send_win_resource),
        nccl_cute.Window(recv_win_resource),
    )
    device.sync()

    expected = float(nranks * (nranks + 1) // 2)
    mismatches = int(cp.count_nonzero(recv_buf != expected).item())
    if mismatches == 0:
        print(f"[rank {rank}] [SUCCESS] all {NUM_ELEMS} elements == {expected}")
    else:
        print(f"[rank {rank}] [ERROR] {mismatches} / {NUM_ELEMS} mismatches, "
              f"recv[0]={float(recv_buf[0])} expected {expected}", flush=True)


    dev_comm_resource.close()
    send_win_resource.close()
    recv_win_resource.close()
    nccl_comm.destroy()
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
