#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""1 MiB NCCL GIN put example using CuTeDSL ``cute.Tensor`` views.

Demonstrates the canonical workflow for the nccl4py CuTeDSL device API:
register two windows (send / recv) on the host, construct ``cute.Tensor``
views over them inside the kernel via Window.tensor, issue a
single Gin.put with a completion signal, wait on the signal on
the destination rank, and validate the payload host-side.

The same kernel is launched twice, through the two ways of passing NCCL
resources into a ``@cute.jit`` function:

    * :func:`test_nccl_put` annotates the ``nccl_cute`` types and takes
      objects the caller converted;
    * :func:`test_nccl_put_resources` annotates the ``nccl.core``
      resource types and takes them unconverted, letting the registered
      JIT arg adapters do the conversion.

The annotations are what pick the form: CuTeDSL validates an argument
against its annotation before it looks for an adapter, so annotating the
converted type rejects a raw resource outright.

Run with exactly two MPI ranks; the transfer is hardcoded from rank 0 to
rank 1::

    mpirun -n 2 python 00_basic.py
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
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


NAME = os.path.basename(__file__)

# 1 MiB transfer: 131072 Int64 elements * 8 bytes = 1,048,576 bytes.
NUM_ELEMS = 1024 * 1024 // 8
DST_RANK = 1
# One signal per launch, so the second transfer waits on its own arrival
# instead of seeing the first one's signal already raised.
SIGNAL_ID1 = 1
SIGNAL_ID2 = 2

@cute.kernel
def test_nccl_put_kernel(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
    signal_id,
):
    """Issue a 1 MiB GIN put from rank 0 to rank 1 via ``cute.Tensor`` views.

    Runs with exactly 2 ranks. Two separate windows make the data flow
    explicit:

        * ``send_win`` is rank 0's source buffer (pre-filled with
          ``arange`` on the host).
        * ``recv_win`` is rank 1's destination buffer (validated host-side
          after sync).

    NCCL window registrations are collective, so both ranks register both
    windows; rank 0's ``recv_win`` and rank 1's ``send_win`` exist but go
    unused.

    Args:
        dev_comm: Value-mode nccl_cute.DevComm reconstructed while
            tracing from the host-mode instance.
        send_win: CuTeDSL view of the registered source window.
        recv_win: CuTeDSL view of the registered destination window.
        signal_id: GIN signal slot this launch uses; a Python int, so it
            is baked in as a compile-time constant.
    """
    tidx, _, _ = cute.arch.thread_idx()
    team = dev_comm.team_world
    gin = dev_comm.gin(nccl_cute.GinBackendMask.ALL, 0)
    coop = nccl_cute.cta()

    # cute.Tensor views spanning the full 1 MiB of each window.
    send = send_win.tensor(cutlass.Int64, cute.make_layout(NUM_ELEMS))
    recv = recv_win.tensor(cutlass.Int64, cute.make_layout(NUM_ELEMS))

    if team.nRanks >= 2:
        if 0 == team.rank:
            if 0 == tidx:
                cute.printf(f"Before Put: send[0]={send[0]} send[{NUM_ELEMS - 1}]={send[NUM_ELEMS - 1]}")
            gin.put(
                team,
                DST_RANK,
                recv_win, recv,   # destination window + tensor (lives on the peer)
                send_win, send,   # source window + tensor (local)
                coop,
                is_signal=True,
                signal_id=signal_id,
                signal_op=0,
                signal_op_arg=1,
            )
        if 1 == team.rank:
            gin.wait_signal(coop, signal=signal_id, least=1)
            if 0 == tidx:
                cute.printf(f"After Put:  recv[0]={recv[0]} recv[{NUM_ELEMS - 1}]={recv[NUM_ELEMS - 1]}")


# A @cute.jit function can take NCCL resources in two forms, and the
# parameter annotations decide which one the caller may use: as of
# cutlass-dsl 4.6, CuTeDSL type-checks each argument against its annotation
# before it looks for a JIT arg adapter. So annotating the converted type and
# passing a raw resource is an error CuTeDSL reports, not something the
# adapter registered by @cutlass.register_jit_arg_adapter gets to fix.
# Leaving a parameter unannotated skips the check, and then either form works.
#
# Both wrappers below launch the same kernel; only their signatures and call
# sites differ. Each uses its own signal slot so the second transfer waits on
# its own arrival.


@cute.jit
def test_nccl_put(
        dev_comm: nccl_cute.DevComm,
        send_win: nccl_cute.Window,
        recv_win: nccl_cute.Window,
    ):
    """Launch the kernel, taking arguments the caller already converted.

    The caller wraps each resource — nccl_cute.DevComm(resource) /
    nccl_cute.Window(resource) — so no adapter is needed. Costs a line
    per argument but gives better IDE completion and keeps static
    analysis honest, since the annotations name the types the body
    actually sees.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        send_win: CuTeDSL view of the registered source window.
        recv_win: CuTeDSL view of the registered destination window.
    """
    test_nccl_put_kernel(dev_comm, send_win, recv_win, SIGNAL_ID1).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True
    )


@cute.jit
def test_nccl_put_resources(
        dev_comm: nccl.DevCommResource,
        send_win: nccl.RegisteredWindowHandle,
        recv_win: nccl.RegisteredWindowHandle,
    ):
    """Same launch, taking the nccl.core resources unconverted.

    The registered JIT arg adapters convert each argument on the way in,
    so inside this body the parameters are already nccl_cute.DevComm /
    nccl_cute.Window. Saves a conversion per argument at the call site.

    Args:
        dev_comm: DevCommResource as returned by
            nccl_comm.create_dev_comm.
        send_win: source RegisteredWindowHandle as returned by
            nccl_comm.register_window.
        recv_win: destination RegisteredWindowHandle, likewise.
    """
    test_nccl_put_kernel(dev_comm, send_win, recv_win, SIGNAL_ID2).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True
    )


def main():
    """Run a 1 MiB GIN put + signal demo across exactly two MPI ranks.

    Returns:
        Exit code; 1 on a wrong rank count or a payload mismatch, otherwise 0.
        Host-side validation prints ``[SUCCESS]`` or
        ``[ERROR N / NUM_ELEMS mismatches]``.
    """
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    root = 0

    if nranks != 2:
        if rank == root:
            print(f"\n[{NAME}] ERROR: needs exactly 2 ranks, got {nranks}")
        return 1

    if rank == root:
        print(f"\n===== {NAME} =====", flush=True)

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    # The put below needs a GIN transport.
    if not nccl_comm.device_api_support or nccl_comm.gin_type == nccl.NcclGinType.NONE:
        if rank == root:
            print(f"Gin.put needs a GIN transport, which this platform does not "
                  f"provide (device_api_support={nccl_comm.device_api_support}, "
                  f"gin_type={nccl_comm.gin_type.name}); nothing to run")
        nccl_comm.destroy()
        return 0

    if rank == root:
        print(f"Running with {nranks} ranks, transferring {NUM_ELEMS * 8} bytes "
              f"from rank 0 to rank {DST_RANK}...")

    # Rank 0 fills send_buf with a pattern; rank 1's recv_buf starts zeroed
    # so the transfer is visible. Each rank registers both windows because
    # registration is collective, so one of them goes unused.
    send_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    recv_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    if rank == 0:
        send_buf[:] = cp.arange(NUM_ELEMS, dtype='int64')
    else:
        send_buf[:] = 0
    recv_buf[:] = 0
    device.sync()  # make host-side fill visible before the kernel runs

    send_win_resource = nccl_comm.register_window(send_buf)
    recv_win_resource = nccl_comm.register_window(recv_buf)
    assert send_win_resource is not None and send_win_resource.is_valid
    assert recv_win_resource is not None and recv_win_resource.is_valid

    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        gin_signal_count=max(SIGNAL_ID1, SIGNAL_ID2) + 1,
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)
    assert dev_comm_resource.is_valid
    assert dev_comm_resource.ptr != 0

    # Convert here; the annotations on test_nccl_put name these types.
    if rank == root:
        print("Launch 1: passing converted nccl_cute objects")
    test_nccl_put(
        nccl_cute.DevComm(dev_comm_resource),
        nccl_cute.Window(send_win_resource),
        nccl_cute.Window(recv_win_resource),
    )
    device.sync()
    comm_mpi.Barrier()

    # Clear the destination so the second launch has to transfer the payload
    # again rather than inheriting the first one's result.
    if rank == DST_RANK:
        recv_buf[:] = 0
        device.sync()
    comm_mpi.Barrier()

    # Hand the resources over as nccl.core returned them and let the
    # registered JIT arg adapters convert them.
    if rank == root:
        print("Launch 2: passing the resources straight through")
    test_nccl_put_resources(dev_comm_resource, send_win_resource, recv_win_resource)
    device.sync()
    comm_mpi.Barrier()

    # Host-side validation on the receiver — compare the full 1 MiB payload.
    mismatches = 0
    if rank == DST_RANK:
        expected = cp.arange(NUM_ELEMS, dtype='int64')
        mismatches = int((recv_buf != expected).sum().item())
        if mismatches == 0:
            print(f"[rank {rank}] [SUCCESS] {NUM_ELEMS * 8} bytes transferred correctly")
        else:
            print(f"[rank {rank}] [ERROR] {mismatches} / {NUM_ELEMS} mismatches")

    # Only the destination rank validates, so share the verdict for the exit
    # code.
    mismatches = comm_mpi.bcast(mismatches, root=DST_RANK)

    dev_comm_resource.close()
    send_win_resource.close()
    recv_win_resource.close()
    nccl_comm.destroy()

    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
