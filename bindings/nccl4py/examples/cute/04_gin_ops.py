#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#
"""Every GIN (network) operation, end to end.

Rank 0 pushes a payload to rank 1 with a counter for local completion,
appends a scalar with ``put_value``, then raises a remote signal. Rank 1
waits on that signal, inspects the signal and its shadow, pulls the
payload back with ``get``, and resets the slots it used.

``Gin.get`` needs NCCL 2.31.1: older device libraries export
``ncclGinGet`` C++-mangled only, so the stub has nothing to bind to. That
phase is skipped below 2.31.1.

Needs exactly 2 ranks with a GIN-capable network. Two nodes give a real
network path; a single node works when NCCL can set up loopback GIN
connections::

    mpirun -n 2 -N 1 python 04_gin_ops.py
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


# Up to 2.31.0 ncclGinGet had no NCCL_IR_EXTERN_C declaration, so the device
# library exports it C++-mangled only and the stub has nothing to bind to;
# JIT compilation of the kernel fails.
# get_version() reports the loaded libnccl, which is what the device
# bitcode is expected to match.
NCCL_VERSION = nccl.get_version().libnccl.version
NCCL_HAS_GIN_GET = NCCL_VERSION.release >= (2, 31, 1)

NUM_ELEMS = 4096
THREADS = 128
SRC_RANK = 0
DST_RANK = 1
SIGNAL_ID = 0
COUNTER_ID = 0
SCALAR = 0x5EED

# ncclGinSignalOp_t: 0 = Inc (signal_op_arg ignored), 1 = Add.
SIGNAL_OP_INC = 0


@cute.kernel
def gin_ops_kernel(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
    get_win: nccl_cute.Window,
):
    """Drive one full GIN round trip between rank 0 and rank 1.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        send_win: rank 0's payload source, also the remote source of the get.
        recv_win: rank 1's payload destination.
        get_win: rank 1's destination for the data it pulls back.
    """
    tidx, _, _ = cute.arch.thread_idx()
    coop = nccl_cute.cta()
    world = dev_comm.team_world

    # The backend mask selects which transports may back this context; the
    # sharing mode says how widely its network resources are shared.
    gin = dev_comm.gin(
        nccl_cute.GinBackendMask.ALL, 0,
        resource_sharing_mode=nccl_cute.GinResourceSharingMode.GPU)

    payload = cute.make_layout(NUM_ELEMS - 1)
    scalar = cute.make_layout(1)
    send = send_win.tensor(cutlass.Int64, payload)
    recv = recv_win.tensor(cutlass.Int64, payload)
    # The last element sits past the payload; offset is in bytes.
    recv_tail = recv_win.tensor(cutlass.Int64, scalar, offset=(NUM_ELEMS - 1) * 8)

    if SRC_RANK == dev_comm.rank:
        # is_counter bumps a *local* counter on local completion — the
        # sender-side notification, unlike the receiver-side signal below.
        # The release scopes say what the caller has released, and what the
        # transfer needs released before it reads the source.
        gin.put(
            world, DST_RANK,
            recv_win, recv,
            send_win, send,
            coop,
            is_counter=True, counter_id=COUNTER_ID,
            given_release=nccl_cute.ThreadScope.THREAD,
            required_release=nccl_cute.ThreadScope.DEVICE,
        )
        gin.wait_counter(coop, counter=COUNTER_ID, least=1)
        if 0 == tidx:
            cute.printf(f"rank 0: put complete, counter={gin.read_counter(counter=COUNTER_ID)}")

        # No source window: the value travels inline, sized by the
        # destination tensor's element type.
        gin.put_value(world, DST_RANK, recv_win, recv_tail, SCALAR, coop)

        # A signal with no data transfer, ordered after the puts above.
        gin.signal(
            world, DST_RANK,
            True, SIGNAL_ID, SIGNAL_OP_INC, 1,
            coop,
        )
        # Drain this context before the kernel exits.
        gin.flush(coop, nccl_cute.MemoryOrder.RELEASE)

    if DST_RANK == dev_comm.rank:
        # The put and put_value are ordered ahead of the signal, so both
        # have landed once this returns.
        gin.wait_signal(coop, signal=SIGNAL_ID, least=1)

        if 0 == tidx:
            cute.printf(
                f"rank 1: signal={gin.read_signal(signal=SIGNAL_ID)} "
                f"recv[0]={recv[0]} tail={recv_tail[0]}")

            # A user-managed uint64 next to the signal — NCCL never writes
            # it, and reset_signal zeroes both. The natural place to record
            # how much of the signal this rank has consumed.
            shadow = cute.make_tensor(
                gin.signal_shadow_pointer(signal=SIGNAL_ID), scalar)
            shadow[0] = cutlass.Uint64(1)
            cute.printf(f"rank 1: shadow={shadow[0]}")

        if NCCL_HAS_GIN_GET:
            # A get needs no participation from the peer; flush waits for
            # it to land.
            remote = send_win.tensor(cutlass.Int64, payload)
            local = get_win.tensor(cutlass.Int64, payload)
            gin.get(world, SRC_RANK, send_win, remote, get_win, local, coop)
            gin.flush(coop, nccl_cute.MemoryOrder.ACQUIRE)

            if 0 == tidx:
                cute.printf(f"rank 1: get complete, get[0]={local[0]}")

        # Leave the slots reusable. Neither reset may race with an update,
        # hence the single thread and the syncs.
        coop.sync()
        if 0 == tidx:
            gin.reset_signal(signal=SIGNAL_ID)
            gin.reset_counter(counter=COUNTER_ID)
        coop.sync()


@cute.jit
def gin_ops(
    dev_comm: nccl_cute.DevComm,
    send_win: nccl_cute.Window,
    recv_win: nccl_cute.Window,
    get_win: nccl_cute.Window,
):
    """Launch :func:`gin_ops_kernel` on a single CTA.

    Args:
        dev_comm: CuTeDSL view of the NCCL device communicator.
        send_win: payload source window.
        recv_win: payload destination window.
        get_win: destination window for the get.
    """
    gin_ops_kernel(dev_comm, send_win, recv_win, get_win).launch(
        grid=[1, 1, 1],
        block=[THREADS, 1, 1],
        cooperative=True,
    )


def main():
    """Run the GIN round trip across 2 ranks and validate both directions.

    Returns:
        Exit code; 0 on success, 1 on a wrong rank count or bad payload.
    """
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    root = 0

    if rank == root:
        print(f"\n===== {NAME} =====", flush=True)
        if not NCCL_HAS_GIN_GET:
            print(f"WARNING: NCCL {NCCL_VERSION} exports "
                  "ncclGinGet C++-mangled only; skipping Gin.get", flush=True)

    if nranks != 2:
        if rank == root:
            print(f"\n[{NAME}] ERROR: needs exactly 2 ranks, got {nranks}")
        return 1

    device = Device(rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)

    nccl_comm = nccl.Communicator.init(nranks=nranks, rank=rank, unique_id=unique_id)

    # Requesting GIN resources with gin_type NONE fails devcomm creation.
    if not nccl_comm.device_api_support or nccl_comm.gin_type == nccl.NcclGinType.NONE:
        if rank == root:
            print("WARNING: no GIN transport on this system "
                  f"(gin_type={nccl_comm.gin_type.name}); nothing to run", flush=True)
        nccl_comm.destroy()
        return 0

    send_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    recv_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    get_buf = nccl.cupy.empty(NUM_ELEMS, dtype='int64')
    send_buf[:] = cp.arange(NUM_ELEMS, dtype='int64') + (rank + 1)
    recv_buf[:] = 0
    get_buf[:] = 0
    device.sync()

    send_win_resource = nccl_comm.register_window(send_buf)
    recv_win_resource = nccl_comm.register_window(recv_buf)
    get_win_resource = nccl_comm.register_window(get_buf)

    # FULL connects every rank to every other rank over GIN. The signal and
    # counter counts must cover the ids the kernel uses; ids start at 0.
    reqs = nccl.NCCLDevCommRequirements(
        gin_connection_type=nccl.NcclGinConnectionType.FULL,
        gin_signal_count=SIGNAL_ID + 1,
        gin_counter_count=COUNTER_ID + 1,
    )
    dev_comm_resource = nccl_comm.create_dev_comm(requirements=reqs)

    gin_ops(
        nccl_cute.DevComm(dev_comm_resource),
        nccl_cute.Window(send_win_resource),
        nccl_cute.Window(recv_win_resource),
        nccl_cute.Window(get_win_resource),
    )
    device.sync()
    # Rank 1's get reads rank 0's window, so nobody tears down early.
    comm_mpi.Barrier()

    failures = 0
    if rank == DST_RANK:
        expected = cp.arange(NUM_ELEMS - 1, dtype='int64') + 1
        put_mismatches = int(cp.count_nonzero(recv_buf[:NUM_ELEMS - 1] != expected).item())
        tail_matches = int(recv_buf[NUM_ELEMS - 1]) == SCALAR
        get_mismatches = (int(cp.count_nonzero(get_buf[:NUM_ELEMS - 1] != expected).item())
                          if NCCL_HAS_GIN_GET else 0)
        failures = put_mismatches + get_mismatches + (0 if tail_matches else 1)
        checked = ("Gin.put, Gin.put_value and Gin.get" if NCCL_HAS_GIN_GET
                   else "Gin.put and Gin.put_value")
        if failures == 0:
            print(f"[rank {rank}] [SUCCESS] {checked} validated")
        else:
            print(f"[rank {rank}] [ERROR] put mismatches={put_mismatches} "
                  f"tail_matches={tail_matches} get mismatches={get_mismatches}", flush=True)

    # Only the destination rank validates, so share the verdict for the exit
    # code.
    failures = comm_mpi.bcast(failures, root=DST_RANK)

    dev_comm_resource.close()
    send_win_resource.close()
    recv_win_resource.close()
    get_win_resource.close()
    nccl_comm.destroy()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
