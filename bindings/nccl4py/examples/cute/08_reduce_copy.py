# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""LSA, multimem, and local ReduceCopy examples using the nccl4py CuTeDSL device API.

Reduce-sum reads registered source windows into a local destination.
Copy and TMA copy read a local source into registered destination windows.
LSA and multimem reduce-sum-copy use registered source and destination windows.
Local reduce-sum-copy uses ordinary local buffers.
Each example allocates its own buffers. Multimem requires multicast support;
TMA copy requires SM100+ hardware and device bitcode built for SM100+.
Every executed operation validates its destination.

Run with two or more MPI ranks on one node::

    mpirun -n 2 python 08_reduce_copy.py
"""

import os
import sys

try:
    from mpi4py import MPI
except ImportError:
    print("ERROR: mpi4py required. Install with: pip install mpi4py")
    sys.exit(1)

try:
    from cuda.core import Device, system
except ImportError:
    print("ERROR: cuda.core required. Install with: pip install cuda-core")
    sys.exit(1)

try:
    import cupy as cp
except ImportError:
    print("ERROR: cupy required. Install with: pip install cupy-cuda13x (or cupy-cuda12x)")
    sys.exit(1)

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE
from cutlass.cute.runtime import from_dlpack

import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


NAME = os.path.basename(__file__)

NUM_ELEMS = 16
LOCAL_N_SRC = 3
LOCAL_N_DST = 2
LOCAL_SRC_STRIDE = NUM_ELEMS + 4
LOCAL_DST_STRIDE = NUM_ELEMS + 8
TMA_NUM_ELEMS = 4096
TMA_SMEM_BYTES = 4096


@cute.kernel
def lsa_reduce_sum_kernel(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst: cute.Tensor,
):
    src = nccl_cute.SymmetricPointer(src_window, cutlass.Float32)
    nccl_cute.lsa_reduce_sum(
        nccl_cute.Coop.cta(),
        src,
        dst.iterator,
        NUM_ELEMS,
        team=dev_comm.team_lsa,
    )


@cute.kernel
def multimem_reduce_sum_kernel(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst: cute.Tensor,
):
    mc_src = src_window.lsa_multimem_pointer(cutlass.Float32, dev_comm)
    nccl_cute.multimem_reduce_sum(nccl_cute.Coop.cta(), mc_src, dst.iterator, NUM_ELEMS)


@cute.kernel
def lsa_copy_kernel(
    dev_comm: nccl_cute.DevComm,
    src: cute.Tensor,
    dst_window: nccl_cute.Window,
):
    """Broadcast this rank's source into its slot of every LSA peer window."""
    team = dev_comm.team_lsa
    dst = nccl_cute.SymmetricPointer(dst_window, cutlass.Float32) + team.rank * NUM_ELEMS
    nccl_cute.lsa_copy(
        nccl_cute.Coop.cta(),
        src.iterator,
        dst,
        NUM_ELEMS,
        team=team,
    )


@cute.kernel
def lsa_copy_tma_kernel(
    dev_comm: nccl_cute.DevComm,
    src: cute.Tensor,
    dst_window: nccl_cute.Window,
):
    """Same broadcast as :func:`lsa_copy_kernel`, staged through shared memory."""
    allocator = cutlass.memory.SmemAllocator()
    smem = allocator.allocate_array(
        element_type=cutlass.Int8, num_elems=TMA_SMEM_BYTES, byte_alignment=16
    )
    team = dev_comm.team_lsa
    dst = nccl_cute.SymmetricPointer(dst_window, cutlass.Float32) + team.rank * TMA_NUM_ELEMS
    nccl_cute.lsa_copy_tma(
        nccl_cute.Coop.cta(),
        src.iterator,
        dst,
        TMA_NUM_ELEMS,
        team=team,
        smem=smem,
        smem_bytes=TMA_SMEM_BYTES,
    )


@cute.kernel
def multimem_copy_kernel(
    dev_comm: nccl_cute.DevComm,
    src: cute.Tensor,
    dst_window: nccl_cute.Window,
):
    mc_dst = dst_window.lsa_multimem_pointer(cutlass.Float32, dev_comm)
    mc_dst = mc_dst + dev_comm.lsa_rank * NUM_ELEMS
    nccl_cute.multimem_copy(nccl_cute.Coop.cta(), src.iterator, mc_dst, NUM_ELEMS)


@cute.kernel
def lsa_reduce_sum_copy_kernel(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst_window: nccl_cute.Window,
):
    """Each rank reduces and broadcasts a disjoint slice of the windows."""
    team = dev_comm.team_lsa
    offset = team.rank * NUM_ELEMS
    src = nccl_cute.SymmetricPointer(src_window, cutlass.Float32) + offset
    dst = nccl_cute.SymmetricPointer(dst_window, cutlass.Float32) + offset
    nccl_cute.lsa_reduce_sum_copy(nccl_cute.Coop.cta(), src, dst, NUM_ELEMS, team=team)


@cute.kernel
def multimem_reduce_sum_copy_kernel(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst_window: nccl_cute.Window,
):
    """Each rank reduces and multicasts a disjoint slice of the windows."""
    offset = dev_comm.lsa_rank * NUM_ELEMS
    mc_src = src_window.lsa_multimem_pointer(cutlass.Float32, dev_comm) + offset
    mc_dst = dst_window.lsa_multimem_pointer(cutlass.Float32, dev_comm) + offset
    nccl_cute.multimem_reduce_sum_copy(nccl_cute.Coop.cta(), mc_src, mc_dst, NUM_ELEMS)


@cute.kernel
def local_reduce_sum_copy_kernel(src: cute.Tensor, dst: cute.Tensor):
    nccl_cute.local_reduce_sum_copy(nccl_cute.Coop.cta(), src, dst, NUM_ELEMS)


@cute.jit
def run_lsa_reduce_sum(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst: cute.Tensor,
):
    lsa_reduce_sum_kernel(dev_comm, src_window, dst).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


@cute.jit
def run_multimem_reduce_sum(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst: cute.Tensor,
):
    multimem_reduce_sum_kernel(dev_comm, src_window, dst).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


@cute.jit
def run_lsa_copy(
    dev_comm: nccl_cute.DevComm,
    src: cute.Tensor,
    dst_window: nccl_cute.Window,
):
    lsa_copy_kernel(dev_comm, src, dst_window).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


@cute.jit
def run_lsa_copy_tma(
    dev_comm: nccl_cute.DevComm,
    src: cute.Tensor,
    dst_window: nccl_cute.Window,
):
    lsa_copy_tma_kernel(dev_comm, src, dst_window).launch(
        grid=[1, 1, 1],
        block=[128, 1, 1],
        cooperative=True,
    )


@cute.jit
def run_multimem_copy(
    dev_comm: nccl_cute.DevComm,
    src: cute.Tensor,
    dst_window: nccl_cute.Window,
):
    multimem_copy_kernel(dev_comm, src, dst_window).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


@cute.jit
def run_lsa_reduce_sum_copy(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst_window: nccl_cute.Window,
):
    lsa_reduce_sum_copy_kernel(dev_comm, src_window, dst_window).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


@cute.jit
def run_multimem_reduce_sum_copy(
    dev_comm: nccl_cute.DevComm,
    src_window: nccl_cute.Window,
    dst_window: nccl_cute.Window,
):
    multimem_reduce_sum_copy_kernel(dev_comm, src_window, dst_window).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


@cute.jit
def run_local_reduce_sum_copy(src: cute.Tensor, dst: cute.Tensor):
    local_reduce_sum_copy_kernel(src, dst).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
    )


def _check_result(
    name: str, actual: cp.ndarray, expected: float | cp.ndarray, comm_mpi: MPI.Comm
) -> int:
    mismatches = cp.count_nonzero(actual != expected).item()
    mismatches = comm_mpi.allreduce(mismatches, op=MPI.SUM)
    if comm_mpi.Get_rank() == 0:
        if mismatches:
            print(f"[ERROR] {name}: {mismatches} mismatched elements", flush=True)
        else:
            print(f"[SUCCESS] {name}", flush=True)
    return mismatches


def main() -> int:
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

    local_comm = comm_mpi.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = local_comm.Get_rank()

    device = Device(local_rank % system.get_num_devices())
    device.set_current()

    unique_id = nccl.get_unique_id() if rank == root else None
    unique_id = comm_mpi.bcast(unique_id, root=root)
    nccl_comm = nccl.Communicator.init(
        nranks=nranks,
        rank=rank,
        unique_id=unique_id,
    )
    lsa_team = nccl_comm.team_lsa

    if lsa_team.n_ranks != nranks:
        if rank == root:
            print(
                f"\n[{NAME}] ERROR: all {nranks} ranks must be LSA peers "
                f"(lsa team has {lsa_team.n_ranks}); run on one node",
                flush=True,
            )
        nccl_comm.destroy()
        local_comm.Free()
        return 1

    # Without device API support no devcomm can be created at all.
    if not nccl_comm.device_api_support:
        if rank == root:
            print("WARNING: this platform has no device API support; nothing to run", flush=True)
        nccl_comm.destroy()
        local_comm.Free()
        return 0

    dev_comm_resource = nccl_comm.create_dev_comm()
    errors = 0

    # LSA reduce-sum: registered source, local destination.
    src_buf = nccl.cupy.empty(NUM_ELEMS, dtype="float32")
    dst_buf = cp.empty(NUM_ELEMS, dtype="float32")
    src_buf[:] = lsa_team.rank + 1
    dst_buf[:] = 0
    device.sync()

    src_win_resource = nccl_comm.register_window(src_buf)
    assert src_win_resource is not None and src_win_resource.is_valid

    comm_mpi.Barrier()
    run_lsa_reduce_sum(
        nccl_cute.runtime.to_cute(dev_comm_resource),
        nccl_cute.runtime.to_cute(src_win_resource),
        from_dlpack(dst_buf),
    )
    device.sync()
    comm_mpi.Barrier()
    expected_sum = float(lsa_team.n_ranks * (lsa_team.n_ranks + 1) // 2)
    errors += _check_result("lsa_reduce_sum", dst_buf, expected_sum, comm_mpi)
    src_win_resource.close()

    # Multimem reduce-sum: registered multicast source, local destination.
    enable_multimem = comm_mpi.allreduce(nccl_comm.multimem_support, op=MPI.LAND)
    if enable_multimem:
        mm_src = nccl.cupy.empty(NUM_ELEMS, dtype="float32")
        mm_dst = cp.empty(NUM_ELEMS, dtype="float32")
        mm_src[:] = lsa_team.rank + 1
        mm_dst[:] = 0
        device.sync()
        mm_src_win = nccl_comm.register_window(mm_src)
        mm_dev_comm_resource = nccl_comm.create_dev_comm(
            requirements=nccl.NCCLDevCommRequirements(lsa_multimem=True)
        )

        comm_mpi.Barrier()
        run_multimem_reduce_sum(
            nccl_cute.runtime.to_cute(mm_dev_comm_resource),
            nccl_cute.runtime.to_cute(mm_src_win),
            from_dlpack(mm_dst),
        )
        device.sync()
        comm_mpi.Barrier()
        errors += _check_result("multimem_reduce_sum", mm_dst, expected_sum, comm_mpi)
        mm_dev_comm_resource.close()
        mm_src_win.close()
    elif rank == root:
        print("[SKIP] multimem reduce-sum: multicast support is unavailable", flush=True)

    # LSA copy: local source, registered destination with one slot per rank.
    copy_src = cp.empty(NUM_ELEMS, dtype="float32")
    copy_dst = nccl.cupy.empty(lsa_team.n_ranks * NUM_ELEMS, dtype="float32")
    copy_src[:] = lsa_team.rank + 1
    copy_dst[:] = 0
    device.sync()
    copy_dst_win = nccl_comm.register_window(copy_dst)

    comm_mpi.Barrier()
    run_lsa_copy(
        nccl_cute.runtime.to_cute(dev_comm_resource),
        from_dlpack(copy_src),
        nccl_cute.runtime.to_cute(copy_dst_win),
    )
    device.sync()
    comm_mpi.Barrier()
    expected_copy = cp.repeat(cp.arange(1, lsa_team.n_ranks + 1, dtype=cp.float32), NUM_ELEMS)
    errors += _check_result("lsa_copy", copy_dst, expected_copy, comm_mpi)
    copy_dst_win.close()

    major, _ = device.compute_capability
    enable_tma = comm_mpi.allreduce(major >= 10, op=MPI.LAND)
    if enable_tma:
        # TMA copy: local source, registered destination, plus kernel-local shared memory.
        # Four warps process multiple full tiles and a partial final tile.
        tma_src = cp.empty(TMA_NUM_ELEMS, dtype="float32")
        tma_dst = nccl.cupy.empty(lsa_team.n_ranks * TMA_NUM_ELEMS, dtype="float32")
        tma_src[:] = lsa_team.rank * TMA_NUM_ELEMS + cp.arange(
            1, TMA_NUM_ELEMS + 1, dtype=cp.float32
        )
        tma_dst[:] = 0
        device.sync()
        tma_dst_win = nccl_comm.register_window(tma_dst)

        comm_mpi.Barrier()
        run_lsa_copy_tma(
            nccl_cute.runtime.to_cute(dev_comm_resource),
            from_dlpack(tma_src),
            nccl_cute.runtime.to_cute(tma_dst_win),
        )
        device.sync()
        comm_mpi.Barrier()
        expected_tma = cp.arange(1, tma_dst.size + 1, dtype=cp.float32)
        errors += _check_result("lsa_copy_tma", tma_dst, expected_tma, comm_mpi)
        tma_dst_win.close()
    elif rank == root:
        print("[SKIP] lsa_copy_tma: SM100+ is required on all ranks", flush=True)

    # Multimem copy: local source, registered multicast destination.
    if enable_multimem:
        mm_copy_src = cp.empty(NUM_ELEMS, dtype="float32")
        mm_copy_dst = nccl.cupy.empty(lsa_team.n_ranks * NUM_ELEMS, dtype="float32")
        mm_copy_src[:] = lsa_team.rank + 1
        mm_copy_dst[:] = 0
        device.sync()
        mm_copy_dst_win = nccl_comm.register_window(mm_copy_dst)
        mm_copy_dev_comm = nccl_comm.create_dev_comm(
            requirements=nccl.NCCLDevCommRequirements(lsa_multimem=True)
        )

        comm_mpi.Barrier()
        run_multimem_copy(
            nccl_cute.runtime.to_cute(mm_copy_dev_comm),
            from_dlpack(mm_copy_src),
            nccl_cute.runtime.to_cute(mm_copy_dst_win),
        )
        device.sync()
        comm_mpi.Barrier()
        errors += _check_result("multimem_copy", mm_copy_dst, expected_copy, comm_mpi)
        mm_copy_dev_comm.close()
        mm_copy_dst_win.close()
    elif rank == root:
        print("[SKIP] multimem copy: multicast support is unavailable", flush=True)

    # LSA reduce-sum-copy: registered source and destination, one slice per rank.
    reduce_copy_src = nccl.cupy.empty(lsa_team.n_ranks * NUM_ELEMS, dtype="float32")
    reduce_copy_dst = nccl.cupy.empty(lsa_team.n_ranks * NUM_ELEMS, dtype="float32")
    reduce_copy_src[:] = lsa_team.rank + 1
    reduce_copy_dst[:] = 0
    device.sync()
    reduce_copy_src_win = nccl_comm.register_window(reduce_copy_src)
    reduce_copy_dst_win = nccl_comm.register_window(reduce_copy_dst)

    comm_mpi.Barrier()
    run_lsa_reduce_sum_copy(
        nccl_cute.runtime.to_cute(dev_comm_resource),
        nccl_cute.runtime.to_cute(reduce_copy_src_win),
        nccl_cute.runtime.to_cute(reduce_copy_dst_win),
    )
    device.sync()
    comm_mpi.Barrier()
    errors += _check_result("lsa_reduce_sum_copy", reduce_copy_dst, expected_sum, comm_mpi)
    reduce_copy_dst_win.close()
    reduce_copy_src_win.close()

    # Multimem reduce-sum-copy: registered multicast source and destination.
    if enable_multimem:
        mm_reduce_copy_src = nccl.cupy.empty(lsa_team.n_ranks * NUM_ELEMS, dtype="float32")
        mm_reduce_copy_dst = nccl.cupy.empty(lsa_team.n_ranks * NUM_ELEMS, dtype="float32")
        mm_reduce_copy_src[:] = lsa_team.rank + 1
        mm_reduce_copy_dst[:] = 0
        device.sync()
        mm_reduce_copy_src_win = nccl_comm.register_window(mm_reduce_copy_src)
        mm_reduce_copy_dst_win = nccl_comm.register_window(mm_reduce_copy_dst)
        mm_reduce_copy_dev_comm = nccl_comm.create_dev_comm(
            requirements=nccl.NCCLDevCommRequirements(lsa_multimem=True)
        )

        comm_mpi.Barrier()
        run_multimem_reduce_sum_copy(
            nccl_cute.runtime.to_cute(mm_reduce_copy_dev_comm),
            nccl_cute.runtime.to_cute(mm_reduce_copy_src_win),
            nccl_cute.runtime.to_cute(mm_reduce_copy_dst_win),
        )
        device.sync()
        comm_mpi.Barrier()
        errors += _check_result(
            "multimem_reduce_sum_copy", mm_reduce_copy_dst, expected_sum, comm_mpi
        )
        mm_reduce_copy_dev_comm.close()
        mm_reduce_copy_dst_win.close()
        mm_reduce_copy_src_win.close()
    elif rank == root:
        print("[SKIP] multimem reduce-sum-copy: multicast support is unavailable", flush=True)

    # Local reduce-sum-copy: three source rows into two destination rows, preserving padding.
    local_src = cp.empty((LOCAL_N_SRC, LOCAL_SRC_STRIDE), dtype="float32")
    local_dst = cp.empty((LOCAL_N_DST, LOCAL_DST_STRIDE), dtype="float32")
    local_src[:] = -1
    local_src[:, :NUM_ELEMS] = cp.arange(1, LOCAL_N_SRC + 1, dtype=cp.float32)[:, None]
    local_dst[:] = -1
    device.sync()

    run_local_reduce_sum_copy(from_dlpack(local_src), from_dlpack(local_dst))
    device.sync()
    expected_local = cp.full_like(local_dst, -1)
    expected_local[:, :NUM_ELEMS] = LOCAL_N_SRC * (LOCAL_N_SRC + 1) // 2
    errors += _check_result("local_reduce_sum_copy", local_dst, expected_local, comm_mpi)

    dev_comm_resource.close()
    nccl_comm.destroy()
    local_comm.Free()
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
