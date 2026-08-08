"""LSA ReduceSum example using the nccl4py CuTeDSL device API.

Each rank contributes its LSA rank plus one. The kernel reduces the
registered source windows across each LSA team and writes the sum to each
rank's local destination buffer.

Run with two or more MPI ranks::

    mpirun -n 2 python 07_reduce_copy.py
"""

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

import cutlass
import cutlass.cute as cute
from cutlass.cute.arch.nvvm_wrappers import WARP_SIZE

import nccl.core as nccl
import nccl.core.device.cute as nccl_cute


NUM_ELEMS = 16


@cute.kernel
def lsa_reduce_sum_kernel(
    src_window: nccl_cute.Window,
    dst_window: nccl_cute.Window,
    lsa_rank: cutlass.Int32,
    lsa_size: cutlass.Int32,
):
    team = nccl_cute.Team(
        nRanks=lsa_size,
        rank=lsa_rank,
        stride=cutlass.Int32(1),
    )
    layout = cute.make_layout(NUM_ELEMS)
    src = src_window.tensor(cutlass.Float32, layout)
    dst = dst_window.tensor(cutlass.Float32, layout)
    nccl_cute.lsa_reduce_sum(
        nccl_cute.cta(),
        src_window,
        src,
        dst,
        NUM_ELEMS,
        team=team,
    )


@cute.jit
def run_lsa_reduce_sum(
    src_window: nccl_cute.Window,
    dst_window: nccl_cute.Window,
    lsa_rank: cutlass.Int32,
    lsa_size: cutlass.Int32,
):
    lsa_reduce_sum_kernel(src_window, dst_window, lsa_rank, lsa_size).launch(
        grid=[1, 1, 1],
        block=[cute.size(WARP_SIZE, mode=[0]), 1, 1],
        cooperative=True,
    )


def main() -> int:
    comm_mpi = MPI.COMM_WORLD
    rank = comm_mpi.Get_rank()
    nranks = comm_mpi.Get_size()
    root = 0

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

    src_buf = nccl.cupy.empty(NUM_ELEMS, dtype="float32")
    dst_buf = nccl.cupy.empty(NUM_ELEMS, dtype="float32")
    src_buf[:] = lsa_team.rank + 1
    dst_buf[:] = 0
    device.sync()

    src_win_resource = nccl_comm.register_window(src_buf)
    dst_win_resource = nccl_comm.register_window(dst_buf)
    assert src_win_resource is not None and src_win_resource.is_valid
    assert dst_win_resource is not None and dst_win_resource.is_valid

    src_window = nccl_cute.Window(src_win_resource)
    dst_window = nccl_cute.Window(dst_win_resource)

    run_lsa_reduce_sum(
        src_window,
        dst_window,
        lsa_team.rank,
        lsa_team.n_ranks,
    )
    device.sync()

    src_win_resource.close()
    dst_win_resource.close()
    nccl_comm.destroy()
    local_comm.Free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
