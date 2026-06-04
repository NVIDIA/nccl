"""Large-pool regression worker.

Exercises the four collective launchers whose int-vs-int64_t kernel-arg
size mismatch caused IMA on the very first launch (commit 95d5a6c):

  - ubx_alltoall            (out_offset → kernel int64_t lineoffset_out)
  - ubx_alltoall_lamport    (out_offset → kernel int64_t lineoffset_out)
  - ubx_allgather_uc        (out_offset → kernel int64_t lineoffset_out)
  - ubx_allreduce_2shot_uc  (in_offset, out_offset → kernel int64_t lineoffset_*)

The bug surfaces regardless of pool size — `cudaLaunchKernelExC` reads 8
bytes from each kernel-arg slot, so a 4-byte int leaves 4 bytes of
adjacent stack garbage in the upper half of `lineoffset_*`. To catch
future regressions in the int64 arithmetic itself, this worker also
supports `--large-pool` which sizes the pool past the 32 GB / 2^31-uint4
threshold where lineoffset_out genuinely needs 64-bit precision.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")


def _init_distributed():
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        raise RuntimeError("Cannot determine rank — not launched via srun/torchrun")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank,
        )
    return rank, world_size, local_rank


def _alloc_pool(pool_bytes, device, group):
    from ubx import SymmAllocator
    return SymmAllocator(pool_bytes, device, group)


def _run_alltoall(allocator, rank, world_size, device, elems, dtype):
    """alltoall: ubx_alltoall launcher (arg4 was int, kernel int64_t)."""
    size = elems
    torch.manual_seed(42 + rank)
    tensor_in = torch.randn(size, dtype=dtype, device=device)

    chunk = size // world_size
    send_chunks = list(tensor_in.split(chunk))
    recv_chunks = [torch.empty(chunk, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_to_all(recv_chunks, send_chunks)
    ref = torch.cat(recv_chunks, dim=0)

    sym = allocator.create_tensor(tensor_in.shape, dtype)
    sym.copy_(tensor_in)
    out = allocator.alltoall(sym)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=0.001, rtol=0.02)
    return f"alltoall ok elems={size}"


def _run_alltoall_lamport(allocator, rank, world_size, device, elems, dtype):
    """alltoall_lamport: ubx_alltoall_lamport launcher (arg4 was int)."""
    size = elems
    torch.manual_seed(43 + rank)
    tensor_in = torch.randn(size, dtype=dtype, device=device)

    chunk = size // world_size
    send_chunks = list(tensor_in.split(chunk))
    recv_chunks = [torch.empty(chunk, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_to_all(recv_chunks, send_chunks)
    ref = torch.cat(recv_chunks, dim=0)

    sym = allocator.create_tensor(tensor_in.shape, dtype)
    sym.copy_(tensor_in)
    out = allocator.alltoall_lamport(sym)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=0.001, rtol=0.02)
    return f"alltoall_lamport ok elems={size}"


def _run_allgather_uc(allocator, rank, world_size, device, elems, dtype):
    """allgather_uc: ubx_allgather_uc launcher (arg4 was int)."""
    torch.manual_seed(44 + rank)
    local = torch.randn(elems, dtype=dtype, device=device)
    ref_list = [torch.empty_like(local) for _ in range(world_size)]
    dist.all_gather(ref_list, local)
    ref = torch.cat(ref_list, dim=0)

    sym = allocator.create_tensor(local.shape, dtype)
    sym.copy_(local)
    out = allocator.allgather_uc(sym)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, ref, atol=0.001, rtol=0.02)
    return f"allgather_uc ok elems={elems}"


def _run_allreduce_uc(allocator, rank, world_size, device, elems, dtype):
    """allreduce_uc: ubx_allreduce_2shot_uc launcher (arg3, arg4 were int)."""
    torch.manual_seed(45 + rank)
    tensor_in = torch.randn(elems, dtype=dtype, device=device)

    ref = tensor_in.float().clone()
    dist.all_reduce(ref, group=dist.group.WORLD)

    sym = allocator.create_tensor(tensor_in.shape, dtype)
    sym.copy_(tensor_in)
    out = allocator.allreduce_uc(sym)
    torch.cuda.synchronize()
    # bf16 sum across 4 ranks can drift up to ~mantissa * num_ranks per cell;
    # standard ubx bf16 tolerance is (0.001, 0.02), but allreduce stacks
    # rounding so use a slightly looser absolute floor.
    torch.testing.assert_close(out.float(), ref, atol=0.05, rtol=0.05)
    return f"allreduce_uc ok elems={elems}"


_OPS = {
    "alltoall": _run_alltoall,
    "alltoall_lamport": _run_alltoall_lamport,
    "allgather_uc": _run_allgather_uc,
    "allreduce_uc": _run_allreduce_uc,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=list(_OPS) + ["all"])
    parser.add_argument("--elems", type=int, default=8192,
                        help="Data elements per rank (must divide by world_size for a2a/a2a_lamport).")
    parser.add_argument("--pool-bytes", type=int, default=0,
                        help="Pool size in bytes. Default: 6 * tensor_bytes (small).")
    parser.add_argument("--large-pool", action="store_true",
                        help="Use a >32 GB pool to push lineoffset_out past INT_MAX (uint4 units).")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    rank, world_size, local_rank = _init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    elem_size = torch.tensor([], dtype=dtype).element_size()
    base_bytes = max(args.elems * elem_size, 1024)
    if args.large_pool:
        # 34 GB — 32 GB threshold + headroom. With uint4 (16B) lines this is
        # 2^31 + epsilon lines, so lineoffset_out at the high end of the
        # pool requires int64_t precision.
        pool_bytes = 34 * (1024 ** 3)
    elif args.pool_bytes > 0:
        pool_bytes = args.pool_bytes
    else:
        pool_bytes = max(base_bytes * world_size * 6, 64 * 1024 * 1024)

    if rank == 0:
        print(f"[r{rank}] pool_bytes={pool_bytes} ({pool_bytes/(1024**3):.2f} GiB) "
              f"world={world_size} elems={args.elems} dtype={dtype}",
              flush=True)

    try:
        allocator = _alloc_pool(pool_bytes, device, dist.group.WORLD)
    except Exception as e:
        print(f"FAIL rank={rank} SymmAllocator init: {type(e).__name__}: {e}", flush=True)
        sys.exit(2)

    ops = list(_OPS) if args.op == "all" else [args.op]
    for op in ops:
        try:
            msg = _OPS[op](allocator, rank, world_size, device, args.elems, dtype)
            print(f"PASS rank={rank} {msg}", flush=True)
        except AssertionError as e:
            print(f"FAIL rank={rank} op={op}: numeric mismatch: {e}", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"FAIL rank={rank} op={op}: {type(e).__name__}: {e}", flush=True)
            sys.exit(1)

    dist.destroy_process_group()
