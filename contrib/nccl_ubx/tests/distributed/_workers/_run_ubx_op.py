"""Worker script for UB-X distributed tests.

Launched by srun/mpirun/torchrun. Detects rank from environment,
initializes torch.distributed, runs the specified operation, and
compares against a reference result.

Usage:
    python _run_ubx_op.py --op allreduce_mc --size 1048576 --dtype bf16
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

# Tests run in eager mode (no CUDA graph), so allocations go to the non-graph
# pool. The default GRAPH_POOL_SHARE=0.9 leaves only 10% for eager-mode use,
# which is too small for large tensors. Flip it so eager mode gets 90%.
os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")


def get_rank_info():
    """Detect rank from environment variables.

    Check torchrun (RANK) and MPI before Slurm — when running under torchrun
    inside a Slurm allocation, SLURM_PROCID is inherited by all child processes
    and would incorrectly make every rank appear as rank 0.
    """
    # Try torchrun first (sets RANK per-process correctly)
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    # Try MPI
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    # Try Slurm (only when not under torchrun/mpirun)
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        raise RuntimeError("Cannot determine rank — not launched via srun/mpirun/torchrun")
    return rank, world_size, local_rank


def init_distributed():
    """Initialize torch.distributed with NCCL backend."""
    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
    return rank, world_size, local_rank


def run_allreduce_test(args):
    """Run UB-X allreduce and compare against NCCL reference."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    # Create input tensor
    torch.manual_seed(42 + rank)
    size = args.size
    tensor_input = torch.randn(size, dtype=dtype, device=device)

    # NCCL reference — use f32 to get the exact sum, avoiding intermediate
    # bf16 truncation that NCCL's multi-step reduction introduces for >2 ranks.
    print(f"[rank{rank}] running NCCL reference allreduce", flush=True)
    ref = tensor_input.float().clone()
    dist.all_reduce(ref, group=dist.group.WORLD)
    print(f"[rank{rank}] NCCL reference done", flush=True)

    # UB-X allreduce
    from ubx import SymmAllocator
    group = dist.group.WORLD
    print(f"[rank{rank}] creating SymmAllocator", flush=True)
    allocator = SymmAllocator(
        size * tensor_input.element_size() * 6,
        device, group,
    )
    print(f"[rank{rank}] SymmAllocator OK, mc_ptr={allocator.mc0_ptr}", flush=True)
    symm_tensor = allocator.create_tensor(tensor_input.shape, dtype)
    symm_tensor.copy_(tensor_input)
    print(f"[rank{rank}] tensor copied, calling {args.op}", flush=True)

    if args.op == "allreduce_mc":
        result = allocator.allreduce_mc(symm_tensor, smlimit=args.smlimit)
    elif args.op == "allreduce_lamport":
        result = allocator.allreduce_lamport(symm_tensor, smlimit=args.smlimit)
    elif args.op == "allreduce_uc":
        result = allocator.allreduce_uc(symm_tensor, smlimit=args.smlimit)
    elif args.op == "allreduce_auto":
        result = allocator.allreduce(symm_tensor, smlimit=args.smlimit)
    else:
        raise ValueError(f"Unknown op: {args.op}")

    print(f"[rank{rank}] kernel done, synchronizing", flush=True)
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        print(f"FAIL rank={rank} op={args.op} size={size}: CUDA error in synchronize: {e}", flush=True)
        sys.exit(1)
    print(f"[rank{rank}] sync done", flush=True)

    # Compare UB-X bf16 result against f32 reference.
    # MULTIMEM_LD accumulates in f32 then truncates to bf16, so error vs f32
    # truth is at most a few bf16 ULPs (~0.03 for typical magnitudes).
    atol = 0.0625 if dtype == torch.bfloat16 else 0.0001
    rtol = 0.02 if dtype == torch.bfloat16 else 0.001
    try:
        result_f32 = result.detach().float()
        print(f"[rank{rank}] result[:4]={result_f32[:4].cpu().tolist()} ref[:4]={ref[:4].cpu().tolist()}", flush=True)
        max_err = (result_f32.cpu() - ref.cpu()).abs().max().item()
        print(f"[rank{rank}] max_abs_err={max_err:.6f} atol={atol} rtol={rtol}", flush=True)
        torch.testing.assert_close(result_f32, ref, atol=atol, rtol=rtol)
        print(f"PASS rank={rank} op={args.op} size={size}")
    except AssertionError as e:
        print(f"FAIL rank={rank} op={args.op} size={size}: numerical mismatch: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL rank={rank} op={args.op} size={size}: unexpected error in comparison: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


def run_allgather_test(args):
    """Run UB-X allgather and compare against NCCL reference."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dtype = torch.bfloat16
    size = args.size

    torch.manual_seed(42 + rank)
    local_tensor = torch.randn(size, dtype=dtype, device=device)

    # NCCL reference
    ref_list = [torch.empty_like(local_tensor) for _ in range(world_size)]
    dist.all_gather(ref_list, local_tensor)
    ref = torch.cat(ref_list, dim=0)

    # UB-X allgather
    from ubx import SymmAllocator
    group = dist.group.WORLD
    pool_size = size * local_tensor.element_size() * world_size * 6
    allocator = SymmAllocator(pool_size, device, group)
    symm_in = allocator.create_tensor(local_tensor.shape, dtype)
    symm_in.copy_(local_tensor)

    if args.op == "allgather_uc":
        result = allocator.allgather_uc(symm_in, smlimit=args.smlimit)
    else:
        result = allocator.allgather(symm_in, smlimit=args.smlimit)
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(result, ref, atol=0.001, rtol=0.02)
        print(f"PASS rank={rank} op={args.op} size={size}")
    except AssertionError as e:
        print(f"FAIL rank={rank} op={args.op} size={size}: {e}")
        sys.exit(1)

    dist.destroy_process_group()


def run_alltoall_test(args):
    """Run UB-X alltoall and compare against manual reference."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    dtype = torch.bfloat16
    size = args.size  # total size, must be divisible by world_size

    torch.manual_seed(42 + rank)
    tensor_input = torch.randn(size, dtype=dtype, device=device)

    # Manual reference via torch.distributed
    chunk_size = size // world_size
    send_chunks = list(tensor_input.split(chunk_size))
    recv_chunks = [torch.empty(chunk_size, dtype=dtype, device=device) for _ in range(world_size)]
    dist.all_to_all(recv_chunks, send_chunks)
    ref = torch.cat(recv_chunks, dim=0)

    # UB-X alltoall
    from ubx import SymmAllocator
    group = dist.group.WORLD
    pool_size = size * tensor_input.element_size() * 6
    allocator = SymmAllocator(pool_size, device, group)
    symm_in = allocator.create_tensor(tensor_input.shape, dtype)
    symm_in.copy_(tensor_input)

    if args.op == "alltoall_lamport":
        result = allocator.alltoall_lamport(symm_in, smlimit=args.smlimit)
    else:
        result = allocator.alltoall(symm_in, smlimit=args.smlimit)
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(result, ref, atol=0.001, rtol=0.02)
        print(f"PASS rank={rank} op={args.op} size={size}")
    except AssertionError as e:
        print(f"FAIL rank={rank} op={args.op} size={size}: {e}")
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UB-X distributed test worker")
    parser.add_argument("--op", required=True,
                        choices=["allreduce_mc", "allreduce_lamport", "allreduce_uc",
                                 "allreduce_auto", "allgather", "allgather_uc",
                                 "alltoall", "alltoall_lamport"])
    parser.add_argument("--size", type=int, default=1024, help="Number of elements")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--smlimit", type=int, default=0)
    parser.add_argument("--cgasize", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.op.startswith("allreduce"):
        run_allreduce_test(args)
    elif args.op in ("allgather", "allgather_uc"):
        run_allgather_test(args)
    elif args.op in ("alltoall", "alltoall_lamport"):
        run_alltoall_test(args)
