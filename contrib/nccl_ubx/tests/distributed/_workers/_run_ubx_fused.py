"""Worker script for UB-X fused operations (residual + RMSNorm) tests.

Tests the fused paths where residual_in and/or gamma are non-null,
comparing against sequential reference computations.

Usage:
    python _run_ubx_fused.py --mode residual_only --hidden_size 1024
    python _run_ubx_fused.py --mode rmsnorm --hidden_size 4096
    python _run_ubx_fused.py --mode fused_vs_unfused
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")


def get_rank_info():
    """Detect rank from environment variables."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
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
            backend="nccl", init_method="env://",
            world_size=world_size, rank=rank,
        )
    return rank, world_size, local_rank


def run_residual_only_test(args):
    """Test allreduce with residual add but no RMSNorm."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    num_tokens = 8
    hidden_size = args.hidden_size
    shape = (num_tokens, hidden_size)

    torch.manual_seed(42 + rank)
    tensor_input = torch.randn(shape, dtype=dtype, device=device)
    # Residual must be identical on all ranks (it's an allgathered activation)
    torch.manual_seed(999)
    residual_in = torch.randn(shape, dtype=dtype, device=device)

    # NCCL reference in f32
    ref_allreduced = tensor_input.float().clone()
    dist.all_reduce(ref_allreduced, group=dist.group.WORLD)
    ref_result = ref_allreduced + residual_in.float()

    # UB-X fused path
    from ubx import SymmAllocator
    from ubx.fused import allreduce_fused
    group = dist.group.WORLD
    pool_bytes = num_tokens * hidden_size * 2 * 6
    allocator = SymmAllocator(pool_bytes, device, group)

    symm_tensor = allocator.create_tensor(shape, dtype)
    symm_tensor.copy_(tensor_input)

    residual_out = torch.empty(shape, dtype=dtype, device=device)

    result = allreduce_fused(
        allocator, symm_tensor,
        hidden_size=hidden_size,
        residual_in=residual_in,
        residual_out=residual_out,
    )
    torch.cuda.synchronize()

    atol, rtol = 0.125, 0.05
    try:
        torch.testing.assert_close(result.detach().float(), ref_result, atol=atol, rtol=rtol)
        print(f"PASS rank={rank} mode=residual_only hidden_size={hidden_size}")
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=residual_only hidden_size={hidden_size}: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL rank={rank} mode=residual_only hidden_size={hidden_size}: "
              f"{type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


def run_rmsnorm_test(args):
    """Test allreduce with fused residual + RMSNorm."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    num_tokens = 8
    hidden_size = args.hidden_size
    shape = (num_tokens, hidden_size)
    eps = 1e-5

    torch.manual_seed(42 + rank)
    tensor_input = torch.randn(shape, dtype=dtype, device=device)
    # Residual and gamma must be identical on all ranks
    torch.manual_seed(999)
    residual_in = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.ones(hidden_size, dtype=dtype, device=device)

    # NCCL reference in f32
    ref_allreduced = tensor_input.float().clone()
    dist.all_reduce(ref_allreduced, group=dist.group.WORLD)
    ref_pre_norm = ref_allreduced + residual_in.float()

    # RMSNorm: x * gamma / sqrt(mean(x^2, dim=-1) + eps)
    variance = ref_pre_norm.pow(2).mean(dim=-1, keepdim=True)
    ref_normed = ref_pre_norm * torch.rsqrt(variance + eps) * gamma.float()

    # UB-X fused path
    from ubx import SymmAllocator
    from ubx.fused import allreduce_fused
    group = dist.group.WORLD
    pool_bytes = num_tokens * hidden_size * 2 * 6
    allocator = SymmAllocator(pool_bytes, device, group)

    symm_tensor = allocator.create_tensor(shape, dtype)
    symm_tensor.copy_(tensor_input)

    residual_out = torch.empty(shape, dtype=dtype, device=device)

    result = allreduce_fused(
        allocator, symm_tensor,
        hidden_size=hidden_size,
        residual_in=residual_in,
        residual_out=residual_out,
        gamma=gamma,
        eps=eps,
    )
    torch.cuda.synchronize()

    atol, rtol = 0.125, 0.05
    try:
        torch.testing.assert_close(result.detach().float(), ref_normed, atol=atol, rtol=rtol)
        print(f"PASS rank={rank} mode=rmsnorm hidden_size={hidden_size}")
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=rmsnorm hidden_size={hidden_size}: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL rank={rank} mode=rmsnorm hidden_size={hidden_size}: "
              f"{type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


def run_fused_vs_unfused_test(args):
    """Compare fused kernel output vs sequential (unfused) ops."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    num_tokens = 8
    hidden_size = 1024
    shape = (num_tokens, hidden_size)
    eps = 1e-5

    torch.manual_seed(42 + rank)
    tensor_input = torch.randn(shape, dtype=dtype, device=device)
    # Residual and gamma must be identical on all ranks
    torch.manual_seed(999)
    residual_in = torch.randn(shape, dtype=dtype, device=device)
    gamma = torch.randn(hidden_size, dtype=dtype, device=device).abs() + 0.1

    # Unfused path: plain allreduce then manual residual + RMSNorm
    from ubx import SymmAllocator
    from ubx.fused import allreduce_fused
    group = dist.group.WORLD
    pool_bytes = num_tokens * hidden_size * 2 * 12

    allocator1 = SymmAllocator(pool_bytes, device, group)
    symm1 = allocator1.create_tensor(shape, dtype)
    symm1.copy_(tensor_input)
    result_unfused = allocator1.allreduce(symm1)
    torch.cuda.synchronize()

    # Manual residual + RMSNorm in f32
    manual_pre_norm = result_unfused.detach().float() + residual_in.float()
    variance = manual_pre_norm.pow(2).mean(dim=-1, keepdim=True)
    manual_normed = manual_pre_norm * torch.rsqrt(variance + eps) * gamma.float()

    # Fused path: separate allocator to avoid state conflicts
    dist.barrier()
    allocator2 = SymmAllocator(pool_bytes, device, group)
    symm2 = allocator2.create_tensor(shape, dtype)
    symm2.copy_(tensor_input)
    residual_out = torch.empty(shape, dtype=dtype, device=device)

    result_fused = allreduce_fused(
        allocator2, symm2,
        hidden_size=hidden_size,
        residual_in=residual_in,
        residual_out=residual_out,
        gamma=gamma,
        eps=eps,
    )
    torch.cuda.synchronize()

    atol, rtol = 0.125, 0.05
    try:
        torch.testing.assert_close(result_fused.detach().float(), manual_normed, atol=atol, rtol=rtol)
        print(f"PASS rank={rank} mode=fused_vs_unfused")
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=fused_vs_unfused: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL rank={rank} mode=fused_vs_unfused: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UB-X fused ops test worker")
    parser.add_argument("--mode", required=True,
                        choices=["residual_only", "rmsnorm", "fused_vs_unfused"])
    parser.add_argument("--hidden_size", type=int, default=1024)
    args = parser.parse_args()

    if args.mode == "residual_only":
        run_residual_only_test(args)
    elif args.mode == "rmsnorm":
        run_rmsnorm_test(args)
    elif args.mode == "fused_vs_unfused":
        run_fused_vs_unfused_test(args)
