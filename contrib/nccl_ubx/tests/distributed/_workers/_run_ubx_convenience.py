"""Worker script for UB-X convenience function (ops.py) tests.

Tests the global allocator registry: request_allocator, get_sym_tensor,
allreduce, restore, free_residual, mem_stats.

Usage:
    python _run_ubx_convenience.py --mode request_and_get_tensor
    python _run_ubx_convenience.py --mode allreduce_convenience
    python _run_ubx_convenience.py --mode restore_round_trip
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


def run_request_and_get_tensor(args):
    """Test request_allocator() then get_sym_tensor()."""
    rank, world_size, local_rank = init_distributed()

    from ubx.ops import request_allocator, get_sym_tensor, _allocator_map
    from ubx.tensor import SymmTensor

    group = dist.group.WORLD
    shape = (1024,)
    dtype = torch.bfloat16

    # request_allocator registers the shape
    request_allocator(group, shape=shape, dtype=dtype)
    assert group in _allocator_map, "Group should be in allocator map after request"
    max_size, alloc = _allocator_map[group]
    assert alloc is None, "Allocator should be None before first get_sym_tensor"

    # get_sym_tensor lazily creates allocator and returns SymmTensor
    tensor = get_sym_tensor(shape, dtype, group)
    assert tensor is not None, "get_sym_tensor should return a tensor for bf16"
    assert isinstance(tensor, SymmTensor), f"Expected SymmTensor, got {type(tensor)}"
    assert tensor.shape == torch.Size(shape), f"Shape mismatch: {tensor.shape}"
    assert tensor.dtype == dtype, f"Dtype mismatch: {tensor.dtype}"

    # Verify allocator was created
    _, alloc = _allocator_map[group]
    assert alloc is not None, "Allocator should be created after get_sym_tensor"

    # Unsupported dtype returns None
    result_fp32 = get_sym_tensor(shape, torch.float32, group)
    assert result_fp32 is None, "get_sym_tensor should return None for fp32"

    print(f"PASS rank={rank} mode=request_and_get_tensor")
    dist.destroy_process_group()


def run_allreduce_convenience(args):
    """Test module-level allreduce() with auto algorithm selection."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    size = 1024

    torch.manual_seed(42 + rank)
    input_data = torch.randn(size, dtype=dtype, device=device)

    # NCCL reference in f32
    ref = input_data.float().clone()
    dist.all_reduce(ref, group=dist.group.WORLD)

    from ubx.ops import request_allocator, get_sym_tensor
    from ubx.ops import allreduce as ops_allreduce

    group = dist.group.WORLD
    shape = (size,)

    request_allocator(group, shape=shape, dtype=dtype)
    symm_tensor = get_sym_tensor(shape, dtype, group)
    symm_tensor.copy_(input_data)

    result = ops_allreduce(symm_tensor)
    torch.cuda.synchronize()

    atol, rtol = 0.0625, 0.02
    try:
        torch.testing.assert_close(result.detach().float(), ref, atol=atol, rtol=rtol)
        print(f"PASS rank={rank} mode=allreduce_convenience")
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=allreduce_convenience: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL rank={rank} mode=allreduce_convenience: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


def run_restore_round_trip(args):
    """Test restore() wraps a tensor back into SymmTensor if in pool."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    size = 512

    from ubx.ops import request_allocator, get_sym_tensor, restore
    from ubx.tensor import SymmTensor

    group = dist.group.WORLD
    shape = (size,)

    request_allocator(group, shape=shape, dtype=dtype)
    symm_tensor = get_sym_tensor(shape, dtype, group)

    torch.manual_seed(42 + rank)
    data = torch.randn(size, dtype=dtype, device=device)
    symm_tensor.copy_(data)

    # Strip SymmTensor subclass (simulating an op that returns a plain Tensor)
    plain_tensor = symm_tensor.as_subclass(torch.Tensor)
    assert not isinstance(plain_tensor, SymmTensor), "as_subclass should return plain Tensor"

    # restore() should wrap it back
    restored = restore(plain_tensor, group)
    assert isinstance(restored, SymmTensor), f"restore should return SymmTensor, got {type(restored)}"

    # Data should be preserved
    torch.testing.assert_close(restored.float(), data.float(), atol=0, rtol=0)

    # Tensor outside pool should be returned unchanged
    outside_tensor = torch.randn(size, dtype=dtype, device=device)
    not_restored = restore(outside_tensor, group)
    assert not isinstance(not_restored, SymmTensor), \
        "Tensor outside pool should not become SymmTensor"

    print(f"PASS rank={rank} mode=restore_round_trip")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UB-X convenience function test worker")
    parser.add_argument("--mode", required=True,
                        choices=["request_and_get_tensor", "allreduce_convenience",
                                 "restore_round_trip"])
    args = parser.parse_args()

    if args.mode == "request_and_get_tensor":
        run_request_and_get_tensor(args)
    elif args.mode == "allreduce_convenience":
        run_allreduce_convenience(args)
    elif args.mode == "restore_round_trip":
        run_restore_round_trip(args)
