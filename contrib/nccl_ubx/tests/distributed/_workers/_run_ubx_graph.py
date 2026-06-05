"""Worker script for UB-X CUDA graph capture/replay tests.

IMPORTANT: This worker does NOT override UBX_GRAPH_POOL_SHARE.
The default 0.9 gives the graph pool 90% of memory, which is what we need
since graph-mode allocations go to freelist[True].

Usage:
    python _run_ubx_graph.py --mode allreduce_in_graph
    python _run_ubx_graph.py --mode multiple_replays
    python _run_ubx_graph.py --mode graph_pool_allocation
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

# NOTE: do NOT set UBX_GRAPH_POOL_SHARE here.
# Default 0.9 = graph pool gets 90%, which is what graph-mode tests need.


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


def run_allreduce_in_graph(args):
    """Capture allreduce in CUDA graph, replay, verify correctness."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    size = 1024

    from ubx import SymmAllocator
    group = dist.group.WORLD
    pool_bytes = size * 2 * 6
    allocator = SymmAllocator(pool_bytes, device, group)

    torch.manual_seed(42 + rank)
    input_data = torch.randn(size, dtype=dtype, device=device)

    # NCCL reference in f32
    ref = input_data.float().clone()
    dist.all_reduce(ref, group=dist.group.WORLD)

    # Allocate input in eager mode (goes to non-graph pool, 10%)
    symm_tensor = allocator.create_tensor((size,), dtype)
    symm_tensor.copy_(input_data)

    # Capture graph
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            result = allocator.allreduce(symm_tensor)
    torch.cuda.current_stream().wait_stream(s)

    # Replay
    g.replay()
    torch.cuda.synchronize()

    atol, rtol = 0.0625, 0.02
    try:
        torch.testing.assert_close(result.detach().float(), ref, atol=atol, rtol=rtol)
        print(f"PASS rank={rank} mode=allreduce_in_graph")
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=allreduce_in_graph: {e}", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"FAIL rank={rank} mode=allreduce_in_graph: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


def run_multiple_replays(args):
    """10 replays of captured graph, each with fresh data."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    size = 1024

    from ubx import SymmAllocator
    group = dist.group.WORLD
    pool_bytes = size * 2 * 6
    allocator = SymmAllocator(pool_bytes, device, group)

    # Allocate input tensor (eager mode)
    symm_tensor = allocator.create_tensor((size,), dtype)

    # Need initial data for capture pass
    torch.manual_seed(42 + rank)
    symm_tensor.copy_(torch.randn(size, dtype=dtype, device=device))

    # Capture once
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            result = allocator.allreduce(symm_tensor)
    torch.cuda.current_stream().wait_stream(s)

    # Replay 10 times with different data
    for i in range(10):
        torch.manual_seed(100 + rank + i * 1000)
        input_data = torch.randn(size, dtype=dtype, device=device)

        # NCCL reference in f32
        ref = input_data.float().clone()
        dist.all_reduce(ref, group=dist.group.WORLD)

        # Copy new data to same address, then replay
        symm_tensor.copy_(input_data)
        g.replay()
        torch.cuda.synchronize()

        atol, rtol = 0.0625, 0.02
        try:
            torch.testing.assert_close(result.detach().float(), ref, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"FAIL rank={rank} mode=multiple_replays iteration={i}: {e}", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"FAIL rank={rank} mode=multiple_replays iteration={i}: {type(e).__name__}: {e}", flush=True)
            sys.exit(1)

    print(f"PASS rank={rank} mode=multiple_replays (10 iterations)")
    dist.destroy_process_group()


def run_graph_pool_allocation(args):
    """Verify graph-mode allocations go to graph pool, eager to non-graph pool."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16
    size = 512

    from ubx import SymmAllocator
    group = dist.group.WORLD
    pool_bytes = size * 2 * 20  # generous
    allocator = SymmAllocator(pool_bytes, device, group)

    # Eager allocation: should go to non-graph pool (offset >= graph_pool_size)
    eager_tensor = allocator.create_tensor((size,), dtype)
    eager_offset = eager_tensor.data_ptr() - allocator.pool_ptr
    if eager_offset < allocator.graph_pool_size:
        print(f"FAIL rank={rank}: eager alloc at offset {eager_offset} "
              f"should be >= graph_pool_size {allocator.graph_pool_size}", flush=True)
        sys.exit(1)

    # Graph-mode allocation: should go to graph pool (offset < graph_pool_size)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            graph_tensor = allocator.create_tensor((size,), dtype)
    torch.cuda.current_stream().wait_stream(s)

    graph_offset = graph_tensor.data_ptr() - allocator.pool_ptr
    if graph_offset >= allocator.graph_pool_size:
        print(f"FAIL rank={rank}: graph alloc at offset {graph_offset} "
              f"should be < graph_pool_size {allocator.graph_pool_size}", flush=True)
        sys.exit(1)

    print(f"PASS rank={rank} mode=graph_pool_allocation "
          f"eager_offset={eager_offset} graph_offset={graph_offset} "
          f"graph_pool_size={allocator.graph_pool_size}")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UB-X CUDA graph test worker")
    parser.add_argument("--mode", required=True,
                        choices=["allreduce_in_graph", "multiple_replays",
                                 "graph_pool_allocation"])
    args = parser.parse_args()

    if args.mode == "allreduce_in_graph":
        run_allreduce_in_graph(args)
    elif args.mode == "multiple_replays":
        run_multiple_replays(args)
    elif args.mode == "graph_pool_allocation":
        run_graph_pool_allocation(args)
