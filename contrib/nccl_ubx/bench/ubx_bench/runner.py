"""Multi-GPU launch and timing infrastructure."""

from __future__ import annotations

import os

# UBX_GRAPH_POOL_SHARE is set per-benchmark based on whether
# CUDA graph mode is enabled (0.9 for graph, 0.1 for eager).

import torch
import torch.distributed as dist
from typing import Dict, List

from .configs import BenchConfig
from .report import BenchResult, format_table, write_json


def init_distributed():
    """Initialize torch.distributed from environment."""
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

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
    return rank, world_size, local_rank


def get_dtype(name: str) -> torch.dtype:
    """Convert dtype name to torch.dtype."""
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return mapping[name]


def run_benchmark(config: BenchConfig):
    """Run the full benchmark sweep."""
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = get_dtype(config.datatype)

    backends = config.backends_to_test()
    sizes = config.size_sweep()

    results: Dict[str, List[BenchResult]] = {b: [] for b in backends}

    # a2av_mxfp8 sweeps token counts instead of byte sizes
    if config.collective == "a2av_mxfp8":
        if "ubx" in backends:
            from .collectives.a2av_mxfp8 import bench_a2av_mxfp8_ubx
            for ntokens in config.token_sweep():
                r = bench_a2av_mxfp8_ubx(
                    ntokens, config.hidden, config.experts_per_rank,
                    config.topk, device, config.iters, config.warmup_iters,
                    world_size, smlimit=config.smlimit,
                    cudagraph=config.cudagraph,
                )
                results["ubx"].append(r)
        if rank == 0:
            table = format_table(config.collective, world_size,
                                 [b for b in backends if b == "ubx"], results)
            print(table)
            if config.output_file and config.output_file.endswith(".json"):
                write_json(config.output_file, config.collective, world_size,
                           [b for b in backends if b == "ubx"], results)
                print(f"\nJSON output written to {config.output_file}")
        dist.destroy_process_group()
        return

    # a2av_combine sweeps token counts and compares 4 implementations:
    # ubx_bf16, ubx_mxfp8, nccl_ep_ll, nccl_ep_ht.
    if config.collective == "a2av_combine":
        from .collectives.a2av_combine import (
            bench_a2av_combine_ubx_bf16,
            bench_a2av_combine_ubx_mxfp8,
            bench_a2av_combine_ubx_bf16_async,
            bench_a2av_combine_ubx_mxfp8_async,
            bench_a2av_combine_nccl_ep,
        )
        for ntokens in config.token_sweep():
            for backend in backends:
                if backend == "ubx_bf16":
                    r = bench_a2av_combine_ubx_bf16(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                        kernel=config.kernel,
                    )
                elif backend == "ubx_mxfp8":
                    r = bench_a2av_combine_ubx_mxfp8(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                        kernel=config.kernel,
                    )
                elif backend == "ubx_bf16_async":
                    r = bench_a2av_combine_ubx_bf16_async(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                elif backend == "ubx_mxfp8_async":
                    r = bench_a2av_combine_ubx_mxfp8_async(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                elif backend in ("nccl_ep_ll", "nccl_ep_ht"):
                    mode = backend.split("_")[-1]
                    r = bench_a2av_combine_nccl_ep(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, mode=mode, cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                else:
                    continue
                results[backend].append(r)
        if rank == 0:
            table = format_table(config.collective, world_size, backends, results)
            print(table)
            if config.output_file and config.output_file.endswith(".json"):
                write_json(config.output_file, config.collective, world_size, backends, results)
                print(f"\nJSON output written to {config.output_file}")
        dist.destroy_process_group()
        return

    # a2av_dispatch sweeps token counts and compares multiple implementations
    # (ubx_bf16, ubx_mxfp8, nccl_ep_ll, nccl_ep_ht) side-by-side.
    if config.collective == "a2av_dispatch":
        from .collectives.a2av_token_dispatch import (
            bench_a2av_dispatch_ubx_bf16,
            bench_a2av_dispatch_ubx_bf16_topk,
            bench_a2av_dispatch_ubx_mxfp8,
            bench_a2av_dispatch_nccl_ep,
        )
        for ntokens in config.token_sweep():
            for backend in backends:
                if backend == "ubx_bf16":
                    r = bench_a2av_dispatch_ubx_bf16(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                elif backend == "ubx_bf16_topk":
                    r = bench_a2av_dispatch_ubx_bf16_topk(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                elif backend == "ubx_mxfp8":
                    r = bench_a2av_dispatch_ubx_mxfp8(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                elif backend in ("nccl_ep_ll", "nccl_ep_ht"):
                    mode = backend.split("_")[-1]
                    r = bench_a2av_dispatch_nccl_ep(
                        ntokens, config.hidden, config.experts_per_rank,
                        config.topk, device, config.iters, config.warmup_iters,
                        world_size, mode=mode, cudagraph=config.cudagraph,
                        routing_alpha=config.routing_alpha,
                    )
                else:
                    continue
                results[backend].append(r)
        if rank == 0:
            table = format_table(config.collective, world_size, backends, results)
            print(table)
            if config.output_file and config.output_file.endswith(".json"):
                write_json(config.output_file, config.collective, world_size, backends, results)
                print(f"\nJSON output written to {config.output_file}")
        dist.destroy_process_group()
        return

    # alltoallv uses standard size sweep with power-law splits
    if config.collective == "alltoallv":
        for size in sizes:
            for backend in backends:
                if backend == "nccl":
                    from .collectives.alltoallv import bench_alltoallv_nccl
                    r = bench_alltoallv_nccl(size, dtype, device, config.iters,
                                              config.warmup_iters, world_size,
                                              alpha=config.alpha)
                    results[backend].append(r)
                elif backend == "ubx":
                    from .collectives.alltoallv import bench_alltoallv_ubx
                    r = bench_alltoallv_ubx(size, dtype, device, config.iters,
                                                config.warmup_iters, world_size,
                                                smlimit=config.smlimit,
                                                alpha=config.alpha)
                    results[backend].append(r)

        if rank == 0:
            table = format_table(config.collective, world_size, backends, results)
            print(table)
            if config.output_file and config.output_file.endswith(".json"):
                write_json(config.output_file, config.collective, world_size, backends, results)
                print(f"\nJSON output written to {config.output_file}")
        dist.destroy_process_group()
        return

    for size in sizes:
        # AllToAll requires size divisible by (nranks * element_size) for even chunks
        if config.collective == "all_to_all":
            elem_size = torch.tensor(0, dtype=dtype).element_size()
            alignment = world_size * elem_size
            size = ((size + alignment - 1) // alignment) * alignment

        for backend in backends:
            if backend == "nccl":
                from .collectives.all_reduce import bench_allreduce_nccl
                from .collectives.all_gather import bench_allgather_nccl
                from .collectives.reduce_scatter import bench_reducescatter_nccl
                from .collectives.all_to_all import bench_alltoall_nccl

                if config.collective == "all_reduce":
                    r = bench_allreduce_nccl(size, dtype, device, config.iters,
                                             config.warmup_iters, world_size,
                                             cudagraph=config.cudagraph)
                elif config.collective == "all_gather":
                    r = bench_allgather_nccl(size, dtype, device, config.iters,
                                             config.warmup_iters, world_size,
                                             cudagraph=config.cudagraph)
                elif config.collective == "reduce_scatter":
                    r = bench_reducescatter_nccl(size, dtype, device, config.iters,
                                                 config.warmup_iters, world_size,
                                                 cudagraph=config.cudagraph)
                elif config.collective == "all_to_all":
                    r = bench_alltoall_nccl(size, dtype, device, config.iters,
                                            config.warmup_iters, world_size,
                                            cudagraph=config.cudagraph)
                else:
                    continue
                results[backend].append(r)

            elif backend == "ubx":
                if config.collective == "all_reduce":
                    from .collectives.all_reduce import bench_allreduce_ubx
                    r = bench_allreduce_ubx(
                        size, dtype, device, config.iters, config.warmup_iters,
                        world_size, kernel=config.kernel, smlimit=config.smlimit,
                        cgasize=config.cgasize, cudagraph=config.cudagraph,
                    )
                    results[backend].append(r)
                elif config.collective == "all_to_all":
                    from .collectives.all_to_all import bench_alltoall_ubx
                    r = bench_alltoall_ubx(
                        size, dtype, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        nthreads_per_block=config.nthreads_per_block,
                        cudagraph=config.cudagraph, kernel=config.kernel,
                    )
                    results[backend].append(r)
                elif config.collective == "all_gather":
                    from .collectives.all_gather import bench_allgather_ubx
                    r = bench_allgather_ubx(
                        size, dtype, device, config.iters, config.warmup_iters,
                        world_size, smlimit=config.smlimit,
                        cudagraph=config.cudagraph, kernel=config.kernel,
                    )
                    results[backend].append(r)


    # Output results (rank 0 only)
    if rank == 0:
        table = format_table(config.collective, world_size, backends, results)
        print(table)

        if config.output_file and config.output_file.endswith(".json"):
            write_json(config.output_file, config.collective, world_size, backends, results)
            print(f"\nJSON output written to {config.output_file}")

    dist.destroy_process_group()
