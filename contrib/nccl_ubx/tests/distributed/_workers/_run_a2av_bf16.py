"""Worker script for UB-X a2av token bf16->bf16 dispatch tests.

Mirrors _run_a2av_mxfp8.py but the kernel writes raw bf16 (no quantization),
so the verifier reads `output` directly and compares against the source bf16
with exact equality (no fp8 tolerance).

Modes:
  manual  -- 2 tokens (all-ones, all-twos), each routed to a single expert on
             the matching rank.
  random  -- Random bf16 tokens, each routed to exactly one randomly chosen
             expert. Verifies that every received token equals the source bf16.

Usage (2 GPUs):
    torchrun --nproc_per_node=2 _run_a2av_bf16.py --mode manual
    torchrun --nproc_per_node=2 _run_a2av_bf16.py --mode random --ntokens 32 --hidden 128
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

os.environ.setdefault("UBX_GRAPH_POOL_SHARE", "0.1")


def get_rank_info():
    if "RANK" in os.environ:
        rank       = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    elif "OMPI_COMM_WORLD_RANK" in os.environ:
        rank       = int(os.environ["OMPI_COMM_WORLD_RANK"])
        world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
        local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", rank))
    elif "SLURM_PROCID" in os.environ:
        rank       = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        local_rank = int(os.environ.get("SLURM_LOCALID", rank))
    else:
        raise RuntimeError("Cannot determine rank")
    return rank, world_size, local_rank


def init_distributed():
    rank, world_size, local_rank = get_rank_info()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
    return rank, world_size, local_rank


def run_manual_test():
    """2 ranks, 1 expert per rank, 1 token each (all-ones / all-twos)."""
    rank, world_size, local_rank = init_distributed()
    assert world_size == 2, "manual test requires exactly 2 ranks"
    device = torch.device(f"cuda:{local_rank}")

    hidden           = 32
    experts_per_rank = 1
    total_experts    = world_size * experts_per_rank
    ntokens          = 1

    fill_val    = float(rank + 1)
    tokens_bf16 = torch.full((ntokens, hidden), fill_val,
                             dtype=torch.bfloat16, device=device)

    token_offsets = torch.full((ntokens, total_experts), -1,
                               dtype=torch.int32, device=device)
    token_offsets[0, rank] = 0

    pool_bytes = 8 * 1024 * 1024
    from ubx import SymmAllocator
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    max_slots_per_expert = 1
    total_slots = experts_per_rank * max_slots_per_expert
    output = allocator.create_tensor(
        [total_slots, hidden], torch.bfloat16)

    print(f"[rank{rank}] manual test: sending token fill={fill_val}", flush=True)

    output = allocator.a2av_token_bf16_bf16(
        tokens_bf16, token_offsets, experts_per_rank, output,
    )
    torch.cuda.synchronize()

    received = output.float().cpu()

    print(f"[rank{rank}] received slot 0, first 8 values: "
          f"{received[0, :8].tolist()}", flush=True)

    try:
        torch.testing.assert_close(received, torch.full_like(received, fill_val),
                                   atol=0.0, rtol=0.0)
        print(f"PASS rank={rank} mode=manual fill={fill_val}", flush=True)
    except AssertionError as e:
        print(f"FAIL rank={rank} mode=manual: {e}", flush=True)
        sys.exit(1)

    dist.destroy_process_group()


def run_random_test(args):
    """Random bf16 tokens, each routed to exactly one random expert.

    Mirrors the mxfp8 random test but with exact bf16 comparison.
    """
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    hidden           = args.hidden
    local_ntokens    = args.ntokens
    experts_per_rank = args.experts_per_rank
    total_experts    = world_size * experts_per_rank
    global_ntokens   = local_ntokens * world_size
    assert hidden % 32 == 0, "hidden must be a multiple of 32"

    torch.manual_seed(args.seed + rank)
    tokens_bf16 = torch.randn(local_ntokens, hidden, dtype=torch.bfloat16, device=device)

    all_tokens_list = [torch.empty_like(tokens_bf16) for _ in range(world_size)]
    dist.all_gather(all_tokens_list, tokens_bf16)
    all_tokens = torch.cat(all_tokens_list, dim=0)

    torch.manual_seed(args.seed)
    routing = torch.zeros(global_ntokens, total_experts, dtype=torch.uint8, device=device)
    expert_ids = torch.randint(0, total_experts, (global_ntokens,), device=device)
    routing[torch.arange(global_ntokens, device=device), expert_ids] = 1

    from ubx import compute_token_offsets
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, rank, world_size)

    pool_bytes = 16 * 1024 * 1024
    from ubx import SymmAllocator
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    total_slots = max_tokens_per_rank
    output = allocator.create_tensor(
        [total_slots, hidden], torch.bfloat16)

    print(f"[rank{rank}] random test: local_ntokens={local_ntokens} hidden={hidden} "
          f"total_experts={total_experts} max_tokens_per_rank={max_tokens_per_rank}",
          flush=True)

    if getattr(args, 'topk', False):
        from ubx import compute_dispatch_topk_map
        topk_expert, topk_slot, topk_max = compute_dispatch_topk_map(
            routing, token_offsets, experts_per_rank, rank, world_size)
        # NOTE: compute_token_offsets returns the FULL routing's token_offsets
        # (global rows). compute_dispatch_topk_map slices to this rank's local
        # tokens internally — output shape is [local_ntokens, topk_max].
        # The base path takes the FULL token_offsets and slices in the kernel
        # via my_token_start; the topk path is already local.
        print(f"[rank{rank}] topk_max={topk_max} "
              f"topk_expert.shape={list(topk_expert.shape)}", flush=True)
        output = allocator.a2av_token_bf16_bf16_topk(
            tokens_bf16, topk_expert, topk_slot, experts_per_rank, output,
        )
    else:
        output = allocator.a2av_token_bf16_bf16(
            tokens_bf16, token_offsets, experts_per_rank, output,
        )
    torch.cuda.synchronize()

    received = output.float().cpu()

    my_expert_start = rank * experts_per_rank
    my_expert_end   = my_expert_start + experts_per_rank

    errors = []
    for src_rank in range(world_size):
        src_offsets, _, _, _ = compute_token_offsets(
            routing, experts_per_rank, src_rank, world_size)
        src_start = src_rank * local_ntokens
        for local_t in range(local_ntokens):
            global_t = src_start + local_t
            e = expert_ids[global_t].item()
            if not (my_expert_start <= e < my_expert_end):
                continue
            slot = src_offsets[local_t, e].item()
            if slot < 0:
                continue
            orig = all_tokens[global_t].float().cpu()
            got  = received[slot]

            if not torch.allclose(got, orig, atol=0.0, rtol=0.0):
                max_err = (got - orig).abs().max().item()
                errors.append(
                    f"global_token {global_t} expert {e} slot {slot}: "
                    f"max_err={max_err:.4g}"
                )

    if errors:
        for err in errors:
            print(f"FAIL rank={rank} mode=random: {err}", flush=True)
        sys.exit(1)

    print(f"PASS rank={rank} mode=random local_ntokens={local_ntokens} hidden={hidden}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["manual", "random"])
    parser.add_argument("--ntokens",           type=int, default=16)
    parser.add_argument("--hidden",            type=int, default=128)
    parser.add_argument("--experts_per_rank",  type=int, default=2)
    parser.add_argument("--seed",              type=int, default=42)
    parser.add_argument("--topk",              action="store_true",
                        help="Use the top-K LUT kernel "
                             "(ubx_a2av_token_bf16_bf16_topk) instead of the base one")
    args = parser.parse_args()

    if args.mode == "manual":
        run_manual_test()
    else:
        run_random_test(args)
