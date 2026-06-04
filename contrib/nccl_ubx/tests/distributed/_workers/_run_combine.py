"""Worker script for UB-X combine kernel correctness tests.

Combine is the reverse of a2av_token dispatch: gathers expert outputs back
to originating ranks and sums the top-K contributions per token, in bf16.

Modes:
  random      -- Random tokens + routing. Tests combine in isolation by
                 first running dispatch (bf16 wire) to populate expert
                 outputs, then running combine, then verifying combined ==
                 sum over each token's K contributions of original token
                 (since the "expert" is identity).
  gate        -- Same as random but with non-trivial gate_weights drawn
                 from softmax(rand). Reference applies the same weights.
  round_trip  -- Dispatch → identity-expert → combine with gate=1/K.
                 Expected: combined ≈ original tokens.

Wire format selected via --wire {bf16, mxfp8}. bf16 wire is bit-exact;
mxfp8 wire allows fp8 quantization tolerance.

Usage (4 GPUs):
    torchrun --nproc_per_node=4 _run_combine.py --mode random --wire bf16 \\
        --ntokens 32 --hidden 256 --experts_per_rank 2 --topk 2
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


def make_random_routing(global_ntokens: int, total_experts: int, topk: int,
                        device: torch.device, seed: int = 42):
    """Random topk routing matrix [global_ntokens, total_experts] uint8."""
    g = torch.Generator(device=device).manual_seed(seed)
    routing = torch.zeros(global_ntokens, total_experts,
                          dtype=torch.uint8, device=device)
    # Each token picks `topk` distinct experts uniformly at random.
    for t in range(global_ntokens):
        perm = torch.randperm(total_experts, generator=g, device=device)[:topk]
        routing[t, perm] = 1
    return routing


def reference_combine(global_tokens_bf16: torch.Tensor,
                      routing: torch.Tensor,
                      gate_weights_full: torch.Tensor,  # [global_ntokens, total_experts] or None
                      myrank: int,
                      local_ntokens: int) -> torch.Tensor:
    """Compute the expected combine output (top-K weighted sum per token).

    With identity experts: combined[t] = Σ_e (gate[t,e] if non-None else 1) * original[t]
                                       for e in routed-to set.
    """
    src_start = myrank * local_ntokens
    out = torch.zeros(local_ntokens, global_tokens_bf16.shape[1],
                      dtype=torch.float32, device=global_tokens_bf16.device)
    for local_t in range(local_ntokens):
        global_t = src_start + local_t
        for e in range(routing.shape[1]):
            if routing[global_t, e].item() == 0:
                continue
            w = gate_weights_full[global_t, e].item() if gate_weights_full is not None else 1.0
            out[local_t] += global_tokens_bf16[global_t].float() * w
    return out


def run_test(args):
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    hidden           = args.hidden
    local_ntokens    = args.ntokens
    experts_per_rank = args.experts_per_rank
    topk             = args.topk
    total_experts    = world_size * experts_per_rank
    global_ntokens   = local_ntokens * world_size
    assert hidden % 32 == 0, "hidden must be a multiple of 32"

    # Generate per-rank random tokens (bf16).
    torch.manual_seed(args.seed + rank)
    tokens_bf16 = torch.randn(local_ntokens, hidden,
                              dtype=torch.bfloat16, device=device)

    # All ranks see the global token tensor (for reference computation).
    all_tokens_list = [torch.empty_like(tokens_bf16) for _ in range(world_size)]
    dist.all_gather(all_tokens_list, tokens_bf16)
    all_tokens = torch.cat(all_tokens_list, dim=0)

    # Deterministic routing matrix (same on every rank).
    routing = make_random_routing(global_ntokens, total_experts, topk,
                                  device, seed=args.seed)

    from ubx import compute_token_offsets, SymmAllocator
    token_offsets, max_tokens_per_rank, _, _ = compute_token_offsets(
        routing, experts_per_rank, rank, world_size)

    # Optional gate weights.
    gate_weights = None
    gate_weights_full = None
    if args.mode == "gate" or args.mode == "round_trip":
        torch.manual_seed(args.seed + 1)
        # softmax over routed experts to give meaningful per-token weights summing to 1.
        gw_full = torch.full((global_ntokens, total_experts), -1e9,
                             dtype=torch.float32, device=device)
        for t in range(global_ntokens):
            for e in range(total_experts):
                if routing[t, e].item() != 0:
                    gw_full[t, e] = torch.randn(1, generator=None, device=device).item()
        gw_full = torch.softmax(gw_full, dim=1)
        # Zero out unrouted entries for cleanliness (post-softmax they're already ~0).
        gw_full = gw_full * routing.float()
        gate_weights_full = gw_full
        # This rank's slice of gate weights.
        gate_weights = gw_full[rank * local_ntokens : (rank + 1) * local_ntokens].contiguous()

    # Allocate enough symm pool for: (1) dispatch output [max_tpr, hidden] bf16,
    # (2) the temp combine buffer (bf16 or mxfp8) [max_tpr, hidden].
    # Headroom factor 4× for REG0 + alignment.
    bytes_per_dispatch = max_tokens_per_rank * hidden * 2  # bf16
    bytes_per_combine_temp = (
        max_tokens_per_rank * hidden  # fp8 = 1B/elem
        + max_tokens_per_rank * (hidden // 32)  # E8M0 1B/block
        if args.wire == "mxfp8"
        else max_tokens_per_rank * hidden * 2  # bf16
    )
    pool_bytes = max(64 * 1024 * 1024,
                     4 * (bytes_per_dispatch + bytes_per_combine_temp))
    allocator = SymmAllocator(pool_bytes, device, dist.group.WORLD)

    # Step 1: dispatch (bf16 wire) so expert_outputs are populated symmetrically.
    dispatch_out = allocator.create_tensor(
        [max_tokens_per_rank, hidden], torch.bfloat16)
    dispatch_out = allocator.a2av_token_bf16_bf16(
        tokens_bf16, token_offsets, experts_per_rank, dispatch_out)
    torch.cuda.synchronize()

    # The dispatch output IS our expert_outputs (identity FFN).
    # Pass it as a regular tensor: combine doesn't require SymmTensor input.
    expert_outputs = dispatch_out.detach().clone()  # plain torch.Tensor

    # Step 2: combine.
    sync = not args.async_combine
    use_lamport_push = (args.kernel == "lamport_push")
    use_push = (args.kernel == "push")
    if (use_lamport_push or use_push) and args.wire != "bf16":
        print(f"FAIL rank={rank}: --kernel {args.kernel} requires --wire bf16",
              flush=True)
        sys.exit(1)

    # PUSH variants need the inverse routing map and topk_idx.
    inverse_map = None
    topk_idx_t = None
    if use_lamport_push or use_push:
        from ubx import compute_combine_push_map
        inverse_map, topk_idx_t, _ = compute_combine_push_map(
            routing, experts_per_rank, rank, world_size)

    # Lamport-push / push variants run N_iter times to exercise warmup AND
    # steady state across the triple/double buffer rotation.
    n_iters = args.lamport_iters if (use_lamport_push or use_push) else 1
    for _ in range(n_iters):
        if use_push:
            combined = allocator.combine_bf16_bf16_push(
                expert_outputs, inverse_map, topk_idx_t,
                experts_per_rank, max_tokens_per_rank,
                gate_weights=gate_weights)
        elif use_lamport_push:
            combined = allocator.combine_bf16_bf16_lamport_push(
                expert_outputs, inverse_map, topk_idx_t,
                experts_per_rank, max_tokens_per_rank,
                gate_weights=gate_weights)
        elif args.wire == "bf16":
            combined = allocator.combine_bf16_bf16(
                expert_outputs, token_offsets, experts_per_rank,
                max_tokens_per_rank, gate_weights=gate_weights, sync=sync)
        elif args.wire == "mxfp8":
            combined = allocator.combine_mxfp8_bf16(
                expert_outputs, token_offsets, experts_per_rank,
                max_tokens_per_rank, gate_weights=gate_weights, sync=sync)
        else:
            raise ValueError(f"unknown wire format: {args.wire}")
        if not sync:
            allocator.combine_wait()
        torch.cuda.synchronize()

    # Reference.
    ref = reference_combine(all_tokens, routing, gate_weights_full,
                            rank, local_ntokens).cpu()
    got = combined.float().cpu()

    # Tolerances.
    if args.wire == "bf16":
        # bf16 dispatch is bit-exact; combine introduces fp32 accumulator
        # downcast so allow bf16 round-off only.
        atol, rtol = 0.02, 0.05
    else:
        # mxfp8: per-block scale × E4M3 mantissa noise × topk accumulation.
        atol = 0.0625 * topk * 4.0  # 1/16 per element × topk × headroom
        rtol = 0.05

    print(f"[rank{rank}] {args.mode}/{args.wire}: local_n={local_ntokens} "
          f"hidden={hidden} epr={experts_per_rank} topk={topk} "
          f"max_tpr={max_tokens_per_rank}",
          flush=True)

    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        diff = (got - ref).abs()
        max_err = diff.max().item()
        max_err_idx = diff.argmax().item()
        ref_at_max = ref.flatten()[max_err_idx].item()
        got_at_max = got.flatten()[max_err_idx].item()
        # Print up to 5 worst tokens.
        per_token_err = diff.view(local_ntokens, -1).max(dim=1).values
        topbad = per_token_err.argsort(descending=True)[:5]
        print(f"FAIL rank={rank} {args.mode}/{args.wire}: "
              f"max_err={max_err:.4g} (tol={atol}+{rtol}*|ref|)"
              f" at idx={max_err_idx} got={got_at_max:.4g} ref={ref_at_max:.4g}",
              flush=True)
        for t in topbad.tolist()[:5]:
            print(f"  token {t}: max_err={per_token_err[t].item():.4g}", flush=True)
        sys.exit(1)

    print(f"PASS rank={rank} {args.mode}/{args.wire}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["random", "gate", "round_trip"])
    parser.add_argument("--wire", required=True, choices=["bf16", "mxfp8"])
    parser.add_argument("--ntokens",          type=int, default=16)
    parser.add_argument("--hidden",           type=int, default=128)
    parser.add_argument("--experts_per_rank", type=int, default=2)
    parser.add_argument("--topk",             type=int, default=2)
    parser.add_argument("--seed",             type=int, default=42)
    parser.add_argument("--async-combine",    dest="async_combine",
                        action="store_true",
                        help="Use combine_*(sync=False) + combine_wait() path")
    parser.add_argument("--kernel",           default="auto",
                        choices=["auto", "lamport_push", "push"],
                        help="auto=PULL+barrier (combine_bf16_bf16); "
                             "lamport_push=PUSH-Lamport (small-msg); "
                             "push=PUSH+barrier (large-msg)")
    parser.add_argument("--lamport-iters",    dest="lamport_iters",
                        type=int, default=4,
                        help="Iterations to exercise lamport warmup + steady state")
    args = parser.parse_args()
    run_test(args)
