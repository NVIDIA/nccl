"""CLI entry point: python -m ubx_bench <collective> [OPTIONS]

Matches nccl-tests CLI interface for easy comparison.
"""

import argparse
import sys

from .configs import BenchConfig, parse_size
from .runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(
        prog="ubx_bench",
        description="nccl-tests-compatible benchmarking suite for ubx",
    )
    parser.add_argument(
        "collective",
        choices=["all_reduce", "all_gather", "reduce_scatter", "all_to_all", "alltoallv", "a2av_mxfp8", "a2av_dispatch", "a2av_combine", "sendrecv"],
        help="Collective operation to benchmark",
    )

    # Size sweep (matching nccl-tests)
    parser.add_argument("-b", "--minbytes", type=parse_size, default="32M",
                        help="Minimum message size (default: 32M). Accepts K/M/G.")
    parser.add_argument("-e", "--maxbytes", type=parse_size, default="32M",
                        help="Maximum message size (default: 32M)")
    parser.add_argument("-i", "--stepbytes", type=parse_size, default="1M",
                        help="Additive step (default: 1M)")
    parser.add_argument("-f", "--stepfactor", type=int, default=None,
                        help="Multiplicative factor (overrides -i)")

    # Timing
    parser.add_argument("-n", "--iters", type=int, default=20,
                        help="Timed iterations (default: 20)")
    parser.add_argument("-w", "--warmup_iters", type=int, default=5,
                        help="Warmup iterations (default: 5)")

    # GPU/Process
    parser.add_argument("-g", "--ngpus", type=int, default=1,
                        help="GPUs per process (default: 1)")
    parser.add_argument("-t", "--nthreads", type=int, default=1,
                        help="Threads per process (default: 1)")

    # Data
    parser.add_argument("-d", "--datatype", default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="Data type (default: bf16)")
    parser.add_argument("-o", "--op", default="sum",
                        help="Reduction operation (default: sum)")

    # ubx-specific
    parser.add_argument("--backend", default="all",
                        help="ubx, nccl, all (default: all)")
    parser.add_argument("--kernel", default="auto",
                        choices=["mc", "uc", "lamport", "lamport_push", "push",
                                 "rw", "rr", "auto"],
                        help="Kernel variant (default: auto). "
                             "For a2av_combine: auto=PULL+barrier; "
                             "lamport_push=PUSH-Lamport (small-msg sweet spot); "
                             "push=PUSH+barrier (large-msg sweet spot).")
    parser.add_argument("--smlimit", type=int, default=0,
                        help="SM count limit for comm kernels (default: 0 = no limit)")
    parser.add_argument("--nthreads-per-block", type=int, default=0,
                        dest="nthreads_per_block",
                        help="Threads per block for the alltoall kernel "
                             "(default: 0 = launcher default 1024). Smaller "
                             "values reduce per-block overhead at small "
                             "messages; sweet spot varies with message size.")
    parser.add_argument("--cgasize", type=int, default=0,
                        help="CGA cluster size (default: 0 = no clustering)")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="Number of chunks for pipelined ops (default: 1)")
    parser.add_argument("--num-comm-sm", type=int, default=16,
                        help="Communication SMs limit (default: 16)")

    # Fused operations
    parser.add_argument("--fused", default="none",
                        choices=["none", "residual", "residual+rmsnorm"],
                        help="Fused operations (default: none)")
    parser.add_argument("--hidden-size", type=int, default=0,
                        help="Hidden dim for fused ops (required if fused != none)")

    # alltoallv-specific
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Power-law skew for alltoallv splits: 0=uniform, 0.5=moderate, 1.0=Zipf (default: 0.5)")

    # a2av_mxfp8-specific
    parser.add_argument("--min-tokens", type=int, default=128,
                        help="Min token count for a2av_mxfp8 sweep (default: 128)")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max token count for a2av_mxfp8 sweep (default: 8192)")
    parser.add_argument("--token-factor", type=int, default=2,
                        help="Multiplicative step for token sweep (default: 2)")
    parser.add_argument("--hidden", type=int, default=4096,
                        help="Hidden dimension for a2av_mxfp8 (must be multiple of 32, default: 4096)")
    parser.add_argument("--experts-per-rank", type=int, default=1,
                        help="Experts per rank for a2av_mxfp8 (default: 1)")
    parser.add_argument("--topk", type=int, default=2,
                        help="Top-k routing for a2av_mxfp8 (default: 2)")
    parser.add_argument("--routing-alpha", type=float, default=2.0,
                        help="Routing distribution skew for a2av_dispatch / "
                             "a2av_mxfp8: 0.0=uniform round-robin, "
                             "1.0=Zipfian, 2.0=heavily skewed (default; "
                             "approximates real MoE expert load skew)")

    # Graph/Correctness
    parser.add_argument("-G", "--cudagraph", type=int, default=10000,
                        help="CUDA graph replays per measurement (default: 10000). "
                             "Pass 0 or use --no-cudagraph for eager mode.")
    parser.add_argument("--no-cudagraph", dest="cudagraph",
                        action="store_const", const=0,
                        help="Disable CUDA graph capture; run in eager mode.")
    parser.add_argument("-c", "--check", type=int, default=1,
                        help="Correctness check iterations (default: 1)")

    # Output
    parser.add_argument("-J", "--output_file", default=None,
                        help="JSON output (if .json extension)")

    args = parser.parse_args()

    config = BenchConfig(
        collective=args.collective,
        minbytes=args.minbytes,
        maxbytes=args.maxbytes,
        stepbytes=args.stepbytes,
        stepfactor=args.stepfactor,
        iters=args.iters,
        warmup_iters=args.warmup_iters,
        ngpus=args.ngpus,
        nthreads=args.nthreads,
        datatype=args.datatype,
        op=args.op,
        backend=args.backend,
        kernel=args.kernel,
        smlimit=args.smlimit,
        nthreads_per_block=args.nthreads_per_block,
        cgasize=args.cgasize,
        alpha=args.alpha,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        token_factor=args.token_factor,
        hidden=args.hidden,
        experts_per_rank=args.experts_per_rank,
        topk=args.topk,
        routing_alpha=args.routing_alpha,
        num_chunks=args.num_chunks,
        num_comm_sm=args.num_comm_sm,
        fused=args.fused,
        hidden_size=args.hidden_size,
        cudagraph=args.cudagraph,
        check=args.check,
        output_file=args.output_file,
    )

    run_benchmark(config)


if __name__ == "__main__":
    main()
