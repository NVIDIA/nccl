"""Default parameter ranges and configuration."""

from dataclasses import dataclass, field
from typing import List, Optional


def parse_size(s: str) -> int:
    """Parse a size string like '32M', '1G', '8K' into bytes."""
    s = s.strip().upper()
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


@dataclass
class BenchConfig:
    """Benchmark configuration matching nccl-tests CLI interface."""
    collective: str = "all_reduce"

    # Size sweep
    minbytes: int = 32 * 1024 * 1024   # 32M
    maxbytes: int = 32 * 1024 * 1024   # 32M
    stepbytes: int = 1024 * 1024        # 1M
    stepfactor: Optional[int] = None    # multiplicative (overrides stepbytes)

    # Timing
    iters: int = 20
    warmup_iters: int = 5

    # GPU/Process
    ngpus: int = 1
    nthreads: int = 1

    # Data
    datatype: str = "bf16"
    op: str = "sum"

    # ubx-specific
    backend: str = "all"                # ubx, nccl, all
    kernel: str = "auto"                # mc, uc, lamport, rw, rr, auto
    smlimit: int = 0
    nthreads_per_block: int = 0         # 0 = launcher default. Currently used
                                        # by the alltoall paths (uc / lamport /
                                        # auto).
    cgasize: int = 0
    num_chunks: int = 1
    num_comm_sm: int = 16

    # alltoallv-specific
    alpha: float = 0.5

    # a2av_mxfp8-specific
    min_tokens: int = 128
    max_tokens: int = 8192
    token_factor: int = 2
    hidden: int = 4096
    experts_per_rank: int = 1
    topk: int = 2
    routing_alpha: float = 2.0

    # Fused operations
    fused: str = "none"                 # none, residual, residual+rmsnorm
    hidden_size: int = 0

    # Graph/Correctness
    cudagraph: int = 10000
    check: int = 1

    # Output
    output_file: Optional[str] = None

    def size_sweep(self) -> List[int]:
        """Generate the list of message sizes to benchmark."""
        sizes = []
        size = self.minbytes
        while size <= self.maxbytes:
            sizes.append(size)
            if self.stepfactor and self.stepfactor > 1:
                size *= self.stepfactor
            else:
                size += self.stepbytes
        return sizes

    def token_sweep(self) -> List[int]:
        """Generate token count list for a2av_mxfp8 sweep."""
        counts = []
        n = self.min_tokens
        while n <= self.max_tokens:
            counts.append(n)
            n *= self.token_factor
        return counts

    def backends_to_test(self) -> List[str]:
        """Parse the backend string into a list."""
        if self.collective == "a2av_dispatch":
            if self.backend == "all":
                return ["ubx_bf16", "ubx_bf16_topk", "ubx_mxfp8",
                        "nccl_ep_ll", "nccl_ep_ht"]
            return [b.strip() for b in self.backend.split(",")]
        if self.collective == "a2av_combine":
            if self.backend == "all":
                return ["ubx_bf16", "ubx_mxfp8", "nccl_ep_ll", "nccl_ep_ht"]
            return [b.strip() for b in self.backend.split(",")]
        if self.backend == "all":
            return ["ubx", "nccl"]
        return [b.strip() for b in self.backend.split(",")]
