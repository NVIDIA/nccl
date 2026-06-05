"""Multi-GPU tests for UB-X combine kernels (bf16 and mxfp8 wire formats)."""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_combine.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestCombine:
    """Tests for ubx_combine_bf16_bf16 and ubx_combine_mxfp8_bf16."""

    @pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
    @pytest.mark.parametrize("ntokens,hidden,experts_per_rank,topk", [
        (8,    64, 2, 2),
        (16,  128, 2, 2),
        (32,  256, 4, 4),
    ])
    def test_random(self, wire, ntokens, hidden, experts_per_rank, topk):
        """Combine identity expert outputs; verify top-K sum reconstructs."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=[
                "--mode", "random",
                "--wire", wire,
                "--ntokens",          str(ntokens),
                "--hidden",           str(hidden),
                "--experts_per_rank", str(experts_per_rank),
                "--topk",             str(topk),
            ],
        )
        assert result.returncode == 0, (
            f"wire={wire} ntokens={ntokens} hidden={hidden} "
            f"epr={experts_per_rank} topk={topk}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert result.stdout.count("PASS") >= 2

    @pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
    def test_gate_weights(self, wire):
        """Random + non-trivial gate weights (softmax over routed experts)."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=[
                "--mode", "gate",
                "--wire", wire,
                "--ntokens", "16",
                "--hidden",  "128",
                "--experts_per_rank", "2",
                "--topk",    "2",
            ],
        )
        assert result.returncode == 0, (
            f"wire={wire}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert result.stdout.count("PASS") >= 2

    @pytest.mark.parametrize("kernel", ["lamport_push", "push"])
    @pytest.mark.parametrize("ntokens,hidden,experts_per_rank,topk", [
        (8,    64, 2, 2),
        (16,  128, 2, 2),
        (32,  256, 4, 4),
    ])
    def test_push_variants(self, kernel, ntokens, hidden, experts_per_rank, topk):
        """PUSH-semantics combine — Lamport (lamport_push) and barrier (push) variants, 4 iters."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=[
                "--mode", "random",
                "--wire", "bf16",
                "--kernel", kernel,
                "--ntokens",          str(ntokens),
                "--hidden",           str(hidden),
                "--experts_per_rank", str(experts_per_rank),
                "--topk",             str(topk),
                "--lamport-iters",    "4",
            ],
        )
        assert result.returncode == 0, (
            f"kernel={kernel} ntokens={ntokens} hidden={hidden} "
            f"epr={experts_per_rank} topk={topk}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert result.stdout.count("PASS") >= 2

    @pytest.mark.parametrize("wire", ["bf16", "mxfp8"])
    def test_async(self, wire):
        """combine_*(sync=False) + combine_wait() path produces same output."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=[
                "--mode", "random",
                "--wire", wire,
                "--ntokens", "16",
                "--hidden",  "128",
                "--experts_per_rank", "2",
                "--topk",    "2",
                "--async-combine",
            ],
        )
        assert result.returncode == 0, (
            f"wire={wire} async\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert result.stdout.count("PASS") >= 2
