"""Multi-GPU tests for UB-X a2av token bf16→mxfp8 dispatch kernel."""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_a2av_mxfp8.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestA2avTokenBf16Mxfp8:
    """Tests for ubx_kernel_a2av_token_bf16_mxfp8."""

    def test_manual_small(self):
        """2 tokens (all-ones / all-twos), 1 block each, routed to matching rank.

        Prints per-rank dequantized values and verifies they equal the original
        fill values within fp8 tolerance.
        """
        result = run_distributed_test(WORKER, num_procs=2, args=["--mode", "manual"])
        assert result.returncode == 0, (
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert result.stdout.count("PASS") >= 2

    @pytest.mark.parametrize("ntokens,hidden,experts_per_rank", [
        (8,   32, 1),
        (16, 128, 2),
        (32, 256, 2),
    ])
    def test_random(self, ntokens, hidden, experts_per_rank):
        """Random bf16 tokens, each routed to one random expert.

        Verifies that every received token dequantizes within the mxfp8
        quantization error bound.
        """
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=[
                "--mode",            "random",
                "--ntokens",         str(ntokens),
                "--hidden",          str(hidden),
                "--experts_per_rank", str(experts_per_rank),
            ],
        )
        assert result.returncode == 0, (
            f"ntokens={ntokens} hidden={hidden} epr={experts_per_rank}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert result.stdout.count("PASS") >= 2
