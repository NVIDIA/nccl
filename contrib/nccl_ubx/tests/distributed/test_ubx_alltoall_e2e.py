"""E2E validation: UB-X alltoall Lamport vs NCCL with random inputs.

Runs random inputs through both NCCL (dist.all_to_all) and UBX
(allocator.alltoall_lamport), compares outputs. Tests multiple sizes,
seeds, and consecutive calls (warmup + barrier-free steady state).
"""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_alltoall_e2e.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAlltoallLamportE2E:
    """E2E: random inputs, NCCL vs UBX Lamport, multiple sizes and seeds."""

    @pytest.mark.parametrize("size", [512, 1024, 4096, 16384, 65536, 262144, 1048576])
    def test_single_call(self, size):
        """Single Lamport call (call 0 = barrier + memset)."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--size", str(size), "--calls", "1", "--seeds", "3"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "ALL PASS" in result.stdout

    @pytest.mark.parametrize("size", [1024, 16384, 262144])
    def test_steady_state(self, size):
        """10 consecutive calls — covers warmup (barrier) + steady state (barrier-free)."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--size", str(size), "--calls", "10", "--seeds", "5"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "ALL PASS" in result.stdout

    @pytest.mark.parametrize("size", [1024, 65536])
    def test_4gpu(self, size):
        """4 GPUs — validates cross-rank correctness at larger scale."""
        result = run_distributed_test(
            WORKER, num_procs=4,
            args=["--size", str(size), "--calls", "5", "--seeds", "3"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "ALL PASS" in result.stdout
