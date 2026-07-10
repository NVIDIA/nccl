"""Multi-GPU tests for UB-X allreduce variants.

Tests are launched via subprocess using srun/mpirun/torchrun.
Reference: torch.distributed.all_reduce (NCCL backend).
"""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_ubx_op.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllreduceMC:
    """Test multicast allreduce for large messages."""

    @pytest.mark.parametrize("size", [2 * 1024 * 1024, 8 * 1024 * 1024])
    def test_allreduce_mc_large(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allreduce_mc", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllreduceLamport:
    """Test Lamport-synchronized allreduce for small messages."""

    @pytest.mark.parametrize("size", [64, 1024, 512 * 1024, 1024 * 1024])
    def test_allreduce_lamport_small(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allreduce_lamport", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllreduceUC:
    """Test unicast fallback allreduce."""

    @pytest.mark.parametrize("size", [1024, 1024 * 1024])
    def test_allreduce_uc_fallback(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allreduce_uc", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllreduceAuto:
    """Test auto-selecting allreduce (mc for large, lamport for small)."""

    @pytest.mark.parametrize("size", [1024, 2 * 1024 * 1024])
    def test_allreduce_auto(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allreduce_auto", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(4)
@pytest.mark.requires_sm9
class TestAllreduceRankScaling:
    """Test allreduce with different GPU counts."""

    @pytest.mark.parametrize("num_procs", [2, 4])
    def test_allreduce_ranks(self, num_procs):
        result = run_distributed_test(WORKER, num_procs=num_procs,
                                      args=["--op", "allreduce_auto", "--size", "65536"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
