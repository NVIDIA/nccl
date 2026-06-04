"""Multi-GPU tests for UB-X allgather."""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_ubx_op.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllgather:
    """Test multicast allgather."""

    @pytest.mark.parametrize("size", [1024, 8192, 65536])
    def test_allgather_basic(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allgather", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    @pytest.mark.parametrize("batch", [16, 64, 256])
    def test_allgather_shapes(self, batch):
        # Test different batch sizes with fixed hidden_size=128
        size = batch * 128
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allgather", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllgatherUC:
    """Test UC allgather (the multicast-free fallback path).

    This path is the only allgather available on NVLS-disabled EP groups or
    EP groups larger than the multicast hardware limit (36 ranks).
    Validating numerical equivalence to NCCL ensures the UC
    path is a drop-in replacement for the MC path.
    """

    @pytest.mark.parametrize("size", [1024, 8192, 65536])
    def test_allgather_uc_basic(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allgather_uc", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    @pytest.mark.parametrize("batch", [16, 64, 256])
    def test_allgather_uc_shapes(self, batch):
        size = batch * 128
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "allgather_uc", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
