"""Multi-GPU tests for UB-X alltoallv (variable-length alltoall)."""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_alltoallv.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAlltoallv:
    """Test variable-length alltoall with power-law distributions."""

    @pytest.mark.parametrize("total_elems", [1024, 8192, 65536])
    def test_alltoallv_uniform(self, total_elems):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--total-elems", str(total_elems),
                                            "--alpha", "0.0"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    @pytest.mark.parametrize("total_elems", [1024, 8192, 65536])
    def test_alltoallv_moderate_skew(self, total_elems):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--total-elems", str(total_elems),
                                            "--alpha", "0.5"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    @pytest.mark.parametrize("total_elems", [1024, 8192, 65536])
    def test_alltoallv_zipf_skew(self, total_elems):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--total-elems", str(total_elems),
                                            "--alpha", "1.0"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(4)
@pytest.mark.requires_sm9
class TestAlltoallvMultiGPU:
    """Test alltoallv with 4 GPUs."""

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_alltoallv_4gpu(self, alpha):
        result = run_distributed_test(WORKER, num_procs=4,
                                      args=["--total-elems", "16384",
                                            "--alpha", str(alpha)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
