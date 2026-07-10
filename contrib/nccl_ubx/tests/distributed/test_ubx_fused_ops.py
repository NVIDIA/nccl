"""Multi-GPU tests for UB-X fused operations (residual + RMSNorm).

These test the fused paths where gamma/residual_in are non-null,
comparing against sequential (unfused) reference implementations.
"""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

FUSED_WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_ubx_fused.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestFusedOps:
    """Test fused residual + RMSNorm allreduce."""

    @pytest.mark.parametrize("hidden_size", [1024, 4096])
    def test_fused_residual_only(self, hidden_size):
        """Test allreduce with residual add but no RMSNorm."""
        result = run_distributed_test(
            FUSED_WORKER, num_procs=2,
            args=["--mode", "residual_only", "--hidden_size", str(hidden_size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    @pytest.mark.parametrize("hidden_size", [1024, 4096])
    def test_fused_rmsnorm(self, hidden_size):
        """Test allreduce with fused residual + RMSNorm."""
        result = run_distributed_test(
            FUSED_WORKER, num_procs=2,
            args=["--mode", "rmsnorm", "--hidden_size", str(hidden_size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_fused_vs_unfused_numerics(self):
        """Compare fused kernel output vs sequential ops."""
        result = run_distributed_test(
            FUSED_WORKER, num_procs=2,
            args=["--mode", "fused_vs_unfused"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
