"""Multi-GPU tests for UB-X module-level convenience functions.

Tests the global allocator registry (request_allocator, get_sym_tensor,
allreduce, free_residual, restore, mem_stats).
"""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

CONV_WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_ubx_convenience.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestConvenienceFunctions:
    """Test module-level convenience API (global allocator registry)."""

    def test_request_and_get_tensor(self):
        """request_allocator() then get_sym_tensor() should return SymmTensor."""
        result = run_distributed_test(
            CONV_WORKER, num_procs=2,
            args=["--mode", "request_and_get_tensor"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_allreduce_convenience(self):
        """Module-level allreduce() should auto-select algorithm."""
        result = run_distributed_test(
            CONV_WORKER, num_procs=2,
            args=["--mode", "allreduce_convenience"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_restore_round_trip(self):
        """restore() should wrap a tensor back into SymmTensor if in pool."""
        result = run_distributed_test(
            CONV_WORKER, num_procs=2,
            args=["--mode", "restore_round_trip"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
