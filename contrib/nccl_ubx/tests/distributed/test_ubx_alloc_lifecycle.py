"""Regression test: SymmAllocator must not leak its bootstrapped ncclComm_t.

Pre-fix, every SymmAllocator instance leaked a fresh ncclComm_t and its
device-side state. The leak only manifested as OOM at large LSA sizes
but is observable as memory drift at any scale.

The test creates and destroys 20 SymmAllocators in a row and asserts that
GPU free memory stays within a fixed leak budget. With the fix in place
the leak is essentially zero; without it the trend is monotonic and the
test fails on a single node within a few iterations at the configured
budget.
"""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_alloc_lifecycle.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAllocLifecycle:
    """Guard against the SymmAllocator comm-ownership leak."""

    def test_no_leak_across_iters(self):
        """20 SymmAllocators in a row stay within a per-rank leak budget."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--iters", "20", "--pool-mib", "64", "--leak-mb", "100"],
            timeout=300,
        )
        assert result.returncode == 0, (
            f"alloc lifecycle leaked or failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "PASS" in result.stdout

    def test_no_leak_no_multicast(self):
        """Same guard with NCCL_NVLS_ENABLE=0 (NVLS-disabled code path)."""
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--iters", "20", "--pool-mib", "64", "--leak-mb", "100"],
            env_extra={"NCCL_NVLS_ENABLE": "0"},
            timeout=300,
        )
        assert result.returncode == 0, (
            f"alloc lifecycle (no-MC) leaked or failed\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
        assert "PASS" in result.stdout
