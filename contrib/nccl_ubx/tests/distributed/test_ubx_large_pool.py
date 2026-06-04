"""Regression tests for the int / int64_t kernel-arg size mismatch.

Commit 95d5a6c (and earlier 17d0be7) fixed launchers where a 4-byte
`int arg4 = (int)(out_offset / sizeof(uint4))` was passed via
`cudaLaunchKernelExC` to a kernel whose signature declared
`int64_t lineoffset_out`. The kernel read 8 bytes from each kernel-arg
slot, so the upper 4 bytes were adjacent stack garbage → corrupted
lineoffset → OOB write → cudaErrorIllegalAddress.

These tests would have caught the bug at any pool size, because the
upper-half garbage corrupts `lineoffset_out` regardless of its true
value. The `large_pool` variants additionally verify the int64
arithmetic itself by pushing `out_offset` past INT_MAX / sizeof(uint4)
(the 32 GB threshold).
"""

import os

import pytest

from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_large_pool.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestLauncherInt64Fix:
    """Each op exercises a different launcher that was broken pre-95d5a6c."""

    @pytest.mark.parametrize("op", ["alltoall", "alltoall_lamport",
                                    "allgather_uc", "allreduce_uc"])
    def test_op_default_pool(self, op):
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--op", op, "--elems", "8192"],
            timeout=180,
        )
        assert result.returncode == 0, (
            f"op={op} returncode={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout, f"op={op} stdout: {result.stdout}"


@requires_multi_gpu(4)
@pytest.mark.requires_sm9
class TestAllLaunchersOneShot:
    """All four affected launchers in one allocator (catches stale-arg-slot
    interactions). 4-GPU run for the smaller grid that megatron uses."""

    def test_all_ops_default_pool(self):
        result = run_distributed_test(
            WORKER, num_procs=4,
            args=["--op", "all", "--elems", "8192"],
            timeout=240,
        )
        assert result.returncode == 0, (
            f"returncode={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        # Expect PASS for each op per rank.
        assert result.stdout.count("PASS") >= 16, (
            f"stdout: {result.stdout}"
        )


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
@pytest.mark.large_memory
class TestLargePoolInt64:
    """34 GiB pool — lineoffset_out at the high end of the pool exceeds
    INT_MAX / sizeof(uint4), so the kernel signature's int64_t precision
    is exercised end-to-end. Requires GPUs with ≥40 GiB free."""

    @pytest.mark.parametrize("op", ["alltoall", "alltoall_lamport",
                                    "allgather_uc", "allreduce_uc"])
    def test_op_large_pool(self, op):
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--op", op, "--elems", "65536", "--large-pool"],
            timeout=300,
        )
        assert result.returncode == 0, (
            f"op={op} returncode={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "PASS" in result.stdout, f"op={op} stdout: {result.stdout}"
