"""Multi-GPU tests for UB-X alltoall."""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_ubx_op.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAlltoall:
    """Test unicast alltoall."""

    @pytest.mark.parametrize("size", [1024, 8192, 65536])
    def test_alltoall_basic(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "alltoall", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAlltoallLamport:
    """Test unicast alltoall with Lamport polling."""

    @pytest.mark.parametrize("size", [1024, 8192, 65536])
    def test_alltoall_lamport_basic(self, size):
        result = run_distributed_test(WORKER, num_procs=2,
                                      args=["--op", "alltoall_lamport", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(4)
@pytest.mark.requires_sm9
class TestAlltoallFullNode:
    """Test alltoall with 4 GPUs (full node)."""

    @pytest.mark.parametrize("size", [1024, 8192, 65536])
    def test_alltoall_4gpu(self, size):
        result = run_distributed_test(WORKER, num_procs=4,
                                      args=["--op", "alltoall", "--size", str(size)])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAlltoallNoMulticast:
    """Test alltoall with multicast disabled (multicast-disabled simulation)."""

    @pytest.mark.parametrize("size", [1024, 8192, 65536, 262144, 1048576])
    def test_alltoall_no_mc(self, size):
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--op", "alltoall", "--size", str(size)],
            env_extra={"NCCL_NVLS_ENABLE": "0"},
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestAlltoallLamportNoMulticast:
    """Test Lamport alltoall with multicast disabled (multicast-disabled simulation)."""

    @pytest.mark.parametrize("size", [1024, 8192, 65536, 262144, 1048576])
    def test_alltoall_lamport_no_mc(self, size):
        result = run_distributed_test(
            WORKER, num_procs=2,
            args=["--op", "alltoall_lamport", "--size", str(size)],
            env_extra={"NCCL_NVLS_ENABLE": "0"},
        )
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
