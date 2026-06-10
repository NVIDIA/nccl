"""Multi-GPU tests for UB-X CUDA graph capture and replay.

SymmAllocator maintains a separate "graph pool" (90% of total) because CUDA
graph replay requires stable memory addresses. These tests verify graph pool
logic works correctly.
"""

import os
import pytest
from .conftest import run_distributed_test, requires_multi_gpu

GRAPH_WORKER = os.path.join(os.path.dirname(__file__), "_workers", "_run_ubx_graph.py")


@requires_multi_gpu(2)
@pytest.mark.requires_sm9
class TestCudaGraphs:
    """Test CUDA graph capture and replay with UB-X collectives."""

    def test_allreduce_in_graph(self):
        """Capture allreduce in CUDA graph, replay, verify correctness."""
        result = run_distributed_test(
            GRAPH_WORKER, num_procs=2,
            args=["--mode", "allreduce_in_graph"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_graph_multiple_replays(self):
        """10 replays of captured graph should all produce correct results."""
        result = run_distributed_test(
            GRAPH_WORKER, num_procs=2,
            args=["--mode", "multiple_replays"], timeout=180)
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout

    def test_graph_pool_allocation(self):
        """Graph-mode allocations should use graph pool (fixed offsets)."""
        result = run_distributed_test(
            GRAPH_WORKER, num_procs=2,
            args=["--mode", "graph_pool_allocation"])
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
        assert "PASS" in result.stdout
