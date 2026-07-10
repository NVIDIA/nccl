"""Smoke tests for ubx_bench — no GPU required."""

import pytest


class TestConfigs:
    """Test configuration parsing."""

    def test_parse_size_bytes(self):
        from ubx_bench.configs import parse_size
        assert parse_size("1024") == 1024
        assert parse_size("8K") == 8192
        assert parse_size("32M") == 32 * 1024 * 1024
        assert parse_size("1G") == 1024 * 1024 * 1024

    def test_parse_size_case_insensitive(self):
        from ubx_bench.configs import parse_size
        assert parse_size("8k") == 8192
        assert parse_size("32m") == 32 * 1024 * 1024

    def test_bench_config_defaults(self):
        from ubx_bench.configs import BenchConfig
        cfg = BenchConfig()
        assert cfg.collective == "all_reduce"
        assert cfg.iters == 20
        assert cfg.warmup_iters == 5
        assert cfg.datatype == "bf16"

    def test_size_sweep_additive(self):
        from ubx_bench.configs import BenchConfig
        cfg = BenchConfig(minbytes=8, maxbytes=32, stepbytes=8)
        sizes = cfg.size_sweep()
        assert sizes == [8, 16, 24, 32]

    def test_size_sweep_multiplicative(self):
        from ubx_bench.configs import BenchConfig
        cfg = BenchConfig(minbytes=8, maxbytes=64, stepfactor=2)
        sizes = cfg.size_sweep()
        assert sizes == [8, 16, 32, 64]

    def test_backends_to_test_all(self):
        from ubx_bench.configs import BenchConfig
        cfg = BenchConfig(backend="all")
        assert cfg.backends_to_test() == ["ubx", "nccl"]

    def test_backends_to_test_single(self):
        from ubx_bench.configs import BenchConfig
        cfg = BenchConfig(backend="ubx")
        assert cfg.backends_to_test() == ["ubx"]

    def test_backends_to_test_comma_separated(self):
        from ubx_bench.configs import BenchConfig
        cfg = BenchConfig(backend="ubx,nccl")
        assert cfg.backends_to_test() == ["ubx", "nccl"]


class TestReport:
    """Test report formatting."""

    def test_compute_bandwidth_allreduce(self):
        from ubx_bench.report import compute_bandwidth
        alg, bus = compute_bandwidth(1024 * 1024, 100.0, 8, "all_reduce")
        assert alg > 0
        assert bus > alg  # bus should be > alg for allreduce

    def test_compute_bandwidth_zero_time(self):
        from ubx_bench.report import compute_bandwidth
        alg, bus = compute_bandwidth(1024, 0.0, 8, "all_reduce")
        assert alg == 0.0
        assert bus == 0.0

    def test_format_table_basic(self):
        from ubx_bench.report import BenchResult, format_table
        results = {
            "nccl": [
                BenchResult(size_bytes=1024, count=512, dtype="bf16", redop="sum",
                            time_us=10.0, algbw_gbs=0.1, busbw_gbs=0.175),
            ]
        }
        table = format_table("all_reduce", 8, ["nccl"], results)
        assert "ubx_bench" in table
        assert "all_reduce" in table
        assert "nccl" in table
