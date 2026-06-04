"""Tests for top-level imports, version, and hardware detection."""

import pytest


class TestImports:
    """Top-level package surface."""

    def test_version_string(self):
        from ubx import __version__
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3

    def test_top_level_exports(self):
        import ubx
        assert hasattr(ubx, "SymmAllocator")
        assert hasattr(ubx, "SymmTensor")
        assert hasattr(ubx, "compute_token_offsets")
        assert hasattr(ubx, "ops")
        assert hasattr(ubx, "fused")


class TestHardwareDetection:
    """Hardware probe utilities."""

    def test_get_cuda_sm_version_no_crash(self):
        from ubx._common import _get_cuda_sm_version
        major, minor = _get_cuda_sm_version()
        assert isinstance(major, int)
        assert isinstance(minor, int)

    @pytest.mark.requires_cuda
    def test_sm_version_positive(self):
        from ubx._common import _get_cuda_sm_version
        major, minor = _get_cuda_sm_version()
        assert major > 0
