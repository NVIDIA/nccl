"""Shared fixtures and skip markers for UB-X tests."""

import pytest
import torch
import os


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on hardware availability."""
    skip_no_cuda = pytest.mark.skip(reason="No CUDA GPU available")
    skip_no_sm9 = pytest.mark.skip(reason="Requires SM 9.0+ (Hopper/Blackwell)")

    has_cuda = torch.cuda.is_available()
    has_sm9 = False
    if has_cuda:
        props = torch.cuda.get_device_properties(0)
        has_sm9 = props.major >= 9

    for item in items:
        if "requires_cuda" in item.keywords and not has_cuda:
            item.add_marker(skip_no_cuda)
        if "requires_sm9" in item.keywords and not has_sm9:
            item.add_marker(skip_no_sm9)


# Custom markers
requires_cuda = pytest.mark.requires_cuda
requires_sm9 = pytest.mark.requires_sm9


@pytest.fixture
def nccl_backend():
    """UB-X's nccl4py backend module, with one skip-vs-fail policy.

    Skip only when nccl4py is genuinely not installed. When it IS installed but
    UB-X can no longer import what it needs, fail: that is the shape of the
    nccl4py 0.3.0 change, which shipped precisely because a broken build still
    looked healthy. Every test touching the backend goes through here, so the
    two cases can never drift apart between test classes.
    """
    from ubx import _nccl_backend as backend

    if backend.nccl4py_available():
        return backend
    err = backend.nccl4py_import_error()
    if backend.nccl4py_missing():
        pytest.skip(f"nccl4py is not installed here: {err!r}")
    pytest.fail(f"nccl4py is installed but UB-X cannot import from it: {err!r}")


@pytest.fixture
def cuda_device():
    """Provide a CUDA device for testing."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    return torch.device("cuda:0")


@pytest.fixture
def mock_pool(cuda_device):
    """Create a mock memory pool for single-GPU SymmTensor tests.

    Returns a uint8 tensor that simulates the symmetric memory pool
    without requiring distributed setup.
    """
    pool_size = 4 * 1024 * 1024  # 4MB
    pool = torch.zeros(pool_size, dtype=torch.uint8, device=cuda_device)
    return pool
