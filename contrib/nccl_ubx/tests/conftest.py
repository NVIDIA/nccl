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
