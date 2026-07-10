"""Tests for SymmTensor lifecycle — single-GPU, no distributed setup needed.

These tests use a mock allocator that provides the minimal interface
SymmTensor needs, without requiring actual symmetric memory or distributed init.
"""

import pytest
import torch
import weakref


class MockAllocator:
    """Minimal mock allocator for SymmTensor tests without distributed setup."""

    def __init__(self, pool: torch.Tensor):
        self.internal_pool = pool
        self.pool_ptr = pool.data_ptr()
        self.pool_size = pool.numel()
        self.tensors = weakref.WeakSet()
        self._allocated = {}  # offset -> ref_count
        self._freed = []

    def allocated_change(self, ptr: int, change: int):
        offset = ptr - self.pool_ptr
        if offset not in self._allocated:
            self._allocated[offset] = 0
        self._allocated[offset] += change
        if self._allocated[offset] <= 0:
            self._freed.append(offset)
            del self._allocated[offset]

    def free(self, ptr: int):
        self.allocated_change(ptr, -1)


@pytest.mark.requires_cuda
class TestSymmTensorCreation:
    """Test basic SymmTensor creation."""

    def test_create_basic_shapes(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        for shape in [(1,), (8, 8), (4, 16, 32)]:
            t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size(shape),
                                    torch.bfloat16, allocator)
            assert t.shape == torch.Size(shape)
            assert t.dtype == torch.bfloat16

    def test_dtype_support(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        for dtype in [torch.bfloat16, torch.float32, torch.float16]:
            t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([64]),
                                    dtype, allocator)
            assert t.dtype == dtype

    def test_data_ptr_in_pool(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 2048, torch.Size([32]),
                                torch.bfloat16, allocator)
        ptr = t.data_ptr()
        pool_start = mock_pool.data_ptr()
        pool_end = pool_start + mock_pool.numel()
        assert pool_start <= ptr < pool_end

    def test_fill_and_read(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 4096, torch.Size([16]),
                                torch.float32, allocator)
        t.fill_(42.0)
        assert torch.all(t == 42.0)

    def test_clone_is_regular(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 4096, torch.Size([16]),
                                torch.float32, allocator)
        t.fill_(7.0)
        cloned = t.clone()
        # clone() should return a regular tensor, not a SymmTensor
        assert not isinstance(cloned, SymmTensor)
        assert torch.all(cloned == 7.0)


@pytest.mark.requires_cuda
class TestSymmTensorView:
    """Test SymmTensor.view() reshape operations."""

    def test_view_reshape(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([4, 8]),
                                torch.bfloat16, allocator)
        v = t.view(32)
        assert isinstance(v, SymmTensor)
        assert v.shape == torch.Size([32])
        # Same data pointer (same backing memory)
        assert v.data_ptr() == t.data_ptr()

    def test_view_with_minus_one(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([4, 8]),
                                torch.bfloat16, allocator)
        v = t.view(-1, 4)
        assert v.shape == torch.Size([8, 4])
        assert isinstance(v, SymmTensor)

    def test_view_size_mismatch_raises(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([4, 8]),
                                torch.bfloat16, allocator)
        with pytest.raises(RuntimeError, match="View size mismatch"):
            t.view(100)

    def test_view_multiple_minus_one_raises(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([4, 8]),
                                torch.bfloat16, allocator)
        with pytest.raises(RuntimeError, match="Only one dimension"):
            t.view(-1, -1)


@pytest.mark.requires_cuda
class TestSymmTensorLifecycle:
    """Test SymmTensor memory management lifecycle."""

    def test_del_returns_memory(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([32]),
                                torch.bfloat16, allocator)
        assert len(allocator._freed) == 0
        del t
        assert len(allocator._freed) == 1

    def test_refcount_tracking(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        t1 = SymmTensor.__new__(SymmTensor, mock_pool, 1024, torch.Size([32]),
                                 torch.bfloat16, allocator)
        # Create a view — new SymmTensor at same offset
        t2 = t1.view(8, 4)
        # Both should be tracked
        assert len(allocator._allocated) == 1  # same offset, ref_count=2
        offset = 1024
        assert allocator._allocated.get(offset, 0) == 2

        del t1
        # After deleting t1, ref_count should be 1
        assert allocator._allocated.get(offset, 0) == 1
        assert len(allocator._freed) == 0

        del t2
        # After deleting t2, memory should be freed
        assert len(allocator._freed) == 1

    def test_pool_validation(self, mock_pool, cuda_device):
        from ubx.tensor import SymmTensor
        allocator = MockAllocator(mock_pool)

        # Trying to create a tensor that exceeds pool bounds should fail
        pool_size = mock_pool.numel()
        with pytest.raises(AssertionError):
            SymmTensor.__new__(SymmTensor, mock_pool, pool_size - 10,
                                torch.Size([1024]), torch.float32, allocator)

    def test_wrong_pool_dtype_raises(self, cuda_device):
        from ubx.tensor import SymmTensor
        wrong_pool = torch.zeros(1024, dtype=torch.float32, device=cuda_device)
        allocator = MockAllocator(wrong_pool)

        with pytest.raises(AssertionError, match="Expected uint8"):
            SymmTensor.__new__(SymmTensor, wrong_pool, 0, torch.Size([8]),
                                torch.float32, allocator)
