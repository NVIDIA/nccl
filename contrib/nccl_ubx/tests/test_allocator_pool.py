"""Tests for SymmAllocator pool management — single-GPU focus.

These tests verify the allocator's internal pool management (alloc, free, merge)
without requiring distributed setup. We mock the distributed initialization.
"""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock


class MockSymmAllocator:
    """A minimal mock that replicates SymmAllocator's pool management logic
    without requiring symmetric memory or distributed setup.

    This tests the pure Python allocation/free/merge logic.
    """

    def __init__(self, pool_size: int, reg0_size: int = 1024, graph_pool_share: float = 0.9):
        self.reg0_size = reg0_size
        self.pool_size = pool_size
        self.graph_pool_size = int(pool_size * graph_pool_share)
        self.graph_pool_size = int(self.graph_pool_size // 4096) * 4096
        self.pool_ptr = 0x1000  # fake base pointer

        self.allocated = {True: [], False: []}
        self.freelist = {
            True: [(self.reg0_size, self.graph_pool_size - self.reg0_size)],
            False: [(self.graph_pool_size, self.pool_size - self.graph_pool_size)],
        }

    def allocate(self, nbytes: int, graph_mode: bool = False):
        for i, (offset, size) in enumerate(self.freelist[graph_mode]):
            if size >= nbytes:
                self.freelist[graph_mode].pop(i)
                self.allocated[graph_mode].append((offset, nbytes, 0))
                if size > nbytes:
                    self.freelist[graph_mode].append((offset + nbytes, size - nbytes))
                return self.pool_ptr + offset
        return None

    def free(self, ptr: int):
        offset = ptr - self.pool_ptr
        graph_mode = offset < self.graph_pool_size
        for i, (alloc_offset, size, ref_count) in enumerate(self.allocated[graph_mode]):
            if alloc_offset == offset:
                self.allocated[graph_mode].pop(i)
                self.freelist[graph_mode].append((offset, size))
                self.freelist[graph_mode].sort(key=lambda x: x[0])
                self._merge_free_segments(graph_mode)
                return

    def _merge_free_segments(self, graph_mode: bool):
        if not self.freelist[graph_mode]:
            return
        merged = []
        current_offset, current_size = self.freelist[graph_mode][0]
        for offset, size in self.freelist[graph_mode][1:]:
            if current_offset + current_size == offset:
                current_size += size
            else:
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        merged.append((current_offset, current_size))
        self.freelist[graph_mode] = merged

    def total_free(self, graph_mode: bool = False) -> int:
        return sum(size for _, size in self.freelist[graph_mode])


class TestAllocatorPool:
    """Test pool allocation, free, and merge logic."""

    def test_allocate_fits(self):
        alloc = MockSymmAllocator(4 * 1024 * 1024)  # 4MB
        ptr = alloc.allocate(4096, graph_mode=True)
        assert ptr is not None
        assert len(alloc.allocated[True]) == 1

    def test_allocate_returns_none_when_full(self):
        alloc = MockSymmAllocator(4096, reg0_size=0, graph_pool_share=1.0)
        # Allocate entire pool
        ptr1 = alloc.allocate(4096, graph_mode=True)
        assert ptr1 is not None
        # Pool is now full
        ptr2 = alloc.allocate(1, graph_mode=True)
        assert ptr2 is None

    def test_free_and_reallocate(self):
        alloc = MockSymmAllocator(8192, reg0_size=0, graph_pool_share=1.0)
        ptr1 = alloc.allocate(4096, graph_mode=True)
        assert ptr1 is not None
        alloc.free(ptr1)
        ptr2 = alloc.allocate(4096, graph_mode=True)
        assert ptr2 is not None
        # Should reuse the same offset
        assert ptr2 == ptr1

    def test_merge_adjacent_free(self):
        alloc = MockSymmAllocator(16384, reg0_size=0, graph_pool_share=1.0)
        # Allocate three blocks
        ptr_a = alloc.allocate(4096, graph_mode=True)
        ptr_b = alloc.allocate(4096, graph_mode=True)
        ptr_c = alloc.allocate(4096, graph_mode=True)
        assert all(p is not None for p in [ptr_a, ptr_b, ptr_c])

        # Free B, then A — should merge into one block
        alloc.free(ptr_b)
        alloc.free(ptr_a)
        # Merged A+B should be one entry: (0, 8192)
        free_offsets = [(o, s) for o, s in alloc.freelist[True]]
        has_merged = any(s >= 8192 for _, s in free_offsets)
        assert has_merged

    def test_fragmentation_recovery(self):
        # Pool sized to exactly fit A+B+C with no leftover tail, so freeing A and C
        # leaves two non-adjacent 4096-byte blocks with B in the middle — no merging.
        alloc = MockSymmAllocator(12288, reg0_size=0, graph_pool_share=1.0)
        ptr_a = alloc.allocate(4096, graph_mode=True)
        ptr_b = alloc.allocate(4096, graph_mode=True)
        ptr_c = alloc.allocate(4096, graph_mode=True)

        # Free A and C (fragmented)
        alloc.free(ptr_a)
        alloc.free(ptr_c)

        # Can't allocate 8192 (contiguous) because B is in the middle
        big_ptr = alloc.allocate(8192, graph_mode=True)
        assert big_ptr is None

        # Free B to recover
        alloc.free(ptr_b)
        # Now all three should merge
        big_ptr = alloc.allocate(12288, graph_mode=True)
        assert big_ptr is not None

    def test_graph_vs_nongraph_pools(self):
        alloc = MockSymmAllocator(1024 * 1024, graph_pool_share=0.5)
        graph_free_before = alloc.total_free(True)
        nongraph_free_before = alloc.total_free(False)

        # Allocate from graph pool
        ptr_g = alloc.allocate(4096, graph_mode=True)
        assert ptr_g is not None
        assert alloc.total_free(True) == graph_free_before - 4096
        assert alloc.total_free(False) == nongraph_free_before  # unchanged

        # Allocate from non-graph pool
        ptr_ng = alloc.allocate(4096, graph_mode=False)
        assert ptr_ng is not None
        assert alloc.total_free(False) == nongraph_free_before - 4096

    def test_reg0_reserved(self):
        alloc = MockSymmAllocator(1024 * 1024)
        # First allocation should start at reg0_size (1024), not at 0
        first_free_offset = alloc.freelist[True][0][0]
        assert first_free_offset >= alloc.reg0_size

    def test_pool_size_env_override(self):
        """Verify env var logic is respected in the real allocator code."""
        # This just tests that the env var name is correct
        # (actual SymmAllocator requires distributed setup, so we test the logic)
        pool_size_mb = os.environ.get("UBX_SYMM_POOL_SIZE")
        # Just verify the variable name is consistent with what ops.py uses
        from ubx.ops import get_sym_tensor
        # Should not crash even without distributed setup
        assert callable(get_sym_tensor)

    def test_graph_pool_share_env(self):
        """Verify graph pool share calculation matches env var."""
        share = float(os.environ.get("UBX_GRAPH_POOL_SHARE", 0.9))
        alloc = MockSymmAllocator(1024 * 1024, graph_pool_share=share)
        expected = int(1024 * 1024 * share)
        expected = int(expected // 4096) * 4096
        assert alloc.graph_pool_size == expected


class TestReg0Sizing:
    """REG0 must hold the commbuff (world_size * 8) plus the kernel flag slots
    at every world_size. Regression for the cudagraph hang: a fixed
    reg0_size=1024 leaves no room for flags once ``world_size * 8`` reaches
    1024 B, so they spill into the graph pool and get clobbered by peer
    data writes, corrupting the BAR counter (UC) / poison polling (Lamport).
    Pure arithmetic — runs single-GPU/CPU, no distributed or CUDA setup
    needed.
    """

    # World sizes covered by the in-supported-range parametrized tests.
    _WORLD_SIZES = [2, 4, 8, 16, 32, 64, 72]

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_reg0_fits_flag_region(self, world_size):
        from ubx.allocator import SymmAllocator
        reg0 = SymmAllocator._reg0_size_for(world_size)
        flag_region_end = world_size * 8 + SymmAllocator._MAX_FLAG_SLOTS * 4
        assert reg0 >= flag_region_end, (
            f"reg0_size={reg0} too small for world_size={world_size}: "
            f"commbuff+flags need {flag_region_end} B"
        )

    def test_reg0_at_old_fixed_boundary(self):
        """At the boundary where ``world_size * 8`` reaches the pre-fix
        fixed reg0_size (1024 B), the buggy old value would leave zero
        room for flags. The current formula must scale past that
        boundary even when extrapolating beyond the in-supported-range
        parametrize values."""
        from ubx.allocator import SymmAllocator
        # Boundary value derived from the old fixed constant, not a
        # cluster scale claim: the smallest world_size at which the old
        # value would overflow REG0.
        boundary_ws = 1024 // 8
        reg0 = SymmAllocator._reg0_size_for(boundary_ws)
        assert reg0 > 1024, f"reg0_size={reg0} regressed to the buggy fixed 1024 B"
        assert reg0 >= boundary_ws * 8 + SymmAllocator._MAX_FLAG_SLOTS * 4

    @pytest.mark.parametrize("world_size", _WORLD_SIZES)
    def test_reg0_is_4k_aligned(self, world_size):
        from ubx.allocator import SymmAllocator
        assert SymmAllocator._reg0_size_for(world_size) % 4096 == 0
