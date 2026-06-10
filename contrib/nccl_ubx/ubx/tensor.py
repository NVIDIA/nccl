"""SymmTensor: custom tensor subclass backed by symmetric memory pool."""

from __future__ import annotations

import torch
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .allocator import SymmAllocator


class SymmTensor(torch.Tensor):
    """Custom tensor subclass that uses symmetric memory pool backing.

    For blocked formats (e.g. 'mxfp8'), the allocation in the pool is larger
    than the visible tensor data: it includes an aligned data region followed
    by a metadata (scales) region.  The tensor shape and dtype describe only
    the data elements; metadata is accessed via ``metadata_ptr`` /
    ``metadata_offset``.
    """

    @staticmethod
    def __new__(
        cls,
        pool: torch.Tensor,
        offset: int,
        shape: torch.Size,
        dtype: torch.dtype,
        allocator: "SymmAllocator",
        blocked_format: Optional[str] = None,
        metadata_offset: Optional[int] = None,
    ):
        # Calculate number of elements and bytes for the data region only
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor(0, dtype=dtype).element_size()
        nbytes = element_size * num_elements

        # Total bytes reserved in the pool (data + metadata for blocked formats)
        total_nbytes = nbytes
        if blocked_format == "mxfp8":
            if metadata_offset is None:
                raise ValueError("metadata_offset is required when blocked_format='mxfp8'")
            metadata_nbytes = (num_elements + 31) // 32  # 1 scale byte per 32 elements
            total_nbytes = metadata_offset + metadata_nbytes

        # Validate pool
        assert pool.dtype == torch.uint8, f"Expected uint8 pool, got {pool.dtype}"
        assert (
            pool.numel() >= offset + total_nbytes
        ), f"Pool too small: {pool.numel()} bytes, need {offset + total_nbytes}"

        # Slice only the data bytes for the tensor view
        byte_slice = pool[offset : offset + nbytes]

        # Reinterpret the uint8 bytes as the target dtype
        tensor = byte_slice.view(dtype=dtype)
        tensor = tensor.view(*shape)

        # Initialize as a subclass of torch.Tensor
        self = torch.Tensor._make_subclass(cls, tensor)
        # Duck-type check: require the allocator interface without forcing SymmAllocator
        # subclass, so mock allocators work in tests.
        required = ("tensors", "allocated_change", "free", "internal_pool")
        missing = [attr for attr in required if not hasattr(allocator, attr)]
        if missing:
            raise TypeError(
                f"allocator missing required attributes: {missing} "
                f"(got {type(allocator)})"
            )
        self._allocator = allocator
        self._ptr = tensor.data_ptr()
        self._offset = offset
        self._size = total_nbytes
        self._blocked_format = blocked_format
        self._metadata_offset = metadata_offset  # byte offset from data_ptr to scales
        allocator.tensors.add(self)
        allocator.allocated_change(self._ptr, 1)
        return self

    @property
    def blocked_format(self) -> Optional[str]:
        """Blocked data format ('mxfp8') or None for a plain tensor."""
        return self._blocked_format

    @property
    def metadata_offset(self) -> Optional[int]:
        """Byte offset from data_ptr() to the start of the metadata (scales) region.

        None for plain (non-blocked) tensors.
        """
        return self._metadata_offset

    @property
    def metadata_ptr(self) -> Optional[int]:
        """Raw integer pointer to the metadata (scales) region.

        None for plain (non-blocked) tensors.
        """
        if self._metadata_offset is None:
            return None
        return self.data_ptr() + self._metadata_offset

    def clone(self, *, memory_format=torch.preserve_format):
        """Return a plain torch.Tensor copy — clones are not backed by symmetric memory."""
        return self.as_subclass(torch.Tensor).clone(memory_format=memory_format)

    def __del__(self):
        """Custom deallocator to return memory to the pool."""
        if hasattr(self, "_allocator") and hasattr(self, "_ptr"):
            self._allocator.free(self._ptr)

    def view(self, *shape):
        """
        Returns a new SymmTensor view with the same backing memory but different shape.
        Handles -1 inference manually without temporary tensors.

        Note: view is not supported on blocked-format tensors because reshaping
        would invalidate the fixed metadata_offset layout.
        """
        if not hasattr(self, '_allocator'):
            return self.as_subclass(torch.Tensor).view(*shape)
        if self._blocked_format is not None:
            raise RuntimeError(
                f"view() is not supported on blocked-format tensors "
                f"(blocked_format={self._blocked_format!r})"
            )
        # Convert shape to list for modification
        shape_list = list(shape)
        if len(shape_list) == 1 and isinstance(shape_list[0], (tuple, list)):
            shape_list = list(shape_list[0])

        original_numel = self.numel()

        # Handle -1 inference
        has_infer_dim = -1 in shape_list
        if has_infer_dim:
            if shape_list.count(-1) > 1:
                raise RuntimeError("Only one dimension can be inferred (contains -1)")

            infer_idx = shape_list.index(-1)
            known_size = 1
            for i, dim in enumerate(shape_list):
                if i != infer_idx:
                    if dim < 0:
                        raise RuntimeError("Only -1 is supported for inference")
                    known_size *= dim

            if original_numel % known_size != 0:
                raise RuntimeError(
                    f"Shape inference failed: {original_numel} elements not divisible by {known_size}"
                )

            inferred_dim = original_numel // known_size
            shape_list[infer_idx] = inferred_dim
            resolved_shape = torch.Size(shape_list)
        else:
            resolved_shape = torch.Size(shape_list)

        # Validate total elements match
        if resolved_shape.numel() != original_numel:
            raise RuntimeError(
                f"View size mismatch: original has {original_numel} elements, "
                f"resolved shape {resolved_shape} requires {resolved_shape.numel()} elements"
            )

        # Create SymmTensor with resolved shape, same backing memory
        new_tensor = SymmTensor.__new__(
            SymmTensor,
            self._allocator.internal_pool,
            self._offset,
            resolved_shape,
            self.dtype,
            self._allocator,
            blocked_format=None,
            metadata_offset=None,
        )
        return new_tensor


torch.serialization.add_safe_globals([SymmTensor])
