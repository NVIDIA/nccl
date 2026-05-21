# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor descriptor: ``Tensor`` wrapping ``ncclEpTensor_t``."""

from __future__ import annotations

import numpy as np

from nccl.core.typing import NcclInvalid

from nccl.bindings import nccl_ep as _ep_bindings


__all__ = ["Tensor"]


class Tensor:
    """Caller-owned device buffer wrapped as an ``ncclEpTensor_t``.

    The descriptor (and its ``sizes`` array) are library-allocated via
    :c:func:`ncclEpTensorAlloc`; the underlying device buffer (or NCCL
    window registration) is caller-owned and must outlive :meth:`destroy`.
    """

    def __init__(
        self,
        alloc_ptr: int,
        ndim: int = 0,
        sizes: tuple[int, ...] = (),
    ) -> None:
        self._alloc_ptr = alloc_ptr
        self._ndim = ndim
        self._sizes = sizes
        # In-place view over the library-allocated struct (owner=self so
        # the binding wrapper does not copy or free the memory).
        self._view = (
            _ep_bindings.Tensor.from_ptr(alloc_ptr, owner=self)
            if alloc_ptr
            else None
        )

    @classmethod
    def create(
        cls,
        ndim: int,
        datatype: int,
        data: int,
        *sizes: int,
    ) -> Tensor:
        """Wrap *data* in a tensor descriptor of shape *sizes*.

        ``len(sizes)`` must equal ``ndim``.

        Raises:
            ValueError: If ``ndim < 1`` or ``len(sizes) != ndim``.

        See Also:
            :meth:`destroy`.
        """
        if ndim < 1:
            raise ValueError(f"ndim must be >= 1, got {ndim}")
        if len(sizes) != ndim:
            raise ValueError(
                f"ndim={ndim} but {len(sizes)} sizes given"
            )
        sizes_buf = np.asarray(sizes, dtype=np.uintp)
        alloc_ptr = _ep_bindings.tensor_alloc(
            ndim, int(datatype), sizes_buf.ctypes.data, 0,
        )
        obj = cls(alloc_ptr, ndim, tuple(sizes))
        obj._view.data_ = data
        return obj

    @classmethod
    def create_from_window(
        cls,
        ndim: int,
        datatype: int,
        win: int,
        win_offset: int,
        *sizes: int,
    ) -> Tensor:
        """Wrap an offset into a registered NCCL window as a tensor descriptor.

        The local data pointer is resolved lazily on first use with a
        :class:`Group`; :py:attr:`data` returns 0 until that resolution
        happens.

        Raises:
            ValueError: If ``ndim < 1`` or ``len(sizes) != ndim``.
        """
        if ndim < 1:
            raise ValueError(f"ndim must be >= 1, got {ndim}")
        if len(sizes) != ndim:
            raise ValueError(
                f"ndim={ndim} but {len(sizes)} sizes given"
            )
        sizes_buf = np.asarray(sizes, dtype=np.uintp)
        alloc_ptr = _ep_bindings.tensor_alloc(
            ndim, int(datatype), sizes_buf.ctypes.data, 0,
        )
        obj = cls(alloc_ptr, ndim, tuple(sizes))
        obj._view.win_hdl = win
        obj._view.win_offset = win_offset
        return obj

    def _check_valid(self, operation: str) -> None:
        if not self._alloc_ptr:
            raise NcclInvalid(
                f"Cannot {operation}: Tensor is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpTensor_t`` address."""
        self._check_valid("read ptr")
        return self._alloc_ptr

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        self._check_valid("read ndim")
        return self._ndim

    @property
    def data(self) -> int:
        """Address of the underlying device buffer (caller-provided at create).

        Returns 0 for window-backed tensors until resolved by the library.
        """
        self._check_valid("read data")
        return self._view.data_

    @property
    def sizes(self) -> tuple[int, ...]:
        """Sizes per dimension as a tuple of length ``ndim``."""
        self._check_valid("read sizes")
        return self._sizes

    def destroy(self) -> None:
        """Free the descriptor; the underlying buffer is the caller's responsibility."""
        if self._alloc_ptr:
            self._view = None
            _ep_bindings.tensor_destroy(self._alloc_ptr)
            self._alloc_ptr = 0

    def __repr__(self) -> str:
        if self._alloc_ptr:
            return f"<Tensor ptr={self._alloc_ptr:#x}>"
        return "<Tensor destroyed>"
