# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ND-tensor descriptor: ``NDTensor`` wrapping ``ncclNDTensor_t``."""

from __future__ import annotations

import numpy as np

from nccl.core.typing import NcclInvalid

from nccl.ep import bindings as _ep_bindings


__all__ = ["NDTensor"]


class NDTensor:
    """Caller-owned device buffer wrapped as an ``ncclNDTensor_t``.

    The descriptor does not own ``data``; the caller must keep the buffer alive
    until :meth:`destroy` returns.
    """

    def __init__(self, ptr: int, ndim: int = 0, sizes: tuple[int, ...] = ()) -> None:
        self._ptr = ptr
        self._ndim = ndim
        self._sizes = sizes

    @classmethod
    def create(
        cls,
        ndim: int,
        datatype: int,
        data: int,
        *sizes: int,
    ) -> NDTensor:
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
        # Pack into a contiguous size_t array; the C side reads exactly ndim entries.
        sizes_buf = np.asarray(sizes, dtype=np.uintp)
        ptr = _ep_bindings.nccl_ep.ep_tensor_create(
            ndim, int(datatype), data, sizes_buf.ctypes.data,
        )
        return cls(ptr, ndim, tuple(sizes))

    @classmethod
    def create_from_window(
        cls,
        ndim: int,
        datatype: int,
        win: int,
        win_offset: int,
        *sizes: int,
    ) -> NDTensor:
        """Wrap an offset into a registered NCCL window as a tensor descriptor.

        The local data pointer is resolved lazily on first use with an
        :class:`EpGroup`; :py:attr:`data` returns ``ncclInvalidUsage``
        until that resolution happens.

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
        ptr = _ep_bindings.nccl_ep.ep_tensor_create_from_window(
            ndim, int(datatype), win, win_offset, sizes_buf.ctypes.data,
        )
        return cls(ptr, ndim, tuple(sizes))

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: NDTensor is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclNDTensor_t`` address."""
        self._check_valid("read ptr")
        return self._ptr

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        self._check_valid("read ndim")
        return self._ndim

    @property
    def data(self) -> int:
        """Address of the underlying device buffer (caller-provided at create)."""
        self._check_valid("read data")
        return _ep_bindings.nccl_ep.ep_tensor_get_data(self._ptr)

    @property
    def sizes(self) -> tuple[int, ...]:
        """Sizes per dimension as a tuple of length ``ndim``."""
        self._check_valid("read sizes")
        return self._sizes

    def destroy(self) -> None:
        """Free the descriptor; the underlying buffer is the caller's responsibility."""
        if self._ptr:
            _ep_bindings.nccl_ep.ep_tensor_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        if self._ptr:
            return f"<NDTensor ptr={self._ptr:#x}>"
        return "<NDTensor destroyed>"
