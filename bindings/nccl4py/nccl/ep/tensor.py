# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""``Tensor`` — Pythonic wrapper for ``ncclEpTensor_t``."""

from __future__ import annotations

import numbers

import numpy as np

from nccl.core.typing import NcclDataType, NcclInvalid

from nccl.bindings import nccl_ep as _ep_bindings


__all__ = ["Tensor"]


# NCCL_EP_TENSOR_MAGIC — required init cookie for caller-owned descriptors
# (contrib/nccl_ep/include/nccl_ep.h).
_TENSOR_MAGIC = 0xCAFECAFE


def _resolve_buffer(buffer) -> tuple[int, tuple[int, ...], NcclDataType]:
    """Resolve a buffer-like into (ptr, shape, dtype). Torch, then CAI/DLPack."""
    try:
        import torch
        if isinstance(buffer, torch.Tensor):
            from nccl.core.interop.torch import resolve_tensor
            ptr, _count, dtype, _dev = resolve_tensor(buffer)
            return ptr, tuple(buffer.shape), dtype
    except (ImportError, ModuleNotFoundError):
        pass

    from cuda.core.utils import StridedMemoryView, args_viewable_as_strided_memory

    @args_viewable_as_strided_memory((0,))
    def _view(buf, stream_ptr):
        return buf.view(stream_ptr)

    try:
        view = _view(buffer, -1)  # -1 = skip stream sync
    except Exception as e:
        raise NcclInvalid(
            f"Cannot resolve {type(buffer).__name__!r}: expected torch.Tensor, "
            f"cuda.core.Buffer, or an object supporting "
            f"__cuda_array_interface__/__dlpack__. ({e})"
        ) from e
    return view.ptr, tuple(view.shape), NcclDataType(view.dtype)


class Tensor:
    """Caller-owned ``ncclEpTensor_t`` descriptor resolved from a Python buffer.

    Accepts ``torch.Tensor``, ``cuda.core.Buffer``, any CAI/DLPack
    object, or a raw ``int`` device pointer (which requires ``dtype``
    and ``shape`` kwargs). The descriptor struct is Python-managed
    (cybind ``__dealloc__`` frees it on GC); the caller is responsible
    for keeping the underlying buffer alive.
    """

    def __init__(
        self,
        buffer,
        *,
        dtype: NcclDataType | int | None = None,
        shape: tuple[int, ...] | None = None,
    ) -> None:
        if isinstance(buffer, numbers.Integral) and not isinstance(buffer, bool):
            if dtype is None or shape is None:
                raise ValueError(
                    "Raw integer pointer requires `dtype` and `shape` kwargs"
                )
            data_ptr = buffer
            sizes = tuple(shape)
            nccl_dtype = NcclDataType(dtype)
            self._buffer = None
        else:
            if dtype is not None or shape is not None:
                raise ValueError(
                    "`dtype`/`shape` are only valid with a raw integer pointer"
                )
            data_ptr, sizes, nccl_dtype = _resolve_buffer(buffer)
            self._buffer = buffer  # lifetime anchor

        if not sizes:
            raise ValueError("Tensor requires at least one dimension")

        self._sizes = sizes
        self._datatype = nccl_dtype

        self._sizes_buf = np.asarray(sizes, dtype=np.uintp)
        ep_t = _ep_bindings.Tensor()
        ep_t.size_    = _ep_bindings.tensor_dtype.itemsize
        ep_t.magic    = _TENSOR_MAGIC
        ep_t.ndim_    = len(sizes)
        ep_t.datatype = int(nccl_dtype)
        ep_t.data_    = int(data_ptr)
        ep_t.sizes    = int(self._sizes_buf.ctypes.data)
        self._ep_t = ep_t

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpTensor_t`` address."""
        return self._ep_t.ptr

    @property
    def ndim(self) -> int:
        return len(self._sizes)

    @property
    def sizes(self) -> tuple[int, ...]:
        return self._sizes

    @property
    def datatype(self) -> NcclDataType:
        return self._datatype

    @property
    def data(self) -> int:
        """Device buffer address."""
        return self._ep_t.data_

    @property
    def buffer(self):
        """Original buffer passed to the constructor (``None`` for raw-pointer)."""
        return self._buffer

    def __repr__(self) -> str:
        return (
            f"<Tensor sizes={self._sizes} dtype={self._datatype.name} "
            f"data={self.data:#x}>"
        )
