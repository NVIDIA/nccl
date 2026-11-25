# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""CuTeDSL wrappers for the NCCL ReduceCopy device APIs."""

import cutlass
from cutlass import cute
from cutlass.base_dsl import Arch
from cutlass.cutlass_dsl import BaseDSL

from . import _nccl_device_ir as _ir
from ._coop import Coop
from ._team import Team
from ._window import SymmetricPointer

__all__ = [
    "local_reduce_sum_copy",
    "lsa_copy",
    "lsa_copy_tma",
    "lsa_reduce_sum",
    "lsa_reduce_sum_copy",
    "multimem_copy",
    "multimem_reduce_sum",
    "multimem_reduce_sum_copy",
]

_TMA_ALIGN_BYTES = 16


def _check_lsa_copy_tma_reqs(smem: cute.Pointer, smem_bytes: int | cutlass.Integer) -> None:
    BaseDSL._get_dsl().check_arch(lambda arch: arch >= Arch.sm_100)
    if isinstance(smem, cute.Pointer) and smem.alignment < _TMA_ALIGN_BYTES:
        raise ValueError(
            f"lsa_copy_tma: smem must be {_TMA_ALIGN_BYTES}-byte aligned, got "
            f"{smem.alignment}-byte alignment (allocate it with "
            f"byte_alignment={_TMA_ALIGN_BYTES})"
        )
    if isinstance(smem_bytes, int) and smem_bytes % _TMA_ALIGN_BYTES:
        raise ValueError(
            f"lsa_copy_tma: smem_bytes must be a multiple of {_TMA_ALIGN_BYTES}, got {smem_bytes}"
        )


def lsa_reduce_sum(
    coop: Coop,
    src: SymmetricPointer,
    dst: cute.Pointer,
    count: int | cutlass.Integer,
    team: Team,
) -> None:
    """Reduce ``count`` contiguous elements from each team rank into local ``dst``.

    Dtypes must match. The caller ensures valid ranges and synchronization.
    """
    if src.dtype is not dst.dtype:
        raise TypeError("reduction endpoints must have matching dtypes")
    dst_ptr = cutlass.Pointer(dst, dtype=dst.dtype).tospace(cutlass.AddressSpace.generic)
    _ir._nccl_ir_lsa_reduce_sum(
        coop._to_ptr(),
        src.window._to_ptr(),
        _ir._SizeT(src.byte_offset),
        dst_ptr,
        _ir._SizeT(count),
        team._to_ptr(),
        src.dtype,
    )


def multimem_reduce_sum(
    coop: Coop,
    mc_src: cute.Pointer,
    dst: cute.Pointer,
    count: int | cutlass.Integer,
) -> None:
    """Reduce ``count`` contiguous elements from multicast ``mc_src`` into local ``dst``.

    ``mc_src`` must address a multicast mapping, for example from
    ``Window.multimem_pointer``. Dtypes must match; the caller ensures valid
    ranges and synchronization.
    """
    if mc_src.dtype is not dst.dtype:
        raise TypeError("reduction endpoints must have matching dtypes")
    mc_src_ptr = cutlass.Pointer(mc_src, dtype=mc_src.dtype).tospace(cutlass.AddressSpace.generic)
    dst_ptr = cutlass.Pointer(dst, dtype=dst.dtype).tospace(cutlass.AddressSpace.generic)
    _ir._nccl_ir_multimem_reduce_sum(
        coop._to_ptr(),
        mc_src_ptr,
        dst_ptr,
        _ir._SizeT(count),
        mc_src.dtype,
    )


def lsa_copy(
    coop: Coop,
    src: cute.Pointer,
    dst: SymmetricPointer,
    count: int | cutlass.Integer,
    team: Team,
) -> None:
    """Copy ``count`` contiguous elements from local ``src`` to each team rank's ``dst``.

    Dtypes must match. The caller ensures valid ranges and synchronization.
    """
    if src.dtype is not dst.dtype:
        raise TypeError("copy endpoints must have matching dtypes")
    src_ptr = cutlass.Pointer(src, dtype=src.dtype).tospace(cutlass.AddressSpace.generic)
    _ir._nccl_ir_lsa_copy(
        coop._to_ptr(),
        src_ptr,
        dst.window._to_ptr(),
        _ir._SizeT(dst.byte_offset),
        _ir._SizeT(count),
        team._to_ptr(),
        src.dtype,
    )


def lsa_copy_tma(
    coop: Coop,
    src: cute.Pointer,
    dst: SymmetricPointer,
    count: int | cutlass.Integer,
    team: Team,
    *,
    smem: cute.Pointer,
    smem_bytes: int | cutlass.Integer,
) -> None:
    """Copy ``count`` contiguous elements to each team rank's ``dst`` via TMA.

    Requires SM100+ kernel and device bitcode, and a thread, warp, or CTA Coop.
    Source and destination must have matching dtypes, reside in global memory,
    and be 16-byte aligned. ``count * src.dtype.bytes`` must be a multiple of 16.
    ``smem`` is shared scratch for the whole Coop, with a pointer alignment
    guarantee of at least 16 bytes. ``smem_bytes`` is its byte capacity, at
    least 32, including internal barrier storage. Python integer capacities
    must be multiples of 16.
    Concurrent calls must use disjoint scratch. The caller ensures valid ranges
    and synchronization around remote accesses.
    """
    _check_lsa_copy_tma_reqs(smem, smem_bytes)
    if src.dtype is not dst.dtype:
        raise TypeError("copy endpoints must have matching dtypes")
    src_ptr = cutlass.Pointer(src, dtype=src.dtype).tospace(cutlass.AddressSpace.generic)
    smem_ptr = cutlass.Pointer(smem, dtype=smem.dtype).tospace(cutlass.AddressSpace.generic)
    _ir._nccl_ir_lsa_copy_tma(
        coop._to_ptr(),
        src_ptr,
        dst.window._to_ptr(),
        _ir._SizeT(dst.byte_offset),
        _ir._SizeT(count),
        team._to_ptr(),
        smem_ptr,
        cutlass.Int32(smem_bytes),
        src.dtype,
    )


def multimem_copy(
    coop: Coop,
    src: cute.Pointer,
    mc_dst: cute.Pointer,
    count: int | cutlass.Integer,
) -> None:
    """Copy ``count`` contiguous elements from local ``src`` to multicast ``mc_dst``.

    ``mc_dst`` must address a multicast mapping, for example from
    ``Window.multimem_pointer``. Dtypes must match; the caller ensures valid
    ranges and synchronization.
    """
    if src.dtype is not mc_dst.dtype:
        raise TypeError("copy endpoints must have matching dtypes")
    src_ptr = cutlass.Pointer(src, dtype=src.dtype).tospace(cutlass.AddressSpace.generic)
    mc_dst_ptr = cutlass.Pointer(mc_dst, dtype=mc_dst.dtype).tospace(cutlass.AddressSpace.generic)
    _ir._nccl_ir_multimem_copy(
        coop._to_ptr(),
        src_ptr,
        mc_dst_ptr,
        _ir._SizeT(count),
        src.dtype,
    )


def lsa_reduce_sum_copy(
    coop: Coop,
    src: SymmetricPointer,
    dst: SymmetricPointer,
    count: int | cutlass.Integer,
    team: Team,
) -> None:
    """Sum ``count`` contiguous elements of ``src`` across ``team`` into each rank's ``dst``.

    Dtypes must match. The caller ensures valid ranges and synchronization.
    """
    if src.dtype is not dst.dtype:
        raise TypeError("reduction endpoints must have matching dtypes")
    _ir._nccl_ir_lsa_reduce_sum_copy(
        coop._to_ptr(),
        src.window._to_ptr(),
        _ir._SizeT(src.byte_offset),
        dst.window._to_ptr(),
        _ir._SizeT(dst.byte_offset),
        _ir._SizeT(count),
        team._to_ptr(),
        src.dtype,
    )


def multimem_reduce_sum_copy(
    coop: Coop,
    mc_src: cute.Pointer,
    mc_dst: cute.Pointer,
    count: int | cutlass.Integer,
) -> None:
    """Reduce ``count`` contiguous elements from ``mc_src`` and copy to ``mc_dst``.

    Both endpoints must address multicast mappings, for example from
    ``Window.multimem_pointer``. Dtypes must match; the caller ensures valid
    ranges and synchronization.
    """
    if mc_src.dtype is not mc_dst.dtype:
        raise TypeError("reduction endpoints must have matching dtypes")
    mc_src_ptr = cutlass.Pointer(mc_src, dtype=mc_src.dtype).tospace(cutlass.AddressSpace.generic)
    mc_dst_ptr = cutlass.Pointer(mc_dst, dtype=mc_dst.dtype).tospace(cutlass.AddressSpace.generic)
    _ir._nccl_ir_multimem_reduce_sum_copy(
        coop._to_ptr(),
        mc_src_ptr,
        mc_dst_ptr,
        _ir._SizeT(count),
        mc_src.dtype,
    )


def local_reduce_sum_copy(
    coop: Coop,
    src: cute.Tensor,
    dst: cute.Tensor,
    count: int | cutlass.Integer,
) -> None:
    """Sum source rows elementwise and copy the result to every destination row.

    Requires pointer-backed tensors of matching dtype with flat 2-D shape
    ``(rows, capacity)``, nonnegative integer row strides, and a static inner
    stride of 1. Processes ``count`` elements per row; it must fit both capacities.
    The caller ensures valid ranges and synchronization.
    """
    if src.element_type is not dst.element_type:
        raise TypeError("reduction endpoints must have matching dtypes")
    for name, tensor in (("src", src), ("dst", dst)):
        if not isinstance(tensor.iterator, cute.Pointer):
            raise TypeError(f"{name} must be a pointer-backed tensor")
        layout = tensor.layout
        if not isinstance(layout, cute.Layout) or cute.rank(layout) != 2 or cute.depth(layout) != 1:
            raise ValueError(f"{name} must have a flat 2-D layout")
        stride = layout.stride
        if not all(cute.is_integer(step) for step in stride):
            raise ValueError(f"{name} must have integer strides")
        if not cute.is_static(stride[1]) or stride[1] != 1:
            raise ValueError(f"{name} must have a statically known inner stride of 1")

    src_ptr = cutlass.Pointer(src.iterator, dtype=src.element_type).tospace(
        cutlass.AddressSpace.generic
    )
    dst_ptr = cutlass.Pointer(dst.iterator, dtype=dst.element_type).tospace(
        cutlass.AddressSpace.generic
    )
    _ir._nccl_ir_local_reduce_sum_copy(
        coop._to_ptr(),
        cutlass.Int32(src.shape[0]),
        src_ptr,
        _ir._SizeT(src.stride[0]),
        cutlass.Int32(dst.shape[0]),
        dst_ptr,
        _ir._SizeT(dst.stride[0]),
        _ir._SizeT(count),
        src.element_type,
    )
