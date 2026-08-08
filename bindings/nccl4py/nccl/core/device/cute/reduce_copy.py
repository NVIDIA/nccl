"""CuTeDSL wrappers for the NCCL ReduceCopy device APIs.

Data operands are ``cute.Tensor`` instances. Their element type selects the
dtype-specialized device symbol, and LSA window offsets are derived from their
iterators.
"""

import cutlass
import cutlass.cute as cute

from . import _bindings
from ._helpers import _to_coop_value, _to_ptr, _to_value


def _tensor_ptr(tensor):
    return _to_ptr(tensor.iterator.toint())


def _window_offset(window, tensor):
    base = cute.make_ptr(cutlass.Int8, window.local_pointer(0))
    return cutlass.Int64(tensor.iterator.toint() - base.toint())


def lsa_reduce_sum(
    coop, src_window, src, dst, count, *, team,
) -> None:
    """Reduce an LSA window tensor into a rank-local destination tensor."""
    _bindings.nccl_lsa_reduce_sum(
        _to_coop_value(coop),
        src_window.ptr,
        _window_offset(src_window, src),
        _tensor_ptr(dst),
        cutlass.Int64(count),
        _to_value(team),
        src.element_type,
    )


def multimem_reduce_sum(coop, mc_src, dst, count) -> None:
    """Reduce from a multimem tensor into a local destination tensor."""
    _bindings.nccl_multimem_reduce_sum(
        _to_coop_value(coop),
        _tensor_ptr(mc_src),
        _tensor_ptr(dst),
        cutlass.Int64(count),
        mc_src.element_type,
    )


def lsa_copy(
    coop, src, dst_window, dst, count, *, team,
) -> None:
    """Copy a rank-local source tensor into an LSA window tensor."""
    _bindings.nccl_lsa_copy(
        _to_coop_value(coop),
        _tensor_ptr(src),
        dst_window.ptr,
        _window_offset(dst_window, dst),
        cutlass.Int64(count),
        _to_value(team),
        src.element_type,
    )


def multimem_copy(coop, src, mc_dst, count) -> None:
    """Copy a local source tensor into a multimem destination tensor."""
    _bindings.nccl_multimem_copy(
        _to_coop_value(coop),
        _tensor_ptr(src),
        _tensor_ptr(mc_dst),
        cutlass.Int64(count),
        src.element_type,
    )


def lsa_reduce_sum_copy(
    coop, src_window, src, dst_window, dst, count, *, team,
) -> None:
    """Reduce and copy between LSA window tensors in the same team."""
    _bindings.nccl_lsa_reduce_sum_copy(
        _to_coop_value(coop),
        src_window.ptr,
        _window_offset(src_window, src),
        dst_window.ptr,
        _window_offset(dst_window, dst),
        cutlass.Int64(count),
        _to_value(team),
        src.element_type,
    )


def multimem_reduce_sum_copy(coop, mc_src, mc_dst, count) -> None:
    """Reduce from and copy to multimem tensors."""
    _bindings.nccl_multimem_reduce_sum_copy(
        _to_coop_value(coop),
        _tensor_ptr(mc_src),
        _tensor_ptr(mc_dst),
        cutlass.Int64(count),
        mc_src.element_type,
    )


def local_reduce_sum_copy(
    coop,
    n_src,
    src_base,
    src_displ,
    n_dst,
    dst_base,
    dst_displ,
    count,
) -> None:
    """Reduce from local source tensors and copy to destination tensors."""
    _bindings.nccl_local_reduce_sum_copy(
        _to_coop_value(coop),
        cutlass.Int32(n_src),
        _tensor_ptr(src_base),
        cutlass.Int64(src_displ),
        cutlass.Int32(n_dst),
        _tensor_ptr(dst_base),
        cutlass.Int64(dst_displ),
        cutlass.Int64(count),
        src_base.element_type,
    )


__all__ = [
    "lsa_reduce_sum",
    "multimem_reduce_sum",
    "lsa_copy",
    "multimem_copy",
    "lsa_reduce_sum_copy",
    "multimem_reduce_sum_copy",
    "local_reduce_sum_copy",
]
