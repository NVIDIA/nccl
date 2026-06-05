"""Cooperative groups (``ncclCoopAny``) for the device API.

:class:`Coop` wraps a stack-alloca'd ``ncclCoopAny`` storage block.
Construct via the module-level factories::

    coop = nccl_cute.cta()
    coop = nccl_cute.warp()
    coop = nccl_cute.thread()
    coop = nccl_cute.lanes(mask)
    coop = nccl_cute.warp_span(warp0=0, n_warps=2, id=0)

By-value FFI signatures (e.g. :meth:`Gin.put`,
:meth:`LsaBarrierSession.arrive`) accept :class:`Coop` instances directly.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

from . import _bindings as raw
from ._structs import _LLVMPtrType, ncclCoopAny


@dsl_user_op
def _alloca_coop(*, loc=None, ip=None):
    """Alloca uninitialized ``ncclCoopAny`` storage on the kernel stack.

    Returns:
        ``!llvm.ptr`` ir.Value to the storage.
    """
    return llvm.alloca(
        res=ir.Type.parse("!llvm.ptr"),
        array_size=cutlass.Int32(1).ir_value(),
        elem_type=ncclCoopAny._struct_type,
        loc=loc,
        ip=ip,
    )


@cute.native_struct
class Coop:
    """Pointer wrapper for ``ncclCoopAny``.

    Construct via :func:`cta` / :func:`warp` / :func:`thread` /
    :func:`lanes` / :func:`warp_span`. Methods/properties forward to the
    bindings, passing ``self.ptr`` directly so accessors don't re-spill
    the underlying struct on every call.
    """

    ptr: _LLVMPtrType

    # === Read-only accessors (cheap — direct ptr dispatch) ===

    @property
    def thread_rank(self) -> cutlass.Int32:
        return raw.ncclCoopThreadRank(self.ptr)

    @property
    def size(self) -> cutlass.Int32:
        return raw.ncclCoopSize(self.ptr)

    @property
    def num_threads(self) -> cutlass.Int32:
        return raw.ncclCoopNumThreads(self.ptr)

    # === Action ===

    def sync(self) -> None:
        """Synchronize all threads in the coop."""
        raw.ncclCoopSync(self.ptr)


# === Module-level factories ===

def cta() -> Coop:
    """Construct a :class:`Coop` spanning the whole CTA (block).

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_coop()
    raw.ncclCoopAnyInitCta(ptr)
    return Coop(ptr=ptr)


def warp() -> Coop:
    """Construct a :class:`Coop` spanning the calling warp.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_coop()
    raw.ncclCoopAnyInitWarp(ptr)
    return Coop(ptr=ptr)


def thread() -> Coop:
    """Construct a :class:`Coop` containing only the calling thread.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_coop()
    raw.ncclCoopAnyInitThread(ptr)
    return Coop(ptr=ptr)


def lanes(lane_mask: int) -> Coop:
    """Construct a :class:`Coop` over the lanes set in ``lane_mask``.

    Args:
        lane_mask: 32-bit lane mask within the calling warp.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_coop()
    raw.ncclCoopAnyInitLanes(ptr, lane_mask)
    return Coop(ptr=ptr)


def warp_span(warp0: int, n_warps: int, id: int) -> Coop:
    """Construct a :class:`Coop` over ``n_warps`` consecutive warps.

    Args:
        warp0: starting warp index.
        n_warps: number of warps to span.
        id: span id (0..15) distinguishing concurrent spans.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_coop()
    raw.ncclCoopAnyInitWarpSpan(ptr, warp0, n_warps, id)
    return Coop(ptr=ptr)


__all__ = [
    "Coop",
    "cta",
    "warp",
    "thread",
    "lanes",
    "warp_span",
]
