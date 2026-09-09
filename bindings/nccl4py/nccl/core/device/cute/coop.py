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

from . import _bindings
from ._helpers import _alloca_struct
from ._structs import _LLVMPtrType, ncclCoopAny


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
        return cutlass.Int32(_bindings.nccl_coop_thread_rank(self.ptr))

    @property
    def size(self) -> cutlass.Int32:
        return cutlass.Int32(_bindings.nccl_coop_size(self.ptr))

    @property
    def num_threads(self) -> cutlass.Int32:
        return cutlass.Int32(_bindings.nccl_coop_num_threads(self.ptr))

    # === Action ===

    def sync(self) -> None:
        """Synchronize all threads in the coop."""
        _bindings.nccl_coop_sync(self.ptr)


# === Module-level factories ===

def cta() -> Coop:
    """Construct a :class:`Coop` spanning the whole CTA (block).

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_struct(ncclCoopAny)
    _bindings.nccl_coop_any_init_cta(ptr)
    return Coop(ptr=ptr)


def warp() -> Coop:
    """Construct a :class:`Coop` spanning the calling warp.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_struct(ncclCoopAny)
    _bindings.nccl_coop_any_init_warp(ptr)
    return Coop(ptr=ptr)


def thread() -> Coop:
    """Construct a :class:`Coop` containing only the calling thread.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_struct(ncclCoopAny)
    _bindings.nccl_coop_any_init_thread(ptr)
    return Coop(ptr=ptr)


def lanes(lane_mask: int) -> Coop:
    """Construct a :class:`Coop` over the lanes set in ``lane_mask``.

    Args:
        lane_mask: 32-bit lane mask within the calling warp.

    Returns:
        :class:`Coop`.
    """
    ptr = _alloca_struct(ncclCoopAny)
    _bindings.nccl_coop_any_init_lanes(ptr, cutlass.Uint32(lane_mask))
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
    ptr = _alloca_struct(ncclCoopAny)
    _bindings.nccl_coop_any_init_warp_span(
        ptr, cutlass.Int32(warp0), cutlass.Int32(n_warps), cutlass.Int32(id))
    return Coop(ptr=ptr)


__all__ = [
    "Coop",
    "cta",
    "warp",
    "thread",
    "lanes",
    "warp_span",
]
