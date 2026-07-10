"""Pythonic wrappers for ``ncclDevComm``, ``ncclWindow``, and ``ncclTeam``.

Wrap a kernel-arg integer pointer with :class:`DevComm` or :class:`Window`
at kernel entry::

    dev_comm = nccl_cute.DevComm(dev_comm)
    win = nccl_cute.Window(win)

:class:`DevComm` queries are properties (``dev_comm.rank``,
``dev_comm.team_world``, ...). :class:`Window` exposes pointer
translations (``win.local_pointer`` / ``lsa_pointer`` / ``peer_pointer``)
plus :meth:`Window.tensor`, which returns a ``cute.Tensor`` view over the
registered buffer — the canonical input to :meth:`Gin.put`.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

from . import _bindings as raw
from ._helpers import _to_ptr
from ._structs import (
    _LLVMPtrType,
    ncclTeam as Team,
    ncclLsaBarrierHandle,
    ncclGinBarrierHandle,
    ncclMultimemHandle,
)
from .gin import Gin, _alloca_ncclGin_C
from .types import GinBackendMask


def _ptr_wrapper(cls):
    """Augment a ``@cute.native_struct``'s ``__init__`` for single-arg construction.

    A single positional integer or ptr-castable arg auto-converts to
    ``ptr=_to_ptr(x)``. Same-struct-value wrap and no-arg / kwarg modes
    still delegate to the native_struct-generated init.

    Args:
        cls: ``@cute.native_struct`` class to augment.

    Returns:
        ``cls`` (mutated in place).
    """
    native_init = cls.__init__

    @dsl_user_op
    def __init__(self, *args, loc=None, ip=None, **kwargs):
        if len(args) == 1 and not kwargs:
            x = args[0]
            is_struct_value = (
                isinstance(x, ir.Value) and x.type == type(self)._struct_type
            )
            if not is_struct_value:
                native_init(self, ptr=_to_ptr(x), loc=loc, ip=ip)
                return
        native_init(self, *args, loc=loc, ip=ip, **kwargs)

    cls.__init__ = __init__
    return cls


@_ptr_wrapper
@cute.native_struct
class DevComm:
    """Pointer wrapper for ``ncclDevComm``.

    Construct with ``DevComm(<integer kernel arg>)``; the integer is
    converted to ``!llvm.ptr`` via :func:`_helpers._to_ptr` before being
    stored in the struct.
    """

    ptr: _LLVMPtrType

    # === Scalar fields ===

    @property
    def rank(self) -> cutlass.Int32:
        return raw.ncclDevComm_Rank(self.ptr)

    @property
    def n_ranks(self) -> cutlass.Int32:
        return raw.ncclDevComm_NRanks(self.ptr)

    @property
    def lsa_rank(self) -> cutlass.Int32:
        return raw.ncclDevComm_LsaRank(self.ptr)

    @property
    def lsa_size(self) -> cutlass.Int32:
        return raw.ncclDevComm_LsaSize(self.ptr)

    # === Embedded barrier handles ===

    @property
    def lsa_barrier(self) -> ncclLsaBarrierHandle:
        return raw.ncclDevComm_LsaBarrier(self.ptr)

    @property
    def rail_gin_barrier(self) -> ncclGinBarrierHandle:
        return raw.ncclDevComm_RailGinBarrier(self.ptr)

    @property
    def hybrid_lsa_barrier(self) -> ncclLsaBarrierHandle:
        return raw.ncclDevComm_HybridLsaBarrier(self.ptr)

    @property
    def hybrid_rail_gin_barrier(self) -> ncclGinBarrierHandle:
        return raw.ncclDevComm_HybridRailGinBarrier(self.ptr)

    @property
    def world_gin_barrier(self) -> ncclGinBarrierHandle:
        return raw.ncclDevComm_WorldGinBarrier(self.ptr)

    @property
    def lsa_multimem(self) -> ncclMultimemHandle:
        return raw.ncclDevComm_LsaMultimem(self.ptr)

    # === Team factories ===

    @property
    def team_world(self) -> Team:
        return raw.ncclTeamWorld(self.ptr)

    @property
    def team_lsa(self) -> Team:
        return raw.ncclTeamLsa(self.ptr)

    @property
    def team_rail(self) -> Team:
        return raw.ncclTeamRail(self.ptr)

    # === Gin factory ===

    def gin(self, backend: GinBackendMask, context_id: int) -> Gin:
        """Allocate and initialize a :class:`Gin` rooted on this comm.

        Args:
            backend: backend selection mask.
            context_id: GIN context id.

        Returns:
            Initialized :class:`Gin`.
        """
        storage = _alloca_ncclGin_C()
        raw.ncclGin_C_init(storage, backend, self, context_id)
        return Gin(ptr=storage)


@_ptr_wrapper
@cute.native_struct
class Window:
    """Pointer wrapper for ``ncclWindow_t``.

    Construct with ``Window(<integer handle from host>)``. Use
    :meth:`tensor` to obtain a ``cute.Tensor`` view over the registered
    buffer (the canonical input to :meth:`Gin.put`), or the
    ``local_pointer`` / ``lsa_pointer`` / ``peer_pointer`` methods to
    translate ``(offset, peer/team)`` tuples to raw virtual addresses.
    """

    ptr: _LLVMPtrType

    def local_pointer(self, offset: int) -> ir.Value:
        """Translate ``offset`` to the local virtual address.

        Args:
            offset: byte offset within the window.

        Returns:
            ``!llvm.ptr`` ir.Value.
        """
        return raw.ncclGetLocalPointer(self.ptr, offset)

    def lsa_pointer(self, offset: int, peer: int) -> ir.Value:
        """Translate ``offset`` to ``peer``'s LSA virtual address.

        Args:
            offset: byte offset within the window.
            peer: LSA-team peer rank.

        Returns:
            ``!llvm.ptr`` ir.Value.
        """
        return raw.ncclGetLsaPointer(self.ptr, offset, peer)

    def peer_pointer(self, offset: int, peer: int, team: Team = None) -> ir.Value:
        """Translate ``offset`` to ``peer``'s virtual address.

        Args:
            offset: byte offset within the window.
            peer: rank within ``team``.
            team: team to address within; default team if ``None``.

        Returns:
            ``!llvm.ptr`` ir.Value.
        """
        if team is None:
            return raw.ncclGetPeerPointer(self.ptr, offset, peer)
        return raw.ncclGetPeerPointerTeam(self.ptr, offset, team, peer)

    def tensor(self, dtype, layout, offset: int = 0):
        """Construct a ``cute.Tensor`` view over the registered buffer.

        Canonical input to :meth:`Gin.put`: byte offset (relative to the
        window) and transfer size are derived from the tensor's iterator
        address and layout.

        Args:
            dtype: cutlass numeric type (e.g. ``cutlass.Int64``).
            layout: ``cute.Layout`` from ``cute.make_layout(...)``.
            offset: byte offset within the window. Default 0.

        Returns:
            ``cute.Tensor`` view at ``offset``.
        """
        return cute.make_tensor(
            cute.make_ptr(dtype, self.local_pointer(offset)),
            layout,
        )


__all__ = [
    "Team",
    "DevComm",
    "Window",
]
