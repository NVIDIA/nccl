"""Barrier sessions for the device API.

Three session types mirror the C++ wrapper:

  * :class:`LsaBarrierSession` — LSA-only (intra-node, NVLink/peer-access).
  * :class:`GinBarrierSession` — GIN-only (inter-node, network).
  * :class:`BarrierSession`    — hybrid (LSA inner + GIN outer).

Construct via the module-level factories:

  * Explicit — caller supplies the team and barrier handle::

        sess = barrier.lsa_session(coop, dev_comm, team, lsa_handle, index=0)
        sess = barrier.gin_session(coop, gin, team, gin_handle, index=0)
        sess = barrier.hybrid_session(coop, inner_team, outer_team, gin,
                                       lsa_handle, gin_handle, index=0)

  * DevComm-derived (mirror the C++ tag-based barrier constructors)::

        sess = barrier.lsa_default(coop, dev_comm, index=0)
        sess = barrier.world_gin(coop, gin, dev_comm, index=0)
        sess = barrier.rail_gin(coop, gin, dev_comm, index=0)
        sess = barrier.world_hybrid(coop, gin, dev_comm, index=0)
"""

import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

from . import _bindings as raw
from ._helpers import _to_ptr
from ._structs import (
    _LLVMPtrType,
    ncclTeam,
    ncclCoopAny,
    ncclLsaBarrierHandle,
    ncclGinBarrierHandle,
    ncclMultimemHandle,
)
from .comm import DevComm
from .gin import Gin
from .types import MemoryOrder, GinFenceLevel


# Session storage alignment. Conservatively chosen — covers ncclCoopAny
# (ptr-aligned) and any wider fields any future session struct might add.
_SESSION_ALIGN = 16


@dsl_user_op
def _alloca_session(size_value, *, loc=None, ip=None) -> ir.Value:
    """Allocate ``size_value`` bytes of stack storage, 16-byte aligned.

    Args:
        size_value: Int64 from ``raw.ncclXxxSession_C_size()``
            (``sizeof(ncclXxxSession_C)`` on the C++ side).

    Returns:
        ``!llvm.ptr`` ir.Value to the storage.
    """
    i8 = ir.IntegerType.get_signless(8)
    return llvm.alloca(
        res=ir.Type.parse("!llvm.ptr"),
        array_size=size_value.ir_value(),
        elem_type=i8,
        alignment=_SESSION_ALIGN,
        loc=loc,
        ip=ip,
    )


def _zero_multimem_handle() -> ncclMultimemHandle:
    """Zero-initialized ``ncclMultimemHandle`` for the default-arg case.

    Returns:
        ``ncclMultimemHandle`` with ``mcBasePtr=NULL``.
    """
    return ncclMultimemHandle(mcBasePtr=_to_ptr(0))


# === LSA Barrier Session ===

@cute.native_struct
class LsaBarrierSession:
    """LSA (Locally Shared Address) barrier session — intra-node, peer-access
    based. Constructed via :func:`lsa_session`."""

    ptr: _LLVMPtrType

    def arrive(self, coop: ncclCoopAny, order: MemoryOrder) -> None:
        """Issue the barrier arrive phase.

        Args:
            coop: cooperative group issuing the arrive.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
        """
        raw.ncclLsaBarrierSessionArrive(self.ptr, coop, order)

    def wait(self, coop: ncclCoopAny, order: MemoryOrder) -> None:
        """Wait for the barrier to complete.

        Args:
            coop: cooperative group issuing the wait.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
        """
        raw.ncclLsaBarrierSessionWait(self.ptr, coop, order)

    def sync(self, coop: ncclCoopAny, order: MemoryOrder) -> None:
        """Arrive + wait in one call.

        Args:
            coop: cooperative group issuing the sync.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
        """
        raw.ncclLsaBarrierSessionSync(self.ptr, coop, order)


def lsa_session(
    coop: ncclCoopAny,
    dev_comm: DevComm,
    team: ncclTeam,
    handle: ncclLsaBarrierHandle,
    index: int,
    *,
    multimem: bool = False,
    mm_handle: ncclMultimemHandle = None,
) -> "LsaBarrierSession":
    """Create and initialize an :class:`LsaBarrierSession`.

    Args:
        coop: cooperative group running the session.
        dev_comm: owning :class:`DevComm`.
        team: team participating in the barrier.
        handle: LSA barrier handle (typically ``dev_comm.lsa_barrier``).
        index: barrier slot index.
        multimem: use multimem stores for the arrive phase.
        mm_handle: multimem handle; required when ``multimem=True``.

    Returns:
        Initialized :class:`LsaBarrierSession`.
    """
    storage = _alloca_session(raw.ncclLsaBarrierSession_C_size())
    raw.ncclLsaBarrierSessionInit(
        storage, coop, dev_comm, team, handle, index, multimem,
        mm_handle if mm_handle is not None else _zero_multimem_handle(),
    )
    return LsaBarrierSession(ptr=storage)


# === GIN Barrier Session ===

@cute.native_struct
class GinBarrierSession:
    """GIN (network) barrier session — inter-node. Constructed via
    :func:`gin_session`. Only ``sync`` is supported."""

    ptr: _LLVMPtrType

    def sync(self, coop: ncclCoopAny, order: MemoryOrder, fence: GinFenceLevel) -> None:
        """Arrive + wait in one call.

        Args:
            coop: cooperative group issuing the sync.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
            fence: GIN fence level. See :class:`GinFenceLevel`.
        """
        raw.ncclGinBarrierSessionSync(self.ptr, coop, order, fence)


def gin_session(
    coop: ncclCoopAny,
    gin: Gin,
    team: ncclTeam,
    handle: ncclGinBarrierHandle,
    index: int,
) -> "GinBarrierSession":
    """Create and initialize a :class:`GinBarrierSession`.

    The ``ncclGin_C`` value is loaded from ``gin``'s buffer for the
    by-value FFI signature.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin` from ``DevComm.gin(...)``.
        team: team participating in the barrier.
        handle: GIN barrier handle (typically
            ``dev_comm.rail_gin_barrier`` or ``dev_comm.world_gin_barrier``).
        index: barrier slot index.

    Returns:
        Initialized :class:`GinBarrierSession`.
    """
    storage = _alloca_session(raw.ncclGinBarrierSession_C_size())
    raw.ncclGinBarrierSessionInit(
        storage, coop, gin.ptr, team, handle, index,
    )
    return GinBarrierSession(ptr=storage)


# === Hybrid Barrier Session ===

@cute.native_struct
class BarrierSession:
    """Hybrid barrier — LSA inner stage + GIN outer stage. Constructed
    via :func:`hybrid_session`. Only ``sync`` is supported."""

    ptr: _LLVMPtrType

    def sync(self, coop: ncclCoopAny, order: MemoryOrder, fence: GinFenceLevel) -> None:
        """Hybrid sync: LSA inner + GIN outer.

        Args:
            coop: cooperative group issuing the sync.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
            fence: GIN fence level on the outer stage.
                See :class:`GinFenceLevel`.
        """
        raw.ncclBarrierSessionSync(self.ptr, coop, order, fence)


def hybrid_session(
    coop: ncclCoopAny,
    inner_team: ncclTeam,
    outer_team: ncclTeam,
    gin: Gin,
    inner_handle: ncclLsaBarrierHandle,
    outer_handle: ncclGinBarrierHandle,
    index: int,
    *,
    multimem: bool = False,
    inner_mm_handle: ncclMultimemHandle = None,
) -> "BarrierSession":
    """Create and initialize a hybrid :class:`BarrierSession`.

    Args:
        coop: cooperative group running the session.
        inner_team: team for the LSA inner stage.
        outer_team: team for the GIN outer stage.
        gin: :class:`Gin` for the outer stage.
        inner_handle: LSA barrier handle for the inner stage.
        outer_handle: GIN barrier handle for the outer stage.
        index: barrier slot index.
        multimem: use multimem stores in the inner arrive phase.
        inner_mm_handle: inner multimem handle; required when
            ``multimem=True``.

    Returns:
        Initialized :class:`BarrierSession`.
    """
    storage = _alloca_session(raw.ncclBarrierSession_C_size())
    raw.ncclBarrierSessionInit(
        storage, coop, inner_team, outer_team, gin.ptr,
        inner_handle, outer_handle, index, multimem,
        inner_mm_handle if inner_mm_handle is not None else _zero_multimem_handle(),
    )
    return BarrierSession(ptr=storage)


# === Convenience factories ===
# Pull the right team and handle from a DevComm so the user doesn't have to
# wire them up manually. Mirror the C++ tag-based barrier constructors
# (ncclTeamTagWorld / ncclTeamTagRail / ncclTeamTagLsa).

def lsa_default(
    coop: ncclCoopAny,
    dev_comm: DevComm,
    index: int,
    *,
    multimem: bool = False,
    mm_handle: ncclMultimemHandle = None,
) -> "LsaBarrierSession":
    """LSA barrier on the default LSA team using ``dev_comm.lsa_barrier``.

    C++ equivalent:
    ``ncclLsaBarrierSession<...>{coop, dev_comm, ncclTeamTagLsa(), index}``.

    Args:
        coop: cooperative group running the session.
        dev_comm: owning :class:`DevComm`.
        index: barrier slot index.
        multimem: use multimem stores for the arrive phase.
        mm_handle: multimem handle; required when ``multimem=True``.

    Returns:
        Initialized :class:`LsaBarrierSession`.
    """
    return lsa_session(coop, dev_comm, dev_comm.team_lsa, dev_comm.lsa_barrier,
                       index, multimem=multimem, mm_handle=mm_handle)


def world_gin(coop: ncclCoopAny, gin: Gin, dev_comm: DevComm, index: int) -> "GinBarrierSession":
    """GIN barrier on the world team using ``dev_comm.world_gin_barrier``.

    C++ equivalent:
    ``ncclGinBarrierSession<...>{coop, gin, ncclTeamTagWorld(), index}``.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin` instance.
        dev_comm: owning :class:`DevComm`.
        index: barrier slot index.

    Returns:
        Initialized :class:`GinBarrierSession`.
    """
    return gin_session(coop, gin, dev_comm.team_world,
                       dev_comm.world_gin_barrier, index)


def rail_gin(coop: ncclCoopAny, gin: Gin, dev_comm: DevComm, index: int) -> "GinBarrierSession":
    """GIN barrier on the rail team using ``dev_comm.rail_gin_barrier``.

    C++ equivalent:
    ``ncclGinBarrierSession<...>{coop, gin, ncclTeamTagRail(), index}``.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin` instance.
        dev_comm: owning :class:`DevComm`.
        index: barrier slot index.

    Returns:
        Initialized :class:`GinBarrierSession`.
    """
    return gin_session(coop, gin, dev_comm.team_rail,
                       dev_comm.rail_gin_barrier, index)


def world_hybrid(
    coop: ncclCoopAny,
    gin: Gin,
    dev_comm: DevComm,
    index: int,
    *,
    multimem: bool = False,
    inner_mm_handle: ncclMultimemHandle = None,
) -> "BarrierSession":
    """Hybrid barrier (LSA + rail-GIN) using the embedded hybrid handles.

    C++ equivalent:
    ``ncclBarrierSession<...>{coop, ncclTeamTagWorld(), gin, index}``.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin` for the outer GIN stage.
        dev_comm: owning :class:`DevComm`; supplies the inner LSA and
            outer rail-GIN handles via its embedded hybrid handles.
        index: barrier slot index.
        multimem: use multimem stores in the inner arrive phase.
        inner_mm_handle: inner multimem handle; required when
            ``multimem=True``.

    Returns:
        Initialized :class:`BarrierSession`.
    """
    return hybrid_session(coop, dev_comm.team_lsa, dev_comm.team_world, gin,
                     dev_comm.hybrid_lsa_barrier,
                     dev_comm.hybrid_rail_gin_barrier, index,
                     multimem=multimem, inner_mm_handle=inner_mm_handle)


__all__ = [
    "LsaBarrierSession",
    "GinBarrierSession",
    "BarrierSession",
    "lsa_session",
    "gin_session",
    "hybrid_session",
    "lsa_default",
    "world_gin",
    "rail_gin",
    "world_hybrid",
]
