"""Barrier sessions for the device API.

Three session types:

  * :class:`LsaBarrierSession` -- LSA-only (intra-node, NVLink/peer-access).
  * :class:`GinBarrierSession` -- GIN-only (inter-node, network).
  * :class:`BarrierSession`    -- hybrid (LSA inner + GIN outer).

Construct via the module-level factories:

  * Explicit -- caller supplies the team and barrier handle::

        sess = barrier.lsa_session(coop, dev_comm, team, lsa_handle, index=0)
        sess = barrier.gin_session(coop, gin, dev_comm, team, gin_handle,
                                   index=0)
        sess = barrier.hybrid_session(coop, inner_team, outer_team, gin,
                                       lsa_handle, gin_handle, index=0)

  * DevComm-derived -- use the communicator's built-in teams and barriers::

        sess = barrier.lsa_default(coop, dev_comm, index=0)
        sess = barrier.world_gin(coop, gin, dev_comm, index=0)
        sess = barrier.rail_gin(coop, gin, dev_comm, index=0)
        sess = barrier.world_hybrid(coop, gin, dev_comm, index=0)

The three GIN factories also accept :data:`GIN_ALL_CONTEXTS` in place of a
:class:`Gin`, to fence every GIN context on the comm rather than one::

        sess = barrier.world_gin(coop, barrier.GIN_ALL_CONTEXTS, dev_comm,
                                 index=0)

Call ``destroy()`` exactly once after the session's final operation. All
threads in the session's cooperative group must call it from uniform control
flow. For LSA and hybrid sessions, this makes the same handle and index safe to
reuse. Omitting it can cause a later barrier to stop synchronizing without
hanging.
"""

import cutlass
import cutlass.cute as cute

from . import _bindings
from .coop import Coop
from ._helpers import _alloca_struct, _to_coop_ptr, _to_value, _to_ptr
from ._structs import (
    _LLVMPtrType,
    ncclIrLsaBarrierSession,
    ncclIrGinBarrierSession,
    ncclIrBarrierSession,
    ncclTeam,
    ncclMultimemHandle,
)
from .comm import DevComm
from .gin import Gin
from .handles import (
    GinBarrierHandle,
    LsaBarrierHandle,
    MultimemHandle,
)
from .types import MemoryOrder, GinFenceLevel


def _zero_multimem_handle() -> ncclMultimemHandle:
    """Zero-initialized ``ncclMultimemHandle`` for the default-arg case.

    Returns:
        ``ncclMultimemHandle`` with ``mcBasePtr=NULL``.
    """
    return ncclMultimemHandle(mcBasePtr=_to_ptr(0))


# === LSA Barrier Session ===

@cute.native_struct
class LsaBarrierSession:
    """LSA (Load/Store Accessible) barrier session -- intra-node, peer-access
    based. Constructed via :func:`lsa_session`."""

    ptr: _LLVMPtrType

    def arrive(self, coop: Coop, order: MemoryOrder) -> None:
        """Issue the barrier arrive phase.

        Args:
            coop: cooperative group issuing the arrive.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
        """
        _bindings.nccl_lsa_barrier_session_arrive(self.ptr, _to_coop_ptr(coop), cutlass.Int32(int(order)))

    def wait(self, coop: Coop, order: MemoryOrder) -> None:
        """Wait for the barrier to complete.

        Args:
            coop: cooperative group issuing the wait.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
        """
        _bindings.nccl_lsa_barrier_session_wait(self.ptr, _to_coop_ptr(coop), cutlass.Int32(int(order)))

    def sync(self, coop: Coop, order: MemoryOrder) -> None:
        """Arrive + wait in one call.

        Args:
            coop: cooperative group issuing the sync.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
        """
        _bindings.nccl_lsa_barrier_session_sync(self.ptr, _to_coop_ptr(coop), cutlass.Int32(int(order)))

    def destroy(self) -> None:
        """Finalize the session so its handle and index can be safely reused.

        All threads in the session's cooperative group must call this exactly
        once.
        """
        _bindings.nccl_lsa_barrier_session_destroy(self.ptr)


def lsa_session(
    coop: Coop,
    dev_comm: DevComm,
    team: ncclTeam,
    handle: LsaBarrierHandle,
    index: int,
    *,
    multimem: bool = False,
    mm_handle: MultimemHandle | None = None,
) -> "LsaBarrierSession":
    """Create and initialize an :class:`LsaBarrierSession`.

    Args:
        coop: cooperative group running the session.
        dev_comm: owning :class:`DevComm`.
        team: team participating in the barrier.
        handle: LSA barrier handle requested from the host.
        index: barrier slot index.
        multimem: use multimem stores for the arrive phase.
        mm_handle: multimem handle; required when ``multimem=True``.

    Returns:
        Initialized :class:`LsaBarrierSession`.
    """
    storage = _alloca_struct(ncclIrLsaBarrierSession)
    _bindings.nccl_lsa_barrier_session_init(
        storage, _to_coop_ptr(coop), dev_comm.ptr, _to_value(team),
        _to_value(handle), cutlass.Uint32(index), cutlass.Boolean(multimem),
        _to_value(mm_handle if mm_handle is not None else _zero_multimem_handle()),
    )
    return LsaBarrierSession(ptr=storage)


# === GIN Barrier Session ===

@cute.native_struct
class GinBarrierSession:
    """GIN (network) barrier session -- inter-node. Constructed via
    :func:`gin_session`. Only ``sync`` is supported."""

    ptr: _LLVMPtrType

    def sync(self, coop: Coop, order: MemoryOrder, fence: GinFenceLevel) -> None:
        """Arrive + wait in one call.

        Args:
            coop: cooperative group issuing the sync.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
            fence: GIN fence level. See :class:`GinFenceLevel`.
        """
        _bindings.nccl_gin_barrier_session_sync(
            self.ptr, _to_coop_ptr(coop), cutlass.Int32(int(order)), cutlass.Int32(int(fence)))

    def destroy(self) -> None:
        """Finalize the session.

        All threads in the session's cooperative group must call this exactly
        once.
        """
        _bindings.nccl_gin_barrier_session_destroy(self.ptr)


class _GinAllContexts:
    """Type of :data:`GIN_ALL_CONTEXTS`."""

    def __repr__(self) -> str:
        return "GIN_ALL_CONTEXTS"


GIN_ALL_CONTEXTS = _GinAllContexts()
"""Pass in place of a :class:`Gin` to fence every GIN context on the comm."""


def gin_session(
    coop: Coop,
    gin: Gin | _GinAllContexts,
    dev_comm: DevComm,
    team: ncclTeam,
    handle: GinBarrierHandle,
    index: int,
) -> "GinBarrierSession":
    """Create and initialize a :class:`GinBarrierSession`.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin` whose context ``sync`` fences, or
            :data:`GIN_ALL_CONTEXTS` to fence every context on the comm.
            Needed when puts and gets span several contexts, at the cost of
            a flush per ``(context, peer)`` pair instead of per peer.
        dev_comm: owning :class:`DevComm`; unused unless ``gin`` is
            :data:`GIN_ALL_CONTEXTS`.
        team: team participating in the barrier.
        handle: GIN barrier handle requested from the host.
        index: barrier slot index.

    Returns:
        Initialized :class:`GinBarrierSession`.
    """
    storage = _alloca_struct(ncclIrGinBarrierSession)
    if isinstance(gin, _GinAllContexts):
        _bindings.nccl_gin_barrier_session_init_all_contexts(
            storage, _to_coop_ptr(coop), dev_comm.ptr, _to_value(team),
            _to_value(handle), cutlass.Uint32(index),
        )
    else:
        _bindings.nccl_gin_barrier_session_init(
            storage, _to_coop_ptr(coop), gin.ptr, _to_value(team),
            _to_value(handle), cutlass.Uint32(index),
        )
    return GinBarrierSession(ptr=storage)


# === Hybrid Barrier Session ===

@cute.native_struct
class BarrierSession:
    """Hybrid barrier -- LSA inner stage + GIN outer stage. Constructed
    via :func:`hybrid_session`. Only ``sync`` is supported."""

    ptr: _LLVMPtrType

    def sync(self, coop: Coop, order: MemoryOrder, fence: GinFenceLevel) -> None:
        """Hybrid sync: LSA inner + GIN outer.

        Args:
            coop: cooperative group issuing the sync.
            order: ``cuda::memory_order``. See :class:`MemoryOrder`.
            fence: GIN fence level on the outer stage.
                See :class:`GinFenceLevel`.
        """
        _bindings.nccl_barrier_session_sync(
            self.ptr, _to_coop_ptr(coop), cutlass.Int32(int(order)), cutlass.Int32(int(fence)))

    def destroy(self) -> None:
        """Finalize the session so its handles and index can be safely reused.

        All threads in the session's cooperative group must call this exactly
        once.
        """
        _bindings.nccl_barrier_session_destroy(self.ptr)


def hybrid_session(
    coop: Coop,
    inner_team: ncclTeam,
    outer_team: ncclTeam,
    gin: Gin,
    inner_handle: LsaBarrierHandle,
    outer_handle: GinBarrierHandle,
    index: int,
    *,
    multimem: bool = False,
    inner_mm_handle: MultimemHandle | None = None,
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
    storage = _alloca_struct(ncclIrBarrierSession)
    _bindings.nccl_barrier_session_init(
        storage, _to_coop_ptr(coop), _to_value(inner_team), _to_value(outer_team),
        gin.ptr, _to_value(inner_handle), _to_value(outer_handle),
        cutlass.Uint32(index), cutlass.Boolean(multimem),
        _to_value(
            inner_mm_handle if inner_mm_handle is not None else _zero_multimem_handle()),
    )
    return BarrierSession(ptr=storage)


# === Convenience factories ===
# Use native constructors for the communicator's built-in teams and barriers.

def lsa_default(
    coop: Coop,
    dev_comm: DevComm,
    index: int,
    *,
    multimem: bool = False,
) -> "LsaBarrierSession":
    """LSA barrier using the communicator's built-in LSA team and resources.

    Args:
        coop: cooperative group running the session.
        dev_comm: owning :class:`DevComm`.
        index: barrier slot index.
        multimem: use the communicator's LSA multimem mapping for arrive.

    Returns:
        Initialized :class:`LsaBarrierSession`.
    """
    storage = _alloca_struct(ncclIrLsaBarrierSession)
    _bindings.nccl_lsa_barrier_session_init_lsa(
        storage, _to_coop_ptr(coop), dev_comm.ptr, cutlass.Uint32(index),
        cutlass.Boolean(multimem),
    )
    return LsaBarrierSession(ptr=storage)


def world_gin(
    coop: Coop,
    gin: Gin | _GinAllContexts,
    dev_comm: DevComm,
    index: int,
) -> "GinBarrierSession":
    """GIN barrier using the communicator's built-in world-team resources.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin`, or :data:`GIN_ALL_CONTEXTS`. See
            :func:`gin_session`.
        dev_comm: owning :class:`DevComm`; used for :data:`GIN_ALL_CONTEXTS`.
        index: barrier slot index.

    Returns:
        Initialized :class:`GinBarrierSession`.
    """
    storage = _alloca_struct(ncclIrGinBarrierSession)
    if isinstance(gin, _GinAllContexts):
        _bindings.nccl_gin_barrier_session_init_all_contexts_world(
            storage, _to_coop_ptr(coop), dev_comm.ptr, cutlass.Uint32(index),
        )
    else:
        _bindings.nccl_gin_barrier_session_init_world(
            storage, _to_coop_ptr(coop), gin.ptr, cutlass.Uint32(index),
        )
    return GinBarrierSession(ptr=storage)


def rail_gin(
    coop: Coop,
    gin: Gin | _GinAllContexts,
    dev_comm: DevComm,
    index: int,
) -> "GinBarrierSession":
    """GIN barrier using the communicator's built-in rail-team resources.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin`, or :data:`GIN_ALL_CONTEXTS`. See
            :func:`gin_session`.
        dev_comm: owning :class:`DevComm`; used for :data:`GIN_ALL_CONTEXTS`.
        index: barrier slot index.

    Returns:
        Initialized :class:`GinBarrierSession`.
    """
    storage = _alloca_struct(ncclIrGinBarrierSession)
    if isinstance(gin, _GinAllContexts):
        _bindings.nccl_gin_barrier_session_init_all_contexts_rail(
            storage, _to_coop_ptr(coop), dev_comm.ptr, cutlass.Uint32(index),
        )
    else:
        _bindings.nccl_gin_barrier_session_init_rail(
            storage, _to_coop_ptr(coop), gin.ptr, cutlass.Uint32(index),
        )
    return GinBarrierSession(ptr=storage)


def world_hybrid(
    coop: Coop,
    gin: Gin,
    dev_comm: DevComm,
    index: int,
    *,
    multimem: bool = False,
) -> "BarrierSession":
    """World barrier using the communicator's built-in hybrid resources.

    Args:
        coop: cooperative group running the session.
        gin: :class:`Gin` for the outer GIN stage.
        dev_comm: owning :class:`DevComm`; retained for call compatibility.
            The session uses the communicator associated with ``gin``.
        index: barrier slot index.
        multimem: use the communicator's LSA multimem mapping for arrive.

    Returns:
        Initialized :class:`BarrierSession`.
    """
    storage = _alloca_struct(ncclIrBarrierSession)
    _bindings.nccl_barrier_session_init_world(
        storage, _to_coop_ptr(coop), gin.ptr, cutlass.Uint32(index),
        cutlass.Boolean(multimem),
    )
    return BarrierSession(ptr=storage)


__all__ = [
    "LsaBarrierSession",
    "GinBarrierSession",
    "BarrierSession",
    "GIN_ALL_CONTEXTS",
    "lsa_session",
    "gin_session",
    "hybrid_session",
    "lsa_default",
    "world_gin",
    "rail_gin",
    "world_hybrid",
]
