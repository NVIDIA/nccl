"""User-facing types shared by the rest of the device API.

  - :class:`CftLeInfo` -- logical endpoint address returned by CFT queries.
  - :class:`MemoryOrder`   -- argument to barrier ``arrive`` / ``wait`` / ``sync``.
  - :class:`ThreadScope`   -- release-scope argument to GIN ``put`` / ``signal``.
  - :class:`GinFenceLevel` -- fence-level argument to GIN barrier ``sync``.
  - :class:`CftTeamMode` -- team layout for :meth:`DevComm.team_cft`.
  - :class:`GinResourceSharingMode` -- resource-sharing mode for :meth:`DevComm.gin`.
"""

from dataclasses import dataclass
from enum import IntEnum, IntFlag

import cutlass


@dataclass(frozen=True)
class CftLeInfo:
    """CFT logical endpoint address produced while tracing device code."""

    le_id: cutlass.Uint32
    le_offset: cutlass.Uint64


class MemoryOrder(IntEnum):
    """Mirrors C++ ``cuda::memory_order`` from <cuda/std/atomic>.

    Integer values match libcu++ exactly; pass to barrier ``arrive`` /
    ``wait`` / ``sync`` calls. Not the same as
    ``cutlass.cute.nvgpu.MemoryOrder`` (which keys to PTX cache modifiers,
    not C++ memory order).
    """

    RELAXED = 0
    CONSUME = 1
    ACQUIRE = 2
    RELEASE = 3
    ACQ_REL = 4
    SEQ_CST = 5


class ThreadScope(IntEnum):
    """Mirrors C++ ``cuda::thread_scope`` from <cuda/std/atomic>.

    Integer values match libcu++ exactly; pass as the release-scope
    arguments to GIN ``put`` / ``put_value`` / ``signal``. Not the same as
    ``cutlass.cute.nvgpu.MemoryScope`` (which is an MLIR IR scope kind with
    a different taxonomy, not the libcu++ thread scope).
    """

    SYSTEM = 0
    DEVICE = 1
    BLOCK = 2
    THREAD = 10  # libcu++ __ATOMIC_THREAD deliberately leaves a gap.


class GinFenceLevel(IntFlag):
    """Mirrors ``enum class ncclGinFenceLevel``.

    Bit-flag composable: ``GinFenceLevel.PUT | GinFenceLevel.GET`` drains
    both prior puts and prior gets on the bound GIN context.
    """

    NONE = 0           # pure sync, no drain
    PUT = 1 << 0       # 1: drain prior puts
    GET = 1 << 1       # 2: drain prior gets


class CftTeamMode(IntEnum):
    """Mirrors ``enum ncclCftTeamMode_t`` -- the layout
    :meth:`DevComm.team_cft` returns.
    """

    FLAT = 0           # every rank in the CFT unicast group
    HIER_MULTIMEM = 1  # same index across multicast CFT groups
    HIER_LSA = 2       # same index across LSA groups


class GinResourceSharingMode(IntEnum):
    """Mirrors ``enum ncclGinResourceSharingMode``.

    Selects whether GIN network resources are shared across the GPU, within
    each CTA, or are exclusive to each thread. The mode belongs to the
    :class:`Gin` instance and is independent of an operation's cooperative
    group.
    """

    GPU = 0
    CTA = 1
    THREAD = 2


__all__ = [
    "CftLeInfo",
    "MemoryOrder",
    "ThreadScope",
    "GinFenceLevel",
    "GinResourceSharingMode",
    "CftTeamMode",
]
