"""User-facing enums shared by the rest of the device API.

  - :class:`MemoryOrder`   — argument to barrier ``arrive`` / ``wait`` / ``sync``.
  - :class:`GinFenceLevel` — fence-level argument to GIN barrier ``sync``.
  - :class:`GinBackendMask` — backend selection for :meth:`DevComm.gin`.
"""

from enum import IntEnum, IntFlag


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


class GinFenceLevel(IntFlag):
    """Mirrors ``enum class ncclGinFenceLevel``.

    Bit-flag composable: ``GinFenceLevel.PUT | GinFenceLevel.GET`` drains
    both prior puts and prior gets on the bound GIN context.
    """

    NONE = 0           # pure sync, no drain
    PUT = 1 << 0       # 1: drain prior puts
    GET = 1 << 1       # 2: drain prior gets


class GinBackendMask(IntFlag):
    """Mirrors the device-side GIN backend selection mask
    (``NCCL_GIN_BACKEND_MASK_*``). Bit positions match the
    ``NCCL_NET_DEVICE_GIN_*`` enumerators in ``net_device.h``.
    """

    PROXY = 1 << 2     # 4
    GDAKI = 1 << 3     # 8
    GPI = 1 << 4       # 16
    ALL = PROXY | GDAKI | GPI


__all__ = [
    "MemoryOrder",
    "GinFenceLevel",
    "GinBackendMask",
]
