"""Experimental CuTeDSL bindings for the NCCL device API.

.. warning::

    Experimental and unstable. This subpackage carries no semantic-versioning
    guarantee: symbols may change or be removed in any release before nccl4py
    1.0.

    The selected ``libnccl_device.bc`` must match exactly the NCCL version
    nccl4py was built against -- the ``bindings`` version reported by
    ``nccl.core.show_versions()``. Resolution prefers
    ``$NCCL_HOME/lib/libnccl_device.bc`` and otherwise uses the copy from the
    installed ``nvidia-nccl-cu12`` / ``nvidia-nccl-cu13`` wheel. No version
    check is performed; a mismatch can cause link errors or incorrect kernel
    behavior.

See ``examples/cute/00_basic.py`` for a complete, runnable example.
"""

try:
    import cutlass.cute  # noqa: F401
except ImportError as e:
    raise ImportError(
        "nccl.core.device.cute requires the nvidia-cutlass-dsl package. "
        "Install it via the matching extra:\n"
        "    pip install 'nccl4py[cu12]'   # for CUDA 12\n"
        "    pip install 'nccl4py[cu13]'   # for CUDA 13"
    ) from e

from . import types, coop, handles, comm, window, gin, barrier, runtime
from .types import *    # MemoryOrder, ThreadScope, GinFenceLevel, GinBackendMask, GinResourceSharingMode
from .coop import *     # Coop, cta, warp, thread, lanes, warp_span
from .handles import *  # MultimemHandle, LsaBarrierHandle, GinBarrierHandle
from .comm import *     # Team, DevComm
from .window import *   # Window
from .gin import *      # Gin
from .barrier import *  # session classes + factories

__all__ = [
    "types",
    "coop",
    "handles",
    "comm",
    "window",
    "gin",
    "barrier",
    "runtime",
    *types.__all__,
    *coop.__all__,
    *handles.__all__,
    *comm.__all__,
    *window.__all__,
    *gin.__all__,
    *barrier.__all__,
]
