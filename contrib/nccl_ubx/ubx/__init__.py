"""UB-X (Ultra Bandwidth X) — low-latency NVLink collectives library.

Provides GPU communication primitives backed by NCCL's symmetric-memory
device API: an in-pool ``SymmAllocator``, fused residual+RMSNorm allreduce,
multicast/Lamport allreduce variants, allgather, alltoall (regular +
Lamport), alltoallv, and bf16->mxfp8 token dispatch for MoE.

Hardware target: SM 9.0+ (Hopper, Blackwell). Requires NVLink multicast
for the MC kernels; UC paths work with multicast disabled.

Versioning:
  - ``__version__`` is the package version (release-level cadence).
  - ``version`` / ``get_version()`` is the API contract version,
    bumped on every public-API change. Downstream code can call
    ``ubx.query_api(name, params=[...])`` to detect drift early and
    ``ubx.query_supported_hw()`` to discover what transports / SM
    archs are usable in this build.
"""

from .allocator import SymmAllocator
from .tensor import SymmTensor
from .ops import compute_token_offsets, compute_combine_push_map, compute_dispatch_topk_map
from . import ops
from . import fused
from ._api_registry import (
    API_VERSION as version,
    get_version,
    query_api,
    query_supported_hw,
)

__version__ = "1.1.0"


__all__ = [
    "SymmAllocator",
    "SymmTensor",
    "compute_token_offsets",
    "compute_combine_push_map",
    "compute_dispatch_topk_map",
    "ops",
    "fused",
    "__version__",
    # API contract version + query interface (see _api_registry.py).
    "version",
    "get_version",
    "query_api",
    "query_supported_hw",
]
