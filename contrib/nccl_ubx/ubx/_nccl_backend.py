"""NCCL4py-backed primitives for the symmetric memory pool.

Encapsulates the NCCL Communicator / window / devcomm lifecycle so
``SymmAllocator`` can replace its ``torch.distributed._symmetric_memory``
dependency without polluting the rest of the file with NCCL-specific
plumbing.

The migration path is hard-cutover (no env-flag side-by-side with the old
torch._symm path); ``NcclSymPool`` is the single source of truth for the
pool's device pointer, the registered window, and the device comm handle.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence, Tuple

import torch
import torch.distributed as dist


# nccl4py is container-provided. Import is best-effort so unit tests can
# inspect the module shape on machines without nccl4py installed.
try:
    import nccl.bindings as _nccl_bindings
    from nccl.core.communicator import Communicator, NCCLDevCommRequirements
    from nccl.core.constants import WindowFlag
    from nccl.core.utils import UniqueId, get_unique_id

    _NCCL4PY_AVAILABLE = True
    _NCCL4PY_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover - depends on environment
    _NCCL4PY_AVAILABLE = False
    _NCCL4PY_IMPORT_ERROR = _e


def nccl4py_available() -> bool:
    """Whether the nccl4py package is importable in this process."""
    return _NCCL4PY_AVAILABLE


def _debug_enabled() -> bool:
    return bool(os.environ.get("UBX_DEBUG"))


# ---------------------------------------------------------------------------
# Communicator acquisition
# ---------------------------------------------------------------------------

def _try_extract_nccl_comm_from_pg(pg, device: torch.device) -> Optional[int]:
    """Best-effort: pull the raw ncclComm_t (int) out of a torch ProcessGroup.

    Returns the handle on success, or ``None`` on any failure (older
    PyTorch, non-NCCL backend, hidden attribute, etc.). Caller is expected
    to fall back to a bootstrap path on ``None``.
    """
    try:
        backend = pg._get_backend(device)
    except Exception:
        return None
    # Probe attribute names in order of likelihood across PyTorch versions.
    candidates: Sequence[str] = (
        "comm", "_comm",
        "get_handle", "_get_handle",
        "get_nccl_comm", "_get_nccl_comm",
        "nccl_comm", "_nccl_comm",
    )
    for attr in candidates:
        try:
            obj = getattr(backend, attr, None)
            if obj is None:
                continue
            handle = obj() if callable(obj) else obj
            if isinstance(handle, int) and handle != 0:
                return handle
        except Exception:
            continue
    return None


def _bootstrap_nccl_comm(
    pg: "dist.ProcessGroup",
    world_size: int,
    rank: int,
    device: torch.device,
) -> "Communicator":
    """Build a fresh ncclComm_t using ``pg`` for out-of-band uid broadcast."""
    if rank == 0:
        uid = get_unique_id()
        uid_bytes = bytes(uid)
    else:
        uid_bytes = b"\x00" * 128

    # NCCL backend broadcasts must use a GPU tensor; bytes -> uint8 tensor.
    uid_tensor = torch.tensor(list(uid_bytes), dtype=torch.uint8, device=device)
    # `group_src=0` selects rank 0 *within* pg, which matches the
    # `rank == 0` source check above. Passing `src=0` would treat 0 as a
    # global rank, which fails for EP subgroups that don't contain global
    # rank 0.
    dist.broadcast(uid_tensor, group_src=0, group=pg)
    received = bytes(uid_tensor.cpu().tolist())
    received_uid = UniqueId.from_bytes(received)

    comm = Communicator()
    comm.initialize(nranks=world_size, rank=rank, unique_id=received_uid)
    return comm


def get_or_create_nccl_comm(
    pg: "dist.ProcessGroup",
    device: torch.device,
    world_size: int,
    rank: int,
) -> Tuple["Communicator", bool]:
    """Reuse the existing PG's ncclComm_t when possible; otherwise bootstrap.

    Returns ``(comm, owned)``. ``owned=True`` means we bootstrapped a fresh
    ncclComm_t and the caller must call ``comm.destroy()`` to release it.
    ``owned=False`` means we aliased the PG's underlying ncclComm_t and
    destroying it would break the parent PG.
    """
    if not _NCCL4PY_AVAILABLE:
        raise RuntimeError(
            "nccl4py is required for NCCL-backed SymmAllocator but failed "
            f"to import: {_NCCL4PY_IMPORT_ERROR!r}"
        )
    # UBX_FORCE_BOOTSTRAP=1: skip the PG-handle extraction path and always
    # bootstrap a fresh ncclComm_t via uid broadcast. Use this when you
    # suspect that the reused handle is wrong (e.g. multi-PG environments
    # where `_get_backend(device)` may return a handle bound to a different
    # PG than `pg`).
    if os.environ.get("UBX_FORCE_BOOTSTRAP", "0") == "1":
        if _debug_enabled():
            print("[ubx._nccl_backend] UBX_FORCE_BOOTSTRAP=1; skipping reuse")
        return _bootstrap_nccl_comm(pg, world_size, rank, device)
    raw = _try_extract_nccl_comm_from_pg(pg, device)
    if raw is not None:
        if _debug_enabled():
            print(f"[ubx._nccl_backend] reusing PG ncclComm_t = {raw:#x}")
        return Communicator(raw), False
    if _debug_enabled():
        print("[ubx._nccl_backend] PG handle extraction failed; "
              "bootstrapping a fresh ncclComm_t via uid broadcast")
    return _bootstrap_nccl_comm(pg, world_size, rank, device), True


# ---------------------------------------------------------------------------
# Pool: ncclMemAlloc + window register + devcomm create
# ---------------------------------------------------------------------------

def _wrap_device_ptr_as_tensor(
    ptr: int, size: int, device: torch.device,
) -> torch.Tensor:
    """Zero-copy wrap of an externally-allocated device pointer as torch.Tensor.

    Uses the ``__cuda_array_interface__`` protocol so the resulting tensor
    is a view (no copy). The caller is responsible for keeping ``ptr`` alive
    for the tensor's lifetime — ``NcclSymPool.close()`` enforces this.
    """
    class _CAIView:
        __cuda_array_interface__ = {
            "version": 3,
            "shape": (size,),
            "typestr": "|u1",
            "data": (ptr, False),  # not read-only
            "strides": None,
        }

    return torch.as_tensor(_CAIView(), device=device)


class NcclSymPool:
    """NCCL-allocated symmetric memory pool with a registered window + devcomm.

    Drop-in replacement for the (pool buffer, multicast_ptr) pair that
    ``torch.distributed._symmetric_memory.rendezvous`` used to return,
    backed instead by NCCL device-API primitives. Per-peer LSA pointers
    are no longer pre-resolved here — kernels call ncclGetLsaPointer
    directly on the registered window.

    Lifecycle:

    1. ``ncclMemAlloc(size_bytes)`` -> raw device pointer
    2. Wrap as ``torch.Tensor`` via __cuda_array_interface__
    3. ``comm.register_window(...)`` -> RegisteredWindowHandle
    4. ``comm.create_dev_comm(...)`` -> DevCommResource (with LSA multimem)
    5. Resolve LSA multimem ptr (offset 0) and per-rank LSA peer ptrs

    Tear-down (close()): destroy devcomm -> deregister window ->
    ``ncclMemFree``.
    """

    def __init__(
        self,
        size_bytes: int,
        device: torch.device,
        comm: "Communicator",
        world_size: int,
        enable_multicast: bool = True,
    ):
        if not _NCCL4PY_AVAILABLE:
            raise RuntimeError(
                "nccl4py is required for NcclSymPool but failed to import: "
                f"{_NCCL4PY_IMPORT_ERROR!r}"
            )
        self._size = int(size_bytes)
        self._device = device
        self._comm = comm
        self._world_size = int(world_size)
        # Pre-init for safe close() on partial construction failure
        self._raw_ptr: int = 0
        self._internal_pool: Optional[torch.Tensor] = None
        self._window = None
        self._dev_comm = None
        self._mc0_ptr: int = 0

        import os
        _r = int(os.environ.get("RANK", "-1"))
        _diag = os.environ.get("UBX_INIT_DIAG", "0") == "1"

        # 1. ncclMemAlloc
        if _diag:
            print(f"[r{_r} NcclSymPool] step1 PRE mem_alloc({self._size})", flush=True)
        self._raw_ptr = int(_nccl_bindings.mem_alloc(self._size))
        if self._raw_ptr == 0:
            raise RuntimeError(f"ncclMemAlloc({self._size}) returned NULL")
        if _diag:
            print(f"[r{_r} NcclSymPool] step1 POST mem_alloc ptr={self._raw_ptr:#x}", flush=True)

        # 2. wrap as torch.Tensor (zero-copy)
        self._internal_pool = _wrap_device_ptr_as_tensor(
            self._raw_ptr, self._size, device,
        )
        if _diag:
            print(f"[r{_r} NcclSymPool] step2 PRE fill_(0)", flush=True)
        self._internal_pool.fill_(0)
        if _diag:
            print(f"[r{_r} NcclSymPool] step2 POST fill_(0)", flush=True)

        # 3. register as a symmetric window
        if _diag:
            print(f"[r{_r} NcclSymPool] step3 PRE register_window", flush=True)
        self._window = comm.register_window(
            self._internal_pool, flags=WindowFlag.CollSymmetric,
        )
        if _diag:
            print(f"[r{_r} NcclSymPool] step3 POST register_window", flush=True)
        if self._window is None:
            raise RuntimeError(
                "comm.register_window returned None; "
                "this NCCL configuration may not support symmetric windows"
            )

        # 4. devcomm
        reqs = NCCLDevCommRequirements()
        reqs.lsa_multimem = bool(enable_multicast)
        if _diag:
            print(f"[r{_r} NcclSymPool] step4 PRE create_dev_comm lsa_mc={reqs.lsa_multimem}", flush=True)
        self._dev_comm = comm.create_dev_comm(reqs)
        if _diag:
            print(f"[r{_r} NcclSymPool] step4 POST create_dev_comm", flush=True)

        # 5. resolve LSA multimem pointer (host-side, once at init).
        # Per-rank LSA peer pointers are no longer pre-resolved — kernels
        # call ncclGetLsaPointer directly via the registered window, and
        # MC launchers call nccl_lsa_mc_ptr() with the right offset on
        # demand. We just need mc0_ptr cached here so multicast_available
        # reflects whether the window has a usable LSA multimem handle.
        if enable_multicast:
            mc = self._window.get_lsa_multimem_device_pointer(offset=0)
            self._mc0_ptr = int(mc) if mc is not None else 0
        else:
            self._mc0_ptr = 0

        if _debug_enabled():
            print(
                f"[ubx._nccl_backend] NcclSymPool: "
                f"size={self._size} pool_ptr={self._raw_ptr:#x} "
                f"window={self.window_handle:#x} "
                f"devcomm={self.dev_comm_handle:#x} "
                f"mc0_ptr={self._mc0_ptr:#x} mc_avail={self.multicast_available}"
            )

    # -- read-only views the SymmAllocator consumes --------------------------

    @property
    def pool_ptr(self) -> int:
        return self._raw_ptr

    @property
    def internal_pool(self) -> torch.Tensor:
        assert self._internal_pool is not None, "NcclSymPool already closed"
        return self._internal_pool

    @property
    def mc0_ptr(self) -> int:
        return self._mc0_ptr

    @property
    def multicast_available(self) -> bool:
        return self._mc0_ptr != 0

    @property
    def window_handle(self) -> int:
        return int(self._window.handle) if self._window is not None else 0

    @property
    def dev_comm_handle(self) -> int:
        return int(self._dev_comm.ptr) if self._dev_comm is not None else 0

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def size(self) -> int:
        return self._size

    # -- explicit lifecycle --------------------------------------------------

    def close(self) -> None:
        """Tear down devcomm -> window -> pool buffer, in order."""
        # devcomm first (it references the window)
        if self._dev_comm is not None:
            try:
                self._dev_comm.close()
            except Exception:
                pass
            self._dev_comm = None
        if self._window is not None:
            try:
                self._window.close()
            except Exception:
                pass
            self._window = None
        # Drop the tensor view BEFORE freeing the underlying allocation,
        # so torch can't hand the pointer to a kernel after free.
        self._internal_pool = None
        if self._raw_ptr != 0:
            try:
                _nccl_bindings.mem_free(self._raw_ptr)
            except Exception:
                pass
            self._raw_ptr = 0

    def __del__(self):
        # Best-effort; explicit close() is preferred and verifiable in tests.
        try:
            self.close()
        except Exception:
            pass
