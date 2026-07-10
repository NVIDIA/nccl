"""SymmAllocator: symmetric memory pool allocator for UB-X collectives.

Backed by nccl4py (ncclMemAlloc + ncclCommWindowRegister + ncclDevCommCreate)
via :mod:`._nccl_backend`. The allocator exposes ``pool_ptr``,
``internal_pool``, and ``mc0_ptr`` so callers can compute pool-byte
offsets via ``tensor.data_ptr() - pool_ptr``. Kernels resolve LSA peer /
multicast pointers via the NCCL device API on the registered window
(see ``csrc/ubx.cu``).
"""

from __future__ import annotations

import os
import torch
import weakref
from typing import Dict, List, Tuple, Optional
from threading import Lock

from ._nccl_backend import NcclSymPool, get_or_create_nccl_comm
from .tensor import SymmTensor

# Import C extension functions
from ubx._C import (
    ubx_allreduce_2shot_uc,
    ubx_allreduce_2shot_mc,
    ubx_allreduce_2shot_mc_lamport,
    ubx_allgather_mc,
    ubx_allgather_uc,
    ubx_alltoall,
    ubx_alltoall_lamport,
    ubx_a2av_token_bf16_mxfp8,
    ubx_a2av_token_bf16_mxfp8_persistent,
    ubx_a2av_token_bf16_bf16,
    ubx_a2av_token_bf16_bf16_topk,
    ubx_combine_bf16_bf16,
    ubx_combine_mxfp8_bf16,
    ubx_combine_bf16_bf16_lamport_push,
    ubx_combine_bf16_bf16_push,
    ubx_combine_wait_bf16,
    ubx_combine_wait_mxfp8,
    ubx_set_timeout,
)
# ubx_barrier is new — present only in containers built after 2026-05-19.
# Tolerate older containers that don't have it.
try:
    from ubx._C import ubx_barrier  # type: ignore[attr-defined]
except ImportError:
    ubx_barrier = None  # type: ignore[assignment]
# E22: combine v2 — new in containers built after 2026-05-21.
try:
    from ubx._C import (  # type: ignore[attr-defined]
        ubx_combine_v2_phase1_bf16,
        ubx_combine_v2_phase2_bf16,
    )
except ImportError:
    ubx_combine_v2_phase1_bf16 = None  # type: ignore[assignment]
    ubx_combine_v2_phase2_bf16 = None  # type: ignore[assignment]

# Diagnostic: per-rank peer-pointer dump. Container build 2026-05-26+.
try:
    from ubx._C import ubx_peer_ptr_dump  # type: ignore[attr-defined]
except ImportError:
    ubx_peer_ptr_dump = None  # type: ignore[assignment]
try:
    from ubx._C import ubx_peer_atomic_test  # type: ignore[attr-defined]
except ImportError:
    ubx_peer_atomic_test = None  # type: ignore[assignment]
try:
    from ubx._C import (  # type: ignore[attr-defined]
        ubx_combine_push3_phase1_write,
        ubx_combine_push3_phase2_signal,
        ubx_combine_push3_phase3_sum,
    )
except ImportError:
    ubx_combine_push3_phase1_write = None   # type: ignore[assignment]
    ubx_combine_push3_phase2_signal = None  # type: ignore[assignment]
    ubx_combine_push3_phase3_sum = None     # type: ignore[assignment]

# `auto` allreduce / alltoall picks Lamport at or below this total tensor size,
# and the "simple" variant (MC for allreduce, UC for alltoall) above it.
AUTO_SWITCH_BYTES = 256 * 1024


class SymmAllocator:
    """Symmetric memory pool allocator for UB-X communication primitives.

    Manages a pool of symmetric memory accessible by all ranks in a process group.
    Supports both graph-mode and eager-mode allocations with separate pool regions.
    """

    # Number of int32 flag slots reserved in REG0 after the commbuff. Kernels
    # index flags off REG0_FLAG_OFFSET = world_size * sizeof(void*); the highest
    # slot used by any kernel in csrc/ubx.cu is UBX_FLAG_PUSH3_ID = 54. Kept at
    # 128 for margin — if a new flag slot >= this value is added in ubx.cu, bump
    # this constant (and reg0 will grow to fit via _reg0_size_for).
    _MAX_FLAG_SLOTS = 128

    @staticmethod
    def _reg0_size_for(world_size: int) -> int:
        """Bytes reserved for REG0, 4 KiB-aligned: the legacy commbuff
        (``world_size * 8``) plus the kernel flag slots
        (``_MAX_FLAG_SLOTS`` int32s).

        Must be large enough that the flag region never spills past REG0 into
        the graph pool. The pre-fix value was a fixed 1024 B — once
        ``world_size * 8`` reached that bound the flags overlapped the graph
        pool's first user allocation, and under cudagraph mode peer data
        writes then corrupted the BAR counter, hanging UC alltoall (and
        Lamport, which shares the entry-barrier path). Regression:
        tests/test_allocator_pool.py.
        """
        need = world_size * 8 + SymmAllocator._MAX_FLAG_SLOTS * 4
        return ((max(1024, need) + 4095) // 4096) * 4096

    def __init__(self, size_bytes: int, device: torch.device,
                 dist_group: torch.distributed.group):
        """Initialize the allocator with a preallocated NCCL-backed pool."""
        self.device = device
        self.world_size = torch.distributed.get_world_size(dist_group)
        self.myrank = torch.distributed.get_rank(dist_group)
        self.dist_group = dist_group
        # REG0 holds the legacy commbuff followed by the kernel flag slots;
        # it must scale with world_size so the flags stay inside REG0 (see
        # _reg0_size_for for the cudagraph BAR-counter corruption this
        # prevents at large world_size).
        self.reg0_size = self._reg0_size_for(self.world_size)

        alignment = 2 * 1024 * 1024  # NCCL allocates in 2 MB pages
        self.pool_size = int((size_bytes + alignment - 1) / alignment) * alignment

        # NCCL_NVLS_ENABLE=0 → NVLS-disabled mode (no NVLink multicast).
        # Plumbed into NCCLDevCommRequirements.lsa_multimem; also affects
        # NCCL's own collectives so the multicast-disable signal is unified.
        enable_mc = os.environ.get("NCCL_NVLS_ENABLE") != "0"

        # Reuse the PG's underlying ncclComm if available, else bootstrap a
        # fresh one. ``_owns_comm`` controls whether close() destroys it —
        # destroying the PG's comm would break the parent PG.
        self._nccl_comm, self._owns_comm = get_or_create_nccl_comm(
            dist_group, device, self.world_size, self.myrank,
        )

        # ncclMemAlloc + register window + create dev comm.
        self._nccl_pool = NcclSymPool(
            size_bytes=self.pool_size,
            device=device,
            comm=self._nccl_comm,
            world_size=self.world_size,
            enable_multicast=enable_mc,
        )

        # Preserve external contract: pool_ptr / internal_pool / mc0_ptr
        # remain the same shape kernels and tests already depend on.
        self.internal_pool = self._nccl_pool.internal_pool
        self.pool_ptr = self._nccl_pool.pool_ptr
        self.mc0_ptr = self._nccl_pool.mc0_ptr

        # Phase 5 cleanup: the REG0 commbuff[] write is gone. After Phase 4
        # all kernels resolve peer pointers via the NCCL device API
        # (ncclGetLsaPointer for UC, host-side ncclGetLsaMultimemDevicePointer
        # for MC). The first RANKS*sizeof(void*) bytes of REG0 are now
        # reserved-but-unused; sync flags follow at REG0_FLAG_OFFSET. Phase
        # 7 follow-up will compact REG0 (move flags to offset 0).

        # Synchronize all processes before proceeding.
        torch.distributed.barrier(group=dist_group)
        # NCCL prewarm: exercise every collective primitive the training
        # path will use on this group so lazy channel setup happens HERE,
        # not at the first runtime call. Without this, jobs running in
        # NVLS-disabled mode can hit a 10-minute watchdog timeout on the
        # first runtime _ALLGATHER_BASE — one rank's NCCL channel-init
        # for that op is still mid-bootstrap while peers have already
        # enqueued the call.
        if os.environ.get("UBX_PREWARM_NCCL", "1") != "0":
            try:
                _prewarm_elems = int(os.environ.get("UBX_PREWARM_NCCL_ELEMS", "262144"))
                _src = torch.zeros(_prewarm_elems, dtype=torch.bfloat16, device=device)
                _dst = torch.zeros(_prewarm_elems * self.world_size, dtype=torch.bfloat16, device=device)
                torch.distributed.all_gather_into_tensor(_dst, _src, group=dist_group)
                # also exercise alltoall_single (megatron's MoE token a2av)
                _a2a_send = torch.zeros(self.world_size * 16, dtype=torch.bfloat16, device=device)
                _a2a_recv = torch.empty_like(_a2a_send)
                torch.distributed.all_to_all_single(_a2a_recv, _a2a_send, group=dist_group)
                torch.cuda.synchronize()
                del _src, _dst, _a2a_send, _a2a_recv
            except Exception as _e:
                if int(os.environ.get("RANK", "-1")) == 0:
                    print(f"[ubx.allocator] prewarm NCCL skipped: {type(_e).__name__}: {_e}", flush=True)
        torch.distributed.barrier(group=dist_group)
        # Track allocated segments: (offset, size, reference count)
        self.allocated: Dict[bool, List[Tuple[int, int, int]]] = {True: [], False: []}
        # Track free segments: (offset, size)
        self.graph_pool_size = int(
            self.pool_size * float(os.environ.get("UBX_GRAPH_POOL_SHARE", 0.9))
        )
        self.graph_pool_size = int(self.graph_pool_size // 4096) * 4096

        # Pool layout: [ REG0 | graph_pool | non_graph_pool ]
        # Both pool segments MUST start at or beyond reg0_size — REG0 holds
        # the NCCL barrier flag region (NVLS1_ID / NVLS1_BAR / A2AV_*) plus
        # the legacy commbuff[] block. An allocation that overlaps REG0
        # silently corrupts the cross-rank sync protocol: the very kernel
        # that allocated the tensor stomps on the flags during its data
        # write, and the next kernel reads garbage `reduce_id` → barrier
        # protocol breaks → silent data corruption.
        #
        # The previous formula put non_graph at `[graph_pool_size, ...)`
        # which equals `[0, pool_size)` when UBX_GRAPH_POOL_SHARE=0 — i.e.
        # the very first eager allocation lands at pool offset 0 and
        # clobbers REG0. Fixed by clamping non_graph start to ≥ reg0_size.
        graph_lo = self.reg0_size
        graph_hi = max(self.reg0_size, self.graph_pool_size)
        nongraph_lo = graph_hi
        nongraph_hi = self.pool_size
        self.freelist: Dict[bool, List[Tuple[int, int]]] = {
            True:  [(graph_lo, graph_hi - graph_lo)]
                    if graph_hi > graph_lo else [],
            False: [(nongraph_lo, nongraph_hi - nongraph_lo)]
                    if nongraph_hi > nongraph_lo else [],
        }
        self.nextpoisoned: Dict[bool, Optional[SymmTensor]] = {True: None, False: None}
        # alltoall_lamport state: 4 persistent symmetric buffers per
        # (graph_mode, shape, dtype) + a host counter. On the first call
        # for a given key, we allocate 4 buffers and fill buf[0..2]
        # with the Lamport poison sentinel; buf[3] gets its first
        # poison from call N=0's in-kernel clear-write. Only the first
        # call (counter==0) runs with the UC entry barrier engaged;
        # subsequent calls roll without barrier on a 4-buffer cycle.
        # The 2-kernel gap between in-kernel poison-write and the
        # corresponding poll is now safe because the dual-kernel split
        # (write + poll) flushes UC writes at the kernel boundary
        # before phase 2 polls — the older monolithic-kernel race that
        # motivated 5 buffers is gone.
        self.lamport_a2a_state: Dict[Tuple, Dict] = {}
        # Triple buffer for combine PUSH-Lamport (race-free variant).
        # Bufs are sized [local_ntokens, topk_max, hidden] bf16 (NOT
        # [max_tokens_per_rank, hidden] like the PULL variant), so
        # max_tokens_per_rank-driven sizing changes between calls require
        # only routing-shape stability, not max_tpr stability.
        self.combine_push_triple_buf: Dict[bool, List[Optional[SymmTensor]]] = {
            True: [None, None, None], False: [None, None, None],
        }
        self.combine_push_call_count: Dict[bool, int] = {True: 0, False: 0}
        # Double buffer for PUSH non-Lamport combine. Two dest bufs alternating
        # — call N writes/reads bufs[N%2], call N+1 uses the other so peer's
        # call N+1 Phase 1 push doesn't race with my call N Phase 2 read.
        self.combine_push_nl_double_buf: Dict[bool, List[Optional[SymmTensor]]] = {
            True: [None, None], False: [None, None],
        }
        self.combine_push_nl_call_count: Dict[bool, int] = {True: 0, False: 0}
        self.residual = None
        self.residual_global = None
        self.residual_tokens = 0
        self.tensors = weakref.WeakSet()
        self.lock = Lock()
        self.nchunks = 1
        self.current_chunk = 0
        self.block_align = int(os.environ.get("UBX_BLOCK_ALIGN", 4096))
        self.dummy = os.environ.get("UBX_DUMMY")
        self.debug = os.environ.get("UBX_DEBUG")
        # UBX_MAXSM overrides per-kernel compiled SM-grid defaults (32/64).
        # Read once here; forwarded into every kernel launch as `default_sms`
        # so the hot path never touches getenv. 0 = use compiled default.
        try:
            _ubx_maxsm = int(os.environ.get("UBX_MAXSM", "0") or 0)
        except ValueError:
            _ubx_maxsm = 0
        self.default_sms = _ubx_maxsm if 0 < _ubx_maxsm <= 128 else 0

        # UBX_TIMEOUT_SEC: kernel polling-loop timeout (default 30s, assuming
        # ~2 GHz GPU clock).  Effective only when the extension was compiled
        # with -DUB_TIMEOUT_ENABLED (UBX_BUILD_TIMEOUT=1 at pip install time);
        # otherwise the setter is a no-op and kernels poll forever.  Read
        # once here, pushed into __constant__ device memory via the setter.
        try:
            _ubx_timeout_sec = float(os.environ.get("UBX_TIMEOUT_SEC", "30") or 30)
        except ValueError:
            _ubx_timeout_sec = 30.0
        ubx_set_timeout(int(_ubx_timeout_sec * 2_000_000_000))
        if self.debug:
            print(f"Rank {self.myrank} Graph pool size: {self.graph_pool_size}")
            print(f"Rank {self.myrank} Non-graph pool size: {self.pool_size - self.graph_pool_size}")
            print(f"Rank {self.myrank} Reg0 size: {self.reg0_size}")
            print(f"Rank {self.myrank} Total pool size: {self.pool_size}")
        self.used_uc = False
        self.used_simple = False
        self.lamport_out = None
        self.lamport_poisoned = False

    @property
    def rank(self) -> int:
        return self.myrank

    @property
    def multicast_available(self) -> bool:
        return self.mc0_ptr is not None and self.mc0_ptr != 0

    @property
    def dev_comm_handle(self) -> int:
        """Raw ncclDevComm_t for pybind11 plumbing (Phase 3+ kernel rewrite).

        Resolves to ``0`` if the allocator has been closed.
        """
        return self._nccl_pool.dev_comm_handle if self._nccl_pool is not None else 0

    @property
    def window_handle(self) -> int:
        """Raw ncclWindow_t for pybind11 plumbing (Phase 3+ kernel rewrite).

        Resolves to ``0`` if the allocator has been closed.
        """
        return self._nccl_pool.window_handle if self._nccl_pool is not None else 0

    def close(self) -> None:
        """Release the NCCL pool, window, devcomm, and bootstrapped comm.

        Tears down in the correct order: dev_comm → window → buffer free →
        bootstrapped ncclComm_t. Idempotent. Called automatically on
        garbage collection but should be invoked explicitly in tests /
        lifecycle-sensitive code.
        """
        if getattr(self, "_nccl_pool", None) is not None:
            self._nccl_pool.close()
            self._nccl_pool = None
        # Destroy the bootstrapped ncclComm_t, gated on ownership: if we
        # extracted (reused) the PG's comm, destroy() would break the
        # parent PG. Without this, every SymmAllocator instance leaks an
        # ncclComm_t and its device-side state.
        if getattr(self, "_owns_comm", False) and getattr(self, "_nccl_comm", None) is not None:
            try:
                self._nccl_comm.destroy()
            except Exception:
                pass
        self._nccl_comm = None
        self._owns_comm = False
        # Drop external aliases to fail loudly if anyone keeps using them.
        self.internal_pool = None
        self.pool_ptr = 0
        self.mc0_ptr = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def allocate(self, nbytes: int) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Allocate nbytes from the pool, returning a pointer and pool reference."""
        graph_mode = torch.cuda.is_current_stream_capturing()
        with self.lock:
            for i, (offset, size) in enumerate(self.freelist[graph_mode]):
                if size >= nbytes:
                    self.freelist[graph_mode].pop(i)
                    self.allocated[graph_mode].append((offset, nbytes, 0))
                    if size > nbytes:
                        self.freelist[graph_mode].append((offset + nbytes, size - nbytes))
                    if self.debug:
                        print(f"Rank {self.myrank} Allocated {nbytes} bytes at {offset} "
                              f"cudagraph capturing: {graph_mode}")
                    import os as _os_at
                    if _os_at.environ.get("UBX_ALLOC_TRACE", "0") == "1" and self.myrank == 0:
                        print(f"[ALLOC-TRACE r{self.myrank}] ALLOC offset={offset} "
                              f"size={nbytes} end={offset + nbytes} graph={int(graph_mode)}",
                              flush=True)
                    return self.pool_ptr + offset, self.internal_pool
            if self.debug:
                print(f"Rank {self.myrank} No suitable free segment found for {nbytes} bytes, "
                      f"allocated list: {self.allocated[graph_mode]}, "
                      f"free list: {self.freelist[graph_mode]} "
                      f"cudagraph capturing mode: {graph_mode}")
            return None, None

    def allocated_change(self, ptr: int, change: int):
        """Change reference count for allocation at ptr."""
        offset = ptr - self.pool_ptr
        graph_mode = offset < self.graph_pool_size
        with self.lock:
            for i, (alloc_offset, size, ref_count) in enumerate(self.allocated[graph_mode]):
                if alloc_offset == offset:
                    self.allocated[graph_mode].pop(i)
                    ref_count += change
                    if ref_count == 0:
                        self.freelist[graph_mode].append((offset, size))
                        self.freelist[graph_mode].sort(key=lambda x: x[0])
                        self._merge_free_segments(graph_mode)
                        if self.debug:
                            print(f"Rank {self.myrank} Freed {size} bytes at {offset} "
                                  f"cudagraph capturing: {graph_mode}")
                        import os as _os_at
                        if _os_at.environ.get("UBX_ALLOC_TRACE", "0") == "1" and self.myrank == 0:
                            print(f"[ALLOC-TRACE r{self.myrank}] FREE  offset={offset} "
                                  f"size={size} end={offset + size} graph={int(graph_mode)}",
                                  flush=True)
                    else:
                        self.allocated[graph_mode].append((offset, size, ref_count))
                        if self.debug:
                            print(f"Rank {self.myrank} Refcount changed to {ref_count} "
                                  f"for {size} bytes at {offset} cudagraph capturing: {graph_mode}")
                    return
            # Ignore invalid pointers silently
            pass

    def free(self, ptr: int):
        """Free the memory at ptr, returning it to the pool."""
        self.allocated_change(ptr, -1)

    def _merge_free_segments(self, graph_mode: bool):
        """Merge adjacent free segments to reduce fragmentation."""
        if not self.freelist[graph_mode]:
            return
        merged = []
        current_offset, current_size = self.freelist[graph_mode][0]
        for offset, size in self.freelist[graph_mode][1:]:
            if current_offset + current_size == offset:
                current_size += size
            else:
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        merged.append((current_offset, current_size))
        self.freelist[graph_mode] = merged

    _MXFP8_BLOCK_SIZE = 32

    def create_tensor(
        self,
        shape: torch.Size,
        dtype: torch.dtype = torch.float32,
        blocked: Optional[str] = None,
    ) -> Optional[torch.Tensor]:
        """Create a SymmTensor using memory from the pool.

        Args:
            shape: Tensor shape (visible element dimensions).
            dtype: Element dtype.
            blocked: Blocked format. None = plain tensor. 'mxfp8' = MX FP8
                with one scale byte per 32 elements appended after data,
                data region aligned to UBX_BLOCK_ALIGN bytes.
        """
        if blocked is not None and blocked != "mxfp8":
            raise ValueError(f"Unsupported blocked format: {blocked!r}. Only 'mxfp8' is supported.")
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor(0, dtype=dtype).element_size()
        data_nbytes = element_size * num_elements

        metadata_offset: Optional[int] = None
        total_nbytes = data_nbytes

        if blocked == "mxfp8":
            data_nbytes_aligned = (
                (data_nbytes + self.block_align - 1) // self.block_align
            ) * self.block_align
            metadata_nbytes = (num_elements + self._MXFP8_BLOCK_SIZE - 1) // self._MXFP8_BLOCK_SIZE
            metadata_offset = data_nbytes_aligned
            total_nbytes = data_nbytes_aligned + metadata_nbytes

        ptr, pool = self.allocate(total_nbytes)
        if ptr is None:
            return None
        offset = ptr - self.pool_ptr
        tensor = SymmTensor(
            pool, offset, torch.Size(shape), dtype, self,
            blocked_format=blocked, metadata_offset=metadata_offset,
        )
        return tensor

    # === Pure collective operations ===

    @staticmethod
    def _check_not_blocked(tensor: torch.Tensor, op: str) -> None:
        if getattr(tensor, "_blocked_format", None) is not None:
            raise TypeError(
                f"{op} does not support blocked tensor formats "
                f"(got blocked_format={tensor._blocked_format!r})"
            )

    def allreduce_uc(
        self,
        tensor_in: torch.Tensor,
        hidden_size: int = 0,
        residual_in: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        fuse_layernorm: bool = False,
        gamma: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """Performs in-place allreduce using unicast (UC) path."""
        self._check_not_blocked(tensor_in, "allreduce_uc")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        nbytes = tensor_in.numel() * tensor_in.element_size() // tensor_in._allocator.nchunks
        chunk = tensor_in._allocator.current_chunk
        # Byte offset within the symmetric window for this rank's chunk.
        in_offset = (tensor_in.data_ptr() - self.pool_ptr) + nbytes * chunk

        ubx_allreduce_2shot_uc(
            self.world_size,
            self.myrank,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            in_offset, in_offset,  # in-place: out_offset == in_offset
            nbytes,
            (residual_in.data_ptr() + nbytes * chunk * residual_in.element_size()) if residual_in is not None else 0,
            (residual_out.data_ptr() + nbytes * chunk * residual_out.element_size()) if residual_out is not None else 0,
            fuse_layernorm,
            gamma.data_ptr() if gamma is not None else 0,
            eps if eps is not None else 0.0,
            hidden_size,
            self.default_sms,
            smlimit,
            cgasize,
            chunk,
            False,
        )
        tensor_in._allocator.current_chunk += 1
        if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks:
            tensor_in._allocator.current_chunk = 0
            tensor_in._allocator.used_uc = False
        return tensor_in

    def allreduce_mc(
        self,
        tensor_in: torch.Tensor,
        hidden_size: int = 0,
        residual_in: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        fuse_layernorm: bool = False,
        gamma: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """Performs in-place allreduce using multicast (MC) path."""
        self._check_not_blocked(tensor_in, "allreduce_mc")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        nbytes = tensor_in.numel() * tensor_in.element_size() // tensor_in._allocator.nchunks
        chunk = tensor_in._allocator.current_chunk
        in_offset = (tensor_in.data_ptr() - self.pool_ptr) + nbytes * chunk

        ubx_allreduce_2shot_mc(
            self.world_size,
            self.myrank,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            in_offset, in_offset,  # in-place: out_offset == in_offset
            nbytes,
            (residual_in.data_ptr() + nbytes * chunk * residual_in.element_size()) if residual_in is not None else 0,
            (residual_out.data_ptr() + nbytes * chunk * residual_out.element_size()) if residual_out is not None else 0,
            fuse_layernorm,
            gamma.data_ptr() if gamma is not None else 0,
            eps if eps is not None else 0.0,
            hidden_size,
            self.default_sms,
            smlimit,
            cgasize,
            chunk,
            False,
        )
        tensor_in._allocator.current_chunk += 1
        tensor_in._allocator.used_simple = True
        if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks:
            tensor_in._allocator.current_chunk = 0
            tensor_in._allocator.used_simple = False
        return tensor_in

    def allreduce_lamport(
        self,
        tensor_in: torch.Tensor,
        hidden_size: int = 0,
        residual_in: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        fuse_layernorm: bool = False,
        gamma: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """Performs allreduce using 2-shot multicast Lamport variant.

        Falls back to UC if multicast unavailable, or to MC if pool is exhausted.
        """
        self._check_not_blocked(tensor_in, "allreduce_lamport")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"
        if self.mc0_ptr is None or self.mc0_ptr == 0 or tensor_in._allocator.used_uc:
            return self.allreduce_uc(
                tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
                smlimit, cgasize
            )

        graph_mode = torch.cuda.is_current_stream_capturing()
        tensor_out = self.nextpoisoned[graph_mode] if tensor_in._allocator.current_chunk == 0 else self.lamport_out
        poisonedout = True if tensor_in._allocator.current_chunk == 0 else self.lamport_poisoned

        if tensor_in._allocator.current_chunk == 0 and (tensor_out is None or tensor_out.shape != tensor_in.shape):
            if self.nextpoisoned[graph_mode] is not None:
                del self.nextpoisoned[graph_mode]
                self.nextpoisoned[graph_mode] = None
            tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)
            poisonedout = False

        if tensor_out is None or tensor_in._allocator.used_simple:
            return self.allreduce_mc(
                tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
                smlimit, cgasize
            )
        tensor_in._allocator.lamport_out = tensor_out
        tensor_in._allocator.lamport_poisoned = poisonedout
        # allocate potential output for next allreduce (speculative) and poison it now
        if tensor_in._allocator.current_chunk == 0:
            self.nextpoisoned[graph_mode] = self.create_tensor(tensor_in.shape, tensor_in.dtype)

        nbytes = tensor_in.numel() * tensor_in.element_size() // tensor_in._allocator.nchunks
        chunk = tensor_in._allocator.current_chunk
        in_offset = (tensor_in.data_ptr() - self.pool_ptr) + nbytes * chunk
        out_offset = (tensor_out.data_ptr() - self.pool_ptr) + nbytes * chunk
        ucptr_out_arg = tensor_out.data_ptr() + nbytes * chunk
        clear_ptr_arg = (
            self.nextpoisoned[graph_mode].data_ptr() + nbytes * chunk
            if self.nextpoisoned[graph_mode] is not None else 0
        )

        ubx_allreduce_2shot_mc_lamport(
            self.world_size,
            self.myrank,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            ucptr_out_arg,
            in_offset, out_offset,
            clear_ptr_arg,
            nbytes,
            poisonedout,
            (residual_in.data_ptr() + residual_in.numel() // tensor_in._allocator.nchunks * chunk * residual_in.element_size()) if residual_in is not None else 0,
            (residual_out.data_ptr() + residual_out.numel() // tensor_in._allocator.nchunks * chunk * residual_out.element_size()) if residual_out is not None else 0,
            fuse_layernorm,
            gamma.data_ptr() if gamma is not None else 0,
            eps if eps is not None else 0.0,
            hidden_size,
            self.default_sms,
            smlimit,
            cgasize,
            chunk,
            False,
        )
        tensor_in._allocator.current_chunk += 1
        if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks:
            tensor_in._allocator.current_chunk = 0
            self.lamport_out = None
            self.lamport_poisoned = False
        return tensor_out

    def allreduce(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """Auto-selects best allreduce variant based on total tensor size."""
        if tensor_in.numel() * tensor_in.element_size() > AUTO_SWITCH_BYTES:
            return self.allreduce_mc(tensor_in, smlimit=smlimit, cgasize=cgasize)
        else:
            return self.allreduce_lamport(tensor_in, smlimit=smlimit, cgasize=cgasize)

    def allgather(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
    ) -> SymmTensor:
        """Allgather.

        Uses multicast (`ubx_allgather_mc`) when available, falls back to
        UC (`ubx_allgather_uc`) when ``mc0_ptr`` is unset (NVLS-disabled EP
        groups, large EP groups >36 ranks, etc.).
        """
        self._check_not_blocked(tensor_in, "allgather")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        tensor_out = self.create_tensor(
            torch.Size([self.world_size * tensor_in.shape[0], *tensor_in.shape[1:]]),
            tensor_in.dtype,
        )
        out_offset = tensor_out.data_ptr() - self.pool_ptr
        bytes_in = tensor_in.numel() * tensor_in.element_size()
        # UBX_AG_DBG=1: per-call pointer + alignment + first/last byte audit.
        # Used to localize the eager-mode "allgather corrupts layer 2's
        # global_routing_u8" bug.
        if os.environ.get("UBX_AG_DBG", "0") == "1":
            if not hasattr(SymmAllocator, "_ag_dbg_n"):
                SymmAllocator._ag_dbg_n = 0
            SymmAllocator._ag_dbg_n += 1
            _n = SymmAllocator._ag_dbg_n
            if _n <= 6 and self.myrank == 0:
                _in_first = tensor_in.flatten()[:4].tolist()
                _in_last  = tensor_in.flatten()[-4:].tolist()
                _in_sum   = int(tensor_in.int().sum().item())
                _out_first_before = tensor_out.flatten()[:4].tolist()
                print(
                    f"[AG_DBG r0 call#{_n}] tensor_in: dtype={tensor_in.dtype} "
                    f"shape={tuple(tensor_in.shape)} numel={tensor_in.numel()} "
                    f"bytes_in={bytes_in} (16B-aligned: {bytes_in % 16 == 0}) "
                    f"data_ptr={tensor_in.data_ptr():#x} (16B-aligned: {tensor_in.data_ptr() % 16 == 0}) "
                    f"sum={_in_sum} first={_in_first} last={_in_last}",
                    flush=True)
                print(
                    f"[AG_DBG r0 call#{_n}] tensor_out: shape={tuple(tensor_out.shape)} "
                    f"data_ptr={tensor_out.data_ptr():#x} "
                    f"pool_ptr={self.pool_ptr:#x} out_offset={out_offset} "
                    f"(out_offset 16B-aligned: {out_offset % 16 == 0}) "
                    f"out_first_before_kernel={_out_first_before} "
                    f"mc0_ptr={self.mc0_ptr:#x} (will_use={'MC' if self.mc0_ptr else 'UC'})",
                    flush=True)
        if self.mc0_ptr is None or self.mc0_ptr == 0:
            ubx_allgather_uc(
                self.world_size, self.myrank,
                self.dev_comm_handle, self.window_handle, self.pool_ptr,
                tensor_in.data_ptr(), out_offset, bytes_in,
                self.default_sms, smlimit,
            )
        else:
            ubx_allgather_mc(
                self.world_size, self.myrank,
                self.dev_comm_handle, self.window_handle, self.pool_ptr,
                tensor_in.data_ptr(), out_offset, bytes_in,
                self.default_sms, smlimit,
            )
        if os.environ.get("UBX_AG_DBG", "0") == "1":
            torch.cuda.synchronize()
            _n = SymmAllocator._ag_dbg_n
            if _n <= 6 and self.myrank == 0:
                _out_first = tensor_out.flatten()[:4].tolist()
                _out_last = tensor_out.flatten()[-4:].tolist()
                _slice_start = tensor_out[0:tensor_in.shape[0]].int().sum().item()
                _slice_mid = tensor_out[tensor_in.shape[0]:2*tensor_in.shape[0]].int().sum().item()
                _out_sum = int(tensor_out.int().sum().item())
                # max byte
                _max_byte = int(tensor_out.max().item())
                print(
                    f"[AG_DBG r0 call#{_n}] POST-KERNEL tensor_out: "
                    f"sum={_out_sum} max_byte={_max_byte} "
                    f"slice[0]={_slice_start} (expect={tensor_in.int().sum().item()}) "
                    f"slice[1]={_slice_mid} "
                    f"first={_out_first} last={_out_last}",
                    flush=True)
        return tensor_out

    def allgather_uc(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
    ) -> SymmTensor:
        """Force-UC allgather (skips multicast even when available).

        Useful for benchmarking and for verifying correctness of the UC
        path when multicast is also available.
        """
        self._check_not_blocked(tensor_in, "allgather_uc")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        tensor_out = self.create_tensor(
            torch.Size([self.world_size * tensor_in.shape[0], *tensor_in.shape[1:]]),
            tensor_in.dtype,
        )
        out_offset = tensor_out.data_ptr() - self.pool_ptr
        ubx_allgather_uc(
            self.world_size, self.myrank,
            self.dev_comm_handle, self.window_handle, self.pool_ptr,
            tensor_in.data_ptr(), out_offset,
            tensor_in.numel() * tensor_in.element_size(),
            self.default_sms, smlimit,
        )
        return tensor_out

    def barrier(self, smlimit: int = 0) -> None:
        if ubx_barrier is None:
            raise RuntimeError(
                "ubx_barrier kernel not available in this container — rebuild "
                "with the new kernel to use SymmAllocator.barrier()."
            )
        return self._barrier_impl(smlimit)

    def _barrier_impl(self, smlimit: int = 0) -> None:
        """Standalone cross-rank barrier using UBX_FLAG_BARRIER_* flags.

        No data movement — just the atomic-flag protocol identical to
        the combine/dispatch kernels' barriers, but on its own flag
        slots so the sync state doesn't interleave with data-kernel
        flag accounting. Useful for surrounding dispatch/combine kernel
        calls with a hard sync that doesn't share flag slots with the
        data kernel itself, to isolate flag-protocol races.

        Launches on the current PyTorch CUDA stream.
        """
        ubx_barrier(
            self.world_size, self.myrank,
            self._nccl_pool.dev_comm_handle, self._nccl_pool.window_handle,
            self.pool_ptr,
            self.default_sms, smlimit,
        )

    def peer_atomic_test(self, test_id: int = 1) -> None:
        """Run the minimal cross-rank peer atomic-inc diagnostic kernel.

        Each rank's thread i issues ATOMIC_UCINC on peer i's
        UBX_FLAG_DIAG_PEER_TEST slot, then thread 0 polls own flag until
        peer atomics arrive. Same raw cross-rank UC atomic pattern as
        combine's lastSM signaling, isolated from any combine logic.

        If group 0's combine hangs and THIS test also hangs in isolation,
        the bug is at the cross-rank UC atomic level (NCCL window / NVLink
        P2P), not in combine kernel logic.

        Caller is responsible for synchronizing if they want a blocking
        check; otherwise the kernel just queues on the current stream.
        """
        if ubx_peer_atomic_test is None:
            raise RuntimeError(
                "ubx_peer_atomic_test not available in this container — rebuild."
            )
        ubx_peer_atomic_test(
            self.world_size, self.myrank, test_id,
            self._nccl_pool.dev_comm_handle, self._nccl_pool.window_handle,
            self.pool_ptr,
        )

    def dump_peer_ptrs(self) -> "torch.Tensor":
        """Return a HOST tensor [world_size] uint64 with each rank's peer
        LSA pointer as observed by THIS rank's GPU.

        Calls the diagnostic `ubx_peer_ptr_dump` kernel that has each
        thread invoke `ncclGetLsaPointer(window, 0, i)` and write the
        result. Then memcpy-D2H to host. Useful for verifying NCCL's
        peer-ptr cache is populated identically (and non-zero) on all
        ranks of the EP group before the first UBX kernel fires.
        """
        if ubx_peer_ptr_dump is None:
            raise RuntimeError(
                "ubx_peer_ptr_dump not available in this container — rebuild."
            )
        dev_buf = torch.zeros(self.world_size, dtype=torch.int64,
                              device=self.device)
        ubx_peer_ptr_dump(
            self.world_size,
            self._nccl_pool.dev_comm_handle, self._nccl_pool.window_handle,
            dev_buf.data_ptr(),
        )
        torch.cuda.synchronize()
        return dev_buf.cpu()

    def alltoall(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
        nthreads: int = 0,
    ) -> SymmTensor:
        """Performs alltoall using unicast.

        Works with multicast disabled (when multicast is unavailable and mc0_ptr=0).
        For multicast-disabled operation, set NCCL_NVLS_ENABLE=0 before
        initializing the allocator to avoid MC rendezvous failures.

        smlimit sets SM count *exactly* (0 = launcher default 32). nthreads sets
        threads-per-block *exactly* (0 = launcher default 1024). Note this kernel's
        smlimit is set-exact, unlike alltoall_lamport / allreduce / allgather where
        smlimit caps from the default — so callers can sweep both up and down here.
        """
        self._check_not_blocked(tensor_in, "alltoall")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"
        tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)
        out_offset = tensor_out.data_ptr() - self.pool_ptr

        ubx_alltoall(
            self.world_size,
            self.myrank,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tensor_in.data_ptr(),
            out_offset,
            tensor_in.numel() * tensor_in.element_size() // self.world_size,
            self.default_sms,
            smlimit,
            nthreads,
        )
        return tensor_out

    def alltoall_auto(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
        nthreads: int = 0,
    ) -> SymmTensor:
        """Auto-selects best alltoall variant based on total tensor size.

        Below AUTO_SWITCH_BYTES (0.25 MB) → Lamport (barrier-free, lowest latency).
        Above → UC (barrier-based, higher bandwidth at large sizes; works with multicast disabled).
        """
        if tensor_in.numel() * tensor_in.element_size() > AUTO_SWITCH_BYTES:
            return self.alltoall(tensor_in, smlimit=smlimit, nthreads=nthreads)
        return self.alltoall_lamport(tensor_in, smlimit=smlimit, nthreads=nthreads)

    def alltoallv_prepare(
        self,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        dtype: torch.dtype = torch.bfloat16,
    ) -> dict:
        """Precompute offsets and allocate output for alltoallv.

        Call once, then use alltoallv_run() repeatedly in the hot path.

        Args:
            output_split_sizes: int32 tensor [world_size] — elements to recv from each rank.
            input_split_sizes: int32 tensor [world_size] — elements to send to each rank.
            dtype: element dtype.

        Returns:
            dict with precomputed state for alltoallv_run().
        """
        import torch.distributed as dist

        elem_size = torch.tensor(0, dtype=dtype).element_size()

        # All kernel offsets and counts are in BYTES (int64 — pool can be 24 GiB,
        # so byte offsets exceed int32 range). The kernel does a uint4 bulk
        # loop + 2 B tail per source, so arbitrary byte counts are OK.
        send_byte_counts_t = (input_split_sizes.to(torch.int64) * elem_size)
        send_byte_offsets_t = torch.zeros_like(send_byte_counts_t)
        send_byte_offsets_t[1:] = send_byte_counts_t[:-1].cumsum(0)

        # Gather all ranks' input splits to compute remote recv byte offsets
        all_input_splits = [torch.zeros_like(input_split_sizes)
                            for _ in range(self.world_size)]
        dist.all_gather(all_input_splits, input_split_sizes, group=self.dist_group)

        remote_recv_byte_offsets = torch.zeros(self.world_size, dtype=torch.int64,
                                                device=self.device)
        for d in range(self.world_size):
            offset_bytes = 0
            for src in range(self.myrank):
                offset_bytes += int(all_input_splits[src][d].item()) * elem_size
            remote_recv_byte_offsets[d] = offset_bytes

        # Output is element-packed (no per-source padding); shape = total elems.
        total_recv_elems = int(output_split_sizes.sum().item())
        tensor_out = self.create_tensor(torch.Size([total_recv_elems]), dtype)

        # Gather output pool byte offsets from all ranks.
        my_out_pool_offset_b = tensor_out.data_ptr() - self.pool_ptr
        all_out_byte_offsets = torch.zeros(self.world_size, dtype=torch.int64, device=self.device)
        my_offset_t = torch.tensor([my_out_pool_offset_b], dtype=torch.int64, device=self.device)
        dist.all_gather_into_tensor(all_out_byte_offsets, my_offset_t, group=self.dist_group)

        dest_byte_offsets = (all_out_byte_offsets + remote_recv_byte_offsets)

        return {
            "send_byte_offsets": send_byte_offsets_t,
            "send_byte_counts":  send_byte_counts_t,
            "dest_byte_offsets": dest_byte_offsets,
            "tensor_out": tensor_out,
            "total_recv_elems": total_recv_elems,
        }

    def alltoallv_run(self, tensor_in: torch.Tensor, state: dict, smlimit: int = 0):
        """Execute alltoallv using precomputed state from alltoallv_prepare().

        Args:
            tensor_in: 1D contiguous input tensor.
            state: dict returned by alltoallv_prepare().
            smlimit: Optional SM count cap (0 = default).

        Returns:
            SymmTensor with sum(output_split_sizes) elements.
        """
        from ubx._C import ubx_alltoallv

        ubx_alltoallv(
            self.world_size,
            self.myrank,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tensor_in.data_ptr(),
            state["send_byte_offsets"].data_ptr(),
            state["send_byte_counts"].data_ptr(),
            state["dest_byte_offsets"].data_ptr(),
            self.default_sms,
            smlimit,
        )
        return state["tensor_out"][:state["total_recv_elems"]]

    def alltoallv(
        self,
        tensor_in: torch.Tensor,
        output_split_sizes: torch.Tensor,
        input_split_sizes: torch.Tensor,
        smlimit: int = 0,
    ) -> SymmTensor:
        """Variable-length alltoall (convenience wrapper).

        Calls alltoallv_prepare() + alltoallv_run(). For repeated calls with
        the same split sizes, use prepare/run directly to avoid per-call overhead.
        """
        self._check_not_blocked(tensor_in, "alltoallv")
        assert tensor_in.device == self.device, "Tensor device mismatch"
        assert tensor_in.is_contiguous(), "Input must be contiguous"
        state = self.alltoallv_prepare(output_split_sizes, input_split_sizes, tensor_in.dtype)
        return self.alltoallv_run(tensor_in, state, smlimit)

    def alltoall_lamport(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
        nthreads: int = 0,
    ) -> SymmTensor:
        """Performs alltoall using unicast with Lamport polling.

        Four-buffer rolling state per (graph_mode, shape, dtype):
          - First call: allocate 4 persistent buffers, pre-fill
            buf[0..2] with the Lamport poison sentinel (UBX_LAMPORT_INT).
            buf[3] is initialised in-kernel by call N=0's poison-write.
            The first kernel runs with the UC entry barrier engaged
            (skip_barrier=False) so peers observe the pre-poison fills.
          - Subsequent calls: out_idx = N % 4, poison_idx = (N + 3) % 4,
            with no in-kernel barrier (skip_barrier=True). The
            dual-kernel split (write + poll) makes the 2-kernel gap
            safe — peer UC writes are flushed at the write-kernel
            exit before any poll begins.

        Falls back to regular alltoall if pool exhausted.
        """
        self._check_not_blocked(tensor_in, "alltoall_lamport")
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        graph_mode = torch.cuda.is_current_stream_capturing()
        key = (graph_mode, tuple(tensor_in.shape), tensor_in.dtype)
        state = self.lamport_a2a_state.get(key)

        if state is None:
            # First call for this (graph_mode, shape, dtype): allocate 4
            # persistent buffers and pre-poison buf[0..2]. buf[3] is
            # initialised in-kernel by call N=0's clear-write.
            bufs: List[SymmTensor] = []
            for _ in range(4):
                b = self.create_tensor(tensor_in.shape, tensor_in.dtype)
                if b is None:
                    for x in bufs:
                        del x
                    return self.alltoall(tensor_in, smlimit, nthreads)
                bufs.append(b)
            # Pre-poison buf[0..2]. View as int32 to write the full
            # 32-bit sentinel pattern (cudaMemset only sets bytes).
            # 0xFFFAFFFA as a signed int32 == -327686.
            poison = -327686
            for i in range(3):
                bufs[i].data.view(torch.int32).fill_(poison)
            state = {"buffers": bufs, "counter": 0}
            self.lamport_a2a_state[key] = state

        bufs = state["buffers"]
        counter = state["counter"]
        out_idx = counter % 4
        poison_idx = (counter + 3) % 4
        # Engage the UC barrier exactly once (counter==0) so peers
        # observe the three pre-poison fills before any UC writes start
        # landing. Subsequent calls rely on the 2-kernel gap between
        # the in-kernel poison-write and the corresponding poll, which
        # is safe under the dual-kernel design (write kernel exit
        # flushes UC writes before the next poll begins).
        skip_barrier = counter > 0

        tensor_out = bufs[out_idx]
        clear_buf = bufs[poison_idx]
        nbytes_per_rank = tensor_in.numel() * tensor_in.element_size() // self.world_size
        out_offset = tensor_out.data_ptr() - self.pool_ptr

        ubx_alltoall_lamport(
            self.world_size, self.myrank,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tensor_in.data_ptr(),
            out_offset,
            clear_buf.data_ptr(),
            nbytes_per_rank,
            True,  # poisoned: Python handles the upfront fills
            self.default_sms, smlimit, nthreads,
            skip_barrier,
        )

        state["counter"] = counter + 1
        return tensor_out
    def a2av_token_bf16_mxfp8(
        self,
        tokens_bf16: torch.Tensor,
        token_offsets: torch.Tensor,
        experts_per_rank: int,
        output: SymmTensor,
        smlimit: int = 0,
        sync: bool = True,
        expert_start: int = 0,
        expert_count: int = 0,
    ) -> SymmTensor:
        """Dispatch bf16 tokens to remote ranks with mxfp8 quantization.

        Reads tokens_bf16 (any bfloat16 tensor on this rank) and writes
        fp8-quantized data + E8M0 scale bytes into every destination rank's
        symmetric pool via NVLink.

        The caller is responsible for allocating output via
        allocator.create_tensor(..., blocked='mxfp8') before calling this
        function.  A typical sizing:
            max_slots_per_expert = ceil(ntokens / (total_experts * 0.8))
            total_slots = experts_per_rank * max_slots_per_expert
            output = allocator.create_tensor(
                [total_slots, hidden], torch.float8_e4m3fn, blocked='mxfp8')

        Output layout
        -------------
        Shape [experts_per_rank * max_slots_per_expert, hidden], blocked='mxfp8'.
        For experts_per_rank > 1, slot indices in token_offsets must be unique
        per rank — assign expert e_local = e % experts_per_rank the sub-range
        [e_local * max_slots_per_expert, (e_local+1) * max_slots_per_expert).

        Args:
            tokens_bf16: [ntokens, hidden] bfloat16.  hidden must be a multiple
                of 32 (one mxfp8 block = 32 bf16 elements).
            token_offsets: [ntokens, total_experts] int32 on the same CUDA
                device.  Entry [t, e] is the destination slot index (≥ 0) for
                token t at expert e, or -1 if not routed there.
                total_experts must equal world_size * experts_per_rank.
            experts_per_rank: Number of experts hosted on each rank.
            output: Pre-allocated mxfp8 SymmTensor to receive tokens into.
                Must have blocked_format='mxfp8', dtype float8_e4m3fn, and
                shape [total_slots, hidden].
            smlimit: Optional SM count cap (0 = use default of 32 SMs).
            sync: If True (default), kernel polls until all ranks complete.
                If False, kernel returns after signalling the barrier flag.
                Caller must call a2av_wait() before reading output.
            expert_start: First expert index (global, 0-based) to dispatch.
                Default 0.
            expert_count: Number of experts to dispatch. 0 (default) means
                all experts. Use with expert_start to pipeline dispatch
                across multiple kernel launches.

        Returns:
            output (the same tensor passed in).
            Received fp8 data is at output.data_ptr(); E8M0 scale bytes are at
            output.metadata_ptr.
        """
        assert output is not None, \
            "output must be a pre-allocated mxfp8 SymmTensor"
        assert isinstance(output, SymmTensor) and output.blocked_format == "mxfp8", \
            "output must be a SymmTensor with blocked_format='mxfp8'"
        assert tokens_bf16.dtype == torch.bfloat16, "tokens_bf16 must be bfloat16"
        assert token_offsets.dtype == torch.int32,   "token_offsets must be int32"
        assert tokens_bf16.device == self.device,    "tokens_bf16 device mismatch"
        assert token_offsets.device == self.device,  "token_offsets device mismatch"
        assert output.device == self.device,         "output device mismatch"

        ntokens          = tokens_bf16.shape[0]
        hidden           = tokens_bf16.numel() // ntokens
        total_experts    = self.world_size * experts_per_rank
        blocks_per_token = hidden // 32

        assert hidden % 32 == 0, f"hidden ({hidden}) must be a multiple of 32"
        assert output.shape[1] == hidden, \
            f"output.shape[1]={output.shape[1]} != hidden={hidden}"
        assert token_offsets.shape == (ntokens, total_experts), (
            f"token_offsets must be [{ntokens}, {total_experts}], "
            f"got {list(token_offsets.shape)}"
        )

        lineoffset_out    = (output.data_ptr()    - self.pool_ptr) // 16
        lineoffset_scales = (output.metadata_ptr  - self.pool_ptr) // 16

        ubx_a2av_token_bf16_mxfp8(
            self.world_size,
            self.myrank,
            ntokens,
            blocks_per_token,
            experts_per_rank,
            token_offsets.data_ptr(),
            lineoffset_out,
            lineoffset_scales,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tokens_bf16.data_ptr(),
            self.default_sms,
            smlimit,
            1 if sync else 0,
            expert_start,
            expert_count,
        )
        return output

    def a2av_token_bf16_bf16(
        self,
        tokens_bf16: torch.Tensor,
        token_offsets: torch.Tensor,
        experts_per_rank: int,
        output: SymmTensor,
        smlimit: int = 0,
        sync: bool = True,
        expert_start: int = 0,
        expert_count: int = 0,
    ) -> SymmTensor:
        """Dispatch bf16 tokens to remote ranks WITHOUT quantization.

        Same routing/slot scheme as a2av_token_bf16_mxfp8 but writes the raw
        bf16 block to the destination instead of fp8 + E8M0 scale.

        Output layout
        -------------
        Shape [experts_per_rank * max_slots_per_expert, hidden], dtype bf16,
        plain (un-blocked) SymmTensor. For experts_per_rank > 1, slot indices
        in token_offsets must be unique per rank — assign expert
        e_local = e % experts_per_rank the sub-range
        [e_local * max_slots_per_expert, (e_local+1) * max_slots_per_expert).

        Args:
            tokens_bf16: [ntokens, hidden] bfloat16. hidden must be a multiple
                of 32 (one block = 32 bf16 elements).
            token_offsets: [ntokens, total_experts] int32 on the same CUDA
                device. Entry [t, e] is the destination slot index (>= 0) for
                token t at expert e, or -1 if not routed there.
            experts_per_rank: Number of experts hosted on each rank.
            output: Pre-allocated bf16 SymmTensor (NOT mxfp8-blocked) with
                shape [total_slots, hidden].
            smlimit, sync, expert_start, expert_count: see a2av_token_bf16_mxfp8.

        Returns:
            output (the same tensor passed in).
        """
        assert output is not None, \
            "output must be a pre-allocated bf16 SymmTensor"
        assert isinstance(output, SymmTensor), \
            "output must be a SymmTensor"
        assert output.blocked_format is None, \
            "output must NOT be a blocked SymmTensor (no scale region for bf16)"
        assert output.dtype == torch.bfloat16, \
            "output dtype must be bfloat16"
        assert tokens_bf16.dtype == torch.bfloat16, "tokens_bf16 must be bfloat16"
        assert token_offsets.dtype == torch.int32,  "token_offsets must be int32"
        assert tokens_bf16.device == self.device,   "tokens_bf16 device mismatch"
        assert token_offsets.device == self.device, "token_offsets device mismatch"
        assert output.device == self.device,        "output device mismatch"

        ntokens          = tokens_bf16.shape[0]
        hidden           = tokens_bf16.numel() // ntokens
        total_experts    = self.world_size * experts_per_rank
        blocks_per_token = hidden // 32

        assert hidden % 32 == 0, f"hidden ({hidden}) must be a multiple of 32"
        assert output.shape[1] == hidden, \
            f"output.shape[1]={output.shape[1]} != hidden={hidden}"
        assert token_offsets.shape == (ntokens, total_experts), (
            f"token_offsets must be [{ntokens}, {total_experts}], "
            f"got {list(token_offsets.shape)}"
        )

        lineoffset_out = (output.data_ptr() - self.pool_ptr) // 16

        ubx_a2av_token_bf16_bf16(
            self.world_size,
            self.myrank,
            ntokens,
            blocks_per_token,
            experts_per_rank,
            token_offsets.data_ptr(),
            lineoffset_out,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tokens_bf16.data_ptr(),
            self.default_sms,
            smlimit,
            1 if sync else 0,
            expert_start,
            expert_count,
        )
        return output

    def a2av_token_bf16_bf16_topk(
        self,
        tokens_bf16: torch.Tensor,
        topk_expert: torch.Tensor,
        topk_slot: torch.Tensor,
        experts_per_rank: int,
        output: SymmTensor,
        smlimit: int = 0,
        sync: bool = True,
    ) -> SymmTensor:
        """Top-K LUT variant of bf16->bf16 token dispatch.

        Same output contract as a2av_token_bf16_bf16. Inputs replace
        ``token_offsets[ntokens, total_experts]`` with two compact LUTs
        ``[ntokens, topk_max]`` (see ubx.ops.compute_dispatch_topk_map):

        - ``topk_expert[t, k]``: global expert id of t's k-th routed expert
          (sorted ascending), or -1 if unused.
        - ``topk_slot[t, k]``: destination slot index, same -1 mask.

        The kernel's inner loop runs ``topk_max`` iterations per token
        instead of ``total_experts``. For top-K=6 / total_experts=128
        that's a 21x reduction in inner-loop work.
        """
        assert output is not None and isinstance(output, SymmTensor)
        assert output.blocked_format is None
        assert output.dtype == torch.bfloat16
        assert tokens_bf16.dtype == torch.bfloat16
        assert topk_expert.dtype == torch.int32 and topk_slot.dtype == torch.int32
        assert tokens_bf16.device == self.device
        assert topk_expert.device == self.device and topk_slot.device == self.device
        assert output.device == self.device
        assert topk_expert.shape == topk_slot.shape

        ntokens          = tokens_bf16.shape[0]
        hidden           = tokens_bf16.numel() // ntokens
        blocks_per_token = hidden // 32
        topk_max         = topk_expert.shape[1]

        assert hidden % 32 == 0
        assert output.shape[1] == hidden
        assert topk_expert.shape[0] == ntokens, (
            f"topk_expert.shape[0]={topk_expert.shape[0]} != ntokens={ntokens}"
        )

        lineoffset_out = (output.data_ptr() - self.pool_ptr) // 16

        ubx_a2av_token_bf16_bf16_topk(
            self.world_size,
            self.myrank,
            ntokens,
            blocks_per_token,
            experts_per_rank,
            topk_max,
            topk_expert.data_ptr(),
            topk_slot.data_ptr(),
            lineoffset_out,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tokens_bf16.data_ptr(),
            self.default_sms,
            smlimit,
            1 if sync else 0,
        )
        return output

    def a2av_token_bf16_mxfp8_persistent(
        self,
        tokens_bf16: torch.Tensor,
        token_offsets: torch.Tensor,
        experts_per_rank: int,
        output: SymmTensor,
        nexperts_per_chunk: int,
        smlimit: int = 0,
    ) -> SymmTensor:
        """Persistent (chunked) token-dispatch bf16 → mxfp8.

        One kernel launch processes ``nchunks = ceil(experts_per_rank /
        nexperts_per_chunk)`` chunks internally, each with its own cross-rank
        barrier. Chunk c covers local experts
        ``[c*nexperts_per_chunk, min((c+1)*nexperts_per_chunk, experts_per_rank))``
        on every rank. Per-chunk barriers match the same A2AV_ID/BAR protocol
        the non-persistent kernel uses, so the caller's compute stream calls
        ``a2av_wait()`` once per chunk — unchanged API.

        Use this instead of calling :meth:`a2av_token_bf16_mxfp8` N times when
        pipelining dispatch with per-expert GEMMs: avoids N-1 kernel launches
        and the kernel-to-kernel ramp-up gap on the dispatch stream.

        Args:
            tokens_bf16: [ntokens, hidden] bfloat16.
            token_offsets: [ntokens, total_experts] int32.
            experts_per_rank: Number of experts hosted on each rank.
            output: Pre-allocated mxfp8 SymmTensor, as for a2av_token_bf16_mxfp8.
            nexperts_per_chunk: Local experts per chunk. Last chunk is smaller
                if ``experts_per_rank`` is not divisible.
            smlimit: Optional SM count cap (0 = default of 32 SMs).

        Returns:
            output (same tensor passed in).
        """
        assert output is not None and isinstance(output, SymmTensor) \
            and output.blocked_format == "mxfp8", \
            "output must be a pre-allocated mxfp8 SymmTensor"
        assert tokens_bf16.dtype == torch.bfloat16, "tokens_bf16 must be bfloat16"
        assert token_offsets.dtype == torch.int32,   "token_offsets must be int32"
        assert tokens_bf16.device == self.device,    "tokens_bf16 device mismatch"
        assert token_offsets.device == self.device,  "token_offsets device mismatch"
        assert output.device == self.device,         "output device mismatch"
        assert nexperts_per_chunk >= 1, "nexperts_per_chunk must be >= 1"

        nchunks = (experts_per_rank + nexperts_per_chunk - 1) // nexperts_per_chunk
        assert nchunks <= 32, (
            f"nchunks={nchunks} exceeds persistent kernel limit of 32. "
            f"Raise nexperts_per_chunk or reduce experts_per_rank."
        )

        ntokens          = tokens_bf16.shape[0]
        hidden           = tokens_bf16.numel() // ntokens
        total_experts    = self.world_size * experts_per_rank
        blocks_per_token = hidden // 32

        assert hidden % 32 == 0, f"hidden ({hidden}) must be a multiple of 32"
        assert output.shape[1] == hidden, \
            f"output.shape[1]={output.shape[1]} != hidden={hidden}"
        assert token_offsets.shape == (ntokens, total_experts), (
            f"token_offsets must be [{ntokens}, {total_experts}], "
            f"got {list(token_offsets.shape)}"
        )

        lineoffset_out    = (output.data_ptr()    - self.pool_ptr) // 16
        lineoffset_scales = (output.metadata_ptr  - self.pool_ptr) // 16

        ubx_a2av_token_bf16_mxfp8_persistent(
            self.world_size,
            self.myrank,
            ntokens,
            blocks_per_token,
            experts_per_rank,
            token_offsets.data_ptr(),
            lineoffset_out,
            lineoffset_scales,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            tokens_bf16.data_ptr(),
            self.default_sms,
            smlimit,
            nchunks,
            nexperts_per_chunk,
        )
        return output

    def a2av_wait(self) -> None:
        """Wait for async a2av dispatch to complete on all ranks.

        Must be called after a2av_token_bf16_mxfp8(..., sync=False) and
        before reading the output tensor.
        """
        from ubx._C import ubx_a2av_wait
        ubx_a2av_wait(
            self.world_size,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
        )

    # =========================================================================
    # MoE token-combine (reverse of a2av_token dispatch).
    # =========================================================================

    def _check_combine_inputs(self, expert_outputs, token_offsets, gate_weights,
                              experts_per_rank, max_tokens_per_rank):
        """Shared pre-flight for combine_bf16_bf16 / combine_mxfp8_bf16."""
        assert expert_outputs.dtype == torch.bfloat16, \
            "expert_outputs must be bfloat16"
        assert expert_outputs.is_cuda and expert_outputs.device == self.device, \
            "expert_outputs device mismatch"
        assert token_offsets.dtype == torch.int32, "token_offsets must be int32"
        assert token_offsets.device == self.device, "token_offsets device mismatch"

        assert expert_outputs.dim() == 2, "expert_outputs must be 2D [n_recv, hidden]"
        n_recv, hidden = expert_outputs.shape
        assert hidden % 32 == 0, f"hidden ({hidden}) must be a multiple of 32"
        assert n_recv <= max_tokens_per_rank, \
            f"n_recv ({n_recv}) exceeds max_tokens_per_rank ({max_tokens_per_rank})"

        local_ntokens, total_experts = token_offsets.shape
        assert total_experts == self.world_size * experts_per_rank, (
            f"token_offsets.shape[1]={total_experts} != "
            f"world_size*experts_per_rank={self.world_size * experts_per_rank}"
        )

        if gate_weights is not None:
            assert gate_weights.dtype == torch.float32, \
                "gate_weights must be float32"
            assert gate_weights.shape == token_offsets.shape, \
                "gate_weights shape must match token_offsets"
            assert gate_weights.device == self.device, \
                "gate_weights device mismatch"

        return n_recv, hidden, local_ntokens, total_experts

    def combine_bf16_bf16(
        self,
        expert_outputs: torch.Tensor,
        token_offsets: torch.Tensor,
        experts_per_rank: int,
        max_tokens_per_rank: int,
        gate_weights: Optional[torch.Tensor] = None,
        smlimit: int = 0,
        sync: bool = True,
        temp: Optional[SymmTensor] = None,
    ) -> torch.Tensor:
        """Combine MoE expert outputs back to originating tokens (bf16 wire).

        Reverse of a2av_token dispatch. Each rank's expert_outputs (bf16) are
        staged into a temp symm buffer (Phase 1), peers' contributions are
        pulled (Phase 2), gate-weighted, summed in fp32, and downcast to bf16.

        Args:
            expert_outputs: [n_recv, hidden] bf16 LOCAL torch.Tensor with the
                FFN expert outputs, in the same slot order they were produced
                by dispatch (row i ↔ slot i).
            token_offsets: [local_ntokens, total_experts] int32 (precomputed
                via compute_token_offsets) — same matrix passed to dispatch.
            experts_per_rank: Number of experts hosted on each rank.
            max_tokens_per_rank: From compute_token_offsets() — the symmetric
                temp buffer first dim.
            gate_weights: Optional [local_ntokens, total_experts] float32 with
                router gate values. None → unweighted sum.
            smlimit: Cap on SMs used (0 = use launcher default of 64).
            sync: True (default) → kernel does Phase 1 + barrier + Phase 2;
                output is ready on return. False → kernel does ONLY Phase 1
                (signals barrier and returns); caller MUST call combine_wait()
                afterwards to run Phase 2 (which produces the actual output).
                Use False to overlap host or device work between phases.

        Returns:
            [local_ntokens, hidden] bf16 torch.Tensor (allocated by torch).
            When sync=False the tensor is not yet populated — read it only
            after combine_wait() returns.
        """
        n_recv, hidden, local_ntokens, _ = self._check_combine_inputs(
            expert_outputs, token_offsets, gate_weights,
            experts_per_rank, max_tokens_per_rank)
        blocks_per_token = hidden // 32

        # The kernel's `lineoffset_temp` is a single scalar — it ASSUMES every
        # rank has its temp buffer at the same pool offset. When asymmetric
        # allocations (e.g. an alltoallv with rank-dependent split sums) have
        # consumed pool space between dispatch and combine, the caller must
        # supply a pre-allocated symmetric temp via this argument. Otherwise
        # we allocate here and trust the caller to keep allocations symmetric.
        if temp is None:
            temp = self.create_tensor(
                (max_tokens_per_rank, hidden), torch.bfloat16)
            if temp is None:
                raise RuntimeError(
                    f"combine_bf16_bf16: failed to allocate temp symm buffer "
                    f"({max_tokens_per_rank} x {hidden} bf16)")
        else:
            assert isinstance(temp, SymmTensor), \
                "temp must be a SymmTensor from this allocator"
            assert temp.shape == (max_tokens_per_rank, hidden), (
                f"temp shape {tuple(temp.shape)} != "
                f"({max_tokens_per_rank}, {hidden})"
            )
            assert temp.dtype == torch.bfloat16, \
                f"temp dtype {temp.dtype} != torch.bfloat16"

        out = torch.empty((local_ntokens, hidden),
                          dtype=torch.bfloat16, device=self.device)

        lineoffset_temp = (temp.data_ptr() - self.pool_ptr) // 16
        gate_ptr = gate_weights.data_ptr() if gate_weights is not None else 0

        ubx_combine_bf16_bf16(
            self.world_size, self.myrank,
            local_ntokens, n_recv, blocks_per_token,
            experts_per_rank, max_tokens_per_rank,
            token_offsets.data_ptr(), gate_ptr,
            lineoffset_temp,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            expert_outputs.data_ptr(), out.data_ptr(),
            self.default_sms, smlimit, 1 if sync else 0,
        )
        if not sync:
            # Stash everything Phase 2 needs. combine_wait() will replay it
            # against ubx_combine_wait_bf16 (poll barrier + run Phase 2).
            # The temp SymmTensor reference keeps the buffer alive — peers
            # need to read it during Phase 2.
            self._pending_combine = {
                "wire": "bf16",
                "temp": temp,                       # holds symm buffer alive
                "out": out,
                "token_offsets": token_offsets,
                "gate_weights": gate_weights,
                "experts_per_rank": experts_per_rank,
                "local_ntokens": local_ntokens,
                "blocks_per_token": blocks_per_token,
                "lineoffset_temp": lineoffset_temp,
                "smlimit": smlimit,
            }
        # temp goes out of scope here in the sync path (SymmTensor.__del__
        # returns its slot). Stream-order ensures the kernel finishes
        # reading the buffer before the dealloc takes effect.
        return out

    def combine_v2_bf16_bf16(
        self,
        expert_outputs: torch.Tensor,
        token_offsets: torch.Tensor,
        experts_per_rank: int,
        max_tokens_per_rank: int,
        gate_weights: Optional[torch.Tensor] = None,
        smlimit: int = 0,
        temp: Optional[SymmTensor] = None,
    ) -> torch.Tensor:
        """E22: combine via TWO separate kernels.

        Phase 1 = pure data copy (no atomics, no barrier). Kernel exit +
        CUDA stream order is the only synchronization. Avoids the v1 SM-hold
        deadlock where all CTAs spin together on BAR while waiting for an
        intra-kernel atomicAdd sum.

        Phase 2 = PDL grid-dep sync → CTA 0 issues RANKS peer ATOMIC_UCINCs
        to UBX_FLAG_COMBINE2_BAR → all CTAs poll until BAR >= reduce_id*RANKS
        → combine_phase2_bf16 (peer pull + fp32 sum).

        `reduce_id` is host-tracked (`self._combine_v2_call_counter`) and
        passed as a kernel arg.
        """
        if ubx_combine_v2_phase1_bf16 is None or ubx_combine_v2_phase2_bf16 is None:
            raise RuntimeError(
                "combine_v2_bf16_bf16: container does not include the v2 "
                "kernels — rebuild ub-x at HEAD that has UBX_FLAG_COMBINE2_BAR."
            )
        n_recv, hidden, local_ntokens, _ = self._check_combine_inputs(
            expert_outputs, token_offsets, gate_weights,
            experts_per_rank, max_tokens_per_rank)
        blocks_per_token = hidden // 32

        if temp is None:
            temp = self.create_tensor(
                (max_tokens_per_rank, hidden), torch.bfloat16)
            if temp is None:
                raise RuntimeError(
                    f"combine_v2_bf16_bf16: failed to allocate temp symm "
                    f"buffer ({max_tokens_per_rank} x {hidden} bf16)"
                )
        else:
            assert isinstance(temp, SymmTensor)
            assert temp.shape == (max_tokens_per_rank, hidden)
            assert temp.dtype == torch.bfloat16

        out = torch.empty(
            (local_ntokens, hidden), dtype=torch.bfloat16, device=self.device)
        lineoffset_temp = (temp.data_ptr() - self.pool_ptr) // 16
        gate_ptr = gate_weights.data_ptr() if gate_weights is not None else 0

        # Host-tracked per-call counter (per-rank, monotonic).
        if not hasattr(self, "_combine_v2_call_counter"):
            self._combine_v2_call_counter = 0
        self._combine_v2_call_counter += 1
        reduce_id = self._combine_v2_call_counter

        ubx_combine_v2_phase1_bf16(
            n_recv, blocks_per_token,
            self.pool_ptr, lineoffset_temp,
            expert_outputs.data_ptr(),
            self.default_sms, smlimit,
        )
        ubx_combine_v2_phase2_bf16(
            self.world_size, reduce_id, local_ntokens, blocks_per_token,
            experts_per_rank,
            token_offsets.data_ptr(), gate_ptr,
            lineoffset_temp,
            self.dev_comm_handle, self.window_handle, self.pool_ptr,
            out.data_ptr(),
            self.default_sms, smlimit,
        )
        return out

    def combine_mxfp8_bf16(
        self,
        expert_outputs: torch.Tensor,
        token_offsets: torch.Tensor,
        experts_per_rank: int,
        max_tokens_per_rank: int,
        gate_weights: Optional[torch.Tensor] = None,
        smlimit: int = 0,
        sync: bool = True,
    ) -> torch.Tensor:
        """Combine MoE expert outputs (mxfp8 over the wire, bf16 accum/output).

        Same shape as combine_bf16_bf16 but quantizes each rank's expert
        outputs to mxfp8 (E8M0 microscale per 32-elem block) before the
        cross-rank exchange, halving NVLink payload. See combine_bf16_bf16
        for the sync vs async semantics.
        """
        n_recv, hidden, local_ntokens, _ = self._check_combine_inputs(
            expert_outputs, token_offsets, gate_weights,
            experts_per_rank, max_tokens_per_rank)
        blocks_per_token = hidden // 32

        temp = self.create_tensor(
            (max_tokens_per_rank, hidden),
            torch.float8_e4m3fn, blocked='mxfp8')
        if temp is None:
            raise RuntimeError(
                f"combine_mxfp8_bf16: failed to allocate temp mxfp8 symm buffer "
                f"({max_tokens_per_rank} x {hidden})")

        out = torch.empty((local_ntokens, hidden),
                          dtype=torch.bfloat16, device=self.device)

        lineoffset_temp   = (temp.data_ptr()      - self.pool_ptr) // 16
        lineoffset_scales = (temp.metadata_ptr    - self.pool_ptr) // 16
        gate_ptr = gate_weights.data_ptr() if gate_weights is not None else 0

        ubx_combine_mxfp8_bf16(
            self.world_size, self.myrank,
            local_ntokens, n_recv, blocks_per_token,
            experts_per_rank, max_tokens_per_rank,
            token_offsets.data_ptr(), gate_ptr,
            lineoffset_temp, lineoffset_scales,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            expert_outputs.data_ptr(), out.data_ptr(),
            self.default_sms, smlimit, 1 if sync else 0,
        )
        if not sync:
            self._pending_combine = {
                "wire": "mxfp8",
                "temp": temp,
                "out": out,
                "token_offsets": token_offsets,
                "gate_weights": gate_weights,
                "experts_per_rank": experts_per_rank,
                "local_ntokens": local_ntokens,
                "blocks_per_token": blocks_per_token,
                "lineoffset_temp": lineoffset_temp,
                "lineoffset_scales": lineoffset_scales,
                "smlimit": smlimit,
            }
        return out

    def combine_bf16_bf16_lamport_push(
        self,
        expert_outputs: torch.Tensor,
        inverse_map: torch.Tensor,
        topk_idx: torch.Tensor,
        experts_per_rank: int,
        max_tokens_per_rank: int,
        gate_weights: Optional[torch.Tensor] = None,
        smlimit: int = 0,
    ) -> torch.Tensor:
        """PUSH-semantics Lamport combine — race-free variant.

        Mirrors alltoall_lamport's safe pattern: each rank PUSHES its
        expert outputs to peer destination bufs at (origin_token, k_idx)
        coordinates determined by `inverse_map`; readers poll OWN buf
        (no cross-rank race with the triple-buffer re-poison protocol).

        Takes inverse_map and topk_idx (computed via
        ubx.compute_combine_push_map) instead of token_offsets; uses dest
        bufs sized [local_ntokens, topk_max, hidden] bf16. Race-free with
        triple-buffer re-poison since each rank only re-poisons its OWN buf
        (mirror of alltoall_lamport's safe pattern).

        Args:
            expert_outputs: [n_recv, hidden] bf16 LOCAL torch.Tensor — the
                FFN outputs at slot indices that compute_token_offsets
                placed them in (scattered per-local-expert ranges).
            inverse_map: [max_tokens_per_rank, 4] int32 from
                compute_combine_push_map. inverse_map[s] =
                (origin_rank, origin_local_token, k_idx, valid).
            topk_idx: [local_ntokens, topk_max] int32 from
                compute_combine_push_map. Per local token, the global
                expert ids of routed experts (sort order).
            experts_per_rank, max_tokens_per_rank: as for the PULL variant.
            gate_weights: optional [local_ntokens, total_experts] float32
                (full sparse layout); kernel looks up gate_weights[t,
                topk_idx[t,k]] for the k-th contribution.
            smlimit: SM cap (0 = launcher default 128).

        Returns:
            [local_ntokens, hidden] bf16 torch.Tensor.
        """
        assert expert_outputs.dtype == torch.bfloat16, "expert_outputs must be bf16"
        assert expert_outputs.is_cuda and expert_outputs.device == self.device
        assert inverse_map.dtype == torch.int32 and inverse_map.is_cuda
        assert inverse_map.device == self.device
        assert inverse_map.shape == (max_tokens_per_rank, 4), (
            f"inverse_map.shape must be [max_tokens_per_rank, 4]={(max_tokens_per_rank, 4)}, "
            f"got {tuple(inverse_map.shape)}"
        )
        assert topk_idx.dtype == torch.int32 and topk_idx.is_cuda
        assert topk_idx.device == self.device
        local_ntokens, topk_max = topk_idx.shape

        n_recv, hidden = expert_outputs.shape
        assert hidden % 32 == 0, f"hidden ({hidden}) must be a multiple of 32"
        assert n_recv <= max_tokens_per_rank
        blocks_per_token = hidden // 32

        if gate_weights is not None:
            assert gate_weights.dtype == torch.float32
            assert gate_weights.shape == (local_ntokens,
                                          self.world_size * experts_per_rank)
            assert gate_weights.device == self.device

        graph_mode = torch.cuda.is_current_stream_capturing()
        bufs = self.combine_push_triple_buf[graph_mode]
        call_n = self.combine_push_call_count[graph_mode]
        out_idx = call_n % 3
        clear_idx = (call_n + 2) % 3

        # Shape change → reset (different routing topology may yield a
        # different topk_max / local_ntokens combination).
        expected_shape = (local_ntokens, topk_max, hidden)
        if (bufs[out_idx] is not None
                and tuple(bufs[out_idx].shape) != expected_shape):
            for i in range(3):
                if bufs[i] is not None:
                    del bufs[i]
                    bufs[i] = None
            self.combine_push_call_count[graph_mode] = 0
            return self.combine_bf16_bf16_lamport_push(
                expert_outputs, inverse_map, topk_idx,
                experts_per_rank, max_tokens_per_rank,
                gate_weights, smlimit)

        for i in range(3):
            if bufs[i] is None:
                bufs[i] = self.create_tensor(expected_shape, torch.bfloat16)
                if bufs[i] is None:
                    raise RuntimeError(
                        f"combine_bf16_bf16_lamport_push: failed to allocate "
                        f"persistent dest buf {i} of shape {expected_shape}")

        dest_now   = bufs[out_idx]
        dest_clear = bufs[clear_idx]

        out = torch.empty((local_ntokens, hidden),
                          dtype=torch.bfloat16, device=self.device)

        lineoffset_dest = (dest_now.data_ptr() - self.pool_ptr) // 16
        gate_ptr = gate_weights.data_ptr() if gate_weights is not None else 0
        clear_ptr = dest_clear.data_ptr()

        poisoned = call_n >= 2
        skip_warmup_barrier = call_n >= 2

        ubx_combine_bf16_bf16_lamport_push(
            self.world_size, self.myrank,
            local_ntokens, n_recv, blocks_per_token,
            experts_per_rank, max_tokens_per_rank, topk_max,
            inverse_map.data_ptr(), topk_idx.data_ptr(), gate_ptr,
            lineoffset_dest, clear_ptr,
            poisoned, skip_warmup_barrier,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            expert_outputs.data_ptr(), out.data_ptr(),
            self.default_sms, smlimit,
        )

        self.combine_push_call_count[graph_mode] = call_n + 1
        return out

    def combine_bf16_bf16_push(
        self,
        expert_outputs: torch.Tensor,
        inverse_map: torch.Tensor,
        topk_idx: torch.Tensor,
        experts_per_rank: int,
        max_tokens_per_rank: int,
        gate_weights: Optional[torch.Tensor] = None,
        smlimit: int = 0,
    ) -> torch.Tensor:
        """PUSH-semantics combine, barrier-based (NOT Lamport).

        Same routing as combine_bf16_bf16_lamport_push but Phase 2 reads
        OWN dest after a single cross-rank ATOMIC_MCINC barrier (no per-
        element Lamport poll). Wins at large message sizes where the
        polling overhead exceeds the barrier latency. Uses double-
        buffering (2 persistent dest bufs alternating, call_n%2) to keep
        peer's next-call Phase 1 push from racing with my current-call
        Phase 2 read.
        """
        assert expert_outputs.dtype == torch.bfloat16
        assert expert_outputs.is_cuda and expert_outputs.device == self.device
        assert inverse_map.dtype == torch.int32 and inverse_map.is_cuda
        assert inverse_map.shape == (max_tokens_per_rank, 4)
        assert topk_idx.dtype == torch.int32 and topk_idx.is_cuda
        local_ntokens, topk_max = topk_idx.shape

        n_recv, hidden = expert_outputs.shape
        assert hidden % 32 == 0
        assert n_recv <= max_tokens_per_rank
        blocks_per_token = hidden // 32

        if gate_weights is not None:
            assert gate_weights.dtype == torch.float32
            assert gate_weights.shape == (local_ntokens,
                                          self.world_size * experts_per_rank)

        graph_mode = torch.cuda.is_current_stream_capturing()
        bufs = self.combine_push_nl_double_buf[graph_mode]
        call_n = self.combine_push_nl_call_count[graph_mode]
        out_idx = call_n % 2

        expected_shape = (local_ntokens, topk_max, hidden)
        if (bufs[out_idx] is not None
                and tuple(bufs[out_idx].shape) != expected_shape):
            for i in range(2):
                if bufs[i] is not None:
                    del bufs[i]
                    bufs[i] = None
            self.combine_push_nl_call_count[graph_mode] = 0
            return self.combine_bf16_bf16_push(
                expert_outputs, inverse_map, topk_idx,
                experts_per_rank, max_tokens_per_rank,
                gate_weights, smlimit)

        for i in range(2):
            if bufs[i] is None:
                bufs[i] = self.create_tensor(expected_shape, torch.bfloat16)
                if bufs[i] is None:
                    raise RuntimeError(
                        f"combine_bf16_bf16_push: failed to allocate "
                        f"persistent dest buf {i} of shape {expected_shape}")

        dest_now = bufs[out_idx]
        out = torch.empty((local_ntokens, hidden),
                          dtype=torch.bfloat16, device=self.device)
        lineoffset_dest = (dest_now.data_ptr() - self.pool_ptr) // 16
        gate_ptr = gate_weights.data_ptr() if gate_weights is not None else 0

        ubx_combine_bf16_bf16_push(
            self.world_size, self.myrank,
            local_ntokens, n_recv, blocks_per_token,
            experts_per_rank, max_tokens_per_rank, topk_max,
            inverse_map.data_ptr(), topk_idx.data_ptr(), gate_ptr,
            lineoffset_dest,
            self.dev_comm_handle,
            self.window_handle,
            self.pool_ptr,
            expert_outputs.data_ptr(), out.data_ptr(),
            self.default_sms, smlimit,
        )

        self.combine_push_nl_call_count[graph_mode] = call_n + 1
        return out

    def combine_push3_bf16_bf16(
        self,
        expert_outputs: torch.Tensor,
        inverse_map: torch.Tensor,
        topk_idx: torch.Tensor,
        experts_per_rank: int,
        max_tokens_per_rank: int,
        gate_weights: Optional[torch.Tensor] = None,
        smlimit: int = 0,
    ) -> torch.Tensor:
        """push3: 3-kernel PUSH combine. Same routing as
        combine_bf16_bf16_push but split into three independent kernels
        chained on the same stream:

          Kernel 1 (phase1_write): many-CTA cross-rank NVLink writes
              of expert outputs into peer ranks' dest bufs at
              (origin_token, k_idx). No barrier, no atomic, just data.
          Kernel 2 (phase2_signal): 1-CTA RANKS-thread cross-rank UCINC
              into UBX_FLAG_PUSH3_BAR, then spin-on-own-flag with
              compile-in timeout. This is the ONLY kernel that can hang.
          Kernel 3 (phase3_sum): many-CTA purely local read of OWN
              dest buf, sum across topk slots, write output. Zero NVLink.

        reduce_id is host-tracked (self._push3_call_counter).
        Output `out` is a fresh torch.empty in PyTorch's caching allocator.
        """
        if (ubx_combine_push3_phase1_write is None
                or ubx_combine_push3_phase2_signal is None
                or ubx_combine_push3_phase3_sum is None):
            raise RuntimeError(
                "combine_push3_bf16_bf16: push3 kernels missing in this "
                "container — rebuild ub-x."
            )
        assert expert_outputs.dtype == torch.bfloat16
        assert expert_outputs.is_cuda and expert_outputs.device == self.device
        assert inverse_map.dtype == torch.int32 and inverse_map.is_cuda
        assert inverse_map.shape == (max_tokens_per_rank, 4)
        assert topk_idx.dtype == torch.int32 and topk_idx.is_cuda
        local_ntokens, topk_max = topk_idx.shape

        n_recv, hidden = expert_outputs.shape
        assert hidden % 32 == 0
        assert n_recv <= max_tokens_per_rank
        blocks_per_token = hidden // 32
        total_experts = self.world_size * experts_per_rank

        if gate_weights is not None:
            assert gate_weights.dtype == torch.float32
            assert gate_weights.shape == (local_ntokens, total_experts)

        # Double-buffered dest bufs: call N writes/reads bufs[N%2], call N+1
        # uses bufs[(N+1)%2]. This prevents a peer's next-call Phase 1 PUSH
        # from racing with this rank's current-call Phase 3 read (the
        # cross-rank barrier in Phase 2 only orders Phase 1 → Phase 2; it
        # does NOT order Phase 3 of call N vs Phase 1 of call N+1 across
        # ranks). Same double-buf pattern as combine_bf16_bf16_push.
        graph_mode = torch.cuda.is_current_stream_capturing()
        if not hasattr(self, "_push3_dest_bufs"):
            self._push3_dest_bufs = {True: [None, None], False: [None, None]}
            self._push3_call_counter = {True: 0, False: 0}
        expected_shape = (local_ntokens, topk_max, hidden)
        bufs = self._push3_dest_bufs[graph_mode]
        if (bufs[0] is not None
                and tuple(bufs[0].shape) != expected_shape):
            # Shape changed — drop both refs (lets SymmTensor.__del__ return
            # slots to the pool) and reset counter. DON'T `del bufs[i]` —
            # that removes from the list rather than nullifying the slot.
            for i in range(2):
                bufs[i] = None
            self._push3_call_counter[graph_mode] = 0
        # Phase 3 only reads slots where topk_idx[t,k] != -1, and those
        # slots are exactly the ones some peer writes in Phase 1 (both are
        # derived from the same capped routing matrix in
        # compute_combine_push_map). So we do NOT need to zero-init the
        # unused region — Phase 3 skips it. Skipping zero_init removes a
        # ~7 ms / call cost (2 × 5.6 GB bf16 writes) which was making
        # UBX_PUSH3_FORCE_FREE=1 ~2x slower than baseline.
        for i in range(2):
            if bufs[i] is None:
                bufs[i] = self.create_tensor(expected_shape, torch.bfloat16)
                if bufs[i] is None:
                    raise RuntimeError(
                        f"combine_push3: failed to allocate dest buf {i} "
                        f"{expected_shape} bf16"
                    )

        call_n = self._push3_call_counter[graph_mode]
        dest_buf = bufs[call_n % 2]

        out = torch.empty((local_ntokens, hidden),
                          dtype=torch.bfloat16, device=self.device)
        lineoffset_dest = (dest_buf.data_ptr() - self.pool_ptr) // 16
        gate_ptr = gate_weights.data_ptr() if gate_weights is not None else 0

        self._push3_call_counter[graph_mode] = call_n + 1
        reduce_id = call_n + 1

        import os as _os
        if _os.environ.get("UBX_PUSH3_DBG", "0") != "0" and reduce_id <= 3:
            print(
                f"[PUSH3-DBG r{self.myrank}] call={reduce_id} "
                f"pool_ptr=0x{self.pool_ptr:016x} "
                f"buf0_ptr=0x{bufs[0].data_ptr():016x} "
                f"buf1_ptr=0x{bufs[1].data_ptr():016x} "
                f"dest_buf_ptr=0x{dest_buf.data_ptr():016x} "
                f"lineoffset_dest={lineoffset_dest} "
                f"n_recv={n_recv} local_ntokens={local_ntokens} "
                f"topk_max={topk_max} max_tpr={max_tokens_per_rank}",
                flush=True,
            )

        _push3_phase_sync = _os.environ.get("UBX_PUSH3_PHASE_SYNC", "0") == "1"
        # Kernel 1: push expert outputs to peers.
        ubx_combine_push3_phase1_write(
            self.world_size, n_recv, blocks_per_token, topk_max,
            inverse_map.data_ptr(), max_tokens_per_rank, lineoffset_dest,
            self.dev_comm_handle, self.window_handle,
            expert_outputs.data_ptr(),
            self.default_sms, smlimit,
        )
        if _push3_phase_sync:
            torch.cuda.synchronize()
            print(f"[PUSH3-PHASE r{self.myrank}] call={reduce_id} after phase1",
                  flush=True)
        # Kernel 2: signal + spin (only kernel that can hang; has timeout).
        # reduce_id is derived from device-resident UBX_FLAG_PUSH3_ID by the
        # kernel itself — required so that captured graphs replay correctly
        # (a host-passed id would freeze at capture-time value while the
        # BAR flag keeps advancing on each replay → instant pass → race).
        ubx_combine_push3_phase2_signal(
            self.world_size,
            self.dev_comm_handle, self.window_handle, self.pool_ptr,
        )
        if _push3_phase_sync:
            torch.cuda.synchronize()
            print(f"[PUSH3-PHASE r{self.myrank}] call={reduce_id} after phase2",
                  flush=True)
        # Kernel 3: purely local sum + write output.
        ubx_combine_push3_phase3_sum(
            local_ntokens, blocks_per_token, topk_max, total_experts,
            topk_idx.data_ptr(), gate_ptr,
            lineoffset_dest, self.pool_ptr,
            out.data_ptr(),
            self.default_sms, smlimit,
        )
        if _push3_phase_sync:
            torch.cuda.synchronize()
            print(f"[PUSH3-PHASE r{self.myrank}] call={reduce_id} after phase3",
                  flush=True)
        # UBX_PUSH3_FORCE_FREE=1: drop dest_buf refs after every call to
        # test whether holding them across MoE layers conflicts with TE's
        # captured cudagraph that touches the NCCL device-API window.
        # The buffers are re-allocated on the next call (symmetric across
        # ranks since all ranks free+alloc in lockstep).
        if _os.environ.get("UBX_PUSH3_FORCE_FREE", "0") == "1":
            for i in range(2):
                bufs[i] = None
            self._push3_call_counter[graph_mode] = 0
        return out

    def combine_wait(self) -> None:
        """Run Phase 2 of the most recent combine_*(..., sync=False) call.

        Polls UBX_FLAG_COMBINE_BAR until all ranks have finished Phase 1,
        then PULLs from peer temp symm buffers, applies gate weights, sums
        in fp32, downcasts to bf16, and writes the output tensor returned
        by the original combine_* call.

        Must be called exactly once after each combine_*(..., sync=False);
        subsequent reads of the output tensor are valid only after this
        returns (or after a stream sync).
        """
        state = getattr(self, "_pending_combine", None)
        if state is None:
            raise RuntimeError(
                "combine_wait: no pending async combine. Call "
                "combine_bf16_bf16/combine_mxfp8_bf16 with sync=False first.")
        gate_ptr = (state["gate_weights"].data_ptr()
                    if state["gate_weights"] is not None else 0)

        if state["wire"] == "bf16":
            ubx_combine_wait_bf16(
                self.world_size,
                state["local_ntokens"], state["blocks_per_token"],
                state["experts_per_rank"],
                state["token_offsets"].data_ptr(), gate_ptr,
                state["lineoffset_temp"],
                self.dev_comm_handle,
                self.window_handle,
                self.pool_ptr,
                state["out"].data_ptr(),
                self.default_sms, state["smlimit"],
            )
        elif state["wire"] == "mxfp8":
            ubx_combine_wait_mxfp8(
                self.world_size,
                state["local_ntokens"], state["blocks_per_token"],
                state["experts_per_rank"],
                state["token_offsets"].data_ptr(), gate_ptr,
                state["lineoffset_temp"], state["lineoffset_scales"],
                self.dev_comm_handle,
                self.window_handle,
                self.pool_ptr,
                state["out"].data_ptr(),
                self.default_sms, state["smlimit"],
            )
        else:
            raise RuntimeError(f"combine_wait: unknown wire {state['wire']!r}")

        # Drop the pending state. The 'temp' SymmTensor reference falls out
        # of scope here, returning the symm slot to the free list.
        self._pending_combine = None
