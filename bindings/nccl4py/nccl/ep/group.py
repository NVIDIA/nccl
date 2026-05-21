# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EP group lifecycle: ``Group`` and its configuration ``GroupConfig``."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from nccl.core.typing import NcclInvalid

from nccl.core.cuda import get_stream_ptr
from nccl.core.typing import NcclStreamSpec

from nccl.bindings import nccl_ep as _ep_bindings
from nccl.ep._binding_helpers import binding_dataclass
from nccl.ep.allocator import AllocConfig
from nccl.ep.enums import Algorithm, Layout

if TYPE_CHECKING:
    from nccl.core import Communicator
    from nccl.ep.handle import Handle, HandleConfig, LayoutInfo
    from nccl.ep.tensor import Tensor


__all__ = ["Group", "GroupConfig"]


_NCCL_EP_API_VERSION = 1


@binding_dataclass(
    _ep_bindings.GroupConfig,
    size_field_dtype=_ep_bindings.group_config_dtype,
)
class GroupConfig:
    """Pythonic configuration for :py:meth:`Group.create`.

    Mirrors :c:struct:`ncclEpGroupConfig_t`. Fields left at their
    defaults (``0`` or ``LOW_LATENCY``/``AUTO``) forward as
    ``NCCL_EP_AUTO`` where applicable.

    Attributes:
        algorithm: Dispatch/combine algorithm. Default: ``LOW_LATENCY``.
        num_experts: Total number of experts across all ranks. Required.
        max_dispatch_tokens_per_rank: Maximum tokens any single rank will
            dispatch. Must be > 0; ``NCCL_EP_AUTO`` is not yet supported
            even in HT mode.
        max_token_bytes: Token payload size in bytes (independent of
            datatype). Required.
        rdma_buffer_size: RDMA buffer size in bytes. 0 selects a default
            sized for any algorithm.
        num_qp_per_rank: Number of QPs per rank. 0 selects auto.
        num_channels: Channels per rank. 0 selects auto. In HT each
            channel occupies 2 SMs.
        max_recv_tokens_per_rank: Total recv-slot budget per rank.
            HT requires > 0 and ``>= max_dispatch_tokens_per_rank``; LL
            with 0 auto-derives ``n_ranks * max_dispatch_tokens_per_rank``.
        max_num_sms: Maximum SMs to use for EP kernels (dispatch,
            combine, preprocessing). 0 selects an algorithm-dependent
            default.
        version: ABI version. Defaults to ``NCCL_EP_API_VERSION``.
        alloc: Device allocator hooks. Default
            :class:`AllocConfig` selects ``cudaMalloc``/``cudaFree``.
        enable_mask: Enable active-mask support for fault tolerance
            (LL mode only). When ``True``, a per-rank mask buffer is
            allocated; remote ranks that time out during dispatch or
            combine are skipped rather than tripping a GPU trap, and a
            host-visible error flag is set (pollable via the mask APIs).
            Default: ``False``.
        timeout_ns: GPU-side wait-loop timeout in nanoseconds. ``0``
            selects the library default (~100 s). Setting too low risks
            false positives. The ``NCCL_EP_TIMEOUT_MS`` env var
            overrides this field at group creation.

    See Also:
        NCCL EP ``ncclEpGroupConfig_t``:
        ``contrib/nccl_ep/include/nccl_ep.h``
    """

    algorithm: Algorithm = Algorithm.LOW_LATENCY
    num_experts: int = 0
    max_dispatch_tokens_per_rank: int = 0
    max_recv_tokens_per_rank: int = 0
    max_token_bytes: int = 0
    rdma_buffer_size: int = 0
    num_qp_per_rank: int = 0
    num_channels: int = 0
    max_num_sms: int = 0
    version: int = _NCCL_EP_API_VERSION
    alloc: AllocConfig = field(default_factory=AllocConfig)
    enable_mask: bool = False
    timeout_ns: int = 0


class Group:
    """A NCCL EP group built on top of an existing :class:`Communicator`.

    Construct via :meth:`create`; release with :meth:`destroy`.
    """

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    @classmethod
    def create(
        cls,
        comm: Communicator,
        config: GroupConfig,
    ) -> Group:
        """Collectively create an EP group across all ranks of *comm*.

        Args:
            comm: An initialized :class:`nccl.core.Communicator`.
            config: Filled-in :class:`GroupConfig` describing the
                group. Custom allocators live in ``config.alloc``
                (:class:`AllocConfig`) — see :mod:`nccl.ep.allocator`
                for usage and lifetime requirements.

        See Also:
            :meth:`destroy`.
        """
        binding = config.to_binding()  # type: ignore[attr-defined]
        ptr = _ep_bindings.create_group(comm.ptr, binding.ptr)
        return cls(ptr)

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: Group is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpGroup_t`` address."""
        self._check_valid("read ptr")
        return self._ptr

    def create_handle(
        self,
        layout: Layout,
        topk_idx: Tensor,
        *,
        layout_info: LayoutInfo | None = None,
        config: HandleConfig | None = None,
        stream: NcclStreamSpec,
    ) -> Handle:
        """Collectively create and initialize a :class:`Handle` over this group.

        HT mode performs metadata exchange as part of this call.

        Args:
            layout: Receive-buffer layout. Required — must not be
                :py:attr:`Layout.UNSET`. HT supports ``FLAT`` /
                ``EXPERT_MAJOR``; LL supports ``EXPERT_MAJOR`` /
                ``RANK_MAJOR``.
            topk_idx: Top-k expert indices for this step
                (shape ``[num_tokens, top_k]``, int64).
            layout_info: Optional :class:`LayoutInfo`. HT: set
                ``expert_counters`` when ``max_dispatch_tokens_per_rank``
                is ``NCCL_EP_AUTO``. LL mode: must be ``None``.
            config: Optional :class:`HandleConfig`; ``None`` forwards
                NULL (library defaults).
            stream: CUDA stream for the launch.
        """
        from nccl.ep.handle import Handle, _materialize, _ptr_of
        self._check_valid("create_handle")
        # Bind materialized structs to locals so their backing memory
        # outlives the C call (binding __dealloc__ frees the struct).
        layout_b = _materialize(layout_info)
        config_b = _materialize(config)
        ptr = _ep_bindings.create_handle(
            self._ptr,
            int(layout),
            topk_idx.ptr,
            _ptr_of(layout_b),
            _ptr_of(config_b),
            get_stream_ptr(stream),
        )
        return Handle(ptr)

    def destroy(self) -> None:
        """Release the group. Subsequent operations on this object are invalid."""
        if self._ptr:
            _ep_bindings.group_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        if self._ptr:
            return f"<Group ptr={self._ptr:#x}>"
        return "<Group destroyed>"
