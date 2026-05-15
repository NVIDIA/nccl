# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""EP group lifecycle: ``EpGroup`` and its configuration ``EpGroupConfig``."""

from __future__ import annotations

from dataclasses import field
from typing import TYPE_CHECKING

from nccl.core.typing import NcclInvalid

from nccl.ep import bindings as _ep_bindings
from nccl.ep._binding_helpers import binding_dataclass
from nccl.ep.allocator import EpAllocConfig
from nccl.ep.enums import NcclEpAlgorithm, NcclEpLayout

if TYPE_CHECKING:
    from nccl.core import Communicator


__all__ = ["EpGroup", "EpGroupConfig"]


_NCCL_EP_API_VERSION = 1


@binding_dataclass(
    _ep_bindings.nccl_ep.EpGroupConfig,
    size_field_dtype=_ep_bindings.nccl_ep.ep_group_config_dtype,
)
class EpGroupConfig:
    """Pythonic configuration for :py:meth:`EpGroup.create`.

    Mirrors :c:struct:`ncclEpGroupConfig_t`. Fields left at their
    defaults (``0`` or ``LOW_LATENCY``/``AUTO``) forward as
    ``NCCL_EP_AUTO`` where applicable.

    Attributes:
        algorithm: Dispatch/combine algorithm. Default: ``LOW_LATENCY``.
        num_experts: Total number of experts across all ranks. Required.
        max_send_tokens_per_rank: Maximum tokens any single rank will
            dispatch. Must be > 0; ``NCCL_EP_AUTO`` is not yet supported
            even in HT mode.
        max_token_bytes: Token payload size in bytes (independent of
            datatype). Required.
        layout: Receive-buffer layout. ``AUTO`` (default) picks
            ``EXPERT_MAJOR`` for LL and ``FLAT`` for HT.
        rdma_buffer_size: RDMA buffer size in bytes. 0 selects a default
            sized for any algorithm.
        num_qp_per_rank: Number of QPs per rank. 0 selects auto.
        num_channels: Channels per rank. 0 selects auto. In HT each
            channel occupies 2 SMs.
        max_recv_token_slots_per_rank: Total recv-slot budget per rank.
            HT requires > 0 and ``>= max_send_tokens_per_rank``; LL
            with 0 auto-derives ``n_ranks * max_send_tokens_per_rank``.
        max_num_sms: Maximum SMs to use for EP kernels (dispatch,
            combine, preprocessing). 0 selects an algorithm-dependent
            default.
        version: ABI version. Defaults to ``NCCL_EP_API_VERSION``.
        alloc: Device allocator hooks. Default
            :class:`EpAllocConfig` selects ``cudaMalloc``/``cudaFree``.
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

    algorithm: NcclEpAlgorithm = NcclEpAlgorithm.LOW_LATENCY
    num_experts: int = 0
    max_send_tokens_per_rank: int = 0
    max_token_bytes: int = 0
    layout: NcclEpLayout = NcclEpLayout.AUTO
    rdma_buffer_size: int = 0
    num_qp_per_rank: int = 0
    num_channels: int = 0
    max_recv_token_slots_per_rank: int = 0
    max_num_sms: int = 0
    version: int = _NCCL_EP_API_VERSION
    alloc: EpAllocConfig = field(default_factory=EpAllocConfig)
    enable_mask: bool = False
    timeout_ns: int = 0


class EpGroup:
    """A NCCL EP group built on top of an existing :class:`Communicator`.

    Construct via :meth:`create`; release with :meth:`destroy`.
    """

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    @classmethod
    def create(
        cls,
        comm: Communicator,
        config: EpGroupConfig,
    ) -> EpGroup:
        """Collectively create an EP group across all ranks of *comm*.

        Args:
            comm: An initialized :class:`nccl.core.Communicator`.
            config: Filled-in :class:`EpGroupConfig` describing the
                group. Custom allocators live in ``config.alloc``
                (:class:`EpAllocConfig`) — see :mod:`nccl.ep.allocator`
                for usage and lifetime requirements.

        See Also:
            :meth:`destroy`.
        """
        binding = config.to_binding()  # type: ignore[attr-defined]
        ptr = _ep_bindings.nccl_ep.ep_create_group(comm.ptr, binding.ptr)
        return cls(ptr)

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: EpGroup is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpGroup_t`` address."""
        self._check_valid("read ptr")
        return self._ptr

    def destroy(self) -> None:
        """Release the group. Subsequent operations on this object are invalid."""
        if self._ptr:
            _ep_bindings.nccl_ep.ep_group_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        if self._ptr:
            return f"<EpGroup ptr={self._ptr:#x}>"
        return "<EpGroup destroyed>"
