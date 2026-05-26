# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""Per-step EP handle (:class:`Handle`) and the Pythonic dataclasses
that mirror the named-struct ABI used by NCCL EP's dispatch/combine
entry points: :class:`HandleConfig`, :class:`DispatchConfig`,
:class:`CombineConfig`, :class:`LayoutInfo`, :class:`DispatchInputs`,
:class:`DispatchOutputs`, :class:`CombineInputs`,
:class:`CombineOutputs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nccl.core.cuda import get_stream_ptr
from nccl.core.typing import NcclInvalid, NcclStreamSpec


from nccl._binding_helpers import binding_dataclass
from nccl.bindings import nccl_ep as _ep_bindings
from nccl.ep.enums import PassDir

if TYPE_CHECKING:
    from nccl.ep.tensor import Tensor


__all__ = [
    "CombineConfig",
    "CombineInputs",
    "CombineOutputs",
    "DispatchConfig",
    "DispatchInputs",
    "DispatchOutputs",
    "Handle",
    "HandleConfig",
    "LayoutInfo",
]


@binding_dataclass(_ep_bindings.HandleConfig)
class HandleConfig:
    """Pythonic configuration for :py:meth:`Group.create_handle`.

    Mirrors :c:struct:`ncclEpHandleConfig_t`. All fields default to 0;
    constructing :py:class:`HandleConfig` without arguments is
    equivalent to passing ``NULL`` for the C ``config`` argument.

    Attributes:
        dispatch_output_per_expert_alignment: HT expert-major only.
            Per-expert zone alignment in tokens (must be a power of 2;
            0/1 = no padding).

    See Also:
        NCCL EP ``ncclEpHandleConfig_t``:
        ``contrib/nccl_ep/include/nccl_ep.h``
    """

    dispatch_output_per_expert_alignment: int = 0


@binding_dataclass(_ep_bindings.DispatchConfig)
class DispatchConfig:
    """Pythonic configuration for :py:meth:`Handle.dispatch`.

    Mirrors :c:struct:`ncclEpDispatchConfig_t`. All fields default to 0;
    constructing :py:class:`DispatchConfig` without arguments is
    equivalent to passing ``NULL`` for the C ``config`` argument.

    Attributes:
        send_only: If non-zero, only initiate transfers and require a
            subsequent :py:meth:`Handle.complete` call. LL mode only.
        round_scales: If non-zero, round the scaling-factor tensor up to
            a power of 2.
        pass_direction: Forward (default) or backward pass; HT-only.
            ``FWD`` requires ``inputs.topk_weights``; ``BWD`` forbids it
            and also forbids ``outputs.topk_weights`` / ``outputs.topk_idx``.
    """

    send_only: int = 0
    round_scales: int = 0
    pass_direction: PassDir = PassDir.FWD


@binding_dataclass(_ep_bindings.CombineConfig)
class CombineConfig:
    """Pythonic configuration for :py:meth:`Handle.combine`.

    Mirrors :c:struct:`ncclEpCombineConfig_t`.

    Attributes:
        send_only: If non-zero, only initiate transfers and require a
            subsequent :py:meth:`Handle.complete` call. LL mode only.
        pass_direction: Forward (default) or backward pass; HT-only.
            ``FWD`` forbids ``inputs.topk_weights``; ``BWD`` requires
            both ``inputs.topk_weights`` and ``outputs.topk_weights``.
    """

    send_only: int = 0
    pass_direction: PassDir = PassDir.FWD


@binding_dataclass(_ep_bindings.LayoutInfo)
class LayoutInfo:
    """Named local tensors carried alongside dispatch / create_handle.

    Mirrors :c:struct:`ncclEpLayoutInfo_t`. All fields are optional;
    omitting (or leaving ``None``) is the C-side ``NULL`` sentinel and
    is interpreted per the conventions documented in ``nccl_ep.h``.

    Attributes:
        expert_counters: 1D ``[num_local_experts]`` int32 (or int64 in
            HT expert-major).

            * HT (handle time): per-expert received counts.
            * LL expert-major (dispatch time): per-expert received
              token counts written by NCCL EP.
        src_rank_counters: 1D ``[num_ranks]`` int32. LL rank-major only
            (dispatch time): per-source-rank token counts.
        expert_offsets: 1D ``[num_local_experts]`` int32 or int64. HT
            expert-major only: prefix sum of padded per-expert counts.
        recv_total_counter: 1D ``[1]`` int32 or int64. HT scalar total
            received-token count.
    """

    expert_counters: Tensor | None = None
    src_rank_counters: Tensor | None = None
    expert_offsets: Tensor | None = None
    recv_total_counter: Tensor | None = None


@binding_dataclass(_ep_bindings.DispatchInputs)
class DispatchInputs:
    """Input tensor bundle for :py:meth:`Handle.dispatch`.

    Mirrors :c:struct:`ncclEpDispatchInputs_t`. ``tokens`` is required;
    other fields are optional (``None`` → C-side ``NULL``).

    Attributes:
        tokens: 2D ``[num_tokens, hidden]``. Token payload.
        topk_weights: 2D ``[num_tokens, top_k]`` float32. LL rank-major
            per-token routing weights, or HT forward routing weights.
        scales: 2D ``[num_tokens, hidden/128]`` float32. HT FP8 only.
    """

    tokens: Tensor | None = None
    topk_weights: Tensor | None = None
    scales: Tensor | None = None


@binding_dataclass(_ep_bindings.DispatchOutputs)
class DispatchOutputs:
    """Output tensor bundle for :py:meth:`Handle.dispatch`.

    Mirrors :c:struct:`ncclEpDispatchOutputs_t`. ``tokens`` is required;
    other fields are optional (``None`` → C-side ``NULL``).

    See ``contrib/nccl_ep/include/nccl_ep.h`` for the shape conventions
    of each field across LL/HT and the supported layouts.

    Attributes:
        tokens: Received tokens.
        topk_weights: LL rank-major or HT: received top-k weights.
        scales: FP8 only: received per-token scaling factors.
        topk_idx: LL rank-major or HT: received top-k expert indices.
    """

    tokens: Tensor | None = None
    topk_weights: Tensor | None = None
    scales: Tensor | None = None
    topk_idx: Tensor | None = None


@binding_dataclass(_ep_bindings.CombineInputs)
class CombineInputs:
    """Input tensor bundle for :py:meth:`Handle.combine`.

    Mirrors :c:struct:`ncclEpCombineInputs_t`. ``tokens`` is required;
    other fields are optional.

    Attributes:
        tokens: Post-expert activation tensor (shape depends on
            algorithm/layout — see ``nccl_ep.h``).
        topk_weights: 2D ``[num_recv_tokens, top_k]`` float32. HT
            backward combine only.
    """

    tokens: Tensor | None = None
    topk_weights: Tensor | None = None


@binding_dataclass(_ep_bindings.CombineOutputs)
class CombineOutputs:
    """Output tensor bundle for :py:meth:`Handle.combine`.

    Mirrors :c:struct:`ncclEpCombineOutputs_t`. ``tokens`` is required;
    other fields are optional.

    Attributes:
        tokens: 2D ``[num_tokens, hidden]`` combined output, restored to
            original token order.
        topk_weights: 2D ``[num_tokens, top_k]`` float32.

            * LL expert-major: per-token routing weights applied on the
              receive side.
            * HT backward: combined routing weights output.
    """

    tokens: Tensor | None = None
    topk_weights: Tensor | None = None


class Handle:
    """Per-step routing handle for dispatch/combine.

    Construct via :py:meth:`Group.create_handle`; release with :meth:`destroy`.
    """

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: Handle is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpHandle_t`` address."""
        self._check_valid("read ptr")
        return self._ptr

    def update(
        self,
        topk_idx: Tensor,
        *,
        layout_info: LayoutInfo | None = None,
        stream: NcclStreamSpec,
    ) -> None:
        """Rebind ``topk_idx`` for the next dispatch without reallocating buffers.

        Args:
            topk_idx: New top-k indices tensor for the upcoming dispatch.
            stream: CUDA stream for the launch.
            layout_info: Optional :class:`LayoutInfo`. HT: set
                ``expert_counters`` when ``max_dispatch_tokens_per_rank`` is
                ``NCCL_EP_AUTO``. LL mode: must be ``None``.

        See Also:
            :meth:`dispatch`.
        """
        self._check_valid("update")
        _ep_bindings.update_handle(
            self._ptr,
            topk_idx.ptr,
            layout_info._lowpp.ptr if layout_info is not None else 0,  # type: ignore[attr-defined]
            get_stream_ptr(stream),
        )

    def dispatch(
        self,
        inputs: DispatchInputs,
        outputs: DispatchOutputs,
        *,
        layout_info: LayoutInfo | None = None,
        config: DispatchConfig | None = None,
        stream: NcclStreamSpec,
    ) -> None:
        """Dispatch tokens to the experts indicated by this handle's top-k routing.

        Routing is fully encoded in the handle (set at
        :meth:`create` / :meth:`update` time via ``topk_idx``), so no
        ``topk_idx`` argument is taken here.

        Args:
            inputs: Input tensor bundle (:class:`DispatchInputs`).
                ``inputs.tokens`` is required.
            outputs: Pre-allocated output tensor bundle
                (:class:`DispatchOutputs`). ``outputs.tokens`` is
                required; per-layout shape rules in ``nccl_ep.h``.
            stream: CUDA stream for the launch.
            layout_info: Optional :class:`LayoutInfo`. LL expert-major
                writes ``expert_counters``; LL rank-major writes
                ``src_rank_counters``.
            config: Optional :class:`DispatchConfig`. ``None`` forwards
                NULL.

        See Also:
            :meth:`combine`, :meth:`complete`.
        """
        self._check_valid("dispatch")
        _ep_bindings.dispatch(
            self._ptr,
            inputs._lowpp.ptr,  # type: ignore[attr-defined]
            outputs._lowpp.ptr,  # type: ignore[attr-defined]
            layout_info._lowpp.ptr if layout_info is not None else 0,  # type: ignore[attr-defined]
            config._lowpp.ptr if config is not None else 0,  # type: ignore[attr-defined]
            get_stream_ptr(stream),
        )

    def combine(
        self,
        inputs: CombineInputs,
        outputs: CombineOutputs,
        *,
        config: CombineConfig | None = None,
        stream: NcclStreamSpec,
    ) -> None:
        """Gather expert outputs back to each token's home rank.

        Must reuse the same :class:`Handle` from the matching
        :meth:`dispatch`.

        Args:
            inputs: Input tensor bundle (:class:`CombineInputs`).
                ``inputs.tokens`` is required; per-layout shape rules in
                ``nccl_ep.h``.
            outputs: Pre-allocated output tensor bundle
                (:class:`CombineOutputs`). ``outputs.tokens`` is
                required (shape ``[num_tokens, hidden]``).
            stream: CUDA stream for the launch.
            config: Optional :class:`CombineConfig`. ``None`` forwards
                NULL.

        See Also:
            :meth:`dispatch`, :meth:`complete`.
        """
        self._check_valid("combine")
        _ep_bindings.combine(
            self._ptr,
            inputs._lowpp.ptr,  # type: ignore[attr-defined]
            outputs._lowpp.ptr,  # type: ignore[attr-defined]
            config._lowpp.ptr if config is not None else 0,  # type: ignore[attr-defined]
            get_stream_ptr(stream),
        )

    def complete(
        self,
        *,
        config: int = 0,
        stream: NcclStreamSpec,
    ) -> None:
        """Complete a staged dispatch/combine (LL mode with ``send_only=True``).

        Args:
            config: Reserved for future options; must be 0 per current
                ``nccl_ep.h``. Exposed for forward-compatibility so a
                future library can accept a non-zero handle without a
                facade rebuild.
            stream: CUDA stream for the launch.

        See Also:
            :meth:`dispatch`, :meth:`combine`.
        """
        self._check_valid("complete")
        _ep_bindings.complete(self._ptr, config, get_stream_ptr(stream))

    def destroy(self) -> None:
        """Release the handle. Subsequent operations on this object are invalid."""
        if self._ptr:
            _ep_bindings.handle_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        if self._ptr:
            return f"<Handle ptr={self._ptr:#x}>"
        return "<Handle destroyed>"
