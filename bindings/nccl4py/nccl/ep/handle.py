# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-step EP handle (:class:`EpHandle`) and the Pythonic dataclasses
that mirror the named-struct ABI used by NCCL EP's dispatch/combine
entry points: :class:`EpHandleConfig`, :class:`EpDispatchConfig`,
:class:`EpCombineConfig`, :class:`EpLayoutInfo`, :class:`EpDispatchInputs`,
:class:`EpDispatchOutputs`, :class:`EpCombineInputs`,
:class:`EpCombineOutputs`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nccl.core.cuda import get_stream_ptr
from nccl.core.typing import NcclInvalid, NcclStreamSpec

from nccl.ep import bindings as _ep_bindings
from nccl.ep._binding_helpers import binding_dataclass

if TYPE_CHECKING:
    from nccl.ep.group import EpGroup
    from nccl.ep.tensor import NDTensor


__all__ = [
    "EpCombineConfig",
    "EpCombineInputs",
    "EpCombineOutputs",
    "EpDispatchConfig",
    "EpDispatchInputs",
    "EpDispatchOutputs",
    "EpHandle",
    "EpHandleConfig",
    "EpLayoutInfo",
]


@binding_dataclass(
    _ep_bindings.nccl_ep.EpHandleConfig,
    size_field_dtype=_ep_bindings.nccl_ep.ep_handle_config_dtype,
)
class EpHandleConfig:
    """Pythonic configuration for :py:meth:`EpHandle.create`.

    Mirrors :c:struct:`ncclEpHandleConfig_t`. All fields default to 0;
    constructing :py:class:`EpHandleConfig` without arguments is
    equivalent to passing ``NULL`` for the C ``config`` argument.

    Attributes:
        use_fp8: Enable FP8 dispatch (default ``False``).
        dispatch_output_per_expert_alignment: HT expert-major only.
            Per-expert zone alignment in tokens (must be a power of 2;
            0/1 = no padding).

    See Also:
        NCCL EP ``ncclEpHandleConfig_t``:
        ``contrib/nccl_ep/include/nccl_ep.h``
    """

    use_fp8: bool = False
    dispatch_output_per_expert_alignment: int = 0


@binding_dataclass(
    _ep_bindings.nccl_ep.EpDispatchConfig,
    size_field_dtype=_ep_bindings.nccl_ep.ep_dispatch_config_dtype,
)
class EpDispatchConfig:
    """Pythonic configuration for :py:meth:`EpHandle.dispatch`.

    Mirrors :c:struct:`ncclEpDispatchConfig_t`. All fields default to 0;
    constructing :py:class:`EpDispatchConfig` without arguments is
    equivalent to passing ``NULL`` for the C ``config`` argument.

    Attributes:
        send_only: If non-zero, only initiate transfers and require a
            subsequent :py:meth:`EpHandle.complete` call. LL mode only.
        round_scales: If non-zero, round the scaling-factor tensor up to
            a power of 2.
    """

    send_only: int = 0
    round_scales: int = 0


@binding_dataclass(
    _ep_bindings.nccl_ep.EpCombineConfig,
    size_field_dtype=_ep_bindings.nccl_ep.ep_combine_config_dtype,
)
class EpCombineConfig:
    """Pythonic configuration for :py:meth:`EpHandle.combine`.

    Mirrors :c:struct:`ncclEpCombineConfig_t`.

    Attributes:
        send_only: If non-zero, only initiate transfers and require a
            subsequent :py:meth:`EpHandle.complete` call. LL mode only.
    """

    send_only: int = 0


@binding_dataclass(
    _ep_bindings.nccl_ep.EpLayoutInfo,
    size_field_dtype=_ep_bindings.nccl_ep.ep_layout_info_dtype,
)
class EpLayoutInfo:
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

    expert_counters: NDTensor | None = None
    src_rank_counters: NDTensor | None = None
    expert_offsets: NDTensor | None = None
    recv_total_counter: NDTensor | None = None


@binding_dataclass(
    _ep_bindings.nccl_ep.EpDispatchInputs,
    size_field_dtype=_ep_bindings.nccl_ep.ep_dispatch_inputs_dtype,
)
class EpDispatchInputs:
    """Input tensor bundle for :py:meth:`EpHandle.dispatch`.

    Mirrors :c:struct:`ncclEpDispatchInputs_t`. ``tokens`` is required;
    other fields are optional (``None`` → C-side ``NULL``).

    Attributes:
        tokens: 2D ``[num_tokens, hidden]``. Token payload.
        topk_weights: 2D ``[num_tokens, top_k]`` float32. LL rank-major
            per-token routing weights, or HT forward routing weights.
        scales: 2D ``[num_tokens, hidden/128]`` float32. HT FP8 only.
    """

    tokens: NDTensor | None = None
    topk_weights: NDTensor | None = None
    scales: NDTensor | None = None


@binding_dataclass(
    _ep_bindings.nccl_ep.EpDispatchOutputs,
    size_field_dtype=_ep_bindings.nccl_ep.ep_dispatch_outputs_dtype,
)
class EpDispatchOutputs:
    """Output tensor bundle for :py:meth:`EpHandle.dispatch`.

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

    tokens: NDTensor | None = None
    topk_weights: NDTensor | None = None
    scales: NDTensor | None = None
    topk_idx: NDTensor | None = None


@binding_dataclass(
    _ep_bindings.nccl_ep.EpCombineInputs,
    size_field_dtype=_ep_bindings.nccl_ep.ep_combine_inputs_dtype,
)
class EpCombineInputs:
    """Input tensor bundle for :py:meth:`EpHandle.combine`.

    Mirrors :c:struct:`ncclEpCombineInputs_t`. ``tokens`` is required;
    other fields are optional.

    Attributes:
        tokens: Post-expert activation tensor (shape depends on
            algorithm/layout — see ``nccl_ep.h``).
        topk_weights: 2D ``[num_recv_tokens, top_k]`` float32. HT
            backward combine only.
    """

    tokens: NDTensor | None = None
    topk_weights: NDTensor | None = None


@binding_dataclass(
    _ep_bindings.nccl_ep.EpCombineOutputs,
    size_field_dtype=_ep_bindings.nccl_ep.ep_combine_outputs_dtype,
)
class EpCombineOutputs:
    """Output tensor bundle for :py:meth:`EpHandle.combine`.

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

    tokens: NDTensor | None = None
    topk_weights: NDTensor | None = None


def _materialize(value: object) -> object:
    """Materialize a Pythonic struct dataclass into its binding instance.

    Returns ``None`` for a ``None`` input. The returned object owns the
    underlying C struct memory; callers MUST keep it alive across the C
    call (otherwise its ``__dealloc__`` frees the struct and the C side
    reads from freed memory).
    """
    if value is None:
        return None
    return value.to_binding()  # type: ignore[attr-defined]


def _ptr_of(binding: object) -> int:
    """Underlying C struct address from a materialized binding, or 0 for None."""
    return 0 if binding is None else binding.ptr


class EpHandle:
    """Per-step routing handle for dispatch/combine.

    Construct via :meth:`create`; release with :meth:`destroy`.
    """

    def __init__(self, ptr: int) -> None:
        self._ptr = ptr

    @classmethod
    def create(
        cls,
        ep_group: EpGroup,
        topk_idx: NDTensor,
        *,
        layout_info: EpLayoutInfo | None = None,
        config: EpHandleConfig | None = None,
        stream: NcclStreamSpec,
    ) -> EpHandle:
        """Collectively create+initialize an EP handle.

        HT mode performs metadata exchange as part of this call.

        Args:
            ep_group: An initialized :class:`EpGroup`.
            topk_idx: Top-k expert indices for this step
                (shape ``[num_tokens, top_k]``, int64).
            stream: CUDA stream for the launch — accepts a
                :class:`cuda.core.Stream`, an integer stream handle, or
                any object exposing ``__cuda_stream__``.
            layout_info: Optional :class:`EpLayoutInfo`. HT: set
                ``expert_counters`` when ``max_send_tokens_per_rank`` is
                ``NCCL_EP_AUTO``. LL mode: must be ``None``.
            config: Optional :class:`EpHandleConfig`. ``None`` forwards
                NULL (use library defaults).
        """
        # Bind materialized structs to locals so their backing memory
        # outlives the C call (binding __dealloc__ frees the struct).
        layout_b = _materialize(layout_info)
        config_b = _materialize(config)
        ptr = _ep_bindings.nccl_ep.ep_create_handle(
            ep_group.ptr,
            topk_idx.ptr,
            _ptr_of(layout_b),
            _ptr_of(config_b),
            get_stream_ptr(stream),
        )
        return cls(ptr)

    def _check_valid(self, operation: str) -> None:
        if not self._ptr:
            raise NcclInvalid(
                f"Cannot {operation}: EpHandle is not initialized or has been destroyed"
            )

    @property
    def ptr(self) -> int:
        """Raw ``ncclEpHandle_t`` address."""
        self._check_valid("read ptr")
        return self._ptr

    def update(
        self,
        topk_idx: NDTensor,
        *,
        layout_info: EpLayoutInfo | None = None,
        stream: NcclStreamSpec,
    ) -> None:
        """Rebind ``topk_idx`` for the next dispatch without reallocating buffers.

        Args:
            topk_idx: New top-k indices tensor for the upcoming dispatch.
            stream: CUDA stream for the launch.
            layout_info: Optional :class:`EpLayoutInfo`. HT: set
                ``expert_counters`` when ``max_send_tokens_per_rank`` is
                ``NCCL_EP_AUTO``. LL mode: must be ``None``.

        See Also:
            :meth:`dispatch`.
        """
        self._check_valid("update")
        layout_b = _materialize(layout_info)
        _ep_bindings.nccl_ep.ep_update_handle(
            self._ptr,
            topk_idx.ptr,
            _ptr_of(layout_b),
            get_stream_ptr(stream),
        )

    def dispatch(
        self,
        inputs: EpDispatchInputs,
        outputs: EpDispatchOutputs,
        *,
        layout_info: EpLayoutInfo | None = None,
        config: EpDispatchConfig | None = None,
        stream: NcclStreamSpec,
    ) -> None:
        """Dispatch tokens to the experts indicated by this handle's top-k routing.

        Routing is fully encoded in the handle (set at
        :meth:`create` / :meth:`update` time via ``topk_idx``), so no
        ``topk_idx`` argument is taken here.

        Args:
            inputs: Input tensor bundle (:class:`EpDispatchInputs`).
                ``inputs.tokens`` is required.
            outputs: Pre-allocated output tensor bundle
                (:class:`EpDispatchOutputs`). ``outputs.tokens`` is
                required; per-layout shape rules in ``nccl_ep.h``.
            stream: CUDA stream for the launch.
            layout_info: Optional :class:`EpLayoutInfo`. LL expert-major
                writes ``expert_counters``; LL rank-major writes
                ``src_rank_counters``.
            config: Optional :class:`EpDispatchConfig`. ``None`` forwards
                NULL.

        See Also:
            :meth:`combine`, :meth:`complete`.
        """
        self._check_valid("dispatch")
        inputs_b = _materialize(inputs)
        outputs_b = _materialize(outputs)
        layout_b = _materialize(layout_info)
        config_b = _materialize(config)
        _ep_bindings.nccl_ep.ep_dispatch(
            self._ptr,
            _ptr_of(inputs_b),
            _ptr_of(outputs_b),
            _ptr_of(layout_b),
            _ptr_of(config_b),
            get_stream_ptr(stream),
        )

    def combine(
        self,
        inputs: EpCombineInputs,
        outputs: EpCombineOutputs,
        *,
        config: EpCombineConfig | None = None,
        stream: NcclStreamSpec,
    ) -> None:
        """Gather expert outputs back to each token's home rank.

        Must reuse the same :class:`EpHandle` from the matching
        :meth:`dispatch`.

        Args:
            inputs: Input tensor bundle (:class:`EpCombineInputs`).
                ``inputs.tokens`` is required; per-layout shape rules in
                ``nccl_ep.h``.
            outputs: Pre-allocated output tensor bundle
                (:class:`EpCombineOutputs`). ``outputs.tokens`` is
                required (shape ``[num_tokens, hidden]``).
            stream: CUDA stream for the launch.
            config: Optional :class:`EpCombineConfig`. ``None`` forwards
                NULL.

        See Also:
            :meth:`dispatch`, :meth:`complete`.
        """
        self._check_valid("combine")
        inputs_b = _materialize(inputs)
        outputs_b = _materialize(outputs)
        config_b = _materialize(config)
        _ep_bindings.nccl_ep.ep_combine(
            self._ptr,
            _ptr_of(inputs_b),
            _ptr_of(outputs_b),
            _ptr_of(config_b),
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
        _ep_bindings.nccl_ep.ep_complete(self._ptr, config, get_stream_ptr(stream))

    def destroy(self) -> None:
        """Release the handle. Subsequent operations on this object are invalid."""
        if self._ptr:
            _ep_bindings.nccl_ep.ep_handle_destroy(self._ptr)
            self._ptr = 0

    def __repr__(self) -> str:
        if self._ptr:
            return f"<EpHandle ptr={self._ptr:#x}>"
        return "<EpHandle destroyed>"
