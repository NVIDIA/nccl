# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information

"""GIN transfers, signal actions, and completion operations for CuTeDSL."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import TYPE_CHECKING, NamedTuple, overload

import cutlass
from cutlass import cute
from cutlass.cutlass_dsl import ir

from nccl.core.typing import NcclGinType

from . import _nccl_device_ir as _ir
from ._coop import Coop
from ._team import Team
from ._types import MemoryOrder
from ._window import SymmetricPointer

if TYPE_CHECKING:
    from ._comm import DevComm

__all__ = [
    "Gin",
    "GinAllContexts",
    "GinBackendMask",
    "GinCounterInc",
    "GinDescriptorSmem",
    "GinOptFlag",
    "GinRequest",
    "GinResourceSharingMode",
    "GinSegmentType",
    "GinSignalAdd",
    "GinSignalFollowResult",
    "GinSignalInc",
    "ThreadScope",
]


class ThreadScope(IntEnum):
    """Scopes for the ``given_release`` and ``required_release`` arguments."""

    SYSTEM = _ir._CUDA_THREAD_SCOPE_SYSTEM
    DEVICE = _ir._CUDA_THREAD_SCOPE_DEVICE
    BLOCK = _ir._CUDA_THREAD_SCOPE_BLOCK
    THREAD = _ir._CUDA_THREAD_SCOPE_THREAD


class GinBackendMask(IntFlag):
    """Compile-time GIN backend selection; currently only ``ALL`` is supported."""

    PROXY = 1 << NcclGinType.PROXY
    GDAKI = 1 << NcclGinType.GDAKI
    GPI = 1 << NcclGinType.GPI
    EFA_GDA = 1 << NcclGinType.EFA_GDA
    ALL = PROXY | GDAKI | GPI | EFA_GDA


class GinResourceSharingMode(IntEnum):
    """The scope at which GIN network resources are shared."""

    GPU = _ir._NCCL_GIN_RESOURCE_SHARING_GPU
    CTA = _ir._NCCL_GIN_RESOURCE_SHARING_CTA
    THREAD = _ir._NCCL_GIN_RESOURCE_SHARING_THREAD


class GinOptFlag(IntFlag):
    """Per-operation GIN optimization flags."""

    DEFAULT = _ir._NCCL_GIN_OPT_FLAGS_DEFAULT

    MAY_SKIP_CREDIT_CHECK = _ir._NCCL_GIN_OPT_FLAGS_MAY_SKIP_CREDIT_CHECK
    """Allow the backend to skip request-queue credit checks.

    Safety requirements are backend-specific; enable only when they are met.
    """

    AGGREGATE_REQUESTS = _ir._NCCL_GIN_OPT_FLAGS_AGGREGATE_REQUESTS
    """Hint that more requests are expected, allowing deferred submission.

    Publish pending requests with an ordered, non-aggregated submission to
    each affected queue before waiting or leaving the kernel. Flush and wait
    do not substitute for publishing deferred submissions.
    """


class GinSegmentType(IntEnum):
    """Physical backing of both transfer ranges, including any host memory."""

    DEVICE = _ir._NCCL_GIN_SEGMENT_TYPE_DEVICE
    MIXED = _ir._NCCL_GIN_SEGMENT_TYPE_MIXED
    HOST_NUMA = _ir._NCCL_GIN_SEGMENT_TYPE_HOST_NUMA


@dataclass(frozen=True, eq=False)
class GinSignalInc:
    """Increment a remote signal after the attached operation completes.

    Strong signals also order preceding puts to the same peer/context and
    require host setup with gin_strong_signals_required=True.
    Do not mix increment and add operations on a signal without a reset.
    """

    signal_id: int | cutlass.Integer
    strong: bool | cutlass.Boolean = field(kw_only=True)


@dataclass(frozen=True, eq=False)
class GinSignalAdd:
    """Add an unsigned value to a remote signal after the attached operation completes.

    See ``GinSignalInc`` for strong ordering, host setup, and signal reuse requirements.
    """

    signal_id: int | cutlass.Integer
    value: int | cutlass.Integer
    strong: bool | cutlass.Boolean = field(kw_only=True)


@dataclass(frozen=True, eq=False)
class GinCounterInc:
    """Increment a local counter when the attached put has consumed its source.

    This does not complete other puts or establish remote settlement.
    """

    counter_id: int | cutlass.Integer


class GinSignalFollowResult(NamedTuple):
    """The previous shadow and observed progress, both masked to the chosen width."""

    before: cutlass.Uint64
    delta: cutlass.Uint64


class GinDescriptorSmem:
    """CTA-shared descriptor storage for GIN submissions.

    Reuse after all participants finish the submitting call; network completion
    is not required. Overlapping calls need separate descriptor storage.
    """

    def __init__(self) -> None:
        self._ptr = _ir._NcclGinDescriptorSmem._allocate(memory_space=cutlass.AddressSpace.smem)

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self._ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> GinDescriptorSmem:
        obj = object.__new__(type(self))
        obj._ptr = values[0]
        return obj

    def _to_ptr(self) -> ir.Value:
        return self._ptr


class GinRequest:
    """Opaque completion request returned by Gin.flush_async().

    Keep its storage alive and use the original Gin and Coop when waiting.
    A timeout does not cancel the request or allow buffers to be reused.
    Requests cannot be carried into another kernel invocation.
    """

    def __init__(self) -> None:
        raise TypeError("Use Gin.flush_async() to create a request")

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self._ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> GinRequest:
        return type(self)._from_ptr(values[0])

    @classmethod
    def _from_ptr(cls, ptr: ir.Value) -> GinRequest:
        obj = object.__new__(cls)
        obj._ptr = ptr
        return obj

    def _to_ptr(self) -> ir.Value:
        return self._ptr


class GinAllContexts:
    """Select all GIN contexts of a communicator for a barrier."""

    def __init__(self, dev_comm: DevComm) -> None:
        self._dev_comm = dev_comm

    def __extract_mlir_values__(self) -> list[ir.Value] | None:
        return self._dev_comm.__extract_mlir_values__()

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> GinAllContexts:
        return type(self)(self._dev_comm.__new_from_mlir_values__(values))


def _remote_action_to_ir_args(
    action: GinSignalInc | GinSignalAdd | None,
) -> tuple[
    _ir._NcclGinSignalType,
    ir.Value,
    _ir._SizeT,
    _ir._NcclGinSignalT,
    cutlass.Boolean,
    _ir._NcclGinSignalOpT,
    cutlass.Uint64,
]:
    if action is None:
        signal_type, signal_id, strong, op, value = _ir._NCCL_GIN_SIGNAL_TYPE_NONE, 0, False, 0, 0
    elif isinstance(action, GinSignalInc):
        signal_type, signal_id, strong = (
            _ir._NCCL_GIN_SIGNAL_TYPE_INDEXED,
            action.signal_id,
            action.strong,
        )
        op, value = _ir._NCCL_GIN_SIGNAL_INC, 1
    elif isinstance(action, GinSignalAdd):
        signal_type, signal_id, strong = (
            _ir._NCCL_GIN_SIGNAL_TYPE_INDEXED,
            action.signal_id,
            action.strong,
        )
        op, value = _ir._NCCL_GIN_SIGNAL_ADD, action.value
    else:
        raise TypeError("remote_action must be GinSignalInc, GinSignalAdd, or None")
    return (
        _ir._NcclGinSignalType(signal_type),
        cutlass.inttoptr(0, cutlass.AddressSpace.generic, cutlass.Int8),
        _ir._SizeT(0),
        _ir._NcclGinSignalT(signal_id),
        cutlass.Boolean(strong),
        _ir._NcclGinSignalOpT(op),
        cutlass.Uint64(value),
    )


def _local_action_to_ir_args(
    action: GinCounterInc | None,
) -> tuple[cutlass.Boolean, _ir._NcclGinCounterT]:
    if action is None:
        is_counter, counter_id = False, 0
    elif isinstance(action, GinCounterInc):
        is_counter, counter_id = True, action.counter_id
    else:
        raise TypeError("local_action must be GinCounterInc or None")
    return cutlass.Boolean(is_counter), _ir._NcclGinCounterT(counter_id)


def _descriptor_to_ir_args(
    descriptor: GinDescriptorSmem | None,
) -> tuple[cutlass.Boolean, ir.Value]:
    if descriptor is None:
        return cutlass.Boolean(False), cutlass.inttoptr(
            0, cutlass.AddressSpace.generic, cutlass.Int8
        )
    if not isinstance(descriptor, GinDescriptorSmem):
        raise TypeError("descriptor must be GinDescriptorSmem or None")
    return cutlass.Boolean(True), descriptor._to_ptr()


def _validate_bit_width(bits: int | cutlass.Integer, maximum: int) -> None:
    if not isinstance(bits, (int, cutlass.Integer)):
        raise TypeError("bits must be an int or cutlass.Integer")
    static_bits = bits.value if isinstance(bits, cutlass.Integer) else bits
    if isinstance(static_bits, int) and not 1 <= static_bits <= maximum:
        raise ValueError(f"bits must be in [1, {maximum}]")


def _transfer_bytes(
    first: SymmetricPointer,
    second: SymmetricPointer,
    count: int | cutlass.Integer,
) -> cutlass.Uint64:
    if first.dtype is not second.dtype:
        raise TypeError("transfer endpoints must have matching dtypes")
    if not isinstance(count, (int, cutlass.Integer)):
        raise TypeError("count must be an int or cutlass.Integer")
    static_count = count.value if isinstance(count, cutlass.Integer) else count
    if isinstance(static_count, int) and static_count < 0:
        raise ValueError("count must be nonnegative")
    count_value = cutlass.Uint64(count).ir_value()
    element_size_value = cutlass.Uint64(first.element_size).ir_value()
    return cutlass.Uint64(count_value * element_size_value)


class Gin:
    """A view of one pre-created GIN context.

    Construct with ``Gin(dev_comm, ...)`` or ``dev_comm.gin(...)`` inside device code.

    Transfers use contiguous registered endpoints and element counts. Callers
    ensure bounds, alignment, and valid signal/counter IDs from host setup.
    All Coop participants call collective operations uniformly.

    Signal and counter waits compare modulo ``2**bits``. Keep the distance
    between the expected and actual values below half that range.

    Storage belongs to the creating device function. Completion on an aborted
    communicator does not establish data readiness. Timed calls return an NCCL
    status, which may differ across Coop participants. Timeouts do not cancel
    work; resources must remain valid until completion or coordinated abort recovery.
    """

    def __init__(
        self,
        dev_comm: DevComm,
        context_id: int | cutlass.Integer = 0,
        *,
        backend_mask: GinBackendMask = GinBackendMask.ALL,
        resource_sharing_mode: GinResourceSharingMode = GinResourceSharingMode.GPU,
    ) -> None:
        if GinBackendMask(backend_mask) != GinBackendMask.ALL:
            raise NotImplementedError("Only GinBackendMask.ALL is currently supported")
        self._ptr = _ir._NcclIrGin._allocate()
        _ir._nccl_ir_gin_init(
            self._ptr,
            dev_comm._to_ptr(),
            cutlass.Int32(context_id),
            _ir._NcclGinResourceSharingMode(resource_sharing_mode),
        )

    def __extract_mlir_values__(self) -> list[ir.Value]:
        return [self._ptr]

    def __new_from_mlir_values__(self, values: list[ir.Value]) -> Gin:
        return type(self)._from_ptr(values[0])

    @classmethod
    def _from_ptr(cls, ptr: ir.Value) -> Gin:
        obj = object.__new__(cls)
        obj._ptr = ptr
        return obj

    def _to_ptr(self) -> ir.Value:
        return self._ptr

    def put(
        self,
        coop: Coop,
        team: Team,
        peer: int | cutlass.Integer,
        dst: SymmetricPointer,
        src: SymmetricPointer,
        count: int | cutlass.Integer,
        *,
        remote_action: GinSignalInc | GinSignalAdd | None = None,
        local_action: GinCounterInc | None = None,
        descriptor: GinDescriptorSmem | None = None,
        given_release: ThreadScope = ThreadScope.THREAD,
        required_release: ThreadScope = ThreadScope.DEVICE,
        optimization_flags: GinOptFlag = GinOptFlag.DEFAULT,
        segment_type: GinSegmentType = GinSegmentType.DEVICE,
    ) -> None:
        """Put ``count`` elements to ``peer`` and optionally signal completion.

        Both endpoints must have the same dtype. The segment type describes
        both ranges. Release scopes do not replace producer synchronization.
        """
        size = _transfer_bytes(dst, src, count)
        remote_args = _remote_action_to_ir_args(remote_action)
        local_args = _local_action_to_ir_args(local_action)
        descriptor_args = _descriptor_to_ir_args(descriptor)
        _ir._nccl_ir_gin_put(
            self._ptr,
            team._to_ptr(),
            cutlass.Int32(peer),
            dst.window._to_ptr(),
            _ir._SizeT(dst.byte_offset),
            src.window._to_ptr(),
            _ir._SizeT(src.byte_offset),
            size,
            *remote_args,
            *local_args,
            coop._to_ptr(),
            *descriptor_args,
            _ir._CudaThreadScope(given_release),
            _ir._CudaThreadScope(required_release),
            cutlass.Uint32(optimization_flags),
            _ir._NcclGinSegmentTypeT(segment_type),
        )

    def put_value(
        self,
        coop: Coop,
        team: Team,
        peer: int | cutlass.Integer,
        dst: SymmetricPointer,
        value: int | cutlass.Integer,
        *,
        remote_action: GinSignalInc | GinSignalAdd | None = None,
        descriptor: GinDescriptorSmem | None = None,
        given_release: ThreadScope = ThreadScope.THREAD,
        required_release: ThreadScope = ThreadScope.DEVICE,
        optimization_flags: GinOptFlag = GinOptFlag.DEFAULT,
    ) -> None:
        """Write one Uint8, Uint16, Uint32, or Uint64 scalar.

        Python integers must fit ``dst.dtype``. DSL scalars must have the same
        dtype as the destination; convert explicitly when needed.
        """
        if dst.dtype not in (cutlass.Uint8, cutlass.Uint16, cutlass.Uint32, cutlass.Uint64):
            raise TypeError("put_value requires a Uint8, Uint16, Uint32, or Uint64 destination")
        if isinstance(value, cutlass.Integer):
            if type(value) is not dst.dtype:
                raise TypeError(
                    "DSL scalar dtype must match the destination; convert it explicitly"
                )
        elif isinstance(value, int):
            if not 0 <= value < (1 << dst.dtype.width):
                raise OverflowError(f"value is out of range for {dst.dtype.__name__}")
        else:
            raise TypeError("value must be a Python int or a matching unsigned DSL scalar")
        remote_args = _remote_action_to_ir_args(remote_action)
        descriptor_args = _descriptor_to_ir_args(descriptor)
        _ir._nccl_ir_gin_put_value(
            self._ptr,
            team._to_ptr(),
            cutlass.Int32(peer),
            dst.window._to_ptr(),
            _ir._SizeT(dst.byte_offset),
            cutlass.Uint64(value),
            _ir._SizeT(dst.element_size),
            *remote_args,
            coop._to_ptr(),
            *descriptor_args,
            _ir._CudaThreadScope(given_release),
            _ir._CudaThreadScope(required_release),
            cutlass.Uint32(optimization_flags),
        )

    def get(
        self,
        coop: Coop,
        team: Team,
        peer: int | cutlass.Integer,
        remote: SymmetricPointer,
        local: SymmetricPointer,
        count: int | cutlass.Integer,
        *,
        descriptor: GinDescriptorSmem | None = None,
        optimization_flags: GinOptFlag = GinOptFlag.DEFAULT,
        segment_type: GinSegmentType = GinSegmentType.DEVICE,
    ) -> None:
        """Get ``count`` elements from ``peer`` into a registered local endpoint.

        Both endpoints must have the same dtype. Flush or wait for completion
        before consuming the destination.
        """
        size = _transfer_bytes(remote, local, count)
        descriptor_args = _descriptor_to_ir_args(descriptor)
        _ir._nccl_ir_gin_get(
            self._ptr,
            team._to_ptr(),
            cutlass.Int32(peer),
            remote.window._to_ptr(),
            _ir._SizeT(remote.byte_offset),
            local.window._to_ptr(),
            _ir._SizeT(local.byte_offset),
            size,
            coop._to_ptr(),
            *descriptor_args,
            cutlass.Uint32(optimization_flags),
            _ir._NcclGinSegmentTypeT(segment_type),
        )

    def signal(
        self,
        coop: Coop,
        team: Team,
        peer: int | cutlass.Integer,
        remote_action: GinSignalInc | GinSignalAdd,
        *,
        descriptor: GinDescriptorSmem | None = None,
        given_release: ThreadScope = ThreadScope.THREAD,
        required_release: ThreadScope = ThreadScope.DEVICE,
        optimization_flags: GinOptFlag = GinOptFlag.DEFAULT,
    ) -> None:
        """Submit a remote signal action without a data transfer."""
        if remote_action is None:
            raise TypeError("signal requires a remote action")
        remote_args = _remote_action_to_ir_args(remote_action)
        descriptor_args = _descriptor_to_ir_args(descriptor)
        _ir._nccl_ir_gin_signal(
            self._ptr,
            team._to_ptr(),
            cutlass.Int32(peer),
            *remote_args,
            coop._to_ptr(),
            *descriptor_args,
            _ir._CudaThreadScope(given_release),
            _ir._CudaThreadScope(required_release),
            cutlass.Uint32(optimization_flags),
        )

    def read_signal(
        self,
        signal_id: int | cutlass.Integer,
        *,
        bits: int | cutlass.Integer = 64,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> cutlass.Uint64:
        """Read a signal masked to ``bits`` in [1, 64]."""
        _validate_bit_width(bits, 64)
        return cutlass.Uint64(
            _ir._nccl_ir_gin_read_signal(
                self._ptr,
                _ir._NcclGinSignalT(signal_id),
                cutlass.Int32(bits),
                _ir._CudaMemoryOrder(order),
            )
        )

    @overload
    def wait_signal(
        self,
        coop: Coop,
        signal_id: int | cutlass.Integer,
        *,
        least: int | cutlass.Integer,
        bits: int | cutlass.Integer = 64,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        timeout_cycles: None = None,
    ) -> None: ...

    @overload
    def wait_signal(
        self,
        coop: Coop,
        signal_id: int | cutlass.Integer,
        *,
        least: int | cutlass.Integer,
        bits: int | cutlass.Integer = 64,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        timeout_cycles: int | cutlass.Integer,
    ) -> cutlass.Int32: ...
    def wait_signal(
        self,
        coop: Coop,
        signal_id: int | cutlass.Integer,
        *,
        least: int | cutlass.Integer,
        bits: int | cutlass.Integer = 64,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        timeout_cycles: int | cutlass.Integer | None = None,
    ) -> cutlass.Int32 | None:
        """Wait for a signal threshold; a supplied timeout returns an NCCL status."""
        _validate_bit_width(bits, 64)
        args = (
            self._ptr,
            coop._to_ptr(),
            _ir._NcclGinSignalT(signal_id),
            cutlass.Uint64(least),
            cutlass.Int32(bits),
            _ir._CudaMemoryOrder(order),
        )
        if timeout_cycles is None:
            _ir._nccl_ir_gin_wait_signal(*args)
            return None
        return cutlass.Int32(
            _ir._nccl_ir_gin_wait_signal_timeout(*args, cutlass.Uint64(timeout_cycles))
        )

    def increase_signal_shadow(
        self,
        signal_id: int | cutlass.Integer,
        delta: int | cutlass.Integer,
    ) -> None:
        """Increase the signal's shadow with a relaxed CTA-scope atomic add."""
        _ir._nccl_ir_gin_increase_signal_shadow(
            self._ptr, _ir._NcclGinSignalT(signal_id), cutlass.Uint64(delta)
        )

    def wait_signal_meet_shadow(
        self,
        coop: Coop,
        signal_id: int | cutlass.Integer,
        *,
        bits: int | cutlass.Integer = 64,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> None:
        """Wait for the signal to reach its shadow without changing the shadow."""
        _validate_bit_width(bits, 64)
        _ir._nccl_ir_gin_wait_signal_meet_shadow(
            self._ptr,
            coop._to_ptr(),
            _ir._NcclGinSignalT(signal_id),
            cutlass.Int32(bits),
            _ir._CudaMemoryOrder(order),
        )

    def wait_signal_follow_shadow(
        self,
        coop: Coop,
        signal_id: int | cutlass.Integer,
        *,
        least_delta: int | cutlass.Integer = 1,
        bits: int | cutlass.Integer = 64,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> GinSignalFollowResult:
        """Wait for progress and advance the shadow; return its old value and delta."""
        _validate_bit_width(bits, 64)
        before = cutlass.Array(cutlass.Uint64, 1)
        delta = cutlass.Array(cutlass.Uint64, 1)
        _ir._nccl_ir_gin_wait_signal_follow_shadow(
            self._ptr,
            coop._to_ptr(),
            _ir._NcclGinSignalT(signal_id),
            cutlass.Uint64(least_delta),
            before.data_ptr(),
            delta.data_ptr(),
            cutlass.Int32(bits),
            _ir._CudaMemoryOrder(order),
        )
        return GinSignalFollowResult(before[0], delta[0])

    def signal_shadow_pointer(self, signal_id: int | cutlass.Integer) -> cute.Pointer:
        """Return a Uint64 pointer to the user-managed signal shadow."""
        ptr = _ir._nccl_ir_gin_get_signal_shadow_ptr(self._ptr, _ir._NcclGinSignalT(signal_id))
        return cute.make_ptr(cutlass.Uint64, ptr)

    def reset_signal(self, signal_id: int | cutlass.Integer) -> None:
        """Clear the signal and shadow; no outstanding operation may update either."""
        _ir._nccl_ir_gin_reset_signal(self._ptr, _ir._NcclGinSignalT(signal_id))

    def read_counter(
        self,
        counter_id: int | cutlass.Integer,
        *,
        bits: int | cutlass.Integer = 56,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> cutlass.Uint64:
        """Read a local completion counter masked to ``bits`` in [1, 56]."""
        _validate_bit_width(bits, 56)
        return cutlass.Uint64(
            _ir._nccl_ir_gin_read_counter(
                self._ptr,
                _ir._NcclGinCounterT(counter_id),
                cutlass.Int32(bits),
                _ir._CudaMemoryOrder(order),
            )
        )

    @overload
    def wait_counter(
        self,
        coop: Coop,
        counter_id: int | cutlass.Integer,
        *,
        least: int | cutlass.Integer,
        bits: int | cutlass.Integer = 56,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        timeout_cycles: None = None,
    ) -> None: ...

    @overload
    def wait_counter(
        self,
        coop: Coop,
        counter_id: int | cutlass.Integer,
        *,
        least: int | cutlass.Integer,
        bits: int | cutlass.Integer = 56,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        timeout_cycles: int | cutlass.Integer,
    ) -> cutlass.Int32: ...
    def wait_counter(
        self,
        coop: Coop,
        counter_id: int | cutlass.Integer,
        *,
        least: int | cutlass.Integer,
        bits: int | cutlass.Integer = 56,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        timeout_cycles: int | cutlass.Integer | None = None,
    ) -> cutlass.Int32 | None:
        """Wait for local source consumption; a timeout returns an NCCL status."""
        _validate_bit_width(bits, 56)
        args = (
            self._ptr,
            coop._to_ptr(),
            _ir._NcclGinCounterT(counter_id),
            cutlass.Uint64(least),
            cutlass.Int32(bits),
            _ir._CudaMemoryOrder(order),
        )
        if timeout_cycles is None:
            _ir._nccl_ir_gin_wait_counter(*args)
            return None
        return cutlass.Int32(
            _ir._nccl_ir_gin_wait_counter_timeout(*args, cutlass.Uint64(timeout_cycles))
        )

    def reset_counter(self, counter_id: int | cutlass.Integer) -> None:
        """Clear a counter when no outstanding operation can update it."""
        _ir._nccl_ir_gin_reset_counter(self._ptr, _ir._NcclGinCounterT(counter_id))

    @overload
    def flush(
        self,
        coop: Coop,
        *,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        descriptor: GinDescriptorSmem | None = None,
        timeout_cycles: None = None,
    ) -> None: ...

    @overload
    def flush(
        self,
        coop: Coop,
        *,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        descriptor: GinDescriptorSmem | None = None,
        timeout_cycles: int | cutlass.Integer,
    ) -> cutlass.Int32: ...
    def flush(
        self,
        coop: Coop,
        *,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        descriptor: GinDescriptorSmem | None = None,
        timeout_cycles: int | cutlass.Integer | None = None,
    ) -> cutlass.Int32 | None:
        """Complete the Coop's prior work on this context, with an optional timeout.

        Completion allows source reuse for puts and local consumption of gets.
        It does not establish remote settlement of puts or fence other contexts.
        """
        descriptor_args = _descriptor_to_ir_args(descriptor)
        args = (self._ptr, coop._to_ptr(), _ir._CudaMemoryOrder(order), *descriptor_args)
        if timeout_cycles is None:
            _ir._nccl_ir_gin_flush(*args)
            return None
        return cutlass.Int32(_ir._nccl_ir_gin_flush_timeout(*args, cutlass.Uint64(timeout_cycles)))

    def flush_async(
        self,
        coop: Coop,
        team: Team,
        peer: int | cutlass.Integer,
        *,
        descriptor: GinDescriptorSmem | None = None,
        optimization_flags: GinOptFlag = GinOptFlag.DEFAULT,
    ) -> GinRequest:
        """Capture prior work to a peer; wait before reusing resources.

        Requires backend request support. The current EFA_GDA backend does not
        implement requests; use blocking flush on that backend.
        """
        request_ptr = _ir._NcclGinRequestT._allocate()
        descriptor_args = _descriptor_to_ir_args(descriptor)
        _ir._nccl_ir_gin_flush_async(
            self._ptr,
            team._to_ptr(),
            cutlass.Uint32(peer),
            request_ptr,
            coop._to_ptr(),
            cutlass.Uint32(optimization_flags),
            *descriptor_args,
        )
        return GinRequest._from_ptr(request_ptr)

    @overload
    def wait(
        self,
        coop: Coop,
        request: GinRequest,
        *,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        descriptor: GinDescriptorSmem | None = None,
        timeout_cycles: None = None,
    ) -> None: ...

    @overload
    def wait(
        self,
        coop: Coop,
        request: GinRequest,
        *,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        descriptor: GinDescriptorSmem | None = None,
        timeout_cycles: int | cutlass.Integer,
    ) -> cutlass.Int32: ...
    def wait(
        self,
        coop: Coop,
        request: GinRequest,
        *,
        order: MemoryOrder = MemoryOrder.ACQUIRE,
        descriptor: GinDescriptorSmem | None = None,
        timeout_cycles: int | cutlass.Integer | None = None,
    ) -> cutlass.Int32 | None:
        """Complete a request using its original Gin and Coop.

        A timeout does not cancel the request. See ``flush_async()`` for backend support.
        """
        if not isinstance(request, GinRequest):
            raise TypeError("request must be a GinRequest")
        descriptor_args = _descriptor_to_ir_args(descriptor)
        args = (
            self._ptr,
            request._to_ptr(),
            coop._to_ptr(),
            *descriptor_args,
            _ir._CudaMemoryOrder(order),
        )
        if timeout_cycles is None:
            _ir._nccl_ir_gin_wait(*args)
            return None
        return cutlass.Int32(_ir._nccl_ir_gin_wait_timeout(*args, cutlass.Uint64(timeout_cycles)))
