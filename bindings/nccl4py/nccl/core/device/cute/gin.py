"""GIN (network) device API for the CuTeDSL layer.

A :class:`Gin` instance exposes :meth:`Gin.put` and
:meth:`Gin.wait_signal`. ``Gin.put`` takes ``cute.Tensor`` arguments
(constructed via :meth:`Window.tensor`) and derives byte offsets and
transfer size from each tensor.

Construct via :meth:`DevComm.gin`::

    gin = dev_comm.gin(GinBackendMask.ALL, context_id=0)
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from . import _bindings
from ._helpers import _to_ptr, _to_coop_value, _to_value
from ._structs import _LLVMPtrType, ncclGin_C, ncclTeam, ncclCoopAny
from .types import MemoryOrder, ThreadScope


@cute.native_struct
class Gin:
    """Wraps a ``ncclGin_C*`` (pointer to a stack-alloca'd ``ncclGin_C`` struct).

    Instances must be produced through :meth:`DevComm.gin`.
    """

    ptr: _LLVMPtrType

    def put(
        self,
        team: ncclTeam,
        peer: int,
        dst_win,
        dst,
        src_win,
        src,
        coop: ncclCoopAny,
        *,
        is_signal: bool = False,
        signal_id: int = 0,
        signal_op: int = 0,
        signal_op_arg: int = 0,
        is_counter: bool = False,
        counter_id: int = 0,
        is_descriptor: bool = False,
        descriptor_ptr=0,
        given_release: ThreadScope = ThreadScope.THREAD,
        required_release: ThreadScope = ThreadScope.DEVICE,
        opt_flags: int = 0,
    ) -> None:
        """Put ``src`` to ``dst`` on ``peer`` of ``team``.

        ``dst`` / ``src`` are ``cute.Tensor`` instances from
        :meth:`Window.tensor`; byte offsets and transfer size are derived
        from them. ``dst`` and ``src`` must have the same byte size.

        Args:
            team: addressing domain.
            peer: destination rank within ``team``.
            dst_win: destination :class:`Window`.
            dst: destination ``cute.Tensor`` inside ``dst_win``.
            src_win: source :class:`Window` (local).
            src: source ``cute.Tensor`` inside ``src_win``.
            coop: cooperative group issuing the put.
            is_signal: write ``signal_id`` after the transfer completes.
            signal_id: signal slot id (when ``is_signal``).
            signal_op: signal op code per the C ABI (when ``is_signal``).
            signal_op_arg: argument to ``signal_op`` (when ``is_signal``).
            is_counter: increment ``counter_id`` after completion.
            counter_id: counter id (when ``is_counter``).
            is_descriptor: ``descriptor_ptr`` carries an external descriptor.
            descriptor_ptr: descriptor pointer (when ``is_descriptor``).
            given_release: ``cuda::thread_scope`` released by the caller;
                default ``THREAD``. See
                :class:`~nccl.core.device.cute.types.ThreadScope`.
            required_release: ``cuda::thread_scope`` the op requires released;
                default ``DEVICE``.
        """
        # Window.local_pointer returns a raw !llvm.ptr; wrap it via cute.make_ptr
        # so both sides expose .toint() and we can subtract uniformly. dtype is
        # Int8 to make the resulting subtraction a byte offset.
        dst_base = cute.make_ptr(cutlass.Int8, dst_win.local_pointer(0))
        src_base = cute.make_ptr(cutlass.Int8, src_win.local_pointer(0))
        dst_offset = dst.iterator.toint() - dst_base.toint()
        src_offset = src.iterator.toint() - src_base.toint()
        size = (dst.element_type.width // 8) * cute.size(dst)
        _bindings.nccl_gin_put(
            self.ptr, _to_value(team), cutlass.Int32(peer),
            dst_win.ptr, cutlass.Int64(dst_offset),
            src_win.ptr, cutlass.Int64(src_offset),
            cutlass.Int64(size),
            cutlass.Boolean(is_signal), cutlass.Int32(signal_id),
            cutlass.Int32(signal_op), cutlass.Int64(signal_op_arg),
            cutlass.Boolean(is_counter), cutlass.Int32(counter_id),
            _to_coop_value(coop),
            cutlass.Boolean(is_descriptor), _to_ptr(descriptor_ptr),
            cutlass.Int32(int(given_release)), cutlass.Int32(int(required_release)),
            cutlass.Int32(opt_flags),
        )

    def put_value(
        self,
        team: ncclTeam,
        peer: int,
        dst_win,
        dst,
        value: int,
        coop: ncclCoopAny,
        *,
        is_signal: bool = False,
        signal_id: int = 0,
        signal_op: int = 0,
        signal_op_arg: int = 0,
        is_descriptor: bool = False,
        descriptor_ptr=0,
        given_release: ThreadScope = ThreadScope.THREAD,
        required_release: ThreadScope = ThreadScope.DEVICE,
        opt_flags: int = 0,
    ) -> None:
        """Put scalar ``value`` to ``dst`` on ``peer`` of ``team``.

        ``dst`` is a ``cute.Tensor`` instance from :meth:`Window.tensor`;
        its byte offset and element size are derived from it.

        Args:
            team: addressing domain.
            peer: destination rank within ``team``.
            dst_win: destination :class:`Window`.
            dst: destination ``cute.Tensor`` inside ``dst_win``.
            value: scalar value to transfer using ``dst``'s element size.
            coop: cooperative group issuing the put.
            is_signal: write ``signal_id`` after the transfer completes.
            signal_id: signal slot id (when ``is_signal``).
            signal_op: signal op code per the C ABI (when ``is_signal``).
            signal_op_arg: argument to ``signal_op`` (when ``is_signal``).
            is_descriptor: ``descriptor_ptr`` carries an external descriptor.
            descriptor_ptr: descriptor pointer (when ``is_descriptor``).
            given_release: ``cuda::thread_scope`` released by the caller;
                default ``THREAD``. See
                :class:`~nccl.core.device.cute.types.ThreadScope`.
            required_release: ``cuda::thread_scope`` the op requires released;
                default ``DEVICE``.
        """
        # Window.local_pointer returns a raw !llvm.ptr; wrap it via cute.make_ptr
        # so both sides expose .toint() and we can subtract uniformly. dtype is
        # Int8 to make the resulting subtraction a byte offset.
        dst_base = cute.make_ptr(cutlass.Int8, dst_win.local_pointer(0))
        dst_offset = dst.iterator.toint() - dst_base.toint()
        size = dst.element_type.width // 8
        _bindings.nccl_gin_put_value(
            self.ptr, _to_value(team), cutlass.Int32(peer),
            dst_win.ptr, cutlass.Int64(dst_offset),
            cutlass.Int64(value), cutlass.Int64(size),
            cutlass.Boolean(is_signal), cutlass.Int32(signal_id),
            cutlass.Int32(signal_op), cutlass.Int64(signal_op_arg),
            _to_coop_value(coop), cutlass.Boolean(is_descriptor), _to_ptr(descriptor_ptr),
            cutlass.Int32(int(given_release)), cutlass.Int32(int(required_release)),
            cutlass.Int32(opt_flags),
        )

    def get(
        self,
        team: ncclTeam,
        peer: int,
        remote_win,
        remote,
        local_win,
        local,
        coop: ncclCoopAny,
        *,
        is_descriptor: bool = False,
        descriptor_ptr=0,
        opt_flags: int = 0,
    ) -> None:
        """Get ``remote`` to ``local`` from ``peer`` of ``team``.

        ``remote`` / ``local`` are ``cute.Tensor`` instances from
        :meth:`Window.tensor`; byte offsets and transfer size are derived
        from them. ``remote`` and ``remote_win`` must have the same byte size.

        Args:
            team: addressing domain.
            peer: source rank within ``team``.
            remote_win: source :class:`Window` (remote).
            remote: source ``cute.Tensor`` inside ``remote_win``.
            local_win: destination :class:`Window` (local).
            local: destination ``cute.Tensor`` inside ``local_win``.
            coop: cooperative group issuing the get.
            is_descriptor: ``descriptor_ptr`` carries an external descriptor.
            descriptor_ptr: descriptor pointer (when ``is_descriptor``).
            opt_flags: GIN operation flags. Default 0.
        """
        # Window.local_pointer returns a raw !llvm.ptr; wrap it via cute.make_ptr
        # so both sides expose .toint() and we can subtract uniformly. dtype is
        # Int8 to make the resulting subtraction a byte offset.
        remote_base = cute.make_ptr(cutlass.Int8, remote_win.local_pointer(0))
        local_base = cute.make_ptr(cutlass.Int8, local_win.local_pointer(0))
        remote_offset = remote.iterator.toint() - remote_base.toint()
        local_offset = local.iterator.toint() - local_base.toint()
        size = (local.element_type.width // 8) * cute.size(local)
        _bindings.nccl_gin_get(
            self.ptr, _to_value(team), cutlass.Int32(peer),
            remote_win.ptr, cutlass.Int64(remote_offset),
            local_win.ptr, cutlass.Int64(local_offset),
            cutlass.Int64(size),
            _to_coop_value(coop), cutlass.Boolean(is_descriptor), _to_ptr(descriptor_ptr),
            cutlass.Int32(opt_flags),
        )

    def signal(
        self,
        team: ncclTeam,
        peer: int,
        is_signal: bool,
        signal_id: int,
        signal_op: int,
        signal_op_arg: int,
        coop: ncclCoopAny,
        *,
        is_descriptor: bool = False,
        descriptor_ptr=0,
        given_release: ThreadScope = ThreadScope.THREAD,
        required_release: ThreadScope = ThreadScope.DEVICE,
        opt_flags: int = 0,
    ) -> None:
        """Signal a GIN operation.

        Args:
            team: addressing domain.
            peer: destination rank within ``team``.
            is_signal: write ``signal_id`` after the transfer completes.
            signal_id: signal slot id (when ``is_signal``).
            signal_op: signal op code per the C ABI (when ``is_signal``).
            signal_op_arg: argument to ``signal_op`` (when ``is_signal``).
            coop: cooperative group issuing the signal.
            is_descriptor: ``descriptor_ptr`` carries an external descriptor.
            descriptor_ptr: descriptor pointer (when ``is_descriptor``).
            given_release: ``cuda::thread_scope`` released by the caller;
                default ``THREAD``. See
                :class:`~nccl.core.device.cute.types.ThreadScope`.
            required_release: ``cuda::thread_scope`` the op requires released;
                default ``DEVICE``.
        """
        _bindings.nccl_gin_signal(
            self.ptr, _to_value(team), cutlass.Int32(peer),
            cutlass.Boolean(is_signal), cutlass.Int32(signal_id),
            cutlass.Int32(signal_op), cutlass.Int64(signal_op_arg),
            _to_coop_value(coop), cutlass.Boolean(is_descriptor), _to_ptr(descriptor_ptr),
            cutlass.Int32(int(given_release)), cutlass.Int32(int(required_release)),
            cutlass.Int32(opt_flags),
        )

    def read_signal(
        self,
        *,
        signal: int = 0,
        bits: int = 64,
        ord: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> cutlass.Int64:
        """Read the value of a GIN signal slot.

        Args:
            signal: signal slot id (matches ``signal_id`` of the producing
                :meth:`put`). Default 0.
            bits: signal slot width in bits (currently 64).
            ord: ``cuda::memory_order``; default ``ACQUIRE``. See
                :class:`~nccl.core.device.cute.types.MemoryOrder`.

        Returns:
            value of the signal slot.
        """
        return cutlass.Int64(_bindings.nccl_gin_read_signal(
            self.ptr, cutlass.Int32(signal), cutlass.Int32(bits), cutlass.Int32(int(ord))))

    def wait_signal(
        self,
        coop: ncclCoopAny,
        *,
        signal: int = 0,
        least: int = 1,
        bits: int = 64,
        ord: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> None:
        """Wait for a GIN signal slot to reach a threshold value.

        Args:
            coop: cooperative group issuing the wait.
            signal: signal slot id (matches ``signal_id`` of the producing
                :meth:`put`). Default 0.
            least: minimum value the slot must reach. Default 1.
            bits: signal slot width in bits (currently 64).
            ord: ``cuda::memory_order``; default ``ACQUIRE``. See
                :class:`~nccl.core.device.cute.types.MemoryOrder`.
        """
        _bindings.nccl_gin_wait_signal(
            self.ptr, _to_coop_value(coop), cutlass.Int32(signal),
            cutlass.Int64(least), cutlass.Int32(bits), cutlass.Int32(int(ord)))

    def read_counter(
        self,
        *,
        counter: int = 0,
        bits: int = 56,
        ord: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> cutlass.Int64:
        """Read the value of a GIN counter.

        Args:
            counter: counter id (matches ``counter_id`` of the producing
                :meth:`put`). Default 0.
            bits: counter width in bits. Default 56.
            ord: ``cuda::memory_order``; default ``ACQUIRE``. See
                :class:`~nccl.core.device.cute.types.MemoryOrder`.

        Returns:
            value of the counter.
        """
        return cutlass.Int64(_bindings.nccl_gin_read_counter(
            self.ptr, cutlass.Int32(counter), cutlass.Int32(bits), cutlass.Int32(int(ord))))

    def wait_counter(
        self,
        coop: ncclCoopAny,
        *,
        counter: int = 0,
        least: int = 1,
        bits: int = 56,
        ord: MemoryOrder = MemoryOrder.ACQUIRE,
    ) -> None:
        """Wait for a GIN counter to reach a threshold value.

        Args:
            coop: cooperative group issuing the wait.
            counter: counter id (matches ``counter_id`` of the producing
                :meth:`put`). Default 0.
            least: minimum value the counter must reach. Default 1.
            bits: counter width in bits. Default 56.
            ord: ``cuda::memory_order``; default ``ACQUIRE``. See
                :class:`~nccl.core.device.cute.types.MemoryOrder`.
        """
        _bindings.nccl_gin_wait_counter(
            self.ptr, _to_coop_value(coop), cutlass.Int32(counter),
            cutlass.Int64(least), cutlass.Int32(bits), cutlass.Int32(int(ord)))

    def reset_counter(self, *, counter: int) -> None:
        """Reset a GIN counter to zero.

        This operation must not race with concurrent modifications to the
        counter.

        Args:
            counter: counter id to reset.
        """
        _bindings.nccl_gin_reset_counter(self.ptr, cutlass.Int32(counter))

    def reset_signal(self, *, signal: int) -> None:
        """Reset a GIN signal slot and its user-managed shadow to zero.

        This operation must not race with concurrent modifications to the
        signal.

        Args:
            signal: signal slot id to reset.
        """
        _bindings.nccl_gin_reset_signal(self.ptr, cutlass.Int32(signal))

    def signal_shadow_pointer(self, *, signal: int = 0) -> cute.Pointer:
        """Return the device pointer to a signal slot's user-managed shadow.

        The shadow is independent state and is not automatically kept in sync
        with the signal. :meth:`reset_signal` resets both the signal and this
        shadow to zero.

        Args:
            signal: signal slot id. Default 0.

        Returns:
            ``cute.Pointer`` of ``Uint64`` at the ``uint64_t`` shadow.
        """
        return cute.make_ptr(
            cutlass.Uint64,
            _bindings.nccl_gin_get_signal_shadow_ptr(self.ptr, cutlass.Int32(signal)),
        )

    def flush(self, coop: ncclCoopAny, ord: MemoryOrder = MemoryOrder.ACQUIRE) -> None:
        """Flush pending GIN operations.

        Args:
            coop: cooperative group issuing the flush.
            ord: ``cuda::memory_order``; default ``ACQUIRE``. See
                :class:`~nccl.core.device.cute.types.MemoryOrder`.
        """
        _bindings.nccl_gin_flush(self.ptr, _to_coop_value(coop), cutlass.Int32(int(ord)))

    @dsl_user_op
    def value(self, *, loc=None, ip=None) -> ncclGin_C:
        """Load the underlying ``ncclGin_C`` struct value.

        Used by barrier-session factories whose C signatures take it by
        value rather than by pointer.

        Returns:
            ``ncclGin_C`` struct value.
        """
        return ncclGin_C(
            llvm.load(ncclGin_C._struct_type, self.ptr, loc=loc, ip=ip)
        )


__all__ = [
    "Gin",
]
