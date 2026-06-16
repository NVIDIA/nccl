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
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.base_dsl._mlir_helpers.op import dsl_user_op

from . import _bindings as raw
from ._structs import _LLVMPtrType, ncclGin_C, ncclTeam, ncclCoopAny


@dsl_user_op
def _alloca_ncclGin_C(*, loc=None, ip=None) -> ir.Value:
    """Alloca uninitialized ``ncclGin_C`` storage on the kernel stack.

    Returns:
        ``!llvm.ptr`` ir.Value to the storage.
    """
    return llvm.alloca(
        res=ir.Type.parse("!llvm.ptr"),
        array_size=cutlass.Int32(1).ir_value(),
        elem_type=ncclGin_C._struct_type,
        loc=loc,
        ip=ip,
    )


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
        given_release: int = 3,
        required_release: int = 1,
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
            given_release: release-fence flags from the caller.
            required_release: release-fence flags required by the op.
        """
        # Window.local_pointer returns a raw !llvm.ptr; wrap it via cute.make_ptr
        # so both sides expose .toint() and we can subtract uniformly. dtype is
        # Int8 to make the resulting subtraction a byte offset.
        dst_base = cute.make_ptr(cutlass.Int8, dst_win.local_pointer(0))
        src_base = cute.make_ptr(cutlass.Int8, src_win.local_pointer(0))
        dst_offset = dst.iterator.toint() - dst_base.toint()
        src_offset = src.iterator.toint() - src_base.toint()
        size = (dst.element_type.width // 8) * cute.size(dst)
        raw.ncclGinPut(
            self.ptr, team, peer, dst_win, dst_offset, src_win, src_offset, size,
            is_signal, signal_id, signal_op, signal_op_arg,
            is_counter, counter_id, coop,
            is_descriptor, descriptor_ptr,
            given_release, required_release,
        )

    def wait_signal(
        self,
        coop: ncclCoopAny,
        *,
        signal: int = 0,
        least: int = 1,
        bits: int = 64,
        ord: int = 2,
    ) -> None:
        """Wait for a GIN signal slot to reach a threshold value.

        Args:
            coop: cooperative group issuing the wait.
            signal: signal slot id (matches ``signal_id`` of the producing
                :meth:`put`). Default 0.
            least: minimum value the slot must reach. Default 1.
            bits: signal slot width in bits (currently 64).
            ord: ``cuda::memory_order`` integer; default 2 = ``ACQUIRE``.
                See :class:`~nccl.core.device.cute.types.MemoryOrder`.
        """
        raw.ncclGinWaitSignal(self.ptr, coop, signal, least, bits, ord)

    @dsl_user_op
    def value(self, *, loc=None, ip=None) -> ncclGin_C:
        """Load the underlying ``ncclGin_C`` struct value.

        Used by barrier-session factories whose C signatures take it by
        value rather than by pointer.

        Returns:
            ``ncclGin_C`` struct value.
        """
        return ncclGin_C(
            llvm.load(res=ncclGin_C._struct_type, addr=self.ptr, loc=loc, ip=ip)
        )


__all__ = [
    "Gin",
]
