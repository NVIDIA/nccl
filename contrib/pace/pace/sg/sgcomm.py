import torch
import torch.distributed as dist
from typing import List, Tuple, Union, Optional

import pace_cpp
from ..utils import EventOverlap, CommConfig, BaseComm


class SGComm(BaseComm):

    def __init__(self, group : dist.ProcessGroup, num_local_ranks : int, config : CommConfig = None):
        """
            Initialize the SGComm buffer.
            Args:
                group: the process group to exchange information.
                num_local_ranks: the number of local ranks in a node.
                config: the communication configuration.
        """
        self.config = config if config else self.get_recommended_config(num_local_ranks, group.size())
        super().__init__(group, num_local_ranks)

    def _build_runtime(self, topo_key):
        c = self.config
        return pace_cpp.SGComm(self.rank, self.num_ranks, self.num_local_ranks, c.slot_unroll, c.nvl_ring_size, c.rdma_ring_size, c.num_sms, topo_key)

    @staticmethod
    def get_recommended_config(num_local_ranks: int, num_ranks: int) -> CommConfig:
        """
        Recommended data-ring configs, tuned per GPU and node count.

        Selection logic:
          Multinode (any GPU):
              unroll=16, nvl_ring=4, rdma_ring=4, num_sms=32.
          Intranode (H100 sm_90): unroll=4, nvl_ring=4, rdma_ring=2, num_sms=32.
          Intranode (B300 sm_100+): unroll=16, nvl_ring=4, rdma_ring=2, num_sms=32.
        """
        is_multinode = (num_local_ranks != num_ranks)
        if is_multinode:
            return CommConfig(16, 4, 4, 32)

        # Detect GPU type via compute capability (sm_90 = H100, sm_100+ = Blackwell).
        # Fallback to H100 values if detection fails (e.g. no CUDA context yet).
        try:
            props = torch.cuda.get_device_properties(0)
            cc_major = props.major
        except Exception:
            cc_major = 9  # Assume H100.
        is_blackwell = cc_major >= 10

        num_sms = 32
        if is_blackwell:
            return CommConfig(16, 4, 2, num_sms)
        return CommConfig(4, 4, 2, num_sms)
    
    def scatter_gather(self, tensor: Union[torch.Tensor, List[torch.Tensor]], scatter_dim = 0, gather_dim = 1, out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, previous_event: Optional[EventOverlap] = None, async_finish: bool = False):
        """
        Perform scatter-gather on different dimensions

        Input:
            `tensor`: the tensor to be scattered and gathered, shaped like (S, H, D). If it is tensor List, all tensor shall share same shape except gather_dim
            `scatter_dim`: the dimension to be scattered. Either scatter 0 gather 1 or scatter 1 gather 0 is supported.
            `gather_dim`: the dimension to be gathered. Either scatter 0 gather 1 or scatter 1 gather 0 is supported.
            `previous_event`: the previous event to be waited for. If None, it will capture newest event on the current stream.
            `async_finish`: whether to make current stream await sgcomm stream immediately. If not, function will return an event, which can be called when you need to make current stream await sgcomm stream.
            `out`: optional pre-allocated output tensor(s) to write the result into (in-place). Must match `tensor` in count and, per tensor, the computed result shape ((S // num_ranks, H * num_ranks, D) for scatter 0 gather 1, else (S * num_ranks, H // num_ranks, D)), dtype, be contiguous CUDA, and 16-byte (int4) aligned. If None, the library allocates the result.

            Strided input (ulysses CP / USP qkv split): the source may be a
            non-contiguous (S, H, D) view whose only non-contiguous axis is
            dim 0 (S). Required strides (validated before launch):
              stride(2) == 1,  stride(1) == D,  stride(0) >= H*D,
              and (stride(0) * element_size) % 16 == 0,  with size(2)*element_size % 16 == 0.
            Canonical case: Q = storage[:, 0, :, :] of a (S, G, H, D) C-order
            storage, with strides (G*H*D, D, 1). The strided source is read
            directly (no copy-in). See `_validate_sg_input` for the exact checks.

        Output:
            `result` (Union[torch.Tensor, List[torch.Tensor]]): the result tensor, shaped like (S // num_ranks, H * num_ranks, D) if scatter 0 gather 1 else (S * num_ranks, H // num_ranks, D). Same object(s) as `out` when provided.
            `event` (Optional[EventOverlap]): the completion event if async_finish is True, else None.
        """
        assert (scatter_dim, gather_dim) in [(1, 0), (0, 1)], 'Only support scatter_dim=1 and gather_dim=0 or scatter_dim=0 and gather_dim=1.'
        list_trans = False
        if isinstance(tensor, torch.Tensor):
            tensor = [tensor]
            list_trans = True
        else:
            for t in tensor:
                assert isinstance(t, torch.Tensor)
        # Validate each input tensor's layout. Contiguous tensors take the
        # unchanged fast path; non-contiguous tensors must satisfy the strided
        # contract (honored directly by the strided SG kernel, no copy-in).
        for i, t in enumerate(tensor):
            self._validate_sg_input(t, i)
        out_list = None
        if out is not None:
            out_list = [out] if isinstance(out, torch.Tensor) else list(out)
            assert len(out_list) == len(tensor), 'out must provide one tensor per input tensor'
        result, event = self.runtime.scatter_gather(tensor, scatter_dim, gather_dim, getattr(previous_event, 'event', None), async_finish, out_list)
        evt = EventOverlap(event) if event is not None else None
        result = result[0] if list_trans else result
        return result, evt

    # --- Strided-input layout contract (ulysses CP / USP qkv split) ----------
    # The strided axis is S = dim 0 of the (S, H, D) source, in both s1g0 and
    # s0g1. dims 1,2 must be contiguous (the inner H*D block); dim 0 may carry
    # a group stride (e.g. Q = storage[:,0] of (S, G, H, D) C-order, stride
    # (G*H*D, D, 1)). The SG kernels read the source directly, no copy.
    # Mirrors the C++ host checks in sgcomm.cpp ScatterGather; keep these in
    # sync with the kernel's contract.
    _STRIDE_ALIGN = 16  # sizeof(float4) / int4

    def _validate_sg_input(self, t: torch.Tensor, idx: int) -> None:
        if t.is_contiguous():
            return  # unchanged fast path
        if t.dim() != 3:
            raise ValueError(
                f"[SGComm] input {idx}: expected a 3-D (S, H, D) tensor, got dim={t.dim()}")
        D, H, esize = t.size(2), t.size(1), t.element_size()
        st = t.stride()
        reasons = []
        if st[2] != 1:
            reasons.append(f"stride(2)=={st[2]} (must be 1)")
        if st[1] != D:
            reasons.append(f"stride(1)=={st[1]} (must equal size(2)={D})")
        if st[0] < H * D:
            reasons.append(f"stride(0)=={st[0]} (must be >= size(1)*size(2)={H * D})")
        if (st[0] * esize) % self._STRIDE_ALIGN != 0:
            reasons.append(f"stride(0)*element_size={st[0] * esize} (must be %{self._STRIDE_ALIGN} == 0)")
        if (D * esize) % self._STRIDE_ALIGN != 0:
            reasons.append(f"size(2)*element_size={D * esize} (must be %{self._STRIDE_ALIGN} == 0)")
        if reasons:
            raise ValueError(
                f"[SGComm] input {idx}: non-contiguous tensor does not satisfy the strided-S "
                f"contract (the only non-contiguous axis may be dim 0 / S): " + "; ".join(reasons))
        # The contract holds — every SG engine reads the strided-S source
        # directly (no copy-in): the SM-resident kernels honor the S-row stride
        # in the in-Y arg slot, and the 0-SM stream path walks rows at the
        # per-tensor src_row_strides pitch. So a valid strided input is accepted
        # unconditionally (there is no strided-capability gate; kernel selection
        # is gated by num_sms instead, see runtime.num_sms()).
        return

    def get_split_tensors(self, tensors : Union[torch.Tensor, List[torch.Tensor]], output: torch.Tensor):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        outs = []
        tsizes = [t.numel() * t.element_size() for t in tensors]
        align = 16 # int4 aligned
        aligned_sizes = [(s + align - 1) & ~(align - 1) for s in tsizes]
        prefix_sizes = [0]
        psize = 0
        for s in aligned_sizes:
            psize += s // tensors[0].element_size()
            prefix_sizes.append(psize)
        outs = [output[prefix_sizes[i]:prefix_sizes[i] + tensors[i].numel()] for i in range(len(tensors))]
        return outs
    
    def get_comm_slots(self, tensors : Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        tsizes = [t.numel() * t.element_size() // self.num_ranks for t in tensors]
        align = 16 # int4 aligned
        aligned_sizes = [(s + align - 1) & ~(align - 1) for s in tsizes]
        total_size = sum(aligned_sizes)
        slot_bytes = align * 1024 * self.config.slot_unroll
        return (total_size + slot_bytes - 1) // slot_bytes
    
    def is_ring_mode(self):
        return self.runtime.is_ring_mode()