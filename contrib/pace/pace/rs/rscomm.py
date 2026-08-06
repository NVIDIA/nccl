import torch
import torch.distributed as dist
from typing import List, Tuple, Union, Optional

import pace_cpp
from ..utils import EventOverlap, CommConfig, BaseComm
import copy

class RSComm(BaseComm):

    def __init__(self, group : dist.ProcessGroup, num_local_ranks : int, config : CommConfig = None, **kwargs):
        """
            Initialize the RSComm object.
            Args:
                group: the process group to exchange information.
                num_local_ranks: the number of local ranks in a node.
        """
        if config is None:
            self.config = self.get_recommended_config(num_local_ranks, group.size())
        else:
            self.config = copy.deepcopy(config)
        super().__init__(group, num_local_ranks)

    def _build_runtime(self, topo_key):
        c = self.config
        return pace_cpp.RSComm(self.rank, self.num_ranks, self.num_local_ranks, c.slot_unroll, c.nvl_ring_size, c.rdma_ring_size, c.num_sms, c.use_wg, topo_key)

    @staticmethod
    def get_recommended_config(num_local_ranks: int, num_ranks: int) -> CommConfig:
        """
        Get recommended RSComm config.
        """
        if num_local_ranks == 1:
            if num_local_ranks != num_ranks:
                return CommConfig(64, 2, 2, 8)
            else:
                return CommConfig(64, 4, 0, 32)
        else:
            if num_local_ranks != num_ranks:
                # Multi-node: keep prior tuning for the legacy path; ring path
                # has not been multi-node-tuned yet.
                return CommConfig(24, 2, 2, 32)
            else:
                return CommConfig(slot_unroll=128, nvl_ring_size=4, rdma_ring_size=0, num_sms=32)
    
    def out_numel_alignment(self) -> int:
        return self.runtime.out_numel_alignment()

    def reduce_scatter(self, tensor: Union[torch.Tensor, List[torch.Tensor]], red_op : Union[dist.ReduceOp.RedOpType, dist.ReduceOp] = dist.ReduceOp.SUM, extra_mul : float = 1.0, extra_post_mul : float = 1.0, out_dtype : Optional[torch.dtype] = None, out: Optional[torch.Tensor] = None, previous_event: Optional[EventOverlap] = None, async_finish: bool = False):
        """
        Perform reduce-scatter over the first dimension of the input tensor(s).

        Input:
            tensor: the tensor(s) to be reduce-scattered. A single tensor or a list of tensors; each must be contiguous.
            red_op: the reduction op — SUM, AVG, or PREMUL_SUM.
            extra_mul: extra scalar applied before reduction (folded into the PREMUL_SUM factor). default 1.0 (no-op).
            extra_post_mul: extra scalar applied after reduction. default 1.0 (no-op).
            previous_event: the previous event to be waited for. If None, it will capture newest event on the current stream.
            async_finish: whether to make current stream await PACE_RS stream immediately. If not, function will return an event, which can be called when you need to make current stream await PACE_RS stream.
            out_dtype: output dtype. None or torch.float32 -> float32 output; torch.bfloat16 -> bfloat16 output.
            out: optional external output buffer. If provided, must be a 1-D contiguous CUDA tensor whose dtype matches out_dtype and whose numel equals the total output element count computed by the C++ side (aligned + non-aligned, each chunk padded to out_numel_alignment()). It is written in-place and returned as result.

        Output:
            result (torch.Tensor): the reduce-scattered result (`out` if provided, else newly allocated).
            event (Optional[EventOverlap]): the completion event if async_finish is True, else None.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = [tensor]
        elif isinstance(tensor, list):
            assert len(tensor) > 0
            for t in tensor:
                assert isinstance(t, torch.Tensor)
        else:
            raise Exception('input ilegal, must be torch.Tensor or List [torch.Tensor]')
        for t in tensor:
            assert t.is_contiguous(), 'RSComm requires contiguous input tensors'
        red_op_type = red_op if isinstance(red_op, dist.ReduceOp.RedOpType) else red_op.op
        assert red_op_type in [dist.ReduceOp.SUM, dist.ReduceOp.AVG, dist.ReduceOp.PREMUL_SUM], f'RSComm support sum, avg, and premul_sum, but get {red_op}'
        red_op_premul = 1.0
        if red_op_type == dist.ReduceOp.PREMUL_SUM:
            _, pf = red_op.__getstate__()
            red_op_premul = float(pf)
        extra_mul *= red_op_premul
        red_int = 1 if red_op_type in [dist.ReduceOp.SUM, dist.ReduceOp.PREMUL_SUM] else 2
        assert out_dtype in (None, torch.float32, torch.bfloat16), f'RSComm out_dtype must be None, torch.float32, or torch.bfloat16, but got {out_dtype}'
        out_type_int = 1 if out_dtype == torch.bfloat16 else 0 # 0 means no cast, just write float
        result, event = self.runtime.reduce_scatter(tensor, red_int, extra_mul, extra_post_mul, out_type_int, out, getattr(previous_event, 'event', None), async_finish)
        # Wrap the raw event as EventOverlap (feedable as previous_event) and keep the
        # input/output tensors alive until it is observed (async safety; replaces the
        # old separate ref_handle return value -> uniform (result, event)).
        evt = EventOverlap(event, extra_tensors=(list(tensor), result)) if event is not None else None
        return result, evt

    def get_call_times(self, tensors : Union[torch.Tensor, List[torch.Tensor]]) -> Tuple [bool, bool]:
        """
            Get the rs call times for the input
            Args:
                tensors: the tensors to be checked
            Returns:
                aligned: whether there is an aligned call
                nonaligned: whether there is an nonaligned call
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        alignment = self.out_numel_alignment()
        # Classify tensors by alignment (matching C++ rscomm.cpp logic)
        aligned_idx, nonaligned_idx = [], []
        for i, t in enumerate(tensors):
            chunk_el = t.numel() // t.size(0)
            ptr_align = 16 if t.dtype == torch.float32 else 8  # sizeof(float4) / sizeof(float2)
            if (t.data_ptr() % ptr_align == 0) and (chunk_el % 4 == 0):
                aligned_idx.append(i)
            else:
                nonaligned_idx.append(i)
        return len(aligned_idx) > 0, len(nonaligned_idx) > 0

    def get_split_tensors(self, tensors : Union[torch.Tensor, List[torch.Tensor]], output: torch.Tensor, tight : bool = False):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        alignment = self.out_numel_alignment()
        # Classify tensors by alignment (matching C++ rscomm.cpp logic)
        aligned_idx, nonaligned_idx = [], []
        for i, t in enumerate(tensors):
            chunk_el = t.numel() // t.size(0)
            ptr_align = 16 if t.dtype == torch.float32 else 8  # sizeof(float4) / sizeof(float2)
            if (t.data_ptr() % ptr_align == 0) and (chunk_el % 4 == 0):
                aligned_idx.append(i)
            else:
                nonaligned_idx.append(i)
        # Compute offsets: aligned first, then non-aligned (matching C++ output layout)
        outs = [None] * len(tensors)
        cnt = 0
        for idx in aligned_idx + nonaligned_idx:
            t = tensors[idx]
            dim0_chunks = (t.size(0) + self.num_ranks - 1) // self.num_ranks
            numel = dim0_chunks * (t.numel() // t.size(0))
            outs[idx] = output[cnt:cnt + numel]
            if not tight:
                cnt += (numel + alignment - 1) // alignment * alignment
            else:
                cnt += numel
        return outs