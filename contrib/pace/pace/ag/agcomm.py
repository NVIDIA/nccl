import torch
import torch.distributed as dist
from typing import List, Tuple, Union, Optional

import pace_cpp
from ..utils import EventOverlap, CommConfig, BaseComm
import copy

class AGComm(BaseComm):

    def __init__(self, group : dist.ProcessGroup, num_local_ranks : int, config : CommConfig = None):
        """
            Initialize the AGComm object.
            Args:
                group: the process group to exchange information.
                num_local_ranks: the number of local ranks in a node.
                config: optional CommConfig; if None, a recommended config is chosen.
        """
        if config is None:
            self.config = self.get_recommended_config(num_local_ranks, group.size())
        else:
            self.config = copy.deepcopy(config)
        super().__init__(group, num_local_ranks)

    def _build_runtime(self, topo_key):
        c = self.config
        # The 0SM AG path aliases the local send buffer with recv_area[node_id]
        # using a single ring-depth dimension, so nvl_ring and rdma_ring must
        # match. Equalize to the max here so callers can keep passing whichever
        # they have (or different values via CommConfig) without hitting the
        # mem_layout mismatch. The C++ AGComm ctor also takes max defensively.
        ring = max(c.nvl_ring_size, c.rdma_ring_size)
        z = c.ag_zero_sm
        return pace_cpp.AGComm(self.rank, self.num_ranks, self.num_local_ranks, c.slot_unroll, ring, ring, c.num_sms, z.use_ring, z.use_graph, z.force_graph_capture, z.min_graph_nodes, topo_key)

    @staticmethod
    def get_recommended_config(num_local_ranks: int, num_ranks: int, num_sms: int = 0) -> CommConfig:
        """
        Get recommended AGComm config.
        """
        if num_sms == 0:
            if num_local_ranks == 1:
                return CommConfig(1024, 4, 4, 0)
            elif num_local_ranks != num_ranks:
                return CommConfig(512, 2, 2, 0)
            else:
                return CommConfig(8192, 2, 0, 0)
        else:
            return CommConfig(48, 4, 4, num_sms)
    def all_gather(self, tensor: Union[torch.Tensor, List[torch.Tensor]], out: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None, input_dtype_mapping: Optional[Tuple[torch.dtype, torch.dtype]] = None, previous_event: Optional[EventOverlap] = None, async_finish: bool = False):
        """
        Perform all-gather on tensor(s)

        Input:
            tensor: the tensor(s) to be all-gathered. A single tensor or a list of tensors; each must be contiguous.
            out: optional output tensor(s). If provided, the all-gather results will be written to these tensors.
                    - If tensor is a single Tensor, out can be a single Tensor or a list with one Tensor.
                    - If tensor is a list, out must be a list with matching length.
                    Each output tensor should have size = input_tensor.numel() * num_ranks.
                    If None, a new combined tensor will be allocated internally.
            previous_event: the previous event to be waited for. If None, it will capture newest event on the current stream.
            async_finish: whether to make current stream await PACE_AG stream immediately. If not, function will return an event, which can be called when you need to make current stream await PACE_AG stream.
            input_dtype_mapping: optional (storage_dtype, comm_dtype) pair to cast on the wire. Only (torch.float32, torch.bfloat16) is supported and only when config.num_sms > 0; None means no cast.

        Output:
            result: If `out` is provided, returns it (tensor or list). Otherwise returns the internally allocated result tensor.
            event (Optional[EventOverlap]): the completion event if async_finish is True, else None.
        """

        storage_type, comm_type = None, None
        if input_dtype_mapping is not None:
            storage_type, comm_type = input_dtype_mapping
        
        assert (storage_type == None) == (comm_type == None)
        if storage_type == comm_type:
            storage_type = None
            comm_type = None
        
        if storage_type is not None:
            assert self.config.num_sms > 0, "Mapping can only be used with num_sms > 0"
            assert storage_type in [torch.float32], f'storage_type ({storage_type}) must be in [torch.float32]'
            assert comm_type in [torch.float32, torch.bfloat16], f'comm_type ({comm_type}) must be in [torch.float32, torch.bfloat16]'
        
        assert isinstance(tensor, list) or isinstance(tensor, torch.Tensor)
        list_input = isinstance(tensor, list)
        if list_input:
            for t in tensor:
                assert isinstance(t, torch.Tensor)
        if not list_input:
            tensor = [tensor]
        for t in tensor:
            assert t.is_contiguous(), 'AGComm requires contiguous input tensors'

        if storage_type is not None:
            for t in tensor:
                assert t.dtype == storage_type, f"tensor dtype ({t.dtype}) must match mapping requested dtype ({storage_type})"
        
        # Validate and normalize output tensors if provided
        output_list = None  # The list form passed to C++
        single_output = False  # Track if user provided a single tensor
        if out is not None:
            if isinstance(out, torch.Tensor):
                # Single tensor output - only valid when input is also single
                assert not list_input, "out must be a list when input is a list of tensors"
                output_list = [out]
                single_output = True
            else:
                assert isinstance(out, list), "out must be a Tensor or a list of tensors"
                output_list = out
            
            assert len(output_list) == len(tensor), f"output list length ({len(output_list)}) must match input tensor count ({len(tensor)})"
            for i, (inp, o) in enumerate(zip(tensor, output_list)):
                expected_numel = inp.numel() * self.num_ranks
                assert o.numel() == expected_numel, f"output[{i}] has {o.numel()} elements, expected {expected_numel}"
                if comm_type is not None:
                    assert o.dtype == comm_type, f"output[{i}] dtype ({o.dtype}) must match mapping requested dtype ({comm_type})"
                else:
                    assert o.dtype == inp.dtype, f"output[{i}] dtype ({o.dtype}) must match input dtype ({inp.dtype})"
        imapping_int = 0
        if storage_type == torch.float32 and comm_type == torch.bfloat16:
            imapping_int = 1
        result, event = self.runtime.all_gather(tensor, output_list, getattr(previous_event, 'event', None), async_finish, imapping_int)
        evt = EventOverlap(event) if event is not None else None
        # If `out` was provided, return the user's buffer; otherwise the allocated result
        return (out if out is not None else result), evt

    def get_split_tensors(self, tensors : Union[torch.Tensor, List[torch.Tensor]], output: torch.Tensor):
        """
        Split output tensor into per-tensor, per-rank slices.
            
        Output layout (tight, no padding): [tensor0_allgather | tensor1_allgather | ...]
        Each tensor_i_allgather = [rank0_data | rank1_data | ... | rankN_data]
            
        Returns:
            outs: List[List[Tensor]] where outs[i][j] is tensor i's data from rank j
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
            
        # Get the number of elements for each tensor
        tensor_numels = [t.numel() for t in tensors]
            
        # Calculate cumulative offsets for each tensor in output (in elements)
        tensor_offsets = [0]
        offset = 0
        for numel in tensor_numels:
            offset += numel * self.num_ranks
            tensor_offsets.append(offset)
            
        # outs[i][j] = tensor i's data from rank j
        outs = [[output[tensor_offsets[i] + j * tensor_numels[i]:tensor_offsets[i] + j * tensor_numels[i] + tensor_numels[i]] 
                 for j in range(self.num_ranks)] 
                for i in range(len(tensors))]
        return outs
    
    def get_gathered_tensors(self, tensors : Union[torch.Tensor, List[torch.Tensor]], output: torch.Tensor):
        """
        Get complete allgather result for each tensor (keeping num_ranks data contiguous).
            
        Output layout (tight, no padding): [tensor0_allgather | tensor1_allgather | ...]
        Each tensor_i_allgather = [rank0_data | rank1_data | ... | rankN_data]
            
        Returns:
            outs: List[Tensor] where outs[i] is tensor i's complete allgather result
        """
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
            
        # Get the number of elements for each tensor
        tensor_numels = [t.numel() for t in tensors]
            
        # Calculate cumulative offsets for each tensor in output (in elements)
        tensor_offsets = [0]
        offset = 0
        for numel in tensor_numels:
            offset += numel * self.num_ranks
            tensor_offsets.append(offset)
            
        # outs[i] = tensor i's complete allgather result (numel * num_ranks elements)
        outs = [output[tensor_offsets[i]:tensor_offsets[i + 1]] for i in range(len(tensors))]
        return outs
    
    def get_comm_slots(self, tensors : Union[torch.Tensor, List[torch.Tensor]]):
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        tsizes = [t.numel() * t.element_size() for t in tensors]
        align = 16 # int4 aligned
        aligned_sizes = [(s + align - 1) & ~(align - 1) for s in tsizes]
        total_size = sum(aligned_sizes)
        slot_bytes = align * 1024 * self.config.slot_unroll
        return (total_size + slot_bytes - 1) // slot_bytes