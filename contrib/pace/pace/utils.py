import torch
import torch.distributed as dist
from typing import Any, Optional, Tuple, Union
from dataclasses import dataclass, field

# noinspection PyUnresolvedReferences
import pace_cpp
from pace_cpp import EventHandle


def compute_topo_key(group: dist.ProcessGroup):
    """
    Topology fingerprint for a ProcessGroup: the sorted tuple of global ranks
    of all members. Two groups with the same fingerprint can share an
    underlying ncclComm. Returns a list[int] (pybind11-friendly).
    """
    return sorted(dist.get_global_rank(group, i) for i in range(group.size()))


def bootstrap_runtime(runtime, group: dist.ProcessGroup, topo_key):
    """
    Generic NCCL-uniqueId broadcast + Connect.

    If a previously-built ncclComm for this `topo_key` is already cached on
    the C++ side, skip the broadcast entirely (the group may even have been
    destroyed already) and pass an empty head_info to Connect — the C++
    side will reuse the cached comm and ignore head_info.

    The trailing barrier is also skipped on cache hit: ncclCommWindowRegister
    and ncclDevCommCreate (called inside Connect) are themselves collective
    on the shared ncclComm, so cross-rank synchronization is preserved
    without touching `group`.
    """
    if pace_cpp.has_cached_comm(topo_key):
        runtime.connect(bytearray())
        return

    # The head_info is a fixed-size ncclUniqueId byte blob. Broadcast it as a
    # raw tensor instead of dist.broadcast_object_list: on some torch builds
    # (e.g. 2.7.0a0 with device_id passed to init_process_group),
    # broadcast_object_list fails on the receiving rank with
    # "EOFError: Ran out of input". Use CUDA tensors for the NCCL backend
    # (required) and CPU for others (e.g. gloo).
    head = bytearray()
    if group.rank() == 0:
        head = runtime.get_head_info()
    backend = dist.get_backend(group)
    dev = 'cpu' if backend == 'gloo' else torch.device('cuda', torch.cuda.current_device())
    size_t = torch.tensor([len(head)], dtype=torch.long, device=dev)
    dist.broadcast(size_t, src=0, group=group)
    buf = torch.zeros(int(size_t.item()), dtype=torch.uint8, device=dev)
    if group.rank() == 0:
        buf.copy_(torch.frombuffer(bytearray(bytes(head)), dtype=torch.uint8))
    dist.broadcast(buf, src=0, group=group)
    runtime.connect(bytearray(buf.cpu().numpy().tobytes()))
    dist.barrier(group=group)


class EventOverlap:
    """
    A wrapper class to manage CUDA events, also for better overlapping convenience.

    Attributes:
        event: the CUDA event captured.
        extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
    """

    def __init__(self, event: Optional[EventHandle] = None,
                 extra_tensors: Optional[Union[Tuple[torch.Tensor], torch.Tensor]] = None) -> None:
        """
        Initialize the class.

        Arguments:
            event: the CUDA event captured.
            extra_tensors: an easier way to simulate PyTorch tensor `record_stream`, may be useful with CUDA graph.
        """
        self.event = event

        # NOTES: we use extra tensors to achieve stream recording, otherwise,
        # stream recording will be incompatible with CUDA graph.
        self.extra_tensors = extra_tensors

    def current_stream_wait(self) -> None:
        """
        The current stream `torch.cuda.current_stream()` waits for the event to be finished.
        """
        assert self.event is not None
        self.event.current_stream_wait()
        self.extra_tensors = None

    def wait(self, stream : torch.cuda.Stream = None):
        if stream is None:
            self.current_stream_wait()
        else:
            self.event.wait(stream.cuda_stream)
            self.extra_tensors = None

    def host_wait(self) -> None:
        """
        Let CPU waits for the event to be finished.
        """
        assert self.event is not None
        self.event.host_wait()
        self.extra_tensors = None
    
    def host_query(self) -> bool:
        assert self.event is not None
        sync = self.event.host_query()
        if sync:
            self.extra_tensors = None
        return sync

    def __enter__(self) -> Any:
        """
        Utility for overlapping and Python `with` syntax.

        You can overlap the kernels on the current stream with the following example:
        ```python
        event_overlap = event_after_all_to_all_kernels()
        with event_overlap:
            do_something_on_current_stream()
        # After exiting the `with` scope, the current stream with wait the event to be finished.
        ```
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Utility for overlapping and Python `with` syntax.

        Please follow the example in the `__enter__` function.
        """
        if self.event is not None:
            self.event.current_stream_wait()
        self.extra_tensors = None

@dataclass
class AGZeroSMConfig:
    """Knobs for the 0-SM (``num_sms == 0``) All-Gather cord path.

    These used to be process-wide environment variables
    (``PACE_AG_FORCE_RING`` / ``GIN_AG_USE_GRAPH`` / ``GIN_AG_FORCE_CAPTURE`` /
    ``GIN_AG_MIN_NODES``); they are now per-comm config fields.
    """

    # Inter-node data shape: True = ring-forward to the next rail peer; False =
    # broadcast to every other node's rail peer. Only meaningful for num_sms==0
    # multi-node; single-node always behaves as broadcast (no RDMA peers).
    use_ring: bool = True
    # Capture the 0-SM AG into a CUDA graph (the c1/c2/c3 copy schedule, plus a
    # separate cord-kernel graph for multi-node). Off by default → direct
    # submission each call.
    use_graph: bool = False
    # Debug: re-capture + re-instantiate the graph on every call instead of
    # using the cache (GIN_AG_FORCE_CAPTURE).
    force_graph_capture: bool = False
    # Minimum estimated graph-node count before graph mode is used. Tiny ops
    # (few memcpys) pay the capture+instantiate tax without recouping it from
    # cached launches, so they fall back to direct submission.
    min_graph_nodes: int = 32


@dataclass
class CommConfig:
    """Tuning knobs shared by all four collectives (one dataclass, no subclasses).

    Fields not relevant to a given collective are ignored by it (e.g.
    ``ag_zero_sm`` is AG-only, ``use_wg`` is RS-only). SG always uses the
    data-ring."""

    slot_unroll: int = 16
    nvl_ring_size: int = 4
    rdma_ring_size: int = 4
    num_sms: int = 32
    # AG-only: 0-SM (num_sms == 0) cord-path knobs (use_ring, CUDA-graph mode).
    ag_zero_sm: AGZeroSMConfig = field(default_factory=AGZeroSMConfig)
    # Hint for reduce-scatter to use warp-group primitives, which can be more
    # efficient for small messages.
    use_wg: bool = False


class BaseComm:
    """
    Common base for RSComm / SGComm / AGComm.

    Handles the boilerplate that's identical across all three:
      - record (rank, num_ranks, num_local_ranks)
      - compute the topology fingerprint
      - build the C++ runtime via the subclass-provided _build_runtime hook
      - bootstrap (uniqueId broadcast + Connect, or skip on cache hit)
      - destroy / capture / get_stream

    Subclasses must implement `_build_runtime(self, topo_key) -> runtime`.
    By the time `_build_runtime` is called, `self.rank / num_ranks /
    num_local_ranks` plus any subclass-specific fields set BEFORE
    `super().__init__(...)` are available.
    """

    def __init__(self, group: dist.ProcessGroup, num_local_ranks: int):
        self.rank = group.rank()
        self.num_ranks = group.size()
        self.num_local_ranks = num_local_ranks
        assert self.num_ranks % self.num_local_ranks == 0
        topo_key = compute_topo_key(group)
        self.runtime = self._build_runtime(topo_key)
        bootstrap_runtime(self.runtime, group, topo_key)
        torch.cuda.synchronize()

    def _build_runtime(self, topo_key):
        raise NotImplementedError

    def destroy(self):
        if self.runtime is not None:
            self.runtime.destroy()
            self.runtime = None

    @staticmethod
    def capture() -> EventOverlap:
        """Capture a CUDA event on `torch.cuda.current_stream()`."""
        return EventOverlap(EventHandle())

    def get_stream(self) -> torch.Stream:
        """Return the comm's internal stream as a torch.cuda.Stream."""
        stream_id, device_index, device_type = self.runtime.get_stream()
        return torch.cuda.Stream(stream_id=stream_id, device_index=device_index, device_type=device_type)
