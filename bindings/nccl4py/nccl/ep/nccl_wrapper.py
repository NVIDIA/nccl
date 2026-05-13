# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python ctypes wrapper for NCCL EP extensions.

This module provides a zero-compilation interface to NCCL EP by loading the
EP shared library and wrapping its EP functions with ctypes. Base NCCL
operations are provided by :mod:`nccl.core`.
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from typing import Any

import nccl.core as nccl

_PACKAGE_NCCL_EP_LIBRARY = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "lib", "libnccl_ep.so")
)


def _find_nccl_ep_library() -> str:
    """Resolve libnccl_ep.so.

    Mirrors :func:`cuda.pathfinder.load_nvidia_dynamic_lib`'s search precedence,
    with the NVIDIA pip-wheel step replaced by the nccl4py package's bundled
    location (which is where libnccl_ep.so lives if a complete cu13 nccl4py
    wheel was installed).

    Search order:
      1. nccl4py package path (``nccl/ep/lib/libnccl_ep.so``).
      2. ``$CONDA_PREFIX/lib`` and ``$CONDA_PREFIX/lib64``.
      3. Dynamic linker default search (``LD_LIBRARY_PATH``, ``ld.so.cache``,
         standard system paths) via :func:`ctypes.util.find_library`. This also
         covers the "already loaded into the process" case naturally.
      4. ``$CUDA_HOME`` / ``$CUDA_PATH`` ``lib`` and ``lib64`` subdirectories.
      5. SONAME fallback so ``ctypes.CDLL`` does a final ``dlopen`` attempt.
    """
    # 1. nccl4py package (replaces cuda.pathfinder's NVIDIA-pip-wheel step)
    if os.path.exists(_PACKAGE_NCCL_EP_LIBRARY):
        return _PACKAGE_NCCL_EP_LIBRARY

    # 2. Conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        for sub in ("lib", "lib64"):
            candidate = os.path.join(conda_prefix, sub, "libnccl_ep.so")
            if os.path.exists(candidate):
                return candidate

    # 3. Dynamic linker default search (LD_LIBRARY_PATH, ld.so.cache, /lib, ...)
    found = ctypes.util.find_library("nccl_ep")
    if found:
        return found

    # 4. CUDA_HOME / CUDA_PATH
    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env_var)
        if root:
            for sub in ("lib", "lib64"):
                candidate = os.path.join(root, sub, "libnccl_ep.so")
                if os.path.exists(candidate):
                    return candidate

    # 5. SONAME fallback — let dlopen do its own system search; if it fails,
    # ctypes.CDLL raises a clear OSError that the caller wraps as ImportError.
    return "libnccl_ep.so"


_funcs: dict[str, Any] = {}


def _load_nccl_ep_library() -> None:
    """Resolve, dlopen, and bind libnccl_ep.so symbols.

    Called once from :mod:`nccl.ep` at import time. Populates the module-level
    ``_funcs`` table that :class:`NCCLLibrary` instances read from.
    """
    global _funcs
    lib = ctypes.CDLL(_find_nccl_ep_library())
    bound: dict[str, Any] = {}
    for func in NCCLLibrary.exported_functions:
        try:
            f = getattr(lib, func.name)
        except AttributeError as e:
            raise RuntimeError(f"{func.name} is not exported by libnccl_ep.so") from e
        f.restype = func.restype
        f.argtypes = func.argtypes
        bound[func.name] = f
    _funcs = bound


# Optional torch import for communicator creation
try:
    import torch
    import torch.distributed as dist
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None
    dist = None

# Type Definitions
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p

# EP-specific types (opaque pointers)
ncclEpGroup_t = ctypes.c_void_p
ncclEpHandle_t = ctypes.c_void_p

# Allocator callback function types:
# typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size);
# typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr);
ncclEpAllocFn_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
ncclEpFreeFn_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

# cudaError_t values
CUDA_SUCCESS = 0
CUDA_ERROR_MEMORY_ALLOCATION = 2


class ncclEpAlgorithm_t:
    NCCL_EP_ALGO_LOW_LATENCY = 0
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1


class ncclEpLayout_t:
    """Receive buffer layout for dispatch/combine."""
    NCCL_EP_LAYOUT_AUTO = 0
    NCCL_EP_LAYOUT_EXPERT_MAJOR = 1
    NCCL_EP_LAYOUT_RANK_MAJOR = 2
    NCCL_EP_LAYOUT_FLAT = 3


# ncclNDTensor_t is an opaque pointer type
ncclNDTensor_t = ctypes.c_void_p


# Base class for cross-boundary EP structs. The C library expects every such
# struct to start with `size = sizeof(struct)`; this base class auto-fills it
# in __init__ so Python callers never have to set it manually.
class _EpStruct(ctypes.Structure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = ctypes.sizeof(type(self))


class ncclEpGroupConfig_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("algorithm", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("num_experts", ctypes.c_uint),
        ("max_send_tokens_per_rank", ctypes.c_uint),
        ("token_size_bytes", ctypes.c_uint),
        ("rdma_buffer_size", ctypes.c_ulong),
        ("num_qp_per_rank", ctypes.c_uint),
        ("num_channels", ctypes.c_uint),
        ("max_recv_token_slots_per_rank", ctypes.c_uint),
    ]


class ncclEpHandleConfig_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("use_fp8", ctypes.c_bool),
        ("dispatch_output_per_expert_alignment", ctypes.c_size_t),
    ]


class ncclEpDispatchConfig_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("send_only", ctypes.c_uint),
        ("round_scales", ctypes.c_uint),
    ]


class ncclEpCombineConfig_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("send_only", ctypes.c_uint),
    ]


class ncclEpLayoutMarks_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("recv_expert_counter", ncclNDTensor_t),
        ("src_rank_counter", ncclNDTensor_t),
        ("recv_expert_offsets", ncclNDTensor_t),
        ("recv_total_counter", ncclNDTensor_t),
    ]


class ncclEpDispatchInputs_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("tokens", ncclNDTensor_t),
        ("topk_weights", ncclNDTensor_t),
        ("scales", ncclNDTensor_t),
    ]


class ncclEpDispatchOutputs_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("tokens", ncclNDTensor_t),
        ("topk_weights", ncclNDTensor_t),
        ("scales", ncclNDTensor_t),
        ("topk_idx", ncclNDTensor_t),
    ]


class ncclEpCombineInputs_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("tokens", ncclNDTensor_t),
        ("topk_weights", ncclNDTensor_t),
    ]


class ncclEpCombineOutputs_t(_EpStruct):
    _fields_ = [
        ("size", ctypes.c_uint),
        ("tokens", ncclNDTensor_t),
        ("topk_weights", ncclNDTensor_t),
    ]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class NCCLLibrary:
    """ctypes wrapper for the NCCL EP extension library."""

    exported_functions = [
        Function("ncclEpCreateGroup", ncclResult_t, [
            ctypes.POINTER(ncclEpGroup_t), ncclComm_t,
            ctypes.POINTER(ncclEpGroupConfig_t),
            ncclEpAllocFn_t, ncclEpFreeFn_t
        ]),
        Function("ncclEpGroupDestroy", ncclResult_t, [ncclEpGroup_t]),
        Function("ncclEpCreateHandle", ncclResult_t, [
            ctypes.POINTER(ncclEpHandle_t), ncclEpGroup_t,
            ncclNDTensor_t,  # topk_idx (opaque handle)
            ctypes.POINTER(ncclEpLayoutMarks_t),  # marks (NULL = none)
            ctypes.POINTER(ncclEpHandleConfig_t),  # NULL = defaults
            cudaStream_t,
        ]),
        Function("ncclEpHandleDestroy", ncclResult_t, [ncclEpHandle_t]),
        Function("ncclEpUpdateHandle", ncclResult_t, [
            ncclEpHandle_t,
            ncclNDTensor_t,  # topk_idx
            ctypes.POINTER(ncclEpLayoutMarks_t),  # marks (NULL = none)
            cudaStream_t
        ]),
        Function("ncclEpDispatch", ncclResult_t, [
            ncclEpHandle_t,
            ncclNDTensor_t,  # topk_idx (top-level arg; NULL for LL/HT-backward)
            ctypes.POINTER(ncclEpDispatchInputs_t),
            ctypes.POINTER(ncclEpDispatchOutputs_t),
            ctypes.POINTER(ncclEpLayoutMarks_t),  # marks (NULL = none)
            ctypes.POINTER(ncclEpDispatchConfig_t),  # NULL = defaults
            cudaStream_t
        ]),
        Function("ncclEpCombine", ncclResult_t, [
            ncclEpHandle_t,
            ctypes.POINTER(ncclEpCombineInputs_t),
            ctypes.POINTER(ncclEpCombineOutputs_t),
            ctypes.POINTER(ncclEpCombineConfig_t),  # NULL = defaults
            cudaStream_t
        ]),
        Function("ncclEpComplete", ncclResult_t, [
            ncclEpHandle_t, ctypes.c_void_p, cudaStream_t
        ]),
        Function("ncclEpTensorCreate", ncclResult_t, [
            ctypes.POINTER(ncclNDTensor_t),  # OUT tensor handle
            ctypes.c_uint,  # ndim
            ctypes.c_int,   # datatype
            ctypes.c_void_p,  # data (caller-owned device pointer; must be non-null)
            ctypes.c_uint,  # size0
            ctypes.c_uint,  # size1
            ctypes.c_uint,  # size2
            ctypes.c_uint,  # size3
            ctypes.c_uint,  # size4
        ]),
        Function("ncclEpTensorDestroy", ncclResult_t, [
            ncclNDTensor_t,  # tensor handle
        ]),
        Function("ncclEpTensorGetData", ncclResult_t, [
            ncclNDTensor_t,  # tensor handle
            ctypes.POINTER(ctypes.c_void_p),  # OUT data pointer
        ]),
        Function("ncclEpTensorGetSizes", ncclResult_t, [
            ncclNDTensor_t,  # tensor handle
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint)),  # OUT sizes
            ctypes.POINTER(ctypes.c_uint),  # OUT ndim
        ]),
    ]

    def __init__(self) -> None:
        if not _funcs:
            raise RuntimeError(
                "libnccl_ep.so has not been loaded; ensure `import nccl.ep` completed without errors"
            )
        self._funcs = _funcs

    def NCCL_CHECK(self, result):
        if result != 0:
            raise RuntimeError(f"NCCL error: {nccl.get_error_string(result)}")

    def ncclEpCreateGroup(self, comm, config, alloc_fn=None, free_fn=None):
        """Create NCCL EP group for distributed EP operations.

        Args:
            comm: NCCL communicator
            config: EP group configuration (ncclEpGroupConfig_t)
            alloc_fn: Optional custom allocator callback (ncclEpAllocFn_t).
                     If None, uses cudaMalloc/cudaFree.
            free_fn: Optional custom free callback (ncclEpFreeFn_t).
                    If None, uses cudaFree.

        Returns:
            ncclEpGroup_t: Opaque handle to the EP group
        """
        ep_group = ncclEpGroup_t()

        # Convert None to NULL callbacks
        alloc_callback = alloc_fn if alloc_fn is not None else ctypes.cast(None, ncclEpAllocFn_t)
        free_callback = free_fn if free_fn is not None else ctypes.cast(None, ncclEpFreeFn_t)

        self.NCCL_CHECK(self._funcs["ncclEpCreateGroup"](
            ctypes.byref(ep_group), comm.ptr, ctypes.byref(config),
            alloc_callback, free_callback
        ))
        return ep_group

    def ncclEpGroupDestroy(self, ep_group):
        self.NCCL_CHECK(self._funcs["ncclEpGroupDestroy"](ep_group))

    def ncclEpCreateHandle(self, ep_group, topk_tensor, config, stream, marks=None):
        """Create EP handle for a specific dispatch/combine operation.

        This triggers the notify_dispatch phase in HT mode, computing token distribution.

        Args:
            ep_group: NCCL EP group handle
            topk_tensor: ncclNDTensor_t with topk indices
            config: ncclEpHandleConfig_t configuration (or None for defaults).
                    Set config.use_fp8 = True to enable FP8 dispatch.
            stream: CUDA stream
            marks: Optional ncclEpLayoutMarks_t. HT mode: set marks.recv_expert_counter
                   when max_send_tokens_per_rank=NCCL_EP_AUTO; LL: must be None.

        Returns:
            ncclEpHandle_t: Handle for dispatch/combine operations
        """
        handle = ncclEpHandle_t()
        config_ptr = ctypes.byref(config) if config else None
        marks_ptr  = ctypes.byref(marks)  if marks  else None

        self.NCCL_CHECK(self._funcs["ncclEpCreateHandle"](
            ctypes.byref(handle), ep_group, topk_tensor,
            marks_ptr, config_ptr, stream
        ))
        return handle

    def ncclEpHandleDestroy(self, handle):
        self.NCCL_CHECK(self._funcs["ncclEpHandleDestroy"](handle))

    def ncclEpUpdateHandle(self, handle, topk_tensor, stream, marks=None):
        """Rebind topk_idx on an existing handle without reallocating buffers."""
        marks_ptr = ctypes.byref(marks) if marks else None
        self.NCCL_CHECK(self._funcs["ncclEpUpdateHandle"](
            handle, topk_tensor, marks_ptr, stream
        ))

    def ncclEpDispatch(self, handle, topk_idx, inputs, outputs, marks, config, stream):
        """Perform EP dispatch with named-struct args.

        Args:
            handle: ncclEpHandle_t
            topk_idx: ncclNDTensor_t (HT forward) or None (LL / HT backward)
            inputs: ncclEpDispatchInputs_t
            outputs: ncclEpDispatchOutputs_t
            marks: ncclEpLayoutMarks_t or None
            config: ncclEpDispatchConfig_t or None
            stream: CUDA stream
        """
        config_ptr = ctypes.byref(config) if config else None
        marks_ptr  = ctypes.byref(marks)  if marks  else None
        topk_idx_arg = topk_idx if topk_idx is not None else ctypes.c_void_p(0)
        self.NCCL_CHECK(self._funcs["ncclEpDispatch"](
            handle, topk_idx_arg,
            ctypes.byref(inputs), ctypes.byref(outputs),
            marks_ptr, config_ptr, stream
        ))

    def ncclEpCombine(self, handle, inputs, outputs, config, stream):
        config_ptr = ctypes.byref(config) if config else None
        self.NCCL_CHECK(self._funcs["ncclEpCombine"](
            handle, ctypes.byref(inputs), ctypes.byref(outputs),
            config_ptr, stream
        ))

    def ncclEpComplete(self, handle, config, stream):
        """Complete a staged EP operation.

        This must be called after ncclEpDispatch or ncclEpCombine to complete the operation.
        In HT mode, this is always required regardless of send_only setting.

        Args:
            handle: NCCL EP handle
            config: Complete configuration (must be None/NULL)
            stream: CUDA stream

        Note: Internally calls ncclEpComplete (the C++ library symbol name).
        """
        self.NCCL_CHECK(self._funcs["ncclEpComplete"](handle, None, stream))


def get_nccl_comm_from_group(group=None):
    """Create NCCL communicator for the given ProcessGroup.

    Following vLLM's approach, we always create a new NCCL communicator rather than
    extracting from PyTorch's ProcessGroup (which is fragile and version-dependent).

    Args:
        group: PyTorch distributed process group (or None for default group or MPI-only mode).
              If PyTorch is not available, this is ignored and MPI is used.

    Returns:
        nccl.core.Communicator: NCCL communicator.

    Raises:
        RuntimeError: If NCCL communicator cannot be created
    """
    return _create_nccl_comm_for_group(group)


def _create_nccl_comm_for_group(group):
    """Create NCCL communicator using nccl.core.

    Follows vLLM's approach of always creating a new communicator rather than
    extracting from PyTorch's ProcessGroup.

    Args:
        group: PyTorch distributed process group (or None for MPI-only mode).
              If PyTorch is not available, falls back to MPI mode.

    Returns:
        nccl.core.Communicator: NCCL communicator.
    """
    # Get rank and world size
    rank = None
    world_size = None

    if HAVE_TORCH and group is not None:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
    elif HAVE_TORCH and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    # MPI-only mode or PyTorch not available: read from environment
    if rank is None or world_size is None:
        if 'OMPI_COMM_WORLD_RANK' in os.environ:
            rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
            world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        elif 'MV2_COMM_WORLD_RANK' in os.environ:
            rank = int(os.environ['MV2_COMM_WORLD_RANK'])
            world_size = int(os.environ['MV2_COMM_WORLD_SIZE'])
        elif 'SLURM_PROCID' in os.environ:
            rank = int(os.environ['SLURM_PROCID'])
            world_size = int(os.environ['SLURM_NTASKS'])
        else:
            raise RuntimeError(
                "Cannot determine rank/world_size. Run with mpirun/srun or initialize PyTorch distributed."
            )

    # Get CUDA device (use environment variable if torch not available)
    if HAVE_TORCH:
        device = torch.cuda.current_device()
    else:
        # Use CUDA_VISIBLE_DEVICES or local rank from environment
        import subprocess
        try:
            device = int(os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0])
        except:
            device = 0

    # rank 0 generates a unique ID; other ranks start with an empty placeholder
    # that gets filled in via the broadcast below.
    unique_id = nccl.get_unique_id(empty=(rank != 0))

    if HAVE_TORCH and (group is not None or dist.is_initialized()):
        # PyTorch distributed mode: use PyTorch broadcast
        backend_name = "nccl"
        try:
            backend_name = dist.get_backend(group) if group else dist.get_backend()
        except:
            pass

        # bytearray gives torch.frombuffer a writable buffer for in-place broadcast.
        buf = bytearray(bytes(unique_id))
        tensor = torch.frombuffer(buf, dtype=torch.uint8)
        if backend_name == "nccl":
            tensor = tensor.to(device)

        # Broadcast from appropriate source
        if group is not None:
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(tensor, src=ranks[0], group=group)
        else:
            dist.broadcast(tensor, src=0)

        unique_id = nccl.UniqueId.from_bytes(tensor.cpu().numpy().tobytes())
    else:
        # MPI-only mode: use file-based broadcast
        import time
        import binascii

        temp_file = os.path.join(os.getcwd(), '.nccl_unique_id.tmp')
        ready_file = os.path.join(os.getcwd(), '.nccl_unique_id_ready.tmp')

        if rank == 0:
            # Write unique ID to file
            unique_id_hex = binascii.hexlify(unique_id.as_bytes).decode('ascii')
            temp_write = temp_file + '.write'
            with open(temp_write, 'w') as f:
                f.write(unique_id_hex)
                f.flush()
                os.fsync(f.fileno())
            os.rename(temp_write, temp_file)

            with open(ready_file, 'w') as f:
                f.write('ready')
                f.flush()
                os.fsync(f.fileno())
        else:
            # Read unique ID from file
            start_time = time.time()
            while not os.path.exists(ready_file):
                if time.time() - start_time > 30:
                    raise RuntimeError(f"Rank {rank}: Timeout waiting for unique ID")
                time.sleep(0.1)

            with open(temp_file, 'r') as f:
                unique_id_hex = f.read().strip()

            if len(unique_id_hex) != 256:
                raise RuntimeError(f"Rank {rank}: Invalid unique ID length")

            unique_id = nccl.UniqueId.from_bytes(binascii.unhexlify(unique_id_hex))

        # Simple barrier using files
        barrier_file = os.path.join(os.getcwd(), f'.nccl_barrier_r{rank}.tmp')
        with open(barrier_file, 'w') as f:
            f.write('done')
            f.flush()
            os.fsync(f.fileno())

        start_time = time.time()
        for r in range(world_size):
            bf = os.path.join(os.getcwd(), f'.nccl_barrier_r{r}.tmp')
            while not os.path.exists(bf):
                if time.time() - start_time > 30:
                    break
                time.sleep(0.1)

    # Initialize NCCL communicator
    if HAVE_TORCH:
        with torch.cuda.device(device):
            comm = nccl.Communicator.init(world_size, rank, unique_id)
    else:
        # Without torch, just call directly (assumes CUDA device already set)
        comm = nccl.Communicator.init(world_size, rank, unique_id)

    # Cleanup temp files (best effort) - only in MPI mode
    if not HAVE_TORCH or (group is None and not dist.is_initialized()):
        try:
            os.remove(os.path.join(os.getcwd(), f'.nccl_barrier_r{rank}.tmp'))
            if rank == 0:
                for f in [temp_file, ready_file]:
                    if os.path.exists(f):
                        os.remove(f)
                for r in range(world_size):
                    bf = os.path.join(os.getcwd(), f'.nccl_barrier_r{r}.tmp')
                    if os.path.exists(bf):
                        os.remove(bf)
        except:
            pass

    return comm
