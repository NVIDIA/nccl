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
ncclEpCombineConfig_t = ctypes.c_void_p

# Allocator callback function types:
# typedef cudaError_t (*ncclEpAllocFn_t)(void** ptr, size_t size);
# typedef cudaError_t (*ncclEpFreeFn_t)(void* ptr);
ncclEpAllocFn_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t)
ncclEpFreeFn_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)

# cudaError_t values
CUDA_SUCCESS = 0
CUDA_ERROR_MEMORY_ALLOCATION = 2


class ncclEpTensorFlags_t:
    NCCL_EP_TENSOR_FLAG_NONE = 0


class ncclEpTensorTag_t:
    """Tensor tags for NCCL EP operations."""
    NCCL_EP_TENSOR_TAG_NONE = 0
    NCCL_EP_TENSOR_TAG_TOKENS = 1
    NCCL_EP_TENSOR_TAG_TOPK_IDX = 2
    NCCL_EP_TENSOR_TAG_TOPK_WEIGHTS = 3
    NCCL_EP_TENSOR_TAG_SCALES = 4
    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_DEVICE = 5
    NCCL_EP_TENSOR_TAG_TOKENS_PER_EXPERTS = 7


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


class ncclEpGroupConfig_t(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("algorithm", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("num_experts", ctypes.c_uint),
        ("max_tokens_per_rank", ctypes.c_uint),
        ("token_size_bytes", ctypes.c_uint),
        ("rdma_buffer_size", ctypes.c_ulong),
        ("num_qp_per_rank", ctypes.c_uint),
        ("num_channels", ctypes.c_uint),
    ]


class ncclEpHandleConfig_t(ctypes.Structure):
    _fields_ = [("use_fp8", ctypes.c_bool)]


class ncclEpDispatchConfig_t(ctypes.Structure):
    _fields_ = [("round_scales", ctypes.c_uint)]


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
            ctypes.POINTER(ncclNDTensor_t),  # local_tensors array
            ctypes.c_uint,  # num_local_tensors
            ctypes.POINTER(ncclEpHandleConfig_t),
            cudaStream_t,
            ctypes.c_bool  # use_fp8
        ]),
        Function("ncclEpHandleDestroy", ncclResult_t, [ncclEpHandle_t]),
        Function("ncclEpUpdateHandle", ncclResult_t, [
            ncclEpHandle_t,
            ncclNDTensor_t,  # topk_idx
            ctypes.POINTER(ncclNDTensor_t),  # local_tensors array
            ctypes.c_uint,  # num_local_tensors
            cudaStream_t
        ]),
        Function("ncclEpDispatch", ncclResult_t, [
            ncclEpHandle_t,
            ctypes.POINTER(ncclNDTensor_t), ctypes.c_uint,
            ctypes.POINTER(ncclNDTensor_t), ctypes.c_uint,
            ctypes.POINTER(ncclNDTensor_t), ctypes.c_uint,
            ctypes.c_uint, ctypes.POINTER(ncclEpDispatchConfig_t),
            cudaStream_t
        ]),
        Function("ncclEpCombine", ncclResult_t, [
            ncclEpHandle_t,
            ctypes.POINTER(ncclNDTensor_t), ctypes.c_uint,
            ctypes.POINTER(ncclNDTensor_t), ctypes.c_uint,
            ctypes.POINTER(ncclNDTensor_t), ctypes.c_uint,
            ctypes.c_uint, ctypes.POINTER(ncclEpCombineConfig_t), cudaStream_t
        ]),
        Function("ncclEpHandleGetNumRecvTokens", ncclResult_t, [
            ncclEpHandle_t, ctypes.POINTER(ctypes.c_uint)
        ]),
        Function("ncclEpComplete", ncclResult_t, [
            ncclEpHandle_t, ctypes.c_void_p, cudaStream_t
        ]),
        Function("ncclEpTensorCreate", ncclResult_t, [
            ctypes.POINTER(ncclNDTensor_t),  # OUT tensor handle
            ctypes.c_uint,  # ndim
            ctypes.c_int,   # datatype
            ctypes.c_int,   # tag
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

    def ncclEpCreateHandle(self, ep_group, topk_tensor, config, stream, local_tensors=None, use_fp8=False):
        """Create EP handle for a specific dispatch/combine operation.

        This triggers the notify_dispatch phase in HT mode, computing token distribution.

        Args:
            ep_group: NCCL EP group handle
            topk_tensor: ncclNDTensor_t with topk indices
            config: ncclEpHandleConfig_t configuration (reserved, should be None)
            stream: CUDA stream
            local_tensors: Optional list of ncclNDTensor_t for local operations.
                          HT mode: accepts optional RECV_EXPERT_COUNTER tensor (1D, ncclInt32, size=num_local_experts)
                          with tag RECV_EXPERT_COUNTER_DEVICE.
                          Required when max_tokens_per_rank=NCCL_EP_AUTO (0).
                          LL mode: does not accept local tensors (must be None or empty list).
            use_fp8: Enable FP8 for dispatch (default: False)

        Returns:
            ncclEpHandle_t: Handle for dispatch/combine operations
        """
        handle = ncclEpHandle_t()
        config_ptr = ctypes.byref(config) if config else None

        # Prepare local_tensors array
        if local_tensors is None or len(local_tensors) == 0:
            local_tensors_ptr = None
            num_local_tensors = 0
        else:
            # Create array of opaque tensor handles
            tensor_arr = (ncclNDTensor_t * len(local_tensors))()
            for i, tensor in enumerate(local_tensors):
                tensor_arr[i] = tensor
            local_tensors_ptr = ctypes.cast(tensor_arr, ctypes.POINTER(ncclNDTensor_t))
            num_local_tensors = len(local_tensors)

        self.NCCL_CHECK(self._funcs["ncclEpCreateHandle"](
            ctypes.byref(handle), ep_group, topk_tensor,
            local_tensors_ptr, num_local_tensors, config_ptr, stream, use_fp8
        ))
        return handle

    def ncclEpHandleDestroy(self, handle):
        self.NCCL_CHECK(self._funcs["ncclEpHandleDestroy"](handle))

    def ncclEpUpdateHandle(self, handle, topk_tensor, stream, local_tensors=None):
        """Rebind topk_idx on an existing handle without reallocating buffers."""
        local_tensors_ptr = None
        num_local_tensors = 0
        if local_tensors is not None and len(local_tensors) > 0:
            tensor_arr = (ncclNDTensor_t * len(local_tensors))(*local_tensors)
            local_tensors_ptr = ctypes.cast(tensor_arr, ctypes.POINTER(ncclNDTensor_t))
            num_local_tensors = len(local_tensors)

        self.NCCL_CHECK(self._funcs["ncclEpUpdateHandle"](
            handle, topk_tensor, local_tensors_ptr, num_local_tensors, stream
        ))

    def ncclEpDispatch(self, handle, input_tensors, num_in, output_tensors, num_out, local_tensors, num_local, send_only, config, stream):
        config_ptr = ctypes.byref(config) if config else None
        self.NCCL_CHECK(self._funcs["ncclEpDispatch"](
            handle, input_tensors, num_in, output_tensors, num_out,
            local_tensors, num_local, send_only, config_ptr, stream))

    def ncclEpCombine(self, handle, input_tensors, num_in, output_tensors, num_out, local_tensors, num_local, send_only, config, stream):
        self.NCCL_CHECK(self._funcs["ncclEpCombine"](
            handle, input_tensors, num_in, output_tensors, num_out,
            local_tensors, num_local, send_only, config, stream))

    def ncclEpHandleGetNumRecvTokens(self, handle):
        """Get the number of tokens this rank will receive after dispatch.

        In HT mode with max_tokens_per_rank=0, returns actual count from notify_dispatch.
        Otherwise, returns max_tokens_per_rank.
        """
        num_tokens = ctypes.c_uint()
        self.NCCL_CHECK(self._funcs["ncclEpHandleGetNumRecvTokens"](handle, ctypes.byref(num_tokens)))
        return num_tokens.value

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
