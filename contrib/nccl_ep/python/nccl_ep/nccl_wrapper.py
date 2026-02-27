# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Python ctypes wrapper for NCCL with EP extensions.

This module provides a zero-compilation interface to NCCL EP by directly
loading the shared library and wrapping its functions using ctypes.
"""

import ctypes
import ctypes.util
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

# Optional torch import for communicator creation
try:
    import torch
    import torch.distributed as dist
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    torch = None
    dist = None

if TYPE_CHECKING:
    from typing import Type

# Type Definitions
ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int

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


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


class ncclDataTypeEnum:
    """NCCL data type enumerations."""
    ncclInt8 = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt64 = 4
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclFloat64 = 8
    ncclBfloat16 = 9

    @classmethod
    def from_torch(cls, dtype) -> int:
        """Convert torch.dtype to NCCL data type enum.

        Note: Requires PyTorch to be installed.
        """
        if not HAVE_TORCH:
            raise RuntimeError("from_torch() requires PyTorch to be installed")

        dtype_map = {
            torch.int8: cls.ncclInt8,
            torch.uint8: cls.ncclUint8,
            torch.int32: cls.ncclInt32,
            torch.int64: cls.ncclInt64,
            torch.float16: cls.ncclFloat16,
            torch.float32: cls.ncclFloat32,
            torch.float64: cls.ncclFloat64,
            torch.bfloat16: cls.ncclBfloat16,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dtype_map[dtype]


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
    NCCL_EP_TENSOR_TAG_RECV_EXPERT_COUNTER_HOST = 6
    NCCL_EP_TENSOR_TAG_TOKENS_PER_EXPERTS = 7


class ncclEpAlgorithm_t:
    NCCL_EP_ALGO_LOW_LATENCY = 0
    NCCL_EP_ALGO_HIGH_THROUGHPUT = 1


class ncclNDTensor_t(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("ndim", ctypes.c_uint),
        ("sizes", ctypes.POINTER(ctypes.c_uint)),
        ("strides", ctypes.POINTER(ctypes.c_uint)),
        ("datatype", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("tag", ctypes.c_uint),
        ("flags", ctypes.c_int),
    ]


class ncclEpGroupConfig_t(ctypes.Structure):
    _fields_ = [
        ("version", ctypes.c_uint),
        ("algorithm", ctypes.c_int),
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
    """Pure Python wrapper for NCCL library with EP extensions."""

    exported_functions = [
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        Function("ncclCommInitRank", ncclResult_t, [
            ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int
        ]),
        Function("ncclAllReduce", ncclResult_t, [
            buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t,
            ctypes.c_int, ncclComm_t, cudaStream_t
        ]),
        # ncclEpCreateGroup with per-group allocator callbacks
        Function("ncclEpCreateGroup", ncclResult_t, [
            ctypes.POINTER(ncclEpGroup_t), ncclComm_t,
            ctypes.POINTER(ncclEpGroupConfig_t), cudaStream_t,
            ncclEpAllocFn_t, ncclEpFreeFn_t
        ]),
        Function("ncclEpGroupDestroy", ncclResult_t, [ncclEpGroup_t, cudaStream_t]),
        Function("ncclEpCreateHandle", ncclResult_t, [
            ctypes.POINTER(ncclEpHandle_t), ncclEpGroup_t,
            ctypes.POINTER(ncclNDTensor_t),  # topk_idx
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)),  # local_tensors array
            ctypes.c_uint,  # num_local_tensors
            ctypes.POINTER(ncclEpHandleConfig_t),
            cudaStream_t,
            ctypes.c_bool  # use_fp8
        ]),
        Function("ncclEpHandleDestroy", ncclResult_t, [ncclEpHandle_t]),
        Function("ncclEpDispatch", ncclResult_t, [
            ncclEpHandle_t,
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)), ctypes.c_uint,
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)), ctypes.c_uint,
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)), ctypes.c_uint,
            ctypes.c_uint, ctypes.POINTER(ncclEpDispatchConfig_t),
            cudaStream_t
        ]),
        Function("ncclEpCombine", ncclResult_t, [
            ncclEpHandle_t,
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)), ctypes.c_uint,
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)), ctypes.c_uint,
            ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)), ctypes.c_uint,
            ctypes.c_uint, ctypes.POINTER(ncclEpCombineConfig_t), cudaStream_t
        ]),
        Function("ncclEpHandleGetNumRecvTokens", ncclResult_t, [
            ncclEpHandle_t, ctypes.POINTER(ctypes.c_uint)
        ]),
        Function("ncclEpComplete", ncclResult_t, [
            ncclEpHandle_t, ctypes.c_void_p, cudaStream_t
        ]),
    ]

    ep_function_names = [
        "ncclEpCreateGroup", "ncclEpGroupDestroy",
        "ncclEpCreateHandle", "ncclEpHandleDestroy",
        "ncclEpDispatch", "ncclEpCombine", "ncclEpHandleGetNumRecvTokens",
        "ncclEpComplete"
    ]

    path_to_library_cache = {}
    path_to_dict_mapping = {}
    _nccl_base_lib = None  # Cache for base NCCL library (ctypes.CDLL object)
    _nccl_base_lib_path = None  # Path to base NCCL library

    def __init__(self, so_file=None):
        if so_file is None:
            so_file = self._find_nccl_library()

        if so_file not in NCCLLibrary.path_to_library_cache:
            # If loading libnccl_ep.so, first load base NCCL with RTLD_GLOBAL
            # so that its symbols (like ncclCommCuDevice) are available
            if 'libnccl_ep' in so_file:
                self._load_base_nccl_library(so_file)

            lib = ctypes.CDLL(so_file)
            NCCLLibrary.path_to_library_cache[so_file] = lib

        self.lib = NCCLLibrary.path_to_library_cache[so_file]
        self.so_file = so_file

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs = {}

            # For EP library: get standard NCCL functions from base library,
            # and EP functions from EP library
            is_ep_lib = 'libnccl_ep' in so_file
            base_lib = NCCLLibrary._nccl_base_lib if is_ep_lib else None

            for func in NCCLLibrary.exported_functions:
                is_ep_func = func.name in NCCLLibrary.ep_function_names

                # Choose which library to get the function from
                if is_ep_lib and not is_ep_func and base_lib is not None:
                    # Standard NCCL function - get from base library
                    target_lib = base_lib
                else:
                    # EP function or single library mode - get from main lib
                    target_lib = self.lib

                try:
                    f = getattr(target_lib, func.name)
                    f.restype = func.restype
                    f.argtypes = func.argtypes
                    _funcs[func.name] = f
                except AttributeError:
                    if not is_ep_func:
                        raise
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs

        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]
        self.ep_available = all(name in self._funcs for name in NCCLLibrary.ep_function_names)

    def _load_base_nccl_library(self, ep_so_file):
        """Load base NCCL library with RTLD_GLOBAL flag.

        When loading libnccl_ep.so, we need to first load the base libnccl.so
        with RTLD_GLOBAL so that its symbols (like ncclCommCuDevice) are available
        to the EP library.

        Args:
            ep_so_file: Path to the EP library (used to find base NCCL in same dir)
        """
        # Only load once
        if NCCLLibrary._nccl_base_lib is not None:
            return

        # Find base NCCL library
        base_nccl_path = None

        # Try in the same directory as the EP library
        ep_dir = os.path.dirname(ep_so_file)
        for candidate in ['libnccl.so.2', 'libnccl.so']:
            candidate_path = os.path.join(ep_dir, candidate)
            if os.path.exists(candidate_path):
                base_nccl_path = candidate_path
                break

        # Try in NCCL_HOME if specified
        if base_nccl_path is None:
            nccl_home = os.environ.get('NCCL_HOME')
            if nccl_home:
                for candidate in ['libnccl.so.2', 'libnccl.so']:
                    candidate_path = os.path.join(nccl_home, 'lib', candidate)
                    if os.path.exists(candidate_path):
                        base_nccl_path = candidate_path
                        break

        # Try system library path
        if base_nccl_path is None:
            try:
                for lib_name in ['nccl.2', 'nccl']:
                    lib = ctypes.util.find_library(lib_name)
                    if lib:
                        base_nccl_path = lib
                        break
            except:
                pass

        if base_nccl_path is None:
            raise RuntimeError(
                f"Could not find base NCCL library (libnccl.so.2 or libnccl.so) "
                f"to load with EP library {ep_so_file}. "
                f"Ensure NCCL is installed and NCCL_HOME is set."
            )

        # Load with RTLD_GLOBAL flag so symbols are available to EP library
        try:
            NCCLLibrary._nccl_base_lib = ctypes.CDLL(base_nccl_path, mode=ctypes.RTLD_GLOBAL)
            NCCLLibrary._nccl_base_lib_path = base_nccl_path
        except Exception as e:
            raise RuntimeError(
                f"Failed to load base NCCL library from {base_nccl_path}: {e}"
            )

    def _find_nccl_library(self):
        """Find NCCL EP library.

        Search order:
        1. NCCL_HOME/lib/libnccl_ep.so (preferred - dedicated EP library)
        2. NCCL_HOME/lib/libnccl.so (fallback - if EP symbols are in main library)
        3. System library path for nccl_ep or nccl
        """
        # Find NCCL EP library
        nccl_home = os.environ.get('NCCL_HOME')
        if nccl_home:
            # First try the dedicated EP library
            ep_lib_path = os.path.join(nccl_home, 'lib', 'libnccl_ep.so')
            if os.path.exists(ep_lib_path):
                return ep_lib_path

            # Fall back to main NCCL library (if EP symbols are linked in)
            lib_path = os.path.join(nccl_home, 'lib', 'libnccl.so')
            if os.path.exists(lib_path):
                return lib_path

        try:
            # Try nccl_ep first (C library name)
            lib = ctypes.util.find_library('nccl_ep')
            if lib:
                return lib
            # Fall back to nccl
            lib = ctypes.util.find_library('nccl')
            if lib:
                return lib
        except:
            pass

        raise RuntimeError(
            "Could not find NCCL library. Set NCCL_HOME or add libnccl.so to LD_LIBRARY_PATH."
        )

    def ncclGetErrorString(self, result):
        return self._funcs["ncclGetErrorString"](result).decode("utf-8")

    def NCCL_CHECK(self, result):
        if result != 0:
            raise RuntimeError(f"NCCL error: {self.ncclGetErrorString(result)}")

    def ncclGetVersion(self):
        version = ctypes.c_int()
        self.NCCL_CHECK(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        v = str(version.value)
        return f"{v[0]}.{v[1:3].lstrip('0') or '0'}.{v[3:].lstrip('0') or '0'}"

    def ncclGetUniqueId(self):
        unique_id = ncclUniqueId()
        self.NCCL_CHECK(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(self, world_size, unique_id, rank):
        comm = ncclComm_t()
        self.NCCL_CHECK(self._funcs["ncclCommInitRank"](ctypes.byref(comm), world_size, unique_id, rank))
        return comm

    def ncclAllReduce(self, sendbuff, recvbuff, count, datatype, op, comm, stream):
        """Perform an all-reduce operation using NCCL."""
        self.NCCL_CHECK(self._funcs["ncclAllReduce"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def ncclEpCreateGroup(self, comm, config, stream, alloc_fn=None, free_fn=None):
        """Create NCCL EP group for distributed EP operations.

        Args:
            comm: NCCL communicator
            config: EP group configuration (ncclEpGroupConfig_t)
            stream: CUDA stream
            alloc_fn: Optional custom allocator callback (ncclEpAllocFn_t).
                     If None, uses cudaMalloc/cudaFree.
            free_fn: Optional custom free callback (ncclEpFreeFn_t).
                    If None, uses cudaFree.

        Returns:
            ncclEpGroup_t: Opaque handle to the EP group
        """
        if not self.ep_available:
            raise RuntimeError("NCCL EP not available")

        ep_group = ncclEpGroup_t()

        # Convert None to NULL callbacks
        alloc_callback = alloc_fn if alloc_fn is not None else ctypes.cast(None, ncclEpAllocFn_t)
        free_callback = free_fn if free_fn is not None else ctypes.cast(None, ncclEpFreeFn_t)

        self.NCCL_CHECK(self._funcs["ncclEpCreateGroup"](
            ctypes.byref(ep_group), comm, ctypes.byref(config), stream,
            alloc_callback, free_callback
        ))
        return ep_group

    def ncclEpGroupDestroy(self, ep_group, stream):
        if not self.ep_available:
            raise RuntimeError("NCCL EP not available")
        self.NCCL_CHECK(self._funcs["ncclEpGroupDestroy"](ep_group, stream))

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
                          with tag RECV_EXPERT_COUNTER_HOST (pinned+mapped) or _DEVICE.
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
            # Create array of pointers to ncclNDTensor_t
            tensor_ptrs = (ctypes.POINTER(ncclNDTensor_t) * len(local_tensors))()
            for i, tensor in enumerate(local_tensors):
                tensor_ptrs[i] = ctypes.pointer(tensor)
            local_tensors_ptr = ctypes.cast(tensor_ptrs, ctypes.POINTER(ctypes.POINTER(ncclNDTensor_t)))
            num_local_tensors = len(local_tensors)

        self.NCCL_CHECK(self._funcs["ncclEpCreateHandle"](
            ctypes.byref(handle), ep_group, ctypes.byref(topk_tensor),
            local_tensors_ptr, num_local_tensors, config_ptr, stream, use_fp8
        ))
        return handle

    def ncclEpHandleDestroy(self, handle):
        if not self.ep_available:
            raise RuntimeError("NCCL EP not available")
        self.NCCL_CHECK(self._funcs["ncclEpHandleDestroy"](handle))

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
        if not self.ep_available:
            raise RuntimeError("NCCL EP not available")
        self.NCCL_CHECK(self._funcs["ncclEpComplete"](handle, None, stream))


def get_nccl_comm_from_group(group=None, nccl_lib: Optional['NCCLLibrary'] = None) -> ncclComm_t:
    """Create NCCL communicator for the given ProcessGroup.

    Following vLLM's approach, we always create a new NCCL communicator rather than
    extracting from PyTorch's ProcessGroup (which is fragile and version-dependent).

    Args:
        group: PyTorch distributed process group (or None for default group or MPI-only mode).
              If PyTorch is not available, this is ignored and MPI is used.
        nccl_lib: NCCLLibrary instance for creating new communicator (required)

    Returns:
        NCCL communicator pointer (ncclComm_t)

    Raises:
        RuntimeError: If NCCL communicator cannot be created
    """
    if nccl_lib is None:
        raise RuntimeError(
            "Cannot create NCCL communicator without NCCLLibrary instance. "
            "Pass nccl_lib parameter to get_nccl_comm_from_group."
        )

    return _create_nccl_comm_for_group(group, nccl_lib)


def _create_nccl_comm_for_group(group, nccl_lib: 'NCCLLibrary') -> ncclComm_t:
    """Create NCCL communicator using ncclCommInitRank.

    Follows vLLM's approach of always creating a new communicator rather than
    extracting from PyTorch's ProcessGroup.

    Args:
        group: PyTorch distributed process group (or None for MPI-only mode).
              If PyTorch is not available, falls back to MPI mode.
        nccl_lib: NCCLLibrary instance for NCCL API calls

    Returns:
        NCCL communicator pointer (ncclComm_t)
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

    # Create and broadcast unique ID
    if rank == 0:
        unique_id = nccl_lib.ncclGetUniqueId()
    else:
        unique_id = ncclUniqueId()

    if HAVE_TORCH and (group is not None or dist.is_initialized()):
        # PyTorch distributed mode: use PyTorch broadcast
        backend_name = "nccl"
        try:
            backend_name = dist.get_backend(group) if group else dist.get_backend()
        except:
            pass

        # NCCL backend needs GPU tensor, gloo/mpi use CPU
        if backend_name == "nccl":
            tensor = torch.tensor(list(unique_id.internal), dtype=torch.uint8, device=device)
        else:
            tensor = torch.ByteTensor(list(unique_id.internal))

        # Broadcast from appropriate source
        if group is not None:
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(tensor, src=ranks[0], group=group)
        else:
            dist.broadcast(tensor, src=0)

        # Convert back to ncclUniqueId
        byte_list = tensor.cpu().tolist() if backend_name == "nccl" else tensor.tolist()
        for i, byte in enumerate(byte_list):
            unique_id.internal[i] = byte
    else:
        # MPI-only mode: use file-based broadcast
        import time
        import binascii

        temp_file = os.path.join(os.getcwd(), '.nccl_unique_id.tmp')
        ready_file = os.path.join(os.getcwd(), '.nccl_unique_id_ready.tmp')

        if rank == 0:
            # Write unique ID to file
            unique_id_hex = binascii.hexlify(bytes(unique_id.internal)).decode('ascii')
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

            unique_id_bytes = binascii.unhexlify(unique_id_hex)
            for i in range(128):
                unique_id.internal[i] = unique_id_bytes[i]

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
            comm = nccl_lib.ncclCommInitRank(world_size, unique_id, rank)
    else:
        # Without torch, just call directly (assumes CUDA device already set)
        comm = nccl_lib.ncclCommInitRank(world_size, unique_id, rank)

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
