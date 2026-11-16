# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import pytest

from nccl.core.buffer import NcclBuffer
from nccl.core.typing import NcclDataType, NcclInvalid


class DLPackOnly:
    def __init__(self, array):
        self.array = array

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, **kwargs):
        return self.array.__dlpack__(**kwargs)


class CudaArrayInterfaceOnly:
    def __init__(self, array):
        self.array = array

    @property
    def __cuda_array_interface__(self):
        return self.array.__cuda_array_interface__


@pytest.fixture
def framework(request):
    module = pytest.importorskip(request.param)
    if request.param == "torch":
        if not module.cuda.is_available():
            pytest.skip("CUDA is required")
    elif module.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA is required")
    return module


def make_view(framework, shape, strides, offset=0):
    # Keep enough backing storage even for views rejected by the resolver.
    if framework.__name__ == "torch":
        base = framework.arange(128, dtype=framework.float32, device="cuda")
        return base.as_strided(shape, strides, offset)
    base = framework.arange(128, dtype=framework.float32)
    return framework.lib.stride_tricks.as_strided(
        base[offset:], shape=shape, strides=tuple(s * base.itemsize for s in strides)
    )


BUFFER_INPUTS = [
    ("torch", lambda x: x),
    ("torch", DLPackOnly),
    ("cupy", lambda x: x),
    ("cupy", DLPackOnly),
    ("cupy", CudaArrayInterfaceOnly),
]


@pytest.mark.parametrize("framework,wrapper", BUFFER_INPUTS, indirect=["framework"])
@pytest.mark.parametrize(
    "shape,strides,offset",
    [
        ((), (), 0),
        ((0, 3), (99, 7), 0),
        ((4,), (1,), 3),
        ((2, 3), (3, 1), 0),
        ((2, 3), (1, 2), 0),
        ((2, 3, 4), (4, 8, 1), 2),
        ((2, 1, 3), (3, 99, 1), 0),
        ((1,), (0,), 0),
    ],
)
def test_dense_layout(framework, wrapper, shape, strides, offset):
    array = make_view(framework, shape, strides, offset)
    result = NcclBuffer(wrapper(array))
    ptr = array.data_ptr() if framework.__name__ == "torch" else array.data.ptr
    # Empty DLPack tensors may normalize the pointer to zero.
    if math.prod(shape):
        assert result.ptr == ptr
    assert result.count == math.prod(shape)
    assert result.dtype == NcclDataType.FLOAT32
    device_id = array.device.index if framework.__name__ == "torch" else array.device.id
    assert result.device_id == device_id


@pytest.mark.parametrize("framework,wrapper", BUFFER_INPUTS, indirect=["framework"])
@pytest.mark.parametrize(
    "shape,strides", [((4,), (2,)), ((4,), (0,)), ((2, 3), (4, 1)), ((2, 3), (1, 1))]
)
def test_reject_non_dense_layout(framework, wrapper, shape, strides):
    array = make_view(framework, shape, strides, offset=1)
    with pytest.raises(NcclInvalid, match="contiguous, non-overlapping"):
        NcclBuffer(wrapper(array))


@pytest.mark.parametrize("wrapper", [lambda x: x, DLPackOnly, CudaArrayInterfaceOnly])
def test_reject_negative_stride(wrapper):
    cupy = pytest.importorskip("cupy")
    array = cupy.arange(8, dtype=cupy.float32)[::-1]
    with pytest.raises(NcclInvalid, match="contiguous, non-overlapping"):
        NcclBuffer(wrapper(array))


@pytest.mark.parametrize("framework_name", ["torch", "cupy"])
def test_direct_resolver_rejects_strided_view(framework_name):
    framework = pytest.importorskip(framework_name)
    array = make_view(framework, (4,), (2,))
    if framework_name == "torch":
        from nccl.core.interop.torch import resolve_tensor as resolve
    else:
        from nccl.core.interop.cupy import resolve_array as resolve
    with pytest.raises(NcclInvalid, match="contiguous, non-overlapping"):
        resolve(array)


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float8_e4m3fn", "bool"])
def test_torch_rejects_strided_dtypes(dtype_name):
    torch = pytest.importorskip("torch")
    tensor = torch.empty(8, device="cuda", dtype=getattr(torch, dtype_name))[::2]
    with pytest.raises(NcclInvalid, match="contiguous, non-overlapping"):
        NcclBuffer(tensor)


@pytest.fixture(scope="module")
def communicators():
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required")
    from nccl.core import Communicator, group

    comms = Communicator.init_all([0, 1])
    yield comms
    with group():
        for comm in comms:
            comm.finalize()
    for comm in comms:
        comm.destroy()


@pytest.mark.parametrize(
    "shape,strides", [((2, 3), (3, 1)), ((2, 3), (1, 2)), ((2, 3, 4), (4, 8, 1))]
)
@pytest.mark.parametrize("framework_name", ["torch", "cupy"])
def test_allreduce_storage_order(communicators, shape, strides, framework_name):
    import torch

    from nccl.core import NcclRedOp, group

    count = math.prod(shape)
    inputs = [torch.arange(count, device=i, dtype=torch.float32) + i for i in range(2)]
    outputs = [torch.full((count + 2,), -999.0, device=i) for i in range(2)]
    for i in range(2):
        torch.cuda.synchronize(i)
    sends = [x.as_strided(shape, strides) for x in inputs]
    recvs = [x[1:-1].as_strided(shape, strides) for x in outputs]
    if framework_name == "cupy":
        cupy = pytest.importorskip("cupy")
        sends = [cupy.from_dlpack(x) for x in sends]
        recvs = [cupy.from_dlpack(x) for x in recvs]
    with group():
        for i, comm in enumerate(communicators):
            with torch.cuda.device(i):
                comm.allreduce(
                    sends[i],
                    recvs[i],
                    NcclRedOp.SUM,
                )
    expected = torch.arange(count, dtype=torch.float32) * 2 + 1
    for i in range(2):
        torch.cuda.synchronize(i)
        actual = outputs[i].cpu()
        assert torch.equal(actual[1:-1], expected)
        assert actual[0] == actual[-1] == -999.0


@pytest.mark.parametrize("side", ["send", "recv"])
def test_allreduce_rejects_before_enqueue(communicators, monkeypatch, side):
    import torch

    from nccl.bindings import nccl as bindings
    from nccl.core import NcclRedOp

    def unexpected_enqueue(*args):
        pytest.fail("Invalid buffer reached the native all_reduce call")

    monkeypatch.setattr(bindings, "all_reduce", unexpected_enqueue)
    with torch.cuda.device(0):
        send = torch.arange(8, device="cuda", dtype=torch.float32)
        recv = torch.full_like(send, -999.0)
        send_view = send[::2] if side == "send" else send[:4]
        recv_view = recv[::2] if side == "recv" else recv[:4]
        with pytest.raises(NcclInvalid, match="contiguous, non-overlapping"):
            communicators[0].allreduce(send_view, recv_view, NcclRedOp.SUM)
        assert torch.equal(recv.cpu(), torch.full((8,), -999.0))
