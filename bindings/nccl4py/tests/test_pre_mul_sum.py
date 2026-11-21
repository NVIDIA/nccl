# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes

import numpy as np
import pytest

from nccl.bindings import nccl as bindings
from nccl.core import Communicator, NcclDataType


@pytest.mark.parametrize(
    "scalar,requested,expected",
    [
        (np.array(1.5, dtype=np.float64), NcclDataType.FLOAT32, np.float32(1.5)),
        (np.array([3], dtype=np.int32), NcclDataType.FLOAT32, np.float32(3)),
        (np.array([2, 0], dtype=np.float32)[:1], NcclDataType.FLOAT64, np.float64(2)),
        (np.array([[4]], dtype=np.float32), None, np.float32(4)),
        (np.array([5], dtype=np.dtype(np.int32).newbyteorder("S")), None, np.int32(5)),
        (np.array([6], dtype=np.float32), bindings.DataType.Float32, np.float32(6)),
        (np.array([-1], dtype=np.int8), int(NcclDataType.INT8), np.int8(-1)),
    ],
)
def test_pre_mul_sum_host_scalar(monkeypatch, scalar, requested, expected):
    op_handle = 17

    def create(scalar_ptr, datatype, residence, comm):
        assert datatype == NcclDataType(expected.dtype)
        assert residence == bindings.ScalarResidence.HostImmediate
        assert ctypes.string_at(scalar_ptr, expected.itemsize) == expected.tobytes()
        return op_handle

    monkeypatch.setattr(bindings, "red_op_create_pre_mul_sum", create)
    op = Communicator(ptr=1).create_pre_mul_sum(scalar, requested)
    assert int(op) == op_handle
