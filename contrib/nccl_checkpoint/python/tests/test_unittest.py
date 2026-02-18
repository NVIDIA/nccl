# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
import cupy as cp
import nccl.core as nccl
from mpi4py import MPI
import pytest

class ScenarioSkip(RuntimeError):
    """Raised when a scenario cannot run in the current environment."""


REDIS_SERVER = "redis-server"
MPI_COMM = MPI.COMM_WORLD


def _use_shim(request: pytest.FixtureRequest) -> bool:
    return request.config.getoption("--checkpoint-mode") == "shim"


def _pick_local_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return sock.getsockname()[1]
    except PermissionError as err:
        pytest.skip(f"unable to bind a local redis test port in this environment: {err}")


def _wait_for_port(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            try:
                sock.connect((host, port))
            except OSError:
                time.sleep(0.05)
                continue
            return True
    return False


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


@contextlib.contextmanager
def _redis_server(kvs_path, comm, root):

    if comm.Get_rank() != root:
        try:
            yield
        finally:
            comm.Barrier()
        return

    with tempfile.TemporaryDirectory(prefix="nccl-checkpoint-redis-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        stdout_path = tmpdir_path / "redis.stdout"
        stderr_path = tmpdir_path / "redis.stderr"
        port = _pick_local_port()
        host = socket.gethostname()
        cmd = [
            str(REDIS_SERVER),
            "--bind", "0.0.0.0",
            "--port", str(port),
            "--save", "",
            "--protected-mode", "no",
            "--appendonly", "no",
            "--dir", str(tmpdir_path),
        ]

        with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open("w", encoding="utf-8") as stderr_file:
            proc = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
            try:
                if not _wait_for_port(host, port, timeout_s=5.0):
                    proc.terminate()
                    proc.wait(timeout=5)
                    details = [
                        f"Command: {' '.join(cmd)}",
                        "redis-server failed to start",
                    ]
                    stdout_text = _read_text(stdout_path).strip()
                    stderr_text = _read_text(stderr_path).strip()
                    if stdout_text:
                        details.append(f"stdout:\n{stdout_text}")
                    if stderr_text:
                        details.append(f"stderr:\n{stderr_text}")
                    pytest.fail("\n".join(details))
                with open(kvs_path, "w", encoding="utf-8") as f:
                    f.write(f"{host}:{port}\n")
                yield
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
                comm.Barrier()


@contextlib.contextmanager
def root_bcast_tmpdir(prefix, comm, root):
    if comm.Get_rank() == root:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=os.path.realpath(os.getcwd())) as tmpdir:
            comm.bcast(tmpdir, root=root)
            try:
                yield tmpdir
            finally:
                comm.Barrier()
    else:
        tmpdir = comm.bcast(None, root=root)
        try:
            yield tmpdir
        finally:
            comm.Barrier()



@pytest.fixture
def kvs_fixture(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    if not _use_shim(request):
        yield
        return

    with root_bcast_tmpdir(prefix="nccl-checkpoint-kvs-", comm=MPI_COMM, root=0) as tmpdir:
        kvs_path = os.path.join(tmpdir, "checkpoint-kvs.txt")
        monkeypatch.setenv("NCCL_CHECKPOINT_KVS_PATH", kvs_path)

        with _redis_server(kvs_path, MPI_COMM, root=0):
            yield


@pytest.fixture(scope="session")
def rank_info():
    rank = MPI_COMM.Get_rank()
    size = MPI_COMM.Get_size()
    host = socket.gethostname()
    hosts = MPI_COMM.allgather(host)
    local_rank = sum(1 for h in hosts[:rank] if h == host)

    class RankInfo:
        mpi_rank = rank
        mpi_size = size
        nccl_rank = rank
        nccl_size = size
        nccl_local_rank = local_rank

    return RankInfo()


def _select_device(rank_info) -> None:
    try:
        from cuda.core import Device, system
    except Exception as err:
        pytest.skip(f"cuda.core is unavailable: {err}")

    device_count = system.get_num_devices()
    if device_count < 1:
        pytest.skip("at least one CUDA device is required")

    device = Device(rank_info.nccl_local_rank)
    device.set_current()


def _checkpoint_library(required: bool = False):
    try:
        import nccl_checkpoint
    except ImportError as err:
        pytest.skip(f"nccl_checkpoint is unavailable: {err}")
    try:
        nccl_checkpoint.get_version()
    except nccl_checkpoint.NCCLCheckpointPreloadError as err:
        if required:
            pytest.skip(f"libnccl-checkpoint-shim.so is not loaded in this process: {err}")
        return None
    return nccl_checkpoint


def test_checkpointrestore() -> None:
    checkpoint = _checkpoint_library()
    if checkpoint is None:
        return
    checkpoint.checkpoint_prepare()
    checkpoint.checkpoint_restore()


test_checkpointrestore.__test__ = False


def test_checkpoint_version() -> None:
    checkpoint = _checkpoint_library()
    if checkpoint is None:
        return
    version = checkpoint.get_version()
    assert version.checkpoint_version == 100
    assert version.nccl_version > 0


def _expect_restore_unsafe_prepare_failure(request: pytest.FixtureRequest) -> None:
    if not _use_shim(request):
        pytest.skip("restore-unsafe prepare failures are shim-only")
    from nccl_checkpoint import NCCLCheckpointError

    checkpoint = _checkpoint_library(required=True)
    assert checkpoint is not None
    with pytest.raises(NCCLCheckpointError, match="ncclCheckpointPrepare failed"):
        checkpoint.checkpoint_prepare()


def nccl_wait_comms(comms, timeout_s: float = 30.0) -> None:
    import nccl.bindings.nccl as nccl_bindings
    import time

    deadline = time.monotonic() + timeout_s
    for comm in comms:
        while True:
            state = comm.get_async_error()
            if state == nccl_bindings.Result.Success:
                break
            if state != nccl_bindings.Result.InProgress:
                raise RuntimeError(f"communicator async state completed with {state}")
            if time.monotonic() >= deadline:
                raise TimeoutError("communicator async state did not complete before timeout")
            time.sleep(0.01)


def synchronize_or_timeout(comms, timeout_s: float = 5.0) -> None:
    event = cp.cuda.Event()
    event.record(cp.cuda.Stream.null)
    deadline = time.monotonic() + timeout_s

    while True:
        try:
            cp.cuda.runtime.eventQuery(event.ptr)
            break
        except cp.cuda.runtime.CUDARuntimeError as err:
            if err.status != cp.cuda.runtime.cudaErrorNotReady:
                raise
        if time.monotonic() >= deadline:
            for comm in comms:
                try:
                    comm.revoke()
                except Exception:
                    pass
            raise TimeoutError(f"CUDA stream did not complete before timeout ({timeout_s}s)")
        time.sleep(0.01)

    nccl_wait_comms(comms, timeout_s=timeout_s)


def synchronize_multidevice_or_timeout(comms, timeout_s: float = 5.0) -> None:
    events = []
    for comm in comms:
        comm.device.set_current()
        event = cp.cuda.Event()
        event.record(cp.cuda.Stream.null)
        events.append((comm, event))

    pending = set(range(len(events)))
    deadline = time.monotonic() + timeout_s
    while pending:
        for idx in list(pending):
            comm, event = events[idx]
            comm.device.set_current()
            try:
                cp.cuda.runtime.eventQuery(event.ptr)
                pending.remove(idx)
            except cp.cuda.runtime.CUDARuntimeError as err:
                if err.status != cp.cuda.runtime.cudaErrorNotReady:
                    raise
        if not pending:
            break
        if time.monotonic() >= deadline:
            for comm in comms:
                try:
                    comm.revoke()
                except Exception:
                    pass
            raise TimeoutError(f"CUDA streams did not complete before timeout ({timeout_s}s)")
        time.sleep(0.01)

    nccl_wait_comms(comms, timeout_s=timeout_s)


def _make_comm(rank_info, *, blocking: bool = True, split_share: bool | None = None):
    unique_id = nccl.get_unique_id(empty=(rank_info.nccl_rank != 0))
    MPI_COMM.Bcast([unique_id.as_ndarray, MPI.BYTE], root=0)
    config = nccl.NCCLConfig(blocking=blocking, split_share=split_share)
    comm = nccl.Communicator.init(
        nranks=rank_info.nccl_size,
        rank=rank_info.nccl_rank,
        unique_id=unique_id,
        config=config,
    )
    return comm


def _finalize_and_destroy(comm) -> None:
    comm.finalize()
    nccl_wait_comms([comm])
    comm.destroy()


def _finalize_and_destroy_many(comms) -> None:
    if not comms:
        return
    with nccl.group():
        for comm in comms:
            comm.finalize()
    nccl_wait_comms(comms)
    for comm in comms:
        comm.destroy()


def _skip_unsupported_window(comm, reason: str) -> None:
    try:
        nccl_wait_comms([comm])
    except Exception:
        comm.abort()
        raise
    _finalize_and_destroy(comm)
    pytest.skip(reason)


@pytest.fixture(params=[True, False], ids=["blocking", "nonblocking"])
def comm_blocking(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


def test_empty_checkpointrestore(request: pytest.FixtureRequest, kvs_fixture, rank_info) -> None:
    _select_device(rank_info)
    test_checkpointrestore()


def test_basic(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    if os.environ.get("_FORCE_PYTEST_FAILURE") == "1":
        raise AssertionError("_FORCE_PYTEST_FAILURE=1 requested a basic test failure")
    _select_device(rank_info)
    comm = _make_comm(rank_info, blocking=comm_blocking)
    nccl_wait_comms([comm])
    send_data = cp.empty(1, dtype="float32")
    recv_data = cp.empty(1, dtype="float32")
    test_checkpointrestore()
    send_data[0] = rank_info.nccl_rank + 1
    comm.allreduce(send_data, recv_data, nccl.SUM)
    synchronize_or_timeout([comm])
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    assert float(recv_data.get()[0]) == expected
    _finalize_and_destroy(comm)


def test_multiple_comms(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    _select_device(rank_info)
    comm0 = _make_comm(rank_info, blocking=comm_blocking)
    comm1 = _make_comm(rank_info, blocking=comm_blocking)
    nccl_wait_comms([comm0, comm1])
    send0 = cp.empty(1024, dtype="float32")
    recv0 = cp.empty(1024, dtype="float32")
    send1 = cp.empty(1024, dtype="float32")
    recv1 = cp.empty(1024, dtype="float32")
    test_checkpointrestore()
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    send0[0] = rank_info.nccl_rank + 1
    comm0.allreduce(send0, recv0, nccl.SUM)
    send1[0] = rank_info.nccl_rank + 1
    comm1.allreduce(send1, recv1, nccl.SUM)
    synchronize_or_timeout([comm0, comm1])
    assert float(recv0.get()[0]) == expected
    assert float(recv1.get()[0]) == expected
    _finalize_and_destroy(comm0)
    _finalize_and_destroy(comm1)


def test_registration(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    _select_device(rank_info)
    comm = _make_comm(rank_info, blocking=comm_blocking)
    nccl_wait_comms([comm])
    buf0 = nccl.mem_alloc(4096)
    buf1 = nccl.mem_alloc(8192)
    handle0 = comm.register_buffer(buf0)
    nccl_wait_comms([comm])
    handle1 = comm.register_buffer(buf1)
    nccl_wait_comms([comm])
    assert handle0.is_valid
    assert handle1.is_valid
    test_checkpointrestore()
    send_data = cp.empty(1, dtype="float32")
    recv_data = cp.empty(1, dtype="float32")
    send_data[0] = rank_info.nccl_rank + 1
    comm.allreduce(send_data, recv_data, nccl.SUM)
    synchronize_or_timeout([comm])
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    assert float(recv_data.get()[0]) == expected
    handle1.close()
    handle0.close()
    _finalize_and_destroy(comm)


def test_window(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    _select_device(rank_info)
    comm = _make_comm(rank_info, blocking=comm_blocking)
    nccl_wait_comms([comm])
    comm.device.set_current()
    buf = nccl.cupy.empty(256, dtype="float32")
    window = comm.register_window(buf)
    nccl_wait_comms([comm])
    if window is None:
        _skip_unsupported_window(comm, "window registration is not supported in this environment")
    assert window.is_valid
    assert window.user_ptr == buf.data.ptr
    test_checkpointrestore()
    buf.fill(rank_info.nccl_rank + 1)
    comm.allreduce(buf, buf, nccl.SUM)
    synchronize_or_timeout([comm])
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    assert float(buf.get()[0]) == expected
    window.close()
    _finalize_and_destroy(comm)


def test_mixed_resources(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    _select_device(rank_info)
    comm = _make_comm(rank_info, blocking=comm_blocking)
    nccl_wait_comms([comm])
    reg_buf = nccl.mem_alloc(4096)
    comm.device.set_current()
    win_buf = nccl.cupy.empty(256, dtype="float32")
    reg_handle = comm.register_buffer(reg_buf)
    nccl_wait_comms([comm])
    window = comm.register_window(win_buf, flags=nccl.WindowFlag.StrictOrdering)
    nccl_wait_comms([comm])
    if window is None:
        reg_handle.close()
        _skip_unsupported_window(comm, "window registration is not supported in this environment")
    assert reg_handle.is_valid
    assert window.is_valid
    assert window.user_ptr == win_buf.data.ptr
    test_checkpointrestore()
    win_buf.fill(rank_info.nccl_rank + 1)
    comm.allreduce(win_buf, win_buf, nccl.SUM)
    synchronize_or_timeout([comm])
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    assert float(win_buf.get()[0]) == expected
    window.close()
    reg_handle.close()
    _finalize_and_destroy(comm)


def test_restore_unsafe_window_user_ptr_prepare_fails(request: pytest.FixtureRequest, kvs_fixture, rank_info) -> None:
    _select_device(rank_info)
    if not _use_shim(request):
        pytest.skip("restore-unsafe prepare failures are shim-only")
    import nccl.bindings.nccl as nccl_bindings

    comm = _make_comm(rank_info, blocking=True)
    nccl_wait_comms([comm])
    comm.device.set_current()
    buf = nccl.cupy.empty(256, dtype="float32")
    window = None
    try:
        window = comm.register_window(buf)
        nccl_wait_comms([comm])
        if window is None:
            _skip_unsupported_window(comm, "window registration is not supported in this environment")
        assert nccl_bindings.win_get_user_ptr(comm._comm, window.handle) == buf.data.ptr
        _expect_restore_unsafe_prepare_failure(request)
    finally:
        if window is not None:
            window.close()
        _finalize_and_destroy(comm)


def test_restore_unsafe_custom_redop_prepare_fails(request: pytest.FixtureRequest, kvs_fixture, rank_info) -> None:
    _select_device(rank_info)
    if not _use_shim(request):
        pytest.skip("restore-unsafe prepare failures are shim-only")

    comm = _make_comm(rank_info, blocking=True)
    nccl_wait_comms([comm])
    op = None
    try:
        op = comm.create_pre_mul_sum(1.0, datatype=nccl.FLOAT32)
        _expect_restore_unsafe_prepare_failure(request)
    finally:
        if op is not None:
            op.close()
        _finalize_and_destroy(comm)


def test_restore_unsafe_dev_comm_prepare_fails(request: pytest.FixtureRequest, kvs_fixture, rank_info) -> None:
    _select_device(rank_info)
    if not _use_shim(request):
        pytest.skip("restore-unsafe prepare failures are shim-only")

    comm = _make_comm(rank_info, blocking=True)
    nccl_wait_comms([comm])
    dev_comm = None
    try:
        if not comm.device_api_support:
            pytest.skip("NCCL device API is not supported in this environment")
        dev_comm = comm.create_dev_comm()
        _expect_restore_unsafe_prepare_failure(request)
    finally:
        if dev_comm is not None:
            dev_comm.close()
        _finalize_and_destroy(comm)


def test_mixed_comm_modes(request: pytest.FixtureRequest, kvs_fixture, rank_info) -> None:
    _select_device(rank_info)
    blocking_comm = _make_comm(rank_info, blocking=True)
    nonblocking_comm = _make_comm(rank_info, blocking=False)
    nccl_wait_comms([blocking_comm, nonblocking_comm])
    blocking_buf = cp.empty(1, dtype="float32")
    nonblocking_buf = cp.empty(1, dtype="float32")
    test_checkpointrestore()
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    blocking_buf[0] = rank_info.nccl_rank + 1
    blocking_comm.allreduce(blocking_buf, blocking_buf, nccl.SUM)
    nonblocking_buf[0] = rank_info.nccl_rank + 1
    nonblocking_comm.allreduce(nonblocking_buf, nonblocking_buf, nccl.SUM)
    synchronize_or_timeout([blocking_comm, nonblocking_comm])
    assert float(blocking_buf.get()[0]) == expected
    assert float(nonblocking_buf.get()[0]) == expected
    _finalize_and_destroy(blocking_comm)
    _finalize_and_destroy(nonblocking_comm)


def test_init_all_restore(request: pytest.FixtureRequest, kvs_fixture, rank_info) -> None:
    if rank_info.mpi_rank % 2 != 0:
        return

    _select_device(rank_info)

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("two visible CUDA devices are required for ncclCommInitAll restore coverage")

    comms = []
    try:
        comms = nccl.Communicator.init_all([0, 1])
        nccl_wait_comms(comms)

        test_checkpointrestore()

        send_buffers = []
        recv_buffers = []
        for idx, comm in enumerate(comms):
            comm.device.set_current()
            send_buffers.append(cp.full(8, float(idx + 1), dtype="float32"))
            recv_buffers.append(cp.empty(8, dtype="float32"))

        with nccl.group():
            for idx, comm in enumerate(comms):
                comm.allreduce(send_buffers[idx], recv_buffers[idx], nccl.SUM)
        synchronize_multidevice_or_timeout(comms)

        for recv in recv_buffers:
            assert float(recv.get()[0]) == 3.0
    finally:
        _finalize_and_destroy_many(comms)


def test_even_ranks_two_gpus_same_process(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    cuda_error = None
    try:
        from cuda.core import Device, system
    except Exception as err:
        Device = None
        system = None
        cuda_error = str(err)
    cuda_errors = MPI_COMM.allgather(cuda_error)
    if any(cuda_errors):
        pytest.skip(f"cuda.core is unavailable: {next(err for err in cuda_errors if err)}")

    host = socket.gethostname()
    hosts = MPI_COMM.allgather(host)
    active_process = rank_info.mpi_rank % 2 == 0
    active_flags = MPI_COMM.allgather(active_process)
    active_ranks = [rank for rank, active in enumerate(active_flags) if active]
    active_index = active_ranks.index(rank_info.mpi_rank) if active_process else -1
    local_active_index = sum(
        1
        for rank, (rank_host, active) in enumerate(zip(hosts, active_flags))
        if rank < rank_info.mpi_rank and rank_host == host and active
    )
    device_ids = [2 * local_active_index, 2 * local_active_index + 1] if active_process else []
    try:
        device_count = system.get_num_devices()
        device_error = None
    except Exception as err:
        device_count = 0
        device_error = str(err)
    device_errors = MPI_COMM.allgather(device_error)
    if any(device_errors):
        pytest.skip(f"unable to query CUDA device count: {next(err for err in device_errors if err)}")
    can_run = (not active_process) or (device_count >= device_ids[-1] + 1)
    if not all(MPI_COMM.allgather(can_run)):
        pytest.skip("active even ranks need two CUDA devices each on their local host")

    root = active_ranks[0]
    unique_id = nccl.get_unique_id(empty=(rank_info.mpi_rank != root))
    MPI_COMM.Bcast([unique_id.as_ndarray, MPI.BYTE], root=root)

    comms = []
    try:
        if active_process:
            config = nccl.NCCLConfig(blocking=comm_blocking)
            total_nccl_ranks = 2 * len(active_ranks)
            with nccl.group():
                for local_idx, device_id in enumerate(device_ids):
                    Device(device_id).set_current()
                    comms.append(
                        nccl.Communicator.init(
                            nranks=total_nccl_ranks,
                            rank=2 * active_index + local_idx,
                            unique_id=unique_id,
                            config=config,
                        )
                    )
            nccl_wait_comms(comms)

        MPI_COMM.Barrier()
        test_checkpointrestore()
        MPI_COMM.Barrier()

        if active_process:
            send_buffers = []
            recv_buffers = []
            total_nccl_ranks = 2 * len(active_ranks)
            for local_idx, comm in enumerate(comms):
                comm.device.set_current()
                value = float(2 * active_index + local_idx + 1)
                send_buffers.append(cp.full(8, value, dtype="float32"))
                recv_buffers.append(cp.empty(8, dtype="float32"))

            with nccl.group():
                for idx, comm in enumerate(comms):
                    comm.allreduce(send_buffers[idx], recv_buffers[idx], nccl.SUM)
            synchronize_multidevice_or_timeout(comms)

            expected = float(total_nccl_ranks * (total_nccl_ranks + 1) // 2)
            for recv in recv_buffers:
                assert float(recv.get()[0]) == expected
    finally:
        _finalize_and_destroy_many(comms)
        MPI_COMM.Barrier()


def test_split_shared_resources_nocolor(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    _select_device(rank_info)
    parent = _make_comm(rank_info, blocking=comm_blocking, split_share=True)
    nccl_wait_comms([parent])

    color = 1
    active = lambda x: not (x == 0)
    if not active(rank_info.nccl_rank): color = None
    child = parent.split(color=color, key=rank_info.nccl_rank)
    if color is None:
        nccl_wait_comms([parent])
        assert not child.is_valid
    else:
        nccl_wait_comms([parent, child])
        assert child.is_valid
        assert child.nranks == sum([active(r) for r in range(rank_info.nccl_size)])

    test_checkpointrestore()

    if child.is_valid:
        send_data = cp.empty(1, dtype="float32")
        recv_data = cp.empty(1, dtype="float32")
        send_data[0] = rank_info.nccl_rank + 1
        child.allreduce(send_data, recv_data, nccl.SUM)
        synchronize_or_timeout([child])
        expected = sum([r + 1 for r in range(rank_info.nccl_size) if active(r)])
        assert abs(1.0 - recv_data.get()[0]/expected) < 0.001
        _finalize_and_destroy(child)

    MPI_COMM.Barrier()
    _finalize_and_destroy(parent)


def test_shrink_shared_resources_excluded(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    if rank_info.nccl_size < 2:
        pytest.skip("communicator shrink exclusion requires at least two ranks")

    _select_device(rank_info)
    parent = _make_comm(rank_info, blocking=comm_blocking)
    nccl_wait_comms([parent])

    exclude_ranks = [0]
    active = lambda x: x not in exclude_ranks
    child = None
    if active(rank_info.nccl_rank):
        shrink_config = nccl.NCCLConfig(blocking=comm_blocking, shrink_share=True)
        child = parent.shrink(exclude_ranks=exclude_ranks, config=shrink_config)
        nccl_wait_comms([parent])
        nccl_wait_comms([child])
        assert child.is_valid
        assert child.nranks == sum(active(r) for r in range(rank_info.nccl_size))
    else:
        nccl_wait_comms([parent])

    MPI_COMM.Barrier()
    test_checkpointrestore()
    MPI_COMM.Barrier()

    if child is not None:
        send_data = cp.empty(1, dtype="float32")
        recv_data = cp.empty(1, dtype="float32")
        send_data[0] = rank_info.nccl_rank + 1
        child.allreduce(send_data, recv_data, nccl.SUM)
        synchronize_or_timeout([child])
        expected = sum(r + 1 for r in range(rank_info.nccl_size) if active(r))
        assert abs(1.0 - recv_data.get()[0] / expected) < 0.001
        _finalize_and_destroy(child)

    MPI_COMM.Barrier()
    _finalize_and_destroy(parent)


def test_grow(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    if rank_info.nccl_size < 2:
        pytest.skip("communicator grow requires at least two ranks")

    _select_device(rank_info)
    old_size = rank_info.nccl_size - 1
    is_existing_rank = rank_info.nccl_rank < old_size

    init_uid = nccl.get_unique_id(empty=(rank_info.nccl_rank != 0))
    MPI_COMM.Bcast([init_uid.as_ndarray, MPI.BYTE], root=0)

    parent = None
    if is_existing_rank:
        parent = nccl.Communicator.init(
            nranks=old_size,
            rank=rank_info.nccl_rank,
            unique_id=init_uid,
            config=nccl.NCCLConfig(blocking=comm_blocking),
        )
        nccl_wait_comms([parent])

    if rank_info.nccl_rank == 0:
        grow_uid = parent.get_unique_id()
    else:
        grow_uid = nccl.get_unique_id(empty=True)
    MPI_COMM.Bcast([grow_uid.as_ndarray, MPI.BYTE], root=0)

    grow_config = None if comm_blocking else nccl.NCCLConfig(blocking=False)
    if is_existing_rank:
        grown = parent.grow(
            nranks=rank_info.nccl_size,
            unique_id=grow_uid if rank_info.nccl_rank == 0 else None,
            config=grow_config,
        )
    else:
        grown = nccl.Communicator().grow(
            nranks=rank_info.nccl_size,
            unique_id=grow_uid,
            rank=rank_info.nccl_rank,
            config=grow_config,
        )
    nccl_wait_comms([grown])

    test_checkpointrestore()

    send_data = cp.empty(1, dtype="float32")
    recv_data = cp.empty(1, dtype="float32")
    send_data[0] = rank_info.nccl_rank + 1
    grown.allreduce(send_data, recv_data, nccl.SUM)
    synchronize_or_timeout([grown])
    expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    assert float(recv_data.get()[0]) == expected
    _finalize_and_destroy(grown)
    if parent is not None:
        _finalize_and_destroy(parent)


def test_grow_preserves_parent(request: pytest.FixtureRequest, kvs_fixture, comm_blocking: bool, rank_info) -> None:
    if rank_info.nccl_size < 2:
        pytest.skip("communicator grow requires at least two ranks")

    _select_device(rank_info)
    old_size = rank_info.nccl_size - 1
    is_existing_rank = rank_info.nccl_rank < old_size

    init_uid = nccl.get_unique_id(empty=(rank_info.nccl_rank != 0))
    MPI_COMM.Bcast([init_uid.as_ndarray, MPI.BYTE], root=0)

    parent = None
    if is_existing_rank:
        parent = nccl.Communicator.init(
            nranks=old_size,
            rank=rank_info.nccl_rank,
            unique_id=init_uid,
            config=nccl.NCCLConfig(blocking=comm_blocking),
        )
        nccl_wait_comms([parent])

    if rank_info.nccl_rank == 0:
        grow_uid = parent.get_unique_id()
    else:
        grow_uid = nccl.get_unique_id(empty=True)
    MPI_COMM.Bcast([grow_uid.as_ndarray, MPI.BYTE], root=0)

    grow_config = None if comm_blocking else nccl.NCCLConfig(blocking=False)
    if is_existing_rank:
        grown = parent.grow(
            nranks=rank_info.nccl_size,
            unique_id=grow_uid if rank_info.nccl_rank == 0 else None,
            config=grow_config,
        )
        nccl_wait_comms([parent])
    else:
        grown = nccl.Communicator().grow(
            nranks=rank_info.nccl_size,
            unique_id=grow_uid,
            rank=rank_info.nccl_rank,
            config=grow_config,
        )
    nccl_wait_comms([grown])

    test_checkpointrestore()

    grown_send = cp.empty(1, dtype="float32")
    grown_recv = cp.empty(1, dtype="float32")
    grown_send[0] = rank_info.nccl_rank + 1
    grown.allreduce(grown_send, grown_recv, nccl.SUM)
    synchronize_or_timeout([grown])
    grown_expected = float(rank_info.nccl_size * (rank_info.nccl_size + 1) // 2)
    assert float(grown_recv.get()[0]) == grown_expected

    if parent is not None:
        parent_send = cp.empty(1, dtype="float32")
        parent_recv = cp.empty(1, dtype="float32")
        parent_send[0] = rank_info.nccl_rank + 1
        parent.allreduce(parent_send, parent_recv, nccl.SUM)
        synchronize_or_timeout([parent])
        parent_expected = float(old_size * (old_size + 1) // 2)
        assert float(parent_recv.get()[0]) == parent_expected

    _finalize_and_destroy(grown)
    if parent is not None:
        _finalize_and_destroy(parent)
