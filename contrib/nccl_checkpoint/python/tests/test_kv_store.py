# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import pytest
from mpi4py import MPI


REDIS_SERVER = "redis-server"
MPI_COMM = MPI.COMM_WORLD


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _kv_sanity_test() -> Path:
    binary = _repo_root() / "build" / "bin" / "kv_sanity_test"
    if not binary.exists():
        pytest.skip(f"{binary} is missing; run `make -C contrib/nccl_checkpoint`")
    return binary


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _pick_local_port() -> tuple[int | None, str | None]:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("0.0.0.0", 0))
            return sock.getsockname()[1], None
    except OSError as err:
        return None, str(err)


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


@contextlib.contextmanager
def _shared_tmpdir(prefix: str):
    rank = MPI_COMM.Get_rank()
    if rank == 0:
        with tempfile.TemporaryDirectory(prefix=prefix, dir=os.path.realpath(os.getcwd())) as tmpdir:
            MPI_COMM.bcast(tmpdir, root=0)
            try:
                yield Path(tmpdir)
            finally:
                MPI_COMM.Barrier()
    else:
        tmpdir = MPI_COMM.bcast(None, root=0)
        try:
            yield Path(tmpdir)
        finally:
            MPI_COMM.Barrier()


@contextlib.contextmanager
def _redis_server():
    if shutil.which(REDIS_SERVER) is None:
        pytest.skip(f"{REDIS_SERVER} is not available")

    with _shared_tmpdir(prefix="nccl-checkpoint-kvs-test-") as tmpdir:
        kvs_path = tmpdir / "checkpoint-kvs.txt"
        stdout_path = tmpdir / "redis.stdout"
        stderr_path = tmpdir / "redis.stderr"
        proc = None
        skip_reason = None
        fail_reason = None

        if MPI_COMM.Get_rank() == 0:
            port, port_error = _pick_local_port()
            if port is None:
                skip_reason = f"unable to bind a local redis test port: {port_error}"
            else:
                host = socket.gethostname()
                cmd = [
                    str(REDIS_SERVER),
                    "--bind",
                    "0.0.0.0",
                    "--port",
                    str(port),
                    "--save",
                    "",
                    "--protected-mode",
                    "no",
                    "--appendonly",
                    "no",
                    "--dir",
                    str(tmpdir),
                ]
                stdout_file = stdout_path.open("w", encoding="utf-8")
                stderr_file = stderr_path.open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file, text=True)
                stdout_file.close()
                stderr_file.close()
                if not _wait_for_port(host, port, timeout_s=5.0):
                    fail_details = [f"Command: {' '.join(cmd)}", "redis-server failed to start"]
                    stdout_text = _read_text(stdout_path).strip()
                    stderr_text = _read_text(stderr_path).strip()
                    if stdout_text:
                        fail_details.append(f"stdout:\n{stdout_text}")
                    if stderr_text:
                        fail_details.append(f"stderr:\n{stderr_text}")
                    fail_reason = "\n".join(fail_details)
                else:
                    kvs_path.write_text(f"{host}:{port}\n", encoding="utf-8")

        skip_reason = MPI_COMM.bcast(skip_reason, root=0)
        fail_reason = MPI_COMM.bcast(fail_reason, root=0)
        if skip_reason:
            pytest.skip(skip_reason)
        if fail_reason:
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)
            pytest.fail(fail_reason)

        MPI_COMM.Barrier()
        try:
            yield kvs_path
        finally:
            MPI_COMM.Barrier()
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)


def _run_kv_helper(mode: str, kvs_path: Path, timeout_s: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["NCCL_CHECKPOINT_KVS_PATH"] = str(kvs_path)
    env["NCCL_CHECKPOINT_KVS_TIMEOUT"] = timeout_s
    env["WORLD_RANK"] = str(MPI_COMM.Get_rank())
    env["WORLD_SIZE"] = str(MPI_COMM.Get_size())
    return subprocess.run(
        [str(_kv_sanity_test()), mode],
        env=env,
        text=True,
        capture_output=True,
        timeout=max(5.0, float(timeout_s) * 4.0),
        check=False,
    )


def test_kv_set_get_round_trip() -> None:
    with _redis_server() as kvs_path:
        result = _run_kv_helper("set-get", kvs_path, timeout_s="2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KVS [" not in result.stderr
    assert "KVS connect" not in result.stderr


def test_kv_missing_key_timeout_is_quiet_until_timeout() -> None:
    with _redis_server() as kvs_path:
        result = _run_kv_helper("missing-key-timeout", kvs_path, timeout_s="0.05")
    assert result.returncode == 0, result.stdout + result.stderr
    timeout_lines = [
        line for line in result.stderr.splitlines()
        if "KVS [GET key=" in line and "timed out" in line
    ]
    assert len(timeout_lines) == 1, result.stdout + result.stderr
    assert "unexpected reply type" not in result.stderr


def test_kv_connect_timeout_is_reported_once() -> None:
    with _shared_tmpdir(prefix="nccl-checkpoint-kvs-missing-") as tmpdir:
        kvs_path = tmpdir / "checkpoint-kvs.txt"
        if MPI_COMM.Get_rank() == 0:
            kvs_path.write_text("127.0.0.1:1\n", encoding="utf-8")
        MPI_COMM.Barrier()
        result = _run_kv_helper("set-get", kvs_path, timeout_s="0.05")
    assert result.returncode != 0, result.stdout + result.stderr
    timeout_lines = [
        line for line in result.stderr.splitlines()
        if line.startswith("KVS connect(") and "timed out" in line
    ]
    assert len(timeout_lines) == 1, result.stdout + result.stderr
