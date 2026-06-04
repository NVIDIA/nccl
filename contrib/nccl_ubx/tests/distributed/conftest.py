"""Distributed test infrastructure.

Multi-GPU tests use the subprocess pattern:
1. Outer pytest file calls run_distributed_test(script, num_procs, args)
2. Helper launches via srun or mpirun
3. Worker detects rank, initializes torch.distributed, runs op, compares vs reference
4. Outer test asserts on return code + stdout
"""

import os
import sys
import subprocess
import pytest
import shutil


def _find_launcher():
    """Determine the best launcher for distributed tests.

    Prefer srun (sets SLURM_PROCID/SLURM_NTASKS automatically).
    Fall back to torchrun which handles MASTER_ADDR/MASTER_PORT/RANK/LOCAL_RANK
    rendezvous correctly. Avoid mpirun — it does not set the env:// rendezvous
    variables that torch.distributed requires, causing EADDRINUSE on TCPStore.
    """
    if shutil.which("srun"):
        return "srun"
    return "torchrun"


def run_distributed_test(
    script: str,
    num_procs: int = 2,
    args: list = None,
    timeout: int = 120,
    env_extra: dict = None,
) -> subprocess.CompletedProcess:
    """Launch a distributed test script with the appropriate launcher.

    Args:
        script: Path to the worker script.
        num_procs: Number of processes (GPUs) to use.
        args: Additional command-line arguments for the worker.
        timeout: Maximum seconds to wait for completion.
        env_extra: Additional environment variables to set for the subprocess.

    Returns:
        CompletedProcess with stdout/stderr.
    """
    if args is None:
        args = []

    launcher = _find_launcher()
    workers_dir = os.path.join(os.path.dirname(__file__), "_workers")

    if launcher == "srun":
        # Build srun args. Some clusters don't expose GPUs as GRES —
        # `--gpus-per-node` is rejected with "Invalid generic resource
        # specification". Within an existing allocation, ntasks alone is
        # sufficient; the GPUs already belong to the job. Set
        # UBX_TEST_GPUS_FLAG=1 to force the gpus flag for clusters that
        # require it.
        srun_args = ["srun", f"--ntasks={num_procs}"]
        if os.environ.get("UBX_TEST_GPUS_FLAG") == "1":
            srun_args.append(f"--gpus-per-node={num_procs}")
        cmd = srun_args + [sys.executable, script] + args
    elif launcher == "mpirun":
        cmd = [
            "mpirun",
            "--allow-run-as-root",
            "-np", str(num_procs),
            sys.executable, script,
        ] + args
    else:
        # torchrun fallback for local development
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_procs}",
            "--standalone",
            script,
        ] + args

    env = {**os.environ, "PYTHONPATH": os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}
    if env_extra:
        env.update(env_extra)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result


def requires_multi_gpu(min_gpus=2):
    """Skip decorator for tests requiring multiple GPUs."""
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    return pytest.mark.skipif(
        num_gpus < min_gpus,
        reason=f"Requires {min_gpus}+ GPUs, found {num_gpus}",
    )
