"""Tests for top-level imports, version, and hardware detection."""

import pytest


class TestImports:
    """Top-level package surface."""

    def test_version_string(self):
        from ubx import __version__
        assert isinstance(__version__, str)
        parts = __version__.split(".")
        assert len(parts) == 3

    def test_top_level_exports(self):
        import ubx
        assert hasattr(ubx, "SymmAllocator")
        assert hasattr(ubx, "SymmTensor")
        assert hasattr(ubx, "compute_token_offsets")
        assert hasattr(ubx, "ops")
        assert hasattr(ubx, "fused")


class TestHardwareDetection:
    """Hardware probe utilities."""

    def test_get_cuda_sm_version_no_crash(self):
        from ubx._common import _get_cuda_sm_version
        major, minor = _get_cuda_sm_version()
        assert isinstance(major, int)
        assert isinstance(minor, int)

    @pytest.mark.requires_cuda
    def test_sm_version_positive(self):
        from ubx._common import _get_cuda_sm_version
        major, minor = _get_cuda_sm_version()
        assert major > 0



class TestNccl4pyBackend:
    """Backend health -- the checks this module is named for.

    It previously covered only version strings and hardware probes, so it
    stayed green on a build where every UB-X collective died at its first
    symmetric-pool allocation.
    """

    def test_allocator_symbol_is_bound(self, nccl_backend):
        """UB-X's allocation entry point must actually resolve.

        No GPU needed. This is the assertion that separates the broken build
        from the fixed one.
        """
        assert callable(nccl_backend._mem_alloc)

    def test_consumes_versioned_api_only(self):
        """UB-X must reach nccl4py only through its versioned surface.

        ``nccl.core.__init__`` states that semantic-versioning guarantees cover
        exactly the names in its ``__all__``, and that all other modules "are
        internal implementation details and are subject to change without
        notice". That rules out both ``nccl.bindings`` -- the private Cython
        layer whose 0.3.0 reshuffle caused this -- and ``nccl.core.<submodule>``,
        which carries no more promise than ``nccl.bindings`` did. Only bare
        ``nccl.core`` is a contract.
        """
        import ast
        import inspect
        from ubx import _nccl_backend

        tree = ast.parse(inspect.getsource(_nccl_backend))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.add(node.module)

        offenders = sorted(
            m for m in modules if m.split(".")[0] == "nccl" and m != "nccl.core"
        )
        assert not offenders, (
            f"_nccl_backend reaches outside nccl4py's versioned surface: "
            f"{offenders} -- import these from 'nccl.core' instead"
        )

    def test_missing_means_absent_not_relocated(self):
        """``nccl4py_missing`` must not read a moved submodule as "not installed".

        This is the hole that would let the next 0.3.0 through. A deleted or
        renamed ``nccl.core`` submodule raises ``ModuleNotFoundError`` exactly
        like an uninstalled package does; classifying the two alike would skip
        the regression instead of failing on it.
        """
        from ubx import _nccl_backend as backend

        def classify(err):
            saved = backend._NCCL4PY_IMPORT_ERROR
            backend._NCCL4PY_IMPORT_ERROR = err
            try:
                return backend.nccl4py_missing()
            finally:
                backend._NCCL4PY_IMPORT_ERROR = saved

        # nccl4py genuinely absent -> an environment gap, safe to skip.
        assert classify(ModuleNotFoundError("No module named 'nccl'", name="nccl"))
        # nccl4py present but reorganised -> a real breakage, must not skip.
        assert not classify(
            ModuleNotFoundError(
                "No module named 'nccl.core.buffer'", name="nccl.core.buffer"
            )
        )
        # nccl4py present, symbol gone -> likewise.
        assert not classify(
            ImportError("cannot import name 'mem_alloc' from 'nccl.core'")
        )


class TestDeviceIndexResolution:
    """``_resolve_device_index``: torch.device -> the ordinal mem_alloc wants.

    ``mem_alloc`` takes an ``NcclDeviceSpec``, so a ``torch.device`` is not a
    valid argument. An index-less device silently means "current", which would
    otherwise put the pool on a different GPU than the tensor view over it.
    """

    def test_explicit_index_is_preserved(self):
        import torch
        from ubx._nccl_backend import _resolve_device_index
        assert _resolve_device_index(torch.device("cuda:1")) == 1

    def test_accepts_a_device_string(self):
        """Annotated ``torch.device``, but nothing stops a caller passing a str."""
        from ubx._nccl_backend import _resolve_device_index
        assert _resolve_device_index("cuda:2") == 2

    def test_returns_a_plain_int(self):
        from ubx._nccl_backend import _resolve_device_index
        assert type(_resolve_device_index("cuda:0")) is int

    @pytest.mark.requires_cuda
    def test_indexless_device_uses_current(self):
        """Bare "cuda" must resolve to the *current* device, not to 0.

        Asserting against device 0 would also pass a hardcoded ``return 0``, so
        move the current device first. On a single-GPU box the two are
        indistinguishable and this degenerates to the weaker check.
        """
        import torch
        from ubx._nccl_backend import _resolve_device_index

        target = 1 if torch.cuda.device_count() > 1 else 0
        previous = torch.cuda.current_device()
        torch.cuda.set_device(target)
        try:
            assert _resolve_device_index(torch.device("cuda")) == target
        finally:
            torch.cuda.set_device(previous)


class TestBufferOwnership:
    """``mem_alloc`` returns a Buffer that OWNS its allocation.

    Unlike the bare integer pointer it replaced, it frees on ``close()`` *and*
    on garbage collection -- a lifetime constraint the pool now has to respect.
    """

    @pytest.mark.requires_cuda
    def test_handle_converts_to_a_nonzero_address(self, nccl_backend):
        """``int(buf.handle)`` is what ``NcclSymPool`` stores as ``_raw_ptr``.

        ``allocator.py`` does pointer arithmetic on that value and passes it to
        the pybind11 extension, so the conversion has to work and must not
        yield NULL.
        """
        buf = nccl_backend._mem_alloc(1024, device=0)
        try:
            ptr = int(buf.handle)
        finally:
            buf.close()
        assert ptr != 0

    @pytest.mark.requires_cuda
    def test_close_is_idempotent(self, nccl_backend):
        """``NcclSymPool.close()`` drops its Buffer reference even when
        ``close()`` raised, which leaves the finalizer to call it a second
        time. That is only safe if the second call is a no-op.
        """
        buf = nccl_backend._mem_alloc(1024, device=0)
        buf.close()
        buf.close()

    @pytest.mark.requires_cuda
    def test_dropped_buffers_are_reclaimed(self, nccl_backend):
        """Buffers dropped without ``close()`` must still be freed.

        The first half is a positive control. ``mem_get_info`` reports
        driver-level free memory, and nothing guarantees a priori that it
        observes ``ncclMemAlloc`` -- if it did not, a finalizer that freed
        nothing would leave free memory unchanged and pass, which is also
        exactly what success looks like. Proving the allocations are visible
        first is what makes the reclaim assertion mean anything.
        """
        import gc
        import torch

        count, chunk = 50, 8 * 1024 * 1024  # 400 MiB in flight
        torch.cuda.init()
        gc.collect()
        free_start, _ = torch.cuda.mem_get_info()

        held = [nccl_backend._mem_alloc(chunk, device=0) for _ in range(count)]
        free_held, _ = torch.cuda.mem_get_info()
        observed_mib = (free_start - free_held) / 1024 ** 2
        assert observed_mib > 256, (
            f"mem_get_info saw only {observed_mib:.1f} MiB while holding "
            f"{count} x {chunk >> 20} MiB of ncclMemAlloc -- it does not "
            f"observe this allocator, so the reclaim check below is vacuous"
        )

        del held  # deliberately no close(): this is the finalizer's job
        gc.collect()
        free_end, _ = torch.cuda.mem_get_info()

        leaked_mib = (free_start - free_end) / 1024 ** 2
        assert leaked_mib < 64, (
            f"{count} unclosed buffers leaked {leaked_mib:.1f} MiB"
        )

    @pytest.mark.requires_cuda
    def test_finalizer_at_interpreter_exit_is_silent(self, nccl_backend):
        """A Buffer left to shutdown must not spew "Exception ignored in".

        Nothing calls ``close()`` here: the finalizer runs during interpreter
        teardown, by which point the nccl4py module state it reaches through
        may already be gone. If it raises there, Python swallows the traceback
        to stderr and the allocation leaks with no other signal. UB-X cannot
        catch this one -- ``NcclSymPool.close()``'s ``except`` only covers the
        explicit path.
        """
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(
            """
            import torch
            from ubx._nccl_backend import _mem_alloc
            torch.cuda.init()
            buf = _mem_alloc(1024, device=0)   # deliberately never closed
            print("ALLOCATED")
            """
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        )
        assert "ALLOCATED" in proc.stdout, f"probe never ran:\n{proc.stderr}"
        assert "Exception ignored" not in proc.stderr, (
            f"finalizer raised during interpreter shutdown:\n{proc.stderr}"
        )
        assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stderr}"
