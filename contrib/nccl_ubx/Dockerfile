# syntax=docker/dockerfile:1
# Dockerfile for UB-X release build
#
# UB-X ships under contrib/nccl_ubx/ in the NVIDIA/nccl repository. This
# Dockerfile clones NCCL once and builds both libnccl.so and UB-X from
# that single source tree.
#
# Supports linux/arm64 and linux/amd64. CUDA architectures are selected
# automatically based on the target platform unless overridden via --build-arg.
#
# Build arguments:
#   PYTORCH_IMAGE - Base PyTorch container (default: nvcr.io/nvidia/pytorch:26.04-py3)
#   NCCL_REPO     - NCCL git repository URL (default: https://github.com/NVIDIA/nccl.git)
#   NCCL_BRANCH   - NCCL branch/tag — used for BOTH libnccl.so and UB-X.
#                   UB-X lives at contrib/nccl_ubx/ inside this checkout.
#                   The actual libnccl.so shipped in the container is built
#                   from this source, so any feature newer than the PyTorch
#                   base image's bundled libnccl is present at runtime
#                   instead of being shadowed.
#   CUDA_ARCH     - Override CUDA architectures (auto-detected if empty)
#                   arm64 default: 9.0;10.0;10.3a  (H100/B200/GB300 ARM)
#                   amd64 default: 9.0;10.0          (H100/B100)
#
# Examples:
#   docker buildx build --platform linux/arm64 -t pytorch-ubx:latest-arm64 .
#   docker buildx build --platform linux/amd64 -t pytorch-ubx:latest-amd64 .
#   docker buildx build --platform linux/amd64,linux/arm64 -t pytorch-ubx:latest .

ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:26.04-py3

# TARGETARCH is set automatically by docker buildx (amd64 or arm64)
ARG TARGETARCH

FROM --platform=linux/${TARGETARCH} ${PYTORCH_IMAGE}

# Build arguments
ARG TARGETARCH
ARG NCCL_REPO=https://github.com/NVIDIA/nccl.git
ARG NCCL_BRANCH=master
ARG CUDA_ARCH=""
ARG CUDA_HOME=/usr/local/cuda

# Labels
LABEL nccl.branch="${NCCL_BRANCH}" \
      ubx.source="nccl/contrib/nccl_ubx" \
      description="PyTorch container with UB-X and ubx-bench"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# ---------------------------------------------------------------------------
# NCCL: clone, build libnccl.so from source, install system-wide, then
# install matching nccl4py bindings. Done BEFORE UB-X so UB-X compiles
# against the same NCCL headers + library that will be loaded at runtime.
#
# Why this exists: previously only the Python bindings were installed
# (`pip install /opt/nccl/bindings/nccl4py`). The runtime libnccl.so.2 was
# inherited from the PyTorch base image (e.g. 26.04 ships 2.29.7), so any
# feature added past the base-image's NCCL version was silently missing
# even when NCCL_BRANCH advertised a newer version. The verify step below
# makes any such mismatch visible in the build log.
# ---------------------------------------------------------------------------

RUN test -n "${NCCL_BRANCH}" || (echo "ERROR: NCCL_BRANCH build-arg must be set" && exit 1) && \
    git clone --depth 1 --branch ${NCCL_BRANCH} ${NCCL_REPO} /opt/nccl

# Build libnccl from source and install to /usr/local. `src.build` produces
# build/lib/libnccl.so.X.Y.Z + symlinks; `src.install PREFIX=/usr/local`
# copies them to /usr/local/lib + nccl.h to /usr/local/include. The
# ld.so.conf.d entry pins /usr/local/lib ahead of the base image's
# /lib/<triplet> so any consumer that dlopen's `libnccl.so.2` — including
# C++ binaries that bypass torch — resolves to the freshly-built copy.
# ldconfig refreshes the linker cache.
RUN cd /opt/nccl && \
    make -j"$(nproc)" src.build CUDA_HOME=${CUDA_HOME} && \
    make src.install PREFIX=/usr/local && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/00-local-nccl.conf && \
    ldconfig && \
    ls -la /usr/local/lib/libnccl.so* /usr/local/include/nccl.h

# Replace torch's bundled libnccl with symlinks to the freshly-built one.
# PyTorch's wheel ships libnccl.so inside torch/lib/ and the dynamic linker
# may resolve `libnccl.so.2` to that copy via torch's RPATH before checking
# /usr/local/lib. Without this step LD_PRELOAD would be required at runtime
# to get the new libnccl loaded.
RUN TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')" && \
    if [ -d "${TORCH_LIB}" ]; then \
        rm -f "${TORCH_LIB}/libnccl.so" "${TORCH_LIB}/libnccl.so.2" "${TORCH_LIB}"/libnccl.so.2.* ; \
        ln -s /usr/local/lib/libnccl.so.2 "${TORCH_LIB}/libnccl.so.2" ; \
        ln -s /usr/local/lib/libnccl.so.2 "${TORCH_LIB}/libnccl.so" ; \
        echo "Replaced torch libnccl in ${TORCH_LIB}:" ; ls -la "${TORCH_LIB}"/libnccl.so* ; \
    else \
        echo "torch/lib not found at ${TORCH_LIB} — skipping bundle replacement" ; \
    fi

# Install nccl4py (Python bindings) + pytest for tests/distributed/.
# cuda-python (provides cuda.bindings) is pre-installed in the pytorch base.
RUN pip install --no-cache-dir /opt/nccl/bindings/nccl4py && \
    pip install --no-cache-dir pytest

# Verify the *runtime* libnccl + nccl4py + torch all line up. If a future
# PyTorch base image bumps its bundle past NCCL_BRANCH, or torch's lib
# directory layout changes and the symlink step above silently skips,
# this print will surface the mismatch instead of hiding it until the
# first NCCL communicator init at runtime.
RUN python -c "\
import ctypes, torch, nccl, pytest; \
v = ctypes.c_int(); \
ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v)); \
runtime = '%d.%d.%d' % (v.value // 10000, (v.value // 100) % 100, v.value % 100); \
print('runtime libnccl:', runtime, '(version code', v.value, ')'); \
print('torch.cuda.nccl.version:', torch.cuda.nccl.version()); \
print('nccl4py loaded from:', nccl.__file__); \
print('pytest:', pytest.__version__)"

# ---------------------------------------------------------------------------
# UB-X: build against the headers + libnccl just installed.
#
# Uses setup.py build_ext --inplace instead of pip install -e . to avoid
# _dlpack_exchange_api AttributeError in some PyTorch base containers.
# UB-X lives at /opt/nccl/contrib/nccl_ubx in the already-cloned NCCL tree
# (the same checkout used for libnccl build above); ubx-bench is at
# contrib/nccl_ubx/bench/. A /opt/ubx symlink is kept for backwards
# compatibility with anything that still references the old path.
# CUDA_ARCH auto-selects per platform if not explicitly provided.
# ---------------------------------------------------------------------------
RUN set -e; \
    if [ -z "${CUDA_ARCH}" ]; then \
        if [ "${TARGETARCH}" = "amd64" ]; then ARCH="9.0;10.0"; \
        else ARCH="9.0;10.0;10.3a"; fi; \
    else ARCH="${CUDA_ARCH}"; fi; \
    echo "$ARCH" > /opt/.cuda_arch; \
    ln -s /opt/nccl/contrib/nccl_ubx /opt/ubx && \
    cd /opt/nccl/contrib/nccl_ubx && \
    rm -rf build/ && \
    TORCH_CUDA_ARCH_LIST="$ARCH" python setup.py build_ext --inplace

# Set PYTHONPATH so ubx (built inplace) and ubx-bench are importable.
# /opt/ubx is the legacy mount path (symlink to the contrib/nccl_ubx subtree).
ENV PYTHONPATH="/opt/ubx:/opt/ubx/bench"

# Recommended runtime environment variables
ENV CUDA_DEVICE_MAX_CONNECTIONS=1

# Create version info script
RUN cat > /usr/local/bin/show-ubx-info << 'EOF'
#!/bin/bash
echo "=== UB-X ==="
python -c "import ubx; print('Version:', ubx.__version__)"
python -c "import ubx; print('API contract version:', ubx.get_version())"
echo ""
echo "=== Source: nccl/contrib/nccl_ubx ==="
cd /opt/nccl && git log -1 --oneline
echo ""
echo "=== ubx-bench ==="
python -c "import ubx_bench; print('Version:', ubx_bench.__version__)"
echo ""
echo "=== Available Collectives ==="
echo "  UBX-supported (ubx kernels in csrc/ubx.cu):"
echo "    all_reduce  all_gather  all_to_all  alltoallv  a2av_dispatch  a2av_mxfp8"
echo "  NCCL-only (no UBX kernel — bench falls back to torch.distributed):"
echo "    reduce_scatter  sendrecv"
echo ""
echo "=== NCCL ==="
python -c "
import ctypes, torch
v = ctypes.c_int(); ctypes.CDLL('libnccl.so.2').ncclGetVersion(ctypes.byref(v))
print('runtime libnccl: %d.%d.%d' % (v.value // 10000, (v.value // 100) % 100, v.value % 100))
print('torch.cuda.nccl.version:', torch.cuda.nccl.version())
"
echo ""
echo "=== CUDA Architectures ==="
echo "Built for: $(cat /opt/.cuda_arch 2>/dev/null || echo unknown)"
EOF
RUN chmod +x /usr/local/bin/show-ubx-info

WORKDIR /workspace
CMD ["/bin/bash"]
